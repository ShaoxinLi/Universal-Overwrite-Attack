#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime

import torch
import torch.backends.cudnn as cudnn

from .logger import Logger
# from torch.utils.tensorboard import SummaryWriter
from .file import check_dir, get_exp_dir

# from torch.utils.tensorboard.summary import hparams

#
# class MySummaryWriter(SummaryWriter):
#     def add_hparams(self, hparam_dict, metric_dict):
#         torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
#         if type(hparam_dict) is not dict or type(metric_dict) is not dict:
#             raise TypeError('hparam_dict and metric_dict should be dictionary.')
#         exp, ssi, sei = hparams(hparam_dict, metric_dict)
#
#         logdir = self._get_file_writer().get_logdir()
#         with SummaryWriter(log_dir=logdir) as w_hp:
#             w_hp.file_writer.add_summary(exp)
#             w_hp.file_writer.add_summary(ssi)
#             w_hp.file_writer.add_summary(sei)
#             for k, v in metric_dict.items():
#                 w_hp.add_scalar(k, v)


def setup_cfg(args):

    # set the experiment dir path
    args.exp_dir_path = get_exp_dir(args.exp_root_dir, args.exp_type, args.dataset, args.dir_suffix)
    check_dir(args.exp_dir_path)

    # get the experiment start datetime
    if "run_timestamp" in [i[0] for i in args._get_kwargs()] and args.run_timestamp:
        args.exp_datatime = args.run_timestamp
    else:
        now = datetime.datetime.now()
        args.exp_datatime = now.strftime("%Y-%m-%d-%H-%M-%S")

    # set the logger
    args.logger = Logger(
        log_dir_path=os.path.join(args.exp_dir_path, "logs"),
        logger_name=args.exp_type,
        log_file_name=f"log_{args.exp_datatime}.txt"
    )

    # set the tensorboard writer
    args.writer = MySummaryWriter(log_dir=os.path.join(args.exp_dir_path, "runs", f"run-{args.exp_datatime}"))

    # set the device
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)


def log_cfg(args):

    args.logger(f"===========================> System info:")
    python_version = sys.version.replace('\n', ' ')
    args.logger(f"Python version: {python_version}")
    args.logger(f"Torch version: {torch.__version__}")
    args.logger(f"Cudnn version: {torch.backends.cudnn.version()}")
    args.logger(f"===========================> Hyperparameters:",)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in sorted(state.items(), key=lambda x: x[0]):
        args.logger(f"{key}: {value}")
