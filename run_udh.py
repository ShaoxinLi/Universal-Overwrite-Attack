#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch.nn

from src.data import load_dataset
from src.models import get_num_parameters
from src.models.udh import UnetGenerator, RevealNet
from src.noise_layers import Noiser
from src.utils.callbacks import *
from src.utils import setup_cfg, log_cfg
from src.metrics import BitwiseAccuracy, PSNR, SSIM
from src.watermarkings.udh import UDH, ScalarTracker, TensorBoardImageLogger, \
    TensorBoardStatLogger, TensorBoardHparamsLogger, TensorBoardProfiler


def parse_arguments():

    def parse_none(value):
        if value == "":
            return None
        return value

    def parse_boolean(value):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            assert False

    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp-root-dir", type=str, default="./archive")
    parser.add_argument("--dir-suffix", type=str, default="")
    parser.add_argument("--device", type=parse_none, default=None)
    parser.add_argument("--seed", type=int, default=3407)

    # Network args
    parser.add_argument("--message-length", type=int, default=30)

    # Data args
    parser.add_argument("--data-root-dir", type=str, default="./Datasets")
    parser.add_argument("--dataset", type=str, choices=["coco", "imagenet", "cc"], default="coco")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-train-images", type=int, default=-1)

    # Trainer args
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--train-noises", type=str, default="identity")
    parser.add_argument("--train-a", type=float, nargs="+", default=[1., 5.])
    parser.add_argument("--train-sigma", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--train-sigma-blur", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--train-Q", type=int, nargs="+", default=[10, 100])

    # Tester args
    parser.add_argument("--auto-restore", type=parse_boolean, default=True)
    parser.add_argument("--ckpt-file-path", type=str, default="")
    parser.add_argument("--test-noises", type=str, default="")
    parser.add_argument("--test-a", type=float, nargs="+", default=[1., 1.])
    parser.add_argument("--test-sigma", type=float, nargs="+", default=[0.1, 0.1])
    parser.add_argument("--test-sigma-blur", type=float, nargs="+", default=[1.0, 1.0])
    parser.add_argument("--test-Q", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--num-eval-users", type=int, default=1)
    parser.add_argument("--num-total-users", type=int, default=1)
    parser.add_argument("--run-timestamp", type=str, default="")

    # Callback args
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--verbose", type=parse_boolean, default=False)

    args = parser.parse_args()
    return args


def prepare_dataloader(args):

    args.logger(f"===========================> Loading dataset {args.dataset}:")
    train_dataset = load_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, part="victim",
        img_size=args.img_size, message_length=args.message_length, num_train_images=args.num_train_images,
    )

    args.logger(f"# Training images: {len(train_dataset)}")
    args.train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 * torch.cuda.device_count(), pin_memory=True
    )

    test_dataset, test_gt_messages = load_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, part="test",
        img_size=args.img_size, message_length=args.message_length, num_train_images=-1
    )
    args.logger(f"# Testing images: {len(test_dataset)}")
    args.test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False,
        num_workers=4 * torch.cuda.device_count(), pin_memory=False
    )
    args.test_gt_messages = test_gt_messages


def prepare_model(args):

    args.logger(f"===========================> Loading UDH network")
    args.encoder = UnetGenerator(
        input_nc=3, output_nc=3, num_downs=5, norm_layer=torch.nn.BatchNorm2d, output_function=torch.nn.Tanh
    )
    args.decoder = RevealNet(
        input_nc=3, output_nc=3, norm_layer=torch.nn.BatchNorm2d, output_function=torch.nn.Sigmoid,
        message_length=args.message_length
    )
    args.noiser = Noiser(
        noises=args.train_noises, a=args.train_a, sigma=args.train_sigma, sigma_blur=args.train_sigma_blur,
        Q=args.train_Q, decoder=args.decoder, message_length=args.message_length, img_size=args.img_size,
    )

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0)

    args.encoder.apply(weights_init)
    args.decoder.apply(weights_init)

    if args.verbose:
        args.logger(f"===========================> Encoder :\n {args.encoder}")
        args.logger(f"# parameters: {get_num_parameters(args.encoder)}")
        args.logger(f"===========================> Decoder :\n {args.decoder}")
        args.logger(f"# parameters: {get_num_parameters(args.decoder)}")


if __name__ == "__main__":

    args = parse_arguments()
    a = {k: v for k, v in args._get_kwargs()}
    args.exp_type = "wm_udh"
    setup_cfg(args)
    log_cfg(args)

    # preparation
    prepare_dataloader(args)
    prepare_model(args)

    # instantiate a UDH
    udh = UDH(device=args.device, seed=args.seed)

    # train
    callbacks = []
    callbacks.append(ScalarTracker({
        "bitwise_acc": BitwiseAccuracy(),
        "psnr": PSNR(),
        "ssim": SSIM(),
    }))
    # callbacks.append(TensorBoardImageLogger(args.writer, 4, (256, 256)))
    # callbacks.append(TensorBoardProfiler(args.writer))
    callbacks.append(TrainLogger(args.logger, args.num_epochs, args.verbose, args.print_freq))
    callbacks.append(TensorBoardScalarLogger(args.writer))
    callbacks.append(HistoryRecorder(args.exp_dir_path, args.exp_datatime))
    callbacks.append(HistoryPlotter(args.exp_dir_path, args.exp_datatime))
    callbacks.append(CheckpointSaver(args.exp_dir_path, args.exp_datatime, monitor=None))
    udh.train(
        encoder=args.encoder, decoder=args.decoder, noiser=args.noiser, loader=args.train_loader,
        message_length=args.message_length, num_epochs=args.num_epochs,
        lr=args.lr, callbacks=callbacks
    )

    # test
    args.noiser = Noiser(
        noises=args.test_noises, a=args.test_a, sigma=args.test_sigma, sigma_blur=args.test_sigma_blur,
        Q=args.test_Q, decoder=args.decoder, message_length=args.message_length, img_size=args.img_size
    )
    if args.auto_restore:
        best_ckpt_file_path = os.path.join(args.exp_dir_path, "checkpoints", f"run-{args.exp_datatime}", "ckpt.pth.tar")
        args.ckpt_file_path = best_ckpt_file_path
    callbacks = []
    callbacks.append(ScalarTracker({
        "bitwise_acc": BitwiseAccuracy(),
        "psnr": PSNR(),
        "ssim": SSIM(),
    }))
    # callbacks.append(TensorBoardImageLogger(args.writer, 4, (256, 256)))
    callbacks.append(TensorBoardStatLogger(args.writer, args.message_length, args.num_total_users))
    callbacks.append(TensorBoardHparamsLogger(args.writer, {k: v for k, v in args._get_kwargs()}))
    udh.test(
        encoder=args.encoder, decoder=args.decoder, noiser=args.noiser, loader=args.test_loader,
        gt_messages=args.test_gt_messages, num_eval_users=args.num_eval_users,
        num_total_users=args.num_total_users, ckpt_file_path=args.ckpt_file_path, callbacks=callbacks
    )
    args.logger(f"Ground truth message set at testing: {udh.state.message_set.tolist()}")
    for metric_name, meter in udh.state.eval_metric_meters.items():
        args.logger(f"Test {metric_name}: {meter.avg.item()}")
