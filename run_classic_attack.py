#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse

from src.data import load_dataset
from src.noise_layers import Noiser
from src.models import *
from src.metrics import BitwiseAccuracy, AttackTime, PSNR, SSIM
from src.utils import CheckpointIO, setup_cfg, log_cfg
from src.attacks.classic_attack import ClassicAttack, ScalarTracker, TensorBoardImageLogger, \
    TensorBoardStatLogger, TensorBoardHparamsLogger, TestLogger


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
    parser.add_argument("--watermarking-model", type=str, choices=["hd", "udh", "ssl", "ros", "criw", "resnet"], default="hd")
    parser.add_argument("--ckpt-file-path", type=str, default="")

    # Data args
    parser.add_argument("--data-root-dir", type=str, default="./Datasets")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "imagenet", "cc"])
    parser.add_argument("--message-length", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=128)

    # Attacker args
    parser.add_argument("--method", type=str, choices=["identity", "brightness", "gaussian", "blur", "jpeg"], default="identity")
    parser.add_argument("--a", type=float, nargs="+", default=[1, 1])
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.1, 0.1])
    parser.add_argument("--sigma-blur", type=float, nargs="+", default=[1.0, 1.0])
    parser.add_argument("--Q", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--num-eval-users", type=int, default=1)
    parser.add_argument("--num-total-users", type=int, default=1)
    parser.add_argument("--num-tests", type=int, default=1)

    args = parser.parse_args()
    return args


def prepare_dataloader(args):

    args.logger(f"===========================> Loading dataset {args.dataset}:")
    test_dataset, test_gt_messages = load_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, part="test",
        img_size=args.img_size, message_length=args.message_length, num_train_images=-1
    )
    args.logger(f"# Testing images: {len(test_dataset)}")
    args.loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False,
        num_workers=4 * torch.cuda.device_count(), pin_memory=False
    )
    args.gt_messages = test_gt_messages


def prepare_model(args):
    args.model = load_model(
        wm_model=args.watermarking_model, ckpt_file_path=args.wm_ckpt_file_path,
        device=args.device,
    )
    args.noiser = Noiser(
        noises=args.method, a=args.a, sigma=args.sigma, sigma_blur=args.sigma_blur,
        Q=args.Q, decoder=args.decoder, message_length=args.message_length, img_size=args.img_size
    )


if __name__ == "__main__":

    args = parse_arguments()
    args.exp_type = f"classic_attack_{args.watermarking_model}"
    setup_cfg(args)
    log_cfg(args)

    # preparation
    prepare_dataloader(args)
    prepare_model(args)

    # instantiate an attacker
    attacker = ClassicAttack(device=args.device, seed=args.seed)

    # attack
    callbacks = []
    callbacks.append(ScalarTracker({
        "bitwise_acc": BitwiseAccuracy(),
        "psnr": PSNR(),
        "ssim": SSIM(),
        "attack_time": AttackTime(),
    }))
    # callbacks.append(TensorBoardImageLogger(args.writer, (256, 256)))
    callbacks.append(TensorBoardStatLogger(args.writer, args.message_length, args.num_total_users))
    callbacks.append(TestLogger(args.logger))
    callbacks.append(TensorBoardHparamsLogger(args.writer, {k: v for k, v in args._get_kwargs()}))
    attacker.attack(
        encoder=args.encoder, decoder=args.decoder, noiser=args.noiser, loader=args.loader,
        gt_messages=args.gt_messages, num_eval_users=args.num_eval_users,
        num_total_users=args.num_total_users, num_tests=args.num_tests,
        callbacks=callbacks
    )
    for metric_name, meter in attacker.state.eval_metric_meters.items():
        args.logger(f"Averaged test {metric_name}: {meter.avg.item():.3f}")
    drop_n4 = attacker.state.eval_metric_meters["identify_acc_n4"].avg.item() - attacker.state.eval_metric_meters[
        "identify_acc_n4_adv"].avg.item()
    drop_n6 = attacker.state.eval_metric_meters["identify_acc_n6"].avg.item() - attacker.state.eval_metric_meters[
        "identify_acc_n6_adv"].avg.item()
    args.logger(f"Averaged identify_acc_drop_n4: {drop_n4:.3f}")
    args.logger(f"Averaged identify_acc_drop_n6: {drop_n6:.3f}")