#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torchvision
from src.data import load_dataset
from src.metrics import BitwiseAccuracy, PerturbationNorm, AttackTime, PSNR, SSIM
from src.utils.callbacks import *
from src.models import *
from src.utils import setup_cfg, log_cfg
from src.attacks.auoa import AUOA, ScalarTracker, TensorBoardHparamsLogger, \
    TensorBoardStatLogger, TensorBoardProfiler, TestLogger


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
    parser.add_argument("--wm-ckpt-file-path", type=str, default="")

    # Data args
    parser.add_argument("--data-root-dir", type=str, default="./Datasets")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "imagenet", "cc"])
    parser.add_argument("--message-length", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-train-images", type=int, default=-1)

    # Trainer args
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--xi", type=float, default=1000.0)

    # Tester args
    parser.add_argument("--auto-restore", type=parse_boolean, default=True)
    parser.add_argument("--ckpt-file-path", type=str, default="")
    parser.add_argument("--run-timestamp", type=str, default="")
    parser.add_argument("--test-num-eval-users", type=int, default=1)
    parser.add_argument("--test-num-total-users", type=int, default=1)

    # Callback args
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--verbose", type=parse_boolean, default=False)

    args = parser.parse_args()
    return args


def prepare_dataloader(args):

    args.logger(f"===========================> Loading dataset {args.dataset}:")
    train_dataset = load_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, part="train",
        img_size=args.img_size, message_length=args.message_length, num_train_images=args.num_train_images,
    )
    args.logger(f"# Training images: {len(train_dataset)}")
    args.train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4 * torch.cuda.device_count(), pin_memory=True
    )
    test_dataset = load_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, part="test",
        img_size=args.img_size, message_length=args.message_length, num_train_images=-1
    )
    args.logger(f"# Testing images: {len(test_dataset)}")
    args.test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False,
        num_workers=4 * torch.cuda.device_count(), pin_memory=True
    )


if __name__ == "__main__":

    args = parse_arguments()
    args.exp_type = f"auoa_attack_{args.watermarking_model}"
    setup_cfg(args)
    log_cfg(args)

    # preparation
    prepare_dataloader(args)
    args.model = load_model(
        wm_model=args.watermarking_model, ckpt_file_path=args.wm_ckpt_file_path,
        device=args.device,
    )
    args.snn = torchvision.models.squeezenet1_1(pretrained=False, progress=True)
    args.snn.classifier[1] = torch.nn.Conv2d(512, args.k, kernel_size=(1, 1), stride=(1, 1))

    # instantiate an attacker
    attacker = AUOA(device=args.device, seed=args.seed)

    # train
    callbacks = []
    callbacks.append(ScalarTracker({
        "bitwise_acc": BitwiseAccuracy(),
        "psnr": PSNR(),
        "ssim": SSIM(),
    }))
    callbacks.append(TensorBoardProfiler(args.writer))
    callbacks.append(TrainLogger(args.logger, args.num_epochs, args.verbose, args.print_freq))
    callbacks.append(TensorBoardScalarLogger(args.writer))
    callbacks.append(HistoryRecorder(args.exp_dir_path, args.exp_datatime))
    callbacks.append(HistoryPlotter(args.exp_dir_path, args.exp_datatime))
    callbacks.append(CheckpointSaver(args.exp_dir_path, args.exp_datatime, monitor=None))
    attacker.train(
        k=args.k, snn=args.snn, decoder=args.model, loader=args.train_loader, img_size=args.img_size,
        message_length=args.message_length, num_epochs=args.num_epochs, lr=args.lr,
        xi=args.xi, callbacks=callbacks
    )

    # test
    if args.auto_restore:
        best_ckpt_file_path = os.path.join(args.exp_dir_path, "checkpoints", f"run-{args.exp_datatime}", "ckpt.pth.tar")
        args.ckpt_file_path = best_ckpt_file_path
    callbacks = []
    callbacks.append(ScalarTracker({
        "bitwise_acc": BitwiseAccuracy(),
        "psnr": PSNR(),
        "ssim": SSIM(),
        "perturbation_norm": PerturbationNorm(),
        "attack_time": AttackTime(),
    }))
    callbacks.append(TensorBoardStatLogger(args.writer, args.message_length, args.test_num_eval_users))
    callbacks.append(TestLogger(args.logger))
    callbacks.append(TensorBoardHparamsLogger(args.writer, {k: v for k, v in args._get_kwargs()}))
    attacker.test(
        snn=args.snn, decoder=args.decoder, loader=args.test_loader, xi=args.xi,
        ckpt_file_path=args.ckpt_file_path, callbacks=callbacks
    )
    for metric_name, meter in attacker.state.eval_metric_meters.items():
        args.logger(f"Averaged test {metric_name}: {meter.avg.item():.3f}")
    drop_n4 = attacker.state.eval_metric_meters["identify_acc_n4"].avg.item() - attacker.state.eval_metric_meters["identify_acc_n4_adv"].avg.item()
    drop_n6 = attacker.state.eval_metric_meters["identify_acc_n6"].avg.item() - attacker.state.eval_metric_meters["identify_acc_n6_adv"].avg.item()
    args.logger(f"Averaged identify_acc_drop_n4: {drop_n4:.3f}")
    args.logger(f"Averaged identify_acc_drop_n6: {drop_n6:.3f}")