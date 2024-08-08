#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import shutil
import time
import torch
import torchvision
from wmattacker import DiffWMAttacker
from src.utils.train import AverageMeter
from src.metrics import BitwiseAccuracy, PSNR, SSIM, AttackTime
from collections import defaultdict
from src.utils.file import check_dir, list_files
from src.diffusers import ReSDPipeline


class MyImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
pipe.set_progress_bar_config(disable=True)
pipe.to(device)


def test(logger, std=1.0, num_total_users=10, num_tests=10):

    attacker = DiffWMAttacker(pipe, batch_size=5, noise_step=60, std=std, captions={})

    # metric
    bitacc = BitwiseAccuracy()
    psnr = PSNR()
    ssim = SSIM()
    attack_time = AttackTime()

    assert num_total_users in [1, 10, 100, 1000]
    if num_total_users == 1:
        tau_n4 = 25 / 30.
        tau_n6 = 27 / 30.
    elif num_total_users == 10:
        tau_n4 = 26. / 30.
        tau_n6 = 28. / 30.
    elif num_total_users == 100:
        tau_n4 = 27. / 30.
        tau_n6 = 29. / 30.
    elif num_total_users == 1000:
        tau_n4 = 28. / 30.
        tau_n6 = 29. / 30.
    else:
        assert False

    metric_meters = defaultdict(AverageMeter)
    eval_metric_meters = defaultdict(AverageMeter)

    for test_idx in range(num_tests):

        for meter in metric_meters.values():
            meter.reset()

        org_image_list = []
        accs_dict = defaultdict(list)
        pred_messages_dict = defaultdict(list)
        gt_message_list = []

        # load gt messages
        message_set = all_messages[test_idx * num_total_users: (test_idx + 1) * num_total_users]
        n, m = divmod(len(org_dataset), message_set.size(0))
        messages = message_set.repeat(n, 1)
        messages = torch.cat([messages, message_set[:m]], dim=0) if m != 0 else messages

        temp_watermarked_image_folder_path = f"./Datasets/imagenet10k/{watermarking_model}/test_n{num_total_users}/c{test_idx + 1}_watermarked/0"
        if os.path.exists(temp_watermarked_image_folder_path):
            shutil.rmtree(temp_watermarked_image_folder_path)
        check_dir(temp_watermarked_image_folder_path)

        # process original images
        for i, (org_images, indices, _) in enumerate(org_loader):
            org_images = org_images.to(device, non_blocking=True)
            gt_messages = messages[indices]

            # decode original image
            org_messages = decoder(org_images)

            accs = bitacc(org_messages, message_set=message_set)
            accs_org, max_indices = torch.max(accs, dim=1)
            pred_messages_org = message_set[max_indices]
            accs_dict["org"].append(accs_org)

            org_image_list.append(org_images)

            # encode images
            watermarked_images = encoder(org_images, gt_messages)
            watermarked_images = watermarked_images.clamp_(0., 1.)

            watermarked_messages = decoder(watermarked_images)

            metric_meters["acc_wm_gt"].update(bitacc(watermarked_messages, gt_messages=gt_messages, mean=True), watermarked_images.size(0))
            accs = bitacc(watermarked_messages, message_set=message_set)
            accs_wm, max_indices = torch.max(accs, dim=1)
            pred_messages_wm = message_set[max_indices]
            accs_dict["wm"].append(accs_wm)
            pred_messages_dict["wm"].append(pred_messages_wm)

            # save watermarked images
            for index, image in zip(indices, watermarked_images):
                wm_image_path = os.path.join(os.path.join(temp_watermarked_image_folder_path, f"{int(index):05d}.png"))
                torchvision.utils.save_image(image, fp=wm_image_path)

        org_image_list = torch.cat(org_image_list, dim=0)

        # attack
        watermarked_image_paths = list_files(temp_watermarked_image_folder_path)
        watermarked_image_paths = sorted(watermarked_image_paths)
        temp_attacked_image_folder_path = f"./Datasets/imagenet10k/{watermarking_model}/test_n{num_total_users}/c{test_idx + 1}_attacked/0"

        if os.path.exists(temp_attacked_image_folder_path):
            shutil.rmtree(temp_attacked_image_folder_path)
        check_dir(temp_attacked_image_folder_path)

        for path in watermarked_image_paths:
            img_name = os.path.split(path)[-1]
            att_img_path = os.path.join(temp_attacked_image_folder_path, img_name)

            attack_start_time = int(round(time.time() * 1000))
            attacker.attack([path], [att_img_path])
            attack_end_time = int(round(time.time() * 1000))

            metric_meters["attack_time"].update(attack_time(attack_start_time, attack_end_time), 1)

        # process attacked images
        temp_attacked_image_folder_path = os.path.split(temp_attacked_image_folder_path)[0]
        adv_dataset = MyImageFolder(temp_attacked_image_folder_path, transforms)
        adv_dataloader = torch.utils.data.DataLoader(
            dataset=adv_dataset, batch_size=16, shuffle=False,
            num_workers=4 * torch.cuda.device_count(), pin_memory=False
        )

        counter = 0
        for i, (adv_images, indices) in enumerate(adv_dataloader):
            adv_images = adv_images.to(device, non_blocking=True)
            gt_messages = messages[counter: counter + adv_images.size(0)]
            counter = counter + adv_images.size(0)

            # decode adv image
            adv_messages = decoder(adv_images)

            metric_meters["acc_adv_gt"].update(bitacc(adv_messages, gt_messages=gt_messages, mean=True), adv_images.size(0))
            metric_meters["psnr"].update(psnr(adv_images, org_image_list[indices]), adv_images.size(0))
            metric_meters["ssim"].update(ssim(adv_images, org_image_list[indices]), adv_images.size(0))

            accs = bitacc(adv_messages, message_set=message_set)
            accs_adv, max_indices = torch.max(accs, dim=1)
            pred_messages_adv = message_set[max_indices]
            accs_dict["adv"].append(accs_adv)
            pred_messages_dict["adv"].append(pred_messages_adv)

            gt_message_list.append(gt_messages)

        for metric_name, meter in metric_meters.items():
            eval_metric_meters[metric_name].update(meter.avg, meter.count)

        accs_org = torch.cat(accs_dict["org"], dim=0)
        accs_wm = torch.cat(accs_dict["wm"], dim=0)
        accs_adv = torch.cat(accs_dict["adv"], dim=0)
        pred_messages_wm = torch.cat(pred_messages_dict["wm"], dim=0)
        pred_messages_adv = torch.cat(pred_messages_dict["adv"], dim=0)
        gt_message_list = torch.cat(gt_message_list, dim=0)

        # unattacked
        probs = torch.cat([accs_org, accs_wm], dim=0)
        labels = torch.cat([
            torch.zeros(accs_org.size(0), device=probs.device),
            torch.ones(accs_wm.size(0), device=probs.device)
        ], dim=0)
        for des, tau in [("n4", tau_n4), ("n6", tau_n6)]:
            predictions = torch.where(probs >= tau, 1, 0)
            fp = torch.sum((predictions == 1) & (labels == 0))
            tp = torch.sum((predictions == 1) & (labels == 1))
            fn = torch.sum((predictions == 0) & (labels == 1))
            tn = torch.sum((predictions == 0) & (labels == 0))
            tp_indices = predictions[labels == 1] == 1
            tidt = torch.sum(torch.all(torch.eq(pred_messages_wm[tp_indices], gt_message_list[tp_indices]), dim=1))

            eval_metric_meters[f"far_{des}"].update(1. - tidt / tp, 1)
            eval_metric_meters[f"fnr_{des}"].update(fn / (tp + fn), 1)
            eval_metric_meters[f"identify_acc_{des}"].update(tidt / (tp + fn), 1)

        # attacked
        probs = torch.cat([accs_org, accs_adv], dim=0)
        labels = torch.cat([
            torch.zeros(accs_org.size(0), device=probs.device),
            torch.ones(accs_adv.size(0), device=probs.device)
        ], dim=0)
        for des, tau in [("n4", tau_n4), ("n6", tau_n6)]:
            predictions = torch.where(probs >= tau, 1, 0)
            fp = torch.sum((predictions == 1) & (labels == 0))
            tp = torch.sum((predictions == 1) & (labels == 1))
            fn = torch.sum((predictions == 0) & (labels == 1))
            tn = torch.sum((predictions == 0) & (labels == 0))
            tp_indices = predictions[labels == 1] == 1
            tidt = torch.sum(torch.all(torch.eq(pred_messages_adv[tp_indices], gt_message_list[tp_indices]), dim=1))

            eval_metric_meters[f"far_{des}_adv"].update(1. - tidt / tp, 1)
            eval_metric_meters[f"fnr_{des}_adv"].update(fn / (tp + fn), 1)
            eval_metric_meters[f"identify_acc_{des}_adv"].update(tidt / (tp + fn), 1)

        # logger(f"{test_idx + 1}-th testing ground truth message set: {message_set.tolist()}")
        for metric_name, meter in eval_metric_meters.items():
            logger(f"{test_idx + 1}-th testing {metric_name}: {meter.val.item()}")

        # delete
        # shutil.rmtree(temp_attacked_image_folder_path)
        # shutil.rmtree(os.path.split(temp_watermarked_image_folder_path)[0])

    for metric_name, meter in eval_metric_meters.items():
        logger(f"Averaged test {metric_name}: {meter.avg.item():.3f}")
    drop_n4 = eval_metric_meters["identify_acc_n4"].avg.item() - eval_metric_meters[
        "identify_acc_n4_adv"].avg.item()
    drop_n6 = eval_metric_meters["identify_acc_n6"].avg.item() - eval_metric_meters[
        "identify_acc_n6_adv"].avg.item()
    logger(f"Averaged identify_acc_drop_n4: {drop_n4:.3f}")
    logger(f"Averaged identify_acc_drop_n6: {drop_n6:.3f}")
