#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import tqdm
import time
import torch
import torchvision
import numpy as np
from src.utils.train import AverageMeter
from src.metrics import BitwiseAccuracy, PSNR, SSIM, AttackTime
from src.metrics.ssim import create_window, _ssim
from collections import defaultdict
from src.noise_layers import Noiser
from src.utils.logger import Logger
from src.utils import check_dir



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE_IMAGENET = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1).to(device)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1).to(device)
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    NORMALIZE_IMAGENET,
])


class NormLayerWrapper(torch.nn.Module):
    def __init__(self, backbone, head):
        super(NormLayerWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        output = self.backbone(x)
        return self.head(output)


class MyImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index, path


class NewBitwiseAccuracy(BitwiseAccuracy):
    def __init__(self):
        super(NewBitwiseAccuracy, self).__init__()

    @torch.no_grad()
    def __call__(self, pred_messages, gt_messages=None, message_set=None, mean=True):
        if gt_messages is not None:
            pred_messages = torch.sign(pred_messages)   # {-1, 1}
            pred_messages[pred_messages == -1.] = 0.    # {0, 1}
            accs = 1. - torch.sum(torch.abs(pred_messages - gt_messages), dim=1) / gt_messages.size(1)
            if mean:
                return torch.sum(accs) / gt_messages.size(0)
            else:
                return accs
        elif message_set is not None:
            pred_messages = pred_messages.unsqueeze(1)
            message_set = message_set.unsqueeze(0)
            pred_messages = torch.sign(pred_messages)
            pred_messages[pred_messages == -1.] = 0.
            accs = 1. - torch.sum(torch.abs(pred_messages - message_set), dim=2) / pred_messages.size(-1)
            return accs


class NewPSNR(PSNR):
    def __init__(self):
        super(NewPSNR, self).__init__()

    @torch.no_grad()
    def __call__(self, post_processed_images, images):
        post_processed_images = post_processed_images * image_std + image_mean
        post_processed_images = torch.clamp(post_processed_images, 0., 1.)

        images = images * image_std + image_mean
        images = torch.clamp(images, 0., 1.)
        delta = (post_processed_images - images) * 255.
        psnr = 20. * np.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))
        return torch.mean(psnr)


class NewSSIM(SSIM):
    def __init__(self):
        super(NewSSIM, self).__init__()

    @torch.no_grad()
    def __call__(self, img1, img2):
        img1 = img1 * image_std + image_mean
        img1 = torch.clamp(img1, 0., 1.)
        img2 = img2 * image_std + image_mean
        img2 = torch.clamp(img2, 0., 1.)

        (_, channel, _, _) = img1.size()
        window = create_window(self.window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# load backbone
model_path = "./Datasets/coco10k/ssl/models/dino_r50_plus.pth"
backbone = torchvision.models.resnet50(pretrained=True)
backbone.head = torch.nn.Identity()
backbone.fc = torch.nn.Identity()
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint
for ckpt_key in ["state_dict", "model_state_dict", "teacher"]:
    if ckpt_key in checkpoint:
        state_dict = checkpoint[ckpt_key]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
backbone.load_state_dict(state_dict, strict=False)
backbone = backbone.to(device, non_blocking=True)

# load norm layer
norm_layer_path = "./Datasets/coco10k/ssl/normlayers/out2048_coco_resized.pth"
checkpoint = torch.load(norm_layer_path, map_location=device)
D = checkpoint["weight"].shape[1]
weight = torch.nn.Parameter(D * checkpoint["weight"])
bias = torch.nn.Parameter(D * checkpoint["bias"])
dim_out, dim_in = weight.shape
norm_layer = torch.nn.Linear(dim_in, dim_out)
norm_layer.weight = torch.nn.Parameter(weight)
norm_layer.bias = torch.nn.Parameter(bias)
norm_layer = norm_layer.to(device, non_blocking=True)

# complete the model
model = NormLayerWrapper(backbone, norm_layer)
for p in model.parameters():
    p.requires_grad = False
model.eval()

# load carriers
carrier_path = "./coco10k/ssl/carriers/carrier_30_2048.pth"
carriers = torch.load(carrier_path, map_location=device)


def test(logger, method, a=1.0, sigma=0.05, sigma_blur=1.0, Q=50, num_total_users=10, num_tests=10):

    noiser = Noiser(
        noises=method, a=(a, a), sigma=(sigma, sigma), sigma_blur=(sigma_blur, sigma_blur),
        Q=(Q, Q), decoder=None, message_length=None, img_size=None
    )

    # metric
    bitacc = NewBitwiseAccuracy()
    psnr = NewPSNR()
    ssim = NewSSIM()
    attack_time = AttackTime()

    # load gt messages
    all_messages = torch.from_numpy(np.load("./Datasets/imagenet10k/test/watermarks.npy")).float()
    all_messages = all_messages[:, :30]
    all_messages = all_messages.to(device, non_blocking=True)

    # load original images
    org_image_folder_path = "./Datasets/imagenet10k/test_resized"
    org_dataset = MyImageFolder(org_image_folder_path, transforms)
    org_dataloader = torch.utils.data.DataLoader(
        dataset=org_dataset, batch_size=16, shuffle=False,
        num_workers=4 * torch.cuda.device_count(), pin_memory=True
    )

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

        # process original images
        for i, (org_images, _, paths) in enumerate(org_dataloader):
            org_images = org_images.to(device, non_blocking=True)

            # decode original image
            ft = model(org_images)
            org_messages = ft @ carriers.T

            accs = bitacc(org_messages, message_set=message_set)
            accs_org, max_indices = torch.max(accs, dim=1)
            pred_messages_org = message_set[max_indices]
            accs_dict["org"].append(accs_org)

            org_image_list.append(org_images)
        org_image_list = torch.cat(org_image_list, dim=0)

        # process watermarked images
        # load images
        watermarked_image_folder_path = f"./Datasets/imagenet10k/ssl_watermarked_jpeg/test_n{num_total_users}/c{test_idx + 1}"
        wm_dataset = MyImageFolder(watermarked_image_folder_path, transforms)
        wm_dataloader = torch.utils.data.DataLoader(
            dataset=wm_dataset, batch_size=1, shuffle=False,
            num_workers=4 * torch.cuda.device_count(), pin_memory=False
        )

        counter = 0
        with tqdm.tqdm(wm_dataloader) as tq:
            for i, (watermarked_images, _, paths) in enumerate(tq):
                watermarked_images = watermarked_images.to(device, non_blocking=True)
                gt_messages = messages[counter: counter + watermarked_images.size(0)]
                counter = counter + watermarked_images.size(0)

                # decode watermarked image
                ft = model(watermarked_images)
                watermarked_messages = ft @ carriers.T

                # generate adv images
                watermarked_images = watermarked_images * image_std + image_mean
                attack_start_time = int(round(time.time() * 1000))
                adv_images = noiser(watermarked_images)
                adv_images = torch.clamp(adv_images, 0., 1.)  # [0, 1]
                attack_end_time = int(round(time.time() * 1000))

                # save adv images
                img_names = [os.path.split(path)[-1] for path in paths]
                img_names = [name.split(".")[0] for name in img_names]
                for adv_img, img_name in zip(adv_images, img_names):
                    img_path = os.path.join(f"./Datasets/imagenet10k/image_examples/ssl/jpeg", f"{img_name}.pdf")
                    torchvision.utils.save_image(adv_img, img_path)

                adv_images = (adv_images - image_mean) / image_std
                watermarked_images = (watermarked_images - image_mean) / image_std

                # decode adv image
                ft = model(adv_images)
                adv_messages = ft @ carriers.T

                metric_meters["acc_wm_gt"].update(bitacc(watermarked_messages, gt_messages=gt_messages, mean=True), watermarked_images.size(0))
                metric_meters["acc_adv_gt"].update(bitacc(adv_messages, gt_messages=gt_messages, mean=True), watermarked_images.size(0))
                metric_meters["psnr"].update(psnr(adv_images, org_image_list[i].unsqueeze(0)), adv_images.size(0))
                metric_meters["ssim"].update(ssim(adv_images, org_image_list[i].unsqueeze(0)), adv_images.size(0))
                metric_meters["attack_time"].update(attack_time(attack_start_time, attack_end_time), watermarked_images.size(0))

                tq.set_postfix(
                    {
                        "Bit acc": f"{metric_meters['acc_adv_gt'].avg.item():.3f}",
                        "PSNR": f"{metric_meters['psnr'].avg.item():.3f}",
                        "SSIM": f"{metric_meters['ssim'].avg.item():.3f}",
                        "Attack time": f"{metric_meters['attack_time'].avg.item():.3f}",
                    },
                    refresh=True
                )

                accs = bitacc(watermarked_messages, message_set=message_set)
                accs_wm, max_indices = torch.max(accs, dim=1)
                pred_messages_wm = message_set[max_indices]
                accs_dict["wm"].append(accs_wm)
                pred_messages_dict["wm"].append(pred_messages_wm)

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

    for metric_name, meter in eval_metric_meters.items():
        logger(f"Averaged test {metric_name}: {meter.avg.item():.3f}")
    drop_n4 = eval_metric_meters["identify_acc_n4"].avg.item() - eval_metric_meters[
        "identify_acc_n4_adv"].avg.item()
    drop_n6 = eval_metric_meters["identify_acc_n6"].avg.item() - eval_metric_meters[
        "identify_acc_n6_adv"].avg.item()
    logger(f"Averaged identify_acc_drop_n4: {drop_n4:.3f}")
    logger(f"Averaged identify_acc_drop_n6: {drop_n6:.3f}")
