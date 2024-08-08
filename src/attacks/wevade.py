#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import tqdm
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from collections import defaultdict
from ..metrics import BitwiseAccuracy
from ..utils import seed_everything, AverageMeter, Callbacks, Callback, check_dir


class WEvade(object):
    def __init__(self, device, seed=None):
        super(WEvade, self).__init__()
        self.device = device
        self.seed = seed

    def attack(self, encoder, decoder, loader, gt_messages, num_eval_users, num_total_users, num_tests,
               max_iter, lr, xi, epsilon, callbacks):

        self.encoder = encoder.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.gt_messages = gt_messages.to(self.device, non_blocking=True)
        self.encoder.eval()
        self.decoder.eval()

        xi = xi / 255.
        self.criterion = torch.nn.MSELoss().to(self.device, non_blocking=True)
        seed_everything(self.seed)

        # callbacks
        self.state = RunnerState()
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)

        num_eval_users = self.gt_messages.size(0) if num_eval_users == -1 else num_eval_users
        num_total_users = self.gt_messages.size(0) if num_total_users == -1 else num_total_users
        assert 1 <= num_eval_users <= num_total_users

        for i in range(num_tests):

            self.state.test_idx = i
            self.callbacks.on_start()
            self.callbacks.on_epoch_start()
            self.callbacks.on_loader_start()

            self.state.message_set = self.gt_messages[self.state.test_idx * num_total_users: (self.state.test_idx + 1) * num_total_users]
            message_set_ = self.state.message_set[:num_eval_users]
            n, m = divmod(self.gt_messages.size(0), message_set_.size(0))
            gt_messages = message_set_.repeat(n, 1)
            gt_messages = torch.cat([gt_messages, message_set_[:m]], dim=0) if m != 0 else gt_messages

            with tqdm.tqdm(loader) as tq:
                for j, (image, _) in enumerate(tq):
                    self.state.image_idx = j

                    image = image.to(self.device, non_blocking=True)
                    message = gt_messages[j].unsqueeze(0)
                    self.callbacks.on_batch_start()

                    # encode
                    watermarked_image = self.encoder(image, message).detach()
                    watermarked_image = watermarked_image.clamp_(0., 1.)

                    attack_start_time = int(round(time.time() * 1000))
                    self.wevade_w(image, watermarked_image, message, max_iter, lr, xi, epsilon)
                    attack_end_time = int(round(time.time() * 1000))

                    self.state.attack_start_time = attack_start_time
                    self.state.attack_end_time = attack_end_time

                    self.callbacks.on_batch_end()
                    tq.set_postfix(
                        {
                            "Bit acc": f"{self.state.metric_meters['acc_adv_gt'].avg.item():.3f}",
                            "PSNR": f"{self.state.metric_meters['psnr'].avg.item():.3f}",
                            "SSIM": f"{self.state.metric_meters['ssim'].avg.item():.3f}",
                            "Attack time": f"{self.state.metric_meters['attack_time'].avg.item():.3f}",
                        },
                        refresh=True
                    )

            self.callbacks.on_loader_end()
            self.callbacks.on_epoch_end()
            self.callbacks.on_end()

    def wevade_w(self, image, adv_image, message, max_iter, lr, xi, epsilon):

        watermarked_image = adv_image.clone()

        # decode original image
        org_message = self.decoder(image)

        # decode watermarked image
        watermarked_message = self.decoder(watermarked_image)

        # set the target message
        target_message = torch.tensor(np.random.choice([0., 1.], (1, message.size(1))), device=self.device, dtype=torch.float32)

        for k in range(max_iter):
            adv_image = adv_image.requires_grad_(True)
            min_value, max_value = torch.min(adv_image), torch.max(adv_image)

            # decode adv image
            adv_message = self.decoder(adv_image)

            # update adv image
            loss = self.criterion(adv_message, target_message)
            grads = torch.autograd.grad(loss, adv_image)
            with torch.no_grad():
                adv_image = adv_image - lr * grads[0]
                adv_image = torch.clamp(adv_image, min_value, max_value)

            # projection adv encoded image
            perturbation_norm = torch.norm(adv_image - watermarked_image, float("inf"))
            if perturbation_norm > xi:
                c = xi / perturbation_norm
                adv_image = self.project(adv_image, watermarked_image, c)

            adv_image = torch.clamp(adv_image, 0., 1.)

            # check if the updated adv image meets early stopping criterion
            adv_message = torch.clamp(torch.round(self.decoder(adv_image)), 0, 1)
            acc_adv_target = 1. - torch.sum(torch.abs(adv_message - target_message)) / message.size(1)
            if acc_adv_target >= 1 - epsilon:
                break

        # update state
        self.state.gt_message = message
        self.state.org_message = org_message
        self.state.watermarked_message = watermarked_message
        self.state.adv_message = adv_message.detach()
        self.state.target_message = target_message

        self.state.image = image
        self.state.watermarked_image = watermarked_image
        self.state.adv_image = adv_image.detach()
        self.state.perturbation = ((adv_image.detach() - watermarked_image) / xi + 1.) / 2.

    def wevade_w_binary_search(self, image, adv_image, message, max_iter, lr, xi, epsilon):

        watermarked_image = adv_image.clone()

        # decode original image
        org_message = self.decoder(image)

        # decode encoded image
        watermarked_message = self.decoder(watermarked_image)

        # set the target message
        target_message = torch.tensor(np.random.choice([0., 1.], (1, message.size(1))), device=self.device, dtype=torch.float32)

        xi_upper, xi_lower = xi, 0
        while (xi_upper - xi_lower) >= 0.001:
            xi = (xi_lower + xi_upper) / 2.
            success = False

            for k in range(max_iter):
                adv_image = adv_image.requires_grad_(True)
                min_value, max_value = torch.min(adv_image), torch.max(adv_image)

                # decode adv image
                adv_message = self.decoder(adv_image)

                # update adv image
                loss = self.criterion(adv_message, target_message)
                grads = torch.autograd.grad(loss, adv_image)
                with torch.no_grad():
                    adv_image = adv_image - lr * grads[0]
                    adv_image = torch.clamp(adv_image, min_value, max_value)

                # projection adv encoded image
                perturbation_norm = torch.norm(adv_image - watermarked_image, float("inf"))
                if perturbation_norm > xi:
                    c = xi / perturbation_norm
                    adv_image = self.project(adv_image, watermarked_image, c)

                adv_image = torch.clamp(adv_image, 0., 1.)

                # check if the updated adv image meets early stopping criterion
                adv_message = torch.clamp(torch.round(self.decoder(adv_image)), 0, 1)
                acc_adv_target = 1. - torch.sum(torch.abs(adv_message - target_message)) / message.size(1)
                if acc_adv_target >= 1 - epsilon:
                    success = True
                    break

            if success:
                xi_upper = xi
            else:
                xi_lower = xi

        # update state
        self.state.gt_message = message
        self.state.org_message = org_message
        self.state.watermarked_message = watermarked_message
        self.state.adv_message = adv_message.detach()
        self.state.target_message = target_message

        self.state.image = image
        self.state.watermarked_image = watermarked_image
        self.state.adv_image = adv_image.detach()
        self.state.perturbation = ((adv_image.detach() - watermarked_image) / xi + 1.) / 2.

    @staticmethod
    def project(image, orig_image, epsilon):
        # If the perturbation exceeds the upper bound, project it back.
        delta = image - orig_image
        delta = epsilon * delta
        return orig_image + delta


class RunnerState(object):
    def __init__(self):
        super(RunnerState, self).__init__()

        self.image_idx = 0
        self.metric_meters = defaultdict(AverageMeter)
        self.eval_metric_meters = defaultdict(AverageMeter)

        self.image = None
        self.watermarked_image = None
        self.adv_image = None
        self.perturbation = None

        self.attack_start_time = None
        self.attack_end_time = None

        self.gt_message = None
        self.org_message = None
        self.watermarked_message = None
        self.adv_message = None
        self.target_message = None
        self.message_set = None

        self.test_idx = 0


class ScalarTracker(Callback):
    def __init__(self, metric_fn_dict):
        super(ScalarTracker, self).__init__()
        self.metric_fn_dict = metric_fn_dict

    def on_loader_start(self):
        for metric_name, meter in self.state.metric_meters.items():
            meter.reset()

    @torch.no_grad()
    def on_batch_end(self):
        for metric_name, metric_fn in self.metric_fn_dict.items():
            if metric_name == "bitwise_acc":
                metric_value = metric_fn(self.state.adv_message, gt_messages=self.state.gt_message, mean=True)
                self.state.metric_meters["acc_adv_gt"].update(metric_value, 1)
            elif metric_name == "attack_time":
                metric_value = metric_fn(self.state.attack_start_time, self.state.attack_end_time)
                self.state.metric_meters[metric_name].update(metric_value, 1)
            elif metric_name == "psnr":
                metric_value = metric_fn(self.state.adv_image, self.state.image)
                self.state.metric_meters["psnr"].update(metric_value, 1)
            elif metric_name == "ssim":
                metric_value = metric_fn(self.state.adv_image, self.state.image)
                self.state.metric_meters["ssim"].update(metric_value, 1)
            else:
                assert False

    def on_loader_end(self):
        for metric_name, meter in self.state.metric_meters.items():
            self.state.eval_metric_meters[metric_name].update(meter.avg, meter.count)


class TensorBoardImageLogger(Callback):
    def __init__(self, writer, resize_to):
        super(TensorBoardImageLogger, self).__init__()
        self.writer = writer
        self.resize_to = resize_to

    @torch.no_grad()
    def on_batch_end(self):
        image = self.state.image
        watermarked_image = self.state.watermarked_image
        adv_image = self.state.adv_image
        if self.resize_to is not None:
            image = F.interpolate(image, size=self.resize_to)
            watermarked_image = F.interpolate(watermarked_image, size=self.resize_to)
            adv_image = F.interpolate(adv_image, size=self.resize_to)

        all_images = torch.cat([image, watermarked_image, adv_image], dim=0)
        grid = torchvision.utils.make_grid(all_images, nrow=all_images.size(0))
        self.writer.add_image("Images (Original v.s. Watermarked v.s. Adv)", grid, self.state.image_idx + 1)


class TensorBoardStatLogger(Callback):
    def __init__(self, writer, message_length, num_users):
        super(TensorBoardStatLogger, self).__init__()
        self.writer = writer
        self.acc = BitwiseAccuracy()

        assert message_length in [30, 256]
        assert num_users in [1, 10, 100, 1000]
        if message_length == 30:
            if num_users == 1:
                self.tau_n4 = 25 / 30.
                self.tau_n6 = 27 / 30.
            elif num_users == 10:
                self.tau_n4 = 26. / 30.
                self.tau_n6 = 28. / 30.
            elif num_users == 100:
                self.tau_n4 = 27. / 30.
                self.tau_n6 = 29. / 30.
            elif num_users == 1000:
                self.tau_n4 = 28. / 30.
                self.tau_n6 = 29. / 30.
            else:
                assert False
        elif message_length == 256:
            if num_users == 1:
                self.tau_n4 = 158 / 256.
                self.tau_n6 = 166 / 256.
            elif num_users == 10:
                self.tau_n4 = 162. / 256.
                self.tau_n6 = 169. / 256.
            elif num_users == 100:
                self.tau_n4 = 166. / 256.
                self.tau_n6 = 172. / 256.
            elif num_users == 1000:
                self.tau_n4 = 169. / 256.
                self.tau_n6 = 175. / 256.
            else:
                assert False

    def on_epoch_start(self):
        self.accs_dict = defaultdict(list)
        self.pred_messages_dict = defaultdict(list)
        self.gt_messages = []

    @torch.no_grad()
    def on_batch_end(self):
        acc = self.acc(self.state.org_message, message_set=self.state.message_set)
        acc_org, max_index = torch.max(acc, dim=1)
        pred_message_org = self.state.message_set[max_index]
        self.accs_dict["org"].append(acc_org)

        acc = self.acc(self.state.watermarked_message, message_set=self.state.message_set)
        acc_wm, max_index = torch.max(acc, dim=1)
        pred_message_wm = self.state.message_set[max_index]
        self.accs_dict["wm"].append(acc_wm)
        self.pred_messages_dict["wm"].append(pred_message_wm)

        acc = self.acc(self.state.adv_message, message_set=self.state.message_set)
        acc_adv, max_index = torch.max(acc, dim=1)
        pred_message_adv = self.state.message_set[max_index]
        self.accs_dict["adv"].append(acc_adv)
        self.pred_messages_dict["adv"].append(pred_message_adv)

        self.gt_messages.append(self.state.gt_message)

    @torch.no_grad()
    def on_epoch_end(self):
        accs_org = torch.cat(self.accs_dict["org"], dim=0)
        accs_wm = torch.cat(self.accs_dict["wm"], dim=0)
        accs_adv = torch.cat(self.accs_dict["adv"], dim=0)
        pred_messages_wm = torch.cat(self.pred_messages_dict["wm"], dim=0)
        pred_messages_adv = torch.cat(self.pred_messages_dict["adv"], dim=0)
        gt_messages = torch.cat(self.gt_messages, dim=0)

        # unattacked
        probs = torch.cat([accs_org, accs_wm], dim=0)
        labels = torch.cat([
            torch.zeros(accs_org.size(0), device=probs.device),
            torch.ones(accs_wm.size(0), device=probs.device)
        ], dim=0)
        for des, tau in [("n4", self.tau_n4), ("n6", self.tau_n6)]:
            predictions = torch.where(probs >= tau, 1, 0)
            fp = torch.sum((predictions == 1) & (labels == 0))
            tp = torch.sum((predictions == 1) & (labels == 1))
            fn = torch.sum((predictions == 0) & (labels == 1))
            tn = torch.sum((predictions == 0) & (labels == 0))
            tp_indices = predictions[labels == 1] == 1
            tidt = torch.sum(torch.all(torch.eq(pred_messages_wm[tp_indices], gt_messages[tp_indices]), dim=1))

            self.state.eval_metric_meters[f"far_{des}"].update(1. - tidt / tp, 1)
            self.state.eval_metric_meters[f"fnr_{des}"].update(fn / (tp + fn), 1)
            self.state.eval_metric_meters[f"identify_acc_{des}"].update(tidt / (tp + fn), 1)

        # attacked
        probs = torch.cat([accs_org, accs_adv], dim=0)
        labels = torch.cat([
            torch.zeros(accs_org.size(0), device=probs.device),
            torch.ones(accs_adv.size(0), device=probs.device)
        ], dim=0)
        for des, tau in [("n4", self.tau_n4), ("n6", self.tau_n6)]:
            predictions = torch.where(probs >= tau, 1, 0)
            fp = torch.sum((predictions == 1) & (labels == 0))
            tp = torch.sum((predictions == 1) & (labels == 1))
            fn = torch.sum((predictions == 0) & (labels == 1))
            tn = torch.sum((predictions == 0) & (labels == 0))
            tp_indices = predictions[labels == 1] == 1
            tidt = torch.sum(torch.all(torch.eq(pred_messages_adv[tp_indices], gt_messages[tp_indices]), dim=1))

            self.state.eval_metric_meters[f"far_{des}_adv"].update(1. - tidt / tp, 1)
            self.state.eval_metric_meters[f"fnr_{des}_adv"].update(fn / (tp + fn), 1)
            self.state.eval_metric_meters[f"identify_acc_{des}_adv"].update(tidt / (tp + fn), 1)


class TestLogger(Callback):
    def __init__(self, logger):
        super(TestLogger, self).__init__()
        self.logger = logger

    def on_epoch_end(self):
        self.logger(
            f"{self.state.test_idx + 1}-th testing ground truth message set: {self.state.message_set.tolist()}"
        )
        for metric_name, meter in self.state.metric_meters.items():
            self.logger(f"{self.state.test_idx + 1}-th testing {metric_name}: {meter.val.item()}")


class TensorBoardHparamsLogger(Callback):
    def __init__(self, writer, args):
        super(TensorBoardHparamsLogger, self).__init__()
        self.writer = writer
        self.args = args

    def on_end(self):
        self.args = {k: v for k, v in sorted(self.args.items(), key=lambda x: x[0])}
        hparams = {}
        for k, v in self.args.items():
            if isinstance(v, (float, int)):
                hparams[k] = v
            elif isinstance(v, str) and k in ["watermarking_model"]:
                hparams[k] = v
        metric_dict = {}
        for metric_name, meter in self.state.metric_meters.items():
            metric_dict[f"hparam/{metric_name}"] = meter.avg.item()
        metric_dict[f"hparam/identify_acc_drop_n4"] = self.state.eval_metric_meters["identify_acc_n4"].avg.item() - \
                                                      self.state.eval_metric_meters["identify_acc_n4_adv"].avg.item()
        metric_dict[f"hparam/identify_acc_drop_n6"] = self.state.eval_metric_meters["identify_acc_n6"].avg.item() - \
                                                      self.state.eval_metric_meters["identify_acc_n6_adv"].avg.item()
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)


class PerturbationSaver(Callback):
    def __init__(self, exp_dir_path, exp_datatime):
        super(PerturbationSaver, self).__init__()
        self.perturbation_dir_path = os.path.join(exp_dir_path, "perturbations", f"run-{exp_datatime}")
        check_dir(self.perturbation_dir_path)

    def on_epoch_start(self):
        self.perturbations = []

    def on_batch_end(self):
        self.perturbations.append(self.state.adv_image - self.state.watermarked_image)

    def on_epoch_end(self):
        self.perturbations = torch.cat(self.perturbations, dim=0)
        self.perturbations = self.perturbations.cpu().numpy()
        save_path = os.path.join(self.perturbation_dir_path, f"test{self.state.test_idx + 1:03d}.npy")
        np.save(save_path, self.perturbations)
