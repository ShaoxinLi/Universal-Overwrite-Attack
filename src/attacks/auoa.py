#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import copy
import torch
import torch.profiler
import torch.nn.functional as F
from collections import defaultdict
from ..metrics import BitwiseAccuracy
from ..utils import seed_everything, AverageMeter, Callbacks, Callback, CheckpointIO
from ..utils.plot import plot_bar


class AUOA(object):
    def __init__(self, device, seed=None):
        super(AUOA, self).__init__()
        self.device = device
        self.seed = seed

    def train(self, k, snn, decoder, loader, img_size, message_length, num_epochs, lr, xi, callbacks):

        self.k = k
        self.snn = snn.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.decoder.eval()
        self.p = torch.zeros(self.k, 3, img_size, img_size, device=self.device, dtype=torch.float32)
        self.p.requires_grad_(True)
        self.tau = 1.0
        self.num_iters = 1.0

        self.optimizer_p = torch.optim.AdamW([self.p], lr=lr)
        self.optimizer_snn = torch.optim.AdamW(list(self.snn.parameters()), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device, non_blocking=True)

        self.message_length = message_length
        self.xi = xi / 255.
        seed_everything(self.seed)

        # callbacks
        self.state = RunnerState()
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)

        # start training
        self.callbacks.on_start()
        for epoch in range(num_epochs):
            self.state.epoch = epoch
            self.callbacks.on_epoch_start()

            # train one epoch
            self._train_epoch(loader)

            # update state
            self.state.state_dict = {
                "p": self.p.data.detach(),
                "snn": self.snn.state_dict(),
                "optimizer_p": self.optimizer_p,
                "optimizer_snn": self.optimizer_snn
            }
            self.callbacks.on_epoch_end()

            # check early stopping
            if self.state.early_stop:
                break

        self.callbacks.on_end()

    def _train_epoch(self, loader):

        self.state.is_train = True
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        for i, (images, messages) in enumerate(loader):
            self.state.iteration = i
            self.state.num_samples_in_batch = images.size(0)

            images = images.to(self.device, non_blocking=True)
            messages = messages.to(self.device, non_blocking=True)

            self.callbacks.on_batch_start()
            if i % 2 == 0:
                self._train_step_p(images, messages)
            else:
                self._train_step_snn(images, messages)

            self.callbacks.on_batch_end()
        self.callbacks.on_loader_end()

    def _train_step_p(self, images, messages):

        self.optimizer_p.zero_grad(set_to_none=True)
        self.optimizer_snn.zero_grad(set_to_none=True)

        logits = self.snn(images)
        g_logits = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        g_uoas = []
        for i in g_logits:
            g_uoa = i * self.p
            g_uoa = torch.sum(g_uoa, dim=0)
            g_uoas.append(g_uoa)
        g_uoas = torch.stack(g_uoas)

        # generate adv images
        adv_images = images + g_uoas
        adv_images = torch.clamp(adv_images, 0., 1.)

        # compute loss and backward
        adv_messages = self.decoder(adv_images)
        loss = -self.criterion(adv_messages, messages)

        loss.backward()
        self.callbacks.on_after_backward()
        self.optimizer_p.step()

        # project
        self.p.data = self.p.data * min(1, self.xi / (torch.linalg.norm(self.p.data, dim=0)))

        # update states
        self.state.losses = {"loss": loss.detach()}

        self.state.gt_messages = messages
        self.state.adv_messages = adv_messages.detach()

        self.state.images = images
        self.state.adv_images = adv_images.detach()

    def _train_step_snn(self, images, messages):

        self.optimizer_p.zero_grad(set_to_none=True)
        self.optimizer_snn.zero_grad(set_to_none=True)

        logits = self.snn(images)
        g_logits = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        g_uoas = []
        for i in g_logits:
            g_uoa = i * self.p
            g_uoa = torch.sum(g_uoa, dim=0)
            g_uoas.append(g_uoa)
        g_uoas = torch.stack(g_uoas)

        # generate adv images
        adv_images = images + g_uoas
        adv_images = torch.clamp(adv_images, 0., 1.)

        # compute loss and backward
        adv_messages = self.decoder(adv_images)
        loss = -self.criterion(adv_messages, messages)

        loss.backward()
        self.callbacks.on_after_backward()
        self.optimizer_snn.step()

        # update states
        self.state.losses = {"loss": loss.detach()}
        if self.num_iters % 2000 == 0:
            self.tau = torch.max(torch.cat([torch.exp(torch.tensor(0.0001, device=self.device) * self.num_iters), torch.tensor(0.1, device=self.device)]))
        self.num_iters += 1

    def _eval_epoch(self, loader):

        self.state.is_train = False
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        p = copy.deepcopy(self.p.detach())
        snn = copy.deepcopy(self.snn.detach())
        for i, (images, messages) in enumerate(loader):
            self.state.iteration = i
            self.state.image_idx = i
            self.state.num_samples_in_batch = images.size(0)

            images = images.to(self.device, non_blocking=True)
            messages = messages.to(self.device, non_blocking=True)

            self.callbacks.on_batch_start()
            self._eval_step(images, messages, p, snn)
            self.callbacks.on_batch_end()

        self.callbacks.on_loader_end()

    @torch.no_grad()
    def _eval_step(self, images, messages, p, snn):

        # generate adv images
        attack_start_time = int(round(time.time()))
        logits = snn(images)
        idxs = torch.argmax(logits, dim=1)
        uoas = []
        for idx in idxs:
            uoa = p[idx]
            uoas.append(uoa)
        uoas = torch.stack(uoas)
        adv_images = images + uoas
        adv_images = torch.clamp(adv_images, 0., 1.)
        attack_end_time = int(round(time.time()))

        # compute loss
        adv_messages = self.decoder(adv_images)
        loss = -self.criterion(adv_messages, messages)

        # update states
        self.state.losses = {"loss": loss}

        self.state.gt_messages = messages
        self.state.adv_messages = adv_messages

        self.state.images = images
        self.state.adv_images = adv_images

        self.state.attack_start_time = attack_start_time
        self.state.attack_end_time = attack_end_time

    def test(self, snn, decoder, loader, xi, ckpt_file_path, callbacks):

        self.snn = snn.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.p = None
        self.decoder.eval()

        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device, non_blocking=True)
        self.xi = xi / 255.

        assert ckpt_file_path
        ckptio = CheckpointIO(
            ckpt_file_path=ckpt_file_path, device=self.device, p=self.p, snn=self.snn
        )
        self.p = ckptio.load()["p"]
        self.snn = ckptio.load()

        # callbacks
        self.state = RunnerState()
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)

        # start testing
        self.callbacks.on_start()
        self.callbacks.on_epoch_start()
        self._eval_epoch(loader)
        self.callbacks.on_epoch_end()
        self.callbacks.on_end()


class RunnerState(object):
    def __init__(self):
        super(RunnerState, self).__init__()
        self.epoch = 0
        self.state_dict = {}
        self.early_stop = False

        self.num_batches = 0
        self.is_train = True
        self.train_loss_meters = defaultdict(AverageMeter)
        self.train_metric_meters = defaultdict(AverageMeter)
        self.eval_loss_meters = defaultdict(AverageMeter)
        self.eval_metric_meters = defaultdict(AverageMeter)

        self.iteration = 0
        self.num_samples_in_batch = 0
        self.losses = {}
        self.loss_meters = defaultdict(AverageMeter)
        self.metric_meters = defaultdict(AverageMeter)

        self.gt_messages = None
        self.adv_messages = None

        self.p = None
        self.images = None
        self.adv_images = None

        self.image_idx = 0

        self.attack_start_time = None
        self.attack_end_time = None


class ScalarTracker(Callback):
    def __init__(self, metric_fn_dict):
        super(ScalarTracker, self).__init__()
        self.metric_fn_dict = metric_fn_dict

    def on_loader_start(self):
        for meter in self.state.loss_meters.values():
            meter.reset()
        for meter in self.state.metric_meters.values():
            meter.reset()

    @torch.no_grad()
    def on_batch_end(self):
        for loss_name, loss in self.state.losses.items():
            self.state.loss_meters[loss_name].update(loss, n=self.state.num_samples_in_batch)
        for metric_name, metric_fn in self.metric_fn_dict.items():
            if metric_name == "bitwise_acc":
                metric_value = metric_fn(self.state.adv_messages, gt_messages=self.state.gt_messages, mean=True)
                self.state.metric_meters["acc_adv_gt"].update(metric_value, self.state.num_samples_in_batch)
            elif metric_name == "perturbation_norm":
                metric_value = metric_fn(self.state.adv_images, self.state.watermarked_images)
                self.state.metric_meters[metric_name].update(metric_value, self.state.num_samples_in_batch)
            elif metric_name == "attack_time":  # for testing
                metric_value = metric_fn(self.state.attack_start_time, self.state.attack_end_time)
                self.state.metric_meters[metric_name].update(metric_value, 1)
            elif metric_name == "psnr":
                metric_value = metric_fn(self.state.adv_images, self.state.images)
                self.state.metric_meters["psnr"].update(metric_value, self.state.num_samples_in_batch)
            elif metric_name == "ssim":
                metric_value = metric_fn(self.state.adv_images, self.state.images)
                self.state.metric_meters["ssim"].update(metric_value, self.state.num_samples_in_batch)
            else:
                assert False

    def on_loader_end(self):
        if self.state.is_train:
            self.state.train_loss_meters = copy.deepcopy(self.state.loss_meters)
            self.state.train_metric_meters = copy.deepcopy(self.state.metric_meters)
        else:
            for loss_name, meter in self.state.loss_meters.items():
                self.state.eval_loss_meters[loss_name].update(meter.avg, meter.count)
            for metric_name, meter in self.state.metric_meters.items():
                self.state.eval_metric_meters[metric_name].update(meter.avg, meter.count)


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
        self.adv_messages = []

    @torch.no_grad()
    def on_batch_end(self):
        if not self.state.is_train:
            accs = self.acc(self.state.org_messages, message_set=self.state.message_set)
            accs_org, max_indices = torch.max(accs, dim=1)
            pred_messages_org = self.state.message_set[max_indices]
            self.accs_dict["org"].append(accs_org)

            accs = self.acc(self.state.watermarked_messages, message_set=self.state.message_set)
            accs_wm, max_indices = torch.max(accs, dim=1)
            pred_messages_wm = self.state.message_set[max_indices]
            self.accs_dict["wm"].append(accs_wm)
            self.pred_messages_dict["wm"].append(pred_messages_wm)

            accs = self.acc(self.state.adv_messages, message_set=self.state.message_set)
            accs_adv, max_indices = torch.max(accs, dim=1)
            pred_messages_adv = self.state.message_set[max_indices]
            self.accs_dict["adv"].append(accs_adv)
            self.pred_messages_dict["adv"].append(pred_messages_adv)

            self.gt_messages.append(self.state.gt_messages)
            self.adv_messages.append(torch.clamp(torch.round(self.state.adv_messages), 0, 1))

    @torch.no_grad()
    def on_epoch_end(self):
        if not self.state.is_train:
            accs_org = torch.cat(self.accs_dict["org"], dim=0)
            accs_wm = torch.cat(self.accs_dict["wm"], dim=0)
            accs_adv = torch.cat(self.accs_dict["adv"], dim=0)
            pred_messages_wm = torch.cat(self.pred_messages_dict["wm"], dim=0)
            pred_messages_adv = torch.cat(self.pred_messages_dict["adv"], dim=0)
            gt_messages = torch.cat(self.gt_messages, dim=0)
            adv_messages = torch.cat(self.adv_messages, dim=0)

            self.writer.add_histogram("Test Bitwise Acc Distribution/org", accs_org, self.state.test_idx + 1)
            self.writer.add_histogram("Test Bitwise Acc Distribution/wm", accs_wm, self.state.test_idx + 1)
            self.writer.add_histogram("Test Bitwise Acc Distribution/adv", accs_adv, self.state.test_idx + 1)

            adv_message_mean = torch.mean(adv_messages, dim=0).cpu().numpy()
            adv_message_std = torch.std(adv_messages, dim=0).cpu().numpy()
            fig = plot_bar(adv_message_mean, adv_message_std)
            self.writer.add_figure("Test Message Distribution/adv", fig, self.state.test_idx + 1)

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
        for metric_name, meter in self.state.eval_metric_meters.items():
            self.logger(f"{self.state.test_idx + 1}-th testing {metric_name}: {meter.val.item()}")


class TensorBoardProfiler(Callback):
    def __init__(self, writer):
        super(TensorBoardProfiler, self).__init__()
        log_dir = writer.log_dir
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=20, repeat=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

    def on_loader_start(self):
        if self.state.is_train and self.state.epoch + 1 == 5:
            self.prof.start()

    def on_batch_end(self):
        if self.state.is_train and self.state.epoch + 1 == 5:
            self.prof.step()

    def on_loader_end(self):
        if self.state.is_train and self.state.epoch + 1 == 5:
            self.prof.stop()


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
        for metric_name, meter in self.state.eval_metric_meters.items():
            metric_dict[f"hparam/{metric_name}"] = meter.avg.item()
        metric_dict[f"hparam/identify_acc_drop_n4"] = self.state.eval_metric_meters["identify_acc_n4"].avg.item() - \
                                                      self.state.eval_metric_meters["identify_acc_n4_adv"].avg.item()
        metric_dict[f"hparam/identify_acc_drop_n6"] = self.state.eval_metric_meters["identify_acc_n6"].avg.item() - \
                                                      self.state.eval_metric_meters["identify_acc_n6_adv"].avg.item()
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)
