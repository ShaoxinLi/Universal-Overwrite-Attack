#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import torch
import torchvision
import torch.profiler
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from ..metrics import BitwiseAccuracy
from ..utils import seed_everything, Callbacks, AverageMeter, CheckpointIO, Callback


class HiDDeN(object):
    def __init__(self, device, seed=None):
        super(HiDDeN, self).__init__()
        self.device = device
        self.seed = seed

    def train(self, encoder, decoder, noiser, discriminator, loader,
              message_length, num_epochs, lr, callbacks):

        # init optimizers
        self.encoder = encoder.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.noiser = noiser.to(self.device, non_blocking=True)
        self.discriminator = discriminator.to(self.device, non_blocking=True)
        self.optimizer_enc_dec = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr
        )
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # define losses
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss().to(self.device, non_blocking=True)
        self.mse_loss = torch.nn.MSELoss().to(self.device, non_blocking=True)

        self.message_length = message_length
        self.lambda_adv = 1e-3
        self.lambda_distortion = 0.7
        self.lambda_message = 1.0
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
                "encoder": self.encoder,
                "decoder": self.decoder,
                "discriminator": self.discriminator,
                "optimizer_enc_dec": self.optimizer_enc_dec,
                "optimizer_discriminator": self.optimizer_discriminator
            }
            self.callbacks.on_epoch_end()

            # check early stopping
            if self.state.early_stop:
                break

        self.callbacks.on_end()

    def _train_epoch(self, loader):

        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        self.state.is_train = True
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        for i, (images, _) in enumerate(loader):
            self.state.iteration = i
            self.state.num_samples_in_batch = images.size(0)

            images = images.to(self.device, non_blocking=True)
            gt_messages = torch.tensor(
                np.random.choice([0., 1.], (images.size(0), self.message_length)),
                device=self.device, dtype=torch.float32
            )

            self.callbacks.on_batch_start()
            self._train_step(images, gt_messages)
            self.callbacks.on_batch_end()

        self.callbacks.on_loader_end()

    def _train_step(self, images, messages):

        # train the discriminator
        self.optimizer_discriminator.zero_grad(set_to_none=True)

        d_target_label_cover = torch.full((images.size(0), 1), 1.0, device=self.device)
        d_target_label_watermarked = torch.full((images.size(0), 1), 0.0, device=self.device)

        d_on_cover = self.discriminator(images)
        d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
        d_loss_on_cover.backward()
        self.callbacks.on_after_backward()

        # watermarking
        watermarked_images = self.encoder(images, messages)     # ~ [0, 1]
        watermarked_images = watermarked_images.clamp_(0., 1.)  # [0, 1]

        # add noise
        noised_images = self.noiser(watermarked_images)

        # decode
        self.decoder.train()        # because decoder maybe set to eval mode when add noises to watermarked image
        watermarked_messages = self.decoder(noised_images)

        d_on_watermarked = self.discriminator(watermarked_images.detach())
        d_loss_on_watermarked = self.bce_with_logits_loss(d_on_watermarked, d_target_label_watermarked)
        d_loss_on_watermarked.backward()
        self.callbacks.on_after_backward()

        self.optimizer_discriminator.step()

        # train the encoder-decoder
        self.optimizer_enc_dec.zero_grad(set_to_none=True)

        d_target_label_watermarked = torch.full((images.size(0), 1), 1.0, device=self.device)
        d_on_watermarked_for_enc = self.discriminator(watermarked_images)

        g_loss_adv = self.bce_with_logits_loss(d_on_watermarked_for_enc, d_target_label_watermarked)
        g_loss_enc = self.mse_loss(watermarked_images, images)
        g_loss_dec = self.mse_loss(watermarked_messages, messages)

        g_loss = self.lambda_adv * g_loss_adv + self.lambda_distortion * g_loss_enc + self.lambda_message * g_loss_dec
        g_loss.backward()
        self.callbacks.on_after_backward()

        self.optimizer_enc_dec.step()

        # update states
        self.state.losses = {
            "g_loss": g_loss.detach(),
        }

        self.state.gt_messages = messages
        self.state.watermarked_messages = watermarked_messages.detach()

        self.state.images = images
        self.state.watermarked_images = watermarked_images.detach()
        self.state.diff_images = torch.abs(watermarked_images.detach() - images)
        self.state.noised_images = noised_images.detach()

    def _eval_epoch(self, loader, gt_messages):

        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        self.state.is_train = False
        self.state.num_batches = len(loader)
        self.callbacks.on_loader_start()

        counter = 0
        for i, (images, _) in enumerate(loader):
            self.state.iteration = i
            self.state.num_samples_in_batch = images.size(0)

            images = images.to(self.device, non_blocking=True)
            messages = gt_messages[counter: counter + images.size(0)]
            counter = counter + images.size(0)

            self.callbacks.on_batch_start()
            self._eval_step(images, messages)
            self.callbacks.on_batch_end()

        self.callbacks.on_loader_end()

    @torch.no_grad()
    def _eval_step(self, images, messages):

        d_target_label_cover = torch.full((images.size(0), 1), 1.0, device=self.device)
        d_target_label_watermarked = torch.full((images.size(0), 1), 0.0, device=self.device)

        d_on_cover = self.discriminator(images)
        d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

        # watermarking
        watermarked_images = self.encoder(images, messages)  # ~ [0, 1]
        watermarked_images = watermarked_images.clamp_(0., 1.)  # [0, 1]

        # add noise
        noised_images = self.noiser(watermarked_images)

        # decode
        watermarked_messages = self.decoder(noised_images)

        d_on_watermarked = self.discriminator(watermarked_images)
        d_loss_on_watermarked = self.bce_with_logits_loss(d_on_watermarked, d_target_label_watermarked)

        d_target_label_watermarked = torch.full((images.size(0), 1), 1.0, device=self.device)
        d_on_watermarked_for_enc = self.discriminator(watermarked_images)

        g_loss_adv = self.bce_with_logits_loss(d_on_watermarked_for_enc, d_target_label_watermarked)
        g_loss_enc = self.mse_loss(watermarked_images, images)
        g_loss_dec = self.mse_loss(watermarked_messages, messages)
        g_loss = self.lambda_adv * g_loss_adv + self.lambda_distortion * g_loss_enc + self.lambda_message * g_loss_dec

        # decode original images
        org_messages = self.decoder(images)

        # update states
        self.state.losses = {
            "g_loss": g_loss,
        }

        self.state.gt_messages = messages
        self.state.watermarked_messages = watermarked_messages
        self.state.org_messages = org_messages

        self.state.images = images
        self.state.watermarked_images = watermarked_images
        self.state.diff_images = torch.abs(watermarked_images - images)
        self.state.noised_images = noised_images

    def test(self, encoder, decoder, noiser, discriminator, loader, gt_messages,
             num_eval_users, num_total_users, ckpt_file_path, callbacks):

        self.encoder = encoder.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.noiser = noiser.to(self.device, non_blocking=True)
        self.discriminator = discriminator.to(self.device, non_blocking=True)

        # define losses
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss().to(self.device, non_blocking=True)
        self.mse_loss = torch.nn.MSELoss().to(self.device, non_blocking=True)

        self.lambda_adv = 1e-3
        self.lambda_distortion = 0.7
        self.lambda_message = 1.0

        assert ckpt_file_path
        ckptio = CheckpointIO(
            ckpt_file_path=ckpt_file_path, device=self.device,
            encoder=self.encoder, decoder=self.decoder, discriminator=self.discriminator
        )
        ckptio.load()

        # callbacks
        self.state = RunnerState()
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_state(self.state)

        # start testing
        self.callbacks.on_start()
        self.callbacks.on_epoch_start()

        self.gt_messages = gt_messages.to(self.device, non_blocking=True)
        num_eval_users = self.gt_messages.size(0) if num_eval_users == -1 else num_eval_users
        num_total_users = self.gt_messages.size(0) if num_total_users == -1 else num_total_users
        assert num_total_users >= 1 and 1 <= num_eval_users <= num_total_users

        self.state.message_set = self.gt_messages[:num_total_users]
        message_set_ = self.state.message_set[:num_eval_users]
        n, m = divmod(self.gt_messages.size(0), message_set_.size(0))
        gt_messages = message_set_.repeat(n, 1)
        gt_messages = torch.cat([gt_messages, message_set_[:m]], dim=0) if m != 0 else gt_messages

        self._eval_epoch(loader, gt_messages)
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

        self.images = None
        self.watermarked_images = None
        self.diff_images = None
        self.noised_images = None

        self.gt_messages = None
        self.org_messages = None
        self.watermarked_messages = None
        self.message_set = None


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
                metric_value = metric_fn(self.state.watermarked_messages, gt_messages=self.state.gt_messages, mean=True)
                self.state.metric_meters["acc_wm_gt"].update(metric_value, self.state.num_samples_in_batch)
            elif metric_name == "psnr":
                metric_value = metric_fn(self.state.watermarked_images, self.state.images)
                self.state.metric_meters["psnr"].update(metric_value, self.state.num_samples_in_batch)
            elif metric_name == "ssim":
                metric_value = metric_fn(self.state.watermarked_images, self.state.images)
                self.state.metric_meters["ssim"].update(metric_value, self.state.num_samples_in_batch)
            else:
                assert False

    def on_loader_end(self):
        if self.state.is_train:
            self.state.train_loss_meters = copy.deepcopy(self.state.loss_meters)
            self.state.train_metric_meters = copy.deepcopy(self.state.metric_meters)
        else:
            self.state.eval_loss_meters = copy.deepcopy(self.state.loss_meters)
            self.state.eval_metric_meters = copy.deepcopy(self.state.metric_meters)


class TensorBoardImageLogger(Callback):
    def __init__(self, writer, num_images, resize_to):
        super(TensorBoardImageLogger, self).__init__()
        self.writer = writer
        self.num_images = num_images
        self.resize_to = resize_to

    @torch.no_grad()
    def on_epoch_end(self):
        if self.state.is_train:
            images = self.state.images[:self.num_images]
            watermarked_images = self.state.watermarked_images[:self.num_images]
            diff_images = self.state.diff_images[:self.num_images] * 15.0
            noised_images = self.state.noised_images[:self.num_images]
            if self.resize_to is not None:
                images = F.interpolate(images, size=self.resize_to)
                watermarked_images = F.interpolate(watermarked_images, size=self.resize_to)
                diff_images = F.interpolate(diff_images, size=self.resize_to)
                noised_images = F.interpolate(noised_images, size=self.resize_to)

            all_images = torch.cat([images, watermarked_images, diff_images, noised_images], dim=0)
            grid = torchvision.utils.make_grid(all_images, nrow=self.num_images)
            self.writer.add_image("Images (Original v.s. Watermarked v.s. Diff v.s. Noised)/train", grid, self.state.epoch + 1)

    @torch.no_grad()
    def on_batch_end(self):
        if not self.state.is_train:
            images = self.state.images[:self.num_images]
            watermarked_images = self.state.watermarked_images[:self.num_images]
            diff_images = self.state.diff_images[:self.num_images] * 15.0
            noised_images = self.state.noised_images[:self.num_images]
            if self.resize_to is not None:
                images = F.interpolate(images, size=self.resize_to)
                watermarked_images = F.interpolate(watermarked_images, size=self.resize_to)
                diff_images = F.interpolate(diff_images, size=self.resize_to)
                noised_images = F.interpolate(noised_images, size=self.resize_to)

            all_images = torch.cat([images, watermarked_images, diff_images, noised_images], dim=0)
            grid = torchvision.utils.make_grid(all_images, nrow=self.num_images)
            self.writer.add_image("Images (Original v.s. Watermarked v.s. Diff v.s. Noised)/test", grid, self.state.iteration + 1)


class TensorBoardStatLogger(Callback):
    def __init__(self, writer, message_length, num_users):
        super(TensorBoardStatLogger, self).__init__()
        self.writer = writer
        self.acc = BitwiseAccuracy()

        assert message_length == 30
        assert num_users in [1, 10, 100, 1000]
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

    def on_epoch_start(self):
        self.accs_dict = defaultdict(list)
        self.pred_messages_dict = defaultdict(list)
        self.gt_messages = []

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

            self.gt_messages.append(self.state.gt_messages)

    @torch.no_grad()
    def on_epoch_end(self):
        if not self.state.is_train:
            accs_org = torch.cat(self.accs_dict["org"], dim=0)
            accs_wm = torch.cat(self.accs_dict["wm"], dim=0)
            pred_messages_wm = torch.cat(self.pred_messages_dict["wm"], dim=0)
            gt_messages = torch.cat(self.gt_messages, dim=0)

            self.writer.add_histogram("Test Bitwise Acc Distribution/org", accs_org)
            self.writer.add_histogram("Test Bitwise Acc Distribution/wm", accs_wm)

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

            # fpr_list, far_list, fnr_list, identify_acc_list = [], [], [], []
            # for tau in np.arange(0.5, 1.01, 0.01):
            #     predictions = torch.where(probs >= tau, 1, 0)
            #     fp = torch.sum((predictions == 1) & (labels == 0))
            #     tp = torch.sum((predictions == 1) & (labels == 1))
            #     fn = torch.sum((predictions == 0) & (labels == 1))
            #     tn = torch.sum((predictions == 0) & (labels == 0))
            #     tp_indices = predictions[labels == 1] == 1
            #     tidt = torch.sum(torch.all(torch.eq(pred_messages_wm[tp_indices], gt_messages[tp_indices]), dim=1))
            #     fpr_list.append(fp / (fp + tn))
            #     far_list.append(1. - tidt / tp)
            #     fnr_list.append(fn / (tp + fn))
            #     identify_acc_list.append(tidt / (tp + fn))      # equal to tpr when only one user
            # fpr_list = torch.as_tensor(fpr_list, dtype=probs.dtype, device=probs.device)
            # far_list = torch.as_tensor(far_list, dtype=probs.dtype, device=probs.device)
            # fnr_list = torch.as_tensor(fnr_list, dtype=probs.dtype, device=probs.device)
            # identify_acc_list = torch.as_tensor(identify_acc_list, dtype=probs.dtype, device=probs.device)
            # self.state.eval_metric_meters["fpr"].update(fpr_list, 1)
            # self.state.eval_metric_meters["far"].update(far_list, 1)
            # self.state.eval_metric_meters["fnr"].update(fnr_list, 1)
            # self.state.eval_metric_meters["identify_acc"].update(identify_acc_list, 1)


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
            elif isinstance(v, str) and k in ["part", "train_noises", "test_noises"]:
                hparams[k] = v
            elif isinstance(v, list):
                hparams[k] = str(v)
        metric_dict = {}
        for metric_name, meter in self.state.eval_metric_meters.items():
            metric_dict[f"hparam/{metric_name}"] = meter.avg.item()
        self.writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)


class ConvBNRelu(torch.nn.Module):

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(torch.nn.Module):

    def __init__(self, H, W, encoder_channels, encoder_blocks, message_length):
        super(Encoder, self).__init__()

        self.H = H
        self.W = W
        layers = [ConvBNRelu(3, encoder_channels)]

        for _ in range(encoder_blocks - 1):
            layer = ConvBNRelu(encoder_channels, encoder_channels)
            layers.append(layer)

        self.conv_layers = torch.nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(encoder_channels + 3 + message_length, encoder_channels)
        self.final_layer = torch.nn.Conv2d(encoder_channels, 3, kernel_size=1)

    def forward(self, images, messages):
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_messages = messages.unsqueeze(-1)
        expanded_messages.unsqueeze_(-1)

        expanded_messages = expanded_messages.expand(-1, -1, self.H, self.W)
        encoded_images = self.conv_layers(images)
        # concatenate expanded message and image
        encoded_images = torch.cat([expanded_messages, encoded_images, images], dim=1)
        encoded_images = self.after_concat_layer(encoded_images)
        encoded_images = self.final_layer(encoded_images)
        return encoded_images


class Decoder(torch.nn.Module):

    def __init__(self, decoder_channels, decoder_blocks, message_length):
        super(Decoder, self).__init__()

        layers = [ConvBNRelu(3, decoder_channels)]
        for _ in range(decoder_blocks - 1):
            layers.append(ConvBNRelu(decoder_channels, decoder_channels))
        layers.append(ConvBNRelu(decoder_channels, message_length))

        layers.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Linear(message_length, message_length)

    def forward(self, encoded_images):
        decoded_messages = self.layers(encoded_images)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        decoded_messages.squeeze_(3).squeeze_(2)
        decoded_messages = self.linear(decoded_messages)
        return decoded_messages


class Discriminator(torch.nn.Module):

    def __init__(self, discriminator_channels, discriminator_blocks):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, discriminator_channels)]
        for _ in range(discriminator_blocks-1):
            layers.append(ConvBNRelu(discriminator_channels, discriminator_channels))

        layers.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Linear(discriminator_channels, 1)

    def forward(self, images):
        x = self.before_linear(images)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
