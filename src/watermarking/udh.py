#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import math
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


class UDH(object):
    def __init__(self, device, seed=None):
        super(UDH, self).__init__()
        self.device = device
        self.seed = seed

    def train(self, encoder, decoder, noiser, loader, message_length,
              num_epochs, lr, callbacks):

        self.encoder = encoder.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.noiser = noiser.to(self.device, non_blocking=True)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr,
            betas=(0.5, 0.999)
        )

        self.mse_loss = torch.nn.MSELoss().to(self.device, non_blocking=True)
        self.message_length = message_length
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

            # adjust learning rate
            cur_lr = lr * (0.1 ** (epoch // 30))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = cur_lr

            # train one epoch
            self._train_epoch(loader)

            # update state
            self.state.state_dict = {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "optimizer": self.optimizer,
            }
            self.callbacks.on_epoch_end()

            # check early stopping
            if self.state.early_stop:
                break

        self.callbacks.on_end()

    def _train_epoch(self, loader):

        self.encoder.train()
        self.decoder.train()

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

        self.optimizer.zero_grad(set_to_none=True)

        # watermarking
        watermarked_images = self.encoder(images, messages)         # ~ (0, 1)
        watermarked_images = watermarked_images.clamp_(0., 1.)      # (0, 1)
        encoder_loss = self.mse_loss(watermarked_images, images)

        # add noise
        noised_images = self.noiser(watermarked_images)

        # decode
        self.decoder.train()  # because decoder maybe set to eval mode when add noises to watermarked image
        watermarked_messages = self.decoder(noised_images)
        decoder_loss = self.mse_loss(watermarked_messages, messages)

        total_loss = encoder_loss + 0.75 * decoder_loss
        total_loss.backward()
        self.callbacks.on_after_backward()

        self.optimizer.step()

        # update states
        self.state.losses = {
            "loss": total_loss.detach()
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

        # encode
        watermarked_images = self.encoder(images, messages)
        watermarked_images = watermarked_images.clamp_(0., 1.)  # (0, 1)
        encoder_loss = self.mse_loss(watermarked_images, images)

        # add noise
        noised_images = self.noiser(watermarked_images)

        # decode
        watermarked_messages = self.decoder(noised_images)
        decoder_loss = self.mse_loss(watermarked_messages, messages)

        # decode original images
        org_messages = self.decoder(images)

        total_loss = encoder_loss + 0.75 * decoder_loss

        # update states
        self.state.losses = {
            "loss": total_loss,
        }

        self.state.gt_messages = messages
        self.state.watermarked_messages = watermarked_messages
        self.state.org_messages = org_messages

        self.state.images = images
        self.state.watermarked_images = watermarked_images
        self.state.diff_images = torch.abs(watermarked_images - images)
        self.state.noised_images = noised_images

    def test(self, encoder, decoder, noiser, loader, gt_messages,
             num_eval_users, num_total_users, ckpt_file_path, callbacks):

        self.encoder = encoder.to(self.device, non_blocking=True)
        self.decoder = decoder.to(self.device, non_blocking=True)
        self.noiser = noiser.to(self.device, non_blocking=True)

        self.mse_loss = torch.nn.MSELoss().to(self.device, non_blocking=True)

        assert ckpt_file_path
        ckptio = CheckpointIO(
            ckpt_file_path=ckpt_file_path, device=self.device,
            encoder=self.encoder, decoder=self.decoder
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

        assert message_length == 256
        assert num_users in [1, 10, 100, 1000]
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



class UnetGenerator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=None, use_dropout=False, output_function=torch.nn.Sigmoid):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block
        self.tanh = output_function == torch.nn.Tanh
        if self.tanh:
            # self.factor = 10 / 255
            self.factor = 2. / 255.
        else:
            self.factor = 1.0

    def forward(self, images, messages):
        messages = messages_to_images(messages, images.size(-1))        # [0, 1]
        transformed_messages = self.factor * self.model(messages)       # [-2 / 255, 2 / 255]
        encoded_images = images + transformed_messages                  # ~ [0, 1]
        return encoded_images


class UnetSkipConnectionBlock(torch.nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False,
                 norm_layer=None, use_dropout=False, output_function=torch.nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d
        if norm_layer == None:
            use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        downconv = torch.nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        uprelu = torch.nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if output_function == torch.nn.Tanh:
                up = [uprelu, upconv, torch.nn.Tanh()]
            else:
                up = [uprelu, upconv, torch.nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = torch.nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if norm_layer == None:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downrelu, downconv]
                up = [uprelu, upconv]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [torch.nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RevealNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, nhf=64, norm_layer=None, output_function=torch.nn.Sigmoid, message_length=256):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
        self.message_length = message_length

        self.conv1 = torch.nn.Conv2d(input_nc, nhf, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
        self.conv6 = torch.nn.Conv2d(nhf, output_nc, 3, 1, 1)
        self.output = output_function()
        self.relu = torch.nn.ReLU(True)

        self.norm_layer = norm_layer
        if norm_layer != None:
            self.norm1 = norm_layer(nhf)
            self.norm2 = norm_layer(nhf * 2)
            self.norm3 = norm_layer(nhf * 4)
            self.norm4 = norm_layer(nhf * 2)
            self.norm5 = norm_layer(nhf)

    def forward(self, images):
        if self.norm_layer != None:
            x = self.relu(self.norm1(self.conv1(images)))
            x = self.relu(self.norm2(self.conv2(x)))
            x = self.relu(self.norm3(self.conv3(x)))
            x = self.relu(self.norm4(self.conv4(x)))
            x = self.relu(self.norm5(self.conv5(x)))
            x = self.output(self.conv6(x))
        else:
            x = self.relu(self.conv1(images))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.relu(self.conv5(x))
            x = self.output(self.conv6(x))
        messages = images_to_messages(x, self.message_length)       # [0, 1]
        return messages


def messages_to_images(messages, image_size):
    message_length = messages.size(1)

    num_patches_per_row = math.sqrt(message_length)
    assert num_patches_per_row.is_integer()
    num_patches_per_row = int(num_patches_per_row)

    patch_size = image_size / num_patches_per_row
    assert patch_size.is_integer()
    patch_size = int(patch_size)

    images = messages.repeat_interleave(patch_size, dim=1)
    images = images.reshape(-1, num_patches_per_row, num_patches_per_row * patch_size)
    images = images.repeat_interleave(patch_size, dim=1)
    images = images.unsqueeze(1).repeat(1, 3, 1, 1)
    return images


def images_to_messages(images, message_length):
    num_patches_per_row = math.sqrt(message_length)
    assert num_patches_per_row.is_integer()
    num_patches_per_row = int(num_patches_per_row)

    patch_size = images.size(-1) / num_patches_per_row
    assert patch_size.is_integer()
    patch_size = int(patch_size)

    conv = torch.nn.Conv2d(3, 1, patch_size, patch_size, bias=False, device=images.device)
    conv.weight = torch.nn.Parameter(torch.ones_like(conv.weight), requires_grad=False)

    messages = conv(images)
    messages = messages / (3 * patch_size * patch_size)
    messages = messages.reshape(images.size(0), -1)
    return messages
