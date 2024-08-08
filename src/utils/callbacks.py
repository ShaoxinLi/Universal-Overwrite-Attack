#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ..utils.train import TimeMeter
from ..utils.file import check_dir
from ..utils.time import convert_secs2time
from ..utils.plot import plot_record


class Callback(object):

    def set_state(self, state):
        self.state = state

    def on_start(self):
        pass

    def on_epoch_start(self):
        pass

    def on_loader_start(self):
        pass

    def on_batch_start(self):
        pass

    def on_batch_end(self):
        pass

    def on_loader_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_end(self):
        pass

    def on_after_backward(self):
        """Called after ``loss.backward()`` but before optimizer does anything."""
        pass


class Callbacks(Callback):
    """Class that combines multiple callbacks into one. For internal use only"""

    def __init__(self, callbacks):
        super(Callbacks, self).__init__()
        if callbacks is None:
            callbacks = [Callback()]
        self.callbacks = callbacks

    def set_state(self, state):
        for callback in self.callbacks:
            callback.set_state(state)

    def on_start(self):
        for callback in self.callbacks:
            callback.on_start()

    def on_epoch_start(self):
        for callback in self.callbacks:
            callback.on_epoch_start()

    def on_loader_start(self):
        for callback in self.callbacks:
            callback.on_loader_start()

    def on_batch_start(self):
        for callback in self.callbacks:
            callback.on_batch_start()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_loader_end(self):
        for callback in self.callbacks:
            callback.on_loader_end()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_after_backward(self):
        for callback in self.callbacks:
            callback.on_after_backward()


class TrainLogger(Callback):
    def __init__(self, logger, num_epochs, verbose=False, print_freq=100):
        super(TrainLogger, self).__init__()
        self.logger = logger
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_freq = print_freq
        self.timer = TimeMeter()

    def on_start(self):
        self.timer.reset_at_start()
        # if hasattr(self.state, "random_message") and self.state.random_message is not None:
        #     self.logger(f"Random target message: {self.state.random_message.tolist()[0]}")

    def on_epoch_start(self):
        if self.state.epoch > 0:
            need_hour, need_mins, need_secs = convert_secs2time(self.timer.epoch_time.avg * (self.num_epochs - self.state.epoch))
            self.need_time = f"[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]"
        else:
            self.need_time = f"[Need: 99:99:99]"

    def on_loader_start(self):
        self.timer.reset_at_loader_start()

    def on_batch_start(self):
        self.timer.clock_at_batch_start()

    def on_batch_end(self):
        self.timer.clock_at_batch_end()
        if self.verbose and self.state.is_train and (self.state.iteration + 1) % self.print_freq == 0:
            desc = f"[Epoch={self.state.epoch + 1:04d}/{self.num_epochs:04d}][{self.state.iteration + 1:04d}/{self.state.num_batches:04d}]"
            desc += self._format_meters(self.state.loss_meters, self.state.metric_meters)
            self.logger(desc)

    def on_loader_end(self):
        if self.state.is_train and self.verbose:
            d_time = self.timer.data_time.avg
            b_time = self.timer.batch_time.avg
            self.logger(f"[Epoch={self.state.epoch + 1:04d}/{self.num_epochs:04d}] Data Time: {d_time:.3f}s Batch Time: {b_time:.3f}s")
        # if hasattr(self.state, "message_set") and self.state.message_set is not None:
        #     if self.state.is_train:
        #         self.logger(f"Ground truth message set at training epoch {self.state.epoch + 1}: {self.state.message_set.tolist()}")
        #     else:
        #         self.logger(f"Ground truth message set at validating epoch {self.state.epoch + 1}: {self.state.message_set.tolist()}")

    def on_epoch_end(self):
        self.timer.clock_at_epoch_end()
        desc = f"[Epoch={self.state.epoch + 1:04d}/{self.num_epochs:04d}]"
        desc += f"{self.need_time:s}"
        desc += f"{self._format_meters(self.state.train_loss_meters, self.state.train_metric_meters, 'Train')}"
        if self.state.eval_loss_meters is not None:
            desc += f"{self._format_meters(self.state.eval_loss_meters, self.state.eval_metric_meters, 'Val')}"
        self.logger(desc)

    @staticmethod
    def _format_meters(loss_meters, metric_meters, prefix=""):
        s = ""
        for loss_name, meter in loss_meters.items():
            if prefix:
                s += f" | {prefix}_{loss_name}: {meter.avg.item():.3f}"
            else:
                s += f" | {loss_name}: {meter.avg.item():.3f}"
        for metric_name, meter in metric_meters.items():
            if prefix:
                if len(meter.avg.size()) == 0:
                    s += f" | {prefix}_{metric_name}: {meter.avg.item():.3f}"
                else:
                    s += f" | {prefix}_{metric_name}: {meter.avg[10].item():.3f}"
            else:
                if len(meter.avg.size()) == 0:
                    s += f" | {metric_name}: {meter.avg.item():.3f}"
                else:
                    s += f" | {metric_name}: {meter.avg[10].item():.3f}"
        return s


class TensorBoardScalarLogger(Callback):
    def __init__(self, writer):
        super(TensorBoardScalarLogger, self).__init__()
        self.writer = writer

    def on_epoch_end(self):

        for loss_name, meter in self.state.train_loss_meters.items():
            self.writer.add_scalar(f"{loss_name}/train", meter.avg.item(), self.state.epoch + 1)
        if self.state.eval_loss_meters is not None:
            for loss_name, meter in self.state.eval_loss_meters.items():
                self.writer.add_scalar(f"{loss_name}/val", meter.avg.item(), self.state.epoch + 1)
        for metric_name, meter in self.state.train_metric_meters.items():
            if len(meter.avg.size()) == 0:
                self.writer.add_scalar(f"{metric_name}/train", meter.avg.item(), self.state.epoch + 1)
            else:
                self.writer.add_scalar(f"{metric_name}/train", meter.avg[10].item(), self.state.epoch + 1)
        if self.state.eval_metric_meters is not None:
            for metric_name, meter in self.state.eval_metric_meters.items():
                if len(meter.avg.size()) == 0:
                    self.writer.add_scalar(f"{metric_name}/val", meter.avg.item(), self.state.epoch + 1)
                else:
                    self.writer.add_scalar(f"{metric_name}/val", meter.avg[10].item(), self.state.epoch + 1)
        self.writer.flush()


class HistoryRecorder(Callback):

    def __init__(self, exp_dir_path, exp_datatime):
        super(HistoryRecorder, self).__init__()
        self.exp_dir_path = exp_dir_path
        self.exp_datatime = exp_datatime

    def on_start(self):
        csv_dir_path = os.path.join(self.exp_dir_path, "csv_files")
        check_dir(csv_dir_path)
        self.his_file_path = os.path.join(csv_dir_path, f"run-{self.exp_datatime}.csv")

    def on_epoch_end(self):
        record = {"epoch": self.state.epoch + 1}
        for loss_name, meter in self.state.train_loss_meters.items():
            record.update({f"train_{loss_name}": meter.avg.item()})
        for metric_name, meter in self.state.train_metric_meters.items():
            if len(meter.avg.size()) == 0:
                record.update({f"train_{metric_name}": meter.avg.item()})
            else:
                record.update({f"train_{metric_name}": meter.avg[10].item()})
        if self.state.eval_loss_meters is not None:
            for loss_name, meter in self.state.eval_loss_meters.items():
                record.update({f"val_{loss_name}": meter.avg.item()})
        if self.state.eval_metric_meters is not None:
            for metric_name, meter in self.state.eval_metric_meters.items():
                if len(meter.avg.size()) == 0:
                    record.update({f"val_{metric_name}": meter.avg.item()})
                else:
                    record.update({f"val_{metric_name}": meter.avg[10].item()})
        self.save_record(self.his_file_path, **record)

    @staticmethod
    def save_record(csv_file, **kwargs):
        """Save records to a csv file"""

        record = {k: v for k, v in kwargs.items() if v is not None}
        file_existence = os.path.exists(csv_file)
        with open(csv_file, "a+") as f:
            writer = csv.writer(f)
            if not file_existence:
                if "epoch" in record.keys():
                    other_keys = list(record.keys())
                    other_keys.remove("epoch")
                    header = ["epoch"] + other_keys
                else:
                    header = list(record.keys())
                writer.writerow(header)
            else:
                with open(csv_file, "r") as g:
                    reader = csv.reader(g)
                    header = next(reader)
            row = [f"{record[i]:.3f}" if isinstance(record[i], float) else f"{record[i]}" for i in header]
            writer.writerow(row)


class HistoryPlotter(Callback):
    def __init__(self, exp_dir_path, exp_datatime):
        super(HistoryPlotter, self).__init__()
        self.exp_datatime = exp_datatime
        self.csv_dir_path = os.path.join(exp_dir_path, "csv_files")
        self.image_dir_path = os.path.join(exp_dir_path, "images", f"run-{self.exp_datatime}")

    def on_end(self):
        check_dir(self.image_dir_path)
        his_file_path = os.path.join(self.csv_dir_path, f"run-{self.exp_datatime}.csv")
        records = {}
        with open(his_file_path, "r") as f:
            reader = csv.DictReader(f)
            for d in reader:
                for k, v in d.items():
                    v = int(v) if k == "epoch" else float(v)
                    if k not in records.keys():
                        records[k] = [v]
                    else:
                        records[k].append(v)
        d = {}
        for loss_name in self.state.train_loss_meters.keys():
            d.update({f"history_{loss_name}": [f"train_{loss_name}"]})
            if self.state.eval_loss_meters is not None and loss_name in self.state.eval_loss_meters.keys():
                d[f"history_{loss_name}"].append(f"val_{loss_name}")
        for metric_name in self.state.train_metric_meters.keys():
            d.update({f"history_{metric_name}": [f"train_{metric_name}"]})
            if self.state.eval_metric_meters is not None and metric_name in self.state.eval_metric_meters.keys():
                d[f"history_{metric_name}"].append(f"val_{metric_name}")
        for img_name, keys in d.items():
            fig = plot_record(records, keys, use_marker=False, show_legend=True)
            plt.savefig(os.path.join(self.image_dir_path, img_name), bbox_inches="tight", dpi=300)
            plt.close("all")


class CheckpointSaver(Callback):

    def __init__(self, exp_dir_path, exp_datatime, ckpt_name_template="ckpt_ep{epoch:04d}_{monitor}{value:.3f}.pth.tar",
                 monitor=None, mode="min"):
        super(CheckpointSaver, self).__init__()
        self.ckpt_dir_path = os.path.join(exp_dir_path, "checkpoints", f"run-{exp_datatime}")
        self.ckpt_name_template = ckpt_name_template
        self.monitor = monitor
        if mode == "min":
            self.best = torch.tensor(999999999.)
            self.monitor_op = torch.less
        elif mode == "max":
            self.best = -torch.tensor(999999999.)
            self.monitor_op = torch.greater

    def on_start(self):
        check_dir(self.ckpt_dir_path)

    def on_epoch_end(self):
        if self.monitor is not None:
            current = self.get_monitor_value()
            if self.monitor_op(current, self.best):
                self.best = current
                ckpt_name = self.ckpt_name_template.format(epoch=self.state.epoch + 1, monitor=self.monitor, value=current.item())
                ckpt_file_path = os.path.join(self.ckpt_dir_path, ckpt_name)
                self._save_checkpoint(self.state.state_dict, ckpt_file_path)
        else:
            ckpt_file_path = os.path.join(self.ckpt_dir_path, "ckpt.pth.tar")
            self._save_checkpoint(self.state.state_dict, ckpt_file_path)

    def on_end(self):
        if not os.path.exists(os.path.join(self.ckpt_dir_path, "ckpt.pth.tar")):
            exist_files = [f for f in os.listdir(self.ckpt_dir_path) if os.path.isfile(os.path.join(self.ckpt_dir_path, f))]
            ckpt_files = [f for f in exist_files if f.startswith("ckpt")]
            if ckpt_files:
                epochs = np.array([int(re.findall(r"ep(\d+)", f)[0]) for f in ckpt_files])
                src_path = os.path.join(self.ckpt_dir_path, ckpt_files[int(epochs.argmax())])
                des_path = os.path.join(self.ckpt_dir_path, "ckpt.pth.tar")
                shutil.copyfile(src_path, des_path)
            else:
                raise FileNotFoundError(f"Can't find any checkpoint files in {self.ckpt_dir_path}")

    @staticmethod
    def _save_checkpoint(state_dict, ckpt_file_path):

        # status is a dict
        for k, v in state_dict.items():
            if isinstance(v, torch.nn.Module) or isinstance(v, torch.optim.Optimizer):
                if hasattr(v, "module"):    # used for saving DDP models
                    state_dict[k] = v.module.state_dict()
                else:
                    state_dict[k] = v.state_dict()
        torch.save(state_dict, ckpt_file_path)

    def get_monitor_value(self):
        value = None
        for loss_name, meter in self.state.loss_meters.items():
            if loss_name == self.monitor:
                value = meter.avg
                break
        if value is None:
            for metric_name, metric_meter in self.state.metric_meters.items():
                if metric_name == self.monitor:
                    if len(metric_meter.avg.size()) == 0:
                        value = metric_meter.avg
                    else:
                        value = metric_meter.avg[10]
                    break
        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value


class EarlyStopper(Callback):

    def __init__(self, monitor, patience, mode="min", delta=0.):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        if mode == "min":
            self.best = torch.tensor(99999.)
            self.monitor_op = torch.less
        elif mode == "max":
            self.best = -torch.tensor(99999.)
            self.monitor_op = torch.greater
        self.delta = delta

    def on_start(self):
        self._reset()

    def on_epoch_end(self):

        current = self.get_monitor_value()
        if self.mode == "min":
            _current = current + self.delta
        else:
            _current = current - self.delta
        if self.monitor_op(_current, self.best):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.state.early_stop = True

    def _reset(self):
        self.counter = 0
        self.state.early_stop = False

    def get_monitor_value(self):
        value = None
        for loss_name, meter in self.state.loss_meters.items():
            if loss_name == self.monitor:
                value = meter.avg
                break
        if value is None:
            for metric_name, metric_meter in self.state.metric_meters.items():
                if metric_name == self.monitor:
                    if len(metric_meter.avg.size()) == 0:
                        value = metric_meter.avg
                    else:
                        value = metric_meter.avg[10]
                    break
        if value is None:
            raise ValueError(f"EarlyStopper can't find {self.monitor} value to monitor")
        return value