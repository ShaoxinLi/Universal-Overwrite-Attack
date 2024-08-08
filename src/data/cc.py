#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import torchvision


class MyImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index, path


def cc(dataset_dir_path, part, img_size, message_length):

    assert part in ["victim", "attack", "test"]
    if part != "test":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((img_size, img_size), pad_if_needed=True),
            torchvision.transforms.ToTensor()
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((img_size, img_size)),
            torchvision.transforms.ToTensor()
        ])

    dir_path = os.path.join(dataset_dir_path, part)
    if not os.path.exists(dir_path):
        dir_path = os.path.join("./Datasets/ConceptualCaption", part)
    dataset = MyImageFolder(dir_path, transforms)

    if part == "attack" or part == "test":
        message_path = os.path.join(dataset_dir_path, part, "watermarks.npy")
        if not os.path.exists(message_path):
            message_path = os.path.join("./Datasets/ConceptualCaption", part, "watermarks.npy")
        messages = torch.from_numpy(np.load(message_path)).float()
        messages = messages[:, :message_length]
        return dataset, messages
    else:
        return dataset
