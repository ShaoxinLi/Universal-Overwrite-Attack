#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from .coco import coco
from .imagenet import imagenet
from .cc import cc


def load_dataset(data_root_dir, dataset_name, part, img_size, message_length, num_train_images):
    if dataset_name == "coco":
        dataset_dir_path = os.path.join(data_root_dir, "coco10k")
        dataset = coco(dataset_dir_path, part, img_size, message_length)
    elif dataset_name == "imagenet":
        dataset_dir_path = os.path.join(data_root_dir, "imagenet10k")
        dataset = imagenet(dataset_dir_path, part, img_size, message_length)
    elif dataset_name == "cc":
        dataset_dir_path = os.path.join(data_root_dir, "ConceptualCaption")
        dataset = imagenet(dataset_dir_path, part, img_size, message_length)
    else:
        assert False
    if num_train_images > 0:
        if part == "train":
            dataset, messages = dataset
            dataset = torch.utils.data.Subset(dataset, np.arange(num_train_images))
            # messages = messages[:num_train_images]
            return dataset, messages
        elif part == "victim":
            dataset = torch.utils.data.Subset(dataset, np.arange(num_train_images))
            return dataset
        else:
            return dataset
    else:
        return dataset
