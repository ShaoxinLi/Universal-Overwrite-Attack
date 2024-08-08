#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import random


class Gaussian(torch.nn.Module):
    def __init__(self, std_min, std_max):
        super(Gaussian, self).__init__()
        self.std_min = std_min
        self.std_max = std_max

    def forward(self, encoded_images):
        min_value = torch.min(encoded_images)
        max_value = torch.max(encoded_images)
        noise_level = random.uniform(self.std_min, self.std_max)

        gaussian = torch.randn_like(encoded_images, device=encoded_images.device)
        noised_images = encoded_images + noise_level * gaussian
        noised_images = noised_images.clamp(min_value, max_value)
        return noised_images

