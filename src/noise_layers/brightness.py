#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import random


class Brightness(torch.nn.Module):
    def __init__(self, contrast_min, contrast_max):
        super(Brightness, self).__init__()
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        self.brightness = 0.2

    def forward(self, encoded_images):
        min_value = torch.min(encoded_images)
        max_value = torch.max(encoded_images)
        contrast = random.uniform(self.contrast_min, self.contrast_max)
        noised_images = torch.clamp((contrast * encoded_images + self.brightness), min_value, max_value)
        return noised_images