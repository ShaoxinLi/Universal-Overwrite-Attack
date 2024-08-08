#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


class PSNR(object):

    def __init__(self):
        super(PSNR, self).__init__()

    @torch.no_grad()
    def __call__(self, post_processed_images, images):
        post_processed_images = torch.clamp(post_processed_images, 0, 1)
        images = torch.clamp(images, 0, 1)

        delta = (post_processed_images - images) * 255.
        psnr = 20. * np.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))
        return torch.mean(psnr)
