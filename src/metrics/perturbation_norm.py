#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class PerturbationNorm(object):

    def __init__(self):
        super(PerturbationNorm, self).__init__()

    @torch.no_grad()
    def __call__(self, post_processed_images, images):
        norm = 0.0
        for post_process_image, image in zip(post_processed_images, images):
            # norm += torch.norm(post_process_image - image, float("inf"))
            norm += torch.norm(post_process_image - image, 2)
        norm = norm / images.size(0) * 255.
        return norm
