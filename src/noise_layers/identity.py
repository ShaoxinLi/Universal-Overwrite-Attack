#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, encoded_images):
        return encoded_images