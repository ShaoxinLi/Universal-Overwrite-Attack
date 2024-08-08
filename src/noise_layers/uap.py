#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


class UAP(torch.nn.Module):
    def __init__(self, decoder, lr, xi, message_length, img_size):
        super(UAP, self).__init__()
        self.decoder = decoder
        self.lr = lr
        self.xi = xi / 255.
        self.message_length = message_length

        self.uap = torch.zeros(1, 3, img_size, img_size, device=decoder.device, dtype=torch.float32)
        self.uap.requires_grad_(True)

        self.optimizer = torch.optim.AdamW([self.uap], lr=lr)
        self.criterion = torch.nn.MSELoss().to(self.device, non_blocking=True)

    def forward(self, encoded_images):
        self.decoder.eval()
        self.optimizer.zero_grad(set_to_none=True)

        # set target messages
        target_messages = torch.tensor(np.random.choice([0., 1.], (encoded_images.size(0), self.message_length)),
                                       device=encoded_images.device, dtype=torch.float32)
        # generate noised images
        noised_images = encoded_images + self.uap
        noised_images = torch.clamp(noised_images, 0., 1.0)

        # compute loss and backward
        decoded_messages = self.decoder(noised_images)
        loss = self.criterion(decoded_messages, target_messages)
        loss.backward()
        self.optimizer.step()

        # project
        self.uap.data = torch.clamp(self.uap.data, -self.xi, self.xi)
        return noised_images