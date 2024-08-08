#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


class WEvade(torch.nn.Module):
    def __init__(self, decoder, max_iter, lr, xi, epsilon, message_length):
        super(WEvade, self).__init__()
        self.decoder = decoder
        self.max_iter = max_iter
        self.lr = lr
        self.xi = xi / 255.
        self.epsilon = epsilon
        self.message_length = message_length
        self.criterion = torch.nn.MSELoss().to(self.device, non_blocking=True)

    def forward(self, encoded_images):
        self.decoder.eval()
        noised_images = []
        for encoded_image in encoded_images:
            encoded_image = encoded_image.unsqueeze(0)
            orig_encoded_image = encoded_image.clone()

            # set the target message
            target_message = torch.tensor(np.random.choice([0., 1.], (encoded_image.size(0), self.message_length)),
                                          device=encoded_image.device, dtype=torch.float32)

            for i in range(self.max_iter):
                encoded_image = encoded_image.requires_grad_(True)
                min_value, max_value = torch.min(encoded_image), torch.max(encoded_image)

                # decode original encoded image
                decoded_message = self.decoder(encoded_image)

                # generate adv encoded image
                loss = self.criterion(decoded_message, target_message)
                grads = torch.autograd.grad(loss, encoded_image)
                with torch.no_grad():
                    encoded_image = encoded_image - self.lr * grads[0]
                    encoded_image = torch.clamp(encoded_image, min_value, max_value)

                # projection adv encoded image
                perturbation_norm = torch.norm(encoded_image - orig_encoded_image, float("inf"))
                if perturbation_norm > self.xi:
                    c = self.xi / perturbation_norm
                    encoded_image = self.project(encoded_image, orig_encoded_image, c)

                encoded_image = torch.clamp(encoded_image, 0., 1.0)

                # decode adv encoded image
                decoded_message = torch.clamp(torch.round(self.decoder(encoded_image)), 0, 1)

                # early stopping.
                bitwise_acc_target = 1. - torch.sum(torch.abs(decoded_message - target_message)) / self.message_length
                if bitwise_acc_target >= 1 - self.epsilon:
                    break

            noised_images.append(encoded_image)
        noised_images = torch.cat(noised_images, dim=0)
        return noised_images

    @staticmethod
    def project(image, orig_image, epsilon):
        # If the perturbation exceeds the upper bound, project it back.
        delta = image - orig_image
        delta = epsilon * delta
        return orig_image + delta