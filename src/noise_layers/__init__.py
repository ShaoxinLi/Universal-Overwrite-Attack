#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from .identity import Identity
from .brightness import Brightness
from .gaussian import Gaussian
from .gaussian_blur import GaussianBlur
# from .jpeg import DiffJPEG
from .wevade import WEvade
from .uap import UAP


class Noiser(torch.nn.Module):

    def __init__(self, noises, a, sigma, sigma_blur, Q, decoder, message_length, img_size):
        super(Noiser, self).__init__()
        noises = noises.split("+") if noises else ["identity"]
        self.noise_layers = {}
        for noise in noises:
            if noise == "identity":
                self.noise_layers["identity"] = Identity()
            elif noise == "brightness":
                self.noise_layers["brightness"] = Brightness(a[0], a[1])
            elif noise == "gaussian":
                self.noise_layers["gaussian"] = Gaussian(sigma[0], sigma[1])
            elif noise == "blur":
                self.noise_layers["blur"] = GaussianBlur(sigma_blur[0], sigma_blur[1])
            elif noise == "jpeg":
                self.noise_layers["jpeg"] = DiffJPEG(Q[0], Q[1], differentiable=True)
            # elif noise == "wevade":
            #     self.noise_layers["wevade"] = WEvade(
            #         decoder=decoder, max_iter=5000, lr=0.01, xi=2.0, epsilon=0.01,
            #         message_length=message_length
            #     )
            # elif noise == "uap":
            #     self.noise_layers["uap"] = UAP(
            #         decoder=decoder, lr=0.001, xi=2.0, message_length=message_length,
            #         img_size=img_size
            #     )
            else:
                raise ValueError("Noise not recognized: \n{}".format(noise))
        self.noise_types = list(self.noise_layers.keys())

    def forward(self, encoded_images):
        noise = np.random.choice(self.noise_types)
        random_noise_layer = self.noise_layers[noise]
        noised_images = random_noise_layer(encoded_images=encoded_images)
        return noised_images
