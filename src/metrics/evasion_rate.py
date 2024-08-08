#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


class EvasionRate(object):

    def __init__(self):
        super(EvasionRate, self).__init__()
        self.taus = np.arange(0.5, 1.01, 0.01)

    @torch.no_grad()
    def __call__(self, decoded_messages, messages, mean):

        evasion_rates = []
        decoded_rounded_messages = torch.clamp(torch.round(decoded_messages), 0, 1)
        accs = 1. - torch.sum(torch.abs(decoded_rounded_messages - messages), dim=1) / messages.size(1)
        for tau in self.taus:
            evasion_rate = torch.sum(accs <= tau) / messages.size(0)
            evasion_rates.append(evasion_rate)
        evasion_rates = torch.as_tensor(evasion_rates, dtype=messages.dtype, device=messages.device)
        return evasion_rates
