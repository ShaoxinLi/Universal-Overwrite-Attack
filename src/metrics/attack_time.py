#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class AttackTime(object):

    def __init__(self):
        super(AttackTime, self).__init__()

    def __call__(self, attack_start_time, attack_end_time):
        return torch.tensor(attack_end_time - attack_start_time)
