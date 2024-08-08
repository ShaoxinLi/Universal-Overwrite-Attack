#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def set_parameter_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters())


def get_num_trainable_parameters(net):
    net_parameters = filter(lambda p: p.requires_grad is True, net.parameters())
    return sum([np.prod(p.size()) for p in net_parameters])


def get_num_non_trainable_parameters(net):
    net_parameters = filter(lambda p: p.requires_grad is False, net.parameters())
    return sum([np.prod(p.size()) for p in net_parameters])