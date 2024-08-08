#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class BitwiseAccuracy(object):

    def __init__(self):
        super(BitwiseAccuracy, self).__init__()

    @torch.no_grad()
    def __call__(self, pred_messages, gt_messages=None, message_set=None, mean=True):
        if gt_messages is not None:
            pred_messages = torch.clamp(torch.round(pred_messages), 0, 1)
            accs = 1. - torch.sum(torch.abs(pred_messages - gt_messages), dim=1) / gt_messages.size(1)
            if mean:
                return torch.sum(accs) / gt_messages.size(0)
            else:
                return accs
        elif message_set is not None:
            pred_messages = pred_messages.unsqueeze(1)
            message_set = message_set.unsqueeze(0)
            pred_messages = torch.clamp(torch.round(pred_messages), 0, 1)
            accs = 1. - torch.sum(torch.abs(pred_messages - message_set), dim=2) / pred_messages.size(-1)
            return accs
