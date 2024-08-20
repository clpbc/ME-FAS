# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import torch


def make_optimizer(cfg, model, log):

    log.write(f"\nMethod:{cfg['model']['mode']}\n", is_file = True)
    total_learnable_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.write(f"Total learnable parameters num:{total_learnable_param_num}\n\n", is_file = True)

    lr = cfg['optimizer']['params']['lr']
    weight_decay = cfg['optimizer']['params']['weight_decay']

    params_optimize = [
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay},
    ]

    optimizer = getattr(torch.optim, cfg['optimizer']['name'])(params_optimize)
    return optimizer