# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import torch

def create_lr_scheduler(optimizer,
                        use = True,
                        step_each_epoch = 1,
                        total_epochs = 1,
                        warmup = True,
                        warmup_epochs = 1,
                        warmup_factor = 1e-3):
    assert step_each_epoch > 0 and total_epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if use:
            """
            x为step
            根据step数返回一个学习率倍率因子，
            注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
            """
            if warmup is True and x <= (warmup_epochs * step_each_epoch):
                alpha = float(x) / (warmup_epochs * step_each_epoch)
                # warmup过程中lr倍率因子从warmup_factor -> 1
                return warmup_factor * (1 - alpha) + alpha
            else:
                # warmup后lr倍率因子从1 -> 0
                # 参考deeplab_v2: Learning rate policy
                return (1 - (x - warmup_epochs * step_each_epoch) / ((total_epochs - warmup_epochs) * step_each_epoch)) ** 0.9
        
        else:   # 如果不适用自定义学习率，倍率默认为1
            return 1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = f)