# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

from .mefas import MEFas


def BuildModel(cfg):

    model = MEFas(cfg)

    return model

