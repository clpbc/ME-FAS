# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

from .flip import Flip
from .jpgfas import JpgFas


def BuildModel(cfg):

    if cfg['model']['mode'] == 'flip':
        model = Flip(cfg)
    
    elif cfg['model']['mode'] == 'jpgfas':
        model = JpgFas(cfg)

    return model

