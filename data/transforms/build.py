# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import numpy as np
import imgaug.augmenters as iaa
import torchvision.transforms as T

from .transforms import RandomErasing


class FasTransforms:
    def __init__(self, cfg, isTrain):
        self.mean = cfg['transforms']['mean']
        self.std = cfg['transforms']['std']
        self.imgSize = cfg['transforms']['imgSize']

        if isTrain:
            self.seq = iaa.Sequential([
                iaa.Fliplr(0.5), 
                iaa.MultiplyAndAddToBrightness(mul = (0.8, 1.2), add = (-3, 3)),
                # iaa.CoarseDropout(p = 0.5, size_percent = 0.0625),
                # iaa.AdditiveGaussianNoise(scale = 0.05*255, per_channel = True), 
                iaa.Resize({"shorter-side": self.imgSize, "longer-side": "keep-aspect-ratio"}),
                iaa.CropToFixedSize(width = self.imgSize, height = self.imgSize, position = "center"),
                # iaa.Resize({'longer-side': self.img_size, 'shorter-side': 'keep-aspect-ratio'}),
                # iaa.PadToFixedSize(width = self.img_size, height = self.img_size, position = "center")
            ])
            
        else:
            self.seq = iaa.Sequential([
                iaa.Resize({"shorter-side": self.imgSize, "longer-side": "keep-aspect-ratio"}),
                iaa.CropToFixedSize(width = self.imgSize, height = self.imgSize, position = "center")
            ])
        
        self.aug_seq = iaa.Sequential([
            iaa.Crop(percent = (0, 0.2), keep_size = True),
            iaa.Sometimes(
                0.6,
                iaa.Sequential([iaa.MultiplyBrightness((0.8, 1.2)), iaa.MultiplyHueAndSaturation(mul_saturation = (0.8, 1.2), mul_hue = (0.95, 1.05))])
            ),
            iaa.GaussianBlur(sigma = (0.5, 1)),
            iaa.Fliplr(0.5),
        ])

        self.trans = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean = self.mean, std = self.std)
                    ])
        
    def __call__(self, img):
        img = np.array(img)

        img = self.seq(image = img)
        aug1_img = self.aug_seq(image = img)
        aug2_img = self.aug_seq(image = img)
 
        return self.trans(img), self.trans(aug1_img), self.trans(aug2_img)



