# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import os, torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from clip import clip
from .prompt_templates import FLIP_real_templates, FLIP_spoof_templates


def LoadClip(cfg):

    backbone_name = cfg['model']['backbone']

    model_path = clip._download(clip._MODELS[backbone_name], os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location = cfg['device']).eval()
        state_dict = None  
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False

    # CoOp使用默认CLIP架构即可
    model = clip.build_model(state_dict or model.state_dict())

    return model


class ProjHead(nn.Module):
    def __init__(self, in_dim = 512, mlp_dim = 4096, out_dim = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace = True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace = True),
            nn.Linear(mlp_dim, out_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        return self.mlp(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip = clip_model
        self.logit_scale = clip_model.logit_scale
        self.projection_head = ProjHead(in_dim = 512, mlp_dim = 4096, out_dim = 256)

    def forward(self, origin_img, aug1_img, aug2_img, labels):

        ### image brunch
        image = torch.cat([origin_img, aug1_img, aug2_img], dim = 0)

        img_feat = self.clip.visual(image)
        img_feat = img_feat / img_feat.norm(dim = -1, keepdim = True)
        
        origin_feat = img_feat[: origin_img.shape[0]]
        aug1_feat = img_feat[origin_img.shape[0]: origin_img.shape[0] + aug1_img.shape[0]]
        aug2_feat = img_feat[-1 * aug2_img.shape[0]: ]
        ### 

        ### text brunch
        prompt_list = FLIP_spoof_templates + FLIP_real_templates
        tokenized_prompts = clip.tokenize(prompt_list).cuda(non_blocking = True, device = self.cfg['device'])
        text_feat = self.clip.encode_text(tokenized_prompts)

        spoof_prompts_feat = text_feat[: len(FLIP_spoof_templates)]
        real_prompts_feat = text_feat[len(FLIP_spoof_templates): ]

        mean_spoof_prompt_feat = spoof_prompts_feat.mean(dim = 0)
        mean_real_prompt_feat = real_prompts_feat.mean(dim = 0)

        ensemble_prompt_feat = torch.stack([mean_spoof_prompt_feat, mean_real_prompt_feat], dim = 0)
        ensemble_prompt_feat = ensemble_prompt_feat / ensemble_prompt_feat.norm(dim = -1, keepdim = True)
        ###

        ### SimCLR features
        out_aug1_feat = self.projection_head(aug1_feat)
        out_aug2_feat = self.projection_head(aug2_feat)
        ###

        ### cls loss
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * origin_feat @ ensemble_prompt_feat.t()
        ###

        ### MSE loss
        text_embedding_v1 = []
        text_embedding_v2 = []

        for label in labels:

            if label == 0:
                available_indices = np.arange(0, len(FLIP_spoof_templates))
                pair_1 = np.random.choice(available_indices, len(FLIP_spoof_templates) // 2)
                pair_2 = np.setdiff1d(available_indices, pair_1)

                spoof_texts_v1 = [spoof_prompts_feat[i] for i in pair_1]
                spoof_texts_v2 = [spoof_prompts_feat[i] for i in pair_2]

                spoof_texts_v1 = torch.stack(spoof_texts_v1, dim = 0)
                spoof_texts_v2 = torch.stack(spoof_texts_v2, dim = 0)

                text_embedding_v1.append(spoof_texts_v1.mean(dim = 0))
                text_embedding_v2.append(spoof_texts_v2.mean(dim = 0))
            
            elif label == 1:
                available_indices = np.arange(0, len(FLIP_real_templates))
                pair_1 = np.random.choice(available_indices, len(FLIP_real_templates) // 2)
                pair_2 = np.setdiff1d(available_indices, pair_1)

                real_texts_v1 = [real_prompts_feat[i] for i in pair_1]
                real_texts_v2 = [real_prompts_feat[i] for i in pair_2]

                real_texts_v1 = torch.stack(real_texts_v1, dim = 0)
                real_texts_v2 = torch.stack(real_texts_v2, dim = 0)

                text_embedding_v1.append(real_texts_v1.mean(dim = 0))
                text_embedding_v2.append(real_texts_v2.mean(dim = 0))

        text_embed_v1 = torch.stack(text_embedding_v1, dim = 0)
        text_embed_v2 = torch.stack(text_embedding_v2, dim = 0)


        text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim = -1, keepdim = True)
        text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim = -1, keepdim = True)

        aug1_text_dot_product = F.cosine_similarity(aug1_feat, text_embed_v1_norm)
        aug2_text_dot_product = F.cosine_similarity(aug2_feat, text_embed_v2_norm)
        ### 

        return logits, out_aug1_feat, out_aug2_feat, aug1_text_dot_product, aug2_text_dot_product


class Flip(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        clipModel = LoadClip(cfg)
        self.cfg = cfg

        self.model = CustomCLIP(cfg, clip_model)
        print("Turning on gradients in both the image and the text encoder")


    def forward(self, aug_img, aug1_img, aug2_img, label):

        return self.model(aug_img, aug1_img, aug2_img, label)
