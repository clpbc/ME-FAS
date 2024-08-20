# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import torch, os
from torch import nn
import torch.nn.functional as F


from clip import clip


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

    model = clip.build_model(state_dict or model.state_dict(), cfg)

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


class ClassifierFC(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)  # 最后一层的输出大小为类别数
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MultiPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        classnames = cfg['dataset']['classname']
        self.n_cls = len(classnames)
        self.n_ctx = cfg['model']['nctx']
        ctx_init = cfg['model']['language_init']

        ctx_dim = 512
        vis_dim = 768

        if ctx_init and (self.n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std = 0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = clip.tokenize(prompts)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer('token_prefix', embedding[:, : 1, :])  # SOS
        self.register_buffer('token_suffix', embedding[:, 1 + self.n_ctx: , :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # vision proj
        self.proj = nn.Linear(ctx_dim, vis_dim)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim = 1,
        )

        return prompts

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        prompts = self.construct_prompts(ctx, prefix, suffix)
        shared_ctx = self.proj(self.ctx)  

        return prompts, shared_ctx


class JpgFas(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.clipModel = LoadClip(cfg)
        self.prompt_learner = MultiPromptLearner(cfg, self.clipModel)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.logit_scale = self.clipModel.logit_scale
        self.projection_head = ProjHead(in_dim = 512, mlp_dim = 2048, out_dim = 256)
        # self.classifier_fc = ClassifierFC()

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ['prompt_learner', 'classifier_fc', 'prompt_projections_image', 'image_projections_prompt', 'projection_head', 'text_projection_image']

        # for name, param in self.clip_model.named_parameters():
        #     if not any(name_part in name for name_part in name_to_update):
        #         param.requires_grad_(False)

    def forward(self, origin_img, aug1_img, aug2_img, labels, Type = 'train'):
        prompts, shared_ctx = self.prompt_learner()

        image = torch.cat([origin_img, aug1_img, aug2_img], dim = 0)

        img_feat, text_feat = self.clipModel(image, prompts, shared_ctx, self.tokenized_prompts, labels, Type)

        img_feat = img_feat / img_feat.norm(dim = -1, keepdim = True)
        text_feat = text_feat / text_feat.norm(dim = -1, keepdim = True)

        origin_feat = img_feat[: origin_img.shape[0]]
        aug1_feat = img_feat[origin_img.shape[0]: origin_img.shape[0] + aug1_img.shape[0]]
        aug2_feat = img_feat[-1 * aug2_img.shape[0]: ]

        ### SimCLR features
        out_aug1_feat = self.projection_head(aug1_feat)
        out_aug2_feat = self.projection_head(aug2_feat)
        ###

        ### cls loss
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * origin_feat @ text_feat.t()
        ###

        ### MSE loss
        text_embedding_v1 = []
        text_embedding_v2 = []

        for label in labels:

            if label == 0:
                text_embedding_v1.append(text_feat[0])
                text_embedding_v2.append(text_feat[0])
            
            elif label == 1:
                text_embedding_v1.append(text_feat[1])
                text_embedding_v2.append(text_feat[1])

        text_embed_v1 = torch.stack(text_embedding_v1, dim = 0)
        text_embed_v2 = torch.stack(text_embedding_v2, dim = 0)


        text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim = -1, keepdim = True)
        text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim = -1, keepdim = True)

        aug1_text_dot_product = F.cosine_similarity(aug1_feat, text_embed_v1_norm)
        aug2_text_dot_product = F.cosine_similarity(aug2_feat, text_embed_v2_norm)
        ### 

        return logits, out_aug1_feat, out_aug2_feat, aug1_text_dot_product, aug2_text_dot_product