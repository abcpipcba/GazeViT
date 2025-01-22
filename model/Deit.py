# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.pyplot as plt


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, crop=False, attention=False, save=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))   # 1*1*384

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))  # 1*(N+2)*384
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()  # self.embed_dim:384  self.num_classes:1000

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.save = save
        self.crop = crop
        self.attention = attention
        # self.crop_rate = 0.64 # keep rate: 0.53 352, 0.64 320, 0.79 288


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]   # batch_size: B
        x = self.patch_embed(x)    

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks  # cls_token: [B, 1, 384]
        dist_token = self.dist_token.expand(B, -1, -1)   # dist_token: [B, 1, 384]
        x = torch.cat((cls_tokens, dist_token, x), dim=1)  

        x = x + self.pos_embed  # x: [B, N+2, 384]
        x = self.pos_drop(x)   

        for i, blk in enumerate(self.blocks):    # blocks: Encoder Block
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]  # x[:, 0]: Class Token, size: [B, 1, 384]; x[:, 1]: dist_token, size: [B, 1, 384]

    def forward_features_save(self, x, indexes=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x_shape = x.shape  # x.size: [B, 3, H, W]

        B = x.shape[0]
        x = self.patch_embed(x)   # x.size:[B, N, 384]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)    # x.size: [B, N+2, 384]

        x = x + self.pos_embed
        x = self.pos_drop(x)    # x.size:[B, N+2, 384]

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks)-1:  
                y = blk.norm1(x)   
                B, N, C = y.shape  # y.shape:[B, N+2, 384]
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)   # attn.size: [B, 6, N+2, N+2]

                last_map = (att[:, :, :2, 2:].detach().cpu().numpy()).sum(axis=1).sum(axis=1)   
                last_map = last_map.reshape(
                    [last_map.shape[0],x_shape[2] // 16, x_shape[3] // 16])     # last_map.size: [B, 16, 16]

            x = blk(x)

        for j, index in enumerate(indexes.cpu().numpy()):
            plt.imsave(os.path.join(self.save, str(indexes[j].cpu().numpy()) + '.png'),
                np.tile(np.expand_dims(last_map[j]/ np.max(last_map[j]), 2), [1, 1, 3]))

        x = self.norm(x)

        return x[:, 0], x[:, 1]   

    def forward(self, x, atten=None, indexes=None, heatmap=None):
        if self.save is not None:
            x, x_dist = self.forward_features_save(x, indexes)
        elif self.attention:
            if heatmap is None:
                heatmap = atten
            hybrid_map = self.forward_features_hybrid(heatmap, atten)
            
        elif self.crop:
            if atten is None:
                atten = torch.zeros_like(x).cuda()   # atten.size: [B, 3, H, W]
            x, x_dist = self.forward_features_crop(x, hybrid_map) 
            
        else:
            x, x_dist = self.forward_features(x)   

        x = self.head(x)
        x_dist = self.head_dist(x_dist)

        return (x + x_dist) / 2

@register_model
def deit_small_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        
        # resize the positional embedding
        # taken from https: //github.com/Jeff-Zilence/TransGeo2022
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))    # permute: [B, H, W, C]->[B, C, H, W]
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])    # permute: [B, C, H, W]->[B, H, W, C]    reshape: [B, H, W, C]->[B, H*W, C]
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)  
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
    return model

