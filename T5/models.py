# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.helpers import named_apply
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.eps
        return x / rms 
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))

        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        
        # Added: 2D convolution (image convolution)
       
        self.q_conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim   # depthwise over full dim
        )
        self.k_conv2d = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )


    def gradient_silu_glu(self, k, v, w1, w2, w3):
        # forward
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z1)
        a1 = z1 * sig
        a2 = a1 * z2
        # backward
        e = - v / float(v.shape[2]) * self.scale
        da2 = e @ w3.transpose(-2, -1)
        g1 = k.transpose(-2, -1) @ (da2 * z2 * (sig * (1.0 + z1 * (1.0 - sig))))
        g2 = k.transpose(-2, -1) @ (da2 * a1)
        g3 = a2.transpose(-2, -1) @ e

        return g1, g2, g3
    


    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = F.instance_norm(k.transpose(1, 2).unsqueeze(-1), eps=1.0)
        k = k.reshape(B, self.num_heads, C // self.num_heads, N).transpose(2, 3)

        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # v: (B,Hd,N,h)

        # 2D convolution processing for v

        # ===== Q FULL-DIM CONV =====
        q_flat = q.transpose(1, 2).reshape(B, N, C)      # (B, N, dim)
        q2 = q_flat.transpose(1, 2).reshape(B, C, H, W)  # (B, dim, H, W)
        q2 = q2 + self.q_conv2d(q2)                      # full-dim conv
        q2 = q2.reshape(B, C, N).transpose(1, 2)         # (B, N, dim)
        q2 = q2.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        # ===== K FULL-DIM CONV =====
        k_flat = k.transpose(1, 2).reshape(B, N, C)
        k2 = k_flat.transpose(1, 2).reshape(B, C, H, W)
        k2 = k2 + self.k_conv2d(k2)
        k2 = k2.reshape(B, C, N).transpose(1, 2)
        k2 = k2.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
  

        # The fast-weight update uses the convolved v2
        w1, w2, w3 = self.w1, self.w2, self.w3
        g1, g2, g3 = self.gradient_silu_glu(k2, v, w1, w2, w3)
        w1, w2, w3 = w1 - g1, w2 - g2, w3 - g3

        x = (F.silu(q2 @ w1) * (q2 @ w2)) @ w3

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone using T5 attention."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 draft_dim=64, meta_depth=3, version=1,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): whether to include a distillation token and auxiliary head
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        raise NotImplementedError('Pretrained .npz loading is not included in this simplified T5 model file.')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed',  'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x.mean(dim=1))


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """Initialize ViT/T5 model weights."""
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _create_vision_transformer(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError('Pretrained loading is not supported in this simplified T5 model file.')
    return VisionTransformer(**kwargs)


@register_model
def T5_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, meta_depth=3, num_heads=3, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, meta_depth=6, num_heads=6, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small_b12(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, meta_depth=3, num_heads=6, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small_b48(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=48, meta_depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small_l1(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=256, depth=24, meta_depth=12, num_heads=4, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small_l2(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=256, depth=24, meta_depth=8, num_heads=4, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small_l4(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=256, depth=24, meta_depth=5, num_heads=4, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_small_l5(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=256, depth=24, meta_depth=4, num_heads=4, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def T5_base(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=512, depth=24, meta_depth=6, num_heads=8, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
