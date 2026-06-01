# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import natten
from natten.functional import na2d_qk, na2d_av
from natten import NeighborhoodAttention2D as NeighborhoodAttention
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class MixedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 kernel_size=5, dilation=1, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Shared QKV projection for both attention mechanisms
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # TTT attention parameters
        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        
        # TTT q,k rescaling parameters
        self.qk_scale_ttt = nn.Parameter(torch.ones(1, 1, dim, 2))
        self.qk_offset_ttt = nn.Parameter(torch.zeros(1, 1, dim, 2))
        
        # Neighborhood Attention parameters
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Added: 2D convolution for image features
        # Apply convolution over the full dim
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

    def _rescale_qk(self, q, k):
        """
        Rescale and reshift q, k for TTT attention
        """
        qk_scale = self.qk_scale_ttt
        qk_offset = self.qk_offset_ttt
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k

    def gradient(self, k, v, w1, w2):
        """
        TTT gradient computation
        """
        # Forward pass
        z = k @ w1
        sig = F.sigmoid(z)
        s = z * sig
        y = s @ w2

        # Backward pass
        e = (y - v) / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ ((e @ w2.transpose(-2, -1)) * (sig * (1.0 + z * (1.0 - sig))))
        g2 = s.transpose(-2, -1) @ e

        return g1, g2

    def forward(self, x, img_size=None):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        # Shared QKV computation
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        
        # ============ TTT Attention ============
        # Rescale q, k for TTT
        q_ttt, k_ttt = self._rescale_qk(q, k)
        
        # Reshape for multi-head attention (for TTT)
        q_ttt = q_ttt.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ttt = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # Process k for TTT (after rescaling)
        k_ttt = F.instance_norm(k_ttt.transpose(1, 2).unsqueeze(-1), eps=1.0)
        k_ttt = k_ttt.reshape(B, self.num_heads, C // self.num_heads, N).transpose(2, 3)

        # ===== Q FULL-DIM CONV =====
        q_flat = q_ttt.transpose(1, 2).reshape(B, N, C)      # (B, N, dim)
        q2 = q_flat.transpose(1, 2).reshape(B, C, H, W)  # (B, dim, H, W)
        q2 = q2 + self.q_conv2d(q2)                      # full-dim conv
        q2 = q2.reshape(B, C, N).transpose(1, 2)         # (B, N, dim)
        q2 = q2.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        # ===== K FULL-DIM CONV =====
        k_flat = k_ttt.transpose(1, 2).reshape(B, N, C)
        k2 = k_flat.transpose(1, 2).reshape(B, C, H, W)
        k2 = k2 + self.k_conv2d(k2)
        k2 = k2.reshape(B, C, N).transpose(1, 2)
        k2 = k2.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # TTT gradient update
        w1, w2 = self.w1, self.w2
        g1, g2 = self.gradient(k2, v_ttt, w1, w2)
        w1_updated = w1 - g1
        w2_updated = w2 - g2
        
        # TTT output
        x_ttt = F.silu(q2 @ w1_updated) @ w2_updated
        x_ttt = x_ttt.transpose(1, 2).reshape(B, N, C)
        
        # ============ Neighborhood Attention ============
        # Determine spatial dimensions
        if img_size is not None:
            H, W = img_size
        else:
            # Assume square patches
            H = W = int(math.sqrt(N))
            assert H * W == N, f"Cannot determine spatial dimensions for N={N}"
        
        # Reshape to 4D for NeighborhoodAttention
        qkv_4d = qkv.reshape(B, H, W, 3 * C)
        q_na, k_na, v_na = torch.chunk(qkv_4d, chunks=3, dim=-1)
        
        # Reshape for neighborhood attention
        q_na = q_na.reshape(B, H, W, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)
        k_na = k_na.reshape(B, H, W, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)
        v_na = v_na.reshape(B, H, W, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)
        
        # Compute neighborhood attention
        q_na = q_na * self.scale
        attn = na2d_qk(q_na, k_na, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_na = na2d_av(attn, v_na, self.kernel_size, self.dilation)
        
        # Reshape back
        x_na = x_na.permute(0, 2, 3, 1, 4).reshape(B, H * W, C)
        
        # ============ Mix the outputs ============
        # Average of TTT and Neighborhood attention
        x_mixed = 0.5 * x_na + 0.5 * x_ttt
        
        # Final projection
        x_mixed = self.proj(x_mixed)
        x_mixed = self.proj_drop(x_mixed)
        
        return x_mixed

class TTTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.w1 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, num_heads, head_dim, head_dim))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Added: 2D convolution for image features
        # Apply convolution over the full dim
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

        

    def gradient(self, k, v, w1, w2):
        z = k @ w1
        sig = torch.sigmoid(z)
        s = z * sig
        y = s @ w2

        e = (y - v) / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ ((e @ w2.transpose(-2, -1)) * (sig * (1.0 + z * (1.0 - sig))))
        g2 = s.transpose(-2, -1) @ e
        return g1, g2

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        k = F.instance_norm(k.transpose(1, 2).unsqueeze(-1), eps=1.0)
        k = k.reshape(B, self.num_heads, C // self.num_heads, N).transpose(2, 3)

        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # v: (B,Hd,N,h)

       

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
  

        # Fast weights operate on the convolved v2
        w1, w2 = self.w1, self.w2
        g1, g2 = self.gradient(k2, v, w1, w2)
        w1, w2 = w1 - g1, w2 - g2

        x = F.silu(q2 @ w1) @ w2
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
       
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MixedAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


