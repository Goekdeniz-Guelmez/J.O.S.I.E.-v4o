from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from encoder2.models.helpers import RMSNorm


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = (qkv[0], qkv[1], qkv[2])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MLP(nn.Module):
    def __init__( self, dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class EncoderTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()
        self.attn = attn_target()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim)
        
        self.layer_scale_type = layer_scale_type
        if layer_scale_type:
            gamma_shape = [1, 1, dim] if layer_scale_type == "per_channel" else [1, 1, 1]
            self.layer_scale_gamma1 = nn.Parameter(torch.ones(size=gamma_shape) * layer_scale_init_value, requires_grad=True)
            self.layer_scale_gamma2 = nn.Parameter(torch.ones(size=gamma_shape) * layer_scale_init_value, requires_grad=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_mask)) * self.layer_scale_gamma1
            x = x + self.drop_path(self.mlp(self.norm2(x))) * self.layer_scale_gamma2
        return x
    


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4
    ):
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        self.post_transformer_layer = post_transformer_layer
        
        if drop_path_type == "progressive":
            dpr = torch.linspace(0, drop_path_rate, num_blocks).tolist()
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate] * num_blocks
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.ModuleList([
            EncoderTransformerBlock(
                dim=embed_dim,
                attn_target=attn_target,
                drop_path=dpr[i],
                layer_scale_type=layer_scale_type,
                layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(num_blocks)
        ])

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor = None):
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)

        for blk_id, blk in enumerate(self.blocks):
            tokens = blk(tokens, attn_mask=attn_mask)

        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)

        return tokens