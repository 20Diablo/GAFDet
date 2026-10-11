from random import random

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
import numpy as np
import torch.nn.functional as F
from natten.functional import natten2dav, natten2dqkrpb

from natten import NeighborhoodAttention2D


class NeighborhoodCrossAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=True,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert (
                kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
                dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        B, Hp, Wp, C = q.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape

        q = self.q(q).reshape(B, H, W, 1, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5).squeeze(
            0)  # [1, B, num_head, H, W, head_dim]
        kv = self.kv(kv).reshape(B, H, W, 2, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
                f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
                + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
                + f"rel_pos_bias={self.rpb is not None}"
        )


class Attention(nn.Module):
    def __init__(self, embed_dims, kernel_size=5):
        super(Attention, self).__init__()
        self.neighbor_atttention = NeighborhoodCrossAttention2D(
            dim=embed_dims, num_heads=1, kernel_size=kernel_size, bias=True, qkv_bias=True
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, q, kv):
        """
        q: b, c, h, w
        k: b, c, h, w
        """
        q = q.permute(0, 2, 3, 1)
        kv = kv.permute(0, 2, 3, 1)
        proj_value = self.neighbor_atttention(q, kv)
        out = self.gamma * proj_value + kv
        return out.permute(0, 3, 1, 2)

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.attention = Attention(80, 5)
    def forward(self, x, y):

        result = self.attention(x,y)
        return result

x = torch.randn(6,80,70,44, dtype=torch.float32)
y = torch.randn(6,80,16,44, dtype=torch.float32)
attention_scores = torch.matmul(x, y.transpose(1, 2))
# 计算注意力权重（softmax）
attention_weights = F.softmax(attention_scores, dim=-1)  # [B*N*W, H, D]

# 用注意力权重加权值（V）
output = torch.matmul(attention_weights, y)  # [B*N*W, H, C]

print(output.shape)  # 输出: [B*N*W, H, C]

# test1 = test()
# print(test1(x,y))