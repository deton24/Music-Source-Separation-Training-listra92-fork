"""HTRoFormer (Hybrid Transformer RoFormer) — single-file MSST payload (bs_roformer_custom).

Merged from Sucial/drumsep-ht-roformer (infer/bs_roformer/bs_roformer.py + ht_roformer.py)
and adapted for Music-Source-Separation-Training (BSRoformer class + MSST forward/loss interface).
"""

from contextlib import nullcontext
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from models.bs_roformer.attend import Attend


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


# --------------------------------------------------------------------------- #
# Building blocks from Sucial's bs_roformer.py (with shared bias support)
# --------------------------------------------------------------------------- #

class RotaryEmbedding(Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("freqs", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, t):
        """t: (..., seq, dim) -> phase table (seq, dim); each adjacent pair shares a phase."""
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
        freqs = positions[:, None] * self.freqs[None]
        return freqs.repeat_interleave(2, dim=-1)

    @staticmethod
    def rotate_half(t):
        t = t.reshape(*t.shape[:-1], -1, 2)
        return torch.stack((-t[..., 1], t[..., 0]), dim=-1).flatten(-2)

    def rotate_queries_or_keys(self, t):
        """Rotate (b, h, seq, d) in adjacent pairs; phases are computed in fp32 then cast back."""
        dtype = t.dtype
        freqs = self.forward(t).float()
        out = t.float() * freqs.cos() + self.rotate_half(t.float()) * freqs.sin()
        return out.to(dtype)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        shared_qkv_bias=None,
        shared_out_bias=None,
        rotary_embed=None,
        flash=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout),
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x):
        x = self.norm(x)

        b, n = x.shape[:2]
        q, k, v = self.to_qkv(x).view(b, n, 3, self.heads, -1).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (b, h, n, d)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        # head-wise sigmoid gate
        gates = self.to_gates(x).sigmoid().transpose(1, 2).unsqueeze(-1)   # (b, h, n, 1)
        out = out * gates

        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        shared_qkv_bias=None,
        shared_out_bias=None,
    ):
        super().__init__()
        self.layers = ModuleList([
            ModuleList([
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    dropout=attn_dropout,
                    shared_qkv_bias=shared_qkv_bias,
                    shared_out_bias=shared_out_bias,
                    rotary_embed=rotary_embed,
                    flash=flash_attn,
                ),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs
        ])

    def forward(self, x):
        outs = [to_feature(split_input)
                for split_input, to_feature in zip(x.split(self.dim_inputs, dim=-1), self.to_features)]
        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    net = []
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)
        net.append(nn.Linear(layer_dim_in, layer_dim_out))
        if is_last:
            continue
        net.append(activation())
    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(self, dim, dim_inputs, depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        dim_hidden = dim * mlp_expansion_factor
        self.to_freqs = ModuleList([
            nn.Sequential(MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1))
            for dim_in in dim_inputs
        ])

    def forward(self, x):
        outs = [mlp(band_features) for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs)]
        return torch.cat(outs, dim=-1)


# 62 bands: 2*24 + 4*12 + 12*8 + 24*8 + 48*8 + 128 + 129 = 1025 bins
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


# --------------------------------------------------------------------------- #
# HTRoFormer time branch & cross-bridges (from ht_roformer.py)
# --------------------------------------------------------------------------- #

ROFORMER_BLOCKS = (0, 3, 6, 9)
TIME_KERNEL = 16
TIME_CROP = TIME_KERNEL // 2


class DConvBlock(Module):
    """ConvNeXt-ish gated block: C -> C/4 (k3, optionally dilated) -> 2C (k1) -> GN -> GLU, scaled residual."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.dconv_layers = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, channels // 4),
            nn.GELU(),
            nn.Conv1d(channels // 4, channels * 2, 1),
        )
        self.norm2 = nn.GroupNorm(1, channels * 2)    # normalised before the GLU, no activation after it
        self.res_scale = nn.Parameter(torch.ones(channels, 1))

    def forward(self, x):
        h = self.norm2(self.dconv_layers(x))
        value, gate = h.chunk(2, dim=1)
        return x + self.res_scale * (value * gate.sigmoid())


class TimeEncoderStage(Module):
    """Zero-pad k/2, stride-kernel conv, GroupNorm+GELU, then two gated blocks."""

    def __init__(self, in_channels, out_channels, stride, time_kernel=TIME_KERNEL):
        super().__init__()
        self.time_crop = time_kernel // 2
        self.conv = nn.Conv1d(in_channels, out_channels, time_kernel, stride=stride)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dconv_layers = ModuleList([DConvBlock(out_channels, 1), DConvBlock(out_channels, 2)])

    def forward(self, x):
        x = F.pad(x, (self.time_crop, self.time_crop))
        x = F.gelu(self.norm(self.conv(x)))
        for block in self.dconv_layers:
            x = block(x)
        return x


class TimeDecoderStage(Module):
    """Transposed conv, crop to the skip length, concat skip, 1x1 fuse, then two gated blocks."""

    def __init__(self, in_channels, out_channels, stride, time_kernel=TIME_KERNEL):
        super().__init__()
        self.time_crop = time_kernel // 2
        self.conv_tr = nn.ConvTranspose1d(in_channels, out_channels, time_kernel, stride=stride)
        self.skip_proj = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dconv_layers = ModuleList([DConvBlock(out_channels, 1), DConvBlock(out_channels, 2)])

    def forward(self, x, skip):
        x = self.conv_tr(x)[..., self.time_crop:self.time_crop + skip.shape[-1]]
        x = torch.cat((x, skip), dim=1)
        x = F.gelu(self.norm(self.skip_proj(x)))
        for block in self.dconv_layers:
            x = block(x)
        return x


class CrossAttention(Module):
    """Queries from one branch, keys/values from the other, output scaled per channel."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 shared_q_bias=None, shared_kv_bias=None, shared_out_bias=None):
        super().__init__()
        self.heads,
