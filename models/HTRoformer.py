from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from models.bs_roformer.attend import Attend
from models.bs_roformer.bs_roformer import (
    DEFAULT_FREQS_PER_BANDS,
    BandSplit,
    FeedForward,
    MaskEstimator,
    RMSNorm,
    default,
)

#: bridges are inserted after these spectral layers (12 layers -> 4 bridges)
ROFORMER_BLOCKS = (0, 3, 6, 9)
TIME_KERNEL = 16


# --------------------------------------------------------------------------- #
# Logic-specific RoPE / Attention / Transformer (with shared biases)
# --------------------------------------------------------------------------- #

class LogicRotaryEmbedding(Module):
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


class LogicAttention(Module):
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

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        # head-wise sigmoid gate
        gates = self.to_gates(x).sigmoid().transpose(1, 2).unsqueeze(-1)   # (b, h, n, 1)
        out = out * gates

        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class LogicTransformer(Module):
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
                LogicAttention(
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


# --------------------------------------------------------------------------- #
# Dual-branch time modules & cross-bridges
# --------------------------------------------------------------------------- #

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
        self.heads, self.dim_head = heads, dim_head
        self.dropout = dropout
        inner = heads * dim_head
        self.norm = RMSNorm(dim)
        self.norm_ctx = RMSNorm(dim)
        self.q = nn.Linear(dim, inner, bias=(shared_q_bias is not None))
        self.kv = nn.Linear(dim, inner * 2, bias=(shared_kv_bias is not None))
        if shared_q_bias is not None:
            self.q.bias = shared_q_bias
        if shared_kv_bias is not None:
            self.kv.bias = shared_kv_bias
        self.out = nn.Linear(inner, dim, bias=(shared_out_bias is not None))
        if shared_out_bias is not None:
            self.out.bias = shared_out_bias
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, context):
        b, n = x.shape[:2]
        q = self.q(self.norm(x)).view(b, n, self.heads, self.dim_head).transpose(1, 2)
        kv = self.kv(self.norm_ctx(context)).view(b, n, 2, self.heads, self.dim_head)
        k, v = kv.unbind(dim=2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.out(out) * self.scale


class CrossBridge(Module):
    """Time self-attention + band-mean spectral context, then cross attention in both directions."""

    def __init__(self, dim, heads, dim_head, rotary_embed, shared_qkv_bias, shared_out_bias,
                 shared_q_bias, shared_kv_bias, attn_dropout=0., ff_dropout=0.,
                 ff_mult=4, flash_attn=True):
        super().__init__()
        self.time_self_attn = LogicAttention(
            dim, heads=heads, dim_head=dim_head, dropout=attn_dropout,
            shared_qkv_bias=shared_qkv_bias, shared_out_bias=shared_out_bias,
            rotary_embed=rotary_embed, flash=flash_attn,
        )
        self.time_self_ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.freq_to_time_proj = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim))
        self.time_cross = CrossAttention(dim, heads, dim_head, attn_dropout,
                                         shared_q_bias, shared_kv_bias, shared_out_bias)
        self.freq_cross = CrossAttention(dim, heads, dim_head, attn_dropout,
                                         shared_q_bias, shared_kv_bias, shared_out_bias)

    def forward(self, x_time, x_freq):
        t = x_time + self.time_self_attn(x_time)
        t = t + self.time_self_ff(t)

        context = self.freq_to_time_proj(x_freq.mean(dim=2))          # band mean -> time-side context
        x_time = t + self.time_cross(t, context)

        b, n, f, d = x_freq.shape
        flat = x_freq.permute(0, 2, 1, 3).reshape(b * f, n, d)        # one sequence per band
        freq_ctx = x_time.unsqueeze(1).expand(b, f, n, d).reshape(b * f, n, d)
        x_freq = (flat + self.freq_cross(flat, freq_ctx)).reshape(b, f, n, d).permute(0, 2, 1, 3)
        return x_time, x_freq


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #

class LogicDrumSplitter(Module):
    """Dual-branch drum separation: STFT mask head + time-domain residual head, 6 drums."""

    def __init__(
        self,
        dim=256,
        *,
        depth=12,
        stereo=True,
        num_stems=6,
        heads=8,
        dim_head=64,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4,
        flash_attn=True,
        mlp_expansion_factor=4,
        mask_estimator_depth=2,
        use_shared_bias=True,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        stft_n_fft=2048,
        stft_hop_length=1024,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn=None,
        bridge_layers=ROFORMER_BLOCKS,
        time_channels=(64, 128, 192, 256),
        time_strides=(8, 8, 16),
        time_kernel=TIME_KERNEL,
        output_mode="sum",
        standardize_eps=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.output_mode = output_mode
        self.standardize_eps = standardize_eps
        self.time_channels = tuple(time_channels)
        self.time_strides = tuple(time_strides)
        self.stride_product = (
            self.time_strides[0] * self.time_strides[1] * self.time_strides[2]
        )
        self.bridge_layers = tuple(bridge_layers)
        freqs_per_bands = tuple(freqs_per_bands)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        # four shared biases: qkv / out / cross-q / cross-kv
        inner = heads * dim_head
        if use_shared_bias:
            self.linear_62_bias_0 = nn.Parameter(torch.ones(inner * 3))
            self.linear_64_bias_0 = nn.Parameter(torch.ones(dim))
            self.linear_78_bias_0 = nn.Parameter(torch.ones(inner))
            self.linear_79_bias_0 = nn.Parameter(torch.ones(inner * 2))
        else:
            self.linear_62_bias_0 = None
            self.linear_64_bias_0 = None
            self.linear_78_bias_0 = None
            self.linear_79_bias_0 = None

        # ---- spectral branch -------------------------------------------------
        self.rotary_embed = LogicRotaryEmbedding(dim=dim_head)
        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
            flash_attn=flash_attn,
            norm_output=False,
            shared_qkv_bias=self.linear_62_bias_0,
            shared_out_bias=self.linear_64_bias_0,
        )
        self.layers = ModuleList([
            ModuleList([
                LogicTransformer(depth=time_transformer_depth, rotary_embed=self.rotary_embed, **transformer_kwargs),
                LogicTransformer(depth=freq_transformer_depth, rotary_embed=self.rotary_embed, **transformer_kwargs),
            ]) for _ in range(depth)
        ])
        self.cross_bridges = ModuleList([
            CrossBridge(
                dim, heads, dim_head, self.rotary_embed,
                self.linear_62_bias_0, self.linear_64_bias_0,
                self.linear_78_bias_0, self.linear_79_bias_0,
                attn_dropout, ff_dropout, ff_mult=ff_mult, flash_attn=flash_attn,
            ) for _ in self.bridge_layers
        ])

        n_bins = stft_win_length // 2 + 1
        assert sum(freqs_per_bands) == n_bins, f'freqs_per_bands must cover {n_bins} bins'
        dim_inputs = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        self.final_norm = RMSNorm(dim)
        self.mask_estimators = ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dim_inputs, depth=mask_estimator_depth,
                          mlp_expansion_factor=mlp_expansion_factor)
            for _ in range(num_stems)
        ])

        # ---- time branch -----------------------------------------------------
        c0, c1, c2, c3 = self.time_channels
        self.time_input_conv = nn.Sequential(
            nn.Conv1d(self.audio_channels, c0, 7, padding=3),
            nn.GroupNorm(1, c0),
            nn.GELU(),
        )
        self.time_encoder = ModuleList([
            TimeEncoderStage(c0, c1, self.time_strides[0], time_kernel=time_kernel),
            TimeEncoderStage(c1, c2, self.time_strides[1], time_kernel=time_kernel),
            TimeEncoderStage(c2, c3, self.time_strides[2], time_kernel=time_kernel),
        ])
        self.time_decoder = ModuleList([
            TimeDecoderStage(c3, c2, self.time_strides[2], time_kernel=time_kernel),
            TimeDecoderStage(c2, c1, self.time_strides[1], time_kernel=time_kernel),
            TimeDecoderStage(c1, c0, self.time_strides[0], time_kernel=time_kernel),
        ])
        self.time_output_conv = nn.Conv1d(c0, num_stems * self.audio_channels, 7, padding=3)

    # ------------------------------------------------------------------ #
    def forward(self, raw_audio):
        """raw_audio: (b, channels, length) -> (b, num_stems, channels, length)."""
        device = raw_audio.device
        x_is_mps = device.type == "mps"

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        batch, channels, length = raw_audio.shape
        assert channels == self.audio_channels, f'expected {self.audio_channels} channels, got {channels}'
        assert length % self.stride_product == 0, (
            f'length must be a multiple of {self.stride_product} '
            f'(time branch strides {self.time_strides}), got {length}'
        )

        # standardise the whole mixture; the time branch re-applies the scale on output
        mean = raw_audio.mean(dim=(1, 2), keepdim=True)
        std = raw_audio.var(dim=(1, 2), unbiased=False, keepdim=True).sqrt()
        normalized = (raw_audio - mean) / (std + self.standardize_eps)

        # ---- STFT ------------------------------------------------------------
        stft_window = self.stft_window_fn(device=device)
        flat = raw_audio.reshape(batch * channels, length)
        try:
            spec = torch.stft(flat, **self.stft_kwargs, window=stft_window, return_complex=True)
        except RuntimeError:
            _flat = flat.cpu() if x_is_mps else flat
            _window = stft_window.cpu() if x_is_mps else stft_window
            spec = torch.stft(_flat, **self.stft_kwargs, window=_window, return_complex=True).to(device)

        n_bins, frames = spec.shape[-2], spec.shape[-1]
        spec = spec.view(batch, channels, n_bins, frames)

        # band-split input: (b, t, f*s*2), freq major, then channel, real/imag last
        feats = torch.view_as_real(spec).permute(0, 3, 2, 1, 4).reshape(batch, frames, -1)

        # ---- time branch -----------------------------------------------------
        hidden = self.time_input_conv(normalized)
        skips = [hidden]
        for stage in self.time_encoder:
            hidden = stage(hidden)
            skips.append(hidden)
        hidden = hidden.transpose(1, 2)                       # (b, t, dim)

        # ---- spectral branch + bridges ---------------------------------------
        x = self.band_split(feats)
        bridge_index = 0
        for index, (time_transformer, freq_transformer) in enumerate(self.layers):
            b, t, f, d = x.shape

            x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
            x = time_transformer(x)
            x = x.reshape(b, f, t, d).permute(0, 2, 1, 3)

            x = x.reshape(b * t, f, d)
            x = freq_transformer(x)
            x = x.reshape(b, t, f, d)

            if index in self.bridge_layers:
                hidden, x = self.cross_bridges[bridge_index](hidden, x)
                bridge_index += 1

        x = self.final_norm(x)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        # (b, n, t, f*s*2) -> (b, n, f*s, t, 2)
        mask = mask.reshape(batch, self.num_stems, frames, -1, 2).permute(0, 1, 3, 2, 4)
        spec_ri = torch.view_as_real(spec).permute(0, 2, 1, 3, 4).reshape(batch, 1, n_bins * channels, frames, 2)
        masked = torch.view_as_complex(spec_ri.contiguous()) * torch.view_as_complex(mask.contiguous())

        # ---- time decoder ----------------------------------------------------
        hidden = hidden.transpose(1, 2)                       # (b, dim, t)
        for stage, skip in zip(self.time_decoder, (skips[2], skips[1], skips[0])):
            hidden = stage(hidden, skip)
        residual = self.time_output_conv(hidden)
        residual = residual.view(batch, self.num_stems, self.audio_channels, length)
        residual = residual * std.view(-1, 1, 1, 1) + mean.view(-1, 1, 1, 1)

        # ---- back to audio ---------------------------------------------------
        spectral = torch.view_as_real(masked).view(batch, self.num_stems, n_bins, channels, frames, 2)
        spectral = spectral.permute(0, 1, 3, 2, 4, 5).reshape(batch * self.num_stems * channels, n_bins, frames, 2)
        spectral = torch.view_as_complex(spectral.contiguous())
        try:
            spectral = torch.istft(spectral, **self.stft_kwargs, window=stft_window,
                                   return_complex=False, length=length)
        except RuntimeError:
            _spec = spectral.cpu() if x_is_mps else spectral
            _window = stft_window.cpu() if x_is_mps else stft_window
            spectral = torch.istft(_spec, **self.stft_kwargs, window=_window,
                                   return_complex=False, length=length).to(device)
        spectral = spectral.view(batch, self.num_stems, self.audio_channels, length)

        if self.output_mode == "spectral":
            return spectral
        if self.output_mode == "time":
            return residual
        return spectral + residual