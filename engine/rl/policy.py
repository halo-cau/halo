"""Spatial Maskable policy for ``DataCenterEnv`` (theory-grounded inductive bias).

Why this architecture (each part maps to a property of rack placement)
----------------------------------------------------------------------
The task is dense spatial prediction: given a bird's-eye state grid, score every
cell and direction as a place to start a rack row. Three facts drive the design.

1. Local airflow geometry is translation invariant -- a rack's intake/exhaust
   interaction with its neighbours looks the same anywhere on the floor. So: a
   residual CONVOLUTIONAL encoder (weight sharing, locality, residual depth).

2. The hot/cold-aisle relation is global and pairwise -- a cell's quality depends
   on a rack possibly far across an aisle whose exhaust faces it, and on distance
   to coolers anywhere. Convolution has a bounded receptive field and cannot
   compare two distant racks. So: a global SELF-ATTENTION bottleneck on a coarse
   token grid (the relational bias of a transformer).

3. The action is per-cell at full resolution. Collapsing to a small bottleneck
   and rebuilding the map with a dense layer (NatureCNN + Linear(512->10000))
   throws the geometry away. So: the full-resolution conv features are kept on a
   skip path, fused with the upsampled attention context, and the logits come
   from a 1x1 CONVOLUTION (a fully convolutional action head).

The extractor returns the fused map flattened, since Stable Baselines3 expects a
2-D ``(batch, features_dim)`` tensor; the heads reshape it to ``(C, G, G)``. The
action-head flatten order matches the env: ``index = (gx*grid + gy)*n_dirs + dir``.
Width and depth are sized for capacity (8 GB headroom) and are the knobs to grow.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    """A GroupNorm group count that divides ``channels`` (falls back to 1)."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class _ResBlock(nn.Module):
    """Pre-activation residual conv block at constant resolution."""

    def __init__(self, channels: int):
        super().__init__()
        g = _groups(channels)
        self.body = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.body(x)


class _GlobalAttention(nn.Module):
    """Self-attention over a coarse token grid: global pairwise reasoning.

    The feature map is average-pooled to ``tokens x tokens`` tokens, given a
    learned positional embedding, run through transformer encoder layers, and
    returned at the coarse resolution (the caller upsamples and fuses). The coarse
    grid keeps the quadratic attention cost small while still giving every region
    a path to every other region.
    """

    def __init__(self, channels: int, tokens: int, heads: int, layers: int):
        super().__init__()
        self.tokens = tokens
        self.pool = nn.AdaptiveAvgPool2d(tokens)
        self.pos = nn.Parameter(th.zeros(1, tokens * tokens, channels))
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=4 * channels,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.shape
        coarse = self.pool(x)                              # (B, C, T, T)
        tok = coarse.flatten(2).transpose(1, 2) + self.pos  # (B, T*T, C)
        tok = self.encoder(tok)
        return tok.transpose(1, 2).reshape(b, c, self.tokens, self.tokens)


class SpatialFeatures(BaseFeaturesExtractor):
    """Residual-conv encoder + global-attention bottleneck + full-resolution fusion.

    Input ``(C_in, G, G)`` -> flat ``(out_channels * G * G)``; resolution preserved.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        out_channels: int = 96,
        depth: int = 6,
        attn_tokens: int = 12,
        attn_heads: int = 6,
        attn_layers: int = 3,
    ):
        c_in, gx, gy = observation_space.shape
        if gx != gy:
            raise ValueError(f"SpatialFeatures expects a square grid, got {gx}x{gy}")
        super().__init__(observation_space, features_dim=out_channels * gx * gy)
        self.out_channels = out_channels
        self.grid = gx

        self.stem = nn.Conv2d(c_in, out_channels, kernel_size=3, padding=1)
        self.encoder = nn.Sequential(*[_ResBlock(out_channels) for _ in range(depth)])
        self.attn = _GlobalAttention(out_channels, attn_tokens, attn_heads, attn_layers)
        # Fuse local full-res features with upsampled global context (U-Net skip).
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        h = self.encoder(self.stem(obs))                   # (B, C, G, G) local features
        g = self.attn(h)                                   # (B, C, T, T) global context
        g = F.interpolate(
            g, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        fused = self.fuse(th.cat([h, g], dim=1))           # (B, C, G, G)
        return fused.reshape(fused.shape[0], -1)


class _SpatialActionHead(nn.Module):
    """``(B, C*G*G)`` -> per-cell, per-dir logits ``(B, G*G*n_dirs)``."""

    def __init__(self, channels: int, grid: int, n_dirs: int):
        super().__init__()
        self.channels = channels
        self.grid = grid
        self.n_dirs = n_dirs
        self.head = nn.Conv2d(channels, n_dirs, kernel_size=1)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        b = latent.shape[0]
        x = latent.view(b, self.channels, self.grid, self.grid)   # (B, C, gx, gy)
        logits = self.head(x)                                     # (B, n_dirs, gx, gy)
        logits = logits.permute(0, 2, 3, 1).contiguous()          # (B, gx, gy, n_dirs)
        # index = (gx*grid + gy)*n_dirs + dir, matching action = pos*4 + dir in the env.
        return logits.view(b, -1)


class _SpatialValueHead(nn.Module):
    """``(B, C*G*G)`` -> ``(B, 1)`` via global average pooling + a small MLP."""

    def __init__(self, channels: int, grid: int):
        super().__init__()
        self.channels = channels
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(channels, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, latent: th.Tensor) -> th.Tensor:
        b = latent.shape[0]
        x = latent.view(b, self.channels, self.grid, self.grid)
        pooled = self.pool(x).reshape(b, self.channels)
        return self.mlp(pooled)


class SpatialMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy: conv + global-attention backbone, conv action head.

    Use it like ``"CnnPolicy"``: ``MaskablePPO(SpatialMaskablePolicy, env, ...)``.
    The feature extractor, the identity MLP extractor (so the heads read the
    spatial features directly), and ``normalize_images=False`` are wired here.
    Backbone size is exposed through ``policy_kwargs`` (``out_channels``,
    ``depth``, ``attn_tokens``, ``attn_heads``, ``attn_layers``).
    """

    def __init__(
        self,
        *args,
        out_channels: int = 96,
        depth: int = 6,
        attn_tokens: int = 12,
        attn_heads: int = 6,
        attn_layers: int = 3,
        **kwargs,
    ):
        kwargs["features_extractor_class"] = SpatialFeatures
        fe_kwargs = dict(kwargs.get("features_extractor_kwargs") or {})
        fe_kwargs.setdefault("out_channels", out_channels)
        fe_kwargs.setdefault("depth", depth)
        fe_kwargs.setdefault("attn_tokens", attn_tokens)
        fe_kwargs.setdefault("attn_heads", attn_heads)
        fe_kwargs.setdefault("attn_layers", attn_layers)
        kwargs["features_extractor_kwargs"] = fe_kwargs
        # Empty net_arch -> identity MLP extractor; conv heads read the features.
        kwargs["net_arch"] = []
        kwargs["normalize_images"] = False
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # The shared feature extractor already exists here (built in
        # ActorCriticPolicy.__init__ before _build); swap the base dense
        # action/value heads for spatial ones.
        self._build_mlp_extractor()  # identity, since net_arch == []
        channels = self.features_extractor.out_channels
        grid = self.features_extractor.grid
        n_dirs = int(self.action_space.n // (grid * grid))
        self.action_net = _SpatialActionHead(channels, grid, n_dirs)
        self.value_net = _SpatialValueHead(channels, grid)
        if self.ortho_init:
            for module, gain in {
                self.features_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
            }.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
