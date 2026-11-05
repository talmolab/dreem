"""Rotary position embedding for the association head.

`Transformer` builds `RotaryXYT` from `embedding_meta["rope"]` (off by default) and the
attention head applies it to its q and k. The positional part of each association logit
then depends only on the differences (dy, dx, dt) between a query and a reference
instance, whatever the layers did to the tokens, and sharing each frequency between space
and time lets the spatial tolerance change with the frame gap.

`fraction` < 1 rotates only that share of the feature pairs and leaves the rest as they
are, so the head can still compare appearance at any displacement.

`RotaryXYT.omega` is marked `_geom = True`; `GTRRunner.configure_optimizers` gives it its
own parameter group with no weight decay and the optimizer's `geom_lr`.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def get_centroids(instances: list) -> torch.Tensor:
    """Box centres in isotropic normalized coordinates (pixels / max(H, W)), as (y, x).

    Unlike `get_boxes`, both axes share one scale, so distances are proportional to
    pixel distances on non-square frames. Missing anchors are skipped; an instance with
    no anchor gets (0, 0).

    Args:
        instances: List of Instance objects.

    Returns:
        An (n_instances, 2) float tensor.
    """
    boxes = torch.cat([instance.bbox for instance in instances], dim=0)  # N, A, 4 (px)
    cen = torch.stack(
        [(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2],
        dim=-1,
    )  # N, A, 2
    cen = torch.nan_to_num(torch.nanmean(cen, dim=1), 0.0)
    scale = torch.tensor(
        [float(max(i.frame.img_shape[1], i.frame.img_shape[2])) for i in instances],
        dtype=cen.dtype,
        device=cen.device,
    )
    return cen / scale[:, None]


class RotaryXYT(nn.Module):
    """Rotary position embedding over (y, x, t) with learned frequencies.

    Feature pair m of a vector at position p = (y, x, t) is rotated by angle
    `omega_m . p`, so `<R(p_i) q, R(p_j) k>` depends on p only through `p_i - p_j`.
    Frequencies start as random directions at log-spaced scales (space in normalized
    coordinates, time in frames), each pair with its own space and time scale.
    """

    def __init__(
        self,
        dim: int,
        sigma_space: tuple[float, float] = (0.01, 0.3),
        sigma_time: tuple[float, float] = (2.0, 64.0),
        fraction: float = 1.0,
    ):
        """Initialize the rotary embedding.

        Args:
            dim: Feature dimension of q and k (even).
            sigma_space: Smallest and largest spatial scale at init (normalized units).
            sigma_time: Smallest and largest temporal scale at init (frames).
            fraction: Share of the dim / 2 feature pairs that are rotated; the first
                `round(fraction * dim / 2)` pairs are, the others pass through unchanged.
        """
        super().__init__()
        if dim % 2:
            raise ValueError(f"RotaryXYT requires even `dim`; got {dim}")
        if not 0 < fraction <= 1:
            raise ValueError(f"`fraction` must be in (0, 1], got {fraction}")
        n = max(1, round(fraction * dim / 2))
        self.n_rot = n

        def scales(lo, hi):
            return torch.logspace(math.log10(lo), math.log10(hi), n)[torch.randperm(n)]

        s, t = scales(*sigma_space), scales(*sigma_time)
        omega = torch.cat(
            [torch.randn(n, 2) / s[:, None], torch.randn(n, 1) / t[:, None]], 1
        )
        self.omega = nn.Parameter(omega)  # (n, 3): (y, x, t) frequency of each pair
        self.omega._geom = True

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rotate the first `n_rot` feature pairs of x (..., N, dim) by pos (N, 3) = (y, x, t)."""
        ang = pos.to(x.dtype) @ self.omega.T  # N, n_rot
        cos, sin = torch.cos(ang), torch.sin(ang)
        rot, rest = x[..., : 2 * self.n_rot], x[..., 2 * self.n_rot :]
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(
            -2
        )
        return torch.cat([rot, rest], dim=-1)
