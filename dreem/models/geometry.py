"""Position terms that reach the association score directly, not only through the tokens.

`Transformer` builds these from `embedding_meta` keys (all off by default):

- `joint` -> `JointRFF`: one random-Fourier-feature embedding over the box coordinates
  and the frame offset together. The token dot product then approximates a joint
  space-time kernel, where separate pos and temp blocks give a sum of a space kernel and
  a time kernel.
- `rope` -> `RotaryXYT`: rotary position embedding over (y, x, t) with learned
  frequencies, applied to the association head's q and k. The positional part of q.k
  then depends only on the differences (dy, dx, dt), whatever the layers did to the
  tokens, and sharing each frequency between space and time lets the spatial tolerance
  change with the gap.
- `dist_bias` -> `DistanceKernelBias`: adds `-d^2 / 2 v(dt)` to every logit, with
  `v(dt) = s0^2 + s1^2 |dt| + s2^2 dt^2` learned (a diffusion-like spread around the last
  position), so the network only has to learn what position does not explain.

Parameters of `RotaryXYT` and `DistanceKernelBias` are marked `_geom = True`;
`GTRRunner.configure_optimizers` gives them their own group with no weight decay and the
optimizer's `geom_lr`.
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
        [(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2], dim=-1
    )  # N, A, 2
    cen = torch.nan_to_num(torch.nanmean(cen, dim=1), 0.0)
    scale = torch.tensor(
        [float(max(i.frame.img_shape[1], i.frame.img_shape[2])) for i in instances],
        dtype=cen.dtype,
        device=cen.device,
    )
    return cen / scale[:, None]


class JointRFF(nn.Module):
    """Random Fourier features over (box coords, warped frame offset) jointly.

    `<f(a), f(b)> ~ exp(-|box_a - box_b|^2 / 2 sigma_pos^2) * exp(-(g_a - g_b)^2 / 2 sigma_temp^2)`
    where `g = L tanh(t / L)`, `L = 3 sigma_temp` is the warp the separate RFF temporal
    embedding uses. Same scaling as the separate RFF embeddings (unit norm). Frozen.
    """

    def __init__(
        self, features: int, n_points: int = 1, sigma_pos: float = 0.02, sigma_temp: float = 20.0
    ):
        """Initialize the joint embedding.

        Args:
            features: Output dimension (even).
            n_points: Anchors per box (4 coords each).
            sigma_pos: Kernel width over box coordinates (normalized units, as RFF pos).
            sigma_temp: Kernel width over the warped frame offset (frames, as RFF temp).
        """
        super().__init__()
        if features % 2:
            raise ValueError(f"JointRFF requires even `features`; got {features}")
        self.features = features
        self.sigma_pos = float(sigma_pos)
        self.sigma_temp = float(sigma_temp)
        self.register_buffer("omega", torch.randn(4 * n_points + 1, features // 2))

    def forward(self, boxes: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Embed boxes (N, n_points, 4), normalized as `get_boxes`, and offsets (N,)."""
        n = boxes.shape[0]
        coords = boxes.reshape(n, -1).float() / self.sigma_pos
        L = 3.0 * self.sigma_temp
        g = L * torch.tanh(times.reshape(-1).float() / L) / self.sigma_temp
        ang = torch.cat([coords, g[:, None]], dim=-1) @ self.omega
        n_freq = self.omega.shape[1]
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1) / math.sqrt(n_freq)


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
    ):
        """Initialize the rotary embedding.

        Args:
            dim: Feature dimension of q and k (even).
            sigma_space: Smallest and largest spatial scale at init (normalized units).
            sigma_time: Smallest and largest temporal scale at init (frames).
        """
        super().__init__()
        if dim % 2:
            raise ValueError(f"RotaryXYT requires even `dim`; got {dim}")
        n = dim // 2

        def scales(lo, hi):
            return torch.logspace(math.log10(lo), math.log10(hi), n)[torch.randperm(n)]

        s, t = scales(*sigma_space), scales(*sigma_time)
        omega = torch.cat([torch.randn(n, 2) / s[:, None], torch.randn(n, 1) / t[:, None]], 1)
        self.omega = nn.Parameter(omega)  # (n, 3): (y, x, t) frequency of each pair
        self.omega._geom = True

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Rotate x (..., N, dim) by positions pos (N, 3) = (y, x, t)."""
        ang = pos.to(x.dtype) @ self.omega.T  # N, n
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class DistanceKernelBias(nn.Module):
    """Logit bias `-|c_q - c_r|^2 / 2 v(dt)`, `v = s0^2 + s1^2 |dt| + s2^2 dt^2`, s learned.

    With `normalize`, it is the 2-d Gaussian log-density up to a constant,
    `-|c_q - c_r|^2 / 2 v(dt) - log(v(dt) / v(1))`. Without the normaliser, a reference
    seen long ago has a wide kernel and scores near 0 at any distance, like a perfect
    match; the tracker compares tracks last seen at different gaps, so this matters there
    (and against the loss's fixed background logit), not within one reference frame.
    """

    def __init__(
        self, s0: float = 0.01, s1: float = 0.01, s2: float = 0.005, normalize: bool = False
    ):
        """Initialize the kernel.

        Args:
            s0: Spread at zero gap (normalized units, `get_centroids`).
            s1: Spread growing with sqrt(gap) (random-walk term).
            s2: Spread growing with the gap (velocity term).
            normalize: Add the Gaussian normaliser `-log(v(dt) / v(1))`.
        """
        super().__init__()
        self.log_s = nn.Parameter(torch.tensor([s0, s1, s2], dtype=torch.float32).log())
        self.log_s._geom = True
        self.normalize = normalize

    def forward(
        self, cq: torch.Tensor, cr: torch.Tensor, tq: torch.Tensor, tr: torch.Tensor
    ) -> torch.Tensor:
        """Bias (Nq, Nr) from centroids cq (Nq, 2), cr (Nr, 2) and offsets tq (Nq,), tr (Nr,)."""
        d2 = ((cq[:, None, :] - cr[None, :, :]) ** 2).sum(-1)
        dt = (tq.reshape(-1, 1) - tr.reshape(1, -1)).abs().to(d2.dtype)
        s2 = torch.exp(2 * self.log_s)
        v = s2[0] + s2[1] * dt + s2[2] * dt**2
        if self.normalize:
            return -d2 / (2 * v) - torch.log(v / s2.sum())
        return -d2 / (2 * v)
