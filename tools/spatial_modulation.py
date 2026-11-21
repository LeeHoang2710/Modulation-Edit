"""Utilities for spatially composing SharedAdaLN transformations.

Infinity flattens each ``(t, h, w)`` scale with width as the fastest-changing
coordinate.  The helpers here preserve that ordering, so a modulation vector
placed on the left/right side remains aligned with the corresponding attention
output token.  All six vectors are composed together: SA/FFN residual gates,
scales, and shifts.
"""
from __future__ import annotations

from typing import Sequence

import torch


def horizontal_region_weights(
    scale_shape: Sequence[int],
    split_ratio: float = 0.5,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return left-region weights in Infinity's flattened token order.

    A token receives the fraction of its column cell lying left of the split.
    Consequently even-width scales produce a hard half/half partition, while a
    one-column scale receives a 50/50 mixture at the default split.  The latter
    is necessary because the coarsest autoregressive scale cannot be divided
    into two spatial tokens.
    """
    if len(scale_shape) != 3:
        raise ValueError(f"expected a (t, h, w) scale, got {tuple(scale_shape)}")
    t, h, w = (int(value) for value in scale_shape)
    if min(t, h, w) <= 0:
        raise ValueError(f"scale dimensions must be positive, got {(t, h, w)}")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")

    # Each discrete column represents [j / w, (j + 1) / w].  Measuring its
    # overlap with [0, split_ratio] gives exact hard halves when w is even and
    # a principled boundary blend otherwise.
    columns = torch.arange(w, device=device, dtype=dtype)
    left_edge = columns / w
    right_edge = (columns + 1) / w
    column_weights = (torch.minimum(right_edge, torch.as_tensor(split_ratio, device=device, dtype=dtype))
                      - left_edge).clamp_(0, 1 / w) * w
    return column_weights.view(1, 1, w).expand(t, h, w).reshape(t * h * w, 1, 1)


def compose_horizontal_modulation(
    left: torch.Tensor,
    right: torch.Tensor,
    scale_shape: Sequence[int],
    split_ratio: float = 0.5,
) -> torch.Tensor:
    """Compose left/right ``(B, 1, 6, C)`` transformations token-wise."""
    if left.ndim != 4 or right.ndim != 4 or left.shape[1] != 1 or right.shape[1] != 1:
        raise ValueError("left and right must have shape (B, 1, 6, C)")
    if left.shape != right.shape or left.shape[2] != 6:
        raise ValueError(f"left/right shapes must match with six vectors, got {left.shape} and {right.shape}")
    weights = horizontal_region_weights(
        scale_shape, split_ratio, device=left.device, dtype=left.dtype
    ).unsqueeze(0)
    return weights * left + (1 - weights) * right


def scale_offset(offset: torch.Tensor, scale_index: int) -> torch.Tensor:
    """Select a shared or per-scale checkpoint offset as ``(1, 1, 6, C)``."""
    if offset.ndim != 4 or offset.shape[1] != 1 or offset.shape[2] != 6:
        raise ValueError(f"offset must have shape (S, 1, 6, C), got {tuple(offset.shape)}")
    if offset.shape[0] == 1:
        return offset
    if not 0 <= scale_index < offset.shape[0]:
        return torch.zeros_like(offset[:1])
    return offset[scale_index:scale_index + 1]
