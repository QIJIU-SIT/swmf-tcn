"""Losses used by every HSLF Stage 1 candidate."""

from __future__ import annotations

from typing import Iterable

import torch


DEFAULT_HORIZON_GROUPS: tuple[tuple[int, int], ...] = ((0, 4), (4, 24), (24, 48), (48, 96))


def masked_horizon_balanced_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    groups: Iterable[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Average L1 within each non-empty horizon group, then average groups."""
    if prediction.shape != target.shape or prediction.shape != target_mask.shape:
        raise ValueError("prediction, target and target_mask must have identical shapes")
    if prediction.ndim != 2:
        raise ValueError("prediction and target must be [batch, horizon]")
    if not bool(target_mask.any()):
        raise ValueError("empty target mask")
    group_losses = []
    for start, end in (tuple(groups) if groups is not None else DEFAULT_HORIZON_GROUPS):
        group_mask = target_mask[:, start:end]
        if not bool(group_mask.any()):
            continue
        error = (prediction[:, start:end] - target[:, start:end]).abs()
        group_losses.append(error[group_mask].mean())
    if not group_losses:
        raise ValueError("empty target mask")
    return torch.stack(group_losses).mean()
