"""Immutable protocol-v2 data and prediction semantics."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import pandas as pd
import torch


DO_FEATURE_NAME = "water:DO_mgl"
PRIMARY_HORIZON = 96
PRIMARY_METRIC = "h96_physical_station_macro_mae"


class NativeOutput(str, Enum):
    STANDARDIZED_ABSOLUTE_DO = "standardized_absolute_do"
    STANDARDIZED_DO_DELTA = "standardized_do_delta"


def resolve_do_channel(feature_names: Sequence[str]) -> int:
    names = tuple(str(name) for name in feature_names)
    matches = [index for index, name in enumerate(names) if name == DO_FEATURE_NAME]
    if len(matches) != 1:
        raise ValueError(f"feature order must contain exactly one {DO_FEATURE_NAME}")
    return matches[0]


def adapt_native_do(native: torch.Tensor, last_do: torch.Tensor, semantics: NativeOutput | str) -> torch.Tensor:
    semantic = NativeOutput(semantics)
    if native.ndim != 2 or last_do.ndim != 1 or native.shape[0] != last_do.shape[0]:
        raise ValueError("native DO and last_do must have shapes [batch, horizon] and [batch]")
    if semantic is NativeOutput.STANDARDIZED_ABSOLUTE_DO:
        return native - last_do[:, None]
    return native


def reconstruct_physical_do(
    delta: torch.Tensor, last_do: torch.Tensor, *, mean: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    if delta.ndim != 2 or last_do.ndim != 1 or mean.ndim != 1 or scale.ndim != 1:
        raise ValueError("DO reconstruction expects [batch, horizon] and rank-one station tensors")
    if not (delta.shape[0] == last_do.shape[0] == mean.shape[0] == scale.shape[0]):
        raise ValueError("DO reconstruction batch dimensions must match")
    if not torch.isfinite(delta).all() or not torch.isfinite(last_do).all() or not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("DO reconstruction inputs must be finite")
    return (delta + last_do[:, None]) * scale[:, None] + mean[:, None]


def window_split(
    origin: pd.Timestamp, target_times: Sequence[pd.Timestamp], train_end: pd.Timestamp, val_end: pd.Timestamp,
) -> str | None:
    """Return a split only when origin and the complete target axis share it."""
    origin = pd.Timestamp(origin)
    targets = pd.DatetimeIndex(target_times)
    if len(targets) != PRIMARY_HORIZON:
        raise ValueError("protocol v2 windows require exactly 96 target timestamps")

    def split(timestamp: pd.Timestamp) -> str:
        if timestamp <= pd.Timestamp(train_end):
            return "train"
        if timestamp <= pd.Timestamp(val_end):
            return "val"
        return "test"

    parts = {split(value) for value in [origin, *targets]}
    return next(iter(parts)) if len(parts) == 1 else None


def h96_physical_station_macro_mae(frame: pd.DataFrame) -> float:
    required = {"station", "horizon", "target", "prediction", "target_mask"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("prediction table missing columns: " + ", ".join(missing))
    valid = frame.loc[(frame["horizon"].astype(int) == PRIMARY_HORIZON) & frame["target_mask"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("prediction table has no valid horizon-96 targets")
    per_station = (valid.assign(abs_error=(valid["prediction"] - valid["target"]).abs())
                   .groupby("station", sort=True)["abs_error"].mean())
    return float(per_station.mean())
