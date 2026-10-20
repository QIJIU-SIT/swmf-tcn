"""Validated, model-neutral objects shared by all benchmark baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Sequence, TypeAlias

import pandas as pd
import torch


SplitName = Literal["train", "val", "test"]


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            str(key): _frozen_mapping(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    )


@dataclass(frozen=True)
class PredictionKey:
    """One uniquely pairable forecast target."""

    station: str
    origin: pd.Timestamp
    target_time: pd.Timestamp
    horizon: int
    frequency_minutes: int = 15

    def __post_init__(self) -> None:
        origin = pd.Timestamp(self.origin)
        target_time = pd.Timestamp(self.target_time)
        if not self.station:
            raise ValueError("prediction key station must be non-empty")
        if self.horizon < 1:
            raise ValueError("prediction key horizon must be positive")
        if self.frequency_minutes < 1:
            raise ValueError("prediction key frequency_minutes must be positive")
        expected = origin + pd.Timedelta(minutes=self.horizon * self.frequency_minutes)
        if target_time != expected:
            raise ValueError("prediction key target_time does not match origin and horizon")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "target_time", target_time)

    def as_tuple(self) -> tuple[str, pd.Timestamp, pd.Timestamp, int]:
        return (self.station, self.origin, self.target_time, self.horizon)


@dataclass(frozen=True)
class CanonicalBatch:
    """A batched 48-hour history and 24-hour target without model-specific fields."""

    history: torch.Tensor
    history_mask: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    station_id: torch.Tensor
    reserve_id: torch.Tensor
    station: Sequence[str]
    reserve: Sequence[str]
    origin: Sequence[pd.Timestamp]
    target_times: Sequence[Sequence[pd.Timestamp]]
    last_do: torch.Tensor | None = None
    last_do_physical: torch.Tensor | None = None
    frequency_minutes: int = 15

    def __post_init__(self) -> None:
        if self.frequency_minutes != 15:
            raise ValueError("canonical benchmark requires 15-minute frequency")
        if self.history.ndim != 3:
            raise ValueError("history must have rank 3: batch, steps, features")
        if self.target.ndim != 2:
            raise ValueError("target must have rank 2: batch, steps")
        if self.history.shape[1] != 192:
            raise ValueError("history must contain exactly 192 steps")
        if self.target.shape[1] != 96:
            raise ValueError("target must contain exactly 96 steps")
        if self.history_mask.shape != self.history.shape:
            raise ValueError("history_mask must match history shape")
        if self.target_mask.shape != self.target.shape:
            raise ValueError("target_mask must match target shape")
        if self.history_mask.dtype != torch.bool or self.target_mask.dtype != torch.bool:
            raise ValueError("masks must be boolean tensors")
        batch_size = self.history.shape[0]
        if self.target.shape[0] != batch_size:
            raise ValueError("history and target batch dimensions must match")
        if self.last_do is None:
            object.__setattr__(self, "last_do", self.history[:, -1, -1].detach().clone())
        if self.last_do_physical is None:
            object.__setattr__(self, "last_do_physical", self.last_do.detach().clone())
        for name, value in (("last_do", self.last_do), ("last_do_physical", self.last_do_physical)):
            if value.ndim != 1 or value.shape[0] != batch_size:
                raise ValueError(f"{name} must be a rank-1 tensor matching batch size")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        for name, value in (("station_id", self.station_id), ("reserve_id", self.reserve_id)):
            if value.ndim != 1 or value.shape[0] != batch_size:
                raise ValueError(f"{name} must be a rank-1 tensor matching batch size")
            if (value <= 0).any():
                raise ValueError(f"{name} values must be positive")
        if len(self.station) != batch_size or len(self.reserve) != batch_size:
            raise ValueError("station and reserve identities must match batch size")
        if any(not value for value in self.station) or any(not value for value in self.reserve):
            raise ValueError("station and reserve IDs must be non-empty")
        if len(self.origin) != batch_size or len(self.target_times) != batch_size:
            raise ValueError("timestamp identities must match batch size")
        if not torch.isfinite(self.history[self.history_mask]).all():
            raise ValueError("masked history values must be finite")
        if not torch.isfinite(self.target[self.target_mask]).all():
            raise ValueError("masked target values must be finite")

        origins = tuple(pd.Timestamp(value) for value in self.origin)
        timestamps = tuple(pd.DatetimeIndex(value) for value in self.target_times)
        keys: list[PredictionKey] = []
        for row, (station, origin, row_times) in enumerate(zip(self.station, origins, timestamps, strict=True)):
            if len(row_times) != 96:
                raise ValueError("target_times must contain exactly 96 timestamps")
            for step, target_time in enumerate(row_times, start=1):
                key = PredictionKey(station, origin, target_time, step, self.frequency_minutes)
                if self.target_mask[row, step - 1]:
                    keys.append(key)
        if len({key.as_tuple() for key in keys}) != len(keys):
            raise ValueError("prediction keys must be unique")
        object.__setattr__(self, "origin", origins)
        object.__setattr__(self, "target_times", timestamps)

    def prediction_keys(self) -> tuple[PredictionKey, ...]:
        keys: list[PredictionKey] = []
        for row, (station, origin, times) in enumerate(zip(self.station, self.origin, self.target_times, strict=True)):
            keys.extend(
                PredictionKey(station, origin, target_time, step, self.frequency_minutes)
                for step, target_time in enumerate(times, start=1)
                if self.target_mask[row, step - 1]
            )
        return tuple(keys)


@dataclass(frozen=True)
class DeviceBatch:
    """Trusted tensor transfer of an already validated ``CanonicalBatch``."""

    history: torch.Tensor
    history_mask: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    last_do: torch.Tensor
    last_do_physical: torch.Tensor
    station_id: torch.Tensor
    reserve_id: torch.Tensor
    station: Sequence[str]
    reserve: Sequence[str]
    origin: Sequence[pd.Timestamp]
    target_times: Sequence[Sequence[pd.Timestamp]]
    frequency_minutes: int = 15


ModelBatch: TypeAlias = CanonicalBatch | DeviceBatch


def to_device_batch(
    batch: CanonicalBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> DeviceBatch:
    """Transfer a validated canonical batch without repeating schema checks."""
    if not isinstance(batch, CanonicalBatch):
        raise TypeError("to_device_batch requires a validated CanonicalBatch")
    return DeviceBatch(
        history=batch.history.to(device, non_blocking=non_blocking),
        history_mask=batch.history_mask.to(device, non_blocking=non_blocking),
        target=batch.target.to(device, non_blocking=non_blocking),
        target_mask=batch.target_mask.to(device, non_blocking=non_blocking),
        last_do=batch.last_do.to(device, non_blocking=non_blocking),
        last_do_physical=batch.last_do_physical.to(device, non_blocking=non_blocking),
        station_id=batch.station_id.to(device, non_blocking=non_blocking),
        reserve_id=batch.reserve_id.to(device, non_blocking=non_blocking),
        station=batch.station,
        reserve=batch.reserve,
        origin=batch.origin,
        target_times=batch.target_times,
        frequency_minutes=batch.frequency_minutes,
    )


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Immutable provenance and feature metadata for canonical benchmark data."""

    feature_names: tuple[str, ...]
    station_ids: Mapping[str, int]
    reserve_ids: Mapping[str, int]
    normalization: Mapping[str, Mapping[str, Mapping[str, float]]]
    history_steps: int = 192
    target_steps: int = 96
    frequency_minutes: int = 15
    protocol: str = "legacy"
    do_feature_name: str = "water:DO_mgl"
    do_channel: int | None = None
    target_representation: str = "standardized_delta_from_origin"
    normalization_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        feature_names = tuple(str(name) for name in self.feature_names)
        if self.history_steps != 192 or self.target_steps != 96:
            raise ValueError("canonical benchmark requires 192 history and 96 target steps")
        if self.frequency_minutes != 15:
            raise ValueError("canonical benchmark requires 15-minute frequency")
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if self.protocol == "protocol_v2":
            if self.do_feature_name not in feature_names:
                raise ValueError("protocol v2 metadata must contain the named DO feature")
            resolved_channel = feature_names.index(self.do_feature_name)
            if self.do_channel != resolved_channel:
                raise ValueError("protocol v2 DO channel does not match feature order")
            if self.target_representation != "standardized_delta_from_origin":
                raise ValueError("protocol v2 requires standardized_delta_from_origin targets")
        if not self.station_ids or not self.reserve_ids:
            raise ValueError("station_ids and reserve_ids must be non-empty")
        if any(not name or int(identifier) <= 0 for name, identifier in self.station_ids.items()):
            raise ValueError("station_ids must have non-empty names and positive values")
        if any(not name or int(identifier) <= 0 for name, identifier in self.reserve_ids.items()):
            raise ValueError("reserve_ids must have non-empty names and positive values")
        frozen_normalization = _frozen_mapping(self.normalization)
        canonical = json.dumps(self.normalization, sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "station_ids", _frozen_mapping(self.station_ids))
        object.__setattr__(self, "reserve_ids", _frozen_mapping(self.reserve_ids))
        object.__setattr__(self, "normalization", frozen_normalization)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "normalization_fingerprint", hashlib.sha256(canonical.encode("utf-8")).hexdigest())
