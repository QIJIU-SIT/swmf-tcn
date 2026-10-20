"""Small, model-neutral protocol for Stage 2 forecasting adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from .schema import ModelBatch


@runtime_checkable
class AdapterProtocol(Protocol):
    """A baseline implementation which consumes only canonical batches."""

    def build(self) -> "AdapterProtocol": ...
    def forward(self, batch: ModelBatch) -> torch.Tensor: ...
    def loss(self, prediction: torch.Tensor, batch: ModelBatch) -> torch.Tensor: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def describe(self) -> dict[str, Any]: ...


def validate_prediction_output(prediction: torch.Tensor, batch: ModelBatch) -> torch.Tensor:
    """Reject non-canonical adapter output before it can be evaluated or frozen."""
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("adapter forward must return a torch.Tensor")
    if prediction.ndim != 2 or tuple(prediction.shape) != tuple(batch.target.shape):
        raise ValueError("adapter prediction must have shape [batch, 96]")
    if not torch.isfinite(prediction).all():
        raise ValueError("adapter prediction must be finite")
    return prediction
