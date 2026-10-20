"""Target-conditioned multivariate wrapper around the official PatchTST core."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..adapter import validate_prediction_output
from ..schema import DeviceBatch
from ...protocol_v2 import DO_FEATURE_NAME, NativeOutput, adapt_native_do
from .neon_multivariate import EFFECTIVE_INPUTS, FEATURE_NAMES, multivariate_history
from .patchtst import PatchTSTAdapter


class PatchTSTMVAdapter(nn.Module):
    def __init__(self, *, source_root: str | Path, station_count: int, seq_len: int = 256, pred_len: int = 96, model_config: Mapping[str, Any] | None = None):
        super().__init__()
        if int(seq_len) != 256 or int(pred_len) != 96:
            raise ValueError("PatchTST-MV requires 256 history and 96 prediction steps")
        self.seq_len, self.pred_len, self.station_count = int(seq_len), int(pred_len), int(station_count)
        self.correction_limit = 0.25
        self.value_gate = nn.Sequential(nn.Linear(14, 16), nn.GELU(), nn.Linear(16, 1))
        self.patchtst = PatchTSTAdapter(source_root=source_root, input_size=1, seq_len=seq_len, pred_len=pred_len, model_config=model_config, feature_names=(DO_FEATURE_NAME,), do_channel=0, protocol="protocol_v2")
        self.station_bias = nn.Embedding(self.station_count, pred_len)
        nn.init.zeros_(self.station_bias.weight)
        self.amp = False

    def _device_batch(self, batch: Mapping[str, Tensor]) -> DeviceBatch:
        history = multivariate_history(batch)
        # PatchTST + RevIN returns values on the scale of its input channel.
        # Keep that channel on the standardized absolute-DO scale and let the
        # multivariate projection learn only a correction on the same scale.
        do_history = torch.where(
            batch["water_mask"][..., 0].bool(),
            torch.nan_to_num(batch["water"][..., 0]),
            torch.zeros_like(batch["water"][..., 0]),
        ).unsqueeze(-1)
        correction = self.correction_limit * torch.tanh(self.value_gate(history))
        fused = do_history + correction
        return DeviceBatch(history=fused, history_mask=torch.ones_like(fused, dtype=torch.bool), target=batch["target"], target_mask=batch["target_mask"].bool(), last_do=batch["last_do"], last_do_physical=batch.get("last_do_physical", batch["last_do"]), station_id=batch["station_id"], reserve_id=batch.get("reserve_id", torch.ones_like(batch["station_id"])), station=batch.get("position", ["unknown"] * len(batch["station_id"])), reserve=batch.get("site", ["unknown"] * len(batch["station_id"])), origin=batch.get("origin", []), target_times=batch.get("target_times", []), frequency_minutes=15)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        if batch["water"].shape[1] != self.seq_len:
            raise ValueError("PatchTST-MV history length mismatch")
        device_batch = self._device_batch(batch)
        # The official adapter returns a standardized delta after converting its
        # absolute output with last_do; the fused one-channel history is the
        # adapted target-conditioned sequence, not a new scientific feature.
        delta = self.patchtst(device_batch)
        prediction = delta + self.station_bias(batch["station_id"].long() - 1)
        return validate_prediction_output(prediction, device_batch)

    def loss(self, prediction: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        mask = batch["target_mask"].bool()
        if not mask.any(): raise ValueError("NEON batch contains no valid targets")
        return (prediction[mask] - batch["target"][mask]).abs().mean()

    def build(self): return self

    def describe(self) -> dict[str, Any]:
        return {"name": "patchtst_mv", "parameter_count": sum(parameter.numel() for parameter in self.parameters()), "feature_names": FEATURE_NAMES, "effective_inputs": EFFECTIVE_INPUTS, "adaptation": "bounded_target_conditioned_residual_fusion", "correction_limit_standardized_do": self.correction_limit, "amp": False, "native_output_semantics": NativeOutput.STANDARDIZED_DO_DELTA.value, "official_source_hash": self.patchtst.source_hash, "config": copy.deepcopy(self.patchtst.model_config), "station_count": self.station_count, "seq_len": self.seq_len, "pred_len": self.pred_len}

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"description": self.describe(), "state_dict": self.state_dict()}, temporary); os.replace(temporary, path)

    def load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("description") != self.describe(): raise ValueError("PatchTST-MV checkpoint provenance/configuration does not match adapter")
        self.load_state_dict(payload["state_dict"])
