"""Shared station-conditioned GRU baseline for the canonical benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..adapter import validate_prediction_output
from ..schema import CanonicalBatch, DeviceBatch, ModelBatch


class SharedGRU(nn.Module):
    """Encode a 48-hour history and predict all 96 future steps directly."""

    def __init__(
        self,
        *,
        input_size: int,
        num_stations: int,
        hidden_size: int,
        station_embedding_size: int,
        layers: int = 1,
        dropout: float = 0.0,
        pred_len: int = 96,
        seq_len: int = 192,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size),
            ("num_stations", num_stations),
            ("hidden_size", hidden_size),
            ("station_embedding_size", station_embedding_size),
            ("layers", layers),
            ("pred_len", pred_len),
            ("seq_len", seq_len),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = {
            "input_size": input_size,
            "num_stations": num_stations,
            "hidden_size": hidden_size,
            "station_embedding_size": station_embedding_size,
            "layers": layers,
            "dropout": float(dropout),
            "pred_len": pred_len,
            "seq_len": seq_len,
        }
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=float(dropout) if layers > 1 else 0.0,
            batch_first=True,
        )
        self.station_embedding = nn.Embedding(num_stations, station_embedding_size)
        self.head = nn.Linear(hidden_size + station_embedding_size, pred_len)

    def forward(
        self,
        history: Tensor,
        station_id: Tensor,
        *,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        if history.ndim != 3 or history.shape[1] != self.config["seq_len"]:
            raise ValueError(f"history must have shape [batch, {self.config['seq_len']}, channels]")
        if station_id.ndim != 1 or station_id.shape[0] != history.shape[0]:
            raise ValueError("station_id must be rank-1 and match history batch size")
        if history_mask is not None and history_mask.shape != history.shape:
            raise ValueError("history_mask must match history shape")
        clean_history = torch.nan_to_num(history)
        if history_mask is not None:
            clean_history = torch.where(history_mask.bool(), clean_history, torch.zeros_like(clean_history))
        station_id = station_id.long()
        if (station_id < 1).any() or (station_id > self.config["num_stations"]).any():
            raise ValueError("station_id values must be in the canonical positive ID range")
        _, hidden = self.encoder(clean_history)
        station = self.station_embedding(station_id - 1)
        return self.head(torch.cat((hidden[-1], station), dim=-1))


class SharedGRUAdapter(nn.Module):
    """Stage 2 adapter exposing SharedGRU through the model-neutral protocol."""

    def __init__(self, **model_config: Any) -> None:
        super().__init__()
        self.model = SharedGRU(**model_config)

    def build(self) -> "SharedGRUAdapter":
        return self

    def forward(self, batch: ModelBatch) -> Tensor:
        if not isinstance(batch, (CanonicalBatch, DeviceBatch)):
            raise TypeError("SharedGRUAdapter expects a validated model batch")
        prediction = self.model(batch.history, batch.station_id, history_mask=batch.history_mask)
        return validate_prediction_output(prediction, batch)

    def loss(self, prediction: Tensor, batch: ModelBatch) -> Tensor:
        validate_prediction_output(prediction, batch)
        mask = batch.target_mask
        if not mask.any():
            raise ValueError("canonical batch contains no valid target values")
        return (prediction[mask] - batch.target[mask]).abs().mean()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"config": self.model.config, "state_dict": self.state_dict()}, temporary)
        os.replace(temporary, path)

    def load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("config") != self.model.config:
            raise ValueError("SharedGRU checkpoint configuration does not match adapter")
        self.load_state_dict(payload["state_dict"])

    def describe(self) -> dict[str, Any]:
        return {
            "name": "shared_gru",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "effective_inputs": "all canonical value channels plus explicit history mask",
            "native_output_semantics": "standardized_do_delta",
            "config": dict(self.model.config),
        }
