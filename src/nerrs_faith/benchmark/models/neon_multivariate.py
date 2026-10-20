"""Small multivariate baselines for the isolated NEON protocol-v0 screen."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FEATURE_NAMES = (
    "water:DO_mgl", "water:pH", "water:specificConductance", "water:turbidity",
    "met:tempRHMean", "met:RHMean", "met:precipBulk",
)
EFFECTIVE_INPUTS = FEATURE_NAMES + tuple(f"{name}:observed_mask" for name in FEATURE_NAMES)


def multivariate_history(batch: Mapping[str, Tensor]) -> Tensor:
    """Return [B, 256, 14]: masked values followed by explicit masks."""
    water, met = batch["water"], batch["met"]
    water_mask, met_mask = batch["water_mask"].bool(), batch["met_mask"].bool()
    values = torch.cat((water, met), dim=-1)
    masks = torch.cat((water_mask, met_mask), dim=-1)
    if values.ndim != 3 or values.shape[1] != 256 or values.shape[2] != 7:
        raise ValueError("NEON multivariate history must have shape [batch, 256, 7]")
    if masks.shape != values.shape:
        raise ValueError("NEON value and mask channels must have identical shapes")
    return torch.cat((torch.where(masks, torch.nan_to_num(values), torch.zeros_like(values)), masks.float()), dim=-1)


def _validate_batch(batch: Mapping[str, Tensor], *, station_count: int) -> None:
    history = multivariate_history(batch)
    target = batch["target"]
    if target.ndim != 2 or target.shape[1] != 96 or target.shape[0] != history.shape[0]:
        raise ValueError("NEON target must have shape [batch, 96]")
    station_id = batch["station_id"].long()
    if station_id.ndim != 1 or station_id.shape[0] != history.shape[0] or (station_id < 1).any() or (station_id > station_count).any():
        raise ValueError("station_id must be positive and within station_count")

def _validate_output(prediction: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
    if prediction.ndim != 2 or tuple(prediction.shape) != tuple(batch["target"].shape) or not torch.isfinite(prediction).all():
        raise ValueError("NEON adapter prediction must be finite with shape [batch, 96]")
    return prediction


class _MVBase(nn.Module):
    def __init__(self, *, station_count: int, pred_len: int = 96, seq_len: int = 256, feature_names: Sequence[str] = FEATURE_NAMES):
        super().__init__()
        if tuple(feature_names) != FEATURE_NAMES:
            raise ValueError("NEON expanded baselines require the frozen feature order")
        self.station_count, self.pred_len, self.seq_len = int(station_count), int(pred_len), int(seq_len)
        if self.seq_len != 256 or self.pred_len != 96 or self.station_count < 1:
            raise ValueError("NEON expanded baselines require 256 history and 96 prediction steps")
        self.station_embedding = nn.Embedding(self.station_count, 4)

    def _station(self, batch):
        station_id = batch["station_id"].long()
        return self.station_embedding(station_id - 1)

    def loss(self, prediction: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        prediction = _validate_output(prediction, batch)
        mask = batch["target_mask"].bool()
        if not mask.any():
            raise ValueError("NEON batch contains no valid targets")
        return (prediction[mask] - batch["target"][mask]).abs().mean()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "feature_names": FEATURE_NAMES,
            "effective_inputs": EFFECTIVE_INPUTS,
            "native_output_semantics": "standardized_do_delta",
            "config": self.config(),
        }

    def config(self) -> dict[str, Any]:
        return {"station_count": self.station_count, "seq_len": self.seq_len, "pred_len": self.pred_len, "feature_names": FEATURE_NAMES}

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"description": self.describe(), "state_dict": self.state_dict()}, temporary)
        os.replace(temporary, path)

    def load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("description") != self.describe():
            raise ValueError("NEON multivariate checkpoint provenance/configuration does not match adapter")
        self.load_state_dict(payload["state_dict"])


class NeonForecastAdapter(nn.Module):
    """Model-neutral seam used by the NEON trainer and checkpoint validator."""
    def __init__(self, model: _MVBase):
        super().__init__()
        self.model = model

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.model(batch)

    def loss(self, prediction: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        return self.model.loss(prediction, batch)

    def build(self):
        return self

    def save(self, path: Path) -> None:
        self.model.save(path)

    def load(self, path: Path) -> None:
        self.model.load(path)

    def describe(self):
        return self.model.describe()


class RecurrentMVAdapter(_MVBase):
    def __init__(self, *, cell: str, input_channels: int = 14, hidden_size: int = 32, station_count: int, station_embedding_size: int = 4, pred_len: int = 96, seq_len: int = 256, feature_names: Sequence[str] = FEATURE_NAMES):
        super().__init__(station_count=station_count, pred_len=pred_len, seq_len=seq_len, feature_names=feature_names)
        if cell not in {"rnn", "lstm"} or input_channels != 14:
            raise ValueError("cell must be rnn or lstm and input_channels must be 14")
        self.cell, self.input_channels, self.hidden_size = cell, int(input_channels), int(hidden_size)
        recurrent = nn.RNN if cell == "rnn" else nn.LSTM
        self.encoder = recurrent(input_channels, hidden_size, batch_first=True)
        self.station_embedding = nn.Embedding(station_count, station_embedding_size)
        self.head = nn.Linear(hidden_size + station_embedding_size, pred_len)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        _validate_batch(batch, station_count=self.station_count)
        encoded = self.encoder(multivariate_history(batch))
        hidden = encoded[1][0][-1] if self.cell == "lstm" else encoded[1][-1]
        return _validate_output(self.head(torch.cat((hidden, self._station(batch)), dim=-1)), batch)

    def config(self):
        return {**super().config(), "cell": self.cell, "input_channels": self.input_channels, "hidden_size": self.hidden_size, "station_embedding_size": int(self.station_embedding.embedding_dim)}


class TCNMVAdapter(_MVBase):
    def __init__(self, *, input_channels: int = 14, channels: int = 32, dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64), station_count: int, pred_len: int = 96, seq_len: int = 256, feature_names: Sequence[str] = FEATURE_NAMES):
        super().__init__(station_count=station_count, pred_len=pred_len, seq_len=seq_len, feature_names=feature_names)
        self.input_channels, self.channels, self.dilations = int(input_channels), int(channels), tuple(int(value) for value in dilations)
        self.receptive_field = 1 + 2 * sum(self.dilations)
        if self.input_channels != 14 or self.receptive_field < 255:
            raise ValueError("TCN-MV requires 14 channels and receptive field >= 255")
        layers = []
        current = self.input_channels
        for dilation in self.dilations:
            layers.extend((nn.Conv1d(current, channels, 3, dilation=dilation), nn.GELU()))
            current = channels
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(channels + 4, pred_len)

    def encode_history(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Encode the frozen 14-channel history into the final TCN state."""
        _validate_batch(batch, station_count=self.station_count)
        value = multivariate_history(batch).transpose(1, 2)
        index = 0
        for dilation in self.dilations:
            value = F.pad(value, (2 * dilation, 0))
            value = self.layers[index + 1](self.layers[index](value)); index += 2
        return value.transpose(1, 2)[:, -1]

    def predict_from_hidden(self, hidden: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        """Apply the unchanged station-conditioned TCN-MV prediction head."""
        return _validate_output(self.head(torch.cat((hidden, self._station(batch)), dim=-1)), batch)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.predict_from_hidden(self.encode_history(batch), batch)

    def config(self):
        return {**super().config(), "input_channels": self.input_channels, "channels": self.channels, "dilations": self.dilations}


class DLinearMVAdapter(_MVBase):
    def __init__(self, *, input_channels: int = 14, seq_len: int = 256, pred_len: int = 96, moving_average: int = 25, station_count: int, feature_names: Sequence[str] = FEATURE_NAMES):
        super().__init__(station_count=station_count, pred_len=pred_len, seq_len=seq_len, feature_names=feature_names)
        if input_channels != 14 or moving_average % 2 == 0:
            raise ValueError("DLinear-MV requires 14 channels and an odd moving-average kernel")
        self.input_channels, self.moving_average = int(input_channels), int(moving_average)
        self.trend_projection = nn.Linear(seq_len, pred_len)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)
        self.channel_fusion = nn.Linear(input_channels, 1)
        self.station_bias = nn.Embedding(station_count, pred_len)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        _validate_batch(batch, station_count=self.station_count)
        value = multivariate_history(batch).transpose(1, 2)
        radius = self.moving_average // 2
        trend = F.avg_pool1d(F.pad(value, (radius, radius), mode="replicate"), self.moving_average, stride=1)
        seasonal = value - trend
        forecast = self.trend_projection(trend) + self.seasonal_projection(seasonal)
        forecast = self.channel_fusion(forecast.transpose(1, 2)).squeeze(-1)
        return _validate_output(forecast + self.station_bias(batch["station_id"].long() - 1), batch)

    def config(self):
        return {**super().config(), "input_channels": self.input_channels, "moving_average": self.moving_average}


def build_recurrent_mv(cell: str, **kwargs):
    return RecurrentMVAdapter(cell=cell, **kwargs)


def build_tcn_mv(**kwargs):
    return TCNMVAdapter(**kwargs)


def build_dlinear_mv(**kwargs):
    return DLinearMVAdapter(**kwargs)
