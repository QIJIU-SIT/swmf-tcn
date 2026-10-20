"""F4 weather-fusion TCN backbone factorial for the NEON validation pilot."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm as apply_weight_norm

from nerrs_faith.benchmark.models.fusion_depth_tcn import (
    MET_INDICES,
    WATER_EFFECTIVE_INPUTS,
    WATER_INDICES,
)
from nerrs_faith.benchmark.models.neon_multivariate import (
    EFFECTIVE_INPUTS,
    FEATURE_NAMES,
    _MVBase,
    _validate_batch,
    _validate_output,
    multivariate_history,
)


F4_TCN_MODELS = (
    "f4_tcn_base",
    "f4_tcn_residual",
    "f4_tcn_double_res",
    "f4_tcn_classic",
)
F4_FUSION_AFTER_BLOCKS = 4
_LONG_DILATIONS = (1, 2, 4, 8, 16, 32, 64)
_SHORT_DILATIONS = (1, 2, 4, 8, 16, 32)
_BLOCK_KINDS = {"base", "residual", "double_res"}


def _block_mask(channels: int) -> Tensor:
    if channels % 2:
        raise ValueError("F4 TCN channels must be even")
    mask = torch.zeros(channels, channels, 1)
    half = channels // 2
    mask[:half, :half] = 1
    mask[half:, half:] = 1
    return mask


class MaskedCausalBlock(nn.Module):
    """One causal temporal unit with a frozen water/weather connectivity mask."""

    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        kind: str,
        cross_group: bool,
        weight_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kind not in _BLOCK_KINDS:
            raise ValueError(f"unknown F4 TCN block kind: {kind}")
        if int(dilation) <= 0:
            raise ValueError("dilation must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.channels = int(channels)
        self.dilation = int(dilation)
        self.kind = kind
        self.cross_group = bool(cross_group)
        self.weight_norm = bool(weight_norm)
        self.dropout_rate = float(dropout)
        convolution_count = 2 if kind == "double_res" else 1
        self.convs = nn.ModuleList()
        for _ in range(convolution_count):
            conv = nn.Conv1d(self.channels, self.channels, 3, dilation=self.dilation)
            if self.weight_norm:
                apply_weight_norm(conv, name="weight")
            self.convs.append(conv)
        self.register_buffer("block_mask", _block_mask(self.channels), persistent=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate else nn.Identity()
        self.receptive_field_increment = 2 * self.dilation * convolution_count

    def _conv(self, value: Tensor, conv: nn.Conv1d) -> Tensor:
        weight = conv.weight if self.cross_group else conv.weight * self.block_mask
        value = F.pad(value, (2 * self.dilation, 0))
        return F.conv1d(value, weight, conv.bias, dilation=self.dilation)

    def forward(self, value: Tensor) -> Tensor:
        if self.kind == "base":
            return self.dropout(self.activation(self._conv(value, self.convs[0])))
        if self.kind == "residual":
            branch = self._conv(value, self.convs[0])
            return self.dropout(self.activation(value + branch))
        branch = self.dropout(self.activation(self._conv(value, self.convs[0])))
        branch = self._conv(branch, self.convs[1])
        return self.activation(value + branch)

    def config(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "dilation": self.dilation,
            "kind": self.kind,
            "cross_group": self.cross_group,
            "convolution_count": len(self.convs),
            "weight_norm": self.weight_norm,
            "dropout": self.dropout_rate,
            "receptive_field_increment": self.receptive_field_increment,
            "block_mask": "fixed_two_stream_block_diagonal_before_fusion",
        }


_DEFAULT_SPECS: dict[str, dict[str, Any]] = {
    "f4_tcn_base": {"kind": "base", "dilations": _LONG_DILATIONS, "weight_norm": False, "dropout": 0.0},
    "f4_tcn_residual": {"kind": "residual", "dilations": _LONG_DILATIONS, "weight_norm": False, "dropout": 0.0},
    "f4_tcn_double_res": {"kind": "double_res", "dilations": _SHORT_DILATIONS, "weight_norm": False, "dropout": 0.0},
    "f4_tcn_classic": {"kind": "double_res", "dilations": _SHORT_DILATIONS, "weight_norm": True, "dropout": 0.1},
}


class F4TCNFactorialAdapter(_MVBase):
    """The four fixed F4 backbone variants behind a common model interface."""

    def __init__(
        self,
        *,
        station_count: int,
        name: str,
        kind: str,
        dilations: Sequence[int],
        weight_norm: bool,
        dropout: float,
        channels: int = 32,
        fusion_after_blocks: int = F4_FUSION_AFTER_BLOCKS,
    ) -> None:
        super().__init__(station_count=station_count)
        if name not in F4_TCN_MODELS:
            raise ValueError(f"unknown F4 TCN model: {name}")
        if kind not in _BLOCK_KINDS:
            raise ValueError(f"unknown F4 TCN block kind: {kind}")
        if int(channels) != 32 or int(fusion_after_blocks) != F4_FUSION_AFTER_BLOCKS:
            raise ValueError("F4 TCN requires channels=32 and fusion_after_blocks=4")
        self.name = name
        self.kind = kind
        self.channels = int(channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.weight_norm = bool(weight_norm)
        self.dropout_rate = float(dropout)
        self.fusion_after_blocks = int(fusion_after_blocks)
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("F4 TCN dilations must be positive and non-empty")
        if self.kind == "double_res" and len(self.dilations) != 6:
            raise ValueError("double_res F4 TCN requires six dilation blocks")
        if self.kind != "double_res" and len(self.dilations) != 7:
            raise ValueError("single-convolution F4 TCN requires seven dilation blocks")
        self.receptive_field = 1 + sum(2 * value * (2 if self.kind == "double_res" else 1) for value in self.dilations)
        self.water_stem = nn.Conv1d(8, self.channels // 2, 1)
        self.met_stem = nn.Conv1d(6, self.channels // 2, 1)
        self.blocks = nn.ModuleList(
            MaskedCausalBlock(
                self.channels,
                dilation=dilation,
                kind=self.kind,
                cross_group=index >= self.fusion_after_blocks,
                weight_norm=self.weight_norm,
                dropout=self.dropout_rate,
            )
            for index, dilation in enumerate(self.dilations)
        )
        self.head = nn.Linear(self.channels + 4, 96)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        _validate_batch(batch, station_count=self.station_count)
        history = multivariate_history(batch)
        water = history[..., WATER_INDICES].transpose(1, 2)
        met = history[..., MET_INDICES].transpose(1, 2)
        value = torch.cat((self.water_stem(water), self.met_stem(met)), dim=1)
        for block in self.blocks:
            value = block(value)
        hidden = value[:, :, -1]
        return _validate_output(self.head(torch.cat((hidden, self._station(batch)), dim=-1)), batch)

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "name": self.name,
            "block_kind": self.kind,
            "convolution_count": sum(len(block.convs) for block in self.blocks),
            "dilations": self.dilations,
            "channels": self.channels,
            "fusion_after_blocks": self.fusion_after_blocks,
            "weight_norm": self.weight_norm,
            "dropout": self.dropout_rate,
            "receptive_field": self.receptive_field,
            "effective_inputs": EFFECTIVE_INPUTS,
            "water_indices": WATER_INDICES,
            "met_indices": MET_INDICES,
            "block_configs": [block.config() for block in self.blocks],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "feature_names": FEATURE_NAMES,
            "effective_inputs": list(EFFECTIVE_INPUTS),
            "native_output_semantics": "standardized_do_delta",
            "config": self.config(),
        }


def _resolve_spec(name: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    if name not in _DEFAULT_SPECS:
        raise ValueError(f"unknown F4 TCN model: {name}")
    spec = dict(_DEFAULT_SPECS[name])
    if config is not None:
        expected = _DEFAULT_SPECS[name]
        if set(config) != {"kind", "dilations", "weight_norm", "dropout"}:
            raise ValueError(f"F4 TCN configuration has unknown keys for {name}")
        for key in ("kind", "weight_norm", "dropout"):
            if key not in config or config[key] != expected[key]:
                raise ValueError(f"F4 TCN configuration mismatch for {name}: {key}")
        if tuple(int(value) for value in config.get("dilations", ())) != tuple(expected["dilations"]):
            raise ValueError(f"F4 TCN configuration mismatch for {name}: dilations")
    return spec


def build_f4_tcn_factorial(
    name: str,
    *,
    station_count: int,
    config: Mapping[str, Any] | None = None,
) -> F4TCNFactorialAdapter:
    spec = _resolve_spec(name, config)
    return F4TCNFactorialAdapter(station_count=station_count, name=name, **spec)
