"""Parameter-matched fusion-depth controls for the frozen F4 backbone."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from nerrs_faith.benchmark.models.f4_tcn_factorial import MET_INDICES, WATER_INDICES, MaskedCausalBlock
from nerrs_faith.benchmark.models.neon_multivariate import EFFECTIVE_INPUTS, FEATURE_NAMES, _MVBase, _validate_batch, _validate_output, multivariate_history

F4_DEPTH_MODELS = ("f4_depth_f0_early", "f4_depth_f4_mid", "f4_depth_f6_late", "f4_depth_water_only")
_DILATIONS = (1, 2, 4, 8, 16, 32)
_DEPTHS = {"f4_depth_f0_early": 0, "f4_depth_f4_mid": 4, "f4_depth_f6_late": 6, "f4_depth_water_only": 6}


class F4DepthDoubleResAdapter(_MVBase):
    """The six-block F4 graph with only its cross-group boundary changed."""

    def __init__(self, *, station_count: int, name: str | None = None, fusion_after: int | None = None, fusion_after_blocks: int | None = None, water_only: bool | None = None) -> None:
        super().__init__(station_count=station_count)
        if fusion_after is not None:
            if fusion_after_blocks is not None and int(fusion_after) != int(fusion_after_blocks):
                raise ValueError("F4 depth fusion boundaries disagree")
            fusion_after_blocks = int(fusion_after)
        if name is None:
            inferred = {0: "f4_depth_f0_early", 4: "f4_depth_f4_mid", 6: "f4_depth_f6_late"}.get(int(fusion_after_blocks) if fusion_after_blocks is not None else -1)
            if bool(water_only) and int(fusion_after_blocks or 6) == 6:
                inferred = "f4_depth_water_only"
            if inferred is None:
                raise ValueError("F4 depth name cannot be inferred from the supplied boundary")
            name = inferred
        if name not in F4_DEPTH_MODELS:
            raise ValueError(f"unknown F4 depth model: {name}")
        expected_depth = _DEPTHS[name]
        expected_water_only = name == "f4_depth_water_only"
        if fusion_after_blocks is not None and int(fusion_after_blocks) != expected_depth:
            raise ValueError("F4 depth fusion boundary does not match frozen model name")
        if water_only is not None and bool(water_only) != expected_water_only:
            raise ValueError("F4 depth water_only flag does not match frozen model name")
        self.name = name
        self.fusion_after_blocks = expected_depth
        self.water_only = expected_water_only
        self.channels = 32
        self.dilations = _DILATIONS
        self.receptive_field = 1 + 4 * sum(self.dilations)
        self.water_stem = nn.Conv1d(8, 16, 1)
        self.met_stem = nn.Conv1d(6, 16, 1)
        self.blocks = nn.ModuleList(
            MaskedCausalBlock(32, dilation=dilation, kind="double_res", cross_group=index >= self.fusion_after_blocks)
            for index, dilation in enumerate(self.dilations)
        )
        self.head = nn.Linear(36, 96)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        _validate_batch(batch, station_count=self.station_count)
        history = multivariate_history(batch)
        water = self.water_stem(history[..., WATER_INDICES].transpose(1, 2))
        met_values = history[..., MET_INDICES]
        if self.water_only:
            met_values = torch.zeros_like(met_values)
        weather = self.met_stem(met_values.transpose(1, 2))
        value = torch.cat((water, weather), dim=1)
        for block in self.blocks:
            value = block(value)
        hidden = value[:, :, -1]
        return _validate_output(self.head(torch.cat((hidden, self._station(batch)), dim=-1)), batch)

    def config(self) -> dict[str, Any]:
        return {**super().config(), "name": self.name, "block_kind": "double_res", "dilations": self.dilations, "channels": self.channels, "fusion_after_blocks": self.fusion_after_blocks, "water_only": self.water_only, "receptive_field": self.receptive_field, "effective_inputs": EFFECTIVE_INPUTS, "water_indices": WATER_INDICES, "met_indices": MET_INDICES, "block_configs": [block.config() for block in self.blocks]}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "parameter_count": sum(parameter.numel() for parameter in self.parameters()), "feature_names": FEATURE_NAMES, "effective_inputs": list(EFFECTIVE_INPUTS), "native_output_semantics": "standardized_do_delta", "config": self.config()}


def build_f4_depth_double_res(name: str, *, station_count: int, config: Mapping[str, Any] | None = None) -> F4DepthDoubleResAdapter:
    if name not in _DEPTHS:
        raise ValueError(f"unknown F4 depth model: {name}")
    expected = {"fusion_after_blocks": _DEPTHS[name], "water_only": name == "f4_depth_water_only"}
    if config is not None and dict(config) != expected:
        raise ValueError(f"F4 depth configuration mismatch for {name}")
    return F4DepthDoubleResAdapter(station_count=station_count, name=name, **expected)
