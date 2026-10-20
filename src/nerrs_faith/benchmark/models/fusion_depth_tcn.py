"""Parameter-matched TCNs for the NEON weather-fusion-depth experiment."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nerrs_faith.benchmark.models.neon_multivariate import (
    EFFECTIVE_INPUTS,
    FEATURE_NAMES,
    _MVBase,
    _validate_batch,
    _validate_output,
    multivariate_history,
)


WATER_INDICES = (0, 1, 2, 3, 7, 8, 9, 10)
MET_INDICES = (4, 5, 6, 11, 12, 13)
DEFAULT_DILATIONS = (1, 2, 4, 8, 16, 32, 64)
WATER_EFFECTIVE_INPUTS = FEATURE_NAMES[:4] + tuple(f"{name}:observed_mask" for name in FEATURE_NAMES[:4])


class FusionDepthTCNAdapter(_MVBase):
    """A causal TCN whose fixed masks control the first fusion layer."""

    def __init__(
        self,
        *,
        station_count: int,
        fusion_after: int,
        channels: int = 32,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        water_only: bool = False,
    ) -> None:
        super().__init__(station_count=station_count)
        if int(fusion_after) not in {0, 2, 4, 7}:
            raise ValueError("fusion_after must be one of 0, 2, 4, 7")
        if int(channels) != 32 or int(channels) % 2:
            raise ValueError("fusion-depth TCN requires an even width of 32")
        if tuple(int(value) for value in dilations) != DEFAULT_DILATIONS:
            raise ValueError("fusion-depth TCN dilations are frozen to [1, 2, 4, 8, 16, 32, 64]")
        self.fusion_after = int(fusion_after)
        self.channels = int(channels)
        self.dilations = tuple(int(value) for value in dilations)
        self.water_only = bool(water_only)
        self.receptive_field = 1 + 2 * sum(self.dilations)
        self.water_stem = nn.Conv1d(8, self.channels // 2, 1)
        self.met_stem = nn.Conv1d(6, self.channels // 2, 1)
        self.convs = nn.ModuleList(
            nn.Conv1d(self.channels, self.channels, 3, dilation=dilation)
            for dilation in self.dilations
        )
        block_mask = torch.zeros(self.channels, self.channels, 1)
        half = self.channels // 2
        block_mask[:half, :half] = 1
        block_mask[half:, half:] = 1
        self.register_buffer("block_mask", block_mask, persistent=True)
        self.activation = nn.GELU()
        self.head = nn.Linear(self.channels + 4, 96)

    def _masked_conv(self, value: Tensor, layer_index: int, dilation: int) -> Tensor:
        conv = self.convs[layer_index]
        weight = conv.weight
        if layer_index < self.fusion_after:
            weight = weight * self.block_mask
        value = F.pad(value, (2 * dilation, 0))
        return self.activation(F.conv1d(value, weight, conv.bias, dilation=dilation))

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        _validate_batch(batch, station_count=self.station_count)
        history = multivariate_history(batch)
        water = history[..., WATER_INDICES]
        met = history[..., MET_INDICES]
        if self.water_only:
            met = torch.zeros_like(met)
        value = torch.cat(
            (self.water_stem(water.transpose(1, 2)), self.met_stem(met.transpose(1, 2))),
            dim=1,
        )
        for index, dilation in enumerate(self.dilations):
            value = self._masked_conv(value, index, dilation)
        hidden = value[:, :, -1]
        prediction = self.head(torch.cat((hidden, self._station(batch)), dim=-1))
        return _validate_output(prediction, batch)

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "input_channels": 14,
            "water_stem_channels": 8,
            "met_stem_channels": 6,
            "channels": self.channels,
            "dilations": self.dilations,
            "fusion_after": self.fusion_after,
            "water_only": self.water_only,
            "group_widths": [self.channels // 2, self.channels // 2],
            "receptive_field": self.receptive_field,
            "block_mask": "fixed_two_stream_block_diagonal_before_fusion",
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": {
                0: "fusion_f0_early", 2: "fusion_f2_mid", 4: "fusion_f4_mid", 7: "fusion_f7_late"
            }[self.fusion_after] if not self.water_only else "fusion_water_only",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "feature_names": FEATURE_NAMES,
            "effective_inputs": list(WATER_EFFECTIVE_INPUTS if self.water_only else EFFECTIVE_INPUTS),
            "native_output_semantics": "standardized_do_delta",
            "config": self.config(),
        }
