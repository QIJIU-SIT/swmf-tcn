"""GPU-resident streaming metrics for large forecast validation sets."""
from __future__ import annotations

import torch


class StationHorizonMAE:
    """Accumulate physical absolute error without per-target CPU transfers."""

    def __init__(self, *, station_count: int, pred_len: int, scales: torch.Tensor, device: torch.device):
        self.station_count = int(station_count)
        self.pred_len = int(pred_len)
        self.scales = scales.to(device=device, dtype=torch.float64)
        self.error_sum = torch.zeros((station_count, pred_len), dtype=torch.float64, device=device)
        self.count = torch.zeros_like(self.error_sum)

    def update(self, prediction, target, mask, station_id) -> None:
        station_index = station_id.long() - 1
        error = (prediction - target).abs().to(torch.float64) * self.scales[station_index].unsqueeze(1)
        valid = mask.to(torch.float64)
        self.error_sum.index_add_(0, station_index, error * valid)
        self.count.index_add_(0, station_index, valid)

    def summarize(self) -> dict:
        means = self.error_sum / self.count.clamp_min(1)
        valid = self.count > 0
        station_total = self.error_sum.sum(1) / self.count.sum(1).clamp_min(1)
        station_valid = self.count.sum(1) > 0
        trajectory = station_total[station_valid].mean().item() if station_valid.any() else float("nan")
        horizons = {}
        for horizon in range(self.pred_len):
            horizon_valid = valid[:, horizon]
            horizons[horizon + 1] = means[horizon_valid, horizon].mean().item() if horizon_valid.any() else float("nan")
        return {"trajectory": trajectory, "horizons": horizons}
