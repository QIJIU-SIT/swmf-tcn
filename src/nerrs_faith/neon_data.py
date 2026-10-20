"""Lazy 15-minute NEON monitoring-position windows."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WATER = ("dissolvedOxygen", "pH", "specificConductance", "turbidity")
MET = ("tempRHMean", "RHMean", "precipBulk")
_CONTROLLED_TEST_CAPABILITY = object()

def _missing_duration(mask: np.ndarray, frequency_minutes: int) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=np.float32)
    for column in range(mask.shape[1]):
        elapsed = 0.0
        for row in range(mask.shape[0]):
            elapsed = 0.0 if mask[row, column] else elapsed + frequency_minutes / 60.0
            result[row, column] = elapsed
    return result

class NeonWindowDataset(Dataset):
    def __init__(self, prepared, *, split: str, seq_len: int = 256, pred_len: int = 96, frequency_minutes: int = 15, scaling_stats=None, allow_test: bool = False):
        if split == "test" and allow_test is not _CONTROLLED_TEST_CAPABILITY:
            raise PermissionError("NEON 2024 test windows are sealed in protocol_v0")
        self.prepared = {}; self.split = split; self.seq_len = int(seq_len); self.pred_len = int(pred_len); self.frequency_minutes = int(frequency_minutes)
        self.scaling_stats = scaling_stats or {}
        self.positions = sorted(prepared); self.rows = []; self.water_missing = {}; self.arrays = {}
        for position in self.positions:
            frame = prepared[position].copy(); frame["date"] = pd.to_datetime(frame.get("date", frame.index), utc=True); frame = frame.sort_values("date").reset_index(drop=True); self.prepared[position] = frame
            mask = frame.get("dissolvedOxygen_observed", pd.Series(True, index=frame.index)).astype(bool)
            water_mask = np.column_stack([frame.get(f"{c}_observed", pd.Series(True, index=frame.index)).to_numpy(bool) for c in WATER])
            self.water_missing[position] = _missing_duration(water_mask, self.frequency_minutes)
            def column(name, *, met=False):
                candidates = [name, f"humidity_{name}", f"precipitation_{name}"] if met else [name]
                for candidate in candidates:
                    if candidate in frame: return frame[candidate]
                return pd.Series(np.nan, index=frame.index)
            water = np.column_stack([column(c).to_numpy(np.float32) for c in WATER])
            met = np.column_stack([column(c, met=True).to_numpy(np.float32) for c in MET])
            precipitation = column("precipBulk", met=True).to_numpy(np.float32)
            precipitation_observed = frame.get("precipBulk_observed", frame.get("precipitation_precipBulk_observed", pd.Series(True, index=frame.index))).to_numpy(bool)
            met_mask = np.column_stack([frame.get(f"{c}_observed", frame.get(f"humidity_{c}_observed", frame.get(f"precipitation_{c}_observed", pd.Series(True, index=frame.index)))).to_numpy(bool) for c in MET])
            stats = self.scaling_stats.get(position, {})
            for idx, name in enumerate(WATER):
                item = stats.get("water", {}).get(name, {})
                water[:, idx] = (water[:, idx] - float(item.get("mean", 0.0))) / float(item.get("scale", 1.0) or 1.0)
            for idx, name in enumerate(MET):
                item = stats.get("meteorology", {}).get(name, {})
                met[:, idx] = (met[:, idx] - float(item.get("mean", 0.0))) / float(item.get("scale", 1.0) or 1.0)
            self.arrays[position] = {
                "water": np.nan_to_num(water), "water_mask": water_mask,
                "met": np.nan_to_num(met), "met_mask": met_mask,
                "precipitation": np.nan_to_num(precipitation), "precipitation_observed": precipitation_observed,
                "do": frame["dissolvedOxygen"].to_numpy(np.float32),
                "do_observed": mask.to_numpy(bool), "date": frame["date"].to_numpy(copy=True),
            }
            for origin in range(self.seq_len - 1, len(frame) - self.pred_len):
                if str(frame.loc[origin, "split"]) != split: continue
                target_split = frame.loc[origin + 1:origin + self.pred_len, "split"]
                if len(target_split) != self.pred_len or not target_split.eq(split).all(): continue
                if not bool(mask.iloc[origin]): continue
                self.rows.append((position, origin))
        self.station_ids = {position: index + 1 for index, position in enumerate(self.positions)}
        self.reserve_ids = {site: index + 1 for index, site in enumerate(sorted({position.split("-", 1)[0] for position in self.positions}))}
    def __len__(self): return len(self.rows)
    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawned Windows workers only need compact arrays; the main-process
        # dataset retains frames for key-manifest generation.
        state["prepared"] = {}
        return state
    def key_frame(self) -> pd.DataFrame:
        records = []
        for position, origin in self.rows:
            frame = self.prepared[position]
            for horizon, target_time in enumerate(frame.loc[origin + 1:origin + self.pred_len, "date"], start=1):
                records.append({"station": position, "aquatic_site": position.split("-", 1)[0], "origin": frame.loc[origin, "date"], "target_time": target_time, "horizon": horizon, "target_mask": bool(frame.loc[origin + horizon, "dissolvedOxygen_observed"])})
        return pd.DataFrame(records)
    def __getitem__(self, index):
        position, origin = self.rows[index]; arrays = self.arrays[position]
        if not bool(arrays["do_observed"][origin]): raise RuntimeError("invalid terminal DO")
        start = origin - self.seq_len + 1; stop = origin + 1; future_stop = origin + self.pred_len + 1
        water = arrays["water"][start:stop]; met = arrays["met"][start:stop]
        water_mask = arrays["water_mask"][start:stop]; met_mask = arrays["met_mask"][start:stop]
        target = arrays["do"][origin + 1:future_stop].astype(float); origin_do = float(arrays["do"][origin])
        target_mask = arrays["do_observed"][origin + 1:future_stop]
        seasonal_steps = int(24 * 60 / self.frequency_minutes)
        seasonal_index = np.arange(origin + 1, origin + self.pred_len + 1) - seasonal_steps
        seasonal = arrays["do"][seasonal_index]
        seasonal_mask = arrays["do_observed"][seasonal_index]
        stats = self.scaling_stats.get(position, {})
        water_stats = stats.get("water", {})
        met_stats = stats.get("meteorology", {})
        do_item = water_stats.get("dissolvedOxygen", {})
        do_mean, do_scale = float(do_item.get("mean", 0.0)), float(do_item.get("scale", 1.0) or 1.0)
        target = (target - origin_do) / do_scale
        origin_do_scaled = (origin_do - do_mean) / do_scale
        seasonal_scaled = (seasonal - do_mean) / do_scale
        missing = self.water_missing[position][origin - self.seq_len + 1:origin + 1]
        rain_start = max(0, origin - seasonal_steps)
        rain_values = arrays["precipitation"][rain_start:origin]
        rain_mask = arrays["precipitation_observed"][rain_start:origin]
        rain_label_available = bool(rain_mask.all()) if len(rain_mask) else False
        rain_label = bool(rain_label_available and np.any((rain_values > 0) & rain_mask))
        return {"water": torch.from_numpy(water).float(), "water_mask": torch.from_numpy(water_mask), "water_missing": torch.from_numpy(missing).float(), "met": torch.from_numpy(met).float(), "met_mask": torch.from_numpy(met_mask), "target": torch.from_numpy(np.nan_to_num(target)).float(), "target_mask": torch.from_numpy(target_mask), "last_do": torch.tensor(origin_do_scaled).float(), "last_do_physical": torch.tensor(origin_do).float(), "seasonal_persistence": torch.from_numpy(np.nan_to_num(seasonal_scaled)).float(), "seasonal_persistence_mask": torch.from_numpy(seasonal_mask), "station_id": torch.tensor(self.station_ids[position]), "reserve_id": torch.tensor(self.reserve_ids[position.split("-", 1)[0]]), "position": position, "site": position.split("-", 1)[0], "station": position, "origin": arrays["date"][origin], "target_times": list(arrays["date"][origin + 1:future_stop]), "input_missing_rate": float((~water_mask).mean()), "rain_label": rain_label, "rain_label_available": rain_label_available}

def future_or_history_values(frame, columns): return frame.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=True)
