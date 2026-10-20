"""Physical persistence baselines on the frozen NEON keys."""
from __future__ import annotations
import numpy as np
import pandas as pd
from nerrs_faith.neon_audit import key_hash

def build_baseline_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"site", "position", "origin", "target_time", "dissolvedOxygen", "dissolvedOxygen_observed", "dissolvedOxygen_lag96", "dissolvedOxygen_lag96_observed", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing: raise ValueError("baseline input missing: " + ", ".join(missing))
    out = pd.DataFrame({"split": frame["split"], "station": frame["position"], "aquatic_site": frame["site"], "origin": frame["origin"], "target_time": frame["target_time"], "horizon": 96, "target": frame["dissolvedOxygen"], "target_mask": frame["dissolvedOxygen_observed"].astype(bool), "persistence": frame["dissolvedOxygen"], "seasonal_persistence": frame["dissolvedOxygen_lag96"], "seasonal_persistence_mask": frame["dissolvedOxygen_lag96_observed"].astype(bool)})
    out["prediction"] = np.where(out["seasonal_persistence_mask"], out["seasonal_persistence"], out["persistence"])
    out["model"] = "seasonal_24h_with_persistence_fallback"
    return out

def build_dataset_baseline_predictions(dataset) -> pd.DataFrame:
    """Produce both baselines on every complete validation trajectory."""
    chunks = list(iter_dataset_baseline_predictions(dataset))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def iter_dataset_baseline_predictions(dataset, *, chunk_origins: int = 1024):
    """Yield bounded chunks without retaining the full validation key table."""
    seasonal_steps = int(24 * 60 / dataset.frequency_minutes)
    horizons = np.arange(1, dataset.pred_len + 1, dtype=np.int16)
    for position in dataset.positions:
        origins = np.asarray([origin for row_position, origin in dataset.rows if row_position == position], dtype=np.int64)
        frame = dataset.prepared[position]
        dates = frame["date"].to_numpy()
        dissolved_oxygen = frame["dissolvedOxygen"].to_numpy(float)
        observed = frame.get("dissolvedOxygen_observed", pd.Series(True, index=frame.index)).to_numpy(bool)
        for start in range(0, len(origins), chunk_origins):
            block = origins[start:start + chunk_origins]
            target_indices = block[:, None] + horizons[None, :]
            seasonal_indices = target_indices - seasonal_steps
            origin_values = dissolved_oxygen[block]
            seasonal_valid = observed[seasonal_indices]
            seasonal_values = dissolved_oxygen[seasonal_indices]
            persistence = np.repeat(origin_values, dataset.pred_len)
            seasonal_flat = seasonal_values.reshape(-1)
            seasonal_mask = seasonal_valid.reshape(-1)
            prediction = np.where(seasonal_mask, seasonal_flat, persistence)
            yield pd.DataFrame({
                "split": dataset.split,
                "station": position,
                "aquatic_site": position.split("-", 1)[0],
                "origin": np.repeat(dates[block], dataset.pred_len),
                "target_time": dates[target_indices].reshape(-1),
                "horizon": np.tile(horizons, len(block)),
                "target": dissolved_oxygen[target_indices].reshape(-1),
                "target_mask": observed[target_indices].reshape(-1),
                "persistence": persistence,
                "seasonal_persistence": np.where(seasonal_mask, seasonal_flat, persistence),
                "seasonal_persistence_mask": seasonal_mask,
                "prediction": prediction,
                "model": "seasonal_24h_with_persistence_fallback",
            })

def baseline_key_hash(frame): return key_hash(frame)
