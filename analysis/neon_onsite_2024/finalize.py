"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PAIR_KEY_COLUMNS = ["aquatic_site", "station", "origin", "target_time", "horizon"]
HORIZON_BANDS = {"0_1h": (1, 4), "1_6h": (5, 24), "6_12h": (25, 48), "12_24h": (49, 96)}

def _iter_batches(path: Path, columns: list[str], batch_rows: int):
    if int(batch_rows) <= 0:
        raise ValueError("batch_rows must be positive")
    parquet = pq.ParquetFile(path)
    missing = sorted(set(columns).difference(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"prediction artifact missing columns: {', '.join(missing)}")
    yield from parquet.iter_batches(columns=columns, batch_size=int(batch_rows))

def _bootstrap(blocks: pd.DataFrame, *, training_seed: int, resamples: int, statistic: str = "f4_mae_minus_tcn_mv_mae") -> dict[str, Any]:
    if blocks.empty or not np.isfinite(blocks[["sum", "count"]].to_numpy(float)).all():
        raise ValueError("paired bootstrap has no finite station-date blocks")
    bootstrap_seed = 2026 + int(training_seed)
    rng = np.random.default_rng(bootstrap_seed)
    station_samples = []
    station_estimates = []
    for station in sorted(blocks["station"].astype(str).unique()):
        station_blocks = blocks.loc[blocks["station"].astype(str) == station]
        sums = station_blocks["sum"].to_numpy(float)
        counts = station_blocks["count"].to_numpy(float)
        station_estimates.append(float(sums.sum() / counts.sum()))
        indices = rng.integers(0, len(station_blocks), size=(int(resamples), len(station_blocks)))
        station_samples.append(sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
    sampled = np.mean(np.stack(station_samples, axis=1), axis=1)
    return {
        "estimate": float(np.mean(station_estimates)),
        "lower": float(np.percentile(sampled, 2.5)),
        "upper": float(np.percentile(sampled, 97.5)),
        "blocks": int(len(blocks)),
        "positions": int(len(station_estimates)),
        "resamples": int(resamples),
        "training_seed": int(training_seed),
        "bootstrap_seed": bootstrap_seed,
        "statistic": statistic,
    }
