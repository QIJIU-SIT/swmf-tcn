"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from nerrs_faith.neon_data import NeonWindowDataset

def _validation_data(output_root: Path, config: Mapping[str, Any]):
    import pandas as pd
    data_dir = output_root / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    scalers = json.loads((data_dir / "scaler.json").read_text(encoding="utf-8"))
    val_end = pd.Timestamp(config["splits"]["val_end"])
    tables = {}
    for position in manifest["positions"]:
        frame = pd.read_parquet(data_dir / f"{position}.parquet", filters=[("date", "<=", val_end.to_pydatetime())])
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
        tables[position] = frame
    kwargs = {"seq_len": int(config["window"]["seq_len"]), "pred_len": int(config["window"]["pred_len"]), "frequency_minutes": int(config["aggregation"]["frequency_minutes"]), "scaling_stats": scalers}
    return NeonWindowDataset(tables, split="train", **kwargs), NeonWindowDataset(tables, split="val", **kwargs), scalers, manifest
