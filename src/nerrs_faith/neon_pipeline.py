"""Causal QA-aware NEON aggregation and lightweight position-table helpers."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

def aggregate_15min(frame: pd.DataFrame, value: str, qf: str, *, method: str = "mean", minimum: int = 12) -> pd.DataFrame:
    if value not in frame or qf not in frame or "startDateTime" not in frame:
        raise ValueError("raw NEON frame requires startDateTime, value, and final QA columns")
    source = frame.copy(); source["timestamp"] = pd.to_datetime(source["startDateTime"], utc=True)
    source[value] = pd.to_numeric(source[value], errors="coerce")
    source["valid"] = source[qf].astype(str).isin({"0", "0.0"}) & source[value].notna()
    source["valid_value"] = source[value].where(source["valid"])
    source["dependency_candidate"] = source["timestamp"].where(source["valid"])
    grouped = source.set_index("timestamp").sort_index().resample("15min", label="right", closed="right")
    count = grouped["valid"].sum().astype(int)
    observed = count >= int(minimum)
    values = grouped["valid_value"].agg("sum" if method == "sum" else "mean").where(observed)
    result = pd.DataFrame({"value": values, "valid_count": count, "observed": observed})
    result["dependency_max_time"] = grouped["dependency_candidate"].max().where(observed)
    return result

def aggregate_mean_15min(frame, value, qf, *, minimum=12):
    return aggregate_15min(frame, value, qf, method="mean", minimum=minimum)

def aggregate_sum_15min(frame, value, qf, *, minimum=12):
    return aggregate_15min(frame, value, qf, method="sum", minimum=minimum)

def _split_label(index: pd.Timestamp, splits: Mapping[str, Any]) -> str:
    index = pd.Timestamp(index)
    for split in ("train", "val", "test"):
        start = pd.Timestamp(splits[f"{split}_start"])
        end = pd.Timestamp(splits[f"{split}_end"])
        if start <= index <= end:
            return split
    return "excluded"

def build_position_table(site: str, position: str, tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> pd.DataFrame:
    water = tables["water"].copy(); water.index = pd.to_datetime(water.index, utc=True)
    humidity = tables["humidity"].copy(); humidity.index = pd.to_datetime(humidity.index, utc=True)
    precip = tables["precipitation"].copy(); precip.index = pd.to_datetime(precip.index, utc=True)
    result = water.join(humidity.add_prefix("humidity_"), how="outer").join(precip.add_prefix("precipitation_"), how="outer").sort_index()
    result["site"], result["position"] = site, f"{site}-{position}"
    site_cfg = config["sites"][site]
    result["met_site_humidity"] = site_cfg["humidity_site"]
    result["met_site_precipitation"] = site_cfg["precipitation_site"]
    result["precipitation_distance_km"] = float(site_cfg.get("precipitation_distance_km", float("nan")))
    result["split"] = [_split_label(ts, config["splits"]) for ts in result.index]
    return result

def fit_training_scalers(table: pd.DataFrame, variables, *, train_end) -> dict[str, dict[str, float]]:
    end = pd.Timestamp(train_end); out = {}
    for variable in variables:
        selection = table.index <= end
        if "split" in table:
            selection = selection & table["split"].eq("train").to_numpy()
        observed_column = f"{variable}_observed"
        if observed_column in table:
            selection = selection & table[observed_column].astype(bool).to_numpy()
        valid = pd.to_numeric(table.loc[selection, variable], errors="coerce").dropna()
        out[variable] = {"mean": float(valid.mean()) if len(valid) else 0.0, "scale": float(valid.std(ddof=0)) if len(valid) and valid.std(ddof=0) > 0 else 1.0}
    return out

def artifact_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

def dataframe_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    if "date" not in canonical:
        canonical = canonical.reset_index().rename(columns={canonical.index.name or "index": "date"})
    else:
        canonical = canonical.reset_index(drop=True)
    return hashlib.sha256(canonical.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()).hexdigest()

def _atomic_text(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)

def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)

def load_one_minute_product(paths, *, columns, value_columns, qf_columns, methods, minimum=12):
    frames = []
    for path in sorted(paths):
        usecols = ["startDateTime", *value_columns, *qf_columns]
        frame = pd.read_csv(path, usecols=lambda name: name in usecols)
        frames.append(frame)
    if not frames: raise ValueError("no NEON one-minute files selected")
    combined = pd.concat(frames, ignore_index=True)
    outputs = []
    for value, qf, method in zip(value_columns, qf_columns, methods):
        aggregated = aggregate_15min(combined, value, qf, method=method, minimum=minimum).rename(columns={"value": value})
        aggregated[f"{value}_observed"] = aggregated["observed"]
        aggregated[f"{value}_dependency_max_time"] = aggregated["dependency_max_time"]
        outputs.append(aggregated[[value, "valid_count", f"{value}_observed", f"{value}_dependency_max_time"]].rename(columns={"valid_count": f"{value}_valid_count"}))
    return pd.concat(outputs, axis=1).sort_index()

def build_neon_position_tables(config: Mapping[str, Any], inventory) -> dict[str, pd.DataFrame]:
    """Build the four aligned position tables from a discovered inventory."""
    import re
    tables = {}
    for site, site_cfg in config["sites"].items():
        water_paths = inventory.files["water"][site]
        humidity_paths = inventory.files["humidity"][site_cfg["humidity_site"]]
        precip_paths = inventory.files["precipitation"][site_cfg["precipitation_site"]]
        humidity = load_one_minute_product(humidity_paths, columns=(), value_columns=["tempRHMean", "RHMean"], qf_columns=["tempRHFinalQF", "RHFinalQF"], methods=["mean", "mean"], minimum=int(config["aggregation"]["required_valid_minutes"]))
        precipitation = load_one_minute_product(precip_paths, columns=(), value_columns=["precipBulk"], qf_columns=["finalQF"], methods=["sum"], minimum=int(config["aggregation"]["required_valid_minutes"]))
        for position in site_cfg["positions"]:
            selected = [path for path in water_paths if re.search(rf"\.{re.escape(position)}\.waq_", Path(path).name)]
            if not selected: raise ValueError(f"missing water position files: {site}-{position}")
            water = load_one_minute_product(selected, columns=(), value_columns=["dissolvedOxygen", "pH", "specificConductance", "turbidity"], qf_columns=["dissolvedOxygenFinalQF", "pHFinalQF", "specificCondFinalQF", "turbidityFinalQF"], methods=["mean"] * 4, minimum=int(config["aggregation"]["required_valid_minutes"]))
            table = build_position_table(site, position, {"water": water, "humidity": humidity, "precipitation": precipitation}, config)
            table["input_dependency_max_time"] = pd.concat([table.get(f"{v}_dependency_max_time", pd.Series(pd.NaT, index=table.index)) for v in ["dissolvedOxygen", "pH", "specificConductance", "turbidity", "humidity_tempRHMean", "humidity_RHMean", "precipitation_precipBulk"]], axis=1).max(axis=1)
            tables[f"{site}-{position}"] = table
    return tables

def fit_neon_scalers(tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> dict[str, Any]:
    train_end = pd.Timestamp(config["splits"]["train_end"])
    result = {}
    water = config["variables"]["water"]; meteorology = config["variables"]["meteorology"]
    for position, table in tables.items():
        met_columns = [f"humidity_{v}" if v in {"tempRHMean", "RHMean"} else f"precipitation_{v}" for v in meteorology]
        met_stats = fit_training_scalers(table, met_columns, train_end=train_end)
        result[position] = {
            "water": fit_training_scalers(table, water, train_end=train_end),
            "meteorology": {name: met_stats[column] for name, column in zip(meteorology, met_columns)},
        }
    return result


def coverage_metadata(table: pd.DataFrame, *, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Summarize raw/QC/aggregation coverage without redefining split boundaries."""
    frame = table.copy()
    result = {"rows": int(len(frame)), "split_counts": {split: int(frame.get("split", pd.Series(dtype=str)).eq(split).sum()) for split in ("train", "val", "test")}}
    for column in frame.columns:
        if column.endswith("_observed"):
            result.setdefault("observed_counts", {})[column] = int(frame[column].fillna(False).astype(bool).sum())
    if "precipitation_precipBulk" in frame:
        result["positive_precipitation_rows"] = int((pd.to_numeric(frame["precipitation_precipBulk"], errors="coerce").fillna(0) > 0).sum())
    return result

def write_neon_artifacts(tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], output: str | Path, *, source_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(output); data_dir = root / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    scalers = fit_neon_scalers(tables, config)
    config_hash = artifact_hash({"aggregation": config["aggregation"], "variables": config["variables"], "splits": config["splits"], "window": config["window"]})
    source_metadata = dict(source_metadata or {})
    source_hash = source_metadata.get("source_hash")
    existing = data_dir / "manifest.json"
    if existing.is_file():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        if prior.get("config_hash") != config_hash or sorted(prior.get("positions", [])) != sorted(tables) or prior.get("source_hash") != source_hash:
            raise FileExistsError("existing NEON artifacts have an incompatible fingerprint")
        return prior
    for position, table in tables.items():
        stored = table.reset_index()
        if "date" not in stored:
            stored = stored.rename(columns={stored.columns[0]: "date"})
        _atomic_parquet(stored, data_dir / f"{position}.parquet")
    _atomic_text(data_dir / "scaler.json", scalers)
    lock = root / "test_lock.json"
    if lock.is_file():
        existing_lock = json.loads(lock.read_text(encoding="utf-8"))
        if existing_lock.get("status") != "sealed":
            raise PermissionError("existing NEON test lock is not sealed")
    else:
        _atomic_text(lock, {"protocol": "neon_protocol_v0", "status": "sealed"})
    scaler_hash = hashlib.sha256((data_dir / "scaler.json").read_bytes()).hexdigest()
    manifest = {"protocol": "neon_protocol_v0", "positions": sorted(tables), "config_hash": config_hash, "scaler_hash": scaler_hash, "axis_hashes": {position: artifact_hash([str(value) for value in pd.to_datetime(table.index, utc=True)]) for position, table in tables.items()}, "table_hashes": {position: dataframe_hash(table) for position, table in tables.items()}, **source_metadata}
    _atomic_text(data_dir / "manifest.json", manifest)
    return manifest
