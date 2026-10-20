"""Causal prepared-data and common-key helpers for the onsite experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import re

from nerrs_faith.neon_audit import key_hash
from nerrs_faith.neon_data import NeonWindowDataset
from nerrs_faith.neon_pipeline import dataframe_hash, fit_neon_scalers
from nerrs_faith.neon_io import NeonSourceInventory


def inventory_through_month(inventory: NeonSourceInventory, end_month: str, *, start_month: str | None = None) -> NeonSourceInventory:
    """Return an inventory view whose observation files lie in the closed month range."""
    pattern = re.compile(r"\.(\d{4}-\d{2})\.basic", re.I)
    files = {}
    for kind, sites in inventory.files.items():
        files[kind] = {}
        for site, paths in sites.items():
            selected = []
            for path in paths:
                match = pattern.search(str(path))
                if match and (start_month is None or match.group(1) >= start_month) and match.group(1) <= end_month:
                    selected.append(path)
            if not selected:
                raise ValueError(f"no {kind} files for {site} in requested month range")
            files[kind][site] = tuple(selected)
    return NeonSourceInventory(inventory.products, inventory.precipitation_mapping, files, inventory.releases, inventory.source_hash)


def validate_causal_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    for position, frame in tables.items():
        date = pd.to_datetime(frame.get("date", frame.index), utc=True)
        if date.duplicated().any():
            raise ValueError(f"duplicate dates in {position}")
        if "input_dependency_max_time" in frame:
            dependency = pd.to_datetime(frame["input_dependency_max_time"], utc=True)
            if bool((dependency > date).fillna(False).any()):
                raise ValueError(f"future dependency in {position}")
        if "target_imputed" in frame and bool(frame["target_imputed"].fillna(False).astype(bool).any()):
            raise ValueError(f"imputed target in {position}")
        for column in frame.columns:
            if column.endswith("_observed"):
                values = frame[column].dropna().astype(bool)
                if values.empty:
                    raise ValueError(f"missing observed provenance in {position}: {column}")


def _simple_keys(frame: pd.DataFrame, position: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame.get("date", frame.index), utc=True)
    mask = frame.get("target_mask", frame.get("dissolvedOxygen_observed", pd.Series(True, index=frame.index))).astype(bool)
    return pd.DataFrame({
        "station": position,
        "aquatic_site": position.split("-", 1)[0],
        "origin": dates,
        "target_time": dates,
        "horizon": 0,
        "target_mask": mask.to_numpy(bool),
    })


def build_validation_key_manifest(tables: Mapping[str, pd.DataFrame], *, split: str, identity: Mapping[str, Any], key_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    frames = []
    if key_frame is not None:
        frames.append(key_frame.copy())
    else:
        frames.extend(_simple_keys(frame[frame.get("split", pd.Series(split, index=frame.index)).eq(split)], position) for position, frame in sorted(tables.items()))
    keys = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["station", "origin", "target_time", "horizon", "target_mask"])
    if keys.empty:
        raise ValueError("cannot freeze an empty validation key set")
    required = ["station", "origin", "target_time", "horizon", "target_mask"]
    missing = [column for column in required if column not in keys]
    if missing:
        raise ValueError("validation keys missing " + ", ".join(missing))
    if "aquatic_site" not in keys:
        keys["aquatic_site"] = keys["station"].astype(str).str.split("-", n=1).str[0]
    keys["origin"] = pd.to_datetime(keys["origin"], utc=True)
    keys["target_time"] = pd.to_datetime(keys["target_time"], utc=True)
    keys = keys.sort_values(required[:-1]).reset_index(drop=True)
    if keys.duplicated(required).any():
        raise ValueError("validation keys contain duplicates")
    payload = {**dict(identity), "split": split, "rows": int(len(keys)), "key_hash": key_hash(keys[["aquatic_site", *required]]), "key_chunk_rows": 262144}
    return payload


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_onsite_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve the physical timestamp when normalizing indexed pipeline tables."""
    normalized = frame.copy()
    if "date" in normalized:
        normalized["date"] = pd.to_datetime(normalized["date"], utc=True)
    else:
        normalized["date"] = pd.to_datetime(normalized.index, utc=True)
    normalized = normalized.set_index("date", drop=False)
    for name in ("tempRHMean", "RHMean"):
        if name in normalized and f"humidity_{name}" not in normalized:
            normalized[f"humidity_{name}"] = normalized[name]
            if f"{name}_observed" in normalized:
                normalized[f"humidity_{name}_observed"] = normalized[f"{name}_observed"]
    if "precipBulk" in normalized and "precipitation_precipBulk" not in normalized:
        normalized["precipitation_precipBulk"] = normalized["precipBulk"]
        if "precipBulk_observed" in normalized:
            normalized["precipitation_precipBulk_observed"] = normalized["precipBulk_observed"]
    return normalized


def write_onsite_artifacts(tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], output: str | Path, *, source_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validate_causal_tables(tables)
    root = Path(output); data_dir = root / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    normalized_tables = {}
    for position, frame in tables.items():
        normalized_tables[position] = normalize_onsite_table(frame)
    scalers = fit_neon_scalers(normalized_tables, config)
    source_audit = dict(source_audit or {})
    source_hash = source_audit.get("source_hash")
    if not source_hash:
        source_hash = hashlib.sha256(json.dumps(source_audit, sort_keys=True, default=str).encode()).hexdigest()
    table_hashes = {}
    axis_hashes = {}
    for position, frame in sorted(normalized_tables.items()):
        stored = frame.copy().reset_index(drop=True)
        if "date" not in stored:
            raise ValueError(f"normalized onsite table has no date column: {position}")
        stored.to_parquet(data_dir / f"{position}.parquet", index=False)
        table_hashes[position] = dataframe_hash(stored)
        axis_hashes[position] = hashlib.sha256(pd.to_datetime(stored["date"], utc=True).astype(str).str.cat(sep="\n").encode()).hexdigest()
    scaler_path = data_dir / "scaler.json"; _atomic_json(scalers, scaler_path)
    manifest = {"protocol": config["protocol"], "positions": sorted(tables), "source_hash": source_hash, "scaler_hash": hashlib.sha256(scaler_path.read_bytes()).hexdigest(), "table_hashes": table_hashes, "axis_hashes": axis_hashes, "config_hash": hashlib.sha256(json.dumps(dict(config), sort_keys=True, default=str).encode()).hexdigest()}
    _atomic_json(manifest, data_dir / "manifest.json")
    prepared = {position: pd.read_parquet(data_dir / f"{position}.parquet") for position in normalized_tables}
    dataset = NeonWindowDataset(prepared, split="val", seq_len=int(config["window"]["seq_len"]), pred_len=int(config["window"]["pred_len"]), frequency_minutes=int(config["aggregation"]["frequency_minutes"]), scaling_stats=scalers)
    key_frame = dataset.key_frame()
    key_manifest = build_validation_key_manifest(prepared, split="val", identity={"source_hash": source_hash, "scaler_hash": manifest["scaler_hash"], "axis_hashes": axis_hashes, "table_hashes": table_hashes}, key_frame=key_frame)
    key_frame.to_parquet(data_dir / "validation_keys.parquet", index=False)
    _atomic_json(key_manifest, data_dir / "validation_key_manifest.json")
    _atomic_json({"protocol": config["protocol"], "status": "sealed"}, root / "test_lock.json")
    return {**manifest, "validation_key_hash": key_manifest["key_hash"], "validation_key_rows": key_manifest["rows"]}
