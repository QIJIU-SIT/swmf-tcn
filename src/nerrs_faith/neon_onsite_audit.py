"""Admission audits for the isolated WLOU/CUPE/SYCA experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nerrs_faith.neon_io import discover_neon_sources, discover_sensor_positions, select_active_positions
from nerrs_faith.neon_pipeline import artifact_hash


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    radius = 6371.0088
    lat_a, lat_b = math.radians(float(latitude_a)), math.radians(float(latitude_b))
    dlat = lat_b - lat_a
    dlon = math.radians(float(longitude_b) - float(longitude_a))
    value = math.sin(dlat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(dlon / 2) ** 2
    return float(2 * radius * math.asin(math.sqrt(min(1.0, value))))


def _split_summary(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    splits = config["splits"]
    return {
        "train": {"start": str(splits.get("train_start", "")), "end": str(splits["train_end"])},
        "val": {"start": str(splits.get("val_start", "")), "end": str(splits["val_end"])},
        "test": {"start": str(splits.get("test_start", "")), "end": str(splits["test_end"])},
    }


def audit_onsite_tables(
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    reasons: list[str] = []
    split_report = _split_summary(config)
    positions: dict[str, Any] = {}
    seq_len = int(config.get("window", {}).get("seq_len", 256))
    pred_len = int(config.get("window", {}).get("pred_len", 96))
    for position, original in tables.items():
        frame = original.copy()
        date = pd.to_datetime(frame.get("date", frame.index), utc=True)
        if date.duplicated().any():
            reasons.append(f"duplicate timestamps in {position}")
        if "input_dependency_max_time" in frame:
            dependency = pd.to_datetime(frame["input_dependency_max_time"], utc=True)
            if bool((dependency > date).fillna(False).any()):
                reasons.append(f"future input dependency in {position}")
        if "target_imputed" in frame and bool(frame["target_imputed"].fillna(False).astype(bool).any()):
            reasons.append(f"imputed target in {position}")
        split_counts = {name: int(frame.get("split", pd.Series("excluded", index=frame.index)).eq(name).sum()) for name in split_report}
        observed_counts = {column: int(frame[column].fillna(False).astype(bool).sum()) for column in frame if column.endswith("_observed")}
        window_counts = {name: 0 for name in split_report}
        horizon96_counts = {name: 0 for name in split_report}
        rain_counts = {"rain": 0, "non_rain": 0, "unavailable": 0}
        split_values = frame.get("split", pd.Series("excluded", index=frame.index)).astype(str).to_numpy()
        target_observed = frame.get("dissolvedOxygen_observed", pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy()
        precip = pd.to_numeric(frame.get("precipitation_precipBulk", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy()
        precip_observed = frame.get("precipitation_precipBulk_observed", pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy()
        for origin in range(seq_len - 1, len(frame) - pred_len):
            split = split_values[origin]
            if split not in window_counts or not np.all(split_values[origin + 1:origin + pred_len + 1] == split):
                continue
            window_counts[split] += 1
            horizon96_counts[split] += int(target_observed[origin + pred_len])
            if split in {"val", "test"}:
                history_mask = precip_observed[max(0, origin - 96):origin]
                if len(history_mask) != 96 or not history_mask.all():
                    rain_counts["unavailable"] += 1
                elif np.any(precip[max(0, origin - 96):origin] > 0):
                    rain_counts["rain"] += 1
                else:
                    rain_counts["non_rain"] += 1
        for split in required_splits:
            if split not in split_report:
                raise ValueError(f"unknown required split: {split}")
            if window_counts[split] == 0 or horizon96_counts[split] == 0:
                reasons.append(f"{position} has no admissible {split} windows or horizon-96 targets")
        positions[position] = {"rows": int(len(frame)), "split_counts": split_counts, "observed_counts": observed_counts, "window_counts": window_counts, "horizon96_target_counts": horizon96_counts, "weather_counts": rain_counts}
    return {"status": "passed" if not reasons else "failed", "reasons": reasons, "splits": split_report, "positions": positions}


def audit_onsite_sources(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Read source inventories and provenance; never writes prepared data."""
    if isinstance(config, (str, Path)):
        import yaml
        config = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    reasons: list[str] = []
    inventory = None
    source_records: dict[str, Any] = {}
    try:
        inventory = discover_neon_sources(config)
        for site, site_cfg in config["sites"].items():
            selected = {}
            water_records = discover_sensor_positions(config["raw"]["water_roots"][site], site=site, product=config["raw"]["products"]["water"])
            selected["water"] = select_active_positions(
                water_records, site=site,
                start=config["splits"]["train_start"], end=config["splits"]["test_end"],
            )
            for kind, root_key, product_key, source_site in (
                ("humidity", "humidity_roots", "humidity", site_cfg["humidity_site"]),
                ("precipitation", "precipitation_roots", "precipitation", site_cfg["precipitation_site"]),
            ):
                try:
                    records = discover_sensor_positions(config["raw"][root_key][source_site], site=source_site, product=config["raw"]["products"][product_key])
                    selected[kind] = records
                except (FileNotFoundError, ValueError) as exc:
                    reasons.append(f"{site} {kind} sensor-position provenance: {exc}")
            distances = []
            for water in selected["water"]:
                for kind in ("humidity", "precipitation"):
                    for met in selected.get(kind, []):
                        distances.append({"water_hor_ver": water["hor_ver"], "meteorology": kind, "meteorology_hor_ver": met["hor_ver"], "distance_km": haversine_km(water["latitude"], water["longitude"], met["latitude"], met["longitude"])})
            source_records[site] = {"positions": selected, "distances": distances}
    except (FileNotFoundError, ValueError) as exc:
        reasons.append(str(exc))
    payload = {
        "protocol": config.get("protocol"),
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "splits": _split_summary(config),
        "products": config.get("raw", {}).get("products", {}),
        "source_hash": getattr(inventory, "source_hash", None),
        "sources": source_records,
        "observation_values_loaded": False,
    }
    payload["audit_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return payload


def write_source_audit(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    payload = audit_onsite_sources(config)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload
