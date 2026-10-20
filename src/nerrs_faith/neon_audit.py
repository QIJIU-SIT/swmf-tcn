"""Fail-closed NEON leakage, key, and sealed-test checks."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import hashlib
import numpy as np

def audit_neon_table(frame: pd.DataFrame) -> dict:
    if "input_dependency_max_time" in frame and "origin" in frame:
        dependency = pd.to_datetime(frame["input_dependency_max_time"], utc=True)
        origin = pd.to_datetime(frame["origin"], utc=True)
        if (dependency > origin).any(): raise ValueError("future input dependency detected")
    if "split" in frame and set(frame["split"].dropna().astype(str)) - {"train", "val", "test", "excluded"}:
        raise ValueError("unknown split")
    return {"rows": int(len(frame)), "split_counts": frame.get("split", pd.Series(dtype=str)).value_counts().to_dict()}

def require_sealed_test_lock(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "sealed": raise PermissionError("NEON test lock is not sealed")
    return payload

def key_hash(frame: pd.DataFrame) -> str:
    columns = ["aquatic_site", "station", "origin", "target_time", "horizon", "target_mask"]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError("prediction table missing key columns: " + ", ".join(missing))
    values = frame[columns].copy()
    for column in ("origin", "target_time"):
        values[column] = pd.to_datetime(values[column], utc=True).astype("int64")
    values["target_mask"] = values["target_mask"].astype(bool)
    values = values.sort_values(columns).drop_duplicates()
    if len(values) != len(frame):
        raise ValueError("prediction table contains duplicate evaluation keys")
    return hashlib.sha256(values.to_csv(index=False, lineterminator="\n").encode()).hexdigest()

def chunked_key_hash(frame: pd.DataFrame, chunk_rows: int) -> str:
    hashes = []
    for station in sorted(frame["station"].astype(str).unique()):
        partition = frame[frame["station"].astype(str) == station]
        hashes.extend(key_hash(partition.iloc[start:start + chunk_rows]) for start in range(0, len(partition), chunk_rows))
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()

def parquet_key_hash(path: str | Path) -> tuple[str, int]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    hashes = []
    rows = 0
    columns = ["aquatic_site", "station", "origin", "target_time", "horizon", "target_mask"]
    for index in range(parquet.num_row_groups):
        frame = parquet.read_row_group(index, columns=columns).to_pandas()
        hashes.append(key_hash(frame)); rows += len(frame)
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest(), rows


def parquet_chunked_key_hash(path: str | Path, chunk_rows: int) -> str:
    """Compute ``chunked_key_hash`` without loading a complete Parquet table."""
    import pyarrow.parquet as pq

    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    columns = ["aquatic_site", "station", "origin", "target_time", "horizon", "target_mask"]
    buffers: dict[str, list[pd.DataFrame]] = {}
    counts: dict[str, int] = {}
    hashes: dict[str, list[str]] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=int(chunk_rows)):
        frame = batch.to_pandas()
        for station, partition in frame.groupby(frame["station"].astype(str), sort=False):
            buffers.setdefault(station, []).append(partition)
            counts[station] = counts.get(station, 0) + len(partition)
            while counts[station] >= int(chunk_rows):
                combined = pd.concat(buffers[station], ignore_index=True)
                hashes.setdefault(station, []).append(key_hash(combined.iloc[:chunk_rows]))
                remainder = combined.iloc[chunk_rows:].copy()
                buffers[station] = [remainder] if len(remainder) else []
                counts[station] = len(remainder)
    for station, count in counts.items():
        if count:
            hashes.setdefault(station, []).append(key_hash(pd.concat(buffers[station], ignore_index=True)))
    ordered = [value for station in sorted(hashes) for value in hashes[station]]
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()


def parquet_streaming_key_hash(path: str | Path, batch_rows: int = 262_144) -> str:
    """Compute the legacy full-table ``key_hash`` for globally sorted Parquet."""
    import pyarrow.parquet as pq

    if int(batch_rows) <= 0:
        raise ValueError("batch_rows must be positive")
    columns = ["aquatic_site", "station", "origin", "target_time", "horizon", "target_mask"]
    digest = hashlib.sha256()
    first = True
    last_key = None
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=int(batch_rows)):
        values = batch.to_pandas()
        for column in ("origin", "target_time"):
            values[column] = pd.to_datetime(values[column], utc=True).astype("int64")
        values["target_mask"] = values["target_mask"].astype(bool)
        ordered = values.sort_values(columns).reset_index(drop=True)
        if not values.reset_index(drop=True).equals(ordered):
            raise ValueError("Parquet evaluation keys are not globally sorted")
        tuples = list(ordered.itertuples(index=False, name=None))
        if tuples:
            if last_key is not None and tuples[0] <= last_key:
                raise ValueError("Parquet evaluation keys are duplicated or not globally sorted")
            last_key = tuples[-1]
        digest.update(ordered.to_csv(index=False, header=first, lineterminator="\n").encode())
        first = False
        rows += len(ordered)
    if rows == 0:
        raise ValueError("Parquet evaluation keys are empty")
    return digest.hexdigest()

def audit_neon_artifacts(output: str | Path, *, config=None) -> dict:
    root = Path(output); require_sealed_test_lock(root / "test_lock.json")
    data_dir = root / "data"
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("NEON derived manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config is not None:
        required_manifest = {"source_hash", "scaler_hash", "axis_hashes", "table_hashes"}
        missing_manifest = sorted(required_manifest.difference(manifest))
        if missing_manifest:
            raise ValueError("NEON manifest missing frozen fingerprints: " + ", ".join(missing_manifest))
        scaler_path = data_dir / "scaler.json"
        if hashlib.sha256(scaler_path.read_bytes()).hexdigest() != manifest["scaler_hash"]:
            raise ValueError("NEON scaler fingerprint mismatch")
    summaries = {}
    for position in manifest.get("positions", []):
        table = pd.read_parquet(data_dir / f"{position}.parquet")
        if config is not None:
            from nerrs_faith.neon_pipeline import artifact_hash, dataframe_hash
            if dataframe_hash(table) != manifest["table_hashes"].get(position):
                raise ValueError(f"table fingerprint mismatch: {position}")
            dates_for_hash = table.index if "date" not in table else table["date"]
            if artifact_hash([str(value) for value in pd.to_datetime(dates_for_hash, utc=True)]) != manifest["axis_hashes"].get(position):
                raise ValueError(f"axis fingerprint mismatch: {position}")
        audit_neon_table(table.reset_index() if "date" not in table else table)
        if "input_dependency_max_time" in table:
            dependency = pd.to_datetime(table["input_dependency_max_time"], utc=True)
            dates = pd.to_datetime(table.index if "date" not in table else table["date"], utc=True)
            if (dependency > dates).any():
                raise ValueError(f"future dependency in {position}")
        coverage = {split: {"slots": int((table["split"] == split).sum())} for split in ("train", "val", "test")}
        if "dissolvedOxygen_observed" in table:
            coverage["train"]["qa_valid_do_slots"] = int(((table["split"] == "train") & table["dissolvedOxygen_observed"].astype(bool)).sum())
            coverage["val"]["qa_valid_do_slots"] = int(((table["split"] == "val") & table["dissolvedOxygen_observed"].astype(bool)).sum())
            coverage["test"]["qa_valid_do_slots"] = int(((table["split"] == "test") & table["dissolvedOxygen_observed"].astype(bool)).sum())
        summaries[position] = {"rows": int(len(table)), "split_counts": table["split"].value_counts().to_dict(), "coverage": coverage}
    if config is not None:
        from nerrs_faith.neon_data import NeonWindowDataset
        scalers = json.loads((data_dir / "scaler.json").read_text(encoding="utf-8")) if (data_dir / "scaler.json").is_file() else {}
        seq_len, pred_len = int(config["window"]["seq_len"]), int(config["window"]["pred_len"])
        tables = {position: pd.read_parquet(data_dir / f"{position}.parquet") for position in manifest.get("positions", [])}
        for position, table in tables.items():
            for split in ("train", "val"):
                dataset = NeonWindowDataset({position: table}, split=split, seq_len=seq_len, pred_len=pred_len, frequency_minutes=int(config["aggregation"]["frequency_minutes"]), scaling_stats=scalers)
                summaries[position]["coverage"][split]["complete_history_windows"] = int(len(dataset))
                summaries[position]["coverage"][split]["valid_forecast_windows"] = int(len(dataset))
                if split == "val":
                    summaries[position]["coverage"][split]["h96_valid_targets"] = int(sum(bool(table.iloc[origin + pred_len].get("dissolvedOxygen_observed", True)) for _, origin in dataset.rows))
        key_path = data_dir / "validation_keys.parquet"
        key_manifest_path = data_dir / "validation_key_manifest.json"
        if not key_path.is_file() or not key_manifest_path.is_file():
            raise FileNotFoundError("frozen NEON validation key manifest is missing")
        key_manifest = json.loads(key_manifest_path.read_text(encoding="utf-8"))
        keys_hash, key_rows = parquet_key_hash(key_path)
        if keys_hash != key_manifest.get("key_hash") or key_rows != int(key_manifest.get("rows", -1)):
            raise ValueError("frozen NEON validation key fingerprint mismatch")
        for field in ("source_hash", "scaler_hash", "axis_hashes", "table_hashes"):
            if key_manifest.get(field) != manifest.get(field):
                raise ValueError(f"validation key manifest {field} mismatch")
    return {"status": "passed", "positions": summaries, "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
