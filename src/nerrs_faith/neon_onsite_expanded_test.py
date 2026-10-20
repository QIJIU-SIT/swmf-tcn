"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from nerrs_faith.neon_full_smoke import resolve_neon_data_root
from nerrs_faith.neon_data import NeonWindowDataset, _CONTROLLED_TEST_CAPABILITY
from nerrs_faith.neon_io import discover_neon_sources, discover_sensor_positions, resolve_observation_positions, select_active_positions
from nerrs_faith.neon_onsite_data import inventory_through_month
from nerrs_faith.neon_pipeline import build_neon_position_tables

def _controlled_test_dataset(tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], scalers: Mapping[str, Any]) -> NeonWindowDataset:
    return NeonWindowDataset(
        tables,
        split="test",
        seq_len=int(config["window"]["seq_len"]),
        pred_len=int(config["window"]["pred_len"]),
        frequency_minutes=int(config["aggregation"]["frequency_minutes"]),
        scaling_stats=scalers,
        allow_test=_CONTROLLED_TEST_CAPABILITY,
    )

def _build_controlled_test_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Rebuild the sealed 2024 tables through the authorized raw-data path."""
    inventory = inventory_through_month(discover_neon_sources(config), "2024-12", start_month="2024-01")
    for site, site_config in config["sites"].items():
        records = discover_sensor_positions(
            config["raw"]["water_roots"][site],
            site=site,
            product=config["raw"]["products"]["water"],
        )
        active = select_active_positions(
            records,
            site=site,
            start=config["splits"]["test_start"],
            end=config["splits"]["test_end"],
        )
        site_config["positions"] = resolve_observation_positions(
            [item["hor_ver"] for item in active], inventory.files["water"][site]
        )
        site_config["precipitation_distance_km"] = 0.0
    tables = build_neon_position_tables(config, inventory)
    return {
        position: frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        for position, frame in tables.items()
    }

def _prepare_test_dataset(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    data_root = resolve_neon_data_root(config, repo_root=repo_root)
    data_dir = data_root / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    scalers = json.loads((data_dir / "scaler.json").read_text(encoding="utf-8"))
    tables = _build_controlled_test_tables(config)
    if set(tables) != set(manifest["positions"]):
        raise ValueError("controlled 2024 positions do not match the frozen prepared manifest")
    dataset = _controlled_test_dataset(tables, config, scalers)
    if not len(dataset):
        raise ValueError("controlled 2024 rebuild produced no test windows")
    return {"dataset": dataset, "scalers": scalers}
