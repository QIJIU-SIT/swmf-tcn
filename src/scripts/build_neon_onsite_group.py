from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from nerrs_faith.neon_io import discover_neon_sources, discover_sensor_positions, resolve_observation_positions, select_active_positions
from nerrs_faith.neon_onsite_audit import audit_onsite_sources, audit_onsite_tables
from nerrs_faith.neon_onsite_data import write_onsite_artifacts, inventory_through_month
from nerrs_faith.neon_pipeline import build_neon_position_tables


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neon_onsite_group_v1.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config_path = Path(args.config); config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    source_audit = audit_onsite_sources(config)
    output = Path(args.output or config["prepared_root"])
    if source_audit["status"] != "passed":
        destination = output.parent / "source_audit.json"; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(source_audit, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(source_audit, indent=2, sort_keys=True, default=str)); return 1
    inventory = inventory_through_month(discover_neon_sources(config), "2023-12")
    for site, site_cfg in config["sites"].items():
        records = discover_sensor_positions(config["raw"]["water_roots"][site], site=site, product=config["raw"]["products"]["water"])
        sensor_positions = [item["hor_ver"] for item in select_active_positions(
            records, site=site, start=config["splits"]["train_start"], end=config["splits"]["test_end"]
        )]
        site_cfg["positions"] = resolve_observation_positions(sensor_positions, inventory.files["water"][site])
        distances = [item["distance_km"] for item in source_audit["sources"].get(site, {}).get("distances", []) if item["meteorology"] == "precipitation"]
        if not distances:
            print(json.dumps({"status": "failed", "reasons": [f"{site} has no precipitation distance provenance"]}, indent=2)); return 1
        site_cfg["precipitation_distance_km"] = min(distances)
    tables = build_neon_position_tables(config, inventory)
    # The prepared package is deliberately sealed at 2023-12.  Test
    # admissibility is checked only by the controlled 2024 test entry point.
    table_audit = audit_onsite_tables(tables, config, required_splits=("train", "val"))
    if table_audit["status"] != "passed":
        print(json.dumps(table_audit, indent=2, sort_keys=True, default=str)); return 1
    manifest = write_onsite_artifacts(tables, config, output, source_audit=source_audit)
    (output.parent / "source_audit.json").write_text(json.dumps(source_audit, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps({**table_audit, "source_audit_hash": source_audit["audit_hash"], "status": "passed"}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
