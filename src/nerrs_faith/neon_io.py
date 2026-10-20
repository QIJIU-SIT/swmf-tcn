"""Read-only discovery of the fixed NEON protocol-v0 source inventory."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


_DIR_RE = re.compile(r"NEON\.(?P<domain>D\d+)\.(?P<site>[A-Z0-9]+)\.(?P<product>DP1\.\d+\.\d+)\.(?P<month>\d{4}-\d{2})", re.I)
_POSITION_RE = re.compile(r"\.sensor_positions(?:\.|$)", re.I)


@dataclass(frozen=True)
class NeonSourceInventory:
    products: dict[str, str]
    precipitation_mapping: dict[str, str]
    files: dict[str, dict[str, tuple[str, ...]]]
    releases: dict[str, tuple[str, ...]]
    source_hash: str


@dataclass(frozen=True)
class SensorPositionRecord:
    site: str
    hor_ver: str
    latitude: float
    longitude: float
    start: str | None
    end: str | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "hor_ver": self.hor_ver,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }


def _config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        import yaml
        return yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    return dict(config)


def _manifest(root: Path) -> dict[str, Any]:
    manifests = sorted(root.glob("*.manifest*.json"))
    if not manifests:
        return {}
    return json.loads(manifests[-1].read_text(encoding="utf-8"))


def _first_column(frame, names: tuple[str, ...], *, required: bool = True):
    for name in names:
        if name in frame.columns:
            return frame[name]
    if required:
        raise ValueError("sensor-position table is missing " + "/".join(names))
    return None


def discover_sensor_positions(root: str | Path, *, site: str, product: str) -> list[dict[str, Any]]:
    """Read sensor-position provenance without falling back to observations."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"sensor-position root is missing or unsafe: {root}")
    records: dict[tuple[Any, ...], SensorPositionRecord] = {}
    for path in sorted(root.rglob("*.csv")):
        if path.is_symlink() or not _POSITION_RE.search(path.name):
            continue
        if str(product) not in path.name:
            continue
        if f".{site}." not in path.name:
            continue
        frame = pd.read_csv(path, dtype=str)
        hor_col = _first_column(frame, ("HOR.VER", "hor_ver", "horizontalVertical"))
        site_col = _first_column(frame, ("siteID", "site", "site_id"), required=False)
        lat_col = _first_column(frame, ("locationReferenceLatitude", "referenceLatitude", "latitude", "lat"))
        lon_col = _first_column(frame, ("locationReferenceLongitude", "referenceLongitude", "longitude", "lon", "lng"))
        start_col = _first_column(frame, ("positionStartDateTime", "startDate", "startDateTime", "start"), required=False)
        end_col = _first_column(frame, ("positionEndDateTime", "endDate", "endDateTime", "end"), required=False)
        for row_index, row in frame.iterrows():
            row_site = str(site_col.iloc[row_index]).strip() if site_col is not None else str(site)
            if row_site != str(site):
                continue
            try:
                latitude = float(lat_col.iloc[row_index]); longitude = float(lon_col.iloc[row_index])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-finite sensor coordinates in {path}") from exc
            if not np.isfinite(latitude) or not np.isfinite(longitude):
                raise ValueError(f"non-finite sensor coordinates in {path}")
            def normalized(column):
                if column is None or column.iloc[row_index] is None:
                    return None
                value = str(column.iloc[row_index]).strip()
                if not value or value.lower() in {"nan", "nat", "none"}:
                    return None
                stamp = pd.Timestamp(value)
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize("UTC")
                else:
                    stamp = stamp.tz_convert("UTC")
                return stamp.isoformat()
            record = SensorPositionRecord(row_site, str(hor_col.iloc[row_index]).strip(), latitude, longitude, normalized(start_col), normalized(end_col), str(path.resolve()))
            records[(record.site, record.hor_ver, record.latitude, record.longitude, record.start, record.end)] = record
    if not records:
        raise ValueError(f"no sensor-position records found for {site} and {product}")
    return [record.as_dict() for record in sorted(records.values(), key=lambda item: (item.hor_ver, item.start or "", item.source))]


def select_active_positions(records: list[Mapping[str, Any]], *, site: str, count: int = 2, start: Any | None = None, end: Any | None = None) -> list[dict[str, Any]]:
    """Select exactly ``count`` unique position IDs, failing closed on ambiguity."""
    def utc(value):
        if value is None:
            return None
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    period_start = utc(start)
    period_end = utc(end)
    candidates = []
    for raw in records:
        record = dict(raw)
        if str(record.get("site")) != str(site):
            continue
        record_start = pd.Timestamp(record["start"]) if record.get("start") else None
        record_end = pd.Timestamp(record["end"]) if record.get("end") else None
        if period_start is not None and record_end is not None and record_end < period_start:
            continue
        if period_end is not None and record_start is not None and record_start > period_end:
            continue
        candidates.append(record)
    by_position: dict[str, dict[str, Any]] = {}
    for record in candidates:
        key = str(record["hor_ver"])
        previous = by_position.get(key)
        if previous is None or str(record.get("start") or "") > str(previous.get("start") or ""):
            by_position[key] = record
    if len(by_position) != int(count):
        raise ValueError(f"{site} must resolve exactly two active positions; found {len(by_position)}")
    return [by_position[key] for key in sorted(by_position)]


def resolve_observation_positions(sensor_positions: list[str], observation_paths: list[str] | tuple[str, ...]) -> list[str]:
    """Map HOR.VER sensor positions to the unique full position IDs used by water files."""
    available = set()
    pattern = re.compile(r"\.(\d+\.\d+\.\d+)\.waq_", re.I)
    for path in observation_paths:
        match = pattern.search(Path(path).name)
        if match:
            available.add(match.group(1))
    resolved = []
    for sensor_position in sensor_positions:
        matches = sorted(value for value in available if value == sensor_position or value.startswith(f"{sensor_position}."))
        if len(matches) != 1:
            raise ValueError(f"sensor position {sensor_position} resolves to {len(matches)} water observation IDs")
        resolved.append(matches[0])
    if len(set(resolved)) != len(resolved):
        raise ValueError("water observation position mapping is not one-to-one")
    return resolved


def discover_neon_sources(config: Mapping[str, Any] | str | Path) -> NeonSourceInventory:
    cfg = _config(config)
    if cfg.get("protocol") not in {"neon_protocol_v0", "neon_onsite_group_v1"}:
        raise ValueError("NEON protocol identifier is required")
    raw, products_cfg, sites = cfg.get("raw", {}), cfg.get("raw", {}).get("products", {}), cfg.get("sites", {})
    if not isinstance(sites, Mapping) or not sites:
        raise ValueError("the pilot requires at least one declared aquatic site")
    wanted_sites = {"water": set(sites), "humidity": {v["humidity_site"] for v in sites.values()}, "precipitation": {v["precipitation_site"] for v in sites.values()}}
    files: dict[str, dict[str, tuple[str, ...]]] = {}
    releases: dict[str, tuple[str, ...]] = {}
    digest = hashlib.sha256()
    for kind in ("water", "humidity", "precipitation"):
        expected_product = str(products_cfg.get(kind, ""))
        selected: dict[str, list[str]] = {site: [] for site in wanted_sites[kind]}
        configured_roots = raw.get(f"{kind}_roots")
        root_by_site = dict(configured_roots) if isinstance(configured_roots, Mapping) else {
            site: raw.get(f"{kind}_root", "") for site in selected
        }
        if set(root_by_site) != set(selected):
            raise ValueError(f"{kind} roots must map every required source site exactly once")
        roots: dict[Path, set[str]] = {}
        for site, root_value in root_by_site.items():
            root_ref = Path(root_value)
            if root_ref.is_symlink():
                raise FileNotFoundError(f"NEON source root is a symlink: {root_ref}")
            root = root_ref.resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"NEON source root is missing or unsafe: {root}")
            roots.setdefault(root, set()).add(site)
        all_months = set()
        for root, root_sites in sorted(roots.items(), key=lambda item: str(item[0])):
            manifest = _manifest(root)
            if manifest and str(manifest.get("productCode", expected_product)) != expected_product:
                raise ValueError(f"unexpected {kind} product")
            for path in sorted(root.rglob("*.csv")):
                if path.is_symlink() or path.name.endswith("variables.csv") or ".variables." in path.name or ".sensor_positions." in path.name:
                    continue
                match = _DIR_RE.search(path.parent.name)
                if not match or match.group("product") != expected_product or match.group("site") not in root_sites:
                    continue
                all_months.add(match.group("month"))
                if "1min" not in path.name and kind != "water":
                    continue
                selected[match.group("site")].append(str(path.resolve()))
                digest.update(f"{kind}:{match.group('site')}:".encode())
                digest.update(path.relative_to(root).as_posix().encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        if any(not paths for paths in selected.values()):
            raise ValueError(f"missing one-minute {kind} files for " + ", ".join(sorted(site for site, paths in selected.items() if not paths)))
        files[kind] = {site: tuple(sorted(paths)) for site, paths in selected.items()}
        required = set(cfg.get("splits", {}).get("required_months", []))
        if required and not required.issubset(all_months):
            raise ValueError("missing source month(s): " + ", ".join(sorted(required - all_months)))
        release_names = set()
        for paths in selected.values():
            for raw_path in paths:
                match = re.search(r"RELEASE-[^./]+", str(Path(raw_path).parent))
                if match:
                    release_names.add(match.group(0))
        releases[kind] = tuple(sorted(release_names))
    mapping = {site: str(value["precipitation_site"]) for site, value in sites.items()}
    return NeonSourceInventory({k: str(v) for k, v in products_cfg.items()}, mapping, files, releases, digest.hexdigest())
