"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from torch.utils.data import BatchSampler
from nerrs_faith.neon_data import NeonWindowDataset

def resolve_onsite_loader_config(config: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    """Resolve the frozen worker policy without enabling multiprocessing on CPU."""
    if str(device) != "cuda":
        return {"num_workers": 0}
    loader = dict(config.get("loader", {}))
    workers = int(loader.get("num_workers", 4))
    resolved = {"num_workers": workers}
    if workers:
        resolved.update({"persistent_workers": bool(loader.get("persistent_workers", True)), "prefetch_factor": int(loader.get("prefetch_factor", 4))})
    return resolved

class StationBalancedBatchSampler(BatchSampler):
    """Interleave station queues so each early batch sees every site."""

    def __init__(self, dataset: NeonWindowDataset, batch_size: int, *, seed: int = 2026, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        groups: dict[str, list[int]] = {}
        for index, (position, _origin) in enumerate(dataset.rows):
            groups.setdefault(str(position).split("-", 1)[0], []).append(index)
        if not groups:
            raise ValueError("station-balanced sampler requires non-empty dataset")
        generator = np.random.default_rng(int(seed))
        self.groups = {site: list(generator.permutation(indices)) for site, indices in sorted(groups.items())}
        self.sites = tuple(self.groups)
        self.samples_per_site = max(len(values) for values in self.groups.values())

    def __iter__(self):
        cursors = {site: 0 for site in self.sites}
        emitted = {site: 0 for site in self.sites}
        remaining = self.samples_per_site * len(self.sites)
        site_cursor = 0
        while remaining:
            batch: list[int] = []
            while len(batch) < self.batch_size and remaining:
                site = self.sites[site_cursor % len(self.sites)]
                site_cursor += 1
                if emitted[site] >= self.samples_per_site:
                    continue
                values = self.groups[site]
                batch.append(int(values[cursors[site] % len(values)]))
                cursors[site] += 1
                emitted[site] += 1
                remaining -= 1
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self) -> int:
        total = self.samples_per_site * len(self.sites)
        return total // self.batch_size if self.drop_last else (total + self.batch_size - 1) // self.batch_size
