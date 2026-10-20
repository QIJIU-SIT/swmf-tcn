"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
import hashlib
from collections.abc import Mapping
from typing import Any
import torch

def perturb_batch(batch: Mapping[str, Any], perturbation: str, *, seed: int, block_steps: int) -> dict[str, Any]:
    """Return a shallow batch copy with a deterministic weather perturbation."""
    variable_channels = {"temp_block_shuffle": 0, "rh_block_shuffle": 1, "precip_block_shuffle": 2}
    if perturbation not in {"met_zero", "met_block_shuffle", *variable_channels}:
        raise ValueError("unknown meteorological perturbation")
    if int(block_steps) <= 0:
        raise ValueError("block_steps must be positive")
    transformed = dict(batch)
    if not torch.is_tensor(batch["met"]) or not torch.is_tensor(batch["met_mask"]):
        raise TypeError("fusion-depth perturbations require tensor met and met_mask")
    if batch["met"].shape != batch["met_mask"].shape:
        raise ValueError("met and met_mask must have identical shapes")
    if perturbation == "met_zero":
        transformed["met"] = torch.zeros_like(batch["met"])
        transformed["met_mask"] = torch.zeros_like(batch["met_mask"])
        return transformed

    values = batch["met"].clone()
    masks = batch["met_mask"].clone()
    selected_channel = variable_channels.get(perturbation)
    if selected_channel is not None:
        channels = [selected_channel]
    else:
        channels = list(range(values.shape[-1]))
    stations = batch.get("station_id")
    origins = batch.get("origin")
    for row in range(values.shape[0]):
        station = int(stations[row]) if torch.is_tensor(stations) else str(stations[row])
        origin = origins[row] if origins is not None else row
        block_count = values.shape[1] // int(block_steps)
        blocks = [(start, start + int(block_steps)) for start in range(0, block_count * int(block_steps), int(block_steps))]
        if perturbation == "met_block_shuffle":
            order = sorted(
                range(len(blocks)),
                key=lambda index: hashlib.sha256(f"{seed}:{station}:{origin}:{index}".encode()).digest(),
            )
        else:
            if len(blocks) <= 1:
                continue
            generator = torch.Generator(device="cpu")
            digest = hashlib.sha256(f"{seed}:{station}:{origin}:{perturbation}".encode()).digest()
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
            order = torch.randperm(len(blocks), generator=generator).tolist()
            if order == list(range(len(blocks))):
                order = order[1:] + order[:1]
        original_values = values[row].clone()
        original_masks = masks[row].clone()
        for channel in channels:
            for destination, source in zip(blocks, (blocks[index] for index in order)):
                dst_start, dst_end = destination
                src_start, src_end = source
                values[row, dst_start:dst_end, channel] = original_values[src_start:src_end, channel]
                if perturbation == "met_block_shuffle":
                    masks[row, dst_start:dst_end, channel] = original_masks[src_start:src_end, channel]
    transformed["met"] = values
    transformed["met_mask"] = masks
    return transformed
