"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
from typing import Any
import torch

def hslf_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors while retaining timestamps for leakage assertions/logging."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    result: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result
