"""Canonical adapter for the supplied official PatchTST supervised source."""

from __future__ import annotations

import copy
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ..adapter import validate_prediction_output
from ..schema import ModelBatch
from ...protocol_v2 import DO_FEATURE_NAME, NativeOutput, adapt_native_do, resolve_do_channel
from .source_loader import load_external_module, source_tree_sha256


class PatchTSTAdapter(nn.Module):
    def __init__(self, *, source_root: str | Path, input_size: int, pred_len: int = 96, seq_len: int = 192, model_config: Mapping[str, Any] | None = None, do_channel: int | None = None, feature_names: tuple[str, ...] | None = None, protocol: str = "legacy") -> None:
        super().__init__()
        root = Path(source_root).resolve()
        if (root / "PatchTST_supervised").is_dir():
            root = root / "PatchTST_supervised"
        required = ("models/PatchTST.py", "layers/PatchTST_backbone.py", "layers/PatchTST_layers.py", "layers/RevIN.py")
        module = load_external_module(root, family="patchtst", module="models.PatchTST", required_files=required)
        self.source_root = root
        self.source_hash = source_tree_sha256(root)
        self.license = "Apache-2.0"
        values = dict(model_config or {})
        self.pred_len = int(pred_len)
        self.input_size = int(input_size)
        if protocol == "protocol_v2" and feature_names is None:
            raise ValueError("Protocol v2 requires explicit feature_names")
        self.feature_names = tuple(feature_names) if feature_names is not None else tuple(
            [f"feature:{index}" for index in range(self.input_size - 1)] + [DO_FEATURE_NAME]
        )
        self.do_channel = resolve_do_channel(self.feature_names) if do_channel is None else int(do_channel)
        if not 0 <= self.do_channel < self.input_size:
            raise ValueError("do_channel must be within the canonical feature range")
        if self.feature_names[self.do_channel] != DO_FEATURE_NAME:
            raise ValueError("do_channel must identify water:DO_mgl")
        defaults = {
            "enc_in": 1, "seq_len": 192, "pred_len": self.pred_len,
            "e_layers": 1, "n_heads": 4, "d_model": 16, "d_ff": 32,
            "dropout": 0.0, "fc_dropout": 0.0, "head_dropout": 0.0,
            "individual": False, "patch_len": 16, "stride": 8, "padding_patch": None,
            "revin": 1, "affine": True, "subtract_last": False,
            "decomposition": False, "kernel_size": 25,
        }
        defaults.update(values)
        defaults["enc_in"] = 1
        defaults["seq_len"] = int(seq_len)
        defaults["pred_len"] = self.pred_len
        self.model_config = copy.deepcopy(defaults)
        self.core = module.Model(Namespace(**defaults))

    def forward(self, batch: ModelBatch) -> Tensor:
        history = torch.where(batch.history_mask, torch.nan_to_num(batch.history), torch.zeros_like(batch.history))
        history = history[:, :, self.do_channel:self.do_channel + 1]
        output = self.core(history)
        if output.ndim != 3 or output.shape[1] != self.pred_len:
            raise ValueError("official PatchTST output must have shape [batch, 96, channels]")
        native = output[:, :, 0]
        return validate_prediction_output(adapt_native_do(native, batch.last_do, NativeOutput.STANDARDIZED_ABSOLUTE_DO), batch)

    @staticmethod
    def masked_l1(prediction: Tensor, batch: ModelBatch) -> Tensor:
        validate_prediction_output(prediction, batch)
        if not batch.target_mask.any():
            raise ValueError("canonical batch contains no valid target values")
        return (prediction[batch.target_mask] - batch.target[batch.target_mask]).abs().mean()

    def loss(self, prediction: Tensor, batch: ModelBatch) -> Tensor:
        return self.masked_l1(prediction, batch)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"source_root": str(self.source_root), "source_hash": self.source_hash, "model_config": self.model_config, "feature_names": self.feature_names, "do_feature_name": DO_FEATURE_NAME, "do_channel": self.do_channel, "effective_inputs": [DO_FEATURE_NAME], "native_output_semantics": NativeOutput.STANDARDIZED_ABSOLUTE_DO.value, "state_dict": self.state_dict()}, temporary)
        os.replace(temporary, path)

    def load(self, path: Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {"source_root": str(self.source_root), "source_hash": self.source_hash, "model_config": self.model_config, "feature_names": self.feature_names, "do_feature_name": DO_FEATURE_NAME, "do_channel": self.do_channel, "effective_inputs": [DO_FEATURE_NAME], "native_output_semantics": NativeOutput.STANDARDIZED_ABSOLUTE_DO.value}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("PatchTST checkpoint provenance/configuration does not match adapter")
        self.load_state_dict(payload["state_dict"])

    def build(self) -> "PatchTSTAdapter":
        return self

    def describe(self) -> dict[str, Any]:
        return {"name": "patchtst", "parameter_count": sum(p.numel() for p in self.parameters()), "source_root": str(self.source_root), "source_hash": self.source_hash, "license": self.license, "feature_names": self.feature_names, "do_feature_name": DO_FEATURE_NAME, "do_channel": self.do_channel, "effective_inputs": [DO_FEATURE_NAME], "native_output_semantics": NativeOutput.STANDARDIZED_ABSOLUTE_DO.value, "config": copy.deepcopy(self.model_config)}


def build_patchtst_adapter(*, source_root: str | Path | None, input_size: int = 1, pred_len: int = 96, seq_len: int = 192, model_config: Mapping[str, Any] | None = None, do_channel: int | None = None, feature_names: tuple[str, ...] | None = None, protocol: str = "legacy") -> PatchTSTAdapter:
    if source_root is None:
        raise FileNotFoundError("official PatchTST source root is required")
    return PatchTSTAdapter(source_root=source_root, input_size=input_size, pred_len=pred_len, seq_len=seq_len, model_config=model_config, do_channel=do_channel, feature_names=feature_names, protocol=protocol)


class PatchTSTSourceUnavailable(RuntimeError):
    """Backward-compatible name for unavailable source errors."""
