"""One trainer for all HSLF variants, with resumable atomic artifacts."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from nerrs_faith.hslf_data import hslf_collate
from nerrs_faith.hslf_loss import masked_horizon_balanced_l1
from nerrs_faith.streaming_metrics import StationHorizonMAE


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def make_hslf_loader(dataset, *, batch_size: int | None = None, shuffle: bool = False, device: str = "auto", **kwargs):
    if "collate_fn" in kwargs:
        raise ValueError("make_hslf_loader uses the frozen hslf_collate function")
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    workers = int(kwargs.pop("num_workers", 0))
    batch_sampler = kwargs.pop("batch_sampler", None)
    loader_kwargs: dict[str, Any] = {
        "collate_fn": hslf_collate,
        "pin_memory": bool(use_cuda),
        "num_workers": workers,
    }
    if batch_sampler is None:
        loader_kwargs.update(batch_size=128 if batch_size is None else batch_size, shuffle=shuffle)
    else:
        if batch_size is not None:
            raise ValueError("batch_size cannot be combined with batch_sampler")
        if shuffle:
            raise ValueError("shuffle cannot be combined with batch_sampler")
        loader_kwargs["batch_sampler"] = batch_sampler
    if workers:
        loader_kwargs["persistent_workers"] = bool(kwargs.pop("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(kwargs.pop("prefetch_factor", 2))
    else:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
    loader_kwargs.update(kwargs)
    return DataLoader(dataset, **loader_kwargs)


def next_batch_size_after_oom(batch_size: int, *, already_retried: bool) -> int:
    if already_retried or batch_size <= 64:
        raise RuntimeError("OOM after the single permitted batch-size reduction")
    return batch_size // 2


def collect_predictions(
    model: nn.Module,
    loader: Iterable,
    *,
    device: str | torch.device = "cpu",
    metadata: Mapping[str, Any] | None = None,
    split: str = "val",
    diagnostic_aggregator: Any | None = None,
    batch_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Collect a prediction table without coupling evaluation to model internals."""
    target_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model = model.to(target_device)
    model.eval()
    metadata = metadata or {}
    id_to_station = {int(value): key for key, value in metadata.get("station_ids", {}).items()}
    id_to_reserve = {int(value): key for key, value in metadata.get("reserve_ids", {}).items()}
    scaling = metadata.get("scaling_stats", {})
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            transformed = batch_transform(batch) if batch_transform is not None else batch
            moved = _move_batch(transformed, target_device)
            kwargs = dict(model_kwargs or {})
            gate_override = kwargs.get("hierarchy_gate_override")
            if torch.is_tensor(gate_override) and gate_override.ndim == 1:
                kwargs["hierarchy_gate_override"] = gate_override.unsqueeze(0).expand(moved["station_id"].shape[0], -1)
            prediction, diagnostics = model(moved, **kwargs)
            if diagnostic_aggregator is not None:
                diagnostic_aggregator.update(diagnostics, moved)
            target = moved["target"].detach().cpu().numpy()
            prediction = prediction.detach().cpu().numpy()
            mask = moved["target_mask"].detach().cpu().numpy().astype(bool)
            station_ids = moved["station_id"].detach().cpu().numpy()
            reserve_ids = moved.get("reserve_id", torch.zeros_like(moved["station_id"])).detach().cpu().numpy()
            last_do = moved["last_do"].detach().cpu().numpy() if "last_do" in moved else np.zeros(len(station_ids))
            last_do_physical = moved.get("last_do_physical")
            if torch.is_tensor(last_do_physical):
                last_do_physical = last_do_physical.detach().cpu().numpy()
            origins = moved.get("origin", [None] * len(station_ids))
            target_times = moved.get("target_times", [None] * len(station_ids))
            seasons = moved.get("season", [None] * len(station_ids))
            day_nights = moved.get("day_night", [None] * len(station_ids))
            missing_rates = moved.get("input_missing_rate", [None] * len(station_ids))
            rain_labels = moved.get("rain_label", [False] * len(station_ids))
            rain_label_available = moved.get("rain_label_available", [True] * len(station_ids))
            high_temp_labels = moved.get("high_temp_label", [False] * len(station_ids))
            seasonal_values = moved.get("seasonal_persistence")
            seasonal_masks = moved.get("seasonal_persistence_mask")
            if torch.is_tensor(seasonal_values):
                seasonal_values = seasonal_values.detach().cpu().numpy()
            if torch.is_tensor(seasonal_masks):
                seasonal_masks = seasonal_masks.detach().cpu().numpy().astype(bool)
            for row_index, station_id in enumerate(station_ids):
                station = id_to_station.get(int(station_id), str(int(station_id)))
                reserve = id_to_reserve.get(int(reserve_ids[row_index]), str(int(reserve_ids[row_index])))
                do_stats = scaling.get(station, {}).get("water", {})
                mean = float(do_stats.get("mean", {}).get("DO_mgl", 0.0))
                scale = float(do_stats.get("scale", {}).get("DO_mgl", 1.0))
                for horizon in range(target.shape[1]):
                    target_time = target_times[row_index][horizon] if target_times[row_index] is not None else None
                    persistence = float(last_do_physical[row_index]) if last_do_physical is not None else float((last_do[row_index] * scale) + mean)
                    seasonal_valid = bool(seasonal_masks[row_index, horizon]) if seasonal_masks is not None else False
                    rows.append(
                        {
                            "split": split,
                            "station": station,
                            "reserve": reserve,
                            "origin": origins[row_index],
                            "target_time": target_time,
                            "horizon": horizon + 1,
                            "target": float((target[row_index, horizon] + last_do[row_index]) * scale + mean),
                            "prediction": float((prediction[row_index, horizon] + last_do[row_index]) * scale + mean),
                            "target_mask": bool(mask[row_index, horizon]),
                            "seed": metadata.get("seed"),
                            "variant": metadata.get("variant"),
                            "season": seasons[row_index],
                            "day_night": day_nights[row_index],
                            "input_missing_rate": missing_rates[row_index],
                            "rain_label": bool(rain_labels[row_index]),
                            "rain_label_available": bool(rain_label_available[row_index]),
                            "high_temp_label": bool(high_temp_labels[row_index]),
                            "persistence": persistence,
                            "seasonal_persistence": float(seasonal_values[row_index, horizon]) if seasonal_values is not None else np.nan,
                            "seasonal_persistence_mask": seasonal_valid,
                        }
                    )
    return pd.DataFrame(rows)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary JSON already exists: {temporary}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_hslf_checkpoint(path: str | Path, model: nn.Module, *, data_hash: str, config: Mapping[str, Any], optimizer=None) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("data_hash") != data_hash:
        raise ValueError("data hash mismatch while restoring checkpoint")
    if payload.get("config_hash") != config_hash(config):
        raise ValueError("configuration hash mismatch while restoring checkpoint")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if payload.get("rng_state") is not None:
        _restore_rng(payload["rng_state"])
    return payload


class HSLFTrainer:
    def __init__(self, model: nn.Module, train_loader: Iterable, val_loader: Iterable, *, run_dir: str | Path, data_hash: str, config: Mapping[str, Any], device: str = "auto", resume_checkpoint: str | Path | None = None, oom_retry_factory: Callable[[int], tuple[Iterable, Iterable]] | None = None, physical_metric_fn: Callable[[torch.Tensor, Mapping[str, Any]], float] | None = None, physical_scaling_stats: Mapping[int, tuple[float, float]] | None = None, primary_metric_name: str = "physical_station_macro_mae", selection_scope: Literal["h96", "trajectory"] = "h96") -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.data_hash = data_hash
        self.oom_retry_factory = oom_retry_factory
        self.physical_metric_fn = physical_metric_fn
        self.physical_scaling_stats = dict(physical_scaling_stats or {})
        self.primary_metric_name = str(primary_metric_name)
        if selection_scope not in {"h96", "trajectory"}:
            raise ValueError("selection_scope must be 'h96' or 'trajectory'")
        self.selection_scope = selection_scope
        self.oom_retried = False
        self.config = {
            "max_epochs": 25,
            "patience": 5,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "amp": True,
            **dict(config),
        }
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        amp_enabled = bool(self.config.get("amp", True)) and self.device.type == "cuda"
        self.amp_enabled = amp_enabled
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.start_epoch = 0
        self.resume_history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_physical_endpoint_mae": []}
        self.resume_best_metric = float("inf")
        self.resume_stale_epochs = 0
        if resume_checkpoint is not None:
            payload = load_hslf_checkpoint(
                resume_checkpoint,
                self.model,
                data_hash=self.data_hash,
                config=self.config,
                optimizer=self.optimizer,
            )
            checkpoint_scope = str(payload.get("selection_scope", "h96"))
            if checkpoint_scope != self.selection_scope:
                raise ValueError("checkpoint selection scope does not match trainer")
            self.scheduler.load_state_dict(payload["scheduler_state"])
            self.scaler.load_state_dict(payload["scaler_state"])
            completed_epoch = str(payload.get("status")) == "running"
            self.start_epoch = int(payload["epoch"]) + (1 if completed_epoch else 0)
            self.resume_history = payload.get("history", self.resume_history)
            self.resume_best_metric = float(payload.get("best_metric", float("inf")))
            self.resume_stale_epochs = int(payload.get("stale_epochs", 0))

    def _checkpoint(self, name: str, *, epoch: int, status: str, history: dict[str, list[float]], best_metric: float, stale_epochs: int) -> None:
        payload = {
            "epoch": epoch,
            "status": status,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "data_hash": self.data_hash,
            "config": self.config,
            "config_hash": config_hash(self.config),
            "history": history,
            "best_metric": best_metric,
            "stale_epochs": int(stale_epochs),
            "early_stopping_state": {"stale_epochs": int(stale_epochs), "patience": int(self.config["patience"])},
            "primary_metric_name": self.primary_metric_name,
            "selection_scope": self.selection_scope,
            "rng_state": _rng_state(),
        }
        _atomic_torch_save(payload, self.run_dir / name)
        _atomic_json({"status": status, "epoch": epoch, "best_metric": best_metric, "stale_epochs": int(stale_epochs), "primary_metric_name": self.primary_metric_name}, self.run_dir / "status.json")

    def _batch_loss(self, batch: Mapping[str, Any], *, train: bool) -> torch.Tensor:
        batch = _move_batch(batch, self.device)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            prediction, _ = self.model(batch)
            loss = masked_horizon_balanced_l1(prediction, batch["target"], batch["target_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite HSLF loss")
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["grad_clip"]))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss.detach()

    def _validate(self) -> tuple[float, float, dict[str, float]]:
        started = time.monotonic()
        self.model.eval()
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_batches = 0
        loader_wait_seconds = 0.0
        compute_started = time.monotonic()
        station_errors: dict[int, list[float]] = {}
        physical_station_errors: dict[int, dict[int, list[float]]] = {}
        callback_values: list[float] = []
        metric = None
        if self.physical_scaling_stats:
            station_count = max(self.physical_scaling_stats)
            scales = torch.ones(station_count)
            for station_id, (_, scale) in self.physical_scaling_stats.items():
                scales[int(station_id) - 1] = float(scale)
            metric = StationHorizonMAE(station_count=station_count, pred_len=96, scales=scales, device=self.device)
        with torch.no_grad():
            previous_batch_end = time.monotonic()
            for batch in self.val_loader:
                loader_wait_seconds += time.monotonic() - previous_batch_end
                moved = _move_batch(batch, self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    prediction, _ = self.model(moved)
                    loss = masked_horizon_balanced_l1(prediction, moved["target"], moved["target_mask"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite HSLF validation loss")
                loss_sum += loss.detach().to(torch.float64)
                loss_batches += 1
                if self.physical_metric_fn is not None:
                    callback_values.append(float(self.physical_metric_fn(prediction, moved)))
                if metric is not None:
                    metric.update(prediction, moved["target"], moved["target_mask"], moved["station_id"])
                if "station_id" in moved and metric is None:
                    endpoint_mask = moved["target_mask"][:, -1]
                    endpoint_error = (prediction[:, -1] - moved["target"][:, -1]).abs()
                    for row_index, (station, error, valid) in enumerate(zip(moved["station_id"], endpoint_error, endpoint_mask)):
                        if bool(valid):
                            station_errors.setdefault(int(station), []).append(float(error.detach().cpu()))
                previous_batch_end = time.monotonic()
        if not loss_batches:
            raise ValueError("validation loader is empty")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        val_loss = float((loss_sum / loss_batches).cpu())
        def macro_at(horizon: int) -> float:
            station_means = []
            for errors in physical_station_errors.values():
                values_at_horizon = errors.get(horizon, [])
                if values_at_horizon:
                    station_means.append(float(np.mean(values_at_horizon)))
            return float(np.mean(station_means)) if station_means else float("nan")

        trajectory_values = [
            float(np.mean(values_at_horizon))
            for errors in physical_station_errors.values()
            for values_at_horizon in [
                [error for horizon_values in errors.values() for error in horizon_values]
            ]
            if values_at_horizon
        ]
        trajectory_metric = float(np.mean(trajectory_values)) if trajectory_values else float("nan")
        h96_metric = macro_at(96)
        if metric is not None:
            summary = metric.summarize()
            trajectory_metric = float(summary["trajectory"])
            macro_at = lambda horizon: float(summary["horizons"][horizon])
            h96_metric = macro_at(96)
        metrics = {
            "val_loss": val_loss,
            "trajectory_physical_station_macro_mae": trajectory_metric,
            "h96_physical_station_macro_mae": h96_metric,
            "mae_1h": macro_at(4),
            "mae_6h": macro_at(24),
            "mae_12h": macro_at(48),
            "mae_24h": macro_at(96),
            "validation_seconds": time.monotonic() - started,
            "validation_loader_wait_seconds": loader_wait_seconds,
            "validation_compute_seconds": time.monotonic() - compute_started - loader_wait_seconds,
        }
        if self.physical_metric_fn is not None:
            selection_metric = float(np.mean(callback_values))
        elif self.selection_scope == "trajectory" and np.isfinite(trajectory_metric):
            selection_metric = trajectory_metric
        elif self.physical_scaling_stats and np.isfinite(h96_metric):
            selection_metric = h96_metric
        elif station_errors:
            selection_metric = float(np.mean([np.mean(errors) for errors in station_errors.values()]))
        else:
            selection_metric = val_loss
        return val_loss, selection_metric, metrics

    def fit(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        max_epochs = int(self.config["max_epochs"])
        patience = int(self.config["patience"])
        if max_epochs <= 0 or patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        time_limit = self.config.get("time_limit_seconds")
        started = time.monotonic()
        history: dict[str, list[float]] = self.resume_history
        history.setdefault("val_physical_endpoint_mae", [])
        for key in (
            "val_trajectory_physical_station_macro_mae",
            "val_h96_physical_station_macro_mae",
            "mae_1h", "mae_6h", "mae_12h", "mae_24h",
            "train_seconds", "validation_seconds",
            "validation_loader_wait_seconds", "validation_compute_seconds",
            "train_loader_wait_seconds", "train_compute_seconds",
        ):
            history.setdefault(key, [])
        best_metric = self.resume_best_metric
        stale = self.resume_stale_epochs
        status = "completed"
        for epoch in range(self.start_epoch, max_epochs):
            self.model.train()
            train_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            train_batches = 0
            train_loader_wait_seconds = 0.0
            train_started = time.monotonic()
            try:
                if time_limit is not None and float(time_limit) <= 0:
                    status = "time_limited"
                    self._checkpoint("last.pt", epoch=epoch, status=status, history=history, best_metric=best_metric, stale_epochs=stale)
                    break
                previous_batch_end = time.monotonic()
                for batch in self.train_loader:
                    train_loader_wait_seconds += time.monotonic() - previous_batch_end
                    if time_limit is not None and time.monotonic() - started >= float(time_limit):
                        status = "time_limited"
                        break
                    train_loss_sum += self._batch_loss(batch, train=True).to(torch.float64)
                    train_batches += 1
                    previous_batch_end = time.monotonic()
                if status == "time_limited":
                    self._checkpoint("last.pt", epoch=epoch, status=status, history=history, best_metric=best_metric, stale_epochs=stale)
                    break
                train_seconds = time.monotonic() - train_started
                val_loss, selection_metric, validation_metrics = self._validate()
                train_metric = float((train_loss_sum / train_batches).cpu()) if train_batches else float("inf")
                history["train_loss"].append(train_metric)
                history["val_loss"].append(val_loss)
                history["train_seconds"].append(train_seconds)
                history["train_loader_wait_seconds"].append(train_loader_wait_seconds)
                history["train_compute_seconds"].append(train_seconds - train_loader_wait_seconds)
                for key in (
                    "val_trajectory_physical_station_macro_mae",
                    "val_h96_physical_station_macro_mae",
                    "mae_1h", "mae_6h", "mae_12h", "mae_24h",
                    "validation_seconds",
                    "validation_loader_wait_seconds", "validation_compute_seconds",
                ):
                    source_key = key.removeprefix("val_")
                    history[key].append(validation_metrics[source_key])
                if self.physical_metric_fn is not None or self.physical_scaling_stats:
                    history["val_physical_endpoint_mae"].append(selection_metric)
                self.scheduler.step(selection_metric)
                if selection_metric < best_metric:
                    best_metric = selection_metric
                    stale = 0
                    self._checkpoint("best.pt", epoch=epoch, status="running", history=history, best_metric=best_metric, stale_epochs=stale)
                else:
                    stale += 1
                self._checkpoint("last.pt", epoch=epoch, status="running", history=history, best_metric=best_metric, stale_epochs=stale)
                if stale >= patience:
                    break
            except KeyboardInterrupt:
                status = "interrupted"
                self._checkpoint("last.pt", epoch=epoch, status=status, history=history, best_metric=best_metric, stale_epochs=stale)
                break
            except RuntimeError as error:
                is_oom = "out of memory" in str(error).lower() or "cuda error" in str(error).lower() and "memory" in str(error).lower()
                if is_oom and self.oom_retry_factory is not None and not self.oom_retried:
                    current_batch_size = int(self.config.get("batch_size", 128))
                    reduced_batch_size = next_batch_size_after_oom(current_batch_size, already_retried=False)
                    self.train_loader, self.val_loader = self.oom_retry_factory(reduced_batch_size)
                    self.config["batch_size"] = reduced_batch_size
                    self.oom_retried = True
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                status = "failed"
                self._checkpoint("last.pt", epoch=epoch, status=status, history=history, best_metric=best_metric, stale_epochs=stale)
                raise
            except Exception:
                status = "failed"
                self._checkpoint("last.pt", epoch=epoch, status=status, history=history, best_metric=best_metric, stale_epochs=stale)
                raise
        if status == "completed":
            self._checkpoint("last.pt", epoch=len(history["train_loss"]) - 1, status=status, history=history, best_metric=best_metric, stale_epochs=stale)
        elapsed_seconds = time.monotonic() - started
        result = {
            "status": status,
            "history": history,
            "best_metric": best_metric,
            "primary_metric_name": self.primary_metric_name,
            "config": self.config,
            "data_hash": self.data_hash,
            "epochs": len(history["train_loss"]),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "elapsed_seconds": elapsed_seconds,
        }
        _atomic_json(result, self.run_dir / "history.json")
        return result
