"""Independent budget/stop/checkpoint loop, using frozen batch loss and validation.

No dataset construction, prediction export or test evaluation. Resume is epoch-boundary only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path
import time

import torch

from nerrs_faith.hslf_training import HSLFTrainer, _rng_state, _restore_rng, _atomic_torch_save, config_hash

METRIC = 'val_trajectory_physical_station_macro_mae'
RULES = dict(max_epochs=15, min_epochs=8, patience=5, min_delta=.0001, post_lr_epochs=2)
SUCCESS = {'EARLY_STOPPED', 'BUDGET_CAPPED_UNRESOLVED'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


@dataclass
class StopController:
    anchor: float | None = None
    stale: int = 0
    last_lr_epoch: int | None = None
    completed_epochs: int = 0

    def observe(self, epoch, value, lr_reduced):
        if epoch != self.completed_epochs + 1 or epoch > 15 or not math.isfinite(value):
            raise ValueError('sequential complete epochs and finite validation MAE required')
        self.completed_epochs = epoch
        if self.anchor is None or value < self.anchor - RULES['min_delta']:
            self.anchor, self.stale = value, 0
        else:
            self.stale += 1
        if lr_reduced:
            self.last_lr_epoch = epoch
        eligible = (epoch >= 8 and self.stale >= 5 and
                    (self.last_lr_epoch is None or epoch - self.last_lr_epoch >= 2))
        if eligible:
            return 'EARLY_STOPPED'
        if epoch == 15:
            # Conservative: no settled stopping condition at the cap is unresolved,
            # including clear continuing improvement or insufficient post-LR time.
            return 'BUDGET_CAPPED_UNRESOLVED'
        return None


def steps(optimizer):
    return sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})


class PersistentBoundaryLoader:
    """Avoid an extra global/sampler RNG draw when replacing persistent workers.

    Only valid for deterministic map datasets (the frozen NeonWindowDataset).
    Worker RNG itself is not restored; no augmentation or worker randomness is used.
    The original RandomSampler keeps its original restored generator.
    """
    def __init__(self, loader):
        self.loader, self.first = loader, True

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.first and getattr(self.loader, 'num_workers', 0) and self.loader.persistent_workers:
            self.first = False
            original = self.loader.generator
            state = original.get_state() if original is not None else torch.get_rng_state()
            self.loader.generator = torch.Generator().set_state(state)
            try:
                iterator = iter(self.loader)
            finally:
                self.loader.generator = original
            return iterator
        self.first = False
        return iter(self.loader)


class ExtensionTrainer(HSLFTrainer):
    def __init__(self, *args, **kwargs):
        if kwargs.get('resume_checkpoint') is not None or kwargs.get('oom_retry_factory') is not None:
            raise ValueError('fresh runs only; automatic resume/OOM retry forbidden')
        config = dict(kwargs.pop('config', {}))
        for key, value in RULES.items():
            if key in config and config[key] != value:
                raise ValueError('fixed extension rule mismatch: ' + key)
        config.update(RULES)
        super().__init__(*args, config=config, **kwargs)
        if self.selection_scope != 'trajectory' or self.physical_metric_fn is not None:
            raise ValueError('full physical trajectory selection required')
        self.stop = StopController()
        self.trace = []
        self._resume_payload = None
        self.best_payload = None

    def restore_boundary(self):
        """Validate and stage only last.pt; do not mutate files or run forward."""
        path = self.run_dir / 'last.pt'
        if not path.is_file():
            raise ValueError('no complete epoch boundary last.pt; interrupted.pt cannot be resumed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if (payload.get('in_progress_epoch') is not None or
                payload.get('completed_epochs') != len(payload['history'][METRIC]) or
                payload.get('epoch') != payload['completed_epochs'] - 1):
            raise ValueError('invalid completed epoch boundary')
        if payload['completed_epochs']:
            audit_run(self.run_dir, expected_config=self.config, expected_data_hash=self.data_hash,
                      incomplete=True, expected_batches=len(self.train_loader))
            best = payload.get('best_checkpoint')
            if best is None:
                best = torch.load(self.run_dir / 'best.pt', map_location='cpu', weights_only=False)
            self.best_payload = copy.deepcopy(best)
        else:
            if payload['config'] != self.config or payload['data_hash'] != self.data_hash:
                raise ValueError('epoch-zero boundary identity mismatch')
        if payload['config_hash'] != config_hash(self.config):
            raise ValueError('resume configuration mismatch')
        if getattr(self.train_loader, 'generator', None) is not None and payload.get('loader_generator_state') is None:
            raise ValueError('shuffle generator state missing at boundary')
        self.model.load_state_dict(payload['model_state'])
        self.optimizer.load_state_dict(payload['optimizer_state'])
        self.scheduler.load_state_dict(payload['scheduler_state'])
        self.scaler.load_state_dict(payload['scaler_state'])
        self.stop = StopController(**payload['stop_state'])
        self.trace = copy.deepcopy(payload['optimization_trace'])
        self._resume_payload = payload
        return dict(completed_epochs=payload['completed_epochs'], next_epoch=payload['completed_epochs'] + 1,
                    source_sha256=sha(path), policy='replay_incomplete_epoch_from_last_complete_boundary')

    def save_state(self, name, history, best, status, *, in_progress=None):
        generator = getattr(self.train_loader, 'generator', None)
        payload = dict(epoch=len(history[METRIC]) - 1,
                       completed_epochs=len(history[METRIC]), in_progress_epoch=in_progress,
                       status=status, model_state=self.model.state_dict(),
                       optimizer_state=self.optimizer.state_dict(), scheduler_state=self.scheduler.state_dict(),
                       scaler_state=self.scaler.state_dict(), rng_state=_rng_state(),
                       loader_generator_state=generator.get_state() if generator is not None else None,
                       exact_resume_supported=False, resume_scope='complete_epoch_deterministic_dataset',
                       config=self.config, config_hash=config_hash(self.config),
                       data_hash=self.data_hash, selection_scope=self.selection_scope,
                       primary_metric_name=self.primary_metric_name, history=history,
                       best_metric=best, stop_state=asdict(self.stop), optimization_trace=self.trace)
        if name == 'best.pt':
            self.best_payload = copy.deepcopy(payload)
        elif name == 'last.pt':
            # An atomic last commit contains its selected best, preventing a best/last split commit.
            payload['best_checkpoint'] = self.best_payload
        _atomic_torch_save(payload, self.run_dir / name)

    def fit(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self._resume_payload is None and any((self.run_dir / name).exists() for name in ['last.pt', 'best.pt', 'interrupted.pt']):
            raise RuntimeError('existing training state: no overwrite or automatic rerun')
        history = {'train_loss': [], 'val_loss': [], METRIC: [], 'train_seconds': [],
                   'validation_seconds': []}
        best = float('inf')
        completed = 0
        previous_seconds = 0.
        terminal = None
        if self._resume_payload is not None:
            payload = self._resume_payload
            archive = self.run_dir / 'recovery_attempts' / uuid.uuid4().hex
            archive.mkdir(parents=True)
            for name in ['best.pt', 'last.pt', 'interrupted.pt', 'history.json', 'optimization_trace.json', 'status.json', 'run_manifest.json']:
                if (self.run_dir / name).is_file():
                    shutil.copy2(self.run_dir / name, archive / name)
            write(archive / 'recovery.json', dict(source_sha256=sha(self.run_dir / 'last.pt'),
                  completed_epochs=payload['completed_epochs'], replay_from_epoch=payload['completed_epochs'] + 1,
                  partial_checkpoint_used=False, policy='epoch_boundary', test_accessed=False))
            manifest_path = self.run_dir / 'run_manifest.json'
            if manifest_path.is_file():
                manifest = read(manifest_path)
                manifest.update(state='RUNNING', resumed_from_completed_epochs=payload['completed_epochs'])
                write(manifest_path, manifest)
            if self.best_payload is not None:
                _atomic_torch_save(self.best_payload, self.run_dir / 'best.pt')
            history = copy.deepcopy(payload['history'])
            best, completed = payload['best_metric'], payload['completed_epochs']
            previous_seconds = sum(history['train_seconds']) + sum(history['validation_seconds'])
            if payload['status'] in SUCCESS:
                terminal = payload['status']
            generator = getattr(self.train_loader, 'generator', None)
            if generator is not None:
                generator.set_state(payload['loader_generator_state'])
            if completed:
                self.train_loader = PersistentBoundaryLoader(self.train_loader)
                self.val_loader = PersistentBoundaryLoader(self.val_loader)
            _restore_rng(payload['rng_state'])
            write(self.run_dir / 'status.json', dict(status=terminal or 'RUNNING', completed_epochs=completed,
                                                    best_metric=best if math.isfinite(best) else None,
                                                    resumed_from_epoch=completed + 1))
            print(f"RESUME model={self.config.get('model', type(self.model).__name__)} seed={self.config.get('seed', '?')} completed_epochs={completed} next_epoch={completed + 1} source=last.pt", flush=True)
        else:
            # Future interruptions before E1 also have a durable initial boundary.
            self.save_state('last.pt', history, best, 'RUNNING')
        started = time.monotonic()
        epoch = completed
        status = terminal or 'RUNNING'
        try:
            for epoch in ([] if terminal else range(completed + 1, 16)):
                self.model.train()
                lr_used = float(self.optimizer.param_groups[0]['lr'])
                before_steps = steps(self.optimizer)
                train_started = time.monotonic()
                loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
                batches = 0
                for batch in self.train_loader:
                    loss_sum += self._batch_loss(batch, train=True).to(torch.float64)
                    batches += 1
                if not batches:
                    raise ValueError('empty training loader')
                if hasattr(self.train_loader, '__len__') and batches != len(self.train_loader):
                    raise ValueError('incomplete training epoch batch count')
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                train_seconds = time.monotonic() - train_started
                val_started = time.monotonic()
                val_loss, value, metrics = self._validate()
                train_loss = float((loss_sum / batches).cpu())
                if not all(math.isfinite(x) for x in [train_loss, val_loss, value]):
                    raise ValueError('finite training and validation metrics required')
                if not math.isclose(value, metrics['trajectory_physical_station_macro_mae'], abs_tol=1e-12):
                    raise ValueError('selection metric differs from full physical trajectory')
                self.scheduler.step(value)
                lr_next = float(self.optimizer.param_groups[0]['lr'])
                reason = self.stop.observe(epoch, value, lr_next < lr_used)
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history[METRIC].append(value)
                history['train_seconds'].append(train_seconds)
                history['validation_seconds'].append(time.monotonic() - val_started)
                for key, item in metrics.items():
                    if key not in ['trajectory_physical_station_macro_mae', 'validation_seconds']:
                        history.setdefault('validation_' + key, []).append(item)
                current_steps = steps(self.optimizer)
                self.trace.append(dict(epoch=epoch, lr_used=lr_used, lr_next=lr_next,
                                       train_batches=batches, optimizer_steps=current_steps,
                                       previous_optimizer_steps=before_steps,
                                       skipped_amp_updates=batches - (min(current_steps) - (min(before_steps) if before_steps else 0)),
                                       stop_state=asdict(self.stop)))
                if value < best:
                    best = value
                    self.save_state('best.pt', history, best, 'RUNNING')
                status = reason or 'RUNNING'
                self.save_state('last.pt', history, best, status)
                write(self.run_dir / 'optimization_trace.json', self.trace)
                write(self.run_dir / 'status.json', dict(status=status, completed_epochs=epoch,
                                                        best_metric=best, stop_state=asdict(self.stop)))
                print(f"model={self.config.get('model', type(self.model).__name__)} seed={self.config.get('seed', '?')} epoch={epoch:02d} val_MAE={value:.8f} steps={current_steps} next_lr={lr_next:g} {status}", flush=True)
                if reason:
                    break
        except BaseException as error:
            status = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED'
            # Partial weights go to a separate file, leaving last completed epoch intact.
            self.save_state('interrupted.pt', history, best, status, in_progress=epoch)
            write(self.run_dir / 'status.json', dict(status=status, completed_epochs=len(history[METRIC]),
                                                    in_progress_epoch=epoch, error=repr(error),
                                                    automatic_retry=False))
            raise
        result = dict(status=status, epochs=len(history[METRIC]), history=history,
                      best_metric=best, config=self.config, data_hash=self.data_hash,
                      primary_metric_name=self.primary_metric_name, selection_scope='trajectory',
                      elapsed_seconds=previous_seconds + time.monotonic() - started)
        if terminal:
            write(self.run_dir / 'optimization_trace.json', self.trace)
            write(self.run_dir / 'status.json', dict(status=status, completed_epochs=completed,
                                                    best_metric=best, stop_state=asdict(self.stop)))
        write(self.run_dir / 'history.json', result)
        return result


def audit_run(path, *, expected_config=None, expected_data_hash=None, expected_batches=None, incomplete=False):
    """No forward pass: audit complete train/2023 history and saved selected state."""
    path = Path(path)
    last = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)
    h = ({k: last[k] for k in ['history', 'config', 'data_hash', 'status']} if incomplete else read(path / 'history.json'))
    if ((expected_config is not None and h['config'] != expected_config) or
            (expected_data_hash is not None and h['data_hash'] != expected_data_hash)):
        raise ValueError('checkpoint/history differs from approved job')
    values = h['history'][METRIC]
    if not (1 if incomplete else 8) <= len(values) <= 15 or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid completed history')
    if h['status'] not in (SUCCESS | {'RUNNING'} if incomplete else SUCCESS):
        raise ValueError('training has not completed')
    selected = min(range(len(values)), key=lambda i: values[i])
    best = last.get('best_checkpoint') if incomplete else None
    if best is None:
        best = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
    for ck in [best, last]:
        if (ck['data_hash'] != h['data_hash'] or ck['config'] != h['config'] or
                ck['config_hash'] != config_hash(ck['config']) or ck['selection_scope'] != 'trajectory'):
            raise ValueError('checkpoint identity/selection mismatch')
        if not all(torch.isfinite(v).all() for v in ck['model_state'].values()):
            raise ValueError('nonfinite checkpoint model state')
        if set(ck['rng_state']) != {'python', 'numpy', 'torch', 'cuda'} or 'scaler_state' not in ck:
            raise ValueError('missing scaler/RNG state')
        if ck['scaler_state'] and ck['scaler_state'].get('scale', 0) <= 0:
            raise ValueError('invalid AMP scale')
        for state in ck['optimizer_state']['state'].values():
            if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
                raise ValueError('nonfinite optimizer state')
    if (best['epoch'] != selected or best['best_metric'] != min(values) or
            best['history'][METRIC] != values[:selected + 1] or
            last['history'] != h['history'] or last['completed_epochs'] != len(values) or
            last['in_progress_epoch'] is not None or last['status'] != h['status']):
        raise ValueError('checkpoint/history epoch or best mismatch')
    controller = StopController()
    trace = last['optimization_trace']
    if len(trace) != len(values) or (not incomplete and trace != read(path / 'optimization_trace.json')):
        raise ValueError('incomplete optimization trace')
    dummy = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([dummy], lr=h['config']['learning_rate'], weight_decay=h['config']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2)
    scheduler_states = []
    previous_steps = []
    for i, (v, row) in enumerate(zip(values, trace), 1):
        if expected_batches is not None and row['train_batches'] != expected_batches:
            raise ValueError('completed epoch batch count differs from approved job')
        expected_used_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v)
        scheduler_states.append(copy.deepcopy(scheduler.state_dict()))
        if (row['epoch'] != i or row['lr_used'] != expected_used_lr or
                row['lr_next'] != optimizer.param_groups[0]['lr'] or
                row['previous_optimizer_steps'] != previous_steps or row['train_batches'] <= 0 or
                len(row['optimizer_steps']) != 1):
            raise ValueError('scheduler/optimizer trace mismatch')
        updates = row['optimizer_steps'][0] - (previous_steps[0] if previous_steps else 0)
        if not 0 <= updates <= row['train_batches'] or row['skipped_amp_updates'] != row['train_batches'] - updates:
            raise ValueError('invalid actual optimization update count')
        previous_steps = row['optimizer_steps']
        reason = controller.observe(i, v, row['lr_next'] < row['lr_used'])
        if reason and i < len(values):
            raise ValueError('continued after fixed stopping rule')
    if (reason or 'RUNNING') != h['status'] or last['stop_state'] != asdict(controller):
        raise ValueError('stop rule/status mismatch')
    for ck, index in [(best, selected), (last, len(values) - 1)]:
        opt = ck['optimizer_state']
        actual_steps = sorted({int(s['step']) for s in opt['state'].values() if 'step' in s})
        if (ck['scheduler_state'] != scheduler_states[index] or actual_steps != trace[index]['optimizer_steps'] or
                len(opt['param_groups']) != 1 or opt['param_groups'][0]['lr'] != trace[index]['lr_next'] or
                opt['param_groups'][0]['weight_decay'] != h['config']['weight_decay']):
            raise ValueError('saved optimizer/scheduler state mismatch')
    status = read(path / 'status.json') if not incomplete else None
    if not incomplete and (status['status'] != h['status'] or status['completed_epochs'] != len(values)):
        raise ValueError('status/history mismatch')
    return dict(audit_passed=True, status=h['status'], epochs=len(values), selected_epoch=selected + 1,
                validation_mae=min(values), selection_split='2023',
                sha256={n: sha(path / n) for n in (['last.pt'] if incomplete else ['best.pt', 'last.pt', 'history.json', 'optimization_trace.json'])})
