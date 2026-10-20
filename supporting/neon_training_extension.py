"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from nerrs_faith.hslf_training import make_hslf_loader
from nerrs_faith.neon_full_smoke import NeonBenchmarkAdapter, _build
from nerrs_faith.neon_pilot import _validation_data
from nerrs_faith.neon_onsite_experiment import StationBalancedBatchSampler, resolve_onsite_loader_config
from nerrs_faith.neon_controlled_weather_ablation import WeatherZeroedAdapter
from nerrs_faith.benchmark.models.f4_depth_double_res import build_f4_depth_double_res
from nerrs_faith.benchmark.models.fusion_depth_tcn import FusionDepthTCNAdapter

ROOT = Path(__file__).resolve().parents[1]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_model(job):
    name = job['model']
    if job['group'] == 'baseline':
        return _build(name, repo_root=ROOT, station_count=6, reserve_count=3, device='cpu')
    if job['group'] == 'ordinary':
        depth = {'fusion_f0_early': 0, 'fusion_f4_mid': 4, 'fusion_f7_late': 7}[name]
        return NeonBenchmarkAdapter(FusionDepthTCNAdapter(station_count=6, fusion_after=depth))
    if job['group'] == 'zeroed':
        return WeatherZeroedAdapter(build_f4_depth_double_res('f4_depth_f4_mid', station_count=6))
    return NeonBenchmarkAdapter(build_f4_depth_double_res(name, station_count=6))

def source_config():
    cfg = yaml.safe_load((ROOT / 'configs/neon_onsite_group_v1.yaml').read_text(encoding='utf-8'))
    # Strip test locations/dates before passing configuration to the data constructor.
    cfg.pop('test_root', None)
    cfg['splits'] = {k: v for k, v in cfg['splits'].items() if not k.startswith('test_')}
    return cfg

def trainval_data():
    cfg = source_config()
    # Predicate-pushdown version of the original P0 "read then <=val_end".
    # Same tables, windows and labels as original; no 2024 dataset is constructed.
    train, val, scalers, _ = _validation_data(ROOT / cfg['prepared_root'], cfg)
    if len(val.station_ids) != 6 or len(val) * 96 != 13820736 or len(train) == 0:
        raise ValueError('train/2023 window identity mismatch')
    return train, val, scalers

def loaders_for(job, dataset, val):
    common = resolve_onsite_loader_config(source_config(), device='cuda')
    if job['group'] == 'baseline':
        # Preserve original DataLoader RandomSampler AND shared sampler/base-seed generator.
        generator = torch.Generator().manual_seed(job['seed'])
        train = make_hslf_loader(dataset, batch_size=512, shuffle=True, generator=generator,
                                 device='cuda', **common)
    else:
        sampler = StationBalancedBatchSampler(dataset, 512, seed=job['seed'])
        train = make_hslf_loader(dataset, batch_sampler=sampler, device='cuda', **common)
    valid = make_hslf_loader(val, batch_size=512, shuffle=False, device='cuda', **common)
    return train, valid
