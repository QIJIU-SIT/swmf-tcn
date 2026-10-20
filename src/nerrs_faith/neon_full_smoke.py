"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from nerrs_faith.benchmark.models.shared_gru import SharedGRUAdapter
from nerrs_faith.benchmark.schema import DeviceBatch
from nerrs_faith.benchmark.models.neon_multivariate import NeonForecastAdapter, RecurrentMVAdapter, TCNMVAdapter, DLinearMVAdapter
from nerrs_faith.benchmark.models.fusion_depth_tcn import FusionDepthTCNAdapter
from nerrs_faith.benchmark.models.f4_tcn_factorial import F4TCNFactorialAdapter
from nerrs_faith.benchmark.models.f4_depth_double_res import F4DepthDoubleResAdapter
from nerrs_faith.benchmark.models.patchtst_mv import PatchTSTMVAdapter

def _build(name, *, repo_root, station_count, reserve_count=3, device='cpu'):
    """Release-only dispatch of the original formal baseline constructors."""
    context = dict(station_count=station_count,
                   patchtst_source=repo_root / 'third_party' / 'PatchTST')
    builders = dict(lstm_mv=_build_lstm, tcn_mv=_build_tcn,
                    dlinear_mv=_build_dlinear, patchtst_mv=_build_patchtst_mv)
    if name == 'shared_gru':
        adapter = SharedGRUAdapter(input_size=7, num_stations=station_count,
                                  hidden_size=32, station_embedding_size=4,
                                  pred_len=96, seq_len=256)
        return NeonBenchmarkAdapter(adapter).to(device)
    if name not in builders:
        raise ValueError('Model is outside the final baseline matrix')
    return builders[name](context).to(device)

def _build_lstm(context):
    return NeonBenchmarkAdapter(RecurrentMVAdapter(cell="lstm", station_count=context["station_count"], hidden_size=32))

def _build_tcn(context):
    return NeonBenchmarkAdapter(TCNMVAdapter(station_count=context["station_count"], channels=32))

def _build_dlinear(context):
    return NeonBenchmarkAdapter(DLinearMVAdapter(station_count=context["station_count"], moving_average=25))

def _build_patchtst_mv(context):
    return NeonBenchmarkAdapter(PatchTSTMVAdapter(source_root=context["patchtst_source"], station_count=context["station_count"], model_config={"patch_len": 16, "stride": 8, "d_model": 128, "n_heads": 8, "e_layers": 3, "d_ff": 256}))

class NeonBenchmarkAdapter(nn.Module):
    """Adapt NEON mapping batches to the trusted 256-step benchmark seam."""
    def __init__(self, adapter: nn.Module):
        super().__init__(); self.adapter = adapter

    def forward(self, batch):
        if isinstance(self.adapter, (NeonForecastAdapter, RecurrentMVAdapter, TCNMVAdapter, DLinearMVAdapter, PatchTSTMVAdapter, FusionDepthTCNAdapter, F4TCNFactorialAdapter, F4DepthDoubleResAdapter)):
            result = self.adapter(batch)
        else:
            model_batch = DeviceBatch(
                history=torch.cat((batch["water"], batch["met"]), dim=-1),
                history_mask=torch.cat((batch["water_mask"], batch["met_mask"]), dim=-1),
                target=batch["target"], target_mask=batch["target_mask"], last_do=batch["last_do"],
                last_do_physical=batch["last_do_physical"], station_id=batch["station_id"],
                reserve_id=batch["reserve_id"], station=batch["position"], reserve=batch["site"],
                origin=batch["origin"], target_times=batch["target_times"], frequency_minutes=15,
            )
            result = self.adapter(model_batch)
        return result if isinstance(result, tuple) else (result, {})

    def describe(self):
        return self.adapter.describe() if hasattr(self.adapter, "describe") else {"name": type(self.adapter).__name__}

def resolve_neon_data_root(config: dict, *, repo_root: Path) -> Path:
    """Resolve prepared data for onsite configs while retaining legacy protocol support."""
    configured = config.get("prepared_root")
    if configured is None:
        configured = config.get("pilot", {}).get("output_root")
    if not configured:
        raise ValueError("NEON config must declare prepared_root or pilot.output_root")
    path = Path(configured)
    return path if path.is_absolute() else repo_root / path
