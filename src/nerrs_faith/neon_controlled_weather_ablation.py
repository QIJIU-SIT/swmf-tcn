"""Selected original definitions; see SOURCE_PROVENANCE.json."""
from __future__ import annotations
from nerrs_faith.neon_full_smoke import NeonBenchmarkAdapter
from nerrs_faith.neon_fusion_depth import perturb_batch

class WeatherZeroedAdapter(NeonBenchmarkAdapter):
    """Keep the original .adapter parameter hierarchy and all trainable weights."""
    def forward(self, batch):
        ablated = perturb_batch(batch, "met_zero", seed=0, block_steps=96)
        return super().forward(ablated)
