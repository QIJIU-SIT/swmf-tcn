Download the three NEON RELEASE-2026 products linked in the repository README. Retain native monthly directories and sensor-position CSVs. Configure the nine `raw.*_roots` paths in `configs/neon_onsite_group_v1.yaml`; defaults point below `data/raw/<site>/<water|humidity|precipitation>/`.

No observations or fabricated example measurements are shipped. `scripts/check_release.py` constructs synthetic data in memory and removes temporary files after checking them. Its outputs are not NEON experiment results.
