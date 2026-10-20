# SWMF-TCN

Code accompanying **A Staged Water–Meteorology Fusion Temporal Convolutional Network for High-Frequency Dissolved Oxygen Forecasting**, prepared for *Environmental Monitoring and Assessment*.

This repository extracts the actual NEON preprocessing, SWMF-TCN and comparator implementations, training loops, and evaluation kernels used for the manuscript. It contains no observations, trained checkpoints, or published prediction files. CPU checks use synthetic inputs only.

## Contents and environment

`src/nerrs_faith/` retains the original implementation namespace. `supporting/` contains the actual extension trainers and final evaluation kernels; `analysis/` contains the original block-bootstrap kernel. `scripts/release.py` is a small new command-line adapter. `SOURCE_PROVENANCE.json` records extraction sources and edits.

Use Python 3.11. The checked environment was Python 3.11.15, PyTorch 2.7.1+cu128, NumPy 2.4.6, pandas 2.2.3, PyArrow 21.0.0 and PyYAML 6.0.3. Requirements pin these Python-package versions (PyTorch's CUDA build is platform dependent). GPU training requires an appropriate CUDA PyTorch installation. A fresh dependency installation and full GPU runs were not tested for this release.

Run all commands from the repository root:

```sh
python -m pip install -r requirements.txt
python scripts/check_release.py
```

For imports outside these scripts, installation is optional:

```sh
python -m pip install --no-deps -e .
python -c "from nerrs_faith.benchmark.models.f4_depth_double_res import build_f4_depth_double_res; print(build_f4_depth_double_res('f4_depth_f4_mid', station_count=6).describe()['parameter_count'])"
```

## Data and preprocessing

Download **RELEASE-2026**, sites **CUPE, SYCA, WLOU**, years **2019–2024**, retaining native basic-product CSV filenames, monthly directories and sensor-position metadata. These are the three manuscript products:

| Product | Official source | DOI |
|---|---|---|
| Water quality, DP1.20288.001 | [NEON](https://data.neonscience.org/data-products/DP1.20288.001/RELEASE-2026) | [10.48443/15k3-aj25](https://doi.org/10.48443/15k3-aj25) |
| Relative humidity and air temperature, DP1.00098.001 | [NEON](https://data.neonscience.org/data-products/DP1.00098.001/RELEASE-2026) | [10.48443/kjyf-eq90](https://doi.org/10.48443/kjyf-eq90) |
| Tipping-bucket precipitation, DP1.00045.001 | [NEON](https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026) | [10.48443/mfds-m209](https://doi.org/10.48443/mfds-m209) |

Set the nine `raw.*_roots` entries in `configs/neon_onsite_group_v1.yaml` to your extracted data directories. Defaults are relative directories such as `data/raw/CUPE/water`; each root must retain NEON's product/site/month directory layout. Sensor-position CSV metadata must be available for all three products. The reader checks product identity, active water positions, meteorological sensor provenance, QA flags and UTC alignment; it does not download data or invent missing metadata.

```sh
python scripts/release.py prepare
```

Preparation retains final-QA-zero observations, right-closed/right-labelled 15-minute bins with at least 12 valid minutes, means for water/temperature/humidity and sums for precipitation. Missing values retain masks; targets are not interpolated. Position-specific scalers use 1 April 2019–31 December 2022 training observations. Prepared training/validation tables stop at December 2023. The controlled prediction path separately rebuilds 2024 tables, matching the actual final experiment.

Seven values plus their seven masks form 14 channels. Histories have 256 steps and outputs 96 steps. The six positions are CUPE/SYCA/WLOU × 101.100.100/102.100.100. `configs/data_contract.json` binds the actual frozen prepared-table, scaler and validation-key identities. Training/prediction stop on disagreement. These hashes are provenance, not included data. Different releases, incomplete downloads, metadata changes or numerical/serialization differences may prevent matching; no real-data rebuild was performed for this package.

## Training and experiment matrix

The final manuscript uses **36 unique learned runs**, seeds **2026, 2027, 2028**. `configs/final_runs.json` is the training authority: SWMF and its fusion/weather controls use the actual v1 loop (maximum 15 epochs); Shared-GRU and LSTM-MV use v2 (maximum 25); TCN-MV, DLinear-MV and PatchTST-MV retain the original five-epoch protocol. Do not substitute the historical five-epoch SWMF experiment. Minimum eight epochs, patience five, min_delta 0.0001 mg/L and two epochs after an LR reduction apply to extension runs. Seven final extension runs were accepted at budget caps; sufficient convergence was not established for those runs.

Inspect the three experiment subsets without reading observations or starting training:

```sh
python scripts/release.py plan --experiment benchmark
python scripts/release.py plan --experiment fusion
python scripts/release.py plan --experiment weather
```

Train the union once (the shared Intermediate/Full checkpoint is trained only once):

```sh
python scripts/release.py train --experiment all --device cuda
```

Alternatively, a single fresh run can be selected, for example:

```sh
python scripts/release.py train --experiment weather --model f4_depth_f4_mid_weather_zeroed --seed 2026 --device cuda
```

These are alternatives: an existing run directory is never overwritten or automatically retrained. Outputs go to `outputs/release_training/<model>/seed-<seed>/`. The CLI retains original model initialization, samplers, CUDA/AMP policy, masked horizon-balanced L1 loss, AdamW and ReduceLROnPlateau. The selected `best.pt` minimizes **2023 full-trajectory physical-unit position-macro MAE**, not the last epoch or 2024 performance. Historical author-path hashes are replaced by local release provenance hashes; algorithmic definitions are preserved. The fresh-run adapter does not include the internal approval/resume system.

The benchmark includes SWMF-TCN, TCN-MV, Shared-GRU, LSTM-MV, DLinear-MV, PatchTST-MV, Persistence and 24-hour seasonal persistence with Persistence fallback. SWMF-TCN has 41,080 stored trainable parameters, six double-convolution causal residual blocks, four source-separated blocks then two interacting blocks, and a four-dimensional monitoring-position embedding. SWMF fusion controls separate 0/4/6 blocks; ordinary-TCN controls separate 0/4/7 layers. Weather-zeroed is retrained with standardized weather values **and masks** zeroed; it is not the historical six-block-separated Water-only control.

## Prediction and evaluation

These commands require the trained checkpoints produced above. No command can directly recover the paper's reported numbers from the assets shipped here.

```sh
python scripts/release.py predict --experiment all --split val --device cuda
python scripts/release.py evaluate --prediction-root outputs/release_predictions/val --output outputs/release_metrics/val
python scripts/release.py predict --experiment benchmark --split test --device cuda
python scripts/release.py evaluate --prediction-root outputs/release_predictions/test --output outputs/release_metrics/test
```

Prediction is FP32 evaluation without autocast, using the original collector and physical reconstruction `(predicted_delta + standardized_origin_DO) * training_scale + training_mean`. Fusion/weather controls are restricted to 2023 validation. The 2024 benchmark requires the original 141,358-origin cohort. The same dataset generates physical-unit deterministic references through the original baseline exporter.

Evaluation writes overall, position and site CSVs, including per-seed and seed-summary values for all 96 steps and four horizon ranges (1–4, 5–24, 25–48, 49–96). MAE is averaged equally over positions; RMSE is the mean of position-specific square roots. R² and MAPE are pooled over valid rows; exactly zero targets are excluded from MAPE. Learned summaries use arithmetic seed means and sample SD (ddof=1); deterministic references have no seed SD.

For paired intervals against TCN-MV, run the following command separately for each seed; change both paths and `--seed` together for 2027 and 2028. Other learned comparators use the same command with their generated prediction path.

```sh
python scripts/release.py bootstrap --candidate outputs/release_predictions/test/f4_depth_f4_mid/seed-2026.parquet --comparator outputs/release_predictions/test/tcn_mv/seed-2026.parquet --seed 2026 --output outputs/release_metrics/test/bootstrap_tcn_seed2026.json
```

The unchanged bootstrap uses 10,000 resamples of monitoring-position × UTC origin-date blocks, recomputes count-weighted position means then the equal-position macro difference, and uses RNG seed `2026 + training_seed`. Differences are SWMF minus comparator; intervals are percentile 95% CIs for annual/H1/H2 scopes. Target keys/masks and valid target values must match. Intervals do not remove adjacent-day dependence or adjust for multiple comparisons.

## Scope and omitted assets

This is a minimal source release candidate. Fourteen synthetic CPU checks cover imports, model initialization/forward shapes, fixed masks/causality, data functions, physical reconstruction, metrics, bootstrap, trainer configuration, CLI loading and the prediction adapter with a temporary untrained synthetic checkpoint. The latter stubs the real-data identity gate and is only a controlled orchestration test. Original-versus-extracted model comparisons and relocation checks are documented in the accompanying author audit. Full preparation, GPU training, real-data inference and end-to-end numerical reproduction were not rerun. Hardware, library builds, source-data bytes and non-deterministic GPU kernels can affect results.

No raw NEON observations, prepared tables, scalers, checkpoints, large prediction Parquets, manuscript assets, SHAP work, plotting intermediates or internal approval records are included. Optional checkpoint distribution and the license for author-owned code await author decisions. Third-party licensing and the exact four retained PatchTST files are documented in `THIRD_PARTY_NOTICES.md`; upstream licenses and attribution are retained.
