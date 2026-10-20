# SWMF-TCN

SWMF-TCN (Staged Water–Meteorology Fusion Temporal Convolutional Network) is a deep learning model for high-frequency dissolved oxygen (DO) forecasting. It uses fixed source-aware convolutional connection masks to control the interaction between water-quality and meteorological information at different stages of temporal encoding.

This repository provides the model implementation and the main workflow for data preprocessing, model training, benchmarking, ablation experiments, and evaluation.

## Features

- **SWMF-TCN model:** A causal temporal convolutional network with separate water-quality and meteorological encoders, staged source interaction, and direct multi-step DO prediction.
- **Baseline comparison:** Includes TCN, GRU, LSTM, DLinear, PatchTST, and persistence-based forecasting methods.
- **Ablation experiments:** Supports early, intermediate, and late fusion comparisons, as well as meteorological-input ablation.
- **Data preprocessing:** Processes high-frequency water-quality and meteorological observations, including missing-value masks and temporal alignment.
- **Model evaluation:** Provides MAE, RMSE, R², MAPE, forecast-horizon analysis, and paired bootstrap evaluation.

## Project Structure
```bash
swmf-tcn/
├── src/nerrs_faith/
│   ├── benchmark/models/   # Forecasting models
│   ├── neon_io.py          # Data loading
│   ├── neon_pipeline.py    # Data preprocessing
│   └── hslf_training.py    # Training framework
├── supporting/             # Training and evaluation
├── analysis/               # Statistical evaluation
├── configs/                # Experiment configurations
├── scripts/
│   ├── release.py          # Main experiment interface
│   └── check_release.py    # Basic validation
├── third_party/            # Third-party implementations
├── data/                   # Local data directory
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11
- PyTorch
- NumPy, pandas, PyArrow and PyYAML
- CUDA-compatible GPU recommended for training

Install dependencies:
```
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

Place the NEON observations in the local data directory and configure the data paths in `configs/neon_onsite_group_v1.yaml`.
```arduino
python scripts/release.py prepare
```

### 2. Model Training

Train the forecasting models and experimental variants:
```css
python scripts/release.py train --experiment all --device cuda
```

### 3. Prediction

Generate predictions using the trained benchmark models:
```bash
python scripts/release.py predict --experiment benchmark --split test --device cuda
```

### 4. Evaluation

Evaluate the model predictions:
```bash
python scripts/release.py evaluate --prediction-root outputs/release_predictions/test --output outputs/release_metrics/test
```

### 5. Basic Verification

Run the included CPU checks:
```
python scripts/check_release.py
```

## Dataset

The study uses high-frequency aquatic monitoring data from  National Ecological Observatory Network (NEON) sites

**Note:** Raw data and trained model checkpoints are not included in this repository. Training and evaluation require the corresponding data and generated checkpoints.

## Third-Party Software

Third-party code and its licensing information are documented in `THIRD_PARTY_NOTICES.md`.
