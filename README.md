# FDS_IDS_CNN — Unsupervised Intrusion Detection (UNSW-NB15)

This academic project implements an unsupervised Intrusion Detection System (IDS) based on two main approaches:
* Convolutional Autoencoder (ConvAE) to detect anomalies via reconstruction error
* Isolation Forest for anomaly detection on scaled features

The data come from the UNSW-NB15 dataset and are prepared through an extraction script that selects a subset focused on the top 5 attack categories and creates consistent splits for training and evaluation.

For a complete description of the objectives, design choices, and experimental results, please refer to the report: Report.pdf.

## Project Structure
* Dataset/: original CSV files and generated train/eval files
* src/: source code (model, training, testing, utils, data preparation)
* Model/: saved models and thresholds (AE and IF, plus scalers)
* Plots/: training curves and score distribution plots
* Logs/: execution logs with metrics and confusion matrices
* run_unsup.py: single entry point for training and testing (AE / Isolation Forest)
* requirements.txt: Python dependencies

## Data and Preparation

The data preparation script merges the UNSW-NB15 files and produces two CSV files:
* Dataset/UNSW_top5_train.csv: training set (numerical features only, no text columns, no labels)
* Dataset/UNSW_top5_eval.csv: evaluation set (numerical features + Label and required metadata)

The procedure can be automated directly from the runner or executed by calling the extraction module.

Run via the runner (recommended option):
```bash
python run_unsup.py --mode train --type ae --build_dataset
```
Or manually:
```bash
python -c "from src.data_extraction import build_datasets; build_datasets()"
```

Operational details of CSV parsing:
* Removal of common non-numerical columns (e.g., srcip, dstip, proto, service, attack_cat, state)
* Robust numeric conversion using errors='coerce' and fillna(0)
* Application of StandardScaler when required

## Methods

### Convolutional Autoencoder (ConvAE)
* 1D architecture with Conv1d + BatchNorm + MaxPool, linear bottleneck, and a decoder with ConvTranspose1d
* Trained on scaled features using MSE loss
* Anomaly threshold defined as the 95th percentile of the reconstruction error on the training set
* Evaluation on UNSW_top5_eval.csv with classification metrics (accuracy, classification_report, confusion_matrix)

Saved outputs:
* Model/conv_autoencoder.pth: model weights
* Model/recon_threshold.json: reconstruction error threshold
* Model/scaler.pkl: preprocessing scaler
* Plots/unsup_loss.png: training loss curve

### Isolation Forest (IF)
* Preprocessing with StandardScaler on features (preferably fit on samples considered “normal” when available)
* Training of sklearn.ensemble.IsolationForest on scaled features
* Anomaly threshold set to the percentile 100 * (1 - contamination) of anomaly scores on the training set
* Evaluation on UNSW_top5_eval.csv using classification metrics

Saved outputs:
* Model/isolation_forest.pkl: Isolation Forest model
* Model/isoforest_threshold.json: threshold and metadata
* Model/scaler_isoforest.pkl: scaler used for IF
* Plots/isoforest_scores.png: score histogram with threshold

## Requirements and Setup

Prerequisites:
* Python 3.10+ (recommended)
* macOS with optional Apple Silicon support (MPS acceleration if available)

Dependency installation:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Quick Usage (Unified Runner)

Main examples (AE and IF):
```bash
# 1) Dataset construction + Autoencoder training
python run_unsup.py --mode train --type ae --build_dataset --epochs 30 --batch_size 256 --lr 1e-3 --latent_dim 64

# 2) Autoencoder evaluation (uses defaults in Model/ if not specified)
python run_unsup.py --mode test --type ae --eval_csv Dataset/UNSW_top5_eval.csv

# 3) Isolation Forest training
python run_unsup.py --mode train --type isoforest --build_dataset --contamination 0.01 --n_estimators 300

# 4) Isolation Forest evaluation
python run_unsup.py --mode test --type isoforest --eval_csv Dataset/UNSW_top5_eval.csv
```
Useful options:
*	--model_out, --threshold_out: override output paths
*	--model_in, --threshold_in: explicit paths for evaluation
*	--device: force cpu/cuda/mps; automatically selected by default

## Notes on Labels and Predictions

During evaluation, predictions are derived by comparing errors/scores with the selected threshold:
* AE: low reconstruction error → inlier (benign); high → anomaly
* IF: higher scores indicate more anomalous samples; threshold comparison determines the class

## Metrics are reported in the logs together with the confusion matrix.

Output Directories
* Model/: models, thresholds, scalers
* Plots/: loss curves and score distributions
* Logs/: log files with metrics and reports

## Development and References
* Core model code: src/network_unsup.py
* AE training/evaluation: src/train_unsup.py, src/test_unsup.py
* IF training/evaluation: src/train_isoforest.py, src/test_isoforest.py
* Data preparation: src/data_extraction.py
* Runner: run_unsup.py

For methodological details, design choices, and experimental results, see Report.pdf.
