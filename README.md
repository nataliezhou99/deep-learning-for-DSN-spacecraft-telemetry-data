# Deep Learning for DSN Spacecraft Telemetry Data

> End-to-end pipeline for anomaly detection on NASA Deep Space Network (DSN) telemetry (JWST & MRO).  
> Combines feature learning with deep models (CNN/BiLSTM/Attention & Transformer) and classical ML (Random Forest), plus robust preprocessing and evaluation.

---

## TL;DR

- **Goal:** Detect & localize anomalies in spacecraft telemetry time series from DSN passes
- **Data:** Mission telemetry + incident reports (per-track labels). *(Raw data not included.)*
- **Pipeline:**
  1. **Preprocess** & align tracks → engineered features → train/val/test manifest  
  2. **Train DL model** to predict high-variance targets from inputs  
  3. **Extract latent features** + **prediction error** per window  
  4. **Train Random Forest** on extracted features → anomaly scores  
  5. **Evaluate** with point-adjusted F1, ROC/PR, confusion matrix
- **Missions:** JWST and MRO (separate scripts, same pattern)
- **Extras:** Optuna hyper-param search, SMOTE class balancing, clear artifacts & logs

---

## Repository Map

```
deep-learning-for-DSN-spacecraft-telemetry-data/
├─ JWST/
│  ├─ data_pipeline_jwst.py              # Build manifest, scale/encode, save per-track parquet + labels
│  ├─ data_utils_jwst.py                 # PyTorch dataset + dataloaders for sliding windows
│  ├─ train_hybrid_autoencoder_jwst.py   # CNN + BiLSTM + Attention encoder/decoder training
│  ├─ autoencoder_extract_features_jwst.py # Latents for manifest test tracks (per-track arrays)
│  ├─ extract_features_jwst.py           # Latents for train/test splits (mirrors training architecture)
│  └─ train_random_forest_jwst.py        # Random Forest training & evaluation with point-adjusted metrics
│
└─ MRO/
   ├─ data_pipeline_mro.py              # Build manifest, scale/encode, save per-track parquet + labels
   ├─ data_utils_hybrid_vae_mro.py      # PyTorch dataset + dataloaders for sliding windows
   ├─ train_hybrid_vae_mro.py           # Transformer encoder + MLP decoder (prediction task)
   ├─ extract_features_mro.py           # Latents + prediction error → features for classical ML
   └─ vae_train_random_forest_mro.py    # Random Forest training with stratified splits, Optuna, SMOTE & diagnostics
```


---

## What’s Interesting (Highlights)

- **Time-aware features:** `seconds_since_start`, cyclical time (`hour_sin/cos`, `dayofweek_sin/cos`), per-track alignment to a **global start**
- **Data quality:** high-cardinality categorical drop, correlated-feature pruning, missingness thresholding, **StandardScaler** on numerics
- **Windowed labels:** window label = `any(anomaly)` → precise **point-adjusted scoring** during eval
- **Two-stage modeling:**
  - Stage 1: DL model learns **compressed representations** while predicting hardest (high-variance) targets
  - Stage 2: Classical ML on **latent features + error** (strong anomaly signal)
- **Imbalance handling:** SMOTE + threshold search for best **point-adjusted F1**
- **Reproducible artifacts:** per-track parquet features & `.npy` labels, `manifest.json`, saved models, plots, CSV logs

---

## Quick Start

> ⚠️ **Update paths** — scripts currently use hardcoded `PROJECT_DIR` like `/home/nzhou/...`.  
> Change these to your local repo path before running.

### 1) Environment

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy pandas scikit-learn imbalanced-learn optuna pyarrow fastparquet tqdm matplotlib seaborn
```

### 2) Prepare Data

Each script defines a mission-specific `PROJECT_DIR` (currently pointing to `/home/nzhou/...`).
Update those constants to wherever you store the DSN telemetry on your machine. The code expects the
following structure under each project directory:

- **JWST** (`PROJECT_DIR` default: `/home/nzhou/updated_dsn_project/JWSTData`)
  - Raw chunked telemetry pickles and DRS CSVs alongside the pipeline script expects files like `chunk_*_mon_JWST.pkl.gz`
  - Outputs written to `processed_diffusion_style/`
- **MRO** (`PROJECT_DIR` default: `/home/nzhou/updated_dsn_project/MRODataSet`)
  - Raw telemetry/incident pickles (`mons.pkl`, `drs.pkl`) in the project root
  - Outputs written to `data_files/` and `processed_data/`

Required inputs (see scripts for exact column names):
- DSN monitoring/telemetry per-track data (parquet or pickle after ingestion)
- Incident/DRS reports mapping track-time ranges → anomaly labels

The pipeline scripts will:
- Merge/align telemetry + incidents
- Engineer features & scale numerics
- Write per-track parquet & *_labels.npy
- Emit processed_data/manifest.json with train/val/test track lists

## Run the Full Pipeline
### A) JWST
#### 1) Build dataset & manifest
`python JWST/data_pipeline_jwst.py`

#### 2) Train deep model (CNN + BiLSTM + Attention)
`python JWST/train_hybrid_autoencoder_jwst.py`

#### 3) Extract features (latent encodings)
`python JWST/autoencoder_extract_features_jwst.py`

> The `JWST/extract_features_jwst.py` script additionally exports latent features for
> both train and test tracks into mission work directories when you want to mirror
> the MRO workflow.

#### 4) Train Random Forest on extracted features (SMOTE + evaluation)
`python JWST/train_random_forest_jwst.py`

### B) MRO
#### 1) Build dataset & manifest
`python MRO/data_pipeline_mro.py`

#### 2) Train deep model (Transformer encoder + MLP decoder)
`python MRO/train_hybrid_vae_mro.py`

#### 3) Extract features (latent + prediction error)
`python MRO/extract_features_mro.py`

#### 4) Train Random Forest on extracted features (Optuna + SMOTE)
`python MRO/vae_train_random_forest_mro.py`

## Key Artifacts & Outputs

- `processed_data/`
  - `manifest.json` — track IDs for train/val/test
  - `tracks/*.parquet` — per-track features
  - `tracks/*_labels.npy` — per-timestamp anomaly labels
- `jwst_vae_work/<dataset>/best_prediction_model_<dataset>.pth` and `MRODataSet/best_prediction_model.pth` — trained DL models
- `xgboost_data/{train,test}/*_features.npy` & `*_labels.npy` — extracted features
- `plots/` — ROC, PR, confusion matrix, feature importances
- CSV & console logs for training/eval

---

## Modeling Details

- **JWST:** Hybrid CNN → BiLSTM → **Attention** encoder with a decoder head to predict selected targets
- **MRO:** **Transformer** encoder (positional encoding) + MLP decoder
- **Targets:** Top-variance numeric telemetry signals (auto-selected from training split)  
- **Features to RF:** JWST → latent state (encoder output); MRO → latent state **+** prediction **error** (Huber-style)
- **Optimization:** Optuna search for RF (`n_estimators`, `max_depth`, etc.)
- **Class imbalance:** Oversampling via SMOTE

---

## Evaluation

- **Primary:** **Point-adjusted F1** — if any timestamp in an incident window is hit, count the whole window as detected
- **Secondary:** Precision/Recall, ROC-AUC, confusion matrix (threshold chosen by best point-adjusted F1)
- **Why point-adjusted?** Detecting *events* matters more than every single timestamp

---

## Reproducibility & Portability

- Set a consistent `RANDOM_SEED` in the pipelines (present in code)
- All transformations (drop rules, scaler fitting, selected columns) are derived **only** from the training split & serialized alongside outputs

---

## Hardware

- Runs on CPU or GPU (`cuda` auto-detect)
- Preprocessing is I/O-heavy; DL training benefits from GPU

---

## Notes for Reviewers

This repo demonstrates: data engineering at scale, sequential modeling, representation learning, classic + deep hybrid systems, careful evaluation under imbalanced labels, and production-style logging/artifacts.  
Raw DSN telemetry are **not** included; scripts and structure show how to reproduce on similar data.

---

## Author

**Natalie Zhou** — Caltech CS, DSN/JPL research, computational biology, and applied ML.  
Feel free to reach out on GitHub for details.
