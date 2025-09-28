"""
MRO Telemetry Processing Pipeline
------------------------------------------------------
Purpose
    Prepare Mars Reconnaissance Orbiter (MRO) Deep Space Network telemetry for
    downstream deep‑learning anomaly detection. The pipeline:
      • Loads raw MONS (monitoring) tracks and DRS (incident reports) from
        pickled payloads.
      • Computes a global time origin and derives time features per sample.
      • Learns preprocessing transforms (missingness drop, high‑cardinality
        categorical drop, one‑hot encoding, correlation pruning) on TRAIN only.
      • Scales features (StandardScaler) fitted on TRAIN statistics.
      • Materializes per‑track feature Parquet files and binary label masks.
      • Emits a dataset manifest mapping tracks to label files per split.

Inputs
    PROJECT_DIR/
      ├─ data_files/                (created if missing; per‑track outputs)
      ├─ mons.pkl                   (list[(track_id, DataFrame)])
      └─ drs.pkl                    (list[(track_id, DataFrame)] with incident windows)

Outputs
    PROJECT_DIR/
      ├─ processed_data/
      │   ├─ manifest.json          (per‑split index of track + label files)
      │   └─ scaler.pkl             (StandardScaler fitted on TRAIN)
      ├─ data_files/
      │   ├─ track_<id>.parquet     (numeric, scaled features)
      │   └─ track_<id>_labels.npy  (binary mask aligned to rows)
      ├─ pipeline_debug.log         (structured run log)
      └─ PIPELINE_SUCCESS.FLAG      (sentinel on success)

Notes
    • Timestamp column is AOJE; it is converted to pandas datetime then to
      seconds since a global start (for cyclic time features).
    • DRS windows are applied per track; any overlap inflates a contiguous 1s
      region in the label mask.
"""

import os
import pickle
import numpy as np
import pandas as pd
import sys
import logging
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================
# --- 1. CONFIGURATION ---
# ==========================
PROJECT_DIR = Path("/home/nzhou/MRO")
DATA_FILES_DIR = PROJECT_DIR / "data_files"
OUTPUT_DIR = PROJECT_DIR / "processed_data"
MONS_FILE, DRS_FILE = "mons.pkl", "drs.pkl"
LOG_FILE = PROJECT_DIR / "pipeline_debug.log"
SUCCESS_FLAG_FILE = PROJECT_DIR / "PIPELINE_SUCCESS.FLAG"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
SCALER_FILE = OUTPUT_DIR / "scaler.pkl"

# ======================================
# --- 2. MODELING & PROCESSING PARAMS ---
# ======================================
TIMESTAMP_COLUMN_NAME = 'AOJE'
TRAIN_RATIO, VALIDATION_RATIO = 0.70, 0.15
MISSINGNESS_THRESHOLD, CORRELATION_THRESHOLD = 0.01, 0.95
RANDOM_SEED = 42

# Structured logging: file + stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)])


def load_raw_dsn_data(data_dir: Path, mons_fn: str, drs_fn: str) -> tuple[list, list]:
    """Load raw MONS and DRS payloads from pickled files.

    Expects:
        mons.pkl: list of (track_id, DataFrame)
        drs.pkl : list of (track_id, DataFrame) with INCIDENT_* columns
    """
    logging.info(f"Loading raw data from {data_dir}...")
    mons_path, drs_path = data_dir / mons_fn, data_dir / drs_fn
    if not mons_path.exists() or not drs_path.exists():
        raise FileNotFoundError(f"Raw data files not found in {data_dir}")
    with open(mons_path, 'rb') as f:
        mons = pickle.load(f)
    with open(drs_path, 'rb') as f:
        drs = pickle.load(f)
    return mons, drs


def find_global_start_time(mons: list, timestamp_col: str) -> pd.Timestamp:
    """Compute global earliest timestamp across all MONS tracks.

    Returns pandas.Timestamp to anchor seconds_since_start.
    """
    logging.info("Finding the global start time...")
    min_time = pd.Timestamp.max
    for _, track_df in mons:
        current_min = pd.to_datetime(track_df[timestamp_col]).min()
        if current_min < min_time:
            min_time = current_min
    return min_time


def split_tracks_by_id(mons: list, drs: list, train_ratio: float, val_ratio: float, seed: int) -> tuple[list, list, list]:
    """Split track IDs into train/val/test; force anomalies into test.

    • Normal IDs are partitioned by ratio.
    • Any ID that appears in DRS is appended to test.
    """
    logging.info("Splitting track IDs...")
    np.random.seed(seed)
    all_track_ids = {track[0] for track in mons}
    anomalous_track_ids = {report[0] for report in drs}
    normal_track_ids = list(all_track_ids - anomalous_track_ids)
    np.random.shuffle(normal_track_ids)

    num_normal = len(normal_track_ids)
    train_end = int(train_ratio * num_normal)
    val_end = train_end + int(val_ratio * num_normal)

    train_ids, val_ids, test_ids = normal_track_ids[:train_end], normal_track_ids[train_end:val_end], normal_track_ids[val_end:] + list(anomalous_track_ids)
    return train_ids, val_ids, test_ids


def process_and_save_track(track_id, mons_map, drs_map, global_start, transformations, scaler):
    """Apply learned transforms to a single track and persist artifacts.

    Saves:
        data_files/track_<id>.parquet     — scaled numeric features
        data_files/track_<id>_labels.npy  — binary anomaly mask (T×1)
    """
    track_df = mons_map[track_id].copy()

    # --- Initialize label mask (binary per row) ---
    label_mask = np.zeros((len(track_df), 1), dtype=np.float32)
    if track_id in drs_map:
        for dr_df in drs_map[track_id]:
            start_time, end_time = pd.to_datetime(dr_df['INCIDENT_START_TIME_DT'].iloc[0]), pd.to_datetime(dr_df['INCIDENT_END_TIME_DT'].iloc[0])
            track_times = pd.to_datetime(track_df[TIMESTAMP_COLUMN_NAME])
            anomaly_indices = track_df.index[(track_times >= start_time) & (track_times <= end_time)]
            if not anomaly_indices.empty:
                label_mask[anomaly_indices.min():anomaly_indices.max() + 1] = 1
    
    # --- Time features ---
    dt_series = pd.to_datetime(track_df[TIMESTAMP_COLUMN_NAME])
    track_df['seconds_since_start'] = (dt_series - global_start).dt.total_seconds()
    track_df['hour_sin'], track_df['hour_cos'] = np.sin(2*np.pi*dt_series.dt.hour/24), np.cos(2*np.pi*dt_series.dt.hour/24)
    track_df['dayofweek_sin'], track_df['dayofweek_cos'] = np.sin(2*np.pi*dt_series.dt.dayofweek/7), np.cos(2*np.pi*dt_series.dt.dayofweek/7)
    track_df.drop(columns=[TIMESTAMP_COLUMN_NAME], inplace=True)

    # --- Column pruning ---
    for col_list_name in ['high_card_features_to_drop', 'columns_to_drop_missing', 'columns_to_drop_correlated']:
        if transformations.get(col_list_name):
            track_df.drop(columns=transformations[col_list_name], inplace=True, errors='ignore')
    
    # --- One‑hot encode categoricals (no explicit NaN bucket) ---
    categorical_cols_to_encode = [col for col in track_df.columns if track_df[col].dtype == 'object']
    if categorical_cols_to_encode:
        track_df = pd.get_dummies(track_df, columns=categorical_cols_to_encode, dummy_na=False, dtype=float)

    # Align to training‑time column set; unseen → 0
    if transformations.get('final_columns'):
        track_df = track_df.reindex(columns=transformations['final_columns'], fill_value=0.0)
    track_df.fillna(0.0, inplace=True)

    # --- Scaling ---
    if not track_df.empty:
        scaled_data = scaler.transform(track_df)
        processed_df = pd.DataFrame(scaled_data, columns=track_df.columns)
    else:
        processed_df = pd.DataFrame(columns=track_df.columns)

    # --- Persist per‑track artifacts ---
    track_filename = DATA_FILES_DIR / f"track_{track_id}.parquet"
    labels_filename = DATA_FILES_DIR / f"track_{track_id}_labels.npy"
    processed_df.to_parquet(track_filename)
    np.save(labels_filename, label_mask)
    return track_filename.name, labels_filename.name


# ===============================
# --- Main Execution (CLI)     ---
# ===============================
if __name__ == "__main__":
    try:
        # Ensure output dirs exist for idempotent runs
        OUTPUT_DIR.mkdir(exist_ok=True)
        DATA_FILES_DIR.mkdir(exist_ok=True)
        logging.info("--- Starting Data Processing Pipeline ---")
        
        # 1) Load raw payloads
        mons_data, drs_data = load_raw_dsn_data(PROJECT_DIR, MONS_FILE, DRS_FILE)
        if not mons_data:
            raise ValueError("No monitoring tracks loaded.")
        
        # 2) Split by track ID (anomalies forced into test)
        train_ids, val_ids, test_ids = split_tracks_by_id(mons_data, drs_data, TRAIN_RATIO, VALIDATION_RATIO, RANDOM_SEED)
        
        # 3) Construct ID→DF maps for fast lookup
        mons_map = {track[0]: track[1] for track in mons_data}
        # drs_map: id → list of DRS DataFrames for that id
        drs_map = {dr_id: [df for i, (id, df) in enumerate(drs_data) if id == dr_id] for dr_id, _ in drs_data}

        # 4) Learn transforms from TRAIN only
        logging.info("Analyzing training set to learn transformations...")
        full_train_df = pd.concat([mons_map[tid] for tid in train_ids], ignore_index=True)
        
        transformations = {}
        # Drop columns with excessive missingness (>1%)
        missing_pct = full_train_df.isna().sum() / len(full_train_df)
        transformations['columns_to_drop_missing'] = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
        
        # Drop high‑cardinality categoricals to avoid OHE explosion
        categorical_cols = [col for col in full_train_df.columns if col.dtype == 'object']
        if categorical_cols:
            cardinality_threshold = int(np.percentile(full_train_df[categorical_cols].nunique(), 90))
            transformations['high_card_features_to_drop'] = [col for col in categorical_cols if full_train_df[col].nunique() > cardinality_threshold]
        
        # One‑hot encode remaining categoricals; then correlation pruning
        temp_df = full_train_df.drop(columns=transformations.get('high_card_features_to_drop', []) + transformations.get('columns_to_drop_missing', []), errors='ignore')
        categorical_cols_to_encode = [col for col in temp_df.columns if temp_df[col].dtype == 'object']
        temp_df = pd.get_dummies(temp_df, columns=categorical_cols_to_encode, dummy_na=False, dtype=float)
        
        corr_matrix = temp_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        transformations['columns_to_drop_correlated'] = [column for column in upper_tri.columns if any(upper_tri[column] > CORRELATION_THRESHOLD)]
        final_df = temp_df.drop(columns=transformations['columns_to_drop_correlated'])
        
        # Freeze the training‑time column universe
        transformations['final_columns'] = final_df.columns.tolist()

        # Fit scaler on TRAIN only
        scaler = StandardScaler()
        logging.info("Fitting scaler on training data...")
        scaler.fit(final_df)

        # 5) Materialize per‑track artifacts and build manifest
        manifest = {"train": [], "val": [], "test": []}
        global_start = find_global_start_time(mons_data, TIMESTAMP_COLUMN_NAME)
        
        for name, ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
            logging.info(f"--- Processing {name} Tracks ---")
            for track_id in tqdm(ids, desc=f"Processing {name} Tracks"):
                f, l = process_and_save_track(track_id, mons_map, drs_map, global_start, transformations, scaler)
                # Manifest uses consistent keys for downstream tooling
                manifest[name.lower()].append({"track": f, "labels": l})
        
        # 6) Persist pipeline artifacts
        with open(MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=4)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        SUCCESS_FLAG_FILE.touch()
        logging.info("--- Pipeline execution finished successfully! ---")

    except Exception as e:
        logging.error("--- FATAL ERROR: The data processing pipeline failed ---", exc_info=True)
        sys.exit(1)
