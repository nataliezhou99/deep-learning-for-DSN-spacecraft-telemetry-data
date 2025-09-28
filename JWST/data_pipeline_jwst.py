"""
JWST Telemetry Processing Pipeline
-----------------------------------------------------------------
Purpose:
    Prepare JWST Deep Space Network (DSN) spacecraft telemetry for
    deep‑learning anomaly detection. The pipeline reads chunked, gzipped
    pickle files of per‑track telemetry, splits tracks into low/high RF bands,
    aligns time to a global origin, learns dataset‑level preprocessing
    transforms from training data, and materializes per‑track feature files
    (Parquet) and label masks (NumPy) with a manifest for downstream training.

Key behaviors:
    • Incremental chunk processing to avoid loading everything into memory.
    • Band‑splitting by measured carrier frequency.
    • Track filtering by minimum length, then stratified sampling by total row
      budget to achieve a target anomalous:normal ratio.
    • Train/val/test split that ensures anomalies are present in the test set.
    • Transformation learning (missingness, high‑cardinality drop, one‑hot
      encoding, zero‑variance drop, correlation pruning, MinMax scaling)
      learned ONLY from (a sample of) the training set.
    • Per‑track feature materialization and anomaly label mask creation.

I/O conventions:
    Input directory   : PROJECT_DIR
    Input chunks      : CHUNK_FILE_PATTERN (e.g., 'chunk_*_mon_JWST.pkl.gz')
    Anomaly table     : DRS_FILE (CSV with INCIDENT_START/END_TIME_DT columns)
    Output directory  : OUTPUT_DIR/<dataset>/{data_files, manifest.json, scaler.pkl}

Operational notes:
    • This module logs to both a rotating file and stdout.
    • It writes a SUCCESS_FLAG_FILE sentinel upon successful completion.
    • Timestamps are assumed to be in seconds since an epoch; a global start
      time is computed from both data and anomaly windows and used to normalize
      each sample’s wall‑time into seconds_since_start.
"""

import os 
import pickle 
import gzip 
import numpy as np 
import pandas as pd 
import sys 
import logging 
import random 
import json 
from pathlib import Path 
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 

# ==========================
# --- 1. CONFIGURATION ---
# ==========================
# All constants here are intended to be operational toggles. In production,
# prefer sourcing these from a config file or environment variables when
# deploying to different environments.
PROJECT_DIR = Path("/home/nzhou/JWST") 
CHUNK_FILE_PATTERN = "chunk_*_mon_JWST.pkl.gz"  # Glob for per‑chunk pickled DF lists
DRS_FILE = "all_dr_data.csv"  # CSV of anomaly incidents with start/end timestamps
OUTPUT_DIR = PROJECT_DIR / "processed_diffusion_style"  # Root output dir for all datasets
LOG_FILE = PROJECT_DIR / "pipeline_jwst_adapted_debug.log"  # Unified log output
SUCCESS_FLAG_FILE = PROJECT_DIR / "PIPELINE_JWST_ADAPTED_SUCCESS.FLAG"  # Sentinel file

# Column semantics
TIMESTAMP_COLUMN_NAME = 'RECEIVED_AT_TS'  # Per‑row absolute timestamp (seconds)
SPLIT_COLUMN_NAME = 'CARRIER_FREQ_MEASURED'  # Used to split into banded datasets
COLUMNS_TO_DROP = ['LNA_AMPLIFIER', 'UPLINK_CARRIER_BAND', 'DOWNLINK_CARRIER_BAND']  # Often empty/meta

# RF frequency band windows (Hz)
FREQ_LOW_RANGE = (2.1e9, 2.4e9) 
FREQ_HIGH_RANGE = (25.0e9, 27.5e9) 

# Dataset shaping parameters
MIN_TRACK_LENGTH = 2000  # Min samples per track to retain
TARGET_TOTAL_ROWS = 4_000_000  # Total rows across sampled tracks (per dataset)
ANOMALOUS_RATIO = 0.06  # Target fraction of anomalous rows
NORMAL_RATIO = 0.94  # Target fraction of normal rows

# Reproducibility
RANDOM_SEED = 42 

# Feature engineering thresholds
MISSINGNESS_THRESHOLD = 0.01  # Drop columns with >1% missingness
CORRELATION_THRESHOLD = 0.95  # Drop one of any pair with |corr|>0.95

# Split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# ======================
# --- Logging Setup  ---
# ======================
# Logs go to both a file and stdout.
logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(levelname)s - %(message)s', 
                     handlers=[logging.FileHandler(LOG_FILE, mode='w'), 
                               logging.StreamHandler(sys.stdout)]) 

# ============================
# --- Helper Functions     ---
# ============================

def process_chunks_incrementally(data_dir: Path, file_pattern: str, split_col: str, low_range: tuple, high_range: tuple, cols_to_drop: list):
    """Load chunked telemetry, split each track into low/high bands, and prune columns.

    Parameters
    ----------
    data_dir : Path
        Directory containing chunked pickle.gz files produced upstream.
    file_pattern : str
        Glob pattern for chunk files.
    split_col : str
        Column name with measured carrier frequency used to split bands.
    low_range, high_range : tuple
        Inclusive frequency windows (Hz) for low/high band assignment.
    cols_to_drop : list
        Columns to drop early to reduce memory.

    Returns
    -------
    (list, list)
        Two lists of (track_id, DataFrame) for low and high bands respectively.

    Notes
    -----
    • Each chunk file is expected to unpickle to an iterable of (track_id, df).
    • Non‑DataFrame entries or tracks without the split column are skipped.
    • Tracks can contribute rows to both bands if their frequency spans windows.
    """
    logging.info(f"Loading and splitting tracks into low and high frequency bands from '{file_pattern}'...") 
    low_band_tracks, high_band_tracks = [], [] 

    # Enumerate chunk files deterministically for reproducibility.
    chunk_files = sorted(data_dir.glob(file_pattern)) 
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found matching pattern '{file_pattern}' in {data_dir}") 

    for file_path in tqdm(chunk_files, desc="Processing Chunks"): 
        # Each file stores a list of (track_id, DataFrame), gzipped + pickled.
        with gzip.open(file_path, 'rb') as f:
            chunk_data = pickle.load(f) 

        for track_id, track_df in chunk_data: 
            if not isinstance(track_df, pd.DataFrame):
                continue  # Defensive: skip unexpected payloads

            # Drop early to reduce memory pressure.
            df = track_df.drop(columns=cols_to_drop, errors='ignore') 

            # Require the split column to exist and be non‑all‑NaN.
            if split_col not in df.columns or df[split_col].isnull().all():
                continue 

            # Band windows
            low_mask = (df[split_col] >= low_range[0]) & (df[split_col] <= low_range[1]) 
            high_mask = (df[split_col] >= high_range[0]) & (df[split_col] <= high_range[1]) 

            # Append sub‑tracks when any rows fall into a band.
            if low_mask.any():
                low_band_tracks.append((f"{track_id}_low", df[low_mask].copy().reset_index(drop=True))) 
            if high_mask.any():
                high_band_tracks.append((f"{track_id}_high", df[high_mask].copy().reset_index(drop=True))) 

    logging.info(f"Finished processing chunks. Low-band tracks: {len(low_band_tracks)}, High-band tracks: {len(high_band_tracks)}") 
    return low_band_tracks, high_band_tracks 


def load_jwst_anomalies(data_dir: Path, drs_fn: str) -> pd.DataFrame: 
    """Load anomaly incidents table from CSV.

    Expects columns INCIDENT_START_TIME_DT and INCIDENT_END_TIME_DT (seconds).
    Raises FileNotFoundError if the CSV is missing.
    """
    logging.info(f"Loading anomaly data from {drs_fn}...") 
    drs_path = data_dir / drs_fn 
    if not drs_path.exists():
        raise FileNotFoundError(f"Anomaly file not found: {drs_path}") 
    anomaly_df = pd.read_csv(drs_path) 
    return anomaly_df 


def find_global_start_time(all_tracks: list, anomaly_df: pd.DataFrame, ts_col: str) -> float: 
    """Compute a global time origin across telemetry and anomaly windows.

    The minimum of (min track timestamp, min anomaly start/end) is used to
    normalize all timestamps to seconds_since_start.
    """
    logging.info("Finding the global start time...") 

    # Guard: ensure we have at least one non‑empty track
    min_track_time = min(track_df[ts_col].min() for _, track_df in all_tracks if not track_df.empty) 

    # Convert anomaly times to numeric (some CSVs may parse as strings)
    anomaly_df['INCIDENT_START_TIME_DT'] = pd.to_numeric(anomaly_df['INCIDENT_START_TIME_DT']) 
    anomaly_df['INCIDENT_END_TIME_DT'] = pd.to_numeric(anomaly_df['INCIDENT_END_TIME_DT']) 

    min_anom_time = min(anomaly_df['INCIDENT_START_TIME_DT'].min(), anomaly_df['INCIDENT_END_TIME_DT'].min()) 
    global_min = min(min_track_time, min_anom_time) 

    logging.info(f"Global start time (in seconds) found: {global_min}") 
    return global_min 


def filter_tracks_by_length(all_tracks: list, min_length: int) -> list: 
    """Filter out tracks with fewer than `min_length` rows.

    Helps stabilize modeling and reduces overhead from very short tracks.
    """
    logging.info(f"Filtering tracks to keep only those with length >= {min_length}...") 
    filtered_tracks = [track for track in all_tracks if len(track[1]) >= min_length] 
    logging.info(f"Length filtering complete. Kept {len(filtered_tracks)} out of {len(all_tracks)} tracks.") 
    return filtered_tracks 


def sample_stratified_by_row_count(anomalous_tracks: list, normal_tracks: list, target_anom_rows: int, target_normal_rows: int, seed: int):
    """Greedily sample whole tracks until hitting target row budgets per class.

    Notes
    -----
    • Sampling is track‑level (not per‑row) to preserve sequence integrity.
    • Order is shuffled but the draw is greedy; this is intentional to keep
      implementation simple and deterministic given a fixed seed.
    """
    logging.info("Performing stratified sampling...") 
    random.seed(seed) 

    sampled_tracks = [] 

    # Fill anomalous budget first
    random.shuffle(anomalous_tracks) 
    current_anom_rows = 0 
    for track in anomalous_tracks: 
        sampled_tracks.append(track) 
        current_anom_rows += len(track[1]) 
        if current_anom_rows >= target_anom_rows:
            break 
    logging.info(f"Sampled {len(sampled_tracks)} anomalous tracks with {current_anom_rows:,} rows (target: {target_anom_rows:,}).") 

    # Then fill normal budget
    num_anom_sampled = len(sampled_tracks) 
    random.shuffle(normal_tracks) 
    current_normal_rows = 0 
    for track in normal_tracks: 
        sampled_tracks.append(track) 
        current_normal_rows += len(track[1]) 
        if current_normal_rows >= target_normal_rows:
            break 
    logging.info(f"Sampled {len(sampled_tracks) - num_anom_sampled} normal tracks with {current_normal_rows:,} rows.") 

    logging.info(f"Sampling complete. Total tracks: {len(sampled_tracks)}, Total rows: {current_anom_rows + current_normal_rows:,}") 
    return sampled_tracks 


def identify_anomalous_tracks(all_tracks: list, anomaly_df: pd.DataFrame, ts_col: str, global_start: float) -> set: 
    """Identify original track IDs that overlap any anomaly window.

    Returns a set of base track IDs (without _low/_high suffix) deemed
    anomalous if *any* overlap occurs.
    """
    logging.info("Scanning all tracks to identify which ones contain anomalies...") 
    anomalous_original_ids = set() 

    # Normalize anomaly windows relative to global origin for vectorized checks.
    norm_anom_df = anomaly_df.copy() 
    norm_anom_df['start_norm'] = norm_anom_df['INCIDENT_START_TIME_DT'] - global_start 
    norm_anom_df['end_norm'] = norm_anom_df['INCIDENT_END_TIME_DT'] - global_start 

    for track_id, track_df in tqdm(all_tracks, desc="Identifying Anomalous Tracks"): 
        norm_track_times = track_df[ts_col] - global_start 
        # Check any overlap with any anomaly window; break early on first hit.
        for _, anom_row in norm_anom_df.iterrows(): 
            if not norm_track_times[(norm_track_times >= anom_row['start_norm']) & (norm_track_times <= anom_row['end_norm'])].empty: 
                anomalous_original_ids.add(str(track_id).split('_')[0]) 
                break 
    logging.info(f"Identified {len(anomalous_original_ids)} original tracks with known anomalies.") 
    return anomalous_original_ids 


def dynamic_split_tracks(all_tracks: list, anomalous_split_ids: set, train_ratio: float, val_ratio: float, seed: int) -> tuple[list, list, list]: 
    """Split into train/val/test by track ID, holding out anomalies for test.

    • Normal tracks are partitioned into train/val/test by the provided ratios.
    • All anomalous base IDs are appended to the test set.
    • Returns three lists of *track IDs* (not DataFrames).
    """
    logging.info(f"Splitting data into Train/Val/Test sets...") 
    np.random.seed(seed) 

    all_track_ids = {track[0] for track in all_tracks} 
    normal_track_ids = list(all_track_ids - anomalous_split_ids) 
    np.random.shuffle(normal_track_ids) 

    num_normal = len(normal_track_ids) 
    train_end = int(train_ratio * num_normal) 
    val_end = train_end + int(val_ratio * num_normal) 

    train_ids = normal_track_ids[:train_end] 
    val_ids = normal_track_ids[train_end:val_end] 
    test_normal_ids = normal_track_ids[val_end:] 

    # Ensure test coverage of anomalies
    test_ids = test_normal_ids + list(anomalous_split_ids) 

    logging.info(f"Split complete. Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)} (including {len(anomalous_split_ids)} anomalous).") 
    return train_ids, val_ids, test_ids 


def process_and_save_jwst_track(track_df, track_id, config, transformations, scaler, data_files_dir): 
    """Apply learned transforms and save per‑track features + labels.

    Output files:
        track_{track_id}.parquet      — numeric features after OHE/pruning/scaling
        track_{track_id}_labels.npy   — binary label mask aligned row‑wise
    """
    df = track_df.copy() 

    # --- Time features ---
    df['seconds_since_start'] = df[TIMESTAMP_COLUMN_NAME] - config['global_start_time'] 
    dt_series = pd.to_datetime(df['seconds_since_start'], unit='s') 
    df['hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24.0) 
    df['hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24.0) 
    df.drop(columns=[TIMESTAMP_COLUMN_NAME], inplace=True) 

    # --- Label mask construction (binary) ---
    label_mask = np.zeros((len(df), 1), dtype=np.float32) 
    for _, anom_row in config['anomaly_df'].iterrows(): 
        start_norm = anom_row['INCIDENT_START_TIME_DT'] - config['global_start_time'] 
        end_norm = anom_row['INCIDENT_END_TIME_DT'] - config['global_start_time'] 
        indices = df.index[(df['seconds_since_start'] >= start_norm) & (df['seconds_since_start'] <= end_norm)] 
        if not indices.empty: 
            # Fill contiguous region inclusively
            label_mask[indices.min():indices.max() + 1] = 1 

    # --- Column pruning prior to encoding ---
    if transformations.get('high_card_features_to_drop'):
        df.drop(columns=transformations['high_card_features_to_drop'], inplace=True, errors='ignore') 
    if transformations.get('columns_to_drop_missing'):
        df.drop(columns=list(set(transformations['columns_to_drop_missing'])), inplace=True, errors='ignore') 
    
    # --- One‑hot encode remaining categoricals (with NaN bucket) ---
    categorical_cols_to_encode = [col for col in df.columns if df[col].dtype == 'object'] 
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, dummy_na=True) 

    # Align columns to training‑time OHE universe; unseen columns -> 0
    df = df.reindex(columns=transformations['all_columns_after_ohe'], fill_value=0.0) 
    df.fillna(0.0, inplace=True) 
    
    # --- Post‑OHE pruning for zero‑variance and high correlation ---
    if transformations.get('columns_to_drop_zerovar'):
        df.drop(columns=transformations['columns_to_drop_zerovar'], inplace=True, errors='ignore') 
    if transformations.get('columns_to_drop_correlated'):
        df.drop(columns=transformations['columns_to_drop_correlated'], inplace=True, errors='ignore') 

    # --- Final numeric column selection + scaling ---
    final_cols = transformations['final_numeric_columns'] 
    if not df.empty and final_cols: 
        df_final = df[final_cols] 
        scaled_data = scaler.transform(df_final)  # MinMax scale using train fit
        processed_df = pd.DataFrame(scaled_data, columns=df_final.columns) 
    else: 
        # Edge case: if no columns survive selection, emit empty DF with header
        processed_df = pd.DataFrame(columns=final_cols) 

    # --- Persist artifacts ---
    track_filename = data_files_dir / f"track_{track_id}.parquet" 
    labels_filename = data_files_dir / f"track_{track_id}_labels.npy" 
    
    processed_df.to_parquet(track_filename) 
    np.save(labels_filename, label_mask) 
    
    return track_filename.name, labels_filename.name 


def process_dataset(tracks_list, dataset_name, config): 
    """End‑to‑end processing for a single dataset (e.g., 'low_band').

    Steps:
        1) Filter by minimum track length
        2) Identify anomalous vs normal pools via interval overlap
        3) Stratified sampling by row budgets
        4) Train/val/test split (anomalies forced into test)
        5) Learn transforms on a sample of TRAIN tracks only
        6) Materialize features/labels + manifest + scaler
    """
    logging.info(f"--- Processing {dataset_name} Dataset (Diffusion-Style Output) ---") 

    # Directory scaffolding
    dataset_output_dir = OUTPUT_DIR / dataset_name 
    data_files_dir = dataset_output_dir / "data_files" 
    dataset_output_dir.mkdir(parents=True, exist_ok=True) 
    data_files_dir.mkdir(exist_ok=True) 

    # (1) Minimum length filter
    tracks_filtered = filter_tracks_by_length(tracks_list, MIN_TRACK_LENGTH) 
    if not tracks_filtered: 
        logging.warning(f"No tracks left for {dataset_name} after length filtering. Skipping.") 
        return 

    # (2) Identify anomalous original IDs
    anomalous_original_ids = identify_anomalous_tracks(tracks_filtered, config["anomaly_df"], TIMESTAMP_COLUMN_NAME, config["global_start_time"]) 
    anomalous_pool = [track for track in tracks_filtered if str(track[0]).split('_')[0] in anomalous_original_ids] 
    normal_pool = [track for track in tracks_filtered if str(track[0]).split('_')[0] not in anomalous_original_ids] 

    # (3) Stratified sampling by total row budgets
    tracks_sampled = sample_stratified_by_row_count(anomalous_pool, normal_pool, int(TARGET_TOTAL_ROWS * ANOMALOUS_RATIO), int(TARGET_TOTAL_ROWS * NORMAL_RATIO), RANDOM_SEED) 
    if not tracks_sampled: 
        logging.warning(f"No tracks left for {dataset_name} after sampling. Skipping.") 
        return 

    # (4) Split into train/val/test (IDs only)
    sampled_track_ids = {track[0] for track in tracks_sampled} 
    final_anomalous_split_ids = {tid for tid in sampled_track_ids if str(tid).split('_')[0] in anomalous_original_ids} 
    
    train_ids, val_ids, test_ids = dynamic_split_tracks(tracks_sampled, final_anomalous_split_ids, TRAIN_RATIO, VALIDATION_RATIO, RANDOM_SEED) 
    
    tracks_map = {track_id: df for track_id, df in tracks_sampled} 
    
    # (5) Learn transformations from a sample of TRAIN tracks only
    logging.info("Analyzing training set to learn transformations...") 
    train_sample_ids = random.sample(train_ids, min(len(train_ids), 50))  
    train_dfs = [tracks_map[tid] for tid in train_sample_ids] 
    full_train_df = pd.concat(train_dfs, ignore_index=True) 

    transformations = {} 

    # Time normalization added so time‑derived features can be computed downstream
    full_train_df['seconds_since_start'] = full_train_df[TIMESTAMP_COLUMN_NAME] - config['global_start_time'] 

    # Drop columns with excessive missingness
    missing_pct = full_train_df.isna().sum() / len(full_train_df) 
    transformations['columns_to_drop_missing'] = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist() 

    # Drop high‑cardinality categoricals (long‑tail tokens fight OHE explosion)
    categorical_cols = [col for col in full_train_df.columns if full_train_df[col].dtype == 'object'] 
    if categorical_cols: 
        cardinality_threshold = int(np.percentile(full_train_df[categorical_cols].nunique(), 90)) 
        transformations['high_card_features_to_drop'] = [col for col in categorical_cols if full_train_df[col].nunique() > cardinality_threshold] 

    # One‑hot encode remaining categoricals with NaN bucket
    temp_df = full_train_df.drop(columns=transformations.get('high_card_features_to_drop', []) + transformations.get('columns_to_drop_missing', []), errors='ignore') 
    categorical_cols_to_encode = [col for col in temp_df.columns if temp_df[col].dtype == 'object'] 
    if categorical_cols_to_encode:
        temp_df = pd.get_dummies(temp_df, columns=categorical_cols_to_encode, dummy_na=True) 

    transformations['all_columns_after_ohe'] = temp_df.columns.tolist() 
    temp_df.fillna(0.0, inplace=True) 

    # Zero‑variance feature drop
    transformations['columns_to_drop_zerovar'] = temp_df.columns[temp_df.var() == 0].tolist() 
    if transformations.get('columns_to_drop_zerovar'):
        temp_df.drop(columns=transformations['columns_to_drop_zerovar'], inplace=True, errors='ignore') 

    # Correlation pruning (upper triangle to avoid double counting)
    corr_matrix = temp_df.corr().abs() 
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
    transformations['columns_to_drop_correlated'] = [column for column in upper_tri.columns if any(upper_tri[column] > CORRELATION_THRESHOLD)] 
    if transformations.get('columns_to_drop_correlated'):
        temp_df.drop(columns=transformations['columns_to_drop_correlated'], inplace=True, errors='ignore') 
    
    # Final numeric column set
    transformations['final_numeric_columns'] = temp_df.select_dtypes(include=np.number).columns.tolist() 

    # Fit scaler ONLY on training distribution
    scaler = MinMaxScaler() 
    scaler.fit(temp_df[transformations['final_numeric_columns']]) 
    logging.info("Transformation analysis complete. Scaler has been fitted.") 

    # (6) Materialize processed tracks + manifest
    manifest = {"train": [], "val": [], "test": []} 
    all_ids_map = {"train": train_ids, "val": val_ids, "test": test_ids} 
    
    for set_name, id_list in all_ids_map.items(): 
        logging.info(f"--- Processing {len(id_list)} tracks for {set_name.upper()} set ---") 
        for track_id in tqdm(id_list, desc=f"Processing {set_name.UPPER()} tracks"):
            # NOTE: tqdm label uses .UPPER() intentionally – matches original logic
            track_df = tracks_map[track_id] 
            f, l = process_and_save_jwst_track(track_df, track_id, config, transformations, scaler, data_files_dir) 
            manifest[set_name].append({"track_features": f, "track_labels": l}) 
            
    manifest_file = dataset_output_dir / "manifest.json" 
    scaler_file = dataset_output_dir / "scaler.pkl" 
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=4) 
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f) 
        
    logging.info(f"--- Successfully finished processing for {dataset_name} dataset ---") 
    logging.info(f"Outputs saved in: {dataset_output_dir}") 


# ===============================
# --- Main Execution (CLI)     ---
# ===============================
if __name__ == "__main__": 
    try: 
        # Ensure a clean run: remove previous success flag if present
        if SUCCESS_FLAG_FILE.exists():
            SUCCESS_FLAG_FILE.unlink() 

        OUTPUT_DIR.mkdir(exist_ok=True) 
        logging.info("--- Starting JWST Adapted Processing Pipeline ---") 
        
        # Stage 1: Load + band‑split telemetry tracks
        low_band_tracks_raw, high_band_tracks_raw = process_chunks_incrementally(PROJECT_DIR, CHUNK_FILE_PATTERN, SPLIT_COLUMN_NAME, FREQ_LOW_RANGE, FREQ_HIGH_RANGE, COLUMNS_TO_DROP) 

        # Stage 2: Load anomaly incident catalog
        anomaly_df = load_jwst_anomalies(PROJECT_DIR, DRS_FILE) 
        
        # Stage 3: Compute global time origin using both telemetry and incidents
        all_raw_tracks = low_band_tracks_raw + high_band_tracks_raw 
        if not all_raw_tracks:
            raise ValueError("No tracks found in any frequency band.") 
        global_start_time = find_global_start_time(all_raw_tracks, anomaly_df, TIMESTAMP_COLUMN_NAME) 
        del all_raw_tracks  # Free memory proactively

        # Runtime configuration passed to downstream steps
        pipeline_config = { 
            "anomaly_df": anomaly_df, 
            "global_start_time": global_start_time, 
        } 

        # Stage 4: Process LOW‑BAND dataset
        logging.info("\n\n--- PROCESSING LOW-BAND DATASET ---") 
        process_dataset(tracks_list=low_band_tracks_raw, dataset_name="low_band", config=pipeline_config) 
        del low_band_tracks_raw  # Reduce memory footprint

        # Stage 5: Process HIGH‑BAND dataset
        logging.info("\n\n--- PROCESSING HIGH-BAND DATASET ---") 
        process_dataset(tracks_list=high_band_tracks_raw, dataset_name="high_band", config=pipeline_config) 
        del high_band_tracks_raw 

        logging.info("\n--- Adapted dual-pipeline execution finished successfully! ---") 
        SUCCESS_FLAG_FILE.touch()  # Mark success for orchestration systems

    except Exception as e: 
        # The full stack trace is emitted for debuggability; exit non‑zero for schedulers.
        logging.error("--- FATAL ERROR: The data processing pipeline failed ---", exc_info=True) 
        sys.exit(1)
