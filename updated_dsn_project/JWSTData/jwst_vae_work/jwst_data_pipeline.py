# jwst_data_pipeline_adapted.py

import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json
import sys
import os
import gc
import shutil
import random
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import gzip
import pyarrow.parquet as pq

# --- 1. CONFIGURATION ---
BASE_PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData")
CHUNK_FILE_PATTERN = "chunk_*_mon_JWST.pkl.gz"
DRS_FILE = "all_dr_data.csv"

OUTPUT_DIR = BASE_PROJECT_DIR / "processed_diffusion_style"
LOG_FILE = BASE_PROJECT_DIR / "pipeline_final_debug.log"
SUCCESS_FLAG_FILE = BASE_PROJECT_DIR / "PIPELINE_FINAL_SUCCESS.FLAG"

TIMESTAMP_COLUMN_NAME = 'RECEIVED_AT_TS'
ELEVATION_COLUMN_NAME = 'ELEVATION_ANGLE'
SPLIT_COLUMN_NAME = 'CARRIER_FREQ_MEASURED'
COLUMNS_TO_DROP = ['LNA_AMPLIFIER', 'UPLINK_CARRIER_BAND', 'DOWNLINK_CARRIER_BAND']
FREQ_LOW_RANGE = (2.1e9, 2.4e9)
FREQ_HIGH_RANGE = (25.0e9, 27.5e9)
MIN_TRACK_LENGTH = 2000
RANDOM_SEED = 42
MISSINGNESS_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)])

# --- 2. HELPER FUNCTIONS ---
def process_elevation_angle(track, min_length):
    track_id, df = track
    if ELEVATION_COLUMN_NAME not in df.columns:
        return [track]

    valid_mask = df[ELEVATION_COLUMN_NAME] != 90
    
    if valid_mask.all(): return [track]
    if not valid_mask.any(): return []

    block_ids = (valid_mask.astype(int).diff().ne(0)).cumsum()
    valid_blocks = df[valid_mask]
    
    new_tracks = []
    for block_num, group_df in valid_blocks.groupby(block_ids[valid_mask]):
        if len(group_df) >= min_length:
            new_track_id = f"{track_id}_p{block_num}"
            new_tracks.append((new_track_id, group_df.reset_index(drop=True)))
    return new_tracks

def process_chunks_incrementally(data_dir: Path, file_pattern: str, split_col: str, low_range: tuple, high_range: tuple, cols_to_drop: list):
    logging.info(f"Loading and splitting tracks from '{file_pattern}'...")
    processed_tracks = []
    chunk_files = sorted(data_dir.glob(file_pattern))
    if not chunk_files: raise FileNotFoundError(f"No chunk files found: {data_dir}/{file_pattern}")
    
    logging.info("Applying elevation angle trimming, splitting, and length filtering...")
    for file_path in tqdm(chunk_files, desc="Processing Chunks"):
        with gzip.open(file_path, 'rb') as f:
            chunk_data = pickle.load(f)
        for track_id, track_df in chunk_data:
            if not isinstance(track_df, pd.DataFrame): continue
            processed_sub_tracks = process_elevation_angle((track_id, track_df), MIN_TRACK_LENGTH)
            if processed_sub_tracks:
                processed_tracks.extend(processed_sub_tracks)

    logging.info(f"Initial processing complete. Found {len(processed_tracks)} valid tracks/sub-tracks.")

    low_band_tracks, high_band_tracks = [], []
    for track_id, df in processed_tracks:
        df = df.drop(columns=cols_to_drop, errors='ignore')
        if split_col not in df.columns or df[split_col].isnull().all(): continue
        low_mask = (df[split_col] >= low_range[0]) & (df[split_col] <= low_range[1])
        high_mask = (df[split_col] >= high_range[0]) & (df[split_col] <= high_range[1])
        if low_mask.any(): low_band_tracks.append((f"{track_id}_low", df[low_mask].copy().reset_index(drop=True)))
        if high_mask.any(): high_band_tracks.append((f"{track_id}_high", df[high_mask].copy().reset_index(drop=True)))
        
    return low_band_tracks, high_band_tracks

def load_jwst_anomalies(data_dir: Path, drs_fn: str) -> pd.DataFrame:
    drs_path = data_dir / drs_fn
    if not drs_path.exists(): raise FileNotFoundError(f"Anomaly file not found: {drs_path}")
    return pd.read_csv(drs_path)

def find_global_start_time(all_tracks: list, anomaly_df: pd.DataFrame, ts_col: str) -> float:
    min_track_time = min(track_df[ts_col].min() for _, track_df in all_tracks if not track_df.empty)
    anomaly_df['INCIDENT_START_TIME_DT'] = pd.to_numeric(anomaly_df['INCIDENT_START_TIME_DT'])
    anomaly_df['INCIDENT_END_TIME_DT'] = pd.to_numeric(anomaly_df['INCIDENT_END_TIME_DT'])
    min_anom_time = min(anomaly_df['INCIDENT_START_TIME_DT'].min(), anomaly_df['INCIDENT_END_TIME_DT'].min())
    return min(min_track_time, min_anom_time)

def identify_anomalous_tracks(all_tracks: list, anomaly_df: pd.DataFrame, ts_col: str, global_start: float) -> set:
    anomalous_original_ids = set()
    norm_anom_df = anomaly_df.copy()
    norm_anom_df['start_norm'] = norm_anom_df['INCIDENT_START_TIME_DT'] - global_start
    norm_anom_df['end_norm'] = norm_anom_df['INCIDENT_END_TIME_DT'] - global_start
    for track_id, track_df in all_tracks:
        original_id = str(track_id).split('_')[0].split('_p')[0]
        norm_track_times = track_df[ts_col] - global_start
        for _, anom_row in norm_anom_df.iterrows():
            if not norm_track_times[(norm_track_times >= anom_row['start_norm']) & (norm_track_times <= anom_row['end_norm'])].empty:
                anomalous_original_ids.add(original_id)
                break
    return anomalous_original_ids

def dynamic_split_tracks(all_tracks: list, anomalous_original_ids: set, train_ratio: float, val_ratio: float, seed: int):
    logging.info("Performing stratified group split on original track IDs...")
    random.seed(seed)
    all_original_ids = sorted(list(set(str(t[0]).split('_')[0].split('_p')[0] for t in all_tracks)))
    anomalous_ids_pool = sorted(list(anomalous_original_ids))
    normal_ids_pool = sorted(list(set(all_original_ids) - anomalous_original_ids))
    random.shuffle(anomalous_ids_pool); random.shuffle(normal_ids_pool)
    train_end_anom = int(train_ratio * len(anomalous_ids_pool)); val_end_anom = train_end_anom + int(val_ratio * len(anomalous_ids_pool))
    train_ids_anom = anomalous_ids_pool[:train_end_anom]; val_ids_anom = anomalous_ids_pool[train_end_anom:val_end_anom]; test_ids_anom = anomalous_ids_pool[val_end_anom:]
    train_end_norm = int(train_ratio * len(normal_ids_pool)); val_end_norm = train_end_norm + int(val_ratio * len(normal_ids_pool))
    train_ids_norm = normal_ids_pool[:train_end_norm]; val_ids_norm = normal_ids_pool[train_end_norm:val_end_norm]; test_ids_norm = normal_ids_pool[val_end_norm:]
    train_ids = set(train_ids_anom + train_ids_norm); val_ids = set(val_ids_anom + val_ids_norm); test_ids = set(test_ids_anom + test_ids_norm)
    logging.info(f"Split complete. Train tracks: {len(train_ids)}, Val tracks: {len(val_ids)}, Test tracks: {len(test_ids)}")
    return train_ids, val_ids, test_ids

def process_and_save_jwst_track(track_df, track_id, config, transformations, scaler, data_files_dir):
    df = track_df.copy()
    if TIMESTAMP_COLUMN_NAME not in df.columns: return
    df['seconds_since_start'] = df[TIMESTAMP_COLUMN_NAME] - config['global_start_time']
    label_mask = np.zeros((len(df), 1), dtype=np.float32)
    for _, anom_row in config['anomaly_df'].iterrows():
        start_norm = anom_row['INCIDENT_START_TIME_DT'] - config['global_start_time']
        end_norm = anom_row['INCIDENT_END_TIME_DT'] - config['global_start_time']
        indices = df.index[(df['seconds_since_start'] >= start_norm) & (df['seconds_since_start'] <= end_norm)]
        if not indices.empty: label_mask[indices.min():indices.max() + 1] = 1
    if transformations.get('high_card_features_to_drop'): df.drop(columns=transformations['high_card_features_to_drop'], inplace=True, errors='ignore')
    if transformations.get('columns_to_drop_missing'): df.drop(columns=list(set(transformations['columns_to_drop_missing'])), inplace=True, errors='ignore')
    categorical_cols_to_encode = [col for col in df.columns if df[col].dtype == 'object']
    if categorical_cols_to_encode: df = pd.get_dummies(df, columns=categorical_cols_to_encode, dummy_na=True)
    df = df.reindex(columns=transformations['all_columns_after_ohe'], fill_value=0.0); df.fillna(0.0, inplace=True)
    if transformations.get('columns_to_drop_zerovar'): df.drop(columns=transformations['columns_to_drop_zerovar'], inplace=True, errors='ignore')
    if transformations.get('columns_to_drop_correlated'): df.drop(columns=transformations['columns_to_drop_correlated'], inplace=True, errors='ignore')
    final_cols = transformations['final_numeric_columns']
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty and final_cols:
        df_final = numeric_df.reindex(columns=final_cols, fill_value=0.0)
        scaled_data = scaler.transform(df_final)
        processed_df = pd.DataFrame(scaled_data, columns=df_final.columns)
    else: processed_df = pd.DataFrame(columns=final_cols)
    track_filename = data_files_dir / f"{track_id}.parquet"
    labels_filename = data_files_dir / f"{track_id}_labels.npy"
    processed_df.to_parquet(track_filename); np.save(labels_filename, label_mask)

def process_dataset(tracks_list, dataset_name, config):
    logging.info(f"--- Processing {dataset_name} Dataset ---")
    dataset_output_dir = OUTPUT_DIR / dataset_name; data_files_dir = dataset_output_dir / "data_files"
    dataset_output_dir.mkdir(parents=True, exist_ok=True); data_files_dir.mkdir(exist_ok=True)
    if not tracks_list: logging.warning(f"No tracks provided for {dataset_name}. Skipping."); return

    final_anomalous_ids = identify_anomalous_tracks(tracks_list, config["anomaly_df"], TIMESTAMP_COLUMN_NAME, config["global_start_time"])
    train_ids, val_ids, test_ids = dynamic_split_tracks(tracks_list, final_anomalous_ids, TRAIN_RATIO, VALIDATION_RATIO, RANDOM_SEED)
    
    manifest = {"train": [], "val": [], "test": []}
    train_tracks_for_fitting = []
    for track_id, track_df in tracks_list:
        original_id = str(track_id).split('_')[0].split('_p')[0]
        entry = {"track_features": f"{track_id}.parquet", "track_labels": f"{track_id}_labels.npy"}
        if original_id in train_ids:
            manifest["train"].append(entry); train_tracks_for_fitting.append(track_df)
        elif original_id in val_ids: manifest["val"].append(entry)
        elif original_id in test_ids: manifest["test"].append(entry)

    logging.info("Analyzing training set to learn transformations...")
    train_sample_df = pd.concat(random.sample(train_tracks_for_fitting, min(len(train_tracks_for_fitting), 50)), ignore_index=True)
    transformations = {}; scaler = MinMaxScaler()
    categorical_cols = train_sample_df.select_dtypes(include=['object', 'category']).columns.tolist()
    transformations['high_card_features_to_drop'] = [col for col in categorical_cols if train_sample_df[col].nunique() > 50]
    missing_pct = train_sample_df.isna().sum() / len(train_sample_df)
    transformations['columns_to_drop_missing'] = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist()
    temp_df = train_sample_df.drop(columns=transformations.get('high_card_features_to_drop', []) + transformations.get('columns_to_drop_missing', []), errors='ignore')
    categorical_cols_to_encode = temp_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols_to_encode: temp_df = pd.get_dummies(temp_df, columns=categorical_cols_to_encode, dummy_na=True)
    transformations['all_columns_after_ohe'] = temp_df.columns.tolist(); temp_df.fillna(0.0, inplace=True)
    transformations['columns_to_drop_zerovar'] = temp_df.columns[temp_df.var() < 1e-6].tolist()
    corr_matrix = temp_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    transformations['columns_to_drop_correlated'] = [column for column in upper_tri.columns if any(upper_tri[column] > CORRELATION_THRESHOLD)]
    temp_df.drop(columns=transformations.get('columns_to_drop_correlated', []), inplace=True, errors='ignore')
    transformations['final_numeric_columns'] = temp_df.select_dtypes(include=np.number).columns.tolist()
    scaler.fit(temp_df[transformations['final_numeric_columns']])

    logging.info(f"--- Processing and saving {len(tracks_list)} total tracks ---")
    for track_id, track_df in tqdm(tracks_list, desc=f"Saving individual tracks"):
        process_and_save_jwst_track(track_df, track_id, config, transformations, scaler, data_files_dir)
            
    manifest_file = dataset_output_dir / "manifest.json"
    scaler_file = dataset_output_dir / "scaler.pkl"
    with open(manifest_file, 'w') as f: json.dump(manifest, f, indent=4)
    with open(scaler_file, 'wb') as f: pickle.dump(scaler, f)
    logging.info(f"Outputs saved in: {dataset_output_dir}")

if __name__ == "__main__":
    if SUCCESS_FLAG_FILE.exists(): SUCCESS_FLAG_FILE.unlink()
    logging.info("--- Starting JWST Data Processing Pipeline ---")
    low_band, high_band = process_chunks_incrementally(BASE_PROJECT_DIR, CHUNK_FILE_PATTERN, SPLIT_COLUMN_NAME, FREQ_LOW_RANGE, FREQ_HIGH_RANGE, COLUMNS_TO_DROP)
    anomaly_data = load_jwst_anomalies(BASE_PROJECT_DIR, DRS_FILE)
    all_tracks = low_band + high_band
    if not all_tracks: raise ValueError("No tracks found.")
    global_start = find_global_start_time(all_tracks, anomaly_data, TIMESTAMP_COLUMN_NAME)
    pipeline_config = {"anomaly_df": anomaly_data, "global_start_time": global_start}
    process_dataset(low_band, "low_band", pipeline_config)
    process_dataset(high_band, "high_band", pipeline_config)
    SUCCESS_FLAG_FILE.touch()
    logging.info("--- Pipeline finished successfully! ---")
