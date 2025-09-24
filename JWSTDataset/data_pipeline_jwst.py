# jwst_data_pipeline.py 

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

# --- 1. CONFIGURATION --- 
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData") 
CHUNK_FILE_PATTERN = "chunk_*_mon_JWST.pkl.gz" 
DRS_FILE = "all_dr_data.csv" 
OUTPUT_DIR = PROJECT_DIR / "processed_diffusion_style" 
LOG_FILE = PROJECT_DIR / "pipeline_jwst_adapted_debug.log" 
SUCCESS_FLAG_FILE = PROJECT_DIR / "PIPELINE_JWST_ADAPTED_SUCCESS.FLAG" 
TIMESTAMP_COLUMN_NAME = 'RECEIVED_AT_TS' 
SPLIT_COLUMN_NAME = 'CARRIER_FREQ_MEASURED' 
COLUMNS_TO_DROP = ['LNA_AMPLIFIER', 'UPLINK_CARRIER_BAND', 'DOWNLINK_CARRIER_BAND'] 
FREQ_LOW_RANGE = (2.1e9, 2.4e9) 
FREQ_HIGH_RANGE = (25.0e9, 27.5e9) 
MIN_TRACK_LENGTH = 2000 
TARGET_TOTAL_ROWS = 4_000_000 
ANOMALOUS_RATIO = 0.06 
NORMAL_RATIO = 0.94 
RANDOM_SEED = 42 
MISSINGNESS_THRESHOLD = 0.01 
CORRELATION_THRESHOLD = 0.95 
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

# --- Setup Logging --- 
logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(levelname)s - %(message)s', 
                     handlers=[logging.FileHandler(LOG_FILE, mode='w'), 
                               logging.StreamHandler(sys.stdout)]) 

# --- Helper Functions --- 
def process_chunks_incrementally(data_dir: Path, file_pattern: str, split_col: str, low_range: tuple, high_range: tuple, cols_to_drop: list):
    logging.info(f"Loading and splitting tracks into low and high frequency bands from '{file_pattern}'...") 
    low_band_tracks, high_band_tracks = [], [] 
    chunk_files = sorted(data_dir.glob(file_pattern)) 
    if not chunk_files: raise FileNotFoundError(f"No chunk files found matching pattern '{file_pattern}' in {data_dir}") 
    for file_path in tqdm(chunk_files, desc="Processing Chunks"): 
        with gzip.open(file_path, 'rb') as f: chunk_data = pickle.load(f) 
        for track_id, track_df in chunk_data: 
            if not isinstance(track_df, pd.DataFrame): continue 
            df = track_df.drop(columns=cols_to_drop, errors='ignore') 
            if split_col not in df.columns or df[split_col].isnull().all(): continue 
            low_mask = (df[split_col] >= low_range[0]) & (df[split_col] <= low_range[1]) 
            high_mask = (df[split_col] >= high_range[0]) & (df[split_col] <= high_range[1]) 
            if low_mask.any(): low_band_tracks.append((f"{track_id}_low", df[low_mask].copy().reset_index(drop=True))) 
            if high_mask.any(): high_band_tracks.append((f"{track_id}_high", df[high_mask].copy().reset_index(drop=True))) 
    logging.info(f"Finished processing chunks. Low-band tracks: {len(low_band_tracks)}, High-band tracks: {len(high_band_tracks)}") 
    return low_band_tracks, high_band_tracks 

def load_jwst_anomalies(data_dir: Path, drs_fn: str) -> pd.DataFrame: 
    logging.info(f"Loading anomaly data from {drs_fn}...") 
    drs_path = data_dir / drs_fn 
    if not drs_path.exists(): raise FileNotFoundError(f"Anomaly file not found: {drs_path}") 
    anomaly_df = pd.read_csv(drs_path) 
    return anomaly_df 

def find_global_start_time(all_tracks: list, anomaly_df: pd.DataFrame, ts_col: str) -> float: 
    logging.info("Finding the global start time...") 
    min_track_time = min(track_df[ts_col].min() for _, track_df in all_tracks if not track_df.empty) 
    anomaly_df['INCIDENT_START_TIME_DT'] = pd.to_numeric(anomaly_df['INCIDENT_START_TIME_DT']) 
    anomaly_df['INCIDENT_END_TIME_DT'] = pd.to_numeric(anomaly_df['INCIDENT_END_TIME_DT']) 
    min_anom_time = min(anomaly_df['INCIDENT_START_TIME_DT'].min(), anomaly_df['INCIDENT_END_TIME_DT'].min()) 
    global_min = min(min_track_time, min_anom_time) 
    logging.info(f"Global start time (in seconds) found: {global_min}") 
    return global_min 

def filter_tracks_by_length(all_tracks: list, min_length: int) -> list: 
    logging.info(f"Filtering tracks to keep only those with length >= {min_length}...") 
    filtered_tracks = [track for track in all_tracks if len(track[1]) >= min_length] 
    logging.info(f"Length filtering complete. Kept {len(filtered_tracks)} out of {len(all_tracks)} tracks.") 
    return filtered_tracks 

def sample_stratified_by_row_count(anomalous_tracks: list, normal_tracks: list, target_anom_rows: int, target_normal_rows: int, seed: int):
    logging.info("Performing stratified sampling...") 
    random.seed(seed) 
    sampled_tracks = [] 
    random.shuffle(anomalous_tracks) 
    current_anom_rows = 0 
    for track in anomalous_tracks: 
        sampled_tracks.append(track) 
        current_anom_rows += len(track[1]) 
        if current_anom_rows >= target_anom_rows: break 
    logging.info(f"Sampled {len(sampled_tracks)} anomalous tracks with {current_anom_rows:,} rows (target: {target_anom_rows:,}).") 
    num_anom_sampled = len(sampled_tracks) 
    random.shuffle(normal_tracks) 
    current_normal_rows = 0 
    for track in normal_tracks: 
        sampled_tracks.append(track) 
        current_normal_rows += len(track[1]) 
        if current_normal_rows >= target_normal_rows: break 
    logging.info(f"Sampled {len(sampled_tracks) - num_anom_sampled} normal tracks with {current_normal_rows:,} rows.") 
    logging.info(f"Sampling complete. Total tracks: {len(sampled_tracks)}, Total rows: {current_anom_rows + current_normal_rows:,}") 
    return sampled_tracks 

def identify_anomalous_tracks(all_tracks: list, anomaly_df: pd.DataFrame, ts_col: str, global_start: float) -> set: 
    logging.info("Scanning all tracks to identify which ones contain anomalies...") 
    anomalous_original_ids = set() 
    norm_anom_df = anomaly_df.copy() 
    norm_anom_df['start_norm'] = norm_anom_df['INCIDENT_START_TIME_DT'] - global_start 
    norm_anom_df['end_norm'] = norm_anom_df['INCIDENT_END_TIME_DT'] - global_start 
    for track_id, track_df in tqdm(all_tracks, desc="Identifying Anomalous Tracks"): 
        norm_track_times = track_df[ts_col] - global_start 
        for _, anom_row in norm_anom_df.iterrows(): 
            if not norm_track_times[(norm_track_times >= anom_row['start_norm']) & (norm_track_times <= anom_row['end_norm'])].empty: 
                anomalous_original_ids.add(str(track_id).split('_')[0]) 
                break 
    logging.info(f"Identified {len(anomalous_original_ids)} original tracks with known anomalies.") 
    return anomalous_original_ids 

def dynamic_split_tracks(all_tracks: list, anomalous_split_ids: set, train_ratio: float, val_ratio: float, seed: int) -> tuple[list, list, list]: 
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
    test_ids = test_normal_ids + list(anomalous_split_ids) 
    logging.info(f"Split complete. Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)} (including {len(anomalous_split_ids)} anomalous).") 
    return train_ids, val_ids, test_ids 

def process_and_save_jwst_track(track_df, track_id, config, transformations, scaler, data_files_dir): 
    df = track_df.copy() 
    df['seconds_since_start'] = df[TIMESTAMP_COLUMN_NAME] - config['global_start_time'] 
    dt_series = pd.to_datetime(df['seconds_since_start'], unit='s') 
    df['hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24.0) 
    df['hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24.0) 
    df.drop(columns=[TIMESTAMP_COLUMN_NAME], inplace=True) 

    label_mask = np.zeros((len(df), 1), dtype=np.float32) 
    for _, anom_row in config['anomaly_df'].iterrows(): 
        start_norm = anom_row['INCIDENT_START_TIME_DT'] - config['global_start_time'] 
        end_norm = anom_row['INCIDENT_END_TIME_DT'] - config['global_start_time'] 
        indices = df.index[(df['seconds_since_start'] >= start_norm) & (df['seconds_since_start'] <= end_norm)] 
        if not indices.empty: 
            label_mask[indices.min():indices.max() + 1] = 1 

    if transformations.get('high_card_features_to_drop'): df.drop(columns=transformations['high_card_features_to_drop'], inplace=True, errors='ignore') 
    if transformations.get('columns_to_drop_missing'): df.drop(columns=list(set(transformations['columns_to_drop_missing'])), inplace=True, errors='ignore') 
    
    categorical_cols_to_encode = [col for col in df.columns if df[col].dtype == 'object'] 
    if categorical_cols_to_encode: df = pd.get_dummies(df, columns=categorical_cols_to_encode, dummy_na=True) 

    df = df.reindex(columns=transformations['all_columns_after_ohe'], fill_value=0.0) 
    df.fillna(0.0, inplace=True) 
    
    if transformations.get('columns_to_drop_zerovar'): df.drop(columns=transformations['columns_to_drop_zerovar'], inplace=True, errors='ignore') 
    if transformations.get('columns_to_drop_correlated'): df.drop(columns=transformations['columns_to_drop_correlated'], inplace=True, errors='ignore') 

    final_cols = transformations['final_numeric_columns'] 
    if not df.empty and final_cols: 
        df_final = df[final_cols] 
        scaled_data = scaler.transform(df_final) 
        processed_df = pd.DataFrame(scaled_data, columns=df_final.columns) 
    else: 
        processed_df = pd.DataFrame(columns=final_cols) 

    track_filename = data_files_dir / f"track_{track_id}.parquet" 
    labels_filename = data_files_dir / f"track_{track_id}_labels.npy" 
    
    processed_df.to_parquet(track_filename) 
    np.save(labels_filename, label_mask) 
    
    return track_filename.name, labels_filename.name 

def process_dataset(tracks_list, dataset_name, config): 
    logging.info(f"--- Processing {dataset_name} Dataset (Diffusion-Style Output) ---") 
    dataset_output_dir = OUTPUT_DIR / dataset_name 
    data_files_dir = dataset_output_dir / "data_files" 
    dataset_output_dir.mkdir(parents=True, exist_ok=True) 
    data_files_dir.mkdir(exist_ok=True) 

    tracks_filtered = filter_tracks_by_length(tracks_list, MIN_TRACK_LENGTH) 
    if not tracks_filtered: 
        logging.warning(f"No tracks left for {dataset_name} after length filtering. Skipping.") 
        return 

    anomalous_original_ids = identify_anomalous_tracks(tracks_filtered, config["anomaly_df"], TIMESTAMP_COLUMN_NAME, config["global_start_time"]) 
    anomalous_pool = [track for track in tracks_filtered if str(track[0]).split('_')[0] in anomalous_original_ids] 
    normal_pool = [track for track in tracks_filtered if str(track[0]).split('_')[0] not in anomalous_original_ids] 

    tracks_sampled = sample_stratified_by_row_count(anomalous_pool, normal_pool, int(TARGET_TOTAL_ROWS * ANOMALOUS_RATIO), int(TARGET_TOTAL_ROWS * NORMAL_RATIO), RANDOM_SEED) 
    if not tracks_sampled: 
        logging.warning(f"No tracks left for {dataset_name} after sampling. Skipping.") 
        return 

    sampled_track_ids = {track[0] for track in tracks_sampled} 
    final_anomalous_split_ids = {tid for tid in sampled_track_ids if str(tid).split('_')[0] in anomalous_original_ids} 
    
    train_ids, val_ids, test_ids = dynamic_split_tracks(tracks_sampled, final_anomalous_split_ids, TRAIN_RATIO, VALIDATION_RATIO, RANDOM_SEED) 
    
    tracks_map = {track_id: df for track_id, df in tracks_sampled} 
    
    logging.info("Analyzing training set to learn transformations...") 
    train_sample_ids = random.sample(train_ids, min(len(train_ids), 50))  
    train_dfs = [tracks_map[tid] for tid in train_sample_ids] 
    full_train_df = pd.concat(train_dfs, ignore_index=True) 

    transformations = {} 
    full_train_df['seconds_since_start'] = full_train_df[TIMESTAMP_COLUMN_NAME] - config['global_start_time'] 
    missing_pct = full_train_df.isna().sum() / len(full_train_df) 
    transformations['columns_to_drop_missing'] = missing_pct[missing_pct > MISSINGNESS_THRESHOLD].index.tolist() 
    categorical_cols = [col for col in full_train_df.columns if full_train_df[col].dtype == 'object'] 
    if categorical_cols: 
        cardinality_threshold = int(np.percentile(full_train_df[categorical_cols].nunique(), 90)) 
        transformations['high_card_features_to_drop'] = [col for col in categorical_cols if full_train_df[col].nunique() > cardinality_threshold] 
    temp_df = full_train_df.drop(columns=transformations.get('high_card_features_to_drop', []) + transformations.get('columns_to_drop_missing', []), errors='ignore') 
    categorical_cols_to_encode = [col for col in temp_df.columns if temp_df[col].dtype == 'object'] 
    if categorical_cols_to_encode: temp_df = pd.get_dummies(temp_df, columns=categorical_cols_to_encode, dummy_na=True) 
    transformations['all_columns_after_ohe'] = temp_df.columns.tolist() 
    temp_df.fillna(0.0, inplace=True) 
    transformations['columns_to_drop_zerovar'] = temp_df.columns[temp_df.var() == 0].tolist() 
    if transformations.get('columns_to_drop_zerovar'): temp_df.drop(columns=transformations['columns_to_drop_zerovar'], inplace=True, errors='ignore') 
    corr_matrix = temp_df.corr().abs() 
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
    transformations['columns_to_drop_correlated'] = [column for column in upper_tri.columns if any(upper_tri[column] > CORRELATION_THRESHOLD)] 
    if transformations.get('columns_to_drop_correlated'): temp_df.drop(columns=transformations['columns_to_drop_correlated'], inplace=True, errors='ignore') 
    
    transformations['final_numeric_columns'] = temp_df.select_dtypes(include=np.number).columns.tolist() 

    scaler = MinMaxScaler() 
    scaler.fit(temp_df[transformations['final_numeric_columns']]) 
    logging.info("Transformation analysis complete. Scaler has been fitted.") 

    manifest = {"train": [], "val": [], "test": []} 
    all_ids_map = {"train": train_ids, "val": val_ids, "test": test_ids} 
    
    for set_name, id_list in all_ids_map.items(): 
        logging.info(f"--- Processing {len(id_list)} tracks for {set_name.upper()} set ---") 
        for track_id in tqdm(id_list, desc=f"Processing {set_name.upper()} tracks"): 
            track_df = tracks_map[track_id] 
            f, l = process_and_save_jwst_track(track_df, track_id, config, transformations, scaler, data_files_dir) 
            manifest[set_name].append({"track_features": f, "track_labels": l}) 
            
    manifest_file = dataset_output_dir / "manifest.json" 
    scaler_file = dataset_output_dir / "scaler.pkl" 
    
    with open(manifest_file, 'w') as f: json.dump(manifest, f, indent=4) 
    with open(scaler_file, 'wb') as f: pickle.dump(scaler, f) 
        
    logging.info(f"--- Successfully finished processing for {dataset_name} dataset ---") 
    logging.info(f"Outputs saved in: {dataset_output_dir}") 

# --- Main Execution Block --- 
if __name__ == "__main__": 
    try: 
        if SUCCESS_FLAG_FILE.exists(): SUCCESS_FLAG_FILE.unlink() 
        OUTPUT_DIR.mkdir(exist_ok=True) 
        logging.info("--- Starting JWST Adapted Processing Pipeline ---") 
        
        low_band_tracks_raw, high_band_tracks_raw = process_chunks_incrementally(PROJECT_DIR, CHUNK_FILE_PATTERN, SPLIT_COLUMN_NAME, FREQ_LOW_RANGE, FREQ_HIGH_RANGE, COLUMNS_TO_DROP) 
        anomaly_df = load_jwst_anomalies(PROJECT_DIR, DRS_FILE) 
        
        all_raw_tracks = low_band_tracks_raw + high_band_tracks_raw 
        if not all_raw_tracks: raise ValueError("No tracks found in any frequency band.") 
        global_start_time = find_global_start_time(all_raw_tracks, anomaly_df, TIMESTAMP_COLUMN_NAME) 
        del all_raw_tracks 

        pipeline_config = { 
            "anomaly_df": anomaly_df, 
            "global_start_time": global_start_time, 
        } 

        logging.info("\n\n--- PROCESSING LOW-BAND DATASET ---") 
        process_dataset(tracks_list=low_band_tracks_raw, dataset_name="low_band", config=pipeline_config) 
        del low_band_tracks_raw 

        logging.info("\n\n--- PROCESSING HIGH-BAND DATASET ---") 
        process_dataset(tracks_list=high_band_tracks_raw, dataset_name="high_band", config=pipeline_config) 
        del high_band_tracks_raw 

        logging.info("\n--- Adapted dual-pipeline execution finished successfully! ---") 
        SUCCESS_FLAG_FILE.touch() 

    except Exception as e: 
        logging.error("--- FATAL ERROR: The data processing pipeline failed ---", exc_info=True) 
        sys.exit(1)
