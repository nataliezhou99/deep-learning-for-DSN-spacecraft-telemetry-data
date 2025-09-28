"""
Train Random Forest on JWST Encoder Features
-----------------------------------------------------------------------
Purpose
    Load per‑window latent features exported by the hybrid autoencoder encoder
    (see hybrid_autoencoder_extract_features_jwst.py) and train a classical
    RandomForest classifier to predict anomaly presence. Evaluate using
    *point‑adjusted* F1 (credit any detection within a true anomaly interval as
    catching the whole interval), plus AUC and a confusion matrix.

Splitting strategies
    • stratified_track_split
        Split *tracks* (not windows) into train/eval with stratification by
        presence of anomalies, so both sets contain a mix of anomalous & normal
        tracks. Avoids temporal leakage within a track.
    • within_track_chronological_split
        For each track, take first 70% windows as train and last 30% as eval.
        Preserves chronology but risks leakage if windows are highly correlated.

Inputs
    • PROJECT_DIR/processed_diffusion_style/<dataset>/manifest.json
    • PROJECT_DIR/<dataset>/xgboost_features_per_track/<track>_{features,labels}.npy

Outputs
    • PROJECT_DIR/<dataset>/best_rf_model_<dataset>_<strategy>.pkl
    • PROJECT_DIR/<dataset>/rf_evaluation_<dataset>_<strategy>.log (this file)

Notes
    • SMOTE is applied only to the training split to address class imbalance.
    • No logic changes versus provided code. Added documentation and comments
      for production clarity and maintainability.
"""

import numpy as np
from pathlib import Path
import logging
import sys
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

# =========================
# --- CONFIGURATION ---
# =========================
DATASET_TO_USE = "low_band"

# Choose your splitting strategy here
# Options: "stratified_track_split", "within_track_chronological_split"
SPLITTING_STRATEGY = "stratified_track_split" 

PROJECT_DIR = Path("/home/nzhou/JWST")
BASE_INPUT_DIR = PROJECT_DIR / "processed_diffusion_style"
BASE_OUTPUT_DIR = PROJECT_DIR
INPUT_DATASET_DIR = BASE_INPUT_DIR / DATASET_TO_USE
OUTPUT_SUBDIR = BASE_OUTPUT_DIR / DATASET_TO_USE
XGB_FEATURES_DIR = OUTPUT_SUBDIR / "xgboost_features_per_track"
MANIFEST_PATH = INPUT_DATASET_DIR / "manifest.json"

# Output file paths are dynamic and named for Random Forest (rf)
MODEL_SAVE_PATH = OUTPUT_SUBDIR / f"best_rf_model_{DATASET_TO_USE}_{SPLITTING_STRATEGY}.pkl"
EVAL_LOG_FILE = OUTPUT_SUBDIR / f"rf_evaluation_{DATASET_TO_USE}_{SPLITTING_STRATEGY}.log"


# Structured logging: file + stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(EVAL_LOG_FILE, mode='w'),
                              logging.StreamHandler(sys.stdout)])

# ======================================
# --- EVALUATION & DATA LOADING UTILS ---
# ======================================

def _get_events(y_true):
    """Return list of (start_idx, end_idx) contiguous positive segments in y.

    Used for point‑adjusted scoring: if any prediction is positive within a
    true event span, credit the entire span as detected.
    """
    events = []
    y_true_diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(y_true_diff == 1)[0]
    ends = np.where(y_true_diff == -1)[0]
    for start, end in zip(starts, ends):
        events.append((start, end))
    return events


def find_best_f1_point_adjusted(labels, scores):
    """Grid‑search a threshold to maximize *point‑adjusted* F1.

    If labels are single‑class, returns zeros and an empty confusion matrix.
    """
    if len(np.unique(labels)) < 2:
        logging.warning("Evaluation set has only one class. Cannot calculate F1 score. Returning 0.")
        return 0.0, 0.0, 0.0, 0.0, np.array([])
        
    best_f1, best_threshold = -1, -1
    true_events = _get_events(labels)
    
    # Candidate thresholds across score distribution
    thresholds = np.percentile(scores, np.linspace(0, 100, 500))
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        adjusted_predictions = predictions.copy()
        # Inflate predictions to cover full true event spans if any hit occurs
        for start, end in true_events:
            if np.any(predictions[start:end] == 1):
                adjusted_predictions[start:end] = 1
        f1 = f1_score(labels, adjusted_predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    # Compute metrics at best threshold
    final_predictions = (scores > best_threshold).astype(int)
    final_adjusted_predictions = final_predictions.copy()
    for start, end in true_events:
        if np.any(final_predictions[start:end] == 1):
            final_adjusted_predictions[start:end] = 1
    precision = precision_score(labels, final_adjusted_predictions, zero_division=0)
    recall = recall_score(labels, final_adjusted_predictions, zero_division=0)
    cm = confusion_matrix(labels, final_adjusted_predictions)
    return best_f1, precision, recall, best_threshold, cm


def load_data_from_tracks(track_list, desc):
    """Load and concatenate per‑track feature/label arrays for a list of tracks."""
    all_features = []
    all_labels = []
    for track_info in tqdm(track_list, desc=desc):
        track_id = Path(track_info['track_features']).stem
        feature_file = XGB_FEATURES_DIR / f"{track_id}_features.npy"
        label_file = XGB_FEATURES_DIR / f"{track_id}_labels.npy"
        
        if feature_file.exists() and label_file.exists():
            all_features.append(np.load(feature_file))
            all_labels.append(np.load(label_file))
    
    if not all_features:
        return np.array([]), np.array([])
        
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# ======================
# --- MAIN EXECUTION ---
# ======================
if __name__ == "__main__":
    logging.info(f"--- Stage 2: Training RandomForest Classifier for '{DATASET_TO_USE}' ---")
    logging.info(f"--- Using Splitting Strategy: {SPLITTING_STRATEGY} ---")
    
    logging.info(f"Loading track list from manifest: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    test_tracks = manifest['test']

    # Choose split strategy
    if SPLITTING_STRATEGY == "stratified_track_split":
        logging.info("Identifying anomalous and normal tracks for stratification...")
        anomalous_tracks = []
        normal_tracks = []
        for track_info in tqdm(test_tracks, desc="Scanning track labels"):
            track_id = Path(track_info['track_features']).stem
            label_file = XGB_FEATURES_DIR / f"{track_id}_labels.npy"
            if label_file.exists():
                labels = np.load(label_file)
                if np.sum(labels) > 0:
                    anomalous_tracks.append(track_info)
                else:
                    normal_tracks.append(track_info)
        
        logging.info(f"Found {len(anomalous_tracks)} anomalous tracks and {len(normal_tracks)} normal tracks.")
        train_anom_tracks, eval_anom_tracks = train_test_split(anomalous_tracks, test_size=0.30, random_state=42)
        train_norm_tracks, eval_norm_tracks = train_test_split(normal_tracks, test_size=0.30, random_state=42)
        train_tracks = train_anom_tracks + train_norm_tracks
        eval_tracks = eval_anom_tracks + eval_norm_tracks
        
        logging.info(f"Final training set: {len(train_anom_tracks)} anomalous tracks, {len(train_norm_tracks)} normal tracks.")
        logging.info(f"Final evaluation set: {len(eval_anom_tracks)} anomalous tracks, {len(eval_norm_tracks)} normal tracks.")

        X_xgb_train, y_xgb_train = load_data_from_tracks(train_tracks, desc="Loading training track data")
        X_xgb_eval, y_xgb_eval = load_data_from_tracks(eval_tracks, desc="Loading evaluation track data")

    elif SPLITTING_STRATEGY == "within_track_chronological_split":
        logging.info("Performing within-track chronological split (70% train, 30% eval for each track)...")
        X_train_list, y_train_list = [], []
        X_eval_list, y_eval_list = [], []
        for track_info in tqdm(test_tracks, desc="Splitting tracks chronologically"):
            track_id = Path(track_info['track_features']).stem
            feature_file = XGB_FEATURES_DIR / f"{track_id}_features.npy"
            label_file = XGB_FEATURES_DIR / f"{track_id}_labels.npy"
            if feature_file.exists() and label_file.exists():
                X_track, y_track = np.load(feature_file), np.load(label_file)
                if len(X_track) == 0: continue
                split_idx = int(0.7 * len(X_track))
                X_train_list.append(X_track[:split_idx])
                y_train_list.append(y_track[:split_idx])
                X_eval_list.append(X_track[split_idx:])
                y_eval_list.append(y_track[split_idx:])
        
        X_xgb_train = np.concatenate(X_train_list, axis=0)
        y_xgb_train = np.concatenate(y_train_list, axis=0)
        X_xgb_eval = np.concatenate(X_eval_list, axis=0)
        y_xgb_eval = np.concatenate(y_eval_list, axis=0)
    
    else:
        raise ValueError(f"Unknown SPLITTING_STRATEGY: '{SPLITTING_STRATEGY}'. Please choose a valid option.")

    # Sanity checks
    if X_xgb_train.size == 0 or X_xgb_eval.size == 0:
        raise ValueError("Data loading resulted in empty arrays. Check if feature files were generated.")
    
    logging.info(f"Clean training data shape: {X_xgb_train.shape}")
    logging.info(f"Clean hold-out evaluation data shape: {X_xgb_eval.shape}")

    # Handle class imbalance on training set only
    logging.info("Applying SMOTE to the training data to balance classes...")
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_xgb_train, y_xgb_train)
    logging.info(f"Resampled (balanced) training data shape: {X_train_resampled.shape}")
    
    # Model hyperparameters (first reasonable pass; tune separately if needed)
    logging.info("Using a predefined set of reasonable hyperparameters.")
    model_params = {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Train and persist model
    logging.info(f"Training final RandomForest model with parameters: {model_params}")
    final_model = RandomForestClassifier(**model_params)
    final_model.fit(X_train_resampled, y_train_resampled)

    logging.info(f"Saving trained RandomForest model to {MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    
    # Evaluation under point‑adjusted F1
    logging.info("Evaluating final model on the hold-out evaluation set...")
    test_scores = final_model.predict_proba(X_xgb_eval)[:, 1]
    
    best_f1, prec, rec, f1_thresh, f1_cm = find_best_f1_point_adjusted(y_xgb_eval, test_scores)
    try: 
        auc = roc_auc_score(y_xgb_eval, test_scores)
    except ValueError: 
        auc = -1.0
    
    logging.info("\n--- Final RandomForest Evaluation Results ---")
    logging.info(f"AUC: {auc:.4f}")
    logging.info(f"Best Point-Adjusted F1 Score: {best_f1:.4f}")
    logging.info(f"  Point-Adjusted Precision: {prec:.4f}, Point-Adjusted Recall: {rec:.4f}, Threshold: {f1_thresh:.6f}")
    logging.info(f"Point-Adjusted Confusion Matrix:\n{f1_cm}")
    
    # Feature importances (indices only; mapping to column names occurs upstream)
    logging.info("\n--- Top 10 Most Important Features ---")
    feature_importances = final_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    for i in range(min(10, len(sorted_idx))):
        logging.info(f"Feature {sorted_idx[i]}: Importance = {feature_importances[sorted_idx[i]]:.4f}")
