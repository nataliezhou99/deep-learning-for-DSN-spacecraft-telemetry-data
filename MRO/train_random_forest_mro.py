"""
Train Random Forest on MRO Features
----------------------------------------------------------------------------------
Purpose
    Train and evaluate a RandomForest classifier on per‑window features exported
    by the predictor (latent state + reconstruction error).

    The pipeline supports two splitting strategies and performs hyperparameter
    search with Optuna, class balancing with SMOTE, and point‑adjusted metrics
    to more fairly score event‑level anomaly detections.

High‑level flow
    1) Choose split strategy and assemble TRAIN/TEST windows from per‑track
       features written by feature extraction.
    2) Balance the training set via SMOTE (training split only).
    3) Tune RF hyperparameters with Optuna (3‑fold CV, objective = point‑adjusted F1).
    4) Train final RF with best params; evaluate on the hold‑out TEST set.
    5) Persist the model and save visual diagnostics (feature importances, ROC,
       PR, confusion matrix).

Inputs
    PROJECT_DIR/
      ├─ random_forest_boost_data/
      │   ├─ train/  (<track>_{features|labels}.npy)  # normal tracks (TRAIN‑only)
      │   └─ test/   (<track>_{features|labels}.npy)  # mixed normal/anomalous (TEST)

Outputs
    PROJECT_DIR/
      ├─ best_random_forest_model.pkl
      ├─ random_forest_evaluation.log
      └─ plots/
          ├─ feature_importance.png
          ├─ roc_curve.png
          ├─ pr_curve.png
          └─ confusion_matrix.png

Notes
    • Point‑adjusted scoring: if any positive prediction occurs anywhere within
      a true event window, we mark the *entire* event as detected when
      computing F1/precision/recall. This better reflects event‑level detection
      use‑cases compared to raw per‑timestep metrics.
    • No functional changes vs. the provided implementation—added
      documentation and inline comments for production clarity.
"""

import numpy as np
from pathlib import Path
import logging
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, make_scorer, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# --- CONFIGURATION ---
# =========================
PROJECT_DIR = Path("/home/nzhou/MRO")
RANDOM_FOREST_DATA_DIR = PROJECT_DIR / "random_forest_data"
RANDOM_FOREST_TRAIN_DIR_NORMAL = RANDOM_FOREST_DATA_DIR / "train"
RANDOM_FOREST_TEST_DIR_MIXED = RANDOM_FOREST_DATA_DIR / "test"
MODEL_SAVE_PATH = PROJECT_DIR / "best_random_forest_model.pkl"
EVAL_LOG_FILE = PROJECT_DIR / "random_forest_evaluation.log"
PLOTS_DIR = PROJECT_DIR / "plots"

# --- SPLITTING STRATEGY ---
# Options: 'within_track_chronological_split' or 'stratified_track_split'
SPLIT_STRATEGY = 'stratified_track_split' 

# --- Optuna Hyperparameters ---
N_TRIALS = 50

# Structured logging: file + stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(EVAL_LOG_FILE, mode='w'),
                              logging.StreamHandler(sys.stdout)])

# =============================
# --- Visualization Helpers ---
# =============================

def plot_feature_importance(model, save_path):
    """Plot and save the top‑20 RandomForest feature importances (by index).

    Note: Indices reflect the embedding+error feature space. If a feature map
    is available, extend to show names instead of indices.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    features = [f"Feature {i}" for i in indices]
    
    plt.figure(figsize=(10, 8))
    plt.title('Top 20 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Feature importance plot saved to {save_path}")


def plot_roc_curve(labels, scores, save_path):
    """Plot and save the ROC curve derived from probability scores."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"ROC curve plot saved to {save_path}")


def plot_pr_curve(labels, scores, save_path):
    """Plot and save the Precision‑Recall curve from probability scores."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Precision-Recall curve plot saved to {save_path}")


def plot_confusion_matrix(cm, save_path):
    """Plot and save a heatmap of the (point‑adjusted) confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Point-Adjusted Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {save_path}")


# ===================================
# --- EVALUATION & UTIL FUNCTIONS ---
# ===================================

def _get_events(y_true):
    """Return list of (start_idx, end_idx) contiguous positive segments in y."""
    events = []
    y_true_diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(y_true_diff == 1)[0]
    ends = np.where(y_true_diff == -1)[0]
    for start, end in zip(starts, ends):
        events.append((start, end))
    return events


def find_best_f1_point_adjusted(labels, scores):
    """Grid‑search threshold maximizing point‑adjusted F1.

    If no positive events exist in labels, default to threshold=0.5 and warn.
    Returns (best_f1, precision, recall, threshold, confusion_matrix, final_preds).
    """
    best_f1, best_threshold = -1, -1
    true_events = _get_events(labels)
    if not true_events:
        logging.warning("No anomalous events found in the evaluation set.")
        return 0, 0, 0, 0, confusion_matrix(labels, scores > 0.5), (scores > 0.5).astype(int)

    thresholds = np.percentile(scores, np.linspace(0, 100, 500))
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        adjusted_predictions = predictions.copy()
        # Inflate any prediction that hits an event to cover the full event span
        for start, end in true_events:
            if np.any(predictions[start:end] == 1):
                adjusted_predictions[start:end] = 1
        f1 = f1_score(labels, adjusted_predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    final_predictions = (scores > best_threshold).astype(int)
    final_adjusted_predictions = final_predictions.copy()
    for start, end in true_events:
        if np.any(final_predictions[start:end] == 1):
            final_adjusted_predictions[start:end] = 1
    precision = precision_score(labels, final_adjusted_predictions, zero_division=0)
    recall = recall_score(labels, final_adjusted_predictions, zero_division=0)
    cm = confusion_matrix(labels, final_adjusted_predictions)
    return best_f1, precision, recall, best_threshold, cm, final_adjusted_predictions
    

def load_data_from_ids(id_list, base_dir):
    """Load and concatenate (<track>_features.npy, <track>_labels.npy) for IDs."""
    all_features, all_labels = [], []
    for track_id in id_list:
        try:
            all_features.append(np.load(base_dir / f"{track_id}_features.npy"))
            all_labels.append(np.load(base_dir / f"{track_id}_labels.npy"))
        except FileNotFoundError:
            logging.warning(f"File for track ID {track_id} not found in {base_dir}. Skipping.")
    if not all_features:
        return np.array([]), np.array([])
    return np.concatenate(all_features), np.concatenate(all_labels)


def check_track_has_anomaly(track_id, base_dir):
    """Return True if the labels file for a track contains any positives."""
    try:
        labels = np.load(base_dir / f"{track_id}_labels.npy")
        return np.any(labels == 1)
    except FileNotFoundError:
        return False


def objective(trial, X, y):
    """Optuna objective: maximize mean CV point‑adjusted F1 across 3 folds."""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'random_state': 42,
        'n_jobs': -1
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        model = RandomForestClassifier(**param)
        model.fit(X_train_fold, y_train_fold)
        y_scores_fold = model.predict_proba(X_val_fold)[:, 1]
        f1, _, _, _, _ = find_best_f1_point_adjusted(y_val_fold, y_scores_fold)
        scores.append(f1)
    return np.mean(scores)


# ======================
# --- Main Execution ---
# ======================
if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)
    logging.info(f"--- Stage 2: Training Random Forest with '{SPLIT_STRATEGY}' and Optuna Tuning ---")
    
    # --- Assemble training/testing windows according to split strategy ---
    if SPLIT_STRATEGY == 'within_track_chronological_split':
        logging.info("Loading and splitting each track chronologically...")
        track_ids = sorted([f.stem.replace('_features', '') for f in RANDOM_FOREST_TEST_DIR_MIXED.glob('*_features.npy')])
        X_train_pieces, y_train_pieces, X_test_pieces, y_test_pieces = [], [], [], []

        # Normal tracks from TRAIN dir (all used for training)
        X_train_normal, y_train_normal = load_data_from_ids(
            sorted([f.stem.replace('_features', '') for f in RANDOM_FOREST_TRAIN_DIR_NORMAL.glob('*_features.npy')]),
            RANDOM_FOREST_TRAIN_DIR_NORMAL
        )
        X_train_pieces.append(X_train_normal)
        y_train_pieces.append(y_train_normal)

        # For each mixed track, first 70% → train, last 30% → test
        for track_id in track_ids:
            try:
                X_track = np.load(RANDOM_FOREST_TEST_DIR_MIXED / f"{track_id}_features.npy")
                y_track = np.load(RANDOM_FOREST_TEST_DIR_MIXED / f"{track_id}_labels.npy")
                if len(X_track) < 10:
                    continue
                split_point = int(0.7 * len(X_track))
                X_train_pieces.append(X_track[:split_point])
                y_train_pieces.append(y_track[:split_point])
                X_test_pieces.append(X_track[split_point:])
                y_test_pieces.append(y_track[split_point:])
            except FileNotFoundError:
                logging.warning(f"File for track ID {track_id} not found. Skipping.")
        X_train, y_train = np.concatenate(X_train_pieces), np.concatenate(y_train_pieces)
        X_test, y_test = np.concatenate(X_test_pieces), np.concatenate(y_test_pieces)

    elif SPLIT_STRATEGY == 'stratified_track_split':
        logging.info("Splitting tracks using stratification...")
        train_normal_track_ids = sorted([f.stem.replace('_features', '') for f in RANDOM_FOREST_TRAIN_DIR_NORMAL.glob('*_features.npy')])
        test_mixed_track_ids = sorted([f.stem.replace('_features', '') for f in RANDOM_FOREST_TEST_DIR_MIXED.glob('*_features.npy')])
        mixed_track_labels = [1 if check_track_has_anomaly(tid, RANDOM_FOREST_TEST_DIR_MIXED) else 0 for tid in test_mixed_track_ids]
        random_forest_train_ids_from_test, random_forest_test_ids = train_test_split(
            test_mixed_track_ids, test_size=0.4, random_state=42, stratify=mixed_track_labels
        )
        logging.info("Loading data for training and testing based on track IDs...")
        X_train_normal, y_train_normal = load_data_from_ids(train_normal_track_ids, RANDOM_FOREST_TRAIN_DIR_NORMAL)
        X_train_mixed, y_train_mixed = load_data_from_ids(random_forest_train_ids_from_test, RANDOM_FOREST_TEST_DIR_MIXED)
        X_test, y_test = load_data_from_ids(random_forest_test_ids, RANDOM_FOREST_TEST_DIR_MIXED)
        X_train = np.concatenate((X_train_normal, X_train_mixed))
        y_train = np.concatenate((y_train_normal, y_train_mixed))
    else:
        raise ValueError(f"Unknown SPLIT_STRATEGY: {SPLIT_STRATEGY}")

    logging.info(f"Final training data shape: {X_train.shape}, Final test data shape: {X_test.shape}")

    # --- Balance training set only ---
    if np.sum(y_train == 1) == 0:
        logging.error("The generated training set contains no anomalies.")
        sys.exit(1)
    logging.info("Balancing training data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"Resampled training data shape after SMOTE: {X_train_resampled.shape}")
    
    # --- Hyperparameter tuning with Optuna ---
    logging.info(f"Starting hyperparameter optimization with Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train_resampled, y_train_resampled), n_trials=N_TRIALS)
    
    logging.info(f"Optimization complete. Best point-adjusted F1 score found: {study.best_value:.4f}")
    logging.info(f"Best parameters found: {study.best_params}")
    
    # --- Train final model with best params ---
    logging.info("Training final RandomForest model with best parameters on resampled data...")
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train_resampled, y_train_resampled)
    logging.info("Model training complete.")
    
    # Persist model
    logging.info(f"Saving trained RandomForest model to {MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    
    # --- Evaluate on hold‑out TEST set ---
    logging.info("Evaluating best model on the hold-out test set...")
    test_scores = final_model.predict_proba(X_test)[:, 1]
    
    best_f1, prec, rec, f1_thresh, f1_cm, final_adjusted_preds = find_best_f1_point_adjusted(y_test, test_scores)
    try:
        auc_val = roc_auc_score(y_test, test_scores)
    except ValueError:
        auc_val = -1.0
    
    logging.info("\n--- Final RandomForest Evaluation Results ---")
    logging.info(f"AUC: {auc_val:.4f}")
    logging.info(f"Best Point-Adjusted F1 Score: {best_f1:.4f}")
    logging.info(f"  Point-Adjusted Precision: {prec:.4f}, Point-Adjusted Recall: {rec:.4f}, Threshold: {f1_thresh:.6f}")
    logging.info(f"Point-Adjusted Confusion Matrix:\n{f1_cm}")
    
    logging.info("\n--- Detailed Point-Adjusted Classification Report ---")
    report = classification_report(y_test, final_adjusted_preds, target_names=['Normal', 'Anomaly'])
    logging.info(f"\n{report}")
    
    logging.info("\n--- All Feature Importances (Most to Least Important) ---")
    feature_importances = final_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    for i in range(len(sorted_idx)):
        feature_index = sorted_idx[i]
        importance_value = feature_importances[feature_index]
        logging.info(f"Feature {feature_index}: Importance = {importance_value:.4f}")

    # --- Plots ---
    logging.info("\n--- Generating Evaluation Plots ---")
    plot_feature_importance(final_model, PLOTS_DIR / "feature_importance.png")
    plot_roc_curve(y_test, test_scores, PLOTS_DIR / "roc_curve.png")
    plot_pr_curve(y_test, test_scores, PLOTS_DIR / "pr_curve.png")
    plot_confusion_matrix(f1_cm, PLOTS_DIR / "confusion_matrix.png")
