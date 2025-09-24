import numpy as np
from pathlib import Path
import logging
import sys
import pickle
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData/jwst_vae_work")
XGB_DATA_DIR = PROJECT_DIR / "xgboost_data"
XGB_TRAIN_DIR_NORMAL = XGB_DATA_DIR / "train"
XGB_TEST_DIR_MIXED = XGB_DATA_DIR / "test"

# --- Switch between 'stratified_track_split' and 'within_track_chronological_split' ---
SPLIT_STRATEGY = 'stratified_track_split' 
N_TRIALS = 20

# --- ⬇️ MODIFIED: DYNAMIC FILENAME GENERATION WITH SUFFIX ⬇️ ---
MODEL_SAVE_PATH = PROJECT_DIR / f"best_random_forest_model_{SPLIT_STRATEGY}_hyp_opt.pkl"
EVAL_LOG_FILE = PROJECT_DIR / f"random_forest_evaluation_{SPLIT_STRATEGY}_hyp_opt.log"
# --- ⬆️ END OF MODIFICATION ⬆️ ---

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(EVAL_LOG_FILE, mode='w'),
                              logging.StreamHandler(sys.stdout)])

# --- EVALUATION FUNCTIONS (Point-Adjust) ---
def _get_events(y_true):
    events = []
    y_true_diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(y_true_diff == 1)[0]
    ends = np.where(y_true_diff == -1)[0]
    for start, end in zip(starts, ends):
        events.append((start, end))
    return events

def find_best_f1_point_adjusted(labels, scores):
    best_f1, best_threshold = -1, -1
    true_events = _get_events(labels)
    if not true_events:
        if np.any(labels): logging.warning("No anomalous events found in the evaluation set.")
        cm = confusion_matrix(labels, scores > 0.5)
        return 0, 0, 0, 0, cm, (scores > 0.5).astype(int)

    thresholds = np.percentile(scores, np.linspace(0, 100, 500))
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        adjusted_predictions = predictions.copy()
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
    all_features, all_labels = [], []
    for track_id in id_list:
        try:
            all_features.append(np.load(base_dir / f"{track_id}_features.npy"))
            all_labels.append(np.load(base_dir / f"{track_id}_labels.npy"))
        except FileNotFoundError:
            logging.warning(f"File for track ID {track_id} not found in {base_dir}. Skipping.")
    if not all_features: return np.array([]), np.array([])
    return np.concatenate(all_features), np.concatenate(all_labels)

def check_track_has_anomaly(track_id, base_dir):
    try:
        labels = np.load(base_dir / f"{track_id}_labels.npy")
        return np.any(labels == 1)
    except FileNotFoundError:
        return False

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    val_scores = model.predict_proba(X_val)[:, 1]
    f1, _, _, _, _, _ = find_best_f1_point_adjusted(y_val, val_scores)
    return f1

# --- PLOTTING FUNCTIONS ---
def plot_feature_importance(model, save_path):
    logging.info("Generating Feature Importance plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Feature Importance plot saved to {save_path}")

def plot_roc_curve(y_true, y_scores, auc, save_path):
    logging.info("Generating ROC Curve plot...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
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
    logging.info(f"ROC Curve plot saved to {save_path}")

def plot_precision_recall_curve(y_true, y_scores, save_path):
    logging.info("Generating Precision-Recall Curve plot...")
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Precision-Recall Curve plot saved to {save_path}")

def plot_confusion_matrix(cm, save_path):
    logging.info("Generating Confusion Matrix plot...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix (Point-Adjusted)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Confusion Matrix plot saved to {save_path}")


if __name__ == "__main__":
    logging.info(f"--- Stage 2: Training Random Forest with '{SPLIT_STRATEGY}' and Optuna Tuning ---")

    if SPLIT_STRATEGY == 'within_track_chronological_split':
        logging.info("Loading and splitting each track chronologically...")
        track_ids = sorted([f.stem.replace('_features', '') for f in XGB_TEST_DIR_MIXED.glob('*_features.npy')])
        
        X_train_pieces, y_train_pieces = [], []
        X_test_pieces, y_test_pieces = [], []
        
        X_train_normal, y_train_normal = load_data_from_ids(
            sorted([f.stem.replace('_features', '') for f in XGB_TRAIN_DIR_NORMAL.glob('*_features.npy')]),
            XGB_TRAIN_DIR_NORMAL
        )
        X_train_pieces.append(X_train_normal)
        y_train_pieces.append(y_train_normal)

        for track_id in track_ids:
            try:
                X_track = np.load(XGB_TEST_DIR_MIXED / f"{track_id}_features.npy")
                y_track = np.load(XGB_TEST_DIR_MIXED / f"{track_id}_labels.npy")
                if len(X_track) < 10: continue
                split_point = int(0.7 * len(X_track))
                X_train_pieces.append(X_track[:split_point])
                y_train_pieces.append(y_track[:split_point])
                X_test_pieces.append(X_track[split_point:])
                y_test_pieces.append(y_track[split_point:])
            except FileNotFoundError:
                logging.warning(f"File for track ID {track_id} not found. Skipping.")

        X_train = np.concatenate(X_train_pieces)
        y_train = np.concatenate(y_train_pieces)
        X_test = np.concatenate(X_test_pieces)
        y_test = np.concatenate(y_test_pieces)

    elif SPLIT_STRATEGY == 'stratified_track_split':
        logging.info("Splitting tracks using stratification...")
        train_normal_track_ids = sorted([f.stem.replace('_features', '') for f in XGB_TRAIN_DIR_NORMAL.glob('*_features.npy')])
        test_mixed_track_ids = sorted([f.stem.replace('_features', '') for f in XGB_TEST_DIR_MIXED.glob('*_features.npy')])
        
        mixed_track_labels = [1 if check_track_has_anomaly(tid, XGB_TEST_DIR_MIXED) else 0 for tid in test_mixed_track_ids]

        # Split all mixed tracks into a main training pool and a final hold-out test set
        train_mixed_ids_pool, test_ids = train_test_split(
            test_mixed_track_ids, test_size=0.4, random_state=42, stratify=mixed_track_labels
        )
        
        # --- MODIFIED SECTION: Create a robust validation set for Optuna ---
        logging.info("Splitting training track pool into Optuna-train and Optuna-validation sets...")

        # Further split the training pool of mixed tracks into a smaller set for Optuna training
        # and a set of whole tracks for Optuna validation.
        train_mixed_ids_labels = [1 if check_track_has_anomaly(tid, XGB_TEST_DIR_MIXED) else 0 for tid in train_mixed_ids_pool]
        optuna_train_mixed_ids, optuna_val_mixed_ids = train_test_split(
            train_mixed_ids_pool, test_size=0.25, random_state=42, stratify=train_mixed_ids_labels
        )

        # Also split the normal tracks, holding some out for the Optuna validation set.
        optuna_train_normal_ids, optuna_val_normal_ids = train_test_split(
            train_normal_track_ids, test_size=0.25, random_state=42
        )
        
        # Load data specifically for the Optuna run based on the new track ID splits
        logging.info("Loading data for Optuna hyperparameter search...")
        X_train_opt_normal, y_train_opt_normal = load_data_from_ids(optuna_train_normal_ids, XGB_TRAIN_DIR_NORMAL)
        X_train_opt_mixed, y_train_opt_mixed = load_data_from_ids(optuna_train_mixed_ids, XGB_TEST_DIR_MIXED)
        X_train_opt = np.concatenate((X_train_opt_normal, X_train_opt_mixed))
        y_train_opt = np.concatenate((y_train_opt_normal, y_train_opt_mixed))

        X_val_opt_normal, y_val_opt_normal = load_data_from_ids(optuna_val_normal_ids, XGB_TRAIN_DIR_NORMAL)
        X_val_opt_mixed, y_val_opt_mixed = load_data_from_ids(optuna_val_mixed_ids, XGB_TEST_DIR_MIXED)
        X_val_opt = np.concatenate((X_val_opt_normal, X_val_opt_mixed))
        y_val_opt = np.concatenate((y_val_opt_normal, y_val_opt_mixed))
        
        logging.info(f"Optuna train set shape: {X_train_opt.shape}, Optuna validation set shape: {X_val_opt.shape}")
        
        # --- END OF MODIFIED SECTION ---

        # Load the FULL training and test data for the final model
        logging.info("Loading full training and test data for final model...")
        X_train_normal_full, y_train_normal_full = load_data_from_ids(train_normal_track_ids, XGB_TRAIN_DIR_NORMAL)
        X_train_mixed_full, y_train_mixed_full = load_data_from_ids(train_mixed_ids_pool, XGB_TEST_DIR_MIXED)
        X_train = np.concatenate((X_train_normal_full, X_train_mixed_full))
        y_train = np.concatenate((y_train_normal_full, y_train_mixed_full))
        X_test, y_test = load_data_from_ids(test_ids, XGB_TEST_DIR_MIXED)

    else:
        raise ValueError(f"Unknown SPLIT_STRATEGY: {SPLIT_STRATEGY}")

    logging.info(f"Total training data shape: {X_train.shape}, Total test data shape: {X_test.shape}")
    if np.sum(y_train == 1) == 0:
        logging.error("The generated training set contains no anomalies.")
        sys.exit(1)

    logging.info(f"Starting Optuna hyperparameter search for {N_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt), n_trials=N_TRIALS)

    logging.info("Optuna search complete.")
    logging.info(f"Best trial F1 score: {study.best_value:.4f}")
    logging.info(f"Best parameters found: {study.best_params}")

    logging.info("Balancing full training data with SMOTE for final model...")
    smote_final = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote_final.fit_resample(X_train, y_train)
    logging.info(f"Resampled training data shape after SMOTE: {X_train_resampled.shape}")

    logging.info("Training final Random Forest model with best parameters...")
    best_params = study.best_params
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    
    final_model.fit(X_train_resampled, y_train_resampled)
    logging.info("Final model training complete.")
    
    logging.info(f"Saving trained Random Forest model to {MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    
    logging.info("Evaluating final model on the hold-out test set...")
    test_scores = final_model.predict_proba(X_test)[:, 1]
    
    best_f1, prec, rec, f1_thresh, f1_cm, final_adjusted_preds = find_best_f1_point_adjusted(y_test, test_scores)
    try: 
        auc = roc_auc_score(y_test, test_scores)
    except ValueError: 
        auc = -1.0
    
    logging.info("\n--- Final Random Forest Evaluation Results (with Optuna tuning) ---")
    logging.info(f"AUC: {auc:.4f}")
    logging.info(f"Best Point-Adjusted F1 Score: {best_f1:.4f}")
    logging.info(f"  Point-Adjusted Precision: {prec:.4f}, Point-Adjusted Recall: {rec:.4f}, Threshold: {f1_thresh:.6f}")
    logging.info(f"Point-Adjusted Confusion Matrix:\n{f1_cm}")

    logging.info("\n--- Detailed Classification Report (Point-Adjusted) ---")
    report = classification_report(y_test, final_adjusted_preds, target_names=['Normal (Class 0)', 'Anomaly (Class 1)'])
    logging.info(f"\n{report}")

    logging.info("\n--- Full Feature Importances (Most to Least Important) ---")
    feature_importances = final_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    for i in sorted_idx:
        logging.info(f"Feature {i}: Importance = {feature_importances[i]:.6f}")

    # --- ⬇️ MODIFIED: DYNAMIC FILENAME GENERATION FOR PLOTS WITH SUFFIX ⬇️ ---
    logging.info("\n--- Generating Evaluation Plots ---")
    plot_feature_importance(final_model, PROJECT_DIR / f"feature_importance_{SPLIT_STRATEGY}_hyp_opt.png")
    plot_roc_curve(y_test, test_scores, auc, PROJECT_DIR / f"roc_curve_{SPLIT_STRATEGY}_hyp_opt.png")
    plot_precision_recall_curve(y_test, test_scores, PROJECT_DIR / f"precision_recall_curve_{SPLIT_STRATEGY}_hyp_opt.png")
    plot_confusion_matrix(f1_cm, PROJECT_DIR / f"confusion_matrix_{SPLIT_STRATEGY}_hyp_opt.png")
    # --- ⬆️ END OF MODIFICATION ⬆️ ---
