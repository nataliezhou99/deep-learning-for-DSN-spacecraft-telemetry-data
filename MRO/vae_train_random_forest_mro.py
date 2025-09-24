"""Train a Random Forest classifier on MRO transformer features."""

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


# --- CONFIGURATION ---
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/MRODataSet")
MANIFEST_PATH = PROJECT_DIR / "processed_data" / "manifest.json"
XGB_DATA_DIR = PROJECT_DIR / "xgboost_data"
XGB_TRAIN_DIR_NORMAL = XGB_DATA_DIR / "train"
XGB_TEST_DIR_MIXED = XGB_DATA_DIR / "test"
MODEL_SAVE_PATH = PROJECT_DIR / "best_random_forest_model.pkl"
EVAL_LOG_FILE = PROJECT_DIR / "random_forest_evaluation.log"
PLOTS_DIR = PROJECT_DIR / "plots"

SPLIT_STRATEGY = "stratified_track_split"
N_TRIALS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(EVAL_LOG_FILE, mode="w"), logging.StreamHandler(sys.stdout)],
)


def plot_feature_importance(model: RandomForestClassifier, save_path: Path) -> None:
    """Persist a bar chart of the most important features."""

    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    features = [f"Feature {i}" for i in indices]
    plt.figure(figsize=(10, 8))
    plt.title("Top 20 Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), features)
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info("Feature importance plot saved to %s", save_path)


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, save_path: Path) -> None:
    """Plot the ROC curve for the evaluation set."""

    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info("ROC curve plot saved to %s", save_path)


def plot_pr_curve(labels: np.ndarray, scores: np.ndarray, save_path: Path) -> None:
    """Plot the precision-recall curve for the evaluation set."""

    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info("Precision-recall curve plot saved to %s", save_path)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    """Render the confusion matrix heatmap."""

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title("Point-Adjusted Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    plt.close()
    logging.info("Confusion matrix plot saved to %s", save_path)


def _get_events(y_true: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous anomaly windows in the label sequence."""

    events: list[tuple[int, int]] = []
    y_true_diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(y_true_diff == 1)[0]
    ends = np.where(y_true_diff == -1)[0]
    for start, end in zip(starts, ends):
        events.append((start, end))
    return events


def find_best_f1_point_adjusted(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Search thresholds and return the best point-adjusted F1 statistics."""

    best_f1, best_threshold = -1.0, -1.0
    true_events = _get_events(labels)
    if not true_events:
        return 0.0, 0.0, 0.0, 0.0, confusion_matrix(labels, scores > 0.5), (scores > 0.5).astype(int)

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


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective that maximises point-adjusted F1 via cross-validation."""

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**param)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores: list[float] = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        model.fit(X_train_fold, y_train_fold)
        y_scores_fold = model.predict_proba(X_val_fold)[:, 1]
        f1, _, _, _, _, _ = find_best_f1_point_adjusted(y_val_fold, y_scores_fold)
        scores.append(f1)
    return float(np.mean(scores))


def load_data_from_tracks(track_list: list[dict], base_dir: Path, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """Load concatenated feature and label arrays for the provided tracks."""

    all_features, all_labels = [], []
    for track_info in tqdm(track_list, desc=desc, leave=False):
        track_id = Path(track_info["track"]).stem
        feature_file = base_dir / f"{track_id}_features.npy"
        label_file = base_dir / f"{track_id}_labels.npy"
        if feature_file.exists() and label_file.exists():
            all_features.append(np.load(feature_file))
            all_labels.append(np.load(label_file))
        else:
            logging.warning("Missing feature or label file for %s in %s", track_id, base_dir)
    if not all_features:
        return np.array([]), np.array([])
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def load_individual_track(track_info: dict, base_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load features and labels for a single track, returning None when missing."""

    track_id = Path(track_info["track"]).stem
    feature_file = base_dir / f"{track_id}_features.npy"
    label_file = base_dir / f"{track_id}_labels.npy"
    if not feature_file.exists() or not label_file.exists():
        logging.warning("Missing feature or label file for %s in %s", track_id, base_dir)
        return None, None
    features = np.load(feature_file)
    labels = np.load(label_file)
    if len(features) == 0:
        return None, None
    return features, labels


if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)
    logging.info("--- Stage 2: Training Random Forest with '%s' and Optuna Tuning ---", SPLIT_STRATEGY)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    train_tracks = manifest.get("train", [])
    test_tracks = manifest.get("test", [])

    X_train_normal, y_train_normal = load_data_from_tracks(train_tracks, XGB_TRAIN_DIR_NORMAL, "Loading baseline training tracks")

    if SPLIT_STRATEGY == "within_track_chronological_split":
        logging.info("Applying within-track chronological split (70/30 per track).")
        X_train_list, y_train_list, X_eval_list, y_eval_list = [], [], [], []
        for track_info in tqdm(test_tracks, desc="Splitting test tracks", leave=False):
            features, labels = load_individual_track(track_info, XGB_TEST_DIR_MIXED)
            if features is None:
                continue
            split_idx = int(0.7 * len(features))
            X_train_list.append(features[:split_idx])
            y_train_list.append(labels[:split_idx])
            X_eval_list.append(features[split_idx:])
            y_eval_list.append(labels[split_idx:])

        if X_train_normal.size:
            X_train_list.append(X_train_normal)
            y_train_list.append(y_train_normal)

        if not X_train_list or not X_eval_list:
            raise ValueError("Insufficient data after chronological split.")

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_eval_list, axis=0)
        y_test = np.concatenate(y_eval_list, axis=0)

    elif SPLIT_STRATEGY == "stratified_track_split":
        logging.info("Applying stratified split across tracks to balance anomalies and normals.")
        anomalous_tracks, normal_tracks = [], []
        for track_info in tqdm(test_tracks, desc="Categorising test tracks", leave=False):
            _, labels = load_individual_track(track_info, XGB_TEST_DIR_MIXED)
            if labels is None:
                continue
            if np.sum(labels) > 0:
                anomalous_tracks.append(track_info)
            else:
                normal_tracks.append(track_info)

        if not anomalous_tracks:
            raise ValueError("No anomalous tracks found for stratified split.")

        train_anom_tracks, eval_anom_tracks = train_test_split(anomalous_tracks, test_size=0.30, random_state=42)
        train_norm_tracks, eval_norm_tracks = train_test_split(normal_tracks, test_size=0.30, random_state=42)

        logging.info(
            "Training tracks: %d anomalous / %d normal | Evaluation tracks: %d anomalous / %d normal",
            len(train_anom_tracks),
            len(train_norm_tracks),
            len(eval_anom_tracks),
            len(eval_norm_tracks),
        )

        X_train_split, y_train_split = load_data_from_tracks(
            train_anom_tracks + train_norm_tracks,
            XGB_TEST_DIR_MIXED,
            "Loading stratified training tracks",
        )
        X_eval_split, y_eval_split = load_data_from_tracks(
            eval_anom_tracks + eval_norm_tracks,
            XGB_TEST_DIR_MIXED,
            "Loading stratified evaluation tracks",
        )

        train_features = [arr for arr in [X_train_normal, X_train_split] if arr.size]
        train_labels = [arr for arr in [y_train_normal, y_train_split] if arr.size]

        if not train_features or not X_eval_split.size:
            raise ValueError("Failed to assemble training or evaluation data for stratified split.")

        X_train = np.concatenate(train_features, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        X_test = X_eval_split
        y_test = y_eval_split

    else:
        raise ValueError(f"Unknown SPLIT_STRATEGY: {SPLIT_STRATEGY}")

    logging.info("Training data shape: %s | Evaluation data shape: %s", X_train.shape, X_test.shape)

    if np.sum(y_train == 1) == 0:
        logging.error("The generated training set contains no anomalies.")
        sys.exit(1)
    logging.info("Balancing training data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info("Resampled training data shape after SMOTE: %s", X_train_resampled.shape)

    logging.info("Starting hyperparameter optimization with Optuna (%d trials)...", N_TRIALS)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train_resampled, y_train_resampled), n_trials=N_TRIALS)
    logging.info("Optimization complete. Best point-adjusted F1 score found: %.4f", study.best_value)
    logging.info("Best parameters found: %s", study.best_params)

    logging.info("Training final Random Forest model with best parameters on resampled data...")
    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train_resampled, y_train_resampled)
    logging.info("Model training complete.")

    logging.info("Saving trained Random Forest model to %s", MODEL_SAVE_PATH)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(final_model, f)

    logging.info("Evaluating best model on the hold-out test set...")
    test_scores = final_model.predict_proba(X_test)[:, 1]
    best_f1, prec, rec, f1_thresh, f1_cm, final_adjusted_preds = find_best_f1_point_adjusted(y_test, test_scores)
    try:
        auc_val = roc_auc_score(y_test, test_scores)
    except ValueError:
        auc_val = -1.0

    logging.info("\n--- Final Random Forest Evaluation Results ---")
    logging.info("AUC: %.4f", auc_val)
    logging.info("Best Point-Adjusted F1 Score: %.4f", best_f1)
    logging.info("  Point-Adjusted Precision: %.4f, Point-Adjusted Recall: %.4f, Threshold: %.6f", prec, rec, f1_thresh)
    logging.info("Point-Adjusted Confusion Matrix:\n%s", f1_cm)

    logging.info("\n--- Detailed Point-Adjusted Classification Report ---")
    report = classification_report(y_test, final_adjusted_preds, target_names=["Normal", "Anomaly"])
    logging.info("\n%s", report)

    logging.info("\n--- All Feature Importances (Most to Least Important) ---")
    feature_importances = final_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    for feature_index in sorted_idx:
        importance_value = feature_importances[feature_index]
        logging.info("Feature %d: Importance = %.4f", feature_index, importance_value)

    logging.info("\n--- Generating Evaluation Plots ---")
    plot_feature_importance(final_model, PLOTS_DIR / "feature_importance.png")
    plot_roc_curve(y_test, test_scores, PLOTS_DIR / "roc_curve.png")
    plot_pr_curve(y_test, test_scores, PLOTS_DIR / "pr_curve.png")
    plot_confusion_matrix(f1_cm, PLOTS_DIR / "confusion_matrix.png")
