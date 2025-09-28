"""
Hybrid Autoencoder Feature Extraction — JWST DSN Telemetry
----------------------------------------------------------
Purpose
    Load a trained semi‑supervised prediction model (CNN → BiLSTM → Attention → MLP)
    and run inference over the TEST split to export per‑window latent features
    (encoder outputs) and corresponding binary labels per track. These features
    can be used downstream by a classical classifier (RandomForest) to compute
    anomaly scores.

Key behaviors
    • Selects input/target columns from training data by variance ranking, while
      excluding time features (seconds_since_start, harmonics) from targets.
    • Builds DataLoaders from a preprocessing manifest (see data_utils_jwst.py).
    • Loads model weights + params from checkpoint and performs forward pass to
      obtain encoder states (no gradients).
    • Saves features/labels per track as separate .npy artifacts.

I/O conventions
    Input  : PROJECT_DIR/processed_data/<dataset>/manifest.json
             PROJECT_DIR/processed_data/<dataset>/data_files/*.parquet
             PROJECT_DIR/<dataset>/best_prediction_model_<dataset>.pth
    Output : PROJECT_DIR/<dataset>/random_forest_features_per_track/<track>_features.npy
             PROJECT_DIR/<dataset>/random_forest_features_per_track/<track>_labels.npy

Notes
    • This script does *not* train the model—only extracts embeddings/features.
    • The model forward returns only the encoder state by design (see class).
    • No functional changes have been made versus the supplied implementation;
      only documentation and comments for production clarity.
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import json
import logging
import sys

from data_utils_jwst import create_prediction_dataloaders

# =================
# --- MODEL ---
# =================
class Attention(nn.Module):
    """Additive attention over BiLSTM outputs.

    Expects input of shape [B, T, 2H] (bidirectional hidden concat) and returns
    a context vector of shape [B, 2H] computed via softmax over timesteps.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: [B, T, 2H]
        attention_scores = self.attention_net(lstm_output).squeeze(2)     # [B, T]
        attention_weights = F.softmax(attention_scores, dim=1)             # [B, T]
        # Weighted sum across time using batch matrix multiply
        context_vector = torch.bmm(
            lstm_output.transpose(1, 2),                                   # [B, 2H, T]
            attention_weights.unsqueeze(2)                                  # [B, T, 1]
        ).squeeze(2)                                                       # [B, 2H]
        return context_vector


class Encoder(nn.Module):
    """Temporal feature encoder: 1D CNN → BiLSTM → Attention.

    • CNN extracts local temporal patterns (kernel_size=7).
    • BiLSTM captures longer dependencies bidirectionally.
    • Attention aggregates the sequence into a fixed‑size vector.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3), nn.ReLU()
        )
        self.lstm = nn.LSTM(
            64, hidden_dim, num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0, bidirectional=True
        )
        self.attention = Attention(hidden_dim)

    def forward(self, x):
        # x: [B, T, F_in] → permute for Conv1d which expects [B, C_in, L]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                       # [B, 64, T]
        x = x.permute(0, 2, 1)                # [B, T, 64]
        lstm_out, _ = self.lstm(x)            # [B, T, 2H]
        context_vector = self.attention(lstm_out)  # [B, 2H]
        return context_vector


class Decoder(nn.Module):
    """Deterministic MLP head to map encoder state → regression targets."""
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.mlp(z)


class PredictionModel(nn.Module):
    """Semi‑supervised model: encoder, decoder, plus binary head for anomalies.

    In this script we only utilize the encoder (embedding extractor). The
    decoder and classification head are defined for completeness and training
    compatibility with the checkpoint.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
        super(PredictionModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
        self.decoder = Decoder(hidden_dim, output_dim)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Intentionally return only the encoder state for downstream feature use
        encoded_state = self.encoder(x)
        return encoded_state


# =========================
# --- CONFIGURATION ---
# =========================
DATASET_TO_USE = "low_band"
PROJECT_DIR = Path("/home/nzhou/JWST")
BASE_INPUT_DIR = PROJECT_DIR / "processed_data"
BASE_OUTPUT_DIR = PROJECT_DIR
INPUT_DATASET_DIR = BASE_INPUT_DIR / DATASET_TO_USE
OUTPUT_SUBDIR = BASE_OUTPUT_DIR / DATASET_TO_USE
DATA_DIR = INPUT_DATASET_DIR / "data_files"
MANIFEST_PATH = INPUT_DATASET_DIR / "manifest.json"
TRAINED_DL_MODEL_PATH = OUTPUT_SUBDIR / f"best_prediction_model_{DATASET_TO_USE}.pth"

# --- Output Files ---
RANDOM_FOREST_FEATURES_DIR = OUTPUT_SUBDIR / "random_forest_features_per_track"
TEST_FEAT_PATH = RANDOM_FOREST_FEATURES_DIR / "test_features.npy"   # Reference only; script saves per‑track
TEST_LABELS_PATH = RANDOM_FOREST_FEATURES_DIR / "test_labels.npy"
TEST_BOUNDARIES_PATH = RANDOM_FOREST_FEATURES_DIR / "test_boundaries.npy"

# --- Hyperparameters ---
BATCH_SIZE = 512
WINDOW_SIZE = 200
NUM_TARGET_FEATURES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Unified console logging for orchestration and notebook runs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


def get_target_columns(manifest_path, data_dir, num_targets):
    """Heuristically pick `num_targets` most‑variant numeric columns as targets.

    Excludes explicit time‑derived features to keep the forecasting target
    semantically meaningful. Returns (input_columns, target_columns).
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Sample a subset of train files to estimate variance robustly
    train_files = [data_dir / item['track_features'] for item in manifest['train']]
    df_list = [pd.read_parquet(f) for f in train_files[:50]]
    full_df = pd.concat(df_list, ignore_index=True)

    numeric_cols = full_df.select_dtypes(include=np.number).columns.tolist()
    time_features_to_exclude = ['seconds_since_start', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']

    # Candidate target pool excludes time harmonics and time index
    candidate_cols = [col for col in numeric_cols if col not in time_features_to_exclude]

    # Highest variance columns are assumed most informative to predict
    variances = full_df[candidate_cols].var().sort_values()
    target_columns = variances.tail(num_targets).index.tolist()

    # Inputs are everything else (model will learn to reconstruct/predict targets)
    all_columns = full_df.columns.tolist()
    input_columns = [col for col in all_columns if col not in target_columns]
    return input_columns, target_columns


if __name__ == "__main__":
    # Ensure output directory exists; avoid failure due to missing parent
    RANDOM_FOREST_FEATURES_DIR.mkdir(exist_ok=True)
    
    # Sanity check: trained model must be present
    if not TRAINED_DL_MODEL_PATH.exists():
        logging.error(f"Deep learning model not found at {TRAINED_DL_MODEL_PATH}. Please run the training script first.")
        sys.exit(1)

    logging.info(f"--- Stage 1: Feature Extraction for '{DATASET_TO_USE}' ---")
    
    # Derive input/target columns from training distribution
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)

    # Build dataloaders; only the TEST loader is used for exporting per‑track artifacts
    dataloaders = create_prediction_dataloaders(
        manifest_path=MANIFEST_PATH, data_dir=DATA_DIR, input_cols=input_cols, target_cols=target_cols,
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, debug_mode=False
    )
    test_loader = dataloaders['test']
    
    logging.info(f"Loading trained semi-supervised model from {TRAINED_DL_MODEL_PATH}")
    checkpoint = torch.load(TRAINED_DL_MODEL_PATH, map_location=DEVICE)
    model_params = checkpoint['model_params']
    
    # Recreate model with saved hyperparameters and restore weights
    model = PredictionModel(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logging.info(f"Generating and saving features for each track to {RANDOM_FOREST_FEATURES_DIR}...")
    with torch.no_grad():
        # Iterate per‑track dataset inside the concatenated TEST dataset
        for track_dataset in tqdm(test_loader.dataset.datasets, desc="Processing individual tracks"):
            if len(track_dataset) == 0:
                continue  # Skip empty windows edge case
            
            track_id = track_dataset.track_path.stem
            single_track_loader = torch.utils.data.DataLoader(track_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            features_list, labels_list = [], []
            for (inputs, _), labels in single_track_loader:
                inputs = inputs.to(DEVICE)
                features = model(inputs)  # Forward pass returns encoder state only
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
            
            if not features_list:
                continue

            track_features = np.concatenate(features_list)
            track_labels = np.concatenate(labels_list)

            # Persist per‑track artifacts; downstream code can glob by *_features.npy
            np.save(RANDOM_FOREST_FEATURES_DIR / f"{track_id}_features.npy", track_features)
            np.save(RANDOM_FOREST_FEATURES_DIR / f"{track_id}_labels.npy", track_labels)

    logging.info("--- Per-track feature extraction complete! ---")
