"""
Hybrid Autoencoder Feature Extraction — MRO Telemetry
--------------------------------------------------------------------------
Purpose
    Load a trained prediction model and run inference over TRAIN and 
    TEST tracks to export per‑window features that combine:
      • Latent embedding from the encoder (last token representation)
      • Reconstruction/prediction error (Huber loss) at the window’s final step

    These features are saved to disk (along with labels) for downstream
    classical models (Random Forest) to perform anomaly detection.

I/O conventions
    Input  : PROJECT_DIR/processed_data/manifest.json
             PROJECT_DIR/data_files/track_*.parquet and *_labels.npy
             PROJECT_DIR/best_prediction_model.pth (contains model_params + weights)
    Output : PROJECT_DIR/random_forest_data/{train,test}/<track>_{features,labels}.npy

Notes
    • No functional changes vs. the provided implementation; this file adds
      docstrings and inline comments for production clarity only.
    • The PredictionModel returns only the predicted target vector during
      standard forward(). For feature export we separately call encoder/decoder
      to compute both latent state and per‑window error.
"""

import torch
import numpy as np
from pathlib import Path
from torch import nn
from tqdm import tqdm
import json
import logging
import sys
import pandas as pd
import math

from data_utils_mro import SingleTrackPredictionDataset

# =================
# --- MODEL ---
# =================
class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Transformer‑style).

    Adds position‑dependent sine/cosine terms to token embeddings. Registered
    as a buffer to avoid training updates.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]; add position encodings for the first T steps
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder that returns the representation of the last token.

    Embedding projects input_dim → d_model. Positional encoding is added before
    standard nn.TransformerEncoder layers. The forward() returns the last time
    step representation (akin to using a CLS‑like token if last step is target).
    """
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src):
        # src: [B, T, F_in]
        src = self.embedding(src) * math.sqrt(self.d_model)  # scale to stabilize
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)               # [B, T, d_model]
        return output[:, -1, :]                              # last token


class Decoder(nn.Module):
    """MLP mapping encoder state → target feature vector (regression)."""
    def __init__(self, d_model, output_dim):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, output_dim)
        )
    def forward(self, z):
        return self.mlp(z)


class PredictionModel(nn.Module):
    """Transformer encoder + MLP decoder for last‑step target prediction."""
    def __init__(self, input_dim, output_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout):
        super(PredictionModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = Decoder(d_model, output_dim)
    def forward(self, x):
        encoded_state = self.encoder(x)
        predicted_target = self.decoder(encoded_state)
        return predicted_target


# =========================
# --- CONFIGURATION ---
# =========================
PROJECT_DIR = Path("/home/nzhou/MRO")
OUTPUT_DIR = PROJECT_DIR / "processed_data"
DATA_DIR = PROJECT_DIR / "data_files"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TRAINED_DL_MODEL_PATH = PROJECT_DIR / "best_prediction_model.pth"

# Output subdirs for downstream (tree is created if missing)
RANDOM_FOREST_DATA_DIR = PROJECT_DIR / "random_forest_data"
RANDOM_FOREST_TRAIN_DIR = RANDOM_FOREST_DATA_DIR / "train"
RANDOM_FOReST_TEST_DIR = RANDOM_FOREST_DATA_DIR / "test"

# Runtime knobs
BATCH_SIZE = 512
WINDOW_SIZE = 100
NUM_TARGET_FEATURES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Unified console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


def get_target_columns(manifest_path, data_dir, num_targets):
    """Heuristically select `num_targets` high‑variance numeric columns as targets.

    Time‑derived columns are excluded from candidate targets. Inputs are all
    remaining columns. Returns (input_columns, target_columns).
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    train_files = [data_dir / item['track'] for item in manifest['train']]
    df_list = [pd.read_parquet(f) for f in train_files[:50]]
    full_df = pd.concat(df_list, ignore_index=True)

    numeric_cols = full_df.select_dtypes(include=np.number).columns.tolist()
    time_features_to_exclude = ['seconds_since_start', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    candidate_cols = [col for col in numeric_cols if col not in time_features_to_exclude]

    variances = full_df[candidate_cols].var().sort_values()
    target_columns = variances.tail(num_targets).index.tolist()

    all_columns = full_df.columns.tolist()
    input_columns = [col for col in all_columns if col not in target_columns]
    return input_columns, target_columns


def generate_features_for_track(model, track_dataset, device):
    """Generate per‑window features for a single track.

    For each window: compute encoder latent state and Huber reconstruction error
    at the final timestep, then concatenate them → [latent | error]. Returns
    (features_array, labels_array) or (None, None) if the track yields no
    windows.
    """
    model.eval()
    track_loader = torch.utils.data.DataLoader(track_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features, all_labels = [], []
    loss_fn = nn.HuberLoss(reduction='none')
    with torch.no_grad():
        for (inputs, targets), labels in track_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            latent_features = model.encoder(inputs)                     # [B, d_model]
            predictions = model.decoder(latent_features)                # [B, F_out]
            error = torch.mean(loss_fn(predictions, targets), dim=1, keepdim=True)  # [B, 1]
            combined_features = torch.cat((latent_features, error), dim=1)         # [B, d_model+1]
            all_features.append(combined_features.cpu().numpy())
            all_labels.append(labels.numpy())
    if not all_features:
        return None, None
    return np.concatenate(all_features), np.concatenate(all_labels)


# ======================
# --- Main Execution ---
# ======================
if __name__ == "__main__":
    # Prepare output directories for idempotent runs
    RANDOM_FOREST_DATA_DIR.mkdir(exist_ok=True)
    RANDOM_FOREST_TRAIN_DIR.mkdir(exist_ok=True)
    RANDOM_FOREST_TEST_DIR.mkdir(exist_ok=True)

    # Ensure checkpoint exists before heavy work
    if not TRAINED_DL_MODEL_PATH.exists():
        logging.error(f"Deep learning model not found at {TRAINED_DL_MODEL_PATH}. Please run the training script first.")
        sys.exit(1)

    logging.info("--- Stage 1: Feature Extraction (Latent Features + Prediction Error) ---")

    # Determine model inputs/targets from TRAIN distribution
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)

    # Load manifest and model checkpoint
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    logging.info(f"Loading trained Transformer model from {TRAINED_DL_MODEL_PATH}")

    checkpoint = torch.load(TRAINED_DL_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model_params = checkpoint['model_params']
    model = PredictionModel(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Iterate over TRAIN and TEST splits, save per‑track artifacts
    for split_name, tracks in [('train', manifest['train']), ('test', manifest['test'])]:
        logging.info(f"Generating features for the {split_name} set...")
        output_dir = RANDOM_FOREST_TRAIN_DIR if split_name == 'train' else RANDOM_FOREST_TEST_DIR
        for track_info in tqdm(tracks, desc=f"Processing {split_name} tracks"):
            track_name = Path(track_info['track']).stem
            track_dataset = SingleTrackPredictionDataset(
                DATA_DIR / track_info['track'],
                DATA_DIR / track_info['labels'],
                input_cols,
                target_cols,
                WINDOW_SIZE,
            )
            if len(track_dataset) == 0:
                continue  # Skip tracks with fewer than window_size rows
            features, labels = generate_features_for_track(model, track_dataset, DEVICE)
            if features is not None:
                np.save(output_dir / f"{track_name}_features.npy", features)
                np.save(output_dir / f"{track_name}_labels.npy", labels)

    logging.info("--- Feature extraction complete! ---")
