"""Generate JWST latent features for downstream classical models."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from data_utils_jwst import create_prediction_dataloaders


class Attention(nn.Module):
    """Attention module matching the training architecture."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention_net(lstm_output).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(lstm_output.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return context_vector


class Encoder(nn.Module):
    """CNN + BiLSTM encoder with attention pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout_rate: float) -> None:
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(input_dim, 64, kernel_size=7, padding=3), nn.ReLU())
        self.lstm = nn.LSTM(
            64,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = Attention(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        return context_vector


class Decoder(nn.Module):
    """MLP decoder retained for checkpoint compatibility."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class PredictionModel(nn.Module):
    """Encoder-only inference model for latent extraction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout_rate: float) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
        self.decoder = Decoder(hidden_dim, output_dim)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_state = self.encoder(x)
        return encoded_state


# --- CONFIGURATION ---
DATASET_TO_USE = "low_band"
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData")
BASE_INPUT_DIR = PROJECT_DIR / "processed_diffusion_style"
BASE_OUTPUT_DIR = PROJECT_DIR / "jwst_vae_work"
INPUT_DATASET_DIR = BASE_INPUT_DIR / DATASET_TO_USE
OUTPUT_SUBDIR = BASE_OUTPUT_DIR / DATASET_TO_USE
DATA_DIR = INPUT_DATASET_DIR / "data_files"
MANIFEST_PATH = INPUT_DATASET_DIR / "manifest.json"
TRAINED_DL_MODEL_PATH = OUTPUT_SUBDIR / f"best_prediction_model_{DATASET_TO_USE}.pth"

# --- Output Files ---
XGB_FEATURES_DIR = OUTPUT_SUBDIR / "xgboost_features_per_track"
TEST_FEAT_PATH = XGB_FEATURES_DIR / "test_features.npy"  # This path is for reference, script saves per track
TEST_LABELS_PATH = XGB_FEATURES_DIR / "test_labels.npy"
TEST_BOUNDARIES_PATH = XGB_FEATURES_DIR / "test_boundaries.npy"

# --- Hyperparameters ---
BATCH_SIZE = 512
WINDOW_SIZE = 200
NUM_TARGET_FEATURES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def get_target_columns(manifest_path: Path, data_dir: Path, num_targets: int) -> tuple[list[str], list[str]]:
    """Replicate the training-time target selection."""

    with open(manifest_path, 'r') as f: manifest = json.load(f)
    train_files = [data_dir / item['track_features'] for item in manifest['train']]
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

if __name__ == "__main__":
    XGB_FEATURES_DIR.mkdir(exist_ok=True)
    
    if not TRAINED_DL_MODEL_PATH.exists():
        logging.error(f"Deep learning model not found at {TRAINED_DL_MODEL_PATH}. Please run the training script first.")
        sys.exit(1)

    logging.info(f"--- Stage 1: Feature Extraction for '{DATASET_TO_USE}' ---")
    
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)

    dataloaders = create_prediction_dataloaders(
        manifest_path=MANIFEST_PATH, data_dir=DATA_DIR, input_cols=input_cols, target_cols=target_cols,
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, debug_mode=False
    )
    test_loader = dataloaders['test']
    
    logging.info(f"Loading trained semi-supervised model from {TRAINED_DL_MODEL_PATH}")
    checkpoint = torch.load(TRAINED_DL_MODEL_PATH, map_location=DEVICE)
    model_params = checkpoint['model_params']
    
    model = PredictionModel(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logging.info(f"Generating and saving features for each track to {XGB_FEATURES_DIR}...")
    with torch.no_grad():
        for track_dataset in tqdm(test_loader.dataset.datasets, desc="Processing individual tracks"):
            if len(track_dataset) == 0:
                continue
            
            track_id = track_dataset.track_path.stem
            single_track_loader = torch.utils.data.DataLoader(track_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            features_list, labels_list = [], []
            for (inputs, _), labels in single_track_loader:
                inputs = inputs.to(DEVICE)
                features = model(inputs) # Forward pass only returns encoder state
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
            
            if not features_list:
                continue

            track_features = np.concatenate(features_list)
            track_labels = np.concatenate(labels_list)

            np.save(XGB_FEATURES_DIR / f"{track_id}_features.npy", track_features)
            np.save(XGB_FEATURES_DIR / f"{track_id}_labels.npy", track_labels)

    logging.info("--- Per-track feature extraction complete! ---")
