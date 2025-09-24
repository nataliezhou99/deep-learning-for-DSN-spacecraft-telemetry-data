# extract_features.py

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

from data_utils_hybrid_vae_mro import SingleTrackPredictionDataset

# --- MODEL DEFINITION (Must match the trained model) ---
class PositionalEncoding(nn.Module):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output[:, -1, :]

class Decoder(nn.Module):
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
    def __init__(self, input_dim, output_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout):
        super(PredictionModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = Decoder(d_model, output_dim)
    def forward(self, x):
        encoded_state = self.encoder(x)
        predicted_target = self.decoder(encoded_state)
        return predicted_target

# --- CONFIGURATION ---
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/MRODataSet")
OUTPUT_DIR = PROJECT_DIR / "processed_data"
DATA_DIR = PROJECT_DIR / "data_files"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TRAINED_DL_MODEL_PATH = PROJECT_DIR / "best_prediction_model.pth"
XGB_DATA_DIR = PROJECT_DIR / "xgboost_data"
XGB_TRAIN_DIR = XGB_DATA_DIR / "train"
XGB_TEST_DIR = XGB_DATA_DIR / "test"
BATCH_SIZE = 512
WINDOW_SIZE = 100
NUM_TARGET_FEATURES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def get_target_columns(manifest_path, data_dir, num_targets):
    with open(manifest_path, 'r') as f: manifest = json.load(f)
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
    model.eval()
    track_loader = torch.utils.data.DataLoader(track_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features, all_labels = [], []
    loss_fn = nn.HuberLoss(reduction='none')
    with torch.no_grad():
        for (inputs, targets), labels in track_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            latent_features = model.encoder(inputs)
            predictions = model.decoder(latent_features)
            error = torch.mean(loss_fn(predictions, targets), dim=1, keepdim=True)
            combined_features = torch.cat((latent_features, error), dim=1)
            all_features.append(combined_features.cpu().numpy())
            all_labels.append(labels.numpy())
    if not all_features: return None, None
    return np.concatenate(all_features), np.concatenate(all_labels)

if __name__ == "__main__":
    XGB_DATA_DIR.mkdir(exist_ok=True)
    XGB_TRAIN_DIR.mkdir(exist_ok=True)
    XGB_TEST_DIR.mkdir(exist_ok=True)
    if not TRAINED_DL_MODEL_PATH.exists():
        logging.error(f"Deep learning model not found at {TRAINED_DL_MODEL_PATH}. Please run the training script first.")
        sys.exit(1)
    logging.info("--- Stage 1: Feature Extraction (Latent Features + Prediction Error) ---")
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)
    with open(MANIFEST_PATH, 'r') as f: manifest = json.load(f)
    logging.info(f"Loading trained Transformer model from {TRAINED_DL_MODEL_PATH}")
    checkpoint = torch.load(TRAINED_DL_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model_params = checkpoint['model_params']
    model = PredictionModel(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    for split_name, tracks in [('train', manifest['train']), ('test', manifest['test'])]:
        logging.info(f"Generating features for the {split_name} set...")
        output_dir = XGB_TRAIN_DIR if split_name == 'train' else XGB_TEST_DIR
        for track_info in tqdm(tracks, desc=f"Processing {split_name} tracks"):
            track_name = Path(track_info['track']).stem
            track_dataset = SingleTrackPredictionDataset(DATA_DIR / track_info['track'], DATA_DIR / track_info['labels'], input_cols, target_cols, WINDOW_SIZE)
            if len(track_dataset) == 0: continue
            features, labels = generate_features_for_track(model, track_dataset, DEVICE)
            if features is not None:
                np.save(output_dir / f"{track_name}_features.npy", features)
                np.save(output_dir / f"{track_name}_labels.npy", labels)
    logging.info("--- Feature extraction complete! ---")
