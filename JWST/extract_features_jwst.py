import torch
import numpy as np
from pathlib import Path
from torch import nn
from tqdm import tqdm
import json
import logging
import sys
import pandas as pd
import random

from data_utils_hybrid_vae_jwst_newest import SingleTrackPredictionDataset

# --- MODEL DEFINITION ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0.0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0) # --- ⬇️ FIX: Get the dynamic batch size
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        
        v_view = self.v.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        
        attn_weights = torch.bmm(v_view, energy).squeeze(1)
        soft_attn_weights = nn.functional.softmax(attn_weights, dim=1)
        
        context = torch.bmm(encoder_outputs.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3), nn.ReLU()
        )
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0, 
                            bidirectional=True)
        self.attention = Attention(hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, _) = self.lstm(x)
        
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        context_vector = self.attention(hidden_final, lstm_out)
        return context_vector

class Decoder(nn.Module):
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
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
        super(PredictionModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
        self.decoder = Decoder(hidden_dim, output_dim)
    def forward(self, x):
        encoded_state = self.encoder(x)
        predicted_target = self.decoder(encoded_state)
        return predicted_target

# --- CONFIGURATION ---
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData/jwst_vae_work")
DATA_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData/processed_diffusion_style/low_band/data_files")
OUTPUT_DIR = PROJECT_DIR / "processed_data"
MANIFEST_PATH = Path("/home/nzhou/updated_dsn_project/JWSTData/processed_diffusion_style/low_band/manifest.json")
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

def generate_features_for_track(model, track_dataset, device):
    model.eval()
    track_loader = torch.utils.data.DataLoader(track_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_features, all_labels = [], []
    with torch.no_grad():
        for (inputs, _), labels in track_loader:
            inputs = inputs.to(device)
            features = model.encoder(inputs)
            all_features.append(features.cpu().numpy())
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
    logging.info("--- Stage 1: Feature Extraction (Track-by-Track) ---")
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)
    with open(MANIFEST_PATH, 'r') as f: manifest = json.load(f)
    logging.info(f"Loading trained BiLSTM+Attention model from {TRAINED_DL_MODEL_PATH}")
    checkpoint = torch.load(TRAINED_DL_MODEL_PATH, map_location=DEVICE)
    model_params = checkpoint['model_params']
    model = PredictionModel(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    for split_name, tracks in [('train', manifest['train']), ('test', manifest['test'])]:
        logging.info(f"Generating features for the {split_name} set...")
        output_dir = XGB_TRAIN_DIR if split_name == 'train' else XGB_TEST_DIR
        for track_info in tqdm(tracks, desc=f"Processing {split_name} tracks"):
            track_name = Path(track_info['track_features']).stem
            track_dataset = SingleTrackPredictionDataset(DATA_DIR / track_info['track_features'], DATA_DIR / track_info['track_labels'], input_cols, target_cols, WINDOW_SIZE)
            if len(track_dataset) == 0: continue
            features, labels = generate_features_for_track(model, track_dataset, DEVICE)
            if features is not None:
                np.save(output_dir / f"{track_name}_features.npy", features)
                np.save(output_dir / f"{track_name}_labels.npy", labels)
    logging.info("--- Feature extraction complete! ---")
