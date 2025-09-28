"""
Training Prediction Model for MRO Telemetry
--------------------------------------------------------
Purpose
    Train a predictor that maps sliding windows of telemetry to the target
    feature vector at the last timestep of each window (Huber regression).
    Dataloaders are built from the preprocessing manifest and
    windowing utilities in `data_utils_mro.py`.

Workflow
    1) Auto‑select target columns from the TRAIN distribution via variance
       ranking (time‑derived columns excluded). Inputs are the remaining cols.
    2) Build TRAIN and VAL DataLoaders (windowed, with labels unused here).
    3) Train a Transformer encoder + MLP decoder with HuberLoss.
    4) Step LR via ReduceLROnPlateau on validation loss.
    5) Early stop with patience; checkpoint the best model + hyperparams.

I/O Conventions
    Input  : PROJECT_DIR/processed_data/manifest.json
             PROJECT_DIR/data_files/track_*.parquet + *_labels.npy
    Output : PROJECT_DIR/best_prediction_model.pth
             PROJECT_DIR/training_console_transformer.log
             PROJECT_DIR/training_log_transformer.csv (epoch metrics)

Notes
    • This file adds documentation and inline comments only; no logic changes.
    • The dataset returns ((window[T×F_in], target_last_step[F_out]), label);
      labels are not used for training here (pure regression objective).
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import logging
import sys
import math

from data_utils_mro import create_prediction_dataloaders

# =================
# --- MODEL ---
# =================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout.

    Registered as a buffer so it isn't updated by the optimizer.
    Expects input in shape [B, T, d_model].
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder returning the last token representation."""
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src):
        # src: [B, T, F_in] → embed → add PE → encode → take last token
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output[:, -1, :]


class Decoder(nn.Module):
    """MLP head mapping encoder state → regression targets."""
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
    """Transformer encoder + MLP decoder (last‑step target prediction)."""
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
DEBUG_MODE = False
PROJECT_DIR = Path("/home/nzhou/MRO")
OUTPUT_DIR = PROJECT_DIR / "processed_data"
DATA_DIR = PROJECT_DIR / "data_files"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
MODEL_SAVE_PATH = PROJECT_DIR / "best_prediction_model.pth"
LOG_SAVE_PATH = PROJECT_DIR / "training_log_transformer.csv"
TRAIN_LOG_FILE = PROJECT_DIR / "training_console_transformer.log"
NUM_TARGET_FEATURES = 10 
BATCH_SIZE = 128
EPOCHS = 50
WINDOW_SIZE = 100
EARLY_STOPPING_PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Transformer Hyperparameters ---
LEARNING_RATE = 0.0001
D_MODEL = 64
N_HEADS = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT_RATE = 0.2

# Structured logging to file + stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(TRAIN_LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)])
logging.info(f"Using device: {DEVICE}")
if DEBUG_MODE:
    logging.info("\n!!! DEBUG MODE IS ACTIVE !!!\n")


def get_target_columns(manifest_path, data_dir, num_targets):
    """Heuristically select high‑variance target columns from TRAIN files.

    Time‑derived columns are excluded from target candidates; inputs are all
    remaining columns. Returns (input_columns, target_columns).
    """
    logging.info("Identifying target columns from training data...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    train_files = [data_dir / item['track'] for item in manifest['train']]

    # Sample a reasonable subset to estimate variance robustly
    df_list = [pd.read_parquet(f) for f in train_files[:50]]
    full_df = pd.concat(df_list, ignore_index=True)

    numeric_cols = full_df.select_dtypes(include=np.number).columns.tolist()
    time_features_to_exclude = ['seconds_since_start', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    candidate_cols = [col for col in numeric_cols if col not in time_features_to_exclude]
    variances = full_df[candidate_cols].var().sort_values()
    target_columns = variances.tail(num_targets).index.tolist()

    all_columns = full_df.columns.tolist()
    input_columns = [col for col in all_columns if col not in target_columns]
    logging.info(f"Selected {len(target_columns)} target columns: {target_columns}")
    return input_columns, target_columns


def evaluate_model(model, loader, loss_fn, device):
    """Compute average loss over a validation DataLoader (no gradients)."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (inputs, targets), _ in tqdm(loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


# ======================
# --- Main Execution ---
# ======================
if __name__ == "__main__":
    logging.info("--- Training Transformer Prediction Model ---")

    # (1) Feature column selection
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)
    
    # (2) Build dataloaders
    dataloaders = create_prediction_dataloaders(
        manifest_path=MANIFEST_PATH, data_dir=DATA_DIR, input_cols=input_cols, target_cols=target_cols,
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, debug_mode=DEBUG_MODE
    )
    train_loader, val_loader = dataloaders['train'], dataloaders['val']
    if train_loader is None or val_loader is None:
        raise ValueError("Dataloaders could not be created.")

    # (3) Instantiate model + optimizer + scheduler + loss
    input_dim, output_dim = len(input_cols), len(target_cols)
    logging.info(f"Model dimensions: input_dim={input_dim}, output_dim={output_dim}")

    model_hyperparams = {'input_dim': input_dim, 'output_dim': output_dim, 'd_model': D_MODEL, 'n_heads': N_HEADS, 'num_encoder_layers': NUM_ENCODER_LAYERS, 'dim_feedforward': DIM_FEEDFORWARD, 'dropout': DROPOUT_RATE}
    model = PredictionModel(**model_hyperparams).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=False)
    loss_fn = nn.HuberLoss()
    
    best_val_loss, epochs_no_improve = float('inf'), 0

    # (4) Train/validate loop with early stopping and CSV logging
    with open(LOG_SAVE_PATH, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss,lr\n")
        
        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
            for (inputs, targets), _ in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            avg_train_loss = total_train_loss / len(pbar)

            # Validation pass
            avg_val_loss = evaluate_model(model, val_loader, loss_fn, DEVICE)
            
            # Scheduler step + metric logging
            scheduler.step(avg_val_loss)
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {lr:.2e}")
            log_file.write(f"{epoch+1},{avg_train_loss},{avg_val_loss},{lr}\n")
            
            # Early stopping + checkpoint best
            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve = avg_val_loss, 0
                logging.info(f"Validation loss improved. Saving best model to {MODEL_SAVE_PATH}")
                torch.save({'model_params': model_hyperparams, 'model_state_dict': model.state_dict()}, MODEL_SAVE_PATH)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
    logging.info("\n--- Training Complete ---")
