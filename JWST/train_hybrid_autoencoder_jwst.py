"""
Training — Hybrid Autoencoder for JWST DSN Telemetry (Semi‑Supervised)
---------------------------------------------------------------------
Purpose
    Train a CNN→BiLSTM→Attention encoder with dual heads:
      • Decoder (regression) to reconstruct/predict selected target features at
        the last timestep of each sliding window.
      • Binary classification head (logits) to detect anomaly presence within
        the window. During training, windows may use pseudo‑labels; validation
        relies on original labels.

Workflow
    1) Derive input/target feature columns from the training distribution using
       a variance heuristic (excluding explicit time features from targets).
    2) Build DataLoaders from the preprocessing manifest (see data_utils_jwst).
       • TRAIN uses pseudo‑labels if present in PSEUDO_LABELS_DIR (via
         label_dir/label_suffix override).
       • VAL uses original labels from the preprocessing pipeline.
    3) Optimize combined loss = Huber(reconstruction) + λ * BCEWithLogits(class).
    4) Track validation reconstruction loss for early stopping and checkpointing.

I/O conventions
    Input  : PROJECT_DIR/processed_data/<dataset>/manifest.json
             PROJECT_DIR/processed_data/<dataset>/data_files/*.parquet
             PROJECT_DIR/<dataset>/pseudo_labels_per_track/*_pseudo_labels.npy (optional)
    Output : PROJECT_DIR/<dataset>/best_prediction_model_<dataset>.pth (state_dict + params)
             PROJECT_DIR/<dataset>/training_console_<dataset>.log

Notes
    • No functional changes were made; only comments/docstrings and log framing
      for production clarity. Hyperparameters and behavior are unchanged.
    • BCEWithLogitsLoss expects raw logits (no sigmoid). Metrics can be added in
      a follow‑up eval pass using thresholds and ROC/AUC as needed.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch import optim, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
import json
import logging
import sys
import random

from data_utils_jwst import create_prediction_dataloaders

# =================
# --- MODEL ---
# =================
class Attention(nn.Module):
    """Additive attention over BiLSTM outputs.

    Input  : [B, T, 2H] sequence of hidden states
    Output : [B, 2H] context vector via softmax over timesteps
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_output):
        attention_scores = self.attention_net(lstm_output).squeeze(2)   # [B, T]
        attention_weights = F.softmax(attention_scores, dim=1)          # [B, T]
        # Weighted sum across time → context
        context_vector = torch.bmm(
            lstm_output.transpose(1, 2),                                # [B, 2H, T]
            attention_weights.unsqueeze(2)                               # [B, T, 1]
        ).squeeze(2)                                                    # [B, 2H]
        return context_vector


class Encoder(nn.Module):
    """Temporal encoder: Conv1d → BiLSTM → Attention."""
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
        # x: [B, T, F_in] → Conv1d expects [B, C_in, L]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                  # [B, 64, T]
        x = x.permute(0, 2, 1)           # [B, T, 64]
        lstm_out, _ = self.lstm(x)       # [B, T, 2H]
        context_vector = self.attention(lstm_out)
        return context_vector


class Decoder(nn.Module):
    """MLP head mapping encoder state → regression targets."""
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
    """Encoder‑decoder with auxiliary binary classification head.

    Forward returns (reconstruction, classification_logits).
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
        encoded_state = self.encoder(x)                        # [B, 2H]
        reconstruction = self.decoder(encoded_state)           # [B, F_out]
        classification_logits = self.classification_head(encoded_state).squeeze(1)  # [B]
        return reconstruction, classification_logits


# =========================
# --- CONFIGURATION ---
# =========================
DATASET_TO_USE = "low_band"
PROJECT_DIR = Path("/home/nzhou/JWST")
BASE_INPUT_DIR = PROJECT_DIR / "processed_data"
BASE_OUTPUT_DIR = PROJECT_DIR
INPUT_DATASET_DIR = BASE_INPUT_DIR / DATASET_TO_USE
OUTPUT_SUBDIR = BASE_OUTPUT_DIR / DATASET_TO_USE
OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = INPUT_DATASET_DIR / "data_files"
MANIFEST_PATH = INPUT_DATASET_DIR / "manifest.json"
MODEL_SAVE_PATH = OUTPUT_SUBDIR / f"best_prediction_model_{DATASET_TO_USE}.pth"
TRAIN_LOG_FILE = OUTPUT_SUBDIR / f"training_console_{DATASET_TO_USE}.log"
NUM_TARGET_FEATURES = 10
PSEUDO_LABELS_DIR = OUTPUT_SUBDIR / "pseudo_labels_per_track"

# --- Hyperparameters ---
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
HIDDEN_DIM = 128
NUM_LAYERS = 4
WINDOW_SIZE = 200
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-5
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LAMBDA_CLASSIFICATION = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Structured logging: file + stdout
logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s - %(message)s',
                      handlers=[logging.FileHandler(TRAIN_LOG_FILE, mode='w'),
                                logging.StreamHandler(sys.stdout)])


def get_target_columns(manifest_path, data_dir, num_targets):
    """Select `num_targets` high‑variance numeric columns as prediction targets.

    Excludes explicit time‑derived features from the candidate pool. Returns
    (input_columns, target_columns).
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    train_files = [data_dir / item['track_features'] for item in manifest['train']]

    # Sample a subset of files to estimate statistics efficiently
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
    logging.info("--- Training Semi-Supervised Feature Extractor ---")

    # (1) Feature column selection
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)
    
    # (2) TRAIN: use pseudo‑labels (if present); VAL: original labels
    logging.info("Creating training dataloader with pseudo-labels...")
    train_dataloaders = create_prediction_dataloaders(
        manifest_path=MANIFEST_PATH, data_dir=DATA_DIR, input_cols=input_cols, target_cols=target_cols,
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
        label_dir=PSEUDO_LABELS_DIR, label_suffix='_pseudo_labels.npy'
    )
    train_loader = train_dataloaders['train']
    
    logging.info("Creating validation/test dataloaders with original labels...")
    val_test_dataloaders = create_prediction_dataloaders(
        manifest_path=MANIFEST_PATH, data_dir=DATA_DIR, input_cols=input_cols, target_cols=target_cols,
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE
    )
    val_loader = val_test_dataloaders['val']
    
    # (3) Model/optimizer/losses
    model_hyperparams = {'input_dim': len(input_cols), 'output_dim': len(target_cols), 'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS, 'dropout_rate': DROPOUT_RATE}
    model = PredictionModel(**model_hyperparams).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    recon_loss_fn = nn.HuberLoss()
    class_loss_fn = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # (4) Training/validation loop with early stopping on val recon loss
    for epoch in range(EPOCHS):
        model.train()
        total_recon_loss, total_class_loss, total_combined_loss = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for (inputs, targets), labels in pbar:
            inputs, targets, labels = inputs.to(DEVICE), targets.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            recon_preds, class_logits = model(inputs)
            
            # Multi‑task objective: reconstruction + anomaly classification
            loss_recon = recon_loss_fn(recon_preds, targets)
            loss_class = class_loss_fn(class_logits, labels)
            combined_loss = loss_recon + (LAMBDA_CLASSIFICATION * loss_class)
            
            combined_loss.backward()
            optimizer.step()
            
            total_recon_loss += loss_recon.item()
            total_class_loss += loss_class.item()
            total_combined_loss += combined_loss.item()
            pbar.set_postfix(loss=combined_loss.item())

        # Average losses across training batches
        avg_recon_loss = total_recon_loss / len(pbar)
        avg_class_loss = total_class_loss / len(pbar)

        # Validation on reconstruction only (labels may differ from pseudo‑labels)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for (inputs, targets), _ in val_pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                recon_preds, _ = model(inputs)
                loss = recon_loss_fn(recon_preds, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_pbar)
        
        logging.info(f"[Epoch {epoch+1}/{EPOCHS}] Train Recon Loss: {avg_recon_loss:.6f} | Train Class Loss: {avg_class_loss:.6f} | Val Recon Loss: {avg_val_loss:.6f}")

        # (5) Checkpoint the best model by validation reconstruction loss
        if avg_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = avg_val_loss, 0
            logging.info(f"Validation loss improved. Saving best model to {MODEL_SAVE_PATH}")
            torch.save({'model_params': model_hyperparams, 'model_state_dict': model.state_dict()}, MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    logging.info("\n--- Semi-Supervised Training Complete ---")
