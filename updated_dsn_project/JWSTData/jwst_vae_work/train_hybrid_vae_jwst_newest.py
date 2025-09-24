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
import optuna

from data_utils_hybrid_vae_jwst_newest import create_prediction_dataloaders

# --- MODEL: CNN + Bidirectional LSTM + Attention ---
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
        
        # --- ⬇️ FIX: Repeat the 'v' vector for every item in the batch ⬇️ ---
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

# --- 1. CONFIGURATION ---
DEBUG_MODE = False
PROJECT_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData/jwst_vae_work")
DATA_DIR = Path("/home/nzhou/updated_dsn_project/JWSTData/processed_diffusion_style/low_band/data_files")
OUTPUT_DIR = PROJECT_DIR / "processed_data"
MANIFEST_PATH = Path("/home/nzhou/updated_dsn_project/JWSTData/processed_diffusion_style/low_band/manifest.json")
MODEL_SAVE_PATH = PROJECT_DIR / "best_prediction_model.pth"
LOG_SAVE_PATH = PROJECT_DIR / "training_log_prediction.csv"
TRAIN_LOG_FILE = PROJECT_DIR / "training_console.log"
NUM_TARGET_FEATURES = 10 
BATCH_SIZE = 128
EPOCHS = 50
WINDOW_SIZE = 100
EARLY_STOPPING_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS = 20

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(TRAIN_LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)])
logging.info(f"Using device: {DEVICE}")
if DEBUG_MODE: logging.info("\n!!! DEBUG MODE IS ACTIVE !!!\n")

def get_target_columns(manifest_path, data_dir, num_targets):
    logging.info("Identifying target columns from training data...")
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
    logging.info(f"Selected {len(target_columns)} target columns: {target_columns}")
    return input_columns, target_columns

def objective(trial, input_dim, output_dim, train_loader, val_loader):
    params = {
        'LEARNING_RATE': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'HIDDEN_DIM': trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        'NUM_LAYERS': trial.suggest_int("num_layers", 1, 3),
        'DROPOUT_RATE': trial.suggest_float("dropout", 0.1, 0.5),
        'WEIGHT_DECAY': trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    }
    model_hyperparams = {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': params['HIDDEN_DIM'], 'num_layers': params['NUM_LAYERS'], 'dropout_rate': params['DROPOUT_RATE']}
    model = PredictionModel(**model_hyperparams).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=False)
    loss_fn = nn.HuberLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}", leave=False)
        for (inputs, targets), _ in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for (inputs, targets), _ in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Trial {trial.number} stopped early at epoch {epoch+1}.")
            break
    return best_val_loss

if __name__ == "__main__":
    logging.info("--- Training Prediction Model (BiLSTM + Attention) with Optuna ---")
    input_cols, target_cols = get_target_columns(MANIFEST_PATH, DATA_DIR, NUM_TARGET_FEATURES)
    dataloaders = create_prediction_dataloaders(
        manifest_path=MANIFEST_PATH, data_dir=DATA_DIR, input_cols=input_cols, target_cols=target_cols,
        batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, debug_mode=DEBUG_MODE
    )
    train_loader, val_loader = dataloaders['train'], dataloaders['val']
    if train_loader is None or val_loader is None: raise ValueError("Dataloaders could not be created.")
    input_dim, output_dim = len(input_cols), len(target_cols)
    logging.info(f"Model dimensions: input_dim={input_dim}, output_dim={output_dim}")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, input_dim, output_dim, train_loader, val_loader), n_trials=N_TRIALS)
    logging.info("Optimization complete.")
    logging.info(f"Best trial validation loss: {study.best_value:.6f}")
    logging.info(f"Best parameters found: {study.best_params}")
    logging.info("Training final model with best parameters...")
    best_params = study.best_params
    model_hyperparams = {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': best_params['hidden_dim'], 'num_layers': best_params['num_layers'], 'dropout_rate': best_params['dropout']}
    final_model = PredictionModel(**model_hyperparams).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, verbose=False)
    loss_fn = nn.HuberLoss()
    best_val_loss, epochs_no_improve = float('inf'), 0
    with open(LOG_SAVE_PATH, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss,lr\n")
        for epoch in range(EPOCHS):
            final_model.train()
            total_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}/{EPOCHS}")
            for (inputs, targets), _ in pbar:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                predictions = final_model(inputs)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            avg_train_loss = total_train_loss / len(pbar)
            final_model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for (inputs, targets), _ in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    predictions = final_model(inputs)
                    loss = loss_fn(predictions, targets)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"[Final Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {lr:.2e}")
            log_file.write(f"{epoch+1},{avg_train_loss},{avg_val_loss},{lr}\n")
            if avg_val_loss < best_val_loss:
                best_val_loss, epochs_no_improve = avg_val_loss, 0
                logging.info(f"Final model validation loss improved. Saving best model to {MODEL_SAVE_PATH}")
                torch.save({'model_params': model_hyperparams, 'model_state_dict': final_model.state_dict()}, MODEL_SAVE_PATH)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Final model early stopping triggered after {epoch+1} epochs.")
                    break
    logging.info("\n--- Training Complete ---")
