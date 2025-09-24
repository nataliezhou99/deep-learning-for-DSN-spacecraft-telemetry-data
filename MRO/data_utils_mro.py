import torch
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SingleTrackPredictionDataset(Dataset):
    """Dataset for a single track, separating input and target features on the fly."""
    def __init__(self, track_path, labels_path, input_cols, target_cols, window_size):
        self.window_size = window_size
        
        df = pd.read_parquet(track_path)
        
        self.input_features = df[input_cols].to_numpy(dtype=np.float32)
        self.target_features = df[target_cols].to_numpy(dtype=np.float32)
        self.labels = np.load(labels_path).astype(np.float32)

    def __len__(self):
        return max(0, len(self.input_features) - self.window_size + 1)

    def __getitem__(self, index):
        window_end = index + self.window_size
        
        input_window = self.input_features[index:window_end]
        target_at_last_step = self.target_features[window_end - 1] 
        
        label_window = self.labels[index:window_end]
        label = 1.0 if np.any(label_window) else 0.0
        
        return (torch.tensor(input_window, dtype=torch.float32), 
                torch.tensor(target_at_last_step, dtype=torch.float32)), \
               torch.tensor(label, dtype=torch.float32)

def create_prediction_dataloaders(
    manifest_path: Path, data_dir: Path, input_cols: list, target_cols: list, 
    batch_size: int, window_size: int, num_workers: int = 0, debug_mode: bool = False, seed: int = 99
):
    """Creates DataLoaders for the feature prediction task."""
    logging.info(f"Creating Prediction DataLoaders from manifest: {manifest_path}")
    with open(manifest_path, 'r') as f: manifest = json.load(f)
    
    random.seed(seed)
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        tracks = manifest[split]
        if debug_mode:
            logging.warning(f"DEBUG MODE: Using a small subset of {split} tracks.")
            random.shuffle(tracks)
            tracks = tracks[:20] if split != 'test' else tracks[:10]

        datasets = [
            SingleTrackPredictionDataset(data_dir / t['track'], data_dir / t['labels'], input_cols, target_cols, window_size)
            for t in tracks
        ]
        
        if not datasets:
            logging.warning(f"No data for split: {split}. Skipping DataLoader creation.")
            dataloaders[split] = None
            continue
            
        full_dataset = ConcatDataset(datasets)
        
        is_train = split == 'train'
        dataloaders[split] = DataLoader(
            full_dataset, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=True, drop_last=is_train
        )
        logging.info(f"Created {split} loader with {len(full_dataset)} samples.")

    logging.info("All Prediction DataLoaders created successfully.")
    return dataloaders
