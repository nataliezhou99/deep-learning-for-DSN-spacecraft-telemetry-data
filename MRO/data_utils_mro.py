"""
MRO Prediction Data Utilities
--------------------------------------
Purpose
    Build sliding‑window datasets and DataLoaders for the MRO telemetry
    feature‑prediction + anomaly window classification task.

Key behaviors
    • Loads per‑track features (Parquet) and per‑row binary labels (NumPy).
    • Emits windows of length `window_size`; the model predicts target features
      at the *last* timestep of each window.
    • Window label = 1 if any timestep inside the window is anomalous; else 0.

Inputs
    • manifest_path: JSON with keys 'train'|'val'|'test'; each value is a list of
      objects containing file names under `data_dir` with keys: 'track', 'labels'.
    • data_dir: base directory holding per‑track Parquet + NPY files.
    • input_cols/target_cols: column lists for model inputs/targets.

Notes
    • No functional changes; comments/docstrings only for production clarity.
    • Uses ConcatDataset to treat windows from different tracks as independent
      samples while keeping per‑track windowing logic inside the dataset class.
"""

import torch
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random

# Unified logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SingleTrackPredictionDataset(Dataset):
    """Sliding‑window dataset built from a single preprocessed track.

    Returns ((input_window[T×F_in], target_last_step[F_out]), label), where
    `label` is a scalar in {0.0, 1.0} indicating if any timestep within the
    window is anomalous.

    Parameters
    ----------
    track_path : str | Path
        Path to the per‑track Parquet file of features.
    labels_path : str | Path
        Path to the per‑track labels .npy (T×1 binary mask).
    input_cols : list[str]
        Column names used as model inputs.
    target_cols : list[str]
        Column names used as prediction targets at the last timestep.
    window_size : int
        Sliding window length (timesteps) for each example.
    """
    def __init__(self, track_path, labels_path, input_cols, target_cols, window_size):
        self.window_size = window_size
        
        # Load per‑track features; enforce float32 for GPU efficiency
        df = pd.read_parquet(track_path)
        
        self.input_features = df[input_cols].to_numpy(dtype=np.float32)
        self.target_features = df[target_cols].to_numpy(dtype=np.float32)
        self.labels = np.load(labels_path).astype(np.float32)

    def __len__(self):
        """Number of available sliding windows for this track."""
        return max(0, len(self.input_features) - self.window_size + 1)

    def __getitem__(self, index):
        """Return one training example at a given start index.

        The target is taken at the last timestep of the window; window label is
        1 if any timestep in the window is labeled anomalous.
        """
        window_end = index + self.window_size
        
        # Inputs across the full window
        input_window = self.input_features[index:window_end]
        
        # Regression/forecasting target at final step of the window
        target_at_last_step = self.target_features[window_end - 1]
        
        # Window‑level anomaly label: any positive within window → 1
        label_window = self.labels[index:window_end]
        label = 1.0 if np.any(label_window) else 0.0
        
        return (torch.tensor(input_window, dtype=torch.float32), 
                torch.tensor(target_at_last_step, dtype=torch.float32)), \
               torch.tensor(label, dtype=torch.float32)


def create_prediction_dataloaders(
    manifest_path: Path, data_dir: Path, input_cols: list, target_cols: list, 
    batch_size: int, window_size: int, num_workers: int = 0, debug_mode: bool = False, seed: int = 99
):
    """Create train/val/test DataLoaders from a manifest and on‑disk files.

    Parameters
    ----------
    manifest_path : Path
        Path to JSON with lists per split containing 'track' and 'labels' names.
    data_dir : Path
        Base directory where those files live.
    input_cols, target_cols : list[str]
        Model inputs and last‑step target columns.
    batch_size : int
        Batch size used for all splits (shuffle only for train).
    window_size : int
        Window length passed to the dataset instances.
    num_workers : int
        PyTorch DataLoader workers.
    debug_mode : bool
        If True, downsample each split for quicker iteration.
    seed : int
        RNG seed for debug subset shuffling.

    Returns
    -------
    dict[str, DataLoader | None]
        Mapping from split to DataLoader; None if no data.
    """
    logging.info(f"Creating Prediction DataLoaders from manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    random.seed(seed)
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        tracks = manifest[split]

        # Optional split downsampling for debug cycles
        if debug_mode:
            logging.warning(f"DEBUG MODE: Using a small subset of {split} tracks.")
            random.shuffle(tracks)
            tracks = tracks[:20] if split != 'test' else tracks[:10]

        # Build one dataset per track; later concatenated into a single dataset
        datasets = [
            SingleTrackPredictionDataset(
                data_dir / t['track'],
                data_dir / t['labels'],
                input_cols,
                target_cols,
                window_size,
            )
            for t in tracks
        ]
        
        if not datasets:
            logging.warning(f"No data for split: {split}. Skipping DataLoader creation.")
            dataloaders[split] = None
            continue
            
        # Concatenate per‑track datasets so each window is an independent sample
        full_dataset = ConcatDataset(datasets)
        
        is_train = split == 'train'
        dataloaders[split] = DataLoader(
            full_dataset, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=True, drop_last=is_train
        )
        logging.info(f"Created {split} loader with {len(full_dataset)} samples.")

    logging.info("All Prediction DataLoaders created successfully.")
    return dataloaders
