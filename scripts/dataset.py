import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class TTCDataset(Dataset):
    def __init__(self, data_dir, seq_len=10, resize_shape=(64, 256)):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.resize_shape = resize_shape
        
        # Get sorted lists of CSV and NPZ files
        self.csv_files = sorted(glob.glob(os.path.join(data_dir, "*_data.csv")))
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*_visuals.npz")))
        
        assert len(self.csv_files) == len(self.npz_files), f"Mismatch: {len(self.csv_files)} CSVs and {len(self.npz_files)} NPZs"
        
        self.num_episodes = len(self.csv_files)
        self.steps_per_episode = 101 # Constant in this dataset
        self.total_samples = self.num_episodes * self.steps_per_episode
        
        # Cache for loaded and resized episodes
        # Cache stores: episode_idx -> (visuals_array, ttc_array)
        self.cache = {}

    def __len__(self):
        return self.total_samples

    def _load_episode(self, ep_idx):
        if ep_idx in self.cache:
            return self.cache[ep_idx]
            
        csv_path = self.csv_files[ep_idx]
        npz_path = self.npz_files[ep_idx]
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        ttc = df['obs_ttc'].values.astype(np.float32)
        
        # Load NPZ visuals
        npz = np.load(npz_path)
        visuals = npz['visuals'] # (101, 150, 600, 3) uint8
        
        # Resize visuals if shape is different
        if self.resize_shape is not None:
            resized_visuals = []
            for t in range(visuals.shape[0]):
                img = Image.fromarray(visuals[t])
                img = img.resize((self.resize_shape[1], self.resize_shape[0]), Image.BILINEAR)
                resized_visuals.append(np.array(img))
            visuals = np.stack(resized_visuals, axis=0) # (101, H_new, W_new, 3)
            
        self.cache[ep_idx] = (visuals, ttc)
        return visuals, ttc

    def __getitem__(self, idx):
        # Decode absolute index to episode and step index
        ep_idx = idx // self.steps_per_episode
        step_idx = idx % self.steps_per_episode
        
        visuals, ttc = self._load_episode(ep_idx)
        
        # Extract a sequence ending at step_idx of length self.seq_len
        # If step_idx < seq_len - 1, we pad by repeating the first frame
        start_idx = step_idx - self.seq_len + 1
        
        if start_idx < 0:
            seq_visuals = []
            num_pads = -start_idx
            for _ in range(num_pads):
                seq_visuals.append(visuals[0])
            for t in range(0, step_idx + 1):
                seq_visuals.append(visuals[t])
            seq_visuals = np.stack(seq_visuals, axis=0)
        else:
            seq_visuals = visuals[start_idx : step_idx + 1]
            
        # Target TTC is the value at step_idx
        target_ttc = ttc[step_idx]
        
        # Format visuals to tensor: (seq_len, channels, H, W)
        # Normalize to [0, 1]
        seq_tensor = torch.from_numpy(seq_visuals).permute(0, 3, 1, 2).float() / 255.0
        
        # ImageNet normalization
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        seq_tensor = (seq_tensor - mean) / std
        
        return seq_tensor, torch.tensor([target_ttc], dtype=torch.float32)
