import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class TTCDataset(Dataset):
    def __init__(self, data_dir, seq_len=20, pred_horizon=10, resize_shape=(64, 256), use_cache=True, return_weights=False, stack_frames=False, num_stacked_frames=1):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.resize_shape = resize_shape
        self.use_cache = use_cache
        self.return_weights = return_weights
        
        if num_stacked_frames > 1:
            self.num_stacked_frames = num_stacked_frames
        elif stack_frames:
            self.num_stacked_frames = 2
        else:
            self.num_stacked_frames = 1
            
        self.stack_frames = self.num_stacked_frames > 1
        
        # Get sorted lists of CSV and NPZ files
        self.csv_files = sorted(glob.glob(os.path.join(data_dir, "*_data.csv")))
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*_visuals.npz")))
        
        assert len(self.csv_files) == len(self.npz_files), f"Mismatch: {len(self.csv_files)} CSVs and {len(self.npz_files)} NPZs"
        
        self.num_episodes = len(self.csv_files)
        self.steps_per_episode = 101 # Constant in this dataset
        self.total_samples = self.num_episodes * self.steps_per_episode
        
        if self.use_cache:
            print(f"Preloading and resizing {self.num_episodes} episodes in parallel from {data_dir}...")
            # Pre-allocate shared memory PyTorch tensors to prevent copy-on-write RAM inflation
            self.visuals = torch.zeros((self.num_episodes, 101, 3, self.resize_shape[0], self.resize_shape[1]), dtype=torch.uint8)
            self.ttc = torch.zeros((self.num_episodes, 101), dtype=torch.float32)
            self.actions = torch.zeros((self.num_episodes, 101), dtype=torch.long)
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                executor.map(self._preload_episode, range(self.num_episodes))
            print("Preloading complete!")
        else:
            # LRU-like single cache to prevent 101x redundant loading when caching is disabled
            self.last_ep_idx = -1
            self.last_visuals = None
            self.last_ttc = None
            self.last_actions = None

    def _preload_episode(self, ep_idx):
        csv_path = self.csv_files[ep_idx]
        npz_path = self.npz_files[ep_idx]
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        ttc = df['obs_ttc'].values.astype(np.float32)
        actions = df['action'].values.astype(np.int64)
        
        # Load NPZ visuals
        npz = np.load(npz_path)
        visuals = npz['visuals'] # (101, 150, 600, 3) uint8
        
        # Resize visuals
        resized_visuals = []
        for t in range(visuals.shape[0]):
            img = Image.fromarray(visuals[t])
            img = img.resize((self.resize_shape[1], self.resize_shape[0]), Image.BILINEAR)
            resized_visuals.append(np.array(img))
        visuals_resized = np.stack(resized_visuals, axis=0) # (101, H, W, 3)
        
        # Permute to channels-first (101, 3, H, W) and store
        self.visuals[ep_idx] = torch.from_numpy(visuals_resized).permute(0, 3, 1, 2)
        self.ttc[ep_idx] = torch.from_numpy(ttc)
        self.actions[ep_idx] = torch.from_numpy(actions)

    def _load_episode_on_the_fly(self, ep_idx):
        if ep_idx == self.last_ep_idx:
            return self.last_visuals, self.last_ttc, self.last_actions
            
        csv_path = self.csv_files[ep_idx]
        npz_path = self.npz_files[ep_idx]
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        ttc = df['obs_ttc'].values.astype(np.float32)
        actions = df['action'].values.astype(np.int64)
        
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
            
        self.last_ep_idx = ep_idx
        self.last_visuals = visuals
        self.last_ttc = ttc
        self.last_actions = actions
        return visuals, ttc, actions

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        ep_idx = idx // self.steps_per_episode
        step_idx = idx % self.steps_per_episode
        
        start_idx = step_idx - self.seq_len + 1
        base_indices = list(range(max(0, start_idx), step_idx + 1))
        if start_idx < 0:
            base_indices = [0] * (-start_idx) + base_indices
            
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if self.use_cache:
            seq_tensors = []
            for k in range(self.num_stacked_frames):
                k_indices = [max(0, idx_val - k) for idx_val in base_indices]
                k_tensor = self.visuals[ep_idx, k_indices].float() / 255.0
                k_tensor = (k_tensor - mean) / std
                seq_tensors.append(k_tensor)
                
            seq_tensor = torch.cat(seq_tensors, dim=1) # (20, 3*num_stacked_frames, H, W)
            seq_actions = self.actions[ep_idx, base_indices]
            
            target_step_idx = min(step_idx + self.pred_horizon, self.steps_per_episode - 1)
            target_ttc = self.ttc[ep_idx, target_step_idx]
        else:
            visuals, ttc, actions = self._load_episode_on_the_fly(ep_idx)
            seq_tensors = []
            for k in range(self.num_stacked_frames):
                k_indices = [max(0, idx_val - k) for idx_val in base_indices]
                k_visuals = visuals[k_indices]
                k_tensor = torch.from_numpy(k_visuals).permute(0, 3, 1, 2).float() / 255.0
                k_tensor = (k_tensor - mean) / std
                seq_tensors.append(k_tensor)
                
            seq_tensor = torch.cat(seq_tensors, dim=1)
            seq_actions = torch.from_numpy(actions[base_indices]).long()
            
            target_step_idx = min(step_idx + self.pred_horizon, self.steps_per_episode - 1)
            target_ttc = ttc[target_step_idx]
        
        if self.return_weights:
            num_real_frames = min(step_idx + 1, self.seq_len)
            weight = num_real_frames / self.seq_len
            return seq_tensor, seq_actions, torch.tensor([target_ttc], dtype=torch.float32), torch.tensor([weight], dtype=torch.float32)
            
        return seq_tensor, seq_actions, torch.tensor([target_ttc], dtype=torch.float32)
