import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

class TTCDataset(Dataset):
    def __init__(self, data_dir, seq_len=20, pred_horizon=10, resize_shape=(64, 256), use_cache=True, return_weights=False, stack_frames=False):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.resize_shape = resize_shape
        self.use_cache = use_cache
        self.return_weights = return_weights
        self.stack_frames = stack_frames
        
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
        # Decode absolute index to episode and step index
        ep_idx = idx // self.steps_per_episode
        step_idx = idx % self.steps_per_episode
        
        if self.use_cache:
            start_idx = step_idx - self.seq_len + 1
            if start_idx < 0:
                indices = [0] * (-start_idx) + list(range(0, step_idx + 1))
                seq_tensor = self.visuals[ep_idx, indices]
                seq_actions = self.actions[ep_idx, indices]
                if self.stack_frames:
                    prev_indices = [0] * (-start_idx + 1) + list(range(0, step_idx))
                    prev_seq_tensor = self.visuals[ep_idx, prev_indices]
            else:
                seq_tensor = self.visuals[ep_idx, start_idx : step_idx + 1]
                seq_actions = self.actions[ep_idx, start_idx : step_idx + 1]
                if self.stack_frames:
                    prev_start_idx = max(0, start_idx - 1)
                    if start_idx == 0:
                        prev_indices = [0] + list(range(0, step_idx))
                        prev_seq_tensor = self.visuals[ep_idx, prev_indices]
                    else:
                        prev_seq_tensor = self.visuals[ep_idx, start_idx - 1 : step_idx]
                
            target_step_idx = min(step_idx + self.pred_horizon, self.steps_per_episode - 1)
            target_ttc = self.ttc[ep_idx, target_step_idx]
            seq_tensor = seq_tensor.float() / 255.0
            if self.stack_frames:
                prev_seq_tensor = prev_seq_tensor.float() / 255.0
        else:
            visuals, ttc, actions = self._load_episode_on_the_fly(ep_idx)
            start_idx = step_idx - self.seq_len + 1
            if start_idx < 0:
                seq_visuals = []
                seq_acts = []
                num_pads = -start_idx
                for _ in range(num_pads):
                    seq_visuals.append(visuals[0])
                    seq_acts.append(actions[0])
                for t in range(0, step_idx + 1):
                    seq_visuals.append(visuals[t])
                    seq_acts.append(actions[t])
                seq_visuals = np.stack(seq_visuals, axis=0)
                seq_actions = torch.tensor(seq_acts, dtype=torch.long)
            else:
                seq_visuals = visuals[start_idx : step_idx + 1]
                seq_actions = torch.from_numpy(actions[start_idx : step_idx + 1]).long()
                
            target_step_idx = min(step_idx + self.pred_horizon, self.steps_per_episode - 1)
            target_ttc = ttc[target_step_idx]
            seq_tensor = torch.from_numpy(seq_visuals).permute(0, 3, 1, 2).float() / 255.0
            if self.stack_frames:
                prev_indices = [max(0, i - 1) for i in (range(start_idx, step_idx + 1) if start_idx >= 0 else [0]*(-start_idx) + list(range(0, step_idx + 1)))]
                prev_visuals = visuals[prev_indices]
                prev_seq_tensor = torch.from_numpy(prev_visuals).permute(0, 3, 1, 2).float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        seq_tensor = (seq_tensor - mean) / std
        
        if self.stack_frames:
            prev_seq_tensor = (prev_seq_tensor - mean) / std
            # Concatenate frame_t and frame_t-1 along channel dimension (dim=1 of shape (20, 3, H, W)) -> (20, 6, H, W)
            seq_tensor = torch.cat([seq_tensor, prev_seq_tensor], dim=1)
        
        if self.return_weights:
            # Proportional weight: (number of real frames) / seq_len
            num_real_frames = min(step_idx + 1, self.seq_len)
            weight = num_real_frames / self.seq_len
            return seq_tensor, seq_actions, torch.tensor([target_ttc], dtype=torch.float32), torch.tensor([weight], dtype=torch.float32)
            
        return seq_tensor, seq_actions, torch.tensor([target_ttc], dtype=torch.float32)
