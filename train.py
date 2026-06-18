import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scripts.dataset import TTCDataset
from scripts.model import VideoTTCPredictor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train TTC Predictor model")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Path to training data directory")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Path to test data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=10, help="Temporal sequence length")
    parser.add_argument("--resize-h", type=int, default=64, help="Height to resize frames")
    parser.add_argument("--resize-w", type=int, default=256, help="Width to resize frames")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension size")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze the ResNet feature extractor weights")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false", help="Do not freeze the ResNet weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="File path to save the best model weights")
    parser.add_argument("--num-train-episodes", type=int, default=None, help="Number of training episodes to use (None for all)")
    parser.add_argument("--num-test-episodes", type=int, default=None, help="Number of test episodes to use (None for all)")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Instantiate datasets
    print("Loading training dataset...")
    train_dataset = TTCDataset(
        data_dir=args.train_dir,
        seq_len=args.seq_len,
        resize_shape=(args.resize_h, args.resize_w)
    )
    
    print("Loading test dataset...")
    test_dataset = TTCDataset(
        data_dir=args.test_dir,
        seq_len=args.seq_len,
        resize_shape=(args.resize_h, args.resize_w)
    )
    
    # Apply limit on episodes if specified
    if args.num_train_episodes is not None:
        train_dataset.csv_files = train_dataset.csv_files[:args.num_train_episodes]
        train_dataset.npz_files = train_dataset.npz_files[:args.num_train_episodes]
        train_dataset.num_episodes = len(train_dataset.csv_files)
        train_dataset.total_samples = train_dataset.num_episodes * train_dataset.steps_per_episode
        print(f"Limiting training to {train_dataset.num_episodes} episodes ({len(train_dataset)} samples)")
        
    if args.num_test_episodes is not None:
        test_dataset.csv_files = test_dataset.csv_files[:args.num_test_episodes]
        test_dataset.npz_files = test_dataset.npz_files[:args.num_test_episodes]
        test_dataset.num_episodes = len(test_dataset.csv_files)
        test_dataset.total_samples = test_dataset.num_episodes * test_dataset.steps_per_episode
        print(f"Limiting test to {test_dataset.num_episodes} episodes ({len(test_dataset)} samples)")
        
    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Instantiate Model
    print(f"Creating VideoTTCPredictor model with hidden_dim={args.hidden_dim}...")
    model = VideoTTCPredictor(hidden_dim=args.hidden_dim)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing CNN feature extractor backbone weights...")
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
            
    model = model.to(device)
    
    # 3. Define Optimizer and Loss Function
    criterion = nn.MSELoss()
    # Only optimize parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    best_val_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_mae += torch.abs(outputs - targets).sum().item()
                
        epoch_val_loss = val_loss / len(test_dataset)
        epoch_val_mae = val_mae / len(test_dataset)
        
        print(f"=== Epoch {epoch} Complete ===")
        print(f"Train MSE: {epoch_train_loss:.4f}")
        print(f"Val MSE:   {epoch_val_loss:.4f} | Val RMSE: {np.sqrt(epoch_val_loss):.4f} | Val MAE: {epoch_val_mae:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved to {args.save_path} (Val MSE: {best_val_loss:.4f})")
        print()
        
    print("Training finished!")

if __name__ == "__main__":
    main()
