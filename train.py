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
    parser.add_argument("--seq-len", type=int, default=20, help="Temporal sequence length (context length)")
    parser.add_argument("--pred-horizon", type=int, default=10, help="Prediction horizon in steps")
    parser.add_argument("--resize-h", type=int, default=64, help="Height to resize frames")
    parser.add_argument("--resize-w", type=int, default=256, help="Width to resize frames")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension size")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze the ResNet feature extractor weights")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false", help="Do not freeze the ResNet weights")
    parser.add_argument("--backbone-mode", type=str, default="partial", choices=["frozen", "partial", "unfrozen"], help="Backbone training mode: frozen (all layers frozen), partial (only layer4 unfrozen, default), unfrozen (all layers unfrozen)")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1, help="Learning rate multiplier for the unfrozen backbone layers (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="File path to save the best model weights")
    parser.add_argument("--num-train-episodes", type=int, default=None, help="Number of training episodes to use (None for all)")
    parser.add_argument("--num-test-episodes", type=int, default=None, help="Number of test episodes to use (None for all)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=True, help="Disable caching resized dataset in RAM")
    parser.add_argument("--restart", action="store_true", default=False, help="Ignore existing checkpoint.pth and restart training from epoch 1")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability (default: 0.2)")
    parser.add_argument("--loss-fn", type=str, default="huber", choices=["mse", "huber", "l1"], help="Loss function to use (default: huber)")
    parser.add_argument("--use-scheduler", action="store_true", default=True, help="Use Cosine Annealing learning rate scheduler")
    parser.add_argument("--no-scheduler", dest="use_scheduler", action="store_false", help="Disable learning rate scheduler")
    parser.add_argument("--action-dim", type=int, default=16, help="Action embedding dimension size")
    parser.add_argument("--no-actions", dest="use_actions", action="store_false", default=True, help="Disable vehicle action input in the model")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer (default: 1e-4)")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Instantiate datasets
    print("Loading training dataset...")
    train_dataset = TTCDataset(
        data_dir=args.train_dir,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        resize_shape=(args.resize_h, args.resize_w),
        use_cache=args.use_cache,
        return_weights=True
    )
    
    print("Loading test dataset...")
    test_dataset = TTCDataset(
        data_dir=args.test_dir,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        resize_shape=(args.resize_h, args.resize_w),
        use_cache=args.use_cache
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
    print(f"Creating VideoTTCPredictor model with hidden_dim={args.hidden_dim}, dropout={args.dropout}, action_dim={args.action_dim}, and use_actions={args.use_actions}...")
    model = VideoTTCPredictor(hidden_dim=args.hidden_dim, dropout=args.dropout, action_dim=args.action_dim, use_actions=args.use_actions)
    
    # Set backbone gradients based on mode and flags
    backbone_mode = args.backbone_mode
    if not args.freeze_backbone:  # Overridden by --unfreeze-backbone
        backbone_mode = "unfrozen"
        
    if backbone_mode == "frozen":
        print("Freezing all CNN backbone weights...")
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
    elif backbone_mode == "partial":
        print("Freezing early CNN backbone layers, keeping only layer4 unfrozen...")
        for name, param in model.feature_extractor.named_parameters():
            if name.startswith('7.'):  # layer4 is index 7
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        print("Keeping all CNN backbone weights unfrozen (full fine-tuning)...")
        for param in model.feature_extractor.parameters():
            param.requires_grad = True
            
    model = model.to(device)
    
    # 3. Define Optimizer and Loss Function
    if args.loss_fn == "huber":
        criterion = nn.HuberLoss(delta=1.0, reduction='none')
        print("Using Huber Loss (Smooth L1 Loss) with reduction='none'")
    elif args.loss_fn == "l1":
        criterion = nn.L1Loss(reduction='none')
        print("Using L1 Loss (MAE) with reduction='none'")
    else:
        criterion = nn.MSELoss(reduction='none')
        print("Using MSE Loss with reduction='none'")
        
    # Setup parameters for optimizer, potentially with differential learning rates
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('feature_extractor.'):
                backbone_params.append(param)
            else:
                other_params.append(param)
                
    trainable_params = backbone_params + other_params
    
    if len(backbone_params) > 0 and args.backbone_lr_mult != 1.0:
        backbone_lr = args.lr * args.backbone_lr_mult
        print(f"Using differential learning rates: backbone LR = {backbone_lr:.2e}, other layers LR = {args.lr:.2e}")
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': other_params, 'lr': args.lr}
        ]
    else:
        print(f"Using uniform learning rate: {args.lr:.2e} for all layers")
        param_groups = [{'params': backbone_params + other_params, 'lr': args.lr}]
        
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print("Using Cosine Annealing learning rate scheduler")
        
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    checkpoint_path = "checkpoint.pth"
    start_epoch = 1
    best_val_loss = float('inf')
    
    if os.path.exists(checkpoint_path) and not args.restart:
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming from Epoch {start_epoch} (Best Val Loss so far: {best_val_loss:.4f})")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            
    # 4. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, actions, targets, weights) in enumerate(train_loader):
            inputs, actions, targets, weights = inputs.to(device), actions.to(device), targets.to(device), weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, actions)
            loss = (criterion(outputs, targets) * weights).mean()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        epoch_train_loss = train_loss / len(train_dataset)
        if scheduler is not None:
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for inputs, actions, targets in test_loader:
                inputs, actions, targets = inputs.to(device), actions.to(device), targets.to(device)
                outputs = model(inputs, actions)
                loss = criterion(outputs, targets).mean()
                val_loss += loss.item() * inputs.size(0)
                val_mse += ((outputs - targets) ** 2).sum().item()
                val_mae += torch.abs(outputs - targets).sum().item()
                
        epoch_val_loss = val_loss / len(test_dataset)
        epoch_val_mse = val_mse / len(test_dataset)
        epoch_val_mae = val_mae / len(test_dataset)
        
        print(f"=== Epoch {epoch} Complete ===")
        print(f"Train Loss ({args.loss_fn}): {epoch_train_loss:.4f}")
        print(f"Val Loss ({args.loss_fn}):   {epoch_val_loss:.4f}")
        print(f"Val MSE:   {epoch_val_mse:.4f} | Val RMSE: {np.sqrt(epoch_val_mse):.4f} | Val MAE: {epoch_val_mae:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved to {args.save_path} (Val Loss: {best_val_loss:.4f})")
            
        # Save checkpoint for resuming
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        print()
        
    print("Training finished!")
    
    # Safety Save: Always save the final epoch's weights
    final_save_path = args.save_path.replace(".pth", "_final.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final epoch model weights to {final_save_path}")
    
    # Only remove checkpoint if the best model or final model was successfully saved
    if os.path.exists(checkpoint_path):
        if os.path.exists(args.save_path) or os.path.exists(final_save_path):
            os.remove(checkpoint_path)
            print("Removed temporary checkpoint file.")
        else:
            print("Warning: Model files were not found. Preserving checkpoint.pth for safety.")

if __name__ == "__main__":
    main()
