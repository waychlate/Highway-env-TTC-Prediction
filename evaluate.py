import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scripts.dataset import TTCDataset;
from scripts.model import VideoTTCPredictor;

def main():
    parser = argparse.ArgumentParser(description="Evaluate TTC Predictor model")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Path to test data directory")
    parser.add_argument("--model-path", type=str, default="best_model.pth", help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length (context length)")
    parser.add_argument("--pred-horizon", type=int, default=10, help="Prediction horizon in steps")
    parser.add_argument("--resize-h", type=int, default=64, help="Height to resize frames")
    parser.add_argument("--resize-w", type=int, default=256, help="Width to resize frames")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension size")
    parser.add_argument("--plot-episodes", type=int, default=3, help="Number of episodes to plot comparisons for")
    parser.add_argument("--output-plot", type=str, default="ttc_predictions_comparison.png", help="Filename for the comparison plot")
    parser.add_argument("--action-dim", type=int, default=16, help="Action embedding dimension size")
    parser.add_argument("--no-actions", dest="use_actions", action="store_false", default=True, help="Disable vehicle action input in the model")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers (default: 2)")
    parser.add_argument("--backbone", type=str, default="custom", choices=["custom", "resnet18"], help="CNN backbone architecture (default: custom)")
    parser.add_argument("--stack-frames", action="store_true", default=False, help="Stack current and previous frames (2 frames = 6 channels) for input")
    parser.add_argument("--num-stacked-frames", type=int, default=1, help="Number of consecutive frames to stack along channel dimension (default: 1, e.g. 3 = 9 channels)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=True, help="Disable caching resized dataset in RAM")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading test dataset for evaluation...")
    dataset = TTCDataset(
        data_dir=args.test_dir,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        resize_shape=(args.resize_h, args.resize_w),
        use_cache=args.use_cache,
        stack_frames=args.stack_frames,
        num_stacked_frames=args.num_stacked_frames
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 2. Instantiate and Load Model
    in_channels = 3 * dataset.num_stacked_frames
    model = VideoTTCPredictor(hidden_dim=args.hidden_dim, action_dim=args.action_dim, use_actions=args.use_actions, num_layers=args.lstm_layers, backbone_type=args.backbone, in_channels=in_channels)
    model_path = args.model_path
    if not os.path.exists(model_path):
        fallback_path = model_path.replace(".pth", "_final.pth")
        if os.path.exists(fallback_path):
            print(f"Warning: {model_path} not found. Falling back to final epoch weights at {fallback_path}...")
            model_path = fallback_path
        else:
            print(f"Error: Model weights not found at {model_path} or fallback {fallback_path}. Please train the model first.")
            return
        
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 3. Collect predictions and targets
    all_preds = []
    all_targets = []
    
    print("Running evaluation on test cases...")
    with torch.no_grad():
        for inputs, actions, targets in loader:
            inputs = inputs.to(device)
            actions = actions.to(device)
            outputs = model(inputs, actions)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate overall metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    # R-squared metric
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    print("\n================ EVALUATION RESULTS ================")
    print(f"Test MSE:   {mse:.4f}")
    print(f"Test RMSE:  {rmse:.4f} (seconds error)")
    print(f"Test MAE:   {mae:.4f} (seconds error)")
    print(f"R² Score:   {r2:.4f}")
    print("====================================================\n")
    
    # 4. Generate visual plots for specific episodes
    # Each episode has steps_per_episode = 101 steps
    steps = dataset.steps_per_episode
    num_episodes = len(all_preds) // steps
    episodes_to_plot = min(args.plot_episodes, num_episodes)
    
    plt.figure(figsize=(12, 4 * episodes_to_plot))
    
    for i in range(episodes_to_plot):
        start_idx = i * steps
        end_idx = start_idx + steps
        
        ep_targets = all_targets[start_idx:end_idx]
        ep_preds = all_preds[start_idx:end_idx]
        
        plt.subplot(episodes_to_plot, 1, i + 1)
        plt.plot(ep_targets, label="Ground Truth (Actual TTC)", color="green", linewidth=2.5)
        plt.plot(ep_preds, label="Predicted TTC", color="red", linestyle="--", linewidth=2.0)
        plt.title(f"Episode {i+1} TTC Prediction over Time")
        plt.xlabel("Step")
        plt.ylabel("TTC (seconds)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    print(f"Saved comparison plot to: {args.output_plot}")

if __name__ == "__main__":
    main()
