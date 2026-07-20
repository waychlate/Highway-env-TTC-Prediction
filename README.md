# Highway-env TTC Prediction

Predicting **Time-to-Collision (TTC)** of a vehicle in a simulated highway environment (`highway-env`). This model uses visual frames and actions to predict the future TTC of the vehicle.

---

### 1. Model Architecture (`VideoTTCPredictor`)

- **Spatial Feature Extractor:** A pre-trained `ResNet-18` backbone (minus the final classification layer) encodes raw 64x256 pixels from each frame into a dense 512-dimensional vector.
- **Action Embedder:** Discrete actions (0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER) are mapped through a trainable `nn.Embedding` layer to a 16-dimensional vector.
- **Temporal Model:** The spatial and action features are concatenated step-by-step to form a `528`-dimensional vector sequence. This sequence is processed by a 2-layer LSTM (`hidden_dim=256`) to model temporal kinematics.
- **MLP Regression Head:** To predict a single TTC value representing the current context window, we extract the LSTM hidden state at the **very last frame** (`[-1]`). This final temporal representation is decoded into a single continuous scalar (TTC in seconds) via a Multi-Layer Perceptron.

### 2. Temporal Configuration

The model is configured to use a lookahead prediction window:

- **Context Length (`seq_len` = 20):** The model is given a history of 20 frames.
- **Prediction Horizon (`pred_horizon` = 10):** The model decodes from the end of the context sequence, but is evaluated against the ground-truth TTC value located 10 steps into the future.

---

## Dataset Format

The dataset (stored in `data/train` and `data/test`) consists of:

- **`*_visuals.npz`:** Contains raw camera visuals (`visuals` key) formatted as `(101, 150, 600, 3)` uint8 arrays.
- **`*_data.csv`:** Contains CSV tables with step-by-step simulator attributes:
  - `action`: Categorical action ID (0-4) taken at that frame.
  - `obs_ttc`: The computed Time-to-Collision (seconds) at that step. Bounded between `0.2` and `10.0`.
  - `crashed`: Binary collision flag.

---

## Environment Setup

Since this codebase relies on PyTorch 2.0.1 (standard on older HiPerGator configurations), ensure you downgrade NumPy to 1.x to avoid C-API mismatch crashes:

```bash
pip install --user "numpy<2"
```

---

## Training the Model

### 1. Local Training

To start a new training run locally:

```bash
python train.py --train-dir data/train --test-dir data/test --epochs 10 --seq-len 20 --pred-horizon 10
```

### 2. Resuming Training

The training script automatically checkpoints at the end of each epoch to `checkpoint.pth`. If the script stops, you can resume training by running without the `--restart` flag:

```bash
python train.py --train-dir data/train --test-dir data/test --epochs 10 --seq-len 20 --pred-horizon 10
```

### 3. Hyperparameters & Options

* `--hidden-dim`: Dimension of the LSTM hidden layers (default: `256`).
* `--lstm-layers`: Number of stacked LSTM layers (default: `2`).
* `--action-dim`: Dimension of the action embedding layer (default: `16`).
* `--no-actions`: Disable vehicle action inputs in the model (runs in **frames-only** mode).
* `--backbone-mode`: Set training mode for CNN backbone: `frozen` (all layers frozen), `partial` (only `layer4` trainable, default), `unfrozen` (fully trainable).
* `--backbone-lr-mult`: Learning rate multiplier for unfrozen CNN backbone layers (default: `0.1`, which multiplies the main learning rate by `0.1` for transfer learning).
* `--weight-decay`: L2 regularization strength for the AdamW optimizer (default: `1e-4`).
* `--loss-fn`: Loss criterion (`huber`, `mse`, `l1`).
* `--restart`: Use this flag if you want to explicitly ignore any existing checkpoints and start from epoch 1.

---

## Evaluation

To evaluate a trained model checkpoint:

```bash
python evaluate.py --test-dir data/test --model-path best_model.pth --seq-len 20 --pred-horizon 10
```

### Evaluation Output Metrics

- **Test MSE / RMSE:** Mean Squared Error and Root Mean Squared Error (in seconds).
- **Test MAE:** Mean Absolute Error (in seconds).
- **$R^2$ Score:** Coefficient of determination indicating how well the model predicts variance.
- **Comparison Plot:** Saves a visual line plot comparing ground-truth TTC vs. Predicted TTC over time to `ttc_predictions_comparison.png`.
