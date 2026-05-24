"""
Experiment_3B_Jena_HCEF_LSTM_Model.py

Experiment 3B:
Sequence-aware Temporal HCE-F model using LSTM.

Task:
Predict today's mean temperature from previous 7 days of multivariate daily climate features.

Input:
jena_daily_window_7d.csv

Output:
Experiment_3B_Jena_HCEF_LSTM_Model
"""

from pathlib import Path
import json
import random
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments")

DATA_FILE = (
    RESULTS_ROOT /
    "Experiment_1B_Jena_Daily_Window_Generation" /
    "Generated_Datasets" /
    "jena_daily_window_7d.csv"
)

EXP_DIR = RESULTS_ROOT / "Experiment_3B_Jena_HCEF_LSTM_Model"

TABLES_DIR = EXP_DIR / "Tables"
FIGURES_DIR = EXP_DIR / "Figures"
REPORTS_DIR = EXP_DIR / "Reports"
MODELS_DIR = EXP_DIR / "Models"
PRED_DIR = EXP_DIR / "Predictions"

for d in [EXP_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

DATE_COL = "date"
TARGET_COL = "target_temperature_today"

LOOKBACK_DAYS = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 250
PATIENCE = 35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

LSTM_HIDDEN = 96
LSTM_LAYERS = 2
PROJECTION_DIM = 64
DROPOUT = 0.25

CONTRASTIVE_WEIGHT = 0.02

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-8
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mape_safe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape_safe(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred))
    }


def parse_lag_feature_columns(feature_cols):
    """
    Expected column format:
    original_feature_lag_k
    Example:
    T (degC)_mean_lag_1
    """
    lag_map = {}

    for c in feature_cols:
        if "_lag_" not in c:
            continue

        base, lag_str = c.rsplit("_lag_", 1)

        try:
            lag = int(lag_str)
        except ValueError:
            continue

        lag_map.setdefault(lag, []).append(c)

    lags = sorted(lag_map.keys(), reverse=True)

    if len(lags) != LOOKBACK_DAYS:
        print(f"Warning: expected {LOOKBACK_DAYS} lags, detected {len(lags)} lags: {lags}")

    base_features = []

    for c in lag_map[lags[0]]:
        base, _ = c.rsplit("_lag_", 1)
        base_features.append(base)

    base_features = sorted(base_features)

    ordered_columns = []

    for lag in lags:
        current_cols = []

        for base in base_features:
            col = f"{base}_lag_{lag}"
            if col in feature_cols:
                current_cols.append(col)

        ordered_columns.extend(current_cols)

    return lags, base_features, ordered_columns


def reshape_to_sequence(df, ordered_columns, n_lags, n_daily_features):
    X_flat = df[ordered_columns].values.astype(np.float32)
    X_seq = X_flat.reshape(len(df), n_lags, n_daily_features)
    return X_seq


# ============================================================
# DATASET CLASS
# ============================================================

class ClimateSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================

class TemporalHCEFLSTM(nn.Module):
    def __init__(self, daily_feature_dim):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(daily_feature_dim, LSTM_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )

        self.lstm = nn.LSTM(
            input_size=LSTM_HIDDEN,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
            bidirectional=False
        )

        self.residual_gate = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN),
            nn.Sigmoid()
        )

        self.projection_head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, PROJECTION_DIM),
            nn.ReLU(),
            nn.Linear(PROJECTION_DIM, PROJECTION_DIM)
        )

        self.regressor_1 = nn.Sequential(
            nn.Linear(PROJECTION_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

        self.regressor_2 = nn.Sequential(
            nn.Linear(PROJECTION_DIM, 64),
            nn.Tanh(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

        self.ensemble_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        x_proj = self.input_projection(x)

        lstm_out, _ = self.lstm(x_proj)

        h_last = lstm_out[:, -1, :]

        residual_source = x_proj[:, -1, :]
        gate = self.residual_gate(h_last)

        h = h_last + gate * residual_source

        z = self.projection_head(h)

        y1 = self.regressor_1(z).squeeze(-1)
        y2 = self.regressor_2(z).squeeze(-1)

        weights = torch.softmax(self.ensemble_logits, dim=0)

        y_hat = weights[0] * y1 + weights[1] * y2

        return y_hat, z, weights


def temporal_contrastive_loss(z, y, temperature=0.2):
    """
    Regression-friendly supervised contrastive loss.
    Samples with closer target values are treated as more similar.
    """
    if z.shape[0] < 3:
        return torch.tensor(0.0, device=z.device)

    z = nn.functional.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature

    y = y.view(-1, 1)
    dist = torch.abs(y - y.T)

    sigma = torch.std(y).detach() + 1e-6
    target_sim = torch.exp(-dist / sigma)

    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)

    sim = sim[mask]
    target_sim = target_sim[mask]

    pred_log_prob = torch.log_softmax(sim.view(z.shape[0], -1), dim=1)
    target_prob = target_sim.view(z.shape[0], -1)
    target_prob = target_prob / (target_prob.sum(dim=1, keepdim=True) + 1e-8)

    loss = -(target_prob * pred_log_prob).sum(dim=1).mean()

    return loss


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("Experiment 3B: Temporal HCE-F LSTM Model")
print("=" * 90)
print(f"Device: {DEVICE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in [DATE_COL, TARGET_COL]]

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

for c in feature_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

lags, base_features, ordered_columns = parse_lag_feature_columns(feature_cols)

daily_feature_dim = len(base_features)
n_lags = len(lags)

print(f"Dataset shape: {df.shape}")
print(f"Detected lags: {lags}")
print(f"Daily feature dimension: {daily_feature_dim}")
print(f"Ordered flattened features: {len(ordered_columns)}")

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

split_summary = pd.DataFrame([
    {
        "split": "train",
        "rows": len(train_df),
        "start_date": str(train_df[DATE_COL].min()),
        "end_date": str(train_df[DATE_COL].max())
    },
    {
        "split": "validation",
        "rows": len(val_df),
        "start_date": str(val_df[DATE_COL].min()),
        "end_date": str(val_df[DATE_COL].max())
    },
    {
        "split": "test",
        "rows": len(test_df),
        "start_date": str(test_df[DATE_COL].min()),
        "end_date": str(test_df[DATE_COL].max())
    }
])
split_summary.to_csv(TABLES_DIR / "chronological_split_summary.csv", index=False)

# ============================================================
# SCALING WITHOUT TEMPORAL LEAKAGE
# ============================================================

scaler = StandardScaler()

scaler.fit(train_df[ordered_columns])

train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[ordered_columns] = scaler.transform(train_df[ordered_columns])
val_scaled[ordered_columns] = scaler.transform(val_df[ordered_columns])
test_scaled[ordered_columns] = scaler.transform(test_df[ordered_columns])

X_train = reshape_to_sequence(train_scaled, ordered_columns, n_lags, daily_feature_dim)
X_val = reshape_to_sequence(val_scaled, ordered_columns, n_lags, daily_feature_dim)
X_test = reshape_to_sequence(test_scaled, ordered_columns, n_lags, daily_feature_dim)

y_train = train_df[TARGET_COL].values.astype(np.float32)
y_val = val_df[TARGET_COL].values.astype(np.float32)
y_test = test_df[TARGET_COL].values.astype(np.float32)

train_loader = DataLoader(
    ClimateSequenceDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    ClimateSequenceDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    ClimateSequenceDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# TRAINING
# ============================================================

model = TemporalHCEFLSTM(daily_feature_dim=daily_feature_dim).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

mse_loss = nn.MSELoss()

best_val_rmse = np.inf
best_epoch = 0
epochs_without_improvement = 0

history = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()

    train_losses = []
    train_mse_losses = []
    train_contrastive_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        pred, z, weights = model(xb)

        loss_mse = mse_loss(pred, yb)
        loss_contrastive = temporal_contrastive_loss(z, yb)

        loss = loss_mse + CONTRASTIVE_WEIGHT * loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        train_losses.append(loss.item())
        train_mse_losses.append(loss_mse.item())
        train_contrastive_losses.append(loss_contrastive.item())

    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            pred, _, _ = model(xb)

            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(yb.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    val_metrics = evaluate(val_targets, val_preds)

    row = {
        "epoch": epoch,
        "train_total_loss": float(np.mean(train_losses)),
        "train_mse_loss": float(np.mean(train_mse_losses)),
        "train_contrastive_loss": float(np.mean(train_contrastive_losses)),
        "validation_MAE": val_metrics["MAE"],
        "validation_RMSE": val_metrics["RMSE"],
        "validation_SMAPE": val_metrics["SMAPE"],
        "validation_R2": val_metrics["R2"],
        "ensemble_weight_1": float(torch.softmax(model.ensemble_logits, dim=0)[0].detach().cpu().item()),
        "ensemble_weight_2": float(torch.softmax(model.ensemble_logits, dim=0)[1].detach().cpu().item())
    }

    history.append(row)

    improved = val_metrics["RMSE"] < best_val_rmse

    if improved:
        best_val_rmse = val_metrics["RMSE"]
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODELS_DIR / "best_hcef_lstm_model.pt")
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:03d} | "
        f"TrainLoss={row['train_total_loss']:.4f} | "
        f"ValRMSE={val_metrics['RMSE']:.4f} | "
        f"ValMAE={val_metrics['MAE']:.4f} | "
        f"ValR2={val_metrics['R2']:.4f} | "
        f"Weights=({row['ensemble_weight_1']:.3f}, {row['ensemble_weight_2']:.3f})"
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break

training_time = time.time() - start_time

history_df = pd.DataFrame(history)
history_df.to_csv(TABLES_DIR / "training_history.csv", index=False)

# ============================================================
# TEST EVALUATION
# ============================================================

model.load_state_dict(torch.load(MODELS_DIR / "best_hcef_lstm_model.pt", map_location=DEVICE))
model.eval()

test_preds = []
test_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred, _, weights = model(xb)

        test_preds.extend(pred.cpu().numpy())
        test_targets.extend(yb.numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)

test_metrics = evaluate(test_targets, test_preds)

final_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu().numpy()

metrics_row = {
    "model": "Temporal_HCEF_LSTM",
    "lookback_days": LOOKBACK_DAYS,
    "daily_feature_dim": daily_feature_dim,
    "sequence_shape": f"{n_lags} x {daily_feature_dim}",
    "best_epoch": best_epoch,
    "training_time_seconds": training_time,
    "test_MAE": test_metrics["MAE"],
    "test_RMSE": test_metrics["RMSE"],
    "test_MAPE": test_metrics["MAPE"],
    "test_SMAPE": test_metrics["SMAPE"],
    "test_R2": test_metrics["R2"],
    "ensemble_weight_1": float(final_weights[0]),
    "ensemble_weight_2": float(final_weights[1])
}

pd.DataFrame([metrics_row]).to_csv(TABLES_DIR / "hcef_lstm_test_metrics.csv", index=False)

pred_df = pd.DataFrame({
    "date": test_df[DATE_COL].values,
    "observed": test_targets,
    "predicted": test_preds,
    "residual": test_targets - test_preds
})

pred_df.to_csv(PRED_DIR / "hcef_lstm_test_predictions.csv", index=False)

summary = {
    "experiment": "Experiment_3B_Jena_HCEF_LSTM_Model",
    "status": "completed",
    "input_dataset": str(DATA_FILE),
    "output_folder": str(EXP_DIR),
    "device": DEVICE,
    "lookback_days": LOOKBACK_DAYS,
    "daily_feature_dim": daily_feature_dim,
    "best_epoch": best_epoch,
    "training_time_seconds": training_time,
    "test_metrics": test_metrics,
    "ensemble_weights": {
        "head_1": float(final_weights[0]),
        "head_2": float(final_weights[1])
    }
}

with open(REPORTS_DIR / "experiment_3b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# ============================================================
# FIGURES
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["train_mse_loss"], label="Training MSE")
plt.plot(history_df["epoch"], history_df["validation_RMSE"] ** 2, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Temporal HCE-F LSTM Training Dynamics")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "training_dynamics_mse.png")

plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["validation_RMSE"], label="Validation RMSE")
plt.plot(history_df["epoch"], history_df["validation_MAE"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Temporal HCE-F LSTM Validation Errors")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "validation_error_dynamics.png")

plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["ensemble_weight_1"], label="Head 1")
plt.plot(history_df["epoch"], history_df["ensemble_weight_2"], label="Head 2")
plt.xlabel("Epoch")
plt.ylabel("Ensemble Weight")
plt.title("Learned Ensemble Weight Dynamics")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "ensemble_weight_dynamics.png")

plt.figure(figsize=(14, 5))
plt.plot(pred_df["date"], pred_df["observed"], label="Observed", linewidth=1.5)
plt.plot(pred_df["date"], pred_df["predicted"], label="Predicted", linewidth=1.2)
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Temporal HCE-F LSTM Forecasting Results")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_lstm_observed_vs_predicted.png")

plt.figure(figsize=(7, 7))
plt.scatter(pred_df["observed"], pred_df["predicted"], s=18, alpha=0.7)
min_v = min(pred_df["observed"].min(), pred_df["predicted"].min())
max_v = max(pred_df["observed"].max(), pred_df["predicted"].max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
plt.xlabel("Observed Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Observed vs Predicted Scatter - Temporal HCE-F LSTM")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_lstm_observed_vs_predicted_scatter.png")

plt.figure(figsize=(8, 6))
plt.scatter(pred_df["predicted"], pred_df["residual"], s=18, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Temperature")
plt.ylabel("Residual")
plt.title("Residual Plot - Temporal HCE-F LSTM")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_lstm_residuals.png")

# ============================================================
# README AND METHODS TEXT
# ============================================================

readme_text = f"""
# Experiment 3B: Temporal HCE-F LSTM Model

## Purpose
This experiment evaluates a sequence-aware HCE-F model for predicting today's mean temperature from the previous 7 days of multivariate daily climate observations.

## Input Dataset
{DATA_FILE}

## Model Components
- Daily input projection
- LSTM temporal encoder
- Residual-gated temporal representation
- Contrastive projection head
- Two-head ensemble regression output

## Split Strategy
Chronological 70/15/15 train-validation-test split.

## Metrics
- MAE
- RMSE
- MAPE
- SMAPE
- R2

## Main Outputs
- hcef_lstm_test_metrics.csv
- training_history.csv
- hcef_lstm_test_predictions.csv
- training dynamics figures
- ensemble weight dynamics
- observed-vs-predicted figures
- residual plots
- saved best model checkpoint

## Reproducibility
Run:

python Experiment_3B_Jena_HCEF_LSTM_Model.py

## Required Packages
- pandas
- numpy
- scikit-learn
- matplotlib
- torch

## Output Folder
{EXP_DIR}
"""

with open(REPORTS_DIR / "README_Experiment_3B_Jena_HCEF_LSTM_Model.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

methods_text = f"""
### Experiment 3B: Sequence-Aware Temporal HCE-F Model

The third experiment was extended with a sequence-aware version of the HCE-F framework to better match the temporal structure of the Jena Climate forecasting task. The flattened seven-day sliding-window dataset was reshaped into a three-dimensional tensor of shape samples by days by daily features. This representation preserved the chronological ordering of the previous seven days and enabled the model to learn cross-day dependencies rather than treating the window as an unordered vector.

The proposed temporal HCE-F model consisted of a daily input projection layer, an LSTM temporal encoder, a residual-gated temporal representation, a contrastive projection head, and a two-head ensemble regression output. The residual-gated connection was used to stabilize temporal representation learning by combining the final LSTM state with the most recent projected daily state. The contrastive projection head encouraged windows with similar target temperatures to occupy nearby latent regions, while the learned ensemble fusion combined two regression heads using trainable probabilistic weights. The model was trained using a weighted sum of mean squared error and a regression-aware temporal contrastive loss. Evaluation used the same chronological train-validation-test split and the same metrics as the baseline forecasting experiment, allowing direct comparison with the conventional models.
"""

with open(REPORTS_DIR / "METHODS_TEXT_Experiment_3B_Jena_HCEF_LSTM_Model.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 90)
print("Experiment 3B completed successfully.")
print("=" * 90)
print(f"Best epoch: {best_epoch}")
print(f"Test MAE: {test_metrics['MAE']:.4f}")
print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
print(f"Test SMAPE: {test_metrics['SMAPE']:.4f}")
print(f"Test R2: {test_metrics['R2']:.4f}")
print(f"Ensemble weights: {final_weights}")
print(f"Output folder: {EXP_DIR}")
print("=" * 90)