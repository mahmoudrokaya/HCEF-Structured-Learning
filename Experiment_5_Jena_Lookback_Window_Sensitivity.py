"""
Experiment_5_Jena_Lookback_Window_Sensitivity.py

Experiment 5:
Lookback-window sensitivity analysis for the full Temporal HCE-F LSTM model.

Windows:
3, 7, 14, 30 days
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

DATASETS_DIR = (
    RESULTS_ROOT /
    "Experiment_1B_Jena_Daily_Window_Generation" /
    "Generated_Datasets"
)

EXP_DIR = RESULTS_ROOT / "Experiment_5_Jena_Lookback_Window_Sensitivity"

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

WINDOWS = [3, 7, 14, 30]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 220
PATIENCE = 35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

LSTM_HIDDEN = 96
LSTM_LAYERS = 2
PROJECTION_DIM = 64
DROPOUT = 0.25
CONTRASTIVE_WEIGHT = 0.02

# ============================================================
# HELPERS
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

    base_features = []
    for c in lag_map[lags[0]]:
        base, _ = c.rsplit("_lag_", 1)
        base_features.append(base)

    base_features = sorted(base_features)

    ordered_columns = []
    for lag in lags:
        for base in base_features:
            col = f"{base}_lag_{lag}"
            if col in feature_cols:
                ordered_columns.append(col)

    return lags, base_features, ordered_columns


def reshape_to_sequence(df, ordered_columns, n_lags, n_daily_features):
    X_flat = df[ordered_columns].values.astype(np.float32)
    return X_flat.reshape(len(df), n_lags, n_daily_features)


def temporal_contrastive_loss(z, y, temperature=0.2):
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

    return -(target_prob * pred_log_prob).sum(dim=1).mean()


# ============================================================
# DATASET
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
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0
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


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def run_window_experiment(window_days):
    print("-" * 90)
    print(f"Running lookback window: {window_days} days")
    print("-" * 90)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_file = DATASETS_DIR / f"jena_daily_window_{window_days}d.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    df = pd.read_csv(data_file)

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

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

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

    model_path = MODELS_DIR / f"best_hcef_lstm_{window_days}d.pt"

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

        ensemble_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu().numpy()

        row = {
            "window_days": window_days,
            "epoch": epoch,
            "train_total_loss": float(np.mean(train_losses)),
            "train_mse_loss": float(np.mean(train_mse_losses)),
            "train_contrastive_loss": float(np.mean(train_contrastive_losses)),
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_SMAPE": val_metrics["SMAPE"],
            "validation_R2": val_metrics["R2"],
            "ensemble_weight_1": float(ensemble_weights[0]),
            "ensemble_weight_2": float(ensemble_weights[1])
        }

        history.append(row)

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Window {window_days:02d}d | Epoch {epoch:03d} | "
            f"ValRMSE={val_metrics['RMSE']:.4f} | "
            f"ValMAE={val_metrics['MAE']:.4f} | "
            f"ValR2={val_metrics['R2']:.4f}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping window {window_days}d at epoch {epoch}. Best epoch: {best_epoch}")
            break

    training_time = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_df.to_csv(TABLES_DIR / f"training_history_{window_days}d.csv", index=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pred, _, _ = model(xb)

            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(yb.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    test_metrics = evaluate(test_targets, test_preds)

    final_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu().numpy()

    pred_df = pd.DataFrame({
        "date": test_df[DATE_COL].values,
        "observed": test_targets,
        "predicted": test_preds,
        "residual": test_targets - test_preds,
        "window_days": window_days
    })

    pred_df.to_csv(PRED_DIR / f"predictions_{window_days}d.csv", index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(pred_df["date"], pred_df["observed"], label="Observed", linewidth=1.5)
    plt.plot(pred_df["date"], pred_df["predicted"], label=f"Predicted {window_days}d", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title(f"Observed vs Predicted Temperature - {window_days}-Day Lookback")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(FIGURES_DIR / f"observed_vs_predicted_{window_days}d.png")

    return {
        "window_days": window_days,
        "dataset_file": str(data_file),
        "samples": int(len(df)),
        "features": int(len(feature_cols)),
        "sequence_shape": f"{n_lags} x {daily_feature_dim}",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "best_epoch": int(best_epoch),
        "training_time_seconds": float(training_time),
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_SMAPE": test_metrics["SMAPE"],
        "test_R2": test_metrics["R2"],
        "ensemble_weight_1": float(final_weights[0]),
        "ensemble_weight_2": float(final_weights[1])
    }


# ============================================================
# RUN ALL WINDOWS
# ============================================================

print("=" * 90)
print("Experiment 5: Lookback Window Sensitivity Analysis")
print("=" * 90)
print(f"Device: {DEVICE}")

all_results = []

for window in WINDOWS:
    result = run_window_experiment(window)
    all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("test_RMSE", ascending=True)
results_df.to_csv(TABLES_DIR / "lookback_window_sensitivity_metrics.csv", index=False)

best_window = results_df.iloc[0].to_dict()

# ============================================================
# FIGURES
# ============================================================

plot_df = results_df.sort_values("window_days")

plt.figure(figsize=(8, 5))
plt.plot(plot_df["window_days"], plot_df["test_RMSE"], marker="o")
plt.xlabel("Lookback Window Days")
plt.ylabel("Test RMSE")
plt.title("Lookback Window Sensitivity - RMSE")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "lookback_window_rmse.png")

plt.figure(figsize=(8, 5))
plt.plot(plot_df["window_days"], plot_df["test_MAE"], marker="o")
plt.xlabel("Lookback Window Days")
plt.ylabel("Test MAE")
plt.title("Lookback Window Sensitivity - MAE")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "lookback_window_mae.png")

plt.figure(figsize=(8, 5))
plt.plot(plot_df["window_days"], plot_df["test_R2"], marker="o")
plt.xlabel("Lookback Window Days")
plt.ylabel("Test R2")
plt.title("Lookback Window Sensitivity - R2")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "lookback_window_r2.png")

plt.figure(figsize=(9, 6))
plt.barh(results_df["window_days"].astype(str) + " days", results_df["test_RMSE"])
plt.xlabel("Test RMSE")
plt.ylabel("Lookback Window")
plt.title("Lookback Window Comparison by Test RMSE")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "lookback_window_rmse_bar.png")

# Combined validation RMSE dynamics
plt.figure(figsize=(10, 6))

for window in WINDOWS:
    hist_file = TABLES_DIR / f"training_history_{window}d.csv"
    if hist_file.exists():
        hist = pd.read_csv(hist_file)
        plt.plot(hist["epoch"], hist["validation_RMSE"], label=f"{window}d")

plt.xlabel("Epoch")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE Dynamics Across Lookback Windows")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "lookback_window_validation_rmse_dynamics.png")

# ============================================================
# SUMMARY FILES
# ============================================================

summary = {
    "experiment": "Experiment_5_Jena_Lookback_Window_Sensitivity",
    "status": "completed",
    "input_dataset_folder": str(DATASETS_DIR),
    "output_folder": str(EXP_DIR),
    "device": DEVICE,
    "windows_evaluated": WINDOWS,
    "best_window_days": int(best_window["window_days"]),
    "best_test_RMSE": float(best_window["test_RMSE"]),
    "best_test_MAE": float(best_window["test_MAE"]),
    "best_test_R2": float(best_window["test_R2"])
}

with open(REPORTS_DIR / "experiment_5_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

readme_text = f"""
# Experiment 5: Lookback Window Sensitivity Analysis

## Purpose
This experiment evaluates how the Temporal HCE-F LSTM model responds to different historical lookback windows.

## Windows Evaluated
- 3 days
- 7 days
- 14 days
- 30 days

## Input Dataset Folder
{DATASETS_DIR}

## Model
Full Temporal HCE-F LSTM:
- daily input projection
- LSTM temporal encoder
- residual-gated representation
- contrastive projection head
- two-head ensemble regression output

## Metrics
- MAE
- RMSE
- MAPE
- SMAPE
- R2

## Outputs
- lookback_window_sensitivity_metrics.csv
- per-window training history files
- per-window prediction files
- RMSE, MAE, and R2 comparison figures
- validation RMSE dynamics figure

## Reproducibility
Run:

python Experiment_5_Jena_Lookback_Window_Sensitivity.py
"""

with open(REPORTS_DIR / "README_Experiment_5_Jena_Lookback_Window_Sensitivity.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

methods_text = f"""
### Experiment 5: Lookback Window Sensitivity Analysis

The fifth experiment evaluated the sensitivity of the full Temporal HCE-F LSTM model to the length of historical temporal context. Four sliding-window datasets were evaluated using 3-day, 7-day, 14-day, and 30-day lookback periods. Each dataset used the same target definition: prediction of the current day's mean temperature from previous multivariate daily climate observations.

For each lookback window, the flattened daily-window representation was reshaped into a sequence tensor preserving chronological order. The same full Temporal HCE-F architecture, consisting of daily input projection, LSTM temporal encoding, residual-gated representation, contrastive projection, and two-head ensemble regression, was trained under the same chronological train-validation-test split and optimization settings. Forecasting performance was compared using MAE, RMSE, SMAPE, MAPE, and R2. This experiment tested whether shorter or longer temporal histories provided better predictive information and assessed the robustness of the proposed model across different temporal memory lengths.
"""

with open(REPORTS_DIR / "METHODS_TEXT_Experiment_5_Jena_Lookback_Window_Sensitivity.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

print("=" * 90)
print("Experiment 5 completed successfully.")
print("=" * 90)
print(f"Output folder: {EXP_DIR}")
print(f"Best window: {int(best_window['window_days'])} days")
print(f"Best Test RMSE: {best_window['test_RMSE']:.4f}")
print(f"Best Test MAE: {best_window['test_MAE']:.4f}")
print(f"Best Test R2: {best_window['test_R2']:.4f}")
print(f"Metrics: {TABLES_DIR / 'lookback_window_sensitivity_metrics.csv'}")
print("=" * 90)