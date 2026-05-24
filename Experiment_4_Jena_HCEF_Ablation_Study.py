"""
Experiment_4_Jena_HCEF_Ablation_Study.py

Experiment 4:
Ablation study for Temporal HCE-F LSTM on Jena daily forecasting.

Ablation variants:
1. Full_Temporal_HCEF_LSTM
2. No_Contrastive_Loss
3. No_Residual_Gate
4. Single_Head_No_Ensemble
5. LSTM_Only

Input:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//New_3_Experiments//Experiment_1B_Jena_Daily_Window_Generation//Generated_Datasets//jena_daily_window_7d.csv

Output:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//New_3_Experiments//Experiment_4_Jena_HCEF_Ablation_Study
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

EXP_DIR = RESULTS_ROOT / "Experiment_4_Jena_HCEF_Ablation_Study"

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
EPOCHS = 220
PATIENCE = 35
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

LSTM_HIDDEN = 96
LSTM_LAYERS = 2
PROJECTION_DIM = 64
DROPOUT = 0.25

DEFAULT_CONTRASTIVE_WEIGHT = 0.02

ABLATIONS = [
    {
        "name": "Full_Temporal_HCEF_LSTM",
        "use_residual_gate": True,
        "use_projection": True,
        "use_ensemble": True,
        "contrastive_weight": DEFAULT_CONTRASTIVE_WEIGHT
    },
    {
        "name": "No_Contrastive_Loss",
        "use_residual_gate": True,
        "use_projection": True,
        "use_ensemble": True,
        "contrastive_weight": 0.0
    },
    {
        "name": "No_Residual_Gate",
        "use_residual_gate": False,
        "use_projection": True,
        "use_ensemble": True,
        "contrastive_weight": DEFAULT_CONTRASTIVE_WEIGHT
    },
    {
        "name": "Single_Head_No_Ensemble",
        "use_residual_gate": True,
        "use_projection": True,
        "use_ensemble": False,
        "contrastive_weight": DEFAULT_CONTRASTIVE_WEIGHT
    },
    {
        "name": "LSTM_Only",
        "use_residual_gate": False,
        "use_projection": False,
        "use_ensemble": False,
        "contrastive_weight": 0.0
    }
]

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

class AblationTemporalHCEF(nn.Module):
    def __init__(
        self,
        daily_feature_dim,
        use_residual_gate=True,
        use_projection=True,
        use_ensemble=True
    ):
        super().__init__()

        self.use_residual_gate = use_residual_gate
        self.use_projection = use_projection
        self.use_ensemble = use_ensemble

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

        if use_residual_gate:
            self.residual_gate = nn.Sequential(
                nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN),
                nn.Sigmoid()
            )
        else:
            self.residual_gate = None

        if use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(LSTM_HIDDEN, PROJECTION_DIM),
                nn.ReLU(),
                nn.Linear(PROJECTION_DIM, PROJECTION_DIM)
            )
            final_dim = PROJECTION_DIM
        else:
            self.projection_head = None
            final_dim = LSTM_HIDDEN

        self.regressor_1 = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

        if use_ensemble:
            self.regressor_2 = nn.Sequential(
                nn.Linear(final_dim, 64),
                nn.Tanh(),
                nn.Dropout(DROPOUT),
                nn.Linear(64, 1)
            )
            self.ensemble_logits = nn.Parameter(torch.zeros(2))
        else:
            self.regressor_2 = None
            self.ensemble_logits = None

    def forward(self, x):
        x_proj = self.input_projection(x)

        lstm_out, _ = self.lstm(x_proj)

        h_last = lstm_out[:, -1, :]

        if self.use_residual_gate:
            residual_source = x_proj[:, -1, :]
            gate = self.residual_gate(h_last)
            h = h_last + gate * residual_source
        else:
            h = h_last

        if self.use_projection:
            z = self.projection_head(h)
        else:
            z = h

        y1 = self.regressor_1(z).squeeze(-1)

        if self.use_ensemble:
            y2 = self.regressor_2(z).squeeze(-1)
            weights = torch.softmax(self.ensemble_logits, dim=0)
            y_hat = weights[0] * y1 + weights[1] * y2
        else:
            weights = torch.tensor([1.0, 0.0], device=x.device)
            y_hat = y1

        return y_hat, z, weights


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("=" * 90)
print("Experiment 4: Jena HCE-F Ablation Study")
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

# Chronological split
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

# Scaling using training only
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
# TRAINING FUNCTION
# ============================================================

def train_one_variant(config):
    variant_name = config["name"]

    print("-" * 90)
    print(f"Running ablation variant: {variant_name}")
    print("-" * 90)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = AblationTemporalHCEF(
        daily_feature_dim=daily_feature_dim,
        use_residual_gate=config["use_residual_gate"],
        use_projection=config["use_projection"],
        use_ensemble=config["use_ensemble"]
    ).to(DEVICE)

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

    model_path = MODELS_DIR / f"best_{variant_name}.pt"

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

            if config["contrastive_weight"] > 0:
                loss_contrastive = temporal_contrastive_loss(z, yb)
            else:
                loss_contrastive = torch.tensor(0.0, device=DEVICE)

            loss = loss_mse + config["contrastive_weight"] * loss_contrastive

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

        current_weights = torch.tensor([1.0, 0.0])
        if config["use_ensemble"]:
            current_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu()

        row = {
            "variant": variant_name,
            "epoch": epoch,
            "train_total_loss": float(np.mean(train_losses)),
            "train_mse_loss": float(np.mean(train_mse_losses)),
            "train_contrastive_loss": float(np.mean(train_contrastive_losses)),
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_SMAPE": val_metrics["SMAPE"],
            "validation_R2": val_metrics["R2"],
            "ensemble_weight_1": float(current_weights[0]),
            "ensemble_weight_2": float(current_weights[1])
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
            f"{variant_name} | Epoch {epoch:03d} | "
            f"TrainLoss={row['train_total_loss']:.4f} | "
            f"ValRMSE={val_metrics['RMSE']:.4f} | "
            f"ValMAE={val_metrics['MAE']:.4f} | "
            f"ValR2={val_metrics['R2']:.4f}"
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping: {variant_name} at epoch {epoch}. Best epoch: {best_epoch}")
            break

    training_time = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_df.to_csv(TABLES_DIR / f"training_history_{variant_name}.csv", index=False)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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

    final_weights = np.array([1.0, 0.0])
    if config["use_ensemble"]:
        final_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu().numpy()

    prediction_df = pd.DataFrame({
        "date": test_df[DATE_COL].values,
        "observed": test_targets,
        "predicted": test_preds,
        "residual": test_targets - test_preds,
        "variant": variant_name
    })

    prediction_df.to_csv(PRED_DIR / f"predictions_{variant_name}.csv", index=False)

    # Variant-specific forecast plot
    plt.figure(figsize=(14, 5))
    plt.plot(prediction_df["date"], prediction_df["observed"], label="Observed", linewidth=1.5)
    plt.plot(prediction_df["date"], prediction_df["predicted"], label=f"Predicted: {variant_name}", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title(f"Observed vs Predicted - {variant_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(FIGURES_DIR / f"observed_vs_predicted_{variant_name}.png")

    return {
        "variant": variant_name,
        "use_residual_gate": config["use_residual_gate"],
        "use_projection": config["use_projection"],
        "use_ensemble": config["use_ensemble"],
        "contrastive_weight": config["contrastive_weight"],
        "best_epoch": best_epoch,
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
# RUN ABLATIONS
# ============================================================

all_results = []

for config in ABLATIONS:
    result = train_one_variant(config)
    all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("test_RMSE", ascending=True)
results_df.to_csv(TABLES_DIR / "ablation_test_metrics.csv", index=False)

# ============================================================
# CONTRIBUTION ANALYSIS
# ============================================================

full_row = results_df[results_df["variant"] == "Full_Temporal_HCEF_LSTM"].iloc[0]

contribution_rows = []

for _, row in results_df.iterrows():
    if row["variant"] == "Full_Temporal_HCEF_LSTM":
        continue

    contribution_rows.append({
        "comparison": f"Full_Temporal_HCEF_LSTM vs {row['variant']}",
        "ablated_variant": row["variant"],
        "RMSE_difference_ablated_minus_full": float(row["test_RMSE"] - full_row["test_RMSE"]),
        "MAE_difference_ablated_minus_full": float(row["test_MAE"] - full_row["test_MAE"]),
        "R2_difference_full_minus_ablated": float(full_row["test_R2"] - row["test_R2"])
    })

contribution_df = pd.DataFrame(contribution_rows)
contribution_df.to_csv(TABLES_DIR / "component_contribution_analysis.csv", index=False)

# ============================================================
# SUMMARY FIGURES
# ============================================================

plt.figure(figsize=(10, 6))
plt.barh(results_df["variant"], results_df["test_RMSE"])
plt.xlabel("Test RMSE")
plt.ylabel("Variant")
plt.title("Ablation Study - Test RMSE")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "ablation_test_rmse.png")

plt.figure(figsize=(10, 6))
plt.barh(results_df["variant"], results_df["test_MAE"])
plt.xlabel("Test MAE")
plt.ylabel("Variant")
plt.title("Ablation Study - Test MAE")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "ablation_test_mae.png")

plt.figure(figsize=(10, 6))
plt.barh(results_df["variant"], results_df["test_R2"])
plt.xlabel("Test R2")
plt.ylabel("Variant")
plt.title("Ablation Study - Test R2")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "ablation_test_r2.png")

# Combined validation RMSE dynamics
plt.figure(figsize=(10, 6))

for variant in results_df["variant"]:
    history_file = TABLES_DIR / f"training_history_{variant}.csv"
    if history_file.exists():
        hist = pd.read_csv(history_file)
        plt.plot(hist["epoch"], hist["validation_RMSE"], label=variant)

plt.xlabel("Epoch")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE Dynamics Across Ablation Variants")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "ablation_validation_rmse_dynamics.png")

# ============================================================
# SUMMARY, README, METHODS
# ============================================================

best_variant = results_df.iloc[0].to_dict()

summary = {
    "experiment": "Experiment_4_Jena_HCEF_Ablation_Study",
    "status": "completed",
    "input_dataset": str(DATA_FILE),
    "output_folder": str(EXP_DIR),
    "device": DEVICE,
    "lookback_days": LOOKBACK_DAYS,
    "daily_feature_dim": daily_feature_dim,
    "best_variant": best_variant["variant"],
    "best_test_RMSE": float(best_variant["test_RMSE"]),
    "best_test_MAE": float(best_variant["test_MAE"]),
    "best_test_R2": float(best_variant["test_R2"]),
    "variants": ABLATIONS
}

with open(REPORTS_DIR / "experiment_4_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

readme_text = f"""
# Experiment 4: Jena HCE-F Ablation Study

## Purpose
This experiment quantifies the contribution of each major component in the Temporal HCE-F LSTM model.

## Input Dataset
{DATA_FILE}

## Ablation Variants
1. Full_Temporal_HCEF_LSTM
2. No_Contrastive_Loss
3. No_Residual_Gate
4. Single_Head_No_Ensemble
5. LSTM_Only

## Split Strategy
Chronological 70/15/15 train-validation-test split.

## Metrics
- MAE
- RMSE
- MAPE
- SMAPE
- R2

## Main Outputs
- ablation_test_metrics.csv
- component_contribution_analysis.csv
- training_history files for all variants
- prediction files for all variants
- ablation comparison figures
- validation RMSE dynamics figure

## Reproducibility
Run:

python Experiment_4_Jena_HCEF_Ablation_Study.py

## Required Packages
- pandas
- numpy
- scikit-learn
- matplotlib
- torch

## Output Folder
{EXP_DIR}
"""

with open(REPORTS_DIR / "README_Experiment_4_Jena_HCEF_Ablation_Study.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

methods_text = f"""
### Experiment 4: Ablation Study and Component Contribution Analysis

The fourth experiment quantified the contribution of each component in the proposed sequence-aware HCE-F model. The same seven-day Jena Climate forecasting dataset and chronological train-validation-test split used in the baseline and full temporal HCE-F experiments were retained to ensure direct comparability. Five controlled variants were evaluated: the full Temporal HCE-F LSTM model, a version without contrastive loss, a version without the residual-gated connection, a single-head version without ensemble fusion, and a plain LSTM-only baseline.

Each variant was trained using the same optimization settings, early-stopping criterion, and evaluation metrics. Forecasting performance was assessed using MAE, RMSE, MAPE, SMAPE, and R2. Component contribution was estimated by comparing each ablated variant against the full Temporal HCE-F model, where increases in error or decreases in R2 after removing a component were interpreted as evidence of that component's positive contribution. This ablation design directly supports the methodological claim that temporal encoding, residual stabilization, contrastive latent alignment, and ensemble fusion jointly contribute to forecasting robustness.
"""

with open(REPORTS_DIR / "METHODS_TEXT_Experiment_4_Jena_HCEF_Ablation_Study.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

print("=" * 90)
print("Experiment 4 completed successfully.")
print("=" * 90)
print(f"Output folder: {EXP_DIR}")
print(f"Best variant: {best_variant['variant']}")
print(f"Best Test RMSE: {best_variant['test_RMSE']:.4f}")
print(f"Best Test MAE: {best_variant['test_MAE']:.4f}")
print(f"Best Test R2: {best_variant['test_R2']:.4f}")
print(f"Metrics: {TABLES_DIR / 'ablation_test_metrics.csv'}")
print("=" * 90)