import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# PATHS
# =====================================================

DATA_FILE = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_1B_Jena_Daily_Window_Generation\Generated_Datasets\jena_daily_window_7d.csv"
)

RESULTS_ROOT = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments"
)

EXP_DIR = RESULTS_ROOT / "Experiment_3_Jena_HCEF_Model"

TABLES_DIR = EXP_DIR / "Tables"
FIGURES_DIR = EXP_DIR / "Figures"
MODELS_DIR = EXP_DIR / "Models"
REPORTS_DIR = EXP_DIR / "Reports"
PRED_DIR = EXP_DIR / "Predictions"

for d in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_temperature_today"
DATE_COL = "date"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 7
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
HIDDEN_DIM = 128

# =====================================================
# DATA
# =====================================================

df = pd.read_csv(DATA_FILE)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in [DATE_COL, TARGET_COL]]

X = df[feature_cols].values
y = df[TARGET_COL].values

# chronological split
n = len(df)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

dates_test = df.iloc[val_end:][DATE_COL]

# scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =====================================================
# DATASET
# =====================================================

class ClimateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    ClimateDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    ClimateDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    ClimateDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)

# =====================================================
# HCE-F MODEL
# =====================================================

class HCEFRegressor(nn.Module):
    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )

        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        h = self.encoder(x)

        z = self.projection(h)

        y_hat = self.regressor(z)

        return y_hat.squeeze()


model = HCEFRegressor(X_train.shape[1]).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

criterion = nn.MSELoss()

# =====================================================
# TRAINING
# =====================================================

train_losses = []
val_losses = []

best_val = np.inf

for epoch in range(EPOCHS):

    model.train()

    batch_losses = []

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        pred = model(xb)

        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    model.eval()

    val_batch = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)

            loss = criterion(pred, yb)

            val_batch.append(loss.item())

    val_loss = np.mean(val_batch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            model.state_dict(),
            MODELS_DIR / "best_hcef_model.pt"
        )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train={train_loss:.4f} | "
        f"Val={val_loss:.4f}"
    )

# =====================================================
# TEST
# =====================================================

model.load_state_dict(
    torch.load(
        MODELS_DIR / "best_hcef_model.pt"
    )
)

model.eval()

preds = []

with torch.no_grad():

    for xb, _ in test_loader:

        xb = xb.to(DEVICE)

        p = model(xb)

        preds.extend(
            p.cpu().numpy()
        )

preds = np.array(preds)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

mape = np.mean(
    np.abs((y_test - preds) / y_test)
) * 100

# =====================================================
# SAVE
# =====================================================

metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "MAPE": float(mape),
    "R2": float(r2)
}

pd.DataFrame([metrics]).to_csv(
    TABLES_DIR / "hcef_test_metrics.csv",
    index=False
)

pred_df = pd.DataFrame({
    "date": dates_test,
    "actual": y_test,
    "predicted": preds
})

pred_df.to_csv(
    PRED_DIR / "hcef_predictions.csv",
    index=False
)

with open(
    REPORTS_DIR / "experiment_3_summary.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

# loss plot
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("HCE-F Training Curve")
plt.grid(alpha=0.3)
plt.savefig(FIGURES_DIR / "training_curve.png", dpi=300)
plt.close()

# prediction plot
plt.figure(figsize=(14,5))
plt.plot(dates_test, y_test, label="Observed")
plt.plot(dates_test, preds, label="Predicted")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("HCE-F Forecasting Results")
plt.grid(alpha=0.3)
plt.savefig(FIGURES_DIR / "hcef_forecast.png", dpi=300)
plt.close()

print("="*90)
print("Experiment 3 completed successfully.")
print(metrics)
print("="*90)