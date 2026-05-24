"""
Experiment_2_Jena_Baseline_Forecasting.py

Experiment 2:
Baseline forecasting models for Jena daily temperature prediction.

Task:
Predict today's mean temperature using previous 7 days of multivariate daily climate features.

Input:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//New_3_Experiments//Experiment_1B_Jena_Daily_Window_Generation//Generated_Datasets//jena_daily_window_7d.csv

Output:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//New_3_Experiments//Experiment_2_Jena_Baseline_Forecasting
"""

from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

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

EXP_DIR = RESULTS_ROOT / "Experiment_2_Jena_Baseline_Forecasting"

TABLES_DIR = EXP_DIR / "Tables"
FIGURES_DIR = EXP_DIR / "Figures"
REPORTS_DIR = EXP_DIR / "Reports"
PREDICTIONS_DIR = EXP_DIR / "Predictions"
MODELS_DIR = EXP_DIR / "Model_Outputs"

for d in [EXP_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_temperature_today"
DATE_COL = "date"
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# HELPERS
# ============================================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 1e-8
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def evaluate_regression(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "MAPE": float(mape(y_true, y_pred)),
        "SMAPE": float(smape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }


def chronological_split(df):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def get_last_day_temperature_feature(feature_columns):
    candidates = [
        c for c in feature_columns
        if c.startswith("T (degC)_mean") and c.endswith("_lag_1")
    ]
    if candidates:
        return candidates[0]

    candidates = [
        c for c in feature_columns
        if "T (degC)_mean" in c and "lag_1" in c
    ]
    if candidates:
        return candidates[0]

    return None


def plot_predictions(dates, y_true, y_pred, model_name, filename):
    plt.figure(figsize=(14, 5))
    plt.plot(dates, y_true, label="Observed", linewidth=1.5)
    plt.plot(dates, y_pred, label="Predicted", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title(f"Observed vs Predicted Temperature - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(FIGURES_DIR / filename)


def plot_residuals(y_true, y_pred, model_name, filename):
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, s=18, alpha=0.7)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted Temperature")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot - {model_name}")
    plt.grid(alpha=0.3)
    save_fig(FIGURES_DIR / filename)


def plot_scatter(y_true, y_pred, model_name, filename):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)
    plt.xlabel("Observed Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title(f"Observed vs Predicted Scatter - {model_name}")
    plt.grid(alpha=0.3)
    save_fig(FIGURES_DIR / filename)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("Experiment 2: Jena Baseline Forecasting Models")
print("=" * 90)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

if DATE_COL not in df.columns:
    raise ValueError(f"Date column not found: {DATE_COL}")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column not found: {TARGET_COL}")

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

print(f"Loaded dataset: {DATA_FILE}")
print(f"Dataset shape: {df.shape}")
print(f"Feature count: {len(feature_cols)}")

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df, val_df, test_df = chronological_split(df)

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

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]

X_val = val_df[feature_cols]
y_val = val_df[TARGET_COL]

X_test = test_df[feature_cols]
y_test = test_df[TARGET_COL]

# ============================================================
# BASELINES
# ============================================================

models = {
    "Linear_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ]),
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=RANDOM_STATE, max_iter=10000))
    ]),
    "Random_Forest": RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    ),
    "Extra_Trees": ExtraTreesRegressor(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    ),
    "Gradient_Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3
    ),
    "MLP_Regressor": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE
        ))
    ])
}

# Persistence baseline: today's temperature = previous day mean temperature
persistence_feature = get_last_day_temperature_feature(feature_cols)

results = []
prediction_frames = []

if persistence_feature is not None:
    print(f"Using persistence feature: {persistence_feature}")

    val_pred = val_df[persistence_feature].values
    test_pred = test_df[persistence_feature].values

    val_metrics = evaluate_regression(y_val, val_pred)
    test_metrics = evaluate_regression(y_test, test_pred)

    row = {
        "model": "Persistence_Previous_Day_Temperature",
        "training_time_seconds": 0.0
    }

    for k, v in val_metrics.items():
        row[f"validation_{k}"] = v

    for k, v in test_metrics.items():
        row[f"test_{k}"] = v

    results.append(row)

    pred_df = pd.DataFrame({
        "date": test_df[DATE_COL],
        "observed": y_test.values,
        "predicted": test_pred,
        "residual": y_test.values - test_pred,
        "model": "Persistence_Previous_Day_Temperature"
    })

    prediction_frames.append(pred_df)
    pred_df.to_csv(PREDICTIONS_DIR / "predictions_Persistence_Previous_Day_Temperature.csv", index=False)

    plot_predictions(test_df[DATE_COL], y_test.values, test_pred, "Persistence_Previous_Day_Temperature", "predictions_Persistence_Previous_Day_Temperature.png")
    plot_residuals(y_test.values, test_pred, "Persistence_Previous_Day_Temperature", "residuals_Persistence_Previous_Day_Temperature.png")
    plot_scatter(y_test.values, test_pred, "Persistence_Previous_Day_Temperature", "scatter_Persistence_Previous_Day_Temperature.png")
else:
    print("Warning: Persistence feature was not detected.")

# Train ML baselines
for model_name, model in models.items():
    print("-" * 90)
    print(f"Training model: {model_name}")

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate_regression(y_val, val_pred)
    test_metrics = evaluate_regression(y_test, test_pred)

    row = {
        "model": model_name,
        "training_time_seconds": float(training_time)
    }

    for k, v in val_metrics.items():
        row[f"validation_{k}"] = v

    for k, v in test_metrics.items():
        row[f"test_{k}"] = v

    results.append(row)

    pred_df = pd.DataFrame({
        "date": test_df[DATE_COL],
        "observed": y_test.values,
        "predicted": test_pred,
        "residual": y_test.values - test_pred,
        "model": model_name
    })

    prediction_frames.append(pred_df)
    pred_df.to_csv(PREDICTIONS_DIR / f"predictions_{model_name}.csv", index=False)

    plot_predictions(test_df[DATE_COL], y_test.values, test_pred, model_name, f"predictions_{model_name}.png")
    plot_residuals(y_test.values, test_pred, model_name, f"residuals_{model_name}.png")
    plot_scatter(y_test.values, test_pred, model_name, f"scatter_{model_name}.png")

    print(
        f"{model_name} | "
        f"Val RMSE={val_metrics['RMSE']:.4f}, Val MAE={val_metrics['MAE']:.4f}, Val R2={val_metrics['R2']:.4f} | "
        f"Test RMSE={test_metrics['RMSE']:.4f}, Test MAE={test_metrics['MAE']:.4f}, Test R2={test_metrics['R2']:.4f}"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("test_RMSE", ascending=True)
results_df.to_csv(TABLES_DIR / "baseline_forecasting_metrics.csv", index=False)

if prediction_frames:
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(PREDICTIONS_DIR / "all_model_test_predictions.csv", index=False)

best_model = results_df.iloc[0].to_dict()

with open(REPORTS_DIR / "best_baseline_model.json", "w", encoding="utf-8") as f:
    json.dump(best_model, f, indent=4)

# ============================================================
# SUMMARY FIGURES
# ============================================================

plt.figure(figsize=(10, 6))
plt.barh(results_df["model"], results_df["test_RMSE"])
plt.xlabel("Test RMSE")
plt.ylabel("Model")
plt.title("Baseline Forecasting Models - Test RMSE")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "baseline_models_test_rmse.png")

plt.figure(figsize=(10, 6))
plt.barh(results_df["model"], results_df["test_MAE"])
plt.xlabel("Test MAE")
plt.ylabel("Model")
plt.title("Baseline Forecasting Models - Test MAE")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "baseline_models_test_mae.png")

plt.figure(figsize=(10, 6))
plt.barh(results_df["model"], results_df["test_R2"])
plt.xlabel("Test R2")
plt.ylabel("Model")
plt.title("Baseline Forecasting Models - Test R2")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "baseline_models_test_r2.png")

# Best model detailed comparison
best_name = best_model["model"]
best_pred_file = PREDICTIONS_DIR / f"predictions_{best_name}.csv"

if best_pred_file.exists():
    best_pred_df = pd.read_csv(best_pred_file)
    best_pred_df["date"] = pd.to_datetime(best_pred_df["date"], errors="coerce")

    plt.figure(figsize=(14, 5))
    plt.plot(best_pred_df["date"], best_pred_df["observed"], label="Observed", linewidth=1.5)
    plt.plot(best_pred_df["date"], best_pred_df["predicted"], label=f"Predicted: {best_name}", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title(f"Best Baseline Test Forecast: {best_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(FIGURES_DIR / "best_baseline_observed_vs_predicted.png")

# ============================================================
# README AND METHODS TEXT
# ============================================================

experiment_summary = {
    "experiment": "Experiment_2_Jena_Baseline_Forecasting",
    "status": "completed",
    "input_dataset": str(DATA_FILE),
    "output_folder": str(EXP_DIR),
    "target": TARGET_COL,
    "date_column": DATE_COL,
    "features": len(feature_cols),
    "samples": int(len(df)),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "split_strategy": "chronological 70/15/15 split",
    "best_baseline_model": best_name,
    "best_test_RMSE": float(best_model["test_RMSE"]),
    "best_test_MAE": float(best_model["test_MAE"]),
    "best_test_R2": float(best_model["test_R2"])
}

with open(REPORTS_DIR / "experiment_2_summary.json", "w", encoding="utf-8") as f:
    json.dump(experiment_summary, f, indent=4)

readme_text = f"""
# Experiment 2: Jena Baseline Forecasting Models

## Purpose
This experiment evaluates classical forecasting baselines for predicting today's mean temperature using the previous 7 days of multivariate daily climate observations.

## Input Dataset
{DATA_FILE}

## Target
{TARGET_COL}

## Split Strategy
Chronological split:
- 70% training
- 15% validation
- 15% testing

This avoids temporal leakage from future observations into training.

## Models
- Persistence previous-day temperature baseline
- Linear Regression
- Ridge Regression
- ElasticNet
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- MLP Regressor

## Metrics
- MAE
- RMSE
- MAPE
- SMAPE
- R2

## Main Outputs
- baseline_forecasting_metrics.csv
- chronological_split_summary.csv
- best_baseline_model.json
- per-model prediction files
- observed-vs-predicted plots
- residual plots
- model comparison plots

## Reproducibility
Run:

python Experiment_2_Jena_Baseline_Forecasting.py

## Required Packages
- pandas
- numpy
- scikit-learn
- matplotlib

## Output Folder
{EXP_DIR}
"""

with open(REPORTS_DIR / "README_Experiment_2_Jena_Baseline_Forecasting.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

methods_text = f"""
### Experiment 2: Baseline Forecasting Models

The second experiment established conventional baseline performance for the daily Jena Climate forecasting task. The generated seven-day sliding-window dataset was used as the primary modeling dataset, where each sample represented multivariate climate information from the previous seven days and the target represented the current day's mean temperature. The data were divided using a chronological 70/15/15 train-validation-test split to avoid temporal leakage.

The evaluated baselines included a persistence model that used the previous day's mean temperature as the prediction, linear regression, ridge regression, ElasticNet, random forest regression, extra trees regression, gradient boosting regression, and a multilayer perceptron regressor. Standardization was applied to models requiring scale normalization. Forecasting performance was assessed using mean absolute error, root mean squared error, mean absolute percentage error, symmetric mean absolute percentage error, and coefficient of determination. The resulting baseline metrics define the reference level against which the proposed temporal HCE-F model and later ablation experiments are compared.
"""

with open(REPORTS_DIR / "METHODS_TEXT_Experiment_2_Jena_Baseline_Forecasting.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 90)
print("Experiment 2 completed successfully.")
print("=" * 90)
print(f"Input dataset: {DATA_FILE}")
print(f"Output folder: {EXP_DIR}")
print(f"Metrics: {TABLES_DIR / 'baseline_forecasting_metrics.csv'}")
print(f"Best model: {best_name}")
print(f"Best Test RMSE: {best_model['test_RMSE']:.4f}")
print(f"Best Test MAE: {best_model['test_MAE']:.4f}")
print(f"Best Test R2: {best_model['test_R2']:.4f}")
print("=" * 90)