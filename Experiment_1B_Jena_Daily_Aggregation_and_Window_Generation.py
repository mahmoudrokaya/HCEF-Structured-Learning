"""
Experiment_1B_Jena_Daily_Aggregation_and_Window_Generation.py

Creates daily aggregated Jena climate forecasting datasets.

Task:
Predict today's mean temperature using previous n days.

Lookback windows:
3, 7, 14, 30 days
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\Baseline\jena_climate_2009_2016_Excel.xlsx"
)

RESULTS_ROOT = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments"
)

EXP_DIR = RESULTS_ROOT / "Experiment_1B_Jena_Daily_Window_Generation"

TABLES_DIR = EXP_DIR / "Tables"
DATASETS_DIR = EXP_DIR / "Generated_Datasets"
REPORTS_DIR = EXP_DIR / "Reports"

for d in [EXP_DIR, TABLES_DIR, DATASETS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOOKBACK_WINDOWS = [3, 7, 14, 30]

TARGET_COL = "T (degC)"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("Experiment 1B: Daily Aggregation and Sliding Window Generation")
print("=" * 90)

df = pd.read_excel(DATA_FILE)

# detect datetime
datetime_col = None

for c in df.columns:
    if "date" in c.lower() or "time" in c.lower():
        datetime_col = c
        break

if datetime_col is None:
    raise ValueError("Datetime column not detected.")

df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
df = df.dropna(subset=[datetime_col])

df = df.sort_values(datetime_col)

# ============================================================
# DAILY AGGREGATION
# ============================================================

df["date_only"] = df[datetime_col].dt.date

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COL not in numeric_cols:
    raise ValueError(f"{TARGET_COL} not found.")

daily_df = df.groupby("date_only")[numeric_cols].agg([
    "mean",
    "min",
    "max",
    "std"
])

# flatten columns
daily_df.columns = [
    f"{col[0]}_{col[1]}"
    for col in daily_df.columns
]

daily_df = daily_df.reset_index()

# create target
daily_df["target_temperature_today"] = daily_df[f"{TARGET_COL}_mean"]

daily_df.to_csv(
    DATASETS_DIR / "jena_daily_aggregated.csv",
    index=False
)

print(f"Daily dataset shape: {daily_df.shape}")

# ============================================================
# WINDOW GENERATION
# ============================================================

summary_rows = []

feature_cols = [
    c for c in daily_df.columns
    if c not in ["date_only", "target_temperature_today"]
]

for window in LOOKBACK_WINDOWS:

    X_list = []
    y_list = []
    date_list = []

    for i in range(window, len(daily_df)):

        window_data = daily_df.iloc[i-window:i][feature_cols].values.flatten()

        target = daily_df.iloc[i]["target_temperature_today"]

        current_date = daily_df.iloc[i]["date_only"]

        X_list.append(window_data)
        y_list.append(target)
        date_list.append(current_date)

    X_array = np.array(X_list)
    y_array = np.array(y_list)

    feature_names = []

    for lag in range(window):

        for f in feature_cols:
            feature_names.append(
                f"{f}_lag_{window-lag}"
            )

    X_df = pd.DataFrame(X_array, columns=feature_names)

    X_df["date"] = date_list
    X_df["target_temperature_today"] = y_array

    output_file = (
        DATASETS_DIR /
        f"jena_daily_window_{window}d.csv"
    )

    X_df.to_csv(output_file, index=False)

    summary_rows.append({
        "window_days": window,
        "samples": len(X_df),
        "features": len(feature_names),
        "output_file": str(output_file)
    })

    print(
        f"Window {window}d -> "
        f"samples={len(X_df)} | "
        f"features={len(feature_names)}"
    )

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    TABLES_DIR / "generated_window_datasets_summary.csv",
    index=False
)

summary_json = {
    "experiment": "Experiment_1B_Jena_Daily_Window_Generation",
    "input_file": str(DATA_FILE),
    "daily_rows": int(daily_df.shape[0]),
    "target": "target_temperature_today",
    "lookback_windows": LOOKBACK_WINDOWS,
    "generated_datasets": summary_rows
}

with open(
    REPORTS_DIR / "experiment_1b_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary_json, f, indent=4)

print("=" * 90)
print("Experiment 1B completed successfully.")
print("=" * 90)
print(f"Output folder: {EXP_DIR}")
print("=" * 90)