"""
Experiment_1_Jena_Data_Audit.py

Experiment 1:
Jena Climate Time-Series Data Audit and Temporal Structure Analysis

Input:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//Data//Baseline//jena_climate_2009_2016_Excel.xlsx

Outputs:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//New_3_Experiments//Experiment_1_Jena_Data_Audit
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ============================================================
# PATH SETTINGS
# ============================================================

DATA_FILE = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\Baseline\jena_climate_2009_2016_Excel.xlsx"
)

RESULTS_ROOT = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments"
)

EXP_DIR = RESULTS_ROOT / "Experiment_1_Jena_Data_Audit"
CODES_DIR = RESULTS_ROOT / "Codes"

TABLES_DIR = EXP_DIR / "Tables"
FIGURES_DIR = EXP_DIR / "Figures"
REPORTS_DIR = EXP_DIR / "Reports"
PROCESSED_DIR = EXP_DIR / "Processed_Data"

for d in [RESULTS_ROOT, CODES_DIR, EXP_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_CANDIDATE = "T (degC)"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def clean_column_names(df):
    df = df.copy()
    df.columns = [
        str(c).strip()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("  ", " ")
        for c in df.columns
    ]
    return df


def detect_datetime_column(df):
    candidates = [
        "Date Time",
        "DateTime",
        "datetime",
        "date_time",
        "date",
        "time",
        "timestamp"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        lower = str(c).lower()
        if "date" in lower or "time" in lower:
            return c

    return None


def detect_numeric_columns(df, exclude_cols=None):
    exclude_cols = exclude_cols or []
    numeric_cols = []

    for c in df.columns:
        if c in exclude_cols:
            continue

        converted = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.90:
            numeric_cols.append(c)

    return numeric_cols


def safe_autocorr(series, lag):
    try:
        return series.autocorr(lag=lag)
    except Exception:
        return np.nan


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("Experiment 1: Jena Climate Data Audit and Temporal Structure Analysis")
print("=" * 90)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)
df = clean_column_names(df)

print(f"Loaded file: {DATA_FILE}")
print(f"Raw shape: {df.shape}")

# ============================================================
# DATETIME HANDLING
# ============================================================

datetime_col = detect_datetime_column(df)

if datetime_col is None:
    raise ValueError(
        "No datetime column detected. Please check whether the file includes a Date Time column."
    )

df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

invalid_datetime_count = int(df[datetime_col].isna().sum())

df = df.dropna(subset=[datetime_col]).copy()
df = df.sort_values(datetime_col).reset_index(drop=True)

df["year"] = df[datetime_col].dt.year
df["month"] = df[datetime_col].dt.month
df["day"] = df[datetime_col].dt.day
df["hour"] = df[datetime_col].dt.hour
df["dayofyear"] = df[datetime_col].dt.dayofyear

# ============================================================
# NUMERIC COLUMN DETECTION
# ============================================================

numeric_cols = detect_numeric_columns(df, exclude_cols=[datetime_col])

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if TARGET_CANDIDATE not in df.columns:
    possible_temp = [c for c in df.columns if "deg" in c.lower() or "temp" in c.lower() or c.lower().startswith("t")]
    TARGET_CANDIDATE = possible_temp[0] if possible_temp else numeric_cols[0]

print(f"Datetime column: {datetime_col}")
print(f"Target candidate: {TARGET_CANDIDATE}")
print(f"Numeric variables: {len(numeric_cols)}")

# ============================================================
# BASIC AUDIT
# ============================================================

time_diffs = df[datetime_col].diff().dropna()
median_interval_seconds = float(time_diffs.dt.total_seconds().median()) if len(time_diffs) else None

basic_summary = {
    "input_file": str(DATA_FILE),
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "datetime_column": datetime_col,
    "invalid_datetime_rows_removed": invalid_datetime_count,
    "start_time": str(df[datetime_col].min()),
    "end_time": str(df[datetime_col].max()),
    "median_sampling_interval_seconds": median_interval_seconds,
    "median_sampling_interval_minutes": median_interval_seconds / 60 if median_interval_seconds else None,
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_timestamps": int(df[datetime_col].duplicated().sum()),
    "total_missing_values": int(df.isna().sum().sum()),
    "missing_percentage": float(round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 6)),
    "numeric_columns_detected": len(numeric_cols),
    "target_candidate": TARGET_CANDIDATE
}

pd.DataFrame([basic_summary]).to_csv(TABLES_DIR / "basic_dataset_summary.csv", index=False)

with open(REPORTS_DIR / "experiment_1_basic_summary.json", "w", encoding="utf-8") as f:
    json.dump(basic_summary, f, indent=4)

# ============================================================
# COLUMN PROFILE
# ============================================================

profile_rows = []

for col in df.columns:
    s = df[col]

    row = {
        "column": col,
        "dtype": str(s.dtype),
        "missing_count": int(s.isna().sum()),
        "missing_percent": round(s.isna().mean() * 100, 6),
        "unique_values": int(s.nunique(dropna=True))
    }

    if pd.api.types.is_numeric_dtype(s):
        row.update({
            "mean": round(float(s.mean()), 6) if s.notna().any() else np.nan,
            "std": round(float(s.std()), 6) if s.notna().any() else np.nan,
            "min": round(float(s.min()), 6) if s.notna().any() else np.nan,
            "median": round(float(s.median()), 6) if s.notna().any() else np.nan,
            "max": round(float(s.max()), 6) if s.notna().any() else np.nan
        })

    profile_rows.append(row)

pd.DataFrame(profile_rows).to_csv(TABLES_DIR / "column_profile.csv", index=False)

# ============================================================
# MISSING VALUES
# ============================================================

missing_table = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": np.round(df.isna().mean().values * 100, 6)
}).sort_values("missing_percent", ascending=False)

missing_table.to_csv(TABLES_DIR / "missing_values_by_column.csv", index=False)

plt.figure(figsize=(10, 8))
top_missing = missing_table.head(25)
plt.barh(top_missing["column"].astype(str), top_missing["missing_percent"])
plt.xlabel("Missing Percentage")
plt.ylabel("Column")
plt.title("Top Missing-Value Columns")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "missing_values_top_columns.png")

# ============================================================
# SAVE PROCESSED DATA
# ============================================================

processed_file = PROCESSED_DIR / "jena_climate_processed_audit_ready.csv"
df.to_csv(processed_file, index=False)

# ============================================================
# TEMPORAL COVERAGE BY YEAR AND MONTH
# ============================================================

year_counts = df["year"].value_counts().sort_index().reset_index()
year_counts.columns = ["year", "row_count"]
year_counts.to_csv(TABLES_DIR / "records_by_year.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(year_counts["year"].astype(str), year_counts["row_count"])
plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.title("Jena Climate Records by Year")
plt.grid(axis="y", alpha=0.3)
save_fig(FIGURES_DIR / "records_by_year.png")

monthly_counts = df.groupby(["year", "month"]).size().reset_index(name="row_count")
monthly_counts.to_csv(TABLES_DIR / "records_by_year_month.csv", index=False)

# ============================================================
# TARGET TEMPORAL TREND
# ============================================================

target_series = pd.to_numeric(df[TARGET_CANDIDATE], errors="coerce")

plt.figure(figsize=(14, 5))
plt.plot(df[datetime_col], target_series, linewidth=0.5)
plt.xlabel("Date Time")
plt.ylabel(TARGET_CANDIDATE)
plt.title(f"Temporal Trend of {TARGET_CANDIDATE}")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "target_temporal_trend.png")

monthly_target = df.set_index(datetime_col)[TARGET_CANDIDATE].resample("M").mean().reset_index()
monthly_target.to_csv(TABLES_DIR / "monthly_target_mean.csv", index=False)

plt.figure(figsize=(14, 5))
plt.plot(monthly_target[datetime_col], monthly_target[TARGET_CANDIDATE], marker="o", linewidth=1)
plt.xlabel("Date Time")
plt.ylabel(f"Monthly Mean {TARGET_CANDIDATE}")
plt.title(f"Monthly Mean Trend of {TARGET_CANDIDATE}")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "monthly_target_mean_trend.png")

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

numeric_df = df[numeric_cols].copy()
numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

corr = numeric_df.corr(numeric_only=True)
corr.to_csv(TABLES_DIR / "numeric_correlation_matrix.csv")

plt.figure(figsize=(12, 10))
plt.imshow(corr.values, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
plt.title("Correlation Matrix of Jena Climate Numerical Variables")
save_fig(FIGURES_DIR / "numeric_correlation_matrix.png")

if TARGET_CANDIDATE in corr.columns:
    target_corr = corr[TARGET_CANDIDATE].drop(TARGET_CANDIDATE).sort_values(
        key=lambda x: np.abs(x),
        ascending=False
    )
    target_corr_df = target_corr.reset_index()
    target_corr_df.columns = ["feature", "correlation_with_target"]
    target_corr_df.to_csv(TABLES_DIR / "feature_target_correlation.csv", index=False)

    plt.figure(figsize=(9, 7))
    top_corr = target_corr_df.head(20)
    plt.barh(top_corr["feature"], top_corr["correlation_with_target"])
    plt.xlabel(f"Correlation with {TARGET_CANDIDATE}")
    plt.ylabel("Feature")
    plt.title(f"Top Correlations with {TARGET_CANDIDATE}")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    save_fig(FIGURES_DIR / "top_feature_target_correlations.png")

# ============================================================
# AUTOCORRELATION ANALYSIS
# ============================================================

lags = [1, 6, 12, 24, 72, 144, 288, 1008]
autocorr_rows = []

for lag in lags:
    autocorr_rows.append({
        "target": TARGET_CANDIDATE,
        "lag_steps": lag,
        "autocorrelation": safe_autocorr(target_series, lag)
    })

autocorr_df = pd.DataFrame(autocorr_rows)
autocorr_df.to_csv(TABLES_DIR / "target_autocorrelation_selected_lags.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(autocorr_df["lag_steps"], autocorr_df["autocorrelation"], marker="o")
plt.xlabel("Lag Steps")
plt.ylabel("Autocorrelation")
plt.title(f"Selected-Lag Autocorrelation of {TARGET_CANDIDATE}")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "target_autocorrelation_selected_lags.png")

full_lags = list(range(1, 289))
full_autocorr = [safe_autocorr(target_series, lag) for lag in full_lags]

full_autocorr_df = pd.DataFrame({
    "lag_steps": full_lags,
    "autocorrelation": full_autocorr
})
full_autocorr_df.to_csv(TABLES_DIR / "target_autocorrelation_1_to_288_lags.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(full_lags, full_autocorr, linewidth=1)
plt.xlabel("Lag Steps")
plt.ylabel("Autocorrelation")
plt.title(f"Autocorrelation of {TARGET_CANDIDATE}, Lags 1-288")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "target_autocorrelation_1_to_288_lags.png")

# ============================================================
# LAGGED TARGET CORRELATION
# ============================================================

lag_rows = []

for lag in lags:
    shifted = target_series.shift(lag)
    valid = target_series.notna() & shifted.notna()
    lag_corr = target_series[valid].corr(shifted[valid])

    lag_rows.append({
        "target": TARGET_CANDIDATE,
        "lag_steps": lag,
        "lag_correlation": lag_corr
    })

pd.DataFrame(lag_rows).to_csv(TABLES_DIR / "target_lag_correlation.csv", index=False)

# ============================================================
# PCA AND t-SNE STRUCTURE
# ============================================================

X_num = numeric_df.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "year": df["year"],
    "month": df["month"],
    TARGET_CANDIDATE: target_series
})
pca_df.to_csv(TABLES_DIR / "pca_coordinates.csv", index=False)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df[TARGET_CANDIDATE],
    s=4,
    alpha=0.6
)
plt.colorbar(scatter, label=TARGET_CANDIDATE)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)")
plt.title("PCA Structure Colored by Temperature")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "pca_temperature_structure.png")

sample_size = min(5000, X_scaled.shape[0])
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(X_scaled.shape[0], sample_size, replace=False)

tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    perplexity=30,
    learning_rate="auto",
    init="pca"
)

X_tsne = tsne.fit_transform(X_scaled[sample_idx])

tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "year": df.iloc[sample_idx]["year"].values,
    "month": df.iloc[sample_idx]["month"].values,
    TARGET_CANDIDATE: target_series.iloc[sample_idx].values
})
tsne_df.to_csv(TABLES_DIR / "tsne_coordinates_sample.csv", index=False)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    tsne_df["TSNE1"],
    tsne_df["TSNE2"],
    c=tsne_df[TARGET_CANDIDATE],
    s=5,
    alpha=0.6
)
plt.colorbar(scatter, label=TARGET_CANDIDATE)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Structure Colored by Temperature")
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "tsne_temperature_structure.png")

# ============================================================
# SEASONAL FEATURE SUMMARY
# ============================================================

season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}
df["season"] = df["month"].map(season_map)

season_summary = df.groupby("season")[numeric_cols].agg(["mean", "std", "min", "max"])
season_summary.to_csv(TABLES_DIR / "seasonal_numeric_summary.csv")

season_target = df.groupby("season")[TARGET_CANDIDATE].agg(["mean", "std", "min", "max"]).reset_index()
season_target.to_csv(TABLES_DIR / "seasonal_target_summary.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(season_target["season"], season_target["mean"])
plt.xlabel("Season")
plt.ylabel(f"Mean {TARGET_CANDIDATE}")
plt.title(f"Seasonal Mean of {TARGET_CANDIDATE}")
plt.grid(axis="y", alpha=0.3)
save_fig(FIGURES_DIR / "seasonal_target_mean.png")

# ============================================================
# FINAL SUMMARY, README, METHODS TEXT
# ============================================================

final_summary = {
    "experiment": "Experiment_1_Jena_Data_Audit",
    "status": "completed",
    "input_file": str(DATA_FILE),
    "output_folder": str(EXP_DIR),
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "datetime_column": datetime_col,
    "target_candidate": TARGET_CANDIDATE,
    "start_time": str(df[datetime_col].min()),
    "end_time": str(df[datetime_col].max()),
    "median_sampling_interval_minutes": basic_summary["median_sampling_interval_minutes"],
    "numeric_columns_detected": len(numeric_cols),
    "duplicate_rows": basic_summary["duplicate_rows"],
    "duplicate_timestamps": basic_summary["duplicate_timestamps"],
    "missing_percentage": basic_summary["missing_percentage"],
    "pca_explained_variance_ratio": list(map(float, pca.explained_variance_ratio_)),
    "processed_file": str(processed_file)
}

with open(REPORTS_DIR / "experiment_1_final_summary.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=4)

readme_text = f"""
# Experiment 1: Jena Climate Data Audit and Temporal Structure Analysis

## Purpose
This experiment audits the Jena Climate time-series dataset and characterizes its temporal structure before model training.

## Input Dataset
{DATA_FILE}

## Output Folder
{EXP_DIR}

## Target Candidate
{TARGET_CANDIDATE}

## Main Outputs
- basic_dataset_summary.csv
- column_profile.csv
- missing_values_by_column.csv
- records_by_year.csv
- records_by_year_month.csv
- numeric_correlation_matrix.csv
- feature_target_correlation.csv
- target_autocorrelation_selected_lags.csv
- target_autocorrelation_1_to_288_lags.csv
- target_lag_correlation.csv
- pca_coordinates.csv
- tsne_coordinates_sample.csv
- seasonal_numeric_summary.csv
- seasonal_target_summary.csv
- jena_climate_processed_audit_ready.csv

## Figures
- target_temporal_trend.png
- monthly_target_mean_trend.png
- numeric_correlation_matrix.png
- top_feature_target_correlations.png
- target_autocorrelation_selected_lags.png
- target_autocorrelation_1_to_288_lags.png
- pca_temperature_structure.png
- tsne_temperature_structure.png
- seasonal_target_mean.png

## Reproducibility
Run:

python Experiment_1_Jena_Data_Audit.py

## Required Packages
- pandas
- numpy
- scikit-learn
- matplotlib
- openpyxl
"""

with open(REPORTS_DIR / "README_Experiment_1_Jena_Data_Audit.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

methods_text = f"""
### Experiment 1: Jena Climate Data Audit and Temporal Structure Analysis

The first experiment audited the Jena Climate multivariate time-series dataset before model development. The dataset was loaded from `jena_climate_2009_2016_Excel.xlsx`, located in the baseline data folder. The datetime column was automatically detected, converted to a standard datetime format, and used to sort the observations chronologically. Invalid datetime rows, duplicate timestamps, missing values, and sampling intervals were quantified to verify temporal consistency.

The experiment then characterized the temporal and statistical structure of the dataset. Numerical variables were detected and converted to numeric format when possible, and `T (degC)` was used as the primary target candidate for subsequent forecasting experiments. The audit generated column-level descriptive statistics, missing-value summaries, yearly and monthly record counts, temporal temperature trends, monthly mean temperature trends, correlation matrices, feature-target correlations, selected-lag autocorrelation, full 1-to-288 lag autocorrelation, lagged target correlations, PCA projections, t-SNE projections, and seasonal summaries. These outputs establish the empirical basis for later baseline forecasting, temporal HCE-F model evaluation, ablation analysis, and robustness testing.
"""

with open(REPORTS_DIR / "METHODS_TEXT_Experiment_1_Jena_Data_Audit.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

print("=" * 90)
print("Experiment 1 completed successfully.")
print("=" * 90)
print(f"Input file: {DATA_FILE}")
print(f"Output folder: {EXP_DIR}")
print(f"Processed file: {processed_file}")
print(f"Summary: {REPORTS_DIR / 'experiment_1_final_summary.json'}")
print("=" * 90)