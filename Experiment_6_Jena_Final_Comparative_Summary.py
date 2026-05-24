"""
Experiment_6_Jena_Final_Comparative_Summary.py

Final comparative benchmarking summary across:
- Experiment 2: classical forecasting baselines
- Experiment 3: HCEF dense
- Experiment 3B: HCEF LSTM
- Experiment 4: ablation study
- Experiment 5: lookback window sensitivity
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments")

EXP2 = ROOT / "Experiment_2_Jena_Baseline_Forecasting"
EXP4 = ROOT / "Experiment_4_Jena_HCEF_Ablation_Study"
EXP5 = ROOT / "Experiment_5_Jena_Lookback_Window_Sensitivity"
EXP6 = ROOT / "Experiment_6_Final_Comparative_Summary"

TABLES = EXP6 / "Tables"
FIGURES = EXP6 / "Figures"
REPORTS = EXP6 / "Reports"

for d in [EXP6, TABLES, FIGURES, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD RESULTS
# ============================================================

baseline_file = EXP2 / "Tables" / "baseline_forecasting_metrics.csv"
ablation_file = EXP4 / "Tables" / "ablation_test_metrics.csv"
window_file = EXP5 / "Tables" / "lookback_window_sensitivity_metrics.csv"

baseline_df = pd.read_csv(baseline_file)
ablation_df = pd.read_csv(ablation_file)
window_df = pd.read_csv(window_file)

# ============================================================
# BEST BASELINE
# ============================================================

best_baseline = baseline_df.sort_values("test_RMSE").iloc[0]

# ============================================================
# HCEF DENSE (from Experiment 3 user output)
# ============================================================

hcef_dense = {
    "model": "HCEF_Dense",
    "test_MAE": 3.9443,
    "test_RMSE": 4.7752,
    "test_R2": 0.5988
}

# ============================================================
# HCEF LSTM (Experiment 3B)
# best from previous experiments
# ============================================================

hcef_lstm = {
    "model": "HCEF_LSTM_7d",
    "test_MAE": 3.18,
    "test_RMSE": 3.96,
    "test_R2": 0.7172
}

# ============================================================
# BEST ABLATION
# ============================================================

best_ablation = ablation_df.sort_values("test_RMSE").iloc[0]

# ============================================================
# BEST WINDOW
# ============================================================

best_window = window_df.sort_values("test_RMSE").iloc[0]

# ============================================================
# COMPARATIVE TABLE
# ============================================================

comparison = pd.DataFrame([
    {
        "Model": best_baseline["model"],
        "Category": "Classical Baseline",
        "RMSE": best_baseline["test_RMSE"],
        "MAE": best_baseline["test_MAE"],
        "R2": best_baseline["test_R2"]
    },
    {
        "Model": hcef_dense["model"],
        "Category": "Proposed HCEF Dense",
        "RMSE": hcef_dense["test_RMSE"],
        "MAE": hcef_dense["test_MAE"],
        "R2": hcef_dense["test_R2"]
    },
    {
        "Model": hcef_lstm["model"],
        "Category": "Proposed Temporal HCEF",
        "RMSE": hcef_lstm["test_RMSE"],
        "MAE": hcef_lstm["test_MAE"],
        "R2": hcef_lstm["test_R2"]
    },
    {
        "Model": best_ablation["variant"],
        "Category": "Best Ablation Variant",
        "RMSE": best_ablation["test_RMSE"],
        "MAE": best_ablation["test_MAE"],
        "R2": best_ablation["test_R2"]
    },
    {
        "Model": f"HCEF Best Window ({int(best_window['window_days'])}d)",
        "Category": "Temporal Window Sensitivity",
        "RMSE": best_window["test_RMSE"],
        "MAE": best_window["test_MAE"],
        "R2": best_window["test_R2"]
    }
])

comparison = comparison.sort_values("RMSE")

comparison.to_csv(
    TABLES / "final_model_comparison_summary.csv",
    index=False
)

# ============================================================
# FIGURES
# ============================================================

plt.figure(figsize=(10, 6))
plt.barh(comparison["Model"], comparison["RMSE"])
plt.xlabel("Test RMSE")
plt.ylabel("Model")
plt.title("Final Comparative Model Benchmark")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "final_comparison_rmse.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(comparison["Model"], comparison["MAE"])
plt.xlabel("Test MAE")
plt.ylabel("Model")
plt.title("Final Comparative MAE Benchmark")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "final_comparison_mae.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.barh(comparison["Model"], comparison["R2"])
plt.xlabel("Test R2")
plt.ylabel("Model")
plt.title("Final Comparative R2 Benchmark")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "final_comparison_r2.png", dpi=300)
plt.close()

# ============================================================
# SUMMARY JSON
# ============================================================

best_model = comparison.sort_values("RMSE").iloc[0]

summary = {
    "experiment": "Experiment_6_Final_Comparative_Summary",
    "best_model": best_model["Model"],
    "best_rmse": float(best_model["RMSE"]),
    "best_mae": float(best_model["MAE"]),
    "best_r2": float(best_model["R2"])
}

with open(REPORTS / "experiment_6_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# ============================================================
# METHODS TEXT
# ============================================================

methods_text = """
### Experiment 6: Final Comparative Benchmarking Summary

The final experiment consolidated all previous evaluations into a unified comparative benchmark. Performance from classical machine learning forecasting baselines, the HCEF dense architecture, the Temporal HCEF LSTM model, ablation variants, and lookback-window sensitivity experiments were aggregated into a single analysis framework.

Models were compared using RMSE, MAE, and coefficient of determination (R2). Lower RMSE and MAE indicate better forecasting precision, whereas higher R2 reflects stronger explained variance. This final experiment provides an integrated quantitative comparison demonstrating the relative performance of the proposed framework against both conventional baselines and internal architectural variants.
"""

with open(REPORTS / "METHODS_TEXT_Experiment_6_Final_Comparative_Summary.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

print("="*90)
print("Experiment 6 completed successfully.")
print("="*90)
print(f"Saved to: {EXP6}")
print(f"Best model: {best_model['Model']}")
print(f"Best RMSE: {best_model['RMSE']:.4f}")
print("="*90)