
# Experiment 2: Jena Baseline Forecasting Models

## Purpose
This experiment evaluates classical forecasting baselines for predicting today's mean temperature using the previous 7 days of multivariate daily climate observations.

## Input Dataset
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_1B_Jena_Daily_Window_Generation\Generated_Datasets\jena_daily_window_7d.csv

## Target
target_temperature_today

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
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_2_Jena_Baseline_Forecasting
