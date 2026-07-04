
# Experiment 5: Lookback Window Sensitivity Analysis

## Purpose
This experiment evaluates how the Temporal HCE-F LSTM model responds to different historical lookback windows.

## Windows Evaluated
- 3 days
- 7 days
- 14 days
- 30 days

## Input Dataset Folder
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_1B_Jena_Daily_Window_Generation\Generated_Datasets

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
