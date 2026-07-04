
# Experiment 3B: Temporal HCE-F LSTM Model

## Purpose
This experiment evaluates a sequence-aware HCE-F model for predicting today's mean temperature from the previous 7 days of multivariate daily climate observations.

## Input Dataset
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_1B_Jena_Daily_Window_Generation\Generated_Datasets\jena_daily_window_7d.csv

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
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_3B_Jena_HCEF_LSTM_Model
