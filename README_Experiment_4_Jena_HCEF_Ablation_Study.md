
# Experiment 4: Jena HCE-F Ablation Study

## Purpose
This experiment quantifies the contribution of each major component in the Temporal HCE-F LSTM model.

## Input Dataset
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_1B_Jena_Daily_Window_Generation\Generated_Datasets\jena_daily_window_7d.csv

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
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_4_Jena_HCEF_Ablation_Study
