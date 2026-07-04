
# Experiment 1: Jena Climate Data Audit and Temporal Structure Analysis

## Purpose
This experiment audits the Jena Climate time-series dataset and characterizes its temporal structure before model training.

## Input Dataset
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\Baseline\jena_climate_2009_2016_Excel.xlsx

## Output Folder
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments\Experiment_1_Jena_Data_Audit

## Target Candidate
T (degC)

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
