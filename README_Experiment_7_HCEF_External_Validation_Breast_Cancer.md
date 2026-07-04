
# Experiment 7: HCE-F External Validation - Breast Cancer Wisconsin

## Purpose
This experiment evaluates cross-domain external validation of the HCE-F framework on the Breast Cancer Wisconsin Diagnostic Dataset.

## Input Folder
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\Breast Cancer Wisconsin Diagnostic Dataset

## Detected Dataset File
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\Breast Cancer Wisconsin Diagnostic Dataset\data.csv

## Target Column
diagnosis

## Models
- HCE-F tabular classifier
- Logistic Regression
- SVM RBF
- Random Forest
- Extra Trees
- Gradient Boosting
- MLP

## Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## Main Outputs
- dataset_summary.csv
- classical_baseline_metrics.csv
- hcef_external_validation_metrics.csv
- external_validation_model_comparison.csv
- hcef_confusion_matrix.csv
- ROC and PR curve coordinates
- model comparison figures
- training and validation dynamics figures

## Reproducibility
Run:

python Experiment_7_HCEF_External_Validation_Breast_Cancer.py
