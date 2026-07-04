
# Experiment 8: HCE-F External Validation - UCI Heart Disease

## Purpose
This experiment evaluates cross-domain external validation of the HCE-F framework on the UCI Heart Disease Dataset.

## Input Folder
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\UCI Heart Disease Data

## Detected Dataset File
D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\UCI Heart Disease Data\heart_disease_uci.csv

## Target Column
num

## Target Conversion
The UCI target is converted into binary classification:
- 0 = no disease
- >0 = disease

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

python Experiment_8_HCEF_External_Validation_Heart_Disease.py
