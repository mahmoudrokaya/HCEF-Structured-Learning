# HCEF – Hybrid Contrastive–Ensemble Framework for Structured Learning and Cross-Domain Generalization

## Overview

This repository contains the official implementation of the **Hybrid Contrastive–Ensemble Framework (HCE-F)** proposed in the manuscript:

### *A Unified Mathematical Framework for Tensorized Residual, Contrastive, and Ensemble Learning for Structured Prediction and Cross-Domain Generalization*

HCE-F is a unified machine learning framework designed for robust structured learning under heterogeneous data conditions. The framework integrates:

- **Tensorized Residual Encoding**
- **Supervised Contrastive Projection Learning**
- **Probabilistic Ensemble Fusion**

within a single optimization pipeline.

The proposed method is designed to support:

- robust prediction under noisy structured data,
- stable learning dynamics,
- cross-domain transferability,
- interpretable representation learning,
- efficient deployment under moderate computational resources.

---

# Repository Structure

```bash
HCEF-Structured-Learning/
│
├── Data/
│
├── Baseline/
│   └── jena_climate_2009_2016_Excel.xlsx
│
├── Breast Cancer Wisconsin Diagnostic Dataset/
│
├── UCI Heart Disease Data/
│
├── New_3_Experiments/
│   ├── Codes/
│   ├── Results/
│   ├── Tables/
│   ├── Figures/
│   ├── Logs/
│   └── Models/
│
├── README.md
└── requirements.txt
Dataset Information
1. Internal Dataset
Jena Climate Dataset

Used as the primary benchmark for sequential structured forecasting experiments.

Source:

Max Planck Institute for Biogeochemistry

Official URL:

https://www.kaggle.com/datasets/mnassrib/jena-climate

Original dataset:

jena_climate_2009_2016_Excel.xlsx

Used for:

daily aggregation
temporal window generation
forecasting experiments
sensitivity analysis
2. External Validation Dataset A
Breast Cancer Wisconsin Diagnostic Dataset

Used for external cross-domain classification validation.

Source:

UCI Machine Learning Repository

DOI:
https://doi.org/10.24432/C5DW2B

URL:
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

3. External Validation Dataset B
UCI Heart Disease Dataset

Used for independent external validation under domain shift.

Source:

UCI Machine Learning Repository

DOI:
https://doi.org/10.24432/C52P4X

URL:
https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Code Information

All experiment scripts are located in:

New_3_Experiments/Codes/

Main scripts include:

Experiment_1B_Jena_Daily_Window_Generation.py

Purpose:

preprocess Jena climate data
generate sliding temporal windows

Generated windows:

3-day
7-day
14-day
30-day
Experiment_2_Baseline_Forecasting.py

Baseline forecasting models:

Linear Regression
Ridge Regression
ElasticNet
Random Forest
Extra Trees
Gradient Boosting
MLP Regressor
Experiment_3_HCEF_Dense_Model.py

Dense HCE-F forecasting model:

tensorized residual representation
supervised contrastive embedding
probabilistic fusion
Experiment_3B_HCEF_LSTM.py

Temporal HCE-F-LSTM implementation.

Best-performing internal model.

Experiment_4_HCEF_Ablation_Study.py

Evaluates contribution of:

contrastive loss
residual temporal gate
probabilistic ensemble fusion
Experiment_5_Window_Sensitivity.py

Evaluates performance across temporal lookback windows.

Experiment_6_Final_Comparative_Summary.py

Final comparative aggregation of all internal experiments.

Experiment_7_HCEF_External_Validation_Breast_Cancer.py

External validation on:

Breast Cancer Wisconsin Diagnostic dataset.

Experiment_8_HCEF_External_Validation_Heart_Disease.py

External validation on:

UCI Heart Disease dataset.

Installation Requirements

Recommended Python version:

Python 3.10+

Required libraries:

numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
scipy
openpyxl
joblib
seaborn

Install using:

pip install -r requirements.txt
Usage Instructions
Step 1 — Clone repository
git clone https://github.com/mahmoudrokaya/HCEF-Structured-Learning.git
cd HCEF-Structured-Learning
Step 2 — Install dependencies
pip install -r requirements.txt
Step 3 — Run experiments

Example:

python New_3_Experiments/Codes/Experiment_1B_Jena_Daily_Window_Generation.py

Then:

python New_3_Experiments/Codes/Experiment_2_Baseline_Forecasting.py

Continue sequentially through Experiment 8.

Methodology

The HCE-F framework consists of three major components:

1. Tensorized Residual Encoding

Captures nonlinear structured interactions between variables using residual tensor mappings while preserving information flow across layers.

2. Supervised Contrastive Projection

Optimizes latent-space geometry through:

intra-class compactness
inter-class separation

using supervised contrastive loss.

3. Probabilistic Ensemble Fusion

Aggregates model outputs through weighted probabilistic fusion to improve:

stability
robustness
generalization
Evaluation Protocol

Evaluation is conducted through:

Baseline Training

Comparison against conventional machine learning models.

Learning Dynamics Analysis

Monitoring:

training loss
validation loss
convergence behavior
Ablation Study

Component-wise contribution analysis.

Sensitivity Analysis

Temporal lookback window comparison.

External Cross-Domain Validation

Independent evaluation on unseen biomedical datasets.

Main Results Summary
Best Internal Model
HCEF_LSTM_7d

Performance:

RMSE = 3.96
MAE = 3.18
R² = 0.7172
External Validation — Breast Cancer
ROC-AUC = 0.9974
F1-score = 0.9647
External Validation — Heart Disease
ROC-AUC = 0.9327
F1-score = 0.8744
Reproducibility Statement

This repository contains:

complete source code
experiment scripts
preprocessing workflow
result generation pipeline
figure generation pipeline
evaluation scripts

All experiments can be reproduced from raw datasets using the provided code.

Citation

If you use this code or data in academic work, please cite:

Mahmoud Rokaya, Dalia I. Hemdan, Ibrahim Gad, Nadiah A. Baghdadi, Elsayed Atlam.

A Unified Mathematical Framework for Tensorized Residual, Contrastive, and Ensemble Learning for Structured Prediction and Cross-Domain Generalization.

Code Availability

The complete source code supporting this study is publicly available at:

https://github.com/mahmoudrokaya/HCEF-Structured-Learning

This repository includes all code necessary to reproduce the experiments, figures, tables, and reported results.

Data Availability

All datasets used in this study are publicly available from the referenced original sources listed above.

Processed experiment outputs generated during the study are included in the repository under:

New_3_Experiments/Results/
License

This repository is released for academic and research use.

For reuse, citation in derivative work is appreciated.

Contact

Mahmoud Rokaya

Department of Information Technology
College of Computers and Information Technology
Taif University
Saudi Arabia

Email:

mahmoudrokaya@tu.edu.sa


---

### Recommended next step

After saving this README, the strongest next section to write is:

## **Methods → Evaluation Method subsection**

because the journal explicitly requested:

> “Evaluation method must appear in Materials & Methods, not only in Experiments and Results.”

That section should include:

- baseline training protocol
- learning dynamics monitoring
- ablation study
- component contribution analysis
- external validation protocol
- statistical metrics

And it should match Experiments 1–8 exactly.