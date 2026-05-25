# HCEF-Structured-Learning

## Hybrid Contrastive–Ensemble Framework for Structured Learning and Cross-Domain Generalization

---

## Description

This repository contains the complete implementation of the **Hybrid Contrastive–Ensemble Framework (HCE-F)** proposed in the study:

**A Unified Mathematical Framework for Structured Learning and Cross-Domain Generalization Using Tensorized Residual Encoding, Supervised Contrastive Representation Learning, and Probabilistic Ensemble Fusion**

The proposed HCE-F framework integrates:

- **Tensorized residual encoding** for nonlinear multivariate feature modeling
- **Supervised contrastive representation learning** for latent-space geometric regularization
- **Probabilistic ensemble fusion** for stable prediction and variance-aware decision aggregation

The framework was evaluated through structured temporal forecasting experiments and external cross-domain validation on independent biomedical datasets.

This repository provides the full implementation workflow required to reproduce all experiments, results, figures, and tables reported in the manuscript.

---

# Dataset Information

## 1. Jena Climate Dataset

Primary dataset used for temporal forecasting experiments.

Used for:
- daily aggregation
- sliding-window generation
- baseline forecasting
- HCE-F forecasting
- HCEF-LSTM evaluation

Prediction target:
- daily temperature forecasting from previous historical observations

Source:

https://www.kaggle.com/datasets/mnassrib/jena-climate

Original source:
Max Planck Institute for Biogeochemistry, Jena, Germany

---

## 2. Breast Cancer Wisconsin Diagnostic Dataset

Used for external validation.

Samples:
- 569

Features:
- 30

Target:
- diagnosis

Source:

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

---

## 3. UCI Heart Disease Dataset

Used for external validation.

Samples:
- 920

Features:
- 15

Target:
- num

Source:

https://archive.ics.uci.edu/dataset/45/heart+disease

---

# Repository Structure

```text
HCEF-Structured-Learning/
│
├── Data/
│
├── Codes/
│   ├── Experiment_1_Jena_Data_Audit.py
│   ├── Experiment_1B_Jena_Daily_Aggregation_and_Window_Generation.py
│   ├── Experiment_2_Jena_Baseline_Forecasting.py
│   ├── Experiment_3_Jena_HCEF_Model.py
│   ├── Experiment_3B_Jena_HCEF_LSTM_Model.py
│   ├── Experiment_4_Jena_HCEF_Ablation_Study.py
│   ├── Experiment_5_Jena_Lookback_Window_Sensitivity.py
│   ├── Experiment_6_Jena_Final_Comparative_Summary.py
│   ├── Experiment_7_HCEF_External_Validation_Breast_Cancer.py
│   └── Experiment_8_HCEF_External_Validation_Heart_Disease.py
│
├── Results/
├── Figures/
├── Tables/
└── README.md
