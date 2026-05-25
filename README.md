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
# Hyperparameter Configuration

## Hyperparameter Configuration

The following hyperparameter settings were used in the implementation of the proposed Hybrid Contrastive–Ensemble Framework (HCE-F). These configurations correspond to the experiments reported in the manuscript and are provided to support full computational reproducibility.

---

## HCE-F Forecasting Model (Experiment 3)

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 20 |
| Contrastive temperature (τ) | 0.2 |
| Validation split | 0.20 |
| Loss function | Supervised Contrastive Loss + Ensemble Fusion Loss |
| Ensemble weighting strategy | Validation-performance based |
| Residual activation | ReLU |
| Output activation | Linear |
| Random seed | 42 |

---

## HCEF-LSTM Forecasting Model (Experiment 3B)

| Parameter | Value |
|---|---:|
| Model type | LSTM + Dense regression head |
| LSTM units | 64 |
| Dense units | 32 |
| Activation | ReLU |
| Output activation | Linear |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epochs | 50 |
| Validation split | 0.20 |
| Best-performing lookback window | 7 days |
| Early stopping | Enabled |
| Random seed | 42 |

---

## Baseline Forecasting Models (Experiment 2)

| Model | Hyperparameter Configuration |
|---|---|
| Linear Regression | Default scikit-learn settings |
| Ridge Regression | alpha = 1.0 |
| ElasticNet | alpha = 1.0, l1_ratio = 0.5 |
| Random Forest Regressor | n_estimators = 200 |
| Extra Trees Regressor | n_estimators = 200 |
| Gradient Boosting Regressor | n_estimators = 100 |
| MLP Regressor | hidden_layer_sizes = (100,), max_iter = 500 |

---

## Lookback Window Sensitivity Analysis (Experiment 5)

Evaluated temporal input windows:

| Window Size |
|---:|
| 3-day |
| 7-day |
| 14-day |
| 30-day |

The 7-day and 14-day temporal windows yielded the strongest overall forecasting performance.

---

## External Validation Experiments (Experiments 7 and 8)

| Parameter | Value |
|---|---:|
| Train/Test Split | 80/20 |
| Cross-validation | 5-fold |
| Random seed | 42 |
| Evaluation metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC |

External validation datasets:

- Breast Cancer Wisconsin Diagnostic Dataset
- UCI Heart Disease Dataset

---

## Computational Environment

The experiments were executed under the following software and hardware environment.

### Software

| Component | Version |
|---|---:|
| Python | 3.10+ |
| NumPy | Latest stable |
| pandas | Latest stable |
| scikit-learn | Latest stable |
| TensorFlow | 2.x |
| Keras | 2.x |
| Matplotlib | Latest stable |
| Seaborn | Latest stable |
| openpyxl | Latest stable |

---

### Hardware

| Component | Specification |
|---|---:|
| Processor | Intel Core i7 |
| RAM | 16 GB |
| GPU | CPU-based execution (no dedicated GPU required) |
| Operating System | Windows |

All experiments were designed to run under modest computational resources and were successfully executed on standard desktop hardware.
