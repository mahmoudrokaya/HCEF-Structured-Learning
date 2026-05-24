"""
Experiment_7_HCEF_External_Validation_Breast_Cancer.py

Experiment 7:
Cross-domain external validation on the Breast Cancer Wisconsin Diagnostic Dataset.

Input folder:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//Data//Breast Cancer Wisconsin Diagnostic Dataset

Output:
D://47//471//New Papers//Paper 3 IJOCTA//Sub//New_3_Experiments//Experiment_7_HCEF_External_Validation_Breast_Cancer
"""

from pathlib import Path
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\Data\Breast Cancer Wisconsin Diagnostic Dataset"
)

RESULTS_ROOT = Path(
    r"D:\47\471\New Papers\Paper 3 IJOCTA\Sub\New_3_Experiments"
)

EXP_DIR = RESULTS_ROOT / "Experiment_7_HCEF_External_Validation_Breast_Cancer"

TABLES_DIR = EXP_DIR / "Tables"
FIGURES_DIR = EXP_DIR / "Figures"
REPORTS_DIR = EXP_DIR / "Reports"
PRED_DIR = EXP_DIR / "Predictions"
MODELS_DIR = EXP_DIR / "Models"
PROCESSED_DIR = EXP_DIR / "Processed_Data"

for d in [EXP_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, PRED_DIR, MODELS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# HCE-F SETTINGS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 45
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 96
PROJECTION_DIM = 48
DROPOUT = 0.20
CONTRASTIVE_WEIGHT = 0.03

# ============================================================
# HELPERS
# ============================================================

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def find_dataset_file(data_dir):
    candidates = []
    for ext in ["*.csv", "*.xlsx", "*.xls", "*.data", "*.txt"]:
        candidates.extend(list(data_dir.rglob(ext)))

    if not candidates:
        raise FileNotFoundError(f"No supported dataset file found in: {data_dir}")

    preferred_keywords = ["breast", "cancer", "wisconsin", "diagnostic", "wdbc", "data"]

    def score_file(p):
        name = p.name.lower()
        score = sum(1 for k in preferred_keywords if k in name)
        if p.suffix.lower() == ".csv":
            score += 2
        if p.suffix.lower() in [".xlsx", ".xls"]:
            score += 1
        return score

    candidates = sorted(candidates, key=score_file, reverse=True)
    return candidates[0]


def load_dataset(file_path):
    ext = file_path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    if ext in [".data", ".txt"]:
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, header=None)

    raise ValueError(f"Unsupported file type: {file_path}")


def clean_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("  ", " ")
        for c in df.columns
    ]
    return df


def detect_target_column(df):
    target_candidates = [
        "diagnosis",
        "Diagnosis",
        "target",
        "Target",
        "class",
        "Class",
        "label",
        "Label",
        "outcome",
        "Outcome"
    ]

    for c in target_candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        lower = str(c).lower()
        if "diagnosis" in lower or "target" in lower or "class" in lower or "label" in lower:
            return c

    low_cardinality = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 5:
            low_cardinality.append((c, nunique))

    if low_cardinality:
        return low_cardinality[0][0]

    raise ValueError("Could not detect target column automatically.")


def prepare_breast_cancer_dataframe(df):
    """
    Handles common WDBC formats:
    1. named columns including diagnosis
    2. raw UCI wdbc.data format without headers:
       ID, diagnosis, 30 numeric features
    """
    df = clean_columns(df)

    # If raw WDBC data loaded without headers, pandas may name columns as integers.
    if all(isinstance(c, int) or str(c).isdigit() for c in df.columns):
        if df.shape[1] == 32:
            feature_names = [
                "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
                "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
            ]
            df.columns = ["id", "diagnosis"] + feature_names

    # Remove unnamed columns
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Drop ID-like columns
    id_like = []
    for c in df.columns:
        lower = str(c).lower()
        if lower in ["id", "sample_id", "patient_id"] or lower.endswith("_id"):
            id_like.append(c)

    if id_like:
        df = df.drop(columns=id_like)

    return df


def preprocess_features(df, target_col):
    y_raw = df[target_col].astype(str)
    X_raw = df.drop(columns=[target_col]).copy()

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_raw)

    X = pd.DataFrame(index=X_raw.index)
    encoding_rows = []

    for c in X_raw.columns:
        s = X_raw[c]

        numeric_s = pd.to_numeric(s, errors="coerce")
        numeric_ratio = numeric_s.notna().mean()

        if numeric_ratio >= 0.90:
            X[c] = numeric_s
            encoding_rows.append({
                "feature": c,
                "encoding": "numeric",
                "unique_values": int(s.nunique(dropna=True))
            })
        else:
            enc = LabelEncoder()
            filled = s.astype(str).fillna("MISSING_VALUE")
            X[c] = enc.fit_transform(filled)
            encoding_rows.append({
                "feature": c,
                "encoding": "label_encoded",
                "unique_values": int(s.nunique(dropna=True))
            })

    X = X.replace([np.inf, -np.inf], np.nan)

    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    return X, y, y_encoder, pd.DataFrame(encoding_rows)


def evaluate_classification(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
    }


def supervised_contrastive_loss(z, y, temperature=0.2):
    if z.shape[0] < 3:
        return torch.tensor(0.0, device=z.device)

    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature

    y = y.view(-1, 1)
    positive_mask = torch.eq(y, y.T).float()
    self_mask = torch.eye(z.shape[0], device=z.device)
    positive_mask = positive_mask - self_mask

    exp_sim = torch.exp(sim) * (1.0 - self_mask)

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    positives_per_row = positive_mask.sum(dim=1)

    valid = positives_per_row > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    loss = -((positive_mask * log_prob).sum(dim=1) / (positives_per_row + 1e-8))
    loss = loss[valid].mean()

    return loss


# ============================================================
# DATASET CLASS
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# HCE-F CLASSIFIER
# ============================================================

class HCEFTabularClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.tensorized_residual_encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU()
        )

        self.residual_projection = nn.Linear(input_dim, HIDDEN_DIM)

        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Sigmoid()
        )

        self.contrastive_projection = nn.Sequential(
            nn.Linear(HIDDEN_DIM, PROJECTION_DIM),
            nn.ReLU(),
            nn.Linear(PROJECTION_DIM, PROJECTION_DIM)
        )

        self.classifier_head_1 = nn.Sequential(
            nn.Linear(PROJECTION_DIM, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 2)
        )

        self.classifier_head_2 = nn.Sequential(
            nn.Linear(PROJECTION_DIM, 32),
            nn.Tanh(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 2)
        )

        self.ensemble_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        h = self.tensorized_residual_encoder(x)

        residual = self.residual_projection(x)
        gate = self.gate(h)

        h = h + gate * residual

        z = self.contrastive_projection(h)

        logits_1 = self.classifier_head_1(z)
        logits_2 = self.classifier_head_2(z)

        weights = torch.softmax(self.ensemble_logits, dim=0)

        logits = weights[0] * logits_1 + weights[1] * logits_2

        return logits, z, weights


# ============================================================
# MAIN
# ============================================================

print("=" * 90)
print("Experiment 7: HCE-F External Validation - Breast Cancer Wisconsin")
print("=" * 90)
print(f"Device: {DEVICE}")

dataset_file = find_dataset_file(DATA_DIR)
df_raw = load_dataset(dataset_file)
df = prepare_breast_cancer_dataframe(df_raw)

target_col = detect_target_column(df)

X, y, y_encoder, encoding_report = preprocess_features(df, target_col)

if len(np.unique(y)) != 2:
    raise ValueError(f"Expected binary target, detected classes: {np.unique(y)}")

encoding_report.to_csv(TABLES_DIR / "encoding_report.csv", index=False)

processed_df = X.copy()
processed_df["target"] = y
processed_df.to_csv(PROCESSED_DIR / "breast_cancer_processed.csv", index=False)

dataset_summary = {
    "dataset_file": str(dataset_file),
    "rows": int(df.shape[0]),
    "columns_after_cleanup": int(df.shape[1]),
    "target_column": target_col,
    "target_classes": list(map(str, y_encoder.classes_)),
    "feature_count": int(X.shape[1]),
    "missing_values_after_processing": int(X.isna().sum().sum())
}

with open(REPORTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(dataset_summary, f, indent=4)

pd.DataFrame([dataset_summary]).to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

target_counts = pd.Series(y).value_counts().sort_index().reset_index()
target_counts.columns = ["encoded_class", "count"]
target_counts["class_label"] = target_counts["encoded_class"].apply(lambda i: y_encoder.classes_[i])
target_counts.to_csv(TABLES_DIR / "target_distribution.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(target_counts["class_label"].astype(str), target_counts["count"])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Breast Cancer Wisconsin Target Distribution")
plt.grid(axis="y", alpha=0.3)
save_fig(FIGURES_DIR / "target_distribution.png")

# ============================================================
# TRAIN/VALIDATION/TEST SPLIT
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    stratify=y_train_val,
    random_state=SEED
)

split_summary = pd.DataFrame([
    {"split": "train", "rows": len(X_train)},
    {"split": "validation", "rows": len(X_val)},
    {"split": "test", "rows": len(X_test)}
])
split_summary.to_csv(TABLES_DIR / "split_summary.csv", index=False)

# ============================================================
# CLASSICAL BASELINES
# ============================================================

baselines = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=SEED))
    ]),
    "Random_Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Extra_Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=SEED,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Gradient_Boosting": GradientBoostingClassifier(
        random_state=SEED
    ),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            early_stopping=True,
            random_state=SEED
        ))
    ])
}

baseline_rows = []

for name, model in baselines.items():
    print("-" * 90)
    print(f"Training baseline: {name}")

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        decision = model.decision_function(X_test)
        prob = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)

    metrics = evaluate_classification(y_test, pred, prob)

    row = {
        "model": name,
        "training_time_seconds": float(training_time)
    }
    row.update(metrics)
    baseline_rows.append(row)

    pd.DataFrame({
        "observed": y_test,
        "predicted": pred,
        "probability_positive": prob,
        "model": name
    }).to_csv(PRED_DIR / f"predictions_{name}.csv", index=False)

baseline_df = pd.DataFrame(baseline_rows).sort_values("roc_auc", ascending=False)
baseline_df.to_csv(TABLES_DIR / "classical_baseline_metrics.csv", index=False)

# ============================================================
# HCE-F TRAINING
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_loader = DataLoader(
    TabularDataset(X_train_scaled, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    TabularDataset(X_val_scaled, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    TabularDataset(X_test_scaled, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = HCEFTabularClassifier(input_dim=X_train_scaled.shape[1]).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

ce_loss = nn.CrossEntropyLoss()

best_val_auc = -np.inf
best_epoch = 0
epochs_without_improvement = 0

history = []

model_path = MODELS_DIR / "best_hcef_breast_cancer.pt"

start_training = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()

    train_losses = []
    train_ce_losses = []
    train_contrastive_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits, z, weights = model(xb)

        loss_ce = ce_loss(logits, yb)
        loss_contrastive = supervised_contrastive_loss(z, yb)

        loss = loss_ce + CONTRASTIVE_WEIGHT * loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        train_losses.append(loss.item())
        train_ce_losses.append(loss_ce.item())
        train_contrastive_losses.append(loss_contrastive.item())

    model.eval()

    val_probs = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)

            logits, _, _ = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            val_probs.extend(probs.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(yb.numpy())

    val_probs = np.array(val_probs)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    val_auc = roc_auc_score(val_targets, val_probs)
    val_f1 = f1_score(val_targets, val_preds, zero_division=0)

    ensemble_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu().numpy()

    row = {
        "epoch": epoch,
        "train_total_loss": float(np.mean(train_losses)),
        "train_ce_loss": float(np.mean(train_ce_losses)),
        "train_contrastive_loss": float(np.mean(train_contrastive_losses)),
        "validation_auc": float(val_auc),
        "validation_f1": float(val_f1),
        "ensemble_weight_1": float(ensemble_weights[0]),
        "ensemble_weight_2": float(ensemble_weights[1])
    }

    history.append(row)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model.state_dict(), model_path)
    else:
        epochs_without_improvement += 1

    print(
        f"HCE-F | Epoch {epoch:03d} | "
        f"Loss={row['train_total_loss']:.4f} | "
        f"ValAUC={val_auc:.4f} | "
        f"ValF1={val_f1:.4f} | "
        f"Weights=({ensemble_weights[0]:.3f}, {ensemble_weights[1]:.3f})"
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break

training_time = time.time() - start_training

history_df = pd.DataFrame(history)
history_df.to_csv(TABLES_DIR / "hcef_training_history.csv", index=False)

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

test_probs = []
test_preds = []
test_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)

        logits, _, weights = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        test_probs.extend(probs.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(yb.numpy())

test_probs = np.array(test_probs)
test_preds = np.array(test_preds)
test_targets = np.array(test_targets)

hcef_metrics = evaluate_classification(test_targets, test_preds, test_probs)

final_weights = torch.softmax(model.ensemble_logits, dim=0).detach().cpu().numpy()

hcef_row = {
    "model": "HCE-F_Tabular_Breast_Cancer",
    "best_epoch": int(best_epoch),
    "training_time_seconds": float(training_time),
    "ensemble_weight_1": float(final_weights[0]),
    "ensemble_weight_2": float(final_weights[1])
}
hcef_row.update(hcef_metrics)

pd.DataFrame([hcef_row]).to_csv(TABLES_DIR / "hcef_external_validation_metrics.csv", index=False)

pd.DataFrame({
    "observed": test_targets,
    "predicted": test_preds,
    "probability_positive": test_probs,
    "model": "HCE-F_Tabular_Breast_Cancer"
}).to_csv(PRED_DIR / "predictions_HCE-F_Tabular_Breast_Cancer.csv", index=False)

# ============================================================
# COMBINED COMPARISON
# ============================================================

combined_df = pd.concat([
    baseline_df,
    pd.DataFrame([{
        "model": hcef_row["model"],
        "training_time_seconds": hcef_row["training_time_seconds"],
        "accuracy": hcef_row["accuracy"],
        "precision": hcef_row["precision"],
        "recall": hcef_row["recall"],
        "f1": hcef_row["f1"],
        "roc_auc": hcef_row["roc_auc"],
        "pr_auc": hcef_row["pr_auc"]
    }])
], ignore_index=True)

combined_df = combined_df.sort_values("roc_auc", ascending=False)
combined_df.to_csv(TABLES_DIR / "external_validation_model_comparison.csv", index=False)

# ============================================================
# FIGURES
# ============================================================

plt.figure(figsize=(9, 6))
plt.barh(combined_df["model"], combined_df["roc_auc"])
plt.xlabel("ROC-AUC")
plt.ylabel("Model")
plt.title("Breast Cancer External Validation - ROC-AUC")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "model_comparison_roc_auc.png")

plt.figure(figsize=(9, 6))
plt.barh(combined_df["model"], combined_df["f1"])
plt.xlabel("F1-score")
plt.ylabel("Model")
plt.title("Breast Cancer External Validation - F1-score")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
save_fig(FIGURES_DIR / "model_comparison_f1.png")

cm = confusion_matrix(test_targets, test_preds)
pd.DataFrame(cm).to_csv(TABLES_DIR / "hcef_confusion_matrix.csv", index=False)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.colorbar(label="Count")
plt.xticks([0, 1], y_encoder.classes_)
plt.yticks([0, 1], y_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.title("HCE-F Confusion Matrix - Breast Cancer")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

save_fig(FIGURES_DIR / "hcef_confusion_matrix.png")

fpr, tpr, _ = roc_curve(test_targets, test_probs)
precision_curve, recall_curve, _ = precision_recall_curve(test_targets, test_probs)

pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(TABLES_DIR / "hcef_roc_curve_coordinates.csv", index=False)
pd.DataFrame({"precision": precision_curve, "recall": recall_curve}).to_csv(TABLES_DIR / "hcef_pr_curve_coordinates.csv", index=False)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {hcef_metrics['roc_auc']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("HCE-F ROC Curve - Breast Cancer")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_roc_curve.png")

plt.figure(figsize=(7, 6))
plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {hcef_metrics['pr_auc']:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("HCE-F Precision-Recall Curve - Breast Cancer")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_pr_curve.png")

plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["train_ce_loss"], label="Cross-Entropy Loss")
plt.plot(history_df["epoch"], history_df["train_contrastive_loss"], label="Contrastive Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("HCE-F Training Dynamics - Breast Cancer")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_training_dynamics.png")

plt.figure(figsize=(9, 5))
plt.plot(history_df["epoch"], history_df["validation_auc"], label="Validation ROC-AUC")
plt.plot(history_df["epoch"], history_df["validation_f1"], label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("HCE-F Validation Dynamics - Breast Cancer")
plt.legend()
plt.grid(alpha=0.3)
save_fig(FIGURES_DIR / "hcef_validation_dynamics.png")

# ============================================================
# SUMMARY, METHODS, README
# ============================================================

best_model = combined_df.iloc[0].to_dict()

summary = {
    "experiment": "Experiment_7_HCEF_External_Validation_Breast_Cancer",
    "status": "completed",
    "dataset_folder": str(DATA_DIR),
    "dataset_file": str(dataset_file),
    "output_folder": str(EXP_DIR),
    "device": DEVICE,
    "rows": int(df.shape[0]),
    "feature_count": int(X.shape[1]),
    "target_column": target_col,
    "target_classes": list(map(str, y_encoder.classes_)),
    "hcef_metrics": hcef_row,
    "best_model_by_roc_auc": best_model["model"],
    "best_roc_auc": float(best_model["roc_auc"]),
    "best_f1": float(best_model["f1"])
}

with open(REPORTS_DIR / "experiment_7_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

methods_text = f"""
### Experiment 7: External Cross-Domain Validation on the Breast Cancer Wisconsin Diagnostic Dataset

The seventh experiment evaluated the cross-domain generalization ability of the proposed HCE-F framework using the Breast Cancer Wisconsin Diagnostic Dataset. This dataset was selected because it differs substantially from the Jena climate forecasting task in domain, target semantics, sample structure, and feature distribution. The task was formulated as binary classification of diagnostic status using structured numerical features.

The dataset was loaded from the local external validation folder, cleaned to remove identifier-like columns, and processed using automatic target detection and feature encoding. Numerical variables were retained, non-numeric variables were label-encoded when necessary, and missing values were median-imputed. The data were split into stratified training, validation, and test partitions. HCE-F was implemented as a tabular classifier consisting of a tensorized residual encoder, residual-gated feature stabilization, supervised contrastive projection, and two-head probabilistic ensemble classification. Performance was evaluated using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrix analysis, ROC curves, and precision-recall curves. Classical machine-learning baselines were also trained on the same split to contextualize external validation performance.
"""

with open(REPORTS_DIR / "METHODS_TEXT_Experiment_7_HCEF_External_Validation_Breast_Cancer.md", "w", encoding="utf-8") as f:
    f.write(methods_text)

readme_text = f"""
# Experiment 7: HCE-F External Validation - Breast Cancer Wisconsin

## Purpose
This experiment evaluates cross-domain external validation of the HCE-F framework on the Breast Cancer Wisconsin Diagnostic Dataset.

## Input Folder
{DATA_DIR}

## Detected Dataset File
{dataset_file}

## Target Column
{target_col}

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
"""

with open(REPORTS_DIR / "README_Experiment_7_HCEF_External_Validation_Breast_Cancer.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

print("=" * 90)
print("Experiment 7 completed successfully.")
print("=" * 90)
print(f"Dataset file: {dataset_file}")
print(f"Target column: {target_col}")
print(f"Rows: {df.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"HCE-F ROC-AUC: {hcef_metrics['roc_auc']:.4f}")
print(f"HCE-F F1: {hcef_metrics['f1']:.4f}")
print(f"Best model by ROC-AUC: {best_model['model']}")
print(f"Output folder: {EXP_DIR}")
print("=" * 90)