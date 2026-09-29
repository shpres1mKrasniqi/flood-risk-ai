"""One metric definition shared by the baseline and every candidate model."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml import config

LABELS: list[int] = sorted(config.RISK_LABELS)

# Macro F1 is the primary metric: it weights Low, Medium and High equally,
# so a model cannot score well by predicting only the majority class.
PRIMARY_METRIC = "f1_macro"


def compute_metrics(y_true, y_pred) -> dict[str, float]:

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }
    per_class = {
        "precision": precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0),
    }
    for name, values in per_class.items():
        for label, value in zip(LABELS, values):
            metrics[f"{name}_{config.RISK_LABELS[label].lower()}"] = value
    return {k: float(v) for k, v in metrics.items()}


def pooled_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)
