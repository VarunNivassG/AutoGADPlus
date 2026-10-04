from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(np.unique(labels)) < 2:
        raise ValueError("ROC-AUC needs both classes")
    return float(roc_auc_score(labels, scores))


def sensitivity(aucs: np.ndarray) -> float:
    a = np.asarray(aucs, dtype=float)
    mx = float(np.max(a))
    return float((mx - np.min(a)) / mx) if mx else 0.0


def gain_over_min(selected_auc: float, aucs: np.ndarray) -> float:
    mn = float(np.min(aucs))
    return float((selected_auc - mn) / mn) if mn else float("inf")


def gain_over_median(selected_auc: float, aucs: np.ndarray) -> float:
    med = float(np.median(aucs))
    return float((selected_auc - med) / med) if med else float("inf")


def gain_over_max(selected_auc: float, aucs: np.ndarray) -> float:
    mx = float(np.max(aucs))
    return float((selected_auc - mx) / mx) if mx else 0.0
