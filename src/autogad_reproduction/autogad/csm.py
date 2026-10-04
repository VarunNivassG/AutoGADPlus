from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CSMResult:
    csm: float
    k: int
    anomaly_mean: float
    anomaly_variance: float
    normal_mean: float
    normal_variance: float


def compute_csm(scores: np.ndarray, k: int) -> CSMResult:
    """AutoGAD's modified CSM (Equation 3):

    (mu_O - mu_I) / sqrt(var_O + var_I)

    O = top-k scores; I = all remaining scores.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = len(s)
    if n < 2:
        raise ValueError("CSM needs at least two nodes")
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < n, got k={k}, n={n}")
    order = np.argsort(-s, kind="mergesort")
    anomaly = s[order[:k]]
    normal = s[order[k:]]
    mu_o = float(np.mean(anomaly))
    mu_i = float(np.mean(normal))
    var_o = float(np.var(anomaly, ddof=0))
    var_i = float(np.var(normal, ddof=0))
    denom = float(np.sqrt(max(var_o + var_i, 0.0)))
    if denom == 0:
        csm = float("inf") if mu_o > mu_i else 0.0
    else:
        csm = (mu_o - mu_i) / denom
    return CSMResult(csm, k, mu_o, var_o, mu_i, var_i)
