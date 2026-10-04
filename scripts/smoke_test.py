from __future__ import annotations

import numpy as np
from autogad_reproduction.data import GraphData
from autogad_reproduction.autogad.csm import compute_csm
from autogad_reproduction.models.anemone import ANEMONE
from autogad_reproduction.autogad.search import grid_search
from autogad_reproduction.evaluation import roc_auc
from autogad_reproduction.seed import set_seed
import scipy.sparse as sp


def make_toy(seed=42):
    rng = np.random.default_rng(seed)
    n = 40
    f = 8
    labels = np.zeros(n, dtype=np.uint8)
    labels[-4:] = 1
    X = rng.normal(size=(n,f)).astype(np.float32)
    X[-4:] += 4.0
    A = np.zeros((n,n), dtype=np.float32)
    # two communities + a few cross links
    for i in range(n):
        for _ in range(3):
            j = int(rng.integers(0,n))
            if i != j:
                A[i,j] = A[j,i] = 1
    return GraphData(X, sp.csr_matrix(A), labels=labels, name="toy")


def main():
    set_seed(42)
    graph = make_toy()
    print(f"Toy graph: N={graph.num_nodes}, F={graph.num_features}, E={graph.num_edges}")
    model_kwargs = dict(hidden_dim=16, epochs=4, batch_size=8, inference_rounds=3, lr=2e-3, device="cpu", seed=42)
    space = {"K": [2,3], "alpha": [0.2,0.8]}
    calls = {"n": 0}
    def score_fn(params, config_id=None):
        calls["n"] += 1
        model = ANEMONE(**model_kwargs)
        return model.fit_score(graph, params)
    result = grid_search(score_fn, space, anomaly_k=4, verbose=True)
    print("Best:", result.best_params, "CSM=", result.best_csm)
    auc = roc_auc(result.best_scores, graph.labels)
    print("Evaluation ROC-AUC:", auc)
    assert calls["n"] == 4
    assert np.isfinite(result.best_csm)
    assert np.isfinite(auc)
    csm = compute_csm(result.best_scores, 4)
    print("CSM detail:", csm)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
