from __future__ import annotations

import random
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from autogad_reproduction.data import GraphData


def inject_cola_style(
    graph: GraphData,
    m: int = 15,
    n_groups: int | None = None,
    candidate_k: int = 50,
    seed: int = 1,
) -> GraphData:
    """Mirror the public CoLA citation/social injection procedure.

    m nodes per structural clique; n_groups cliques. The same number of
    attribute/context anomalies is created. Structural and attribute anomalies
    are disjoint.
    """
    rng = random.Random(seed)
    X = graph.features.copy()
    A = graph.adjacency.toarray().astype(np.float32)
    N = graph.num_nodes
    if n_groups is None:
        n_groups = max(1, int(round(150 / m)))
    total_each = m * n_groups
    if 2 * total_each >= N:
        raise ValueError("Not enough nodes for requested injection")
    all_idx = list(range(N))
    rng.shuffle(all_idx)
    structural = all_idx[:total_each]
    attribute = all_idx[total_each:2*total_each]
    labels = np.zeros(N, dtype=np.uint8)
    labels[structural] = 1
    labels[attribute] = 1
    str_labels = np.zeros(N, dtype=np.uint8); str_labels[structural] = 1
    attr_labels = np.zeros(N, dtype=np.uint8); attr_labels[attribute] = 1
    original_A = A.copy()
    for g in range(n_groups):
        current = structural[g*m:(g+1)*m]
        A[np.ix_(current, current)] = 1.0
        A[current, current] = 0.0
    for i in attribute:
        candidate_pool = [x for x in all_idx if x != i]
        picks = rng.sample(candidate_pool, min(candidate_k, len(candidate_pool)))
        # Euclidean distance over attribute vectors; choose the farthest candidate.
        d = np.linalg.norm(X[picks] - X[i][None, :], axis=1)
        max_idx = picks[int(np.argmax(d))]
        X[i] = X[max_idx]
    return GraphData(
        features=X,
        adjacency=sp.csr_matrix(A),
        labels=labels,
        class_labels=graph.class_labels,
        structure_labels=str_labels,
        attribute_labels=attr_labels,
        name=f"{graph.name}_cola_injected",
    )
