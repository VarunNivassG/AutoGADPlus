import numpy as np
import scipy.sparse as sp
from autogad_reproduction.data import GraphData
from autogad_reproduction.injection import inject_cola_style


def test_injection_counts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    A = sp.csr_matrix((100,100), dtype=np.float32)
    for i in range(99):
        A[i,i+1] = A[i+1,i] = 1
    g = GraphData(X, A)
    out = inject_cola_style(g, m=5, n_groups=4, candidate_k=10, seed=1)
    assert out.labels.sum() == 40
    assert out.structure_labels.sum() == 20
    assert out.attribute_labels.sum() == 20
