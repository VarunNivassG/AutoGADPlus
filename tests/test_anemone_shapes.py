import numpy as np
import scipy.sparse as sp
from autogad_reproduction.data import GraphData
from autogad_reproduction.models.anemone import ANEMONE


def test_anemone_smoke():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10,4)).astype(np.float32)
    A = np.zeros((10,10), dtype=np.float32)
    for i in range(9):
        A[i,i+1] = A[i+1,i] = 1
    g = GraphData(X, sp.csr_matrix(A))
    m = ANEMONE(hidden_dim=8, epochs=1, batch_size=5, inference_rounds=1, device="cpu", seed=1)
    scores = m.fit_score(g, {"K":3,"alpha":0.8})
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))


def test_anemone_multiple_negative_rounds():
    import torch

    core = __import__(
        "autogad_reproduction.models.anemone",
        fromlist=["_ANEMONECore"],
    )._ANEMONECore(4, 8)
    patch_adj = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
    patch_x = torch.randn(4, 3, 4)
    context_adj = patch_adj.clone()
    context_x = patch_x.clone()
    target_x = torch.randn(4, 4)

    pos_p, neg_p, pos_c, neg_c = core.forward_batch(
        patch_adj, patch_x, context_adj, context_x, target_x, 2, 3
    )

    assert pos_p.shape == (4,)
    assert neg_p.shape == (2, 4)
    assert pos_c.shape == (4,)
    assert neg_c.shape == (3, 4)
    assert torch.isfinite(neg_p).all()
    assert torch.isfinite(neg_c).all()
