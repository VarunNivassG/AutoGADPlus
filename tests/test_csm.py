import numpy as np
from autogad_reproduction.autogad.csm import compute_csm


def test_csm_separation():
    scores = np.array([0.95, 0.92, 0.90, 0.20, 0.18, 0.15])
    r = compute_csm(scores, 3)
    assert r.csm > 3
    assert r.anomaly_mean > r.normal_mean
