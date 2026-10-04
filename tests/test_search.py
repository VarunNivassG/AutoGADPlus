import numpy as np
from autogad_reproduction.autogad.search import grid_search


def test_grid_search():
    def score_fn(p, config_id=None):
        return np.array([p["x"]*3, p["x"]*2, 0.1, 0.0])
    result = grid_search(score_fn, {"x": [0.1,0.5,1.0]}, anomaly_k=2, verbose=False)
    assert result.best_params["x"] == 1.0
    assert len(result.table) == 3
