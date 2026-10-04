from __future__ import annotations
from typing import Any, Mapping
import subprocess
from pathlib import Path
import tempfile
import numpy as np
from autogad_reproduction.models.base import GraphAnomalyDetector
from autogad_reproduction.data import GraphData


class OfficialRepoAdapter(GraphAnomalyDetector):
    """Adapter boundary for the official legacy implementations.

    The upstream projects use separate DGL-era runners and command-line interfaces.
    This class deliberately fails loudly until a checkout and runner command are configured,
    instead of silently substituting another detector while claiming reproduction.
    """
    def __init__(self, name: str, repository: str, checkout_dir: str | None = None, runner: list[str] | None = None):
        self.name = name
        self.repository = repository
        self.checkout_dir = Path(checkout_dir) if checkout_dir else None
        self.runner = runner
        self._scores = None

    def fit(self, graph: GraphData, params: Mapping[str, Any]):
        if self.checkout_dir is None:
            raise RuntimeError(
                f"{self.name}: configure checkout_dir for the official repository {self.repository}. "
                "The Phase-1 repository intentionally does not copy third-party source trees."
            )
        if not self.checkout_dir.exists():
            raise FileNotFoundError(self.checkout_dir)
        if not self.runner:
            raise RuntimeError(f"{self.name}: configure the upstream runner command in the experiment config")
        completed = subprocess.run(self.runner, cwd=self.checkout_dir, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"{self.name} runner failed:\n{completed.stdout}\n{completed.stderr}")
        raise RuntimeError(
            f"{self.name}: upstream execution completed but no score-file parser is configured. "
            "Add a parser for the exact upstream Output format before using it in AutoGAD."
        )

    def score(self, graph: GraphData) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("fit() must succeed before score()")
        return self._scores


class PyGODAdapter(GraphAnomalyDetector):
    def __init__(self, name: str, class_name: str):
        self.name = name
        self.class_name = class_name
        self.detector = None
        self._scores = None

    def fit(self, graph: GraphData, params: Mapping[str, Any]):
        try:
            from pygod.models import DOMINANT, AnomalyDAE, GUIDE, GAAN, CONAD
            import torch
            from torch_geometric.data import Data
        except Exception as exc:
            raise RuntimeError("PyGOD adapter requires optional torch-geometric and pygod packages") from exc
        cls = {"DOMINANT": DOMINANT, "AnomalyDAE": AnomalyDAE, "GUIDE": GUIDE, "GAAN": GAAN, "CONAD": CONAD}[self.class_name]
        x = torch.from_numpy(graph.features).float()
        coo = graph.adjacency.tocoo()
        edge_index = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        data = Data(x=x, edge_index=edge_index)
        detector = cls(**dict(params))
        detector.fit(data)
        scores = detector.decision_score_
        self.detector = detector
        self._scores = np.asarray(scores).reshape(-1)
        return self

    def score(self, graph: GraphData) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("Call fit() first")
        return self._scores
