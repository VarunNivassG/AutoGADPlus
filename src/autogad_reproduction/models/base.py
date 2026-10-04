from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping
import numpy as np
from autogad_reproduction.data import GraphData


class GraphAnomalyDetector(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, graph: GraphData, params: Mapping[str, Any]) -> "GraphAnomalyDetector":
        raise NotImplementedError

    @abstractmethod
    def score(self, graph: GraphData) -> np.ndarray:
        raise NotImplementedError

    def fit_score(self, graph: GraphData, params: Mapping[str, Any]) -> np.ndarray:
        self.fit(graph, params)
        return self.score(graph)
