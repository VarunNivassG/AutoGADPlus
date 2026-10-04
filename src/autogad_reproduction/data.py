from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


@dataclass
class GraphData:
    features: np.ndarray
    adjacency: sp.csr_matrix
    labels: Optional[np.ndarray] = None
    class_labels: Optional[np.ndarray] = None
    structure_labels: Optional[np.ndarray] = None
    attribute_labels: Optional[np.ndarray] = None
    name: str = "graph"

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        if self.features.ndim != 2:
            raise ValueError("features must be [N,F]")
        if not sp.issparse(self.adjacency):
            self.adjacency = sp.csr_matrix(self.adjacency)
        else:
            self.adjacency = self.adjacency.tocsr()
        if self.adjacency.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("adjacency shape must match number of nodes")
        self.adjacency = ((self.adjacency + self.adjacency.T) > 0).astype(np.float32).tocsr()
        self.adjacency.setdiag(0)
        self.adjacency.eliminate_zeros()
        for field in ("labels", "class_labels", "structure_labels", "attribute_labels"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, np.asarray(value).reshape(-1))
                if len(getattr(self, field)) != self.num_nodes:
                    raise ValueError(f"{field} length must equal N")

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_edges(self) -> int:
        return int(self.adjacency.nnz // 2)


def load_mat(path: str | Path, name: Optional[str] = None) -> GraphData:
    path = Path(path)
    mat = sio.loadmat(path)
    def pick(*keys):
        for key in keys:
            if key in mat:
                return mat[key]
        return None
    features = pick("Attributes", "X", "attribute", "features")
    adjacency = pick("Network", "A", "adj", "Adjacency")
    if features is None or adjacency is None:
        raise ValueError(f"{path} must contain an attribute matrix and adjacency matrix")
    if sp.issparse(features):
        features = features.toarray()
    if sp.issparse(adjacency):
        adjacency = adjacency.tocsr()
    else:
        adjacency = sp.csr_matrix(adjacency)
    labels = pick("Label", "labels", "y")
    class_labels = pick("Class", "class", "class_labels")
    structure_labels = pick("str_anomaly_label", "structure_labels")
    attribute_labels = pick("attr_anomaly_label", "attribute_labels")
    return GraphData(
        features=np.asarray(features),
        adjacency=adjacency,
        labels=labels,
        class_labels=class_labels,
        structure_labels=structure_labels,
        attribute_labels=attribute_labels,
        name=name or path.stem,
    )


def save_mat(graph: GraphData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Network": graph.adjacency.tocsc(),
        "Attributes": sp.csc_matrix(graph.features),
    }
    if graph.labels is not None:
        payload["Label"] = graph.labels.reshape(-1, 1).astype(np.uint8)
    if graph.class_labels is not None:
        payload["Class"] = graph.class_labels.reshape(-1, 1)
    if graph.structure_labels is not None:
        payload["str_anomaly_label"] = graph.structure_labels.reshape(-1, 1).astype(np.uint8)
    if graph.attribute_labels is not None:
        payload["attr_anomaly_label"] = graph.attribute_labels.reshape(-1, 1).astype(np.uint8)
    sio.savemat(path, payload)
