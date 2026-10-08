"""
Data preprocessing module for CORA and other citation network datasets.

This module provides utilities for loading, cleaning, normalizing, and preprocessing
graph data from the CORA dataset and similar citation networks for use with AutoGAD.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Optional, Tuple
import logging

from .data import GraphData

logger = logging.getLogger(__name__)


class CoraPreprocessor:
    """Preprocessor for CORA citation network dataset."""
    
    def __init__(self, normalize_features: bool = True, 
                 remove_self_loops: bool = True,
                 symmetrize_adjacency: bool = True):
        """
        Initialize CORA preprocessor.
        
        Args:
            normalize_features: Whether to normalize node features (row-wise L2 normalization)
            remove_self_loops: Whether to remove self-loops from adjacency matrix
            symmetrize_adjacency: Whether to symmetrize adjacency matrix
        """
        self.normalize_features = normalize_features
        self.remove_self_loops = remove_self_loops
        self.symmetrize_adjacency = symmetrize_adjacency
        
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess node features using row-wise L2 normalization.
        
        Args:
            features: Feature matrix of shape [N, F]
            
        Returns:
            Normalized feature matrix of shape [N, F]
        """
        if not self.normalize_features:
            return features.astype(np.float32)
        
        logger.info(f"Normalizing features: shape {features.shape}")
        
        # Row-wise L2 normalization
        features = features.astype(np.float32)
        row_norms = np.linalg.norm(features, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        features_normalized = features / row_norms
        
        logger.info(f"Features normalized: min={features_normalized.min():.4f}, "
                   f"max={features_normalized.max():.4f}, mean={features_normalized.mean():.4f}")
        
        return features_normalized
    
    def preprocess_adjacency(self, adjacency: sp.csr_matrix) -> sp.csr_matrix:
        """
        Preprocess adjacency matrix: symmetrize and optionally remove self-loops.
        
        Args:
            adjacency: Adjacency matrix in sparse format
            
        Returns:
            Preprocessed adjacency matrix in CSR format
        """
        logger.info(f"Preprocessing adjacency: shape {adjacency.shape}, nnz={adjacency.nnz}")
        
        # Ensure CSR format
        if not sp.issparse(adjacency):
            adjacency = sp.csr_matrix(adjacency)
        else:
            adjacency = adjacency.tocsr()
        
        # Symmetrize if requested
        if self.symmetrize_adjacency:
            adjacency = (adjacency + adjacency.T > 0).astype(np.float32).tocsr()
            logger.info(f"Adjacency symmetrized: nnz={adjacency.nnz}")
        
        # Remove self-loops if requested
        if self.remove_self_loops:
            adjacency.setdiag(0)
            adjacency.eliminate_zeros()
            logger.info(f"Self-loops removed: nnz={adjacency.nnz}")
        
        adjacency = adjacency.astype(np.float32)
        return adjacency
    
    def preprocess(self, graph_data: GraphData) -> GraphData:
        """
        Apply full preprocessing pipeline to graph data.
        
        Args:
            graph_data: Input GraphData object
            
        Returns:
            Preprocessed GraphData object
        """
        logger.info(f"Starting preprocessing for graph: {graph_data.name}")
        logger.info(f"Input: {graph_data.num_nodes} nodes, {graph_data.num_features} features, "
                   f"{graph_data.num_edges} edges")
        
        # Preprocess features and adjacency
        features = self.preprocess_features(graph_data.features)
        adjacency = self.preprocess_adjacency(graph_data.adjacency)
        
        # Create new GraphData with preprocessed data
        preprocessed = GraphData(
            features=features,
            adjacency=adjacency,
            labels=graph_data.labels,
            class_labels=graph_data.class_labels,
            structure_labels=graph_data.structure_labels,
            attribute_labels=graph_data.attribute_labels,
            name=f"{graph_data.name}_preprocessed"
        )
        
        logger.info(f"Preprocessing complete: {preprocessed.num_nodes} nodes, "
                   f"{preprocessed.num_features} features, {preprocessed.num_edges} edges")
        
        return preprocessed


class FeatureNormalizer:
    """Utility for different feature normalization schemes."""
    
    @staticmethod
    def l2_normalize(features: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalization."""
        row_norms = np.linalg.norm(features, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        return features / row_norms
    
    @staticmethod
    def l1_normalize(features: np.ndarray) -> np.ndarray:
        """Row-wise L1 normalization."""
        row_sums = np.sum(np.abs(features), axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return features / row_sums
    
    @staticmethod
    def standardize(features: np.ndarray) -> np.ndarray:
        """Column-wise standardization (zero mean, unit variance)."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1
        return (features - mean) / std
    
    @staticmethod
    def minmax_scale(features: np.ndarray) -> np.ndarray:
        """Column-wise min-max scaling to [0, 1]."""
        feature_min = np.min(features, axis=0)
        feature_max = np.max(features, axis=0)
        denom = feature_max - feature_min
        denom[denom == 0] = 1
        return (features - feature_min) / denom


class AdjacencyProcessor:
    """Utilities for adjacency matrix processing."""
    
    @staticmethod
    def symmetrize(adjacency: sp.csr_matrix, method: str = 'or') -> sp.csr_matrix:
        """
        Symmetrize adjacency matrix.
        
        Args:
            adjacency: Input adjacency matrix
            method: 'or' (union), 'and' (intersection), or 'avg' (average)
            
        Returns:
            Symmetrized adjacency matrix
        """
        if not sp.issparse(adjacency):
            adjacency = sp.csr_matrix(adjacency)
        else:
            adjacency = adjacency.tocsr()
        
        if method == 'or':
            return (adjacency + adjacency.T > 0).astype(np.float32).tocsr()
        elif method == 'and':
            return (adjacency.multiply(adjacency.T) > 0).astype(np.float32).tocsr()
        elif method == 'avg':
            return ((adjacency + adjacency.T) / 2).astype(np.float32).tocsr()
        else:
            raise ValueError(f"Unknown symmetrization method: {method}")
    
    @staticmethod
    def remove_self_loops(adjacency: sp.csr_matrix) -> sp.csr_matrix:
        """Remove self-loops from adjacency matrix."""
        adjacency = adjacency.tolil()
        adjacency.setdiag(0)
        return adjacency.tocsr().astype(np.float32)
    
    @staticmethod
    def add_self_loops(adjacency: sp.csr_matrix) -> sp.csr_matrix:
        """Add identity matrix to adjacency (add self-loops)."""
        n = adjacency.shape[0]
        identity = sp.eye(n, dtype=np.float32, format='csr')
        return (adjacency + identity).astype(np.float32)
    
    @staticmethod
    def compute_degree(adjacency: sp.csr_matrix) -> np.ndarray:
        """Compute node degree."""
        return np.array(adjacency.sum(axis=1)).flatten()
    
    @staticmethod
    def normalize_by_degree(adjacency: sp.csr_matrix, symmetric: bool = True) -> sp.csr_matrix:
        """
        Normalize adjacency by degree (D^-1 A or D^-1/2 A D^-1/2).
        
        Args:
            adjacency: Input adjacency matrix
            symmetric: If True, use symmetric normalization D^-1/2 A D^-1/2
            
        Returns:
            Degree-normalized adjacency matrix
        """
        degrees = AdjacencyProcessor.compute_degree(adjacency)
        
        if symmetric:
            # D^-1/2
            inv_sqrt_degree = np.power(degrees, -0.5)
            inv_sqrt_degree[np.isinf(inv_sqrt_degree)] = 0
            inv_sqrt_degree_mat = sp.diags(inv_sqrt_degree)
            # D^-1/2 A D^-1/2
            normalized = inv_sqrt_degree_mat @ adjacency @ inv_sqrt_degree_mat
        else:
            # D^-1
            inv_degree = np.power(degrees, -1.0)
            inv_degree[np.isinf(inv_degree)] = 0
            inv_degree_mat = sp.diags(inv_degree)
            # D^-1 A
            normalized = inv_degree_mat @ adjacency
        
        return normalized.astype(np.float32).tocsr()


def preprocess_cora_dataset(input_path: str | Path, 
                            output_path: Optional[str | Path] = None,
                            normalize_features: bool = True,
                            symmetrize_adjacency: bool = True,
                            remove_self_loops: bool = True) -> GraphData:
    """
    Load and preprocess CORA dataset in one step.
    
    Args:
        input_path: Path to .mat file containing CORA data
        output_path: Optional path to save preprocessed data
        normalize_features: Whether to L2-normalize features
        symmetrize_adjacency: Whether to symmetrize adjacency
        remove_self_loops: Whether to remove self-loops
        
    Returns:
        Preprocessed GraphData object
    """
    from .data import load_mat, save_mat
    
    logger.info(f"Loading CORA dataset from {input_path}")
    graph_data = load_mat(input_path, name="cora")
    
    preprocessor = CoraPreprocessor(
        normalize_features=normalize_features,
        remove_self_loops=remove_self_loops,
        symmetrize_adjacency=symmetrize_adjacency
    )
    
    preprocessed = preprocessor.preprocess(graph_data)
    
    if output_path is not None:
        logger.info(f"Saving preprocessed data to {output_path}")
        save_mat(preprocessed, output_path)
    
    return preprocessed


def compute_graph_statistics(graph_data: GraphData) -> dict:
    """
    Compute comprehensive statistics for a graph.
    
    Args:
        graph_data: GraphData object
        
    Returns:
        Dictionary containing graph statistics
    """
    adjacency = graph_data.adjacency
    features = graph_data.features
    
    degrees = AdjacencyProcessor.compute_degree(adjacency)
    
    stats = {
        'num_nodes': graph_data.num_nodes,
        'num_edges': graph_data.num_edges,
        'num_features': graph_data.num_features,
        'density': 2 * graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1)),
        'avg_degree': degrees.mean(),
        'min_degree': degrees.min(),
        'max_degree': degrees.max(),
        'feature_mean': features.mean(),
        'feature_std': features.std(),
        'feature_min': features.min(),
        'feature_max': features.max(),
        'is_connected': sp.csgraph.connected_components(adjacency, directed=False)[0] == 1
    }
    
    logger.info(f"Graph statistics for {graph_data.name}:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return stats
