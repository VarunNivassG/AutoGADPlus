from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def describe(name: str, value) -> None:
    print(f"\n[{name}]")
    print(f"  Python type : {type(value).__name__}")

    if sp.issparse(value):
        print(f"  Sparse      : yes")
        print(f"  Shape       : {value.shape}")
        print(f"  dtype       : {value.dtype}")
        print(f"  nnz         : {value.nnz}")

        if value.shape[0] == value.shape[1]:
            diff = (value - value.T).nnz
            print(f"  symmetric   : {'yes' if diff == 0 else 'no'}")

        if value.nnz > 0:
            data = value.data
            print(
                f"  value range : "
                f"[{data.min():.6g}, {data.max():.6g}]"
            )

    elif isinstance(value, np.ndarray):
        print(f"  Sparse      : no")
        print(f"  Shape       : {value.shape}")
        print(f"  dtype       : {value.dtype}")
        print(f"  ndim        : {value.ndim}")

        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            print(
                f"  value range : "
                f"[{value.min():.6g}, {value.max():.6g}]"
            )

        if value.ndim == 1 or (value.ndim == 2 and 1 in value.shape):
            flat = value.reshape(-1)
            unique = np.unique(flat)

            print(f"  length      : {len(flat)}")

            if len(unique) <= 20:
                print(f"  unique vals : {unique}")

    else:
        print(f"  Value       : {value}")


def infer_cora(path: Path) -> None:
    print("=" * 70)
    print(f"Inspecting: {path}")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    mat = sio.loadmat(path)

    print("\nMATLAB keys:")
    for key in mat:
        if not key.startswith("__"):
            print(f"  - {key}")

    print("\nDetailed contents:")

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        describe(key, value)

    # Try to identify likely graph components.
    def pick(*keys):
        for key in keys:
            if key in mat:
                return mat[key], key
        return None, None

    features, feature_key = pick(
        "Attributes",
        "X",
        "attribute",
        "features",
    )

    adjacency, adjacency_key = pick(
        "Network",
        "A",
        "adj",
        "Adjacency",
    )

    labels, label_key = pick(
        "Label",
        "labels",
        "y",
    )

    print("\n" + "=" * 70)
    print("INFERRED GRAPH STRUCTURE")
    print("=" * 70)

    if features is not None:
        feature_shape = features.shape
        print(f"Features key : {feature_key}")
        print(f"Features shape: {feature_shape}")

        if len(feature_shape) == 2:
            print(f"Number of nodes: {feature_shape[0]}")
            print(f"Feature dimension: {feature_shape[1]}")
    else:
        print("Features      : NOT FOUND")

    if adjacency is not None:
        print(f"Adjacency key: {adjacency_key}")
        print(f"Adjacency shape: {adjacency.shape}")

        if sp.issparse(adjacency):
            print(f"Edges (stored entries): {adjacency.nnz}")

            if adjacency.shape[0] == adjacency.shape[1]:
                print(
                    f"Undirected/symmetric: "
                    f"{(adjacency - adjacency.T).nnz == 0}"
                )
        else:
            if adjacency.ndim == 2:
                # Number of non-zero entries.
                nnz = np.count_nonzero(adjacency)
                print(f"Non-zero entries: {nnz}")

                if adjacency.shape[0] == adjacency.shape[1]:
                    print(
                        f"Undirected/symmetric: "
                        f"{np.array_equal(adjacency, adjacency.T)}"
                    )
                    print(f"Undirected edges: {nnz // 2}")

    else:
        print("Adjacency     : NOT FOUND")

    if labels is not None:
        flat = np.asarray(labels).reshape(-1)

        print(f"\nLabel key     : {label_key}")
        print(f"Number labels : {len(flat)}")

        unique, counts = np.unique(flat, return_counts=True)

        print("Label distribution:")
        for value, count in zip(unique, counts):
            print(f"  {value}: {count}")

    else:
        print("\nAnomaly labels: NOT FOUND")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Cora/graph MATLAB .mat dataset."
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to the .mat file",
    )

    args = parser.parse_args()

    infer_cora(args.path)


if __name__ == "__main__":
    main()