from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cora","citeseer","pubmed"], required=True)
    p.add_argument("--output-dir", default="data/raw")
    args = p.parse_args()
    try:
        from torch_geometric.datasets import Planetoid
    except Exception as exc:
        raise SystemExit("Install optional torch-geometric first: pip install torch-geometric") from exc
    root = Path(args.output_dir) / "planetoid"
    ds = Planetoid(root=str(root), name={"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed"}[args.dataset])
    data = ds[0]
    print(f"Downloaded {args.dataset}: nodes={data.num_nodes}, edges={data.edge_index.shape[1] // 2}, features={data.num_features}")
    print(f"Raw files are under {root}")


if __name__ == "__main__":
    main()
