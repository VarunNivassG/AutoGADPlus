from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from autogad_reproduction.data import GraphData, save_mat


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", choices=["cora","citeseer","pubmed"], required=True)
    p.add_argument("--output", required=True)
    args=p.parse_args()
    from torch_geometric.datasets import Planetoid
    ds=Planetoid(root=args.root, name={"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed"}[args.dataset])
    d=ds[0]
    coo=d.edge_index.cpu().numpy()
    A=sp.coo_matrix((np.ones(coo.shape[1]), (coo[0], coo[1])), shape=(d.num_nodes,d.num_nodes)).tocsr()
    g=GraphData(d.x.cpu().numpy(), A, class_labels=d.y.cpu().numpy(), name=args.dataset)
    save_mat(g,args.output)
    print(f"saved {args.output}: N={g.num_nodes}, E={g.num_edges}, F={g.num_features}")

if __name__=="__main__": main()
