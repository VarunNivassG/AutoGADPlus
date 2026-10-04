from __future__ import annotations
import argparse
from autogad_reproduction.data import load_mat, save_mat
from autogad_reproduction.injection import inject_cola_style


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--m", type=int, default=15)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    g = load_mat(args.input)
    out = inject_cola_style(g, m=args.m, n_groups=args.n, candidate_k=args.candidate_k, seed=args.seed)
    save_mat(out, args.output)
    print(f"Saved {args.output}; anomalies={int(out.labels.sum())}")


if __name__ == "__main__":
    main()
