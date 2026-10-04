from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from autogad_reproduction.data import load_mat
from autogad_reproduction.experiments import run_search, evaluate_selected, dump_result
from autogad_reproduction.research_logger import ResearchLogger


def main():
    p = argparse.ArgumentParser(prog="autogad")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--model", default="anemone")
    run.add_argument("--search-space", default="anemone")
    run.add_argument("--search-config", default="configs/search_spaces.yaml")
    run.add_argument("--anemone-config", default="configs/experiments.yaml")
    run.add_argument("--k", type=int, default=None, help="pseudo-anomaly count used by CSM")
    run.add_argument("--max-configs", type=int, default=None)
    run.add_argument("--output", default="outputs/autogad_result")
    run.add_argument("--device", default=None)
    run.add_argument("--log-batches", action="store_true", help="write per-batch metrics to batch_metrics.csv")
    run.add_argument("--no-plots", action="store_true", help="disable automatic research plots")
    args = p.parse_args()

    if args.command == "run":
        graph = load_mat(args.dataset)
        with open(args.search_config, "r", encoding="utf-8") as f:
            spaces = yaml.safe_load(f)
        if args.search_space not in spaces:
            raise KeyError(f"Search space {args.search_space} not found")
        with open(args.anemone_config, "r", encoding="utf-8") as f:
            exp = yaml.safe_load(f)
        defaults = dict(exp.get("anemone", {}))
        if args.device:
            defaults["device"] = args.device
        anomaly_k = args.k or (int(round(0.05 * graph.num_nodes)))
        logger = ResearchLogger(args.output, log_batches=args.log_batches)
        command = " ".join(__import__("sys").argv)
        dataset_meta = {
            "name": graph.name,
            "num_nodes": graph.num_nodes,
            "num_features": graph.num_features,
            "num_edges": graph.num_edges,
            "labels_available": graph.labels is not None,
        }
        logger.run_started(
            command=command, dataset_path=args.dataset, dataset_meta=dataset_meta,
            model=args.model, search_space=spaces[args.search_space],
            anomaly_k=anomaly_k, model_defaults=defaults,
        )
        try:
            result = run_search(
                graph, args.model, spaces[args.search_space], defaults, anomaly_k, args.max_configs, logger=logger
            )
            dump_result(result, args.output, make_plots=not args.no_plots)
            print(f"Best configuration: {result.best_params}")
            print(f"Best CSM: {result.best_csm:.6f}")
            scores, auc = evaluate_selected(
                graph, args.model, result.best_params, defaults,
                logger=logger, output_dir=None if args.no_plots else args.output,
            )
            if args.no_plots:
                import numpy as np
                np.save(__import__('pathlib').Path(args.output) / "final_scores.npy", np.asarray(scores))
            if auc is not None:
                print(f"Final ROC-AUC (evaluation only): {auc:.6f}")
                logger.log("EVAL", "ROC_AUC", roc_auc=auc)
            logger.close(status="completed")
        except Exception as exc:
            logger.log("ERROR", "RUN_FAILED", error=repr(exc))
            logger.close(status="failed")
            raise


if __name__ == "__main__":
    main()
