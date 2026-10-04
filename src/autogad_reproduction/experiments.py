from __future__ import annotations
from pathlib import Path
from typing import Mapping, Any
import json
import yaml
import numpy as np

from autogad_reproduction.autogad.search import grid_search
from autogad_reproduction.data import GraphData
from autogad_reproduction.evaluation import roc_auc
from autogad_reproduction.models.registry import build_model
from autogad_reproduction.seed import set_seed
from autogad_reproduction.visualization import finalize_visualizations


def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_search(
    graph: GraphData,
    model_name: str,
    search_space: Mapping[str, list],
    model_defaults: Mapping[str, Any],
    anomaly_k: int,
    max_configs: int | None = None,
    logger=None,
):
    def score_fn(params, config_id):
        set_seed(int(model_defaults.get("seed", 42)))
        context = {"phase": "search", "config_id": config_id, "params": dict(params)}
        model = build_model(model_name, logger=logger, run_context=context, **model_defaults)
        return model.fit_score(graph, params)
    result = grid_search(
        score_fn, search_space, anomaly_k=anomaly_k, max_configs=max_configs,
        logger=logger,
    )
    if logger is not None:
        logger.selection(best_params=result.best_params, best_csm=result.best_csm)
    return result


def evaluate_selected(
    graph: GraphData, model_name: str, params: Mapping[str, Any],
    model_defaults: Mapping[str, Any], logger=None, output_dir: str | Path | None = None,
):
    set_seed(int(model_defaults.get("seed", 42)))
    context = {"phase": "final_evaluation", "config_id": "final", "params": dict(params)}
    model = build_model(model_name, logger=logger, run_context=context, **model_defaults)
    scores = model.fit_score(graph, params)
    auc = None if graph.labels is None else roc_auc(scores, graph.labels)
    if logger is not None:
        logger.evaluation(auc=auc, scores=scores, labels=graph.labels)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "final_scores.npy", np.asarray(scores))
        node_rows = {"node_id": np.arange(graph.num_nodes), "score": np.asarray(scores)}
        if graph.labels is not None:
            node_rows["label"] = np.asarray(graph.labels).reshape(-1)
            node_rows["rank"] = np.argsort(-scores, kind="mergesort").argsort() + 1
        import pandas as pd
        pd.DataFrame(node_rows).sort_values("score", ascending=False).to_csv(out / "final_node_scores.csv", index=False)
        finalize_visualizations(out, scores=scores, labels=graph.labels)
    return scores, auc


def dump_result(result, path: str | Path, *, make_plots: bool = True):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(path / "search_results.csv", index=False)
    payload = {"best_params": result.best_params, "best_csm": result.best_csm}
    (path / "search_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(path / "best_scores.npy", np.asarray(result.best_scores))
    if make_plots:
        finalize_visualizations(path)
