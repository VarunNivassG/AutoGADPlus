from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Any
import time

import pandas as pd

from .csm import compute_csm


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_csm: float
    table: pd.DataFrame
    best_scores: Any


def expand_grid(space: Mapping[str, list]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, values)) for values in product(*(space[k] for k in keys))]


def grid_search(
    score_fn: Callable[[dict[str, Any], int], Any],
    search_space: Mapping[str, list],
    anomaly_k: int,
    max_configs: int | None = None,
    verbose: bool = True,
    logger=None,
) -> SearchResult:
    configs = expand_grid(search_space)
    if max_configs is not None:
        configs = configs[:max_configs]
    rows = []
    best_score = float("-inf")
    best_params = None
    best_scores = None
    for idx, params in enumerate(configs):
        t0 = time.perf_counter()
        if logger is not None:
            logger.config_start(idx, params, len(configs))
        scores = score_fn(params, idx)
        csm = compute_csm(scores, anomaly_k)
        elapsed = time.perf_counter() - t0
        row = {
            "config_id": idx,
            **params,
            "csm": csm.csm,
            "anomaly_mean": csm.anomaly_mean,
            "anomaly_variance": csm.anomaly_variance,
            "normal_mean": csm.normal_mean,
            "normal_variance": csm.normal_variance,
            "runtime_sec": elapsed,
        }
        rows.append(row)
        if logger is not None:
            logger.config_end(idx, params, csm, elapsed)
        if verbose:
            print(f"[{idx+1}/{len(configs)}] {params} -> CSM={csm.csm:.6f} ({elapsed:.2f}s)")
        if csm.csm > best_score:
            best_score = csm.csm
            best_params = dict(params)
            best_scores = scores
    if best_params is None:
        raise RuntimeError("Search produced no configuration")
    return SearchResult(best_params, best_score, pd.DataFrame(rows), best_scores)
