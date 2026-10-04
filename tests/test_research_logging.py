import numpy as np
import scipy.sparse as sp

from autogad_reproduction.data import GraphData
from autogad_reproduction.experiments import run_search, dump_result, evaluate_selected
from autogad_reproduction.research_logger import ResearchLogger


def test_research_logging_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    n = 16
    X = rng.normal(size=(n, 5)).astype(np.float32)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    labels = np.zeros(n, dtype=np.uint8)
    labels[-2:] = 1
    graph = GraphData(X, sp.csr_matrix(A), labels=labels, name="logtest")
    out = tmp_path / "run"
    logger = ResearchLogger(out, log_batches=True, console=False)
    defaults = dict(
        hidden_dim=8, epochs=2, batch_size=4, lr=1e-3,
        inference_rounds=2, device="cpu", seed=42,
        negsamp_round_patch=2, negsamp_round_context=1,
        weight_decay=0.0, restart_prob=0.15,
        resample_views_each_epoch=False,
    )
    search_space = {"K": [2], "alpha": [0.8]}
    logger.run_started(
        command="pytest", dataset_path="toy",
        dataset_meta={"name": "logtest", "num_nodes": n, "num_features": 5, "num_edges": n, "labels_available": True},
        model="anemone", search_space=search_space, anomaly_k=2,
        model_defaults=defaults,
    )
    result = run_search(graph, "anemone", search_space, defaults, 2, logger=logger)
    dump_result(result, out, make_plots=True)
    scores, auc = evaluate_selected(graph, "anemone", result.best_params, defaults, logger=logger, output_dir=out)
    logger.close()

    expected = [
        "log.txt", "events.jsonl", "run_metadata.json", "epoch_metrics.csv",
        "batch_metrics.csv", "inference_round_metrics.csv", "search_results.csv",
        "search_results.json", "best_scores.npy", "final_scores.npy", "final_node_scores.csv",
        "plots/search_csm_vs_config.png", "plots/csm_heatmap_K_alpha.png",
        "plots/training_loss_total.png", "plots/final_score_distribution.png",
        "plots/final_roc_curve.png", "plots/inference_round_stability.png",
    ]
    for rel in expected:
        assert (out / rel).exists(), rel
    assert len(np.load(out / "final_scores.npy")) == n
    assert np.isfinite(auc)
