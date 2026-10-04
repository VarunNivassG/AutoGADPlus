from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_search_results(search_csv: str | Path, plot_dir: str | Path) -> None:
    search_csv = Path(search_csv)
    plot_dir = Path(plot_dir)
    df = pd.read_csv(search_csv)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["config_id"], df["csm"], marker="o", linewidth=1)
    best_idx = int(df["csm"].idxmax())
    ax.scatter([df.loc[best_idx, "config_id"]], [df.loc[best_idx, "csm"]], s=80, zorder=3)
    ax.set_title("AutoGAD Search: CSM by Hyperparameter Configuration")
    ax.set_xlabel("Configuration ID")
    ax.set_ylabel("CSM")
    ax.grid(alpha=0.25)
    _save(fig, plot_dir / "search_csm_vs_config.png")

    if "runtime_sec" in df:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["config_id"], df["runtime_sec"], marker="o", linewidth=1)
        ax.set_title("AutoGAD Search Runtime by Configuration")
        ax.set_xlabel("Configuration ID")
        ax.set_ylabel("Runtime (s)")
        ax.grid(alpha=0.25)
        _save(fig, plot_dir / "search_runtime_vs_config.png")

    if {"K", "alpha"}.issubset(df.columns):
        pivot = df.pivot(index="K", columns="alpha", values="csm").sort_index()
        fig, ax = plt.subplots(figsize=(11, 5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto")
        fig.colorbar(image, ax=ax, label="CSM")
        ax.set_title("CSM Heatmap over (K, α)")
        ax.set_xlabel("α")
        ax.set_ylabel("K")
        ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        _save(fig, plot_dir / "csm_heatmap_K_alpha.png")

    columns = [c for c in ["anomaly_mean", "normal_mean", "anomaly_variance", "normal_variance"] if c in df]
    if columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in columns:
            ax.plot(df["config_id"], df[col], marker="o", linewidth=1, label=col)
        ax.set_title("CSM Components Across Search")
        ax.set_xlabel("Configuration ID")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, plot_dir / "csm_components.png")


def plot_training_curves(epoch_csv: str | Path, plot_dir: str | Path, *, phase: str = "final_evaluation") -> None:
    path = Path(epoch_csv)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    if "phase" in df.columns:
        subset = df[df["phase"].astype(str) == phase]
        if subset.empty:
            subset = df
    else:
        subset = df
    # Plot the latest/final training run for a clean research figure.
    group_cols = [c for c in ["config_id"] if c in subset.columns]
    if group_cols:
        last_cfg = subset["config_id"].iloc[-1]
        subset = subset[subset["config_id"] == last_cfg]
    subset = subset.sort_values("epoch")

    for metric, title, filename in [
        ("loss_total", "ANEMONE Training: Total Loss", "training_loss_total.png"),
        ("loss_patch", "ANEMONE Training: Patch / Node-Node Loss", "training_loss_patch.png"),
        ("loss_context", "ANEMONE Training: Context / Node-Subgraph Loss", "training_loss_context.png"),
    ]:
        if metric not in subset.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(subset["epoch"], subset[metric], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        _save(fig, Path(plot_dir) / filename)

    score_metrics = [c for c in ["patch_pos_mean", "patch_neg_mean", "context_pos_mean", "context_neg_mean"] if c in subset.columns]
    if score_metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for metric in score_metrics:
            ax.plot(subset["epoch"], subset[metric], linewidth=1.5, label=metric)
        ax.set_title("ANEMONE Positive / Negative Discriminator Scores")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean score")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, Path(plot_dir) / "training_discriminator_scores.png")


def plot_final_scores(scores: np.ndarray, labels: np.ndarray | None, plot_dir: str | Path) -> None:
    plot_dir = Path(plot_dir)
    scores = np.asarray(scores).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(scores) + 1), np.sort(scores)[::-1], linewidth=1.2)
    ax.set_title("Final Anomaly Scores Ranked by Score")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Anomaly score")
    ax.grid(alpha=0.25)
    _save(fig, plot_dir / "final_score_rank.png")

    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        fig, ax = plt.subplots(figsize=(10, 5))
        normal = scores[labels == 0]
        anomaly = scores[labels == 1]
        ax.hist(normal, bins=30, alpha=0.65, label=f"Normal (n={len(normal)})")
        ax.hist(anomaly, bins=30, alpha=0.65, label=f"Anomaly (n={len(anomaly)})")
        ax.set_title("Final Anomaly-Score Distribution (Evaluation Labels)")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, plot_dir / "final_score_distribution.png")

        if len(np.unique(labels)) >= 2:
            fpr, tpr, _ = roc_curve(labels, scores)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, linewidth=2)
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            ax.set_title("Final ROC Curve (Evaluation Only)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.grid(alpha=0.25)
            _save(fig, plot_dir / "final_roc_curve.png")



def plot_inference_rounds(round_csv: str | Path, plot_dir: str | Path, *, phase: str = "final_evaluation") -> None:
    path = Path(round_csv)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    if "phase" in df.columns:
        subset = df[df["phase"].astype(str) == phase]
    else:
        subset = df
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if "patch_base_mean" in subset:
        ax.plot(subset["round"], subset["patch_base_mean"], marker="o", linewidth=1.4, label="patch base mean")
    if "context_base_mean" in subset:
        ax.plot(subset["round"], subset["context_base_mean"], marker="o", linewidth=1.4, label="context base mean")
    ax.set_title("Inference-Round Anomaly-Signal Stability")
    ax.set_xlabel("Inference round")
    ax.set_ylabel("Negative score − positive score")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, Path(plot_dir) / "inference_round_stability.png")


def finalize_visualizations(output_dir: str | Path, scores: np.ndarray | None = None,
                            labels: np.ndarray | None = None) -> None:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    search_csv = output_dir / "search_results.csv"
    epoch_csv = output_dir / "epoch_metrics.csv"
    round_csv = output_dir / "inference_round_metrics.csv"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if search_csv.exists():
        plot_search_results(search_csv, plot_dir)
    if epoch_csv.exists():
        plot_training_curves(epoch_csv, plot_dir)
    if round_csv.exists():
        plot_inference_rounds(round_csv, plot_dir)
    if scores is not None:
        plot_final_scores(scores, labels, plot_dir)
