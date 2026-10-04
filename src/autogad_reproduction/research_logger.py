from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class ResearchLogger:
    """Structured research logger for reproducible AutoGAD experiments.

    Artifacts written under ``output_dir``:
      - log.txt: human-readable chronological log
      - events.jsonl: machine-readable event stream
      - epoch_metrics.csv: per-epoch training metrics
      - batch_metrics.csv: optional per-batch training metrics
      - run_metadata.json: environment, dataset, command and configuration metadata
      - search_results.csv/json: written by experiments.dump_result
      - plots/: research visualizations written during finalization
    """

    def __init__(self, output_dir: str | Path, *, log_batches: bool = False, console: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "log.txt"
        self.events_path = self.output_dir / "events.jsonl"
        self.epoch_csv_path = self.output_dir / "epoch_metrics.csv"
        self.batch_csv_path = self.output_dir / "batch_metrics.csv"
        self.round_csv_path = self.output_dir / "inference_round_metrics.csv"
        self.metadata_path = self.output_dir / "run_metadata.json"
        self.log_batches = log_batches
        self.console = console
        self._start_time = time.perf_counter()
        self._epoch_fields: list[str] | None = None
        self._batch_fields: list[str] | None = None
        self._round_fields: list[str] | None = None
        self._metadata: dict[str, Any] = {}
        self.log("INFO", "Research logging initialized", output_dir=str(self.output_dir))

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if is_dataclass(value):
            return ResearchLogger._jsonable(asdict(value))
        if isinstance(value, Mapping):
            return {str(k): ResearchLogger._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ResearchLogger._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return str(value)
        return value

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log(self, level: str, message: str, **fields: Any) -> None:
        ts = self._timestamp()
        suffix = ""
        if fields:
            suffix = " | " + " | ".join(
                f"{k}={self._display(v)}" for k, v in fields.items()
            )
        line = f"{ts} | {level:<5} | {message}{suffix}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        event = {"timestamp": ts, "level": level, "message": message, **{k: self._display(v) for k, v in fields.items()}}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.console:
            print(line, flush=True)

    def run_started(self, *, command: str, dataset_path: str, dataset_meta: Mapping[str, Any], model: str,
                    search_space: Mapping[str, Any], anomaly_k: int, model_defaults: Mapping[str, Any]) -> None:
        self._metadata.update({
            "started_at_utc": self._timestamp(),
            "command": command,
            "dataset_path": str(dataset_path),
            "dataset": self._jsonable(dataset_meta),
            "model": model,
            "search_space": self._jsonable(search_space),
            "anomaly_k": anomaly_k,
            "model_defaults": self._jsonable(model_defaults),
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        })
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log("INFO", "RUN_START", command=command, dataset=dataset_path, model=model,
                 nodes=dataset_meta.get("num_nodes"), features=dataset_meta.get("num_features"),
                 edges=dataset_meta.get("num_edges"), anomaly_k=anomaly_k)

    def config_start(self, config_id: int, params: Mapping[str, Any], total_configs: int) -> None:
        self.log("INFO", "CONFIG_START", config_id=config_id, total_configs=total_configs, params=dict(params))

    def config_end(self, config_id: int, params: Mapping[str, Any], csm_result: Any, elapsed_sec: float) -> None:
        self.log(
            "INFO",
            "CONFIG_END",
            config_id=config_id,
            params=dict(params),
            csm=float(csm_result.csm),
            anomaly_mean=float(csm_result.anomaly_mean),
            anomaly_variance=float(csm_result.anomaly_variance),
            normal_mean=float(csm_result.normal_mean),
            normal_variance=float(csm_result.normal_variance),
            elapsed_sec=float(elapsed_sec),
        )

    def batch_metrics(self, row: Mapping[str, Any]) -> None:
        if not self.log_batches:
            return
        self._append_csv(self.batch_csv_path, row, "batch")

    def epoch_metrics(self, row: Mapping[str, Any]) -> None:
        self._append_csv(self.epoch_csv_path, row, "epoch")
        self.log(
            "TRAIN",
            "EPOCH_END",
            phase=row.get("phase"),
            config_id=row.get("config_id"),
            epoch=row.get("epoch"),
            loss_total=row.get("loss_total"),
            loss_patch=row.get("loss_patch"),
            loss_context=row.get("loss_context"),
            patch_pos_mean=row.get("patch_pos_mean"),
            patch_neg_mean=row.get("patch_neg_mean"),
            context_pos_mean=row.get("context_pos_mean"),
            context_neg_mean=row.get("context_neg_mean"),
            grad_norm=row.get("grad_norm"),
            elapsed_sec=row.get("epoch_runtime_sec"),
        )

    def training_start(self, *, phase: str, config_id: int | str, params: Mapping[str, Any], n_nodes: int,
                       n_features: int, n_edges: int, device: str) -> None:
        self.log("TRAIN", "TRAINING_START", phase=phase, config_id=config_id, params=dict(params),
                 nodes=n_nodes, features=n_features, edges=n_edges, device=device)

    def training_views(self, *, phase: str, config_id: int | str, generation_sec: float, K: int,
                       resampled_each_epoch: bool) -> None:
        self.log("TRAIN", "VIEW_GENERATION", phase=phase, config_id=config_id,
                 generation_sec=generation_sec, K=K, resampled_each_epoch=resampled_each_epoch)

    def training_end(self, *, phase: str, config_id: int | str, elapsed_sec: float, epochs: int) -> None:
        self.log("TRAIN", "TRAINING_END", phase=phase, config_id=config_id, elapsed_sec=elapsed_sec, epochs=epochs)

    def scoring_start(self, *, phase: str, config_id: int | str, inference_rounds: int, K: int) -> None:
        self.log("SCORE", "SCORING_START", phase=phase, config_id=config_id,
                 inference_rounds=inference_rounds, K=K)

    def inference_round(self, row: Mapping[str, Any]) -> None:
        self._append_csv(self.round_csv_path, row, "round")
        self.log(
            "SCORE",
            "INFERENCE_ROUND",
            phase=row.get("phase"),
            config_id=row.get("config_id"),
            round=row.get("round"),
            patch_base_mean=row.get("patch_base_mean"),
            patch_base_std=row.get("patch_base_std"),
            context_base_mean=row.get("context_base_mean"),
            context_base_std=row.get("context_base_std"),
            elapsed_sec=row.get("round_runtime_sec"),
        )

    def selection(self, *, best_params: Mapping[str, Any], best_csm: float) -> None:
        self._metadata["selected_params"] = self._jsonable(dict(best_params))
        self._metadata["selected_csm"] = float(best_csm)
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log("SEARCH", "BEST_CONFIGURATION", best_params=dict(best_params), best_csm=float(best_csm))

    def scoring_end(self, *, phase: str, config_id: int | str, elapsed_sec: float,
                    score_mean: float, score_std: float, score_min: float, score_max: float) -> None:
        self.log("SCORE", "SCORING_END", phase=phase, config_id=config_id, elapsed_sec=elapsed_sec,
                 score_mean=score_mean, score_std=score_std, score_min=score_min, score_max=score_max)

    def evaluation(self, *, auc: float | None, scores: np.ndarray, labels: np.ndarray | None) -> None:
        self._metadata["final_roc_auc"] = None if auc is None else float(auc)
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log("EVAL", "FINAL_EVALUATION", roc_auc=auc,
                 score_mean=float(np.mean(scores)), score_std=float(np.std(scores)),
                 score_min=float(np.min(scores)), score_max=float(np.max(scores)),
                 labels_available=labels is not None)

    def close(self, *, status: str = "completed") -> None:
        elapsed = time.perf_counter() - self._start_time
        self._metadata["finished_at_utc"] = self._timestamp()
        self._metadata["status"] = status
        self._metadata["wall_time_sec"] = elapsed
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log("INFO", "RUN_END", status=status, wall_time_sec=elapsed)

    def _append_csv(self, path: Path, row: Mapping[str, Any], kind: str) -> None:
        payload = {k: self._jsonable(v) for k, v in row.items()}
        if kind == "epoch":
            fields = self._epoch_fields
        elif kind == "batch":
            fields = self._batch_fields
        else:
            fields = self._round_fields
        if fields is None:
            fields = list(payload.keys())
            if kind == "epoch":
                self._epoch_fields = fields
            elif kind == "batch":
                self._batch_fields = fields
            else:
                self._round_fields = fields
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({field: payload.get(field) for field in fields})
            return
        new_fields = [f for f in payload if f not in fields]
        if new_fields:
            fields = fields + new_fields
            if kind == "epoch":
                self._epoch_fields = fields
            elif kind == "batch":
                self._batch_fields = fields
            else:
                self._round_fields = fields
            rows = []
            if path.exists():
                with path.open("r", newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow({field: payload.get(field) for field in fields})
            return
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow({field: payload.get(field) for field in fields})

    @staticmethod
    def _display(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            if np.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{float(value):.4f}"

        if isinstance(value, (np.integer,)):
            return str(int(value))

        if isinstance(value, bool):
            return str(value)

        if isinstance(value, dict):
            return str(value)

        return str(value)
