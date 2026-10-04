from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from autogad_reproduction.data import GraphData
from autogad_reproduction.models.base import GraphAnomalyDetector


@dataclass
class ANEMONEConfig:
    K: int = 4
    alpha: float = 0.8
    hidden_dim: int = 64
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    restart_prob: float = 0.15
    inference_rounds: int = 32
    negsamp_round_patch: int = 1
    negsamp_round_context: int = 1
    seed: int = 42
    device: str = "cpu"
    resample_views_each_epoch: bool = False


class _ANEMONECore(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.patch_theta = nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.patch_w = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.context_phi = nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.context_w = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.patch_theta)
        nn.init.xavier_uniform_(self.patch_w)
        nn.init.xavier_uniform_(self.context_phi)
        nn.init.xavier_uniform_(self.context_w)

    @staticmethod
    def _gcn(adj: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # adj: [B,K,K], x: [B,K,F]
        b, k, _ = x.shape
        eye = torch.eye(k, device=x.device, dtype=x.dtype).expand(b, -1, -1)
        a = adj + eye
        deg = a.sum(-1).clamp_min(1.0)
        d = deg.rsqrt()
        norm = d.unsqueeze(-1) * a * d.unsqueeze(-2)
        h = torch.bmm(norm, x)
        h = h @ weight
        return torch.relu(h)

    def patch(self, adj: torch.Tensor, x_masked: torch.Tensor, x_target: torch.Tensor):
        H = self._gcn(adj, x_masked, self.patch_theta)
        h = H[:, 0, :]
        z = torch.relu(x_target @ self.patch_theta)
        return h, z

    def context(self, adj: torch.Tensor, x_masked: torch.Tensor, x_target: torch.Tensor):
        H = self._gcn(adj, x_masked, self.context_phi)
        h = H.mean(dim=1)
        z = torch.relu(x_target @ self.context_phi)
        return h, z

    @staticmethod
    def bilinear(h: torch.Tensor, z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((h @ W * z).sum(-1))

    @staticmethod
    def _negative_indices(batch_size: int, rounds: int, device: torch.device) -> list[torch.Tensor]:
        if rounds < 0:
            raise ValueError("negative-sampling rounds must be >= 0")
        if batch_size < 2 and rounds > 0:
            raise ValueError("negative sampling requires a batch size >= 2")
        base = torch.arange(batch_size, device=device)
        return [torch.roll(base, shifts=r) for r in range(1, rounds + 1)]

    def forward_batch(
        self,
        patch_adj,
        patch_x_masked,
        context_adj,
        context_x_masked,
        x_target,
        negsamp_round_patch: int,
        negsamp_round_context: int,
    ):
        hp, zp = self.patch(patch_adj, patch_x_masked, x_target)
        hc, zc = self.context(context_adj, context_x_masked, x_target)
        sp_pos = self.bilinear(hp, zp, self.patch_w)
        sc_pos = self.bilinear(hc, zc, self.context_w)

        sp_negs = []
        for perm in self._negative_indices(hp.shape[0], negsamp_round_patch, hp.device):
            sp_negs.append(self.bilinear(hp[perm], zp, self.patch_w))

        sc_negs = []
        for perm in self._negative_indices(hc.shape[0], negsamp_round_context, hc.device):
            sc_negs.append(self.bilinear(hc[perm], zc, self.context_w))

        sp_neg = torch.stack(sp_negs, dim=0) if sp_negs else sp_pos.new_empty((0, sp_pos.shape[0]))
        sc_neg = torch.stack(sc_negs, dim=0) if sc_negs else sc_pos.new_empty((0, sc_pos.shape[0]))
        return sp_pos, sp_neg, sc_pos, sc_neg


class ANEMONE(GraphAnomalyDetector):
    name = "anemone"

    def __init__(self, logger=None, run_context: Mapping[str, Any] | None = None, **kwargs):
        self.logger = logger
        self.run_context = dict(run_context or {})
        self.cfg = ANEMONEConfig(**kwargs)
        self.model: _ANEMONECore | None = None
        self.graph: GraphData | None = None
        self.device = torch.device(self.cfg.device)
        self._train_views = None

    @staticmethod
    def _rwr_order(adjacency: sp.csr_matrix, start: int, K: int, restart_prob: float, rng: np.random.Generator) -> np.ndarray:
        n = adjacency.shape[0]
        if K >= n:
            return np.arange(n, dtype=np.int64)
        neighbors = [adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]] for i in range(n)]
        selected = [int(start)]
        current = int(start)
        max_steps = max(1000, K * 50)
        for _ in range(max_steps):
            if len(selected) >= K:
                break
            if rng.random() < restart_prob or len(neighbors[current]) == 0:
                current = int(start)
            else:
                current = int(rng.choice(neighbors[current]))
            if current not in selected:
                selected.append(current)
        if len(selected) < K:
            remaining = np.setdiff1d(np.arange(n), np.array(selected, dtype=np.int64), assume_unique=False)
            extra = rng.choice(remaining, size=K-len(selected), replace=False)
            selected.extend(extra.tolist())
        return np.asarray(selected[:K], dtype=np.int64)

    def _make_views(self, target_ids: np.ndarray, rng: np.random.Generator):
        assert self.graph is not None
        K = min(self.cfg.K, self.graph.num_nodes)
        A = self.graph.adjacency
        X = self.graph.features
        patch_adjs = np.zeros((len(target_ids), K, K), dtype=np.float32)
        patch_xm = np.zeros((len(target_ids), K, X.shape[1]), dtype=np.float32)
        context_adjs = np.zeros((len(target_ids), K, K), dtype=np.float32)
        context_xm = np.zeros((len(target_ids), K, X.shape[1]), dtype=np.float32)
        xt = X[target_ids].copy().astype(np.float32)
        for b, node in enumerate(target_ids):
            patch_idx = self._rwr_order(A, int(node), K, self.cfg.restart_prob, rng)
            context_idx = self._rwr_order(A, int(node), K, self.cfg.restart_prob, rng)
            pa = A[patch_idx][:, patch_idx].toarray().astype(np.float32)
            ca = A[context_idx][:, context_idx].toarray().astype(np.float32)
            px = X[patch_idx].copy().astype(np.float32)
            cx = X[context_idx].copy().astype(np.float32)
            px[0] = 0.0
            cx[0] = 0.0
            patch_adjs[b], patch_xm[b] = pa, px
            context_adjs[b], context_xm[b] = ca, cx
        return patch_adjs, patch_xm, context_adjs, context_xm, xt

    def _train(self):
        assert self.graph is not None and self.model is not None
        cfg = self.cfg
        n = self.graph.num_nodes
        logger = self.logger
        phase = self.run_context.get("phase", "search")
        config_id = self.run_context.get("config_id", "unknown")
        params = self.run_context.get("params", {"K": cfg.K, "alpha": cfg.alpha})
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(cfg.seed)
        train_ids = np.arange(n)
        cached = None
        training_start = time.perf_counter()
        if logger is not None:
            logger.training_start(
                phase=phase, config_id=config_id, params=params,
                n_nodes=n, n_features=self.graph.num_features,
                n_edges=self.graph.num_edges, device=str(self.device),
            )
        if not cfg.resample_views_each_epoch:
            view_t0 = time.perf_counter()
            cached = self._make_views(train_ids, rng)
            view_elapsed = time.perf_counter() - view_t0
            if logger is not None:
                logger.training_views(
                    phase=phase, config_id=config_id, generation_sec=view_elapsed,
                    K=min(cfg.K, n), resampled_each_epoch=False,
                )
        for epoch in range(cfg.epochs):
            epoch_t0 = time.perf_counter()
            rng_epoch = np.random.default_rng(cfg.seed + epoch)
            if cfg.resample_views_each_epoch:
                view_t0 = time.perf_counter()
                cached = self._make_views(train_ids, rng_epoch)
                if logger is not None:
                    logger.training_views(
                        phase=phase, config_id=config_id,
                        generation_sec=time.perf_counter() - view_t0,
                        K=min(cfg.K, n), resampled_each_epoch=True,
                    )
            patch_adjs, patch_xmask, context_adjs, context_xmask, xt = cached
            perm = rng_epoch.permutation(n)
            epoch_acc = {
                "loss_total": 0.0, "loss_patch": 0.0, "loss_context": 0.0,
                "patch_pos_mean": 0.0, "patch_neg_mean": 0.0,
                "context_pos_mean": 0.0, "context_neg_mean": 0.0,
                "grad_norm": 0.0, "batch_count": 0,
            }
            for batch_index, start in enumerate(range(0, n, cfg.batch_size)):
                ids = perm[start:start + cfg.batch_size]
                if len(ids) < 2:
                    continue
                # The cached arrays are indexed in the original node order.
                patch_adj_b = torch.from_numpy(patch_adjs[ids]).to(self.device)
                patch_xm_b = torch.from_numpy(patch_xmask[ids]).to(self.device)
                context_adj_b = torch.from_numpy(context_adjs[ids]).to(self.device)
                context_xm_b = torch.from_numpy(context_xmask[ids]).to(self.device)
                xt_b = torch.from_numpy(xt[ids]).to(self.device)
                sp_pos, sp_neg, sc_pos, sc_neg = self.model.forward_batch(
                    patch_adj_b, patch_xm_b, context_adj_b, context_xm_b, xt_b,
                    cfg.negsamp_round_patch, cfg.negsamp_round_context,
                )
                eps = 1e-8
                loss_p_pos = -torch.log(sp_pos + eps).mean()
                loss_c_pos = -torch.log(sc_pos + eps).mean()
                loss_p_neg = (-torch.log(1.0 - sp_neg + eps).mean()
                              if sp_neg.numel() else sp_pos.new_tensor(0.0))
                loss_c_neg = (-torch.log(1.0 - sc_neg + eps).mean()
                              if sc_neg.numel() else sc_pos.new_tensor(0.0))
                loss_p = 0.5 * (loss_p_pos + loss_p_neg)
                loss_c = 0.5 * (loss_c_pos + loss_c_neg)
                loss = cfg.alpha * loss_c + (1.0 - cfg.alpha) * loss_p
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_sq = 0.0
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        grad_sq += float(parameter.grad.detach().norm(2).item() ** 2)
                grad_norm = math.sqrt(grad_sq)
                optimizer.step()

                batch_count = epoch_acc["batch_count"] + 1
                epoch_acc["loss_total"] += float(loss.item())
                epoch_acc["loss_patch"] += float(loss_p.item())
                epoch_acc["loss_context"] += float(loss_c.item())
                epoch_acc["patch_pos_mean"] += float(sp_pos.detach().mean().item())
                epoch_acc["patch_neg_mean"] += float(sp_neg.detach().mean().item()) if sp_neg.numel() else 0.0
                epoch_acc["context_pos_mean"] += float(sc_pos.detach().mean().item())
                epoch_acc["context_neg_mean"] += float(sc_neg.detach().mean().item()) if sc_neg.numel() else 0.0
                epoch_acc["grad_norm"] += grad_norm
                epoch_acc["batch_count"] = batch_count

                if logger is not None:
                    logger.batch_metrics({
                        "phase": phase, "config_id": config_id, "epoch": epoch + 1,
                        "batch": batch_index + 1, "batch_size": len(ids),
                        "loss_total": float(loss.item()), "loss_patch": float(loss_p.item()),
                        "loss_context": float(loss_c.item()),
                        "patch_pos_mean": float(sp_pos.detach().mean().item()),
                        "patch_neg_mean": float(sp_neg.detach().mean().item()) if sp_neg.numel() else np.nan,
                        "context_pos_mean": float(sc_pos.detach().mean().item()),
                        "context_neg_mean": float(sc_neg.detach().mean().item()) if sc_neg.numel() else np.nan,
                        "grad_norm": grad_norm, "learning_rate": optimizer.param_groups[0]["lr"],
                    })

            count = max(int(epoch_acc["batch_count"]), 1)
            row = {
                "phase": phase, "config_id": config_id, "epoch": epoch + 1,
                "loss_total": epoch_acc["loss_total"] / count,
                "loss_patch": epoch_acc["loss_patch"] / count,
                "loss_context": epoch_acc["loss_context"] / count,
                "patch_pos_mean": epoch_acc["patch_pos_mean"] / count,
                "patch_neg_mean": epoch_acc["patch_neg_mean"] / count,
                "context_pos_mean": epoch_acc["context_pos_mean"] / count,
                "context_neg_mean": epoch_acc["context_neg_mean"] / count,
                "grad_norm": epoch_acc["grad_norm"] / count,
                "batch_count": epoch_acc["batch_count"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_runtime_sec": time.perf_counter() - epoch_t0,
            }
            if logger is not None:
                logger.epoch_metrics(row)

        if logger is not None:
            logger.training_end(
                phase=phase, config_id=config_id, elapsed_sec=time.perf_counter() - training_start,
                epochs=cfg.epochs,
            )

    def fit(self, graph: GraphData, params: Mapping[str, Any]) -> "ANEMONE":
        merged = {**self.cfg.__dict__, **dict(params)}
        self.cfg = ANEMONEConfig(**merged)
        if self.cfg.K < 2:
            raise ValueError("ANEMONE K must be >= 2")
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.graph = graph
        self.device = torch.device(self.cfg.device)
        self.model = _ANEMONECore(graph.num_features, self.cfg.hidden_dim).to(self.device)
        self._train()
        return self

    @torch.no_grad()
    def score(self, graph: GraphData | None = None) -> np.ndarray:
        if graph is None:
            graph = self.graph
        if graph is None or self.model is None:
            raise RuntimeError("Call fit() before score()")
        self.model.eval()
        cfg = self.cfg
        n = graph.num_nodes
        phase = self.run_context.get("phase", "search")
        config_id = self.run_context.get("config_id", "unknown")
        rng = np.random.default_rng(cfg.seed + 100000)
        patch_base = np.zeros((n, cfg.inference_rounds), dtype=np.float64)
        context_base = np.zeros((n, cfg.inference_rounds), dtype=np.float64)
        X = graph.features
        logger = self.logger
        if logger is not None:
            logger.scoring_start(phase=phase, config_id=config_id, inference_rounds=cfg.inference_rounds, K=min(cfg.K, n))
        score_start = time.perf_counter()
        for r in range(cfg.inference_rounds):
            round_t0 = time.perf_counter()
            for i in range(n):
                patch_ids = self._rwr_order(graph.adjacency, i, min(cfg.K, n), cfg.restart_prob, rng)
                context_ids = self._rwr_order(graph.adjacency, i, min(cfg.K, n), cfg.restart_prob, rng)
                patch_A = graph.adjacency[patch_ids][:, patch_ids].toarray().astype(np.float32)[None, ...]
                context_A = graph.adjacency[context_ids][:, context_ids].toarray().astype(np.float32)[None, ...]
                patch_X = X[patch_ids].copy().astype(np.float32)
                context_X = X[context_ids].copy().astype(np.float32)
                patch_X[0] = 0.0
                context_X[0] = 0.0
                Xt_t = torch.from_numpy(X[i][None, ...]).to(self.device)
                hp, zp = self.model.patch(torch.from_numpy(patch_A).to(self.device), torch.from_numpy(patch_X[None]).to(self.device), Xt_t)
                hc, zc = self.model.context(torch.from_numpy(context_A).to(self.device), torch.from_numpy(context_X[None]).to(self.device), Xt_t)
                if n > 1:
                    j = int(rng.integers(0, n - 1))
                    if j >= i:
                        j += 1
                else:
                    j = i
                neg_ids = self._rwr_order(graph.adjacency, j, min(cfg.K, n), cfg.restart_prob, rng)
                A_neg = graph.adjacency[neg_ids][:, neg_ids].toarray().astype(np.float32)[None, ...]
                X_neg = X[neg_ids].copy().astype(np.float32)
                X_neg[0] = 0.0
                hn_p, _ = self.model.patch(torch.from_numpy(A_neg).to(self.device), torch.from_numpy(X_neg[None]).to(self.device), Xt_t)
                hn_c, _ = self.model.context(torch.from_numpy(A_neg).to(self.device), torch.from_numpy(X_neg[None]).to(self.device), Xt_t)
                sp = float(self.model.bilinear(hp, zp, self.model.patch_w).item())
                snp = float(self.model.bilinear(hn_p, zp, self.model.patch_w).item())
                sc = float(self.model.bilinear(hc, zc, self.model.context_w).item())
                snc = float(self.model.bilinear(hn_c, zc, self.model.context_w).item())
                patch_base[i, r] = snp - sp
                context_base[i, r] = snc - sc
            if logger is not None:
                logger.inference_round({
                    "phase": phase, "config_id": config_id, "round": r + 1,
                    "patch_base_mean": float(patch_base[:, r].mean()),
                    "patch_base_std": float(patch_base[:, r].std()),
                    "context_base_mean": float(context_base[:, r].mean()),
                    "context_base_std": float(context_base[:, r].std()),
                    "round_runtime_sec": time.perf_counter() - round_t0,
                })
        yp = patch_base.mean(axis=1) + patch_base.std(axis=1)
        yc = context_base.mean(axis=1) + context_base.std(axis=1)
        scores = cfg.alpha * yc + (1.0 - cfg.alpha) * yp
        if logger is not None:
            logger.scoring_end(
                phase=phase, config_id=config_id, elapsed_sec=time.perf_counter() - score_start,
                score_mean=float(scores.mean()), score_std=float(scores.std()),
                score_min=float(scores.min()), score_max=float(scores.max()),
            )
        return scores

    def fit_score(self, graph: GraphData, params: Mapping[str, Any]) -> np.ndarray:
        return self.fit(graph, params).score(graph)
