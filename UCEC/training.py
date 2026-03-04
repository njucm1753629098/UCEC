from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .graph import RunGraph
from .schema import REL_SPECS, RelationSpec
from .utils import auroc_auprc


@dataclass
class TrainConfig:
    epochs: int = 60
    steps_per_epoch: int = 50
    batch_pos: int = 2048
    neg_per_pos: int = 1
    lr: float = 2e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _collect_nodes_by_type(run: RunGraph) -> Dict[str, np.ndarray]:
    idx = run.index
    nodes: Dict[str, List[int]] = {t: [] for t in idx.num_nodes.keys()}
    for t in idx.num_nodes.keys():
        off = idx.offsets[t]
        nodes[t] = np.arange(off, off + idx.num_nodes[t], dtype=np.int64)
    return nodes


def _build_pos_sets(run: RunGraph) -> Dict[str, set]:
    pos: Dict[str, set] = {}
    for spec in REL_SPECS:
        rel = spec.name
        s = set()
        for split, df in run.splits.edges[rel].items():
            if rel == "HI":
                pairs = zip(df["herb"].astype(str), df["ingredient"].astype(str))
            elif rel == "IP":
                pairs = zip(df["ingredient"].astype(str), df["protein"].astype(str))
            elif rel == "PD":
                pairs = zip(df["protein"].astype(str), df["disease"].astype(str))
            elif rel == "PPi":
                pairs = zip(df["protein"].astype(str), df["protein2"].astype(str))
            elif rel == "PPath":
                pairs = zip(df["protein"].astype(str), df["pathway"].astype(str))
            elif rel == "PathD":
                pairs = zip(df["pathway"].astype(str), df["disease"].astype(str))
            else:
                continue
            for a, b in pairs:
                ha = run.index.id_maps[spec.head_type][a]
                ta = run.index.id_maps[spec.tail_type][b]
                s.add((run.index.offsets[spec.head_type] + ha, run.index.offsets[spec.tail_type] + ta))
        pos[rel] = s
    return pos


def _rel_train_arrays(run: RunGraph) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    idx = run.index
    for spec in REL_SPECS:
        rel = spec.name
        df = run.splits.edges[rel]["train"]
        if rel == "HI":
            h = df["herb"].astype(str).map(idx.id_maps["herb"]).to_numpy()
            t = df["ingredient"].astype(str).map(idx.id_maps["ingredient"]).to_numpy()
            h = h + idx.offsets["herb"]
            t = t + idx.offsets["ingredient"]
        elif rel == "IP":
            h = df["ingredient"].astype(str).map(idx.id_maps["ingredient"]).to_numpy() + idx.offsets["ingredient"]
            t = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
        elif rel == "PD":
            h = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
            t = df["disease"].astype(str).map(idx.id_maps["disease"]).to_numpy() + idx.offsets["disease"]
        elif rel == "PPi":
            h = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
            t = df["protein2"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
        elif rel == "PPath":
            h = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
            t = df["pathway"].astype(str).map(idx.id_maps["pathway"]).to_numpy() + idx.offsets["pathway"]
        elif rel == "PathD":
            h = df["pathway"].astype(str).map(idx.id_maps["pathway"]).to_numpy() + idx.offsets["pathway"]
            t = df["disease"].astype(str).map(idx.id_maps["disease"]).to_numpy() + idx.offsets["disease"]
        else:
            continue
        out[rel] = (h.astype(np.int64), t.astype(np.int64))
    return out


class TypeAwareNegativeSampler:
    def __init__(self, run: RunGraph, seed: int):
        self.run = run
        self.rng = np.random.default_rng(seed)
        self.nodes_by_type = _collect_nodes_by_type(run)
        self.pos_sets = _build_pos_sets(run)

    def sample(self, rel: str, heads: np.ndarray, tails: np.ndarray, n_neg_per_pos: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        spec = next(s for s in REL_SPECS if s.name == rel)
        head_pool = self.nodes_by_type[spec.head_type]
        tail_pool = self.nodes_by_type[spec.tail_type]
        pos_set = self.pos_sets[rel]

        B = len(heads)
        neg_h = np.repeat(heads, n_neg_per_pos).copy()
        neg_t = np.repeat(tails, n_neg_per_pos).copy()

        corrupt_head = self.rng.random(size=B * n_neg_per_pos) < 0.5
        for i in range(B * n_neg_per_pos):
            if corrupt_head[i]:
                for _ in range(50):
                    h = int(self.rng.choice(head_pool))
                    if (h, int(neg_t[i])) not in pos_set:
                        neg_h[i] = h
                        break
            else:
                for _ in range(50):
                    t = int(self.rng.choice(tail_pool))
                    if (int(neg_h[i]), t) not in pos_set:
                        neg_t[i] = t
                        break
        return neg_h, neg_t


def train_gnn_model(
    run: RunGraph,
    model,
    rel_name_to_full_id: Dict[str, int],
    cfg: TrainConfig,
    seed: int,
    use_rel_types_in_encoder: bool,
) -> Dict[str, List[float]]:
    device = torch.device(cfg.device)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rel_train = _rel_train_arrays(run)
    sampler = TypeAwareNegativeSampler(run, seed=seed + 999)

    logs = {"loss": []}

    edge_index = run.train.edge_index.to(device)
    edge_type = run.train.edge_type.to(device)

    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        if use_rel_types_in_encoder:
            z = model.encode(edge_index=edge_index, edge_type=edge_type)
        else:
            z = model.encode(edge_index=edge_index)

        total_loss = 0.0
        rel_names = list(rel_train.keys())
        for step in range(cfg.steps_per_epoch):
            rel = rel_names[step % len(rel_names)]
            h_all, t_all = rel_train[rel]
            if len(h_all) == 0:
                continue
            idxs = np.random.default_rng(seed + epoch * 13 + step).integers(0, len(h_all), size=cfg.batch_pos)
            h_pos = h_all[idxs]
            t_pos = t_all[idxs]
            h_neg, t_neg = sampler.sample(rel, h_pos, t_pos, n_neg_per_pos=cfg.neg_per_pos)

            h = torch.from_numpy(np.concatenate([h_pos, h_neg])).long().to(device)
            t = torch.from_numpy(np.concatenate([t_pos, t_neg])).long().to(device)
            y = torch.from_numpy(np.concatenate([np.ones_like(h_pos), np.zeros_like(h_neg)])).float().to(device)

            rel_id = rel_name_to_full_id[rel]
            rel_ids = torch.full((h.shape[0],), rel_id, dtype=torch.long, device=device)

            logits = model.score_logits_all_rel(z, rel_ids, h, t)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss = total_loss + loss

        total_loss.backward()
        opt.step()
        logs["loss"].append(float(total_loss.detach().cpu().item()))
    return logs


@torch.no_grad()
def eval_link_prediction_binary(
    run: RunGraph,
    model,
    rel_name_to_full_id: Dict[str, int],
    rel: str,
    split: str,
    seed: int,
    n_neg_per_pos: int = 1,
    use_rel_types_in_encoder: bool = True,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()

    edge_index = run.train.edge_index.to(device)
    edge_type = run.train.edge_type.to(device)

    if use_rel_types_in_encoder:
        z = model.encode(edge_index=edge_index, edge_type=edge_type)
    else:
        z = model.encode(edge_index=edge_index)

    # positives in split
    df = run.splits.edges[rel][split]
    idx = run.index
    if rel == "HI":
        h = df["herb"].astype(str).map(idx.id_maps["herb"]).to_numpy() + idx.offsets["herb"]
        t = df["ingredient"].astype(str).map(idx.id_maps["ingredient"]).to_numpy() + idx.offsets["ingredient"]
    elif rel == "IP":
        h = df["ingredient"].astype(str).map(idx.id_maps["ingredient"]).to_numpy() + idx.offsets["ingredient"]
        t = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
    elif rel == "PD":
        h = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
        t = df["disease"].astype(str).map(idx.id_maps["disease"]).to_numpy() + idx.offsets["disease"]
    else:
        # generic
        head_col = df.columns[0]
        tail_col = df.columns[1]
        raise ValueError(f"Unsupported rel for eval: {rel} (need explicit mapping)")

    sampler = TypeAwareNegativeSampler(run, seed=seed + 2024)
    h_neg, t_neg = sampler.sample(rel, h, t, n_neg_per_pos=n_neg_per_pos)

    rel_id = rel_name_to_full_id[rel]
    # score
    h_all = torch.from_numpy(np.concatenate([h, h_neg])).long().to(device)
    t_all = torch.from_numpy(np.concatenate([t, t_neg])).long().to(device)
    y = np.concatenate([np.ones_like(h), np.zeros_like(h_neg)]).astype(int)

    rel_ids = torch.full((h_all.shape[0],), rel_id, dtype=torch.long, device=device)
    logits = model.score_logits_all_rel(z, rel_ids, h_all, t_all)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    return auroc_auprc(y_true=y, y_score=probs)


def train_kge(
    run: RunGraph,
    model,
    rel_name_to_full_id: Dict[str, int],
    cfg: TrainConfig,
    seed: int,
) -> Dict[str, List[float]]:
    device = torch.device(cfg.device)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rel_train = _rel_train_arrays(run)
    sampler = TypeAwareNegativeSampler(run, seed=seed + 999)

    logs = {"loss": []}
    rel_names = list(rel_train.keys())

    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        for step in range(cfg.steps_per_epoch):
            rel = rel_names[step % len(rel_names)]
            h_all, t_all = rel_train[rel]
            if len(h_all) == 0:
                continue
            idxs = np.random.default_rng(seed + epoch * 13 + step).integers(0, len(h_all), size=cfg.batch_pos)
            h_pos = h_all[idxs]
            t_pos = t_all[idxs]
            h_neg, t_neg = sampler.sample(rel, h_pos, t_pos, n_neg_per_pos=cfg.neg_per_pos)

            h = torch.from_numpy(np.concatenate([h_pos, h_neg])).long().to(device)
            t = torch.from_numpy(np.concatenate([t_pos, t_neg])).long().to(device)
            y = torch.from_numpy(np.concatenate([np.ones_like(h_pos), np.zeros_like(h_neg)])).float().to(device)
            rel_id = rel_name_to_full_id[rel]
            rel_ids = torch.full((h.shape[0],), rel_id, dtype=torch.long, device=device)

            logits = model.score_logits(rel_ids=rel_ids, head=h, tail=t)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
        logs["loss"].append(total_loss / max(cfg.steps_per_epoch, 1))
    return logs


@torch.no_grad()
def eval_kge_binary(
    run: RunGraph,
    model,
    rel_name_to_full_id: Dict[str, int],
    rel: str,
    split: str,
    seed: int,
    n_neg_per_pos: int = 1,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()

    df = run.splits.edges[rel][split]
    idx = run.index
    if rel == "IP":
        h = df["ingredient"].astype(str).map(idx.id_maps["ingredient"]).to_numpy() + idx.offsets["ingredient"]
        t = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
    elif rel == "PD":
        h = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
        t = df["disease"].astype(str).map(idx.id_maps["disease"]).to_numpy() + idx.offsets["disease"]
    else:
        raise ValueError(f"Unsupported rel for eval: {rel}")

    sampler = TypeAwareNegativeSampler(run, seed=seed + 2024)
    h_neg, t_neg = sampler.sample(rel, h, t, n_neg_per_pos=n_neg_per_pos)

    h_all = torch.from_numpy(np.concatenate([h, h_neg])).long().to(device)
    t_all = torch.from_numpy(np.concatenate([t, t_neg])).long().to(device)
    y = np.concatenate([np.ones_like(h), np.zeros_like(h_neg)]).astype(int)

    rel_id = rel_name_to_full_id[rel]
    rel_ids = torch.full((h_all.shape[0],), rel_id, dtype=torch.long, device=device)

    logits = model.score_logits(rel_ids=rel_ids, head=h_all, tail=t_all)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return auroc_auprc(y_true=y, y_score=probs)
