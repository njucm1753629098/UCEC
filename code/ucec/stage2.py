from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .graph import RunGraph
from .utils import logit


@dataclass
class RetrievalConfig:
    max_ing_per_herb: int = 30
    max_prot_per_ing: int = 20
    ppi_topk: int = 100
    max_path_per_prot: int = 20
    retrieval_budget: int = 100
    use_ppi_hop: bool = False


@dataclass
class Chain:
    nodes: List[Tuple[str, str]]
    nodes_g: List[int]
    edges: List[Tuple[str, int, int]]
    edge_evidence: List[float]
    pre_score: float


class EvidenceIndex:

    def __init__(self, run: RunGraph, cfg: RetrievalConfig):
        self.run = run
        self.idx = run.index
        self.cfg = cfg
        sp = run.splits.edges

        hi = sp["HI"]["train"][["herb", "ingredient"]].copy()
        ip = sp["IP"]["train"][["ingredient", "protein", "evidence"]].copy()
        pd = sp["PD"]["train"][["protein", "disease", "evidence"]].copy()
        ppi = sp["PPi"]["train"][["protein", "protein2", "evidence"]].copy()
        ppath = sp["PPath"]["train"][["protein", "pathway"]].copy()
        pathd = sp["PathD"]["train"][["pathway", "disease", "evidence"]].copy()

        self.herb_to_ing: Dict[str, List[str]] = {}
        for h, g in hi.groupby("herb"):
            self.herb_to_ing[str(h)] = g["ingredient"].astype(str).tolist()

        self.ing_to_prot: Dict[str, pd.DataFrame] = {}
        for ing, g in ip.groupby("ingredient"):
            gg = g.sort_values("evidence", ascending=False).head(cfg.max_prot_per_ing)
            self.ing_to_prot[str(ing)] = gg

        self.prot_to_ppi: Dict[str, List[Tuple[str, float]]] = {}
        for p, g in ppi.groupby("protein"):
            gg = g.sort_values("evidence", ascending=False).head(cfg.ppi_topk)
            self.prot_to_ppi[str(p)] = list(zip(gg["protein2"].astype(str).tolist(), gg["evidence"].astype(float).tolist()))

        self.prot_to_path: Dict[str, List[str]] = {}
        for p, g in ppath.groupby("protein"):
            self.prot_to_path[str(p)] = g["pathway"].astype(str).tolist()

        # Evidence lookups
        self.pd_ev: Dict[Tuple[str, str], float] = {(r.protein, r.disease): float(r.evidence) for r in pd.itertuples(index=False)}
        self.pathd_ev: Dict[Tuple[str, str], float] = {(r.pathway, r.disease): float(r.evidence) for r in pathd.itertuples(index=False)}

    def _g(self, t: str, rid: str) -> int:
        rid = str(rid)
        return self.idx.offsets[t] + self.idx.id_maps[t][rid]

    def retrieve_chains(self, herb: str, disease: str) -> List[Chain]:
        cfg = self.cfg
        herb = str(herb); disease = str(disease)
        ings = self.herb_to_ing.get(herb, [])
        if not ings:
            return []
        ings = ings[: cfg.max_ing_per_herb]

        chains: List[Chain] = []

        def add_chain(nodes, edges, evs):
            pre = 1.0
            for e in evs:
                pre *= float(e)
            nodes_g = [self._g(t, rid) for t, rid in nodes]
            chains.append(Chain(nodes=nodes, nodes_g=nodes_g, edges=edges, edge_evidence=evs, pre_score=pre))

        ip = self.run.splits.edges["IP"]["train"][["ingredient", "protein", "evidence"]]
        ip = ip[ip["ingredient"].isin(ings)]
        for ing, g in ip.groupby("ingredient"):
            gg = g.sort_values("evidence", ascending=False).head(cfg.max_prot_per_ing)
            for row in gg.itertuples(index=False):
                p = str(row.protein)
                ev_ip = float(row.evidence)
                ev_pd = self.pd_ev.get((p, disease))
                if ev_pd is None:
                    continue
                nodes = [("herb", herb), ("ingredient", str(ing)), ("protein", p), ("disease", disease)]
                edges = [("HI", 0, 1), ("IP", 1, 2), ("PD", 2, 3)]
                evs = [1.0, ev_ip, float(ev_pd)]
                add_chain(nodes, edges, evs)

        for ing, g in ip.groupby("ingredient"):
            gg = g.sort_values("evidence", ascending=False).head(cfg.max_prot_per_ing)
            for row in gg.itertuples(index=False):
                p = str(row.protein)
                ev_ip = float(row.evidence)
                for path in self.prot_to_path.get(p, [])[: cfg.max_path_per_prot]:
                    ev_pathd = self.pathd_ev.get((str(path), disease))
                    if ev_pathd is None:
                        continue
                    nodes = [("herb", herb), ("ingredient", str(ing)), ("protein", p), ("pathway", str(path)), ("disease", disease)]
                    edges = [("HI", 0, 1), ("IP", 1, 2), ("PPath", 2, 3), ("PathD", 3, 4)]
                    evs = [1.0, ev_ip, 1.0, float(ev_pathd)]
                    add_chain(nodes, edges, evs)

        if cfg.use_ppi_hop:
            for ing, g in ip.groupby("ingredient"):
                gg = g.sort_values("evidence", ascending=False).head(cfg.max_prot_per_ing)
                for row in gg.itertuples(index=False):
                    p1 = str(row.protein)
                    ev_ip = float(row.evidence)
                    for p2, ev_ppi in self.prot_to_ppi.get(p1, [])[: cfg.ppi_topk]:
                        ev_pd = self.pd_ev.get((str(p2), disease))
                        if ev_pd is None:
                            continue
                        nodes = [("herb", herb), ("ingredient", str(ing)), ("protein", p1), ("protein", str(p2)), ("disease", disease)]
                        edges = [("HI", 0, 1), ("IP", 1, 2), ("PPi", 2, 3), ("PD", 3, 4)]
                        evs = [1.0, ev_ip, float(ev_ppi), float(ev_pd)]
                        add_chain(nodes, edges, evs)

            for ing, g in ip.groupby("ingredient"):
                gg = g.sort_values("evidence", ascending=False).head(cfg.max_prot_per_ing)
                for row in gg.itertuples(index=False):
                    p1 = str(row.protein)
                    ev_ip = float(row.evidence)
                    for p2, ev_ppi in self.prot_to_ppi.get(p1, [])[: cfg.ppi_topk]:
                        for path in self.prot_to_path.get(str(p2), [])[: cfg.max_path_per_prot]:
                            ev_pathd = self.pathd_ev.get((str(path), disease))
                            if ev_pathd is None:
                                continue
                            nodes = [("herb", herb), ("ingredient", str(ing)), ("protein", p1), ("protein", str(p2)), ("pathway", str(path)), ("disease", disease)]
                            edges = [("HI", 0, 1), ("IP", 1, 2), ("PPi", 2, 3), ("PPath", 3, 4), ("PathD", 4, 5)]
                            evs = [1.0, ev_ip, float(ev_ppi), 1.0, float(ev_pathd)]
                            add_chain(nodes, edges, evs)

        chains.sort(key=lambda c: c.pre_score, reverse=True)
        return chains[: cfg.retrieval_budget]


@dataclass
class PerturbConfig:
    mc_samples: int = 16
    gate_hidden: int = 128
    gate_dropout: float = 0.2
    p_drop_min: float = 0.05
    p_drop_max: float = 0.50
    p_drop_gamma: float = 2.0
    lambda_sigma: float = 0.5
    aggregation_budget: int = 30
    use_evidence_only_gate: bool = False


@dataclass
class Stage2TrainConfig:
    max_epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    print_every: int = 1
    patience: int = 20
    min_delta: float = 1e-4
    monitor_fraction: float = 0.25
    seed: int = 1


class EdgeGate(nn.Module):
    def __init__(self, node_dim: int, rel_names_forward: List[str], cfg: PerturbConfig):
        super().__init__()
        self.cfg = cfg
        self.rel_names = rel_names_forward
        self.rel2id = {r: i for i, r in enumerate(rel_names_forward)}
        self.rel_emb = nn.Embedding(len(rel_names_forward), node_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 3 + 1, cfg.gate_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.gate_dropout),
            nn.Linear(cfg.gate_hidden, 1),
        )

    def forward_logits(self, z_src: torch.Tensor, z_dst: torch.Tensor, rel_ids: torch.LongTensor, evidence: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_evidence_only_gate:
            return torch.log(torch.clamp(evidence, 1e-6, 1.0))
        r = self.rel_emb(rel_ids)
        x = torch.cat([z_src, z_dst, r, evidence.unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1)


def _masked_softmax_grouped(logits: torch.Tensor, group_ptr: torch.LongTensor, keep: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(logits)
    G = group_ptr.numel() - 1
    for g in range(G):
        s = int(group_ptr[g].item())
        e = int(group_ptr[g + 1].item())
        if e <= s:
            continue
        m = keep[s:e]
        if not torch.any(m):
            continue
        l = logits[s:e].clone()
        l[~m] = -1e9
        p = torch.softmax(l, dim=0) * m.to(logits.dtype)
        out[s:e] = p
    return out


def evidence_guided_dropout_prob(evidence: torch.Tensor, cfg: PerturbConfig) -> torch.Tensor:
    e = torch.clamp(evidence, 0.0, 1.0)
    p = cfg.p_drop_min + (cfg.p_drop_max - cfg.p_drop_min) * torch.pow((1.0 - e), cfg.p_drop_gamma)
    return torch.clamp(p, 0.0, 1.0)


@dataclass
class PairEvidenceResult:
    E: float
    U: float
    top_chains: List[Dict]


class UCECEvidenceScorer:
    def __init__(self, run: RunGraph, z: torch.Tensor, retr_cfg: RetrievalConfig, pert_cfg: PerturbConfig, device: str = "cpu"):
        self.run = run
        self.z = z.to(torch.device(device))
        self.retr_cfg = retr_cfg
        self.pert_cfg = pert_cfg
        self.device = torch.device(device)

        self.rel_forward = ["HI", "IP", "PPi", "PPath", "PD", "PathD"]
        self.gate = EdgeGate(node_dim=self.z.shape[1], rel_names_forward=self.rel_forward, cfg=pert_cfg).to(self.device)

        self.ev_index = EvidenceIndex(run, retr_cfg)
        self._pair_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _prepare_pair(self, herb: str, disease: str) -> Optional[Dict[str, Any]]:
        key_hd = (str(herb), str(disease))
        if key_hd in self._pair_cache:
            return self._pair_cache[key_hd]

        chains = self.ev_index.retrieve_chains(str(herb), str(disease))
        if not chains:
            self._pair_cache[key_hd] = None
            return None

        edge_map: Dict[Tuple[int, int, str], int] = {}
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_rel: List[str] = []
        edge_ev: List[float] = []
        chain_edge_ids: List[List[int]] = []

        for c in chains:
            ids = []
            for (rel, si, ti), ev in zip(c.edges, c.edge_evidence):
                sg = c.nodes_g[si]
                tg = c.nodes_g[ti]
                k = (int(sg), int(tg), rel)
                if k not in edge_map:
                    edge_map[k] = len(edge_src)
                    edge_src.append(int(sg))
                    edge_dst.append(int(tg))
                    edge_rel.append(rel)
                    edge_ev.append(float(ev))
                ids.append(edge_map[k])
            chain_edge_ids.append(ids)

        rel_ids = torch.tensor([self.gate.rel2id[r] for r in edge_rel], dtype=torch.long)
        src_t = torch.tensor(edge_src, dtype=torch.long)
        dst_t = torch.tensor(edge_dst, dtype=torch.long)
        ev_t = torch.tensor(edge_ev, dtype=torch.float32)

        order = torch.argsort(src_t * 100 + rel_ids)
        src_t = src_t[order]
        dst_t = dst_t[order]
        rel_ids = rel_ids[order]
        ev_t = ev_t[order]

        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel())
        chain_edge_ids_sorted = [[int(inv[i].item()) for i in ids] for ids in chain_edge_ids]

        key = torch.stack([src_t, rel_ids], dim=1).detach().cpu().numpy()
        group_ptr = [0]
        for i in range(1, len(key)):
            if key[i, 0] != key[i - 1, 0] or key[i, 1] != key[i - 1, 1]:
                group_ptr.append(i)
        group_ptr.append(len(key))

        prepared = {
            "chains": chains,
            "src_t": src_t.cpu(),
            "dst_t": dst_t.cpu(),
            "rel_ids": rel_ids.cpu(),
            "ev_t": ev_t.cpu(),
            "group_ptr": torch.tensor(group_ptr, dtype=torch.long),
            "chain_edge_ids_sorted": chain_edge_ids_sorted,
        }
        self._pair_cache[key_hd] = prepared
        return prepared

    def compute_pair_evidence_expected(self, herb: str, disease: str) -> torch.Tensor:
        prepared = self._prepare_pair(herb, disease)
        if prepared is None:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        src_t = prepared["src_t"].to(self.device)
        dst_t = prepared["dst_t"].to(self.device)
        rel_ids = prepared["rel_ids"].to(self.device)
        ev_t = prepared["ev_t"].to(self.device)
        group_ptr = prepared["group_ptr"].to(self.device)
        chain_edge_ids_sorted = prepared["chain_edge_ids_sorted"]

        z_src = self.z[src_t]
        z_dst = self.z[dst_t]
        logits = self.gate.forward_logits(z_src, z_dst, rel_ids, ev_t)
        keep = torch.ones_like(ev_t, dtype=torch.bool)
        alpha = _masked_softmax_grouped(logits, group_ptr, keep)
        ctilde = torch.clamp(alpha * ev_t, min=0.0)

        chain_scores = []
        for ids in chain_edge_ids_sorted:
            if not ids:
                chain_scores.append(torch.zeros((), dtype=torch.float32, device=self.device))
            else:
                chain_scores.append(ctilde[torch.tensor(ids, device=self.device)].sum())
        scores = torch.stack(chain_scores)
        agg = min(int(self.pert_cfg.aggregation_budget), int(scores.numel()))
        if agg <= 0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        top_scores, _ = torch.topk(scores, k=agg)
        return top_scores.mean()

    def compute_pair_evidence(self, herb: str, disease: str) -> PairEvidenceResult:
        prepared = self._prepare_pair(herb, disease)
        if prepared is None:
            return PairEvidenceResult(E=0.0, U=1.0, top_chains=[])
        chains = prepared["chains"]
        src_t = prepared["src_t"].to(self.device)
        dst_t = prepared["dst_t"].to(self.device)
        rel_ids = prepared["rel_ids"].to(self.device)
        ev_t = prepared["ev_t"].to(self.device)
        group_ptr = prepared["group_ptr"].to(self.device)
        chain_edge_ids_sorted = prepared["chain_edge_ids_sorted"]

        self.gate.train()
        z_src = self.z[src_t]
        z_dst = self.z[dst_t]
        p_drop = evidence_guided_dropout_prob(ev_t, self.pert_cfg)

        T = int(self.pert_cfg.mc_samples)
        contrib = torch.zeros((T, src_t.numel()), dtype=torch.float32, device=self.device)
        for t in range(T):
            keep = torch.rand_like(p_drop) > p_drop
            logits_t = self.gate.forward_logits(z_src, z_dst, rel_ids, ev_t)
            contrib[t] = _masked_softmax_grouped(logits_t, group_ptr, keep)

        mu = contrib.mean(dim=0)
        sigma = contrib.std(dim=0, unbiased=False)

        ctilde = torch.clamp(mu.abs() * ev_t - self.pert_cfg.lambda_sigma * sigma, min=0.0)

        chain_scores = []
        for ids in chain_edge_ids_sorted:
            if not ids:
                chain_scores.append(0.0)
            else:
                chain_scores.append(float(ctilde[torch.tensor(ids, device=self.device)].sum().item()))

        agg = min(int(self.pert_cfg.aggregation_budget), len(chains))
        top_idx = np.argsort(chain_scores)[::-1][:agg]
        top_scores = [chain_scores[i] for i in top_idx]
        E = float(np.mean(top_scores)) if top_scores else 0.0

        top_edge_ids = set()
        for i in top_idx:
            for eid in chain_edge_ids_sorted[i]:
                top_edge_ids.add(eid)
        if top_edge_ids:
            sig = sigma[torch.tensor(sorted(top_edge_ids), device=self.device)].detach().cpu().numpy()
            U = float(np.quantile(sig, 0.75))
        else:
            U = 1.0

        top_chains = []
        for i in top_idx:
            c = chains[i]
            ids = chain_edge_ids_sorted[i]
            edge_info = []
            for (rel, si, ti), ev, eid in zip(c.edges, c.edge_evidence, ids):
                edge_info.append({
                    "rel": rel,
                    "src": f"{c.nodes[si][0]}:{c.nodes[si][1]}",
                    "dst": f"{c.nodes[ti][0]}:{c.nodes[ti][1]}",
                    "evidence": float(ev),
                    "mu": float(mu[eid].item()),
                    "sigma": float(sigma[eid].item()),
                    "ctilde": float(ctilde[eid].item()),
                })
            top_chains.append({
                "nodes": [f"{t}:{rid}" for t, rid in c.nodes],
                "pre_score": float(c.pre_score),
                "score": float(chain_scores[i]),
                "edges": edge_info,
            })

        return PairEvidenceResult(E=E, U=U, top_chains=top_chains)


def corrected_prior_by_disease_bias(prior_probs: np.ndarray, diseases: np.ndarray) -> np.ndarray:

    p = np.clip(prior_probs.astype(float), 1e-6, 1 - 1e-6)
    l = np.log(p) - np.log1p(-p)
    df = pd.DataFrame({"disease": diseases.astype(str), "logit": l})
    bias = df.groupby("disease")["logit"].mean().to_dict()
    l_corr = np.array([lv - float(bias[str(d)]) for lv, d in zip(l, diseases)], dtype=float)
    p_corr = 1 / (1 + np.exp(-l_corr))
    return p_corr.astype(float)


class LogisticCalibrator(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward(self, s0corr: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        x = logit(s0corr)
        return torch.sigmoid(self.a * x + self.b * E + self.c)


def fit_calibrator(
    s0corr: np.ndarray,
    E: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-2,
    steps: int = 2000,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> LogisticCalibrator:
    dev = torch.device(device)
    model = LogisticCalibrator().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    s0_t = torch.tensor(s0corr, dtype=torch.float32, device=dev)
    E_t = torch.tensor(E, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p = model(s0_t, E_t)
        loss = torch.nn.functional.binary_cross_entropy(p, y_t)
        loss.backward()
        opt.step()
    return model


def fit_stage2_posterior(
    scorer: UCECEvidenceScorer,
    s0corr: np.ndarray,
    herbs: List[str],
    diseases: List[str],
    y: np.ndarray,
    cfg: Stage2TrainConfig,
    device: str = "cpu",
) -> Tuple[LogisticCalibrator, Dict[str, List[float]]]:
    dev = torch.device(device)
    scorer.gate = scorer.gate.to(dev)
    scorer.device = dev
    scorer.z = scorer.z.to(dev)

    calib = LogisticCalibrator().to(dev)
    params = list(scorer.gate.parameters()) + list(calib.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: Dict[str, List[float]] = {"train_bce": [], "monitor_bce": []}

    s0_t = torch.tensor(s0corr, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)
    n = len(herbs)
    rng = np.random.default_rng(cfg.seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_monitor = int(round(n * float(cfg.monitor_fraction)))
    n_monitor = min(max(n_monitor, 1), max(n - 1, 1))
    monitor_idx = all_idx[:n_monitor]
    train_idx = all_idx[n_monitor:] if n > 1 else all_idx
    if train_idx.size == 0:
        train_idx = all_idx
    order = train_idx.copy()
    best_monitor = float("inf")
    best_epoch = 0
    best_gate_state = None
    best_calib_state = None
    wait = 0

    for epoch in range(1, cfg.max_epochs + 1):
        rng.shuffle(order)
        scorer.gate.train()
        calib.train()
        losses: List[float] = []
        for start in range(0, len(order), cfg.batch_size):
            batch_idx = order[start : start + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            e_vals = []
            for i in batch_idx:
                e_vals.append(scorer.compute_pair_evidence_expected(herbs[i], diseases[i]))
            E_batch = torch.stack(e_vals)
            p = calib(s0_t[batch_idx], E_batch)
            loss = torch.nn.functional.binary_cross_entropy(p, y_t[batch_idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_bce"].append(mean_loss)

        scorer.gate.eval()
        calib.eval()
        with torch.no_grad():
            monitor_e = torch.stack([scorer.compute_pair_evidence_expected(herbs[i], diseases[i]) for i in monitor_idx])
            monitor_p = calib(s0_t[monitor_idx], monitor_e)
            monitor_loss = torch.nn.functional.binary_cross_entropy(monitor_p, y_t[monitor_idx])
            monitor_bce = float(monitor_loss.detach().cpu().item())
        history["monitor_bce"].append(monitor_bce)

        improved = monitor_bce < best_monitor - float(cfg.min_delta)
        if improved:
            best_monitor = monitor_bce
            best_epoch = epoch
            best_gate_state = copy.deepcopy(scorer.gate.state_dict())
            best_calib_state = copy.deepcopy(calib.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % max(int(cfg.print_every), 1) == 0 or improved or wait >= cfg.patience:
            suffix = " *best" if improved else f" no_improve={wait}/{cfg.patience}"
            print(
                f"[stage2:train] epoch {epoch}/{cfg.max_epochs} "
                f"train_bce={mean_loss:.6f} monitor_bce={monitor_bce:.6f}{suffix}",
                flush=True,
            )
        if wait >= cfg.patience:
            print(f"[stage2:train] early stop at epoch {epoch}; best_epoch={best_epoch} monitor_bce={best_monitor:.6f}", flush=True)
            break

    if best_gate_state is not None:
        scorer.gate.load_state_dict(best_gate_state)
    if best_calib_state is not None:
        calib.load_state_dict(best_calib_state)
    history["best_epoch"] = [float(best_epoch)]
    history["best_monitor_bce"] = [float(best_monitor)]
    return calib.eval(), history
