#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

try:
    from torch_geometric.nn.models import Node2Vec
except Exception:  
    Node2Vec = None

from ucec.data import load_splits
from ucec.graph import build_run_graph
from ucec.models.rgcn import RGCNConfig, RGCNLinkPredictor
from ucec.models.gcn import GCNConfig, GCNLinkPredictor
from ucec.models.kge import KGEConfig, KGEModel
from ucec.training import (
    TrainConfig,
    train_gnn_model,
    eval_link_prediction_binary,
    train_kge,
    eval_kge_binary,
    TypeAwareNegativeSampler,
    auroc_auprc,
)
from ucec.utils import set_seed


def save_node_lists(run, out_path: str):
    idx = run.index
    out = {}
    for t in idx.id_maps.keys():
        inv = [None] * idx.num_nodes[t]
        for rid, li in idx.id_maps[t].items():
            inv[li] = rid
        out[t] = inv
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/seed_k folder containing *_train/val/test.csv and meta.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch_pos", type=int, default=2048)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=None, help="Override seed (otherwise from meta.json).")
    ap.add_argument("--no_node2vec", action="store_true", help="Disable node2vec baseline (enabled by default).")
    ap.add_argument("--n2v_walk_length", type=int, default=20)
    ap.add_argument("--n2v_context_size", type=int, default=10)
    ap.add_argument("--n2v_walks_per_node", type=int, default=10)
    ap.add_argument("--n2v_p", type=float, default=1.0)
    ap.add_argument("--n2v_q", type=float, default=1.0)
    ap.add_argument("--n2v_num_negative_samples", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    splits = load_splits(str(run_dir))
    seed = int(args.seed) if args.seed is not None else int(splits.meta.get("seed", 1))
    set_seed(seed)

    run = build_run_graph(splits)
    save_node_lists(run, str(run_dir / "node_lists.json"))

    num_nodes_total = run.index.num_nodes_total
    num_rel_full = len(run.train.rel2id)

    rel_name_to_full_id = {name: run.train.rel2id[name] for name in ["HI", "IP", "PPi", "PPath", "PD", "PathD"]}

    cfg = TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_pos=args.batch_pos,
        neg_per_pos=args.neg_per_pos,
        lr=args.lr,
        device=args.device,
    )

    metrics = {"seed": seed, "device": args.device, "models": {}}

    #R-GCN
    rgcn = RGCNLinkPredictor(num_nodes_total=num_nodes_total, num_relations=num_rel_full,
                            cfg=RGCNConfig(dim=args.dim, dropout=args.dropout))
    train_logs = train_gnn_model(run, rgcn, rel_name_to_full_id, cfg, seed=seed, use_rel_types_in_encoder=True)
    # eval PD/IP
    rgcn = rgcn.to(torch.device(args.device))
    m = {}
    for split in ["val", "test"]:
        m[f"PD_{split}"] = eval_link_prediction_binary(run, rgcn, rel_name_to_full_id, rel="PD", split=split, seed=seed, use_rel_types_in_encoder=True)
        m[f"IP_{split}"] = eval_link_prediction_binary(run, rgcn, rel_name_to_full_id, rel="IP", split=split, seed=seed, use_rel_types_in_encoder=True)
    metrics["models"]["rgcn"] = {"train": train_logs, "eval": m}

    # save embeddings for stage2
    rgcn.eval()
    with torch.no_grad():
        z = rgcn.encode(run.train.edge_index.to(args.device), run.train.edge_type.to(args.device)).detach().cpu()
    torch.save(z, str(run_dir / "rgcn_embeddings.pt"))
    torch.save(rgcn.state_dict(), str(run_dir / "rgcn_state.pt"))

    gcn = GCNLinkPredictor(num_nodes_total=num_nodes_total, num_relations=num_rel_full,
                           cfg=GCNConfig(dim=args.dim, dropout=args.dropout))
    train_logs = train_gnn_model(run, gcn, rel_name_to_full_id, cfg, seed=seed, use_rel_types_in_encoder=False)
    gcn = gcn.to(torch.device(args.device))
    m = {}
    for split in ["val", "test"]:
        m[f"PD_{split}"] = eval_link_prediction_binary(run, gcn, rel_name_to_full_id, rel="PD", split=split, seed=seed, use_rel_types_in_encoder=False)
        m[f"IP_{split}"] = eval_link_prediction_binary(run, gcn, rel_name_to_full_id, rel="IP", split=split, seed=seed, use_rel_types_in_encoder=False)
    metrics["models"]["gcn"] = {"train": train_logs, "eval": m}
    torch.save(gcn.state_dict(), str(run_dir / "gcn_state.pt"))

    # KGE baselines
    for name, score_fn in [("transe", "transe"), ("distmult", "distmult"), ("complex", "complex")]:
        kge = KGEModel(num_nodes_total=num_nodes_total, num_relations=num_rel_full, cfg=KGEConfig(dim=args.dim, score_fn=score_fn))
        train_logs = train_kge(run, kge, rel_name_to_full_id, cfg, seed=seed)
        kge = kge.to(torch.device(args.device))
        m = {}
        for split in ["val", "test"]:
            m[f"PD_{split}"] = eval_kge_binary(run, kge, rel_name_to_full_id, rel="PD", split=split, seed=seed)
            m[f"IP_{split}"] = eval_kge_binary(run, kge, rel_name_to_full_id, rel="IP", split=split, seed=seed)
        metrics["models"][name] = {"train": train_logs, "eval": m}
        torch.save(kge.state_dict(), str(run_dir / f"{name}_state.pt"))

    # node2vec baseline
    if not args.no_node2vec:
        if Node2Vec is None:
            raise ImportError("torch_geometric is required for --node2vec. Please install PyG.")

        device = torch.device(args.device)
        # Use training graph, remove self-loops for node2vec
        edge_index = run.train.edge_index
        self_id = run.train.rel2id.get("self", None)
        if self_id is not None:
            mask = (run.train.edge_type != self_id)
            edge_index = edge_index[:, mask]

        n2v = Node2Vec(
            edge_index=edge_index,
            embedding_dim=args.dim,
            walk_length=args.n2v_walk_length,
            context_size=args.n2v_context_size,
            walks_per_node=args.n2v_walks_per_node,
            p=args.n2v_p,
            q=args.n2v_q,
            num_negative_samples=args.n2v_num_negative_samples,
            sparse=True,
        ).to(device)

        loader = n2v.loader(batch_size=256, shuffle=True, num_workers=0)
        opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)

        n2v.train()
        n2v_logs = {"loss": []}
        for epoch in range(1, args.epochs + 1):
            total = 0.0
            for pos_rw, neg_rw in loader:
                opt.zero_grad(set_to_none=True)
                loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item())
            n2v_logs["loss"].append(total / max(len(loader), 1))

        # embeddings
        n2v.eval()
        with torch.no_grad():
            z_n2v = n2v().detach().cpu()  # [N, dim]
        torch.save(z_n2v, str(run_dir / "node2vec_embeddings.pt"))
        torch.save(n2v.state_dict(), str(run_dir / "node2vec_state.pt"))

        # Evaluate on PD/IP with dot-product scoring
        sampler = TypeAwareNegativeSampler(run, seed=seed + 2024)
        idx = run.index

        def _get_pos(rel: str, split: str):
            df = run.splits.edges[rel][split]
            if rel == "PD":
                h = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
                t = df["disease"].astype(str).map(idx.id_maps["disease"]).to_numpy() + idx.offsets["disease"]
            elif rel == "IP":
                h = df["ingredient"].astype(str).map(idx.id_maps["ingredient"]).to_numpy() + idx.offsets["ingredient"]
                t = df["protein"].astype(str).map(idx.id_maps["protein"]).to_numpy() + idx.offsets["protein"]
            else:
                raise ValueError(rel)
            return h.astype(np.int64), t.astype(np.int64)

        def _eval(rel: str, split: str):
            h, t = _get_pos(rel, split)
            h_neg, t_neg = sampler.sample(rel, h, t, n_neg_per_pos=args.neg_per_pos)
            h_all = np.concatenate([h, h_neg])
            t_all = np.concatenate([t, t_neg])
            y = np.concatenate([np.ones_like(h), np.zeros_like(h_neg)]).astype(int)
            # sigmoid(dot)
            zz = z_n2v.numpy()
            s = (zz[h_all] * zz[t_all]).sum(axis=1)
            p = 1.0 / (1.0 + np.exp(-s))
            return auroc_auprc(y_true=y, y_score=p)

        m = {
            "PD_val": _eval("PD", "val"),
            "PD_test": _eval("PD", "test"),
            "IP_val": _eval("IP", "val"),
            "IP_test": _eval("IP", "test"),
        }
        metrics["models"]["node2vec"] = {"train": n2v_logs, "eval": m}

    out_path = run_dir / "stage1_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
