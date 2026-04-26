#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ucec.data import load_splits
from ucec.graph import build_run_graph
from ucec.models.rgcn import RGCNConfig, RGCNLinkPredictor
from ucec.training import TrainConfig, TypeAwareNegativeSampler, eval_link_prediction_binary, train_gnn_one_epoch
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


def _print_rel_metrics(tag: str, vals: dict) -> str:
    return (
        f"{tag} AUROC={vals.get('auroc', float('nan')):.4f} "
        f"AUPRC={vals.get('auprc', float('nan')):.4f} "
        f"ACC={vals.get('accuracy', float('nan')):.4f} "
        f"P={vals.get('precision', float('nan')):.4f} "
        f"R={vals.get('recall', float('nan')):.4f} "
        f"F1={vals.get('f1', float('nan')):.4f}"
    )


def _composite_val_score(metrics: dict) -> float:
    pd_val = metrics["PD_val"]
    ip_val = metrics["IP_val"]
    return (
        0.55 * float(pd_val.get("auprc", 0.0))
        + 0.20 * float(pd_val.get("auroc", 0.0))
        + 0.15 * float(ip_val.get("auprc", 0.0))
        + 0.10 * float(ip_val.get("auroc", 0.0))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/seed_k folder containing *_train/val/test.csv and meta.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch_pos", type=int, default=512)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--print_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="Override seed (otherwise from meta.json).")
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
        epochs=1,
        steps_per_epoch=args.steps_per_epoch,
        batch_pos=args.batch_pos,
        neg_per_pos=args.neg_per_pos,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        verbose=False,
        print_every=args.print_every,
    )

    print(
        f"[stage1] training rgcn with early stopping on device={args.device} "
        f"(max_epochs={args.max_epochs}, patience={args.patience})",
        flush=True,
    )
    model = RGCNLinkPredictor(
        num_nodes_total=num_nodes_total,
        num_relations=num_rel_full,
        cfg=RGCNConfig(dim=args.dim, dropout=args.dropout),
    ).to(torch.device(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = TypeAwareNegativeSampler(run, seed=seed + 999)

    best_score = float("-inf")
    best_epoch = 0
    best_metrics = None
    best_state = None
    wait = 0
    loss_history = []
    epoch_records = []

    for epoch in range(1, args.max_epochs + 1):
        loss = train_gnn_one_epoch(
            run=run,
            model=model,
            rel_name_to_full_id=rel_name_to_full_id,
            cfg=cfg,
            seed=seed,
            epoch=epoch,
            opt=optimizer,
            sampler=sampler,
            use_rel_types_in_encoder=True,
        )
        loss_history.append(float(loss))

        if epoch == 1 or epoch % max(int(args.print_every), 1) == 0:
            print(f"[stage1:rgcn] epoch {epoch}/{args.max_epochs} loss={loss:.6f}", flush=True)

        if epoch % max(int(args.eval_every), 1) != 0 and epoch != args.max_epochs:
            continue

        metrics_epoch = {
            "PD_val": eval_link_prediction_binary(run, model, rel_name_to_full_id, rel="PD", split="val", seed=seed, use_rel_types_in_encoder=True),
            "PD_test": eval_link_prediction_binary(run, model, rel_name_to_full_id, rel="PD", split="test", seed=seed, use_rel_types_in_encoder=True),
            "IP_val": eval_link_prediction_binary(run, model, rel_name_to_full_id, rel="IP", split="val", seed=seed, use_rel_types_in_encoder=True),
            "IP_test": eval_link_prediction_binary(run, model, rel_name_to_full_id, rel="IP", split="test", seed=seed, use_rel_types_in_encoder=True),
        }
        val_score = _composite_val_score(metrics_epoch)
        epoch_records.append({"epoch": epoch, "loss": float(loss), "val_score": float(val_score), **metrics_epoch})
        print(
            f"[stage1:rgcn] eval epoch {epoch} val_score={val_score:.6f} | "
            f"{_print_rel_metrics('PD_val', metrics_epoch['PD_val'])} | "
            f"{_print_rel_metrics('IP_val', metrics_epoch['IP_val'])}",
            flush=True,
        )

        if val_score > best_score + float(args.min_delta):
            best_score = float(val_score)
            best_epoch = int(epoch)
            best_metrics = copy.deepcopy(metrics_epoch)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save(best_state, str(run_dir / "rgcn_state_best.pt"))
            print(f"[stage1:rgcn] new best epoch={best_epoch} val_score={best_score:.6f}", flush=True)
        else:
            wait += 1
            print(f"[stage1:rgcn] no improvement for {wait}/{args.patience} evals", flush=True)
            if wait >= args.patience:
                print(f"[stage1:rgcn] early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("RGCN training finished without any evaluated checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encode(run.train.edge_index.to(args.device), run.train.edge_type.to(args.device)).detach().cpu()

    torch.save(z, str(run_dir / "rgcn_embeddings.pt"))
    torch.save(best_state, str(run_dir / "rgcn_state.pt"))

    metrics = {
        "seed": seed,
        "device": args.device,
        "stage1_mode": "rgcn_only_early_stop",
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "configs": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "eval_every": args.eval_every,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_pos": args.batch_pos,
            "neg_per_pos": args.neg_per_pos,
            "dim": args.dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "models": {
            "rgcn": {
                "train": {
                    "loss": loss_history,
                    "epochs_ran": len(loss_history),
                    "epoch_records": epoch_records,
                },
                "eval": best_metrics,
            }
        },
    }

    print(
        f"[stage1:rgcn] best epoch={best_epoch} | "
        f"{_print_rel_metrics('PD_test', best_metrics['PD_test'])} | "
        f"{_print_rel_metrics('IP_test', best_metrics['IP_test'])}",
        flush=True,
    )

    out_path = run_dir / "stage1_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
