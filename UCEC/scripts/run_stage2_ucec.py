#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ucec.data import load_splits
from ucec.graph import build_run_graph
from ucec.proxy import sample_proxy_benchmark
from ucec.stage2 import (
    RetrievalConfig,
    PerturbConfig,
    UCECEvidenceScorer,
    corrected_prior_by_disease_bias,
    fit_calibrator,
)
from ucec.utils import auroc_auprc, expected_calibration_error, brier_score, set_seed


def _global_index(run, node_type: str, rid: str) -> int:
    rid = str(rid)
    return run.index.offsets[node_type] + run.index.id_maps[node_type][rid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    # proxy sampling
    ap.add_argument("--n_herbs", type=int, default=500)
    ap.add_argument("--pos_per_herb", type=int, default=4)
    ap.add_argument("--neg_per_herb", type=int, default=16)
    # retrieval/scoring
    ap.add_argument("--use_ppi_hop", action="store_true")
    ap.add_argument("--mc_samples", type=int, default=16)
    ap.add_argument("--retrieval_budget", type=int, default=100)
    ap.add_argument("--aggregation_budget", type=int, default=30)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    splits = load_splits(str(run_dir))
    seed = int(args.seed) if args.seed is not None else int(splits.meta.get("seed", 1))
    set_seed(seed)

    run = build_run_graph(splits)
    z = torch.load(run_dir / "rgcn_embeddings.pt", map_location="cpu")
    z = z.to(torch.device(args.device))

    retr_cfg = RetrievalConfig(
        max_ing_per_herb=30,
        max_prot_per_ing=20,
        ppi_topk=100,
        max_path_per_prot=20,
        retrieval_budget=args.retrieval_budget,
        use_ppi_hop=args.use_ppi_hop,
    )
    pert_cfg = PerturbConfig(
        mc_samples=args.mc_samples,
        aggregation_budget=args.aggregation_budget,
    )

    scorer = UCECEvidenceScorer(run, z, retr_cfg=retr_cfg, pert_cfg=pert_cfg, device=args.device)

    # validation benchmark (held-out PD val edges)
    bench_val = sample_proxy_benchmark(
        run, seed=seed + 10, n_herbs=args.n_herbs, pos_per_herb=args.pos_per_herb, neg_per_herb=args.neg_per_herb, pd_split="val"
    ).pairs
    # test benchmark (held-out PD test edges)
    bench_test = sample_proxy_benchmark(
        run, seed=seed + 20, n_herbs=args.n_herbs, pos_per_herb=args.pos_per_herb, neg_per_herb=args.neg_per_herb, pd_split="test"
    ).pairs

    def compute_features(df: pd.DataFrame, desc: str):
        herbs = df["herb"].astype(str).tolist()
        diseases = df["disease"].astype(str).tolist()
        y = df["label"].astype(int).to_numpy()

        # prior S0 from dot product of embeddings
        h_idx = torch.tensor([_global_index(run, "herb", h) for h in herbs], dtype=torch.long, device=z.device)
        d_idx = torch.tensor([_global_index(run, "disease", d) for d in diseases], dtype=torch.long, device=z.device)
        with torch.no_grad():
            logits = torch.sum(z[h_idx] * z[d_idx], dim=-1)
            s0 = torch.sigmoid(logits).detach().cpu().numpy()

        s0corr = corrected_prior_by_disease_bias(s0, np.array(diseases, dtype=object))

        # evidence E and uncertainty U
        E = np.zeros(len(df), dtype=float)
        U = np.zeros(len(df), dtype=float)
        topchains = []
        for i, (h, d) in enumerate(tqdm(list(zip(herbs, diseases)), desc=f"Stage2 evidence {desc}")):
            res = scorer.compute_pair_evidence(h, d)
            E[i] = res.E
            U[i] = res.U
            # keep small explanations
            if i < 200:
                topchains.append({"herb": h, "disease": d, "top_chains": res.top_chains})
        return s0, s0corr, E, U, y, topchains

    s0_v, s0c_v, E_v, U_v, y_v, expl_v = compute_features(bench_val, "val")
    s0_t, s0c_t, E_t, U_t, y_t, expl_t = compute_features(bench_test, "test")

    # Fit calibration on validation
    calib = fit_calibrator(s0c_v, E_v, y_v, device="cpu")
    with torch.no_grad():
        a = float(calib.a.item()); b = float(calib.b.item()); c = float(calib.c.item())

    def apply_calib(s0c, E):
        s0c_ten = torch.tensor(s0c, dtype=torch.float32)
        E_ten = torch.tensor(E, dtype=torch.float32)
        with torch.no_grad():
            p = calib(s0c_ten, E_ten).numpy()
        return p

    S_v = apply_calib(s0c_v, E_v)
    S_t = apply_calib(s0c_t, E_t)

    # Metrics
    metrics = {
        "seed": seed,
        "proxy_val": {
            **auroc_auprc(y_v, S_v),
            "ece10": expected_calibration_error(S_v, y_v, n_bins=10),
            "brier": brier_score(S_v, y_v),
        },
        "proxy_test": {
            **auroc_auprc(y_t, S_t),
            "ece10": expected_calibration_error(S_t, y_t, n_bins=10),
            "brier": brier_score(S_t, y_t),
        },
        "calibrator": {"a": a, "b": b, "c": c},
        "configs": {"retrieval": retr_cfg.__dict__, "perturb": pert_cfg.__dict__},
    }

    # Save pair tables
    out_val = bench_val.copy()
    out_val["S0"] = s0_v; out_val["S0corr"] = s0c_v; out_val["E"] = E_v; out_val["U"] = U_v; out_val["S"] = S_v
    out_test = bench_test.copy()
    out_test["S0"] = s0_t; out_test["S0corr"] = s0c_t; out_test["E"] = E_t; out_test["U"] = U_t; out_test["S"] = S_t

    try:
        out_val.to_parquet(run_dir / "proxy_val_pairs.parquet", index=False)
        out_test.to_parquet(run_dir / "proxy_test_pairs.parquet", index=False)
    except Exception as e:
        # Fallback if parquet engine is missing
        out_val.to_csv(run_dir / "proxy_val_pairs.csv", index=False)
        out_test.to_csv(run_dir / "proxy_test_pairs.csv", index=False)
        metrics["warnings"] = metrics.get("warnings", []) + [f"parquet_write_failed: {type(e).__name__}: {e}"]

    # Save explanations
    with open(run_dir / "explanations_val.jsonl", "w", encoding="utf-8") as f:
        for r in expl_v:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(run_dir / "explanations_test.jsonl", "w", encoding="utf-8") as f:
        for r in expl_t:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    K = 5
    drops = []
    for rec in expl_t:
        if not rec.get("top_chains"):
            continue
        best = rec["top_chains"][0]
        edges = best.get("edges", [])
        if not edges:
            continue
        edges_sorted = sorted(edges, key=lambda e: e.get("ctilde", 0.0), reverse=True)

        E0 = float(best.get("score", 0.0))
        mask = (out_test["herb"].astype(str) == rec["herb"]) & (out_test["disease"].astype(str) == rec["disease"])
        if not mask.any():
            continue
        s0c = float(out_test.loc[mask, "S0corr"].iloc[0])
        # baseline posterior
        S0 = float(1/(1+np.exp(-(a*np.log(s0c/(1-s0c)) + b*E0 + c))))
        for k in range(1, K+1):
            removed = edges_sorted[:k]
            Ek = max(E0 - sum(float(e.get("ctilde", 0.0)) for e in removed), 0.0)
            Sk = float(1/(1+np.exp(-(a*np.log(s0c/(1-s0c)) + b*Ek + c))))
            drops.append({"k": k, "drop": S0 - Sk})
    del_df = pd.DataFrame(drops).groupby("k", as_index=False)["drop"].mean()
    del_df.to_csv(run_dir / "deletion_test.csv", index=False)

    with open(run_dir / "proxy_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] wrote {run_dir/'proxy_metrics.json'} and pair tables.")


if __name__ == "__main__":
    main()
