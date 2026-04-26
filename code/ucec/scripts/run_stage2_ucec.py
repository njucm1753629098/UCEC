#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
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
    Stage2TrainConfig,
    UCECEvidenceScorer,
    fit_calibrator,
    fit_stage2_posterior,
)
from ucec.utils import auroc_auprc, expected_calibration_error, brier_score, set_seed


def _global_index(run, node_type: str, rid: str) -> int:
    rid = str(rid)
    return run.index.offsets[node_type] + run.index.id_maps[node_type][rid]


def _identity_entity_maps(run) -> tuple[dict[str, str], dict[str, str]]:
    herb_map = {str(h): str(h) for h in run.index.id_maps["herb"].keys()}
    disease_map = {str(d): str(d) for d in run.index.id_maps["disease"].keys()}
    return herb_map, disease_map


def _build_entity_maps(data_dir: str, run) -> tuple[dict[str, str], dict[str, str]]:
    herb_map, disease_map = _identity_entity_maps(run)

    herb_file = os.path.join(data_dir, "hit2_herbs_ingredients.csv")
    if os.path.exists(herb_file):
        herbs = pd.read_csv(herb_file, low_memory=False)
        if {"Herb ID", "English Name", "Chinese Character"}.issubset(herbs.columns):
            herb_name = (
                herbs[["Herb ID", "English Name", "Chinese Character"]]
                .drop_duplicates("Herb ID")
                .fillna("")
                .assign(
                    display_name=lambda df: np.where(
                        df["English Name"].astype(str).str.strip() != "",
                        df["English Name"],
                        df["Chinese Character"],
                    )
                )
            )
            herb_map.update({str(r["Herb ID"]): str(r["display_name"]) for _, r in herb_name.iterrows()})

    disease_file = os.path.join(data_dir, "disgenet_target_disease.csv")
    if os.path.exists(disease_file):
        diseases = pd.read_csv(disease_file, low_memory=False)
        if {"disease_id", "disease_name"}.issubset(diseases.columns):
            dis_name = diseases[["disease_id", "disease_name"]].drop_duplicates("disease_id").fillna("")
            disease_map.update({str(r["disease_id"]): str(r["disease_name"]) for _, r in dis_name.iterrows()})

    return herb_map, disease_map


def _estimate_disease_logit_bias(run, z: torch.Tensor, herb_batch_size: int = 256) -> dict[str, float]:
    herb_ids = sorted(run.index.id_maps["herb"].keys())
    disease_ids = sorted(run.index.id_maps["disease"].keys())
    herb_gidx = torch.tensor([_global_index(run, "herb", h) for h in herb_ids], dtype=torch.long)
    disease_gidx = torch.tensor([_global_index(run, "disease", d) for d in disease_ids], dtype=torch.long)

    z_cpu = z.detach().cpu()
    herb_z = z_cpu[herb_gidx]
    disease_z = z_cpu[disease_gidx]
    sums = torch.zeros(len(disease_ids), dtype=torch.float32)
    total = 0
    with torch.no_grad():
        for start in range(0, herb_z.shape[0], herb_batch_size):
            batch = herb_z[start : start + herb_batch_size]
            logits = batch @ disease_z.T
            sums += logits.sum(dim=0)
            total += batch.shape[0]
    mean_logits = (sums / max(total, 1)).numpy()
    return {str(disease_ids[i]): float(mean_logits[i]) for i in range(len(disease_ids))}


def _apply_global_disease_bias(prior_probs: np.ndarray, diseases: np.ndarray, disease_bias: dict[str, float]) -> np.ndarray:
    p = np.clip(np.asarray(prior_probs, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p) - np.log1p(-p)
    corr = np.array(
        [logits[i] - float(disease_bias.get(str(diseases[i]), 0.0)) for i in range(len(diseases))],
        dtype=float,
    )
    return 1.0 / (1.0 + np.exp(-corr))


def _annotate_names(df: pd.DataFrame, herb_map: dict[str, str], disease_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["herb"] = out["herb"].astype(str)
    out["disease"] = out["disease"].astype(str)
    out["herb_name"] = out["herb"].map(herb_map).fillna(out["herb"])
    out["disease_name"] = out["disease"].map(disease_map).fillna(out["disease"])
    return out


def _flatten_explanations(records: list[dict], herb_map: dict[str, str], disease_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_rows = []
    edge_rows = []
    for rec in records:
        herb = str(rec.get("herb", ""))
        disease = str(rec.get("disease", ""))
        for chain_rank, chain in enumerate(rec.get("top_chains", []), start=1):
            case_rows.append(
                {
                    "herb": herb,
                    "herb_name": herb_map.get(herb, herb),
                    "disease": disease,
                    "disease_name": disease_map.get(disease, disease),
                    "chain_rank": chain_rank,
                    "chain_score": float(chain.get("score", 0.0)),
                    "chain_pre_score": float(chain.get("pre_score", 0.0)),
                    "nodes": " -> ".join(chain.get("nodes", [])),
                }
            )
            for edge_rank, edge in enumerate(chain.get("edges", []), start=1):
                edge_rows.append(
                    {
                        "herb": herb,
                        "herb_name": herb_map.get(herb, herb),
                        "disease": disease,
                        "disease_name": disease_map.get(disease, disease),
                        "chain_rank": chain_rank,
                        "edge_rank": edge_rank,
                        "rel": edge.get("rel", ""),
                        "src": edge.get("src", ""),
                        "dst": edge.get("dst", ""),
                        "evidence": float(edge.get("evidence", 0.0)),
                        "mu": float(edge.get("mu", 0.0)),
                        "sigma": float(edge.get("sigma", 0.0)),
                        "ctilde": float(edge.get("ctilde", 0.0)),
                    }
                )
    return pd.DataFrame(case_rows), pd.DataFrame(edge_rows)


def _annotate_explanations(records: list[dict], herb_map: dict[str, str], disease_map: dict[str, str]) -> list[dict]:
    out = []
    for rec in records:
        herb = str(rec.get("herb", ""))
        disease = str(rec.get("disease", ""))
        item = dict(rec)
        item["herb_name"] = herb_map.get(herb, herb)
        item["disease_name"] = disease_map.get(disease, disease)
        out.append(item)
    return out


def _diagnose_outputs(df: pd.DataFrame, explanations: list[dict]) -> dict:
    diag = {
        "rows": int(len(df)),
        "score_nan_count": int(df["S"].isna().sum()) if "S" in df.columns else 0,
        "s0corr_mean": float(df["S0corr"].mean()) if "S0corr" in df.columns and len(df) else 0.0,
        "s0corr_std": float(df["S0corr"].std(ddof=0)) if "S0corr" in df.columns and len(df) else 0.0,
        "s0corr_half_ratio": float(np.mean(np.isclose(df["S0corr"].to_numpy(), 0.5))) if "S0corr" in df.columns and len(df) else 0.0,
        "evidence_zero_ratio": float(np.mean(np.isclose(df["E"].to_numpy(), 0.0))) if "E" in df.columns and len(df) else 0.0,
        "uncertainty_zero_ratio": float(np.mean(np.isclose(df["U"].to_numpy(), 0.0))) if "U" in df.columns and len(df) else 0.0,
        "unmapped_disease_name_ratio": float(np.mean(df["disease"].astype(str).eq(df["disease_name"].astype(str)))) if {"disease", "disease_name"}.issubset(df.columns) and len(df) else 0.0,
        "unmapped_herb_name_ratio": float(np.mean(df["herb"].astype(str).eq(df["herb_name"].astype(str)))) if {"herb", "herb_name"}.issubset(df.columns) and len(df) else 0.0,
        "empty_explanation_ratio": float(np.mean([len(rec.get("top_chains", [])) == 0 for rec in explanations])) if explanations else 0.0,
    }
    warnings = []
    if diag["score_nan_count"] > 0:
        warnings.append("score_nan_detected")
    if diag["s0corr_half_ratio"] > 0.95:
        warnings.append("s0corr_nearly_constant_0.5")
    if diag["unmapped_disease_name_ratio"] > 0.95:
        warnings.append("disease_name_mapping_missing")
    if diag["unmapped_herb_name_ratio"] > 0.95:
        warnings.append("herb_name_mapping_missing")
    if diag["evidence_zero_ratio"] > 0.95:
        warnings.append("evidence_mostly_zero")
    diag["warnings"] = warnings
    return diag


def _write_paper_tables(
    run_dir: Path,
    metrics: dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expl_val: list[dict],
    expl_test: list[dict],
    herb_map: dict[str, str],
    disease_map: dict[str, str],
) -> None:
    out_dir = run_dir / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"split": "proxy_val", **metrics["proxy_val"]},
            {"split": "proxy_test", **metrics["proxy_test"]},
        ]
    ).to_csv(out_dir / "table_metrics_summary.csv", index=False)

    ranked_test = test_df.sort_values(["S", "E", "U"], ascending=[False, False, True]).copy()
    ranked_test["rank"] = np.arange(1, len(ranked_test) + 1)
    ranked_test[
        ["rank", "herb", "herb_name", "disease", "disease_name", "label", "S", "E", "U", "S0", "S0corr"]
    ].head(100).to_csv(out_dir / "table_proxy_test_top100.csv", index=False)

    high_conf = ranked_test[(ranked_test["S"] >= 0.7) & (ranked_test["U"] <= ranked_test["U"].median())].copy()
    high_conf.head(100).to_csv(out_dir / "table_high_confidence_pairs.csv", index=False)

    val_cases, val_edges = _flatten_explanations(expl_val, herb_map, disease_map)
    test_cases, test_edges = _flatten_explanations(expl_test, herb_map, disease_map)
    val_cases.to_csv(out_dir / "table_case_studies_val.csv", index=False)
    test_cases.to_csv(out_dir / "table_case_studies_test.csv", index=False)
    val_edges.to_csv(out_dir / "table_case_edges_val.csv", index=False)
    test_edges.to_csv(out_dir / "table_case_edges_test.csv", index=False)


def _export_full_predictions(
    run,
    run_dir: Path,
    z: torch.Tensor,
    scorer: UCECEvidenceScorer,
    calib,
    herb_map: dict[str, str],
    disease_map: dict[str, str],
    shortlist_per_herb: int,
    topk_per_herb: int,
    herb_limit: int,
    explanation_pairs: int,
    disease_bias: dict[str, float],
) -> None:
    out_dir = run_dir / "full_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    herb_ids = sorted(run.index.id_maps["herb"].keys())
    disease_ids = sorted(run.index.id_maps["disease"].keys())
    if herb_limit > 0:
        herb_ids = herb_ids[:herb_limit]

    z_cpu = z.detach().cpu()
    disease_gidx = torch.tensor([_global_index(run, "disease", d) for d in disease_ids], dtype=torch.long)
    disease_z = z_cpu[disease_gidx]

    frames: list[pd.DataFrame] = []
    explanation_records: list[dict] = []
    for herb in tqdm(herb_ids, desc="full candidate ranking"):
        herb_idx = _global_index(run, "herb", herb)
        with torch.no_grad():
            logits = torch.mv(disease_z, z_cpu[herb_idx])
            s0 = torch.sigmoid(logits).detach().cpu().numpy()
        order = np.argsort(s0)[::-1][: max(int(shortlist_per_herb), 1)]
        candidates = pd.DataFrame(
            {
                "herb": [str(herb)] * len(order),
                "disease": [str(disease_ids[i]) for i in order],
                "S0": s0[order],
            }
        )
        if candidates.empty:
            continue

        s0corr = _apply_global_disease_bias(
            candidates["S0"].to_numpy(), candidates["disease"].astype(str).to_numpy(), disease_bias
        )
        E = np.zeros(len(candidates), dtype=float)
        U = np.zeros(len(candidates), dtype=float)
        for i, disease in enumerate(candidates["disease"].astype(str).tolist()):
            res = scorer.compute_pair_evidence(str(herb), disease)
            E[i] = res.E
            U[i] = res.U
            if len(explanation_records) < explanation_pairs:
                explanation_records.append({"herb": str(herb), "disease": disease, "top_chains": res.top_chains})

        with torch.no_grad():
            S = calib(
                torch.tensor(s0corr, dtype=torch.float32),
                torch.tensor(E, dtype=torch.float32),
            ).detach().cpu().numpy()
        out = candidates.copy()
        out["S0corr"] = s0corr
        out["E"] = E
        out["U"] = U
        out["S"] = S
        out = _annotate_names(out, herb_map, disease_map)
        out = out.sort_values(["S", "E", "U"], ascending=[False, False, True]).reset_index(drop=True)
        out["rank_within_herb"] = np.arange(1, len(out) + 1)
        frames.append(out.head(int(topk_per_herb)).copy())

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "predictions_topk_per_herb.csv", index=False)
    combined.sort_values(["S", "E", "U"], ascending=[False, False, True]).head(1000).to_csv(
        out_dir / "predictions_global_top1000.csv", index=False
    )

    full_cases, full_edges = _flatten_explanations(explanation_records, herb_map, disease_map)
    full_cases.to_csv(out_dir / "prediction_case_studies.csv", index=False)
    full_edges.to_csv(out_dir / "prediction_case_edges.csv", index=False)
    with open(out_dir / "prediction_explanations.jsonl", "w", encoding="utf-8") as f:
        for rec in explanation_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "ucec" / "data"))
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
    ap.add_argument("--stage2_max_epochs", type=int, default=200)
    ap.add_argument("--stage2_patience", type=int, default=20)
    ap.add_argument("--stage2_min_delta", type=float, default=1e-4)
    ap.add_argument("--stage2_monitor_fraction", type=float, default=0.25)
    ap.add_argument("--stage2_train_epochs", type=int, default=None, help="Deprecated alias for --stage2_max_epochs.")
    ap.add_argument("--stage2_train_batch_size", type=int, default=64)
    ap.add_argument("--stage2_train_lr", type=float, default=1e-3)
    ap.add_argument("--stage2_print_every", type=int, default=1)
    ap.add_argument("--evidence_only_gate", action="store_true")
    ap.add_argument("--export_full_predictions", action="store_true")
    ap.add_argument("--full_shortlist_per_herb", type=int, default=200)
    ap.add_argument("--full_topk_per_herb", type=int, default=50)
    ap.add_argument("--full_explanation_pairs", type=int, default=300)
    ap.add_argument("--full_herb_limit", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    splits = load_splits(str(run_dir))
    seed = int(args.seed) if args.seed is not None else int(splits.meta.get("seed", 1))
    set_seed(seed)

    run = build_run_graph(splits)
    z = torch.load(run_dir / "rgcn_embeddings.pt", map_location="cpu")
    z = z.to(torch.device(args.device))
    herb_map, disease_map = _build_entity_maps(args.data_dir, run)
    disease_bias = _estimate_disease_logit_bias(run, z)

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
        use_evidence_only_gate=bool(args.evidence_only_gate),
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

        s0corr = _apply_global_disease_bias(s0, np.array(diseases, dtype=object), disease_bias)

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

    # Train stage-2 posterior on validation proxy pairs.
    stage2_max_epochs = int(args.stage2_train_epochs) if args.stage2_train_epochs is not None else int(args.stage2_max_epochs)
    stage2_train_cfg = Stage2TrainConfig(
        max_epochs=stage2_max_epochs,
        batch_size=args.stage2_train_batch_size,
        lr=args.stage2_train_lr,
        print_every=args.stage2_print_every,
        patience=args.stage2_patience,
        min_delta=args.stage2_min_delta,
        monitor_fraction=args.stage2_monitor_fraction,
        seed=seed + 30,
    )
    print(
        f"[stage2] training posterior gate on validation proxy pairs "
        f"(max_epochs={stage2_train_cfg.max_epochs}, patience={stage2_train_cfg.patience}, "
        f"batch_size={stage2_train_cfg.batch_size}, "
        f"evidence_only_gate={pert_cfg.use_evidence_only_gate})",
        flush=True,
    )
    calib, train_hist = fit_stage2_posterior(
        scorer=scorer,
        s0corr=s0c_v,
        herbs=bench_val["herb"].astype(str).tolist(),
        diseases=bench_val["disease"].astype(str).tolist(),
        y=y_v,
        cfg=stage2_train_cfg,
        device=args.device,
    )
    # Recompute validation evidence after gate training for final calibration fit.
    s0_v, s0c_v, E_v, U_v, y_v, expl_v = compute_features(bench_val, "val-refit")
    s0_t, s0c_t, E_t, U_t, y_t, expl_t = compute_features(bench_test, "test")
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
        "configs": {
            "retrieval": retr_cfg.__dict__,
            "perturb": pert_cfg.__dict__,
            "stage2_train": stage2_train_cfg.__dict__,
        },
    }

    # Save pair tables
    out_val = bench_val.copy()
    out_val["S0"] = s0_v; out_val["S0corr"] = s0c_v; out_val["E"] = E_v; out_val["U"] = U_v; out_val["S"] = S_v
    out_test = bench_test.copy()
    out_test["S0"] = s0_t; out_test["S0corr"] = s0c_t; out_test["E"] = E_t; out_test["U"] = U_t; out_test["S"] = S_t
    out_val = _annotate_names(out_val, herb_map, disease_map)
    out_test = _annotate_names(out_test, herb_map, disease_map)
    out_val = out_val.sort_values(["S", "E", "U"], ascending=[False, False, True]).reset_index(drop=True)
    out_test = out_test.sort_values(["S", "E", "U"], ascending=[False, False, True]).reset_index(drop=True)
    out_val["rank"] = np.arange(1, len(out_val) + 1)
    out_test["rank"] = np.arange(1, len(out_test) + 1)

    try:
        out_val.to_parquet(run_dir / "proxy_val_pairs.parquet", index=False)
        out_test.to_parquet(run_dir / "proxy_test_pairs.parquet", index=False)
    except Exception as e:
        # Fallback if parquet engine is missing
        out_val.to_csv(run_dir / "proxy_val_pairs.csv", index=False)
        out_test.to_csv(run_dir / "proxy_test_pairs.csv", index=False)
        metrics["warnings"] = metrics.get("warnings", []) + [f"parquet_write_failed: {type(e).__name__}: {e}"]

    # Save explanations
    expl_v_named = _annotate_explanations(expl_v, herb_map, disease_map)
    expl_t_named = _annotate_explanations(expl_t, herb_map, disease_map)
    with open(run_dir / "explanations_val.jsonl", "w", encoding="utf-8") as f:
        for r in expl_v_named:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(run_dir / "explanations_test.jsonl", "w", encoding="utf-8") as f:
        for r in expl_t_named:
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

    torch.save(scorer.gate.state_dict(), run_dir / "stage2_gate_state.pt")
    torch.save({"a": a, "b": b, "c": c}, run_dir / "stage2_calibrator.pt")
    with open(run_dir / "stage2_training.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": train_hist,
                "stage2_train": stage2_train_cfg.__dict__,
                "retrieval": retr_cfg.__dict__,
                "perturb": pert_cfg.__dict__,
            },
            f,
            indent=2,
        )

    diagnostics = {
        "proxy_val": _diagnose_outputs(out_val, expl_v_named),
        "proxy_test": _diagnose_outputs(out_test, expl_t_named),
    }
    with open(run_dir / "result_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    with open(run_dir / "proxy_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_paper_tables(run_dir, metrics, out_val, out_test, expl_v_named, expl_t_named, herb_map, disease_map)
    if args.export_full_predictions:
        _export_full_predictions(
            run=run,
            run_dir=run_dir,
            z=z,
            scorer=scorer,
            calib=calib,
            herb_map=herb_map,
            disease_map=disease_map,
            shortlist_per_herb=args.full_shortlist_per_herb,
            topk_per_herb=args.full_topk_per_herb,
            herb_limit=args.full_herb_limit,
            explanation_pairs=args.full_explanation_pairs,
            disease_bias=disease_bias,
        )
    print(f"[OK] wrote {run_dir/'proxy_metrics.json'} and pair tables.")


if __name__ == "__main__":
    main()
