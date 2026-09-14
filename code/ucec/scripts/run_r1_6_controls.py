"""Run the reviewer-requested perturbation controls on one fixed test set.

The relevance score and labels are taken from the main UCEC run. Each control
only recomputes U, using the same trained edge gate and cached evidence chains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucec.data import load_splits
from ucec.graph import build_run_graph
from ucec.proxy import sample_proxy_benchmark
from ucec.stage2 import RetrievalConfig, PerturbConfig, UCECEvidenceScorer
from ucec.utils import set_seed


MODES = ["full", "uniform", "shuffled_evidence", "relation_fixed", "mc_only", "evidence_only"]


def _load_pairs(run_dir: Path) -> pd.DataFrame | None:
    parquet = run_dir / "proxy_test_pairs.parquet"
    csv = run_dir / "proxy_test_pairs.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _global_index(run, node_type: str, rid: str) -> int:
    return run.index.offsets[node_type] + run.index.id_maps[node_type][str(rid)]


def _compute_disease_bias(run, z: torch.Tensor) -> dict[str, float]:
    herb_ids = sorted(run.index.id_maps["herb"].keys())
    disease_ids = sorted(run.index.id_maps["disease"].keys())
    hi = torch.tensor([_global_index(run, "herb", h) for h in herb_ids], dtype=torch.long)
    di = torch.tensor([_global_index(run, "disease", d) for d in disease_ids], dtype=torch.long)
    hz = z[hi]
    dz = z[di]
    sums = torch.zeros(len(disease_ids), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(hz), 256):
            sums += (hz[start : start + 256] @ dz.T).sum(dim=0).cpu()
    mean_logits = (sums / max(len(hz), 1)).numpy()
    return {str(disease_ids[i]): float(mean_logits[i]) for i in range(len(disease_ids))}


def _calibrated_scores(run, z: torch.Tensor, pairs: pd.DataFrame, evidence: np.ndarray, run_dir: Path) -> np.ndarray:
    hi = torch.tensor([_global_index(run, "herb", h) for h in pairs["herb"]], dtype=torch.long)
    di = torch.tensor([_global_index(run, "disease", d) for d in pairs["disease"]], dtype=torch.long)
    with torch.no_grad():
        s0 = torch.sigmoid(torch.sum(z[hi] * z[di], dim=-1)).cpu().numpy()
    bias = _compute_disease_bias(run, z.cpu())
    logits = np.log(np.clip(s0, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(s0, 1e-6, 1 - 1e-6))
    s0corr = np.array([logits[i] - bias.get(str(d), 0.0) for i, d in enumerate(pairs["disease"])])
    cal_path = run_dir / "stage2_calibrator.pt"
    if cal_path.exists():
        cal = torch.load(cal_path, map_location="cpu")
        a, b, c = float(cal["a"]), float(cal["b"]), float(cal["c"])
    else:
        a, b, c = 1.0, 1.0, 0.0
    return (1.0 / (1.0 + np.exp(-(a * s0corr + b * evidence + c)))).astype(float)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


def _risk_coverage(risk_score: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
    """Return AURC and risks at 50% and 80% coverage.

    Samples with the lowest risk score are accepted first. For U, this means
    low uncertainty; for -E, this means high evidence.
    """
    order = np.argsort(risk_score, kind="mergesort")
    err = errors[order].astype(float)
    cum_risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    coverage = np.arange(1, len(err) + 1, dtype=float) / len(err)
    aurc = float(np.trapz(np.r_[0.0, cum_risk], np.r_[0.0, coverage]))
    at50 = float(cum_risk[max(0, int(np.ceil(0.50 * len(err))) - 1)])
    at80 = float(cum_risk[max(0, int(np.ceil(0.80 * len(err))) - 1)])
    return aurc, at50, at80


def _metrics(risk: np.ndarray, errors: np.ndarray) -> dict[str, float]:
    return {
        "error_auroc": _auc(risk, errors),
        "error_auprc": _average_precision(risk, errors),
        "aurc": _risk_coverage(risk, errors)[0],
        "risk_at_50pct_coverage": _risk_coverage(risk, errors)[1],
        "risk_at_80pct_coverage": _risk_coverage(risk, errors)[2],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pairs_file", default=None, help="Optional fixed proxy-pair CSV containing label and S.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--mc_samples", type=int, default=16)
    ap.add_argument("--retrieval_budget", type=int, default=100)
    ap.add_argument("--aggregation_budget", type=int, default=30)
    ap.add_argument("--use_ppi_hop", action="store_true")
    ap.add_argument("--max_pairs", type=int, default=0, help="0 means all test pairs")
    ap.add_argument(
        "--top_k_per_herb",
        type=int,
        default=0,
        help="Keep the top K pairs per herb by the fixed relevance score S; 0 disables filtering.",
    )
    ap.add_argument("--n_herbs", type=int, default=100)
    ap.add_argument("--pos_per_herb", type=int, default=2)
    ap.add_argument("--neg_per_herb", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    splits = load_splits(str(run_dir))
    run = build_run_graph(splits)
    z = torch.load(run_dir / "rgcn_embeddings.pt", map_location="cpu")
    pairs = pd.read_csv(args.pairs_file) if args.pairs_file else _load_pairs(run_dir)
    if pairs is None:
        pairs = sample_proxy_benchmark(
            run,
            seed=args.seed + 20,
            n_herbs=args.n_herbs,
            pos_per_herb=args.pos_per_herb,
            neg_per_herb=args.neg_per_herb,
            pd_split="test",
        ).pairs
        pairs.to_csv(out_dir / "fixed_proxy_test_pairs.csv", index=False)
    else:
        pairs = pairs.copy()
    if args.top_k_per_herb > 0:
        if "S" not in pairs.columns:
            raise ValueError("--top_k_per_herb requires the saved fixed relevance score column S.")
        pairs = (
            pairs.sort_values(["herb", "S"], ascending=[True, False])
            .groupby("herb", as_index=False, group_keys=False)
            .head(args.top_k_per_herb)
            .reset_index(drop=True)
        )
    if args.max_pairs > 0:
        pairs = pairs.head(args.max_pairs).copy()
    pairs["herb"] = pairs["herb"].astype(str)
    pairs["disease"] = pairs["disease"].astype(str)
    if "label" not in pairs.columns:
        raise ValueError("The fixed test table must contain a label column.")

    retr_cfg = RetrievalConfig(
        max_ing_per_herb=30,
        max_prot_per_ing=20,
        ppi_topk=100,
        max_path_per_prot=20,
        retrieval_budget=args.retrieval_budget,
        use_ppi_hop=args.use_ppi_hop,
    )
    pert_cfg = PerturbConfig(mc_samples=args.mc_samples, aggregation_budget=args.aggregation_budget)
    scorer = UCECEvidenceScorer(run, z, retr_cfg=retr_cfg, pert_cfg=pert_cfg, device=args.device)
    gate_path = run_dir / "stage2_gate_state.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing trained gate: {gate_path}")
    gate_state = torch.load(gate_path, map_location=args.device)
    scorer.gate.load_state_dict(gate_state)

    herbs = pairs["herb"].tolist()
    diseases = pairs["disease"].tolist()
    values = {}
    for mode in MODES:
        # Reset the RNG per mode so the comparison is reproducible. The cache
        # of retrieved chains remains shared across all modes.
        set_seed(args.seed)
        scorer.pert_cfg.perturb_mode = mode
        u = np.zeros(len(pairs), dtype=float)
        e = np.zeros(len(pairs), dtype=float)
        for i, (herb, disease) in enumerate(tqdm(zip(herbs, diseases), total=len(pairs), desc=f"control {mode}")):
            result = scorer.compute_pair_evidence(herb, disease)
            u[i] = result.U
            e[i] = result.E
        values[f"U_{mode}"] = u
        values[f"E_{mode}"] = e

    if "S" not in pairs.columns:
        pairs["S"] = _calibrated_scores(run, z.cpu(), pairs, values["E_full"], run_dir)

    # The full-run S is held fixed. An error means that its 0.5 decision is
    # inconsistent with the proxy label; U should rank these cases highest.
    y = pairs["label"].to_numpy(dtype=int)
    s = pairs["S"].to_numpy(dtype=float)
    errors = ((s >= 0.5).astype(int) != y).astype(int)
    scores = pairs[["herb", "disease", "label", "S"]].copy()
    scores["error"] = errors
    metrics = {
        "n_pairs": int(len(pairs)),
        "n_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "task_definition": "Fixed full-UCEC S at threshold 0.5; controls recompute only U on identical test pairs and cached chains.",
        "modes": {},
    }
    for mode in MODES:
        u = values[f"U_{mode}"]
        scores[f"U_{mode}"] = u
        scores[f"E_{mode}"] = values[f"E_{mode}"]
        metrics["modes"][mode] = _metrics(u, errors)
    metrics["modes"]["evidence_only_score"] = _metrics(-values["E_full"], errors)

    scores.to_csv(out_dir / "r1_6_control_scores.csv", index=False)
    with open(out_dir / "r1_6_control_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)
    pd.DataFrame(metrics["modes"]).T.reset_index(names="mode").to_csv(
        out_dir / "r1_6_control_metrics.csv", index=False
    )
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"[OK] wrote {out_dir}")


if __name__ == "__main__":
    main()
