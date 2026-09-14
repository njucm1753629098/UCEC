"""Table 1 PPI-hop sensitivity over the full disease universe.

The global score is computed for every herb-disease pair. Expensive local
stochastic scoring is restricted to pairs that can yield at least one chain
under exactly the same retrieval limits used by UCEC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucec.data import load_splits
from ucec.graph import build_run_graph
from ucec.proxy import build_herb_targets, build_proxy_labels_from_heldout_pd
from ucec.scripts.run_stage2_ucec import (
    _apply_global_disease_bias,
    _estimate_disease_logit_bias,
    _global_index,
)
from ucec.stage2 import PerturbConfig, RetrievalConfig, UCECEvidenceScorer
from ucec.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-ppi-hops", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--checkpoint-every-herbs", type=int, default=10)
    parser.add_argument("--retrieval-budget", type=int, default=100)
    parser.add_argument("--aggregation-budget", type=int, default=30)
    parser.add_argument("--ppi-walk-beam", type=int, default=100)
    return parser.parse_args()


def proxy_positives(run) -> dict[str, set[str]]:
    targets = build_herb_targets(run)
    labels = build_proxy_labels_from_heldout_pd(run, targets, pd_split="test")
    output: dict[str, set[str]] = {}
    for herb, disease in labels:
        output.setdefault(str(herb), set()).add(str(disease))
    return output


def validate_reachability(scorer: UCECEvidenceScorer, herb: str, diseases: list[str], reachable: set[str]) -> None:
    rng = np.random.default_rng(abs(hash((herb, scorer.retr_cfg.max_ppi_hops))) % (2**32))
    inside = [d for d in diseases if d in reachable]
    outside = [d for d in diseases if d not in reachable]
    for disease in (inside[:3] + (rng.choice(outside, size=min(10, len(outside)), replace=False).tolist() if outside else [])):
        has_chain = bool(scorer.ev_index.retrieve_chains(herb, str(disease)))
        if has_chain != (str(disease) in reachable):
            raise RuntimeError(f"Reachability mismatch for {herb}, {disease}")


def rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    positive_ranks = ranks[np.flatnonzero(labels == 1)]
    return {
        "AUPRC": float(average_precision_score(labels, scores)),
        "MRR": float(np.mean(1.0 / positive_ranks)),
        "Hits@10": float(np.mean(positive_ranks <= 10)),
        "Hits@20": float(np.mean(positive_ranks <= 20)),
        "Hits@50": float(np.mean(positive_ranks <= 50)),
        "proxy_positives": int(len(positive_ranks)),
    }


def run_hop(args: argparse.Namespace, run, z: torch.Tensor, herbs: list[str], diseases: list[str], positives, disease_bias, hop: int) -> None:
    hop_dir = args.out_dir / f"max_ppi_hops_{hop}" / f"shard_{args.shard_index}_of_{args.shard_count}"
    hop_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = hop_dir / "per_herb_metrics.partial.csv"
    completed = set()
    existing = pd.DataFrame()
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
        completed = set(existing["herb"].astype(str))

    scorer = UCECEvidenceScorer(
        run,
        z,
        RetrievalConfig(
            max_ing_per_herb=30,
            max_prot_per_ing=20,
            ppi_topk=100,
            max_path_per_prot=20,
            retrieval_budget=args.retrieval_budget,
            use_ppi_hop=True,
            max_ppi_hops=hop,
            ppi_walk_beam=args.ppi_walk_beam,
        ),
        PerturbConfig(mc_samples=16, aggregation_budget=args.aggregation_budget),
        device=args.device,
    )
    scorer.gate.load_state_dict(torch.load(args.run_dir / "stage2_gate_state.pt", map_location=args.device))
    scorer.gate.eval()
    calibrator = torch.load(args.run_dir / "stage2_calibrator.pt", map_location="cpu")
    a, b, c = (float(calibrator[key]) for key in ("a", "b", "c"))

    disease_indices = torch.tensor(
        [_global_index(run, "disease", disease) for disease in diseases],
        dtype=torch.long,
        device=z.device,
    )
    disease_z = z[disease_indices]
    disease_to_index = {disease: idx for idx, disease in enumerate(diseases)}
    rows = existing.to_dict("records") if not existing.empty else []
    started = time.perf_counter()

    pending = [herb for herb in herbs if herb not in completed]
    for number, herb in enumerate(tqdm(pending, desc=f"full-universe max PPI hops={hop}"), start=1):
        positive_set = positives.get(herb, set())
        if not positive_set:
            continue
        herb_idx = _global_index(run, "herb", herb)
        with torch.no_grad():
            logits = torch.mv(disease_z, z[herb_idx])
            s0 = torch.sigmoid(logits).cpu().numpy()
        s0corr = _apply_global_disease_bias(s0, np.asarray(diseases, dtype=object), disease_bias)
        evidence = np.zeros(len(diseases), dtype=float)

        reachable = scorer.ev_index.reachable_diseases(herb)
        if number <= 3:
            validate_reachability(scorer, herb, diseases, reachable)
        reachable_indices = [disease_to_index[d] for d in reachable if d in disease_to_index]
        for disease_idx in reachable_indices:
            evidence[disease_idx] = scorer.compute_pair_evidence(herb, diseases[disease_idx]).E

        clipped = np.clip(s0corr, 1e-6, 1 - 1e-6)
        corrected_logit = np.log(clipped) - np.log1p(-clipped)
        score = 1.0 / (1.0 + np.exp(-(a * corrected_logit + b * evidence + c)))
        labels = np.fromiter((int(disease in positive_set) for disease in diseases), dtype=np.int8)
        rows.append(
            {
                "herb": herb,
                "max_ppi_hops": hop,
                "reachable_diseases": len(reachable_indices),
                **rank_metrics(labels, score),
            }
        )
        scorer._pair_cache.clear()
        if number % args.checkpoint_every_herbs == 0:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)

    result = pd.DataFrame(rows).drop_duplicates("herb", keep="last")
    result.to_csv(hop_dir / "per_herb_metrics.csv", index=False)
    result.to_csv(checkpoint, index=False)
    metrics = {key: float(result[key].mean()) for key in ("AUPRC", "MRR", "Hits@10", "Hits@20", "Hits@50")}
    summary = {
        "max_ppi_hops": hop,
        "full_disease_universe": len(diseases),
        "evaluated_herbs": int(len(result)),
        "mean_reachable_diseases": float(result["reachable_diseases"].mean()),
        "metrics": metrics,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
    }
    (hop_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(str(args.run_dir))
    seed = int(args.seed) if args.seed is not None else int(splits.meta.get("seed", 1))
    set_seed(seed)
    run = build_run_graph(splits)
    z = torch.load(args.run_dir / "rgcn_embeddings.pt", map_location=args.device).to(args.device)
    positives = proxy_positives(run)
    herbs = sorted(positives)[args.shard_index :: args.shard_count]
    diseases = sorted(run.index.id_maps["disease"])
    disease_bias = _estimate_disease_logit_bias(run, z)
    for hop in args.max_ppi_hops:
        run_hop(args, run, z, herbs, diseases, positives, disease_bias, int(hop))


if __name__ == "__main__":
    main()
