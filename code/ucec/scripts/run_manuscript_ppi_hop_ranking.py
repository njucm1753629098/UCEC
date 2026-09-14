#!/usr/bin/env python
"""Evaluate 1/2/3-PPI-hop variants with the manuscript Table 1 protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SEEDS = (1, 2, 3, 4, 5)
POS_PER_HERB = 5
NEG_PER_HERB = 300
MAX_HERBS_EVAL = 1250
K_LIST = (10, 20, 50)
ALPHA_PPI = 0.6
PPI_TOPK = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        return 0.0
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def herb_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    positive_ranks = ranks[np.flatnonzero(labels == 1)]
    first_hit = int(positive_ranks.min())
    out = {
        "MRR": float(1.0 / first_hit),
        "AUPRC": average_precision(labels, scores),
        "first_hit_rank": first_hit,
    }
    for k in K_LIST:
        out[f"Hits@{k}"] = float(first_hit <= k)
    return out


def load_matrices(data_dir: Path, derived_dir: Path, split_dir: Path):
    hi = pd.read_csv(data_dir / "hit2_herbs_ingredients.csv", low_memory=False)[
        ["Herb ID", "Related Compound ID"]
    ].dropna()
    hi.columns = ["herb", "compound"]
    ip = pd.read_csv(derived_dir / "IP_literature_counts.csv", low_memory=False)[
        ["Compound ID", "Gene Symbol", "lit_n"]
    ].dropna()
    ip.columns = ["compound", "protein", "lit_n"]
    ip["weight"] = np.log1p(ip["lit_n"].astype(float))
    ip = ip.groupby(["compound", "protein"], as_index=False)["weight"].max()

    hip = hi.merge(ip, on="compound", how="inner")[["herb", "protein", "weight"]]
    hip = hip.groupby(["herb", "protein"], as_index=False)["weight"].max()

    pd_edges = pd.read_csv(derived_dir / "PD_disgenet_scores.csv", low_memory=False)[
        ["gene_symbol", "disease_id", "score"]
    ].dropna()
    pd_edges.columns = ["protein", "disease", "weight"]
    pd_edges["weight"] = pd_edges["weight"].astype(float).clip(0.0, 1.0)
    pd_edges = pd_edges.groupby(["protein", "disease"], as_index=False)["weight"].max()
    pd_all = pd_edges.copy()

    pd_train_keys = pd.read_csv(split_dir / "PD_train.csv", usecols=["protein", "disease"]).astype(str).drop_duplicates()
    pd_test = pd.read_csv(split_dir / "PD_test.csv", usecols=["protein", "disease"]).astype(str).drop_duplicates()
    pd_edges = pd_edges.merge(pd_train_keys, on=["protein", "disease"], how="inner")

    ppi = pd.read_csv(derived_dir / "PPI_before_induced1.tsv", sep="\t", low_memory=False)[
        ["Gene1", "Gene2", "combine_score"]
    ].dropna()
    ppi.columns = ["p1", "p2", "weight"]
    ppi["weight"] = ppi["weight"].astype(float).clip(0.0, 1.0)
    ppi = ppi.sort_values(["p1", "weight"], ascending=[True, False]).groupby("p1", as_index=False).head(PPI_TOPK)

    herbs = sorted(hip["herb"].astype(str).unique())
    proteins = sorted(set(hip["protein"].astype(str)) | set(pd_all["protein"].astype(str)) | set(ppi["p1"].astype(str)) | set(ppi["p2"].astype(str)))
    diseases = pd_all["disease"].astype(str).drop_duplicates().tolist()
    hmap = {x: i for i, x in enumerate(herbs)}
    pmap = {x: i for i, x in enumerate(proteins)}
    dmap = {x: i for i, x in enumerate(diseases)}

    hip["herb"] = hip["herb"].astype(str)
    hip["protein"] = hip["protein"].astype(str)
    pd_edges["protein"] = pd_edges["protein"].astype(str)
    pd_edges["disease"] = pd_edges["disease"].astype(str)
    ppi["p1"] = ppi["p1"].astype(str)
    ppi["p2"] = ppi["p2"].astype(str)

    h = sparse.csr_matrix(
        (hip["weight"], (hip["herb"].map(hmap), hip["protein"].map(pmap))),
        shape=(len(herbs), len(proteins)),
    )
    h_binary = h.copy()
    h_binary.data[:] = 1.0
    d = sparse.csr_matrix(
        (pd_edges["weight"], (pd_edges["protein"].map(pmap), pd_edges["disease"].map(dmap))),
        shape=(len(proteins), len(diseases)),
    )
    d_binary = d.copy()
    d_binary.data[:] = 1.0
    d_test = sparse.csr_matrix(
        (
            np.ones(len(pd_test), dtype=float),
            (pd_test["protein"].map(pmap), pd_test["disease"].map(dmap)),
        ),
        shape=(len(proteins), len(diseases)),
    )
    a = sparse.csr_matrix(
        (ppi["weight"], (ppi["p1"].map(pmap), ppi["p2"].map(pmap))),
        shape=(len(proteins), len(proteins)),
    )
    proxy_positive = (h_binary @ d_test).tocsr()
    proxy_positive.data[:] = 1.0
    return herbs, diseases, h, h_binary, d, d_binary, a, proxy_positive


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    herbs, diseases, h, h_binary, d, d_binary, a, proxy_positive = load_matrices(
        args.data_dir, args.derived_dir, args.split_dir
    )

    direct_scores = (h @ d).tocsr()
    direct_counts = (h_binary @ d_binary).tocsr()
    hop_scores: dict[int, sparse.csr_matrix] = {}
    cumulative = h.copy()
    frontier = h.copy()
    for hop in (1, 2, 3):
        frontier = (ALPHA_PPI * (frontier @ a)).tocsr()
        cumulative = (cumulative + frontier).tocsr()
        hop_scores[hop] = (cumulative @ d).tocsr()

    eligible = [i for i in range(len(herbs)) if proxy_positive.getrow(i).nnz >= POS_PER_HERB]
    rows: list[dict] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        selected = rng.choice(np.asarray(eligible), size=min(MAX_HERBS_EVAL, len(eligible)), replace=False)
        for hi_idx in selected:
            positive_pool = proxy_positive.getrow(int(hi_idx)).indices
            positives = rng.choice(positive_pool, size=POS_PER_HERB, replace=False).tolist()
            occupied = set(positive_pool.tolist())
            negatives: list[int] = []
            tries = 0
            while len(negatives) < NEG_PER_HERB and tries < NEG_PER_HERB * 50:
                di = int(rng.integers(0, len(diseases)))
                if di not in occupied:
                    negatives.append(di)
                    occupied.add(di)
                tries += 1
            candidates = np.asarray(positives + negatives, dtype=np.int64)
            labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
            for hop in (1, 2, 3):
                scores = hop_scores[hop].getrow(int(hi_idx)).toarray().ravel()[candidates]
                rows.append({"run_seed": seed, "herb_id": herbs[int(hi_idx)], "max_ppi_hops": hop, **herb_metrics(labels, scores)})

    per_herb = pd.DataFrame(rows)
    per_herb.to_csv(args.output_dir / "ppi_hop_ranking_per_herb.csv", index=False)
    per_run = per_herb.groupby(["run_seed", "max_ppi_hops"], as_index=False).agg(
        AUPRC=("AUPRC", "mean"), MRR=("MRR", "mean"),
        **{f"Hits@{k}": (f"Hits@{k}", "mean") for k in K_LIST},
    )
    per_run.to_csv(args.output_dir / "ppi_hop_ranking_per_run.csv", index=False)
    summary = per_run.groupby("max_ppi_hops", as_index=False).agg(
        AUPRC=("AUPRC", "mean"), AUPRC_sd=("AUPRC", "std"),
        MRR=("MRR", "mean"), MRR_sd=("MRR", "std"),
        **{f"Hits@{k}": (f"Hits@{k}", "mean") for k in K_LIST},
    )
    summary.to_csv(args.output_dir / "ppi_hop_ranking_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
