from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .graph import RunGraph


@dataclass
class ProxyBenchmark:
    pairs: pd.DataFrame


def build_herb_targets(run: RunGraph, max_targets_per_herb: int = 200) -> Dict[str, List[str]]:

    hi_tr = run.splits.edges["HI"]["train"][["herb", "ingredient"]].copy()
    ip_tr = run.splits.edges["IP"]["train"][["ingredient", "protein", "evidence"]].copy()

    hip = hi_tr.merge(ip_tr, on="ingredient", how="inner")

    out: Dict[str, List[str]] = {}
    for herb, g in hip.groupby("herb"):
        g2 = g.groupby("protein", as_index=False)["evidence"].max()
        g2 = g2.sort_values("evidence", ascending=False).head(max_targets_per_herb)
        out[str(herb)] = g2["protein"].astype(str).tolist()
    return out


def build_proxy_labels_from_heldout_pd(run: RunGraph, herb_targets: Dict[str, List[str]], pd_split: str = "test") -> Dict[Tuple[str, str], int]:

    held = run.splits.edges["PD"][pd_split][["protein", "disease"]].copy()

    dis2prot: Dict[str, set] = {}
    for d, g in held.groupby("disease"):
        dis2prot[str(d)] = set(g["protein"].astype(str).tolist())

    labels: Dict[Tuple[str, str], int] = {}
    for herb, prots in herb_targets.items():
        prot_set = set(prots)
        for d, ps in dis2prot.items():
            if prot_set & ps:
                labels[(herb, d)] = 1
    return labels


def sample_proxy_benchmark(
    run: RunGraph,
    seed: int,
    n_herbs: int = 500,
    pos_per_herb: int = 4,
    neg_per_herb: int = 16,
    pd_split: str = "test",
) -> ProxyBenchmark:
    rng = np.random.default_rng(seed)
    herb_targets = build_herb_targets(run)

    pos_labels = build_proxy_labels_from_heldout_pd(run, herb_targets, pd_split=pd_split)

    diseases_universe = sorted(run.index.id_maps["disease"].keys())
    herbs = sorted(herb_targets.keys())
    if not herbs:
        raise RuntimeError("No herbs with training targets; check HI/IP train splits.")
    herbs_sel = rng.choice(np.array(herbs, dtype=object), size=min(n_herbs, len(herbs)), replace=False)

    rows = []
    for herb in herbs_sel:
        herb = str(herb)
        pos_ds = [d for (h, d), y in pos_labels.items() if h == herb and y == 1]
        if len(pos_ds) == 0:
            continue
        rng.shuffle(pos_ds)
        pos_pick = pos_ds[:min(pos_per_herb, len(pos_ds))]

        pos_set = set(pos_ds)
        neg_candidates = [d for d in diseases_universe if d not in pos_set]
        if len(neg_candidates) == 0:
            continue
        neg_pick = rng.choice(np.array(neg_candidates, dtype=object), size=min(neg_per_herb, len(neg_candidates)), replace=False).tolist()

        for d in pos_pick:
            rows.append((herb, str(d), 1))
        for d in neg_pick:
            rows.append((herb, str(d), 0))

    pairs = pd.DataFrame(rows, columns=["herb", "disease", "label"]).drop_duplicates()
    return ProxyBenchmark(pairs=pairs)
