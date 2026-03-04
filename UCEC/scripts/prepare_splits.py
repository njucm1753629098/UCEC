#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucec.data import load_raw_relations, make_splits_and_derived_edges, save_splits
from ucec.preprocess import preprocess_all
from ucec.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing raw/cached edge files.")
    ap.add_argument("--out_dir", required=True, help="Output runs folder (will create seed_* subfolders).")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds.")
    ap.add_argument("--alpha_fdr", type=float, default=0.05, help="FDR threshold for pathway-disease derivation.")
    ap.add_argument("--auto_build_inputs", action="store_true", help="If cached IP/PD/PPI files are missing, build them from raw inputs.")
    ap.add_argument("--k_ppi_top", type=int, default=100, help="Top-k outgoing PPI edges per protein (used when auto-building).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_build_inputs:
        need = [
            os.path.join(args.data_dir, "IP_literature_counts.csv"),
            os.path.join(args.data_dir, "PD_disgenet_scores.csv"),
            os.path.join(args.data_dir, "PPI_before_induced1.tsv"),
        ]
        if any(not os.path.exists(p) for p in need):
            preprocess_all(args.data_dir, k_ppi_top=args.k_ppi_top)

    raw = load_raw_relations(args.data_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for s in seeds:
        set_seed(s)
        run_dir = out_dir / f"seed_{s}"
        run_dir.mkdir(parents=True, exist_ok=True)
        split_rel = make_splits_and_derived_edges(raw=raw, seed=s, alpha_fdr=args.alpha_fdr)
        save_splits(split_rel, str(run_dir))
        print(f"[OK] saved splits to {run_dir}")

if __name__ == "__main__":
    main()
