#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucec.preprocess import preprocess_all
from ucec.data import derive_pathway_disease_edges


def main():
    ap = argparse.ArgumentParser(
        description="Build cached edge tables (IP/PD/PPI) from raw HIT2/DisGeNET/Reactome/STRING inputs."
    )
    ap.add_argument("--data_dir", required=True, help="Folder containing raw input files.")
    ap.add_argument("--k_ppi_top", type=int, default=100, help="Top-k outgoing PPI edges per protein (default 100).")

    ap.add_argument("--in_hit2_ip", default="hit2_ingredients_targets.csv")
    ap.add_argument("--in_disgenet_pd", default="disgenet_target_disease.csv")
    ap.add_argument("--in_reactome_pp", default="reactome_protein_pathway.csv")
    ap.add_argument("--in_ppi", default="combine_score.tsv")

    ap.add_argument("--out_ip", default="IP_literature_counts.csv")
    ap.add_argument("--out_pd", default="PD_disgenet_scores.csv")
    ap.add_argument("--out_ppi", default="PPI_before_induced1.tsv")
    args = ap.parse_args()

    out_ip, out_pd, out_ppi = preprocess_all(
        data_dir=args.data_dir,
        k_ppi_top=args.k_ppi_top,
        in_hit2_ip=args.in_hit2_ip,
        in_disgenet_pd=args.in_disgenet_pd,
        in_reactome_pp=args.in_reactome_pp,
        in_ppi=args.in_ppi,
        out_ip=args.out_ip,
        out_pd=args.out_pd,
        out_ppi=args.out_ppi,
    )

    print("[OK] generated cached inputs:")
    print(" -", out_ip)
    print(" -", out_pd)
    print(" -", out_ppi)

    try:
        import pandas as pd
        import os

        pp = pd.read_csv(os.path.join(args.data_dir, args.in_reactome_pp), low_memory=False)[
            ["protein", "pathway"]
        ].dropna().astype(str).drop_duplicates()

        pd_df = pd.read_csv(out_pd, low_memory=False)[["gene_symbol", "disease_id", "score"]].dropna().copy()
        pd_df = pd_df.rename(columns={"gene_symbol": "protein", "disease_id": "disease"})
        pd_df["protein"] = pd_df["protein"].astype(str)
        pd_df["disease"] = pd_df["disease"].astype(str)
        pd_df = pd_df.drop_duplicates(subset=["protein", "disease"])

        ip_df = pd.read_csv(out_ip, low_memory=False)[["Gene Symbol"]].dropna().copy()
        bg = set(ip_df["Gene Symbol"].astype(str)) | set(pd_df["protein"].astype(str)) | set(pp["protein"].astype(str))

        pathd = derive_pathway_disease_edges(
            pp=pp,
            pd_train=pd_df[["protein", "disease"]],
            bg_proteins=bg,
            alpha=0.05,
        )
        out_pathd = os.path.join(args.data_dir, "pathway_disease_edges_FDR_lt_0p05.csv")
        pathd.to_csv(out_pathd, index=False)
        print(" -", out_pathd)
    except Exception as e:
        print(f"[WARN] Could not write global PathD file: {e}")


if __name__ == "__main__":
    main()
