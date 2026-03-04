from __future__ import annotations

import os
import re
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd


def _need(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")


def _parse_lit_count(x) -> int:
    if pd.isna(x):
        return 0
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else 0


def build_ip_literature_counts(path_hit2_ip: str) -> pd.DataFrame:
    _need(path_hit2_ip)
    df = pd.read_csv(path_hit2_ip, low_memory=False)
    if "\tCommon name" not in df.columns and "Common name" in df.columns:
        df = df.rename(columns={"Common name": "\tCommon name"})
    cols = ["Compound ID", "\tCommon name", "Gene Symbol", "UniprotID", "No. of Literature Evidence"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"hit2_ingredients_targets.csv missing columns: {missing}")
    df = df[cols].copy()
    df["lit_n"] = df["No. of Literature Evidence"].apply(_parse_lit_count).astype(int)
    df = df.drop(columns=["No. of Literature Evidence"])
    df = df.dropna(subset=["Compound ID", "Gene Symbol"])
    df["Compound ID"] = df["Compound ID"].astype(str)
    df["\tCommon name"] = df["\tCommon name"].astype(str)
    df["Gene Symbol"] = df["Gene Symbol"].astype(str)
    df["UniprotID"] = df["UniprotID"].astype(str).fillna("")
    df = df.groupby(["Compound ID", "\tCommon name", "Gene Symbol", "UniprotID"], as_index=False)["lit_n"].max()
    return df


def build_pd_disgenet_scores(path_disgenet: str) -> pd.DataFrame:
    _need(path_disgenet)
    df = pd.read_csv(path_disgenet, low_memory=False)
    cols = ["gene_symbol", "disease_id", "score"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"disgenet_target_disease.csv missing columns: {missing}")
    df = df[cols].dropna().copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    df["disease_id"] = df["disease_id"].astype(str)
    df["score"] = df["score"].astype(float).clip(0.0, 1.0)
    df = df.groupby(["gene_symbol", "disease_id"], as_index=False)["score"].max()
    return df


def build_ppi_pruned(path_ppi: str, relevant_proteins: Set[str], topk: int = 100) -> pd.DataFrame:

    _need(path_ppi)
    df = pd.read_csv(path_ppi, sep="\t", low_memory=False)
    cols = ["Gene1", "Gene2", "combine_score"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"combine_score.tsv missing columns: {missing}")
    df = df[cols].dropna().copy()
    df["Gene1"] = df["Gene1"].astype(str)
    df["Gene2"] = df["Gene2"].astype(str)
    df["combine_score"] = df["combine_score"].astype(float).clip(0.0, 1.0)
    if relevant_proteins:
        df = df[df["Gene1"].isin(relevant_proteins) & df["Gene2"].isin(relevant_proteins)].copy()
    df1 = df.rename(columns={"Gene1": "src", "Gene2": "dst"})
    df2 = df.rename(columns={"Gene2": "src", "Gene1": "dst"})
    dd = pd.concat([df1, df2], ignore_index=True)
    dd = dd.sort_values(["src", "combine_score"], ascending=[True, False])
    dd = dd.groupby("src", as_index=False).head(topk)

    a = dd["src"].astype(str).to_numpy()
    b = dd["dst"].astype(str).to_numpy()
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    out = pd.DataFrame({"Gene1": mn, "Gene2": mx, "combine_score": dd["combine_score"].to_numpy()})
    out = out.sort_values(["Gene1", "Gene2", "combine_score"], ascending=[True, True, False])
    out = out.drop_duplicates(["Gene1", "Gene2"], keep="first")
    return out


def preprocess_all(
    data_dir: str,
    k_ppi_top: int = 100,
    in_hit2_ip: str = "hit2_ingredients_targets.csv",
    in_disgenet_pd: str = "disgenet_target_disease.csv",
    in_reactome_pp: str = "reactome_protein_pathway.csv",
    in_ppi: str = "combine_score.tsv",
    out_ip: str = "IP_literature_counts.csv",
    out_pd: str = "PD_disgenet_scores.csv",
    out_ppi: str = "PPI_before_induced1.tsv",
) -> Tuple[str, str, str]:

    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    path_hit2_ip = os.path.join(data_dir, in_hit2_ip)
    path_disgenet = os.path.join(data_dir, in_disgenet_pd)
    path_ppath = os.path.join(data_dir, in_reactome_pp)
    path_ppi = os.path.join(data_dir, in_ppi)

    for p in [path_hit2_ip, path_disgenet, path_ppath, path_ppi]:
        _need(p)

    ip_counts = build_ip_literature_counts(path_hit2_ip)
    out_ip_path = os.path.join(data_dir, out_ip)
    ip_counts.to_csv(out_ip_path, index=False)

    pd_scores = build_pd_disgenet_scores(path_disgenet)
    out_pd_path = os.path.join(data_dir, out_pd)
    pd_scores.to_csv(out_pd_path, index=False)

    pp = pd.read_csv(path_ppath, low_memory=False)
    if "protein" not in pp.columns or "pathway" not in pp.columns:
        raise KeyError("reactome_protein_pathway.csv must contain columns: protein, pathway")
    pp = pp[["protein", "pathway"]].dropna().copy()
    pp["protein"] = pp["protein"].astype(str)

    relevant = set(ip_counts["Gene Symbol"].astype(str)) | set(pd_scores["gene_symbol"].astype(str)) | set(pp["protein"].astype(str))
    ppi_pruned = build_ppi_pruned(path_ppi, relevant_proteins=relevant, topk=k_ppi_top)
    out_ppi_path = os.path.join(data_dir, out_ppi)
    ppi_pruned.to_csv(out_ppi_path, sep="\t", index=False, float_format="%.3f")

    return out_ip_path, out_pd_path, out_ppi_path
