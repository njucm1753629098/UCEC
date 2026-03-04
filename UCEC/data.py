from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

from .schema import REL_SPECS, NODE_TYPES


def _need(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")


def _scale01_with_train(train_vals: np.ndarray, vals: np.ndarray) -> np.ndarray:
    train_vals = np.asarray(train_vals, dtype=float)
    vals = np.asarray(vals, dtype=float)
    mn = float(np.min(train_vals)) if train_vals.size else 0.0
    mx = float(np.max(train_vals)) if train_vals.size else 1.0
    if mx - mn < 1e-12:
        return np.zeros_like(vals, dtype=float)
    return (vals - mn) / (mx - mn)


def split_edges(df: pd.DataFrame, seed: int, ratios=(0.8, 0.1, 0.1)) -> Dict[str, pd.DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-9
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    tr = df.iloc[:n_tr].copy()
    va = df.iloc[n_tr:n_tr + n_va].copy()
    te = df.iloc[n_tr + n_va:].copy()
    return {"train": tr, "val": va, "test": te}


def _bh_fdr_for_nonzero(pvals_nonzero: np.ndarray, m_total: int) -> np.ndarray:
    p = np.asarray(pvals_nonzero, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    q = np.clip(q * (m_total / max(n, 1)), 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def derive_pathway_disease_edges(
    pp: pd.DataFrame,
    pd_train: pd.DataFrame,
    bg_proteins: Optional[set[str]] = None,
    alpha: float = 0.05,
    min_disease_genes: int = 10,
    min_pathway_genes: int = 10,
    max_pathway_genes: int = 500,
) -> pd.DataFrame:

    pp = pp[["protein", "pathway"]].dropna().copy()
    pp["protein"] = pp["protein"].astype(str)
    pp["pathway"] = pp["pathway"].astype(str)
    pp = pp.drop_duplicates()

    pd_train = pd_train[["protein", "disease"]].dropna().copy()
    pd_train["protein"] = pd_train["protein"].astype(str)
    pd_train["disease"] = pd_train["disease"].astype(str)
    pd_train = pd_train.drop_duplicates()

    # Filter diseases by gene-set size
    disease_sizes = pd_train.groupby("disease")["protein"].nunique()
    keep_diseases = set(disease_sizes[disease_sizes >= min_disease_genes].index.astype(str))
    pd_train = pd_train[pd_train["disease"].isin(keep_diseases)].copy()

    # Filter pathways by gene-set size
    pathway_sizes = pp.groupby("pathway")["protein"].nunique()
    keep_pathways = set(
        pathway_sizes[
            (pathway_sizes >= min_pathway_genes) & (pathway_sizes <= max_pathway_genes)
        ].index.astype(str)
    )
    pp = pp[pp["pathway"].isin(keep_pathways)].copy()

    if bg_proteins is None:
        proteins = sorted(set(pp["protein"]) | set(pd_train["protein"]))
    else:
        proteins = sorted(set(map(str, bg_proteins)))
    gene_to_idx = {g: i for i, g in enumerate(proteins)}
    N = len(proteins)

    pathways = pp["pathway"].unique()
    diseases = pd_train["disease"].unique()
    p_to_idx = {p: i for i, p in enumerate(pathways)}
    d_to_idx = {d: i for i, d in enumerate(diseases)}

    pg_rows = pp["pathway"].map(p_to_idx).values
    pg_cols = pp["protein"].map(gene_to_idx).values
    pg = sparse.coo_matrix((np.ones(len(pp), dtype=np.int8), (pg_rows, pg_cols)),
                           shape=(len(pathways), len(proteins))).tocsr()

    dg_rows = pd_train["disease"].map(d_to_idx).values
    dg_cols = pd_train["protein"].map(gene_to_idx).values
    dg = sparse.coo_matrix((np.ones(len(pd_train), dtype=np.int8), (dg_rows, dg_cols)),
                           shape=(len(diseases), len(proteins))).tocsr()

    overlap = (pg @ dg.T).tocoo()
    k = overlap.data.astype(int)
    p_idx = overlap.row
    d_idx = overlap.col

    pathway_size = np.asarray(pg.sum(axis=1)).ravel().astype(int)
    disease_size = np.asarray(dg.sum(axis=1)).ravel().astype(int)

    K_arr = pathway_size[p_idx]
    n_arr = disease_size[d_idx]

    pvals = hypergeom.sf(k - 1, N, K_arr, n_arr)

    m_total = len(pathways) * len(diseases)
    qvals = _bh_fdr_for_nonzero(pvals, m_total=m_total)

    sig = qvals < alpha
    out = pd.DataFrame({
        "pathway": pathways[p_idx[sig]],
        "disease": diseases[d_idx[sig]],
        "overlap_k": k[sig],
        "disease_size_n": n_arr[sig],
        "pathway_size_K": K_arr[sig],
        "fdr_q": qvals[sig],
    })
    out["neglog10_fdr"] = (-np.log10(out["fdr_q"].clip(1e-300, 1.0))).astype(float)
    if len(out) == 0:
        out["r_wd"] = np.array([], dtype=float)
    else:
        v = out["neglog10_fdr"].values.astype(float)
        vmin = float(v.min())
        vmax = float(v.max())
        if vmax <= vmin:
            out["r_wd"] = 1.0
        else:
            out["r_wd"] = (v - vmin) / (vmax - vmin)
    out = out.sort_values(["fdr_q", "neglog10_fdr"], ascending=[True, False]).reset_index(drop=True)
    return out


@dataclass
class RawRelations:
    HI: pd.DataFrame
    IP: pd.DataFrame
    PD: pd.DataFrame
    PPi: pd.DataFrame
    PPath: pd.DataFrame


def load_raw_relations(data_dir: str, files: Optional[Dict[str, str]] = None) -> RawRelations:
    files = files or {}
    f_hi = os.path.join(data_dir, files.get("HI", "hit2_herbs_ingredients.csv"))
    f_ip = os.path.join(data_dir, files.get("IP", "IP_literature_counts.csv"))
    f_pd = os.path.join(data_dir, files.get("PD", "PD_disgenet_scores.csv"))
    f_ppi = os.path.join(data_dir, files.get("PPi", "PPI_before_induced1.tsv"))
    f_ppath = os.path.join(data_dir, files.get("PPath", "reactome_protein_pathway.csv"))

    for p in [f_hi, f_ip, f_pd, f_ppi, f_ppath]:
        _need(p)

    hi = pd.read_csv(f_hi, low_memory=False)[["Herb ID", "Related Compound ID"]].dropna()
    hi.columns = ["herb", "ingredient"]
    hi["herb"] = hi["herb"].astype(str)
    hi["ingredient"] = hi["ingredient"].astype(str)
    hi = hi.drop_duplicates()

    ip = pd.read_csv(f_ip, low_memory=False)[["Compound ID", "Gene Symbol", "lit_n"]].dropna()
    ip.columns = ["ingredient", "protein", "lit_n"]
    ip["ingredient"] = ip["ingredient"].astype(str)
    ip["protein"] = ip["protein"].astype(str)
    ip["lit_n"] = ip["lit_n"].astype(float).fillna(0.0).clip(0.0, None)
    ip = ip.groupby(["ingredient", "protein"], as_index=False)["lit_n"].max()

    pd_df = pd.read_csv(f_pd, low_memory=False)[["gene_symbol", "disease_id", "score"]].dropna()
    pd_df.columns = ["protein", "disease", "score"]
    pd_df["protein"] = pd_df["protein"].astype(str)
    pd_df["disease"] = pd_df["disease"].astype(str)
    pd_df["score"] = pd_df["score"].astype(float).clip(0.0, 1.0)
    pd_df = pd_df.groupby(["protein", "disease"], as_index=False)["score"].max()

    ppi = pd.read_csv(f_ppi, sep="\t", low_memory=False)[["Gene1", "Gene2", "combine_score"]].dropna()
    ppi.columns = ["protein", "protein2", "score"]
    ppi["protein"] = ppi["protein"].astype(str)
    ppi["protein2"] = ppi["protein2"].astype(str)
    ppi["score"] = ppi["score"].astype(float).clip(0.0, 1.0)
    ppi = ppi.drop_duplicates(["protein", "protein2"])

    ppath = pd.read_csv(f_ppath, low_memory=False)[["protein", "pathway"]].dropna()
    ppath["protein"] = ppath["protein"].astype(str)
    ppath["pathway"] = ppath["pathway"].astype(str)
    ppath = ppath.drop_duplicates()

    return RawRelations(HI=hi, IP=ip, PD=pd_df, PPi=ppi, PPath=ppath)


@dataclass
class SplitRelations:
    edges: Dict[str, Dict[str, pd.DataFrame]]
    evidence_scalers: Dict[str, Dict[str, float]]


def make_splits_and_derived_edges(
    raw: RawRelations,
    seed: int,
    alpha_fdr: float = 0.05,
    ratios=(0.8, 0.1, 0.1),
) -> SplitRelations:

    edges: Dict[str, Dict[str, pd.DataFrame]] = {}
    evidence_scalers: Dict[str, Dict[str, float]] = {}
    meta: Dict[str, str] = {"seed": str(seed)}

    def _empty(cols):
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    df_hi = raw.HI[["herb", "ingredient"]].copy()
    df_hi["evidence"] = 1.0
    edges.setdefault("HI", {})["train"] = df_hi
    edges["HI"]["val"] = _empty(["herb", "ingredient", "evidence"])
    edges["HI"]["test"] = _empty(["herb", "ingredient", "evidence"])

    s_ip = split_edges(raw.IP, seed=seed + 11, ratios=ratios)
    train_log = np.log1p(s_ip["train"]["lit_n"].values.astype(float))

    mn = float(train_log.min()) if train_log.size else 0.0
    mx = float(train_log.max()) if train_log.size else 1.0
    evidence_scalers["IP"] = {"min": mn, "max": mx}
    for split, df in s_ip.items():
        logv = np.log1p(df["lit_n"].values.astype(float))
        e = _scale01_with_train(train_log, logv)
        df2 = df[["ingredient", "protein"]].copy()
        df2["evidence"] = e.astype(float)
        edges.setdefault("IP", {})[split] = df2

    s_pd = split_edges(raw.PD, seed=seed + 23, ratios=ratios)
    for split, df in s_pd.items():
        df2 = df[["protein", "disease"]].copy()
        df2["evidence"] = df["score"].values.astype(float)
        edges.setdefault("PD", {})[split] = df2

    df_ppi = raw.PPi[["protein", "protein2", "score"]].copy()
    df2 = df_ppi[["protein", "protein2"]].copy()
    df2["evidence"] = df_ppi["score"].values.astype(float)
    edges.setdefault("PPi", {})["train"] = df2
    edges["PPi"]["val"] = _empty(["protein", "protein2", "evidence"])
    edges["PPi"]["test"] = _empty(["protein", "protein2", "evidence"])

    df_ppath = raw.PPath[["protein", "pathway"]].copy()
    df_ppath["evidence"] = 1.0
    edges.setdefault("PPath", {})["train"] = df_ppath
    edges["PPath"]["val"] = _empty(["protein", "pathway", "evidence"])
    edges["PPath"]["test"] = _empty(["protein", "pathway", "evidence"])

    pd_train = edges["PD"]["train"][["protein", "disease"]].copy()
    pp_full = raw.PPath[["protein", "pathway"]].copy()

    bg = set(raw.IP["protein"].astype(str)) | set(raw.PD["protein"].astype(str)) | set(raw.PPath["protein"].astype(str))
    pathd_full = derive_pathway_disease_edges(
        pp=pp_full,
        pd_train=pd_train,
        bg_proteins=bg,
        alpha=alpha_fdr,
    )

    df2 = pathd_full[["pathway", "disease"]].copy()
    df2["evidence"] = pathd_full["r_wd"].values.astype(float)
    edges.setdefault("PathD", {})["train"] = df2
    edges["PathD"]["val"] = _empty(["pathway", "disease", "evidence"])
    edges["PathD"]["test"] = _empty(["pathway", "disease", "evidence"])
    evidence_scalers["PathD"] = {"min": 0.0, "max": 1.0}

    return SplitRelations(edges=edges, evidence_scalers=evidence_scalers, meta=meta)


def save_splits(split_rel: SplitRelations, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Save meta
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"meta": split_rel.meta, "evidence_scalers": split_rel.evidence_scalers}, f, indent=2)

    for rel, dd in split_rel.edges.items():
        for split, df in dd.items():
            df.to_csv(os.path.join(out_dir, f"{rel}_{split}.csv"), index=False)


def load_splits(run_dir: str) -> SplitRelations:
    import json
    meta_path = os.path.join(run_dir, "meta.json")
    _need(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    evidence_scalers = obj.get("evidence_scalers", {})
    meta = obj.get("meta", {})

    edges: Dict[str, Dict[str, pd.DataFrame]] = {}
    for rel in [r.name for r in REL_SPECS]:
        for split in ["train", "val", "test"]:
            p = os.path.join(run_dir, f"{rel}_{split}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, low_memory=False)
                edges.setdefault(rel, {})[split] = df
    return SplitRelations(edges=edges, evidence_scalers=evidence_scalers, meta=meta)
