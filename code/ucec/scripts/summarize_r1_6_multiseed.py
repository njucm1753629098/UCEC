from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


MODES = ["full", "uniform", "shuffled_evidence", "relation_fixed", "mc_only", "evidence_only"]
KEYS = ["herb", "disease", "label", "S"]


def aurc(risk: np.ndarray, error: np.ndarray) -> float:
    order = np.argsort(risk, kind="mergesort")
    cumulative = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    coverage = np.arange(1, len(error) + 1) / len(error)
    return float(np.trapz(np.r_[0.0, cumulative], np.r_[0.0, coverage]))


def metrics(risk: np.ndarray, error: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(error, risk)),
        "auprc": float(average_precision_score(error, risk)),
        "aurc": aurc(risk, error),
    }


def bootstrap_auc_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    auc_a, auc_b, differences = [], [], []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[group] for group in sampled_groups])
        if np.unique(y[idx]).size < 2:
            continue
        a = roc_auc_score(y[idx], score_a[idx])
        b = roc_auc_score(y[idx], score_b[idx])
        auc_a.append(a)
        auc_b.append(b)
        differences.append(a - b)
    return {
        "score_a_auroc": float(roc_auc_score(y, score_a)),
        "score_a_95ci": np.quantile(auc_a, [0.025, 0.975]).tolist(),
        "score_b_auroc": float(roc_auc_score(y, score_b)),
        "score_b_95ci": np.quantile(auc_b, [0.025, 0.975]).tolist(),
        "auroc_difference": float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)),
        "difference_95ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "bootstrap_probability_difference_le_zero": float(np.mean(np.asarray(differences) <= 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--evidence_bins", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    paths = sorted(Path(args.input_root).glob("seed_*/r1_6_control_scores.csv"))
    if not paths:
        raise FileNotFoundError(f"No seed outputs found below {args.input_root}")

    frames = []
    reference = None
    per_seed_rows = []
    for path in paths:
        seed_name = path.parent.name
        df = pd.read_csv(path).sort_values(["herb", "disease"]).reset_index(drop=True)
        if reference is None:
            reference = df[KEYS].copy()
        elif not reference.equals(df[KEYS]):
            raise ValueError(f"Candidate rows differ in {path}")
        error = (df["label"].to_numpy(dtype=int) == 0).astype(int)
        for mode in MODES:
            row = {"seed": seed_name, "mode": mode, **metrics(df[f"U_{mode}"].to_numpy(), error)}
            per_seed_rows.append(row)
        per_seed_rows.append(
            {"seed": seed_name, "mode": "evidence_score_alone", **metrics(-df["E_full"].to_numpy(), error)}
        )
        df["seed"] = seed_name
        frames.append(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(out_dir / "control_metrics_by_seed.csv", index=False)
    summary = per_seed.groupby("mode")[["auroc", "auprc", "aurc"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(out_dir / "control_metrics_summary.csv", index=False)

    combined = pd.concat(frames, ignore_index=True)
    value_cols = [f"U_{mode}" for mode in MODES] + [f"E_{mode}" for mode in MODES]
    aggregate = combined.groupby(KEYS, as_index=False)[value_cols].mean()
    aggregate.to_csv(out_dir / "aggregate_candidate_scores.csv", index=False)

    stratified = aggregate.loc[aggregate["E_full"] > 0].copy()
    error = (stratified["label"].to_numpy(dtype=int) == 0).astype(int)
    stratified["evidence_stratum"] = pd.qcut(
        stratified["E_full"], q=args.evidence_bins, labels=False, duplicates="drop"
    )
    stratified["U_within_evidence_stratum"] = stratified.groupby("evidence_stratum")["U_full"].rank(pct=True)
    stratified_u = stratified["U_within_evidence_stratum"].to_numpy(dtype=float)
    evidence_risk = -stratified["E_full"].to_numpy(dtype=float)
    direct = bootstrap_auc_difference(
        error,
        stratified_u,
        evidence_risk,
        stratified["herb"].astype(str).to_numpy(),
        args.bootstrap,
        args.seed,
    )
    direct.update(
        {
            "n_candidates": int(len(stratified)),
            "n_operationally_unsupported": int(error.sum()),
            "evidence_strata": int(stratified["evidence_stratum"].nunique()),
            "inclusion": "top-three candidates per natural product with at least one retrieved evidence chain",
            "bootstrap_unit": "natural product",
            "score_a": "uncertainty ranked within evidence strata",
            "score_b": "negative pair-level evidence score",
        }
    )

    correlation_rows = []
    for mode in MODES:
        correlation_rows.append(
            {
                "mode": mode,
                "spearman_with_full_U": float(aggregate["U_full"].corr(aggregate[f"U_{mode}"], method="spearman")),
                "spearman_with_pair_evidence": float(aggregate[f"U_{mode}"].corr(aggregate["E_full"], method="spearman")),
                "mean_uncertainty": float(aggregate[f"U_{mode}"].mean()),
            }
        )
    pd.DataFrame(correlation_rows).to_csv(out_dir / "control_rank_correlations.csv", index=False)
    stratified.to_csv(out_dir / "evidence_stratified_candidate_scores.csv", index=False)
    with open(out_dir / "evidence_stratified_result.json", "w", encoding="utf-8") as handle:
        json.dump(direct, handle, indent=2)

    nested_rows = []
    for cv_seed in [2026, 2027, 2028, 2029, 2030]:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cv_seed)
        predictions = {}
        for name, features in {
            "relevance_plus_evidence": ["S", "E_full"],
            "relevance_plus_evidence_plus_uncertainty": ["S", "E_full", "U_full"],
        }.items():
            oof = np.zeros(len(stratified), dtype=float)
            for train_idx, test_idx in cv.split(stratified, error, groups=stratified["herb"]):
                model = make_pipeline(
                    SplineTransformer(n_knots=5, degree=3, include_bias=False),
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced"),
                )
                model.fit(stratified.iloc[train_idx][features], error[train_idx])
                oof[test_idx] = model.predict_proba(stratified.iloc[test_idx][features])[:, 1]
            predictions[name] = oof
        base = metrics(predictions["relevance_plus_evidence"], error)
        extended = metrics(predictions["relevance_plus_evidence_plus_uncertainty"], error)
        nested_rows.append(
            {
                "cv_seed": cv_seed,
                "baseline_auroc": base["auroc"],
                "baseline_auprc": base["auprc"],
                "extended_auroc": extended["auroc"],
                "extended_auprc": extended["auprc"],
                "delta_auroc": extended["auroc"] - base["auroc"],
                "delta_auprc": extended["auprc"] - base["auprc"],
            }
        )
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(out_dir / "incremental_value_grouped_cv.csv", index=False)
    nested_summary = nested.drop(columns="cv_seed").agg(["mean", "std"]).T.reset_index(names="metric")
    nested_summary.to_csv(out_dir / "incremental_value_grouped_cv_summary.csv", index=False)

    print(summary.to_string())
    print(json.dumps(direct, indent=2))
    print(nested_summary.to_string(index=False))


if __name__ == "__main__":
    main()
