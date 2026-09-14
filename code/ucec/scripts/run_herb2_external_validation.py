"""Run positive-only validation on HERB 2.0 clinical-trial associations."""

from __future__ import annotations

import argparse
import json
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
from ucec.scripts.run_r1_6_controls import _calibrated_scores
from ucec.stage2 import PerturbConfig, RetrievalConfig, UCECEvidenceScorer
from ucec.utils import set_seed


KS = (1, 5, 10, 20, 30, 50)


def score_candidates(run, z, artifact_dir: Path, herbs: list[str], diseases: list[str], device: str) -> pd.DataFrame:
    retr_cfg = RetrievalConfig(
        max_ing_per_herb=30,
        max_prot_per_ing=20,
        ppi_topk=100,
        max_path_per_prot=20,
        retrieval_budget=100,
        use_ppi_hop=True,
    )
    pert_cfg = PerturbConfig(mc_samples=16, aggregation_budget=30)
    scorer = UCECEvidenceScorer(run, z, retr_cfg=retr_cfg, pert_cfg=pert_cfg, device=device)
    scorer.gate.load_state_dict(torch.load(artifact_dir / "stage2_gate_state.pt", map_location=device))

    rows = []
    for disease in tqdm(diseases, desc="HERB 2.0 external scoring"):
        for herb in herbs:
            result = scorer.compute_pair_evidence(herb, disease)
            rows.append({"herb": herb, "disease": disease, "E": result.E, "U": result.U})
    scored = pd.DataFrame(rows)
    scored["S"] = _calibrated_scores(
        run,
        z.detach().cpu(),
        scored,
        scored["E"].to_numpy(),
        artifact_dir,
    )
    return scored


def aggregate_subject_scores(scores: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subjects = mapping[["Subject_id", "Subject_name", "model_herb_ids"]].drop_duplicates("Subject_id")
    diseases = sorted(mapping["disease_id"].unique())
    for subject in subjects.itertuples(index=False):
        herb_ids = [value for value in subject.model_herb_ids.split(";") if value]
        subset = scores[scores["herb"].isin(herb_ids)]
        missing = set(diseases).difference(subset["disease"].astype(str))
        if missing:
            raise RuntimeError(f"Missing {len(missing)} disease scores for {subject.Subject_id}")
        for disease, group in subset.groupby("disease", sort=True):
            rows.append(
                {
                    "Subject_id": subject.Subject_id,
                    "Subject_name": subject.Subject_name,
                    "disease_id": str(disease),
                    "mapped_model_herb_nodes": len(herb_ids),
                    "S": float(group["S"].mean()),
                    "E": float(group["E"].mean()),
                    "U": float(group["U"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    result["rank_within_subject"] = result.groupby("Subject_id")["S"].rank(
        method="average", ascending=False
    )
    n_diseases = len(diseases)
    result["rank_percentile"] = 1.0 - (
        (result["rank_within_subject"] - 1.0) / max(n_diseases - 1, 1)
    )
    return result


def bootstrap(values: dict[str, np.ndarray], rng: np.random.Generator, n: int) -> tuple[float, float]:
    ids = np.array(sorted(values), dtype=object)
    estimates = np.empty(n, dtype=float)
    for i in range(n):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        estimates[i] = np.concatenate([values[key] for key in sampled]).mean()
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def analyze_subset(
    name: str,
    mapping: pd.DataFrame,
    subject_scores: pd.DataFrame,
    out_dir: Path,
    seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> dict[str, object]:
    positives = mapping.drop_duplicates(["Subject_id", "disease_id"]).merge(
        subject_scores,
        on=["Subject_id", "disease_id"],
        how="left",
        validate="one_to_one",
    )
    if positives["S"].isna().any():
        raise RuntimeError(f"Missing scores in {name} positive set")

    candidate_by_subject = {
        key: group[["rank_within_subject", "rank_percentile"]].to_numpy(dtype=float)
        for key, group in subject_scores.groupby("Subject_id", sort=True)
    }
    positive_counts = positives.groupby("Subject_id").size().to_dict()
    rng = np.random.default_rng(seed)
    null_percentile = np.empty(n_permutations, dtype=float)
    null_recall = {k: np.empty(n_permutations, dtype=float) for k in KS}
    for i in range(n_permutations):
        sampled = []
        for subject_id, count in positive_counts.items():
            candidates = candidate_by_subject[subject_id]
            idx = rng.choice(len(candidates), size=count, replace=False)
            sampled.append(candidates[idx])
        values = np.concatenate(sampled)
        null_percentile[i] = values[:, 1].mean()
        for k in KS:
            null_recall[k][i] = (values[:, 0] <= k).mean()

    metric_rows = []
    for k in KS:
        observed = float((positives["rank_within_subject"] <= k).mean())
        grouped = {
            key: (group["rank_within_subject"].to_numpy(dtype=float) <= k).astype(float)
            for key, group in positives.groupby("Subject_id", sort=True)
        }
        low, high = bootstrap(grouped, rng, n_bootstrap)
        metric_rows.append(
            {
                "metric": f"Recall@{k}",
                "observed": observed,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "random_expectation": float(min(k / subject_scores["disease_id"].nunique(), 1.0)),
                "permutation_p_one_sided": float(
                    (1 + np.sum(null_recall[k] >= observed)) / (n_permutations + 1)
                ),
            }
        )

    observed_percentile = float(positives["rank_percentile"].mean())
    percentile_groups = {
        key: group["rank_percentile"].to_numpy(dtype=float)
        for key, group in positives.groupby("Subject_id", sort=True)
    }
    percentile_ci = bootstrap(percentile_groups, rng, n_bootstrap)
    percentile_p = float(
        (1 + np.sum(null_percentile >= observed_percentile)) / (n_permutations + 1)
    )
    summary = {
        "source": "HERB 2.0 curated herb-level ClinicalTrials.gov associations",
        "subset": name,
        "negative_labels_constructed": False,
        "unrecorded_pairs_role": "ranking candidates only",
        "source_records": int(len(mapping)),
        "unique_herb2_disease_pairs": int(len(positives)),
        "unique_herb2_subjects": int(positives["Subject_id"].nunique()),
        "unique_nct_ids": int(mapping["NCT_id"].nunique()),
        "candidate_diseases_per_subject": int(subject_scores["disease_id"].nunique()),
        "mean_positive_rank": float(positives["rank_within_subject"].mean()),
        "median_positive_rank": float(positives["rank_within_subject"].median()),
        "mean_positive_rank_percentile": observed_percentile,
        "mean_positive_rank_percentile_bootstrap_95_ci": list(percentile_ci),
        "rank_percentile_permutation_p_one_sided": percentile_p,
        "metrics": metric_rows,
        "interpretation": (
            "Positive-only rank enrichment of HERB 2.0 clinical-trial-linked associations. "
            "Trial registration does not by itself establish therapeutic efficacy."
        ),
    }
    positives.to_csv(out_dir / f"herb2_{name}_positive_pair_ranks.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / f"herb2_{name}_metrics.csv", index=False)
    (out_dir / f"herb2_{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--base", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=20000)
    args = parser.parse_args()

    set_seed(args.seed)
    base = Path(args.base)
    run_dir = Path(args.run_dir)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else run_dir
    out_dir = base / "external_validation/HERB2/results"

    full = pd.read_csv(out_dir / "herb2_clinical_mapped_pairs_all.csv", dtype=str, keep_default_na=False)
    strict_path = out_dir / "herb2_clinical_pairs_excluding_cmaup_nct.csv"
    splits = load_splits(str(run_dir))
    run = build_run_graph(splits)
    z = torch.load(artifact_dir / "rgcn_embeddings.pt", map_location="cpu")
    valid_herbs = set(run.index.id_maps["herb"])
    valid_diseases = set(run.index.id_maps["disease"])

    model_herbs = sorted(
        {
            herb_id
            for values in full["model_herb_ids"]
            for herb_id in values.split(";")
            if herb_id in valid_herbs
        }
    )
    diseases = sorted(set(full["disease_id"]).intersection(valid_diseases))
    scores = score_candidates(run, z, artifact_dir, model_herbs, diseases, args.device)
    scores.to_csv(out_dir / "herb2_model_node_candidate_scores.csv", index=False)
    subject_scores = aggregate_subject_scores(scores, full)
    subject_scores.to_csv(out_dir / "herb2_subject_candidate_scores.csv", index=False)

    summaries = {
        "all_mapped": analyze_subset(
            "all_mapped", full, subject_scores, out_dir, args.seed, args.bootstrap, args.permutations
        )
    }
    if strict_path.exists():
        strict = pd.read_csv(strict_path, dtype=str, keep_default_na=False)
        summaries["excluding_cmaup_nct"] = analyze_subset(
            "excluding_cmaup_nct",
            strict,
            subject_scores[subject_scores["Subject_id"].isin(strict["Subject_id"].unique())],
            out_dir,
            args.seed + 1,
            args.bootstrap,
            args.permutations,
        )
    (out_dir / "herb2_external_validation_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
