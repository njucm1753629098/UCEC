"""Prepare a positive-only HERB 2.0 clinical validation set and audit CMAUP overlap.

Only HERB 2.0 records whose subject type is ``Herb`` are considered. Disease
conditions and herb names are mapped conservatively to nodes already present in
UCEC. Unrecorded pairs are not assigned negative labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def norm(value: object) -> str:
    text = str(value or "").lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]", "", text)


def is_name(value: object) -> bool:
    return norm(value) not in {"", "na", "nan", "none", "null", "unknown"}


# Restricted to common, unambiguous clinical aliases whose target UMLS concept
# is present in the UCEC disease catalog.
CURATED_DISEASE_ALIASES = {
    "alzheimer's": "C0002395",
    "alzheimer's disease": "C0002395",
    "alzheimers disease": "C0002395",
    "rheumatoid arthritis": "C0003873",
    "type 2 diabetes": "C0011849",
    "type 2 diabetes mellitus": "C0011849",
    "type ii diabetes": "C0011849",
    "breast cancer": "C0678222",
    "prostate cancer": "C0600139",
    "parkinson's disease": "C0030567",
    "parkinson disease": "C0030567",
    "metabolic syndrome": "C0524620",
    "lupus nephritis": "C0024143",
    "non-small cell lung cancer": "C0007131",
    "peripheral arterial disease": "C1704436",
    "irritable bowel syndrome": "C0022104",
    "fatty liver disease": "C0015695",
    "polycystic ovary syndrome": "C0032460",
    "polycystic ovarian syndrome": "C0032460",
    "chronic hepatitis c": "C0524910",
    "open angle glaucoma": "C0017612",
    "allergic rhinitis": "C0035457",
    "ankylosing spondylitis": "C0038013",
    "chronic kidney disease": "C1561643",
    "diabetic nephropathy": "C0011881",
    "diabetic neuropathy": "C0011882",
    "diabetic peripheral neuropathy": "C0011882",
    "impaired glucose tolerance": "C0271650",
    "acute respiratory distress syndrome": "C0035222",
    "colorectal cancer": "C0009402",
    "cardiovascular disease": "C0007222",
    "cardiovascular diseases": "C0007222",
    "migraine": "C0149931",
    "chronic migraine": "C0149931",
    "crohn disease": "C0010346",
    "crohn's disease": "C0010346",
    "ulcerative colitis": "C0009324",
    "mild cognitive impairment": "C1270972",
    "sickle cell disease": "C5680749",
    "autism spectrum disorder": "C1510586",
    "bipolar disorder": "C0005586",
    "cystic fibrosis": "C0010674",
    "end stage renal disease": "C2316810",
    "endometriosis": "C0014175",
    "fibromyalgia": "C0016053",
    "multiple sclerosis": "C0026769",
    "osteoporosis": "C0029456",
    "schizophrenia": "C0036341",
    "tourette syndrome": "C0040517",
    "vitiligo": "C0042900",
    "covid-19": "C5203670",
    "covid19": "C5203670",
}
CURATED_DISEASE_ALIASES = {
    norm(key): value for key, value in CURATED_DISEASE_ALIASES.items()
}


NON_DISEASE_TERMS = {
    norm(value)
    for value in [
        "Healthy",
        "Healthy Volunteers",
        "Inflammation",
        "Pain",
        "Fatigue",
        "Nausea",
        "Vomiting",
        "Hyperglycemia",
        "Hyperlipidemia",
        "Dyslipidemia",
        "Insulin Resistance",
        "Body Weight",
        "Arthralgia",
        "Low Back Pain",
    ]
}


def split_conditions(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def add_index(index: dict[str, set[str]], value: object, identifier: str) -> None:
    if is_name(value):
        index.setdefault(norm(value), set()).add(identifier)


def build_herb_mapping(base: Path, herb_info: pd.DataFrame) -> pd.DataFrame:
    model = pd.read_csv(
        base / "source_data/model_graph_data/hit2_herbs_ingredients.csv",
        dtype=str,
        keep_default_na=False,
    )[
        ["Herb ID", "English Name", "Latin Name", "Chinese Pin Yin"]
    ].drop_duplicates()

    indices: dict[str, dict[str, set[str]]] = {
        "english": {},
        "latin": {},
        "pinyin": {},
    }
    for row in model.itertuples(index=False, name=None):
        herb_id, english, latin, pinyin = row
        add_index(indices["english"], english, herb_id)
        add_index(indices["latin"], latin, herb_id)
        add_index(indices["pinyin"], pinyin, herb_id)

    rows = []
    for herb in herb_info.itertuples(index=False):
        matches: dict[str, set[str]] = {}
        fields = {
            "english": getattr(herb, "Herb_en_name"),
            "latin": getattr(herb, "Herb_latin_name"),
            "pinyin": getattr(herb, "Herb_pinyin_name"),
        }
        for field, value in fields.items():
            ids = indices[field].get(norm(value), set()) if is_name(value) else set()
            if ids:
                matches[field] = ids

        all_ids = sorted(set().union(*matches.values())) if matches else []
        # Conflicting identifier sets are retained in the audit but excluded
        # from validation until manually resolved.
        conflict = len({tuple(sorted(ids)) for ids in matches.values()}) > 1
        rows.append(
            {
                "herb2_id": herb.Herb_id,
                "herb2_name": herb.Herb_en_name,
                "herb2_latin": herb.Herb_latin_name,
                "herb2_pinyin": herb.Herb_pinyin_name,
                "model_herb_ids": ";".join(all_ids),
                "matched_fields": ";".join(sorted(matches)),
                "mapping_status": "conflict" if conflict else ("accepted_exact" if all_ids else "unmatched"),
            }
        )
    return pd.DataFrame(rows)


def build_disease_indices(base: Path) -> tuple[dict[str, str], dict[str, str], set[str]]:
    disease = pd.read_csv(
        base / "source_data/model_graph_data/disgenet_target_disease.csv",
        dtype=str,
        keep_default_na=False,
    )[["disease_id", "disease_name"]].drop_duplicates()
    valid_ids = set(disease["disease_id"])
    exact: dict[str, str] = {}
    for row in disease.itertuples(index=False):
        exact.setdefault(norm(row.disease_name), row.disease_id)
    id_to_name = disease.drop_duplicates("disease_id").set_index("disease_id")["disease_name"].to_dict()
    return exact, id_to_name, valid_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(Path(__file__).resolve().parents[3]))
    args = parser.parse_args()

    base = Path(args.base)
    source = base / "external_validation/HERB2/source"
    out_dir = base / "external_validation/HERB2/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(
        source / "HERB_clinical_trials_v2.txt",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    herb_info = pd.read_csv(
        source / "HERB_herb_info_v2.txt",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    herb_mapping = build_herb_mapping(base, herb_info)
    herb_mapping.to_csv(out_dir / "herb2_herb_mapping_audit.csv", index=False)

    herb_trials = clinical[clinical["Subject_type"].str.casefold() == "herb"].copy()
    herb_trials = herb_trials.merge(
        herb_mapping,
        left_on="Subject_id",
        right_on="herb2_id",
        how="left",
        validate="many_to_one",
    )
    herb_trials["herb_mapping_accepted"] = herb_trials["mapping_status"].eq("accepted_exact")

    exact_disease, id_to_name, valid_disease_ids = build_disease_indices(base)
    mapped_rows = []
    for trial in herb_trials.itertuples(index=False):
        if not trial.herb_mapping_accepted:
            continue
        for condition in split_conditions(trial.Study_condition):
            key = norm(condition)
            if key in NON_DISEASE_TERMS:
                continue
            disease_id = exact_disease.get(key) or CURATED_DISEASE_ALIASES.get(key)
            if not disease_id or disease_id not in valid_disease_ids:
                continue
            mapped_rows.append(
                {
                    "Clinical_trial_id": trial.Clinical_trial_id,
                    "NCT_id": trial.NCT_id,
                    "Subject_id": trial.Subject_id,
                    "Subject_name": trial.Subject_name,
                    "model_herb_ids": trial.model_herb_ids,
                    "Study_condition": trial.Study_condition,
                    "condition_component": condition,
                    "disease_id": disease_id,
                    "disease_name": id_to_name.get(disease_id, ""),
                    "disease_mapping_source": "exact_model_name" if key in exact_disease else "curated_alias",
                    "Status": trial.Status,
                    "Phase": trial._8 if hasattr(trial, "_8") else "",
                    "Study_result": trial.Study_result,
                    "Study_type": trial.Study_type,
                    "Intervention_purpose": trial.Intervention_purpose,
                    "Start_date": trial.Start_date,
                    "Completion_date": trial.Completion_date,
                    "URL": trial.URL,
                }
            )

    mapped = pd.DataFrame(mapped_rows)
    if mapped.empty:
        raise RuntimeError("No HERB 2.0 clinical records mapped to UCEC.")
    mapped = mapped.drop_duplicates(["NCT_id", "Subject_id", "disease_id"])

    cmaup_raw_path = base / "external_validation/CMAUP2/source/clinical_trials_associations_full.txt"
    cmaup_mapped_path = base / "external_validation/CMAUP2/results/cmaup_expanded_disease_pairs.csv"
    cmaup_overlap_audited = cmaup_raw_path.exists() and cmaup_mapped_path.exists()
    if cmaup_overlap_audited:
        cmaup_raw = pd.read_csv(
            cmaup_raw_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        cmaup_mapped = pd.read_csv(
            cmaup_mapped_path,
            dtype=str,
            keep_default_na=False,
        )
    else:
        cmaup_raw = pd.DataFrame(columns=["NCT_ID", "Associated_by plant_or_compound"])
        cmaup_mapped = pd.DataFrame(columns=["NCT_ID", "model_herb_ids", "disease_id"])
    cmaup_raw_nct = set(cmaup_raw["NCT_ID"])
    cmaup_direct_plant = cmaup_raw[
        cmaup_raw["Associated_by plant_or_compound"].str.casefold().eq("plant")
    ]
    cmaup_direct_plant_nct = set(cmaup_direct_plant["NCT_ID"])
    cmaup_mapped_nct = set(cmaup_mapped["NCT_ID"])
    mapped["nct_present_in_cmaup_raw"] = mapped["NCT_id"].isin(cmaup_raw_nct)
    mapped["nct_present_in_cmaup_mapped"] = mapped["NCT_id"].isin(cmaup_mapped_nct)

    cmaup_node_pairs = set()
    cmaup_nct_node_pairs = set()
    for row in cmaup_mapped.itertuples(index=False):
        for herb_id in str(row.model_herb_ids).split(";"):
            if herb_id:
                cmaup_node_pairs.add((herb_id, row.disease_id))
                cmaup_nct_node_pairs.add((row.NCT_ID, herb_id, row.disease_id))

    def node_pair_overlap(row: pd.Series) -> bool:
        return any((herb_id, row["disease_id"]) in cmaup_node_pairs for herb_id in row["model_herb_ids"].split(";"))

    def nct_node_pair_overlap(row: pd.Series) -> bool:
        return any(
            (row["NCT_id"], herb_id, row["disease_id"]) in cmaup_nct_node_pairs
            for herb_id in row["model_herb_ids"].split(";")
        )

    mapped["model_herb_disease_pair_in_cmaup"] = mapped.apply(node_pair_overlap, axis=1)
    mapped["same_nct_model_herb_disease_in_cmaup"] = mapped.apply(nct_node_pair_overlap, axis=1)
    mapped.to_csv(out_dir / "herb2_clinical_mapped_pairs_all.csv", index=False)

    unique_nct = mapped[~mapped["nct_present_in_cmaup_raw"]].copy()
    unique_nct.to_csv(out_dir / "herb2_clinical_pairs_excluding_cmaup_nct.csv", index=False)
    overlap = mapped[mapped["nct_present_in_cmaup_raw"]].copy()
    overlap.to_csv(out_dir / "herb2_cmaup_overlap_records.csv", index=False)

    summary = {
        "herb2_release": "HERB 2.0 official clinical trial download",
        "negative_labels_constructed": False,
        "cmaup_overlap_audited": cmaup_overlap_audited,
        "raw_clinical_rows": int(len(clinical)),
        "raw_unique_nct_ids": int(clinical["NCT_id"].nunique()),
        "raw_herb_rows": int(len(herb_trials)),
        "raw_herb_unique_nct_ids": int(herb_trials["NCT_id"].nunique()),
        "cmaup_raw_unique_nct_ids": int(len(cmaup_raw_nct)),
        "cmaup_direct_plant_unique_nct_ids": int(len(cmaup_direct_plant_nct)),
        "raw_herb_nct_overlap_with_cmaup_all": int(
            len(set(herb_trials["NCT_id"]).intersection(cmaup_raw_nct))
        ),
        "raw_herb_nct_overlap_with_cmaup_direct_plant": int(
            len(set(herb_trials["NCT_id"]).intersection(cmaup_direct_plant_nct))
        ),
        "herb2_catalog_herbs": int(len(herb_info)),
        "herb2_herbs_exactly_mapped_to_ucec": int((herb_mapping["mapping_status"] == "accepted_exact").sum()),
        "herb2_herb_mapping_conflicts_excluded": int((herb_mapping["mapping_status"] == "conflict").sum()),
        "mapped_validation_rows": int(len(mapped)),
        "mapped_unique_herb2_disease_pairs": int(len(mapped[["Subject_id", "disease_id"]].drop_duplicates())),
        "mapped_unique_nct_ids": int(mapped["NCT_id"].nunique()),
        "mapped_unique_herb2_ids": int(mapped["Subject_id"].nunique()),
        "mapped_unique_disease_ids": int(mapped["disease_id"].nunique()),
        "nct_overlap_with_cmaup_raw": int(mapped.loc[mapped["nct_present_in_cmaup_raw"], "NCT_id"].nunique()),
        "rows_with_nct_overlap_with_cmaup_raw": int(mapped["nct_present_in_cmaup_raw"].sum()),
        "rows_with_model_herb_disease_pair_overlap": int(mapped["model_herb_disease_pair_in_cmaup"].sum()),
        "rows_with_same_nct_model_herb_disease_overlap": int(mapped["same_nct_model_herb_disease_in_cmaup"].sum()),
        "cmaup_excluded_unique_rows": int(len(unique_nct)),
        "cmaup_excluded_unique_nct_ids": int(unique_nct["NCT_id"].nunique()),
        "cmaup_excluded_unique_herb2_disease_pairs": int(len(unique_nct[["Subject_id", "disease_id"]].drop_duplicates())),
        "policy": (
            "Herb subjects only; exact herb mappings without cross-field conflicts; exact UCEC disease names "
            "plus a restricted alias dictionary; non-disease conditions excluded. The strict subset removes "
            "every NCT identifier present anywhere in the CMAUP source file."
        ),
    }
    (out_dir / "herb2_cmaup_overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
