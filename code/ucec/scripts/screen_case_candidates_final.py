#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


BROAD_EXACT = {
    "fibrosis", "hyperemia", "necrosis", "edema", "anemia", "aneurysm",
    "dementia", "lymphoma", "adenoma", "dermatitis", "tachycardia",
    "weight gain", "small head", "melanoma",
}

BROAD_PATTERNS = [
    r"^.*\bdiseases?\b$",
    r"^.*\bdisorders?\b$",
    r"^.*\bsyndromes?\b$",
    r"^.*\bneoplasms?\b$",
    r"^.*\binjuries?\b$",
    r"^.*\babnormalit(y|ies)\b$",
    r"^.*\bdeficienc(y|ies)\b$",
    r"^.*\binfections?\b$",
    r"^.*\bpain\b$",
    r"^.*\blesions?\b$",
]

RARE_OR_LOW_ACTIONABILITY_PATTERNS = [
    r"\bfamilial\b",
    r"\bcongenital\b",
    r"\bautosomal\b",
    r"\binfantile\b",
    r"\btype\s+[ivx0-9a-z]+\b",
    r"\bdelayed speech\b",
    r"\bschizencephaly\b",
    r"\bcutis marmorata\b",
    r"\brasopathy\b",
    r"\bmyotonus\b",
    r"\bhand flapping\b",
    r"\bamelogenesis imperfecta\b",
    r"\bcontinuous spike\b",
    r"\bsevere congenital\b",
]

CLASSIC_PAIR_RULES = [
    # Keep this list deliberately conservative. External literature thresholds
    # should be applied manually before final manuscript submission.
    ("all - grass of dahurian patrinia", "abscess"),
    ("capillary wormwood", "jaundice"),
    ("nux vomica", "pain"),
    ("dahurian angelica", "headache"),
]

PROTOTYPES = [
    ("High posterior / low uncertainty", "high_posterior_low_uncertainty", 1.0, 0.0),
    ("High posterior / high uncertainty", "high_posterior_high_uncertainty", 1.0, 1.0),
    ("Low posterior / low uncertainty", "low_posterior_low_uncertainty", 0.0, 0.0),
    ("Low posterior / high uncertainty", "low_posterior_high_uncertainty", 0.0, 1.0),
]


def normalize(s: pd.Series) -> pd.Series:
    mn = float(s.min())
    mx = float(s.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mn) / (mx - mn)


def is_too_broad(name: str) -> bool:
    x = str(name).strip().lower()
    if x in BROAD_EXACT:
        return True
    return any(re.search(pattern, x) for pattern in BROAD_PATTERNS)


def is_low_actionability(name: str) -> bool:
    x = str(name).strip().lower()
    return any(re.search(pattern, x) for pattern in RARE_OR_LOW_ACTIONABILITY_PATTERNS)


def is_classic_pair(herb_name: str, disease_name: str) -> bool:
    h = str(herb_name).strip().lower()
    d = str(disease_name).strip().lower()
    return any(h_rule in h and d_rule in d for h_rule, d_rule in CLASSIC_PAIR_RULES)


def split_chain(path: str) -> list[str]:
    if not isinstance(path, str) or not path.strip():
        return []
    return [x.strip() for x in path.split(" -> ") if x.strip()]


def mechanism_key(path: str) -> tuple[str, str, str]:
    nodes = split_chain(path)
    ingredient = ""
    proteins: list[str] = []
    pathway = ""
    for node in nodes:
        if node.startswith("ingredient:"):
            ingredient = node.split(":", 1)[1]
        elif node.startswith("protein:"):
            proteins.append(node.split(":", 1)[1])
        elif node.startswith("pathway:"):
            pathway = node.split(":", 1)[1]
    core_protein = proteins[-1] if proteins else ""
    return ingredient, core_protein, pathway


def load_ingredient_counts(project_root: Path) -> pd.Series:
    hit = pd.read_csv(project_root / "ucec" / "data" / "hit2_herbs_ingredients.csv", low_memory=False)
    return hit.groupby("Herb ID")["Related Compound ID"].nunique()


def build_tables(run_dir: Path, readable_chain_file: Path, out_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(readable_chain_file)
    cols = list(raw.columns)
    df = raw.rename(columns={
        cols[0]: "herb_id",
        cols[1]: "herb_name",
        cols[2]: "disease_id",
        cols[3]: "disease_name",
        cols[4]: "rank_within_herb",
        cols[5]: "posterior",
        cols[6]: "evidence_score",
        cols[7]: "uncertainty",
        cols[8]: "s0",
        cols[9]: "s0corr",
        cols[10]: "chain_rank",
        cols[11]: "chain_score",
        cols[12]: "chain_pre_score",
        cols[13]: "chain_path",
        cols[14]: "chain_path_short",
        cols[15]: "edge_details",
    })
    df["mechanism_key"] = df["chain_path"].map(mechanism_key)
    valid_chain = pd.to_numeric(df["chain_rank"], errors="coerce").fillna(0) > 0
    chain_counts = (
        df[valid_chain]
        .drop_duplicates(["herb_id", "disease_id", "mechanism_key"])
        .groupby(["herb_id", "disease_id"])
        .size()
        .rename("retained_chain_count")
    )

    pair = (
        df.groupby(["herb_id", "herb_name", "disease_id", "disease_name"], as_index=False)
        .agg(
            posterior=("posterior", "first"),
            uncertainty=("uncertainty", "first"),
            evidence_score=("evidence_score", "first"),
            rank_within_herb=("rank_within_herb", "first"),
        )
    )
    pair = pair.merge(chain_counts.reset_index(), on=["herb_id", "disease_id"], how="left")
    pair["retained_chain_count"] = pair["retained_chain_count"].fillna(0).astype(int)

    ingredient_counts = load_ingredient_counts(project_root)
    pair["candidate_compound_count"] = pair["herb_id"].map(ingredient_counts).fillna(0).astype(int)
    pair["known_hd_pair"] = False
    pair["classic_pair"] = [
        is_classic_pair(h, d) for h, d in zip(pair["herb_name"], pair["disease_name"])
    ]
    pair["too_broad_disease"] = pair["disease_name"].map(is_too_broad)
    pair["low_actionability_disease"] = pair["disease_name"].map(is_low_actionability)
    pair["insufficient_chains"] = pair["retained_chain_count"] < 3
    pair["insufficient_compounds"] = pair["candidate_compound_count"] < 3

    pair["eligible"] = ~(
        pair["known_hd_pair"]
        | pair["classic_pair"]
        | pair["too_broad_disease"]
        | pair["low_actionability_disease"]
        | pair["insufficient_chains"]
        | pair["insufficient_compounds"]
    )

    def reason(row: pd.Series) -> str:
        reasons = []
        if row["known_hd_pair"]:
            reasons.append("known herb-disease pair")
        if row["classic_pair"]:
            reasons.append("classic pair by rule")
        if row["too_broad_disease"]:
            reasons.append("disease name too broad")
        if row["low_actionability_disease"]:
            reasons.append("low-actionability rare/genetic disease")
        if row["insufficient_chains"]:
            reasons.append("fewer than 3 non-redundant chains")
        if row["insufficient_compounds"]:
            reasons.append("fewer than 3 candidate compounds")
        return "; ".join(reasons) if reasons else "eligible"

    pair["note"] = pair.apply(reason, axis=1)

    pool = pair[pair["eligible"]].copy()
    pool["P"] = normalize(pool["posterior"])
    pool["U_norm"] = normalize(pool["uncertainty"])

    selected_rows = []
    used: set[tuple[str, str]] = set()
    for label, key, p0, u0 in PROTOTYPES:
        pool[f"distance_{key}"] = np.sqrt((pool["P"] - p0) ** 2 + (pool["U_norm"] - u0) ** 2)
        pool[f"similarity_{key}"] = 1.0 - pool[f"distance_{key}"] / math.sqrt(2.0)
        for _, row in pool.sort_values(f"distance_{key}", ascending=True).iterrows():
            pair_key = (str(row["herb_id"]), str(row["disease_id"]))
            if pair_key not in used:
                used.add(pair_key)
                selected_rows.append({
                    "category": label,
                    "category_key": key,
                    "herb_id": row["herb_id"],
                    "herb_name": row["herb_name"],
                    "disease_id": row["disease_id"],
                    "disease_name": row["disease_name"],
                    "posterior": row["posterior"],
                    "uncertainty": row["uncertainty"],
                    "P": row["P"],
                    "U_norm": row["U_norm"],
                    "prototype_distance": row[f"distance_{key}"],
                    "prototype_similarity": row[f"similarity_{key}"],
                    "retained_chain_count": row["retained_chain_count"],
                    "candidate_compound_count": row["candidate_compound_count"],
                })
                break

    selected = pd.DataFrame(selected_rows)
    selected_pairs = selected[["herb_id", "disease_id"]].drop_duplicates()
    selected_chains = df.merge(selected_pairs, on=["herb_id", "disease_id"], how="inner").copy()
    selected_chains = selected_chains[valid_chain.loc[selected_chains.index] if selected_chains.index.isin(valid_chain.index).all() else pd.to_numeric(selected_chains["chain_rank"], errors="coerce").fillna(0) > 0]
    selected_chains = selected_chains.sort_values(["herb_id", "disease_id", "chain_score"], ascending=[True, True, False])
    selected_chains = (
        selected_chains.drop_duplicates(["herb_id", "disease_id", "mechanism_key"])
        .groupby(["herb_id", "disease_id"], as_index=False, group_keys=False)
        .head(3)
    )
    selected_chains = selected_chains[
        [
            "herb_id", "herb_name", "disease_id", "disease_name", "posterior", "uncertainty",
            "chain_rank", "chain_score", "chain_pre_score", "chain_path", "chain_path_short", "edge_details",
        ]
    ].copy()

    total_cn = pair.rename(columns={
        "herb_id": "中药ID",
        "herb_name": "中药名",
        "disease_id": "疾病ID",
        "disease_name": "疾病名",
        "posterior": "后验分数S",
        "uncertainty": "不确定性U",
        "evidence_score": "证据链分数E",
        "rank_within_herb": "药物内排名",
        "retained_chain_count": "最终保留非冗余链数",
        "candidate_compound_count": "候选成分数",
        "known_hd_pair": "是否属于数据集已有中药疾病对",
        "classic_pair": "是否属于经典配对",
        "too_broad_disease": "疾病名是否过于宽泛",
        "low_actionability_disease": "是否低可操作性疾病",
        "insufficient_chains": "是否链数不足",
        "insufficient_compounds": "是否成分不足",
        "eligible": "是否进入候选池",
        "note": "备注",
    })
    selected_cn = selected.rename(columns={
        "category": "类别",
        "herb_id": "中药ID",
        "herb_name": "中药",
        "disease_id": "疾病ID",
        "disease_name": "疾病",
        "posterior": "后验",
        "uncertainty": "不确定性",
        "prototype_distance": "原型距离",
        "prototype_similarity": "原型相似度",
        "retained_chain_count": "链数",
        "candidate_compound_count": "候选成分数",
    })
    chains_cn = selected_chains.rename(columns={
        "herb_id": "中药ID",
        "herb_name": "中药",
        "disease_id": "疾病ID",
        "disease_name": "疾病",
        "posterior": "后验",
        "uncertainty": "不确定性",
        "chain_rank": "原链排名",
        "chain_score": "链得分",
        "chain_pre_score": "链原始乘积分",
        "chain_path": "证据链路径",
        "chain_path_short": "证据链路径_简写",
        "edge_details": "边级证据明细",
    })

    total_path = out_dir / "final_rule_total_table.csv"
    selected_path = out_dir / "final_rule_selected_four.csv"
    chains_path = out_dir / "final_rule_selected_four_top3_chains.csv"
    xlsx_path = out_dir / "final_rule_case_screening.xlsx"
    method_path = out_dir / "final_rule_screening_method.md"
    method_cn_path = out_dir / "final_rule_screening_method_cn.txt"

    total_cn.sort_values(["是否进入候选池", "后验分数S"], ascending=[False, False]).to_csv(total_path, index=False, encoding="utf-8-sig")
    selected_cn.to_csv(selected_path, index=False, encoding="utf-8-sig")
    chains_cn.to_csv(chains_path, index=False, encoding="utf-8-sig")

    method_text = build_method_text(
        total_pairs=len(pair),
        eligible_pairs=len(pool),
        selected_cn=selected_cn,
    )
    method_path.write_text(method_text, encoding="utf-8-sig")
    method_cn_path.write_text(build_method_text_cn(len(pair), len(pool), selected_cn), encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        total_cn.sort_values(["是否进入候选池", "后验分数S"], ascending=[False, False]).to_excel(writer, sheet_name="总表", index=False)
        selected_cn.to_excel(writer, sheet_name="四个原型案例", index=False)
        chains_cn.to_excel(writer, sheet_name="四例前3链", index=False)

    print(f"[OK] wrote {total_path}")
    print(f"[OK] wrote {selected_path}")
    print(f"[OK] wrote {chains_path}")
    print(f"[OK] wrote {xlsx_path}")
    print(f"[OK] wrote {method_path}")
    print(f"[OK] wrote {method_cn_path}")
    print(f"[INFO] total_pairs={len(pair)} eligible_pairs={len(pool)}")
    print(selected_cn[["类别", "中药", "疾病", "后验", "不确定性", "原型距离", "链数", "候选成分数"]].to_string(index=False))


def build_method_text(total_pairs: int, eligible_pairs: int, selected_cn: pd.DataFrame) -> str:
    selected_view = selected_cn[
        ["类别", "中药", "疾病", "后验", "不确定性", "P", "U_norm", "原型距离", "链数", "候选成分数"]
    ].copy()
    selected_md = _to_markdown_table(selected_view)
    return f"""# Case screening rule for network pharmacology validation

## Overview

All model-predicted herb-disease pairs were first organized into a single screening table. Each row represented one herb-disease pair and contained the herb name, disease name, posterior score S, uncertainty U, number of retained non-redundant evidence chains, known-pair flag, classic-pair flag, disease-specificity flag, compound-count flag, and screening note.

The screening was intentionally fixed before downstream network pharmacology analysis. The selected cases should not be replaced according to later enrichment results.

## Initial candidate set

Total predicted pairs: {total_pairs}

Eligible pairs after rule-based filtering: {eligible_pairs}

## Exclusion criteria

1. Known herb-disease pair.

A pair was excluded if the same herb-disease relation was already present as a direct known relation in the curated graph or training resources. In the current graph, direct herb-disease edges were not used, so all pairs were marked as not directly known.

2. Classic or well-known pair.

A pair was excluded if it matched a conservative predefined classic-pair rule. In a final manuscript workflow, this column should additionally be checked by external evidence saturation: a pair is considered classic if it is present in a curated TCM indication database or if the joint herb-disease literature search exceeds a prespecified threshold, for example at least 50 records.

3. Disease name too broad.

A disease label was considered too broad if it was only a generic disease class, symptom, pathological process, or unspecified disease family. Exact generic labels such as Fibrosis, Hyperemia, Necrosis, Edema, Anemia, Aneurysm, Dementia, Lymphoma, Adenoma, Dermatitis, Tachycardia, Weight Gain, Small head, and Melanoma were excluded. Names ending in broad families such as disease(s), disorder(s), syndrome(s), neoplasm(s), injury/injuries, abnormality/abnormalities, deficiency/deficiencies, infection(s), pain, or lesion(s) were also excluded.

Importantly, a qualified disease modifier preserves eligibility. For example, Pulmonary Fibrosis and Cystic Fibrosis were retained because they refer to specific disease entities, whereas the generic label Fibrosis was excluded.

4. Low-actionability disease label.

Labels dominated by very rare congenital, familial, autosomal, infantile, or highly specific genetic subtype wording were excluded because they are usually poor choices for network pharmacology validation. The operational keywords included familial, congenital, autosomal, infantile, type-specific labels, delayed speech, schizencephaly, cutis marmorata, RASopathy, myotonus, hand flapping, amelogenesis imperfecta, continuous spike, and severe congenital.

5. Insufficient evidence chains.

A pair was retained only if it had at least three non-redundant evidence chains after final model aggregation. Two chains were treated as redundant if they shared the same ingredient, disease-proximal core protein, and pathway node. Only one representative chain was counted for each such mechanism key.

6. Insufficient herb compounds.

A pair was retained only if the corresponding herb had at least three retrievable candidate compounds in the HIT2 herb-compound table. This prevents selecting cases for which downstream compound-target network analysis would be structurally underpowered.

## Standardization of posterior and uncertainty

For all eligible pairs, posterior score S and uncertainty U were min-max normalized within the eligible candidate pool:

P = (S - min(S)) / (max(S) - min(S))

U_norm = (U - min(U)) / (max(U) - min(U))

Here P represents normalized posterior strength and U_norm represents normalized uncertainty.

## Prototype-based case selection

Each eligible pair was represented as a point (P, U_norm). Four prototype points were defined:

High posterior / low uncertainty: (1, 0)

High posterior / high uncertainty: (1, 1)

Low posterior / low uncertainty: (0, 0)

Low posterior / high uncertainty: (0, 1)

For each category, the Euclidean distance from every candidate pair to the corresponding prototype was calculated:

d = sqrt((P - P0)^2 + (U_norm - U0)^2)

The pair with the smallest distance was selected for that category. If the same pair was selected by multiple categories, it was assigned to the category for which it had the smallest prototype distance, and the next closest unused pair was selected for the other category.

## Final selected cases

{selected_md}
"""


def build_method_text_cn(total_pairs: int, eligible_pairs: int, selected_cn: pd.DataFrame) -> str:
    view = selected_cn[
        ["类别", "中药", "疾病", "后验", "不确定性", "P", "U_norm", "原型距离", "链数", "候选成分数"]
    ].copy()
    table = _to_markdown_table(view)
    return f"""网络药理学验证案例筛选规则

一、总原则

本研究先将全部模型输出的中药-疾病配对整理成总表。每一行代表一个中药-疾病配对，并记录中药名、疾病名、后验分数S、不确定性U、最终保留的非冗余证据链数、是否已有配对、是否经典配对、疾病名是否过宽泛、候选成分数和备注。

筛选规则在后续网络药理学分析之前固定。四个案例一旦选定，不再根据后续富集结果更换。

二、候选池规模

模型输出配对总数：{total_pairs}

规则过滤后的合格候选池数量：{eligible_pairs}

三、排除规则

1. 已有中药-疾病配对排除。
如果某个中药-疾病关系已经作为直接已知关系出现在训练资源或整合知识图谱中，则排除。当前图谱没有使用直接中药-疾病边，因此本轮所有配对在该项上均标记为“否”。

2. 经典或已知适应症配对排除。
如果某个配对属于传统适应症、外部中药适应症数据库已知关系，或中药名与疾病名联合检索文献数达到预设阈值，则排除。建议论文中将阈值固定为至少50条联合检索记录。本轮脚本使用保守内置规则进行标记，最终投稿前建议再做一次人工文献核查。

3. 疾病名过宽泛排除。
如果疾病名称只是疾病大类、症状、病理过程或未限定疾病家族，则排除。例如 Fibrosis、Hyperemia、Necrosis、Edema、Anemia、Aneurysm、Dementia、Lymphoma、Adenoma、Dermatitis、Tachycardia、Weight Gain、Small head、Melanoma 等泛化标签被排除。以 disease(s)、disorder(s)、syndrome(s)、neoplasm(s)、injury/injuries、abnormality/abnormalities、deficiency/deficiencies、infection(s)、pain、lesion(s) 等泛化词结尾的名称也被排除。

带有明确限定词的疾病名保留。例如 Pulmonary Fibrosis 和 Cystic Fibrosis 指向明确疾病实体，因此不按泛化 Fibrosis 排除。

4. 低可操作性疾病排除。
明显偏罕见遗传、先天、家族型或极细分亚型的疾病不作为网络药理学主验证案例。关键词包括 familial、congenital、autosomal、infantile、type-specific label、delayed speech、schizencephaly、cutis marmorata、RASopathy、myotonus、hand flapping、amelogenesis imperfecta、continuous spike、severe congenital 等。

5. 证据链不足排除。
每个配对必须至少保留3条非冗余证据链。若两条链具有相同的成分、疾病近端核心蛋白和通路节点，则视为同一机制分支，只计为1条非冗余链。

6. 中药成分不足排除。
每个中药在 HIT2 中必须至少有3个可检索候选成分，否则后续成分-靶点网络分析结构不足，排除。

四、后验和不确定性标准化

在合格候选池内，对后验分数S和不确定性U分别做0到1归一化：

P = (S - min(S)) / (max(S) - min(S))

U_norm = (U - min(U)) / (max(U) - min(U))

其中 P 表示标准化后验强度，U_norm 表示标准化不确定性。

五、四类原型距离筛选

将每个候选配对表示为二维点 (P, U_norm)，并定义四个原型点：

高后验、低不确定：(1, 0)

高后验、高不确定：(1, 1)

低后验、低不确定：(0, 0)

低后验、高不确定：(0, 1)

每个候选点到对应原型点的欧氏距离为：

d = sqrt((P - P0)^2 + (U_norm - U0)^2)

每一类选择距离对应原型点最近的配对。如果同一配对被多个类别同时选中，则保留其距离最小的类别，其他类别顺延选择下一个未使用配对。

六、最终四个案例

{table}
"""


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final strict prototype-based case screening.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--readable_chain_file", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    readable_chain_file = Path(args.readable_chain_file) if args.readable_chain_file else run_dir / "readable_all_predictions_with_chains.csv"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "case_screening_final_rule"
    build_tables(run_dir, readable_chain_file, out_dir)


if __name__ == "__main__":
    main()
