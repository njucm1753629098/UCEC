#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from ucec.data import load_splits
from ucec.graph import build_run_graph
from ucec.stage2 import RetrievalConfig, PerturbConfig, UCECEvidenceScorer


def _load_stage2_configs(run_dir: Path) -> tuple[RetrievalConfig, PerturbConfig]:
    cfg_path = run_dir / "stage2_training.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    retr = cfg.get("retrieval", {})
    pert = cfg.get("perturb", {})
    return RetrievalConfig(**retr), PerturbConfig(**pert)


def _chain_to_text(nodes: list[str]) -> str:
    return " -> ".join(str(x) for x in nodes)


def _edges_to_text(edges: list[dict]) -> str:
    parts = []
    for e in edges:
        rel = e.get("rel", "")
        src = e.get("src", "")
        dst = e.get("dst", "")
        ev = float(e.get("evidence", 0.0))
        mu = float(e.get("mu", 0.0))
        sigma = float(e.get("sigma", 0.0))
        ctilde = float(e.get("ctilde", 0.0))
        parts.append(f"{rel}: {src} => {dst} | evidence={ev:.4f}, contribution={ctilde:.4f}, mu={mu:.4f}, sigma={sigma:.4f}")
    return "\n".join(parts)


def _node_names(nodes: list[str]) -> str:
    readable = []
    for item in nodes:
        text = str(item)
        if ":" in text:
            readable.append(text.split(":", 1)[1])
        else:
            readable.append(text)
    return " -> ".join(readable)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a readable all-prediction table with evidence chains.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--prediction_file", default=None)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_xlsx", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_chains_per_pair", type=int, default=30)
    ap.add_argument("--xlsx_top_chains", type=int, default=5)
    ap.add_argument("--limit_rows", type=int, default=0, help="Debug only. 0 means all rows.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_file = Path(args.prediction_file) if args.prediction_file else run_dir / "full_predictions" / "predictions_topk_per_herb.csv"
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "readable_all_predictions_with_chains.csv"
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else run_dir / "readable_all_predictions_summary.xlsx"

    if not pred_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_file}")

    predictions = pd.read_csv(pred_file, dtype={"herb": str, "disease": str})
    if args.limit_rows and args.limit_rows > 0:
        predictions = predictions.head(args.limit_rows).copy()

    splits = load_splits(str(run_dir))
    run = build_run_graph(splits)
    z = torch.load(run_dir / "rgcn_embeddings.pt", map_location="cpu")
    retr_cfg, pert_cfg = _load_stage2_configs(run_dir)
    scorer = UCECEvidenceScorer(run, z, retr_cfg=retr_cfg, pert_cfg=pert_cfg, device=args.device)
    gate_path = run_dir / "stage2_gate_state.pt"
    if gate_path.exists():
        scorer.gate.load_state_dict(torch.load(gate_path, map_location=args.device))

    rows = []
    for rec in tqdm(predictions.itertuples(index=False), total=len(predictions), desc="export readable chains"):
        herb = str(rec.herb)
        disease = str(rec.disease)
        herb_name = str(getattr(rec, "herb_name", herb))
        disease_name = str(getattr(rec, "disease_name", disease))
        result = scorer.compute_pair_evidence(herb, disease)
        chains = result.top_chains[: max(int(args.max_chains_per_pair), 0)]
        base = {
            "药物ID": herb,
            "药物名称": herb_name,
            "疾病ID": disease,
            "疾病名称": disease_name,
            "药物内排名": int(getattr(rec, "rank_within_herb", 0)),
            "最终分数S": float(getattr(rec, "S")),
            "证据链分数E": float(getattr(rec, "E")),
            "不确定性U": float(getattr(rec, "U")),
            "stage1先验S0": float(getattr(rec, "S0")),
            "校正先验S0corr": float(getattr(rec, "S0corr")),
        }
        if not chains:
            rows.append({
                **base,
                "证据链排名": 0,
                "证据链得分": 0.0,
                "证据链原始乘积分": 0.0,
                "证据链路径": "",
                "证据链路径_简写": "",
                "边级证据明细": "",
            })
            continue
        for rank, chain in enumerate(chains, start=1):
            nodes = chain.get("nodes", [])
            edges = chain.get("edges", [])
            rows.append({
                **base,
                "证据链排名": rank,
                "证据链得分": float(chain.get("score", 0.0)),
                "证据链原始乘积分": float(chain.get("pre_score", 0.0)),
                "证据链路径": _chain_to_text(nodes),
                "证据链路径_简写": _node_names(nodes),
                "边级证据明细": _edges_to_text(edges),
            })

    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_cols = [
        "药物ID", "药物名称", "疾病ID", "疾病名称", "药物内排名",
        "最终分数S", "证据链分数E", "不确定性U", "stage1先验S0", "校正先验S0corr",
    ]
    summary = predictions.rename(
        columns={
            "herb": "药物ID",
            "herb_name": "药物名称",
            "disease": "疾病ID",
            "disease_name": "疾病名称",
            "rank_within_herb": "药物内排名",
            "S": "最终分数S",
            "E": "证据链分数E",
            "U": "不确定性U",
            "S0": "stage1先验S0",
            "S0corr": "校正先验S0corr",
        }
    )[summary_cols].copy()
    xlsx_chains = table[table["证据链排名"].between(1, int(args.xlsx_top_chains))].copy()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="所有预测结果", index=False)
        xlsx_chains.to_excel(writer, sheet_name=f"Top{args.xlsx_top_chains}证据链", index=False)

    print(f"[OK] wrote CSV: {out_csv}")
    print(f"[OK] wrote XLSX: {out_xlsx}")
    print(f"[INFO] prediction rows={len(predictions)}, chain table rows={len(table)}")


if __name__ == "__main__":
    main()
