from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .schema import REL_SPECS, NODE_TYPES, RelationSpec
from .data import SplitRelations


@dataclass
class GraphIndex:
    id_maps: Dict[str, Dict[str, int]]
    offsets: Dict[str, int]
    num_nodes: Dict[str, int]
    num_nodes_total: int
    global_type: List[str]


@dataclass
class GraphTensors:
    edge_index: torch.LongTensor
    edge_type: torch.LongTensor
    edge_evidence: torch.FloatTensor
    rel2id: Dict[str, int]
    id2rel: Dict[int, str]
    rel_specs_full: List[Tuple[str, str, str]]


@dataclass
class RunGraph:
    index: GraphIndex
    train: GraphTensors
    splits: SplitRelations


def build_node_index(splits: SplitRelations) -> GraphIndex:
    nodes: Dict[str, set] = {t: set() for t in NODE_TYPES}

    def add(t: str, vals: pd.Series):
        nodes[t].update(vals.astype(str).tolist())

    for spec in REL_SPECS:
        rel = spec.name
        for split, df in splits.edges.get(rel, {}).items():
            if rel == "HI":
                add("herb", df["herb"])
                add("ingredient", df["ingredient"])
            elif rel == "IP":
                add("ingredient", df["ingredient"])
                add("protein", df["protein"])
            elif rel == "PD":
                add("protein", df["protein"])
                add("disease", df["disease"])
            elif rel == "PPi":
                add("protein", df["protein"])
                add("protein", df["protein2"])
            elif rel == "PPath":
                add("protein", df["protein"])
                add("pathway", df["pathway"])
            elif rel == "PathD":
                add("pathway", df["pathway"])
                add("disease", df["disease"])

    id_maps: Dict[str, Dict[str, int]] = {}
    offsets: Dict[str, int] = {}
    num_nodes: Dict[str, int] = {}
    global_type: List[str] = []
    cursor = 0
    for t in NODE_TYPES:
        ids = sorted(nodes[t])
        id_maps[t] = {rid: i for i, rid in enumerate(ids)}
        offsets[t] = cursor
        n = len(ids)
        num_nodes[t] = n
        global_type.extend([t] * n)
        cursor += n
    return GraphIndex(id_maps=id_maps, offsets=offsets, num_nodes=num_nodes, num_nodes_total=cursor, global_type=global_type)


def _global(idx: GraphIndex, node_type: str, raw_id: str) -> int:
    return idx.offsets[node_type] + idx.id_maps[node_type][str(raw_id)]


def build_train_graph_tensors(idx: GraphIndex, splits: SplitRelations, add_self_loops: bool = True) -> GraphTensors:
    rel_specs_full: List[Tuple[str, str, str]] = []
    rel2id: Dict[str, int] = {}

    def add_rel(name: str, h: str, t: str):
        if name in rel2id:
            return
        rel2id[name] = len(rel2id)
        rel_specs_full.append((name, h, t))

    for spec in REL_SPECS:
        add_rel(spec.name, spec.head_type, spec.tail_type)
        add_rel(f"{spec.name}_rev", spec.tail_type, spec.head_type)

    if add_self_loops:
        add_rel("self", "any", "any")

    edges_src: List[int] = []
    edges_dst: List[int] = []
    edges_type: List[int] = []
    edges_ev: List[float] = []

    def add_edges(rel_name: str, h_type: str, t_type: str, df: pd.DataFrame, src_col: str, dst_col: str):
        rid = rel2id[rel_name]
        for h, t, ev in zip(df[src_col].astype(str).tolist(), df[dst_col].astype(str).tolist(), df["evidence"].astype(float).tolist()):
            edges_src.append(_global(idx, h_type, h))
            edges_dst.append(_global(idx, t_type, t))
            edges_type.append(rid)
            edges_ev.append(float(ev))

    for spec in REL_SPECS:
        df = splits.edges[spec.name]["train"]
        if spec.name == "HI":
            add_edges("HI", "herb", "ingredient", df, "herb", "ingredient")
            add_edges("HI_rev", "ingredient", "herb", df, "ingredient", "herb")
        elif spec.name == "IP":
            add_edges("IP", "ingredient", "protein", df, "ingredient", "protein")
            add_edges("IP_rev", "protein", "ingredient", df, "protein", "ingredient")
        elif spec.name == "PD":
            add_edges("PD", "protein", "disease", df, "protein", "disease")
            add_edges("PD_rev", "disease", "protein", df, "disease", "protein")
        elif spec.name == "PPi":
            add_edges("PPi", "protein", "protein", df, "protein", "protein2")
            add_edges("PPi_rev", "protein", "protein", df, "protein2", "protein")
        elif spec.name == "PPath":
            add_edges("PPath", "protein", "pathway", df, "protein", "pathway")
            add_edges("PPath_rev", "pathway", "protein", df, "pathway", "protein")
        elif spec.name == "PathD":
            add_edges("PathD", "pathway", "disease", df, "pathway", "disease")
            add_edges("PathD_rev", "disease", "pathway", df, "disease", "pathway")

    if add_self_loops:
        rid = rel2id["self"]
        for n in range(idx.num_nodes_total):
            edges_src.append(n)
            edges_dst.append(n)
            edges_type.append(rid)
            edges_ev.append(1.0)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_type = torch.tensor(edges_type, dtype=torch.long)
    edge_ev = torch.tensor(edges_ev, dtype=torch.float32)
    id2rel = {i: r for r, i in rel2id.items()}
    return GraphTensors(edge_index=edge_index, edge_type=edge_type, edge_evidence=edge_ev,
                        rel2id=rel2id, id2rel=id2rel, rel_specs_full=rel_specs_full)


def build_run_graph(splits: SplitRelations) -> RunGraph:
    idx = build_node_index(splits)
    train = build_train_graph_tensors(idx, splits, add_self_loops=True)
    return RunGraph(index=idx, train=train, splits=splits)
