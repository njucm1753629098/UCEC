from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


@dataclass
class GCNConfig:
    dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class GCNLinkPredictor(nn.Module):

    def __init__(self, num_nodes_total: int, num_relations: int, cfg: GCNConfig):
        super().__init__()
        self.cfg = cfg
        self.x0 = nn.Embedding(num_nodes_total, cfg.dim)
        nn.init.normal_(self.x0.weight, mean=0.0, std=0.01)

        self.convs = nn.ModuleList()
        in_dim = cfg.dim
        for _ in range(cfg.num_layers):
            self.convs.append(GCNConv(in_dim, cfg.dim, add_self_loops=False, normalize=True))
            in_dim = cfg.dim

        self.dropout = float(cfg.dropout)
        self.rel_W = nn.Parameter(torch.empty(num_relations, cfg.dim, cfg.dim))
        nn.init.xavier_uniform_(self.rel_W)

    def encode(self, edge_index: torch.LongTensor) -> torch.Tensor:
        x = self.x0.weight
        for li, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if li < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def score_logits_all_rel(self, z: torch.Tensor, rel_ids: torch.LongTensor, head: torch.LongTensor, tail: torch.LongTensor) -> torch.Tensor:
        zh = z[head]
        zt = z[tail]
        W = self.rel_W[rel_ids]
        tmp = torch.bmm(zh.unsqueeze(1), W).squeeze(1)
        return torch.sum(tmp * zt, dim=-1)
