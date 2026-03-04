from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn


@dataclass
class KGEConfig:
    dim: int = 128
    score_fn: Literal["transe", "distmult", "complex"] = "transe"
    transe_p: int = 1  # 1=L1, 2=L2


class KGEModel(nn.Module):

    def __init__(self, num_nodes_total: int, num_relations: int, cfg: KGEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.ent = nn.Embedding(num_nodes_total, d)
        nn.init.normal_(self.ent.weight, mean=0.0, std=0.01)

        if cfg.score_fn == "complex":
            self.ent_im = nn.Embedding(num_nodes_total, d)
            nn.init.normal_(self.ent_im.weight, mean=0.0, std=0.01)
            self.rel_re = nn.Embedding(num_relations, d)
            self.rel_im = nn.Embedding(num_relations, d)
            nn.init.normal_(self.rel_re.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.rel_im.weight, mean=0.0, std=0.01)
        else:
            self.rel = nn.Embedding(num_relations, d)
            nn.init.normal_(self.rel.weight, mean=0.0, std=0.01)

    def score_logits(self, rel_ids: torch.LongTensor, head: torch.LongTensor, tail: torch.LongTensor) -> torch.Tensor:
        d = self.cfg.dim
        h = self.ent(head)
        t = self.ent(tail)
        if self.cfg.score_fn == "transe":
            r = self.rel(rel_ids)
            x = h + r - t
            if self.cfg.transe_p == 2:
                dist = torch.norm(x, p=2, dim=-1)
            else:
                dist = torch.norm(x, p=1, dim=-1)
            return -dist
        elif self.cfg.score_fn == "distmult":
            r = self.rel(rel_ids)
            return torch.sum(h * r * t, dim=-1)
        elif self.cfg.score_fn == "complex":
            hr, hi = h, self.ent_im(head)
            tr, ti = t, self.ent_im(tail)
            rr, ri = self.rel_re(rel_ids), self.rel_im(rel_ids)
            return torch.sum(
                hr * rr * tr
                + hi * rr * ti
                + hr * ri * ti
                - hi * ri * tr,
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown score_fn: {self.cfg.score_fn}")
