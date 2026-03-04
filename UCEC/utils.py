from __future__ import annotations

import random
from typing import Dict, Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


def auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out: Dict[str, float] = {}
    out["auroc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    out["auprc"] = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(m):
            continue
        acc = float(labels[m].mean())
        conf = float(probs[m].mean())
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def hits_mrr(rankings: List[int], ks: Iterable[int] = (10, 20, 50)) -> Dict[str, float]:
    rankings = [int(r) for r in rankings if r > 0]
    if not rankings:
        return {f"hits@{k}": 0.0 for k in ks} | {"mrr": 0.0}
    mrr = float(np.mean([1.0 / r for r in rankings]))
    out: Dict[str, float] = {"mrr": mrr}
    for k in ks:
        out[f"hits@{k}"] = float(np.mean([1.0 if r <= k else 0.0 for r in rankings]))
    return out
