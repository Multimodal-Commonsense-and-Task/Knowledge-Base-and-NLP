"""Binary classification metrics.

The paper reports AUC, accuracy, sensitivity and specificity, and picks the operating
threshold with Youden's index, which weights sensitivity and specificity equally.
Confidence intervals are bootstrap percentiles.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def youden_threshold(y_true, y_score) -> float:
    """The threshold maximizing (sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))])


def binary_metrics(y_true, y_score, threshold: float | None = None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else float("nan")
    if threshold is None:
        threshold = youden_threshold(y_true, y_score) if not np.isnan(auc) else 0.5

    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"auc": auc, "accuracy": (tp + tn) / len(y_true),
            "sensitivity": sens, "specificity": spec,
            "youden_index": (sens + spec - 1) if not np.isnan(sens + spec) else float("nan"),
            "threshold": threshold,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "n": len(y_true)}


def bootstrap_ci(y_true, y_score, metric: str = "auc", n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI, matching the 95% CIs reported in the paper."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx].tolist())) < 2:
            continue
        vals.append(binary_metrics(y_true[idx], y_score[idx])[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def summarize(y_true, y_score, with_ci: bool = True, seed: int = 42) -> dict:
    out = binary_metrics(y_true, y_score)
    if with_ci:
        for m in ("auc", "accuracy"):
            lo, hi = bootstrap_ci(y_true, y_score, metric=m, seed=seed)
            out[f"{m}_ci95"] = [lo, hi]
    return out
