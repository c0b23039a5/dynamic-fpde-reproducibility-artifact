from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i, j in combinations(range(n), 2):
        prod = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        if prod > 0:
            concordant += 1
        elif prod < 0:
            discordant += 1
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom) if denom else float("nan")


def top_k_precision(score: np.ndarray, truth: np.ndarray, k: int) -> float:
    score_top = set(np.argsort(-np.abs(score))[:k].tolist())
    truth_top = set(np.argsort(-np.abs(truth))[:k].tolist())
    return float(len(score_top & truth_top) / max(1, min(k, len(truth_top))))


def top_k_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta = set(np.argsort(-np.abs(a))[:k].tolist())
    tb = set(np.argsort(-np.abs(b))[:k].tolist())
    union = ta | tb
    return float(len(ta & tb) / len(union)) if union else float("nan")


def calibration_metrics(summary, truth: np.ndarray, *, top_k: int = 5) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    mean = summary["posterior_mean"].to_numpy(dtype=float)
    lower = summary["ci_lower_95"].to_numpy(dtype=float)
    upper = summary["ci_upper_95"].to_numpy(dtype=float)
    covered = (truth >= lower) & (truth <= upper)
    return {
        "coverage_95": float(np.mean(covered)),
        "mean_ci_width": float(np.mean(upper - lower)),
        "median_ci_width": float(np.median(upper - lower)),
        "sign_accuracy": float(np.mean(np.sign(mean) == np.sign(truth))),
        "top_k_precision": top_k_precision(mean, truth, min(top_k, truth.size)),
        "spearman_rank_correlation": spearman_corr(np.abs(mean), np.abs(truth)),
        "kendall_tau": kendall_tau(np.abs(mean), np.abs(truth)),
        "posterior_sign_calibration": float(np.mean(np.maximum(summary["p_positive"], summary["p_negative"]))),
    }


def replacement_values(X_train: np.ndarray, y_train: Optional[np.ndarray] = None, strategy: str = "mean") -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    if strategy in {"mean", "class_conditional_mean"}:
        return np.mean(X_train, axis=0)
    if strategy == "median":
        return np.median(X_train, axis=0)
    raise ValueError(f"unsupported replacement strategy: {strategy}")


def deletion_insertion_metrics(
    model: Any,
    x: np.ndarray,
    attribution: np.ndarray,
    baseline: np.ndarray,
    *,
    target_label: Optional[int] = None,
    fractions: Sequence[float] = (0.0, 0.1, 0.2, 0.5, 1.0),
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    attribution = np.asarray(attribution, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    proba0 = np.asarray(model.predict_proba(x.reshape(1, -1))[0], dtype=float)
    classes = np.asarray(getattr(model, "classes_", np.arange(proba0.size)), dtype=int)
    if target_label is None:
        target_label = int(classes[int(np.argmax(proba0))])
    target_idx = int(np.where(classes == int(target_label))[0][0])
    order = np.argsort(-np.abs(attribution))
    xs = []
    deletion = []
    insertion = []
    n_features = x.size
    for frac in fractions:
        k = int(round(float(frac) * n_features))
        idx = order[:k]
        deleted = x.copy()
        deleted[idx] = baseline[idx]
        inserted = baseline.copy()
        inserted[idx] = x[idx]
        xs.append(float(frac))
        deletion.append(float(model.predict_proba(deleted.reshape(1, -1))[0][target_idx]))
        insertion.append(float(model.predict_proba(inserted.reshape(1, -1))[0][target_idx]))
    deletion_auc = float(np.trapezoid(deletion, xs))
    insertion_auc = float(np.trapezoid(insertion, xs))
    return {
        "p0": float(proba0[target_idx]),
        "deletion_auc": deletion_auc,
        "deletion_drop_auc": float(proba0[target_idx] - deletion_auc),
        "insertion_auc": insertion_auc,
        "comprehensiveness": float(proba0[target_idx] - deletion[-1]),
        "sufficiency": float(proba0[target_idx] - insertion[-1]),
        "faithfulness_correlation": spearman_corr(np.abs(attribution), np.abs(x - baseline)),
    }


def stability_metrics(samples: np.ndarray, *, top_k: int = 5) -> Dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        return {
            "mean_spearman_between_runs": float("nan"),
            "mean_kendall_between_runs": float("nan"),
            "top_k_jaccard_between_runs": float("nan"),
            "normalized_std_importance": float("nan"),
            "rank_entropy": float("nan"),
        }
    spear = []
    kendall = []
    jacc = []
    for i, j in combinations(range(samples.shape[0]), 2):
        spear.append(spearman_corr(np.abs(samples[i]), np.abs(samples[j])))
        kendall.append(kendall_tau(np.abs(samples[i]), np.abs(samples[j])))
        jacc.append(top_k_jaccard(samples[i], samples[j], min(top_k, samples.shape[1])))
    abs_samples = np.abs(samples)
    mean_abs = np.mean(abs_samples, axis=0)
    norm_std = np.mean(np.std(abs_samples, axis=0) / np.maximum(mean_abs, 1e-12))
    top = np.argmax(abs_samples, axis=1)
    probs = np.bincount(top, minlength=samples.shape[1]) / samples.shape[0]
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return {
        "mean_spearman_between_runs": float(np.nanmean(spear)),
        "mean_kendall_between_runs": float(np.nanmean(kendall)),
        "top_k_jaccard_between_runs": float(np.nanmean(jacc)),
        "normalized_std_importance": float(norm_std),
        "rank_entropy": entropy,
    }
