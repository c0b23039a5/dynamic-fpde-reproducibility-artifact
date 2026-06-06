from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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


def sign_calibration_arrays(summary, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return sign confidence, correctness, predicted signs, and ignored neutral count.

    Synthetic true attributions can be exactly zero for non-informative features.
    Those true-zero features are treated as neutral and excluded from sign
    calibration because neither positive nor negative sign is correct.
    """
    truth = np.asarray(truth, dtype=float)
    p_pos = summary["p_positive"].to_numpy(dtype=float)
    p_neg = summary["p_negative"].to_numpy(dtype=float)
    confidence = np.maximum(p_pos, p_neg)
    predicted_sign = np.where(p_pos >= p_neg, 1.0, -1.0)
    true_sign = np.sign(truth)
    mask = true_sign != 0
    ignored = int(np.size(true_sign) - np.count_nonzero(mask))
    correct = (predicted_sign[mask] == true_sign[mask]).astype(float)
    return confidence[mask], correct, predicted_sign[mask], ignored


def sign_reliability_bins(summary, truth: np.ndarray, *, n_bins: int = 10, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    confidence, correct, _, ignored = sign_calibration_arrays(summary, truth)
    metadata = dict(metadata or {})
    rows = []
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = int(confidence.size)
    for bin_id in range(int(n_bins)):
        lo = float(edges[bin_id])
        hi = float(edges[bin_id + 1])
        if bin_id == int(n_bins) - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n = int(np.count_nonzero(mask))
        mean_conf = float(np.mean(confidence[mask])) if n else float("nan")
        acc = float(np.mean(correct[mask])) if n else float("nan")
        rows.append(
            {
                **metadata,
                "bin_id": int(bin_id),
                "bin_lower": lo,
                "bin_upper": hi,
                "bin_feature_count": n,
                "bin_weight": float(n / total) if total else 0.0,
                "mean_confidence": mean_conf,
                "sign_accuracy": acc,
                "abs_calibration_error": float(abs(acc - mean_conf)) if n else float("nan"),
                "n_neutral_ignored": ignored,
            }
        )
    return pd.DataFrame(rows)


def sign_calibration_metrics(summary, truth: np.ndarray, *, n_bins: int = 10) -> Dict[str, float]:
    confidence, correct, _, ignored = sign_calibration_arrays(summary, truth)
    if confidence.size == 0:
        return {
            "sign_brier_score": float("nan"),
            "sign_ece": float("nan"),
            "sign_accuracy_at_confidence_0_8": float("nan"),
            "sign_accuracy_at_confidence_0_9": float("nan"),
            "n_sign_calibration_features": 0,
            "n_neutral_sign_features_ignored": ignored,
        }
    bins = sign_reliability_bins(summary, truth, n_bins=n_bins)
    ece = float(np.nansum(bins["bin_weight"].to_numpy(dtype=float) * bins["abs_calibration_error"].to_numpy(dtype=float)))
    out: Dict[str, float] = {
        "sign_brier_score": float(np.mean((confidence - correct) ** 2)),
        "sign_ece": ece,
        "sign_accuracy_at_confidence_0_8": float(np.mean(correct[confidence >= 0.8])) if np.any(confidence >= 0.8) else float("nan"),
        "sign_accuracy_at_confidence_0_9": float(np.mean(correct[confidence >= 0.9])) if np.any(confidence >= 0.9) else float("nan"),
        "n_sign_calibration_features": int(confidence.size),
        "n_neutral_sign_features_ignored": ignored,
    }
    return out


def calibration_metrics(summary, truth: np.ndarray, *, top_k: int = 5, sign_bins: int = 10) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    mean = summary["posterior_mean"].to_numpy(dtype=float)
    lower = summary["ci_lower_95"].to_numpy(dtype=float)
    upper = summary["ci_upper_95"].to_numpy(dtype=float)
    covered = (truth >= lower) & (truth <= upper)
    sign_metrics = sign_calibration_metrics(summary, truth, n_bins=sign_bins)
    return {
        "coverage_95": float(np.mean(covered)),
        "mean_ci_width": float(np.mean(upper - lower)),
        "median_ci_width": float(np.median(upper - lower)),
        "sign_accuracy": float(np.mean(np.sign(mean[truth != 0]) == np.sign(truth[truth != 0]))) if np.any(truth != 0) else float("nan"),
        "top_k_precision": top_k_precision(mean, truth, min(top_k, truth.size)),
        "spearman_rank_correlation": spearman_corr(np.abs(mean), np.abs(truth)),
        "kendall_tau": kendall_tau(np.abs(mean), np.abs(truth)),
        **sign_metrics,
    }


def replacement_values(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    *,
    target_label: Optional[int] = None,
    strategy: str = "mean",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    if strategy == "mean":
        return np.mean(X_train, axis=0)
    if strategy == "median":
        return np.median(X_train, axis=0)
    if strategy == "class_conditional_mean":
        if y_train is None or target_label is None:
            raise ValueError("class_conditional_mean requires y_train and target_label")
        y_arr = np.asarray(y_train)
        mask = y_arr == target_label
        if not np.any(mask):
            raise ValueError(f"class_conditional_mean unavailable for target_label={target_label!r}")
        return np.mean(X_train[mask], axis=0)
    if strategy == "marginal_sampling":
        if rng is None:
            raise ValueError("marginal_sampling requires rng")
        idx = rng.integers(0, X_train.shape[0], size=X_train.shape[1])
        return X_train[idx, np.arange(X_train.shape[1])]
    if strategy == "permutation":
        if rng is None:
            raise ValueError("permutation requires rng")
        values = np.mean(X_train, axis=0).copy()
        for j in range(X_train.shape[1]):
            values[j] = rng.choice(X_train[:, j])
        return values
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
    deltas = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        replaced = x.copy()
        replaced[j] = baseline[j]
        p_replaced = float(model.predict_proba(replaced.reshape(1, -1))[0][target_idx])
        deltas[j] = float(proba0[target_idx] - p_replaced)
    deletion_auc = float(np.trapezoid(deletion, xs))
    insertion_auc = float(np.trapezoid(insertion, xs))
    return {
        "p0": float(proba0[target_idx]),
        "deletion_auc": deletion_auc,
        "deletion_drop_auc": float(proba0[target_idx] - deletion_auc),
        "insertion_auc": insertion_auc,
        "comprehensiveness": float(proba0[target_idx] - deletion[-1]),
        "sufficiency": float(proba0[target_idx] - insertion[-1]),
        "faithfulness_correlation": spearman_corr(np.abs(attribution), np.abs(deltas)),
        "faithfulness_delta_mean": float(np.mean(deltas)),
        "faithfulness_delta_abs_mean": float(np.mean(np.abs(deltas))),
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
