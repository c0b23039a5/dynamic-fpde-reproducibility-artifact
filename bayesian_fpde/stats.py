from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(float(x > y) for x in a for y in b)
    lt = sum(float(x < y) for x in a for y in b)
    return float((gt - lt) / (a.size * b.size))


def method_tests(df: pd.DataFrame, *, metric: str = "combined_score") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    effects = []
    if df.empty or metric not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    keys = [c for c in ["dataset_name", "task_id", "seed", "fold"] if c in df.columns]
    pivot = df.pivot_table(index=keys or None, columns="method", values=metric, aggfunc="mean")
    p_values = []
    pairs = []
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception:
        friedmanchisquare = None
        wilcoxon = None
    methods = list(pivot.columns)
    if len(methods) >= 3 and friedmanchisquare is not None:
        aligned = pivot.dropna()
        if not aligned.empty:
            stat, p_value = friedmanchisquare(*[aligned[m].to_numpy(dtype=float) for m in methods])
            rows.append({"test": "friedman", "metric": metric, "method_a": "all", "method_b": "", "statistic": float(stat), "p_value": float(p_value)})
    for a, b in combinations(methods, 2):
        aligned = pivot[[a, b]].dropna()
        if aligned.empty:
            continue
        if wilcoxon is not None:
            try:
                stat, p_value = wilcoxon(aligned[a], aligned[b], zero_method="wilcox")
            except ValueError:
                stat, p_value = float("nan"), 1.0
        else:
            stat, p_value = float("nan"), float("nan")
        pairs.append((a, b, stat, p_value))
        p_values.append(p_value)
        effects.append({"metric": metric, "method_a": a, "method_b": b, "cliffs_delta": cliffs_delta(aligned[a], aligned[b])})
    adjusted = holm_adjust(p_values) if p_values else []
    for (a, b, stat, p_value), p_adj in zip(pairs, adjusted):
        rows.append({"test": "wilcoxon", "metric": metric, "method_a": a, "method_b": b, "statistic": float(stat), "p_value": float(p_value), "p_holm": float(p_adj)})
    return pd.DataFrame(rows), pd.DataFrame(effects)


def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    *,
    metric: str = "combined_score",
    n_bootstrap: int = 500,
    seed: int = 0,
    unit_level: str = "dataset_seed",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    if unit_level == "dataset_seed":
        unit_cols = [col for col in ["dataset_name", "task_id", "seed", "fold", "method"] if col in df.columns]
        if "method" not in unit_cols:
            unit_cols.append("method")
        unit_df = df.groupby(unit_cols, dropna=False).agg(
            unit_metric=(metric, "mean"),
            n_instance_rows=(metric, "size"),
        ).reset_index()
    elif unit_level == "instance":
        unit_df = df.copy()
        unit_df["unit_metric"] = pd.to_numeric(unit_df[metric], errors="coerce")
        unit_df["n_instance_rows"] = 1
    else:
        raise ValueError("unit_level must be 'dataset_seed' or 'instance'")
    for method, sub in unit_df.groupby("method"):
        vals = sub["unit_metric"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        means = []
        medians = []
        for _ in range(n_bootstrap):
            sample = rng.choice(vals, size=vals.size, replace=True)
            means.append(np.mean(sample))
            medians.append(np.median(sample))
        rows.append(
            {
                "method": method,
                "metric": metric,
                "unit_level": unit_level,
                "n_units": int(vals.size),
                "n_instance_rows": int(sub["n_instance_rows"].sum()),
                "mean": float(np.mean(vals)),
                "mean_ci_lower_95": float(np.quantile(means, 0.025)),
                "mean_ci_upper_95": float(np.quantile(means, 0.975)),
                "median": float(np.median(vals)),
                "median_ci_lower_95": float(np.quantile(medians, 0.025)),
                "median_ci_upper_95": float(np.quantile(medians, 0.975)),
            }
        )
    return pd.DataFrame(rows)
