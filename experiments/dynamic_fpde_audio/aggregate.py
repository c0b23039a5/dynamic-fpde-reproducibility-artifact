"""CSV aggregation helpers for Dynamic-FPDE audio experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


SUMMARY_METRICS = (
    "evidence",
    "abs_exactness_residual",
    "deletion_drop_auc",
    "insertion_gain_auc",
    "combined_score",
    "runtime_sec",
)


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {"mean": "", "std": "", "median": "", "n": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": variance**0.5, "median": median(values), "n": len(values)}


def aggregate_by_method(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["fold"]), str(row["method"]))].append(row)
    out: list[dict[str, object]] = []
    for (dataset, fold, method), group_rows in sorted(groups.items()):
        summary: dict[str, object] = {"dataset": dataset, "fold": fold, "method": method, "n": len(group_rows)}
        for metric in SUMMARY_METRICS:
            values = [value for row in group_rows if (value := _to_float(row.get(metric))) is not None]
            stats = _stats(values)
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
            summary[f"{metric}_median"] = stats["median"]
            summary[f"{metric}_n"] = stats["n"]
        out.append(summary)
    return out


def aggregate_additivity(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", ""))
        if not method.startswith("dynamic_"):
            continue
        value = _to_float(row.get("abs_exactness_residual"))
        if value is not None:
            groups[(str(row["dataset"]), str(row["fold"]), method)].append(value)
    out: list[dict[str, object]] = []
    for (dataset, fold, method), values in sorted(groups.items()):
        stats = _stats(values)
        out.append(
            {
                "dataset": dataset,
                "fold": fold,
                "method": method,
                "abs_exactness_residual_mean": stats["mean"],
                "abs_exactness_residual_std": stats["std"],
                "abs_exactness_residual_median": stats["median"],
                "n": stats["n"],
            }
        )
    return out


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

