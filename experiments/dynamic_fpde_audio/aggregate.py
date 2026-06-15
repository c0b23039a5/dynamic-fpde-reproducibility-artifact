"""CSV aggregation helpers for Dynamic-FPDE audio experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


SUMMARY_METRICS = (
    "evidence",
    "prototype_margin",
    "selection_margin",
    "evaluation_evidence",
    "evaluation_margin",
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
        margins = [value for row in group_rows if (value := _to_float(row.get("prototype_margin"))) is not None]
        positive = [value for value in margins if value > 0.0]
        negative = [value for value in margins if value < 0.0]
        selection_margins = [value for row in group_rows if (value := _to_float(row.get("selection_margin"))) is not None]
        selection_positive = [value for value in selection_margins if value > 0.0]
        selection_negative = [value for value in selection_margins if value < 0.0]
        summary["prototype_margin_mean"] = _stats(margins)["mean"]
        summary["prototype_margin_median"] = _stats(margins)["median"]
        summary["prototype_margin_positive_rate"] = (len(positive) / len(margins)) if margins else ""
        summary["n_positive_margin"] = len(positive)
        summary["n_negative_margin"] = len(negative)
        summary["selection_margin_mean"] = _stats(selection_margins)["mean"]
        summary["selection_margin_median"] = _stats(selection_margins)["median"]
        summary["selection_margin_positive_rate"] = (len(selection_positive) / len(selection_margins)) if selection_margins else ""
        summary["n_selection_positive_margin"] = len(selection_positive)
        summary["n_selection_negative_margin"] = len(selection_negative)
        out.append(summary)
    return out


def selection_positive_margin_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows with positive common Dynamic-Diff selection margin."""

    return [row for row in rows if (value := _to_float(row.get("selection_margin"))) is not None and value > 0.0]


def method_positive_margin_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows with positive method-specific prototype margin."""

    return [row for row in rows if (value := _to_float(row.get("prototype_margin"))) is not None and value > 0.0]


def positive_margin_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows with positive common selection margin.

    This compatibility alias intentionally uses ``selection_margin`` so method
    summaries compare the same sample set across Dynamic-FPDE variants and
    ranking baselines.
    """

    return selection_positive_margin_rows(rows)


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
