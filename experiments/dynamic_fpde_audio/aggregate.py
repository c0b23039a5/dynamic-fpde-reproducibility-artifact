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
    "selection_runtime_sec",
    "native_fpde_runtime_sec",
    "diagnostic_runtime_sec",
    "total_runtime_sec",
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


def average_random_repetitions(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Average random-baseline repetition rows to one row per sample.

    The sample metrics CSV keeps every random repetition. Method summaries use
    this helper so random_baseline has the same effective sample count as the
    Dynamic-FPDE methods and the energy ranking baseline.
    """

    passthrough: list[dict[str, object]] = []
    random_groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("method")) != "random_baseline":
            copied = dict(row)
            copied.setdefault("aggregation_unit", "sample")
            copied.setdefault("n_underlying_rows", 1)
            copied.setdefault("random_repetitions", "")
            passthrough.append(copied)
            continue
        key = (
            str(row.get("dataset", "")),
            str(row.get("fold", "")),
            str(row.get("seed", "")),
            str(row.get("sample_id", "")),
            str(row.get("method", "")),
        )
        random_groups[key].append(row)

    averaged: list[dict[str, object]] = []
    for group_rows in random_groups.values():
        base = dict(group_rows[0])
        for metric in SUMMARY_METRICS:
            values = [value for row in group_rows if (value := _to_float(row.get(metric))) is not None]
            if values:
                base[metric] = sum(values) / len(values)
        for metric in ("T", "F"):
            values = [value for row in group_rows if (value := _to_float(row.get(metric))) is not None]
            if values:
                base[metric] = int(round(sum(values) / len(values)))
        prototype_margin = _to_float(base.get("prototype_margin"))
        selection_margin = _to_float(base.get("selection_margin"))
        if prototype_margin is not None:
            base["prototype_margin_positive"] = prototype_margin > 0.0
            base["prototype_margin_sign"] = _margin_sign(prototype_margin)
        if selection_margin is not None:
            base["selection_margin_positive"] = selection_margin > 0.0
            base["selection_margin_sign"] = _margin_sign(selection_margin)
        base["random_repetition"] = ""
        base["aggregation_unit"] = "sample"
        base["n_underlying_rows"] = len(group_rows)
        base["random_repetitions"] = len(group_rows)
        averaged.append(base)
    return sorted(
        passthrough + averaged,
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("fold", "")),
            str(row.get("sample_id", "")),
            str(row.get("method", "")),
        ),
    )


def _margin_sign(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def aggregate_by_method(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["fold"]), str(row["method"]))].append(row)
    out: list[dict[str, object]] = []
    for (dataset, fold, method), group_rows in sorted(groups.items()):
        sample_ids = {str(row.get("sample_id", idx)) for idx, row in enumerate(group_rows)}
        underlying_counts = [
            int(value)
            for row in group_rows
            if (value := _to_float(row.get("n_underlying_rows"))) is not None
        ]
        random_repetitions = [
            value
            for row in group_rows
            if (value := _to_float(row.get("random_repetitions"))) is not None
        ]
        summary: dict[str, object] = {
            "dataset": dataset,
            "fold": fold,
            "method": method,
            "n": len(group_rows),
            "n_unique_samples": len(sample_ids),
            "n_rows": sum(underlying_counts) if underlying_counts else len(group_rows),
            "random_repetitions_mean": (sum(random_repetitions) / len(random_repetitions)) if random_repetitions else "",
        }
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


def aggregate_rawfeat_samples(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate RawFeat sample metrics by method and hybrid lambda."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("method", "rawfeat_hyb")), str(row.get("lambda_hyb", "")))].append(row)
    output: list[dict[str, object]] = []
    for (method, lambda_hyb), group in sorted(groups.items()):
        exactness = [v for row in group if (v := _to_float(row.get("abs_exactness_residual"))) is not None]
        evidence = [v for row in group if (v := _to_float(row.get("absolute_evidence"))) is not None]
        output.append(
            {
                "method": method,
                "lambda_hyb": lambda_hyb,
                "n": len(group),
                "mean_exactness_residual": _stats(exactness)["mean"],
                "mean_absolute_evidence": _stats(evidence)["mean"],
                "shape_match_rate": sum(str(row.get("shape_match", "")).lower() == "true" for row in group) / len(group),
            }
        )
    return output


def aggregate_rawfeat_generation(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize regenerated-waveform RawFeat audits when generation is enabled."""
    items = list(rows)
    if not items:
        return []
    residuals = [v for row in items if (v := _to_float(row.get("generated_abs_exactness_residual"))) is not None]
    evidence = [v for row in items if (v := _to_float(row.get("generated_absolute_evidence"))) is not None]
    return [
        {
            "n": len(items),
            "mean_generated_exactness_residual": _stats(residuals)["mean"],
            "mean_generated_absolute_evidence": _stats(evidence)["mean"],
            "shape_match_rate": sum(str(row.get("shape_match", "")).lower() == "true" for row in items) / len(items),
            "audit_pass_rate": sum(str(row.get("generated_audit_passed", "")).lower() == "true" for row in items) / len(items),
        }
    ]
