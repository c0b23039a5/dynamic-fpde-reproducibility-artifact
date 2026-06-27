"""Generate IEEE Access RawFeat tables and figures from full-run sample metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_INPUT = Path("outputs/rawfeat_dynamic_fpde_esc50_full/results/rawfeat_sample_metrics.csv")
REQUIRED_COLUMNS = {
    "fold",
    "evidence",
    "absolute_evidence",
    "abs_exactness_residual",
    "audit_passed",
    "shape_match",
    "raw_group_attribution",
    "feature_group_attribution",
    "dt_group_attribution",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"RawFeat sample metrics not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns in {path}: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"RawFeat sample metrics are empty: {path}")
    return rows


def _values(rows: Iterable[dict[str, str]], column: str) -> np.ndarray:
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} must contain finite numeric values")
    return values


def _rate(rows: list[dict[str, str]], column: str) -> float:
    return float(np.mean([str(row[column]).strip().lower() == "true" for row in rows]))


def _write_table(path: Path, header: tuple[str, str], rows: Iterable[tuple[str, str]]) -> None:
    body = [
        r"\begin{tabular}{lr}",
        r"\hline",
        f"{header[0]} & {header[1]} \\\\",
        r"\hline",
    ]
    body.extend(f"{label} & {value} \\\\" for label, value in rows)
    body.extend([r"\hline", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def generate(input_csv: Path, tables_dir: Path, figures_dir: Path) -> list[Path]:
    rows = _read_rows(input_csv)
    evidence = _values(rows, "evidence")
    absolute_evidence = _values(rows, "absolute_evidence")
    residual = _values(rows, "abs_exactness_residual")
    group_columns = (
        ("Raw frames", "raw_group_attribution"),
        ("Acoustic features", "feature_group_attribution"),
        ("Time delta", "dt_group_attribution"),
    )
    group_means = [(label, float(np.mean(_values(rows, column)))) for label, column in group_columns]

    audit_path = tables_dir / "rawfeat_audit_summary.tex"
    _write_table(
        audit_path,
        ("Audit quantity", "Value"),
        (
            ("Samples", str(len(rows))),
            ("Shape-match rate", f"{_rate(rows, 'shape_match'):.4f}"),
            ("Exactness-audit pass rate", f"{_rate(rows, 'audit_passed'):.4f}"),
            ("Maximum absolute residual", f"{float(np.max(residual)):.3e}"),
        ),
    )

    positive = int(np.sum(evidence > 0.0))
    negative = int(np.sum(evidence < 0.0))
    zero = int(np.sum(evidence == 0.0))
    sign_path = tables_dir / "rawfeat_evidence_sign.tex"
    _write_table(
        sign_path,
        ("Evidence sign", "Count (rate)"),
        (
            ("Positive", f"{positive} ({positive / len(rows):.4f})"),
            ("Negative", f"{negative} ({negative / len(rows):.4f})"),
            ("Zero", f"{zero} ({zero / len(rows):.4f})"),
        ),
    )

    group_path = tables_dir / "rawfeat_group_attribution.tex"
    _write_table(
        group_path,
        ("Representation group", "Mean attribution"),
        ((label, f"{value:.6f}") for label, value in group_means),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = figures_dir / "rawfeat_evidence_histogram.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(evidence, bins="auto", color="#4472C4", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set(xlabel="Prototype-contrast evidence", ylabel="Samples")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=200)
    plt.close(fig)

    group_figure_path = figures_dir / "rawfeat_group_attribution_bar.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [item[0] for item in group_means]
    means = [item[1] for item in group_means]
    ax.bar(labels, means, color=["#4472C4", "#ED7D31", "#70AD47"])
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("Mean attribution")
    fig.tight_layout()
    fig.savefig(group_figure_path, dpi=200)
    plt.close(fig)

    fold_figure_path = figures_dir / "rawfeat_abs_evidence_by_fold.png"
    folds = sorted({int(row["fold"]) for row in rows})
    fold_means = []
    fold_errors = []
    for fold in folds:
        selected = absolute_evidence[np.asarray([int(row["fold"]) == fold for row in rows])]
        fold_means.append(float(np.mean(selected)))
        fold_errors.append(float(np.std(selected, ddof=1) / np.sqrt(selected.size)) if selected.size > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(folds, fold_means, yerr=fold_errors, capsize=3, color="#5B9BD5")
    ax.set(xlabel="ESC-50 fold", ylabel="Mean absolute evidence", xticks=folds)
    fig.tight_layout()
    fig.savefig(fold_figure_path, dpi=200)
    plt.close(fig)

    return [audit_path, sign_path, group_path, histogram_path, group_figure_path, fold_figure_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate(args.input, args.tables_dir, args.figures_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
