"""Optional plotting helpers for Dynamic-FPDE audio outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plot generation requires matplotlib.") from exc
    return plt


def save_deletion_insertion_plot(curves: dict[str, Any], output_base: str | Path) -> None:
    plt = _pyplot()
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fractions = np.asarray(curves["fractions"], dtype=float)
    ax.plot(fractions, curves["deletion_drop_curve"], marker="o", label="Deletion drop")
    ax.plot(fractions, curves["insertion_gain_curve"], marker="s", label="Insertion gain")
    ax.set_xlabel("Fraction of frames")
    ax.set_ylabel("Normalized prototype evidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_combined_score_plot(summary_rows: list[dict[str, object]], output_base: str | Path) -> None:
    plt = _pyplot()
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]) for row in summary_rows]
    values = [float(row.get("combined_score_mean") or 0.0) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(np.arange(len(labels)), values, color="#2563eb")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Combined score")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def save_lambda_selection_plot(lambda_rows: list[dict[str, object]], output_base: str | Path) -> None:
    plt = _pyplot()
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(lambda_rows, key=lambda row: (str(row.get("fold")), float(row.get("lambda_hyb") or 0.0)))
    xs = [float(row.get("lambda_hyb") or 0.0) for row in rows]
    ys = [float(row.get("mean_combined_score") or row.get("score") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("lambda_hyb")
    ax.set_ylabel("Mean combined score")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

