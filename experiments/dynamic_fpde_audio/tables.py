"""LaTeX table generation for Dynamic-FPDE audio experiment summaries."""

from __future__ import annotations

import csv
from pathlib import Path


def _read_required_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: object) -> str:
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value).replace("_", r"\_")


def _write_table(path: Path, columns: list[str], rows: list[list[object]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_fmt(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_tables(results_dir: str | Path, tables_dir: str | Path) -> list[Path]:
    results = Path(results_dir)
    tables = Path(tables_dir)
    summary = _read_required_csv(results / "dynamic_fpde_summary_by_method.csv")
    positive_summary = _read_required_csv(results / "dynamic_fpde_summary_positive_margin_by_method.csv")
    additivity = _read_required_csv(results / "dynamic_fpde_additivity_summary.csv")
    lambdas = _read_required_csv(results / "dynamic_fpde_lambda_selection.csv")
    written: list[Path] = []

    main_path = tables / "table_dynamic_fpde_main_results.tex"
    _write_table(
        main_path,
        ["Dataset", "Fold", "Method", "Combined", "Deletion", "Insertion", "N"],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                row.get("method"),
                row.get("combined_score_mean"),
                row.get("deletion_drop_auc_mean"),
                row.get("insertion_gain_auc_mean"),
                row.get("n"),
            ]
            for row in summary
        ],
        "Dynamic-FPDE all-sample prototype-driven deletion and insertion results.",
        "tab:dynamic-fpde-main-results",
    )
    written.append(main_path)

    positive_path = tables / "table_dynamic_fpde_positive_margin_results.tex"
    _write_table(
        positive_path,
        ["Dataset", "Fold", "Method", "Combined", "Deletion", "Insertion", "N"],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                row.get("method"),
                row.get("combined_score_mean"),
                row.get("deletion_drop_auc_mean"),
                row.get("insertion_gain_auc_mean"),
                row.get("n"),
            ]
            for row in positive_summary
        ],
        "Dynamic-FPDE results for samples with positive common Dynamic-Diff selection margin.",
        "tab:dynamic-fpde-positive-margin-results",
    )
    written.append(positive_path)

    margin_path = tables / "table_dynamic_fpde_margin_summary.tex"
    _write_table(
        margin_path,
        [
            "Dataset",
            "Fold",
            "Method",
            "Method Margin",
            "Method Positive Rate",
            "Selection Margin",
            "Selection Positive Rate",
            "Sel. N+",
            "Sel. N-",
        ],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                row.get("method"),
                row.get("prototype_margin_mean"),
                row.get("prototype_margin_positive_rate"),
                row.get("selection_margin_mean"),
                row.get("selection_margin_positive_rate"),
                row.get("n_selection_positive_margin"),
                row.get("n_selection_negative_margin"),
            ]
            for row in summary
        ],
        "Method-specific prototype margins and common Dynamic-Diff selection margins by method.",
        "tab:dynamic-fpde-margin-summary",
    )
    written.append(margin_path)

    additivity_path = tables / "table_dynamic_fpde_additivity.tex"
    _write_table(
        additivity_path,
        ["Dataset", "Fold", "Method", "Abs. Residual", "N"],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                row.get("method"),
                row.get("abs_exactness_residual_mean"),
                row.get("n"),
            ]
            for row in additivity
        ],
        "Dynamic-FPDE auditable attribution-sum residuals.",
        "tab:dynamic-fpde-additivity",
    )
    written.append(additivity_path)

    lambda_path = tables / "table_dynamic_fpde_lambda.tex"
    _write_table(
        lambda_path,
        ["Dataset", "Fold", "Lambda", "Score", "Deletion", "Insertion", "N"],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                row.get("lambda_hyb"),
                row.get("mean_combined_score") or row.get("score"),
                row.get("mean_deletion_drop_auc"),
                row.get("mean_insertion_gain_auc"),
                row.get("n_eval_samples"),
            ]
            for row in lambdas
        ],
        "Dynamic-Hyb lambda selection with normalized prototype-evidence curves.",
        "tab:dynamic-fpde-lambda",
    )
    written.append(lambda_path)
    return written
