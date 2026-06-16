"""LaTeX table generation for Dynamic-FPDE audio experiment summaries."""

from __future__ import annotations

import csv
from pathlib import Path


METHOD_LABELS = {
    "dynamic_diff": "Native-Time Dynamic-Diff",
    "dynamic_cos": "Native-Time Dynamic-Cos",
    "dynamic_hyb": "Native-Time Dynamic-Hyb",
    "energy_baseline_raw": "Raw energy baseline",
    "feature_norm_baseline_standardized": "Standardized feature-norm baseline",
    "random_baseline": "Random baseline",
}


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


def _method_label(value: object) -> str:
    return METHOD_LABELS.get(str(value), str(value))


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
    sample_metrics = _read_required_csv(results / "dynamic_fpde_sample_metrics.csv")
    written: list[Path] = []

    main_path = tables / "table_dynamic_fpde_main_results.tex"
    _write_table(
        main_path,
        ["Dataset", "Fold", "Method", "Combined", "Deletion", "Insertion", "Unique N", "Rows"],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                _method_label(row.get("method")),
                row.get("combined_score_mean"),
                row.get("deletion_drop_auc_mean"),
                row.get("insertion_gain_auc_mean"),
                row.get("n_unique_samples") or row.get("n"),
                row.get("n_rows") or row.get("n"),
            ]
            for row in summary
        ],
        "Legacy/comparison Native-Time Dynamic-FPDE all-sample prototype-evidence removal and recovery diagnostics; random baseline repetitions are averaged per sample.",
        "tab:dynamic-fpde-main-results",
    )
    written.append(main_path)

    positive_path = tables / "table_dynamic_fpde_positive_margin_results.tex"
    _write_table(
        positive_path,
        ["Dataset", "Fold", "Method", "Combined", "Deletion", "Insertion", "Unique N", "Rows"],
        [
            [
                row.get("dataset"),
                row.get("fold"),
                _method_label(row.get("method")),
                row.get("combined_score_mean"),
                row.get("deletion_drop_auc_mean"),
                row.get("insertion_gain_auc_mean"),
                row.get("n_unique_samples") or row.get("n"),
                row.get("n_rows") or row.get("n"),
            ]
            for row in positive_summary
        ],
        "Legacy/comparison Native-Time Dynamic-FPDE selection-positive-margin results using the common Native-Time Dynamic-Diff target/rival pair; random repetitions are averaged per sample.",
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
                _method_label(row.get("method")),
                row.get("prototype_margin_mean"),
                row.get("prototype_margin_positive_rate"),
                row.get("selection_margin_mean"),
                row.get("selection_margin_positive_rate"),
                row.get("n_selection_positive_margin"),
                row.get("n_selection_negative_margin"),
            ]
            for row in summary
        ],
        "Legacy/comparison method-specific prototype margins and common Native-Time Dynamic-Diff selection margins by method.",
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
                _method_label(row.get("method")),
                row.get("abs_exactness_residual_mean"),
                row.get("n"),
            ]
            for row in additivity
        ],
        "Legacy/comparison Native-Time Dynamic-FPDE auditable attribution-sum residuals for Phi.",
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
        "Legacy/comparison Native-Time Dynamic-Hyb lambda selection with normalized prototype-evidence diagnostics.",
        "tab:dynamic-fpde-lambda",
    )
    written.append(lambda_path)

    shape_path = tables / "table_dynamic_fpde_native_time_checks.tex"
    methods = sorted({row.get("method", "") for row in sample_metrics})
    check_rows: list[list[object]] = []
    for method in methods:
        rows = [row for row in sample_metrics if row.get("method") == method]
        n = len(rows)
        shape_ok = sum(str(row.get("shape_preserved", "")).lower() == "true" for row in rows)
        target_meta = sum(bool(row.get("target_prototype_source_sample_id")) for row in rows)
        rival_meta = sum(bool(row.get("rival_prototype_source_sample_id")) for row in rows)
        residuals = []
        for row in rows:
            try:
                residuals.append(float(row.get("abs_exactness_residual") or "nan"))
            except ValueError:
                pass
        check_rows.append(
            [
                _method_label(method),
                n,
                shape_ok,
                target_meta,
                rival_meta,
                max(residuals) if residuals else "",
            ]
        )
    _write_table(
        shape_path,
        ["Method", "N", "Shape OK", "Target Proto Meta", "Rival Proto Meta", "Max Abs. Residual"],
        check_rows,
        "Legacy/comparison Native-Time checks for Phi shape preservation, exemplar prototype metadata availability, and attribution additivity.",
        "tab:dynamic-fpde-native-time-checks",
    )
    written.append(shape_path)
    return written
