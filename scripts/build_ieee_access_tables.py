#!/usr/bin/env python3
"""Generate IEEE Access manuscript tables from completed artifact CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


LATEX_ROW_END = r" \\"

METHOD_NAMES = {
    "hyb_fpde": "Hyb-FPDE",
    "bayesian_hyb_fpde": "Bayesian-FPDE",
    "bootstrap_fpde": "Bootstrap-FPDE",
    "shap": "SHAP",
    "lime": "LIME",
}

DATASET_ORDER = [
    (31, "credit-g"),
    (37, "diabetes"),
    (9946, "wdbc"),
    (10093, "banknote-authentication"),
    (10101, "blood-transfusion-service-center"),
]

REQUIRED_MAIN = [
    "public_uncertainty_validation_method_summary.csv",
    "public_uncertainty_validation_summary.csv",
    "stability_method_summary.csv",
    "training_size_uncertainty_method_summary.csv",
    "faithfulness_method_summary.csv",
    "faithfulness_summary.csv",
    "faithfulness_metrics.csv",
]

REQUIRED_SENSITIVITY = [
    "posterior_samples_sensitivity_summary.csv",
    "lambda_hyb_sensitivity_summary.csv",
]


def latex_escape(value: object) -> str:
    s = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "--" if pd.isna(value) else latex_escape(value)
    if not np.isfinite(value_float):
        return "--"
    return f"{value_float:.{digits}f}"


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    value_float = float(value)
    if not np.isfinite(value_float):
        return "--"
    return str(int(round(value_float)))


def table_env(
    label: str,
    caption: str,
    column_spec: str,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    table_star: bool = False,
) -> str:
    env = "table*" if table_star else "table"
    body = ["        " + " & ".join(header) + LATEX_ROW_END, "        \\hline"]
    body.extend("        " + " & ".join(row) + LATEX_ROW_END for row in rows)
    return (
        f"\\begin{{{env}}}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "        \\hline\n"
        + "\n".join(body)
        + "\n        \\hline\n"
        "\\end{tabular}\n"
        f"\\end{{{env}}}\n"
    )


def resolve_dir(explicit: Path | None, candidates: Sequence[Path], label: str) -> Path:
    if explicit is not None:
        if explicit.is_dir():
            return explicit
        for candidate in candidates:
            if candidate.name == explicit.name and candidate.is_dir():
                return candidate
        raise FileNotFoundError(f"{label} directory does not exist: {explicit}")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    joined = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find {label} directory. Tried: {joined}")


def require_files(root: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required CSVs under {root}: {', '.join(missing)}")


def read_csv(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / name)


def display_method(method: object) -> str:
    return METHOD_NAMES.get(str(method), str(method))


def method_summary_value(df: pd.DataFrame, method: str, column: str) -> object:
    subset = df[df["method"] == method]
    if subset.empty:
        raise ValueError(f"Missing method {method} in method summary")
    return subset.iloc[0][column]


def make_dataset_table(main_root: Path) -> str:
    summary = read_csv(main_root, "faithfulness_summary.csv")
    metrics = read_csv(main_root, "faithfulness_metrics.csv")
    rows = []
    for task_id, dataset in DATASET_ORDER:
        subset = summary[(summary["task_id"] == task_id) & (summary["dataset_name"] == dataset)]
        subset = subset[subset["method"] == "bayesian_hyb_fpde"]
        if subset.empty:
            raise ValueError(f"Missing faithfulness summary row for {task_id} {dataset}")
        row = subset.iloc[0]
        instances = float(row["mean_n_train"]) + float(row["mean_n_test"])
        label_values = metrics[metrics["task_id"] == task_id][["true_label", "pred_label", "target_label"]]
        classes = pd.unique(label_values.to_numpy().ravel())
        classes = [x for x in classes if pd.notna(x)]
        rows.append(
            [
                str(task_id),
                latex_escape(dataset),
                fmt_int(instances),
                fmt_int(row["mean_n_features"]),
                fmt_int(len(classes)),
                "From result metadata",
            ]
        )
    return table_env(
        "tab:datasets",
        "Public benchmark datasets used in the IEEE Access experiments.",
        "r l r r r l",
        ["Task ID", "Dataset", "Instances", "Features", "Classes", "Notes"],
        rows,
    )


def make_experimental_configuration_table() -> str:
    rows = [
        ("Datasets", "5 public benchmark datasets"),
        ("Seeds", "0--9"),
        ("n\\_explain", "50"),
        ("posterior\\_samples", "500"),
        ("bootstrap\\_samples", "50"),
        ("top\\_k", "5"),
        ("lambda\\_hyb", "0.5"),
        ("train\\_fractions", "\\{0.10, 0.25, 0.50, 0.75, 1.00\\}"),
        ("max\\_train\\_rows", "2000"),
        ("max\\_test\\_rows", "500"),
        ("model", "auto"),
        ("main output directory", "results\\_ieee"),
        ("sensitivity output directory", "results\\_ieee\\_sensitivity"),
    ]
    return table_env(
        "tab:experimental_configuration",
        "Experimental configuration for the public-data IEEE Access artifact runs.",
        "l l",
        ["Setting", "Value"],
        rows,
    )


def make_uncertainty_validation_table(main_root: Path) -> str:
    df = read_csv(main_root, "public_uncertainty_validation_method_summary.csv")
    rows = []
    for method in ["bayesian_hyb_fpde", "bootstrap_fpde"]:
        rows.append(
            [
                display_method(method),
                fmt(method_summary_value(df, method, "mean_empirical_reference_coverage_95")),
                fmt(method_summary_value(df, method, "mean_uncertainty_error_correlation")),
                fmt(method_summary_value(df, method, "mean_ci_width_error_correlation")),
                fmt(method_summary_value(df, method, "mean_mean_ci_width")),
                fmt(method_summary_value(df, method, "mean_mean_posterior_std")),
            ]
        )
    return table_env(
        "tab:uncertainty_validation",
        "Uncertainty validation against leave-one-seed empirical references.",
        "l r r r r r",
        [
            "Method",
            "empirical\\_reference\\_coverage\\_95",
            "uncertainty-error corr.",
            "CI width-error corr.",
            "Mean CI width",
            "Mean posterior std.",
        ],
        rows,
        table_star=True,
    )


def make_per_dataset_coverage_table(main_root: Path) -> str:
    df = read_csv(main_root, "public_uncertainty_validation_summary.csv")
    rows = []
    for task_id, dataset in DATASET_ORDER:
        subset = df[(df["task_id"] == task_id) & (df["dataset_name"] == dataset)]
        bayes = subset[subset["method"] == "bayesian_hyb_fpde"].iloc[0]
        boot = subset[subset["method"] == "bootstrap_fpde"].iloc[0]
        rows.append(
            [
                latex_escape(dataset),
                fmt(bayes["mean_empirical_reference_coverage_95"]),
                fmt(boot["mean_empirical_reference_coverage_95"]),
                fmt(bayes["mean_mean_ci_width"]),
                fmt(bayes["mean_mean_posterior_std"]),
            ]
        )
    return table_env(
        "tab:per_dataset_coverage",
        "Per-dataset empirical reference coverage; these values are not true attribution coverage.",
        "l r r r r",
        [
            "Dataset",
            "Bayesian-FPDE coverage",
            "Bootstrap-FPDE coverage",
            "Bayesian-FPDE mean CI width",
            "Bayesian-FPDE posterior std.",
        ],
        rows,
        table_star=True,
    )


def make_stability_table(main_root: Path) -> str:
    df = read_csv(main_root, "stability_method_summary.csv")
    rows = []
    for method in ["bayesian_hyb_fpde", "bootstrap_fpde", "hyb_fpde", "shap", "lime"]:
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            [
                display_method(method),
                fmt(row["mean_mean_spearman_between_seeds"]),
                fmt(row["mean_mean_pearson_between_seeds"]),
                fmt(row["mean_top_k_jaccard_between_seeds"]),
                fmt(row["mean_sign_agreement_between_seeds"]),
            ]
        )
    return table_env(
        "tab:stability",
        "Stability of feature rankings and signs across random seeds.",
        "l r r r r",
        ["Method", "Spearman", "Pearson", "Top-k Jaccard", "Sign agreement"],
        rows,
    )


def make_training_size_table(main_root: Path) -> str:
    df = read_csv(main_root, "training_size_uncertainty_method_summary.csv")
    df = df[df["method"] == "bayesian_hyb_fpde"].copy().sort_values("train_fraction")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                fmt(row["train_fraction"], 2),
                fmt(row["mean_mean_posterior_std"], 4),
                fmt(row["mean_mean_ci_width"], 4),
                fmt(row["mean_attribution_distance_to_full_train"], 4),
                fmt(row["mean_top_k_jaccard_to_full_train"], 3),
            ]
        )
    return table_env(
        "tab:training_size",
        "Bayesian-FPDE uncertainty behavior across training fractions.",
        "r r r r r",
        [
            "Train fraction",
            "Mean posterior std.",
            "Mean CI width",
            "Distance to full-train ref.",
            "Top-k Jaccard to full train",
        ],
        rows,
    )


def make_faithfulness_table(main_root: Path) -> str:
    df = read_csv(main_root, "faithfulness_method_summary.csv")
    rows = []
    for method in ["bayesian_hyb_fpde", "bootstrap_fpde", "hyb_fpde", "shap", "lime"]:
        subset = df[df["method"] == method]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            [
                display_method(method),
                fmt(row["mean_faithfulness_correlation"]),
                fmt(row["mean_deletion_auc"]),
                fmt(row["mean_insertion_auc"]),
                fmt(row["mean_combined_score"]),
            ]
        )
    return table_env(
        "tab:faithfulness",
        "Faithfulness metrics are baseline-dependent and are used as complementary model-output consistency evidence.",
        "l r r r r",
        ["Method", "Faithfulness corr.", "Deletion AUC", "Insertion AUC", "Combined score"],
        rows,
    )


def aggregate_sensitivity(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    numeric_cols = [
        "empirical_reference_coverage_95",
        "mean_posterior_std",
        "mean_ci_width",
        "mean_spearman_between_seeds",
        "top_k_jaccard_between_seeds",
        "attribution_distance_to_default",
        "top_k_jaccard_to_default",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    return df.groupby(group_column, as_index=False)[cols].mean(numeric_only=True).sort_values(group_column)


def make_posterior_sensitivity_table(sensitivity_root: Path) -> str:
    df = read_csv(sensitivity_root, "posterior_samples_sensitivity_summary.csv")
    if "lambda_hyb" in df.columns:
        df = df[np.isclose(df["lambda_hyb"].astype(float), 0.5)]
    grouped = aggregate_sensitivity(df, "posterior_samples")
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            [
                fmt_int(row["posterior_samples"]),
                fmt(row["empirical_reference_coverage_95"]),
                fmt(row["mean_posterior_std"], 4),
                fmt(row["mean_ci_width"], 4),
                fmt(row["mean_spearman_between_seeds"]),
                fmt(row["top_k_jaccard_between_seeds"]),
            ]
        )
    return table_env(
        "tab:posterior_sensitivity",
        "Sensitivity to posterior sample size at lambda\\_hyb = 0.5.",
        "r r r r r r",
        [
            "posterior\\_samples",
            "empirical\\_reference\\_coverage\\_95",
            "Mean posterior std.",
            "Mean CI width",
            "Spearman",
            "Top-k Jaccard",
        ],
        rows,
    )


def make_lambda_sensitivity_table(sensitivity_root: Path) -> str:
    df = read_csv(sensitivity_root, "lambda_hyb_sensitivity_summary.csv")
    if "posterior_samples" in df.columns:
        df = df[df["posterior_samples"].astype(float) == 500.0]
    grouped = aggregate_sensitivity(df, "lambda_hyb")
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            [
                fmt(row["lambda_hyb"], 2),
                fmt(row["empirical_reference_coverage_95"]),
                fmt(row["mean_posterior_std"], 4),
                fmt(row["mean_ci_width"], 4),
                fmt(row["mean_spearman_between_seeds"]),
                fmt(row["top_k_jaccard_between_seeds"]),
                fmt(row["attribution_distance_to_default"], 4),
                fmt(row["top_k_jaccard_to_default"]),
            ]
        )
    return table_env(
        "tab:lambda_sensitivity",
        "Sensitivity to the Hyb-FPDE mixing weight at posterior\\_samples = 500.",
        "r r r r r r r r",
        [
            "lambda\\_hyb",
            "empirical\\_reference\\_coverage\\_95",
            "Mean posterior std.",
            "Mean CI width",
            "Spearman",
            "Top-k Jaccard",
            "Distance to default",
            "Top-k Jaccard to default",
        ],
        rows,
        table_star=True,
    )


def make_reproducibility_artifact_table() -> str:
    rows = [
        ("Main config", "configs/openml\\_public\\_ieee\\_access.yaml", "Main public-data experiment configuration"),
        ("Sensitivity config", "configs/openml\\_public\\_ieee\\_access\\_sensitivity.yaml", "Sensitivity experiment configuration"),
        ("Main workflow", ".github/workflows/ieee-access-bayesian-fpde-experiments.yml", "GitHub Actions workflow for main outputs"),
        ("Sensitivity workflow", ".github/workflows/ieee-access-sensitivity.yml", "GitHub Actions workflow for sensitivity outputs"),
        ("Main results", "results\\_ieee/", "Aggregate CSV outputs for main experiments"),
        ("Sensitivity results", "results\\_ieee\\_sensitivity/", "Aggregate CSV outputs for sensitivity analyses"),
        ("Main figures", "figures\\_ieee/", "Figures generated from main results"),
        ("Sensitivity figures", "figures\\_ieee\\_sensitivity/", "Figures generated from sensitivity results"),
        ("Logs", "logs\\_ieee/, logs\\_ieee\\_sensitivity/", "Workflow execution logs"),
    ]
    return table_env(
        "tab:reproducibility_artifact",
        "Reproducibility artifact components for the IEEE Access experiments.",
        "l l l",
        ["Component", "Path", "Purpose"],
        rows,
        table_star=True,
    )


def write_table(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate IEEE Access LaTeX tables from artifact CSVs.")
    parser.add_argument("--main-results-dir", type=Path, help="Directory containing results_ieee CSVs.")
    parser.add_argument("--sensitivity-results-dir", type=Path, help="Directory containing results_ieee_sensitivity CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper") / "tables", help="Output directory for LaTeX tables.")
    args = parser.parse_args(argv)

    main_root = resolve_dir(
        args.main_results_dir,
        [
            Path("results_ieee"),
            Path("bayesian-fpde-ieee-access-ieee_full") / "results_ieee",
        ],
        "main results",
    )
    sensitivity_root = resolve_dir(
        args.sensitivity_results_dir,
        [
            Path("results_ieee_sensitivity"),
            Path("bayesian-fpde-ieee-access-sensitivity-sensitivity_full") / "results_ieee_sensitivity",
        ],
        "sensitivity results",
    )
    require_files(main_root, REQUIRED_MAIN)
    require_files(sensitivity_root, REQUIRED_SENSITIVITY)

    tables = {
        "table_datasets.tex": make_dataset_table(main_root),
        "table_experimental_configuration.tex": make_experimental_configuration_table(),
        "table_uncertainty_validation.tex": make_uncertainty_validation_table(main_root),
        "table_per_dataset_coverage.tex": make_per_dataset_coverage_table(main_root),
        "table_stability.tex": make_stability_table(main_root),
        "table_training_size.tex": make_training_size_table(main_root),
        "table_faithfulness.tex": make_faithfulness_table(main_root),
        "table_posterior_sensitivity.tex": make_posterior_sensitivity_table(sensitivity_root),
        "table_lambda_sensitivity.tex": make_lambda_sensitivity_table(sensitivity_root),
        "table_reproducibility_artifact.tex": make_reproducibility_artifact_table(),
    }

    written = []
    for name, content in tables.items():
        out_path = args.output_dir / name
        write_table(out_path, content)
        written.append(out_path)

    print(f"Main results CSV directory: {main_root}")
    print(f"Sensitivity results CSV directory: {sensitivity_root}")
    print("Generated tables:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
