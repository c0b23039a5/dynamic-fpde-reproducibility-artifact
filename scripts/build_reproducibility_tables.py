#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_reproducibility_tables.py

Reconstruct the FPDE manuscript tables from raw per-run experiment outputs.

Input:
    - A zip archive or directory containing per-task/per-seed output folders.
    - Each output folder should contain at least:
        summary_by_task.csv
        lambda_distribution.csv
        run_config.json

Output:
    - processed CSV files used for auditing
    - LaTeX tables and reproducibility text snippets

Example:
    python build_reproducibility_tables.py \
        --input ALL_DATA.zip \
        --output generated \
        --expected-tasks 72 \
        --expected-seeds 10
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


METHOD_MAP: Dict[str, str] = {
    "hyb_fpde_grid": "Hyb-FPDE",
    "aime": "AIME",
    "shap": "SHAP",
    "lime": "LIME",
}
METHOD_ORDER = ["Hyb-FPDE", "AIME", "SHAP", "LIME"]
METRICS = ["deletion_drop_auc", "insertion_auc", "combined_score", "runtime_seconds"]
SUBJECT_ORDER = [
    "Biology",
    "Business",
    "Climate and Environment",
    "Computer Science / Computer Vision",
    "Computer Science / Other",
    "Engineering",
    "Games",
    "Health and Medicine",
    "Law",
    "Physics and Chemistry",
    "Social Sciences",
    "Other",
]
LATEX_ROW_END = r" \\"

# Reporting labels only. These labels are not used for model fitting,
# attribution computation, or perturbation evaluation.
TASK_METADATA = [{'task_id': 45,
  'dataset_name': 'splice',
  'subject_area': 'Biology',
  'subject_subarea': 'Genomics and molecular biology'},
 {'task_id': 167140,
  'dataset_name': 'dna',
  'subject_area': 'Biology',
  'subject_subarea': 'Genomics and molecular biology'},
 {'task_id': 9910,
  'dataset_name': 'Bioresponse',
  'subject_area': 'Biology',
  'subject_subarea': 'Molecular bioactivity'},
 {'task_id': 146800,
  'dataset_name': 'MiceProtein',
  'subject_area': 'Biology',
  'subject_subarea': 'Protein expression'},
 {'task_id': 29,
  'dataset_name': 'credit-approval',
  'subject_area': 'Business',
  'subject_subarea': 'Credit and finance'},
 {'task_id': 31,
  'dataset_name': 'credit-g',
  'subject_area': 'Business',
  'subject_subarea': 'Credit and finance'},
 {'task_id': 167120,
  'dataset_name': 'numerai28.6',
  'subject_area': 'Business',
  'subject_subarea': 'Financial market prediction'},
 {'task_id': 9981,
  'dataset_name': 'cnae-9',
  'subject_area': 'Business',
  'subject_subarea': 'Industry and economic activity text classification'},
 {'task_id': 219,
  'dataset_name': 'electricity',
  'subject_area': 'Business',
  'subject_subarea': 'Energy market systems'},
 {'task_id': 10101,
  'dataset_name': 'blood-transfusion-service-center',
  'subject_area': 'Business',
  'subject_subarea': 'Donation behavior and service analytics'},
 {'task_id': 14965,
  'dataset_name': 'bank-marketing',
  'subject_area': 'Business',
  'subject_subarea': 'Marketing, sales, and customer analytics'},
 {'task_id': 125920,
  'dataset_name': 'dresses-sales',
  'subject_area': 'Business',
  'subject_subarea': 'Marketing, sales, and customer analytics'},
 {'task_id': 167141,
  'dataset_name': 'churn',
  'subject_area': 'Business',
  'subject_subarea': 'Marketing, sales, and customer analytics'},
 {'task_id': 2079,
  'dataset_name': 'eucalyptus',
  'subject_area': 'Climate and Environment',
  'subject_subarea': 'Climate, ecology, and environment'},
 {'task_id': 9978,
  'dataset_name': 'ozone-level-8hr',
  'subject_area': 'Climate and Environment',
  'subject_subarea': 'Climate, ecology, and environment'},
 {'task_id': 146819,
  'dataset_name': 'climate-model-simulation-crashes',
  'subject_area': 'Climate and Environment',
  'subject_subarea': 'Climate, ecology, and environment'},
 {'task_id': 2074,
  'dataset_name': 'satimage',
  'subject_area': 'Climate and Environment',
  'subject_subarea': 'Remote sensing and satellite imagery'},
 {'task_id': 146820,
  'dataset_name': 'wilt',
  'subject_area': 'Biology',
  'subject_subarea': 'Plant disease and remote sensing'},
 {'task_id': 6,
  'dataset_name': 'letter',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / letters, digits, and scripts'},
 {'task_id': 28,
  'dataset_name': 'optdigits',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / letters, digits, and scripts'},
 {'task_id': 32,
  'dataset_name': 'pendigits',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / letters, digits, and scripts'},
 {'task_id': 3573,
  'dataset_name': 'mnist_784',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / letters, digits, and scripts'},
 {'task_id': 9964,
  'dataset_name': 'semeion',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / letters, digits, and scripts'},
 {'task_id': 167121,
  'dataset_name': 'Devnagari-Script',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / letters, digits, and scripts'},
 {'task_id': 12,
  'dataset_name': 'mfeat-factors',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 14,
  'dataset_name': 'mfeat-fourier',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 16,
  'dataset_name': 'mfeat-karhunen',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 18,
  'dataset_name': 'mfeat-morphological',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 22,
  'dataset_name': 'mfeat-zernike',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 53,
  'dataset_name': 'vehicle',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 125922,
  'dataset_name': 'texture',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 146822,
  'dataset_name': 'segment',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 146824,
  'dataset_name': 'mfeat-pixel',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 146825,
  'dataset_name': 'Fashion-MNIST',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 167124,
  'dataset_name': 'CIFAR_10',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Computer Vision / image descriptors and natural images'},
 {'task_id': 10093,
  'dataset_name': 'banknote-authentication',
  'subject_area': 'Computer Science / Computer Vision',
  'subject_subarea': 'Image-derived authentication'},
 {'task_id': 3902,
  'dataset_name': 'pc4',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Software engineering quality'},
 {'task_id': 3903,
  'dataset_name': 'pc3',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Software engineering quality'},
 {'task_id': 3904,
  'dataset_name': 'jm1',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Software engineering quality'},
 {'task_id': 3913,
  'dataset_name': 'kc2',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Software engineering quality'},
 {'task_id': 3917,
  'dataset_name': 'kc1',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Software engineering quality'},
 {'task_id': 3918,
  'dataset_name': 'pc1',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Software engineering quality'},
 {'task_id': 3022,
  'dataset_name': 'vowel',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Speech and language'},
 {'task_id': 3481,
  'dataset_name': 'isolet',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Speech and language'},
 {'task_id': 9952,
  'dataset_name': 'phoneme',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Speech and language'},
 {'task_id': 3549,
  'dataset_name': 'analcatdata_authorship',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'NLP / authorship attribution'},
 {'task_id': 9985,
  'dataset_name': 'first-order-theorem-proving',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Symbolic reasoning and theorem proving'},
 {'task_id': 43,
  'dataset_name': 'spambase',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Web, advertising, spam, and security'},
 {'task_id': 9977,
  'dataset_name': 'nomao',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Web entity resolution and record linkage'},
 {'task_id': 14952,
  'dataset_name': 'PhishingWebsites',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Web, advertising, spam, and security'},
 {'task_id': 167125,
  'dataset_name': 'Internet-Advertisements',
  'subject_area': 'Computer Science / Other',
  'subject_subarea': 'Web, advertising, spam, and security'},
 {'task_id': 14954,
  'dataset_name': 'cylinder-bands',
  'subject_area': 'Engineering',
  'subject_subarea': 'Industrial manufacturing and fault detection'},
 {'task_id': 146817,
  'dataset_name': 'steel-plates-fault',
  'subject_area': 'Engineering',
  'subject_subarea': 'Industrial manufacturing and fault detection'},
 {'task_id': 9960,
  'dataset_name': 'wall-robot-navigation',
  'subject_area': 'Engineering',
  'subject_subarea': 'Sensors, gestures, robotics, and activity recognition'},
 {'task_id': 14969,
  'dataset_name': 'GesturePhaseSegmentationProcessed',
  'subject_area': 'Engineering',
  'subject_subarea': 'Sensors, gestures, robotics, and activity recognition'},
 {'task_id': 14970,
  'dataset_name': 'har',
  'subject_area': 'Engineering',
  'subject_subarea': 'Sensors, gestures, robotics, and activity recognition'},
 {'task_id': 3,
  'dataset_name': 'kr-vs-kp',
  'subject_area': 'Games',
  'subject_subarea': 'Board and strategic games'},
 {'task_id': 49,
  'dataset_name': 'tic-tac-toe',
  'subject_area': 'Games',
  'subject_subarea': 'Board and strategic games'},
 {'task_id': 146195,
  'dataset_name': 'connect-4',
  'subject_area': 'Games',
  'subject_subarea': 'Board and strategic games'},
 {'task_id': 167119,
  'dataset_name': 'jungle_chess_2pcs_raw_endgame_complete',
  'subject_area': 'Games',
  'subject_subarea': 'Board and strategic games'},
 {'task_id': 15,
  'dataset_name': 'breast-w',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Clinical and medical diagnosis'},
 {'task_id': 37,
  'dataset_name': 'diabetes',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Clinical and medical diagnosis'},
 {'task_id': 3021,
  'dataset_name': 'sick',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Clinical and medical diagnosis'},
 {'task_id': 9946,
  'dataset_name': 'wdbc',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Clinical and medical diagnosis'},
 {'task_id': 9971,
  'dataset_name': 'ilpd',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Clinical and medical diagnosis'},
 {'task_id': 3560,
  'dataset_name': 'analcatdata_dmft',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Dental and medical survey data'},
 {'task_id': 23,
  'dataset_name': 'cmc',
  'subject_area': 'Health and Medicine',
  'subject_subarea': 'Reproductive health and socioeconomic survey data'},
 {'task_id': 9957,
  'dataset_name': 'qsar-biodeg',
  'subject_area': 'Physics and Chemistry',
  'subject_subarea': 'Chemical molecular descriptors and biodegradation'},
 {'task_id': 11,
  'dataset_name': 'balance-scale',
  'subject_area': 'Social Sciences',
  'subject_subarea': 'Psychological experimental data'},
 {'task_id': 7592,
  'dataset_name': 'adult',
  'subject_area': 'Social Sciences',
  'subject_subarea': 'Census and socioeconomic data'},
 {'task_id': 146821,
  'dataset_name': 'car',
  'subject_area': 'Other',
  'subject_subarea': 'Rule-based decision-model benchmark'},
 {'task_id': 9976,
  'dataset_name': 'madelon',
  'subject_area': 'Other',
  'subject_subarea': 'Synthetic benchmark'}]


@dataclass(frozen=True)
class LoadedResults:
    summary_by_task: pd.DataFrame
    lambda_distribution: pd.DataFrame
    run_configs: pd.DataFrame


def latex_escape(value: object) -> str:
    """Escape a small text cell for LaTeX tabular output."""
    s = str(value)
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


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_mean_sd(mean: float, sd: float, digits: int = 4, bold: bool = False) -> str:
    cell = f"{fmt(mean, digits)} $\\pm$ {fmt(sd, digits)}"
    return f"\\textbf{{{cell}}}" if bold else cell


def fmt_p_value(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    if p < 0.001:
        mantissa, exponent = f"{p:.2e}".split("e")
        return rf"${float(mantissa):.2f}\times 10^{{{int(exponent)}}}$"
    return f"{p:.3f}"


def table_env(label: str, caption: str, tabular: str, *, table_star: bool = True) -> str:
    env = "table*" if table_star else "table"
    return (
        f"\\begin{{{env}}}[!tbp]\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        f"    \\centering\n"
        f"{tabular}\n"
        f"\\end{{{env}}}\n"
    )


def parse_seed_from_config(config_text: str, source_name: str) -> Dict[str, object]:
    cfg = json.loads(config_text)
    if "seed" not in cfg:
        raise ValueError(f"run_config.json has no seed: {source_name}")
    return cfg


def _read_zip(input_path: Path) -> LoadedResults:
    summary_frames: List[pd.DataFrame] = []
    lambda_frames: List[pd.DataFrame] = []
    config_rows: List[Dict[str, object]] = []

    with zipfile.ZipFile(input_path) as zf:
        names = set(zf.namelist())
        for name in sorted(names):
            if not name.endswith("summary_by_task.csv"):
                continue
            folder = name.rsplit("/", 1)[0]
            cfg_name = folder + "/run_config.json"
            if cfg_name not in names:
                raise FileNotFoundError(f"Missing run_config.json next to {name}")
            cfg = parse_seed_from_config(zf.read(cfg_name).decode("utf-8"), cfg_name)
            seed = int(cfg["seed"])

            df = pd.read_csv(io.BytesIO(zf.read(name)))
            df["seed"] = seed
            df["run_folder"] = folder
            summary_frames.append(df)

            config_row = dict(cfg)
            config_row["run_folder"] = folder
            config_rows.append(config_row)

            lambda_name = folder + "/lambda_distribution.csv"
            if lambda_name in names:
                ldf = pd.read_csv(io.BytesIO(zf.read(lambda_name)))
                ldf["seed"] = seed
                ldf["run_folder"] = folder
                lambda_frames.append(ldf)

    if not summary_frames:
        raise FileNotFoundError(f"No summary_by_task.csv files found in {input_path}")
    return LoadedResults(
        summary_by_task=pd.concat(summary_frames, ignore_index=True),
        lambda_distribution=pd.concat(lambda_frames, ignore_index=True) if lambda_frames else pd.DataFrame(),
        run_configs=pd.DataFrame(config_rows),
    )


def _read_directory(input_path: Path) -> LoadedResults:
    summary_frames: List[pd.DataFrame] = []
    lambda_frames: List[pd.DataFrame] = []
    config_rows: List[Dict[str, object]] = []

    for summary_path in sorted(input_path.rglob("summary_by_task.csv")):
        folder = summary_path.parent
        cfg_path = folder / "run_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing run_config.json next to {summary_path}")
        cfg = parse_seed_from_config(cfg_path.read_text(encoding="utf-8"), str(cfg_path))
        seed = int(cfg["seed"])

        df = pd.read_csv(summary_path)
        df["seed"] = seed
        df["run_folder"] = str(folder.relative_to(input_path))
        summary_frames.append(df)

        config_row = dict(cfg)
        config_row["run_folder"] = str(folder.relative_to(input_path))
        config_rows.append(config_row)

        lambda_path = folder / "lambda_distribution.csv"
        if lambda_path.exists():
            ldf = pd.read_csv(lambda_path)
            ldf["seed"] = seed
            ldf["run_folder"] = str(folder.relative_to(input_path))
            lambda_frames.append(ldf)

    if not summary_frames:
        raise FileNotFoundError(f"No summary_by_task.csv files found under {input_path}")
    return LoadedResults(
        summary_by_task=pd.concat(summary_frames, ignore_index=True),
        lambda_distribution=pd.concat(lambda_frames, ignore_index=True) if lambda_frames else pd.DataFrame(),
        run_configs=pd.DataFrame(config_rows),
    )


def load_results(input_path: Path) -> LoadedResults:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return _read_zip(input_path)
    if input_path.is_dir():
        return _read_directory(input_path)
    raise ValueError(f"Input must be a .zip archive or a directory: {input_path}")


def prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"task_id", "dataset_name", "method", "seed", *METRICS}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary_by_task.csv is missing required columns: {missing}")

    df = summary[summary["method"].isin(METHOD_MAP)].copy()
    df["method_display"] = df["method"].map(METHOD_MAP)
    metadata = pd.DataFrame(TASK_METADATA)
    df = df.merge(metadata, on="task_id", how="left", suffixes=("", "_mapped"))
    missing_meta = df[df["subject_area"].isna()]["task_id"].drop_duplicates().tolist()
    if missing_meta:
        raise ValueError(f"Missing TASK_METADATA entries for task_id values: {missing_meta}")
    return df


def validate_completeness(df: pd.DataFrame, expected_tasks: int, expected_seeds: int) -> None:
    tasks = sorted(int(x) for x in df["task_id"].unique())
    seeds = sorted(int(x) for x in df["seed"].unique())
    methods = sorted(df["method_display"].unique().tolist())
    if expected_tasks and len(tasks) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, found {len(tasks)}")
    if expected_seeds and len(seeds) != expected_seeds:
        raise ValueError(f"Expected {expected_seeds} seeds, found {len(seeds)}: {seeds}")
    expected = pd.MultiIndex.from_product([tasks, seeds, METHOD_ORDER], names=["task_id", "seed", "method_display"])
    observed = pd.MultiIndex.from_frame(df[["task_id", "seed", "method_display"]].drop_duplicates())
    missing = expected.difference(observed)
    if len(missing) > 0:
        raise ValueError(f"Missing task/seed/method cells; first missing cells: {list(missing[:10])}")
    extra_methods = sorted(set(methods) - set(METHOD_ORDER))
    if extra_methods:
        raise ValueError(f"Unexpected display methods after filtering: {extra_methods}")


def aggregate_seed_level(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_seed = df.groupby(["seed", "method_display", "subject_area"], as_index=False)[METRICS].mean()
    subject_balanced_seed = subject_seed.groupby(["seed", "method_display"], as_index=False)[METRICS].mean()
    task_balanced_seed = df.groupby(["seed", "method_display"], as_index=False)[METRICS].mean()
    return subject_seed, subject_balanced_seed, task_balanced_seed


def summarize_over_seeds(seed_level: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    agg = seed_level.groupby(list(group_cols)).agg({m: ["mean", "std"] for m in metrics}).reset_index()
    agg.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg


def make_main_results_table(summary: pd.DataFrame, label: str, caption: str) -> str:
    rows = []
    best_method = summary.sort_values("combined_score_mean", ascending=False).iloc[0]["method_display"]
    for method in METHOD_ORDER:
        r = summary[summary["method_display"] == method].iloc[0]
        cells = [latex_escape(method)]
        for metric in METRICS:
            cells.append(fmt_mean_sd(r[f"{metric}_mean"], r[f"{metric}_std"], 4, bold=(method == best_method and metric == "combined_score")))
        rows.append("        " + " & ".join(cells) + LATEX_ROW_END)
    tabular = (
        "    \\begin{tabular}{lcccc}\n"
        "        \\toprule\n"
        "        Method & Deletion-drop AUC & Insertion AUC & Combined score & Runtime (s) " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}"
    )
    return table_env(label, caption, tabular, table_star=True)


def make_subject_area_table(subject_seed: pd.DataFrame) -> str:
    summary = summarize_over_seeds(subject_seed, ["subject_area", "method_display"], metrics=["combined_score"])
    rows = []
    for subject in SUBJECT_ORDER:
        sub = summary[summary["subject_area"] == subject]
        if sub.empty:
            continue
        best = sub.sort_values("combined_score_mean", ascending=False).iloc[0]["method_display"]
        cells = [latex_escape(subject)]
        for method in METHOD_ORDER:
            r = sub[sub["method_display"] == method]
            if r.empty:
                cells.append("--")
            else:
                rr = r.iloc[0]
                cells.append(fmt_mean_sd(rr["combined_score_mean"], rr["combined_score_std"], 4, bold=(method == best)))
        rows.append("        " + " & ".join(cells) + LATEX_ROW_END)
    tabular = (
        "    \\scriptsize\n"
        "    \\setlength{\\tabcolsep}{3pt}\n"
        "    \\begin{tabular}{lcccc}\n"
        "        \\toprule\n"
        "        Subject area & Hyb-FPDE & AIME & SHAP & LIME " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}"
    )
    return table_env(
        "tab:subject_area_combined_scores",
        "Subject-area-wise combined score. Values are mean $\\pm$ sample standard deviation over seeds after averaging tasks within each subject area.",
        tabular,
        table_star=True,
    )


def selected_lambda_by_task_seed(lambda_df: pd.DataFrame) -> pd.DataFrame:
    if lambda_df.empty or "selected_lambda" not in lambda_df.columns:
        return pd.DataFrame(columns=["task_id", "seed", "selected_lambda"])
    required = {"task_id", "seed", "selected_lambda"}
    missing = sorted(required - set(lambda_df.columns))
    if missing:
        raise ValueError(f"lambda_distribution.csv missing columns: {missing}")
    selected = lambda_df[["task_id", "seed", "selected_lambda"]].dropna().drop_duplicates()
    if selected.duplicated(["task_id", "seed"]).any():
        selected = selected.groupby(["task_id", "seed"], as_index=False)["selected_lambda"].first()
    return selected


def taskwise_lambda_summary(selected_lambda: pd.DataFrame) -> pd.DataFrame:
    if selected_lambda.empty:
        return pd.DataFrame(columns=["task_id", "lambda_mean", "lambda_sd"])
    return selected_lambda.groupby("task_id", as_index=False).agg(
        lambda_mean=("selected_lambda", "mean"),
        lambda_sd=("selected_lambda", "std"),
    )


def taskwise_summary(df: pd.DataFrame, selected_lambda: pd.DataFrame) -> pd.DataFrame:
    taskwise = df.groupby(["subject_area", "subject_subarea", "task_id", "dataset_name_mapped", "method_display"], as_index=False).agg(
        combined_mean=("combined_score", "mean"),
        combined_sd=("combined_score", "std"),
    )
    lambda_summary = taskwise_lambda_summary(selected_lambda)
    if not lambda_summary.empty:
        taskwise = taskwise.merge(lambda_summary, on="task_id", how="left")
    else:
        taskwise["lambda_mean"] = np.nan
        taskwise["lambda_sd"] = np.nan
    return taskwise


def make_taskwise_appendix_tables(taskwise: pd.DataFrame) -> str:
    parts = []
    for subject in SUBJECT_ORDER:
        sub = taskwise[taskwise["subject_area"] == subject]
        if sub.empty:
            continue
        pivot_mean = sub.pivot_table(index=["subject_subarea", "task_id", "dataset_name_mapped"], columns="method_display", values="combined_mean")
        pivot_sd = sub.pivot_table(index=["subject_subarea", "task_id", "dataset_name_mapped"], columns="method_display", values="combined_sd")
        lambda_by_task = sub[["subject_subarea", "task_id", "dataset_name_mapped", "lambda_mean", "lambda_sd"]].drop_duplicates()
        lambda_by_task = lambda_by_task.set_index(["subject_subarea", "task_id", "dataset_name_mapped"])
        pivot_mean = pivot_mean.reset_index().sort_values(["subject_subarea", "task_id"])
        pivot_sd = pivot_sd.reset_index().set_index(["subject_subarea", "task_id", "dataset_name_mapped"])
        rows = []
        for _, row in pivot_mean.iterrows():
            idx = (row["subject_subarea"], row["task_id"], row["dataset_name_mapped"])
            displayed = {m: round(float(row[m]), 3) for m in METHOD_ORDER if pd.notna(row.get(m, np.nan))}
            best_value = max(displayed.values())
            lambda_mean = float(lambda_by_task.loc[idx, "lambda_mean"]) if idx in lambda_by_task.index else float("nan")
            lambda_sd = float(lambda_by_task.loc[idx, "lambda_sd"]) if idx in lambda_by_task.index else float("nan")
            cells = [
                latex_escape(row["subject_subarea"]),
                str(int(row["task_id"])),
                latex_escape(row["dataset_name_mapped"]),
                fmt_mean_sd(lambda_mean, lambda_sd, 2),
            ]
            for method in METHOD_ORDER:
                mean = float(row[method])
                sd = float(pivot_sd.loc[idx, method])
                cells.append(fmt_mean_sd(mean, sd, 3, bold=(round(mean, 3) == best_value)))
            rows.append("        " + " & ".join(cells) + LATEX_ROW_END)
        label = "tab:taskwise_combined_scores_sd_" + re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
        label = label.replace("computer_science_computer_vision", "computer_vision")
        label = label.replace("climate_and_environment", "climate_environment")
        label = label.replace("health_and_medicine", "health_medicine")
        label = label.replace("physics_and_chemistry", "physics_chemistry")
        tabular = (
            "    \\scriptsize\n"
            "    \\setlength{\\tabcolsep}{3pt}\n"
            "    \\renewcommand{\\arraystretch}{1.03}\n"
            "    \\begin{tabular}{lrlccccc}\n"
            "        \\toprule\n"
            "        Subject subarea & Task & Dataset & $\\lambda^*$ & Hyb-FPDE & AIME & SHAP & LIME " + LATEX_ROW_END + "\n"
            "        \\midrule\n"
            + "\n".join(rows) + "\n"
            "        \\bottomrule\n"
            "    \\end{tabular}%"
        )
        parts.append(table_env(
            label,
            f"Task-wise combined score for {latex_escape(subject)}, reported as mean $\\pm$ sample standard deviation over seeds, with validation-selected $\\lambda^*$ reported as mean $\\pm$ sample standard deviation over the same seeds. Bold entries indicate the row-wise largest displayed mean score among Hyb-FPDE, AIME, SHAP, and LIME; ties are bolded jointly.",
            tabular,
            table_star=True,
        ))
    return "\n".join(parts)


def compute_winner_counts(taskwise: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = taskwise.pivot_table(index=["subject_area", "task_id"], columns="method_display", values="combined_mean").reset_index()
    rows = []
    for _, r in pivot.iterrows():
        displayed = {m: round(float(r[m]), 3) for m in METHOD_ORDER}
        best = max(displayed.values())
        for m, v in displayed.items():
            if v == best:
                rows.append({"subject_area": r["subject_area"], "task_id": int(r["task_id"]), "method_display": m})
    wins = pd.DataFrame(rows)
    overall = wins.groupby("method_display").size().reindex(METHOD_ORDER, fill_value=0).reset_index(name="best_count")
    subjects = [s for s in SUBJECT_ORDER if s in set(pivot["subject_area"])]
    by_subject = wins.groupby(["subject_area", "method_display"]).size().unstack(fill_value=0).reindex(index=subjects, columns=METHOD_ORDER, fill_value=0)
    task_counts = pivot.groupby("subject_area")["task_id"].nunique().reindex(by_subject.index)
    by_subject.insert(0, "tasks", task_counts.astype(int))
    return overall, by_subject.reset_index()


def make_winner_tables(overall: pd.DataFrame, by_subject: pd.DataFrame) -> Tuple[str, str]:
    rows = []
    for _, r in by_subject.iterrows():
        cells = [latex_escape(r["subject_area"]), str(int(r["tasks"]))] + [str(int(r[m])) for m in METHOD_ORDER]
        rows.append("        " + " & ".join(cells) + LATEX_ROW_END)
    subject_tabular = (
        "    \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l r rrrr@{}}\n"
        "        \\toprule\n"
        "        Subject area & Tasks & Hyb-FPDE & AIME & SHAP & LIME " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular*}"
    )
    subject_tex = table_env(
        "tab:subject_area_winner_counts",
        "Subject-area-wise task-best counts for the predefined four-method comparison. For each task, the method or methods with the largest displayed mean combined score after rounding to three decimals are counted. Ties are counted jointly, so row totals can exceed the number of tasks.",
        subject_tabular,
        table_star=True,
    )
    count_map = dict(zip(overall["method_display"], overall["best_count"]))
    overall_tabular = (
        "    \\begin{tabular}{lrrrr}\n"
        "        \\toprule\n"
        "        Method & Hyb-FPDE & AIME & SHAP & LIME " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        "        Best & " + " & ".join(str(int(count_map.get(m, 0))) for m in METHOD_ORDER) + LATEX_ROW_END + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}"
    )
    overall_tex = table_env(
        "tab:taskwise_winner_counts",
        "Task-wise best counts under the displayed-mean rule for all tasks in the predefined four-method comparison. Ties are counted jointly, so counts need not sum to the number of tasks.",
        overall_tabular,
        table_star=False,
    )
    return subject_tex, overall_tex


def make_sd_summary_table(taskwise: pd.DataFrame) -> str:
    sd = taskwise.groupby("method_display")["combined_sd"].agg(["mean", "median"]).reindex(METHOD_ORDER)
    tabular = (
        "    \\begin{tabular}{lrrrr}\n"
        "        \\toprule\n"
        "        Method & Hyb-FPDE & AIME & SHAP & LIME " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        "        Mean SD & " + " & ".join(fmt(sd.loc[m, "mean"], 3) for m in METHOD_ORDER) + LATEX_ROW_END + "\n"
        "        Median SD & " + " & ".join(fmt(sd.loc[m, "median"], 3) for m in METHOD_ORDER) + LATEX_ROW_END + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}"
    )
    return table_env(
        "tab:taskwise_sd_summary",
        "Average task-wise standard deviation of the combined score across seeds. Smaller values indicate lower seed-to-seed variation at the task level.",
        tabular,
        table_star=False,
    )


def compute_lambda_distribution(lambda_df: pd.DataFrame) -> pd.DataFrame:
    selected = selected_lambda_by_task_seed(lambda_df)
    if selected.empty:
        return pd.DataFrame(columns=["lambda", "count", "percentage"])
    counts = selected["selected_lambda"].value_counts().sort_index().reset_index()
    counts.columns = ["lambda", "count"]
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100.0
    return counts


def make_lambda_table(lambda_counts: pd.DataFrame) -> str:
    rows = []
    for _, r in lambda_counts.iterrows():
        rows.append(f"        {float(r['lambda']):.1f} & {int(r['count'])} & {float(r['percentage']):.1f}\\%" + LATEX_ROW_END)
    tabular = (
        "    \\begin{tabular}{ccc}\n"
        "        \\toprule\n"
        "        $\\lambda^*$ & Count & Percentage " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}"
    )
    return table_env(
        "tab:lambda_distribution",
        "Validation-selected hybrid weights for Hyb-FPDE over the task--seed pairs.",
        tabular,
        table_star=False,
    )


def holm_correction(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p_values[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted.tolist()


def compute_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    if wilcoxon is None:
        raise RuntimeError("scipy is required for Wilcoxon tests. Install scipy or skip statistical tests.")
    pivot = df.pivot_table(index=["task_id", "seed"], columns="method_display", values="combined_score")
    comparisons = [
        ("Hyb-FPDE", "AIME"),
        ("Hyb-FPDE", "SHAP"),
        ("Hyb-FPDE", "LIME"),
        ("AIME", "SHAP"),
        ("AIME", "LIME"),
        ("SHAP", "LIME"),
    ]
    rows = []
    pvals = []
    for ref, other in comparisons:
        diff = (pivot[ref] - pivot[other]).dropna()
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        ties = int((diff == 0).sum())
        if np.allclose(diff.to_numpy(), 0):
            p = 1.0
        else:
            res = wilcoxon(diff.to_numpy(), zero_method="wilcox", alternative="two-sided", mode="auto")
            p = float(res.pvalue)
        pvals.append(p)
        rows.append({"reference": ref, "compared": other, "mean_diff": float(diff.mean()), "wins": wins, "losses": losses, "ties": ties, "p_raw": p})
    for row, p in zip(rows, holm_correction(pvals)):
        row["p_holm"] = p
    return pd.DataFrame(rows)


def make_stat_tests_table(tests: pd.DataFrame) -> str:
    rows = []
    for _, r in tests.iterrows():
        cells = [latex_escape(r["reference"]), latex_escape(r["compared"]), fmt(r["mean_diff"], 4), str(int(r["wins"])), str(int(r["losses"])), str(int(r["ties"])), fmt_p_value(float(r["p_holm"]))]
        rows.append("        " + " & ".join(cells) + LATEX_ROW_END)
    tabular = (
        "    \\begin{tabular}{llccccc}\n"
        "        \\toprule\n"
        "        Reference & Compared method & Mean diff. & Wins & Losses & Ties & $p_{\\mathrm{Holm}}$ " + LATEX_ROW_END + "\n"
        "        \\midrule\n"
        + "\n".join(rows) + "\n"
        "        \\bottomrule\n"
        "    \\end{tabular}"
    )
    return table_env(
        "tab:stat_tests",
        "Descriptive paired Wilcoxon signed-rank tests on the combined score over task--seed pairs. Positive mean differences indicate that the reference method is better than the compared method.",
        tabular,
        table_star=True,
    )


def make_reproducibility_text(df: pd.DataFrame, lambda_counts: pd.DataFrame, configs: pd.DataFrame) -> str:
    n_tasks = df["task_id"].nunique()
    n_seeds = df["seed"].nunique()
    n_pairs = df[["task_id", "seed"]].drop_duplicates().shape[0]
    n_explanations_per_method = int(df[df["method_display"] == METHOD_ORDER[0]]["ok"].sum()) if "ok" in df.columns else int(n_pairs)
    seed_values = ", ".join(str(int(s)) for s in sorted(df["seed"].unique()))
    lambda_sentence = ""
    if not lambda_counts.empty:
        top = lambda_counts.sort_values("count", ascending=False).iloc[0]
        endpoints = int(lambda_counts[lambda_counts["lambda"].isin([0.0, 1.0])]["count"].sum())
        total = int(lambda_counts["count"].sum())
        lambda_sentence = (
            f" The most frequent selected hybrid weight was $\\lambda^*={float(top['lambda']):.1f}$ "
            f"({int(top['count'])}/{total}, {float(top['percentage']):.1f}\\%). "
            f"Endpoint weights $0.0$ or $1.0$ were selected in {endpoints}/{total} task--seed pairs."
        )
    cfg_cols = ["suite_id", "fold", "repeat", "sample", "n_explain", "n_val_select", "n_estimators", "learning_rate", "num_leaves", "aime_local_y"]
    cfg_bits = []
    if not configs.empty:
        for c in cfg_cols:
            if c in configs.columns and configs[c].nunique(dropna=False) == 1:
                cfg_bits.append(f"{c}={configs[c].iloc[0]}")
    cfg_text = ", ".join(cfg_bits)
    base = (
        "% Generated by build_reproducibility_tables.py\n"
        fr"The numerical tables were reconstructed from the per-run result archive rather than typed manually. "
        fr"The script reads each \texttt{{summary\_by\_task.csv}}, \texttt{{lambda\_distribution.csv}}, and \texttt{{run\_config.json}} file, attaches the seed from the run configuration, and verifies that the selected comparison contains {n_tasks} OpenML-CC18 tasks, {n_seeds} seeds ({seed_values}), and the four reported methods. "
        fr"The resulting archive contains {n_pairs} task--seed pairs and {n_explanations_per_method:,} completed explanations per method. "
        r"Subject-area-balanced scores are computed by averaging tasks within each subject area for each seed and method, then averaging the populated subject areas equally. Task-balanced scores are computed by averaging all tasks equally for each seed and method. Reported uncertainty is the sample standard deviation over seeds."
        f"{lambda_sentence}"
    )
    if cfg_text:
        base += f" The common run configuration was: {latex_escape(cfg_text)}."
    return base + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reproducibility tables from FPDE result archives.")
    parser.add_argument("--input", required=True, type=Path, help="Path to ALL_DATA zip archive or extracted result directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for generated CSV and TeX files.")
    parser.add_argument("--expected-tasks", type=int, default=72)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--no-validate", action="store_true", help="Skip completeness validation.")
    args = parser.parse_args(argv)

    loaded = load_results(args.input)
    df = prepare_summary(loaded.summary_by_task)
    if not args.no_validate:
        validate_completeness(df, args.expected_tasks, args.expected_seeds)

    out = args.output
    csv_dir = out / "processed_csv"
    tex_dir = out / "generated_tex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    subject_seed, subject_balanced_seed, task_balanced_seed = aggregate_seed_level(df)
    subject_balanced_summary = summarize_over_seeds(subject_balanced_seed, ["method_display"])
    task_balanced_summary = summarize_over_seeds(task_balanced_seed, ["method_display"])
    selected_lambda = selected_lambda_by_task_seed(loaded.lambda_distribution)
    tw = taskwise_summary(df, selected_lambda)
    overall_wins, subject_wins = compute_winner_counts(tw)
    lambda_counts = compute_lambda_distribution(loaded.lambda_distribution)
    tests = compute_pairwise_tests(df)

    # Audit CSV outputs.
    df.to_csv(csv_dir / "summary_by_task_seed_filtered.csv", index=False, lineterminator="\n")
    subject_balanced_seed.to_csv(csv_dir / "subject_balanced_seed_method.csv", index=False, lineterminator="\n")
    task_balanced_seed.to_csv(csv_dir / "task_balanced_seed_method.csv", index=False, lineterminator="\n")
    tw.to_csv(csv_dir / "taskwise_combined_score_mean_sd.csv", index=False, lineterminator="\n")
    lambda_counts.to_csv(csv_dir / "lambda_selected_distribution.csv", index=False, lineterminator="\n")
    tests.to_csv(csv_dir / "paired_wilcoxon_tests.csv", index=False, lineterminator="\n")
    subject_wins.to_csv(csv_dir / "subject_area_winner_counts.csv", index=False, lineterminator="\n")
    overall_wins.to_csv(csv_dir / "taskwise_winner_counts.csv", index=False, lineterminator="\n")

    # LaTeX outputs.
    write_text(
        tex_dir / "main_results_subject_area_balanced.tex",
        make_main_results_table(
            subject_balanced_summary,
            "tab:main_results",
            "Subject-area-balanced OpenML-CC18 results for the predefined four-method comparison. For each seed and method, tasks are first averaged within each populated subject area and then the subject areas are averaged equally. Values are mean $\\pm$ sample standard deviation over seeds.",
        ),
    )
    write_text(
        tex_dir / "task_weighted_results.tex",
        make_main_results_table(
            task_balanced_summary,
            "tab:task_weighted_results",
            "Task-balanced OpenML-CC18 results for the predefined four-method comparison. For each seed and method, the metric is averaged equally over tasks without subject-area balancing. Values are mean $\\pm$ sample standard deviation over seeds.",
        ),
    )
    write_text(tex_dir / "subject_area_combined_scores.tex", make_subject_area_table(subject_seed))
    write_text(tex_dir / "stat_tests.tex", make_stat_tests_table(tests))
    subject_win_tex, overall_win_tex = make_winner_tables(overall_wins, subject_wins)
    write_text(tex_dir / "subject_area_winner_counts.tex", subject_win_tex)
    write_text(tex_dir / "taskwise_winner_counts.tex", overall_win_tex)
    write_text(tex_dir / "taskwise_sd_summary.tex", make_sd_summary_table(tw))
    write_text(tex_dir / "lambda_distribution.tex", make_lambda_table(lambda_counts))
    write_text(tex_dir / "appendix_taskwise_combined_scores_by_subject.tex", make_taskwise_appendix_tables(tw))
    write_text(tex_dir / "reproducibility_processing_text.tex", make_reproducibility_text(df, lambda_counts, loaded.run_configs))

    outputs = sorted(
        str(p.relative_to(out)).replace("\\", "/")
        for p in out.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    outputs.append("manifest.json")
    manifest = {
        "input": str(args.input),
        "n_summary_rows_raw": int(len(loaded.summary_by_task)),
        "n_summary_rows_reported_methods": int(len(df)),
        "n_tasks": int(df["task_id"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "methods": METHOD_ORDER,
        "outputs": outputs,
    }
    write_text(out / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
