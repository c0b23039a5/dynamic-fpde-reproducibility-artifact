from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from bayesian_fpde.utils import read_csv_preserve_metadata


ROOT = Path(__file__).resolve().parents[1]


def _row(
    *,
    method: str,
    n_samples: int,
    n_features: int,
    class_separation: str,
    class_balance: str,
    seed: int,
    coverage_95: float,
    warning: bool,
) -> dict[str, object]:
    return {
        "method": method,
        "dataset_name": "synthetic_gaussian",
        "task_id": "",
        "seed": str(seed),
        "fold": "synthetic_random_split",
        "split_id": "synthetic_random_split",
        "mode": "full",
        "config_hash": "experimentabc",
        "experiment_config_hash": "experimentabc",
        "workflow_run_id": "workflow1",
        "workflow_run_attempt": "1",
        "workflow_name": "synthetic",
        "workflow_ref": "refs/heads/main",
        "workflow_sha": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
        "runner_invocation_hash": f"runner{seed}",
        "run_config_hash": f"runner{seed}",
        "job_config_hash": f"job{seed}",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "git_commit": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
        "status": "ok",
        "error_message": "",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": 3,
        "n_classes": 2,
        "class_separation": class_separation,
        "feature_correlation": "independent",
        "class_balance": class_balance,
        "posterior_samples": 50,
        "n_explain": 5,
        "requested_n_explain": 5,
        "selection_policy": "correctly_classified_only",
        "top_k": 5,
        "tau": 0.0,
        "lambda_hyb": 0.5,
        "effective_n_explain": 2 if warning else 5,
        "n_explanation_rows": (2 if warning else 5) * n_features,
        "n_unique_explanation_units": 2 if warning else 5,
        "low_effective_n_explain_warning": warning,
        "model": "random_forest",
        "coverage_95": coverage_95,
        "mean_ci_width": 0.8 + 0.01 * n_features,
        "median_ci_width": 0.7 + 0.01 * n_features,
        "sign_accuracy": 0.7,
        "top_k_precision": 0.6,
        "spearman_rank_correlation": 0.5,
        "kendall_tau": 0.4,
        "sign_brier_score": 0.2,
        "sign_ece": 0.1,
        "sign_accuracy_at_confidence_0_8": 0.75,
        "sign_accuracy_at_confidence_0_9": 0.8,
    }


def test_analyze_synthetic_full_creates_tables_and_figures(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    rows = [
        _row(method="bayesian_diff_fpde", n_samples=50, n_features=10, class_separation="small", class_balance="balanced", seed=0, coverage_95=0.92, warning=False),
        _row(method="bayesian_diff_fpde", n_samples=100, n_features=10, class_separation="small", class_balance="imbalanced", seed=1, coverage_95=0.88, warning=True),
        _row(method="bayesian_hyb_fpde", n_samples=50, n_features=50, class_separation="large", class_balance="balanced", seed=0, coverage_95=0.97, warning=False),
        _row(method="bayesian_hyb_fpde", n_samples=100, n_features=50, class_separation="large", class_balance="imbalanced", seed=1, coverage_95=0.96, warning=False),
    ]
    input_path = results / "synthetic_calibration_summary.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False, lineterminator="\n")

    cmd = [
        sys.executable,
        "-m",
        "experiments.analyze_synthetic_full",
        "--input",
        str(input_path),
        "--results-dir",
        str(results),
        "--figures-dir",
        str(figures),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert completed.returncode == 0, completed.stderr

    expected_csvs = [
        "synthetic_full_method_summary.csv",
        "synthetic_full_by_n_samples.csv",
        "synthetic_full_by_n_features.csv",
        "synthetic_full_by_class_separation.csv",
        "synthetic_full_by_effective_warning.csv",
        "synthetic_full_low_effective_warning_summary.csv",
    ]
    for name in expected_csvs:
        path = results / name
        assert path.exists(), name
        df = read_csv_preserve_metadata(path)
        assert not df.empty, name
        assert "low_effective_warning_count" in df.columns
        assert "low_effective_warning_rate" in df.columns

    method_summary = read_csv_preserve_metadata(results / "synthetic_full_method_summary.csv")
    assert {"coverage_gap_from_95", "abs_coverage_gap_from_95", "undercoverage_flag_count", "undercoverage_flag_rate"}.issubset(method_summary.columns)
    diff = method_summary[method_summary["method"] == "bayesian_diff_fpde"].iloc[0]
    assert abs(float(diff["coverage_95_mean"]) - 0.90) < 1e-12
    assert abs(float(diff["coverage_gap_from_95"]) - (-0.05)) < 1e-12
    assert int(diff["undercoverage_count"]) == 1

    warning_summary = read_csv_preserve_metadata(results / "synthetic_full_low_effective_warning_summary.csv")
    assert {"warning_group", "warning_level", "low_effective_warning_count", "low_effective_warning_rate"}.issubset(warning_summary.columns)
    assert set(warning_summary["warning_group"]).issuperset({"n_samples", "class_separation", "class_balance", "method"})
    n100 = warning_summary[(warning_summary["warning_group"] == "n_samples") & (warning_summary["warning_level"].astype(str) == "100")].iloc[0]
    assert abs(float(n100["low_effective_warning_rate"]) - 0.5) < 1e-12

    expected_figures = [
        "synthetic_full_coverage_by_method.png",
        "synthetic_full_coverage_by_n_samples.png",
        "synthetic_full_ci_width_by_n_samples.png",
        "synthetic_full_sign_ece_by_n_samples.png",
        "synthetic_full_topk_precision_by_n_samples.png",
        "synthetic_full_effective_n_explain_by_n_samples.png",
        "synthetic_full_warning_rate_heatmap.png",
    ]
    for name in expected_figures:
        path = figures / name
        assert path.exists(), name
        assert path.stat().st_size > 0
