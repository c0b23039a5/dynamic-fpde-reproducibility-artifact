from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common import _openml_summary
from bayesian_fpde.stats import bootstrap_confidence_intervals, method_tests
from bayesian_fpde.utils import ensure_dirs, git_commit, now_iso


def _read_csvs(results_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ["openml_metrics.csv", "faithfulness_metrics.csv", "synthetic_calibration_summary.csv", "stability_metrics.csv", "training_size_uncertainty.csv", "ablation_metrics.csv"]:
        path = results_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Bayesian-FPDE experiment results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dirs(results_dir, figures_dir)
    df = _read_csvs(results_dir)
    openml_path = results_dir / "openml_metrics.csv"
    if openml_path.exists():
        openml_metrics = pd.read_csv(openml_path)
        _openml_summary(openml_metrics, ["dataset_name", "task_id", "seed", "method"]).to_csv(results_dir / "openml_seed_summary.csv", index=False, lineterminator="\n")
        _openml_summary(openml_metrics, ["dataset_name", "task_id", "method"]).to_csv(results_dir / "openml_global_summary.csv", index=False, lineterminator="\n")
        _openml_summary(openml_metrics, ["method"]).to_csv(results_dir / "openml_method_summary.csv", index=False, lineterminator="\n")
    metric = "combined_score" if "combined_score" in df.columns else "deletion_drop_auc"
    if "combined_score" not in df.columns and {"deletion_drop_auc", "insertion_auc"}.issubset(df.columns):
        df["combined_score"] = (df["deletion_drop_auc"] + df["insertion_auc"]) / 2.0
        metric = "combined_score"
    ok = df[df["status"] == "ok"] if not df.empty and "status" in df.columns else df
    tests, effects = method_tests(ok, metric=metric)
    ci = bootstrap_confidence_intervals(ok, metric=metric, n_bootstrap=200)
    if tests.empty:
        tests = pd.DataFrame(
            [
                {
                    "test": "not_run",
                    "metric": metric,
                    "method_a": "",
                    "method_b": "",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "p_holm": float("nan"),
                    "status": "skipped",
                    "error_message": "insufficient paired method data for statistical tests",
                }
            ]
        )
    if effects.empty:
        effects = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "method_a": "",
                    "method_b": "",
                    "cliffs_delta": float("nan"),
                    "status": "skipped",
                    "error_message": "insufficient paired method data for effect sizes",
                }
            ]
        )
    mode = ""
    config_hash = ""
    if not df.empty:
        if "mode" in df.columns:
            mode = next((str(v) for v in df["mode"] if pd.notna(v) and str(v) != ""), "")
        if "config_hash" in df.columns:
            config_hash = next((str(v) for v in df["config_hash"] if pd.notna(v) and str(v) != ""), "")
    common = {
        "mode": mode,
        "config_hash": config_hash,
        "timestamp": now_iso(),
        "git_commit": git_commit(),
    }
    for frame, default_status in [(tests, "ok"), (effects, "ok"), (ci, "ok")]:
        for key, value in common.items():
            frame[key] = value
        if "status" not in frame.columns:
            frame["status"] = default_status
        if "error_message" not in frame.columns:
            frame["error_message"] = ""
        if "metric_direction" not in frame.columns:
            frame["metric_direction"] = "higher_is_better"

    tests.to_csv(results_dir / "statistical_tests.csv", index=False, lineterminator="\n")
    effects.to_csv(results_dir / "effect_sizes.csv", index=False, lineterminator="\n")
    ci.to_csv(results_dir / "bootstrap_confidence_intervals.csv", index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
