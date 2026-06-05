from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bayesian_fpde.stats import bootstrap_confidence_intervals, method_tests
from bayesian_fpde.utils import ensure_dirs, write_csv


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
    write_csv(tests, results_dir / "statistical_tests.csv")
    write_csv(effects, results_dir / "effect_sizes.csv")
    write_csv(ci, results_dir / "bootstrap_confidence_intervals.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
