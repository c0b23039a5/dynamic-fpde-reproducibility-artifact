from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.stats import bootstrap_confidence_intervals, method_tests
from bayesian_fpde.utils import ensure_dirs, git_commit, now_iso, read_csv_preserve_metadata
from experiments.common import _openml_summary
from experiments.public_aggregate import write_public_experiment_summaries


def _unique_nonempty(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted({str(v) for v in df[column] if pd.notna(v) and str(v) != ""})


def _write_hash_warning(logs_dir: Path, message: str) -> None:
    ensure_dirs(logs_dir)
    with open(logs_dir / "aggregate_results.log", "a", encoding="utf-8") as fh:
        fh.write(f"{now_iso()} WARNING {message}\n")
    print(f"WARNING: {message}")


def _read_csvs(results_dir: Path) -> pd.DataFrame:
    frames = []
    for name in [
        "openml_metrics.csv",
        "public_uncertainty_validation.csv",
        "faithfulness_metrics.csv",
        "stability_metrics.csv",
        "training_size_uncertainty.csv",
        "synthetic_calibration_summary.csv",
        "ablation_metrics.csv",
    ]:
        path = results_dir / name
        if path.exists():
            frames.append(read_csv_preserve_metadata(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def combine_openml_task_outputs(root: str | Path, out: str | Path) -> None:
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["openml_global_summary.csv", "openml_metrics.csv", "openml_runtime.csv"]:
        frames = [read_csv_preserve_metadata(path) for path in root.rglob(f"results/{name}") if path.stat().st_size > 0]
        if frames:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(out / name, index=False, lineterminator="\n")
    local_frames = []
    for path in root.rglob("results/openml_local_explanations.parquet"):
        try:
            local_frames.append(pd.read_parquet(path))
        except Exception:
            pass
    for path in root.rglob("results/openml_local_explanations.parquet.csv"):
        local_frames.append(read_csv_preserve_metadata(path))
    if local_frames:
        pd.concat(local_frames, ignore_index=True, sort=False).to_parquet(out / "openml_local_explanations.parquet", index=False)


def combine_synthetic_task_outputs(root: str | Path, results_dir: str | Path, figures_dir: str | Path) -> None:
    root = Path(root)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    ensure_dirs(results_dir, figures_dir)
    for name in ["synthetic_calibration.csv", "synthetic_calibration_summary.csv", "synthetic_sign_calibration_bins.csv"]:
        frames = [read_csv_preserve_metadata(path) for path in root.rglob(f"results/{name}") if path.stat().st_size > 0]
        if frames:
            pd.concat(frames, ignore_index=True, sort=False).to_csv(results_dir / name, index=False, lineterminator="\n")
    summary_path = results_dir / "synthetic_calibration_summary.csv"
    summary = read_csv_preserve_metadata(summary_path) if summary_path.exists() else pd.DataFrame()
    save_line_plot(summary, x="n_samples", y="coverage_95", group="method", path=figures_dir / "synthetic_coverage_vs_n.png", title="Synthetic coverage vs n")
    save_line_plot(summary, x="n_samples", y="mean_ci_width", group="method", path=figures_dir / "synthetic_ci_width_vs_n.png", title="Synthetic CI width vs n")
    save_line_plot(summary, x="n_samples", y="sign_ece", group="method", path=figures_dir / "synthetic_sign_ece_vs_n.png", title="Synthetic sign ECE vs n")
    save_line_plot(summary, x="n_samples", y="top_k_precision", group="method", path=figures_dir / "synthetic_topk_precision.png", title="Synthetic top-k precision")


def _hash_metadata(df: pd.DataFrame, logs_dir: Path) -> dict[str, object]:
    mode = ""
    if not df.empty and "mode" in df.columns:
        mode = next((str(v) for v in df["mode"] if pd.notna(v) and str(v) != ""), "")
    experiment_hashes = _unique_nonempty(df, "experiment_config_hash") or _unique_nonempty(df, "config_hash")
    workflow_ids = _unique_nonempty(df, "workflow_run_id")
    runner_hashes = _unique_nonempty(df, "runner_invocation_hash") or _unique_nonempty(df, "run_config_hash")
    job_hashes = _unique_nonempty(df, "job_config_hash")
    workflow_attempts = _unique_nonempty(df, "workflow_run_attempt")
    workflow_names = _unique_nonempty(df, "workflow_name")
    workflow_refs = _unique_nonempty(df, "workflow_ref")
    workflow_shas = _unique_nonempty(df, "workflow_sha")
    git_commits = _unique_nonempty(df, "git_commit")
    experiment_hash_consistent = len(experiment_hashes) <= 1
    workflow_consistent = len(workflow_ids) <= 1
    if not experiment_hash_consistent:
        _write_hash_warning(logs_dir, f"aggregate input contains {len(experiment_hashes)} experiment_config_hash values; marking experiment_config_hash as multiple")
    experiment_hash_value = experiment_hashes[0] if experiment_hash_consistent and experiment_hashes else ("multiple" if experiment_hashes else "")
    runner_hash_value = runner_hashes[0] if len(runner_hashes) == 1 else ("multiple" if runner_hashes else "")
    workflow_sha_value = workflow_shas[0] if len(workflow_shas) == 1 else ("multiple" if workflow_shas else "")
    git_commit_value = git_commits[0] if len(git_commits) == 1 else (git_commit() if not git_commits else "multiple")
    return {
        "mode": mode,
        "config_hash": experiment_hash_value,
        "experiment_config_hash": experiment_hash_value,
        "n_experiment_config_hashes": len(experiment_hashes),
        "experiment_config_hash_consistent": bool(experiment_hash_consistent),
        "experiment_config_hashes": ",".join(experiment_hashes) if len(experiment_hashes) <= 10 else "",
        "workflow_run_id": workflow_ids[0] if workflow_consistent and workflow_ids else ("multiple" if workflow_ids else ""),
        "workflow_run_attempt": workflow_attempts[0] if len(workflow_attempts) == 1 else ("multiple" if workflow_attempts else ""),
        "workflow_name": workflow_names[0] if len(workflow_names) == 1 else ("multiple" if workflow_names else ""),
        "workflow_ref": workflow_refs[0] if len(workflow_refs) == 1 else ("multiple" if workflow_refs else ""),
        "workflow_sha": workflow_sha_value,
        "n_workflow_run_ids": len(workflow_ids),
        "workflow_run_id_consistent": bool(workflow_consistent),
        "workflow_run_ids": ",".join(workflow_ids) if len(workflow_ids) <= 10 else "",
        "runner_invocation_hash": runner_hash_value,
        "run_config_hash": runner_hash_value,
        "n_runner_invocation_hashes": len(runner_hashes),
        "runner_invocation_hashes": ",".join(runner_hashes) if len(runner_hashes) <= 10 else "",
        "n_run_config_hashes": len(runner_hashes),
        "run_config_hash_consistent": bool(len(runner_hashes) <= 1),
        "run_config_hashes": ",".join(runner_hashes) if len(runner_hashes) <= 10 else "",
        "n_job_config_hashes": len(job_hashes),
        "job_config_hashes": ",".join(job_hashes) if len(job_hashes) <= 10 else "",
        "timestamp": now_iso(),
        "git_commit": git_commit_value,
    }


def _safe_tests(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"] == "ok"] if not df.empty and "status" in df.columns else df
    tests, effects = method_tests(ok, metric=metric)
    ci = bootstrap_confidence_intervals(ok, metric=metric, n_bootstrap=200, unit_level="dataset_seed")
    if tests.empty:
        tests = pd.DataFrame([{"test": "not_run", "metric": metric, "method_a": "", "method_b": "", "statistic": float("nan"), "p_value": float("nan"), "p_holm": float("nan"), "status": "skipped", "error_message": "insufficient paired method data for statistical tests"}])
    if effects.empty:
        effects = pd.DataFrame([{"metric": metric, "method_a": "", "method_b": "", "cliffs_delta": float("nan"), "status": "skipped", "error_message": "insufficient paired method data for effect sizes"}])
    return tests, effects, ci


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Bayesian-FPDE experiment results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    logs_dir = results_dir.parent / "logs"
    ensure_dirs(results_dir, figures_dir)

    df = _read_csvs(results_dir)
    openml_path = results_dir / "openml_metrics.csv"
    if openml_path.exists():
        openml_metrics = read_csv_preserve_metadata(openml_path)
        _openml_summary(openml_metrics, ["dataset_name", "task_id", "seed", "method"]).to_csv(results_dir / "openml_seed_summary.csv", index=False, lineterminator="\n")
        _openml_summary(openml_metrics, ["dataset_name", "task_id", "method"]).to_csv(results_dir / "openml_global_summary.csv", index=False, lineterminator="\n")
        _openml_summary(openml_metrics, ["method"]).to_csv(results_dir / "openml_method_summary.csv", index=False, lineterminator="\n")
    write_public_experiment_summaries(results_dir, figures_dir)

    metric = "combined_score" if "combined_score" in df.columns else "deletion_drop_auc"
    if "combined_score" not in df.columns and {"deletion_drop_auc", "insertion_auc"}.issubset(df.columns):
        df["combined_score"] = (df["deletion_drop_auc"] + df["insertion_auc"]) / 2.0
        metric = "combined_score"
    tests, effects, ci = _safe_tests(df, metric=metric)
    common = _hash_metadata(df, logs_dir)
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
