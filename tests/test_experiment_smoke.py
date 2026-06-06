from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from bayesian_fpde.utils import read_csv_preserve_metadata
from experiments.aggregate_results import combine_openml_task_outputs


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA = {
    "method",
    "dataset_name",
    "seed",
    "mode",
    "config_hash",
    "experiment_config_hash",
    "workflow_run_id",
    "workflow_run_attempt",
    "workflow_name",
    "workflow_ref",
    "workflow_sha",
    "runner_invocation_hash",
    "run_config_hash",
    "job_config_hash",
    "timestamp",
    "git_commit",
    "status",
    "error_message",
}


def _run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    assert result.returncode == 0, result.stderr


def _csv(path: str, required: set[str]) -> pd.DataFrame:
    target = ROOT / path
    assert target.exists(), path
    df = read_csv_preserve_metadata(target)
    assert not df.empty, path
    assert required.issubset(df.columns), f"{path} missing {sorted(required - set(df.columns))}"
    return df


def test_all_smoke_commands_and_outputs():
    for dirname in ["results", "figures", "logs"]:
        path = ROOT / dirname
        path.mkdir(exist_ok=True)
        for child in path.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    _run("experiments.run_synthetic_calibration", "--config", "configs/synthetic.yaml", "--mode", "smoke")
    synthetic = _csv(
        "results/synthetic_calibration_summary.csv",
        REQUIRED_METADATA
        | {
            "n_samples",
            "n_features",
            "n_informative",
            "n_classes",
            "class_separation",
            "feature_correlation",
            "class_balance",
            "posterior_samples",
            "n_explain",
            "requested_n_explain",
            "effective_n_explain",
            "n_explanation_rows",
            "n_unique_explanation_units",
            "selection_policy",
            "low_effective_n_explain_warning",
            "top_k",
            "tau",
            "lambda_hyb",
            "coverage_95",
            "mean_ci_width",
            "median_ci_width",
            "sign_accuracy",
            "sign_brier_score",
            "sign_ece",
            "sign_accuracy_at_confidence_0_8",
            "sign_accuracy_at_confidence_0_9",
            "top_k_precision",
            "spearman_rank_correlation",
            "kendall_tau",
        },
    )
    synthetic_detail = _csv("results/synthetic_calibration.csv", REQUIRED_METADATA | {"feature", "feature_index", "attribution", "true_attribution"})
    synthetic_bins = _csv("results/synthetic_sign_calibration_bins.csv", REQUIRED_METADATA | {"bin_id", "bin_feature_count", "bin_weight", "mean_confidence", "sign_accuracy", "n_features"})
    assert set(synthetic_bins["n_features"].astype(int).unique()).issubset({8})
    assert synthetic_bins["bin_feature_count"].astype(int).ge(0).all()
    assert set(synthetic_bins["bin_id"].astype(int).unique()) == set(range(10))
    ok_synthetic = synthetic[synthetic["status"] == "ok"].copy()
    assert ok_synthetic["effective_n_explain"].astype(int).le(ok_synthetic["requested_n_explain"].astype(int)).all()
    assert ok_synthetic["effective_n_explain"].astype(int).gt(0).all()
    condition_cols = ["method", "seed", "n_samples", "n_features", "n_informative", "n_classes", "class_separation", "feature_correlation", "class_balance"]
    detail_counts = synthetic_detail.groupby(condition_cols, dropna=False).size().reset_index(name="detail_rows")
    merged_counts = ok_synthetic.merge(detail_counts, on=condition_cols, how="left")
    assert (merged_counts["n_explanation_rows"].astype(int) == merged_counts["detail_rows"].astype(int)).all()
    assert (merged_counts["n_unique_explanation_units"].astype(int) == merged_counts["effective_n_explain"].astype(int)).all()
    assert (ROOT / "figures" / "synthetic_coverage_vs_n.png").exists()
    assert (ROOT / "figures" / "synthetic_ci_width_vs_n.png").exists()
    assert (ROOT / "figures" / "synthetic_sign_ece_vs_n.png").exists()
    assert (ROOT / "figures" / "synthetic_topk_precision.png").exists()

    _run("experiments.run_openml_benchmark", "--config", "configs/openml_cc18.yaml", "--mode", "smoke")
    openml = _csv(
        "results/openml_metrics.csv",
        REQUIRED_METADATA | {"task_id", "fold", "split_id", "faithfulness_correlation", "faithfulness_delta_mean", "dependency_available", "explanation_model_calls", "evaluation_model_calls", "total_model_calls", "number_of_model_calls", "combined_score", "metric_direction", "background_size", "max_background"},
    )
    ok_calls = openml[openml["status"] == "ok"]
    assert (ok_calls["number_of_model_calls"] == ok_calls["total_model_calls"]).all()
    assert (ok_calls["total_model_calls"] == ok_calls["explanation_model_calls"] + ok_calls["evaluation_model_calls"]).all()
    assert ok_calls["evaluation_model_calls"].gt(0).all()
    assert openml["experiment_config_hash"].nunique() == 1
    assert openml["config_hash"].nunique() == 1
    assert (openml["config_hash"] == openml["experiment_config_hash"]).all()
    assert openml["runner_invocation_hash"].nunique() >= 1
    assert (ROOT / "results" / "openml_local_explanations.parquet").exists() or (ROOT / "results" / "openml_local_explanations.parquet.csv").exists()
    seed_summary = _csv("results/openml_seed_summary.csv", {"dataset_name", "task_id", "seed", "method", "n_rows", "n_explanation_rows", "n_unique_explanation_units", "mean_explanation_units_per_dataset_seed", "mean_deletion_auc", "n_experiment_config_hashes", "experiment_config_hash_consistent", "n_runner_invocation_hashes", "n_job_config_hashes", "mean_runtime_seconds_all_rows", "mean_runtime_seconds_ok_only"})
    _csv("results/openml_global_summary.csv", {"dataset_name", "task_id", "method", "n_rows", "n_explanation_rows", "n_unique_explanation_units", "mean_deletion_auc", "n_experiment_config_hashes", "experiment_config_hash_consistent"})
    method_summary = _csv("results/openml_method_summary.csv", {"method", "n_rows", "n_datasets", "n_explanation_rows", "n_unique_explanation_units", "mean_deletion_auc", "mean_runtime_seconds_all_rows", "mean_runtime_seconds_ok_only"})
    assert seed_summary["experiment_config_hash_consistent"].all()
    skipped = openml[openml["method"].isin(["shap", "lime", "aime"]) & (openml["status"] == "skipped")]
    assert not skipped.empty
    assert skipped["error_message"].astype(str).str.len().gt(0).all()
    bayes_skipped = openml[openml["method"].isin(["bayesshap", "bayeslime", "bayesian_aime"])]
    assert set(bayes_skipped["status"]) == {"skipped"}
    assert set(bayes_skipped["error_message"]) == {
        "BayesSHAP adapter is not implemented in this artifact",
        "BayesLIME adapter is not implemented in this artifact",
        "Bayesian-AIME adapter is not implemented in this artifact",
    }
    assert bayes_skipped["dependency_available"].notna().all()
    assert bayes_skipped[["explanation_model_calls", "evaluation_model_calls", "total_model_calls", "number_of_model_calls"]].eq(0).all().all()
    assert bayes_skipped[["deletion_auc", "insertion_auc", "faithfulness_correlation", "combined_score"]].isna().all().all()
    bayes_summary = method_summary[method_summary["method"].isin(["bayesshap", "bayeslime", "bayesian_aime"])]
    assert bayes_summary["mean_runtime_seconds_all_rows"].notna().all()
    assert bayes_summary["mean_runtime_seconds_ok_only"].isna().all()
    shap_ok = openml[(openml["method"] == "shap") & (openml["status"] == "ok")]
    if not shap_ok.empty and shap_ok["n_train"].max() > 1:
        assert shap_ok["background_size"].gt(1).all()
    lime_ok = openml[(openml["method"] == "lime") & (openml["status"] == "ok")]
    if not lime_ok.empty:
        assert (lime_ok["background_size"] == lime_ok["n_train"]).all()

    _run("experiments.run_stability", "--config", "configs/openml_cc18.yaml", "--mode", "smoke")
    _csv("results/stability_metrics.csv", REQUIRED_METADATA | {"mean_spearman_between_runs", "top_k_jaccard_between_runs"})

    _run("experiments.run_faithfulness", "--config", "configs/openml_cc18.yaml", "--mode", "smoke")
    _csv("results/faithfulness_metrics.csv", REQUIRED_METADATA | {"deletion_auc", "insertion_auc", "faithfulness_correlation"})

    _run("experiments.run_training_size_uncertainty", "--config", "configs/synthetic.yaml", "--mode", "smoke")
    _csv("results/training_size_uncertainty.csv", REQUIRED_METADATA | {"training_size", "mean_ci_width", "sign_ece"})

    _run("experiments.run_ablation", "--config", "configs/ablation.yaml", "--mode", "smoke")
    _csv("results/ablation_metrics.csv", REQUIRED_METADATA | {"ablation", "mean_ci_width", "deletion_auc"})

    _run("experiments.run_case_studies", "--config", "configs/case_study.yaml", "--mode", "smoke")
    _csv("results/case_study_breast_cancer.csv", REQUIRED_METADATA | {"posterior_mean", "ci_lower_95", "ci_upper_95"})

    _run("experiments.aggregate_results", "--results-dir", "results", "--figures-dir", "figures")
    _csv("results/statistical_tests.csv", {"test", "metric"})
    _csv("results/effect_sizes.csv", {"metric", "method_a", "method_b"})
    _csv("results/bootstrap_confidence_intervals.csv", {"method", "metric", "mean"})

    assert (ROOT / "figures" / "openml_faithfulness_boxplot.png").exists()
    assert (ROOT / "figures" / "ablation_lambda.png").exists()


def test_synthetic_full_config_contains_paper_grid():
    cfg = yaml.safe_load((ROOT / "configs" / "synthetic_full.yaml").read_text(encoding="utf-8"))
    grid = cfg["grid"]
    assert grid["n_samples"] == [50, 100, 500, 1000]
    assert grid["n_features"] == [10, 50, 100]
    assert grid["n_informative"] == [3, 5, 10]
    assert grid["class_separation"] == ["small", "medium", "large"]
    assert grid["feature_correlation"] == ["independent", "correlated"]
    assert grid["class_balance"] == ["balanced", "imbalanced"]
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert "full" in cfg["modes"]
    pilot = cfg["modes"]["pilot"]
    assert pilot["seeds"] == [0, 1]
    assert pilot["posterior_samples"] == 200
    assert pilot["n_explain"] == 20
    assert pilot["grid"]["n_samples"] == [50, 100, 500]
    assert pilot["grid"]["n_features"] == [10, 50]
    assert pilot["grid"]["n_informative"] == [3, 5]
    assert pilot["grid"]["class_separation"] == ["small", "medium"]
    assert pilot["grid"]["feature_correlation"] == ["independent", "correlated"]
    assert pilot["grid"]["class_balance"] == ["balanced", "imbalanced"]


def test_openml_raw_metrics_have_single_experiment_hash_and_job_hashes_vary(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    logs = tmp_path / "logs"
    config = tmp_path / "openml_local_multi_seed.yaml"
    config.write_text(
        "\n".join(
            [
                f"results_dir: {results.as_posix()}",
                f"figures_dir: {figures.as_posix()}",
                f"logs_dir: {logs.as_posix()}",
                "local_smoke: true",
                "seeds: [0, 1]",
                "n_explain: 2",
                "posterior_samples: 12",
                "bootstrap_samples: 4",
                "top_k: 3",
                "tau: 0.0",
                "lambda_hyb: 0.5",
                "model: random_forest",
                "methods:",
                "  - hyb_fpde",
                "  - bayesian_hyb_fpde",
                "  - bayesshap",
                "  - bayeslime",
                "  - bayesian_aime",
                "modes:",
                "  smoke: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _run("experiments.run_openml_benchmark", "--config", str(config), "--mode", "smoke")
    metrics = read_csv_preserve_metadata(results / "openml_metrics.csv")
    assert metrics["experiment_config_hash"].nunique() == 1
    assert metrics["config_hash"].nunique() == 1
    assert (metrics["config_hash"] == metrics["experiment_config_hash"]).all()
    assert metrics["runner_invocation_hash"].nunique() == 1
    assert (metrics["run_config_hash"] == metrics["runner_invocation_hash"]).all()
    assert metrics["job_config_hash"].nunique() >= 1
    job_units = metrics[["dataset_name", "task_id", "seed", "fold"]].drop_duplicates()
    if len(job_units) > 1:
        assert metrics["job_config_hash"].nunique() > 1

    bayes_skipped = metrics[metrics["method"].isin(["bayesshap", "bayeslime", "bayesian_aime"])]
    assert set(bayes_skipped["status"]) == {"skipped"}
    assert set(bayes_skipped["error_message"]) == {
        "BayesSHAP adapter is not implemented in this artifact",
        "BayesLIME adapter is not implemented in this artifact",
        "Bayesian-AIME adapter is not implemented in this artifact",
    }
    assert bayes_skipped[["explanation_model_calls", "evaluation_model_calls", "total_model_calls", "number_of_model_calls"]].eq(0).all().all()
    assert bayes_skipped[["deletion_auc", "insertion_auc", "faithfulness_correlation", "combined_score"]].isna().all().all()

    _run("experiments.aggregate_results", "--results-dir", str(results), "--figures-dir", str(figures))
    summary = read_csv_preserve_metadata(results / "openml_method_summary.csv")
    assert set(summary["n_experiment_config_hashes"]) == {1}
    assert summary["experiment_config_hash_consistent"].all()
    assert set(summary["experiment_config_hash"]) != {"multiple"}
    assert set(summary["config_hash"]) != {"multiple"}


def test_openml_artifact_combine_preserves_sha_metadata_strings(tmp_path: Path):
    sha = "2777e761b10ba6431735c0f7fb2fb3b05ab08559"
    root = tmp_path / "bayesian_artifacts"
    task_results = root / "task_31_seed_0" / "results"
    task_results.mkdir(parents=True)
    out = tmp_path / "results"
    row = {
        "method": "hyb_fpde",
        "dataset_name": "dummy",
        "task_id": "31",
        "seed": "0",
        "fold": "fold0",
        "split_id": "fold0",
        "mode": "medium",
        "config_hash": "experimentabc",
        "experiment_config_hash": "experimentabc",
        "workflow_run_id": "123456789",
        "workflow_run_attempt": "1",
        "workflow_name": "Bayesian OpenML experiment",
        "workflow_ref": "refs/heads/main",
        "workflow_sha": sha,
        "runner_invocation_hash": "runnerabc",
        "run_config_hash": "runnerabc",
        "job_config_hash": "jobabc",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "git_commit": sha,
        "status": "ok",
        "error_message": "",
        "explained_index": 0,
        "deletion_drop_auc": 0.4,
        "insertion_auc": 0.4,
        "combined_score": 0.4,
        "runtime_seconds": 0.1,
    }
    pd.DataFrame([row]).to_csv(task_results / "openml_metrics.csv", index=False)
    pd.DataFrame([row]).to_csv(task_results / "openml_runtime.csv", index=False)
    combine_openml_task_outputs(root, out)
    combined = read_csv_preserve_metadata(out / "openml_metrics.csv")
    assert str(combined.loc[0, "git_commit"]) == sha
    assert str(combined.loc[0, "workflow_sha"]) == sha
    assert str(combined.loc[0, "git_commit"]) != "2.777e+79"
    assert str(combined.loc[0, "workflow_sha"]) != "2.777e+79"
    assert not isinstance(combined.loc[0, "git_commit"], float)
    assert not isinstance(combined.loc[0, "workflow_sha"], float)


def test_aggregate_openml_summaries_from_dummy_metrics(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    rows = []
    for seed in [0, 1]:
        for explained_index in [0, 1]:
            for method, score in [("hyb_fpde", 0.4), ("bayesian_hyb_fpde", 0.5)]:
                rows.append(
                    {
                        "method": method,
                        "dataset_name": "dummy",
                        "task_id": 1,
                        "seed": seed,
                        "fold": "fold0",
                        "split_id": "fold0",
                        "mode": "test",
                        "config_hash": "experimentabc",
                        "experiment_config_hash": "experimentabc",
                        "workflow_run_id": "workflow1",
                        "workflow_run_attempt": "1",
                        "workflow_name": "Bayesian OpenML experiment",
                        "workflow_ref": "refs/heads/main",
                        "workflow_sha": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
                        "runner_invocation_hash": f"runner{seed}",
                        "run_config_hash": f"runner{seed}",
                        "job_config_hash": f"job{seed}",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "git_commit": "deadbee",
                        "status": "ok",
                        "error_message": "",
                        "explained_index": explained_index,
                        "explained_order": explained_index,
                        "true_label": 0,
                        "pred_label": 0,
                        "target_label": 0,
                        "deletion_auc": score,
                        "deletion_drop_auc": score,
                        "insertion_auc": score,
                        "faithfulness_correlation": score,
                        "runtime_seconds": 0.1,
                        "number_of_model_calls": 3,
                        "explanation_model_calls": 1,
                        "evaluation_model_calls": 2,
                        "total_model_calls": 3,
                        "top_k_jaccard": score,
                        "test_accuracy": 0.9,
                        "combined_score": score,
                        "metric_direction": "higher_is_better",
                    }
                )
    pd.DataFrame(rows).to_csv(results / "openml_metrics.csv", index=False)
    _run("experiments.aggregate_results", "--results-dir", str(results), "--figures-dir", str(figures))
    seed_summary = read_csv_preserve_metadata(results / "openml_seed_summary.csv")
    global_summary = read_csv_preserve_metadata(results / "openml_global_summary.csv")
    method_summary = read_csv_preserve_metadata(results / "openml_method_summary.csv")
    assert not seed_summary.empty
    assert not global_summary.empty
    assert method_summary["method"].nunique() == 2
    assert "explained_index" not in global_summary.columns
    assert "true_label" not in global_summary.columns
    assert {"n_explanation_rows", "n_unique_explanation_units", "mean_explanation_units_per_dataset_seed", "mean_runtime_seconds_all_rows", "mean_runtime_seconds_ok_only"}.issubset(global_summary.columns)
    assert set(method_summary["n_experiment_config_hashes"]) == {1}
    assert method_summary["experiment_config_hash_consistent"].all()
    assert set(method_summary["n_runner_invocation_hashes"]) == {2}
    assert int(method_summary["n_unique_explanation_units"].sum()) == 8
    for name in ["statistical_tests.csv", "effect_sizes.csv", "bootstrap_confidence_intervals.csv"]:
        df = read_csv_preserve_metadata(results / name)
        assert not df.empty
        assert "mode" in df.columns
        assert {"n_experiment_config_hashes", "experiment_config_hash_consistent", "experiment_config_hashes", "n_runner_invocation_hashes", "n_job_config_hashes"}.issubset(df.columns)
        assert df[["mode", "config_hash", "experiment_config_hash", "timestamp", "git_commit", "status"]].notna().any().all()
    ci = read_csv_preserve_metadata(results / "bootstrap_confidence_intervals.csv")
    assert set(ci["unit_level"]) == {"dataset_seed"}
    assert {"n_units", "n_instance_rows"}.issubset(ci.columns)


def test_matrix_metadata_semantics_allow_multiple_runner_and_job_hashes(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    rows = []
    for seed in [0, 1]:
        rows.append(
            {
                "method": "hyb_fpde",
                "dataset_name": f"dummy_{seed}",
                "task_id": str(31 + seed),
                "seed": str(seed),
                "fold": "fold0",
                "split_id": "fold0",
                "mode": "medium",
                "config_hash": "experimentabc",
                "experiment_config_hash": "experimentabc",
                "workflow_run_id": "123456789",
                "workflow_run_attempt": "1",
                "workflow_name": "Bayesian OpenML experiment",
                "workflow_ref": "refs/heads/main",
                "workflow_sha": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
                "runner_invocation_hash": f"runner{seed}",
                "run_config_hash": f"runner{seed}",
                "job_config_hash": f"job{seed}",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "git_commit": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
                "status": "ok",
                "error_message": "",
                "explained_index": seed,
                "deletion_drop_auc": 0.4,
                "insertion_auc": 0.4,
                "combined_score": 0.4,
                "runtime_seconds": 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(results / "openml_metrics.csv", index=False)
    metrics = read_csv_preserve_metadata(results / "openml_metrics.csv")
    assert metrics["experiment_config_hash"].nunique() == 1
    assert metrics["workflow_run_id"].nunique() == 1
    assert metrics["config_hash"].nunique() == 1
    assert metrics["runner_invocation_hash"].nunique() > 1
    assert metrics["job_config_hash"].nunique() > 1
    _run("experiments.aggregate_results", "--results-dir", str(results), "--figures-dir", str(figures))
    summary = read_csv_preserve_metadata(results / "openml_method_summary.csv")
    assert set(summary["n_experiment_config_hashes"]) == {1}
    assert summary["experiment_config_hash_consistent"].all()
    assert set(summary["n_workflow_run_ids"]) == {1}
    assert summary["n_runner_invocation_hashes"].gt(1).all()
    assert summary["n_job_config_hashes"].gt(1).all()


def test_aggregate_marks_inconsistent_experiment_hashes(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    rows = []
    for experiment_hash, seed in [("experiment_a", 0), ("experiment_b", 1)]:
        rows.append(
            {
                "method": "hyb_fpde",
                "dataset_name": "dummy",
                "task_id": 1,
                "seed": seed,
                "fold": "fold0",
                "split_id": "fold0",
                "mode": "test",
                "config_hash": experiment_hash,
                "experiment_config_hash": experiment_hash,
                "workflow_run_id": "workflow1",
                "workflow_run_attempt": "1",
                "workflow_name": "Bayesian OpenML experiment",
                "workflow_ref": "refs/heads/main",
                "workflow_sha": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
                "runner_invocation_hash": f"runner{seed}",
                "run_config_hash": f"runner{seed}",
                "job_config_hash": f"job{seed}",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "git_commit": "deadbee",
                "status": "ok",
                "error_message": "",
                "explained_index": seed,
                "deletion_drop_auc": 0.4,
                "insertion_auc": 0.4,
                "combined_score": 0.4,
                "runtime_seconds": 0.1,
            }
        )
    pd.DataFrame(rows).to_csv(results / "openml_metrics.csv", index=False)
    _run("experiments.aggregate_results", "--results-dir", str(results), "--figures-dir", str(figures))
    summary = read_csv_preserve_metadata(results / "openml_method_summary.csv")
    assert set(summary["n_experiment_config_hashes"]) == {2}
    assert not summary["experiment_config_hash_consistent"].all()
    assert set(summary["experiment_config_hash"]) == {"multiple"}
    assert set(summary["config_hash"]) == {"multiple"}
    assert (tmp_path / "logs" / "aggregate_results.log").exists()
