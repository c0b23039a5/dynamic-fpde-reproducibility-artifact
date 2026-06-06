from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA = {
    "method",
    "dataset_name",
    "seed",
    "mode",
    "config_hash",
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
    df = pd.read_csv(target)
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
        REQUIRED_METADATA | {"coverage_95", "sign_brier_score", "sign_ece", "sign_accuracy_at_confidence_0_8", "sign_accuracy_at_confidence_0_9"},
    )
    _csv("results/synthetic_sign_calibration_bins.csv", REQUIRED_METADATA | {"bin_id", "mean_confidence", "sign_accuracy"})
    assert (ROOT / "figures" / "synthetic_coverage_vs_n.png").exists()

    _run("experiments.run_openml_benchmark", "--config", "configs/openml_cc18.yaml", "--mode", "smoke")
    openml = _csv(
        "results/openml_metrics.csv",
        REQUIRED_METADATA | {"task_id", "fold", "split_id", "faithfulness_correlation", "faithfulness_delta_mean", "dependency_available", "explanation_model_calls", "evaluation_model_calls", "total_model_calls", "number_of_model_calls", "combined_score", "metric_direction", "background_size", "max_background"},
    )
    ok_calls = openml[openml["status"] == "ok"]
    assert (ok_calls["number_of_model_calls"] == ok_calls["total_model_calls"]).all()
    assert (ok_calls["total_model_calls"] == ok_calls["explanation_model_calls"] + ok_calls["evaluation_model_calls"]).all()
    assert ok_calls["evaluation_model_calls"].gt(0).all()
    assert openml["run_config_hash"].nunique() == 1
    assert (openml["config_hash"] == openml["run_config_hash"]).all()
    assert (ROOT / "results" / "openml_local_explanations.parquet").exists() or (ROOT / "results" / "openml_local_explanations.parquet.csv").exists()
    seed_summary = _csv("results/openml_seed_summary.csv", {"dataset_name", "task_id", "seed", "method", "n_rows", "n_explanation_rows", "n_unique_explanation_units", "mean_explanation_units_per_dataset_seed", "mean_deletion_auc", "n_run_config_hashes", "run_config_hash_consistent", "n_job_config_hashes", "mean_runtime_seconds_all_rows", "mean_runtime_seconds_ok_only"})
    _csv("results/openml_global_summary.csv", {"dataset_name", "task_id", "method", "n_rows", "n_explanation_rows", "n_unique_explanation_units", "mean_deletion_auc", "n_run_config_hashes", "run_config_hash_consistent"})
    method_summary = _csv("results/openml_method_summary.csv", {"method", "n_rows", "n_datasets", "n_explanation_rows", "n_unique_explanation_units", "mean_deletion_auc", "mean_runtime_seconds_all_rows", "mean_runtime_seconds_ok_only"})
    assert seed_summary["run_config_hash_consistent"].all()
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
                        "config_hash": "abc123",
                        "run_config_hash": "runabc",
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
    seed_summary = pd.read_csv(results / "openml_seed_summary.csv")
    global_summary = pd.read_csv(results / "openml_global_summary.csv")
    method_summary = pd.read_csv(results / "openml_method_summary.csv")
    assert not seed_summary.empty
    assert not global_summary.empty
    assert method_summary["method"].nunique() == 2
    assert "explained_index" not in global_summary.columns
    assert "true_label" not in global_summary.columns
    assert {"n_explanation_rows", "n_unique_explanation_units", "mean_explanation_units_per_dataset_seed", "mean_runtime_seconds_all_rows", "mean_runtime_seconds_ok_only"}.issubset(global_summary.columns)
    assert int(method_summary["n_unique_explanation_units"].sum()) == 8
    for name in ["statistical_tests.csv", "effect_sizes.csv", "bootstrap_confidence_intervals.csv"]:
        df = pd.read_csv(results / name)
        assert not df.empty
        assert "mode" in df.columns
        assert {"n_run_config_hashes", "run_config_hash_consistent", "run_config_hashes", "n_job_config_hashes"}.issubset(df.columns)
        assert df[["mode", "config_hash", "run_config_hash", "timestamp", "git_commit", "status"]].notna().any().all()
    ci = pd.read_csv(results / "bootstrap_confidence_intervals.csv")
    assert set(ci["unit_level"]) == {"dataset_seed"}
    assert {"n_units", "n_instance_rows"}.issubset(ci.columns)


def test_aggregate_marks_inconsistent_run_hashes(tmp_path: Path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    rows = []
    for run_hash, seed in [("run_a", 0), ("run_b", 1)]:
        rows.append(
            {
                "method": "hyb_fpde",
                "dataset_name": "dummy",
                "task_id": 1,
                "seed": seed,
                "fold": "fold0",
                "split_id": "fold0",
                "mode": "test",
                "config_hash": run_hash,
                "run_config_hash": run_hash,
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
    summary = pd.read_csv(results / "openml_method_summary.csv")
    assert set(summary["n_run_config_hashes"]) == {2}
    assert not summary["run_config_hash_consistent"].all()
    assert set(summary["run_config_hash"]) == {"multiple"}
    assert (tmp_path / "logs" / "aggregate_results.log").exists()
