from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from bayesian_fpde.utils import read_csv_preserve_metadata


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


def _csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    target = Path(path)
    assert target.exists(), str(path)
    df = read_csv_preserve_metadata(target)
    assert not df.empty, str(path)
    assert required.issubset(df.columns), f"{path} missing {sorted(required - set(df.columns))}"
    return df


def test_ieee_access_workflow_config_is_the_public_experiment_source():
    cfg = yaml.safe_load((ROOT / "configs" / "openml_public_ieee_access.yaml").read_text(encoding="utf-8"))
    assert cfg["results_dir"] == "results_ieee"
    assert cfg["figures_dir"] == "figures_ieee"
    assert cfg["logs_dir"] == "logs_ieee"
    assert cfg["task_ids"] == [31, 10101, 9946, 37, 10093]
    assert sorted(cfg["modes"]) == ["ieee_full", "ieee_min"]
    assert cfg["modes"]["ieee_min"]["max_tasks"] == 3
    assert cfg["modes"]["ieee_full"]["max_tasks"] == 5


def test_faithfulness_passes_configured_bootstrap_samples(monkeypatch, tmp_path: Path):
    import experiments.run_faithfulness as runner

    cfg = {
        "results_dir": str(tmp_path / "results"),
        "figures_dir": str(tmp_path / "figures"),
        "logs_dir": str(tmp_path / "logs"),
        "mode": "ieee_min",
        "seeds": [7],
        "faithfulness_methods": ["bootstrap_fpde"],
        "n_explain": 2,
        "posterior_samples": 11,
        "bootstrap_samples": 50,
        "top_k": 3,
        "lambda_hyb": 0.4,
        "max_background": 9,
    }
    calls: list[tuple[str, int]] = []
    hashes = {
        "config_hash": "config",
        "experiment_config_hash": "experiment",
        "workflow_run_id": "run",
        "workflow_run_attempt": "1",
        "workflow_name": "workflow",
        "workflow_ref": "ref",
        "workflow_sha": "sha",
        "runner_invocation_hash": "runner",
        "run_config_hash": "run-config",
        "job_config_hash": "job",
    }

    class Logger:
        def info(self, *args, **kwargs):
            return None

    def fake_hashes_for_job(*args, **kwargs):
        calls.append(("hash", kwargs["bootstrap_samples"]))
        return hashes

    def fake_evaluate_methods_for_dataset(*args, **kwargs):
        calls.append(("evaluate", kwargs["bootstrap_samples"]))
        metrics = pd.DataFrame(
            [
                {
                    "method": "bootstrap_fpde",
                    "explained_order": 0,
                    "deletion_auc": 0.1,
                    "insertion_auc": 0.2,
                }
            ]
        )
        return pd.DataFrame(), metrics, pd.DataFrame()

    payload = ("dummy", "X_train", "y_train", "X_test", "y_test", "model", ["x0"], "model_name")
    monkeypatch.setattr(sys, "argv", ["run_faithfulness.py", "--config", "config.yml", "--mode", "ieee_min"])
    monkeypatch.setattr(runner, "load_mode_config", lambda *args, **kwargs: dict(cfg))
    monkeypatch.setattr(runner, "apply_task_id_filter", lambda loaded_cfg, task_id: loaded_cfg)
    monkeypatch.setattr(runner, "setup_logging", lambda *args, **kwargs: Logger())
    monkeypatch.setattr(runner, "load_tabular_openml_or_local", lambda *args, **kwargs: [("", payload, "local_smoke")])
    monkeypatch.setattr(runner, "config_hashes_for_job", fake_hashes_for_job)
    monkeypatch.setattr(runner, "evaluate_methods_for_dataset", fake_evaluate_methods_for_dataset)
    monkeypatch.setattr(runner, "save_line_plot", lambda *args, **kwargs: None)

    assert runner.main() == 0
    assert calls == [("hash", 50), ("evaluate", 50)]


def test_public_experiments_and_aggregate_on_local_ieee_shape(tmp_path: Path):
    results = tmp_path / "results_ieee"
    figures = tmp_path / "figures_ieee"
    logs = tmp_path / "logs_ieee"
    config = tmp_path / "openml_public_ieee_access_local.yaml"
    config.write_text(
        "\n".join(
            [
                f"results_dir: {results.as_posix()}",
                f"figures_dir: {figures.as_posix()}",
                f"logs_dir: {logs.as_posix()}",
                "posterior_samples: 12",
                "bootstrap_samples: 4",
                "lambda_hyb: 0.5",
                "top_k: 3",
                "tau: 0.0",
                "model: random_forest",
                "train_fractions: [0.5, 1.0]",
                "uncertainty_methods:",
                "  - bayesian_hyb_fpde",
                "faithfulness_methods:",
                "  - hyb_fpde",
                "  - bayesian_hyb_fpde",
                "methods:",
                "  - bayesian_hyb_fpde",
                "modes:",
                "  ieee_min:",
                "    local_smoke: true",
                "    seeds: [0, 1]",
                "    max_tasks: 1",
                "    n_explain: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for module in [
        "experiments.run_public_uncertainty_validation",
        "experiments.run_stability",
        "experiments.run_faithfulness",
        "experiments.run_training_size_uncertainty",
    ]:
        _run(module, "--config", str(config), "--mode", "ieee_min")

    _run("experiments.aggregate_results", "--results-dir", str(results), "--figures-dir", str(figures), "--logs-dir", str(logs))

    public = _csv(
        results / "public_uncertainty_validation.csv",
        REQUIRED_METADATA | {"empirical_reference_coverage_95", "uncertainty_error_correlation"},
    )
    stability = _csv(results / "stability_metrics.csv", REQUIRED_METADATA | {"mean_spearman_between_runs", "top_k_jaccard_between_runs"})
    faithfulness = _csv(results / "faithfulness_metrics.csv", REQUIRED_METADATA | {"deletion_auc", "insertion_auc", "faithfulness_correlation"})
    training_size = _csv(results / "training_size_uncertainty.csv", REQUIRED_METADATA | {"train_fraction", "mean_ci_width", "attribution_distance_to_full_train"})
    assert set(public["mode"]) == {"ieee_min"}
    assert set(stability["mode"]) == {"ieee_min"}
    assert set(faithfulness["mode"]) == {"ieee_min"}
    assert set(training_size["mode"]) == {"ieee_min"}
    for name in ["statistical_tests.csv", "effect_sizes.csv", "bootstrap_confidence_intervals.csv"]:
        _csv(results / name, {"metric", "status", "experiment_config_hash", "git_commit"})


def test_aggregate_marks_inconsistent_experiment_hashes(tmp_path: Path):
    results = tmp_path / "results_ieee"
    figures = tmp_path / "figures_ieee"
    logs = tmp_path / "logs_ieee"
    results.mkdir()
    rows = []
    for experiment_hash, seed in [("experiment_a", 0), ("experiment_b", 1)]:
        rows.append(
            {
                "method": "bayesian_hyb_fpde",
                "dataset_name": "dummy",
                "task_id": 1,
                "seed": seed,
                "fold": "fold0",
                "split_id": "fold0",
                "mode": "ieee_min",
                "config_hash": experiment_hash,
                "experiment_config_hash": experiment_hash,
                "workflow_run_id": "workflow1",
                "workflow_run_attempt": "1",
                "workflow_name": "IEEE Access Bayesian-FPDE experiments",
                "workflow_ref": "refs/heads/main",
                "workflow_sha": "2777e761b10ba6431735c0f7fb2fb3b05ab08559",
                "runner_invocation_hash": f"runner{seed}",
                "run_config_hash": f"runner{seed}",
                "job_config_hash": f"job{seed}",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "git_commit": "deadbee",
                "status": "ok",
                "error_message": "",
                "deletion_auc": 0.4,
                "insertion_auc": 0.4,
                "combined_score": 0.4,
                "metric_direction": "higher_is_better",
            }
        )
    pd.DataFrame(rows).to_csv(results / "faithfulness_metrics.csv", index=False)

    _run("experiments.aggregate_results", "--results-dir", str(results), "--figures-dir", str(figures), "--logs-dir", str(logs))

    tests = read_csv_preserve_metadata(results / "statistical_tests.csv")
    assert set(tests["n_experiment_config_hashes"]) == {2}
    assert not tests["experiment_config_hash_consistent"].all()
    assert set(tests["experiment_config_hash"]) == {"multiple"}
    assert set(tests["config_hash"]) == {"multiple"}
    assert (logs / "aggregate_results.log").exists()


def test_sensitivity_analysis_local_two_seed_run(tmp_path: Path):
    results = tmp_path / "results_ieee_sensitivity"
    figures = tmp_path / "figures_ieee_sensitivity"
    logs = tmp_path / "logs_ieee_sensitivity"
    config = tmp_path / "openml_public_ieee_access_sensitivity_local.yaml"
    config.write_text(
        "\n".join(
            [
                f"results_dir: {results.as_posix()}",
                f"figures_dir: {figures.as_posix()}",
                f"logs_dir: {logs.as_posix()}",
                "posterior_samples: 100",
                "bootstrap_samples: 10",
                "lambda_hyb: 0.5",
                "top_k: 3",
                "tau: 0.0",
                "uncertainty_methods:",
                "  - bayesian_hyb_fpde",
                "modes:",
                "  sensitivity_posterior:",
                "    local_smoke: true",
                "    seeds: [0, 1]",
                "    max_tasks: 1",
                "    n_explain: 2",
                "    posterior_samples_grid: [20, 40]",
                "    lambda_hyb_grid: [0.5]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _run("experiments.run_sensitivity_analysis", "--config", str(config), "--mode", "sensitivity_posterior")

    sensitivity = _csv(
        results / "sensitivity_results.csv",
        REQUIRED_METADATA
        | {
            "sensitivity_type",
            "posterior_samples",
            "lambda_hyb",
            "empirical_reference_coverage_95",
            "mean_posterior_std",
            "mean_ci_width",
            "uncertainty_error_correlation",
            "ci_width_error_correlation",
            "mean_spearman_between_seeds",
            "top_k_jaccard_between_seeds",
            "attribution_distance_to_default",
            "top_k_jaccard_to_default",
            "sign_agreement_to_default",
            "runtime_seconds",
        },
    )
    posterior = _csv(
        results / "posterior_samples_sensitivity.csv",
        REQUIRED_METADATA | {"sensitivity_type", "posterior_samples", "lambda_hyb", "runtime_seconds"},
    )
    posterior_summary = _csv(
        results / "posterior_samples_sensitivity_summary.csv",
        {"sensitivity_type", "dataset_name", "method", "posterior_samples", "lambda_hyb", "n_rows", "n_seeds"},
    )
    lambda_csv = read_csv_preserve_metadata(results / "lambda_hyb_sensitivity.csv")
    lambda_summary = read_csv_preserve_metadata(results / "lambda_hyb_sensitivity_summary.csv")

    assert not sensitivity.empty
    assert not posterior.empty
    assert not posterior_summary.empty
    assert (results / "sensitivity_seed_features.csv").exists()
    assert (results / "sensitivity_local_explanations.parquet").exists() or (results / "sensitivity_local_explanations.parquet.csv").exists()
    assert set(sensitivity["sensitivity_type"]) == {"posterior_samples"}
    assert set(sensitivity["status"]).issubset({"ok", "skipped", "error"})
    assert "ok" in set(sensitivity["status"])
    assert set(sensitivity["posterior_samples"].astype(int)) == {20, 40}
    assert set(sensitivity["lambda_hyb"].astype(float)) == {0.5}
    assert sensitivity["mean_posterior_std"].notna().any()
    assert sensitivity["mean_ci_width"].notna().any()
    assert sensitivity["runtime_seconds"].notna().any()
    assert sensitivity["mean_spearman_between_seeds"].notna().any()
    assert sensitivity["top_k_jaccard_between_seeds"].notna().any()
    assert posterior_summary["n_seeds"].astype(int).eq(2).all()
    assert set(lambda_csv.columns).issuperset({"sensitivity_type", "posterior_samples", "lambda_hyb"})
    assert set(lambda_summary.columns).issuperset({"sensitivity_type", "posterior_samples", "lambda_hyb"})
