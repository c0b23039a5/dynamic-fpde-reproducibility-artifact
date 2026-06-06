from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayesian_fpde.plotting import save_metric_boxplot
from bayesian_fpde.utils import ensure_dirs, setup_logging, write_json
from experiments.common import config_hashes_for_job, evaluate_methods_for_dataset, load_mode_config, load_tabular_openml_or_local, parser_with_config, write_standard_outputs


def main() -> int:
    args = parser_with_config("Run Bayesian-FPDE OpenML-CC18 benchmark.").parse_args()
    cfg = load_mode_config(args.config, args.mode, runner_name="experiments.run_openml_benchmark")
    logger = setup_logging(cfg.get("logs_dir", "logs"), "openml_benchmark")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    methods = cfg.get("methods", ["diff_fpde", "cos_fpde", "hyb_fpde", "bayesian_hyb_fpde", "bootstrap_fpde", "shap", "lime", "aime"])
    all_local = []
    all_metrics = []
    all_runtime = []
    for seed in cfg.get("seeds", [0]):
        for task_id, payload, split_name in load_tabular_openml_or_local(cfg, seed=int(seed), mode=args.mode):
            dataset_name, X_train, y_train, X_test, y_test, model, feature_names, model_name = payload
            logger.info("running dataset=%s task_id=%s seed=%s model=%s", dataset_name, task_id, seed, model_name)
            n_explain = int(cfg.get("n_explain", 100))
            posterior_samples = int(cfg.get("posterior_samples", 200))
            bootstrap_samples = int(cfg.get("bootstrap_samples", 30))
            tau = float(cfg.get("tau", 0.0))
            top_k = int(cfg.get("top_k", 5))
            lambda_hyb = float(cfg.get("lambda_hyb", 0.5))
            max_background = int(cfg.get("max_background", 100))
            hashes = config_hashes_for_job(
                cfg,
                dataset_name=dataset_name,
                task_id=task_id,
                seed=int(seed),
                fold=split_name,
                split_id=split_name,
                methods=methods,
                n_explain=n_explain,
                posterior_samples=posterior_samples,
                bootstrap_samples=bootstrap_samples,
                tau=tau,
                top_k=top_k,
                lambda_hyb=lambda_hyb,
                max_background=max_background,
            )
            local, metrics, runtime = evaluate_methods_for_dataset(
                dataset_name=dataset_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model=model,
                feature_names=feature_names,
                methods=methods,
                seed=int(seed),
                task_id=task_id,
                fold=split_name,
                n_explain=n_explain,
                posterior_samples=posterior_samples,
                bootstrap_samples=bootstrap_samples,
                tau=tau,
                top_k=top_k,
                lambda_hyb=lambda_hyb,
                mode=str(cfg.get("mode", "")),
                config_hash=hashes["config_hash"],
                experiment_config_hash=hashes["experiment_config_hash"],
                workflow_run_id=hashes["workflow_run_id"],
                workflow_run_attempt=hashes["workflow_run_attempt"],
                workflow_name=hashes["workflow_name"],
                workflow_ref=hashes["workflow_ref"],
                workflow_sha=hashes["workflow_sha"],
                runner_invocation_hash=hashes["runner_invocation_hash"],
                run_config_hash=hashes["run_config_hash"],
                job_config_hash=hashes["job_config_hash"],
                max_background=max_background,
            )
            all_local.append(local)
            all_metrics.append(metrics)
            all_runtime.append(runtime)
    local_df = pd.concat(all_local, ignore_index=True) if all_local else pd.DataFrame()
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    runtime_df = pd.concat(all_runtime, ignore_index=True) if all_runtime else pd.DataFrame()
    write_standard_outputs(results_dir, figures_dir, local_df, metrics_df, runtime_df)
    save_metric_boxplot(metrics_df, metric="faithfulness_correlation", path=figures_dir / "openml_faithfulness_boxplot.png", title="OpenML faithfulness")
    runtime_ok_df = runtime_df[runtime_df["status"] == "ok"] if not runtime_df.empty and "status" in runtime_df.columns else runtime_df
    save_metric_boxplot(runtime_ok_df, metric="runtime_seconds", path=figures_dir / "openml_runtime_boxplot.png", title="OpenML runtime")
    rank_table = local_df.groupby(["method", "feature"], as_index=False)["rank_mean"].mean() if not local_df.empty else pd.DataFrame()
    rank_table.to_csv(figures_dir / "openml_average_rank_table.csv", index=False, lineterminator="\n")
    write_json({"config": cfg, "n_metric_rows": int(len(metrics_df))}, results_dir / "openml_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
