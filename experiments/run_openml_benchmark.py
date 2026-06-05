from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayesian_fpde.plotting import save_metric_boxplot
from bayesian_fpde.utils import ensure_dirs, setup_logging, write_json
from experiments.common import evaluate_methods_for_dataset, load_mode_config, load_tabular_openml_or_local, parser_with_config, write_standard_outputs


def main() -> int:
    args = parser_with_config("Run Bayesian-FPDE OpenML-CC18 benchmark.").parse_args()
    cfg = load_mode_config(args.config, args.mode)
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
                n_explain=int(cfg.get("n_explain", 100)),
                posterior_samples=int(cfg.get("posterior_samples", 200)),
                bootstrap_samples=int(cfg.get("bootstrap_samples", 30)),
                tau=float(cfg.get("tau", 0.0)),
                top_k=int(cfg.get("top_k", 5)),
                lambda_hyb=float(cfg.get("lambda_hyb", 0.5)),
                mode=str(cfg.get("mode", "")),
                config_hash=str(cfg.get("config_hash", "")),
            )
            all_local.append(local)
            all_metrics.append(metrics)
            all_runtime.append(runtime)
    local_df = pd.concat(all_local, ignore_index=True) if all_local else pd.DataFrame()
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    runtime_df = pd.concat(all_runtime, ignore_index=True) if all_runtime else pd.DataFrame()
    write_standard_outputs(results_dir, figures_dir, local_df, metrics_df, runtime_df)
    save_metric_boxplot(metrics_df, metric="faithfulness_correlation", path=figures_dir / "openml_faithfulness_boxplot.png", title="OpenML faithfulness")
    save_metric_boxplot(runtime_df, metric="runtime_seconds", path=figures_dir / "openml_runtime_boxplot.png", title="OpenML runtime")
    rank_table = local_df.groupby(["method", "feature"], as_index=False)["rank_mean"].mean() if not local_df.empty else pd.DataFrame()
    rank_table.to_csv(figures_dir / "openml_average_rank_table.csv", index=False, lineterminator="\n")
    write_json({"config": cfg, "n_metric_rows": int(len(metrics_df))}, results_dir / "openml_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
