from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.datasets import generate_synthetic_gaussian, fit_black_box
from bayesian_fpde.metrics import deletion_insertion_metrics
from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import base_metadata, ensure_dirs, write_csv
from experiments.common import load_mode_config, parser_with_config


def main() -> int:
    args = parser_with_config("Run Bayesian-FPDE ablations.").parse_args()
    cfg = load_mode_config(args.config, args.mode)
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    rows = []
    for seed in cfg.get("seeds", [0]):
        data = generate_synthetic_gaussian(n_samples=int(cfg.get("n_samples", 160)), n_features=int(cfg.get("n_features", 10)), n_informative=int(cfg.get("n_informative", 3)), random_seed=int(seed))
        model, model_name = fit_black_box(data.X, data.y, seed=int(seed), model_name=str(cfg.get("model", "random_forest")))
        x = data.X[0]
        baseline = np.mean(data.X, axis=0)
        for posterior_samples in cfg.get("posterior_samples_grid", [100, 500]):
            result = explain_bayesian_fpde(model, x, data.X, data.y, config=BayesianFPDEConfig(n_posterior_samples=int(posterior_samples)), anchor=baseline, feature_names=data.feature_names, seed=int(seed))
            attr = result.summary["posterior_mean"].to_numpy(dtype=float)
            metrics = deletion_insertion_metrics(model, x, attr, baseline, target_label=result.positive_label)
            rows.append({**base_metadata(**{**data.metadata, "method": "bayesian_hyb_fpde", "seed": int(seed), "fold": "synthetic_ablation", "split_id": f"posterior_samples_{posterior_samples}", "mode": str(cfg.get("mode", "")), "config_hash": str(cfg.get("config_hash", "")), "run_config_hash": str(cfg.get("run_config_hash", cfg.get("config_hash", ""))), "job_config_hash": str(cfg.get("job_config_hash", cfg.get("config_hash", ""))), "status": "ok", "error_message": ""}), "ablation": "posterior_samples", "posterior_samples": int(posterior_samples), "lambda_hyb": 0.5, "model": model_name, "mean_ci_width": float(np.mean(result.summary["ci_upper_95"] - result.summary["ci_lower_95"])), **metrics})
        for lambda_hyb in cfg.get("lambda_grid", [0.0, 0.5, 1.0]):
            result = explain_bayesian_fpde(model, x, data.X, data.y, config=BayesianFPDEConfig(n_posterior_samples=int(cfg.get("posterior_samples", 100)), lambda_hyb=float(lambda_hyb)), anchor=baseline, feature_names=data.feature_names, seed=int(seed))
            attr = result.summary["posterior_mean"].to_numpy(dtype=float)
            metrics = deletion_insertion_metrics(model, x, attr, baseline, target_label=result.positive_label)
            rows.append({**base_metadata(**{**data.metadata, "method": "bayesian_hyb_fpde", "seed": int(seed), "fold": "synthetic_ablation", "split_id": f"lambda_hyb_{lambda_hyb}", "mode": str(cfg.get("mode", "")), "config_hash": str(cfg.get("config_hash", "")), "run_config_hash": str(cfg.get("run_config_hash", cfg.get("config_hash", ""))), "job_config_hash": str(cfg.get("job_config_hash", cfg.get("config_hash", ""))), "status": "ok", "error_message": ""}), "ablation": "lambda_hyb", "posterior_samples": int(cfg.get("posterior_samples", 100)), "lambda_hyb": float(lambda_hyb), "model": model_name, "mean_ci_width": float(np.mean(result.summary["ci_upper_95"] - result.summary["ci_lower_95"])), **metrics})
    df = pd.DataFrame(rows)
    write_csv(df, results_dir / "ablation_metrics.csv")
    save_line_plot(df[df["ablation"] == "posterior_samples"], x="posterior_samples", y="mean_ci_width", path=figures_dir / "ablation_posterior_samples.png", title="Posterior sample ablation")
    save_line_plot(df[df["ablation"] == "lambda_hyb"], x="lambda_hyb", y="deletion_drop_auc", path=figures_dir / "ablation_lambda.png", title="Lambda ablation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
