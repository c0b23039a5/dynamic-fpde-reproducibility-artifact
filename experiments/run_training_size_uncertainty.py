from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.datasets import generate_synthetic_gaussian, fit_black_box
from bayesian_fpde.fpde import true_fpde_attribution
from bayesian_fpde.metrics import calibration_metrics
from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import base_metadata, ensure_dirs, write_csv
from experiments.common import config_hashes_for_job, load_mode_config, parser_with_config


def main() -> int:
    args = parser_with_config("Run training-size vs uncertainty experiments.").parse_args()
    cfg = load_mode_config(args.config, args.mode)
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    rows = []
    sizes = cfg.get("training_sizes", [25, 50, 100])
    for seed in cfg.get("seeds", [0]):
        data = generate_synthetic_gaussian(n_samples=max(max(sizes) + 50, 180), n_features=int(cfg.get("n_features", 10)), n_informative=int(cfg.get("n_informative", 3)), random_seed=int(seed))
        rng = np.random.default_rng(int(seed))
        for size in sizes:
            idx = rng.choice(np.arange(data.X.shape[0]), size=min(int(size), data.X.shape[0]), replace=False)
            X_train, y_train = data.X[idx], data.y[idx]
            if np.unique(y_train).size < 2:
                continue
            model, model_name = fit_black_box(X_train, y_train, seed=int(seed), model_name=str(cfg.get("model", "random_forest")))
            x = data.X[-1]
            proba = model.predict_proba(x.reshape(1, -1))[0]
            labels = np.asarray(getattr(model, "classes_", np.arange(proba.size)), dtype=int)
            order = np.argsort(proba)[::-1]
            c_plus, c_minus = int(labels[order[0]]), int(labels[order[1]])
            result = explain_bayesian_fpde(
                model,
                x,
                X_train,
                y_train,
                config=BayesianFPDEConfig(n_posterior_samples=int(cfg.get("posterior_samples", 100))),
                anchor=np.mean(X_train, axis=0),
                feature_names=data.feature_names,
                seed=int(seed),
            )
            truth = true_fpde_attribution(x, data.true_prototypes, data.labels, positive_label=c_plus, negative_label=c_minus, mode="hyb", lambda_hyb=float(cfg.get("lambda_hyb", 0.5)))
            metrics = calibration_metrics(result.summary, truth, top_k=int(cfg.get("top_k", 5)), sign_bins=int(cfg.get("sign_calibration_bins", 10)))
            hashes = config_hashes_for_job(cfg, dataset_name=str(data.metadata.get("dataset_name", "")), seed=int(seed), fold="synthetic_training_size", split_id=f"training_size_{size}", methods=["bayesian_hyb_fpde"], posterior_samples=int(cfg.get("posterior_samples", 100)), top_k=int(cfg.get("top_k", 5)), lambda_hyb=float(cfg.get("lambda_hyb", 0.5)))
            rows.append({**base_metadata(**{**data.metadata, "method": "bayesian_hyb_fpde", "seed": int(seed), "fold": "synthetic_training_size", "split_id": f"training_size_{size}", "mode": str(cfg.get("mode", "")), **hashes, "status": "ok", "error_message": ""}), "model": model_name, "training_size": int(size), "sign_confidence": float(np.mean(np.maximum(result.summary["p_positive"], result.summary["p_negative"]))), "rank_stability": float(np.nanmean(result.summary["rank_probability_top_k"])), **metrics})
    df = pd.DataFrame(rows)
    write_csv(df, results_dir / "training_size_uncertainty.csv")
    save_line_plot(df, x="training_size", y="mean_ci_width", group="method", path=figures_dir / "ci_width_vs_training_size.png", title="CI width vs training size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
