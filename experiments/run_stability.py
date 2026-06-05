from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.datasets import generate_synthetic_gaussian, fit_black_box
from bayesian_fpde.metrics import stability_metrics
from bayesian_fpde.plotting import save_metric_boxplot
from bayesian_fpde.utils import base_metadata, ensure_dirs, setup_logging, write_csv
from experiments.common import load_mode_config, parser_with_config


def main() -> int:
    args = parser_with_config("Run explanation stability experiments.").parse_args()
    cfg = load_mode_config(args.config, args.mode)
    logger = setup_logging(cfg.get("logs_dir", "logs"), "stability")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    rows = []
    B = int(cfg.get("stability_bootstrap", 30))
    top_k = int(cfg.get("top_k", 5))
    for seed in cfg.get("seeds", [0]):
        data = generate_synthetic_gaussian(
            n_samples=int(cfg.get("n_samples", 160)),
            n_features=int(cfg.get("n_features", 12)),
            n_informative=int(cfg.get("n_informative", 4)),
            random_seed=int(seed),
        )
        model, _ = fit_black_box(data.X, data.y, seed=int(seed), model_name=str(cfg.get("model", "random_forest")))
        x = data.X[0]
        samples = []
        rng = np.random.default_rng(int(seed))
        for b in range(B):
            idx = rng.integers(0, data.X.shape[0], size=data.X.shape[0])
            Xb, yb = data.X[idx], data.y[idx]
            if np.unique(yb).size < 2:
                continue
            result = explain_bayesian_fpde(
                model,
                x,
                Xb,
                yb,
                config=BayesianFPDEConfig(n_posterior_samples=int(cfg.get("posterior_samples", 50)), top_k=top_k),
                anchor=np.mean(Xb, axis=0),
                feature_names=data.feature_names,
                seed=int(seed) + b,
            )
            samples.append(result.summary["posterior_mean"].to_numpy(dtype=float))
        metrics = stability_metrics(np.asarray(samples), top_k=top_k)
        rows.append({**base_metadata(**data.metadata), "method": "bayesian_hyb_fpde", "status": "ok", "error": "", "posterior_rank_std": float(np.nanmean([metrics.get("rank_entropy", np.nan)])), **metrics})
        logger.info("stability completed seed=%s", seed)
    df = pd.DataFrame(rows)
    write_csv(df, results_dir / "stability_metrics.csv")
    save_metric_boxplot(df, metric="mean_spearman_between_runs", path=figures_dir / "stability_spearman_boxplot.png", title="Stability Spearman")
    save_metric_boxplot(df, metric="top_k_jaccard_between_runs", path=figures_dir / "stability_topk_jaccard_boxplot.png", title="Stability top-k Jaccard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
