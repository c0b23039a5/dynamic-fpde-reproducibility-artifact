from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde
from bayesian_fpde.datasets import fit_black_box, generate_synthetic_gaussian
from bayesian_fpde.fpde import FPDEConfig, class_prototypes, explain_fpde, true_fpde_attribution
from bayesian_fpde.metrics import calibration_metrics, sign_reliability_bins
from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import base_metadata, ensure_dirs, setup_logging, write_csv, write_json
from experiments.common import config_hashes_for_job, load_mode_config, parser_with_config


def main() -> int:
    args = parser_with_config("Run synthetic Bayesian-FPDE calibration.").parse_args()
    cfg = load_mode_config(args.config, args.mode, runner_name="experiments.run_synthetic_calibration")
    logger = setup_logging(cfg.get("logs_dir", "logs"), "synthetic_calibration")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)

    grid = cfg.get("grid", {})
    n_samples_values = grid.get("n_samples", [100])
    n_features_values = grid.get("n_features", [10])
    n_informative_values = grid.get("n_informative", [3])
    separations = grid.get("class_separation", ["medium"])
    correlations = grid.get("feature_correlation", ["independent"])
    balances = grid.get("class_balance", ["balanced"])
    n_classes_values = grid.get("n_classes", [2])
    seeds = cfg.get("seeds", [0])
    n_explain = int(cfg.get("n_explain", 10))
    posterior_samples = int(cfg.get("posterior_samples", 200))
    top_k = int(cfg.get("top_k", 5))
    tau = float(cfg.get("tau", 0.0))
    lambda_hyb = float(cfg.get("lambda_hyb", 0.5))
    sign_bins = int(cfg.get("sign_calibration_bins", 10))
    methods = cfg.get("methods", ["bayesian_hyb_fpde"])

    rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    bin_rows: List[pd.DataFrame] = []
    for seed in seeds:
        for n_samples in n_samples_values:
            for n_features in n_features_values:
                for n_informative in n_informative_values:
                    if n_informative > n_features:
                        continue
                    for n_classes in n_classes_values:
                        for sep in separations:
                            for corr in correlations:
                                for balance in balances:
                                    data = generate_synthetic_gaussian(
                                        n_samples=int(n_samples),
                                        n_features=int(n_features),
                                        n_informative=int(n_informative),
                                        n_classes=int(n_classes),
                                        class_separation=str(sep),
                                        feature_correlation=str(corr),
                                        class_balance=str(balance),
                                        random_seed=int(seed),
                                    )
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        data.X,
                                        data.y,
                                        test_size=float(cfg.get("test_size", 0.3)),
                                        random_state=int(seed),
                                        stratify=data.y,
                                    )
                                    model, model_name = fit_black_box(X_train, y_train, seed=int(seed), model_name=str(cfg.get("model", "random_forest")))
                                    baseline = np.mean(X_train, axis=0)
                                    prototypes, labels = class_prototypes(X_train, y_train)
                                    pred = model.predict(X_test)
                                    eligible = np.where(pred == y_test)[0]
                                    if eligible.size == 0:
                                        eligible = np.arange(len(y_test))
                                    eligible = eligible[: min(n_explain, eligible.size)]
                                    for method in methods:
                                        mode = "hyb"
                                        if "diff" in method:
                                            mode = "diff"
                                        elif "cos" in method:
                                            mode = "cos"
                                        job_hashes = config_hashes_for_job(
                                            cfg,
                                            dataset_name=str(data.metadata.get("dataset_name", "")),
                                            task_id=str(data.metadata.get("task_id", "")),
                                            seed=int(seed),
                                            fold="synthetic_random_split",
                                            split_id="synthetic_random_split",
                                            methods=[method],
                                            n_explain=n_explain,
                                            posterior_samples=posterior_samples,
                                            tau=tau,
                                            top_k=top_k,
                                            lambda_hyb=lambda_hyb,
                                        )
                                        per_instance_metrics = []
                                        for order, idx in enumerate(eligible.tolist()):
                                            x = X_test[idx]
                                            det_attr, c_plus, c_minus = explain_fpde(
                                                model,
                                                x,
                                                prototypes,
                                                labels,
                                                config=FPDEConfig(mode=mode, lambda_hyb=lambda_hyb),
                                                anchor=baseline,
                                            )
                                            true_attr = true_fpde_attribution(
                                                x,
                                                data.true_prototypes,
                                                data.labels,
                                                positive_label=c_plus,
                                                negative_label=c_minus,
                                                mode=mode,
                                                lambda_hyb=lambda_hyb,
                                                anchor=np.zeros_like(baseline),
                                            )
                                            bcfg = BayesianFPDEConfig(
                                                mode=mode,
                                                lambda_hyb=lambda_hyb,
                                                n_posterior_samples=posterior_samples,
                                                tau=tau,
                                                top_k=top_k,
                                            )
                                            result = explain_bayesian_fpde(
                                                model,
                                                x,
                                                X_train,
                                                y_train,
                                                config=bcfg,
                                                anchor=baseline,
                                                feature_names=data.feature_names,
                                                seed=int(seed) + order,
                                            )
                                            row_meta = base_metadata(
                                                **{
                                                    **data.metadata,
                                                    "method": method,
                                                    "seed": int(seed),
                                                    "fold": "synthetic_random_split",
                                                    "split_id": "synthetic_random_split",
                                                    "mode": str(cfg.get("mode", "")),
                                                    **job_hashes,
                                                    "status": "ok",
                                                    "error_message": "",
                                                }
                                            )
                                            metric = calibration_metrics(result.summary, true_attr, top_k=top_k, sign_bins=sign_bins)
                                            per_instance_metrics.append(metric)
                                            bin_rows.append(
                                                sign_reliability_bins(
                                                    result.summary,
                                                    true_attr,
                                                    n_bins=sign_bins,
                                                    metadata={
                                                        **row_meta,
                                                        "explained_index": int(idx),
                                                        "explained_order": int(order),
                                                    },
                                                )
                                            )
                                            meta = {
                                                **row_meta,
                                                "method": method,
                                                "model": model_name,
                                                "explained_index": int(idx),
                                                "explained_order": int(order),
                                                "positive_label": int(c_plus),
                                                "negative_label": int(c_minus),
                                            }
                                            for _, feature_row in result.summary.iterrows():
                                                out = {**meta, **feature_row.to_dict()}
                                                out["true_attribution"] = float(true_attr[int(feature_row["feature_index"])])
                                                out["deterministic_fpde_score"] = float(det_attr[int(feature_row["feature_index"])])
                                                local_rows.append(out)
                                        if per_instance_metrics:
                                            avg = pd.DataFrame(per_instance_metrics).mean(numeric_only=True).to_dict()
                                            rows.append(
                                                {
                                                    **base_metadata(
                                                        **{
                                                            **data.metadata,
                                                            "method": method,
                                                            "seed": int(seed),
                                                            "fold": "synthetic_random_split",
                                                            "split_id": "synthetic_random_split",
                                                            "mode": str(cfg.get("mode", "")),
                                                            **job_hashes,
                                                            "status": "ok",
                                                            "error_message": "",
                                                        }
                                                    ),
                                                    "model": model_name,
                                                    **avg,
                                                }
                                            )
                                    logger.info("completed synthetic seed=%s n=%s p=%s", seed, n_samples, n_features)

    detail = pd.DataFrame(local_rows)
    summary = pd.DataFrame(rows)
    write_csv(detail, results_dir / "synthetic_calibration.csv")
    write_csv(summary, results_dir / "synthetic_calibration_summary.csv")
    bins = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
    write_csv(bins, results_dir / "synthetic_sign_calibration_bins.csv")
    save_line_plot(summary, x="n_samples", y="coverage_95", group="method", path=figures_dir / "synthetic_coverage_vs_n.png", title="Synthetic coverage vs n")
    save_line_plot(summary, x="n_samples", y="mean_ci_width", group="method", path=figures_dir / "synthetic_ci_width_vs_n.png", title="Synthetic CI width vs n")
    save_line_plot(summary, x="n_samples", y="sign_ece", group="method", path=figures_dir / "synthetic_sign_ece_vs_n.png", title="Synthetic sign ECE vs n")
    save_line_plot(summary, x="n_samples", y="top_k_precision", group="method", path=figures_dir / "synthetic_topk_precision.png", title="Synthetic top-k precision")
    write_json({"config": cfg, "n_rows": int(len(summary))}, results_dir / "synthetic_calibration_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
