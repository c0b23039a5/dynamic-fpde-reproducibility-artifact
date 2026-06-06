from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

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


SYNTHETIC_METRIC_COLUMNS = [
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
]


def _split_filter(value: str | None, cast):
    if value is None or str(value).strip() == "":
        return None
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def _filtered(values: Sequence[Any], selected: Sequence[Any] | None) -> List[Any]:
    return list(values if selected is None else selected)


def _condition_metadata(
    *,
    seed: int,
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_classes: int,
    sep: str,
    corr: str,
    balance: str,
    posterior_samples: int,
    n_explain: int,
    top_k: int,
    tau: float,
    lambda_hyb: float,
) -> Dict[str, Any]:
    return {
        "dataset_name": "synthetic_gaussian",
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_informative": int(n_informative),
        "n_classes": int(n_classes),
        "class_separation": str(sep),
        "feature_correlation": str(corr),
        "class_balance": str(balance),
        "posterior_samples": int(posterior_samples),
        "n_explain": int(n_explain),
        "top_k": int(top_k),
        "tau": float(tau),
        "lambda_hyb": float(lambda_hyb),
    }


def _summary_metadata(
    cfg: Dict[str, Any],
    *,
    condition: Dict[str, Any],
    method: str,
    job_hashes: Dict[str, str],
    status: str,
    error_message: str = "",
) -> Dict[str, Any]:
    return base_metadata(
        **{
            **condition,
            "method": method,
            "fold": "synthetic_random_split",
            "split_id": "synthetic_random_split",
            "mode": str(cfg.get("mode", "")),
            **job_hashes,
            "status": status,
            "error_message": error_message,
        }
    )


def main() -> int:
    parser = parser_with_config("Run synthetic Bayesian-FPDE calibration.")
    parser.add_argument("--seed", default=None, help="Comma-separated seed filter.")
    parser.add_argument("--n-samples", default=None, help="Comma-separated n_samples filter.")
    parser.add_argument("--n-features", default=None, help="Comma-separated n_features filter.")
    parser.add_argument("--n-informative", default=None, help="Comma-separated n_informative filter.")
    parser.add_argument("--class-separation", default=None, help="Comma-separated class_separation filter.")
    parser.add_argument("--feature-correlation", default=None, help="Comma-separated feature_correlation filter.")
    parser.add_argument("--class-balance", default=None, help="Comma-separated class_balance filter.")
    parser.add_argument("--posterior-samples", type=int, default=None, help="Override posterior_samples for this invocation.")
    parser.add_argument("--n-explain", type=int, default=None, help="Override n_explain for this invocation.")
    args = parser.parse_args()
    runner_context = {
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_informative": args.n_informative,
        "class_separation": args.class_separation,
        "feature_correlation": args.feature_correlation,
        "class_balance": args.class_balance,
        "posterior_samples": args.posterior_samples,
        "n_explain": args.n_explain,
    }
    cfg = load_mode_config(
        args.config,
        args.mode,
        runner_name="experiments.run_synthetic_calibration",
        runner_invocation_context={k: v for k, v in runner_context.items() if v is not None},
    )
    logger = setup_logging(cfg.get("logs_dir", "logs"), "synthetic_calibration")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)

    grid = cfg.get("grid", {})
    n_samples_values = _filtered(grid.get("n_samples", [100]), _split_filter(args.n_samples, int))
    n_features_values = _filtered(grid.get("n_features", [10]), _split_filter(args.n_features, int))
    n_informative_values = _filtered(grid.get("n_informative", [3]), _split_filter(args.n_informative, int))
    separations = _filtered(grid.get("class_separation", ["medium"]), _split_filter(args.class_separation, str))
    correlations = _filtered(grid.get("feature_correlation", ["independent"]), _split_filter(args.feature_correlation, str))
    balances = _filtered(grid.get("class_balance", ["balanced"]), _split_filter(args.class_balance, str))
    n_classes_values = grid.get("n_classes", [2])
    seeds = _filtered(cfg.get("seeds", [0]), _split_filter(args.seed, int))
    n_explain = int(args.n_explain if args.n_explain is not None else cfg.get("n_explain", 10))
    posterior_samples = int(args.posterior_samples if args.posterior_samples is not None else cfg.get("posterior_samples", 200))
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
                                    condition = _condition_metadata(
                                        seed=int(seed),
                                        n_samples=int(n_samples),
                                        n_features=int(n_features),
                                        n_informative=int(n_informative),
                                        n_classes=int(n_classes),
                                        sep=str(sep),
                                        corr=str(corr),
                                        balance=str(balance),
                                        posterior_samples=posterior_samples,
                                        n_explain=n_explain,
                                        top_k=top_k,
                                        tau=tau,
                                        lambda_hyb=lambda_hyb,
                                    )
                                    try:
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
                                                extra_context=condition,
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
                                                row_meta = _summary_metadata(cfg, condition=condition, method=method, job_hashes=job_hashes, status="ok")
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
                                                        **_summary_metadata(cfg, condition=condition, method=method, job_hashes=job_hashes, status="ok"),
                                                        "model": model_name,
                                                        **avg,
                                                    }
                                                )
                                    except Exception as exc:
                                        error_message = f"{type(exc).__name__}: {exc}"
                                        logger.exception(
                                            "synthetic condition failed seed=%s n=%s p=%s informative=%s sep=%s corr=%s balance=%s",
                                            seed,
                                            n_samples,
                                            n_features,
                                            n_informative,
                                            sep,
                                            corr,
                                            balance,
                                        )
                                        for method in methods:
                                            job_hashes = config_hashes_for_job(
                                                cfg,
                                                dataset_name="synthetic_gaussian",
                                                seed=int(seed),
                                                fold="synthetic_random_split",
                                                split_id="synthetic_random_split",
                                                methods=[method],
                                                n_explain=n_explain,
                                                posterior_samples=posterior_samples,
                                                tau=tau,
                                                top_k=top_k,
                                                lambda_hyb=lambda_hyb,
                                                extra_context=condition,
                                            )
                                            rows.append(
                                                {
                                                    **_summary_metadata(cfg, condition=condition, method=method, job_hashes=job_hashes, status="error", error_message=error_message),
                                                    "model": str(cfg.get("model", "random_forest")),
                                                    **{name: np.nan for name in SYNTHETIC_METRIC_COLUMNS},
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
