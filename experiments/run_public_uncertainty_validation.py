from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde, summarize_samples
from bayesian_fpde.bootstrap_fpde import bootstrap_fpde_samples
from bayesian_fpde.fpde import FPDEConfig
from bayesian_fpde.metrics import spearman_corr, top_k_jaccard
from bayesian_fpde.plotting import save_metric_boxplot
from bayesian_fpde.utils import base_metadata, ensure_dirs, setup_logging, write_csv, write_parquet_or_csv
from experiments.common import config_hashes_for_job, explain_indices, load_mode_config, load_tabular_openml_or_local, parser_with_config


_BAYESIAN_METHODS: Dict[str, tuple[str, float]] = {
    "bayesian_diff_fpde": ("diff", 1.0),
    "bayesian_cos_fpde": ("cos", 0.0),
    "bayesian_hyb_fpde": ("hyb", 0.5),
}


def _finite_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _method_summary(
    *,
    method: str,
    model: Any,
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    posterior_samples: int,
    bootstrap_samples: int,
    top_k: int,
    tau: float,
    lambda_hyb: float,
    seed: int,
) -> pd.DataFrame:
    baseline = np.mean(X_train, axis=0)
    method_key = method.lower()
    if method_key in _BAYESIAN_METHODS:
        fpde_mode, default_lambda = _BAYESIAN_METHODS[method_key]
        result = explain_bayesian_fpde(
            model,
            x,
            X_train,
            y_train,
            config=BayesianFPDEConfig(
                mode=fpde_mode,
                lambda_hyb=lambda_hyb if fpde_mode == "hyb" else default_lambda,
                n_posterior_samples=posterior_samples,
                tau=tau,
                top_k=top_k,
            ),
            anchor=baseline,
            feature_names=feature_names,
            seed=seed,
        )
        return result.summary

    if method_key == "bootstrap_fpde":
        samples = bootstrap_fpde_samples(
            model,
            x,
            X_train,
            y_train,
            n_bootstrap=bootstrap_samples,
            config=FPDEConfig(mode="hyb", lambda_hyb=lambda_hyb),
            anchor=baseline,
            seed=seed,
        )
        return summarize_samples(samples, feature_names=feature_names, tau=tau, top_k=top_k)

    raise ValueError(f"unsupported public uncertainty method: {method}")


def _append_feature_rows(
    rows: List[Dict[str, Any]],
    *,
    summary: pd.DataFrame,
    metadata: Dict[str, Any],
    method: str,
    explained_index: int,
    explained_order: int,
    n_train: int,
    n_test: int,
) -> None:
    for _, feature_row in summary.iterrows():
        ci_width = float(feature_row["ci_upper_95"] - feature_row["ci_lower_95"])
        rows.append(
            {
                **metadata,
                "method": method,
                "explained_index": int(explained_index),
                "explained_order": int(explained_order),
                "feature": str(feature_row["feature"]),
                "feature_index": int(feature_row["feature_index"]),
                "posterior_mean": float(feature_row["posterior_mean"]),
                "posterior_std": float(feature_row["posterior_std"]),
                "ci_lower_95": float(feature_row["ci_lower_95"]),
                "ci_upper_95": float(feature_row["ci_upper_95"]),
                "ci_width": ci_width,
                "p_positive": float(feature_row["p_positive"]),
                "p_negative": float(feature_row["p_negative"]),
                "p_abs_gt_tau": float(feature_row["p_abs_gt_tau"]),
                "rank_mean": float(feature_row["rank_mean"]),
                "rank_std": float(feature_row["rank_std"]),
                "rank_probability_top_k": float(feature_row["rank_probability_top_k"]),
                "attribution": float(feature_row["posterior_mean"]),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "metric_direction": "higher_is_better",
            }
        )


def _seed_feature_aggregate(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    ok = local[local["status"] == "ok"].copy() if "status" in local.columns else local.copy()
    if ok.empty:
        return pd.DataFrame()
    ok["sign_confidence"] = np.maximum(ok["p_positive"].astype(float), ok["p_negative"].astype(float))
    group_cols = ["dataset_name", "task_id", "fold", "method", "seed", "feature", "feature_index"]
    agg = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            posterior_mean=("posterior_mean", "mean"),
            posterior_std=("posterior_std", "mean"),
            ci_lower_95=("ci_lower_95", "mean"),
            ci_upper_95=("ci_upper_95", "mean"),
            ci_width=("ci_width", "mean"),
            p_positive=("p_positive", "mean"),
            p_negative=("p_negative", "mean"),
            sign_confidence=("sign_confidence", "mean"),
            n_explained_instances=("explained_index", "nunique"),
            n_train=("n_train", "max"),
            n_test=("n_test", "max"),
        )
        .reset_index()
    )
    return agg


def _nan_public_metrics() -> Dict[str, float]:
    return {
        "empirical_reference_coverage_95": float("nan"),
        "uncertainty_error_correlation": float("nan"),
        "ci_width_error_correlation": float("nan"),
        "uncertainty_error_pearson": float("nan"),
        "sign_agreement_at_confidence": float("nan"),
        "mean_ci_width": float("nan"),
        "median_ci_width": float("nan"),
        "mean_posterior_std": float("nan"),
        "mean_abs_error_to_reference": float("nan"),
        "top_k_jaccard_to_reference": float("nan"),
    }


def _compute_public_uncertainty_metrics(seed_features: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    threshold = float(cfg.get("sign_confidence_threshold", 0.8))
    top_k = int(cfg.get("top_k", 5))
    group_cols = ["dataset_name", "task_id", "fold", "method"]
    if seed_features.empty:
        return pd.DataFrame()

    for key, group in seed_features.groupby(group_cols, dropna=False):
        dataset_name, task_id, fold, method = key
        seeds = sorted(group["seed"].astype(str).unique().tolist())
        if len(seeds) < 2:
            hashes = config_hashes_for_job(
                cfg,
                dataset_name=str(dataset_name),
                task_id=task_id,
                seed="all",
                fold=fold,
                split_id="leave_one_seed_reference",
                methods=[str(method)],
                n_explain=int(cfg.get("n_explain", 0)),
                posterior_samples=int(cfg.get("posterior_samples", 0)),
                bootstrap_samples=int(cfg.get("bootstrap_samples", 0)),
                top_k=top_k,
                lambda_hyb=float(cfg.get("lambda_hyb", 0.5)),
            )
            rows.append(
                {
                    **base_metadata(
                        dataset_name=dataset_name,
                        task_id=task_id,
                        seed="all",
                        fold=fold,
                        split_id="leave_one_seed_reference",
                        mode=str(cfg.get("mode", "")),
                        method=str(method),
                        **hashes,
                        status="skipped",
                        error_message="public uncertainty validation requires at least two seeds",
                    ),
                    "n_seeds": len(seeds),
                    "n_reference_seeds": 0,
                    "n_common_features": 0,
                    "sign_confidence_threshold": threshold,
                    "metric_direction": "higher_is_better",
                    **_nan_public_metrics(),
                }
            )
            continue

        for seed in seeds:
            current = group[group["seed"].astype(str) == str(seed)].copy()
            others = group[group["seed"].astype(str) != str(seed)].copy()
            reference = (
                others.groupby(["feature"], dropna=False)
                .agg(reference_phi=("posterior_mean", "mean"))
                .reset_index()
            )
            merged = current.merge(reference, on="feature", how="inner")
            hashes = config_hashes_for_job(
                cfg,
                dataset_name=str(dataset_name),
                task_id=task_id,
                seed=seed,
                fold=fold,
                split_id="leave_one_seed_reference",
                methods=[str(method)],
                n_explain=int(cfg.get("n_explain", 0)),
                posterior_samples=int(cfg.get("posterior_samples", 0)),
                bootstrap_samples=int(cfg.get("bootstrap_samples", 0)),
                top_k=top_k,
                lambda_hyb=float(cfg.get("lambda_hyb", 0.5)),
            )
            if merged.empty:
                status = "skipped"
                error = "no common features across leave-one-seed reference"
                metrics = _nan_public_metrics()
            else:
                error_abs = np.abs(merged["posterior_mean"].to_numpy(dtype=float) - merged["reference_phi"].to_numpy(dtype=float))
                ci_width = merged["ci_width"].to_numpy(dtype=float)
                posterior_std = merged["posterior_std"].to_numpy(dtype=float)
                covered = (merged["reference_phi"] >= merged["ci_lower_95"]) & (merged["reference_phi"] <= merged["ci_upper_95"])
                sign_conf = merged["sign_confidence"].to_numpy(dtype=float)
                conf_mask = (sign_conf >= threshold) & (merged["reference_phi"].to_numpy(dtype=float) != 0.0)
                if np.any(conf_mask):
                    sign_agreement = float(
                        np.mean(
                            np.sign(merged.loc[conf_mask, "posterior_mean"].to_numpy(dtype=float))
                            == np.sign(merged.loc[conf_mask, "reference_phi"].to_numpy(dtype=float))
                        )
                    )
                else:
                    sign_agreement = float("nan")
                metrics = {
                    "empirical_reference_coverage_95": float(np.mean(covered)),
                    "uncertainty_error_correlation": spearman_corr(posterior_std, error_abs),
                    "ci_width_error_correlation": spearman_corr(ci_width, error_abs),
                    "uncertainty_error_pearson": _finite_pearson(posterior_std, error_abs),
                    "sign_agreement_at_confidence": sign_agreement,
                    "mean_ci_width": float(np.nanmean(ci_width)),
                    "median_ci_width": float(np.nanmedian(ci_width)),
                    "mean_posterior_std": float(np.nanmean(posterior_std)),
                    "mean_abs_error_to_reference": float(np.nanmean(error_abs)),
                    "top_k_jaccard_to_reference": top_k_jaccard(
                        merged["posterior_mean"].to_numpy(dtype=float),
                        merged["reference_phi"].to_numpy(dtype=float),
                        min(top_k, merged.shape[0]),
                    ),
                }
                status = "ok"
                error = ""

            rows.append(
                {
                    **base_metadata(
                        dataset_name=dataset_name,
                        task_id=task_id,
                        seed=seed,
                        fold=fold,
                        split_id="leave_one_seed_reference",
                        mode=str(cfg.get("mode", "")),
                        method=str(method),
                        **hashes,
                        status=status,
                        error_message=error,
                    ),
                    "n_seeds": len(seeds),
                    "n_reference_seeds": max(0, len(seeds) - 1),
                    "n_common_features": int(merged.shape[0]),
                    "sign_confidence_threshold": threshold,
                    "metric_direction": "higher_is_better",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parser_with_config("Run public-data uncertainty validation with leave-one-seed empirical references.").parse_args()
    cfg = load_mode_config(args.config, args.mode, runner_name="experiments.run_public_uncertainty_validation")
    logger = setup_logging(cfg.get("logs_dir", "logs"), "public_uncertainty_validation")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)

    methods = cfg.get("uncertainty_methods", ["bayesian_hyb_fpde", "bootstrap_fpde"])
    rows: List[Dict[str, Any]] = []
    for seed in cfg.get("seeds", [0]):
        for task_id, payload, split_name in load_tabular_openml_or_local(cfg, seed=int(seed), mode=args.mode):
            dataset_name, X_train, y_train, X_test, y_test, model, feature_names, _ = payload
            pred = model.predict(X_test)
            indices = explain_indices(y_test, pred, int(cfg.get("n_explain", 20)), seed=int(seed))
            for method in methods:
                hashes = config_hashes_for_job(
                    cfg,
                    dataset_name=dataset_name,
                    task_id=task_id,
                    seed=int(seed),
                    fold=split_name,
                    split_id=split_name,
                    methods=[method],
                    n_explain=int(cfg.get("n_explain", 20)),
                    posterior_samples=int(cfg.get("posterior_samples", 100)),
                    bootstrap_samples=int(cfg.get("bootstrap_samples", 30)),
                    top_k=int(cfg.get("top_k", 5)),
                    lambda_hyb=float(cfg.get("lambda_hyb", 0.5)),
                )
                metadata = base_metadata(
                    dataset_name=dataset_name,
                    task_id=task_id,
                    seed=int(seed),
                    fold=split_name,
                    split_id=split_name,
                    mode=str(cfg.get("mode", "")),
                    method=method,
                    **hashes,
                    status="ok",
                    error_message="",
                )
                for order, idx in enumerate(indices.tolist()):
                    try:
                        summary = _method_summary(
                            method=method,
                            model=model,
                            x=X_test[idx],
                            X_train=X_train,
                            y_train=y_train,
                            feature_names=feature_names,
                            posterior_samples=int(cfg.get("posterior_samples", 100)),
                            bootstrap_samples=int(cfg.get("bootstrap_samples", 30)),
                            top_k=int(cfg.get("top_k", 5)),
                            tau=float(cfg.get("tau", 0.0)),
                            lambda_hyb=float(cfg.get("lambda_hyb", 0.5)),
                            seed=int(seed) + int(order),
                        )
                        _append_feature_rows(
                            rows,
                            summary=summary,
                            metadata=metadata,
                            method=method,
                            explained_index=int(idx),
                            explained_order=int(order),
                            n_train=int(X_train.shape[0]),
                            n_test=int(X_test.shape[0]),
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                **metadata,
                                "status": "error",
                                "error_message": f"{type(exc).__name__}: {exc}",
                                "explained_index": int(idx),
                                "explained_order": int(order),
                                "n_train": int(X_train.shape[0]),
                                "n_test": int(X_test.shape[0]),
                            }
                        )
                logger.info("public uncertainty completed dataset=%s seed=%s method=%s", dataset_name, seed, method)

    local = pd.DataFrame(rows)
    write_parquet_or_csv(local, results_dir / "public_uncertainty_local_explanations.parquet")
    seed_features = _seed_feature_aggregate(local)
    write_csv(seed_features, results_dir / "public_uncertainty_seed_features.csv")
    metrics = _compute_public_uncertainty_metrics(seed_features, cfg)
    write_csv(metrics, results_dir / "public_uncertainty_validation.csv")
    save_metric_boxplot(
        metrics,
        metric="empirical_reference_coverage_95",
        path=figures_dir / "public_uncertainty_empirical_reference_coverage.png",
        title="Empirical reference coverage",
    )
    save_metric_boxplot(
        metrics,
        metric="uncertainty_error_correlation",
        path=figures_dir / "public_uncertainty_error_correlation.png",
        title="Uncertainty-error correlation",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
