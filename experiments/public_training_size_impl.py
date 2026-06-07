from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde, summarize_samples
from bayesian_fpde.bootstrap_fpde import bootstrap_fpde_samples
from bayesian_fpde.datasets import fit_black_box
from bayesian_fpde.fpde import FPDEConfig
from bayesian_fpde.metrics import top_k_jaccard
from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import base_metadata, ensure_dirs, setup_logging, write_csv, write_parquet_or_csv
from experiments.common import config_hashes_for_job, explain_indices, load_mode_config, load_tabular_openml_or_local, parser_with_config


_BAYESIAN_METHODS: Dict[str, tuple[str, float]] = {
    "bayesian_diff_fpde": ("diff", 1.0),
    "bayesian_cos_fpde": ("cos", 0.0),
    "bayesian_hyb_fpde": ("hyb", 0.5),
}


def _fraction_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    y = np.asarray(y)
    if fraction >= 1.0:
        return np.arange(y.size)
    rng = np.random.default_rng(seed)
    parts: List[np.ndarray] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if idx.size:
            n = max(1, min(idx.size, int(round(float(fraction) * idx.size))))
            parts.append(rng.choice(idx, size=n, replace=False))
    out = np.concatenate(parts) if parts else np.arange(y.size)
    rng.shuffle(out)
    return np.asarray(out, dtype=int)


def _summary_for_method(method: str, model: Any, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, feature_names: Sequence[str], cfg: Dict[str, Any], seed: int) -> pd.DataFrame:
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
                lambda_hyb=float(cfg.get("lambda_hyb", 0.5)) if fpde_mode == "hyb" else default_lambda,
                n_posterior_samples=int(cfg.get("posterior_samples", 100)),
                tau=float(cfg.get("tau", 0.0)),
                top_k=int(cfg.get("top_k", 5)),
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
            n_bootstrap=int(cfg.get("bootstrap_samples", 30)),
            config=FPDEConfig(mode="hyb", lambda_hyb=float(cfg.get("lambda_hyb", 0.5))),
            anchor=baseline,
            seed=seed,
        )
        return summarize_samples(samples, feature_names=feature_names, tau=float(cfg.get("tau", 0.0)), top_k=int(cfg.get("top_k", 5)))
    raise ValueError(f"unsupported training-size uncertainty method: {method}")


def _add_rows(rows: List[Dict[str, Any]], summary: pd.DataFrame, metadata: Dict[str, Any], method: str, fraction: float, training_size: int, explained_index: int, explained_order: int, n_test: int) -> None:
    for _, r in summary.iterrows():
        ci_width = float(r["ci_upper_95"] - r["ci_lower_95"])
        rows.append({
            **metadata,
            "method": method,
            "train_fraction": float(fraction),
            "training_size": int(training_size),
            "explained_index": int(explained_index),
            "explained_order": int(explained_order),
            "feature": str(r["feature"]),
            "feature_index": int(r["feature_index"]),
            "posterior_mean": float(r["posterior_mean"]),
            "posterior_std": float(r["posterior_std"]),
            "ci_lower_95": float(r["ci_lower_95"]),
            "ci_upper_95": float(r["ci_upper_95"]),
            "ci_width": ci_width,
            "p_positive": float(r["p_positive"]),
            "p_negative": float(r["p_negative"]),
            "p_abs_gt_tau": float(r["p_abs_gt_tau"]),
            "rank_mean": float(r["rank_mean"]),
            "rank_std": float(r["rank_std"]),
            "rank_probability_top_k": float(r["rank_probability_top_k"]),
            "attribution": float(r["posterior_mean"]),
            "n_train": int(training_size),
            "n_test": int(n_test),
            "metric_direction": "lower_is_better_for_uncertainty_widths",
        })


def fraction_feature_summary(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    df = local[local["status"] == "ok"].copy() if "status" in local.columns else local.copy()
    if df.empty:
        return pd.DataFrame()
    df["sign_confidence"] = np.maximum(df["p_positive"].astype(float), df["p_negative"].astype(float))
    return (
        df.groupby(["dataset_name", "task_id", "fold", "seed", "method", "train_fraction", "training_size", "feature", "feature_index"], dropna=False)
        .agg(
            posterior_mean=("posterior_mean", "mean"),
            posterior_std=("posterior_std", "mean"),
            ci_width=("ci_width", "mean"),
            sign_confidence=("sign_confidence", "mean"),
            n_explained_instances=("explained_index", "nunique"),
        )
        .reset_index()
    )


def training_size_metrics(feature_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    top_k = int(cfg.get("top_k", 5))
    if feature_df.empty:
        return pd.DataFrame()
    for (dataset_name, task_id, fold, seed, method), group in feature_df.groupby(["dataset_name", "task_id", "fold", "seed", "method"], dropna=False):
        full_fraction = float(group["train_fraction"].astype(float).max())
        ref = group[group["train_fraction"].astype(float) == full_fraction][["feature", "posterior_mean"]].rename(columns={"posterior_mean": "full_train_phi"})
        tmp: List[Dict[str, Any]] = []
        for (fraction, training_size), sub in group.groupby(["train_fraction", "training_size"], dropna=False):
            merged = sub.merge(ref, on="feature", how="inner")
            if merged.empty:
                dist = float("nan")
                jacc = float("nan")
            else:
                diff = merged["posterior_mean"].to_numpy(dtype=float) - merged["full_train_phi"].to_numpy(dtype=float)
                dist = float(np.sqrt(np.mean(diff ** 2)))
                jacc = top_k_jaccard(merged["posterior_mean"].to_numpy(dtype=float), merged["full_train_phi"].to_numpy(dtype=float), min(top_k, merged.shape[0]))
            hashes = config_hashes_for_job(cfg, dataset_name=str(dataset_name), task_id=task_id, seed=seed, fold=fold, split_id=f"train_fraction_{float(fraction):.4g}", methods=[str(method)], n_explain=int(cfg.get("n_explain", 0)), posterior_samples=int(cfg.get("posterior_samples", 0)), bootstrap_samples=int(cfg.get("bootstrap_samples", 0)), top_k=top_k, lambda_hyb=float(cfg.get("lambda_hyb", 0.5)), extra_context={"train_fraction": float(fraction), "training_size": int(training_size)})
            tmp.append({
                **base_metadata(dataset_name=dataset_name, task_id=task_id, seed=seed, fold=fold, split_id=f"train_fraction_{float(fraction):.4g}", mode=str(cfg.get("mode", "")), method=str(method), **hashes, status="ok", error_message=""),
                "train_fraction": float(fraction),
                "training_size": int(training_size),
                "mean_posterior_std": float(np.nanmean(sub["posterior_std"].to_numpy(dtype=float))),
                "mean_ci_width": float(np.nanmean(sub["ci_width"].to_numpy(dtype=float))),
                "median_ci_width": float(np.nanmedian(sub["ci_width"].to_numpy(dtype=float))),
                "mean_sign_confidence": float(np.nanmean(sub["sign_confidence"].to_numpy(dtype=float))),
                "attribution_distance_to_full_train": dist,
                "top_k_jaccard_to_full_train": jacc,
                "n_features": int(sub["feature"].nunique()),
                "n_explained_instances": int(sub["n_explained_instances"].max()),
                "metric_direction": "lower_is_better_for_uncertainty_widths",
            })
        if tmp:
            tdf = pd.DataFrame(tmp).sort_values("train_fraction")
            finite = tdf[np.isfinite(tdf["mean_ci_width"].to_numpy(dtype=float))]
            slope = float(np.polyfit(finite["train_fraction"].to_numpy(dtype=float), finite["mean_ci_width"].to_numpy(dtype=float), 1)[0]) if finite.shape[0] >= 2 and finite["train_fraction"].nunique() >= 2 else float("nan")
            for r in tmp:
                r["uncertainty_decrease_slope"] = slope
                r["uncertainty_decrease_rate"] = -slope if np.isfinite(slope) else float("nan")
                rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    args = parser_with_config("Run public-data training-size uncertainty experiments.").parse_args()
    cfg = load_mode_config(args.config, args.mode, runner_name="experiments.run_training_size_uncertainty")
    logger = setup_logging(cfg.get("logs_dir", "logs"), "training_size_uncertainty")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    methods: Sequence[str] = cfg.get("uncertainty_methods", ["bayesian_hyb_fpde", "bootstrap_fpde"])
    fractions = [float(v) for v in cfg.get("train_fractions", [0.1, 0.25, 0.5, 0.75, 1.0])]
    rows: List[Dict[str, Any]] = []
    for seed in cfg.get("seeds", [0]):
        for task_id, payload, split_name in load_tabular_openml_or_local(cfg, seed=int(seed), mode=args.mode):
            dataset_name, X_full, y_full, X_test, y_test, full_model, feature_names, _ = payload
            explained = explain_indices(y_test, full_model.predict(X_test), int(cfg.get("n_explain", 20)), seed=int(seed))
            for fraction in fractions:
                idx = _fraction_indices(y_full, fraction, seed=int(seed) + int(round(fraction * 10000)))
                X_train = X_full[idx]
                y_train = y_full[idx]
                if np.unique(y_train).size < 2:
                    logger.info("skipping dataset=%s seed=%s fraction=%s: one class", dataset_name, seed, fraction)
                    continue
                model, model_name = fit_black_box(X_train, y_train, seed=int(seed), model_name=str(cfg.get("model", "auto")), feature_names_in=feature_names)
                for method in methods:
                    hashes = config_hashes_for_job(cfg, dataset_name=dataset_name, task_id=task_id, seed=int(seed), fold=split_name, split_id=f"train_fraction_{fraction:.4g}", methods=[method], n_explain=int(cfg.get("n_explain", 20)), posterior_samples=int(cfg.get("posterior_samples", 100)), bootstrap_samples=int(cfg.get("bootstrap_samples", 30)), top_k=int(cfg.get("top_k", 5)), lambda_hyb=float(cfg.get("lambda_hyb", 0.5)), extra_context={"train_fraction": fraction, "training_size": int(X_train.shape[0])})
                    metadata = base_metadata(dataset_name=dataset_name, task_id=task_id, seed=int(seed), fold=split_name, split_id=f"train_fraction_{fraction:.4g}", mode=str(cfg.get("mode", "")), method=method, **hashes, status="ok", error_message="", model=model_name)
                    for order, ex_idx in enumerate(explained.tolist()):
                        try:
                            summary = _summary_for_method(method, model, X_test[ex_idx], X_train, y_train, feature_names, cfg, seed=int(seed) + int(order))
                            _add_rows(rows, summary, metadata, method, fraction, int(X_train.shape[0]), int(ex_idx), int(order), int(X_test.shape[0]))
                        except Exception as exc:
                            rows.append({**metadata, "status": "error", "error_message": f"{type(exc).__name__}: {exc}", "train_fraction": fraction, "training_size": int(X_train.shape[0]), "explained_index": int(ex_idx), "explained_order": int(order), "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])})
                logger.info("training-size completed dataset=%s seed=%s fraction=%s", dataset_name, seed, fraction)
    local = pd.DataFrame(rows)
    write_parquet_or_csv(local, results_dir / "training_size_local_explanations.parquet")
    feature_df = fraction_feature_summary(local)
    write_csv(feature_df, results_dir / "training_size_seed_features.csv")
    metrics = training_size_metrics(feature_df, cfg)
    write_csv(metrics, results_dir / "training_size_uncertainty.csv")
    save_line_plot(metrics, x="train_fraction", y="mean_ci_width", group="method", path=figures_dir / "ci_width_vs_training_size.png", title="CI width vs training fraction")
    save_line_plot(metrics, x="train_fraction", y="attribution_distance_to_full_train", group="method", path=figures_dir / "distance_to_full_train_vs_training_size.png", title="Distance to full-training reference")
    return 0
