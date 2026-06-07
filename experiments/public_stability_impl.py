from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from bayesian_fpde.metrics import spearman_corr, top_k_jaccard
from bayesian_fpde.plotting import save_metric_boxplot
from bayesian_fpde.utils import base_metadata, ensure_dirs, setup_logging, write_csv, write_parquet_or_csv
from experiments.common import config_hashes_for_job, evaluate_methods_for_dataset, load_mode_config, load_tabular_openml_or_local, parser_with_config


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
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


def seed_features(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return pd.DataFrame()
    df = local[local["status"] == "ok"].copy() if "status" in local.columns else local.copy()
    if df.empty:
        return pd.DataFrame()
    if "attribution" not in df.columns:
        df["attribution"] = df.get("posterior_mean", np.nan)
    if "ci_width" not in df.columns:
        df["ci_width"] = df["ci_upper_95"].astype(float) - df["ci_lower_95"].astype(float) if {"ci_upper_95", "ci_lower_95"}.issubset(df.columns) else np.nan
    return (
        df.groupby(["dataset_name", "task_id", "fold", "method", "seed", "feature", "feature_index"], dropna=False)
        .agg(
            attribution=("attribution", "mean"),
            abs_attribution=("attribution", lambda s: float(np.mean(np.abs(s.astype(float))))),
            ci_width=("ci_width", "mean"),
            posterior_std=("posterior_std", "mean") if "posterior_std" in df.columns else ("attribution", lambda _: np.nan),
            rank_probability_top_k=("rank_probability_top_k", "mean") if "rank_probability_top_k" in df.columns else ("attribution", lambda _: np.nan),
            n_explained_instances=("explained_index", "nunique"),
        )
        .reset_index()
    )


def _pairwise(group: pd.DataFrame, top_k: int) -> Dict[str, float]:
    spear: List[float] = []
    pear: List[float] = []
    jacc: List[float] = []
    sign: List[float] = []
    ci_stab: List[float] = []
    for a_seed, b_seed in combinations(sorted(group["seed"].astype(str).unique().tolist()), 2):
        a = group[group["seed"].astype(str) == a_seed]
        b = group[group["seed"].astype(str) == b_seed]
        m = a.merge(b, on="feature", suffixes=("_a", "_b"), how="inner")
        if m.empty:
            continue
        aa = m["attribution_a"].to_numpy(dtype=float)
        bb = m["attribution_b"].to_numpy(dtype=float)
        spear.append(spearman_corr(np.abs(aa), np.abs(bb)))
        pear.append(_pearson(aa, bb))
        jacc.append(top_k_jaccard(aa, bb, min(top_k, aa.size)))
        sign.append(float(np.mean(np.sign(aa) == np.sign(bb))))
        ci_stab.append(spearman_corr(m["ci_width_a"].to_numpy(dtype=float), m["ci_width_b"].to_numpy(dtype=float)))
    return {
        "mean_spearman_between_seeds": float(np.nanmean(spear)) if spear else float("nan"),
        "mean_pearson_between_seeds": float(np.nanmean(pear)) if pear else float("nan"),
        "top_k_jaccard_between_seeds": float(np.nanmean(jacc)) if jacc else float("nan"),
        "sign_agreement_between_seeds": float(np.nanmean(sign)) if sign else float("nan"),
        "ci_width_stability": float(np.nanmean(ci_stab)) if ci_stab else float("nan"),
    }


def stability_metrics(seed_feature_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    top_k = int(cfg.get("top_k", 5))
    if seed_feature_df.empty:
        return pd.DataFrame()
    for (dataset_name, task_id, fold, method), group in seed_feature_df.groupby(["dataset_name", "task_id", "fold", "method"], dropna=False):
        seeds = sorted(group["seed"].astype(str).unique().tolist())
        means = group.groupby("feature", dropna=False).agg(mean_abs=("abs_attribution", "mean"), mean_ci=("ci_width", "mean"))
        hashes = config_hashes_for_job(cfg, dataset_name=str(dataset_name), task_id=task_id, seed="all", fold=fold, split_id="across_seeds", methods=[str(method)], n_explain=int(cfg.get("n_explain", 0)), posterior_samples=int(cfg.get("posterior_samples", 0)), bootstrap_samples=int(cfg.get("bootstrap_samples", 0)), top_k=top_k, lambda_hyb=float(cfg.get("lambda_hyb", 0.5)))
        status = "ok" if len(seeds) >= 2 else "skipped"
        row = {
            **base_metadata(dataset_name=dataset_name, task_id=task_id, seed="all", fold=fold, split_id="across_seeds", mode=str(cfg.get("mode", "")), method=str(method), **hashes, status=status, error_message="" if status == "ok" else "stability requires at least two seeds"),
            "n_seeds": int(len(seeds)),
            "n_common_or_observed_features": int(means.shape[0]),
            "uncertainty_rank_correlation": spearman_corr(means["mean_abs"].to_numpy(dtype=float), means["mean_ci"].to_numpy(dtype=float)),
            "metric_direction": "higher_is_better",
            **(_pairwise(group, top_k) if len(seeds) >= 2 else {"mean_spearman_between_seeds": float("nan"), "mean_pearson_between_seeds": float("nan"), "top_k_jaccard_between_seeds": float("nan"), "sign_agreement_between_seeds": float("nan"), "ci_width_stability": float("nan")}),
        }
        row["mean_spearman_between_runs"] = row["mean_spearman_between_seeds"]
        row["top_k_jaccard_between_runs"] = row["top_k_jaccard_between_seeds"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parser_with_config("Run public-data explanation stability experiments.").parse_args()
    cfg = load_mode_config(args.config, args.mode, runner_name="experiments.run_stability")
    logger = setup_logging(cfg.get("logs_dir", "logs"), "stability")
    results_dir = Path(cfg.get("results_dir", "results"))
    figures_dir = Path(cfg.get("figures_dir", "figures"))
    ensure_dirs(results_dir, figures_dir)
    methods: Sequence[str] = cfg.get("methods", ["hyb_fpde", "bayesian_hyb_fpde", "bootstrap_fpde"])
    local_frames: List[pd.DataFrame] = []
    runtime_frames: List[pd.DataFrame] = []
    for seed in cfg.get("seeds", [0]):
        for task_id, payload, split_name in load_tabular_openml_or_local(cfg, seed=int(seed), mode=args.mode):
            dataset_name, X_train, y_train, X_test, y_test, model, feature_names, _ = payload
            hashes = config_hashes_for_job(cfg, dataset_name=dataset_name, task_id=task_id, seed=int(seed), fold=split_name, split_id=split_name, methods=methods, n_explain=int(cfg.get("n_explain", 20)), posterior_samples=int(cfg.get("posterior_samples", 100)), bootstrap_samples=int(cfg.get("bootstrap_samples", 30)), top_k=int(cfg.get("top_k", 5)), lambda_hyb=float(cfg.get("lambda_hyb", 0.5)), max_background=int(cfg.get("max_background", 100)))
            local, _, runtime = evaluate_methods_for_dataset(dataset_name=dataset_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model=model, feature_names=feature_names, methods=methods, seed=int(seed), task_id=task_id, fold=split_name, n_explain=int(cfg.get("n_explain", 20)), posterior_samples=int(cfg.get("posterior_samples", 100)), bootstrap_samples=int(cfg.get("bootstrap_samples", 30)), tau=float(cfg.get("tau", 0.0)), top_k=int(cfg.get("top_k", 5)), lambda_hyb=float(cfg.get("lambda_hyb", 0.5)), mode=str(cfg.get("mode", "")), config_hash=hashes["config_hash"], experiment_config_hash=hashes["experiment_config_hash"], workflow_run_id=hashes["workflow_run_id"], workflow_run_attempt=hashes["workflow_run_attempt"], workflow_name=hashes["workflow_name"], workflow_ref=hashes["workflow_ref"], workflow_sha=hashes["workflow_sha"], runner_invocation_hash=hashes["runner_invocation_hash"], run_config_hash=hashes["run_config_hash"], job_config_hash=hashes["job_config_hash"], max_background=int(cfg.get("max_background", 100)))
            local_frames.append(local)
            runtime_frames.append(runtime)
            logger.info("stability local explanations completed dataset=%s seed=%s", dataset_name, seed)
    local_df = pd.concat(local_frames, ignore_index=True, sort=False) if local_frames else pd.DataFrame()
    runtime_df = pd.concat(runtime_frames, ignore_index=True, sort=False) if runtime_frames else pd.DataFrame()
    write_parquet_or_csv(local_df, results_dir / "stability_local_explanations.parquet")
    write_csv(runtime_df, results_dir / "stability_runtime.csv")
    feature_df = seed_features(local_df)
    write_csv(feature_df, results_dir / "stability_seed_features.csv")
    metrics = stability_metrics(feature_df, cfg)
    write_csv(metrics, results_dir / "stability_metrics.csv")
    save_metric_boxplot(metrics, metric="mean_spearman_between_seeds", path=figures_dir / "stability_spearman_boxplot.png", title="Stability Spearman")
    save_metric_boxplot(metrics, metric="top_k_jaccard_between_seeds", path=figures_dir / "stability_topk_jaccard_boxplot.png", title="Stability top-k Jaccard")
    return 0
