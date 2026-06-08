from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from bayesian_fpde.metrics import top_k_jaccard
from bayesian_fpde.plotting import save_line_plot
from bayesian_fpde.utils import base_metadata, ensure_dirs, write_csv, write_parquet_or_csv
from experiments.common import apply_task_id_filter, config_hashes_for_job, explain_indices, load_mode_config, load_tabular_openml_or_local, parser_with_config
from experiments.public_stability_impl import seed_features as stability_seed_features
from experiments.public_stability_impl import stability_metrics
from experiments.run_public_uncertainty_validation import _compute_public_uncertainty_metrics, _method_summary, _seed_feature_aggregate


REQUIRED_COLUMNS = [
    "sensitivity_type",
    "dataset_name",
    "task_id",
    "seed",
    "method",
    "posterior_samples",
    "lambda_hyb",
    "empirical_reference_coverage_95",
    "mean_posterior_std",
    "mean_ci_width",
    "uncertainty_error_correlation",
    "ci_width_error_correlation",
    "mean_spearman_between_seeds",
    "top_k_jaccard_between_seeds",
    "attribution_distance_to_default",
    "top_k_jaccard_to_default",
    "sign_agreement_to_default",
    "runtime_seconds",
    "status",
    "error_message",
]


@dataclass(frozen=True)
class SensitivityCondition:
    sensitivity_type: str
    posterior_samples: int
    lambda_hyb: float

    @property
    def split_id(self) -> str:
        return f"{self.sensitivity_type}_posterior_{self.posterior_samples}_lambda_{self.lambda_hyb:g}"


def sensitivity_conditions(cfg: Dict[str, Any]) -> List[SensitivityCondition]:
    posterior_grid = [int(v) for v in cfg.get("posterior_samples_grid", [cfg.get("posterior_samples", 500)])]
    lambda_grid = [float(v) for v in cfg.get("lambda_hyb_grid", [cfg.get("lambda_hyb", 0.5)])]
    default_posterior = int(cfg.get("posterior_samples", 500))
    default_lambda = float(cfg.get("lambda_hyb", 0.5))
    mode = str(cfg.get("mode", ""))

    if mode == "sensitivity_posterior":
        return [SensitivityCondition("posterior_samples", posterior, default_lambda) for posterior in posterior_grid]
    if mode == "sensitivity_lambda":
        return [SensitivityCondition("lambda_hyb", default_posterior, lambda_hyb) for lambda_hyb in lambda_grid]
    if mode == "sensitivity_full":
        return [SensitivityCondition("full_grid", posterior, lambda_hyb) for posterior in posterior_grid for lambda_hyb in lambda_grid]
    if mode == "sensitivity_smoke":
        posterior = posterior_grid[0]
        lambda_hyb = lambda_grid[0]
        return [
            SensitivityCondition("posterior_samples", posterior, lambda_hyb),
            SensitivityCondition("lambda_hyb", posterior, lambda_hyb),
        ]
    raise ValueError(f"unsupported sensitivity mode: {mode}")


def _condition_cfg(cfg: Dict[str, Any], condition: SensitivityCondition) -> Dict[str, Any]:
    out = dict(cfg)
    out["posterior_samples"] = int(condition.posterior_samples)
    out["lambda_hyb"] = float(condition.lambda_hyb)
    out["sensitivity_type"] = condition.sensitivity_type
    return out


def _append_summary_rows(
    rows: List[Dict[str, Any]],
    *,
    summary: pd.DataFrame,
    metadata: Dict[str, Any],
    condition: SensitivityCondition,
    method: str,
    explained_index: int,
    explained_order: int,
    n_train: int,
    n_test: int,
    runtime_seconds: float,
) -> None:
    for _, feature_row in summary.iterrows():
        ci_width = float(feature_row["ci_upper_95"] - feature_row["ci_lower_95"])
        rows.append(
            {
                **metadata,
                "sensitivity_type": condition.sensitivity_type,
                "method": method,
                "posterior_samples": int(condition.posterior_samples),
                "lambda_hyb": float(condition.lambda_hyb),
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
                "runtime_seconds": float(runtime_seconds),
                "metric_direction": "higher_is_better",
            }
        )


def _condition_seed_features(local: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if local.empty:
        return pd.DataFrame()
    for _, sub in local.groupby(["sensitivity_type", "posterior_samples", "lambda_hyb"], dropna=False):
        feature_df = _seed_feature_aggregate(sub)
        if feature_df.empty:
            continue
        for col in ["sensitivity_type", "posterior_samples", "lambda_hyb"]:
            feature_df[col] = sub[col].iloc[0]
        if "seed" in feature_df.columns:
            feature_df["seed"] = feature_df["seed"].astype(str)
        frames.append(feature_df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _condition_public_metrics(seed_features: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if seed_features.empty:
        return pd.DataFrame()
    for (sensitivity_type, posterior_samples, lambda_hyb), sub in seed_features.groupby(["sensitivity_type", "posterior_samples", "lambda_hyb"], dropna=False):
        condition_cfg = dict(cfg)
        condition_cfg["posterior_samples"] = int(posterior_samples)
        condition_cfg["lambda_hyb"] = float(lambda_hyb)
        metrics = _compute_public_uncertainty_metrics(sub, condition_cfg)
        if metrics.empty:
            continue
        metrics["sensitivity_type"] = str(sensitivity_type)
        metrics["posterior_samples"] = int(posterior_samples)
        metrics["lambda_hyb"] = float(lambda_hyb)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _condition_stability_metrics(local: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if local.empty:
        return pd.DataFrame()
    for (sensitivity_type, posterior_samples, lambda_hyb), sub in local.groupby(["sensitivity_type", "posterior_samples", "lambda_hyb"], dropna=False):
        feature_df = stability_seed_features(sub)
        condition_cfg = dict(cfg)
        condition_cfg["posterior_samples"] = int(posterior_samples)
        condition_cfg["lambda_hyb"] = float(lambda_hyb)
        metrics = stability_metrics(feature_df, condition_cfg)
        if metrics.empty:
            continue
        metrics["sensitivity_type"] = str(sensitivity_type)
        metrics["posterior_samples"] = int(posterior_samples)
        metrics["lambda_hyb"] = float(lambda_hyb)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _default_comparisons(seed_features: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if seed_features.empty:
        return pd.DataFrame()
    default_posterior = int(cfg.get("posterior_samples", 500))
    default_lambda = float(cfg.get("lambda_hyb", 0.5))
    top_k = int(cfg.get("top_k", 5))
    rows: List[Dict[str, Any]] = []

    group_cols = ["sensitivity_type", "dataset_name", "task_id", "fold", "method", "seed"]
    for key, group in seed_features.groupby(group_cols, dropna=False):
        sensitivity_type, dataset_name, task_id, fold, method, seed = key
        ref = group[
            (group["posterior_samples"].astype(int) == default_posterior)
            & np.isclose(group["lambda_hyb"].astype(float), default_lambda)
        ][["feature", "posterior_mean"]].rename(columns={"posterior_mean": "default_phi"})
        for (posterior_samples, lambda_hyb), sub in group.groupby(["posterior_samples", "lambda_hyb"], dropna=False):
            merged = sub.merge(ref, on="feature", how="inner")
            if merged.empty:
                distance = float("nan")
                jaccard = float("nan")
                sign_agreement = float("nan")
            else:
                current = merged["posterior_mean"].to_numpy(dtype=float)
                default = merged["default_phi"].to_numpy(dtype=float)
                distance = float(np.sqrt(np.mean((current - default) ** 2)))
                jaccard = top_k_jaccard(current, default, min(top_k, current.size))
                sign_agreement = float(np.mean(np.sign(current) == np.sign(default)))
            rows.append(
                {
                    "sensitivity_type": str(sensitivity_type),
                    "dataset_name": dataset_name,
                    "task_id": task_id,
                    "fold": fold,
                    "method": method,
                    "seed": str(seed),
                    "posterior_samples": int(posterior_samples),
                    "lambda_hyb": float(lambda_hyb),
                    "attribution_distance_to_default": distance,
                    "top_k_jaccard_to_default": jaccard,
                    "sign_agreement_to_default": sign_agreement,
                }
            )
    return pd.DataFrame(rows)


def _runtime_by_condition(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty or "runtime_seconds" not in local.columns:
        return pd.DataFrame()
    unit_cols = [
        "sensitivity_type",
        "dataset_name",
        "task_id",
        "fold",
        "seed",
        "method",
        "posterior_samples",
        "lambda_hyb",
        "explained_index",
        "explained_order",
    ]
    units = local.drop_duplicates([col for col in unit_cols if col in local.columns])
    out = (
        units.groupby(["sensitivity_type", "dataset_name", "task_id", "fold", "seed", "method", "posterior_samples", "lambda_hyb"], dropna=False)["runtime_seconds"]
        .sum()
        .reset_index()
    )
    if "seed" in out.columns:
        out["seed"] = out["seed"].astype(str)
    return out


def _assemble_results(local: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    seed_features = _condition_seed_features(local)
    public_metrics = _condition_public_metrics(seed_features, cfg)
    stability = _condition_stability_metrics(local, cfg)
    default_metrics = _default_comparisons(seed_features, cfg)
    runtime = _runtime_by_condition(local)

    if public_metrics.empty:
        base = default_metrics.copy()
    else:
        base = public_metrics.copy()

    merge_keys = ["sensitivity_type", "dataset_name", "task_id", "fold", "method", "posterior_samples", "lambda_hyb"]
    if not stability.empty:
        keep = merge_keys + ["mean_spearman_between_seeds", "top_k_jaccard_between_seeds"]
        base = base.merge(stability[[col for col in keep if col in stability.columns]], on=merge_keys, how="left")

    seed_keys = merge_keys + ["seed"]
    if not default_metrics.empty:
        base = base.merge(default_metrics, on=seed_keys, how="left")
    if not runtime.empty:
        base = base.merge(runtime, on=seed_keys, how="left", suffixes=("", "_measured"))
        if "runtime_seconds_measured" in base.columns:
            base["runtime_seconds"] = base["runtime_seconds_measured"]
            base = base.drop(columns=["runtime_seconds_measured"])

    if "runtime_seconds" not in base.columns:
        base["runtime_seconds"] = np.nan
    for col in REQUIRED_COLUMNS:
        if col not in base.columns:
            base[col] = np.nan
    return base[REQUIRED_COLUMNS + [col for col in base.columns if col not in REQUIRED_COLUMNS]]


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=[col for col in REQUIRED_COLUMNS if col != "seed"])
    group_cols = ["sensitivity_type", "dataset_name", "task_id", "method", "posterior_samples", "lambda_hyb"]
    numeric_cols = [
        "empirical_reference_coverage_95",
        "mean_posterior_std",
        "mean_ci_width",
        "uncertainty_error_correlation",
        "ci_width_error_correlation",
        "mean_spearman_between_seeds",
        "top_k_jaccard_between_seeds",
        "attribution_distance_to_default",
        "top_k_jaccard_to_default",
        "sign_agreement_to_default",
        "runtime_seconds",
    ]
    available_numeric = [col for col in numeric_cols if col in results.columns]
    grouped = results.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n_rows=("method", "size"),
        n_seeds=("seed", "nunique"),
        **{col: (col, "mean") for col in available_numeric},
    ).reset_index()
    status_counts = results.groupby(group_cols + ["status"], dropna=False).size().unstack(fill_value=0).reset_index()
    status_counts = status_counts.rename(columns={value: f"status_{value}" for value in status_counts.columns if value not in group_cols})
    return summary.merge(status_counts, on=group_cols, how="left")


def _write_result_set(results: pd.DataFrame, path: Path, summary_path: Path) -> None:
    write_csv(results, path)
    write_csv(_summary(results), summary_path)


def _filtered_outputs(results: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    default_posterior = int(cfg.get("posterior_samples", 500))
    default_lambda = float(cfg.get("lambda_hyb", 0.5))
    if results.empty:
        return results.copy(), results.copy()
    posterior = results[
        (results["sensitivity_type"] == "posterior_samples")
        | ((results["sensitivity_type"] == "full_grid") & np.isclose(results["lambda_hyb"].astype(float), default_lambda))
    ].copy()
    lambda_df = results[
        (results["sensitivity_type"] == "lambda_hyb")
        | ((results["sensitivity_type"] == "full_grid") & (results["posterior_samples"].astype(int) == default_posterior))
    ].copy()
    return posterior, lambda_df


def run_sensitivity(cfg: Dict[str, Any]) -> pd.DataFrame:
    methods: Sequence[str] = cfg.get("uncertainty_methods", ["bayesian_hyb_fpde"])
    conditions = sensitivity_conditions(cfg)
    rows: List[Dict[str, Any]] = []

    for seed in cfg.get("seeds", [0]):
        for task_id, payload, split_name in load_tabular_openml_or_local(cfg, seed=int(seed), mode=str(cfg.get("mode", ""))):
            dataset_name, X_train, y_train, X_test, y_test, model, feature_names, _ = payload
            pred = model.predict(X_test)
            indices = explain_indices(y_test, pred, int(cfg.get("n_explain", 20)), seed=int(seed))
            for condition in conditions:
                condition_cfg = _condition_cfg(cfg, condition)
                for method in methods:
                    hashes = config_hashes_for_job(
                        condition_cfg,
                        dataset_name=dataset_name,
                        task_id=task_id,
                        seed=int(seed),
                        fold=split_name,
                        split_id=condition.split_id,
                        methods=[method],
                        n_explain=int(condition_cfg.get("n_explain", 20)),
                        posterior_samples=int(condition.posterior_samples),
                        bootstrap_samples=int(condition_cfg.get("bootstrap_samples", 30)),
                        top_k=int(condition_cfg.get("top_k", 5)),
                        lambda_hyb=float(condition.lambda_hyb),
                        extra_context={"sensitivity_type": condition.sensitivity_type},
                    )
                    metadata = base_metadata(
                        dataset_name=dataset_name,
                        task_id=task_id,
                        seed=int(seed),
                        fold=split_name,
                        split_id=condition.split_id,
                        mode=str(cfg.get("mode", "")),
                        method=method,
                        **hashes,
                        status="ok",
                        error_message="",
                    )
                    for order, idx in enumerate(indices.tolist()):
                        start = time.perf_counter()
                        try:
                            summary = _method_summary(
                                method=method,
                                model=model,
                                x=X_test[idx],
                                X_train=X_train,
                                y_train=y_train,
                                feature_names=feature_names,
                                posterior_samples=int(condition.posterior_samples),
                                bootstrap_samples=int(condition_cfg.get("bootstrap_samples", 30)),
                                top_k=int(condition_cfg.get("top_k", 5)),
                                tau=float(condition_cfg.get("tau", 0.0)),
                                lambda_hyb=float(condition.lambda_hyb),
                                seed=int(seed) + int(order),
                            )
                            elapsed = float(time.perf_counter() - start)
                            _append_summary_rows(
                                rows,
                                summary=summary,
                                metadata=metadata,
                                condition=condition,
                                method=method,
                                explained_index=int(idx),
                                explained_order=int(order),
                                n_train=int(X_train.shape[0]),
                                n_test=int(X_test.shape[0]),
                                runtime_seconds=elapsed,
                            )
                        except Exception as exc:
                            rows.append(
                                {
                                    **metadata,
                                    "sensitivity_type": condition.sensitivity_type,
                                    "posterior_samples": int(condition.posterior_samples),
                                    "lambda_hyb": float(condition.lambda_hyb),
                                    "explained_index": int(idx),
                                    "explained_order": int(order),
                                    "n_train": int(X_train.shape[0]),
                                    "n_test": int(X_test.shape[0]),
                                    "runtime_seconds": float(time.perf_counter() - start),
                                    "status": "error",
                                    "error_message": f"{type(exc).__name__}: {exc}",
                                }
                            )
    local = pd.DataFrame(rows)
    results_dir = Path(cfg.get("results_dir", "results_ieee_sensitivity"))
    write_parquet_or_csv(local, results_dir / "sensitivity_local_explanations.parquet")
    seed_features = _condition_seed_features(local)
    write_csv(seed_features, results_dir / "sensitivity_seed_features.csv")
    return _assemble_results(local, cfg)


def main() -> int:
    args = parser_with_config("Run IEEE Access Bayesian-FPDE sensitivity analysis.").parse_args()
    cfg = load_mode_config(args.config, args.mode, runner_name="experiments.run_sensitivity_analysis")
    cfg = apply_task_id_filter(cfg, args.task_id)
    results_dir = Path(cfg.get("results_dir", "results_ieee_sensitivity"))
    figures_dir = Path(cfg.get("figures_dir", "figures_ieee_sensitivity"))
    logs_dir = Path(cfg.get("logs_dir", "logs_ieee_sensitivity"))
    ensure_dirs(results_dir, figures_dir, logs_dir)

    results = run_sensitivity(cfg)
    posterior, lambda_df = _filtered_outputs(results, cfg)

    _write_result_set(results, results_dir / "sensitivity_results.csv", results_dir / "sensitivity_summary.csv")
    _write_result_set(posterior, results_dir / "posterior_samples_sensitivity.csv", results_dir / "posterior_samples_sensitivity_summary.csv")
    _write_result_set(lambda_df, results_dir / "lambda_hyb_sensitivity.csv", results_dir / "lambda_hyb_sensitivity_summary.csv")

    save_line_plot(
        posterior,
        x="posterior_samples",
        y="mean_ci_width",
        group="dataset_name",
        path=figures_dir / "posterior_samples_mean_ci_width.png",
        title="Posterior sample sensitivity",
    )
    save_line_plot(
        lambda_df,
        x="lambda_hyb",
        y="attribution_distance_to_default",
        group="dataset_name",
        path=figures_dir / "lambda_hyb_distance_to_default.png",
        title="Lambda sensitivity",
    )
    return 0
