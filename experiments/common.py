from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from bayesian_fpde.baselines import optional_baseline
from bayesian_fpde.bayesian_fpde import BayesianFPDEConfig, explain_bayesian_fpde, summarize_samples
from bayesian_fpde.bootstrap_fpde import bootstrap_fpde_samples
from bayesian_fpde.datasets import (
    encode_labels,
    fit_black_box,
    generate_synthetic_gaussian,
    get_suite_task_ids,
    load_case_study_dataset,
    load_openml_task,
    preprocess_train_test,
    split_openml_or_stratified,
)
from bayesian_fpde.fpde import FPDEConfig, class_prototypes, explain_fpde
from bayesian_fpde.metrics import deletion_insertion_metrics, stability_metrics, top_k_jaccard
from bayesian_fpde.utils import base_metadata, config_hash as stable_config_hash, ensure_dirs, load_yaml, mode_config, setup_logging, write_csv, write_json, write_parquet_or_csv


DETERMINISTIC_METHODS = {
    "diff_fpde": FPDEConfig(mode="diff"),
    "cos_fpde": FPDEConfig(mode="cos"),
    "hyb_fpde": FPDEConfig(mode="hyb", lambda_hyb=0.5),
    "hyb_fpde_grid": FPDEConfig(mode="hyb", lambda_hyb=0.5),
}

BAYESIAN_METHODS = {
    "bayesian_diff_fpde": ("diff", 1.0),
    "bayesian_cos_fpde": ("cos", 0.0),
    "bayesian_hyb_fpde": ("hyb", 0.5),
}

OPTIONAL_METHODS = {"shap", "lime", "aime", "bayesshap", "bayeslime", "bayesian_aime"}
EXPLANATION_UNIT_COLS = ["dataset_name", "task_id", "seed", "fold", "explained_index"]
DATASET_SEED_UNIT_COLS = ["dataset_name", "task_id", "seed", "fold"]


def parser_with_config(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--task-id", default=None, help="Comma-separated OpenML task_id filter.")
    return parser


def load_mode_config(path: str, mode: str, *, runner_name: str = "", runner_invocation_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = load_yaml(path)
    if runner_invocation_context:
        raw = dict(raw)
        raw["runner_invocation_context"] = runner_invocation_context
    return mode_config(raw, mode, runner_name=runner_name)


def apply_task_id_filter(cfg: Dict[str, Any], task_id_filter: Optional[str]) -> Dict[str, Any]:
    if not task_id_filter:
        return cfg
    task_ids = [int(value.strip()) for value in str(task_id_filter).split(",") if value.strip()]
    if not task_ids:
        return cfg
    filtered = dict(cfg)
    filtered["task_ids"] = task_ids
    max_tasks = filtered.get("max_tasks")
    if max_tasks:
        filtered["max_tasks"] = min(int(max_tasks), len(task_ids))
    return filtered


def job_config_hash_for(
    *,
    experiment_config_hash: str = "",
    runner_invocation_hash: str = "",
    workflow_run_id: str = "",
    run_config_hash: str = "",
    mode: str,
    dataset_name: str,
    task_id: int | str = "",
    seed: int | str = "",
    fold: int | str = "",
    split_id: int | str = "",
    methods: Sequence[str] = (),
    n_explain: int = 0,
    posterior_samples: int = 0,
    bootstrap_samples: int = 0,
    tau: float = 0.0,
    top_k: int = 0,
    lambda_hyb: float = 0.0,
    max_background: int = 0,
    extra_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Hash the job-specific knobs that distinguish dataset/seed/fold work."""

    return stable_config_hash(
        {
            "experiment_config_hash": experiment_config_hash or run_config_hash,
            "workflow_run_id": workflow_run_id,
            "runner_invocation_hash": runner_invocation_hash or run_config_hash,
            "mode": mode,
            "dataset_name": dataset_name,
            "task_id": task_id,
            "seed": seed,
            "fold": fold,
            "split_id": split_id,
            "methods": list(methods),
            "n_explain": int(n_explain),
            "posterior_samples": int(posterior_samples),
            "bootstrap_samples": int(bootstrap_samples),
            "tau": float(tau),
            "top_k": int(top_k),
            "lambda_hyb": float(lambda_hyb),
            "max_background": int(max_background),
            "extra_context": extra_context or {},
        }
    )


def config_hashes_for_job(
    cfg: Dict[str, Any],
    *,
    dataset_name: str,
    task_id: int | str = "",
    seed: int | str = "",
    fold: int | str = "",
    split_id: int | str = "",
    methods: Sequence[str] = (),
    n_explain: int = 0,
    posterior_samples: int = 0,
    bootstrap_samples: int = 0,
    tau: float = 0.0,
    top_k: int = 0,
    lambda_hyb: float = 0.0,
    max_background: int = 0,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    experiment_hash = str(cfg.get("experiment_config_hash", cfg.get("config_hash", "")))
    workflow_run_id = str(cfg.get("workflow_run_id", ""))
    workflow_run_attempt = str(cfg.get("workflow_run_attempt", ""))
    workflow_name = str(cfg.get("workflow_name", ""))
    workflow_ref = str(cfg.get("workflow_ref", ""))
    workflow_sha = str(cfg.get("workflow_sha", ""))
    runner_hash = str(cfg.get("runner_invocation_hash", cfg.get("run_config_hash", experiment_hash)))
    return {
        "config_hash": experiment_hash,
        "experiment_config_hash": experiment_hash,
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
        "workflow_name": workflow_name,
        "workflow_ref": workflow_ref,
        "workflow_sha": workflow_sha,
        "runner_invocation_hash": runner_hash,
        "run_config_hash": runner_hash,
        "job_config_hash": job_config_hash_for(
            experiment_config_hash=experiment_hash,
            workflow_run_id=workflow_run_id,
            runner_invocation_hash=runner_hash,
            mode=str(cfg.get("mode", "")),
            dataset_name=dataset_name,
            task_id=task_id,
            seed=seed,
            fold=fold,
            split_id=split_id,
            methods=methods,
            n_explain=n_explain,
            posterior_samples=posterior_samples,
            bootstrap_samples=bootstrap_samples,
            tau=tau,
            top_k=top_k,
            lambda_hyb=lambda_hyb,
            max_background=max_background,
            extra_context=extra_context,
        ),
    }


def explain_indices(y: np.ndarray, pred: np.ndarray, n_explain: int, *, correct_only: bool = True, seed: int = 0) -> np.ndarray:
    eligible = np.where(pred == y)[0] if correct_only else np.arange(len(y))
    if eligible.size == 0:
        eligible = np.arange(len(y))
    rng = np.random.default_rng(seed)
    if n_explain and eligible.size > n_explain:
        eligible = rng.choice(eligible, size=n_explain, replace=False)
    return np.asarray(sorted(eligible.tolist()), dtype=int)


def evaluate_methods_for_dataset(
    *,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: Any,
    feature_names: Sequence[str],
    methods: Sequence[str],
    seed: int,
    task_id: int | str = "",
    fold: int | str = "",
    n_explain: int = 10,
    posterior_samples: int = 100,
    bootstrap_samples: int = 30,
    tau: float = 0.0,
    top_k: int = 5,
    lambda_hyb: float = 0.5,
    mode: str = "",
    config_hash: str = "",
    run_config_hash: str = "",
    experiment_config_hash: str = "",
    workflow_run_id: str = "",
    workflow_run_attempt: str = "",
    workflow_name: str = "",
    workflow_ref: str = "",
    workflow_sha: str = "",
    runner_invocation_hash: str = "",
    job_config_hash: str = "",
    max_background: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prototypes, labels = class_prototypes(X_train, y_train)
    baseline = np.mean(X_train, axis=0)
    pred = model.predict(X_test)
    perf = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    indices = explain_indices(y_test, pred, n_explain, seed=seed)
    metric_rows: List[Dict[str, Any]] = []
    local_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    effective_experiment_config_hash = str(experiment_config_hash or config_hash or run_config_hash)
    effective_workflow_run_id = str(workflow_run_id)
    effective_workflow_run_attempt = str(workflow_run_attempt)
    effective_workflow_name = str(workflow_name)
    effective_workflow_ref = str(workflow_ref)
    effective_workflow_sha = str(workflow_sha)
    effective_runner_invocation_hash = str(runner_invocation_hash or run_config_hash or effective_experiment_config_hash)
    effective_job_config_hash = str(
        job_config_hash
        or job_config_hash_for(
            experiment_config_hash=effective_experiment_config_hash,
            workflow_run_id=effective_workflow_run_id,
            runner_invocation_hash=effective_runner_invocation_hash,
            mode=mode,
            dataset_name=dataset_name,
            task_id=task_id,
            seed=seed,
            fold=fold,
            split_id=fold,
            methods=methods,
            n_explain=n_explain,
            posterior_samples=posterior_samples,
            bootstrap_samples=bootstrap_samples,
            tau=tau,
            top_k=top_k,
            lambda_hyb=lambda_hyb,
            max_background=max_background,
        )
    )
    metadata = base_metadata(
        dataset_name=dataset_name,
        task_id=task_id,
        seed=seed,
        fold=fold,
        split_id=fold,
        mode=mode,
        config_hash=effective_experiment_config_hash,
        experiment_config_hash=effective_experiment_config_hash,
        workflow_run_id=effective_workflow_run_id,
        workflow_run_attempt=effective_workflow_run_attempt,
        workflow_name=effective_workflow_name,
        workflow_ref=effective_workflow_ref,
        workflow_sha=effective_workflow_sha,
        runner_invocation_hash=effective_runner_invocation_hash,
        run_config_hash=effective_runner_invocation_hash,
        job_config_hash=effective_job_config_hash,
    )

    for order, idx in enumerate(indices.tolist()):
        x = X_test[idx]
        y_true = int(y_test[idx])
        proba = np.asarray(model.predict_proba(x.reshape(1, -1))[0], dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(proba.size)), dtype=int)
        pred_label = int(classes[int(np.argmax(proba))])
        for method in methods:
            start = time.perf_counter()
            status = "ok"
            error_message = ""
            attr: Optional[np.ndarray] = None
            summary: Optional[pd.DataFrame] = None
            target_label = pred_label
            method_key = method.lower()
            dependency_available = method_key not in OPTIONAL_METHODS
            explanation_model_calls = 0
            background_size = 0
            row_max_background: Optional[int] = None
            if method_key in OPTIONAL_METHODS:
                result = optional_baseline(method_key, model, x, X_train, y_train, feature_names, target_label, seed=seed, max_background=max_background)
                status, error_message, attr = result.status, result.error_message, result.attribution
                dependency_available = bool(result.dependency_available)
                explanation_model_calls = int(result.n_model_calls)
                background_size = int(result.background_size)
                row_max_background = result.max_background
            else:
                try:
                    if method_key in DETERMINISTIC_METHODS:
                        cfg = DETERMINISTIC_METHODS[method_key]
                        if cfg.mode == "hyb":
                            cfg = FPDEConfig(mode="hyb", lambda_hyb=lambda_hyb)
                        attr, target_label, _ = explain_fpde(model, x, prototypes, labels, config=cfg, anchor=baseline)
                        explanation_model_calls = 1
                    elif method_key in BAYESIAN_METHODS:
                        fpde_mode, default_lambda = BAYESIAN_METHODS[method_key]
                        cfg = BayesianFPDEConfig(
                            mode=fpde_mode,
                            lambda_hyb=lambda_hyb if fpde_mode == "hyb" else default_lambda,
                            n_posterior_samples=posterior_samples,
                            tau=tau,
                            top_k=top_k,
                        )
                        result = explain_bayesian_fpde(
                            model,
                            x,
                            X_train,
                            y_train,
                            config=cfg,
                            anchor=baseline,
                            feature_names=feature_names,
                            seed=seed + order,
                        )
                        attr = result.summary["posterior_mean"].to_numpy(dtype=float)
                        target_label = result.positive_label
                        summary = result.summary
                        explanation_model_calls = 1
                    elif method_key == "bootstrap_fpde":
                        samples = bootstrap_fpde_samples(
                            model,
                            x,
                            X_train,
                            y_train,
                            n_bootstrap=bootstrap_samples,
                            config=FPDEConfig(mode="hyb", lambda_hyb=lambda_hyb),
                            anchor=baseline,
                            seed=seed + order,
                        )
                        summary = summarize_samples(samples, feature_names=feature_names, tau=tau, top_k=top_k)
                        attr = summary["posterior_mean"].to_numpy(dtype=float)
                        explanation_model_calls = int(samples.shape[0])
                    else:
                        status = "skipped"
                        error_message = f"unknown or optional method not enabled: {method}"
                except Exception as exc:
                    status = "error"
                    error_message = f"{type(exc).__name__}: {exc}"
            elapsed = float(time.perf_counter() - start)
            row_base = {
                **metadata,
                "method": method,
                "status": status,
                "error_message": error_message,
                "dependency_available": dependency_available,
                "explained_index": int(idx),
                "explained_order": int(order),
                "true_label": y_true,
                "pred_label": pred_label,
                "target_label": int(target_label),
                "n_features": int(X_train.shape[1]),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "runtime_seconds": elapsed,
                "explanation_model_calls": explanation_model_calls,
                "evaluation_model_calls": 0,
                "total_model_calls": explanation_model_calls,
                "n_model_calls": explanation_model_calls,
                "number_of_model_calls": explanation_model_calls,
                "background_size": background_size,
                "max_background": row_max_background,
                **perf,
            }
            runtime_rows.append(dict(row_base))
            if status == "ok" and attr is not None:
                metric = deletion_insertion_metrics(model, x, attr, baseline, target_label=target_label)
                evaluation_model_calls = int(2 * len((0.0, 0.1, 0.2, 0.5, 1.0)) + 1 + x.size)
                total_model_calls = int(explanation_model_calls + evaluation_model_calls)
                metric["explanation_model_calls"] = explanation_model_calls
                metric["evaluation_model_calls"] = evaluation_model_calls
                metric["total_model_calls"] = total_model_calls
                metric["number_of_model_calls"] = total_model_calls
                metric["n_model_calls"] = total_model_calls
                # Higher is better for both terms:
                # deletion_drop_auc = p0 - deletion_auc, so larger probability drop is better;
                # insertion_auc rewards recovering target probability early as features are inserted.
                metric["combined_score"] = float((metric["deletion_drop_auc"] + metric["insertion_auc"]) / 2.0)
                metric["metric_direction"] = "higher_is_better"
                row_base = {**row_base, "evaluation_model_calls": evaluation_model_calls, "total_model_calls": total_model_calls, "n_model_calls": total_model_calls, "number_of_model_calls": total_model_calls}
                runtime_rows[-1].update(row_base)
                metric_rows.append({**row_base, **metric, "top_k_jaccard": np.nan})
                if summary is not None:
                    for _, feature_row in summary.iterrows():
                        local_rows.append({**row_base, **feature_row.to_dict(), "attribution": float(feature_row["posterior_mean"])})
                else:
                    for j, value in enumerate(attr):
                        local_rows.append(
                            {
                                **row_base,
                                "feature": str(feature_names[j]),
                                "feature_index": int(j),
                                "attribution": float(value),
                                "posterior_mean": np.nan,
                                "posterior_std": np.nan,
                                "ci_lower_95": np.nan,
                                "ci_upper_95": np.nan,
                                "p_positive": np.nan,
                                "p_negative": np.nan,
                                "p_abs_gt_tau": np.nan,
                                "rank_mean": np.nan,
                                "rank_std": np.nan,
                                "rank_probability_top_k": np.nan,
                            }
                        )
            else:
                metric_rows.append({**row_base, "p0": np.nan, "deletion_auc": np.nan, "deletion_drop_auc": np.nan, "insertion_auc": np.nan, "comprehensiveness": np.nan, "sufficiency": np.nan, "faithfulness_correlation": np.nan, "faithfulness_delta_mean": np.nan, "faithfulness_delta_abs_mean": np.nan, "combined_score": np.nan, "metric_direction": "higher_is_better", "top_k_jaccard": np.nan})
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        det = metrics[metrics["method"].isin(["hyb_fpde", "hyb_fpde_grid"])][["explained_index", "deletion_drop_auc"]]
        # top-k Jaccard is computed from local rows when a deterministic reference exists.
        ref_local = pd.DataFrame(local_rows)
        if not ref_local.empty:
            for key, sub in ref_local.groupby(["explained_index"]):
                ref = sub[sub["method"].isin(["hyb_fpde", "hyb_fpde_grid"])]
                if ref.empty:
                    continue
                ref_attr = ref.sort_values("feature_index")["attribution"].to_numpy(dtype=float)
                for method, msub in sub.groupby("method"):
                    attr = msub.sort_values("feature_index")["attribution"].to_numpy(dtype=float)
                    value = top_k_jaccard(attr, ref_attr, min(top_k, attr.size))
                    mask = (metrics["explained_index"] == key) & (metrics["method"] == method)
                    metrics.loc[mask, "top_k_jaccard"] = value
    return pd.DataFrame(local_rows), metrics, pd.DataFrame(runtime_rows)


def make_local_smoke_dataset(seed: int = 0):
    data = generate_synthetic_gaussian(n_samples=120, n_features=8, n_informative=3, n_classes=2, random_seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.25, random_state=seed, stratify=data.y)
    model, model_name = fit_black_box(X_train, y_train, seed=seed, model_name="random_forest")
    return "local_smoke_gaussian", X_train, y_train, X_test, y_test, model, data.feature_names, model_name


def load_tabular_openml_or_local(cfg: Dict[str, Any], *, seed: int, mode: str):
    if cfg.get("local_smoke", False):
        yield ("", make_local_smoke_dataset(seed), "local_smoke")
        return
    task_ids = cfg.get("task_ids")
    if not task_ids:
        task_ids = get_suite_task_ids(int(cfg.get("suite_id", 99)))
    max_tasks = cfg.get("max_tasks")
    if max_tasks:
        task_ids = task_ids[: int(max_tasks)]
    for task_id in task_ids:
        task, X_df, y_ser, dataset_name = load_openml_task(int(task_id))
        mask = ~pd.isna(y_ser)
        X_df = X_df.loc[mask].reset_index(drop=True)
        y = encode_labels(y_ser.loc[mask].reset_index(drop=True))
        train_idx, test_idx, split_name = split_openml_or_stratified(
            task,
            X_df,
            y,
            seed=seed,
            fold=int(cfg.get("fold", 0)),
            repeat=int(cfg.get("repeat", 0)),
            sample=int(cfg.get("sample", 0)),
        )
        max_train = int(cfg.get("max_train_rows", 0) or 0)
        max_test = int(cfg.get("max_test_rows", 0) or 0)
        if max_train and train_idx.size > max_train:
            train_idx = np.random.default_rng(seed).choice(train_idx, size=max_train, replace=False)
        if max_test and test_idx.size > max_test:
            test_idx = np.random.default_rng(seed + 1).choice(test_idx, size=max_test, replace=False)
        X_train, X_test, names, _ = preprocess_train_test(
            X_df.iloc[train_idx].reset_index(drop=True),
            X_df.iloc[test_idx].reset_index(drop=True),
        )
        y_train, y_test = y[train_idx], y[test_idx]
        model, model_name = fit_black_box(X_train, y_train, seed=seed, model_name=str(cfg.get("model", "auto")), feature_names_in=names)
        yield (int(task_id), (dataset_name, X_train, y_train, X_test, y_test, model, names, model_name), split_name)


def _status_counts(group: pd.DataFrame) -> pd.Series:
    counts = group["status"].value_counts() if "status" in group.columns else pd.Series(dtype=int)
    return pd.Series(
        {
            "status_ok": int(counts.get("ok", 0)),
            "status_skipped": int(counts.get("skipped", 0)),
            "status_error": int(counts.get("error", 0)),
        }
    )


def _unique_nonempty(values: pd.Series) -> List[str]:
    return sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})


def _unique_explanation_units(group: pd.DataFrame) -> int:
    unit_cols = [col for col in EXPLANATION_UNIT_COLS if col in group.columns]
    if unit_cols:
        return int(group[unit_cols].drop_duplicates().shape[0])
    return int(len(group))


def _hash_consistency(group: pd.DataFrame) -> pd.Series:
    experiment_hashes = _unique_nonempty(group["experiment_config_hash"]) if "experiment_config_hash" in group.columns else []
    if not experiment_hashes:
        experiment_hashes = _unique_nonempty(group["config_hash"]) if "config_hash" in group.columns else []
    workflow_ids = _unique_nonempty(group["workflow_run_id"]) if "workflow_run_id" in group.columns else []
    runner_hashes = _unique_nonempty(group["runner_invocation_hash"]) if "runner_invocation_hash" in group.columns else []
    if not runner_hashes:
        runner_hashes = _unique_nonempty(group["run_config_hash"]) if "run_config_hash" in group.columns else []
    job_hashes = _unique_nonempty(group["job_config_hash"]) if "job_config_hash" in group.columns else []
    experiment_consistent = len(experiment_hashes) <= 1
    workflow_consistent = len(workflow_ids) <= 1
    experiment_hash = experiment_hashes[0] if experiment_consistent and experiment_hashes else ("multiple" if experiment_hashes else "")
    runner_hash = runner_hashes[0] if len(runner_hashes) == 1 else ("multiple" if runner_hashes else "")
    return pd.Series(
        {
            "config_hash": experiment_hash,
            "experiment_config_hash": experiment_hash,
            "n_experiment_config_hashes": int(len(experiment_hashes)),
            "experiment_config_hash_consistent": bool(experiment_consistent),
            "experiment_config_hashes": ",".join(experiment_hashes) if len(experiment_hashes) <= 10 else "",
            "workflow_run_id": workflow_ids[0] if workflow_consistent and workflow_ids else ("multiple" if workflow_ids else ""),
            "n_workflow_run_ids": int(len(workflow_ids)),
            "workflow_run_id_consistent": bool(workflow_consistent),
            "workflow_run_ids": ",".join(workflow_ids) if len(workflow_ids) <= 10 else "",
            "runner_invocation_hash": runner_hash,
            "run_config_hash": runner_hash,
            "n_runner_invocation_hashes": int(len(runner_hashes)),
            "runner_invocation_hashes": ",".join(runner_hashes) if len(runner_hashes) <= 10 else "",
            # Deprecated aliases retained for old downstream readers.
            "n_run_config_hashes": int(len(runner_hashes)),
            "run_config_hash_consistent": bool(len(runner_hashes) <= 1),
            "run_config_hashes": ",".join(runner_hashes) if len(runner_hashes) <= 10 else "",
            "job_config_hash": job_hashes[0] if job_hashes else "",
            "n_job_config_hashes": int(len(job_hashes)),
            "job_config_hashes": ",".join(job_hashes) if len(job_hashes) <= 10 else "",
        }
    )


def _openml_summary(metrics: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    mean_cols = [
        "deletion_auc",
        "deletion_drop_auc",
        "insertion_auc",
        "faithfulness_correlation",
        "number_of_model_calls",
        "explanation_model_calls",
        "evaluation_model_calls",
        "total_model_calls",
        "top_k_jaccard",
        "test_accuracy",
        "combined_score",
        "background_size",
        "max_background",
    ]
    available_mean_cols = [col for col in mean_cols if col in metrics.columns]
    grouped = metrics.groupby(list(group_cols), dropna=False)
    summary = grouped.agg(
        n_rows=("method", "size"),
        n_explanation_rows=("method", "size"),
        n_datasets=("dataset_name", "nunique"),
        n_seeds=("seed", "nunique"),
        **{f"mean_{col}": (col, "mean") for col in available_mean_cols},
    ).reset_index()
    unique_units = grouped.apply(_unique_explanation_units).reset_index(name="n_unique_explanation_units")
    summary = summary.merge(unique_units, on=list(group_cols), how="left")
    summary["n_unique_explained_indices"] = summary["n_unique_explanation_units"]
    summary["n_explain_instances"] = summary["n_unique_explanation_units"]

    unit_parent_cols = list(dict.fromkeys(list(group_cols) + [col for col in DATASET_SEED_UNIT_COLS if col in metrics.columns]))
    per_dataset_seed = metrics.groupby(unit_parent_cols, dropna=False).apply(_unique_explanation_units).reset_index(name="explanation_units_per_dataset_seed")
    per_dataset_seed_grouped = per_dataset_seed.groupby(list(group_cols), dropna=False)["explanation_units_per_dataset_seed"].agg(
        mean_explanation_units_per_dataset_seed="mean",
        min_explanation_units_per_dataset_seed="min",
        max_explanation_units_per_dataset_seed="max",
    ).reset_index()
    summary = summary.merge(per_dataset_seed_grouped, on=list(group_cols), how="left")
    summary["mean_explain_instances_per_seed"] = summary["mean_explanation_units_per_dataset_seed"]
    summary["min_explain_instances_per_seed"] = summary["min_explanation_units_per_dataset_seed"]
    summary["max_explain_instances_per_seed"] = summary["max_explanation_units_per_dataset_seed"]

    if "runtime_seconds" in metrics.columns:
        runtime_all = grouped["runtime_seconds"].mean().reset_index(name="mean_runtime_seconds_all_rows")
        summary = summary.merge(runtime_all, on=list(group_cols), how="left")
        if "status" in metrics.columns:
            ok_runtime = metrics[metrics["status"] == "ok"].groupby(list(group_cols), dropna=False)["runtime_seconds"].mean().reset_index(name="mean_runtime_seconds_ok_only")
        else:
            ok_runtime = runtime_all.rename(columns={"mean_runtime_seconds_all_rows": "mean_runtime_seconds_ok_only"})
        summary = summary.merge(ok_runtime, on=list(group_cols), how="left")
        summary["mean_runtime_seconds"] = summary["mean_runtime_seconds_ok_only"]

    status = grouped.apply(_status_counts).reset_index()
    summary = summary.merge(status, on=list(group_cols), how="left")
    summary["metric_direction"] = "higher_is_better"
    for meta_col in ["mode", "git_commit", "workflow_run_attempt", "workflow_name", "workflow_ref", "workflow_sha"]:
        if meta_col in metrics.columns and meta_col not in summary.columns:
            meta = grouped[meta_col].agg(lambda s: next((str(v) for v in s if pd.notna(v) and str(v) != ""), "")).reset_index(name=meta_col)
            summary = summary.merge(meta, on=list(group_cols), how="left")
    hash_meta = grouped.apply(_hash_consistency).reset_index()
    summary = summary.merge(hash_meta, on=list(group_cols), how="left")
    return summary


def write_standard_outputs(results_dir: str | Path, figures_dir: str | Path, local: pd.DataFrame, metrics: pd.DataFrame, runtime: pd.DataFrame) -> None:
    ensure_dirs(results_dir, figures_dir)
    write_parquet_or_csv(local, Path(results_dir) / "openml_local_explanations.parquet")
    write_csv(_openml_summary(metrics, ["dataset_name", "task_id", "seed", "method"]), Path(results_dir) / "openml_seed_summary.csv")
    write_csv(_openml_summary(metrics, ["dataset_name", "task_id", "method"]), Path(results_dir) / "openml_global_summary.csv")
    write_csv(_openml_summary(metrics, ["method"]), Path(results_dir) / "openml_method_summary.csv")
    write_csv(metrics, Path(results_dir) / "openml_metrics.csv")
    write_csv(runtime, Path(results_dir) / "openml_runtime.csv")
