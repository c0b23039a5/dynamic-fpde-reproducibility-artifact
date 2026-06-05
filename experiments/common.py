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
from bayesian_fpde.utils import base_metadata, ensure_dirs, load_yaml, mode_config, setup_logging, write_csv, write_json, write_parquet_or_csv


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


def parser_with_config(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="smoke")
    return parser


def load_mode_config(path: str, mode: str) -> Dict[str, Any]:
    return mode_config(load_yaml(path), mode)


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
    metadata = base_metadata(dataset_name=dataset_name, task_id=task_id, seed=seed, fold=fold)

    for order, idx in enumerate(indices.tolist()):
        x = X_test[idx]
        y_true = int(y_test[idx])
        proba = np.asarray(model.predict_proba(x.reshape(1, -1))[0], dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(proba.size)), dtype=int)
        pred_label = int(classes[int(np.argmax(proba))])
        for method in methods:
            start = time.perf_counter()
            status = "ok"
            error = ""
            attr: Optional[np.ndarray] = None
            summary: Optional[pd.DataFrame] = None
            target_label = pred_label
            method_key = method.lower()
            if method_key in OPTIONAL_METHODS:
                base_name = {"bayesshap": "shap", "bayeslime": "lime", "bayesian_aime": "aime"}.get(method_key, method_key)
                result = optional_baseline(base_name, model, x, X_train, y_train, feature_names, target_label, seed=seed)
                status, error, attr = result.status, result.error, result.attribution
            else:
                try:
                    if method_key in DETERMINISTIC_METHODS:
                        cfg = DETERMINISTIC_METHODS[method_key]
                        if cfg.mode == "hyb":
                            cfg = FPDEConfig(mode="hyb", lambda_hyb=lambda_hyb)
                        attr, target_label, _ = explain_fpde(model, x, prototypes, labels, config=cfg, anchor=baseline)
                    elif method_key in BAYESIAN_METHODS:
                        mode, default_lambda = BAYESIAN_METHODS[method_key]
                        cfg = BayesianFPDEConfig(
                            mode=mode,
                            lambda_hyb=lambda_hyb if mode == "hyb" else default_lambda,
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
                    else:
                        status = "skipped"
                        error = f"unknown or optional method not enabled: {method}"
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
            elapsed = float(time.perf_counter() - start)
            row_base = {
                **metadata,
                "method": method,
                "status": status,
                "error": error,
                "explained_index": int(idx),
                "explained_order": int(order),
                "true_label": y_true,
                "pred_label": pred_label,
                "target_label": int(target_label),
                "n_features": int(X_train.shape[1]),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "runtime_seconds": elapsed,
                **perf,
            }
            runtime_rows.append({k: row_base[k] for k in row_base if k not in {"error"}} | {"error": error})
            if status == "ok" and attr is not None:
                metric = deletion_insertion_metrics(model, x, attr, baseline, target_label=target_label)
                metric["number_of_model_calls"] = int(2 * len((0.0, 0.1, 0.2, 0.5, 1.0)) + 1)
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
                metric_rows.append({**row_base, "p0": np.nan, "deletion_auc": np.nan, "deletion_drop_auc": np.nan, "insertion_auc": np.nan, "comprehensiveness": np.nan, "sufficiency": np.nan, "faithfulness_correlation": np.nan, "number_of_model_calls": 0, "top_k_jaccard": np.nan})
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


def write_standard_outputs(results_dir: str | Path, figures_dir: str | Path, local: pd.DataFrame, metrics: pd.DataFrame, runtime: pd.DataFrame) -> None:
    ensure_dirs(results_dir, figures_dir)
    write_parquet_or_csv(local, Path(results_dir) / "openml_local_explanations.parquet")
    summary = metrics[metrics["status"] == "ok"].groupby(["dataset_name", "method"], as_index=False).mean(numeric_only=True) if not metrics.empty else pd.DataFrame()
    write_csv(summary, Path(results_dir) / "openml_global_summary.csv")
    write_csv(metrics, Path(results_dir) / "openml_metrics.csv")
    write_csv(runtime, Path(results_dir) / "openml_runtime.csv")
