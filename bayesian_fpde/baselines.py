from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Dict, Optional, Sequence

import numpy as np


class MethodUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineResult:
    attribution: Optional[np.ndarray]
    status: str
    error_message: str = ""
    dependency_available: bool = False
    n_model_calls: int = 0
    background_size: int = 0
    max_background: Optional[int] = None


_DEPENDENCY_CACHE: Dict[str, bool] = {}
_ADAPTER_SKIP_CACHE: Dict[str, str] = {}


def dependency_available(module_name: str) -> bool:
    if module_name not in _DEPENDENCY_CACHE:
        try:
            _DEPENDENCY_CACHE[module_name] = importlib.util.find_spec(module_name) is not None
        except ValueError:
            _DEPENDENCY_CACHE[module_name] = True
    return _DEPENDENCY_CACHE[module_name]


def shap_background_sample(
    X_train: np.ndarray,
    *,
    seed: int = 0,
    max_background: int = 100,
) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D")
    if X_train.shape[0] == 0:
        raise ValueError("X_train must contain at least one row")
    n_background = min(int(max_background), X_train.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_train.shape[0], size=n_background, replace=False)
    return X_train[np.sort(idx)]


def explain_shap(
    model: Any,
    x: np.ndarray,
    X_train: np.ndarray,
    target_label: int,
    *,
    seed: int = 0,
    max_background: int = 100,
) -> np.ndarray:
    try:
        import shap
    except Exception as exc:
        raise MethodUnavailable("SHAP is not installed") from exc
    background = shap_background_sample(X_train, seed=seed, max_background=max_background)
    explainer = shap.Explainer(model.predict_proba, background)
    x_2d = np.asarray(x, dtype=float).reshape(1, -1)
    values = explainer(x_2d)
    arr = np.asarray(values.values)
    classes = np.asarray(getattr(model, "classes_", np.arange(arr.shape[-1])), dtype=int)
    idx = int(np.where(classes == int(target_label))[0][0]) if arr.ndim == 3 else 0
    if arr.ndim == 3:
        out = np.asarray(arr[0, :, idx], dtype=float)
    else:
        out = np.asarray(arr[0], dtype=float)
    if out.shape != np.asarray(x).shape:
        raise ValueError(f"SHAP attribution shape {out.shape} does not match x shape {np.asarray(x).shape}")
    return out


def explain_lime(model: Any, x: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, feature_names: Sequence[str], target_label: int, *, seed: int = 0) -> np.ndarray:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise MethodUnavailable("LIME is not installed") from exc
    classes = [str(c) for c in getattr(model, "classes_", np.unique(y_train))]
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=list(feature_names),
        class_names=classes,
        discretize_continuous=False,
        random_state=seed,
        mode="classification",
    )
    exp = explainer.explain_instance(x, model.predict_proba, num_features=x.size, labels=[int(target_label)])
    attr = np.zeros(x.size, dtype=float)
    for j, value in exp.as_map().get(int(target_label), []):
        if 0 <= int(j) < attr.size:
            attr[int(j)] = float(value)
    return attr


def explain_aime(*args: Any, **kwargs: Any) -> np.ndarray:
    try:
        import aime_xai  # noqa: F401
    except Exception as exc:
        raise MethodUnavailable("AIME is not installed") from exc
    raise MethodUnavailable("AIME dependency is installed but the adapter is not implemented in this artifact")


def optional_baseline(
    method: str,
    model: Any,
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    target_label: int,
    *,
    seed: int = 0,
    max_background: int = 100,
) -> BaselineResult:
    dependency_name = {
        "shap": "shap",
        "lime": "lime",
        "aime": "aime_xai",
        "bayesshap": "shap",
        "bayeslime": "lime",
        "bayesian_aime": "aime_xai",
    }.get(method, method)
    dep_ok = dependency_available(dependency_name)
    not_implemented = {
        "bayesshap": "BayesSHAP adapter is not implemented in this artifact",
        "bayeslime": "BayesLIME adapter is not implemented in this artifact",
        "bayesian_aime": "Bayesian-AIME adapter is not implemented in this artifact",
    }
    if method in not_implemented:
        message = _ADAPTER_SKIP_CACHE.setdefault(method, not_implemented[method])
        return BaselineResult(None, "skipped", message, dep_ok, 0, 0, max_background)
    try:
        if method == "shap":
            attr = explain_shap(model, x, X_train, target_label, seed=seed, max_background=max_background)
            n_model_calls = min(int(max_background), int(np.asarray(X_train).shape[0])) + 1
            background_size = min(int(max_background), int(np.asarray(X_train).shape[0]))
        elif method == "lime":
            attr = explain_lime(model, x, X_train, y_train, feature_names, target_label, seed=seed)
            n_model_calls = 5000
            background_size = int(np.asarray(X_train).shape[0])
        elif method == "aime":
            attr = explain_aime(model, x, X_train, y_train, feature_names, target_label)
            n_model_calls = 0
            background_size = int(np.asarray(X_train).shape[0])
        else:
            raise ValueError(f"unknown baseline method: {method}")
        if attr.shape != x.shape:
            raise ValueError(f"attribution shape {attr.shape} does not match {x.shape}")
        return BaselineResult(np.asarray(attr, dtype=float), "ok", "", dep_ok, n_model_calls, background_size, max_background)
    except MethodUnavailable as exc:
        return BaselineResult(None, "skipped", str(exc), dep_ok, 0, 0, max_background)
    except Exception as exc:
        return BaselineResult(None, "error", f"{type(exc).__name__}: {exc}", dep_ok, 0, 0, max_background)
