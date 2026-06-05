from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


class MethodUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineResult:
    attribution: Optional[np.ndarray]
    status: str
    error: str = ""


def explain_shap(model: Any, x: np.ndarray, target_label: int) -> np.ndarray:
    try:
        import shap
    except Exception as exc:
        raise MethodUnavailable("SHAP is not installed") from exc
    explainer = shap.Explainer(model.predict_proba, np.asarray(x).reshape(1, -1))
    values = explainer(np.asarray(x).reshape(1, -1))
    arr = np.asarray(values.values)
    classes = np.asarray(getattr(model, "classes_", np.arange(arr.shape[-1])), dtype=int)
    idx = int(np.where(classes == int(target_label))[0][0]) if arr.ndim == 3 else 0
    if arr.ndim == 3:
        return np.asarray(arr[0, :, idx], dtype=float)
    return np.asarray(arr[0], dtype=float)


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
    raise MethodUnavailable("AIME adapter is not enabled in the Bayesian pipeline v1")


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
) -> BaselineResult:
    try:
        if method == "shap":
            attr = explain_shap(model, x, target_label)
        elif method == "lime":
            attr = explain_lime(model, x, X_train, y_train, feature_names, target_label, seed=seed)
        elif method == "aime":
            attr = explain_aime(model, x, X_train, y_train, feature_names, target_label)
        else:
            raise ValueError(f"unknown baseline method: {method}")
        if attr.shape != x.shape:
            raise ValueError(f"attribution shape {attr.shape} does not match {x.shape}")
        return BaselineResult(np.asarray(attr, dtype=float), "ok", "")
    except MethodUnavailable as exc:
        return BaselineResult(None, "skipped", str(exc))
    except Exception as exc:
        return BaselineResult(None, "error", f"{type(exc).__name__}: {exc}")
