from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FPDEConfig:
    mode: str = "hyb"
    lambda_hyb: float = 0.5
    contrast: str = "runner_up"
    normalize: bool = False


def class_prototypes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    labels = np.asarray(sorted(np.unique(y)), dtype=int)
    if labels.size < 2:
        raise ValueError("at least two classes are required")
    prototypes = np.vstack([X[y == label].mean(axis=0) for label in labels])
    return prototypes, labels


def _label_index(labels: Sequence[int], label: int) -> int:
    idx = np.where(np.asarray(labels) == int(label))[0]
    if idx.size == 0:
        raise ValueError(f"label not found in prototypes: {label}")
    return int(idx[0])


def top_two_classes(model: Any, x: np.ndarray, labels: Optional[Sequence[int]] = None) -> Tuple[int, int, np.ndarray]:
    proba = np.asarray(model.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0], dtype=float)
    model_labels = np.asarray(getattr(model, "classes_", np.arange(proba.shape[0])), dtype=int)
    order = np.argsort(proba)[::-1]
    c_plus = int(model_labels[order[0]])
    c_minus = int(model_labels[order[1]]) if order.size > 1 else c_plus
    if labels is not None:
        available = set(int(v) for v in labels)
        if c_plus not in available:
            c_plus = int(next(iter(available)))
        if c_minus not in available or c_minus == c_plus:
            c_minus = int(next(v for v in available if v != c_plus))
    return c_plus, c_minus, proba


def fpde_attribution(
    x: np.ndarray,
    prototypes: np.ndarray,
    labels: Sequence[int],
    *,
    positive_label: int,
    negative_label: int,
    mode: str = "hyb",
    lambda_hyb: float = 0.5,
    anchor: Optional[np.ndarray] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Compute clean-room prototype-contrast feature attributions."""
    x = np.asarray(x, dtype=float)
    prototypes = np.asarray(prototypes, dtype=float)
    p_plus = prototypes[_label_index(labels, positive_label)]
    p_minus = prototypes[_label_index(labels, negative_label)]
    anchor_arr = np.zeros_like(x) if anchor is None else np.asarray(anchor, dtype=float)
    direction = p_plus - p_minus
    diff = (x - anchor_arr) * direction
    denom = float(np.linalg.norm(direction) * max(np.linalg.norm(x - anchor_arr), 1e-12))
    cos = diff / denom if denom > 0 else np.zeros_like(diff)
    mode = mode.lower()
    if mode in {"diff", "diff_fpde"}:
        attr = diff
    elif mode in {"cos", "cos_fpde"}:
        attr = cos
    elif mode in {"hyb", "hyb_fpde", "hyb_fpde_grid"}:
        attr = float(lambda_hyb) * diff + (1.0 - float(lambda_hyb)) * cos
    else:
        raise ValueError(f"unknown FPDE mode: {mode}")
    if normalize:
        scale = np.sum(np.abs(attr))
        if scale > 0:
            attr = attr / scale
    return np.asarray(attr, dtype=float)


def explain_fpde(
    model: Any,
    x: np.ndarray,
    prototypes: np.ndarray,
    labels: Sequence[int],
    *,
    config: FPDEConfig | None = None,
    anchor: Optional[np.ndarray] = None,
    true_label: Optional[int] = None,
) -> Tuple[np.ndarray, int, int]:
    config = config or FPDEConfig()
    c_plus, c_minus, _ = top_two_classes(model, x, labels)
    if config.contrast == "true_class" and true_label is not None:
        c_minus = int(true_label)
        if c_minus == c_plus:
            c_minus = int(next(label for label in labels if int(label) != c_plus))
    attr = fpde_attribution(
        x,
        prototypes,
        labels,
        positive_label=c_plus,
        negative_label=c_minus,
        mode=config.mode,
        lambda_hyb=config.lambda_hyb,
        anchor=anchor,
        normalize=config.normalize,
    )
    return attr, c_plus, c_minus


def true_fpde_attribution(
    x: np.ndarray,
    true_prototypes: np.ndarray,
    labels: Sequence[int],
    *,
    positive_label: int,
    negative_label: int,
    mode: str = "hyb",
    lambda_hyb: float = 0.5,
    anchor: Optional[np.ndarray] = None,
) -> np.ndarray:
    return fpde_attribution(
        x,
        true_prototypes,
        labels,
        positive_label=positive_label,
        negative_label=negative_label,
        mode=mode,
        lambda_hyb=lambda_hyb,
        anchor=anchor,
    )
