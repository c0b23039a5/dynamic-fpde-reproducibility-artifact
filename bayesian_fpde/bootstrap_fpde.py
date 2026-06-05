from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .fpde import FPDEConfig, class_prototypes, explain_fpde


def bootstrap_fpde_samples(
    model: Any,
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_bootstrap: int = 100,
    config: Optional[FPDEConfig] = None,
    anchor: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    out = []
    n = X_train.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb = X_train[idx]
        yb = y_train[idx]
        if np.unique(yb).size < 2:
            continue
        prototypes, labels = class_prototypes(Xb, yb)
        attr, _, _ = explain_fpde(model, x, prototypes, labels, config=config, anchor=anchor)
        out.append(attr)
    if not out:
        raise ValueError("no valid bootstrap samples contained at least two classes")
    return np.asarray(out, dtype=float)
