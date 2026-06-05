from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fpde import FPDEConfig, fpde_attribution, top_two_classes
from .prototypes import NIGPrior, fit_nig_posteriors, sample_posterior_prototypes


@dataclass(frozen=True)
class BayesianFPDEConfig:
    mode: str = "hyb"
    lambda_hyb: float = 0.5
    n_posterior_samples: int = 200
    tau: float = 0.0
    top_k: int = 5
    prior: NIGPrior = NIGPrior()
    normalize: bool = False


@dataclass(frozen=True)
class BayesianFPDEResult:
    samples: np.ndarray
    summary: pd.DataFrame
    positive_label: int
    negative_label: int


def summarize_samples(
    samples: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    tau: float = 0.0,
    top_k: int = 5,
) -> pd.DataFrame:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_features)")
    n_samples, n_features = samples.shape
    names = list(feature_names or [f"x{j}" for j in range(n_features)])
    if len(names) != n_features:
        raise ValueError("feature_names length mismatch")

    order = np.argsort(-np.abs(samples), axis=1)
    ranks = np.empty_like(order, dtype=float)
    for i in range(n_samples):
        ranks[i, order[i]] = np.arange(1, n_features + 1)

    rows = []
    for j, name in enumerate(names):
        vals = samples[:, j]
        rows.append(
            {
                "feature": str(name),
                "feature_index": int(j),
                "posterior_mean": float(np.mean(vals)),
                "posterior_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "ci_lower_95": float(np.quantile(vals, 0.025)),
                "ci_upper_95": float(np.quantile(vals, 0.975)),
                "p_positive": float(np.mean(vals > 0)),
                "p_negative": float(np.mean(vals < 0)),
                "p_abs_gt_tau": float(np.mean(np.abs(vals) > tau)),
                "rank_mean": float(np.mean(ranks[:, j])),
                "rank_std": float(np.std(ranks[:, j], ddof=1)) if vals.size > 1 else 0.0,
                "rank_probability_top_k": float(np.mean(ranks[:, j] <= min(top_k, n_features))),
            }
        )
    return pd.DataFrame(rows)


def explain_bayesian_fpde(
    model: Any,
    x: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: BayesianFPDEConfig | None = None,
    anchor: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> BayesianFPDEResult:
    config = config or BayesianFPDEConfig()
    posteriors = fit_nig_posteriors(X_train, y_train, config.prior)
    prototype_samples, labels = sample_posterior_prototypes(
        posteriors,
        config.n_posterior_samples,
        seed=seed,
    )
    c_plus, c_minus, _ = top_two_classes(model, x, labels)
    attr_samples = []
    for proto in prototype_samples:
        attr_samples.append(
            fpde_attribution(
                x,
                proto,
                labels,
                positive_label=c_plus,
                negative_label=c_minus,
                mode=config.mode,
                lambda_hyb=config.lambda_hyb,
                anchor=anchor,
                normalize=config.normalize,
            )
        )
    samples = np.asarray(attr_samples, dtype=float)
    summary = summarize_samples(
        samples,
        feature_names=feature_names,
        tau=config.tau,
        top_k=config.top_k,
    )
    return BayesianFPDEResult(samples=samples, summary=summary, positive_label=c_plus, negative_label=c_minus)


def posterior_summary_for_samples(
    samples: np.ndarray,
    metadata: Dict[str, Any],
    *,
    feature_names: Optional[Sequence[str]] = None,
    tau: float = 0.0,
    top_k: int = 5,
) -> pd.DataFrame:
    df = summarize_samples(samples, feature_names=feature_names, tau=tau, top_k=top_k)
    for key, value in metadata.items():
        df[key] = value
    return df
