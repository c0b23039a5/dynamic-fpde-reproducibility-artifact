from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class NIGPrior:
    m0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 2.0


@dataclass(frozen=True)
class NIGPosterior:
    label: int
    n: int
    m_n: np.ndarray
    kappa_n: float
    alpha_n: float
    beta_n: np.ndarray


def fit_nig_posteriors(X: np.ndarray, y: np.ndarray, prior: NIGPrior | None = None) -> Dict[int, NIGPosterior]:
    """Fit independent Normal-Inverse-Gamma posteriors per class and feature."""
    prior = prior or NIGPrior()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")
    if prior.kappa0 <= 0 or prior.alpha0 <= 0 or prior.beta0 <= 0:
        raise ValueError("NIG prior parameters must be positive except m0")

    posteriors: Dict[int, NIGPosterior] = {}
    for label in np.unique(y):
        Xc = X[y == label]
        n = int(Xc.shape[0])
        if n == 0:
            continue
        mean = Xc.mean(axis=0)
        centered_ss = np.sum((Xc - mean) ** 2, axis=0)
        kappa_n = prior.kappa0 + n
        alpha_n = prior.alpha0 + n / 2.0
        m_n = (prior.kappa0 * prior.m0 + n * mean) / kappa_n
        beta_n = (
            prior.beta0
            + 0.5 * centered_ss
            + (prior.kappa0 * n * (mean - prior.m0) ** 2) / (2.0 * kappa_n)
        )
        posteriors[int(label)] = NIGPosterior(
            label=int(label),
            n=n,
            m_n=np.asarray(m_n, dtype=float),
            kappa_n=float(kappa_n),
            alpha_n=float(alpha_n),
            beta_n=np.asarray(beta_n, dtype=float),
        )
    if len(posteriors) < 2:
        raise ValueError("at least two classes are required")
    return posteriors


def posterior_mean_prototypes(posteriors: Dict[int, NIGPosterior]) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sorted(posteriors), dtype=int)
    prototypes = np.vstack([posteriors[int(label)].m_n for label in labels])
    return prototypes, labels


def sample_posterior_prototypes(
    posteriors: Dict[int, NIGPosterior],
    n_samples: int,
    *,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw prototype means from p(mu_cj | D).

    Returns an array with shape (n_samples, n_classes, n_features) and the class labels.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    gen = np.random.default_rng(seed)
    labels = np.asarray(sorted(posteriors), dtype=int)
    draws = []
    for _ in range(n_samples):
        per_class = []
        for label in labels:
            post = posteriors[int(label)]
            gamma_draw = gen.gamma(shape=post.alpha_n, scale=1.0 / post.beta_n)
            sigma2 = 1.0 / gamma_draw
            mu = gen.normal(loc=post.m_n, scale=np.sqrt(sigma2 / post.kappa_n))
            per_class.append(mu)
        draws.append(np.vstack(per_class))
    return np.asarray(draws, dtype=float), labels


def class_counts(y: Iterable[int]) -> Dict[int, int]:
    labels, counts = np.unique(np.asarray(list(y)), return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}
