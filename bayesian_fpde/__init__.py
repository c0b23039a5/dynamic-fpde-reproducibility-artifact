"""Clean-room Bayesian-FPDE reproducibility utilities."""

from .bayesian_fpde import BayesianFPDEConfig, BayesianFPDEResult, explain_bayesian_fpde
from .fpde import FPDEConfig, class_prototypes, explain_fpde, true_fpde_attribution
from .prototypes import NIGPrior, fit_nig_posteriors, sample_posterior_prototypes

__all__ = [
    "BayesianFPDEConfig",
    "BayesianFPDEResult",
    "FPDEConfig",
    "NIGPrior",
    "class_prototypes",
    "explain_bayesian_fpde",
    "explain_fpde",
    "fit_nig_posteriors",
    "sample_posterior_prototypes",
    "true_fpde_attribution",
]
