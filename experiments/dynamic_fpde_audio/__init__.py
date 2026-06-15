"""Audio experiment helpers for Dynamic-FPDE reproducibility runs."""

from .aggregate import aggregate_additivity, aggregate_by_method
from .baselines import energy_frame_scores, random_frame_scores
from .datasets import ESCSample, ModeConfig, get_mode_config, read_esc50_metadata, split_esc50
from .features import FeatureConfig, fit_standardizer, transform_features

__all__ = [
    "ESCSample",
    "FeatureConfig",
    "ModeConfig",
    "aggregate_additivity",
    "aggregate_by_method",
    "energy_frame_scores",
    "fit_standardizer",
    "get_mode_config",
    "random_frame_scores",
    "read_esc50_metadata",
    "split_esc50",
    "transform_features",
]

