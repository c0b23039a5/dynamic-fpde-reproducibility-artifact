"""Audio experiment helpers for Dynamic-FPDE reproducibility runs."""

from .aggregate import aggregate_additivity, aggregate_by_method
from .baselines import energy_frame_scores, random_frame_scores
from .datasets import ESCSample, ModeConfig, get_mode_config, read_esc50_metadata, split_esc50
from .features import FeatureConfig, fit_standardizer, transform_features
from .native_metrics import native_temporal_deletion_insertion_curves
from .native_prototypes import select_native_exemplar_prototype, validate_feature_matrix

__all__ = [
    "ESCSample",
    "FeatureConfig",
    "ModeConfig",
    "aggregate_additivity",
    "aggregate_by_method",
    "energy_frame_scores",
    "fit_standardizer",
    "get_mode_config",
    "native_temporal_deletion_insertion_curves",
    "random_frame_scores",
    "read_esc50_metadata",
    "select_native_exemplar_prototype",
    "split_esc50",
    "transform_features",
    "validate_feature_matrix",
]
