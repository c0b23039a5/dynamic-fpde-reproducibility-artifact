"""Native-Time Dynamic-FPDE exemplar prototype selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from experiments.dynamic_fpde_audio.features import FeatureConfig


@dataclass(frozen=True)
class NativePrototype:
    vector: np.ndarray
    metadata: dict[str, object]


def validate_feature_matrix(X: np.ndarray, F_expected: int | None = None) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"feature matrix must be 2D, got shape={arr.shape}")
    if arr.shape[0] <= 0:
        raise ValueError("feature matrix must contain at least one frame")
    if arr.shape[1] <= 0:
        raise ValueError("feature matrix must contain at least one feature")
    if F_expected is not None and arr.shape[1] != int(F_expected):
        raise ValueError(f"feature dimension mismatch: expected {int(F_expected)}, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("feature matrix contains NaN or inf")
    return arr


def _labels_by_sample(sample_ids: Sequence[str], labels: Mapping[str, str] | Sequence[str]) -> dict[str, str]:
    if isinstance(labels, Mapping):
        return {str(sample_id): str(labels[sample_id]) for sample_id in sample_ids}
    label_items = [str(label) for label in labels]
    if len(label_items) != len(sample_ids):
        raise ValueError("number of sample ids and labels differ")
    return {str(sample_id): label for sample_id, label in zip(sample_ids, label_items, strict=True)}


def _frame_time_sec(frame_index: int, feature_config: FeatureConfig | None) -> float:
    if feature_config is None:
        return float(frame_index)
    return feature_config.frame_time_sec(frame_index)


def select_native_exemplar_prototype(
    feature_matrices: Mapping[str, np.ndarray],
    labels: Mapping[str, str] | Sequence[str],
    *,
    label: str,
    selection_rule: str = "nearest_to_class_centroid_frame",
    feature_config: FeatureConfig | None = None,
    feature_names: Sequence[str] | None = None,
    prototype_mode: str = "selected_exemplar",
) -> NativePrototype:
    """Select one real frame vector for ``label`` from variable-length samples."""

    sample_ids = list(feature_matrices.keys())
    label_map = _labels_by_sample(sample_ids, labels)
    selected: list[tuple[str, int, np.ndarray]] = []
    F_expected: int | None = None
    for sample_id in sample_ids:
        X = validate_feature_matrix(feature_matrices[sample_id], F_expected)
        F_expected = X.shape[1]
        if label_map[sample_id] != str(label):
            continue
        for frame_index in range(X.shape[0]):
            selected.append((sample_id, frame_index, X[frame_index].copy()))
    if not selected:
        raise ValueError(f"no frames found for label={label!r}")

    frame_stack = np.vstack([item[2] for item in selected])
    if selection_rule == "nearest_to_class_centroid_frame":
        centroid = np.mean(frame_stack, axis=0)
        distances = np.sum((frame_stack - centroid) ** 2, axis=1)
        selected_index = int(np.argmin(distances))
    elif selection_rule == "medoid_frame":
        distances = np.sum((frame_stack[:, None, :] - frame_stack[None, :, :]) ** 2, axis=2)
        selected_index = int(np.argmin(np.sum(distances, axis=1)))
    else:
        raise ValueError("prototype-selection must be medoid_frame or nearest_to_class_centroid_frame")

    source_sample_id, source_frame_index, vector = selected[selected_index]
    metadata: dict[str, object] = {
        "source_sample_id": source_sample_id,
        "source_frame_index": int(source_frame_index),
        "source_time_sec": _frame_time_sec(source_frame_index, feature_config),
        "label": str(label),
        "prototype_mode": prototype_mode,
        "selection_rule": selection_rule,
    }
    if feature_names is not None:
        metadata["feature_names"] = list(feature_names)
    return NativePrototype(vector=vector.astype(float, copy=True), metadata=metadata)


def select_native_target_rival_prototypes(
    feature_matrices: Mapping[str, np.ndarray],
    labels: Mapping[str, str] | Sequence[str],
    *,
    target_label: str,
    sample_matrix: np.ndarray,
    selection_rule: str = "nearest_to_class_centroid_frame",
    feature_config: FeatureConfig | None = None,
    feature_names: Sequence[str] | None = None,
) -> tuple[NativePrototype, NativePrototype, str]:
    """Select target and closest non-target exemplar prototypes for one sample."""

    sample_ids = list(feature_matrices.keys())
    label_map = _labels_by_sample(sample_ids, labels)
    labels_unique = sorted(set(label_map.values()))
    if str(target_label) not in labels_unique:
        raise ValueError(f"target label has no prototype frames: {target_label!r}")
    if len(labels_unique) < 2:
        raise ValueError("at least two labels are required to select a rival prototype")

    X = validate_feature_matrix(sample_matrix)
    prototypes = {
        label: select_native_exemplar_prototype(
            feature_matrices,
            label_map,
            label=label,
            selection_rule=selection_rule,
            feature_config=feature_config,
            feature_names=feature_names,
        )
        for label in labels_unique
    }
    F = X.shape[1]
    target = prototypes[str(target_label)]
    validate_feature_matrix(target.vector.reshape(1, -1), F)

    rival_distances: dict[str, float] = {}
    for label, proto in prototypes.items():
        if label == str(target_label):
            continue
        p = np.asarray(proto.vector, dtype=float)
        if p.shape != (F,):
            raise ValueError(f"rival prototype for {label!r} has shape {p.shape}, expected {(F,)}")
        rival_distances[label] = float(np.mean(np.sum((X - p) ** 2, axis=1)))
    rival_label = min(rival_distances, key=rival_distances.get)
    return target, prototypes[rival_label], rival_label


def select_class_exemplar_prototypes(
    feature_matrices: Mapping[str, np.ndarray],
    labels: Mapping[str, str] | Sequence[str],
    *,
    selection_rule: str = "nearest_to_class_centroid_frame",
    feature_config: FeatureConfig | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, NativePrototype]:
    sample_ids = list(feature_matrices.keys())
    label_map = _labels_by_sample(sample_ids, labels)
    return {
        label: select_native_exemplar_prototype(
            feature_matrices,
            label_map,
            label=label,
            selection_rule=selection_rule,
            feature_config=feature_config,
            feature_names=feature_names,
        )
        for label in sorted(set(label_map.values()))
    }
