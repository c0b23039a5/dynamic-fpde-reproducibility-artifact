"""Frame-level audio feature extraction and standardization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Deterministic frame-feature extraction settings."""

    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 13


@dataclass(frozen=True)
class FeatureStandardizer:
    """Training-set feature standardization statistics."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.feature_names),
        }


def _librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Audio feature extraction requires librosa. Install with "
            '`python -m pip install -e ".[dynamic-audio]"`.'
        ) from exc
    return librosa


def extract_frame_features(
    audio_path: str | Path,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> tuple[np.ndarray, list[str]]:
    """Extract a deterministic frame-level acoustic feature matrix.

    Dynamic-FPDE receives the returned ``(T, F)`` feature matrix. It does not
    explain individual raw waveform samples. Non-finite feature values are
    converted with ``np.nan_to_num`` and recorded in run metadata.
    """

    librosa = _librosa()
    path = Path(audio_path)
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        y = np.zeros(2, dtype=float)
    elif y.size == 1:
        y = np.pad(y.astype(float, copy=False), (0, 1), mode="edge")
    else:
        y = y.astype(float, copy=False)

    frame_length = int(min(max(2, y.size), max(2, n_fft)))
    fft_length = frame_length
    kwargs = {"y": y, "hop_length": hop_length}
    rms = librosa.feature.rms(frame_length=frame_length, **kwargs)
    zcr = librosa.feature.zero_crossing_rate(frame_length=frame_length, **kwargs)
    centroid = librosa.feature.spectral_centroid(sr=sr, n_fft=fft_length, **kwargs)
    bandwidth = librosa.feature.spectral_bandwidth(sr=sr, n_fft=fft_length, **kwargs)
    rolloff = librosa.feature.spectral_rolloff(sr=sr, n_fft=fft_length, **kwargs)
    flatness = librosa.feature.spectral_flatness(n_fft=fft_length, **kwargs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=fft_length, hop_length=hop_length, n_mfcc=n_mfcc)

    blocks = [rms, zcr, centroid, bandwidth, rolloff, flatness, mfcc]
    min_frames = min(block.shape[1] for block in blocks)
    X = np.vstack([block[:, :min_frames] for block in blocks]).T
    X = np.nan_to_num(X.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = [
        "rms",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flatness",
        *[f"mfcc_{i}" for i in range(1, n_mfcc + 1)],
    ]
    return X, feature_names


def feature_cache_path(cache_dir: str | Path, dataset: str, sample_id: str, config: FeatureConfig) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in sample_id)
    cfg = f"sr{config.sr}_fft{config.n_fft}_hop{config.hop_length}_mfcc{config.n_mfcc}"
    return Path(cache_dir) / f"{dataset}_{safe_id}_{cfg}.npz"


def load_or_extract_features(
    audio_path: str | Path,
    *,
    dataset: str,
    sample_id: str,
    cache_dir: str | Path,
    config: FeatureConfig,
) -> tuple[np.ndarray, list[str], bool]:
    cache_path = feature_cache_path(cache_dir, dataset, sample_id, config)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return data["X"].astype(float, copy=False), data["feature_names"].astype(str).tolist(), True
    X, feature_names = extract_frame_features(audio_path, **asdict(config))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, feature_names=np.asarray(feature_names, dtype=str))
    return X, feature_names, False


def fit_standardizer(
    train_matrices: Sequence[np.ndarray],
    feature_names: Sequence[str],
    *,
    eps: float = 1e-12,
) -> FeatureStandardizer:
    matrices = [np.asarray(X, dtype=float) for X in train_matrices]
    if not matrices:
        raise ValueError("train_matrices must contain at least one matrix")
    n_features = matrices[0].shape[1]
    for i, X in enumerate(matrices):
        if X.ndim != 2:
            raise ValueError(f"train_matrices[{i}] must be 2D")
        if X.shape[1] != n_features:
            raise ValueError("all train matrices must have the same feature dimension")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"train_matrices[{i}] contains NaN or inf")
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match feature dimension")
    flat = np.vstack(matrices)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std <= eps, 1.0, std)
    return FeatureStandardizer(mean=mean.astype(float), std=std.astype(float), feature_names=tuple(feature_names))


def transform_features(matrix: np.ndarray, standardizer: FeatureStandardizer) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2D")
    if X.shape[1] != standardizer.mean.shape[0]:
        raise ValueError("matrix feature dimension does not match standardizer")
    out = (X - standardizer.mean) / standardizer.std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

