"""Frame-level acoustic feature extraction for Native-Time Dynamic-FPDE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Audio-to-frame-feature settings.

    Audio is decoded, converted to mono, resampled to ``target_sr``, and then
    converted to a variable-length matrix of frame-level acoustic features.
    The frame axis is never cropped, padded, or resampled to force a common
    length across clips.
    """

    target_sr: int = 16000
    frame_length: int = 1024
    hop_length: int = 512
    normalize_audio: bool = True

    def frame_time_sec(self, frame_index: int) -> float:
        return float(int(frame_index) * int(self.hop_length) / int(self.target_sr))


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


def _load_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Dynamic-FPDE audio feature extraction requires soundfile. Install with "
            '`python -m pip install -e ".[dynamic-audio]"`.'
        ) from exc

    data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    mono = np.mean(data, axis=1, dtype=np.float64).astype(np.float32, copy=False)
    return mono, int(sample_rate)


def _resample(y: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return y.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    gcd = int(np.gcd(original_sr, target_sr))
    up = target_sr // gcd
    down = original_sr // gcd
    return resample_poly(y, up, down).astype(np.float32, copy=False)


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if y.size == 0:
        y = np.zeros(1, dtype=np.float32)
    if y.size < frame_length:
        padded = np.pad(y, (0, frame_length - y.size))
        return padded.reshape(1, frame_length)

    n_frames = 1 + (y.size - frame_length) // hop_length
    if n_frames <= 0:
        n_frames = 1
    shape = (n_frames, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides).copy()


def extract_frame_features(
    audio_path: str | Path,
    *,
    target_sr: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 512,
    normalize_audio: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return a variable-length ``(T, F)`` acoustic feature matrix."""

    y, original_sr = _load_audio(audio_path)
    y = _resample(y, original_sr, int(target_sr))
    if normalize_audio:
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if np.isfinite(peak) and peak > 0.0:
            y = y / max(peak, 1e-8)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)

    frames = _frame_signal(y, int(frame_length), int(hop_length))
    window = np.hanning(int(frame_length)).astype(float)
    windowed = frames * window
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    power = spectrum * spectrum
    freqs = np.fft.rfftfreq(int(frame_length), d=1.0 / int(target_sr))
    power_sum = np.sum(power, axis=1) + 1e-12

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    log_energy = np.log(power_sum / float(frame_length))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    centroid = np.sum(power * freqs, axis=1) / power_sum
    bandwidth = np.sqrt(np.sum(power * (freqs - centroid[:, None]) ** 2, axis=1) / power_sum)
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * power_sum[:, None], axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (np.mean(power, axis=1) + 1e-12)

    X = np.column_stack([rms, log_energy, zcr, centroid, bandwidth, rolloff, flatness])
    scale = np.asarray([1.0, 1.0, 1.0, max(target_sr, 1), max(target_sr, 1), max(target_sr, 1), 1.0], dtype=float)
    X = X / scale
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(float, copy=False)
    return X, [
        "rms",
        "log_energy",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff85",
        "spectral_flatness",
    ]


def feature_cache_path(cache_dir: str | Path, dataset: str, sample_id: str, config: FeatureConfig) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in sample_id)
    cfg = f"sr{config.target_sr}_fl{config.frame_length}_hop{config.hop_length}_acoustic"
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
        if X.shape[0] == 0:
            raise ValueError(f"train_matrices[{i}] must contain at least one frame")
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
