"""Raw-waveform-frame plus acoustic-feature representation utilities."""

from __future__ import annotations

from dataclasses import asdict
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .features import FeatureConfig, extract_frame_features


def load_mono_resampled_audio(audio_path: str | Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode finite audio, mix it to mono, and resample it to ``target_sr``."""
    import soundfile as sf
    from scipy.signal import resample_poly

    if int(target_sr) <= 0:
        raise ValueError("target_sr must be positive")
    data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float64")
    if data.shape[0] == 0:
        raise ValueError(f"audio is empty: {audio_path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"audio contains NaN or inf: {audio_path}")
    mono = np.mean(data, axis=1, dtype=np.float64)
    if int(sample_rate) != int(target_sr):
        common = gcd(int(sample_rate), int(target_sr))
        mono = resample_poly(mono, int(target_sr) // common, int(sample_rate) // common)
    mono = np.asarray(mono, dtype=np.float64)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError(f"resampled audio is empty or non-finite: {audio_path}")
    return mono, int(target_sr)


def frame_waveform(y: np.ndarray, frame_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame one waveform without imposing a fixed temporal length."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("y must be a one-dimensional waveform")
    if values.size == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("y contains NaN or inf")
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if values.size < frame_length:
        frames = np.pad(values, (0, frame_length - values.size))[None, :]
    else:
        starts = list(range(0, values.size - frame_length + 1, hop_length))
        final_start = values.size - frame_length
        if starts[-1] != final_start:
            starts.append(final_start)
        frames = np.stack([values[start : start + frame_length] for start in starts], axis=0)
    return frames.astype(np.float64, copy=False), np.ones(frames.shape[0], dtype=bool)


def frame_times_sec(T: int, hop_length: int, target_sr: int) -> np.ndarray:
    if int(T) <= 0 or int(hop_length) <= 0 or int(target_sr) <= 0:
        raise ValueError("T, hop_length, and target_sr must be positive")
    return np.arange(int(T), dtype=np.float64) * (float(hop_length) / float(target_sr))


def make_dt(timestamps_sec: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps_sec, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps_sec must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps_sec contains NaN or inf")
    dt = np.diff(timestamps, prepend=timestamps[0])
    if np.any(dt < 0.0) or not np.all(np.isfinite(dt)):
        raise ValueError("timestamps_sec must be finite and non-decreasing")
    return dt.astype(np.float64, copy=False)


def overlap_add_frames(
    frames: np.ndarray,
    frame_length: int,
    hop_length: int,
    output_length: int | None = None,
) -> np.ndarray:
    """Average overlapping generated frames back into a finite waveform."""
    values = np.asarray(frames, dtype=np.float64)
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != frame_length:
        raise ValueError(f"frames must have shape (T, {frame_length})")
    if frame_length <= 0 or hop_length <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("frames must be finite and frame/hop lengths must be positive")
    natural_length = (values.shape[0] - 1) * hop_length + frame_length
    length = natural_length if output_length is None else int(output_length)
    if length <= 0:
        raise ValueError("output_length must be positive")
    waveform = np.zeros(max(length, natural_length), dtype=np.float64)
    weights = np.zeros_like(waveform)
    for index, frame in enumerate(values):
        start = index * hop_length
        stop = start + frame_length
        waveform[start:stop] += frame
        weights[start:stop] += 1.0
    np.divide(waveform, weights, out=waveform, where=weights > 0.0)
    waveform = waveform[:length] if length <= waveform.size else np.pad(waveform, (0, length - waveform.size))
    return np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def build_rawfeat_input(
    audio_path: str | Path,
    feature_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build aligned raw frames, acoustic features, time deltas, and mask."""
    y, sample_rate = load_mono_resampled_audio(audio_path, feature_config.target_sr)
    if feature_config.normalize_audio:
        peak = float(np.max(np.abs(y)))
        if peak > 0.0:
            y = y / max(peak, 1e-8)
    raw_frames, mask = frame_waveform(y, feature_config.frame_length, feature_config.hop_length)
    features, feature_names = extract_frame_features(audio_path, **asdict(feature_config))
    if raw_frames.shape[0] != features.shape[0]:
        raise RuntimeError(
            f"raw/features time mismatch: {raw_frames.shape[0]} vs {features.shape[0]}"
        )
    timestamps = frame_times_sec(raw_frames.shape[0], feature_config.hop_length, sample_rate)
    dt = make_dt(timestamps)
    metadata: dict[str, Any] = {
        "audio_path": str(Path(audio_path)),
        "sample_rate": sample_rate,
        "waveform_length": int(y.size),
        "duration_sec": float(y.size / sample_rate),
        "frame_length": int(feature_config.frame_length),
        "hop_length": int(feature_config.hop_length),
        "feature_names": feature_names,
        "timestamps_sec": timestamps.tolist(),
    }
    return raw_frames, features.astype(np.float64, copy=False), dt, mask, metadata


__all__ = [
    "build_rawfeat_input",
    "frame_times_sec",
    "frame_waveform",
    "load_mono_resampled_audio",
    "make_dt",
    "overlap_add_frames",
]
