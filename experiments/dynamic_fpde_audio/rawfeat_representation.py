"""Raw-waveform-frame plus acoustic-feature representation utilities."""

from __future__ import annotations

from dataclasses import asdict
from math import gcd
from pathlib import Path
from typing import Any, Literal, overload

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


@overload
def frame_waveform(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    return_starts: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def frame_waveform(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    return_starts: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def frame_waveform(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    return_starts: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        starts = [0]
        frames = np.pad(values, (0, frame_length - values.size))[None, :]
    else:
        starts = list(range(0, values.size - frame_length + 1, hop_length))
        final_start = values.size - frame_length
        if starts[-1] != final_start:
            starts.append(final_start)
        frames = np.stack([values[start : start + frame_length] for start in starts], axis=0)
    output = (frames.astype(np.float64, copy=False), np.ones(frames.shape[0], dtype=bool))
    if return_starts:
        return (*output, np.asarray(starts, dtype=np.int64))
    return output


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
    frame_starts: np.ndarray | None = None,
) -> np.ndarray:
    """Average overlapping generated frames back into a finite waveform."""
    values = np.asarray(frames, dtype=np.float64)
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != frame_length:
        raise ValueError(f"frames must have shape (T, {frame_length})")
    if frame_length <= 0 or hop_length <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("frames must be finite and frame/hop lengths must be positive")
    if frame_starts is None:
        starts = np.arange(values.shape[0], dtype=np.int64) * hop_length
    else:
        raw_starts = np.asarray(frame_starts)
        if raw_starts.ndim != 1 or raw_starts.shape[0] != values.shape[0]:
            raise ValueError(f"frame_starts must have shape ({values.shape[0]},)")
        if not np.all(np.isfinite(raw_starts)) or not np.all(raw_starts == np.floor(raw_starts)):
            raise ValueError("frame_starts must contain finite integer sample offsets")
        starts = raw_starts.astype(np.int64)
        if np.any(starts < 0) or np.any(np.diff(starts) < 0):
            raise ValueError("frame_starts must be non-negative and non-decreasing")
    natural_length = int(starts[-1]) + frame_length
    length = natural_length if output_length is None else int(output_length)
    if length <= 0:
        raise ValueError("output_length must be positive")
    waveform = np.zeros(max(length, natural_length), dtype=np.float64)
    weights = np.zeros_like(waveform)
    for start, frame in zip(starts.tolist(), values, strict=True):
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
    raw_frames, mask, frame_starts = frame_waveform(
        y,
        feature_config.frame_length,
        feature_config.hop_length,
        return_starts=True,
    )
    features, feature_names = extract_frame_features(audio_path, **asdict(feature_config))
    if raw_frames.shape[0] != features.shape[0]:
        raise RuntimeError(
            f"raw/features time mismatch: {raw_frames.shape[0]} vs {features.shape[0]}"
        )
    timestamps = frame_starts.astype(np.float64) / float(sample_rate)
    dt = make_dt(timestamps)
    metadata: dict[str, Any] = {
        "audio_path": str(Path(audio_path)),
        "sample_rate": sample_rate,
        "waveform_length": int(y.size),
        "original_waveform_length": int(y.size),
        "frame_starts": frame_starts.tolist(),
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
