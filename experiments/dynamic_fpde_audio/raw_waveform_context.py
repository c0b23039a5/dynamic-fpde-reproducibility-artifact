"""Fast Raw-Waveform Dynamic-FPDE context construction for ESC-50 runs."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RawContextBuildResult:
    context: Any
    timings: dict[str, float]
    cache_hit: bool = False


def _validate_sample_rate(name: str, sample_rate: int) -> int:
    value = int(sample_rate)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _as_raw_waveform(name: str, waveform: np.ndarray | Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(waveform, dtype=float)
    if arr.ndim == 2:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.shape[1] <= 8:
            arr = np.mean(arr, axis=1)
        elif arr.shape[0] <= 8:
            arr = np.mean(arr, axis=0)
        else:
            raise ValueError(f"{name} stereo waveform must have a recognizable channel axis")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D raw waveform or 2D stereo waveform, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr.astype(float, copy=True)


def _resampler_version() -> str:
    try:
        import scipy

        return f"scipy.signal.resample_poly:{scipy.__version__}"
    except Exception:
        return "scipy.signal.resample_poly:unknown"


def _resample_raw_waveform(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    source = _validate_sample_rate("source_sr", source_sr)
    target = _validate_sample_rate("target_sr", target_sr)
    if source == target:
        return waveform.astype(float, copy=True)
    from scipy.signal import resample_poly

    g = gcd(source, target)
    return resample_poly(waveform, up=target // g, down=source // g).astype(float, copy=False)


def resample_raw_waveform_polyphase(
    waveform: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    *,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Return mono raw waveform resampled with ``scipy.signal.resample_poly``."""

    return _resample_raw_waveform(_as_raw_waveform("waveform", waveform), source_sr, target_sr)


def _validate_time_params(target_sr: int, segment_sec: float, hop_sec: float) -> tuple[int, int, float, float]:
    sr = _validate_sample_rate("target_sr", target_sr)
    seg = float(segment_sec)
    hop = float(hop_sec)
    if not np.isfinite(seg) or seg <= 0.0:
        raise ValueError("segment_sec must be positive")
    if not np.isfinite(hop) or hop <= 0.0:
        raise ValueError("hop_sec must be positive")
    segment_length = int(round(seg * sr))
    hop_length = int(round(hop * sr))
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    return segment_length, hop_length, seg, hop


def _raw_windows(waveform: np.ndarray, segment_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(waveform.shape[0])
    if n_samples < segment_length:
        window = np.zeros(segment_length, dtype=float)
        mask = np.zeros(segment_length, dtype=bool)
        window[:n_samples] = waveform
        mask[:n_samples] = True
        return window[None, :], mask[None, :], np.array([0], dtype=np.intp), np.array([n_samples], dtype=np.intp)

    starts = list(range(0, n_samples - segment_length + 1, hop_length))
    final_start = n_samples - segment_length
    if starts[-1] != final_start:
        starts.append(final_start)
    starts_arr = np.asarray(starts, dtype=np.intp)
    windows = np.stack([waveform[start : start + segment_length] for start in starts_arr], axis=0)
    masks = np.ones_like(windows, dtype=bool)
    lengths = np.full(starts_arr.shape, segment_length, dtype=np.intp)
    return windows.astype(float, copy=True), masks, starts_arr, lengths


def _resolve_xp(context_device: str) -> tuple[Any, str]:
    if context_device == "cpu":
        return np, "cpu"
    if context_device == "cuda":
        import cupy as cp  # type: ignore[import-not-found]

        return cp, "cuda"
    if context_device == "auto":
        try:
            import cupy as cp  # type: ignore[import-not-found]

            cp.cuda.runtime.getDeviceCount()
            return cp, "cuda"
        except Exception:
            return np, "cpu"
    raise ValueError("context_device must be cpu, cuda, or auto")


def _labels_unique(labels: np.ndarray) -> np.ndarray:
    out: list[Any] = []
    for label in labels.tolist():
        if label not in out:
            out.append(label)
    return np.asarray(out, dtype=object)


def _asnumpy(value: Any, xp: Any) -> np.ndarray:
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def _blockwise_full_mask_totals(windows: np.ndarray, *, block_size: int, context_device: str) -> tuple[np.ndarray, str]:
    xp, device_name = _resolve_xp(context_device)
    n_items, length = windows.shape
    totals = np.zeros(n_items, dtype=float)
    sq = np.sum(windows * windows, axis=1)
    ref_cache: dict[int, Any] = {}
    sq_cache: dict[int, Any] = {}
    for row_start in range(0, n_items, block_size):
        row_end = min(row_start + block_size, n_items)
        rows = xp.asarray(windows[row_start:row_end], dtype=float)
        row_sq = xp.asarray(sq[row_start:row_end], dtype=float)
        block_total = xp.zeros(row_end - row_start, dtype=float)
        for col_start in range(0, n_items, block_size):
            col_end = min(col_start + block_size, n_items)
            if col_start not in ref_cache:
                ref_cache[col_start] = xp.asarray(windows[col_start:col_end], dtype=float)
                sq_cache[col_start] = xp.asarray(sq[col_start:col_end], dtype=float)
            refs = ref_cache[col_start]
            ref_sq = sq_cache[col_start]
            dist = (row_sq[:, None] + ref_sq[None, :] - 2.0 * (rows @ refs.T)) / float(length)
            block_total += xp.sum(xp.maximum(dist, 0.0), axis=1)
        totals[row_start:row_end] = _asnumpy(block_total, xp).astype(float, copy=False)
    return totals, device_name


def _blockwise_full_mask_candidate_totals(
    candidate_windows: np.ndarray,
    windows: np.ndarray,
    *,
    block_size: int,
    context_device: str,
) -> tuple[np.ndarray, str]:
    xp, device_name = _resolve_xp(context_device)
    n_candidates, length = candidate_windows.shape
    totals = np.zeros(n_candidates, dtype=float)
    ref_sq_np = np.sum(windows * windows, axis=1)
    candidate_sq_np = np.sum(candidate_windows * candidate_windows, axis=1)
    ref_cache: dict[int, Any] = {}
    ref_sq_cache: dict[int, Any] = {}
    for row_start in range(0, n_candidates, block_size):
        row_end = min(row_start + block_size, n_candidates)
        rows = xp.asarray(candidate_windows[row_start:row_end], dtype=float)
        row_sq = xp.asarray(candidate_sq_np[row_start:row_end], dtype=float)
        block_total = xp.zeros(row_end - row_start, dtype=float)
        for col_start in range(0, windows.shape[0], block_size):
            col_end = min(col_start + block_size, windows.shape[0])
            if col_start not in ref_cache:
                ref_cache[col_start] = xp.asarray(windows[col_start:col_end], dtype=float)
                ref_sq_cache[col_start] = xp.asarray(ref_sq_np[col_start:col_end], dtype=float)
            refs = ref_cache[col_start]
            ref_sq = ref_sq_cache[col_start]
            dist = (row_sq[:, None] + ref_sq[None, :] - 2.0 * (rows @ refs.T)) / float(length)
            block_total += xp.sum(xp.maximum(dist, 0.0), axis=1)
        totals[row_start:row_end] = _asnumpy(block_total, xp).astype(float, copy=False)
    return totals, device_name


def _blockwise_masked_mean_totals(windows: np.ndarray, masks: np.ndarray, *, block_size: int, context_device: str) -> tuple[np.ndarray, str]:
    xp, device_name = _resolve_xp(context_device)
    n_items = windows.shape[0]
    totals = np.zeros(n_items, dtype=float)
    for row_start in range(0, n_items, block_size):
        row_end = min(row_start + block_size, n_items)
        rows = xp.asarray(windows[row_start:row_end], dtype=float)
        row_masks = xp.asarray(masks[row_start:row_end], dtype=bool)
        block_total = xp.zeros(row_end - row_start, dtype=float)
        for col_start in range(0, n_items, block_size):
            col_end = min(col_start + block_size, n_items)
            refs = xp.asarray(windows[col_start:col_end], dtype=float)
            ref_masks = xp.asarray(masks[col_start:col_end], dtype=bool)
            valid = row_masks[:, None, :] & ref_masks[None, :, :]
            counts = xp.sum(valid, axis=2)
            diff = xp.where(valid, rows[:, None, :] - refs[None, :, :], 0.0)
            dist = xp.sum(diff * diff, axis=2) / xp.maximum(counts, 1)
            dist = xp.where(counts > 0, dist, xp.inf)
            block_total += xp.sum(dist, axis=1)
        totals[row_start:row_end] = _asnumpy(block_total, xp).astype(float, copy=False)
    return totals, device_name


def _candidate_indices(
    n_items: int,
    *,
    max_candidates_per_label: int | None,
    prototype_selection: str,
    seed: int,
    label: Any,
) -> np.ndarray:
    if prototype_selection == "exact_medoid":
        return np.arange(n_items, dtype=np.intp)
    if max_candidates_per_label is None or int(max_candidates_per_label) <= 0 or int(max_candidates_per_label) >= n_items:
        return np.arange(n_items, dtype=np.intp)
    limit = int(max_candidates_per_label)
    payload = f"{label}|{int(seed)}".encode("utf-8")
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little")
    rng = np.random.default_rng(digest)
    return np.sort(rng.choice(n_items, size=limit, replace=False).astype(np.intp))


def _choose_medoid(
    windows: np.ndarray,
    masks: np.ndarray,
    *,
    prototype_selection: str,
    medoid_block_size: int,
    max_candidates_per_label: int | None,
    context_device: str,
    seed: int,
    label: Any,
) -> tuple[int, dict[str, Any]]:
    if windows.shape[0] == 1:
        return 0, {"n_candidates": 1, "medoid_distance_total": 0.0, "context_device_resolved": "cpu"}
    if prototype_selection not in {"exact_medoid", "sampled_medoid"}:
        raise ValueError("prototype_selection must be exact_medoid or sampled_medoid")
    block_size = int(medoid_block_size)
    if block_size <= 0:
        raise ValueError("medoid_block_size must be positive")
    candidates = _candidate_indices(
        windows.shape[0],
        max_candidates_per_label=max_candidates_per_label,
        prototype_selection=prototype_selection,
        seed=seed,
        label=label,
    )
    candidate_windows = windows[candidates]
    candidate_masks = masks[candidates]
    if np.all(candidate_masks) and np.all(masks):
        totals, resolved_device = _blockwise_full_mask_candidate_totals(
            candidate_windows,
            windows,
            block_size=block_size,
            context_device=context_device,
        )
    else:
        totals = np.zeros(candidates.shape[0], dtype=float)
        _, resolved_device = _resolve_xp(context_device)
        xp, _ = _resolve_xp(context_device)
        for row_start in range(0, candidates.shape[0], block_size):
            row_end = min(row_start + block_size, candidates.shape[0])
            rows = xp.asarray(candidate_windows[row_start:row_end], dtype=float)
            row_masks = xp.asarray(candidate_masks[row_start:row_end], dtype=bool)
            block_total = xp.zeros(row_end - row_start, dtype=float)
            for col_start in range(0, windows.shape[0], block_size):
                col_end = min(col_start + block_size, windows.shape[0])
                refs = xp.asarray(windows[col_start:col_end], dtype=float)
                ref_masks = xp.asarray(masks[col_start:col_end], dtype=bool)
                valid = row_masks[:, None, :] & ref_masks[None, :, :]
                counts = xp.sum(valid, axis=2)
                diff = xp.where(valid, rows[:, None, :] - refs[None, :, :], 0.0)
                dist = xp.sum(diff * diff, axis=2) / xp.maximum(counts, 1)
                dist = xp.where(counts > 0, dist, xp.inf)
                block_total += xp.sum(dist, axis=1)
            totals[row_start:row_end] = _asnumpy(block_total, xp).astype(float, copy=False)
    local_idx = int(np.argmin(totals))
    medoid_idx = int(candidates[local_idx])
    return medoid_idx, {
        "n_candidates": int(candidates.shape[0]),
        "n_label_windows": int(windows.shape[0]),
        "candidate_fraction": float(candidates.shape[0] / windows.shape[0]),
        "medoid_distance_total": float(totals[local_idx]),
        "distance_metric": "masked_mean_squared_distance",
        "context_device_resolved": resolved_device,
    }


def save_raw_context_cache(path: Path, context: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in context.prototype_labels.tolist()]
    arrays: dict[str, Any] = {
        "labels": np.asarray(labels, dtype=str),
        "prototype_indices": np.asarray([context.prototype_indices[label] for label in context.prototype_labels.tolist()], dtype=np.intp),
    }
    for i, label in enumerate(context.prototype_labels.tolist()):
        arrays[f"prototype_{i}"] = np.asarray(context.prototypes[label], dtype=float)
        arrays[f"prototype_mask_{i}"] = np.asarray(context.prototype_masks[label], dtype=bool)
    metadata = {
        "target_sr": int(context.target_sr),
        "segment_length": int(context.segment_length),
        "hop_length": int(context.hop_length),
        "segment_sec": float(context.segment_sec),
        "hop_sec": float(context.hop_sec),
        "details": context.details,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata, sort_keys=True), **arrays)
        handle.flush()
    tmp.replace(path)


def load_raw_context_cache(path: Path, fpde: Any) -> Any:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item()))
    labels = data["labels"].astype(str).tolist()
    prototype_labels = np.asarray(labels, dtype=object)
    prototypes = {label: data[f"prototype_{i}"].astype(float, copy=True) for i, label in enumerate(labels)}
    prototype_masks = {label: data[f"prototype_mask_{i}"].astype(bool, copy=True) for i, label in enumerate(labels)}
    prototype_indices = {label: int(value) for label, value in zip(labels, data["prototype_indices"].tolist(), strict=True)}
    return fpde.RawWaveformFPDEContext(
        segment_banks={},
        segment_masks={},
        prototype_labels=prototype_labels,
        prototypes=prototypes,
        prototype_masks=prototype_masks,
        prototype_indices=prototype_indices,
        target_sr=int(metadata["target_sr"]),
        segment_length=int(metadata["segment_length"]),
        hop_length=int(metadata["hop_length"]),
        segment_sec=float(metadata["segment_sec"]),
        hop_sec=float(metadata["hop_sec"]),
        details=dict(metadata["details"]),
    )


def prepare_fast_raw_waveform_fpde_context(
    fpde: Any,
    waveforms: Sequence[np.ndarray | Sequence[float] | Sequence[Sequence[float]]],
    labels: Sequence[Any],
    *,
    sample_rates: Sequence[int] | int,
    sample_ids: Sequence[str] | None = None,
    target_sr: int = 16000,
    segment_sec: float = 0.5,
    hop_sec: float = 0.1,
    prototype_selection: str = "exact_medoid",
    medoid_block_size: int = 128,
    max_candidates_per_label: int | None = None,
    context_device: str = "cpu",
    retain_segment_banks: bool = False,
    seed: int = 0,
) -> RawContextBuildResult:
    """Build a compact ``fpde.RawWaveformFPDEContext`` with fast medoid selection."""

    total_start = perf_counter()
    segment_length, hop_length, seg, hop = _validate_time_params(target_sr, segment_sec, hop_sec)
    waveform_items = list(waveforms)
    if not waveform_items:
        raise ValueError("waveforms must contain at least one sample")
    labels_arr = np.asarray(labels, dtype=object)
    if labels_arr.ndim != 1 or labels_arr.shape[0] != len(waveform_items):
        raise ValueError(f"number of waveforms and labels differ: {len(waveform_items)} vs {labels_arr.shape[0]}")
    if isinstance(sample_rates, (int, np.integer)):
        rate_items = [int(sample_rates)] * len(waveform_items)
    else:
        rate_items = [int(value) for value in sample_rates]
    if len(rate_items) != len(waveform_items):
        raise ValueError(f"number of waveforms and sample_rates differ: {len(waveform_items)} vs {len(rate_items)}")
    id_items = [str(i) for i in range(len(waveform_items))] if sample_ids is None else [str(value) for value in sample_ids]
    if len(id_items) != len(waveform_items):
        raise ValueError("number of waveforms and sample_ids differ")

    timings = {
        "audio_load_runtime_sec": 0.0,
        "resample_runtime_sec": 0.0,
        "windowing_runtime_sec": 0.0,
        "bank_build_runtime_sec": 0.0,
        "medoid_runtime_sec": 0.0,
    }
    prototype_labels = _labels_unique(labels_arr)
    segment_banks: dict[Any, np.ndarray] = {}
    segment_masks: dict[Any, np.ndarray] = {}
    prototypes: dict[Any, np.ndarray] = {}
    prototype_masks: dict[Any, np.ndarray] = {}
    prototype_indices: dict[Any, int] = {}
    prototype_provenance: dict[str, dict[str, Any]] = {}
    class_counts: dict[str, int] = {}
    input_lengths: list[int] = []
    resampled_lengths: list[int] = []
    source_sample_rates: list[int] = []
    medoid_details: dict[str, Any] = {}

    for label in prototype_labels.tolist():
        windows_by_label: list[np.ndarray] = []
        masks_by_label: list[np.ndarray] = []
        provenance: list[dict[str, Any]] = []
        label_indices = [i for i, value in enumerate(labels_arr.tolist()) if value == label]
        class_counts[str(label)] = len(label_indices)
        for i in label_indices:
            raw = _as_raw_waveform(f"waveforms[{i}]", waveform_items[i])
            input_lengths.append(int(raw.shape[0]))
            source_sample_rates.append(int(rate_items[i]))
            resample_start = perf_counter()
            resampled = _resample_raw_waveform(raw, rate_items[i], target_sr)
            timings["resample_runtime_sec"] += perf_counter() - resample_start
            resampled_lengths.append(int(resampled.shape[0]))
            window_start = perf_counter()
            windows, masks, starts, lengths = _raw_windows(resampled, segment_length, hop_length)
            timings["windowing_runtime_sec"] += perf_counter() - window_start
            windows_by_label.extend([windows[j].copy() for j in range(windows.shape[0])])
            masks_by_label.extend([masks[j].copy() for j in range(masks.shape[0])])
            for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
                provenance.append(
                    {
                        "source_sample_id": id_items[i],
                        "source_sr": int(rate_items[i]),
                        "target_sr": int(target_sr),
                        "start_sample": int(start),
                        "end_sample": int(start + length),
                    }
                )
        if not windows_by_label:
            raise ValueError(f"no windows found for label={label!r}")
        bank_start = perf_counter()
        windows_arr = np.stack(windows_by_label, axis=0).astype(float, copy=False)
        masks_arr = np.stack(masks_by_label, axis=0).astype(bool, copy=False)
        timings["bank_build_runtime_sec"] += perf_counter() - bank_start
        medoid_start = perf_counter()
        medoid_idx, details = _choose_medoid(
            windows_arr,
            masks_arr,
            prototype_selection=prototype_selection,
            medoid_block_size=medoid_block_size,
            max_candidates_per_label=max_candidates_per_label,
            context_device=context_device,
            seed=seed,
            label=label,
        )
        timings["medoid_runtime_sec"] += perf_counter() - medoid_start
        if retain_segment_banks:
            segment_banks[label] = windows_arr
            segment_masks[label] = masks_arr
        prototypes[label] = windows_arr[medoid_idx].copy()
        prototype_masks[label] = masks_arr[medoid_idx].copy()
        prototype_indices[label] = int(medoid_idx)
        medoid_details[str(label)] = details
        prototype_provenance[str(label)] = provenance[medoid_idx]

    timings["context_runtime_sec"] = perf_counter() - total_start
    details = {
        "time_mode": "raw_waveform",
        "uses_acoustic_features": False,
        "uses_spectrogram": False,
        "uses_mfcc": False,
        "waveform_normalization": False,
        "prototype_kind": "label_medoid_raw_segment",
        "prototype_selection": prototype_selection,
        "distance_metric": "masked_mean_squared_distance",
        "medoid_block_size": int(medoid_block_size),
        "max_candidates_per_label": "" if max_candidates_per_label is None else int(max_candidates_per_label),
        "retain_segment_banks": bool(retain_segment_banks),
        "n_train_samples": len(waveform_items),
        "input_lengths": tuple(input_lengths),
        "resampled_lengths": tuple(resampled_lengths),
        "class_counts": class_counts,
        "prototype_provenance": prototype_provenance,
        "medoid_details": medoid_details,
        "resample_method": "scipy.signal.resample_poly",
        "source_sr": tuple(source_sample_rates),
        "target_sr": int(target_sr),
        "resampler_version": _resampler_version(),
        "runtime_accounting": "exclusive_timers",
    }
    context = fpde.RawWaveformFPDEContext(
        segment_banks=segment_banks,
        segment_masks=segment_masks,
        prototype_labels=prototype_labels.copy(),
        prototypes=prototypes,
        prototype_masks=prototype_masks,
        prototype_indices=prototype_indices,
        target_sr=int(target_sr),
        segment_length=int(segment_length),
        hop_length=int(hop_length),
        segment_sec=float(seg),
        hop_sec=float(hop),
        details=details,
    )
    return RawContextBuildResult(context=context, timings=timings, cache_hit=False)
