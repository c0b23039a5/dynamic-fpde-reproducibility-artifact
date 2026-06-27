from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.dynamic_fpde_audio.features import FeatureConfig
from experiments.dynamic_fpde_audio.rawfeat_representation import (
    build_rawfeat_input,
    frame_waveform,
    overlap_add_frames,
)


def test_frame_waveform_preserves_variable_length_and_adds_end_aligned_frame():
    frames, mask = frame_waveform(np.arange(11, dtype=float), frame_length=4, hop_length=3)

    assert frames.shape == (4, 4)
    np.testing.assert_array_equal(frames[-1], np.arange(7, 11, dtype=float))
    assert mask.shape == (4,)
    assert mask.all()


def test_short_waveform_produces_one_padded_frame():
    frames, mask = frame_waveform(np.array([1.0, 2.0]), frame_length=5, hop_length=2)

    assert frames.shape == (1, 5)
    np.testing.assert_array_equal(frames[0], [1.0, 2.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(mask, [True])


def test_build_rawfeat_input_has_matching_time_axes(tmp_path: Path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "stereo.wav"
    t = np.arange(173, dtype=float) / 8000.0
    stereo = np.column_stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)])
    sf.write(path, stereo, 8000)

    raw, features, dt, mask, metadata = build_rawfeat_input(
        path,
        FeatureConfig(target_sr=4000, frame_length=32, hop_length=13),
    )

    assert raw.shape[0] == features.shape[0] == dt.shape[0] == mask.shape[0]
    assert raw.shape[1] == 32
    assert features.shape[1] == 7
    assert dt[0] == 0.0
    assert np.all(np.isfinite(raw))
    assert np.all(np.isfinite(features))
    assert np.all(np.isfinite(dt))
    assert metadata["sample_rate"] == 4000


def test_overlap_add_frames_returns_finite_waveform():
    frames = np.arange(24, dtype=float).reshape(4, 6)
    waveform = overlap_add_frames(frames, frame_length=6, hop_length=3)

    assert waveform.shape == (15,)
    assert np.all(np.isfinite(waveform))


def test_rawfeat_public_api_and_generated_raw_audit():
    from fpde.dynamic import (
        DynamicFPDEEngine,
        DynamicFPDEResult,
        PrototypeRawGenerator,
        pad_sequences,
        select_lambda_dynamic,
        split_representation,
        validate_sequence_inputs,
    )

    assert DynamicFPDEResult is not None
    assert callable(pad_sequences)
    assert callable(validate_sequence_inputs)
    assert callable(select_lambda_dynamic)
    assert callable(split_representation)
    raw = [
        np.zeros((3, 4)),
        np.full((3, 4), 0.1),
        np.ones((3, 4)),
        np.full((3, 4), 0.9),
    ]
    features = [np.column_stack([item.mean(axis=1), item.std(axis=1)]) for item in raw]
    dt = [np.array([[0.0], [0.1], [0.1]]) for _ in raw]
    labels = ["a", "a", "b", "b"]
    engine = DynamicFPDEEngine(lambda_hyb=0.5).fit(raw=raw, features=features, dt=dt, y=labels)
    generator = PrototypeRawGenerator().fit(raw=raw, features=features, y=labels)
    generated = generator.generate_with_metadata(
        label="a", length=3, condition_features=features[0], noise_scale=0.0
    )["raw"]
    waveform = overlap_add_frames(generated, frame_length=4, hop_length=2)
    regenerated, mask = frame_waveform(waveform, frame_length=4, hop_length=2)
    regenerated_features = np.column_stack([regenerated.mean(axis=1), regenerated.std(axis=1)])
    result = engine.explain_one(
        raw=regenerated,
        features=regenerated_features,
        dt=np.array([[0.0], [0.1], [0.1]]),
        mask=mask,
        target_class="a",
        rival_class="b",
    )

    assert generated.shape == (3, 4)
    assert np.isfinite(result.audit["abs_error"])
    assert result.audit["abs_error"] < 1e-9
    assert result.evidence == pytest.approx(np.sum(result.attributions), abs=1e-9)
