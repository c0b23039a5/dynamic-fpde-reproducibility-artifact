from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.run_group_local_bayesian_rawfeat_dynamic_fpde import main


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _write_rawfeat(
    path: Path, *, seed: int, time_length: int, include_raw: bool = True
) -> None:
    rng = np.random.default_rng(seed)
    arrays = {
        "features": rng.normal(size=(time_length, 3)).astype(np.float32),
        "dt": np.linspace(0.01, 0.02, time_length, dtype=np.float32),
        "mask": np.ones(time_length, dtype=bool),
        "frame_starts": (np.arange(time_length) * 128).astype(np.int64),
    }
    if include_raw:
        arrays["raw"] = rng.normal(size=(time_length, 2)).astype(np.float32)
    np.savez(path, **arrays)


def _write_input_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _two_sample_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "like",
            "cover_group_id": "g",
            "label": "Like",
            "rawfeat_npz_path": "like.npz",
            "rel_path": "covers/like",
            "eligible_group_local": True,
        },
        {
            "sample_id": "dislike",
            "cover_group_id": "g",
            "label": "Dislike",
            "rawfeat_npz_path": "dislike.npz",
            "rel_path": "covers/dislike",
            "eligible_group_local": True,
        },
    ]


def test_group_local_bayesian_rawfeat_runner_smoke(tmp_path: Path) -> None:
    lengths = {"like-a": 3, "dislike-a": 5, "like-only": 4}
    for seed, (sample_id, time_length) in enumerate(lengths.items()):
        _write_rawfeat(tmp_path / f"{sample_id}.npz", seed=seed, time_length=time_length)

    input_csv = tmp_path / "group_local_runner_input.csv"
    pd.DataFrame(
        [
            {"sample_id": "like-a", "cover_group_id": "eligible", "label": "Like", "rawfeat_npz_path": "like-a.npz", "rel_path": "covers/like-a", "eligible_group_local": True},
            {"sample_id": "dislike-a", "cover_group_id": "eligible", "label": "Dislike", "rawfeat_npz_path": "dislike-a.npz", "rel_path": "covers/dislike-a", "eligible_group_local": True},
            {"sample_id": "like-only", "cover_group_id": "ineligible", "label": "Like", "rawfeat_npz_path": "like-only.npz", "rel_path": "covers/like-only", "eligible_group_local": True},
        ]
    ).to_csv(input_csv, index=False)

    output_dir = tmp_path / "output"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5", "--save-attributions",
    ]) == 0

    per_sample_path = output_dir / "results" / "per_sample_hyb.csv"
    skipped_path = output_dir / "results" / "skipped_groups.csv"
    assert per_sample_path.exists()
    assert skipped_path.exists()

    per_sample = pd.read_csv(per_sample_path)
    skipped = pd.read_csv(skipped_path)
    assert set(per_sample["sample_id"]) == {"like-a", "dislike-a"}
    assert set(per_sample["T"]) == {3, 5}
    assert np.isfinite(per_sample["max_abs_exactness_residual"]).all()
    assert skipped.loc[0, "cover_group_id"] == "ineligible"
    like_attribution = output_dir / "attributions" / "eligible__like-a__hyb_summary.npz"
    dislike_attribution = output_dir / "attributions" / "eligible__dislike-a__hyb_summary.npz"
    assert like_attribution.exists()
    assert dislike_attribution.exists()
    with np.load(like_attribution) as data:
        assert data["posterior_mean"].shape == (3, 4)
    with np.load(dislike_attribution) as data:
        assert data["posterior_mean"].shape == (5, 4)
    assert (per_sample["evidence_input"] == "features_dt").all()
    assert (per_sample["input_dim"] == 4).all()
    assert (per_sample["raw_dim"] == 0).all()
    assert (per_sample["feature_dim"] == 3).all()
    assert (per_sample["dt_dim"] == 1).all()
    assert (per_sample["raw_included"] == False).all()  # noqa: E712
    assert (per_sample["dt_included"] == True).all()  # noqa: E712
    assert (per_sample["uses_raw_for_evidence"] == False).all()  # noqa: E712

    config = json.loads((output_dir / "logs" / "run_config.json").read_text(encoding="utf-8"))
    assert config["evidence_input"] == "features_dt"
    assert config["uses_raw_for_evidence"] is False
    assert config["uses_raw_for_generation"] is False
    assert config["global_temporal_resampling"] is False
    assert config["uses_original_song_data"] is False
    assert config["draw_chunk_size"] == 8
    assert config["free_cuda_memory_pool"] is False
    timing = pd.read_csv(output_dir / "results" / "timing_summary.csv")
    assert list(timing.columns) == [
        "cover_group_id", "load_sec", "posterior_fit_sec", "explain_sec",
        "samples_per_sec", "draws_per_sec", "device",
    ]
    assert set(timing["cover_group_id"]) == {"eligible", "ineligible"}
    assert (timing["device"] == "cpu").all()


def test_feature_only_npz_works_with_features_input(tmp_path: Path) -> None:
    _write_rawfeat(tmp_path / "like.npz", seed=40, time_length=4, include_raw=False)
    _write_rawfeat(tmp_path / "dislike.npz", seed=41, time_length=6, include_raw=False)
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv, _two_sample_rows())

    output_dir = tmp_path / "features"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5", "--evidence-input", "features",
    ]) == 0

    result = pd.read_csv(output_dir / "results" / "per_sample_hyb.csv")
    assert dict(zip(result["sample_id"], result["T"])) == {"like": 4, "dislike": 6}
    assert (result["evidence_input"] == "features").all()
    assert (result["D"] == 3).all()
    assert (result["input_dim"] == 3).all()
    assert (result["raw_dim"] == 0).all()
    assert (result["feature_dim"] == 3).all()
    assert (result["dt_dim"] == 0).all()
    assert (result["uses_raw_for_evidence"] == False).all()  # noqa: E712
    config = json.loads((output_dir / "logs" / "run_config.json").read_text(encoding="utf-8"))
    assert config["global_temporal_resampling"] is False


def test_feature_only_npz_works_with_features_dt_default(tmp_path: Path) -> None:
    _write_rawfeat(tmp_path / "like.npz", seed=50, time_length=5, include_raw=False)
    _write_rawfeat(tmp_path / "dislike.npz", seed=51, time_length=8, include_raw=False)
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv, _two_sample_rows())

    output_dir = tmp_path / "features-dt"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5",
    ]) == 0

    result = pd.read_csv(output_dir / "results" / "per_sample_hyb.csv")
    assert dict(zip(result["sample_id"], result["T"])) == {"like": 5, "dislike": 8}
    assert (result["evidence_input"] == "features_dt").all()
    assert (result["D"] == 4).all()
    assert (result["input_dim"] == 4).all()
    assert (result["raw_dim"] == 0).all()
    assert (result["feature_dim"] == 3).all()
    assert (result["dt_dim"] == 1).all()
    assert (result["raw_included"] == False).all()  # noqa: E712
    assert (result["dt_included"] == True).all()  # noqa: E712
    assert (result["uses_raw_for_evidence"] == False).all()  # noqa: E712
    config = json.loads((output_dir / "logs" / "run_config.json").read_text(encoding="utf-8"))
    assert config["evidence_input"] == "features_dt"
    assert config["uses_raw_for_evidence"] is False
    assert config["uses_raw_for_generation"] is False
    assert config["uses_original_song_data"] is False
    assert config["global_temporal_resampling"] is False


def test_rawfeat_mode_requires_raw_and_uses_raw_feature_dt_dimension(tmp_path: Path) -> None:
    _write_rawfeat(tmp_path / "like.npz", seed=60, time_length=4, include_raw=True)
    _write_rawfeat(tmp_path / "dislike.npz", seed=61, time_length=4, include_raw=True)
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv, _two_sample_rows())

    output_dir = tmp_path / "rawfeat"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5", "--evidence-input", "rawfeat",
    ]) == 0
    result = pd.read_csv(output_dir / "results" / "per_sample_hyb.csv")
    assert (result["D"] == 6).all()
    assert (result["input_dim"] == 6).all()
    assert (result["raw_dim"] == 2).all()
    assert (result["feature_dim"] == 3).all()
    assert (result["dt_dim"] == 1).all()
    assert (result["raw_included"] == True).all()  # noqa: E712
    assert (result["uses_raw_for_evidence"] == True).all()  # noqa: E712

    _write_rawfeat(tmp_path / "like-missing.npz", seed=62, time_length=4, include_raw=False)
    _write_rawfeat(tmp_path / "dislike-missing.npz", seed=63, time_length=4, include_raw=False)
    missing_input_csv = tmp_path / "missing-input.csv"
    _write_input_csv(
        missing_input_csv,
        [
            {**_two_sample_rows()[0], "rawfeat_npz_path": "like-missing.npz"},
            {**_two_sample_rows()[1], "rawfeat_npz_path": "dislike-missing.npz"},
        ],
    )
    missing_output_dir = tmp_path / "rawfeat-missing"
    assert main([
        "--input-csv", str(missing_input_csv), "--output-dir", str(missing_output_dir),
        "--n-samples", "5", "--evidence-input", "rawfeat",
    ]) == 0
    errors = pd.read_csv(missing_output_dir / "results" / "errors.csv")
    assert set(errors["stage"]) == {"load_npz"}
    assert errors["error"].str.contains("missing NPZ arrays: raw", regex=False).all()


def test_group_local_low_memory_preserves_native_time_and_skips_coordinates(
    tmp_path: Path,
) -> None:
    _write_rawfeat(tmp_path / "like.npz", seed=10, time_length=4)
    _write_rawfeat(tmp_path / "dislike.npz", seed=11, time_length=7)
    input_csv = tmp_path / "group_local_runner_input.csv"
    pd.DataFrame(
        [
            {"sample_id": "like", "cover_group_id": "g", "label": "Like", "rawfeat_npz_path": "like.npz", "rel_path": "covers/like", "eligible_group_local": True},
            {"sample_id": "dislike", "cover_group_id": "g", "label": "Dislike", "rawfeat_npz_path": "dislike.npz", "rel_path": "covers/dislike", "eligible_group_local": True},
        ]
    ).to_csv(input_csv, index=False)

    output_dir = tmp_path / "low-memory"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5", "--low-memory", "--save-attributions",
    ]) == 0

    result_path = output_dir / "results" / "per_sample_hyb.csv"
    assert result_path.exists()
    result = pd.read_csv(result_path)
    assert dict(zip(result["sample_id"], result["T"])) == {"like": 4, "dislike": 7}
    scalar_columns = [
        "evidence_mean", "evidence_ci_low", "evidence_ci_high",
        "evidence_probability_positive", "evidence_sign_stability",
        "raw_group_mean", "raw_group_ci_low", "raw_group_ci_high",
        "feature_group_mean", "feature_group_ci_low", "feature_group_ci_high",
        "dt_group_mean", "dt_group_ci_low", "dt_group_ci_high",
        "max_abs_exactness_residual", "mean_abs_exactness_residual",
        "max_abs_group_sum_residual",
    ]
    assert np.isfinite(result[scalar_columns].to_numpy(dtype=float)).all()
    assert not list((output_dir / "attributions").glob("*.npz"))

    config = json.loads((output_dir / "logs" / "run_config.json").read_text(encoding="utf-8"))
    assert config["low_memory_streaming"] is True
    assert config["global_temporal_resampling"] is False


def test_low_memory_coordinate_summary_requires_explicit_flag(tmp_path: Path) -> None:
    _write_rawfeat(tmp_path / "like.npz", seed=20, time_length=3)
    _write_rawfeat(tmp_path / "dislike.npz", seed=21, time_length=5)
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {"sample_id": "like", "cover_group_id": "g", "label": "Like", "rawfeat_npz_path": "like.npz", "rel_path": "like", "eligible_group_local": True},
            {"sample_id": "dislike", "cover_group_id": "g", "label": "Dislike", "rawfeat_npz_path": "dislike.npz", "rel_path": "dislike", "eligible_group_local": True},
        ]
    ).to_csv(input_csv, index=False)

    output_dir = tmp_path / "coordinate-opt-in"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5", "--low-memory", "--save-coordinate-summary",
    ]) == 0
    assert (output_dir / "attributions" / "g__like__hyb_summary.npz").exists()


@pytest.mark.skipif(not _cuda_available(), reason="CuPy CUDA device is unavailable")
def test_low_memory_cuda_smoke(tmp_path: Path) -> None:
    _write_rawfeat(tmp_path / "like.npz", seed=30, time_length=4)
    _write_rawfeat(tmp_path / "dislike.npz", seed=31, time_length=6)
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {"sample_id": "like", "cover_group_id": "gpu", "label": "Like", "rawfeat_npz_path": "like.npz", "rel_path": "like", "eligible_group_local": True},
            {"sample_id": "dislike", "cover_group_id": "gpu", "label": "Dislike", "rawfeat_npz_path": "dislike.npz", "rel_path": "dislike", "eligible_group_local": True},
        ]
    ).to_csv(input_csv, index=False)

    output_dir = tmp_path / "cuda"
    assert main([
        "--input-csv", str(input_csv), "--output-dir", str(output_dir),
        "--n-samples", "5", "--device", "cuda", "--draw-chunk-size", "2",
    ]) == 0
    result = pd.read_csv(output_dir / "results" / "per_sample_hyb.csv")
    assert dict(zip(result["sample_id"], result["T"])) == {"like": 4, "dislike": 6}
    assert np.isfinite(result["evidence_mean"]).all()
    assert np.isfinite(result["max_abs_exactness_residual"]).all()
    timing = pd.read_csv(output_dir / "results" / "timing_summary.csv")
    assert np.isfinite(timing[["load_sec", "posterior_fit_sec", "explain_sec"]].to_numpy()).all()
    assert (timing["device"] == "cuda").all()
    config = json.loads((output_dir / "logs" / "run_config.json").read_text(encoding="utf-8"))
    assert config["device"] == "cuda"
    assert config["draw_chunk_size"] == 2
    assert config["free_cuda_memory_pool"] is False
    assert config["cuda_device_name"]
