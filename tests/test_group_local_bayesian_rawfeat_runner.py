from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_group_local_bayesian_rawfeat_dynamic_fpde import main


def _write_rawfeat(path: Path, *, seed: int, time_length: int) -> None:
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        raw=rng.normal(size=(time_length, 2)),
        features=rng.normal(size=(time_length, 3)),
        dt=np.linspace(0.01, 0.02, time_length),
        mask=np.ones(time_length, dtype=bool),
        frame_starts=np.arange(time_length) * 128,
    )


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
        assert data["posterior_mean"].shape == (3, 6)
    with np.load(dislike_attribution) as data:
        assert data["posterior_mean"].shape == (5, 6)

    config = json.loads((output_dir / "logs" / "run_config.json").read_text(encoding="utf-8"))
    assert config["global_temporal_resampling"] is False
    assert config["uses_original_song_data"] is False
