# Dynamic-FPDE Reproducibility Artifact

This repository contains reproducible **Raw-Waveform Dynamic-FPDE**
experiments for time-resolved prototype-directional audio explanations.

Raw-Waveform Dynamic-FPDE is the primary confirmed ESC-50 workflow in this
artifact. It uses only raw waveform samples and labels. It does not extract
acoustic features, spectrograms, or MFCCs, and it does not apply peak, RMS, or
loudness waveform normalization. Audio is decoded, stereo is converted to mono
inside the FPDE Raw API, and sample rate is converted to `target_sr`; clip
durations remain variable.

The previous Native-Time/frame-level acoustic feature runner is preserved as a
legacy/comparison path. The older resampled-time, `prototype_length`-based
Dynamic-FPDE path remains legacy and benchmark-oriented only.

## Installation

The artifact installs FPDE from the `dynamic` branch of `fpde-xai/fpde`:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dynamic-audio,plot]"
python -m pytest
```

For NVIDIA CUDA acceleration, install the CUDA extra with the CuPy wheel that
matches CUDA 13.x:

```bash
python -m pip install -e ".[dev,dynamic-audio,plot,cuda]"
```

On this Windows/OneDrive checkout, `uv` may be more reliable:

```bash
uv --system-certs run --link-mode=copy --extra dev --extra dynamic-audio --extra plot python -m pytest
```

## ESC-50 Raw-Waveform Experiments

The runner expects ESC-50 to be available locally and does not redistribute or
download the raw dataset.

Smoke run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0
```

Full 5-fold run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_full \
  --mode full \
  --folds 1,2,3,4,5 \
  --seed 0
```

Raw defaults are `--target-sr 16000`, `--segment-sec 0.5`, `--hop-sec 0.1`,
and the full `--lambda-grid` of `0.0, 0.1, ..., 1.0`. Use `--device cpu`,
`--device cuda`, or `--device auto` for the package Raw-Waveform computation.
There is intentionally no waveform-normalization option.

Short waveforms are zero-padded only when they are shorter than one raw segment,
and the valid mask excludes padded samples from distance, evidence, overlap-add,
and exported segments. Longer waveforms remain variable-length and get an
end-aligned final window when needed.

Optional label-conditioned RAW generation is supplied with:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --raw-generator my_package.my_module:generator
```

The generator signature is
`generator(label, lambda_hyb, segment, sample_rate, role, metadata)`. It is
called only after Raw-Hyb has selected the top positive or negative segment. If
no generator is provided, generation is recorded as `skipped`.

## Main Outputs

The Raw-Waveform ESC-50 runner writes:

- `raw_waveform_config.json`
- `results/raw_waveform_sample_metrics.csv`
- `results/raw_waveform_summary_by_lambda.csv`
- `samples/<sample_id>/summary.csv`
- `samples/<sample_id>/raw_hyb_lambda_X/window_evidence.csv`
- `samples/<sample_id>/raw_hyb_lambda_X/top_positive_segment.wav`
- `samples/<sample_id>/raw_hyb_lambda_X/top_negative_segment.wav`
- `samples/<sample_id>/raw_hyb_lambda_X/generated_target_lambda_X.wav`, when a generator is provided
- `samples/<sample_id>/raw_hyb_lambda_X/generated_rival_lambda_X.wav`, when a generator is provided
- `samples/<sample_id>/raw_hyb_lambda_X/waveform_phi_hyb.png`
- `samples/<sample_id>/raw_hyb_lambda_X/comparison_positive.png`
- `samples/<sample_id>/raw_hyb_lambda_X/comparison_negative.png`
- `samples/<sample_id>/raw_hyb_lambda_X/metrics.json`

Feature caches under `outputs/**/cache/features/` are ignored by default
because they are derived from the legacy Native-Time feature runner, not the
primary Raw-Waveform runner.

## Interpretation Limits

Raw-Waveform Dynamic-FPDE explains raw-sample prototype evidence after
converting audio to a common `target_sr`. It is not a causal explanation, does
not claim black-box model faithfulness, does not perform DTW alignment, and
does not claim sampling-rate invariance.

Raw-Diff, Raw-Cos, and Raw-Hyb are computed against raw segment prototypes from
label-specific segment banks. Positive evidence supports the target label;
negative evidence supports the rival label. Total evidence can depend on clip
length and window coverage, so compare across clips carefully.

The legacy Native-Time runner still writes the older
`dynamic_fpde_sample_metrics.csv` and LaTeX tables for frame-level feature
comparison experiments. Those outputs are no longer the primary Raw-Waveform
surface.

See `docs/dynamic_fpde_experiments.md` for the Raw-Waveform processing
contract, output schema, generator hook, and legacy Native-Time comparison
runner.

## Citation

If you use this artifact, cite the repository and the associated Dynamic-FPDE
paper or manuscript when available. Citation metadata is provided in
`CITATION.cff`.

## License

Unless otherwise stated, this repository is distributed under the Apache
License 2.0. ESC-50 is distributed separately under Creative Commons
Attribution-NonCommercial terms; follow the dataset license when using raw
audio or derived feature caches.
