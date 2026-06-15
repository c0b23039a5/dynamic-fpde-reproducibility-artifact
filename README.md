# Dynamic-FPDE Reproducibility Artifact

This repository contains reproducible **Native-Time Dynamic-FPDE** experiments
for time-resolved prototype-directional audio explanations.

Native-Time Dynamic-FPDE is the intended Dynamic-FPDE formulation in this
artifact. It operates on frame-level acoustic feature matrices, not raw
waveform samples. For each clip, the input is `X_i` with shape `(T_i, F)` and
the primary output is `Phi_i` with the same shape. Longer clips produce longer
`Phi_i` matrices. `time_importance` and `feature_importance` are auxiliary
summaries of `Phi_i`, not replacements for it.

The older resampled-time, `prototype_length`-based Dynamic-FPDE path is legacy
and benchmark-oriented only. The primary ESC-50 runner rejects
`--prototype-length` instead of silently falling back to fixed-length temporal
resampling.

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

## ESC-50 Experiments

The runner expects ESC-50 to be available locally and does not redistribute or
download the raw dataset.

Smoke run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/native_time_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --prototype-mode exemplar \
  --prototype-selection nearest_to_class_centroid_frame \
  --anchor zero \
  --normalize none \
  --lambda-hyb 0.5 \
  --backend cpu \
  --make-figures
```

Full 5-fold run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/native_time_dynamic_fpde_esc50_full \
  --mode full \
  --folds 1,2,3,4,5 \
  --seed 0 \
  --prototype-mode exemplar \
  --prototype-selection nearest_to_class_centroid_frame \
  --anchor zero \
  --normalize none \
  --lambda-hyb 0.5 \
  --backend cpu \
  --make-figures
```

Audio decode, target-sample-rate resampling, mono conversion, and acoustic
feature extraction, feature standardization, and deletion/insertion diagnostics
stay on CPU. Native-Time FPDE computation can run on CPU or, with
`--backend cuda`, as CuPy elementwise attribution tensor computation. CUDA mode
never pads or fixed-length-resamples variable-length clips; it groups only
samples that naturally share the same `(T, F)`.

Feature extraction may zero-pad a clip shorter than one analysis frame so that
the clip yields `T == 1`. This is intra-clip minimum-frame padding only; it is
not temporal alignment, not fixed-length resampling, and not dense tensor
batching. ESC-50 clips are normally longer than one frame.

## Main Outputs

The ESC-50 runner writes:

- `results/dynamic_fpde_sample_metrics.csv`
- `results/dynamic_fpde_summary_by_method.csv`
- `results/dynamic_fpde_summary_positive_margin_by_method.csv`
- `results/dynamic_fpde_lambda_selection.csv`
- `results/dynamic_fpde_additivity_summary.csv`
- `tables/table_dynamic_fpde_main_results.tex`
- `tables/table_dynamic_fpde_positive_margin_results.tex`
- `tables/table_dynamic_fpde_margin_summary.tex`
- `tables/table_dynamic_fpde_additivity.tex`
- `tables/table_dynamic_fpde_lambda.tex`
- `tables/table_dynamic_fpde_native_time_checks.tex`

Feature caches under `outputs/**/cache/features/` are ignored by default
because they are derived from ESC-50 audio.

`dynamic_fpde_sample_metrics.csv` keeps `runtime_sec` as a compatibility alias
for `total_runtime_sec`. It also reports `native_fpde_runtime_sec` and
`diagnostic_runtime_sec` separately.

## Interpretation Limits

Dynamic-FPDE explains prototype evidence in frame-level acoustic feature space.
It does not claim raw waveform attribution, causal explanation, black-box model
faithfulness, DTW alignment, or verse/chorus alignment. Dynamic-FPDE itself is
not sampling-rate invariant; raw audio is decoded, resampled to `target_sr`,
converted to mono, converted to frame-level features, and then explained.

Total evidence can depend on the number of frames, so do not directly compare
total evidence across clips with different lengths without normalization or
careful interpretation. Prototype frames are auditable through
`source_sample_id`, `source_frame_index`, and `source_time_sec`.

The runner includes ranking baselines named `energy_baseline_raw`,
`feature_norm_baseline_standardized`, and `random_baseline`. They are frame
ranking diagnostics, not FPDE attribution methods.

See `docs/dynamic_fpde_experiments.md` for dataset assumptions, output schemas,
metric definitions, CUDA details, and table generation.

## Citation

If you use this artifact, cite the repository and the associated Dynamic-FPDE
paper or manuscript when available. Citation metadata is provided in
`CITATION.cff`.

## License

Unless otherwise stated, this repository is distributed under the Apache
License 2.0. ESC-50 is distributed separately under Creative Commons
Attribution-NonCommercial terms; follow the dataset license when using raw
audio or derived feature caches.
