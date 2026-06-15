# Dynamic-FPDE Reproducibility Artifact

This repository contains reproducible **Dynamic-FPDE** experiments for
time-resolved prototype-directional audio explanations.

Dynamic-FPDE operates on frame-level acoustic feature matrices and explains
prototype evidence for a target prototype over a rival prototype. This artifact
does not implement raw waveform attribution, Delta-Dynamic-FPDE,
AIME/SHAP/LIME comparisons, recommender-specific logic, or causal
explanations.

## Installation

The artifact installs FPDE from the `dynamic` branch of `fpde-xai/fpde`:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dynamic-audio,plot]"
python -m pytest
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
  --output-dir outputs/dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --prototype-length 64 \
  --backend cpu \
  --make-figures
```

Full 5-fold run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/dynamic_fpde_esc50_full \
  --mode full \
  --folds 1,2,3,4,5 \
  --seed 0 \
  --prototype-length 128 \
  --backend cpu \
  --make-figures
```

`--backend` accepts `cpu` or `cuda` and defaults to `cpu`. Feature extraction
and temporal resampling remain CPU-side for both backends. In CUDA mode, the
runner batches already-resampled Dynamic-FPDE tensors and calls
`dynamic_diff_fpde_gpu`, `dynamic_cos_fpde_gpu`, and `dynamic_hyb_fpde_gpu`;
outputs are converted back to NumPy before metrics and CSV writing. If CUDA is
requested but CuPy or a usable CUDA device is unavailable, the run fails with a
clear error instead of falling back to CPU.

Runtime excludes CPU feature extraction and CPU temporal resampling unless
otherwise stated. CUDA acceleration applies only to batched Dynamic-FPDE tensor
operations.

See `docs/dynamic_fpde_experiments.md` for dataset assumptions, output files,
metric definitions, margin diagnostics, LaTeX table generation, and
interpretation limits.

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

Feature caches under `outputs/**/cache/features/` are ignored by default
because they are derived from ESC-50 audio.

## Citation

If you use this artifact, cite the repository and the associated Dynamic-FPDE
paper or manuscript when available. Citation metadata is provided in
`CITATION.cff`.

## License

Unless otherwise stated, this repository is distributed under the Apache
License 2.0. ESC-50 is distributed separately under Creative Commons
Attribution-NonCommercial terms; follow the dataset license when using raw
audio or derived feature caches.
