# Dynamic-FPDE Reproducibility Artifact

This repository's primary ESC-50 workflow is **RawFeat Dynamic-FPDE**. It
decomposes prototype-contrast evidence over a concatenated, variable-length
temporal representation containing:

- framed raw waveform: `raw_frames.shape == (T, frame_length)`
- frame-level acoustic descriptors: `features.shape == (T, F)`
- time deltas and validity mask: `dt.shape == mask.shape == (T,)`

Raw framing and feature extraction use the same sample rate, frame length, and
hop length. Clips are not globally resampled to a fixed `T`. The existing
Raw-Waveform-only runner is retained as a legacy/comparison workflow, and the
Native-Time feature-only runner is retained as another comparison workflow.

## Installation

The artifact pins the Phase 5/6 RawFeat API from `fpde-xai/fpde`:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dynamic-audio,plot]"
python -m pytest
```

RawFeat itself is NumPy/SciPy/SoundFile based and adds no deep-learning
dependency. Existing comparison experiments retain their original PyTorch and
CuPy extras. CUDA 13 comparison runs can be installed with:

```bash
python -m pip install -e ".[dev,dynamic-audio,plot,cuda]"
```

## RawFeat ESC-50 runs

The runner expects `audio/` and `meta/esc50.csv` below the dataset root. It
does not download or redistribute ESC-50.

Smoke run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_rawfeat_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/rawfeat_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --lambda-hyb 0.5 \
  --normalize l1
```

Generation/audit run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_rawfeat_dynamic_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/rawfeat_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --generation-scope selected \
  --summary-scaling standard \
  --noise-scale 0.0
```

`--normalize l1` affects exported time-importance figures only; the audit CSV
always stores the engine's unmodified evidence and attribution sum. Generated
RAW is an inspection/audit artifact, not an explanation input. The runner
overlap-adds generated raw frames to a waveform, saves `generated_target.wav`,
then re-runs that waveform through the acoustic feature extractor before its
generated-sample audit.

## RawFeat outputs

- `rawfeat_config.json`
- `results/rawfeat_sample_metrics.csv`
- `results/rawfeat_sample_summary.csv`
- `results/rawfeat_generation_metrics.csv`
- `results/rawfeat_generation_summary.csv`
- `results/rawfeat_errors.csv`, when `--skip-errors` records failures
- `samples/<sample_id>/summary.csv`
- `samples/<sample_id>/rawfeat_hyb_lambda_<lambda>/metrics.json`
- `samples/<sample_id>/rawfeat_hyb_lambda_<lambda>/generated_target.wav`, when enabled

The sample schema records evidence, attribution sum, exactness residual, group
attributions, raw/feature/attribution/time-importance shapes, and a shape-match
flag. Aggregates report mean absolute evidence and mean exactness residual by
method/lambda. Generation aggregates report the same audit quantities after
waveform reconstruction and feature reprocessing.

## Interpretation limits

RawFeat Dynamic-FPDE explains prototype-contrast evidence in the concatenated
raw-frame, acoustic-feature, and time-delta representation. It does not claim a
causal explanation, ground-truth explanation, black-box classifier
faithfulness, or raw-waveform attribution beyond this prototype evidence
decomposition. The core audit is `evidence ~= attributions.sum()`.

`PrototypeRawGenerator` retains training-derived prototypes, feature summaries,
and residual information. Treat generator state and generated audio as
potentially sensitive; do not publish them without checking privacy, licensing,
membership-inference, and data-leakage risks.

## Comparison workflows

- Raw-Waveform-only legacy/comparison:
  `experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py`
- Native-Time feature-only comparison:
  `experiments/dynamic_fpde_audio/run_esc50_dynamic_fpde.py`

See [RawFeat experiment contract](docs/rawfeat_dynamic_fpde_experiments.md) for
the primary workflow and [Raw-Waveform comparison contract](docs/dynamic_fpde_experiments.md)
for the preserved comparison runner.

## Citation and license

Citation metadata is provided in `CITATION.cff`. Unless otherwise stated, this
repository is distributed under Apache License 2.0. ESC-50 is distributed
separately under Creative Commons Attribution-NonCommercial terms.
