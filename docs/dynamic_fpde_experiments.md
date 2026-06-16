# Raw-Waveform Dynamic-FPDE Audio Experiments

This artifact treats **Raw-Waveform Dynamic-FPDE** as the primary confirmed
ESC-50 workflow. The input is only a raw waveform and its label. The legacy
Native-Time/frame-level feature runner is retained for comparison experiments,
but it is not the main Raw-Waveform surface.

## Dataset

The first supported dataset is ESC-50. The runner expects:

```text
data/ESC-50/
  audio/
  meta/esc50.csv
```

It reads `meta/esc50.csv`, uses `fold` for ESC-50 splits, and uses `category`
as the class label. The artifact does not download or redistribute ESC-50.

## Raw Processing Contract

The Raw-Waveform runner uses `fpde-xai/fpde@dynamic` as the source of truth for
Raw-Diff, Raw-Cos, Raw-Hyb, masks, overlap-add, and lambda-wise saving.

Allowed preprocessing is deliberately narrow:

- decode audio
- reject empty, NaN, or inf waveforms
- convert stereo to mono
- convert sample rate to `target_sr`
- keep each waveform variable-length
- split into sliding raw windows
- zero-pad only waveforms shorter than one segment, with a valid mask

The runner does not use:

- acoustic feature extraction
- spectrograms
- MFCCs
- peak, RMS, or loudness waveform normalization
- fixed-duration alignment across clips

Default raw settings are:

```text
target_sr = 16000
segment_sec = 0.5
hop_sec = 0.1
lambda_grid = 0.0, 0.1, ..., 1.0
device = cuda
```

For the default sample rate this gives `segment_length = 8000` and
`hop_length = 1600`. `device = cuda` uses CUDA 13 through `cupy-cuda13x`; the
runner fails clearly if the CUDA 13/CuPy backend is unavailable. Use
`--device cpu` only for portable smoke tests or debugging.

## Raw Prototypes And Evidence

Training waveforms are converted to raw sliding windows. For each label, the
FPDE Raw API builds a raw segment bank and selects a label-medoid raw segment
prototype:

```text
B_c = {w_1, w_2, ..., w_M}
p_target in R^L
p_rival in R^L
```

For each test waveform, the target label is the sample label. If no rival label
is explicitly supplied, the Raw API selects a non-target rival prototype. Each
window is compared with the target and rival prototypes.

Raw-Diff:

```text
phi_diff[j] =
    (w[j] - p_rival[j]) ** 2
  - (w[j] - p_target[j]) ** 2
```

Raw-Cos:

```text
phi_cos[j] =
    w[j] * p_target[j] / (||w|| ||p_target|| + eps)
  - w[j] * p_rival[j]  / (||w|| ||p_rival||  + eps)
```

Raw-Hyb:

```text
phi_hyb(lambda) =
    lambda * scale(phi_diff)
  + (1 - lambda) * scale(phi_cos)
```

Positive evidence supports the target label. Negative evidence supports the
rival label. Padding positions are masked out of distance, evidence,
aggregation, and exported segments.

## Run

Smoke run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --device cuda
```

Full 5-fold run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_full \
  --mode full \
  --folds 1,2,3,4,5 \
  --seed 0 \
  --device cuda
```

Optional label-conditioned RAW generation is connected with:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --device cuda \
  --raw-generator my_package.my_module:generator
```

The hook signature is:

```python
generator(label, lambda_hyb, segment, sample_rate, role, metadata) -> waveform
```

This hook is called only after Raw-Hyb has selected the important positive or
negative segment. If omitted, generated RAW artifacts are skipped and recorded
as `skipped`.

## Outputs

The Raw-Waveform runner writes:

- `raw_waveform_config.json`
- `results/raw_waveform_sample_metrics.csv`
- `results/raw_waveform_summary_by_lambda.csv`
- `results/raw_waveform_errors.csv`, only when errors are skipped
- `samples/<sample_id>/summary.csv`
- `samples/<sample_id>/raw_hyb_lambda_X/window_evidence.csv`
- `samples/<sample_id>/raw_hyb_lambda_X/top_positive_segment.wav`
- `samples/<sample_id>/raw_hyb_lambda_X/top_negative_segment.wav`
- optional generated target/rival WAV files
- `samples/<sample_id>/raw_hyb_lambda_X/waveform_phi_hyb.png`
- `samples/<sample_id>/raw_hyb_lambda_X/comparison_positive.png`
- `samples/<sample_id>/raw_hyb_lambda_X/comparison_negative.png`
- `samples/<sample_id>/raw_hyb_lambda_X/metrics.json`

Sample metrics include `sample_id`, `fold`, `target_label`, `rival_label`,
`lambda_hyb`, `evidence`, `n_windows`, `input_length`, `sample_rate`,
`phi_shape`, `shape_match`, generation status, top positive segment metadata,
top negative segment metadata, runtime columns, `device`, `segment_length`, and
`hop_length`.

Every lambda result must satisfy:

```text
raw waveform shape == phi_hyb(lambda) shape
```

## Legacy Native-Time Comparison Runner

The previous frame-level runner remains available:

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

That runner operates on frame-level acoustic feature matrices and writes the
older `dynamic_fpde_*` CSV and LaTeX table files. Its generated tables should
be interpreted as Native-Time feature-space comparison outputs, not as the
primary Raw-Waveform evidence surface.

## Interpretation Limits

Raw-Waveform Dynamic-FPDE is prototype evidence decomposition over raw samples.
It is not a causal explanation, does not claim black-box model faithfulness,
does not perform DTW alignment, and does not claim sampling-rate invariance.
Audio is converted to `target_sr` before raw sliding-window evidence is
computed.
