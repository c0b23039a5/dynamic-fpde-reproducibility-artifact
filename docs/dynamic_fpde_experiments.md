# Raw-Waveform-only Dynamic-FPDE Comparison Experiments

This document describes the preserved **Raw-Waveform-only Dynamic-FPDE**
legacy/comparison workflow. RawFeat Dynamic-FPDE is now primary and combines
raw waveform frames with aligned frame-level acoustic descriptors. The
Native-Time feature-only runner is also retained as a comparison workflow.

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

The no-alignment Raw-Waveform runner uses `fpde-xai/fpde@dynamic` as the source
of truth for Raw-Diff, Raw-Cos, Raw-Hyb, masks, overlap-add, and lambda-wise
saving. This artifact adds Shift-Robust Raw-Waveform Dynamic-FPDE for
`alignment_mode=hard_bounded` and `alignment_mode=soft_bounded`.

Allowed preprocessing is deliberately narrow:

- decode audio
- reject empty, NaN, or inf waveforms
- convert stereo to mono
- convert sample rate to `target_sr` with `scipy.signal.resample_poly`
- keep each waveform variable-length
- split into sliding raw windows
- zero-pad only waveforms shorter than one segment, with a valid mask

This Raw-Waveform-only comparison runner does not use:

- acoustic feature extraction
- spectrograms
- STFT features
- MFCCs
- RMS features
- spectral centroid
- chroma
- handcrafted acoustic feature matrices
- peak, RMS, or loudness waveform normalization
- z-score waveform normalization
- global DTW alignment
- circular shift
- fixed-duration alignment across clips

Default raw settings are:

```text
target_sr = 16000
segment_sec = 0.5
hop_sec = 0.1
lambda_grid = 0.0, 0.1, ..., 1.0
device = cuda
prototype_selection = exact_medoid
medoid_block_size = 128
alignment_mode = none
```

For the default sample rate this gives `segment_length = 8000` and
`hop_length = 1600`. `device = cuda` uses CUDA 13 through `cupy-cuda13x`; the
runner fails clearly if the CUDA 13/CuPy backend is unavailable. Use
`--device cpu` only for portable smoke tests or debugging.

## Shift-Robust Raw-Waveform Dynamic-FPDE

The compatibility path is `--alignment-mode none`. The shift-robust path uses
`--alignment-mode hard_bounded` or `--alignment-mode soft_bounded`.

Shift-Robust Raw-Waveform Dynamic-FPDE performs bounded local lag alignment
between each input window and the target/rival raw prototypes before computing
prototype-directional evidence. It is not global DTW. It uses non-circular
zero-filled shifts with explicit masks; out-of-range waveform samples are zero
and out-of-range mask samples are false. Padding is excluded from distance,
cosine, evidence, and masked overlap-add.

Coarse-to-fine search first scores a coarse lag grid, selects the top
`coarse_top_k` coarse centers, explores each center within `fine_radius_ms`,
deduplicates fine lags, clips to `shift_max_ms`, and then runs hard or soft
alignment. The lag metrics are precomputed once per candidate and reused for
the fixed lambda grid:

```text
lambda_grid = 0.0, 0.1, ..., 1.0
C_lambda(delta) =
    lambda * normalized_mse(delta)
  + (1 - lambda) * cosine_distance(delta)
  + overlap_penalty_weight * (1 - overlap_ratio(delta))
```

`hard_bounded` chooses the deterministic minimum-cost lag, preferring smaller
absolute lag and then smaller signed lag for ties. `soft_bounded` uses a
numerically stable softmax over valid lags:

```text
q_delta = softmax(-C_lambda(delta) / alignment_temperature)
```

Target and rival prototypes are aligned independently with identical alignment
settings, so `target_best_lag` and `rival_best_lag` may differ. Lambda affects
both the alignment cost and the Raw-Hyb mixture. Therefore bounded Raw-Diff,
Raw-Cos, and Raw-Hyb are lambda-dependent and are written per lambda.

For each lambda:

```text
scaled_diff = phi_diff / (sum(abs(phi_diff)) + eps)
scaled_cos = phi_cos / (sum(abs(phi_cos)) + eps)
phi_hyb = lambda * scaled_diff + (1 - lambda) * scaled_cos
```

This scaling applies only to explanation components. The waveform itself is not
amplitude-normalized.

## Raw Prototypes And Evidence

Training waveforms are converted to raw sliding windows label by label. The
artifact runner builds one label bank at a time, scores candidate medoids with
block-wise matrix operations, and then keeps only the selected prototype unless
`--retain-segment-banks` is set. The medoid distance is masked mean squared
distance, so short padded windows are not favored merely because they contain
fewer valid samples:

```text
B_c = {w_1, w_2, ..., w_M}
p_target in R^L
p_rival in R^L
d(w_i, w_j) = mean((w_i[valid] - w_j[valid]) ** 2)
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

With `alignment_mode=soft_bounded` or `hard_bounded`, Raw-Diff and Raw-Cos use
the shifted target/rival prototypes selected by bounded alignment. Soft
alignment averages per-lag sample contributions by the valid sample's nonzero
alignment-weight sum; samples with no valid aligned contribution remain zero.

## Run

Smoke run:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --fold 1 \
  --seed 0 \
  --alignment-mode none \
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
  --prototype-selection exact_medoid \
  --alignment-mode soft_bounded \
  --shift-max-ms 20 \
  --coarse-step-ms 1 \
  --fine-radius-ms 2 \
  --fine-step-samples 1 \
  --coarse-top-k 3 \
  --minimum-overlap-ratio 0.8 \
  --alignment-temperature 0.05 \
  --overlap-penalty-weight 1.0 \
  --medoid-block-size 128 \
  --context-device cuda \
  --resume \
  --skip-completed-samples \
  --device cuda
```

`--context-cache-dir` defaults to `outputs/.../cache/raw_context`. The cache key
includes the train split hash, fold, seed, target sample rate, segment/hop
seconds, prototype method, medoid block size, candidate cap, context device,
alignment mode, shift bounds, coarse/fine search settings, minimum overlap
ratio, alignment temperature, overlap penalty, installed FPDE version, and
artifact commit. With `--resume --skip-completed-samples`, fold-level checkpoints are
written under `fold_<N>/results/` plus `fold_<N>/completed_samples.txt`.
`--resume` also skips completed samples by default, so rerunning a checkpointed
fold does not duplicate CSV rows. Sample rows deduplicate on fold, seed,
sample ID, lambda, and alignment mode; method rows deduplicate on fold, seed,
sample ID, method, and alignment mode. `--overwrite` ignores checkpoints and caches.
The compact context cache stores prototypes and masks only; when
`--retain-segment-banks` is used, the cache is disabled so the in-memory context
really contains full segment banks. If a compact context cache cannot be read,
the runner deletes it and rebuilds the context.

Optional label-conditioned RAW generation is connected with:

```bash
python experiments/dynamic_fpde_audio/run_esc50_raw_waveform_fpde.py \
  --dataset-root data/ESC-50 \
  --output-dir outputs/raw_waveform_dynamic_fpde_esc50_smoke \
  --mode smoke \
  --device cuda \
  --generation-scope selected \
  --raw-generator my_package.my_module:generator
```

The hook signature is:

```python
generator(label, lambda_hyb, segment, sample_rate, role, metadata) -> waveform
```

This hook is called only after explanation and positive/negative important
segment extraction, and only when `--generation-scope selected` or
`--generation-scope all` is set. Generated RAW artifacts are inspection outputs
and are not used in explanation computation. If omitted, or when
`--generation-scope none` is used, generated RAW artifacts are skipped and
recorded as `skipped`.

## Outputs

The Raw-Waveform runner writes:

- `raw_waveform_config.json`
- `results/raw_waveform_sample_metrics.csv`
- `results/raw_waveform_method_metrics.csv`
- `results/window_alignment_metrics.csv`
- `results/raw_waveform_summary_by_lambda.csv`
- `results/raw_waveform_errors.csv`, only when errors are skipped
- `fold_<N>/results/*.csv` and `fold_<N>/completed_samples.txt`, when resume/checkpoint mode is used
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
`alignment_mode`, `lambda_hyb`, `evidence`, `n_windows`, `input_length`, `sample_rate`,
`phi_shape`, `shape_match`, generation status, top positive segment metadata,
top negative segment metadata, runtime columns, `device`, `segment_length`, and
`hop_length`. They also include `evidence_total`, `evidence_per_window`,
`evidence_per_valid_sample`, positive/negative/absolute evidence,
positive/negative window rates, `n_valid_samples`, `coverage_rate`, medoid
runtime, context cache status, prototype selection settings, and resampler
metadata. For bounded alignment, they also include target/rival lag summaries,
boundary hit rates, confidence means, and alignment valid rate.

`raw_waveform_method_metrics.csv` stores separate rows for
`raw_diff_unscaled_no_alignment`, `raw_cos_unscaled_no_alignment`,
`raw_hyb_l1_no_alignment_lambda_X`, and bounded-mode
`shift_robust_raw_diff_lambda_X`, `shift_robust_raw_cos_lambda_X`, and
`shift_robust_raw_hyb_lambda_X`. This separates method choice, component
scaling, alignment mode, and lambda choice.

`window_alignment_metrics.csv` stores compact per-window alignment details:
target/rival best lag in samples and milliseconds, alignment cost, overlap
ratio, entropy, confidence, and validity.

Runtime columns separate fold-level context cost from sample-level explanation
and save cost. `fold_context_runtime_sec` is recorded for provenance,
`context_runtime_amortized_sec` divides it by the number of test samples, and
`sample_total_runtime_sec` excludes the fold context build. The context builder
uses exclusive timers for resampling, windowing, bank stacking, and medoid
selection.

The runner reports all lambda grid points. It records no test-sample
best-lambda selection for evaluation; `best_lambda` from the lower-level FPDE
object is intentionally ignored by the artifact CSV surface.

Every lambda result must satisfy:

```text
raw waveform shape == phi_hyb(lambda) shape
```

## Sensitivity Script

`experiments/dynamic_fpde_audio/run_shift_robust_sensitivity.py` runs a
synthetic zero-padded-shift sensitivity check over:

- `alignment_mode`: `none`, `hard_bounded`, `soft_bounded`
- `shift_max_ms`: `0`, `5`, `10`, `20`, `50`
- artificial input shifts: `-20`, `-10`, `-5`, `-2`, `-1`, `0`, `1`, `2`, `5`, `10`, `20` ms

It reports evidence differences, sign agreement, window evidence Spearman,
phi Pearson/Spearman, top-k Jaccard, rival label agreement, selected lag error,
boundary hit rate, and alignment confidence. Artificial shifts use zero padding
and masks, not circular shift.

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
primary RawFeat evidence surface.

## Interpretation Limits

Raw-Waveform-only Dynamic-FPDE is prototype evidence decomposition over raw samples.
It is not a causal explanation, is not a ground-truth explanation, does not
claim black-box classifier faithfulness, does not perform global DTW alignment,
and does not claim perfect time invariance or raw waveform reconstruction from
labels. Audio is converted to `target_sr` before raw sliding-window evidence is
computed.
