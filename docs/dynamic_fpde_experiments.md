# Native-Time Dynamic-FPDE Audio Experiments

This artifact treats **Native-Time Dynamic-FPDE** as the intended
Dynamic-FPDE formulation. The older resampled-time, `prototype_length`-based
path is legacy and benchmark-oriented only.

For each audio clip, the runner builds a frame-level acoustic feature matrix
`X_i in R^{T_i x F}`. `T_i` may differ across clips and `F` is shared. The main
explanation output is `Phi_i in R^{T_i x F}` with the invariant
`Phi_i.shape == X_i.shape`. `time_importance = sum_f Phi[t, f]` and
`feature_importance = sum_t Phi[t, f]` are auxiliary summaries only.

The runner does not fixed-length-resample the time axis, temporally average,
max/min/pool, pad variable-length samples into a dense tensor, or use DTW
alignment. It does not claim verse/chorus alignment, raw waveform attribution,
causal explanation, or black-box model faithfulness.

## Dataset

The first supported dataset is ESC-50. The runner expects:

```text
data/ESC-50/
  audio/
  meta/esc50.csv
```

It reads `meta/esc50.csv`, uses `fold` for ESC-50 splits, and uses `category`
as the class label. The artifact does not download or redistribute ESC-50.

## Audio And Features

Dynamic-FPDE operates on frame-level acoustic feature matrices, not waveform
samples. Raw audio is decoded, resampled to a common `target_sr`, converted to
mono, converted into frame-level acoustic features, standardized from the
training fold, and then explained.

Dynamic-FPDE itself is not sampling-rate invariant. The common target sample
rate is a preprocessing convention, not a mathematical invariance claim.

Feature caches are written under `outputs/**/cache/features/` and remain
ignored by git.

If a decoded clip is shorter than `frame_length`, the feature extractor
zero-pads that clip internally to create a single analysis frame (`T == 1`).
This is an intra-clip minimum-frame guard only. It is not temporal alignment,
not fixed-length resampling, not global-duration normalization, and not dense
tensor batching. ESC-50 clips are normally longer than one frame.

## Native Prototypes

Native-Time uses feature-space vector prototypes:

```text
p_target in R^F
p_rival in R^F
```

It does not use time-series prototypes `P in R^{L x F}`. Prototype vectors are
selected from real exemplar frames:

```text
p_target = X_a[t_a, :]
p_rival = X_b[t_b, :]
```

The default selection rule is `nearest_to_class_centroid_frame`: collect all
standardized training frames for a class, compute the class frame centroid in
feature space, and choose the real frame nearest to that centroid. The
alternative `medoid_frame` also returns a real frame. Prototype metadata records
`source_sample_id`, `source_frame_index`, `source_time_sec`, `label`,
`prototype_mode`, `selection_rule`, and feature names when available.

For each test sample, the target prototype comes from the true class. The
common rival prototype is selected from non-target class exemplar prototypes by
nearest mean squared distance over native frames and is recorded as
`common_rival_label`.

## Run

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

`--folds` accepts a comma-separated ESC-50 fold list:

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

`--prototype-length` is rejected with a Native-Time legacy-mode error.

## Formulas

Native-Time Dynamic-Diff:

```text
Phi_diff[t, f] =
    (X[t, f] - p_rival[f]) ** 2
  - (X[t, f] - p_target[f]) ** 2
```

Positive values support the target prototype. Negative values support the rival
prototype.

Native-Time Dynamic-Cos uses an anchor `a in R^F`, defaulting to zero:

```text
z[t] = X[t, :] - a
q_target = p_target - a
q_rival  = p_rival  - a
```

`||z[t]||` is computed per frame. `||q_target||` and `||q_rival||` are
feature-vector norms. The implementation uses finite `eps` stabilization and
must not produce NaN or inf.

Native-Time Dynamic-Hyb:

```text
Phi_hyb = lambda_hyb * Phi_diff + (1 - lambda_hyb) * Phi_cos
```

For Native-Time, the default `normalize` is `none`. Optional `l1`
normalization never alters, resamples, pools, or pads the time axis.

## Backend

`--backend cpu` uses the FPDE Native-Time API from
`fpde-xai/fpde@dynamic`. If those APIs are not installed, the runner fails
clearly and does not fall back to legacy resampled-time Dynamic-FPDE.

`--backend cuda` requires CuPy and a usable CUDA device. CPU work still includes
audio decode, audio resampling to `target_sr`, mono conversion, feature
extraction, feature standardization, and deletion/insertion diagnostics. CUDA
acceleration applies only to Native-Time attribution tensor computation. The
main runner resolves all test samples first, then launches CUDA over grouped
batches. Variable-length samples are not padded, cropped, or resampled; CUDA
grouping is limited to samples that naturally share the same `(T, F)`.

## Outputs

The runner writes:

- `run_config.json`
- `native_time_feature_config.json`
- `environment_info.json`
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

Sample rows include Native-Time-specific checks and metadata:
`T`, `F`, `phi_shape`, `x_shape`, `shape_preserved`,
`target_prototype_source_sample_id`,
`target_prototype_source_frame_index`,
`target_prototype_source_time_sec`, `target_prototype_label`,
`rival_prototype_source_sample_id`,
`rival_prototype_source_frame_index`,
`rival_prototype_source_time_sec`, `rival_prototype_label`,
`prototype_mode`, `prototype_selection_rule`, `normalize`, `anchor`, and
`evidence_role`.

Runtime columns are split into:

- `selection_runtime_sec`, the shared common-rival selection time for the
  sample. It is repeated on every method row for that sample because the
  selected target/rival pair is shared by all Dynamic-FPDE methods and ranking
  diagnostics.
- `native_fpde_runtime_sec`, the Native-Time attribution computation time
  (`native_fpde_runtime_sec` is baseline ranking construction time for ranking
  diagnostics)
- `diagnostic_runtime_sec`, the CPU prototype-evidence diagnostic time
- `total_runtime_sec`, `selection_runtime_sec + native_fpde_runtime_sec +
  diagnostic_runtime_sec`
- `runtime_sec`, retained as a compatibility alias for `total_runtime_sec`

Before rows are written, the runner checks that `Phi.shape == X.shape`,
prototypes and anchors have shape `(F,)`, and `X`, prototypes, anchors, `Phi`,
and evidence are finite. Native-Time result rows do not contain
`prototype_length` or `resampled_length` fields.

## Diagnostics

Deletion/insertion curves are normalized prototype-evidence removal/recovery
diagnostics over native frame indices. They rank frames by signed, absolute, or
positive frame evidence depending on the configured diagnostic mode. The
default runner path uses signed evidence. The curves are not causal
faithfulness and do not imply black-box model faithfulness.

Total evidence can depend on the number of frames. Do not directly compare
total evidence across clips with different lengths without normalization or
careful interpretation.

## Tables

```bash
python scripts/make_dynamic_fpde_tables.py \
  --results-dir outputs/native_time_dynamic_fpde_esc50_smoke/results \
  --tables-dir outputs/native_time_dynamic_fpde_esc50_smoke/tables
```

Generated tables use Native-Time terminology:

- Native-Time Dynamic-Diff
- Native-Time Dynamic-Cos
- Native-Time Dynamic-Hyb
- Raw energy baseline
- Standardized feature-norm baseline
- Random baseline

`energy_baseline_raw` ranks native frames by raw acoustic feature-vector norm.
`feature_norm_baseline_standardized` ranks native frames by standardized
feature-vector norm. Neither is an FPDE attribution method or attribution
evidence; both are ranking baselines evaluated with the common prototype
evidence diagnostic. Internally these use `raw_feature_norm_scores` and
`standardized_feature_norm_scores`; the lower-level helper is
`frame_norm_scores`, with `energy_frame_scores` retained only as a
backward-compatible alias.

`table_dynamic_fpde_native_time_checks.tex` reports shape preservation,
prototype metadata availability, and additivity residuals.
