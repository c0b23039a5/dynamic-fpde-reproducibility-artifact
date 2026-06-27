# RawFeat Dynamic-FPDE Audio Experiments

RawFeat Dynamic-FPDE is the primary ESC-50 workflow in this artifact. It uses
raw waveform frames together with frame-level acoustic descriptors to explain
prototype-contrast evidence over one concatenated temporal representation.

## Processing contract

For sample `i`, the input contract is:

```text
raw_i       shape: (T_i, frame_length)
features_i  shape: (T_i, F)
dt_i        shape: (T_i,)
mask_i      shape: (T_i,)
```

Audio is decoded with SoundFile, mixed to mono, checked for empty/non-finite
samples, and converted to `target_sr` with `scipy.signal.resample_poly`. Raw
framing and acoustic feature extraction use exactly the same `target_sr`,
`frame_length`, and `hop_length`, so their time dimensions must match. A clip
shorter than one frame is zero-padded only to create one frame. Longer clips
remain variable-length, and an end-aligned final frame is included when a
partial tail would otherwise be dropped.

There is no global fixed-length temporal resampling. Padding across samples is
used only inside the FPDE batch representation and is accompanied by a boolean
mask. `dt[0]` is zero and later entries are finite frame-time increments.

The engine fits masked class prototypes over concatenated raw, feature, and
time-delta channels. For every explanation, the runner checks:

```text
evidence ~= attributions.sum()
raw attribution shape == raw frame shape
feature attribution shape == feature shape
time importance shape == (T_i,)
```

## Run

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

Use `--folds 1,2,3,4,5` for multiple folds. `smoke`, `pilot`, and `full` use
the existing deterministic ESC-50 split limits. The RawFeat runner is
NumPy/SciPy/SoundFile based and does not require CUDA or new deep-learning
dependencies.

`--normalize none|l1` controls plotted time-importance scaling. It does not
modify the evidence or attributions used by the exactness audit.

## Generated RAW audit

`--generation-scope none` disables generation. `selected` and `all` enable the
single configured RawFeat hybrid lambda in the current runner. The generator
is fit on training raw frames, labels, acoustic features, and masks. It creates
`(T, frame_length)` raw frames conditioned on the test feature sequence.

Generated RAW is for inspection and audit only. It is not fed back into the
original explanation and is not evidence of faithful waveform reconstruction.
The runner overlap-adds the frames, saves `generated_target.wav`, reads that WAV
again, re-extracts acoustic features, and only then calls
`DynamicFPDEEngine.explain_one` for the generated audit.

`PrototypeRawGenerator` contains training-derived prototypes, feature
summaries, nearest-neighbor metadata, and residual information. Its state and
outputs may leak membership or source characteristics. Apply dataset licensing,
privacy review, access control, and publication review before sharing either.

## Output schema

`rawfeat_config.json` records the processing contract and interpretation limit.
`results/rawfeat_sample_metrics.csv` contains one row per explained sample with:

- dataset/fold/seed/sample identity and target/rival labels
- method, lambda, and export normalization
- evidence, absolute evidence, attribution sum, and exactness residual
- raw, feature, and time-delta group attributions
- raw, feature, attribution, and time-importance shapes
- shape-match and audit-pass flags
- time/channel dimensions and runtime

`results/rawfeat_sample_summary.csv` aggregates by method/lambda and includes
mean exactness residual and mean absolute evidence.

When generation is enabled, `results/rawfeat_generation_metrics.csv` records
the generated/reprocessed shapes, reconstructed WAV path, generated evidence,
generated exactness residual, and generator-neighbor audit metadata.
`results/rawfeat_generation_summary.csv` aggregates those audits.

Per-sample files are:

```text
samples/<sample_id>/summary.csv
samples/<sample_id>/rawfeat_hyb_lambda_<lambda>/metrics.json
samples/<sample_id>/rawfeat_hyb_lambda_<lambda>/generated_target.wav  # optional
samples/<sample_id>/rawfeat_hyb_lambda_<lambda>/time_importance.png   # optional
```

With `--skip-errors`, failures are isolated and written to
`results/rawfeat_errors.csv`. Without it, the first failure stops the run.

## Interpretation limits

RawFeat Dynamic-FPDE is a prototype evidence decomposition. It is not a causal
or ground-truth explanation, does not establish black-box classifier
faithfulness, and does not claim raw-waveform attribution beyond the stated
concatenated-representation evidence. Generated audio is an inspection/audit
surface only.

The Raw-Waveform-only workflow is preserved as a legacy/comparison runner, and
the Native-Time feature-only workflow is preserved as a comparison runner. See
`docs/dynamic_fpde_experiments.md` for the former.
