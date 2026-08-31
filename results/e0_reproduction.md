# E0 — Reproduction of ESM2-650M + HP baseline

Ran 2026-08-30 on a single RTX 3090 (24 GB), repo `finetune-esm`, branch `ppb-affinity`,
commit prior to this file's own commit.

## Commands executed (verbatim, in order)

```bash
uv run python -m finetune_esm.train build-cache --backbone esm2-650m --device cuda --batch-size 8

uv run pytest tests/test_cache.py -v -m slow

for seed in 0 1 2; do
  uv run python -m finetune_esm.train train \
    --backbone esm2-650m --fusion hp --split author --seed "$seed" \
    --num-epochs 30 --batch-size 32 --learning-rate 1e-3
done
```

No hyperparameters, split, or code were changed from the brief. Batch size for the cache
build (8) and training (32) were used exactly as specified; no batch-size reduction was
needed (GPU had 2.3/24 GB used before the run, and no OOM occurred at any step).

## Step 1 — cache build

Output: `cached 10304 unique chains to /home/ytian/github/finetune-esm/embedding_cache/esm2-650m`

- Wall-clock: **2m54.1s** (`real 2m54.065s`)
- `residues.f16.npy`: 4.7 GiB (`du -sh`; matches the expected ~5.00 GB for
  1,953,372 truncated residues × 1280 dims × fp16)
- Also produced `pooled.f16.npy` (26 MB), `index.parquet` (549 KB), `meta.json`

## Step 2 — cache verification against live forward pass

`uv run pytest tests/test_cache.py -v -m slow`

```
tests/test_cache.py::test_cache_round_trips_and_matches_live_forward_pass PASSED
tests/test_cache.py::test_pooled_equals_mean_over_residues PASSED
tests/test_cache.py::test_duplicate_sequences_are_encoded_once PASSED
tests/test_cache.py::test_mismatched_metadata_is_rejected PASSED
tests/test_cache.py::test_unknown_sequence_raises PASSED
tests/test_cache.py::test_corrupt_index_is_rejected PASSED
tests/test_cache.py::test_expect_ignores_n_chains PASSED
tests/test_cache.py::test_crashed_rebuild_leaves_no_valid_meta PASSED

8 passed, 1 deselected in 9.17s
```

All slow cache tests passed, including the direct comparison against a live ESM-2
forward pass. Proceeded to training.

## Step 3 — three-seed training on the author split

| Seed | Spearman | Pearson | RMSE | Wall-clock | Best-epoch checkpoint |
|---|---|---|---|---|---|
| 0 | 0.4310 | 0.4225 | 1.5766 | 3m14.8s | `epoch=2-step=609.ckpt` |
| 1 | 0.4253 | 0.4146 | 1.6196 | 2m47.4s | `epoch=1-step=406.ckpt` |
| 2 | 0.4290 | 0.4164 | 1.6173 | 5m29.0s | `epoch=9-step=2030.ckpt` |
| **Mean** | **0.4284** | **0.4178** | **1.6045** | | |
| **Sample SD (n=3)** | **0.0029** | **0.0041** | **0.0242** | | |

Raw per-seed test-set values (full precision, as printed by Lightning's test loop):

```
esm2-650m/hp/author seed=0: spearman=0.430962473154068 pearson=0.4224647283554077 rmse=1.5765711069107056
esm2-650m/hp/author seed=1: spearman=0.42529916763305664 pearson=0.4145525097846985 rmse=1.6196300983428955
esm2-650m/hp/author seed=2: spearman=0.4289807379245758 pearson=0.4164023697376251 rmse=1.6173431873321533
```

**Epochs completed.** The run uses Lightning's `EarlyStopping` on `val_spearman` with
`patience=5` and `ModelCheckpoint(save_top_k=1)` on the same metric, with `--num-epochs 30`
as the cap. The checkpoint filename Lightning restored for testing names the *best* epoch
(0-indexed) for each seed: seed 0 → epoch 2, seed 1 → epoch 1, seed 2 → epoch 9. Because
stdout was captured non-interactively, Lightning's per-epoch progress bars were not written
to the log, and no `metrics.csv` / TensorBoard log was produced (the training entrypoint
does not configure a file logger), so the exact epoch at which each run *stopped* (as
opposed to its best epoch) is not recoverable from the captured output. Given
`patience=5` and no OOM/crash, all three runs stopped short of the 30-epoch cap via early
stopping (each restored-checkpoint epoch is well under 30, and none of the three checkpoint
files carries an epoch index anywhere near 29).

Total training wall-clock across the three seeds: 11m31.2s. Combined with cache build
(2m54.1s) and cache verification (9.2s), the full harness ran in **~14m34s**.

## Comparison to reference values

| Source | Spearman | Pearson | RMSE |
|---|---|---|---|
| **This run (mean of 3 seeds, MLP head)** | **0.4284** | **0.4178** | **1.6045** |
| Published: ESM2-650M + HP, frozen + ConvBERT head (arXiv:2505.20036) | 0.47 | 0.48 | 1.74 |
| Published best on benchmark: ProtT5 + PAD | 0.48 | 0.51 | 1.42 |
| This project's trivial-feature floor (no PLM) | 0.229 | 0.226 | 1.896 |

**Note on architecture.** This harness's head is a plain MLP on top of hierarchical-pooled
(HP) fusion, frozen ESM2-650M embeddings. The published ESM2-650M + HP number used a
ConvBERT head instead of an MLP. Some deviation between our number and the published
number is therefore expected from this head-architecture difference alone, on top of any
other sources of variance (fusion implementation details, optimizer/schedule, exact
early-stopping behavior). The published number, not the number produced here, remains the
bar to beat for downstream work.

No gate verdict is recorded in this file.

## Gate verdict (recorded by the controller, after the run)

**ACCEPTED — reproduction succeeded.** Mean Spearman 0.4284 differs from the published
0.47 by 0.0416, inside the plan's ±0.05 acceptance band. Research may proceed.

Supporting evidence beyond the headline number:

- **Well clear of the floor.** 0.4284 is nearly double this project's trivial-feature
  baseline of 0.229, so the PLM is contributing real signal rather than re-deriving
  length and composition.
- **Stable.** Sample SD across three seeds is 0.0029 — the pipeline is deterministic and
  the number is not seed luck.
- **Cache proven, not assumed.** The cached embeddings were compared against a live ESM-2
  forward pass before any training ran.
- **The residual gap has a named cause.** Our head is a plain MLP; the published run used
  a ConvBERT head. The published number remains the bar.

### An observation worth carrying into the research phase

Predicting a single constant for every test complex gives RMSE 1.9162 using the train mean,
or **1.6145** using the test mean (the best any constant could do — it is just the test
standard deviation).

| Model | RMSE | vs. oracle constant (1.6145) |
|---|---|---|
| This run (MLP head) | 1.6045 | beats it, but only just |
| Published ESM2-650M + HP (ConvBERT head) | 1.74 | **worse than a constant** |
| Published best, ProtT5 + PAD | 1.42 | beats it clearly |
| Trivial features (no PLM) | 1.896 | worse than a constant |

Two things follow. First, our run trails the published Spearman (0.4284 vs 0.47) while
*beating* its RMSE (1.6045 vs 1.74) — better calibrated, worse ranked. That is the
signature of optimising MSE while being scored on Spearman, which is the mismatch experiment
E2 targets. Second, a published result on this benchmark reports an absolute error worse
than predicting the mean, which suggests the one-log train/test label shift documented in
spec §2 is going uncorrected and makes the calibration experiment E7 more attractive than
its position in the ordering implies.

All three runs peaked early — best epochs 2, 1 and 9 of a 30-epoch cap — so the model
converges fast on 6,485 training rows with a frozen backbone.

## Post-gate diagnostic: does the fusion architecture matter here?

Prompted by the final review's observation that `AttentionPool` at initialisation has
effective support over 1013 of 1022 residue positions — i.e. it is near-uniform, so early
in training hierarchical pooling behaves like mean pooling — the other three fusion
strategies were run on the same cache, same split, seed 0.

| Fusion | Spearman | Pearson | RMSE |
|---|---|---|---|
| HP (the E0 configuration) | 0.4310 | 0.4225 | 1.5766 |
| EC (embedding concatenation) | 0.4366 | 0.4331 | 1.5928 |
| SC (sequence concatenation) | 0.4449 | 0.4479 | 1.6049 |
| PAD (pooled attention addition) | 0.4402 | 0.4526 | 1.6327 |

All four land within **0.014** of each other. The benchmark paper's central claim is that
HP and PAD beat the naive concatenation baselines by *up to 12% Spearman*, which on a 0.47
baseline is about 0.056 — roughly four times the total spread observed here. The ordering
is also inverted: **SC, the least structured of the four, scores highest, and HP scores
lowest.**

This does not by itself refute the paper. It is consistent with at least three readings:
our fusion modules may not be learning much beyond mean pooling within the few epochs
early stopping allows; the architecture effect may not survive on this split; or all four
implementations may converge to similar solutions given a frozen backbone and a small head.

It does, however, change what E0 licenses. The gate remains ACCEPTED — 0.4284 is inside the
±0.05 band and nearly double the 0.229 floor — but the explanation offered above for the
residual gap ("our MLP head versus the paper's ConvBERT") is now weakened rather than merely
untested: if the choice of fusion architecture moves the number by only 0.014 in this
harness, the head and fusion story is unlikely to account for a 0.04 gap. Note also that
spec §5.3 called for a ConvBERT head that was never built, so the comparison has still
never been run.

The practical consequence is that **E1 (the scrambled-partner and single-partner controls)
should run before anything else**, and this table should be read as a preview of it: four
different ways of combining the two partners — including one that simply concatenates
everything and pools once — produce the same answer.
