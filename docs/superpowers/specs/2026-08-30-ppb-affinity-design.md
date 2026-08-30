# Design: Protein–Protein Binding Affinity Regression on PPB-Affinity

Date: 2026-08-30
Status: Approved for implementation planning
Supersedes: the CAFA-5 multi-label GO-term classification task in this repo

## 1. Goal

Repurpose `finetune-esm` from CAFA-5 GO-term classification to protein–protein
binding affinity regression on the `proteinea/ppb_affinity` benchmark, and beat
the published state of the art on that benchmark.

Two deliverables, sequenced:

1. **Engineering.** A lean, current, reproducible harness that makes experiments
   fast and trustworthy. Its only job is to serve the research.
2. **Research.** A credible result above the published ceiling, with an
   evaluation protocol honest enough that the result survives scrutiny.

### Non-goals

- Retaining the CAFA-5 classification path. It is deleted; git history preserves it.
- Multi-node distributed training. The workload does not need it (see §5.1).
- Structure-based modelling as a primary axis (see §4.3).
- Serving or deployment. The old README roadmap item stays unchecked.

## 2. The dataset, as measured

All figures below were computed directly from `filtered.csv`
(`huggingface.co/datasets/proteinea/ppb_affinity`, CC-BY-NC-SA-4.0), not taken
from the paper. They drive most of the design decisions that follow.

| Property | Value |
|---|---|
| Rows | 8,207 (train 6,485 / val 965 / test 757) |
| Columns | 21, including `Ligand Sequences`, `Receptor Sequences`, `KD(M)`, `Mutations`, `PDB`, `split` |
| Target | pKd = −log₁₀(KD in M); range 1.39–15.70, mean 7.86, sd 2.07 |
| Unique PDB entries | train 950, val 648, test 573 |
| PDB entries spanning >1 split | **0** (splits are clean) |
| Unique chain sequences | 10,304 (from 20,709 chain instances) |
| Chain length | median 178, p99 892, max 1,634; 0.5% exceed 1,022 |

Three properties matter more than the headline size:

**Train/test composition differs sharply.** Train is 78% SKEMPI v2.0 and 73% of
its rows carry a mutation. Test is 88% PDBbind v2020 and only 2.8% of its rows
carry a mutation. The training signal is dominated by "how does this point
mutation shift affinity"; the test task is "what is the absolute affinity of this
novel complex."

**Training redundancy is severe.** 6,485 train rows span only 950 unique PDB
entries — 6.8× redundancy. The single largest group, 3SGB, contributes 280 rows
(4.3% of training) differing only by mutations at one interface. The ten largest
PDB groups contribute 1,849 rows — **28.5% of the training set** (22.5% of all
rows), and all ten fall entirely within train. Uniform sampling therefore spends
most of the gradient budget on a handful of interfaces.

**Labels are shifted between splits.** Train mean pKd 8.05 vs test 7.02 — a
one-log offset. Predicting the train mean everywhere yields test RMSE 1.916; the
oracle constant (test mean) yields 1.616; the best published model reaches 1.42.
A meaningful share of published RMSE is uncorrected distribution shift.

### 2.1 Baseline calibration

A gradient-boosted tree on sequence lengths, chain counts, amino-acid composition,
and crystal resolution — no PLM — reaches **Spearman 0.212 / Pearson 0.222 /
RMSE 1.934** on the test split. Any claimed PLM result must be read against this
floor, not against zero.

## 3. Published results to beat

On this exact benchmark and split:

| Model | Spearman | Pearson | RMSE |
|---|---|---|---|
| ProtT5 + PAD, frozen + ConvBERT head | 0.48 ± 0.03 | 0.51 | 1.42 |
| Ankh2-Ext1 + HP, fine-tuned | 0.47 | 0.48 | 1.45 |
| ESM2-650M + HP, frozen + ConvBERT | 0.47 | 0.48 | 1.74 |
| Boltz-2-PPI (structure only) | 0.357 | 0.338 | 1.362 |
| ProtT5-PAD + Boltz-2-PPI (hybrid) | ~0.50 | ~0.52 | 1.326 |

Sources: arXiv:2505.20036 (the benchmark paper) and arXiv:2512.06592 (the Boltz-2
PPI follow-up). The follow-up's Pearson/Spearman column labels appear transposed
relative to the source paper; treat its correlations as ±0.02.

**Target: Spearman > 0.50 on the authors' split, with the honest-evaluation
caveats of §6 reported alongside.**

## 4. Where the headroom is

### 4.1 Loss/metric mismatch (highest confidence)

The benchmark trains with MSE and evaluates with Spearman. Three independent
2025–26 results show large effects from switching to a ranking objective:

- ProtAff: Spearman 0.413 with margin ranking vs **0.163** with MSE, same architecture.
- AbLWR: FRA 21.28% with ListMLE vs **3.29%** with MSE.
- GeoStab-suite: soft-Spearman + MSE gives 0.60/0.61 vs 0.44/0.46 for the next best.

No clean pointwise-vs-pairwise-vs-listwise ablation exists for *absolute* PPI
affinity on a sequence-identity-controlled split. That is a well-defined gap.

### 4.2 Data structure (unaddressed by the benchmark)

The redundancy and label shift in §2 are untouched by the published work.
Group-aware sampling or weighting, and modelling the SKEMPI rows as offsets from
a wild-type anchor rather than as independent absolute measurements, both follow
directly from the measured structure of the data.

### 4.3 What not to pursue

**Structure.** Two independent results say it does not pay here. Boltz-2-PPI
reaches 0.357 alone and adds ~+0.02 in hybrid. BindPred's ablation: ESM-2
embeddings alone Pearson 0.860, PyRosetta energy features alone 0.783, both
together 0.861 — a +0.001 gain. Structure earns its keep on ΔΔG, not on absolute
affinity across heterogeneous complexes and assay conditions.

**Backbone scaling as a primary bet.** On this benchmark ProtT5 (0.48) ≈
Ankh2-Ext1 (0.47) ≈ ESM2-650M (0.47). Backbone choice is worth ~0.01; fusion
architecture is worth ~0.05. Newer backbones are a cheap parallel track, not the
main hypothesis.

### 4.4 The control experiment that reframes everything

ProtAff's own target ablation reports Spearman 0.413 with the true target,
**0.432 with a scrambled target**, and 0.510 with an unrelated antigen — the model
scores one sequence in isolation and largely ignores its partner.

The benchmark paper's entire thesis is that partner-fusion architecture matters
(HP/PAD beat naive concatenation by up to 12% Spearman). If a scrambled-partner
control scores comparably on this benchmark, that thesis does not hold and the
published architecture ranking measures something other than interaction
modelling. Combined with the 0.212 trivial-feature floor of §2.1, this is a
cheap experiment with a large information payoff either way. It runs on cached
embeddings in under a day.

## 5. Architecture

### 5.1 Stack

**uv** for packaging and environments, replacing the pinned `requirements.txt`
freeze; this also fixes the flat imports that currently force execution from
inside `finetune_esm/`. **Lightning** for the training loop, with `FSDPStrategy`
available behind a flag for rented multi-GPU nodes. **MLflow** for tracking.
**Optuna** for sweeps.

**Ray is removed.** Its value is multi-*node* orchestration; this workload fits on
one node with room to spare. Lightning handles single-node multi-GPU natively via
`devices=N, strategy="fsdp"`. Removing Ray also removes `CustomRayFSDPStrategy`,
which exists only to patch a Ray-specific state-dict key bug via private
`ray.train.lightning._lightning_utils` symbols that have since moved — the
fragile part disappears rather than needing modernisation. A plain
`torchrun`-compatible entrypoint additionally runs unmodified on EC2, SageMaker,
Lambda Labs, Modal, or RunPod, whereas Ray requires standing up a cluster on each.

FSDP does nothing on a single 3090 — it shards across devices. It is retained for
rented multi-GPU nodes and for the repo's stated "scalable distributed training"
identity. Keeping it means porting to PyTorch's current FSDP2 (`fully_shard`)
interface, not preserving the 2024 code; the exact Lightning-to-FSDP2 path is to
be confirmed during implementation.

### 5.2 Two-tier training

**Screening tier (default).** The backbone is frozen, so its output is a pure
function of the sequence and is computed once into a cache. Per-residue fp16
embeddings for all 10,304 unique chains cost **5.0 GB** at 1,280 dimensions
(ESM-2 650M) and ~10 GB at 2,560 (ESM-2 3B). After one build pass, every
experiment on fusion architecture, loss, sampling scheme, or split trains on
cached tensors in seconds to minutes. This is what allows several hundred
experiments on one consumer GPU where the benchmark authors ran dozens on two
H100s.

**LoRA tier.** The top configurations from screening graduate to LoRA
fine-tuning. This tier exists because the frozen-only assumption is contradicted
by the evidence: ProtAff reports 0.257 frozen vs **0.413** with LoRA on Q/K/V of
the last four layers; Schmirler et al. (*Nat Commun* 2024, 615 method-task
combinations) find fine-tuning "almost always improves downstream predictions"
with PEFT matching it at 4.5× speedup; BALM-PPI's RMSE degrades from 0.994 to
1.192 without PEFT. A counterexample exists — PEER's human PPI accuracy *drops*
from 83.6% to 77.3% under fine-tuning — so the tier is an experiment, not an
assumption.

### 5.3 Package layout

```
finetune_esm/
  data/
    dataset.py     # HF download, parse, pKd derivation, row schema
    cache.py       # embedding cache build + memmap reader
    splits.py      # authors' split, cold-PDB, random, (stretch) interface-based
  models/
    backbones.py   # PLM registry behind one interface
    fusion.py      # EC, SC, HP, PAD, and new fusions
    heads.py       # MLP and ConvBERT-style heads
    lora.py        # repaired hand-rolled LoRA
    module.py      # LightningModule
  losses.py        # MSE, margin ranking, ListMLE, soft-Spearman, hybrids
  metrics.py       # Spearman, Pearson, RMSE
  train.py         # CLI entrypoint
  sweep.py         # Optuna study driver
  config.py
tests/
```

Each unit has one job and a stated interface: `backbones` maps a model id to
`(tokenizer, encoder, hidden_dim)`; `fusion` maps a list of per-chain residue
embeddings plus masks to a single vector; `heads` maps that vector to a scalar;
`losses` maps `(pred, target, group_id)` to a scalar. Fusion strategies are
interchangeable behind one signature, which is what makes the §7 matrix a sweep
rather than a rewrite.

### 5.4 Cache format

Keyed by SHA-1 of the chain sequence, which deduplicates 20,709 chain instances
to 10,304 encoder calls. Per backbone:

- `embeddings.f16.npy` — memmapped `(total_residues, dim)`
- `pooled.f16.npy` — `(n_chains, dim)` mean-pooled, for fusions that need only chain vectors
- `index.parquet` — `chain_sha1 → (offset, length)`
- `meta.json` — backbone id and revision, dtype, dim, truncation limit, tokenizer revision

Truncation at 1,022 residues affects 0.5% of unique chains, so it is the default;
`meta.json` records it so a cache built under different settings is never silently
reused.

### 5.5 LoRA

Kept as a from-scratch implementation rather than replaced by PEFT — it has value
in a showcase repo and the current file is only ~160 lines. Two bugs are fixed:

1. `LoRALayer.forward` scales by `alpha`; the standard formulation scales by
   `alpha / rank`. With the current defaults (alpha=16, rank=8) the effective
   update is 8× the conventional size, making published learning rates
   non-transferable.
2. `apply_lora` freezes all parameters and then wraps `pre_classifier` and
   `classifier`, leaving their base weights at random initialisation permanently
   frozen. The head then learns only through a rank-8 bottleneck starting from a
   random projection. The head's base weights must remain trainable.

Correctness is pinned by a test asserting numerical equivalence to PEFT under
identical initialisation (§8).

## 6. Evaluation protocol

**Primary split:** the authors' `split` column (MMseqs2, ≤30% identity). This is
what makes our numbers directly comparable to the published 0.48.

**Reported alongside:** cold-PDB and random splits, following BALM-PPI, whose own
three-way report (Pearson 0.887 random / 0.730 cold / 0.612 at <30% identity) shows
how much a single number can flatter. Single-split claims on this dataset are close
to meaningless.

**Stretch:** an interface-similarity split via iDist. Bushuiev et al.
(arXiv:2404.10457) show a 30%-identity MMseqs2 split still leaves ~30% leakage,
so even the "honest" 0.48 is optimistic. An interface-based split would be a
stronger claim than anything published on this dataset.

**Metrics:** Spearman (primary, and the checkpoint-selection criterion, matching
the benchmark), Pearson, RMSE in pKd. Mean ± sd over 3 seeds.

**Mandatory controls**, run before any result is believed:

1. **Scrambled partner** — replace the receptor with a random receptor from the same split.
2. **Single partner** — ligand only, and receptor only.
3. **Trivial features** — the 0.212 floor from §2.1.
4. **Label shuffle** — must collapse to ~0.

## 7. Experiment plan

Ordered by expected value. Each stage gates the next.

| # | Experiment | Purpose |
|---|---|---|
| E0 | Reproduce ESM2-650M + HP, frozen | Gate. Expect ≈0.47. If we cannot reproduce it, nothing downstream is credible. |
| E1 | The four controls of §6 | Establish what the benchmark actually measures. |
| E2 | Loss ablation: MSE / margin ranking / ListMLE / soft-Spearman / hybrid | The best-evidenced lever (§4.1). |
| E3 | Group-aware sampling and weighting by PDB cluster | Addresses the 6.8× redundancy (§2). |
| E4 | LoRA tier on the best screened configuration | Tests the frozen ceiling (§5.2). |
| E5 | Backbone sweep: ESM-C 300M/600M/6B, MINT, Ankh3, ESM-2 650M | All unevaluated on this benchmark. |
| E6 | Ensembling across the top configurations | The published top-10 spans 0.04 Spearman — a natural ensemble setup. |
| E7 | Calibration for the one-log train/test label shift | Targets RMSE specifically. |

Notes on E5. ESM-C was re-released by Biohub under MIT on 2026-06-09 including the
6B checkpoint; no PPI affinity paper has used it. Because the screening tier needs
only frozen inference, 6B in fp16 (~12 GB) fits on the 3090 for a one-time cache
build even though training it there would be impossible. MINT (*Nature
Communications* 2025, [10.1038/s41467-025-67971-3](https://doi.org/10.1038/s41467-025-67971-3),
CC-BY-4.0) is an ESM-2 650M backbone pretrained on 96M STRING PPIs with cross-chain
attention inside every layer — structurally different from the benchmark's four
late-fusion strategies, and absent from its model list. Ankh3 (arXiv:2505.20052)
shipped the same week as the benchmark paper from the same lab and was not
evaluated on it.

## 8. Testing

pytest, targeting the failure modes that would silently corrupt a result rather
than aiming at coverage:

- **Split integrity** — no PDB entry spans two splits; row counts match 6,485/965/757.
- **Cache equivalence** — for a random sample of chains, cached embeddings match a live forward pass within fp16 tolerance.
- **Cache invalidation** — a cache built with different `meta.json` settings is rejected, not silently reused.
- **LoRA equivalence** — the hand-rolled implementation matches PEFT numerically under identical initialisation.
- **Metric correctness** — Spearman/Pearson/RMSE against `scipy` on fixtures, including ties and constant-prediction edge cases.
- **Target derivation** — pKd from KD(M) round-trips; the known range 1.39–15.70 is reproduced.
- **Control wiring** — the scrambled-partner control actually permutes partners and does not accidentally leak the true one.

## 9. Deletions

`finetune_esm/train.py` Ray/FSDP orchestration; `CustomRayFSDPStrategy` and the
Ray imports in `utils.py`; `data.py` in its entirety, replaced by the `data/`
package of §5.3; the CAFA-5 `FinetuneESM` classification head and multi-label
metrics; `requirements.txt`;
the stale `finetune_results/` and `mlruns/` artifacts; the broken `data`
symlink (→ `../esm-lora/data/`, which no longer exists); `notebooks/deprecated/`.

`notebooks/cafa5_*.ipynb` are retained as historical record but moved out of the
active path.

## 10. Risks

| Risk | Mitigation |
|---|---|
| E0 fails to reproduce ≈0.47 | Hard gate. Diagnose before proceeding; the discrepancy is itself informative. |
| E1 invalidates the benchmark | Not a failure. Reframes the work as a critique with a corrected protocol — a stronger contribution than +0.02 Spearman. |
| Gains vanish under cold-PDB or interface splits | Expected to some degree; §6 reports all splits so this surfaces early rather than at review. |
| ESM-C 6B cache build too slow on one 3090 | Fall back to 600M. The 300M/600M checkpoints are the documented sweet spot anyway. |
| ProtT5 and Ankh3 are encoder-decoder T5 models | Embedding extraction differs from ESM; the `backbones` interface must abstract this, and it is the reason that abstraction exists. |
| MINT needs bespoke loading, not plain HF | Isolated behind the `backbones` interface; if integration costs more than a day, defer it. |
| Small test set (757 rows) makes ±0.02 differences noise | 3 seeds, report sd, and prefer effect sizes over rankings. |

## 11. Open questions

- Whether to pursue the iDist interface-based split (§6 stretch) in this cycle or defer it.
- Whether the SKEMPI mutational rows are better modelled as ΔΔG offsets from a wild-type anchor than as absolute pKd targets. E3 will inform this.
- Whether to fold in auxiliary training data (StaB-ddG's stability→binding transfer; ProtAff's 362k multi-source aggregation). Out of scope for this cycle, revisit after E4.
