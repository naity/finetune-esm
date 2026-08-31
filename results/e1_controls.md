# E1 — What does this benchmark actually measure?

Run 2026-08-30/31 on a single RTX 3090, branch `research-controls`, against the same
ESM2-650M embedding cache used for the E0 reproduction. All figures are three seeds.

## Method

Each control rewrites the dataset so one source of information is destroyed, and the
model is **trained and evaluated** under that corruption. The question is therefore
"is this task learnable without the removed information?" — a claim about the benchmark,
not about one trained model.

Controls are applied after the split is fixed and independently within each split, so
the partition, its zero-PDB-leakage guarantee, row counts, PDB grouping and the label
multiset are unchanged. Scrambling uses a derangement, so nothing keeps its true partner
by accident.

`fixed_receptor` replaces every receptor with a single constant real sequence, so the
partner carries no per-example information whatsoever — a stricter ablation than
scrambling, which preserves the marginal distribution of partners.

## The four controls (MLP head, authors' ≤30% identity split)

| Control | Spearman | % of baseline |
|---|---|---|
| none (correct pairing) | 0.4284 ± 0.0029 | 100.0% |
| scramble receptor | 0.4129 ± 0.0082 | 96.4% |
| fixed receptor (no partner information) | 0.4026 ± 0.0088 | 94.0% |
| fixed ligand † | 0.3053 | 71.3% |
| **shuffle labels** | **−0.0527** | evaluation is sound |

† one seed.

## Head architecture: the loophole, closed

Spec §5.3 called for a ConvBERT head — the configuration the benchmark paper actually
used — and the harness plan silently dropped it. That left the E0 gap attributed to a
control that did not exist. It now exists (`finetune_esm/models/convbert.py`), placed as
the Ankh/ProtT5 lineage places it: a ConvBERT block over the residue sequence, followed
by pooling.

| Head | none | receptor removed | Δ | paired t(2) |
|---|---|---|---|---|
| MLP | 0.4284 ± 0.0029 | 0.4026 ± 0.0088 | +0.0258 | +5.20 |
| **ConvBERT** | **0.4215 ± 0.0020** | **0.4174 ± 0.0258** | **+0.0042** | **+0.30** |

Two results, and the second is the important one.

**ConvBERT does not close the gap to the published 0.47.** It scores 0.4215, slightly
*below* the MLP's 0.4284. So the head architecture is not what separates this harness
from the published number, and the E0 report's explanation for that gap is now
falsified rather than merely untested. The remaining 0.0485 is unattributed.

**Under the paper's own head, the binding partner contributes nothing measurable.**
Removing the receptor entirely costs 0.0042 Spearman, paired t = 0.30 — indistinguishable
from zero. On one seed the receptor-free model scored *higher* than any run with correct
pairing. The finding therefore holds against the paper's architecture, not only against a
simplification of it.

## Split protocol: the variable that actually matters

The same ablation on a cold-PDB split (whole PDB entries held out, but no
sequence-identity control; each seed builds a different partition, hence the wide spread):

| Split | Baseline | Receptor worth | paired t(2) |
|---|---|---|---|
| cold-PDB (no identity control) | 0.7258 ± 0.1092 | **0.1517** | +4.45 |
| authors' ≤30% identity | 0.4284 ± 0.0029 | **0.0258** | +5.20 |

**The binding partner is worth 5.9× more on the leakier split.** The natural reading is
that when homologous complexes appear on both sides of the split, receptor identity helps
retrieve the near-duplicate — a memorisation route, not interaction modelling. Imposing a
30% identity threshold closes that route and the contribution collapses.

Our cold-split Pearson (0.744) sits close to BALM-PPI's published cold-split figure of
0.730 on this dataset, which is a reasonable sign the splits behave as intended.

## What actually moves the number

| Intervention | Δ Spearman |
|---|---|
| **Changing the split protocol** (cold-PDB → ≤30% identity) | **−0.297** |
| Removing the binding partner, MLP head | −0.026 |
| Swapping all four fusion architectures (total spread) | 0.014 |
| Swapping the head (MLP → ConvBERT) | −0.007 |
| Removing the binding partner, ConvBERT head | −0.004 |

**The evaluation protocol dominates every architectural choice by an order of
magnitude.** The benchmark paper's contribution is a comparison among the interventions
in rows 3 and 4 — differences of 0.01 or less in this harness — while row 1, which the
paper holds fixed, is worth thirty times as much.

## Caveats

- One backbone (ESM2-650M), frozen. Three seeds. The random split is untested.
- Our ConvBERT is a faithful-in-placement reimplementation, not the paper's code; its
  layer count, width and kernel may differ from theirs.
- We reproduce at 0.4215–0.4284 against a published 0.47. The residual is now
  unexplained, and if it were specifically interaction signal these controls would
  understate the partner's importance.
- Controls corrupt training and evaluation together. The complementary experiment —
  train on correct pairs, substitute the partner only at inference — measures a trained
  model's reliance rather than task learnability, and is not answered here.
- `fixed_ligand` is one seed and should not be quoted precisely.

## Consequence

The design spec's goal was to beat the published ~0.50. That is now the less valuable
half of the work: optimising architecture to better predict a quantity that is 99%
recoverable without the binding partner — under the paper's own head — is optimising a
proxy.

The stronger contribution is a critique with a corrected protocol: on its own
leakage-controlled split this benchmark does not require modelling the protein–protein
interaction, architecture comparisons run on it are measuring something else, and the
split protocol is worth an order of magnitude more than any architectural choice
compared in the literature. This independently reproduces, on a different dataset and
under a stricter protocol, ProtAff's 2026 finding that a cross-attention affinity model
scored higher with a scrambled target than with the true one.
