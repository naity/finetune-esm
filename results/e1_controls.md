# E1 — What does this benchmark actually measure?

Run 2026-08-30 on a single RTX 3090, branch `research-controls`, against the same
ESM2-650M embedding cache and the same authors' (≤30% sequence identity) split used
for the E0 reproduction.

## Method

Each control rewrites the dataset so that one specific source of information is
destroyed, then the model is **trained and evaluated** under that corruption. That
phrasing matters: the question is not "does a model trained on correct pairs rely on
the receptor?" but "is this task learnable without the receptor at all?" — a claim
about the benchmark rather than about one trained model.

Controls are applied after the split is fixed and independently within each split, so
the partition, its zero-PDB-leakage guarantee, the row count, the PDB grouping and the
label multiset are all identical to the uncontrolled run. Scrambling uses a derangement,
so no complex accidentally keeps its true partner.

Configuration throughout: ESM2-650M frozen, hierarchical-pooling fusion, MLP head,
MSE loss, 30-epoch cap with early stopping on validation Spearman.

## Results

| Control | Spearman (mean ± sd, 3 seeds) | vs. baseline | % of baseline |
|---|---|---|---|
| **none** (correct pairing) | **0.4284 ± 0.0029** | — | 100.0% |
| scramble receptor (wrong partner) | 0.4129 ± 0.0082 | −0.0156 | 96.4% |
| **fixed receptor** (receptor carries no information) | **0.4026 ± 0.0088** | **−0.0258** | **94.0%** |
| fixed ligand (ligand carries no information) † | 0.3053 | −0.123 | 71.3% |
| shuffle labels | −0.0527 | −0.481 | — |

† one seed only.

For scale, on the same harness and split:

- spread across **all four** fusion architectures (EC/SC/HP/PAD): **0.0139**
- trivial-feature floor, no PLM at all (length, chain count, composition): **0.229**
- label-shuffle floor: **−0.053**

## Reading

**Replacing every receptor with a single constant sequence costs 0.026 Spearman.**
The model keeps 94% of its score when the binding partner is stripped of all
per-example information. Scrambling the partner instead of removing it costs about the
same (0.016), which is the consistency one would want: destroying the pairing and
destroying the partner do comparable, small damage.

So a sequence-based model on this benchmark is close to a **ligand-only predictor**.
The entire contribution of knowing which protein the ligand actually binds is 0.026,
which is 3.9 pooled standard deviations — real, but roughly twice the spread between
four different fusion architectures and about a tenth of the distance from the trivial
floor to the full score.

That explains the fusion diagnostic recorded in `e0_reproduction.md`: EC, SC, HP and
PAD land within 0.014 of one another because there is very little interaction signal
for different fusion strategies to exploit differently. The benchmark paper's central
claim — that hierarchical pooling and pooled-attention-addition beat naive
concatenation by up to 12% Spearman — is a claim about modelling an interaction that,
on this split, contributes about 6% of the score.

**Two supporting readings.** The label shuffle collapses to −0.05, so the evaluation
protocol is sound and nothing is leaking. And the receptor-only run still reaches 0.305,
above the 0.229 trivial floor, so both partners carry independent signal — what is
missing is the *interaction* between them, not the proteins themselves.

## Caveats

- Measured on this harness, which reproduces at 0.4284 against the published 0.47. If
  the missing 0.04 were specifically interaction signal that a ConvBERT head captures
  and ours does not, the picture could differ. The fusion diagnostic makes that
  unlikely — swapping the entire fusion architecture moves the number by 0.014 — but it
  has not been tested, because the ConvBERT head that spec §5.3 called for was never built.
- One fusion (HP), one split (the authors'), three seeds. The cold-PDB and random splits
  are untested here.
- These controls corrupt training and evaluation together. The complementary experiment —
  train on correct pairs, substitute the partner only at inference — measures how much a
  *trained model* relies on the receptor, and would likely show a larger drop. That is a
  different question and is not answered here.
- `fixed_ligand` is a single seed and should not be quoted with precision.

## Consequence for the project

The goal set in the design spec was to beat the published ~0.50 Spearman. That target now
looks like the less valuable half of the work. Optimising fusion architectures, losses and
backbones to better predict a quantity that is 94% recoverable without the binding partner
is optimising a proxy.

The stronger contribution is the critique with a corrected protocol: this benchmark, on its
own leakage-controlled split, does not require modelling the protein–protein interaction,
and any architecture comparison run on it is measuring something else. That is directly
supported by the independent finding (ProtAff, 2026) that a cross-attention affinity model
scored *higher* with a scrambled target than with the true one — reproduced here on a
different dataset under a stricter protocol.
