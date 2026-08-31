# Decomposing the score: does the model rank complexes, or rank within them?

Run 2026-08-31, branch `research-controls`, ESM2-650M frozen + ConvBERT + HP, seed 0.
Per-row test predictions saved via `--save-predictions`.

## Why decompose

A global Spearman of 0.43 can come from two unrelated abilities:

- **between-complex** — ranking whole complexes against each other ("this antibody–antigen
  family binds tighter than that protease–inhibitor family")
- **within-complex** — ranking variants *inside* one complex ("this interface mutation
  helps, that one hurts")

Only the second requires anything interface-like. Within a complex the two partners are
the same proteins differing by a couple of residues, so family priors, source-dataset
priors and length/composition effects are all constant and cancel. The global metric
cannot tell the two apart.

Measurement power differs sharply by split. The authors' ≤30% identity clustering
scatters mutation series, leaving only 128 of 757 test rows in groups of ≥3. Cold-PDB
holds out whole PDB entries, so intact series land in test: **766 of 977 rows** in
evaluable groups. Cold-PDB is therefore the primary measurement here, with the authors'
split reported alongside despite its weakness.

## Result

| Run | global | between-complex | within-complex | groups | rows |
|---|---|---|---|---|---|
| cold-PDB, receptor present | 0.548 | **0.638** | **0.096** | 29 | 698 |
| cold-PDB, receptor removed | 0.651 | 0.611 | 0.067 | 28 | 694 |
| authors' split, receptor present | 0.422 | 0.374 | 0.080 | 8 | 32 |
| authors' split, receptor removed | 0.409 | 0.376 | 0.077 | 8 | 32 |

Within-complex Spearman is size-weighted across groups with ≥3 rows and ≥2 distinct
labels. Between/within is a descriptive split, not an exact variance partition — global
rank correlation over unequal group sizes is not their sum.

**The score is between-complex.** The model ranks whole complexes at 0.64 and ranks
variants inside a complex at 0.10. Removing the receptor entirely moves the within-complex
figure from 0.096 to 0.067, consistent with there being very little interaction signal
present to lose.

## Is there anything to rank? (the control on this control)

Low within-complex correlation would be uninformative if the labels inside a complex
barely varied. They do vary, but modestly:

- median group size 4; median within-group pKd **range 0.64**, max 7.03
- median within-group pKd **sd 0.28**, against a global test sd of **2.25**
- 23 of 48 groups carry ≥5 distinct labels

So a typical complex spans about **12% of the global spread internally**. Within-complex
ranking is a genuinely harder problem with a smaller dynamic range, and some of the 0.10
is attributable to that rather than to model failure. This is a real limit on how hard the
result can be pushed.

It is not, however, an impossible problem. Published structure-based ΔΔG methods report
*per-structure* Spearman of 0.40 (RDE-Network), 0.43 (Prompt-DDG) and 0.50 (Light-DDG) —
and within a complex, ranking absolute pKd is equivalent to ranking ΔΔG, since the
wild-type baseline is a constant offset. Those numbers come from different splits and from
direct ΔΔG supervision, so they are not a like-for-like ceiling. They do establish that
within-complex ranking at 0.4–0.5 is attainable, and that 0.10 reflects this model rather
than the task's floor.

## Reading

The 0.43 headline on this benchmark is very largely the ability to tell protein families
apart. That is a real and useful capability, but it is not binding-affinity modelling, and
it is exactly the capability a ligand-only model already has — which is why E1 found the
partner contributes ~0.004 under this same architecture.

Two consequences follow.

**For interpreting published results.** Any leaderboard on this benchmark that reports a
single global correlation is dominated by between-family variance. Two methods separated
by 0.02 global Spearman may be indistinguishable on the part of the task that involves an
interface. The fusion-architecture comparison the benchmark paper makes — a 0.014 spread
in this harness — sits well inside that ambiguity.

**For building something better.** Making within-complex ranking *the* objective rather
than 12% of the variance is what would force a model to use the interface. That is the
ΔΔG reframing: predict the change relative to a wild-type anchor, so family priors cancel
by construction instead of dominating. This decomposition is the evidence for taking that
route rather than continuing to optimise the global number.

## Caveats

- One seed, one backbone, one head. Cold-PDB split variance is large (E1 measured sd 0.109
  across seeds), so the global figures here should not be quoted precisely.
- Cold-PDB permits homology leakage. That inflates the between-complex figure, but affects
  within-complex much less: family priors are constant inside a group and cancel from the
  within-group correlation regardless of what leaked.
- On the authors' split only 32 rows are evaluable, so its within-complex figure of 0.08 is
  indicative at best.
- The cold-PDB receptor-removed run scored *higher* globally (0.651 vs 0.548) while scoring
  lower on both components. That is possible because the decomposition is descriptive
  rather than additive, and it is a single seed on a split with large variance. It should
  not be read as the receptor hurting.
