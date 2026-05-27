# Branch-Adaptive Quality-First Refinement for Hierarchical Science Maps

Status: internal research note
Audience: collaborators working on SciScape hierarchy construction
Date: 2026-05-04

## Korean Executive Summary

현재 후처리는 작은 cluster는 어느 정도 CPM quality 손실을 감수하고
합치고, 너무 큰 cluster는 local split으로 CPM quality를 키우거나 최소한
보존하는 방식이다. 다음 연구 단계의 핵심은 단순한 전역 size threshold가
아니라, **CPM이 암시하는 critical-gamma gap**으로 병합과 분할이
구조적으로 정당한지 판단하는 것이다.

핵심 수식은 두 community `A`, `B` 사이의 CPM critical resolution이다.

```math
\gamma^*_{A,B} = \frac{e_{AB}}{W_A W_B}
```

현재 global resolution `gamma`에서:

```math
g_{A,B} = \gamma - \gamma^*_{A,B}
```

가 크면 `A`와 `B`를 분리해 두는 것이 CPM상 강하게 지지되고, 작거나
음수이면 병합이 자연스럽다. 다만 이 값은 진짜 dendrogram height가
아니라 local critical-gamma evidence로 봐야 한다. CPM은 ultrametric을
보장하지 않고, `gamma` 변화에 따른 cluster lineage도 항상 단조적이지
않을 수 있다. oversized parent를 여러 child로 쪼갤 때도 같은 방식으로
`gamma*_split`과 normalized `Delta Q / W_between`을 정의할 수 있다.
따라서 Leiden은 여러 seed/gamma perturbation으로 후보 split을 만드는
장치로 사용하고, 최종 채택은 CPM normalized gap, exact delta Q,
continuous stability score, contraction pressure로 판단한다. Semantic
coherence는 main acceptance loop가 아니라 post-hoc explainability check로
둔다.

이 아이디어의 논문 framing은 다음과 같다.

> Scientific fields have locally varying density, so global hierarchy cuts or
> global size thresholds are insufficient for interpretable science maps. A
> branch-adaptive CPM postprocess uses a critical-gamma map and normalized
> quality gaps to decide when local merges and splits are structurally
> justified.

## One-Line Summary

The current hierarchy postprocess repairs clusters by merging clusters that are
too small and splitting clusters that are too large. The next step is to replace
purely global size thresholds with a branch-adaptive rule: use CPM-implied
critical-resolution gaps to decide which local merges or splits are
structurally meaningful.

Important caveat: CPM critical resolutions should not be treated as a true
dendrogram. CPM does not imply an ultrametric hierarchy, and cluster identities
need not evolve monotonically as `gamma` changes. The safer framing is a
`critical-gamma map`: local split/merge candidates are assigned CPM-neutral
resolutions and normalized quality gaps, then compared as branch-level evidence.

## Design Constraints After Methodological Review

The revised protocol explicitly adopts the following constraints:

- Do not frame `gamma*` as a dendrogram height; use "critical-gamma map" or
  "local critical-gamma evidence."
- Use normalized split gain `Delta Q_split / W_between` as the primary
  candidate comparison metric, especially when candidate child count `k`
  varies.
- Report a `tau_split` sensitivity curve rather than a single tuned threshold.
- Use continuous stability scores, such as mean pairwise AMI, rather than a
  binary "same split" rule with nested thresholds. In the first pilot,
  stability is a ranking/tie-breaker and reported diagnostic, not a hard
  acceptance threshold.
- Use no-quality-regression as the default quality policy:
  `epsilon_Q = 0`, so accepted split sets must have non-negative exact CPM
  delta under the original graph evaluator.
- Keep semantic coherence outside the primary acceptance loop; use it for
  post-hoc explanation of accepted, rejected, and unchanged cases.
- Use the configured target cluster weight as the pilot receiver scale, rather
  than global `W_max`; local quantile or median-based scales are later
  sensitivity checks.
- Generate candidates on induced local subgraphs, but evaluate acceptance on
  the original graph to preserve exact CPM accounting with cross-boundary
  edges.
- Keep MCMC adaptive cut as related work. The pilot uses deterministic branch
  selection for reproducibility.

## Motivation

Hierarchical science maps are not only flat partitions. They are inputs to
contraction, visualization, topic interpretation, and downstream field
delineation. A partition that is acceptable by flat CPM quality can still be
problematic as a hierarchy input if it contains:

- many tiny clusters that are not useful as map units;
- one or a few oversized clusters that dominate the contracted next level;
- field-dependent density regimes where a single global threshold over-merges
  dense branches and over-splits sparse branches.

The current postprocess already addresses the first two issues:

```text
raw Leiden
  -> small-cluster repair
  -> oversize split repair
  -> optional hard-cap diagnostic
  -> quality/fallback acceptance
  -> contraction + next-level validation
```

Empirically, `quality_first` is the defensible default:

- it preserves or improves exact CPM quality in the observed source runs;
- it lowers source max/target ratio in many source-seed cases;
- it reduces downstream concentration pressure on average;
- strict `hard_cap` often falls back, so it is diagnostic rather than a robust
  default.

However, this still leaves a conceptual gap. We need a principled rule for when
a merge or split should be considered a true structural decision rather than a
size-driven repair. The proposed answer is to define local critical-gamma gaps
using CPM critical resolutions.

## Relation To Adaptive Cut

Boucherie, Ahn, and Lehmann's "Adaptive cut reveals multiscale complexity in
networks" argues that cutting a dendrogram at a single level can be suboptimal
when the dendrogram is unbalanced. Their adaptive cut uses multi-level cuts and
an objective function to exploit branch-specific structure rather than one
global threshold.

Our setting differs in two ways:

- Leiden does not naturally expose a clean binary dendrogram for the final
  partition.
- We have a specific objective, CPM, and we care about exact CPM quality,
  source-level size constraints, and next-level contraction pressure.

The analogous strategy is:

```text
Use Leiden and local gamma/seed perturbations to generate candidate branch
splits or merges.

Use CPM critical resolution gaps to decide whether those branch decisions are
structurally justified.
```

In short:

```text
Leiden = candidate generator
CPM gap = structural acceptance signal
```

This is only an analogy to adaptive dendrogram cuts. We should avoid claiming
that CPM critical gamma values form a dendrogram height function. Pairwise
critical gamma values need not satisfy ultrametric constraints, and partitions
obtained at different gamma values need not form a nested sequence.

A minimal counterexample for the ultrametric issue is a three-node graph with
unit node weights and edge weights:

```text
e_12 = 0.30, e_23 = 0.40, e_13 = 0.35
```

Then:

```text
gamma*_12 = 0.30
gamma*_23 = 0.40
gamma*_13 = 0.35
```

For an ultrametric distance-like height, two of the three largest values would
need to tie. Here they do not, so CPM critical gamma values should be treated as
pairwise or candidate-level evidence, not as a guaranteed tree metric.

## CPM Objective And Critical Resolution

Assume an undirected weighted graph with node/document weights `w_i`. For a
community `C`, let:

```text
W_C = sum_{i in C} w_i
e_C = total internal edge weight inside C
```

Use the CPM-style objective:

```math
Q_\gamma(P) = \sum_{C \in P} \left(e_C - \gamma \binom{W_C}{2}\right)
```

The exact constant depends on the implementation convention for counting
undirected edges and self terms. The merge/split signs below are the important
part; they should be implemented using the same convention as the exact CPM
evaluator.

For implementation, compute these quantities on the original graph or on a
contracted graph with the following mapping:

- a super-node self-loop stores the internal edge weight of the corresponding
  original community;
- the edge between super-nodes `A` and `B` stores original cross-community edge
  weight `e_AB`;
- node weight `W_A` is the document/node weight represented by super-node `A`.

Example: if `W_A = 40`, `W_B = 20`, `e_AB = 0.9`, and `gamma = 0.001`, then:

```math
\gamma^*_{A,B} = \frac{0.9}{40 \cdot 20} = 0.001125
```

Because `gamma < gamma*`, merging is CPM-positive:

```math
\Delta Q_{\text{merge}} = 0.9 - 0.001 \cdot 800 = 0.1
```

If `gamma = 0.002`, the same merge is CPM-negative:

```math
\Delta Q_{\text{merge}} = 0.9 - 0.002 \cdot 800 = -0.7
```

### Merge Gain

For two communities `A` and `B`, let `e_AB` be the total edge weight between
them. The CPM gain from merging them is:

```math
\Delta Q_{\text{merge}}(A,B)
  = e_{AB} - \gamma W_A W_B
```

The critical resolution at which merge is exactly neutral is:

```math
\gamma^*_{A,B}
  = \frac{e_{AB}}{W_A W_B}
```

Interpretation:

```text
if gamma < gamma*_AB:
    merging A and B increases CPM quality

if gamma > gamma*_AB:
    keeping A and B split increases CPM quality
```

So the local CPM gap is:

```math
g_{A,B} = \gamma - \gamma^*_{A,B}
```

Large positive `g_AB` means the current resolution strongly supports keeping
the two branches separate. Large negative `g_AB` means the current resolution
supports merging them.

### Multiway Split Gain

Suppose an oversized parent cluster `P` is proposed to be split into children:

```text
P -> {C_1, C_2, ..., C_k}
```

Define:

```math
E_{\text{between}}
  = \sum_{i<j} e_{C_i C_j}
```

```math
W_{\text{between}}
  = \sum_{i<j} W_{C_i} W_{C_j}
```

The critical resolution of this split is:

```math
\gamma^*_{\text{split}}
  = \frac{E_{\text{between}}}{W_{\text{between}}}
```

The split gain at the global baseline resolution is:

```math
\Delta Q_{\text{split}}
  = \gamma W_{\text{between}} - E_{\text{between}}
  = (\gamma - \gamma^*_{\text{split}}) W_{\text{between}}
```

The raw split gap is:

```math
g_{\text{split}}
  = \gamma - \gamma^*_{\text{split}}
```

For comparing candidates with different numbers of children `k`, the main
comparison metric should be the scale-normalized split gain:

```math
\tilde{g}_{\text{split}}
  = \frac{\Delta Q_{\text{split}}}{W_{\text{between}}}
```

Interpretation:

- `tilde_g_split > 0`: the proposed child separation is CPM-positive at the current
  resolution.
- `tilde_g_split ~= 0`: the split is weak or borderline; require stability or
  holdout evidence.
- `tilde_g_split < 0`: the split is CPM-negative; only accept under an explicit
  constraint policy, not under `quality_first`.

This normalized value should be the primary reporting axis. The unnormalized
`\Delta Q_split` is still needed for exact quality accounting, but it is not a
fair standalone comparison when candidate splits have different `k`.

### Merge Regret For Small Clusters

Small-cluster repair intentionally accepts some quality loss to avoid unusable
tiny map units. For a small cluster `S` and receiver `R`:

```math
\Delta Q_{\text{merge}}(S,R)
  = e_{SR} - \gamma W_S W_R
```

If this is negative, define merge regret:

```math
\text{regret}(S,R) = -\Delta Q_{\text{merge}}(S,R)
```

and normalized regret:

```math
\tilde{r}(S,R)
  = \frac{-\Delta Q_{\text{merge}}(S,R)}{W_S W_R}
```

Interpretation:

- low regret: the small cluster is weakly separated and can be merged safely;
- high regret: the small cluster may be a real small topic and should not be
  forcibly merged without a stronger operational reason.

This gives us a CPM-grounded way to distinguish "small noise" from "small but
structural topic."

## Critical-Gamma Map, Not A True Dendrogram

For a true dendrogram, a branch height is usually a distance, similarity, or
merge score with nested structure. CPM critical resolutions do not guarantee
that structure. A safer object is a critical-gamma map:

```math
m(A,B) = \gamma^*_{A,B}
```

For a parent split:

```math
m(P -> C_1,\ldots,C_k) = \gamma^*_{\text{split}}
```

The structural evidence around a candidate is then the distance from the active
resolution, preferably normalized:

```math
\text{raw active gap} = \gamma - m
```

Across a local gamma sweep, avoid a separate "close variant" threshold. Use
continuous AMI across gamma slices rather than claiming true dendrogram
persistence:

```math
S_{\gamma\text{-AMI}}(P)
  = \frac{2}{G(G-1)}
    \sum_{1 \leq i < j \leq G}
      \operatorname{AMI}(P_{\gamma_i}, P_{\gamma_j})
```

The important point is that the value is not an arbitrary linkage distance and
not an ultrametric height. It is the CPM resolution at which the local
merge/split candidate becomes neutral.

## Why This Matters For Science Maps

Scientific fields have uneven local density:

- dense biomedical, AI, or materials regions may contain many separable
  subtopics even when a global resolution keeps them together;
- sparse or interdisciplinary regions may look fragmented under the same
  global rule;
- a single global size cap or cut threshold cannot handle both regimes.

A branch-adaptive method is therefore not merely an engineering tweak. It is a
methodological response to heterogeneous topic density in science maps.

## Proposed Policy: `branch_adaptive_quality_first`

The policy extends the current `quality_first` rule.

### Inputs

- baseline global resolution `gamma`;
- raw or small-repaired membership;
- candidate oversized clusters;
- local graph induced by each candidate cluster;
- local Leiden perturbations over seed and gamma multipliers;
- exact CPM evaluator;
- optional semantic/coherence diagnostics for post-hoc explanation, not main
  acceptance.

### Candidate Generation

For each oversized source cluster `P`:

```text
for alpha in gamma_multipliers:
    gamma_local = alpha * gamma
    for seed in local_seeds:
        run Leiden on induced subgraph P at gamma_local
        project candidate children back to global node ids
        repair tiny local fragments if needed
        record candidate split
```

Candidate generation deliberately uses the induced subgraph of `P` to make
local perturbation cheap and focused. Acceptance evaluation must then be run on
the original graph, not only the induced graph, so that cross-boundary edges and
the global CPM convention are included in exact quality accounting.

Suggested first sweep:

```text
alpha in {1.02, 1.05, 1.10, 1.15, 1.20, 1.25}
seed in {11, 42, 73}
```

Use a hierarchical compute budget in the first implementation:

```text
stage 1: alpha in {1.05, 1.15, 1.25}, seeds {11, 42, 73}
stage 2: expand to all six alpha values only for parents with promising
         normalized split gain or contraction improvement
```

This prevents the pilot from becoming `n_oversized_parents * 18` local Leiden
runs by default.

### Split Acceptance

Accept a split candidate only if:

```text
exact Delta Q_split >= 0 under the original graph evaluator
normalized split gain is on the reported tau curve
child size distribution is not pathological
next-level contraction pressure improves or is neutral
```

In math form:

```math
\Delta Q_{\text{split}} \geq \epsilon_Q
```

Default:

```math
\epsilon_Q = 0
```

This fixes the pilot as a no-quality-regression policy. A negative quality
floor can be explored later, but it should not be the default paper claim.

```math
\tilde{g}_{\text{split}}
  = \frac{\Delta Q_{\text{split}}}{W_{\text{between}}}
  \geq \tau_{\text{split}}
```

For paper-quality reporting, do not use a single tuned `tau_split`. Report a
sensitivity curve:

```text
tau_split / gamma in {0.0, 0.001, 0.005, 0.01, 0.05}
```

or the equivalent absolute normalized-gain values. The pilot may choose one
operational default after seeing the curve, but the paper should show that the
method lies on a reasonable trade-off frontier rather than only at a
hand-picked threshold.

Semantic coherence should not be part of the primary acceptance rule. Use it
post hoc to compare accepted, rejected, and unchanged candidates.

Continuous stability scores are also diagnostics and ranking/tie-breakers in
the first pilot rather than hard acceptance thresholds. This avoids adding a
second threshold grid before the split-gain protocol is validated.

### Merge Acceptance

For small clusters, do not merge solely because they are below a minimum size.
Prefer receiver `R` that minimizes normalized regret:

```math
R^*(S)
  = \arg\min_R \tilde{r}(S,R)
```

Accept the merge if:

```math
\tilde{r}(S,R^*) \leq \tau_{\text{merge}}
```

and the receiver does not become an obvious oversized outlier:

```math
W_{R^*} + W_S \leq \tau_{\text{receiver}} \cdot W_{\text{target}}
```

For the pilot, `W_target` is the configured target cluster weight already used
by the hierarchy postprocess, e.g. `target_max_doc_weight`. Local quantiles or
median-based scales can be tested later, but the first implementation should
choose one fixed scale. Do not use global `W_max`; a Pareto-like size
distribution can make it too permissive.

If no receiver satisfies the regret threshold, mark the cluster as:

```text
protected_small_structural_topic
```

rather than blindly merging it.

## Branch-Stability Diagnostics

Because Leiden is stochastic and does not expose a canonical dendrogram, branch
stability should be estimated from repeated local perturbations.

Candidate metrics:

- continuous pairwise adjusted mutual information among local candidate
  partitions;
- continuous average child-overlap or best-match Jaccard across candidates;
- gamma-axis mean AMI, avoiding strong persistence claims;
- rank stability of the largest child branches;
- exact CPM gap stability.

A simple first metric should avoid a nested "same split" threshold. Use the
mean pairwise AMI directly:

```math
S_{\text{AMI}}(P)
  = \frac{2}{M(M-1)}
    \sum_{1 \leq i < j \leq M}
      \operatorname{AMI}(P_i, P_j)
```

For candidates with different `k`, report stability by `k` as well as pooled
stability. This makes two-way and many-way candidate regimes visible instead of
forcing them into one binary notion of sameness.

In the first policy implementation, use these stability values only for
candidate ranking and reporting:

```text
primary gate: exact Delta Q >= 0 and normalized split-gain threshold curve
ranking/tie-breaker: S_AMI, contraction improvement, child-size diagnostics
```

## Contraction-Aware Objective

For hierarchical science maps, source-level quality is not sufficient. A split
should be evaluated by its effect on the contracted graph that feeds the next
level.

Let `B(P)` be a contraction pressure score. Candidate components:

```text
max contracted node weight / target
contracted node weight Gini
parent max-child share
next-level oversize count
next-level cluster-weight entropy
```

A combined acceptance score can be:

```math
F(P)
  = \Delta Q_{\text{source}}
    - \lambda_{\text{size}} \Delta \text{OversizePenalty}
    - \lambda_{\text{conc}} \Delta \text{ConcentrationPenalty}
    - \lambda_{\text{frag}} \Delta \text{FragmentPenalty}
```

For `quality_first`, the safer policy is lexicographic rather than fully
weighted:

```text
1. reject if exact CPM quality is negative under the original graph evaluator;
2. reject if normalized split gain is below the reported tau curve point;
3. prefer candidates that reduce source max/target;
4. break ties using next-level concentration;
5. break remaining ties using continuous stability diagnostics;
6. report semantic coherence post hoc, not as an acceptance criterion.
```

This preserves the current philosophical position: CPM remains the source of
truth, and size/contraction metrics guide proposal choice rather than rewriting
the objective.

## Expected Failure Modes

The branch-adaptive policy should make failure modes more interpretable.

| Failure mode | Numerical criterion | Meaning | Action |
|---|---|---|---|
| `weak_cpm_gap` | `tilde_g_split < tau_split` on the reported tau curve | candidate split is not structurally supported by CPM | keep parent intact |
| `unstable_branch` | low `S_AMI` or low gamma-axis AMI relative to other candidates | split appears only under specific seeds/gamma | down-rank and report |
| `semantic_core_cluster` | strong text coherence and weak normalized split gain | oversized cluster is broad but coherent | keep or label as broad topic |
| `receiver_regret_too_high` | `min_R tilde_r(S,R) > tau_merge` | small cluster is costly to merge | protect as small structural topic |
| `contraction_neutral` | source split accepted but `Delta` next-level concentration is near zero or worse | source split helps locally but not hierarchy | accept only if source benefit is strong |
| `fragmentation_risk` | child-size entropy or below-min child count exceeds threshold | split creates too many tiny children | reject or repair children |

## Minimal Pilot Experiment

### Goal

Compare the existing `quality_first` postprocess with
`branch_adaptive_quality_first`.

### Samples

Use two diagnostic fields first:

- field30: positive downstream case where quality_first already improves
  next-level Gini and parent concentration;
- field26: mixed case where max ratio improves but next-level concentration
  does not clearly improve.

Then expand to all six validation fields.

### Threshold And Holdout Protocol

The pilot should not report one hand-picked `tau_split`. Use a threshold curve:

```text
tau_split / gamma in {0.0, 0.001, 0.005, 0.01, 0.05}
```

Recommended design:

- use field30 and field26 as pilot fields for implementation and qualitative
  debugging;
- freeze the candidate scoring formula and the reported threshold grid before
  running all six fields;
- treat the all-six-field run as the main evidence rather than re-tuning
  thresholds per field;
- report paired deltas for every `tau_split`, not only the chosen operating
  point.
- keep `epsilon_Q = 0` fixed across the pilot and main six-field run unless a
  separate negative-quality-budget experiment is explicitly introduced.
- use stability diagnostics for ranking and reporting, not as a separate hard
  threshold.

If an operational default is needed, select it from the trade-off curve using a
predefined rule such as:

```text
smallest tau_split whose mean source exact delta Q is non-negative and whose
mean source max/target is no worse than current quality_first
```

This reduces cherry-picking risk.

### Candidate Conflict Resolution

Local candidates may overlap. The first implementation should avoid an ILP and
use a deterministic greedy resolver:

```text
1. sort candidates by:
   normalized split gain,
   exact Delta Q,
   source max/target reduction,
   contraction improvement,
   fewer pathological children
2. accept a candidate if none of its parent nodes are already claimed;
3. re-evaluate exact CPM delta after applying the selected non-overlapping set;
4. fallback if the final exact quality floor is violated.
```

The final exact quality floor is non-negative by default:

```text
final exact Delta Q >= 0
```

An exact ILP can be future work if conflicts become common, but deterministic
greedy selection is easier to audit and reproduce.

### Seed Budget

Use paired source seeds and next-level seeds:

```text
source seeds: 11, 42, 73
next-level seeds: 11, 42, 73
```

For final reporting, prefer paired comparisons within the same field/source
seed/next seed rather than unpaired averages. If runtime allows, extend source
seeds to 5 or 10 only after the two-field pilot validates compute cost.

### Power And Generality

Six fields are enough for a methods pilot but not enough for a broad field-level
generality claim. Report:

- paired row counts;
- field-level breakdown;
- sign counts by metric;
- bootstrap confidence intervals over paired deltas if needed;
- explicit field heterogeneity.

The paper should claim "observed six-field evidence" unless another expansion
round adds fields.

### Experimental Conditions

```text
baseline: small_only
current: two_stage_quality_first
diagnostic: two_stage_hard_cap
new: branch_adaptive_quality_first
```

### Metrics

Source-level:

- exact CPM delta Q;
- max doc weight / target;
- oversize count;
- cluster weight Gini;
- number of protected small clusters;
- number of branch-adaptive accepted splits.

Branch-level:

- `gamma*_split`;
- raw `split_gap`;
- normalized split gain `Delta Q_split / W_between` as the primary candidate
  comparison metric;
- branch stability;
- candidate child count `k`;
- child size entropy;
- semantic coherence delta as post-hoc explanation.

Next-level:

- max/target ratio;
- oversize count;
- Gini;
- parent max-child share;
- next-level seed stability.

Semantic sanity:

- title/abstract TF-IDF doc-centroid coherence;
- top-term overlap or representative titles for selected examples.

### Success Criteria

The new policy is useful if it:

- keeps exact CPM delta non-negative or above a configured quality floor;
- improves or matches `quality_first` on source max/target ratio;
- improves next-level concentration in mixed cases such as field26;
- avoids unnecessary splits in no-op cases such as field18;
- explains failures using normalized critical-gamma evidence and stability.

## Paper Framing

The strongest Scientometrics framing is:

> Scientific fields have locally varying density, so global hierarchy cuts or
> global size thresholds are insufficient for interpretable science maps. We
> propose a branch-adaptive CPM postprocess that uses a critical-gamma map and
> normalized quality gaps to decide when local merges and splits are
> structurally justified, preserving exact CPM quality while reducing
> contraction imbalance.

This is stronger than saying:

> We add heuristics to fix cluster sizes.

It positions the method as a CPM-grounded adaptive refinement strategy for
hierarchical science maps. Avoid saying that CPM produces a true adaptive
dendrogram cut.

MCMC-based adaptive cut should remain related work for this manuscript. The
proposed contribution is deterministic branch selection using CPM
critical-gamma evidence, which is easier to reproduce and audit on large
science-map graphs.

## Claims To Avoid

- Do not claim guaranteed balanced clusters.
- Do not claim all next-level metrics improve in every field.
- Do not claim semantic improvement from TF-IDF sanity checks alone.
- Do not treat Leiden move order as a true dendrogram.
- Do not claim CPM critical gamma values form an ultrametric hierarchy.
- Do not accept branch splits solely because they reduce size.
- Do not report only one tuned `tau_split`; show sensitivity curves.
- Do not introduce MCMC into the main method unless a later project explicitly
  targets stochastic adaptive-cut optimization.

## Implementation Sketch

```python
def branch_adaptive_quality_first(graph, membership, gamma, targets):
    membership = small_cluster_repair_with_regret(
        graph=graph,
        membership=membership,
        gamma=gamma,
        min_weight=targets.min_weight,
        max_receiver_weight=targets.receiver_cap,
    )

    oversized = find_oversized_clusters(membership, targets.max_weight)
    candidates = []

    for parent in oversized:
        local_graph = induced_subgraph(graph, parent)
        # Candidate generation uses the induced graph for local perturbation.
        local_candidates = local_gamma_seed_sweep(
            local_graph=local_graph,
            gamma=gamma,
            multipliers=[1.05, 1.15, 1.25],
            seeds=[11, 42, 73],
        )

        for split in local_candidates:
            # Acceptance is evaluated on the original graph so cross-boundary
            # edges and exact CPM accounting are preserved.
            split_stats = cpm_split_gap(
                graph=graph,
                parent=parent,
                children=split.children,
                gamma=gamma,
            )
            stability = mean_pairwise_ami(split, local_candidates)
            if accept_branch_split(split_stats, stability, targets):
                candidates.append(split)

    proposed = greedy_non_overlapping_splits(membership, candidates)
    proposed = repair_fragmentation_if_needed(proposed)

    if exact_cpm_delta(graph, membership, proposed, gamma) >= 0.0:
        return proposed, "accepted"
    return membership, "fallback"
```

## Open Questions

1. What default operating point should be chosen from the `tau_split`
   sensitivity curve?
2. Should `tau_split` be global, field-adaptive, or parent-adaptive after the
   first fixed-grid evaluation?
3. Should merge regret use only CPM loss, or also post-hoc semantic diagnostics
   for explanation?
4. Is mean pairwise AMI sufficient for stability ranking, or do we need child-overlap
   summaries by candidate `k`?
5. Does branch-adaptive refinement improve field26-like mixed cases, or merely
   explain why they should remain unchanged?
6. If the no-quality-regression default is too conservative, should a separate
   negative-quality-budget appendix experiment be added?

## Recommended Next Step

Implement a pilot `branch_adaptive_quality_first` runner for field30 and
field26 only. Save branch-level diagnostics before running the full hierarchy:

```text
branch_adaptive_split_candidates.csv
branch_adaptive_merge_regrets.csv
branch_adaptive_policy_effects.csv
branch_adaptive_next_level_effects.csv
```

The immediate goal is not to beat every metric. The goal is to show that
normalized critical-gamma evidence explains when splitting, merging, or
preserving a cluster is the right structural choice.

## References

- Louis Boucherie, Yong-Yeol Ahn, and Sune Lehmann. "Adaptive cut reveals
  multiscale complexity in networks." arXiv:2512.08741, 2025.
- Current evidence files:
  `research/consensus/results/adaptive_refinement/hierarchy_postprocess_validation/`.
