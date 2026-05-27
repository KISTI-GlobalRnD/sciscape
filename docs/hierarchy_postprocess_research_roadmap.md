# Hierarchy Postprocess Research Roadmap

Status: draft research prioritization
Scope: CPM Leiden hierarchy construction, size-aware postprocess, and contraction-aware validation

## Core Framing

The strongest paper framing is not that SciScape adds another postprocess
heuristic. The stronger claim is:

> CPM Leiden is a strong flat partition optimizer, but scientific landscape
> construction adds hierarchy-specific operational constraints: document-weight
> balance, stable contraction, and interpretability across levels. We propose
> quality-preserving repair layers that address these constraints without
> modifying the CPM objective.

The current implementation already provides the first layer:

```text
raw Leiden
  -> small-cluster repair
  -> oversize split-repair probes
  -> oversize boundary trim
  -> quality/hard-cap acceptance
  -> projection + contraction
```

The next research question is which additions make this a publishable
methodology rather than a useful engineering feature.

## Priority Summary

| Priority | Idea | Paper value | Implementation cost | Main risk |
|---|---|---:|---:|---|
| P0 | Local resolution probing | High | Medium | Needs strong ablations |
| P0 | Contraction-aware evaluation | High | Medium | Proxy metric must be defensible |
| P0 | Oversize failure taxonomy | High | Low | Must avoid overclaiming |
| P1 | Quality-preserving size constraints | Medium-high | Low | Mostly framing, not new code |
| P1 | Hierarchical consistency metrics | Medium-high | Medium | Metric selection can sprawl |
| P1 | Stability across seeds/gamma multipliers | Medium | Medium-high | Runtime cost |
| P2 | Semantic validation layer | Medium-high | Medium | LLM/expert evaluation burden |
| P2 | Adaptive policy selection | Medium | Medium | Could look too heuristic |
| P3 | New Rust fused kernel | Low-medium | High | Engineering, not paper core |

## P0. Local Resolution Probing

### Claim

Oversized clusters are tested by applying higher local CPM resolutions only
inside the problematic source cluster, then repairing back at the global
baseline resolution.

This can be stated as:

```math
\gamma_{\text{local}} = \alpha \gamma,\quad \alpha > 1
```

The global hierarchy keeps one level resolution, but oversized clusters receive
a local stress test. A split is considered meaningful only if it survives repair
at the baseline resolution.

### Why It Matters

Global gamma is a known practical limitation for heterogeneous paper networks:
one density regime may be over-merged while another is over-split. Local probing
addresses this without abandoning CPM or changing the global objective.

### Required Experiments

- Compare raw Leiden, small repair, and local probing on the same levels.
- Vary gamma multipliers: `1.02, 1.05, 1.10, 1.15, 1.20, 1.25`.
- Report how often oversized clusters produce retained non-restored splits.
- Show exact CPM delta before and after repair.
- Compare against simply rerunning the whole graph at higher gamma.

### Success Criteria

- Local probing reduces max document weight or oversize excess.
- Exact CPM quality delta is non-negative or bounded by configured floor.
- It avoids the over-fragmentation seen in global high-gamma reruns.
- The largest escaped fragments are semantically interpretable.

### Paper Position

This is the strongest method contribution. It can be described as
`local-resolution counterfactual testing for hierarchical CPM clustering`.

## P0. Contraction-Aware Evaluation

### Claim

In a hierarchy, current-level quality is not enough. A postprocess should also
be evaluated by its effect on the contracted graph that feeds the next level.

Current acceptance mostly checks:

```math
\Delta Q_\ell \geq \epsilon.
```

A contraction-aware extension would additionally measure:

```math
\Delta B_{\ell+1} > 0,
```

where `B` is a next-level balance or stability proxy.

### Candidate Metrics

- contracted node weight Gini
- max contracted node weight / total weight
- entropy of contracted node weights
- degree concentration of contracted graph
- top-1 / top-10 contracted node mass
- next-level cluster count stability
- next-level max cluster weight

### Required Experiments

- Run hierarchy with and without oversize repair.
- Compare the next-level graph before Leiden:
  - number of super-nodes
  - max super-node weight
  - super-node weight Gini
  - edge concentration around top super-nodes
- Compare the next-level partition:
  - max doc weight
  - cluster count
  - size distribution
  - CPM quality

### Success Criteria

- Proposed repair improves contracted graph balance.
- Improvements persist into the next level partition.
- Current-level CPM quality remains acceptable.

### Paper Position

This turns the feature from "flat postprocess" into a hierarchy methodology.
It is the most important conceptual addition after local resolution probing.

## P0. Oversize Failure Taxonomy

### Claim

Not every oversized cluster should be forcibly split. Some are structurally
inseparable under CPM quality constraints. The method should classify failure
modes rather than silently accepting or rejecting proposals.

### Current / Proposed Failure Modes

- `no_current_oversize_candidates`
- `no_selected_candidates`
- `no_candidate_boundary_moves`
- `quality_floor`
- `move_budget`
- `trim_delta_bound_or_receiver_cap`
- `target_satisfied`
- `restored_source_cluster`
- `no_retained_source_unit`
- `source_not_above_target_max`
- `hard_cap_not_satisfied`

### Required Experiments

- Aggregate failure modes by field and hierarchy level.
- Show whether failures concentrate in specific graph regimes:
  - high internal density
  - low external degree
  - high conductance boundary
  - sparse citation coverage
- Inspect examples from each failure mode.

### Success Criteria

- The taxonomy explains why some large clusters remain large.
- It distinguishes feasible oversize clusters from genuinely broad topics.
- It supports honest reporting for `hard_cap` failures.

### Paper Position

This is low-cost and high-value. It makes the method auditable and protects
against the criticism that the algorithm blindly forces arbitrary splits.

## P1. Quality-Preserving Size Constraints

### Claim

Size constraints should be handled as proposal generation plus CPM quality
filtering, not by rewriting the CPM objective.

The method keeps CPM as the source of truth:

```text
generate size-improving candidate
evaluate exact CPM delta
accept only under quality policy
fallback otherwise
```

### Why It Matters

This is theoretically safer than adding ad hoc size penalties into CPM. It
keeps the resolution-limit-free motivation of CPM intact while still giving
operators practical size diagnostics.

### Required Experiments

- Compare `quality_first` and `hard_cap`.
- Report how much hard-cap mode loses in exact CPM quality.
- Show cases where hard cap is infeasible without unacceptable quality loss.

### Success Criteria

- `quality_first` improves balance with minimal quality loss.
- `hard_cap` is useful diagnostically but not necessary as default.
- The paper can justify not forcing all max-size constraints.

### Paper Position

This is mostly framing plus summary tables. It should be part of the main
method section, but it is not enough by itself.

## P1. Hierarchical Consistency Metrics

### Claim

Because Leiden does not produce a proper dendrogram, an induced Leiden hierarchy
must be evaluated by parent-child consistency and stability across levels.

### Candidate Metrics

- parent-child purity
- child coverage of parent cluster
- child size entropy within each parent
- hierarchy depth utilization
- percentage of documents changing coarse lineage across seeds
- AMI/ARI per level across seeds
- max child-to-parent mass ratio

### Required Experiments

- Compare baseline hierarchy vs postprocessed hierarchy.
- Evaluate consistency from nano to micro and micro to meso.
- Report whether oversize repair prevents one child from dominating a parent.

### Success Criteria

- The proposed method improves child balance without breaking parent coherence.
- Seed stability at upper levels is unchanged or improved.

### Paper Position

This is important if the paper is positioned as hierarchical landscape
construction rather than flat clustering.

## P1. Stability Across Seeds and Gamma Multipliers

### Claim

An oversized split is more credible if it appears repeatedly across optimizer
seeds or local gamma multipliers.

Potential confidence score:

```math
S(C) =
\Pr_{\alpha,s}
[
\text{cluster } C \text{ yields a retained non-oversize split}
].
```

### Required Experiments

- Run local probes across multiple seeds and gamma multipliers.
- Measure repeatability of escaped fragments.
- Compare high-confidence and low-confidence split examples.

### Success Criteria

- Stable probes correlate with better semantic coherence.
- Low-confidence probes are rejected or marked diagnostic.

### Paper Position

This can make the method look less heuristic, but it increases runtime. Treat
as a second-stage experiment after P0 evidence is collected.

## P2. Semantic Validation Layer

### Claim

Structural balance is useful only if semantic coherence is preserved. Since the
application is scientific literature mapping, the repair should be evaluated by
topic coherence and boundary assignment quality.

### Candidate Metrics

- top keyword specificity before/after
- cross-cluster keyword leakage
- cluster label coherence
- LLM blind review of boundary papers
- expert review for selected split fragments
- abstract embedding cohesion within split fragments

### Required Experiments

- Sample documents from clusters changed by postprocess.
- Compare before/after local neighborhoods.
- Run LLM blind review on source cluster vs escaped fragment assignment.

### Success Criteria

- Repaired clusters are at least as coherent as baseline clusters.
- Escaped fragments receive distinct labels or keyword profiles.

### Paper Position

This is important for a scientometrics audience, but it can be expensive.
Run after structural metrics identify promising cases.

## P2. Adaptive Policy Selection

### Claim

Different graph regimes may need different postprocess policies. For example,
high-density broad topics may prefer `quality_first`, while operational map
generation may require `hard_cap`.

### Possible Rules

- Use `quality_first` by default.
- Enable `hard_cap` only when target max violation is severe.
- Increase trim budget only if candidate moves are quality-neutral.
- Skip local probing when oversize cluster has low conductance and no external
  boundary.

### Required Experiments

- Build a policy matrix over:
  - trim delta bound
  - max moves per cluster
  - apply iterations
  - selection mode
- Compare quality/balance tradeoffs.

### Success Criteria

- A simple policy dominates manual tuning in most cases.
- The policy remains explainable.

### Paper Position

Potentially useful, but risky if it becomes a bag of heuristics. Keep it
secondary unless the policy is simple and well-supported.

## P3. New Rust Fused Kernel

### Claim

A fused Rust postprocess kernel could reduce overhead by combining candidate
selection, split repair, trim, quality checks, and diagnostics.

### Why It Is Lower Priority

This improves engineering performance but does not materially improve the
scientific claim unless runtime becomes the bottleneck in field-scale tests.

### Required Experiments

- Profile current Python orchestration on large fields.
- Identify repeated graph scans.
- Compare fused kernel runtime and memory.

### Success Criteria

- Meaningful runtime reduction on million-scale graphs.
- No change in accepted memberships.

### Paper Position

Engineering appendix or software note, not the core contribution.

## Suggested Experimental Order

1. Establish baseline tables:
   - raw Leiden
   - small repair
   - quality-first oversize repair
   - hard-cap diagnostic mode
2. Add local resolution probing ablations:
   - gamma multiplier sweep
   - global high-gamma rerun comparison
3. Add contraction-aware metrics:
   - before/after contracted graph balance
   - next-level cluster distribution
4. Aggregate failure taxonomy:
   - by level
   - by field
   - by oversize severity
5. Add hierarchy consistency metrics:
   - parent-child purity
   - child balance
   - seed stability
6. Run semantic validation only on representative changed clusters.

## Minimal Publishable Package

The smallest coherent paper package would include:

- local resolution probing;
- quality-preserving acceptance;
- oversize failure taxonomy;
- contraction-aware next-level evaluation;
- experiments on at least two fields with different citation densities.

The stronger package would additionally include:

- seed/gamma stability confidence;
- semantic validation with LLM or expert review;
- hierarchy consistency metrics across levels.

## Recommended Near-Term Decision

Prioritize P0 work first:

1. local resolution probing evidence;
2. contraction-aware evaluation;
3. failure taxonomy.

These three form the core thesis. P1/P2 additions should be used to strengthen
the paper only after P0 results show that the method improves hierarchy balance
without damaging CPM quality.
