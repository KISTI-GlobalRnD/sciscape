# Dongdaemun Refinement Algorithm Design

Status: design target for the next algorithmic slice. This document describes
an integrated Leiden-refinement variant. It does not replace the currently
validated Dongdaemun-post manuscript evidence.

## Core Decision

Dongdaemun should be developed in two clearly separated forms.

1. `Dongdaemun-post` is the current validated method. It runs after a Leiden
   membership is available, proposes upper-tail macro refinements, and commits
   only memberships whose exact original-graph CPM audit is non-regressing.

2. `Dongdaemun-refinement` is the next research target. It moves what we learned
   from postprocess candidate exploration into the Leiden refinement stage by
   allocating extra refinement search to oversized or unstable parent
   communities. It leaves the CPM objective unchanged.

The second form is not "postprocess inside the loop". The important change is
where the information is used. Size and instability decide which parent
communities receive extra refinement effort before contraction. They do not add
a size penalty to CPM and they do not directly force cross-parent reassignment.

## Design Principles

CPM remains the only optimization objective.

Oversize is a search-budget signal. It is evidence that a parent community may
hide unresolved internal structure, not proof that the objective is wrong.

Every parent community still receives standard Leiden refinement. Dongdaemun
adds extra refinement attempts for high-priority parents; it does not skip small
or medium parents.

Refined children must be subsets of their local-moving parent community. This
preserves the Leiden refinement invariant needed for contraction and recursive
optimization.

Parent-neighbor information may be used as a diagnostic or priority score, but
not as a direct reassignment rule inside refinement. Cross-parent movement
should be decided later by standard local moving on the reduced graph at the
same `gamma`.

The integrated variant must be judged by Pareto evidence: better upper-tail
structure or downstream hierarchy metrics at comparable or lower runtime, while
preserving CPM quality behavior.

## Notation

Let:

- `G = (V, E, w)` be the current graph.
- `gamma` be the CPM resolution.
- `P_move` be the partition after the Leiden local-moving phase.
- `C` be a parent community in `P_move`.
- `a_v` be the document weight of node `v`.
- `W(C) = sum_{v in C} a_v` be the diagnostic document weight of `C`.
- `T_max` be the target upper-tail document weight.
- `R_std(C)` be the standard Leiden refinement result inside parent `C`.
- `R_ddm(C)` be an optional Dongdaemun extra-refinement result inside `C`.
- `R(C)` be the refined child partition used for contraction.

The basic size-pressure score is:

```text
size_pressure(C) = max(0, W(C) / T_max - 1)
```

A softer eligibility threshold may be used:

```text
eligible_by_size(C) = W(C) >= alpha * T_max
```

where `alpha <= 1`. This lets the algorithm spend budget on parents approaching
the cap without changing the objective. Computing these scores costs one pass
over the parent membership plus a sort over active parents.

## Algorithm

```text
procedure LEIDEN_CPM_WITH_DONGDAEMUN_REFINEMENT(G, gamma, config):
    P = INITIALIZE_PARTITION(G)

    repeat until convergence or max iterations:
        P_move = LEIDEN_LOCAL_MOVE(G, P, gamma)

        parents = COMMUNITIES(P_move)
        R = empty refined partition

        for each parent C in parents:
            R_std(C) = STANDARD_LEIDEN_REFINEMENT_WITHIN_PARENT(G, C, gamma)
            R(C) = R_std(C)

        priorities = SCORE_PARENTS(G, P_move, parents, config)
        selected = SELECT_EXTRA_REFINEMENT_PARENTS(priorities, config.budget)

        for each parent C in selected:
            candidates = GENERATE_PARENT_INTERNAL_CANDIDATES(
                G,
                C,
                R_std(C),
                gamma,
                config,
            )
            candidate = CHOOSE_PARENT_INTERNAL_REFINEMENT(candidates, config)
            if candidate passes structural screens:
                R(C) = candidate

        G_reduced = CONTRACT_BY_REFINED_CHILDREN(G, R)
        P_reduced = LEIDEN_CPM(G_reduced, gamma)
        P = MERGE_REDUCED_PARTITION_BACK(R, P_reduced)

    return P
```

The only new degree of freedom is the extra parent-internal refinement budget.
The reduced graph is still optimized with the original CPM objective.

## Parent Priority

The first implementation should use a deliberately simple priority function:

```text
priority(C) =
    w_size * size_pressure(C)
  + w_instability * instability_pressure(C)
  + w_boundary * boundary_ambiguity(C)
  + w_probe * prior_probe_signal(C)
```

For `v0`, use only `size_pressure` plus deterministic top-`K` selection. This
keeps the first experiment interpretable and cheap.

Later versions can add:

- `instability_pressure`: disagreement across seed perturbations or repeated
  parent-local refinements.
- `boundary_ambiguity`: whether a parent has multiple strong neighboring
  parents on the quotient graph.
- `prior_probe_signal`: cached evidence from previous hierarchy levels or
  previous iterations that a parent produced useful splits.

Small parents are not ignored. They receive `R_std(C)` exactly as in standard
Leiden. Dongdaemun priority only controls extra attempts.

## Candidate Generation

Candidate generation must produce child communities contained in one parent.

Allowed `v0` generators:

- Same-gamma seed perturbation inside the parent.
- Higher-gamma parent-local refinement, followed by baseline-gamma internal
  repair inside the same parent.
- Multiple deterministic seeds for the largest or most unstable parents, capped
  by a strict budget.

Deferred generators:

- Branch-adaptive critical-gamma search.
- Seed ensemble stability scoring.
- Parent-neighbor quotient diagnostics.
- Learned or cached predictor screens.

Not allowed in the refinement stage:

- Moving a child directly into an external neighbor parent.
- Applying boundary trim as a committed reassignment.
- Adding size penalties to the local-move gain formula.

## Candidate Screens

The first integrated version should prefer structural screens over strong
quality claims, because the final quality effect is determined after reduced
graph optimization.

A parent-internal candidate is eligible when:

- it produces at least two non-empty children;
- its largest child fraction is meaningfully below the original parent share;
- singleton or dust mass stays under a configured budget;
- it does not simply restore the original parent after internal repair;
- it does not explode the number of reduced graph nodes beyond the budget.

The current postprocess signals should be reused as diagnostics:

- `net_delta_q` at baseline gamma;
- `restored_source_cluster`;
- `escaped_source_weight`;
- `retained_source_units`;
- `largest_source_unit_fraction`;
- singleton and low-weight child mass;
- receiver oversize increase.

In the integrated design, these signals rank or filter extra-refinement
candidates. They are not a final current-level commit rule, because the
refined children are passed into contraction and the reduced graph decides the
next moves.

## Parent-Neighbor Quotient Diagnostic

The postprocess split-repair evidence showed that useful changes often appear
only after a forced split is repaired at baseline gamma. That suggests a
quotient diagnostic, but it must be used carefully.

For a selected parent `C`, build a small quotient whose nodes are:

- candidate children of `C`;
- neighboring parent communities of `C`;
- optionally a residual "other neighbor" bucket for cheap scoring.

Use this quotient to estimate whether candidate children have plausible
baseline-gamma attachments outside the original parent. The score can raise or
lower parent priority, or select among parent-internal child partitions.

The quotient diagnostic must not directly assign a child to a neighbor in the
refinement phase. That assignment belongs to later local moving on the reduced
graph.

## Configuration Sketch

```text
DongdaemunRefinementConfig:
    enabled: bool = false
    target_max_weight: float
    soft_min_ratio: float = 1.0
    max_extra_parents_per_iteration: int
    max_extra_children_per_parent: int
    max_reduced_node_growth_ratio: float
    gamma_multipliers: [1.02, 1.05, 1.10, 1.15, 1.20, 1.25]
    seed_perturbations: int
    min_child_weight: float
    max_singleton_weight_fraction: float
    max_largest_child_fraction: float
    use_quotient_diagnostic: bool = false
    priority_size_weight: float = 1.0
    priority_instability_weight: float = 0.0
    priority_boundary_weight: float = 0.0
    priority_probe_weight: float = 0.0
```

Default behavior remains unchanged when `enabled=false`.

## Implementation Slices

### Slice 1: Size-Priority Internal Refinement

Implement a Rust-only experimental path in the Leiden refinement stage.

- Score parent communities by `size_pressure`.
- Select the top `K` eligible parents per iteration.
- Run one or more parent-local split generators.
- Apply only parent-internal child partitions.
- Preserve the standard refinement result for all unselected parents.
- Log selected parents, child counts, largest child fractions, and reduced graph
  growth.

This slice answers whether postprocess-learned oversize targeting can improve
the search path before contraction.

### Slice 2: Seed And Gamma Perturbation Budget

Add bounded seed perturbations and the existing fine gamma multiplier schedule.

- Compare same-gamma seed perturbation against high-gamma split plus internal
  baseline repair.
- Track per-parent runtime.
- Track whether the same parent repeatedly produces useful children.

This slice tests the user's concern that some failures are seed-sensitive, not
only high-gamma-sensitive.

### Slice 3: Quotient Diagnostic

Add the parent-neighbor quotient only as a scoring diagnostic.

- Measure whether diagnostic-positive parents lead to better reduced-graph
  outcomes.
- Keep direct neighbor reassignment disabled.
- Compare against size-only priority.

### Slice 4: Branch-Adaptive Search

If the first three slices show promise, add critical-gamma or branch-adaptive
candidate generation.

This should remain a generator improvement, not an objective change.

## Evaluation Design

Every experiment should compare four baselines:

- `standard_leiden`: unchanged Leiden at the same gamma and seed.
- `standard_leiden_extra_budget`: more ordinary Leiden iterations or restarts
  using similar runtime.
- `dongdaemun_post`: current quality-audited postprocess.
- `dongdaemun_refinement`: integrated extra-refinement path.

Primary metrics:

- wall time and peak memory;
- CPM quality on the current graph;
- number of clusters and reduced graph size;
- max document weight, count above `T_max`, and upper-tail shares;
- next-level concentration metrics after contraction;
- seed stability of upper-tail outcomes;
- semantic coherence as post-hoc validation only.

The method is successful only if it finds better or comparable CPM-quality
partitions with better upper-tail and downstream hierarchy metrics under a
competitive runtime budget. Doing more work is not a contribution by itself.

## Expected Contribution If It Works

The contribution would be:

> An objective-preserving adaptive refinement allocation method for Leiden-style
> CPM hierarchy construction. It uses upper-tail size pressure and refinement
> instability to decide where to spend extra parent-local search before
> contraction, while preserving the CPM objective and Leiden's refinement
> subset invariant.

This is stronger than a postprocess because it can change the search state
before reduced-graph optimization. It is also safer than a size-penalized
objective because it keeps quality accounting comparable to standard CPM.

## Risks And Guardrails

High-gamma splits can be hysteresis-only. They may reveal fragments that repair
back to the original parent at baseline gamma. The algorithm must screen for
this rather than assuming every split is useful.

Seed perturbation can be expensive. It must be budgeted per parent and compared
against ordinary Leiden restarts with the same runtime.

Over-fragmentation can make the reduced graph larger without improving the
final partition. Track reduced node growth and cap child counts.

The integrated method does not guarantee strict target satisfaction. It is a
search improvement, not a hard constraint solver.

The first implementation must stay opt-in. It should not alter the default
Python or Rust Leiden behavior.

## Current Recommendation

Keep Dongdaemun-post as the current manuscript-backed method. Use this document
as the implementation target for the next research slice.

The first concrete experiment should be size-priority, parent-internal
Dongdaemun refinement in Rust Leiden:

```text
local move
standard refinement for all parents
extra internal refinement for top oversized parents
contraction by refined children
standard reduced-graph Leiden
merge back
```

If this fails to beat `standard_leiden_extra_budget` and `dongdaemun_post` on
runtime-quality tradeoffs, Dongdaemun should remain a postprocess/fallback
method. If it succeeds, the paper can honestly move from "postprocess" toward
"integrated objective-preserving adaptive refinement".
