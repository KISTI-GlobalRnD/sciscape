# SciSci Adaptive Refinement TODO

## Status

This is a design TODO, not an implementation commitment yet.

The standard Rust Leiden path should stay close to CWTS/Java behavior. SciSci-
specific perturbation should be implemented as a separate experimental stage
after baseline Leiden and before small-cluster postprocess.

## Motivation

Late Leiden iterations can spend several minutes on near-identity recursive
contractions while moving only a small number of aggregate nodes. Randomly
shuffling all nodes again is therefore a poor default perturbation strategy for
SciSci graphs.

SciSci clusters repeatedly show core-periphery structure. Perturbing leaf-like
nodes is usually low value. More useful perturbations should target:

- whether a large core cluster should split,
- whether medium clusters should merge,
- whether ambiguous boundary subclusters should move,
- and only then small local polish.

## Cost-Effectiveness Rule

A perturbation is worth trying only if:

```text
expected_structural_gain / estimated_compute_cost
>
additional_random_leiden_iteration_gain / additional_random_leiden_iteration_cost
```

The current large bcrefresh probe should be used to estimate the denominator:
late iterations around the 1.6M-cluster regime cost several hundred seconds
while changing cluster count by only hundreds of clusters.

## Objective Terms

Use CPM as the primary accept/reject objective:

```text
Q = sum_C [ e_C - gamma * s_C * (s_C - 1) / 2 ]
```

where `s_C` is cluster doc weight and `e_C` is internal edge weight.

For merging two clusters:

```text
delta_Q_merge(A, B) = w_AB - gamma * s_A * s_B
```

For splitting one cluster into parts `P_i`:

```text
delta_Q_split = -cut(P) + gamma * sum_{i<j} s_i * s_j
```

For boundary movement of subcluster `x` from `A` to `B`:

```text
score(x -> B) = w_xB - gamma * s_x * s_B
```

Do not accept candidates solely because they increase cluster count. Penalize
singleton growth and leaf-only movement unless they clearly improve the target
distribution.

## Candidate Utility

Candidate priority should be computed from a utility/cost ratio:

```text
utility =
  delta_Q
  + lambda * size_band_improvement
  + eta * target_cluster_count_improvement
  - mu * singleton_or_leaf_penalty

cost =
  estimated_edges_scanned
  + probe_count * induced_edges

priority = utility / cost
```

The first implementation should log these terms in dry-run mode before applying
any membership changes.

## Stage 1: Cluster Graph Stats

Build reusable cluster-level stats from a baseline membership:

- cluster doc weight,
- cluster block count,
- internal edge weight,
- external edge weight,
- top neighbor weight,
- leafness proxy,
- conductance proxy,
- distance to target size band.

Suggested output:

- per-cluster parquet/csv summary,
- aggregate histogram report,
- top macro-merge and macro-split candidate tables.

## Stage 2: Macro Merge Dry-Run

Start here because it is cheap and works on the cluster graph.

Candidate conditions:

- `delta_Q_merge > -epsilon`,
- merged doc weight moves closer to target band,
- neither side is purely leaf-like unless the merge fixes an obvious fragment,
- no conflicting greedy merge in the same pass.

Implementation should first support:

- dry-run only,
- greedy non-conflicting candidate selection,
- exact predicted `delta_Q_merge`,
- before/after size-band simulation.

## Stage 3: Large Cluster Split Probes

Only probe a small budgeted set of clusters.

Candidate conditions:

- doc weight above target maximum,
- low internal density or high conductance proxy,
- evidence of multi-core structure,
- sufficient induced edge budget.

Budget constraints:

```text
total_induced_edges_to_probe <= 5-10% of original directed edges
top_k_split_candidates <= 500-1000
probe_seeds <= 2-3
probe_gamma_multipliers in {1.25, 1.5, 2.0}
```

Accept split only if it improves CPM or nearly preserves CPM while improving
the target size distribution without creating excessive singleton mass.

## Stage 4: Boundary Refinement

Boundary refinement should operate on subclusters or meaningful ambiguous
regions, not arbitrary leaf nodes.

Candidate conditions:

- top-1 and top-2 destination scores are close,
- the unit connects meaningfully to more than one core cluster,
- doc weight and degree exceed minimum thresholds,
- leafness is below threshold.

Exclude:

- low-degree one-sided leaves,
- tiny doc-weight fragments,
- moves that only increase singleton count.

## Stage 5: Local Polish

After accepted macro perturbations, run a limited polish:

- one standard Leiden iteration, or
- a restricted local move over changed neighborhoods.

Do not let polish turn into another full until-convergence run without a budget.

## Open Questions

- What target should be optimized directly: cluster count, size-band mass, CPM,
  or a weighted combination?
- Should macro merge happen before split for all regimes, or only when current
  cluster count is above target?
- What leafness threshold separates irrelevant leaves from meaningful boundary
  subclusters?
- Should adaptive refinement consume random seeds, or should it stay mostly
  deterministic with randomness only for tie-breaking?
- How should accepted adaptive changes be compared against one extra standard
  Leiden iteration on the same baseline?

## Initial Implementation Order

1. Add profiling observability to standard Leiden.
2. Build cluster graph stats and dry-run report.
3. Implement macro merge dry-run.
4. Validate predicted delta_Q against exact recomputation on small test graphs.
5. Add macro merge apply mode behind an explicit experimental flag.
6. Design split probes after merge dry-run results are available.
