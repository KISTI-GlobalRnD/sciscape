# SciSci Adaptive Refinement TODO

## Status

This is a design TODO, not a production implementation commitment yet.

The standard Rust Leiden path should stay close to CWTS/Java behavior. SciSci-
specific perturbation should be implemented as a separate experimental stage
after baseline Leiden and before small-cluster postprocess.

As of 2026-04-29, the first prerequisite is implemented: standard Leiden now
logs enough progress detail to identify late near-identity contractions and low
movement iterations (`moved_nodes`, recursion `depth`, and contraction
node/edge deltas). A large-graph recursion guard also skips recursive Leiden
calls when contraction has become nearly identity. Cluster-graph diagnostics,
boundary probes, high-gamma split probes, and split-then-repair dry-runs are
implemented. The next step is no longer generic split/merge exploration; it is
an explicit hysteretic split-repair apply path with rollback.

## Latest Observations

Large bcrefresh contracted-graph probes show why this should be a targeted
stage rather than another full random restart:

- `gamma=0.0005`, seed 42, convergence-guard run:
  - final clusters: `1,670,312`
  - elapsed: approximately `66.1 min`
  - high-water memory: approximately `42.9 GB`
  - final doc-size distribution remains very small
- Same input with the large-graph recursion guard:
  - final clusters: `1,670,185`
  - elapsed: approximately `59.2 min`
  - Leiden phase elapsed: approximately `58.4 min`
  - speedup over the guardless convergence run: approximately `414 sec`
    (`10.4%` total wall time)
  - high-water memory: approximately `42.9 GB`
  - CPM quality improved slightly (`+103.37`)
  - final doc-size distribution stayed effectively unchanged
- Cluster-graph dry-run diagnostics on that membership:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/bcrefresh_g0005_recguard_cluster_graph_summary.json`
  - GPU artifact directory:
    `/data/openalex_clusters/rust_profile_bcrefresh_contracted_g0005_recguard_seed42_convergence_guard_20260429/adaptive_refinement_dryrun`
  - graph reload: approximately `50.5 sec`
  - cluster-graph stats pass: approximately `39.0 sec`
  - dry-run high-water memory: approximately `26.2 GB`
  - top `50,000` macro-merge candidates all improved the `250-1500`
    doc-weight band, but none had positive CPM `delta_Q`
  - the top `10,000` exported candidates include `105` exact `delta_Q = 0`
    ties and `1,161` near-neutral candidates with `delta_Q > -1e-4`
- Medium g016 contracted-graph dry-run diagnostics on the
  `gamma=0.0085` membership:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_cluster_graph_summary.json`
  - GPU artifact directory:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_dryrun_selfloops`
  - input graph: `2,925,181` contracted nodes and `169,159,819`
    parquet edge rows
  - graph reload from raw sidecars: approximately `9.0 sec`
  - cluster-graph stats pass: approximately `7.7 sec`
  - dry-run high-water memory: approximately `7.4 GB`
  - active clusters: `1,612,895`
  - clusters in the `250-1500` doc-weight band: `51,845`
  - top `50,000` macro-merge candidates all improved the target size band,
    but none had positive CPM `delta_Q`
  - `217` top candidates had `delta_Q > -1e-4`; `2,321` had
    `delta_Q > -1e-3`
- Exploratory macro-merge policy ensemble on the exported top `10,000`
  g016 candidates:
  - artifact directory:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_macro_merge_ensemble`
  - all selected near-neutral candidates involved at least one singleton
    endpoint; no non-singleton-only policy selected a candidate
  - `epsilon=1e-4` selected `216` non-conflicting pairs with cumulative
    `Q` debt approximately `0.0113`
  - `epsilon=1e-3`, `Q` debt capped at `0.5`, selected `1,493` pairs
  - policies requiring merged weight in the `250-1500` band selected only
    `5` pairs at `epsilon=1e-4`, `23` at `epsilon=3e-4`, and `80` at
    `epsilon=1e-3`
  - no policy increased the count of clusters already in the target band;
    these merges mainly attach tiny singleton fragments to small/medium
    clusters
- Boundary-candidate dry-run on the same g016 membership:
  - artifact directory:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_boundary_candidates`
  - GPU artifact directory:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_boundary_dryrun`
  - graph reload: approximately `9.0 sec`
  - cluster-graph stats pass with second-neighbor metrics: approximately
    `7.5 sec`
  - dry-run high-water memory: approximately `7.4 GB`
  - active clusters with a second neighbor: `1,540,676`
  - neighbor-weight ratio is high (`p50=0.630`, `p90=0.930`), meaning many
    clusters are not one-sided leaves at the cluster-graph level
  - policy-filtered candidate counts:
    - `boundary_nonleaf`: `293,608`
    - `boundary_high_ambiguity`: `264,945`
    - `boundary_band_scale`: `85,204`
    - `boundary_largeish_ambiguous`: `27,453`
    - `boundary_strict`: `10,736`
- Boundary block-move probes on top `1,000` candidates:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_boundary_move_probe/README_summary.json`
  - GPU artifact directories:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_boundary_move_probe_*`
  - each run reloads graph in approximately `9-13 sec`, but the actual
    block-move probe pass takes only `0.09-0.18 sec` for `1,000` clusters
  - tested policies: `boundary_strict`, `boundary_high_ambiguity`,
    `boundary_band_scale`, and `boundary_nonleaf`
  - no policy found a positive-CPM single-block move to the top or second
    neighbor cluster
  - with a relaxed near-neutral threshold of `epsilon=0.05`,
    `boundary_high_ambiguity` found only `21` near-neutral block moves and
    `boundary_band_scale` found `22`
  - best observed single-block move still had negative `delta_Q`
    (`-0.000401`)
- Boundary grouped split/move probes on the same top `1,000` candidate sets:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_boundary_group_probe/README_summary.json`
  - GPU artifact directories:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_boundary_group_probe_*`
  - each run reloads graph in approximately `8.9-9.0 sec`, while the grouped
    probe pass itself takes only `0.08-0.11 sec` for `1,000` clusters
  - tested policies: `boundary_strict`, `boundary_high_ambiguity`,
    `boundary_band_scale`, and `boundary_nonleaf`
  - no policy found a positive-CPM grouped move or split to the top or second
    neighbor cluster
  - the grouped top/second-neighbor heuristic is therefore useful as a cheap
    negative screen, but not sufficient as the adaptive perturbation itself
- Multi-core split probes on the top `500` g016 doc-weight clusters:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_multi_core_split_probe/README_summary.json`
  - GPU artifact directories:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_multi_core_split_probe_large_top500*`
  - probe pass takes approximately `0.07 sec` for `500` clusters times
    `5-6` gamma multipliers; graph reload still dominates at approximately
    `8.9 sec`
  - no induced split is positive at the baseline `gamma=0.0085`
  - high-gamma probes reveal strong hysteresis-only splits:
    - coarse multipliers `{1.25,1.5,2,3,5}`: `2,455 / 2,500` rows are
      positive at the probe gamma but non-positive at the baseline gamma
    - fine multipliers `{1.02,1.05,1.10,1.15,1.20,1.25}`: `1,892 / 3,000`
      rows are hysteresis-only
  - split granularity rises rapidly as gamma increases:
    - at `1.02x`, median parts is `3`, median singleton weight is `2`
    - at `1.10x`, median parts is `9`, median singleton weight is `87.5`
    - at `1.25x`, median parts is `20`, median singleton weight is `306.5`
  - this supports the hysteresis model: raising gamma exposes many internal
    fragments and hub/supernode singletons, but preserving those fragments after
    lowering gamma requires an explicit utility/debt rule rather than pure CPM
    acceptance
- Split-then-repair probes on the same top `500` g016 doc-weight clusters:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_split_merge_repair_probe/README_summary.json`
  - GPU artifact directory:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_split_merge_repair_probe_large_top500_fine`
  - repair policy: force high-gamma split, then greedily merge at baseline
    `gamma=0.0085`; source-source and source-external merges are allowed, while
    external-external merges are blocked
  - probe pass takes approximately `1.19 sec` for `500` clusters times `6`
    gamma multipliers
  - most forced splits repair back to the original source cluster:
    `2,612 / 3,000` rows restore the source cluster
  - `374 / 3,000` rows have source mass escaping into external neighbor
    clusters after repair
  - because restored splits can produce tiny floating-point positive
    `net_delta_Q`, use `net_delta_Q > 1e-6` as the practical positive threshold
  - `144` rows are practically net-positive, `130` of them include escaped
    source mass, and `81` rows exceed `net_delta_Q > 1.0`
  - best observed repaired perturbation has `net_delta_Q = 58.91`
  - this is the first positive evidence for the hysteretic strategy: a split
    that is not acceptable by itself can become useful after baseline repair
- Late iterations can spend several minutes while reducing cluster count by
  only hundreds of clusters.
- This suggests the valuable operations are not random leaf movements, but
  targeted macro merge/split/boundary corrections on the cluster graph.

## Motivation

Late Leiden iterations can spend several minutes on near-identity recursive
contractions while moving only a small number of aggregate nodes. Randomly
shuffling all nodes again is therefore a poor default perturbation strategy for
SciSci graphs.

SciSci clusters repeatedly show core-periphery structure. Perturbing leaf-like
nodes is usually low value, and pure high-gamma splitting is usually not
acceptable at the baseline gamma. The useful perturbation is hysteretic:

- temporarily raise gamma to expose multiple cores and hub/periphery fragments,
- force a local split only inside a small candidate cluster,
- repair that split at the baseline gamma while allowing source fragments to
  reattach to external neighbor clusters,
- accept only the repaired final state, not the forced split itself,
- and then run a limited polish around changed neighborhoods.

Macro merge and boundary movement should therefore be treated as components of
the repair stage, not as independent first-class strategies unless the current
cluster-count regime specifically calls for them.

## Cost-Effectiveness Rule

A perturbation is worth trying only if:

```text
expected_structural_gain / estimated_compute_cost
>
additional_random_leiden_iteration_gain / additional_random_leiden_iteration_cost
```

The current large bcrefresh probe should be used to estimate the denominator.
With recursion guard enabled, late iterations around the 1.6M-cluster regime
still cost approximately `316-320 sec` each while changing cluster count by
only `110-201` clusters and moving `5,040-10,104` aggregate nodes.

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

For split-then-repair:

```text
forced_split_debt = delta_Q_split_at_baseline
repair_gain = sum accepted baseline-gamma merge gains after the forced split
net_delta_Q = forced_split_debt + repair_gain
```

The forced split may be negative. It is only a basin-transition proposal. The
candidate is evaluated after repair, using `net_delta_Q` and structural terms
such as escaped source weight, retained core count, restored-source flag,
singleton mass, and target-size-band movement.

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
  net_delta_Q
  + lambda * size_band_improvement
  + eta * target_cluster_count_improvement
  + rho * escaped_source_weight_if_net_positive
  - mu * singleton_or_leaf_penalty
  - nu * restored_source_cluster_penalty

cost =
  estimated_edges_scanned
  + probe_count * induced_edges
  + repair_quotient_edges

priority = utility / cost
```

The first implementation should log these terms in dry-run mode before applying
any membership changes.

For split-repair candidates, the first acceptance policy should be conservative:

```text
accept if:
  net_delta_Q > 1e-6
  and restored_source_cluster == false
  and (escaped_source_weight > 0 or retained_source_units >= 2)
  and final_small_source_weight <= singleton_budget
  and candidate does not conflict with an already accepted source cluster
```

Candidates with `net_delta_Q > 1.0` should be treated as strong candidates.
Candidates with `0 < net_delta_Q <= 1.0` are useful for diagnosis but should
require a structural gain threshold before apply mode.

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

Implementation status:

- [x] Rust backend method: `RustLeidenGraph.cluster_graph_stats(...)`
- [x] Python report helper:
      `sciscape.clustering.write_adaptive_refinement_report(...)`
- [x] Macro-merge candidate table ranked by predicted CPM `delta_Q`
- [ ] Macro-split candidate table

## Stage 2: Macro Merge Dry-Run

Start here because it is cheap and works on the cluster graph.

Candidate conditions:

- `delta_Q_merge >= -epsilon` rather than strictly positive; the first large
  dry-run found no positive macro-merge candidates but did find exact-tie and
  near-neutral candidates that improve the target size band,
- merged doc weight moves closer to target band,
- neither side is purely leaf-like unless the merge fixes an obvious fragment,
- no conflicting greedy merge in the same pass.

Implementation should first support:

- [x] dry-run only,
- [x] greedy non-conflicting candidate selection,
- [x] exact predicted `delta_Q_merge`,
- [x] before/after size-band simulation,
- [x] policy-ensemble comparison over epsilon/leafness/conductance filters,
- [ ] actual membership perturbation,
- [ ] post-perturb polish and rollback.

## Stage 3: Hysteretic Split-Repair

This is now the primary adaptive-refinement direction. Only probe a small
budgeted set of clusters, and judge the final repaired state rather than the
forced split itself.

Candidate conditions:

- doc weight above target maximum,
- or high doc weight within the target band when searching for hysteretic
  basin transitions,
- low internal density, high conductance proxy, or strong external attachment,
- evidence of multi-core structure after a small gamma increase,
- sufficient induced edge budget.

Budget constraints:

```text
total_induced_edges_to_probe <= 5-10% of original directed edges
top_k_split_candidates <= 500-1000
probe_seeds <= 2-3
probe_gamma_multipliers in {1.02, 1.05, 1.10, 1.15, 1.20, 1.25}
```

Candidate workflow:

1. Run high-gamma induced local merge inside the source cluster.
2. Evaluate forced-split debt at the baseline gamma.
3. Build a local quotient containing split source parts plus external neighbor
   clusters touched by those parts.
4. Repair greedily at the baseline gamma, allowing source-source and
   source-external merges while blocking external-external merges.
5. Accept only the repaired state if it passes `net_delta_Q` and structural
   filters.

Do not accept a split only because probe-gamma `delta_Q` is positive. The g016
pilot showed that probe-positive split rows are mostly hysteresis-only and
often create large singleton/hub fragments.

Implementation status:

- [x] large-cluster candidate export by doc weight
- [x] induced local-merge reclustering for candidate clusters
- [x] local split `delta_Q` evaluation at baseline and probe gamma
- [x] hysteresis-only split diagnostics
- [x] split-then-baseline-repair dry-run with external-neighbor escape metrics
- [ ] split-repair candidate selection table ranked by utility/cost
- [ ] split-repair apply mode for non-conflicting candidates
- [ ] exact quality recomputation and rollback after apply
- [ ] limited polish over affected neighborhoods
- [ ] pilot on a larger graph with clusters above the target maximum

## Stage 4: Boundary Refinement

Boundary refinement is now secondary to split-repair. It should operate on
source fragments created by high-gamma split probes or on meaningful ambiguous
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

Implementation status:

- [x] first-neighbor / second-neighbor cluster-graph metrics
- [x] boundary ambiguity policy summaries
- [x] top boundary candidate CSV export
- [x] single-block top/second-neighbor move probe around candidate clusters
- [x] grouped top/second-neighbor split/move probe around candidate clusters
- [ ] integrate split-repair escaped fragments with boundary candidate scoring
- [ ] accepted boundary perturbation only when it improves repaired partition

## Stage 5: Local Polish

After accepted split-repair perturbations, run a limited polish:

- one standard Leiden iteration, or
- a restricted local move over changed neighborhoods.

Do not let polish turn into another full until-convergence run without a budget.

## Open Questions

- What target should be optimized directly: cluster count, size-band mass, CPM,
  or a weighted combination?
- What `net_delta_Q` and structural-gain thresholds should separate apply-mode
  candidates from diagnostic-only candidates?
- How much singleton/hub mass is acceptable after repair when `net_delta_Q` is
  positive?
- Should the split-repair gamma multiplier be selected per candidate by the
  smallest multiplier that creates an escape, or by best utility/cost?
- Should repair allow near-neutral merges (`repair_epsilon > 0`) or stay
  strictly CPM-positive at baseline gamma?
- Should macro merge happen as a pre-pass only when current cluster count is
  above target, or should it remain only inside split-repair repair?
- Should adaptive refinement consume random seeds, or should it use deterministic
  high-gamma ordering first and reserve randomness for tie-breaking?
- How should accepted adaptive changes be compared against one extra standard
  Leiden iteration on the same baseline?

## Initial Implementation Order

1. [x] Add profiling observability to standard Leiden.
2. [x] Add a large-graph recursion guard for near-identity contraction tails.
3. [x] Build cluster graph stats and dry-run report.
4. [x] Implement macro merge dry-run.
5. [x] Validate predicted `delta_Q` with Rust unit tests and dry-run summaries.
6. [x] Implement boundary move/group probes as negative screens.
7. [x] Implement high-gamma multi-core split probes.
8. [x] Implement split-then-baseline-repair dry-run.
9. [ ] Build split-repair candidate selection and conflict resolution.
10. [ ] Add split-repair apply mode behind an explicit experimental flag.
11. [ ] Run exact quality recomputation plus rollback on accepted candidates.
12. [ ] Pilot on larger graph where clusters exceed the target maximum.

## Non-Goals For Now

- Do not change the default Leiden objective or local-move semantics.
- Do not mix SciSci-specific target-size heuristics into the Java/CWTS parity
  path.
- Do not accept split/merge candidates solely because they move the cluster
  count toward a target; CPM and size-distribution effects must both be logged.
