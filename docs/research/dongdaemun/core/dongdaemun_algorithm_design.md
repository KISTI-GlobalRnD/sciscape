# Dongdaemun Algorithm Design

Status: canonical algorithm specification for the manuscript track.

Naming and claim boundaries are defined in
`docs/research/dongdaemun/core/dongdaemun_naming_contract.md`.

Dongdaemun is an objective-preserving macro-refinement layer for hierarchical
CPM science maps. It is not a new clustering objective, not a size-penalized
CPM variant, and not a replacement for Leiden. It adds a deterministic,
audited macro-move neighborhood around oversized communities that remain after
standard Leiden-style CPM optimization.

The shortest definition is:

> Dongdaemun reopens oversized CPM communities, proposes collective
> refinements, and accepts the resulting membership only when exact
> original-graph CPM quality is non-regressing.

## Algorithmic Contract

Dongdaemun must satisfy four contract points.

1. CPM remains the objective.
   Size thresholds are never inserted into the CPM quality function. They only
   decide where to spend candidate-generation effort.

2. Lower-tail support repair is separate.
   Tiny-cluster consolidation is an auxiliary minimum-support policy. It is the
   baseline from which Dongdaemun starts, not the main contribution.

3. Macro-refinement is audited on the original graph for the level.
   Candidate generators may use local probes, higher probe resolutions, or
   boundary heuristics, but acceptance is based on exact CPM quality computed on
   the graph whose membership will be passed forward.

4. Fallback is part of the algorithm.
   If the audit fails, the effective hierarchy receives the pre-Dongdaemun
   baseline membership. Rejected memberships may be written as diagnostic
   artifacts, but they are not effective results.

## Notation

Let:

- `G = (V, E, w)` be the current hierarchy-level graph.
- `P_raw` be the raw Leiden membership.
- `P_min` be the membership after lower-tail minimum-support repair.
- `P_ddm` be a candidate Dongdaemun-refined membership.
- `P_eff` be the effective membership passed to contraction.
- `gamma` be the CPM resolution.
- `a_v` be the document weight of node `v` for size diagnostics.
- `W(C) = sum_{v in C} a_v` be cluster document weight.
- `T_min` be the lower-tail support threshold.
- `T_max` be the upper-tail target.
- `epsilon_Q` be the quality floor delta, default `0`.

The CPM objective used by the project evaluator is:

```text
Q_gamma(P; G) = sum_C [internal_weight_G(C) - gamma * n_C * (n_C - 1) / 2]
```

where `n_C` follows the graph's CPM node-size convention. The manuscript should
state that all Dongdaemun acceptance decisions use the same exact evaluator as
the Leiden/CPM implementation for the current level.

The Dongdaemun audit delta is:

```text
Delta Q(P_ddm | P_min) = Q_gamma(P_ddm; G) - Q_gamma(P_min; G)
```

For `quality_first`, accept when:

```text
Delta Q(P_ddm | P_min) >= epsilon_Q
```

For `hard_cap`, accept when:

```text
Delta Q(P_ddm | P_min) >= epsilon_Q
and max_C W(C; P_ddm) <= T_max
```

With the default `epsilon_Q = 0`, any accepted `quality_first` output is
non-regressing relative to `P_min`. `hard_cap` either returns a non-regressing
cap-satisfying candidate or falls back to `P_min`.

## What Counts As Dongdaemun

Dongdaemun is the upper-tail macro-refinement stage:

```text
P_raw -> lower-tail support repair -> P_min
P_min -> Dongdaemun upper-tail audit -> P_eff
P_eff -> contraction for next hierarchy level
```

The following are part of Dongdaemun:

- detecting oversized communities in `P_min`,
- generating local macro-refinement candidates for those communities,
- selecting non-conflicting candidates deterministically,
- applying conservative boundary polish,
- auditing the final membership using exact CPM on `G`,
- falling back to `P_min` when acceptance fails.

The following are not Dongdaemun's main contribution:

- raw Leiden optimization,
- tiny-cluster support repair,
- semantic coherence scoring,
- hard-cap diagnostics,
- branch-adaptive local-gamma tuning,
- integrated loop-level acceleration.

## Algorithm 1: Hierarchical CPM With Dongdaemun

```text
procedure BUILD_HIERARCHY_WITH_DONGDAEMUN(G_0, levels, gamma_schedule, T_min, T_max):
    G = G_0
    hierarchy = []

    for level in levels:
        gamma = gamma_schedule[level]

        P_raw = LEIDEN_CPM(G, gamma)
        P_min = LOWER_TAIL_SUPPORT_REPAIR(G, P_raw, T_min)

        P_eff, audit = DONGDAEMUN_REFINE(
            G,
            P_min,
            gamma,
            T_max,
            policy="quality_first",
            epsilon_Q=0,
        )

        hierarchy.append((G, P_raw, P_min, P_eff, audit))
        G = CONTRACT_GRAPH(G, P_eff)

    return hierarchy
```

This is the manuscript-level algorithm. In the frozen evidence, Dongdaemun is
implemented conservatively as a postprocess after each relevant Leiden result.

## Algorithm 2: Dongdaemun Refinement

```text
procedure DONGDAEMUN_REFINE(G, P_min, gamma, T_max, policy, epsilon_Q, R):
    Q_base = Q_gamma(P_min; G)
    S = OVERSIZED_COMMUNITIES(P_min, T_max)

    if S is empty:
        return P_min, audit(status="no_current_oversize_candidates",
                            accepted=true,
                            DeltaQ=0)

    P_cur = P_min
    P_best_cap = P_min if MAX_WEIGHT(P_min) <= T_max else None
    split_log = []
    rejected_signatures = empty set

    for iteration in 1..R:
        S_cur = OVERSIZED_COMMUNITIES(P_cur, T_max)
        if S_cur is empty:
            break

        C = GENERATE_SPLIT_REPAIR_CANDIDATES(
            G,
            P_cur,
            S_cur,
            gamma,
            probe_schedule=PROBE_SCHEDULE(iteration),
        )
        C = C plus BOUNDARY_POLISH_CANDIDATES(G, P_cur, S_cur, gamma)
        C = FILTER_REJECTED_SIGNATURES(C, rejected_signatures)
        A = RANK_CANDIDATES(C, policy)

        if A is empty:
            break

        progress = false
        for candidate in A:
            if CONFLICTS_WITH_COMMITTED(candidate, split_log):
                continue

            P_next = APPLY(candidate, P_cur)
            if P_next == P_cur:
                rejected_signatures.add(SIGNATURE(candidate, P_cur))
                continue

            DeltaQ_step = Q_gamma(P_next; G) - Q_gamma(P_cur; G)
            if DeltaQ_step < 0:
                rejected_signatures.add(SIGNATURE(candidate, P_cur))
                continue

            P_cur = P_next
            split_log.append((candidate, DeltaQ_step))
            progress = true

            if Q_gamma(P_cur; G) >= Q_base + epsilon_Q
               and MAX_WEIGHT(P_cur) <= T_max:
                P_best_cap = P_cur

        if progress is false:
            break

    P_trim = BOUNDARY_POLISH(
        G,
        P_cur,
        gamma,
        T_max,
        min_move_delta = TRIM_DELTA_BOUND(policy),
        quality_floor = Q_base + epsilon_Q,
    )

    DeltaQ_final = Q_gamma(P_trim; G) - Q_base
    cap_ok = max_C W(C; P_trim) <= T_max
    quality_ok = DeltaQ_final >= epsilon_Q

    if policy == "quality_first" and quality_ok:
        return P_trim, audit(status="committed",
                             accepted=true,
                             DeltaQ=DeltaQ_final,
                             cap_ok=cap_ok,
                             split_log=split_log)

    if policy == "hard_cap" and quality_ok and cap_ok:
        return P_trim, audit(status="committed",
                             accepted=true,
                             DeltaQ=DeltaQ_final,
                             cap_ok=true,
                             split_log=split_log)

    if policy == "hard_cap" and P_best_cap is not None:
        return P_best_cap, audit(status="committed_best_cap_state",
                                 accepted=true,
                                 diagnostic_membership=P_trim,
                                 DeltaQ=Q_gamma(P_best_cap; G) - Q_base,
                                 cap_ok=true,
                                 split_log=split_log)

    return P_min, audit(status=FALLBACK_REASON(quality_ok, cap_ok),
                        accepted=false,
                        diagnostic_membership=P_trim if P_trim != P_min else None,
                        DeltaQ=DeltaQ_final,
                        cap_ok=cap_ok,
                        split_log=split_log)
```

The current code implements this pattern in
`sciscape/clustering/hierarchy_postprocess.py` with:

- `apply_iterations = 4`,
- `selection_mode = "oversize_first"` by default,
- probe gamma multipliers `(1.02, 1.05, 1.10, 1.15, 1.20, 1.25)`,
- `quality_floor_delta = 0`,
- `quality_first_trim_min_delta_q = 0`,
- `hard_cap_trim_min_delta_q = -1`,
- `trim_max_moves_per_cluster = 100`.

The Rust Dongdaemun core and PyO3 binding are available as an opt-in hierarchy
fast path through
`HierarchyPostprocessConfig(use_rust_dongdaemun=True, write_artifacts=False)`.
The default remains the Python orchestration path, and artifact-writing
hierarchy runs still use Python until the Rust path reproduces the trim move-row
CSVs.

The probe gamma multipliers are candidate-generation settings. They do not
change the final acceptance objective.

Explicit termination conditions:

- `R = apply_iterations` is exhausted.
- no oversized communities remain.
- no candidates remain after filtering rejected signatures.
- no candidate produces a changed membership.
- no candidate produces non-negative exact step `Delta Q`.

Zero-delta candidates are allowed only if they change the membership and reduce
upper-tail pressure. A zero-delta no-op is rejected by signature so it cannot be
selected repeatedly.

The `hard_cap` best-state rule is a design improvement over a purely lossy
fallback. If an intermediate state satisfies both the cap and quality floor, the
algorithm may keep that as the effective output even when later diagnostic
proposals fail. The frozen evidence should still be reported according to the
effective membership actually used in those runs.

## Candidate Generation

Candidate generation is intentionally permissive; acceptance is strict.

Each candidate generator must return enough information to reconstruct or apply
a proposed membership and to audit it with exact CPM.

Required candidate fields:

- source oversized cluster id,
- probe parameters,
- proposed child labels or moved-node list,
- predicted or local diagnostic delta,
- size diagnostics before and after the proposed move,
- stable candidate id for deterministic tie-breaking.

Allowed generators:

- split-repair probes inside oversized communities,
- repeated same-gamma or fixed-grid probe passes,
- boundary polish from oversized clusters to compatible receivers,
- future branch-adaptive local-resolution probes.

Probe schedule:

- Default fixed schedule: `gamma * [1.02, 1.05, 1.10, 1.15, 1.20, 1.25]`.
- Optional hierarchical schedule: run a coarse multiplier grid first, then run
  finer probes only for parents that produce non-negative or near-neutral
  candidates.
- Compute budget is bounded by:

```text
max_local_runs <= R * n_oversize_candidates * n_gamma_multipliers * n_probe_seeds
```

The first Rust implementation should use one seed and the fixed schedule. Extra
probe seeds are a future robustness option, not a default manuscript claim.

Configuration mapping:

| Config field | Algorithm role |
| --- | --- |
| `resolution` | Final CPM `gamma` for exact audit. |
| `target_max_weight` | Oversize trigger and hard-cap acceptance target. |
| `quality_floor_delta` | `epsilon_Q` in final and step-level quality floors. |
| `apply_iterations` | Maximum refinement iterations `R`. |
| `gamma_multipliers` | Local split-repair candidate-generation schedule. |
| `min_core_weight` | Minimum retained child/core weight before repair treats fragments as small. |
| `randomness` | Local probe stochasticity. |
| `repair_epsilon` | Near-neutral repair merge tolerance inside split-repair. |
| `trim_min_delta_q_quality_first` | Boundary-polish per-move lower bound for `quality_first`. |
| `trim_min_delta_q_hard_cap` | Boundary-polish per-move lower bound for `hard_cap` diagnostics. |
| `trim_max_moves_per_cluster` | Boundary-polish move budget per oversized source cluster. |
| `seed` | Base seed for local probes. |
| `pair_seeded` | Derive stable seeds from `(seed, source_cluster, gamma_multiplier)` so probing and application replay the same local candidate. |

Disallowed as acceptance criteria:

- semantic coherence,
- target satisfaction alone,
- predicted delta without exact CPM audit,
- normalized gain without exact CPM audit.

## Selection Rule

Dongdaemun uses deterministic quality-gated selection.

The recommended manuscript rule is:

```text
filter:
    reject candidates predicted or audited to be below the quality floor
    reject candidates with no membership change

rank quality_first candidates by:
    1. non-negative exact Delta Q when available, otherwise non-negative predicted Delta Q
    2. larger reduction in max/target ratio
    3. larger reduction in oversize count or excess mass
    4. lower fragmentation penalty
    5. larger Delta Q
    6. stable candidate id

rank hard_cap candidates by:
    1. cap satisfaction or larger cap-violation reduction
    2. non-negative exact Delta Q when available, otherwise non-negative predicted Delta Q
    3. larger excess-mass reduction
    4. lower fragmentation penalty
    5. larger Delta Q
    6. stable candidate id

apply sequentially:
    for each ranked candidate:
        skip if it conflicts with already committed state
        apply candidate to current membership
        compute exact Delta Q on G
        commit only if exact Delta Q >= 0 and membership changed
        otherwise rollback candidate and mark its signature rejected
```

Selection can use predicted diagnostics to avoid expensive exhaustive audits.
Every committed candidate must still pass exact CPM accounting after application
to the current membership.

Conflict definition:

```text
conflict(A, B) =
    affected_nodes(A) intersects affected_nodes(B)
    or touched_clusters(A) intersects touched_clusters(B)
```

where `touched_clusters` includes source clusters and receiver/target clusters.
The cluster-set rule is intentionally conservative. It prevents applying a
split of cluster `X` after another candidate has already moved mass into or out
of `X` without regenerating candidates for the new state.

A future implementation may relax the cluster-set rule by regenerating
candidates after every committed move. The exact audit rule remains unchanged.

## Boundary Polish

Boundary polish is a second-stage macro move. It tries to reduce remaining
oversize pressure by moving boundary nodes from oversized communities to
receiver communities.

Boundary polish is not claimed to discover ordinary Leiden local moves that a
fully converged local-moving phase would have already taken. Its role is more
specific:

- lower-tail repair may perturb the membership after the original Leiden pass;
- split-repair may create new boundaries that were not present in `P_min`;
- Dongdaemun targets only oversized source clusters and applies an explicit
  receiver-cap check before moving mass.

This is why boundary polish is better described as targeted post-repair polish
than as a replacement for another full fast-local-move pass.

For `quality_first`:

```text
min_move_delta >= 0
final Delta Q >= epsilon_Q
```

For `hard_cap` diagnostics:

```text
min_move_delta may be negative
final Delta Q must still satisfy the quality floor
cap satisfaction is required for acceptance
```

This distinction matters for the paper. Negative per-move trim budgets are not
part of the default Dongdaemun scientific claim.

## Policy Semantics

| Policy | Effective output | Manuscript role |
| --- | --- | --- |
| `small_only` | `P_min` after lower-tail repair. | Baseline, not Dongdaemun. |
| `oversize_split_only` | Accepted split-repair without the full two-stage polish. | Ablation. |
| `two_stage_quality_first` | Split-repair plus conservative boundary polish if exact `Delta Q >= 0`; otherwise fallback. | Default Dongdaemun-post evidence. |
| `two_stage_hard_cap` | Candidate only if exact `Delta Q >= 0` and cap is satisfied; otherwise fallback. | Diagnostic or operational strict variant. |
| `branch_adaptive_quality_first` | Future generator feeding the same exact audit rule. | Future work. |

## Guarantees Under The Contract

Conditional quality guarantee:

If the exact CPM evaluator is correct and `epsilon_Q >= 0`, any accepted
`quality_first` Dongdaemun output satisfies:

```text
Q_gamma(P_eff; G) >= Q_gamma(P_min; G)
```

Fallback guarantee:

If a candidate fails the audit, the effective output is `P_min`. Therefore the
effective hierarchy never receives a rejected diagnostic membership.

Hard-cap conditional guarantee:

An accepted `hard_cap` output satisfies both the quality floor and the cap. The
policy does not guarantee that a cap-satisfying output exists. If no
cap-satisfying audited state is found, fallback may return `P_min`.

Sequential rollback guarantee:

Batch-predicted candidate gains are never trusted as final evidence. Candidates
are applied sequentially or re-audited after any batch optimization. A candidate
whose exact step `Delta Q` is negative is rolled back and cannot enter the
effective membership.

No semantic guarantee:

Semantic coherence is evaluated after the fact. It is not used to accept or
reject Dongdaemun moves.

No integrated-loop guarantee:

The current evidence validates only conservative Dongdaemun-post behavior.
Loop-level Dongdaemun remains a design target. The current target design is
specified in `docs/research/dongdaemun/refinement/dongdaemun_refinement_algorithm_design.md`.

## Integrated Dongdaemun Loop

The integrated form is future work. The implementation target is not a direct
copy of the postprocess commit loop inside Leiden. It is an objective-preserving
adaptive refinement allocation method:

- local moving still optimizes CPM at the current `gamma`;
- every parent community still receives standard Leiden refinement;
- oversized or unstable parents receive extra parent-internal refinement
  budget;
- refined children must remain subsets of their local-moving parent community;
- parent-neighbor quotient signals may rank candidates but must not directly
  reassign children to external parents;
- contraction and reduced-graph Leiden decide cross-parent movement under the
  unchanged CPM objective.

The full design is maintained in
`docs/research/dongdaemun/refinement/dongdaemun_refinement_algorithm_design.md`. The minimal target loop is:

```text
procedure LEIDEN_CPM_WITH_DONGDAEMUN(G, gamma, T_max):
    P = INITIALIZE_PARTITION(G)

    repeat until convergence or max_outer_iterations exhausted:
        P_move = LEIDEN_LOCAL_MOVE_PASS(G, P, gamma)

        for each parent C in COMMUNITIES(P_move):
            R(C) = STANDARD_LEIDEN_REFINEMENT_WITHIN_PARENT(G, C, gamma)

        S = PRIORITIZE_PARENTS(P_move, T_max)
        for each selected parent C in S:
            A = GENERATE_PARENT_INTERNAL_REFINEMENTS(G, C, R(C), gamma)
            R(C) = BEST_STRUCTURALLY_VALID_REFINEMENT(A, R(C))

        G_reduced = CONTRACT_BY_REFINED_CHILDREN(G, R)
        P_reduced = LEIDEN_CPM(G_reduced, gamma)
        P = MERGE_REDUCED_PARTITION_BACK(R, P_reduced)

    return P
```

Expected future tests:

- Does parent-internal Dongdaemun refinement beat ordinary extra Leiden budget?
- Does integrated Dongdaemun reduce fallback frequency?
- Does it improve source-level upper-tail balance relative to Dongdaemun-post?
- Does it improve next-level concentration metrics?
- Does it change runtime enough to matter?

None of these are current manuscript results.

## Paper-Framing Summary

The clean manuscript framing is:

> Dongdaemun is to oversized hierarchy construction what Leiden was to Louvain
> in spirit: it changes the search neighborhood while preserving the quality
> objective. The current paper validates a conservative postprocess version,
> where every effective macro-refinement is audited by exact CPM accounting.

For the current manuscript evidence:

> Dongdaemun-post is oversize-targeted refinement applied to an already
> available current-level partition. It proposes current-level membership
> changes for oversized communities and accepts only exact CPM non-regressing
> changes.

For the integrated future-work target:

> Dongdaemun-refinement allocates extra parent-internal Leiden refinement budget
> to oversized or unstable local-moving parents. It preserves the refinement
> subset invariant, and reduced-graph Leiden decides cross-parent movement under
> the unchanged CPM objective.

The manuscript should not state or imply:

- Dongdaemun changes CPM.
- Dongdaemun guarantees strict target satisfaction.
- Lower-tail repair is the main algorithmic contribution.
- Hard-cap is the default scientific method.
- Semantic coherence is optimized.
- Branch-adaptive gamma is validated.
- Integrated-loop Dongdaemun is empirically validated.

## Design Issue Disposition

This table records the current decision for the algorithm-review issues raised
before Rust implementation. It separates correctness decisions from future
empirical work so the core algorithm does not quietly absorb unsupported claims.

| Issue | Decision | Rationale | Implementation timing |
| --- | --- | --- | --- |
| Loop convergence was underdefined. | Refined now. | Termination affects correctness. The algorithm now terminates on `apply_iterations`, no oversize, no candidate, no membership change, or no non-negative exact step `Delta Q`. Zero-delta no-ops are rejected by signature. | Required before full core. |
| Candidate conflict resolution was underdefined. | Refined now. | Simultaneous macro moves are unsafe if they touch overlapping nodes or clusters. The first rule is conservative: conflict on affected-node overlap or source/target cluster-set overlap. | Required before full core. |
| Batch candidate application could make summed predicted deltas misleading. | Refined now. | Individual predicted `Delta Q` values are not additive after interacting moves. The core algorithm now uses sequential apply with exact re-audit and rollback. | Required before full core. |
| Probe gamma selection needs justification. | Partially refined. | Fixed multipliers are documented as candidate-generation settings. Hierarchical/fine probe schedules are allowed but not default, because current branch-adaptive evidence is diagnostic rather than validated. | Fixed schedule in first Rust core; hierarchical schedule later. |
| Boundary polish might duplicate fast local move. | Refined conceptually. | Boundary polish is now framed as targeted post-repair polish after lower-tail repair or split-repair creates new boundaries. It is not claimed as a replacement for another full fast-local-move pass. | Toy tests in trim slice; larger ablation later. |
| Hard-cap fallback could discard a useful cap-satisfying intermediate state. | Refined now. | The algorithm now tracks `P_best_cap`, the best audited cap-satisfying intermediate state. Hard-cap can return it instead of falling all the way back to `P_min`. | Required before hard-cap full-core tests. |
| Ranking placed size reduction before quality. | Refined now. | `quality_first` is now a quality-gated policy: candidates must be non-regressing first, then upper-tail improvement ranks among viable candidates. | Required before split selector. |
| Gamma multipliers imply compute cost. | Refined now. | The algorithm now states the budget bound `R * n_oversize_candidates * n_gamma_multipliers * n_probe_seeds`. | First Rust core uses one seed. |
| Config fields were not mapped to algorithm roles. | Refined now. | The config mapping table documents `min_core_weight`, `repair_epsilon`, `pair_seeded`, trim bounds, and probe schedule. | Required before public Rust API stabilizes. |
| `pair_seeded` needed definition. | Refined now. | It derives stable probe seeds from `(seed, source_cluster, gamma_multiplier)` so probe and application replay the same local candidate. | Required before split-repair tests. |
| Macro move novelty relative to Leiden refinement needed sharper language. | Refined now. | Dongdaemun is described as oversize-targeted refinement applied directly to the current partition, while Leiden refinement supports optimization and contraction without directly imposing current-level upper-tail repair. | Manuscript framing. |
| Lower-tail repair interaction needs ablation. | Deferred. | This is an empirical design question, not an algorithm correctness issue. The core still treats `P_min` as input and keeps lower-tail repair separate. | Add experiment later: raw-only, small-only, Dongdaemun-only, combined. |
| Multi-seed probe robustness. | Deferred. | Multi-seed probing increases compute and complicates deterministic tests. The first implementation should use one seed plus pair-seeded replay. | Robustness layer after Rust core is green. |
| Python ranking parity. | Deferred. | Python ranking is report/evidence oriented. Rust should first lock the exact audit contract; byte-for-byte parity can be added only if evidence reproduction requires it. | After Slice 4, if needed. |
