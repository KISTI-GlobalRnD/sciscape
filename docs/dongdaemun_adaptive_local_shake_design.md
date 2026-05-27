# Dongdaemun Adaptive Local Shake Design

Status: research design for the next Dongdaemun-refinement slice. This document
describes a parent-local speculative perturbation portfolio. It does not change
the production defaults and it does not replace the validated Dongdaemun-post
evidence.

## Core Decision

The next refinement prototype should move from a single perturbation family
(`gamma_multipliers` or near-tie randomness) to an adaptive local shake
portfolio.

The design is not a full multibus ensemble. It does not fork the complete Leiden
trajectory, run several full alternatives, and choose the best final membership.
Instead, it opens a short-lived speculative branch at a local decision point,
generates one or more parent-local candidate refinements, scores those
candidates under the original CPM objective, and keeps the current greedy choice
unless a candidate is strictly better.

```text
standard greedy parent decision
        |
        +-- short-lived shake arm A
        +-- short-lived shake arm B
        +-- short-lived shake arm C
        |
score every candidate with original gamma/objective
        |
accept only if candidate improves the current local choice
otherwise keep the original greedy state
```

This targets the local path-dependence of greedy Leiden refinement while
preserving the main CPM objective and the Leiden invariant that refined children
remain inside their parent community.

## Motivation

Recent trajectory and perturbation pilots support three observations.

- Near-tie parent-local refinement can generate changed candidates and at least
  one qf-positive replacement when the commit path allows qf-based replacement.
- Resolution up/down perturbation exposes real boundary sensitivity, but raw
  resolution perturbation is unsafe as a direct policy: it produced both wins
  and sizable losses in the 8-case pilot.
- The useful operation is not "change the objective". The useful operation is
  "use a nearby objective or tie-breaking rule to discover an alternative local
  partition, then judge it by the original objective and guards".

Therefore `gamma` shaking should be only one arm in a broader portfolio. The
system should adaptively choose arms based on parent-local evidence, not sweep
every perturbation everywhere.

## Non-Goals

This design does not:

- add a size penalty to CPM;
- change the final optimization objective;
- move refined children outside their local-moving parent;
- perform full-run ensemble selection inside the Rust core;
- require learned policies in the first implementation;
- commit candidates that are worse than the current local candidate under the
  original objective.

## Terminology

`Decision point` means a parent-local refinement opportunity after the local
moving phase and before contraction.

`Shake arm` means a bounded speculative candidate generator. Each arm is allowed
to perturb how a parent-local refinement is found, but candidate scoring is done
under the original `gamma`.

`Current candidate` means the candidate that the existing greedy/refinement
path would use without the local shake portfolio.

`Local replacement` means replacing the current parent-local candidate before
contraction. It is not a full-trajectory rollback.

## Safety Model

The local shake portfolio has two different safety levels. They must not be
conflated.

`Local-qf-safe` means the replacement candidate improves the parent-local
candidate score under the original CPM objective before contraction.

`Final-qf-safe` means the complete Leiden/Dongdaemun run returns a final
membership whose full quality is not worse than the non-shake baseline.

The v1 local shake commit rule can make a replacement local-qf-safe. It cannot
guarantee final-qf-safety by itself, because changing the refined children can
change the reduced graph and the later greedy trajectory. Final-qf-safety needs
an explicit final guard, audit-and-rollback mode, or a full-run comparison in the
experiment runner.

Therefore:

- `trace_only` is diagnostic only;
- `qf_replace` is local-qf-safe only;
- `pressure_guarded` is local-qf-safe plus upper-tail guard;
- final no-loss claims require `use_final_quality_guard` or an equivalent
  final audit outside the local shake decision.

## Trigger Features

The first implementation should separate required trigger fields from optional
diagnostic fields. Required fields must already be available at the parent-local
candidate point. Optional fields may be absent; they must not silently change a
trigger decision unless the arm explicitly declares that it can use missing
values.

Required v1 fields:

```text
LocalShakeTriggerFeatures {
    depth
    iteration
    parent_id
    parent_visit_index
    parent_size
    parent_weight
    parent_weight_ratio
    standard_n_children
    standard_largest_child_fraction
    standard_singleton_weight_fraction
    local_merge_low_margin_count
    local_merge_min_margin
    local_merge_p10_margin
    current_candidate_delta_q
    current_candidate_valid
    current_candidate_quality_passes
}
```

Optional diagnostic fields:

```text
local_move_low_margin_count
local_move_min_margin
boundary_ambiguity_score
prior_arm_win_rate
first_divergence_phase_hint
```

If an optional field is unavailable, the v1 selector must treat it as
`unknown`, not as zero. This prevents missing instrumentation from suppressing
or activating an arm by accident.

## Shake Arms

The v1 portfolio should support these arms behind an opt-in mode.

### `near_tie_refinement`

Perturb only low-margin local merge/refinement decisions inside the parent.

Trigger:

- parent has at least one low-margin local merge decision;
- margin is below `near_tie_margin_parent_weight * parent_weight`;
- parent size is above a small minimum.

Commit intent:

- useful for greedy tie/path dependence;
- should require `changed_decision_count > 0`.

### `resolution_down`

Run parent-local refinement with a slightly lower probe resolution, then score
the resulting child partition under the original resolution.

Trigger:

- parent appears over-fragmented;
- singleton fraction or tiny-child mass is high;
- prior resolution robustness traces show down-perturbation qf-positive
  candidates for similar parents.

Default probe multipliers:

```text
[0.98]
```

Optional pilot multipliers:

```text
[0.95, 0.98]
```

### `resolution_up`

Run parent-local refinement with a slightly higher probe resolution, then score
the resulting child partition under the original resolution.

Trigger:

- parent is above or near the upper-tail weight target;
- largest child fraction is high;
- standard refinement returns a near-identity or weak split.

Default probe multipliers:

```text
[1.02]
```

Optional pilot multipliers:

```text
[1.02, 1.05]
```

### `seed_local_refinement`

Re-run parent-local refinement at the original resolution with a different
derived seed and/or node order.

Trigger:

- low movement margin;
- repeated trajectory divergence at the same parent/depth;
- current candidate is valid but structurally weak.

This is the cheapest arm conceptually, but it should be budgeted tightly because
it can become a hidden local ensemble if applied broadly.

### `node_order_control`

Change only deterministic ordering/tie-breaking inside the parent-local
refinement.

Trigger:

- low-margin counts are high;
- best and second-best increments are within a small parent-weight-scaled band.

This arm is a diagnostic-first arm. It is useful for proving that the divergence
is caused by local greedy ordering rather than by the resolution probe itself.

### Deferred Arms

The following are plausible but should not be part of the first production-like
prototype:

- `pressure_split_bias`: bias candidate generation toward splitting an
  oversized dominant child, while still scoring by original CPM.
- `repair_merge_bias`: after an over-splitting shake, repair at original gamma
  inside the parent.
- `boundary_move_kick`: allow a bounded low-margin boundary node group to move
  inside a transaction log. This is more invasive because local move side
  effects are less parent-contained than refinement children.
- `quotient_attachment_probe`: use parent-neighbor quotient evidence only as a
  score/trigger, not as a committed external reassignment.

## Adaptive Arm Selection

v1 should use deterministic rules, not a learned policy.

```text
select_arms(features, config):
    arms = []

    near_tie_ready =
        features.local_merge_low_margin_count >= config.near_tie_min_count
        and features.local_merge_min_margin <= config.near_tie_margin_parent_weight
                                           * features.parent_weight

    if near_tie_ready:
        arms.push(near_tie_refinement)

    resolution_up_ready =
        features.parent_weight_ratio >= config.resolution_up_min_parent_ratio
        or features.standard_largest_child_fraction >= config.large_child_fraction

    if resolution_up_ready:
        arms.push(resolution_up)

    resolution_down_ready =
        features.standard_singleton_weight_fraction >= config.singleton_fraction
        and features.parent_weight_ratio <= config.resolution_down_max_parent_ratio

    if resolution_down_ready:
        arms.push(resolution_down)

    seed_ready =
        features.local_merge_low_margin_count >= config.seed_margin_count
        or optional(features.local_move_low_margin_count) >= config.seed_margin_count

    if seed_ready:
        arms.push(seed_local_refinement)

    return budgeted_deterministic_prefix(arms, config.arm_priority)
```

The first budget should be conservative:

```text
max_arms_per_parent = 2
max_candidates_per_parent = 4
max_shake_parents_per_iteration = existing selected parent budget
```

Later versions can replace the deterministic rule with a contextual bandit or
offline-trained ranking model, but only after the trace data is large enough to
estimate per-arm loss risk.

The first selector should avoid a broad `low_margin_count > 0` trigger. The
trajectory pilot showed many low-margin opportunities, so a one-count trigger
would likely consume the budget with near-tie attempts before testing the rest
of the portfolio.

## Candidate Generation Contract

Every arm must return a candidate with this common shape:

```text
LocalShakeCandidate {
    arm
    arm_index
    probe_gamma_multiplier
    probe_randomness
    probe_seed
    child_assignments
    assignment_hash
    n_children
    changed_node_count
    child_count_delta
    largest_child_fraction
    largest_child_fraction_delta
    singleton_weight_fraction
    singleton_weight_fraction_delta
    changed_decision_count
    candidate_delta_q_original_gamma
    valid
    quality_passes
    structural_guard_passes
    rejection_reason
}
```

The candidate must satisfy the parent containment invariant:

```text
forall child in candidate.children:
    child.nodes subset_of parent.nodes
```

All candidate quality values are measured under the original CPM objective, not
under the probe objective that produced the candidate.

## Distinctness

A candidate is distinct from the current candidate when at least one of the
following is true:

```text
candidate.assignment_hash != current.assignment_hash
candidate.changed_node_count > 0
candidate.n_children != current.n_children
abs(candidate.largest_child_fraction - current.largest_child_fraction) > shape_eps
abs(candidate.singleton_weight_fraction - current.singleton_weight_fraction) > shape_eps
candidate.changed_decision_count > 0
```

Recommended initial `shape_eps`:

```text
shape_eps = 1e-12
```

For `near_tie_refinement`, `changed_decision_count > 0` is required in addition
to distinctness. This prevents a near-tie probe that made no actual perturbed
decision from being counted as a useful replacement.

## Hard Constraints And Guard Scores

Hard constraints are conditions that must pass in every commit mode:

- every child is contained inside the current parent;
- the candidate has at least one non-empty child and no orphan assignment;
- the candidate does not exceed the configured reduced-node growth budget;
- candidate quality is finite;
- candidate shape metrics are finite;
- the candidate was produced by an enabled arm.

`pressure_guarded` additionally blocks candidates whose local qf improvement
comes with material upper-tail regression. The first pressure guard should be
simple:

```text
pressure_guard_passes =
    candidate.max_child_weight_ratio <= current.max_child_weight_ratio + pressure_eps
    and candidate.n_above_target <= current.n_above_target
```

If `n_above_target` is unavailable at the parent-local point, v1 should use only
the parent-local max-child ratio and record `pressure_guard_partial = true` in
the trace.

Guard scores used for deterministic reduction should be normalized so they are
tie-breakers, not hidden objectives:

```text
pressure_guard_score =
    current.max_child_weight_ratio - candidate.max_child_weight_ratio

structural_guard_score =
    current.largest_child_fraction - candidate.largest_child_fraction
```

The primary ordering remains qf gain. Guard scores should not let a lower-qf
candidate beat a higher-qf candidate unless the mode explicitly says so.

## Commit Rule

The commit rule should stay simpler than the arm selection logic.

```text
accept candidate if:
    mode is qf_replace
    candidate.valid
    candidate.quality_passes
    candidate.distinct
    candidate_delta_q_original_gamma > current_candidate_delta_q + eps
    hard constraints are not worse
    optional structural guard passes
```

Recommended initial `eps`:

```text
eps = max(config.min_candidate_delta_q, 1e-12 * parent_weight)
```

For pure CPM-qf experiments, the optional structural guard should be off by
default because the previous pilot showed that an overly strict structural guard
can block the only qf-positive replacement. For production-like hierarchy runs,
enable a separate `pressure_guarded` mode that blocks candidates that improve
local qf only by making the upper-tail pressure materially worse.

Rejected candidates must still be traced.

The local commit rule should update only the parent-local choice. It should not
claim that the full run is no-loss unless a final guard runs after the full
membership is produced.

Recommended final guard modes:

```text
none:
    no final rollback; use only for trace/local-qf experiments

quality_guard:
    compare final quality to the non-shake fallback available in the same run;
    return fallback if final quality is below min_final_quality_delta

runner_audit:
    run baseline and local-shake policies as separate pilot rows; compare final
    quality offline; do not expose as a production no-loss guarantee
```

`quality_guard` is the production-like safety target, but it may require keeping
or recomputing enough fallback state to restore the non-shake membership. If
that state is not available cheaply, v1 should report local-qf safety only and
leave final no-loss evidence to `runner_audit`.

## Parallelism

v1 should be sequential-first. Arm evaluation can later be parallelized locally,
but only after candidate generation is factored into pure helpers.

Sequential v1:

```text
for arm in selected_arms in deterministic order:
    generate candidate in parent-local scratch state
    score candidate under original gamma
    append LocalShakeCandidate
reduce candidates deterministically
emit trace rows
update stats
replace current choice if commit rule passes
```

Allowed future parallel region:

```text
for arm in selected_arms parallel:
    generate candidate in parent-local scratch state
    score candidate under original gamma
    return LocalShakeCandidate
```

Sequential region:

```text
sort candidates by deterministic reduce key
emit trace rows
update stats
replace current choice if commit rule passes
```

Trace writers, aggregate stats, RNG streams, and `choice` mutation should not be
shared by worker threads. Each worker should receive a derived deterministic
seed and return a pure candidate record. This avoids non-deterministic trace
ordering and makes tests stable.

The reduce key should be explicit:

```text
(
    commit_eligible desc,
    gain_vs_current desc,
    pressure_guard_score desc,
    structural_guard_score desc,
    arm_priority asc,
    arm_index asc,
    probe_gamma_multiplier asc,
    probe_seed asc,
    assignment_hash asc,
)
```

This tie-break is part of the algorithm contract. Parallel and sequential
candidate evaluation must reduce to the same selected candidate.

## Rust Integration Plan

The implementation should reuse the existing parent-local candidate machinery
instead of adding a separate full-run branch mechanism.

1. Add a config enum:

```text
AdaptiveLocalShakeMode:
    Off
    TraceOnly
    QfReplace
    PressureGuarded
```

`QfReplace` and `PressureGuarded` are local commit modes. They do not imply
final rollback unless paired with a final guard mode.

2. Add an arm enum:

```text
AdaptiveLocalShakeArm:
    NearTieRefinement
    ResolutionDown
    ResolutionUp
    SeedLocalRefinement
    NodeOrderControl
```

3. Add config fields:

```text
adaptive_local_shake_mode
adaptive_local_shake_arms
adaptive_local_shake_max_arms_per_parent
adaptive_local_shake_max_candidates_per_parent
adaptive_local_shake_min_gain_parent_weight
adaptive_local_shake_shape_eps
adaptive_local_shake_arm_priority
adaptive_local_shake_near_tie_min_count
adaptive_local_shake_resolution_down_multipliers
adaptive_local_shake_resolution_up_multipliers
adaptive_local_shake_resolution_up_min_parent_ratio
adaptive_local_shake_resolution_down_max_parent_ratio
adaptive_local_shake_large_child_fraction
adaptive_local_shake_singleton_fraction
adaptive_local_shake_seed_perturbations
adaptive_local_shake_seed_margin_count
adaptive_local_shake_near_tie_margin_parent_weight
adaptive_local_shake_near_tie_randomness
adaptive_local_shake_final_guard_mode
```

4. Refactor candidate handling into a common function:

```text
evaluate_parent_local_candidate(...)
maybe_replace_parent_local_choice(...)
emit_local_shake_candidate_trace(...)
```

5. Keep the existing `adaptive_near_tie_probe_mode` and `gamma_multipliers`
   public behavior stable during the transition. The local shake portfolio can
   initially call the same internal helper paths, then later deprecate duplicate
   research-only knobs.

6. Implement v1 candidate evaluation sequentially. Add Rayon only after the
   candidate helpers are pure and a sequential-vs-parallel equivalence test
   exists.

7. Do not promise final rollback in the first Rust slice unless the fallback
   membership is available cheaply. If fallback state is not retained, expose
   final safety as `runner_audit` only.

## Python Interface Plan

Add opt-in wrapper arguments with safe defaults:

```text
adaptive_local_shake_mode: "off" | "trace_only" | "qf_replace" | "pressure_guarded"
adaptive_local_shake_arms: tuple[str, ...] = ()
adaptive_local_shake_max_arms_per_parent: int = 0
adaptive_local_shake_max_candidates_per_parent: int = 0
adaptive_local_shake_min_gain_parent_weight: float = 0.0
adaptive_local_shake_shape_eps: float = 1e-12
adaptive_local_shake_arm_priority: tuple[str, ...] = ()
adaptive_local_shake_near_tie_min_count: int = 1
adaptive_local_shake_resolution_down_multipliers: tuple[float, ...] = ()
adaptive_local_shake_resolution_up_multipliers: tuple[float, ...] = ()
adaptive_local_shake_resolution_up_min_parent_ratio: float = 1.0
adaptive_local_shake_resolution_down_max_parent_ratio: float = 1.0
adaptive_local_shake_large_child_fraction: float = 0.95
adaptive_local_shake_singleton_fraction: float = 0.05
adaptive_local_shake_seed_perturbations: int = 0
adaptive_local_shake_seed_margin_count: int = 2
adaptive_local_shake_final_guard_mode: "none" | "quality_guard" | "runner_audit" = "none"
```

Defaults must preserve current behavior exactly.

## Trace Schema

Add JSONL events.

### `adaptive_local_shake_trigger`

Fields:

```text
event
run_id
depth
iteration
parent_id
parent_visit_index
parent_size
parent_weight
parent_weight_ratio
standard_n_children
standard_largest_child_fraction
standard_singleton_weight_fraction
local_merge_low_margin_count
local_merge_min_margin
local_merge_p10_margin
local_move_low_margin_count
optional_fields_present
selected_arms
trigger_reason
```

### `adaptive_local_shake_candidate`

Fields:

```text
event
run_id
depth
iteration
parent_id
parent_visit_index
arm
arm_index
probe_gamma_multiplier
probe_seed
assignment_hash
candidate_n_children
changed_node_count
child_count_delta
largest_child_fraction
largest_child_fraction_delta
singleton_weight_fraction
singleton_weight_fraction_delta
candidate_delta_q
current_candidate_delta_q
gain_vs_current
valid
quality_passes
structural_guard_passes
distinct
commit_eligible
committed
commit_block_reason
changed_decision_count
final_guard_mode
```

### `adaptive_local_shake_decision`

Fields:

```text
event
run_id
depth
iteration
parent_id
parent_visit_index
candidate_count
best_arm
best_gain_vs_current
best_assignment_hash
committed
selected_arm
selected_gain_vs_current
selected_assignment_hash
final_guard_mode
```

The analyzer should report:

- arm trigger counts;
- candidate counts by arm;
- valid and qf-positive rates by arm;
- local replacement counts by arm;
- distinctness reasons by arm;
- final qf delta by policy;
- final qf loss count by final guard mode;
- upper-tail pressure changes by policy;
- elapsed overhead.

## Test Plan

Rust unit tests:

- arm selector returns deterministic arms for synthetic trigger features;
- resolution up/down arms score candidates under original gamma, not probe
  gamma;
- local shake never moves a child outside the parent;
- candidate distinctness is false for identical assignments and true for
  changed assignments, child counts, or shape metrics;
- `trace_only` emits candidates but does not replace the current choice;
- `qf_replace` keeps the current choice when all candidates are worse;
- `qf_replace` replaces the current choice when a valid candidate improves qf;
- candidate reduce tie-break is deterministic;
- future parallel candidate evaluation reduces to the same selected candidate as
  the sequential path.

Python tests:

- wrapper forwards all local shake options to Rust;
- invalid mode and invalid arm names raise `ValueError`;
- synthetic trace analyzer writes required columns;
- analyzer separates local replacement rate from final qf loss rate;
- local shake defaults leave existing wrapper calls unchanged.

Validation commands:

```bash
cargo test --manifest-path rust/Cargo.toml --features python --lib local_shake
uv run --extra dev maturin develop --manifest-path rust/Cargo.toml
uv run pytest -q tests/test_leiden_rust.py -k local_shake
uv run pytest -q tests/test_dongdaemun_adaptive_local_shake.py
uv run pytest -q
```

## Pilot Plan

Use the existing 8-case trajectory pilot set.

Policies:

```text
online
near_tie_qf_replace
resolution_trace
local_shake_trace
local_shake_qf_replace
local_shake_pressure_guarded
local_shake_qf_replace_runner_audit
local_shake_pressure_guarded_runner_audit
```

Initial local shake arms:

```text
near_tie_refinement
resolution_up
resolution_down
seed_local_refinement
```

Primary success criteria:

- local shake produces distinct parent-local candidates in divergent cases;
- `qf_replace` records local replacements only when the original-gamma
  parent-local score improves over the current candidate;
- trace identifies which arm generated each local replacement;
- runner audit shows whether local replacements survive to final qf gains;
- overhead stays within `1.20x` for sequential v1 on the pilot.

Final-safety success criteria:

- `runner_audit` has no unexplained final qf losses versus `online`;
- any final qf loss is traceable to a specific local replacement and arm;
- `pressure_guarded` does not materially worsen upper-tail pressure;
- if a `quality_guard` implementation is added, guarded runs must either match
  or exceed `online` final quality within `min_final_quality_delta`.

Secondary criteria:

- stable cases stay unchanged or improve;
- low-margin trigger density is higher for committed/replaced candidates than
  for rejected candidates.

Stop criteria:

- changed candidates are rare after adaptive triggering;
- qf-positive local replacements do not survive to final qf gains;
- local-qf-positive replacements frequently cause final qf losses and no cheap
  guard separates wins from losses;
- the portfolio mostly commits candidates from one arm, making the portfolio
  unnecessary;
- overhead exceeds the budget without higher win rate.

## Expected Interpretation

If the design works, it should be described as a guarded local search extension
to Leiden refinement, not as an objective change. It counteracts known greedy
path-dependence by spending a small speculative budget at unstable local
decisions, while the original CPM objective remains the acceptance criterion.

Resolution perturbation is then a discovery tool for alternate local basins, not
the target criterion. Near-tie randomness, seed perturbation, and node-order
control are complementary discovery tools for the same guarded local search
framework.
