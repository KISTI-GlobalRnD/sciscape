# SciScape Failed Direction Ledger

Status: active guardrail document
Date: 2026-05-27
Scope: research directions that should not be repeated without new evidence or
an explicit mechanism question.

This is not a list of useless work. Many items below produced useful
diagnostics. The purpose is to prevent the same exploratory path from being
re-run as if it were still an open primary direction.

## How To Use This Ledger

- Before launching a new adaptive-refinement run, check whether it matches a
  failed pattern below.
- If it does, do not rerun it unless the proposed run has a new mechanism
  question and a clear pass/fail criterion.
- Keep negative-control summaries indexed even when raw traces are archived.
- A failed direction can be reopened only when the listed revisit condition is
  satisfied.

## Failure Labels

| Label | Meaning |
| --- | --- |
| `closed` | Do not continue this direction as a primary path. |
| `negative-control` | Keep as a control or audit baseline, not as a method. |
| `merge-as-diagnostic` | Keep evidence, but fold it into another track. |
| `needs-screen-first` | Do not run expensive replay until a cheap screen says it is ready. |
| `claim-boundary` | The artifact is useful, but the tempting claim is invalid. |

## Adaptive Refinement Failures

### F1. More Policy Or Threshold Sweeps Without A Mechanism Question

Status: `closed`

Failed hypothesis:

Changing `best_qf`, `risk_adjusted`, threshold, source, or ranking policy would
by itself reveal the next algorithmic improvement.

What happened:

The work repeatedly drifted into candidate-selection and parameter tuning
without proving that the underlying transition rule changed. Positive
`delta_q` or better ranking on a slice was not enough to establish a new
algorithm.

Decision:

Do not run another threshold/source/policy sweep unless the run answers a
mechanism question such as first trajectory divergence, near-tie moves, basin
support movement, or selector readiness.

Revisit only if:

- the proposed sweep has a named mechanism hypothesis;
- it reports material gain, cost-adjusted gain, p5 count, wall time, and memory
  HWM;
- it can change a future operator decision, not just tune a score.

Primary anchors:

- `docs/research/leiden_basin/leiden_multibasin_research_guardrails.md`
- `research/consensus/TODO_SCISCI_ADAPTIVE_REFINEMENT.md`

### F2. Direct-Node Closure Shrink As A Main Basin Operator

Status: `closed`

Failed hypothesis:

Shrinking direct support nodes from vanilla toward a candidate footprint would
produce a compact quality-winning support transition.

What happened:

Closure shrink could produce quality-positive rows, but they stayed
vanilla-near. Diagnostic labels did not find `quality_win_support_shift` rows.
The best rows were quality wins in the wrong basin, not candidate-like
transitions.

Decision:

Do not promote direct-node-only closure shrink as a mechanism. Keep it as a
negative control for basin-transition claims.

Revisit only if:

- a new source has a support-shift gate already passed by direct shrink;
- the row beats seed/iteration controls on material or cost-adjusted value;
- closure context is explicitly priced.

Primary anchors:

- `research/consensus/scripts/leiden_basin/transition_routes/transition_operators/run_leiden_basin_transition_closure_operator_pilot.py`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_operator_pilot_field34_cc/`

### F3. Label-Internal Repair As A Main Basin Operator

Status: `closed`

Failed hypothesis:

Repairing high-ratio candidate labels internally would produce a distinct
candidate-near or support-shift basin transition.

What happened:

The rows stayed candidate-near or quality-negative. Polish rows were dominated
by seed controls, and no `quality_win_support_shift` rows were found.

Decision:

Archive as a negative control. Do not run more label-internal repair pilots as
the main path.

Revisit only if:

- a selector-screen row identifies a specific label with multiple positive
  local handles;
- the proposed repair is a joint bundle, not isolated label-internal polish;
- aligned support and endpoint-distance metrics are included.

Primary anchors:

- `research/consensus/scripts/leiden_basin/transition_routes/closure_context/run_leiden_basin_transition_label_internal_repair_pilot.py`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_label_internal_repair_pilot_field34_cc/`

### F4. Sequential Stage2 Recovery After Compact Target Move

Status: `closed`

Failed hypothesis:

After a compact target-only tunneling step, local context opening or local
candidate-label transplant around those selected target nodes would recover the
remaining QF lag.

What happened:

Context-only rows were exact no-ops. Candidate/current/vanilla label
transplants were no-ops in most cases, and boundary-shell transplants created
large pre-polish QF debt that bounded polish reverted. The stage2 rows remained
`stage2_no_recovery`.

Decision:

Do not continue sequential "target move then local recovery" as the main
operator family. Use joint bundle selection before polish instead.

Revisit only if:

- a new source-screen says the source has coherent label-completion handles;
- the action activates target and companion context together;
- the output compares against stage1 and seed controls.

Primary anchors:

- `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/run_leiden_basin_attachment_margin_stage2_recovery.py`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_stage2_recovery_field34_cc_c0_p6_p8_p10_v2/`

### F5. Broad Gate-Only Context Release As The Operator

Status: `merge-as-diagnostic`

Failed hypothesis:

Opening a broad vanilla-label gate would be the useful recovery operator.

What happened:

The broad gate exposed a real signal, but the useful endpoint was reproduced
more cheaply by compact target-mutability. The gate was diagnostic: it revealed
nodes such as `2890`, but it was too expensive to be the operator.

Decision:

Do not treat broad gate release as the method. Use it only to discover target
handles or source-side context for selector features.

Revisit only if:

- compact target-handle replay fails but gate context has a unique material
  quality gain;
- the gate can be narrowed by a coherent component or label rule;
- mutable-node cost is explicitly compared.

Primary anchors:

- `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/profile_leiden_basin_post_gate_gate_trace.py`
- `research/consensus/scripts/leiden_basin/operator_probes/gate_release/run_leiden_basin_gate_release_operator_probe.py`

### F6. Low-Wall Side Routes As Finished Operators

Status: `negative-control`

Failed hypothesis:

Lower QF wall side routes should be better operator candidates than a higher
wall route.

What happened:

Side routes could cross the support gate with lower wall, but stayed
quality-negative. The higher-wall p9 route was more useful because it recovered
quickly. Low wall alone was not a success metric.

Decision:

Keep low-wall side routes as recovery-target diagnostics, not as finished
operator rows.

Revisit only if:

- a post-gate recovery move turns a low-wall route QF-positive;
- debt area and recovery slope beat the high-wall shortcut;
- the row passes seed/iteration controls.

Primary anchors:

- `research/consensus/scripts/leiden_basin/transition_routes/route_wall/profile_leiden_basin_side_route_expansion.py`
- `research/consensus/scripts/leiden_basin/transition_routes/tunneling_pathways/profile_leiden_basin_pathway_debt_area.py`
- `research/consensus/scripts/leiden_basin/transition_routes/tunneling_pathways/profile_leiden_basin_tunneling_evidence.py`

### F7. Repeated c2 p6/p8 Selector Replay Without Readiness

Status: `needs-screen-first`

Failed hypothesis:

Rerunning the local handle selector on c2 variants would test whether the c0
selector generalizes.

What happened:

c2 variants were already recovered controls or had too few positive non-source
handles. They did not expose the c0 label-competition or label-completion
mechanism.

Decision:

Do not spend replay budget on c2 p6/p8 again unless the source-screen reports
`selector_test_ready` or `coherent_label_completion_probe`.

Revisit only if:

- a fresh non-c0 source slice passes `recovery_contexts` source screening;
- multiple positive-margin handles or a same-label completion group is present;
- evaluation-core leakage is disabled for non-c0 rows.

Primary anchors:

- `research/consensus/scripts/leiden_basin/operator_probes/selector_sources/screen_leiden_basin_selector_sources.py`
- `research/consensus/scripts/leiden_basin/operator_probes/selector_sources/run_leiden_basin_selector_source_screen_batch.py`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_selector_source_screen_batch_field34_cc_non_c0_v0/`

### F8. Raw Exact Changed-Node Counts As Basin Evidence

Status: `claim-boundary`

Failed hypothesis:

Large exact `changed_node_count` values represented large basin movement.

What happened:

Focused replay showed large exact changes were often label namespace accounting.
Label-invariant aligned support and endpoint-distance metrics told a much
smaller and more accurate story.

Decision:

Do not make basin-level claims from raw exact changed-node counts. Require
aligned support, alignment-error, changed-support, or endpoint-distance columns.

Revisit only if:

- exact labels are canonicalized or paired with best-partner alignment;
- exact and aligned metrics are reported side by side;
- the claim explicitly states whether it is about implementation labels or
  basin support.

Primary anchors:

- `research/consensus/scripts/leiden_basin/evidence_panels/audits/audit_leiden_basin_evaluation_metrics.py`
- `research/consensus/scripts/leiden_basin/basin_signatures/endpoint_flips/summarize_leiden_basin_recomputed_operator_metrics.py`

### F9. Naive Pull-Ranked Unit Growth

Status: `closed`

Failed hypothesis:

Coherent target units selected by simple pull-ranked top-k would beat
node-level staged target growth.

What happened:

Unit definitions were diagnostically useful, but naive unit growth was weaker
than node-level staged target growth and expensive to branch deeply.

Decision:

Do not run deeper unit-aware beam searches without cached or incremental
scoring and a cost-aware unit selector.

Revisit only if:

- target-unit scoring is cached or incremental;
- marginal support progress per mutable node is reported before polish;
- the run is narrower than the previous full unit branching attempts.

Primary anchors:

- `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/profile_leiden_basin_target_units.py`
- `research/consensus/scripts/leiden_basin/transition_routes/closure_context/search_leiden_basin_transitions.py`

### F10. Raw Max-Gap Elbow Without Pull-Fraction Guard

Status: `closed`

Failed hypothesis:

The largest pull-score gap alone can choose a good target-growth elbow.

What happened:

The raw max-gap rule often selected `k=1` with only about `15-22%` cumulative
pull. It was too aggressive and removed nodes needed to cross the support gate.

Decision:

Use guarded elbow rules only, and treat elbow as a scheduler rather than an
acceptance policy.

Revisit only if:

- the selection retains enough cumulative pull;
- escalation/backfill variants remain alive in the branch search;
- support-gate and QF recovery are both reported.

Primary anchors:

- `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/profile_leiden_basin_target_elbows.py`
- `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/evaluate_leiden_basin_target_elbow_polish.py`
- `research/consensus/scripts/leiden_basin/basin_signatures/branch_growth/search_leiden_basin_branch_target_growth.py`

## Dongdaemun-Post And Hierarchy Repair Boundaries

### F11. Branch-Adaptive Tau As A Validated Default Method

Status: `claim-boundary`

Failed hypothesis:

Branch-adaptive critical-gamma tau selection was ready to become a validated
default hierarchy method.

What happened:

The frozen branch-adaptive pilot is negative/diagnostic: no tau setting
selected accepted candidates. The idea remains useful as a structural framing
and future diagnostic, not as a current validated method.

Decision:

Merge branch-adaptive critical-gamma into Track B as framing or future work.
Do not present it as a validated Dongdaemun-post method.

Revisit only if:

- branch-adaptive candidates are accepted under exact CPM audit;
- tau sensitivity is reported, not tuned to one point;
- results beat quality-first postprocess on a paper-facing criterion.

Primary anchors:

- `docs/research/leiden_basin/branch_adaptive_quality_first_research_note.md`
- `docs/research/dongdaemun/manuscript/dongdaemun_evidence_map.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/`

### F12. Hard-Cap As The Main Dongdaemun-Post Claim

Status: `claim-boundary`

Failed hypothesis:

Strict hard-cap mode should be the main hierarchy repair policy.

What happened:

Hard-cap often falls back or requires accepting worse tradeoffs. It is useful
diagnostically, but quality-first is the stronger validated default.

Decision:

Keep hard-cap as a diagnostic policy and failure-mode explanation. Do not make
it the main paper method.

Revisit only if:

- a hard-cap variant achieves the cap without material CPM or semantic loss;
- fallback rates are low across fields/seeds;
- quality-first cannot satisfy a required operational constraint.

Primary anchors:

- `docs/research/dongdaemun/manuscript/dongdaemun_evidence_map.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table1_policy_comparison.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md`

### F13. Rust Fast Path As A Research Claim

Status: `merge-as-diagnostic`

Failed hypothesis:

The Rust Dongdaemun fast path could be presented as a speed or method claim
from the first field12 validation.

What happened:

The Rust and Python paths committed different valid results, and timings
reflected different backend behavior rather than equivalent kernels.

Decision:

Keep Rust as implementation support. Do not use first fast-path timing as a
research claim or fused-kernel decision.

Revisit only if:

- Python/Rust phase-level parity is established or intentionally separated;
- timings compare equivalent kernels;
- artifact-writing and move-row parity are addressed when needed.

Primary anchors:

- `research/consensus/results/adaptive_refinement/rust_dongdaemun_fast_path_validation/`
- `docs/research/dongdaemun/core/dongdaemun_rust_test_module_plan.md`

## Leiden Hysteresis Boundaries

### F14. Hysteresis Work Acceleration As Basin Escape

Status: `claim-boundary`

Failed hypothesis:

Boundary perturbation work-acceleration rows prove a new multi-basin escape
algorithm.

What happened:

The strongest claim is smaller: selected perturbations can reduce refinement
work to a target QF. Structural divergence was small or not fully inspected,
and savings must distinguish refinement work from total elapsed cost.

Decision:

Merge hysteresis evidence into Track C as cost/instrumentation support. Do not
present it as a standalone basin-escape method.

Revisit only if:

- structural divergence is large and durable;
- total elapsed cost remains favorable after candidate evaluation;
- the transition rule, not just candidate selection, is changed.

Primary anchors:

- `docs/research/leiden_basin/leiden_hysteresis_work_acceleration_note.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_budget123_20260513/`

## Current Positive Direction To Protect

The current positive Track C direction is not any failed item above. It is:

```text
screen source readiness
  -> select label-coherent handles from local features
  -> test compact forced transplant or joint bundle
  -> evaluate aligned support, endpoint distance, QF, target progress, cost,
     and seed/iteration controls
```

A proposed experiment that bypasses the screen or uses exact changed-node count
as the main success metric should be rejected before it runs.
