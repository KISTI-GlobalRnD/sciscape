# Leiden Basin Surface Claim Schema

Status: design draft
Date: 2026-06-08
Scope: Track C basin-definition and claim-promotion vocabulary only

This document fixes the current working definition of basin evidence after the
`014`/`016`/`005` surface split. It is a claim schema, not an algorithm,
route runner, wall proof, quality evaluation, or method result.

## Core Definition

A basin is not first defined as a final cluster object, but as evidence on a
specified partition-state surface. Stronger basin claims require promotion from
recurrent state signatures to certified local objects, endpoint objects, and
finally typed wall relations.

Operationally:

```text
surface_qualified_basin_evidence:
  Evidence that a partition-state family is recurrent and distinguishable on a
  declared object surface.

surface_qualified_basin_claim:
  A basin claim whose strength is explicitly limited by the highest surface
  level and relation gate that has passed.
```

The word `basin` should therefore be qualified every time it appears in Track C
claim language. Preferred forms are:

- `signature-surface basin evidence`
- `local signature-object evidence`
- `endpoint-object basin evidence`
- `wall-relation evidence`
- `quality evidence`

Unqualified basin language is allowed only in high-level research questions or
when the sentence immediately states the active surface level.

## Three Axes

The current evidence must keep three axes separate.

| Axis | Question | Examples | Claim boundary |
| --- | --- | --- | --- |
| `state` | Is a partition state or signature observed and recurrent? | endpoint signature, route-state signature, transient band signature | Does not imply object identity. |
| `object` | Can that state be interpreted as a local object or endpoint object? | local signature-object, source component, target object, nonendpoint transient | Does not imply a wall. |
| `relation` | What relation connects states or objects? | direct leg, typed ladder, clean object relation, collapse | Does not imply quality or method success. |

Quality and cost are downstream axes. They are not part of basin definition,
object certification, or wall promotion.

## Promotion Ladder

Every case should be located on this ladder before stronger wording is used.

| Level | Name | Required evidence | Allowed wording | Explicitly blocked wording |
| --- | --- | --- | --- | --- |
| L1 | `observed_state` | A partition state or signature is observed under a declared protocol. | observed state | basin, object, wall |
| L2 | `recurrent_state` | The state recurs across declared seeds, starts, fractions, or perturbations. | recurrent state/signature | object, endpoint basin, wall |
| L3 | `local_signature_object` | A recurrent signature can be used as a stable local object on the declared surface. | local signature-object, signature-surface evidence | endpoint-object, wall |
| L4 | `endpoint_object` | Source and/or target endpoint objects are certified under a declared endpoint protocol. | endpoint-object evidence | wall unless relation gate passes |
| L5 | `typed_relation` | Relations are classified as clean, ladder, direct-only, collapse, or unresolved. | typed relation, transition ladder, collapse control | wall-ready unless G6 passes |
| L6 | `wall_ready_relation` | A declared wall-relation gate passes for accepted objects. | wall-ready relation, primitive wall evidence | method, quality, general wall |
| L7 | `method_quality_claim` | Quality/cost/replay/generalization gates pass after wall evidence. | method/quality claim with scope | basin definition shortcut |

Promotion is monotone only in evidence accounting, not in claim language. A
case can pass L3 and L5 while failing L4 or L6. Lower-level success never opens
higher-level claims automatically.

## Required Audit Columns

Every future basin-surface audit should expose these columns, even if the table
also has richer case-specific fields.

| Column | Allowed values | Meaning |
| --- | --- | --- |
| `surface_level` | `state`, `signature_object`, `endpoint_object`, `relation`, `quality` | Highest surface currently being evaluated. |
| `object_status` | `certified`, `split`, `nonendpoint`, `collapse`, `unknown`, `not_applicable` | Object interpretation at that surface. |
| `relation_status` | `clean`, `ladder`, `collapse`, `direct_only`, `unresolved`, `not_applicable` | Relation classification, separate from quality. |
| `claim_status` | `open`, `diagnostic_only`, `blocked`, `closed` | What claim language is currently allowed. |

Recommended companion columns:

- `promotion_level`: one of `L1` through `L7`;
- `promotion_blocker`: short blocker such as `source_family_split`,
  `transient_nonendpoint`, `external_membership_absent`, or `quality_not_tested`;
- `allowed_wording`: concise phrase that can be used in reports;
- `blocked_wording`: concise phrase that must not be used.

## Current Case Map

| pair | highest stable surface | object_status | relation_status | claim_status | Current wording |
| --- | --- | --- | --- | --- | --- |
| `014` | `endpoint_object` | `certified` | `clean` candidate | `diagnostic_only` | endpoint-object primitive wall candidate under the direct-only/recovery-loop surface |
| `016` | `signature_object` | `split` plus `nonendpoint` | `direct_only` plus `ladder` | `blocked` | signature-object transition-band case with certified target local object |
| `005` | `endpoint_object` boundary | `collapse` | `collapse` | `closed` | boundary/collapse control |

The `014` and `016` positives are not contradictory. They are positive on
different surfaces:

- `014` is positive on the endpoint-object / direct-recovery surface.
- `016` is positive on the stable-plateau / signature-object transition-band
  surface.

Neither surface by itself opens a method, quality, full-replay, or general wall
claim.

## Current 016 Interpretation

The current `016` wording should be:

> `016` is a signature-object transition-band case with a certified target
> local object, split source-family components, and a recurrent non-endpoint
> transient; object-wall evidence remains closed.

This wording is intentionally weaker than `endpoint basin`, `wall`, or
`pathway`. The local signature-object certificate gives useful state/object
vocabulary, but it does not resolve endpoint-object identity.

## Design Consequences

1. The next step is not another route execution.
2. The next design gate must decide which object surface is valid for which
   claim tier.
3. Local signature-objects may be accepted as a primitive diagnostic surface,
   but not as endpoint-object or wall evidence.
4. True wall language requires an explicit relation gate over an accepted object
   surface.
5. Quality and cost claims remain out of scope until basin/object/relation
   evidence is fixed.

## Reopen Conditions

Route execution can resume only after one of these design gates passes:

- `signature_object_surface_rule_accepts_diagnostic_only`: local
  signature-objects are accepted as a diagnostic basin-state surface, with
  blocked wall language recorded.
- `endpoint_object_membership_required_for_wall`: a true symmetric
  endpoint-object membership audit is required before wall language.
- `typed_ladder_wall_rule_predeclared`: a typed ladder can be considered
  wall-ready only under a predeclared relation rule and negative controls.

Any new runner must report the required audit columns above and preserve the
claim boundary in its summary JSON and report. New basin-surface audits should
use
`research/consensus/scripts/leiden_basin/materialization/surface_claim_schema_adapter.py`
for the shared required-column vocabulary, value validation, and case mapping.

## First Application Audit

The first read-only application of this schema is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_claim_schema_application_gamma1e5_20260608/`.

It applies the required columns to the current `014`/`016`/`005` surface split:

| pair | `surface_level` | `object_status` | `relation_status` | `claim_status` | `promotion_level` |
| --- | --- | --- | --- | --- | --- |
| `014` | `endpoint_object` | `certified` | `clean` | `diagnostic_only` | `L5_typed_relation` |
| `016` | `signature_object` | `split` | `ladder` | `blocked` | `L3_local_signature_object` |
| `005` | `endpoint_object` | `collapse` | `collapse` | `closed` | `L5_typed_relation` |

All six application gates pass. This does not promote any wall, pathway,
method, quality/cost, full-replay, or route-execution claim. It only fixes the
shared comparison vocabulary for future basin-surface audits. The audit now
records `schema_adapter=surface_claim_schema_adapter.py` in summary/config so
the next audit can reuse the same validation contract rather than re-declaring
it locally.

## Object-Surface Rule Decision

The first adapter-backed rule decision is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_object_surface_rule_decision_gamma1e5_20260608/`.

It applies the schema adapter to the current surface rows and fixes the
object-surface rule:

- local signature-objects are accepted as diagnostic basin-state surfaces only;
- endpoint-object membership is still required for object-wall wording;
- typed ladder evidence is not a wall rule until a separate rule and controls
  are predeclared.

All seven decision gates pass. The rule accepts `016` as a diagnostic
signature-object transition-band surface, retains `014` as existing local
primitive object-wall evidence, and keeps `005` as the closed collapse guard.
It does not promote pathway, wall, method, quality/cost, full-replay, or
route-execution claims.

## Panel Readiness Audit

The first adapter-backed panel-readiness audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_gamma1e5_20260609/`.

It applies the object-surface rule to the current 23-pair first-pass panel.
All nine readiness gates pass. The audit materializes all 23 rows as
schema-valid, but grants scoreable status only to `016`, `014`, `009`, `012`,
`020`, and `005`. `016` is the single diagnostic transition-band reference;
`014` is a different object-wall surface; `009`, `012`, and `020` are strict
analog negative guards; `005` remains the closed collapse guard; and 17 rows
remain not-scoreable surface gaps. This is readiness, not panel-level
generality, and it promotes no wall/pathway/method/quality/full-replay/
route-execution claim.

## Gap-Fill Contract

The next design-only gap-fill contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract_gamma1e5_20260609/`.

It does not broaden the 23-pair panel. It opens only `001` and `007`, because
they are diagnostic-not-scoreable non-strict local-signature rows. It locks
`016` as the single diagnostic transition-band reference and locks `014`,
`009`, `012`, `020`, and `005` as fixed guards. The other 15 screened gaps
remain excluded. All seven contract gates pass. The planned execution surface
is exactly 54 route rows over allowed starts and the fixed bridge-fraction
schedule. A future execution can only classify `001`/`007` as diagnostic
recurrence, scoreable negative, or residual gap; it cannot promote
panel-generality, wall, pathway, method, quality/cost, or full-replay claims.

The execution and audit are materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_gamma1e5_20260609/`
and
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_gamma1e5_20260609/`.

The trace runs all 54 route rows over 8 seeds, producing 432 local fraction rows
and 48 pair/start/seed readouts. Both `001` and `007` retain source-family
starts but show zero finite single-side bands and zero target-like final
states. The audit therefore classifies both rows as
`gap_fill_scoreable_negative_no_recurrence_guard`. The scoreable surface is now
8 rows, the remaining not-scoreable screened gaps are 15, and `016` remains the
single diagnostic transition-band reference.

A follow-up schedule-boundary audit is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_gamma1e5_20260609/`,
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_gamma1e5_20260609/`,
and
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_gamma1e5_20260609/`.
It reopens only `001` and `007` to test whether the 0.5 lower bound created a
schedule artifact. The trace runs 30 route rows over 8 seeds, producing 240
local fraction rows and 48 pair/start/seed readouts. Both pairs become
target-like below 0.5, but neither shows a finite single-side band or
diagnostic recurrence. The audit therefore qualifies both as
`low_fraction_late_target_collapse_guard`. This changes the interpretation of
the two guards from simple no-recurrence negatives to late target-collapse
guards; it still does not promote wall, pathway, panel-generality, method,
quality/cost, full-replay, or route-execution claims.

The direction-setting transition-type panel contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_transition_type_panel_contract_gamma1e5_20260609/`.
It reads the `016` pathway-shape audit, the `001`/`007` low-fraction boundary
audit, and the surface-rule panel. All six contract gates pass. The contract
freezes the rule that target-like collapse is not enough for 016-like evidence:
the positive reference requires a finite adjacent typed transient/single-side
band. It keeps object-surface promotion separate, recommends a typed-ladder
relation-rule contract as the next 016 relation gate, and keeps screened-gap
expansion blocked.

The typed-ladder relation-rule contract is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_gamma1e5_20260609/`.
All six gates pass. This is the first explicit `relation_status=ladder`
wording rule after the transition-type panel: it permits `016` to be described
as a diagnostic typed-ladder relation over local signature-object states, but
it does not permit wall or pathway language. The controls are now explicit:
`001`/`007` block target-endpoint-only widening, `009`/`012`/`020` block
nonfinite strict-analog widening, `005` blocks boundary-collapse widening, and
`014` blocks object-wall transfer to `016`.

The typed-ladder relation-rule application is materialized at
`../../../research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_gamma1e5_20260609/`.
All six gates pass. This is the first schema-valid application of that ladder
rule to the eight-row scoreable surface: `016` moves to
`surface_level=relation`, `object_status=split`, `relation_status=ladder`,
and `claim_status=diagnostic_only`. The application does not change the wall
boundary: `014` remains a separate diagnostic object-wall surface, five rows
remain blocked controls, `005` remains closed, and the 15 screened rows remain
not-scoreable.
