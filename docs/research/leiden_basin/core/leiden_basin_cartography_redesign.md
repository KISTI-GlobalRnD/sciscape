# Leiden Objective-Landscape And Basin Diagnostics

Status: Track C redesign draft
Date: 2026-05-27
Scope: standalone Leiden basin R&D under the SciScape research workspace

This document resets Track C away from c0-local operator continuation. It also
keeps Track C independent from Track A and Track B. The research object here is
not multi-layer consensus and not hierarchy repair. It is the optimization
landscape exposed by Leiden-style CPM search.

The strongest version of Track C is:

> Leiden can converge to multiple endpoint identities under controlled seeds,
> iterations, candidate perturbations, or local edit probes. If those endpoints
> can be grouped into observed basins and if walls between those basins can be
> measured, then refinement can become a measured landscape-cartography problem
> rather than blind restart, policy sweeping, or one-fixture operator tuning.

This preserves the original basin-tunneling idea, but the first contribution is
diagnostic: define the basin objects, test whether walls exist between them,
and classify how routes cross or fail to cross those walls. Basin quality is a
later annotation, not the definition.

## Completion Target

Track C is complete when it can be closed as one of three explicit outcomes:

1. `C-basin-definition-paper`: a standalone study that defines Leiden endpoint
   identities, global observed basin candidates, support-local basin candidates,
   basin relations, and route traces.
2. `C-wall-cartography-paper`: a study that shows when observed basin candidates
   are separated by objective walls, support incompatibility, polish reversion,
   or failed direct paths.
3. `C-directed-basin-search-paper`: a later method that reaches a predeclared
   basin candidate more deliberately than broad restart or seed/iteration
   variation.

The default completion target is `C-basin-definition-paper`. Escalate to
`C-wall-cartography-paper` or `C-directed-basin-search-paper` only when the
evidence gates below are met. This prevents Track C from staying open
indefinitely while chasing a general algorithm.

## Claim Boundary

This is not a Track A claim and not a Track B claim.

- Do not use Track A local-review, taxonomy, layer-disagreement, or boundary
  epistemics evidence as Track C predictors.
- Do not use Track B Dongdaemun-post repair evidence, hard-cap behavior, or
  hierarchy-pressure evidence as Track C predictors.
- Do not present Track C results as `Dongdaemun-post` or validated
  `Dongdaemun-refinement` evidence.

Track C may use the same input graph families as fixtures, but it must generate
and evaluate its own optimizer-native evidence: memberships, signatures, traces,
candidate rows, support movement, QF debt/recovery, wall time, p5 evaluations,
and memory HWM.

Existing c0 branch target-growth, tunneling, attachment-margin, and local-handle
artifacts are evidence sources for a redesign. They are not current algorithm
proof.

## Why The Previous Direction Was Too Narrow

The old ordering was:

1. find a promising c0 transition;
2. refine local handles or selector readiness;
3. try to generalize the operator.

That path risks producing a tuned case study. It also makes the hardest claim
first: that a compact operator works generally.

The new ordering is:

1. audit artifact metrics and label-invariant basin evidence;
2. inventory endpoint identities across seeds, iterations, candidates, and graph
   slices;
3. define global observed basin candidates and support-local basin candidates;
4. test whether walls exist between basin candidates;
5. classify routes by whether they cross, bounce, collapse, or remain unknown;
6. evaluate basin quality only after the basin/wall/route map exists.

This changes Track C from "make the c0 selector work elsewhere" to "explain
what basin objects are visible, whether they are separated by walls, and how
routes move between them."

## Central Research Questions

1. What is the primitive basin object visible in the existing artifacts:
   endpoint identity, global observed basin, support-local basin, or only a
   basin-relation proxy?
2. Once basins are defined, is there evidence for a wall between them:
   objective drop, path debt, polish reversion, failed direct transition, or
   incompatible support movement?
3. If walls exist, what kinds of moves cross them: restart, seed variation,
   candidate perturbation, local handle, prefix growth, or another controlled
   route?
4. Which wall-crossing routes are repeatable across fields, graph layers, seeds,
   iterations, or candidate budgets?
5. Only after basin identity, wall existence, and crossing routes are defined:
   which discovered basins are better or worse under quality and cost criteria?
6. Can a later matched intervention reach a chosen basin more deliberately than
   broad restart or seed/iteration variation?

## Methodology v0

This section defines the method that must exist before Track C can be called
complete. It is intentionally narrower than a full operator. The method is a
diagnostic pipeline for Leiden objective-landscape analysis.

### Unit Of Analysis

A Track C case is:

`case_id = graph_slice + graph_kind + seed_family + candidate_budget + polish_mode`

Required case metadata:

- graph slice or fixture identifier;
- graph kind or edge construction used by the Leiden run;
- baseline Leiden settings;
- seed and iteration schedule;
- candidate-budget schedule when candidate perturbations are used;
- p1/p5 or cheap/full evaluation mode;
- artifact root and rerun command.

Case selection must include:

- at least two non-c0 cases for any positive generalization statement;
- at least one negative-control or already-recovered case;
- c0 only as a known diagnostic reference, never as the main proof.

### Primitive Basin Definition

Track C should start from a deliberately primitive definition. A basin is not
yet a full mathematical attraction region. The current object is an observed
endpoint basin:

`observed_basin = cluster(final_polished_endpoint_memberships | fixed case_id, fixed endpoint protocol)`

This definition is intentionally empirical:

- it only covers endpoints that were actually observed under the indexed seed,
  iteration, restart, candidate, or probe protocols;
- it does not claim to characterize the full set of initial conditions that
  would converge to the same local optimum;
- it can be refined as stronger membership, signature, or trajectory evidence
  becomes available;
- unknown parts of the basin boundary should be reported as unknown, not filled
  in by support-distance intuition.

The primitive definition separates three objects that were previously easy to
mix:

- `basin_identity`: whether two final endpoints belong to the same
  label-invariant endpoint cluster;
- `basin_relation`: whether an endpoint is vanilla-near, candidate-like, or
  support-shifted relative to reference endpoints;
- `transition_route`: whether a path pays and recovers objective debt while
  moving through support space.

Only `basin_identity` can support a distinct-basin claim. `basin_relation` and
`transition_route` are useful diagnostic signals, but they must remain proxies
until they are connected to endpoint membership or strong signature evidence.

### Basin-First Exploration Order

Track C must not start by asking which basin is better. It must first decide
what a basin is and whether basins are separated by walls.

The required order is:

1. `endpoint_inventory`: identify final endpoint identities without ranking them
   by quality.
2. `basin_definition`: group endpoint identities into global observed basins and
   support-local basin candidates under declared metrics and thresholds.
3. `wall_detection`: test whether different basin groups are separated by an
   objective wall, support-incompatibility, polish reversion, or failed direct
   path.
4. `wall_crossing`: characterize which route types can cross or fail to cross
   the wall.
5. `basin_evaluation`: only after the first four steps, evaluate which basin is
   better, cheaper, or more useful.

Quality, materiality, and cost are forbidden as basin-definition criteria. They
may annotate a basin only after the basin identity and wall evidence are fixed.

### Endpoint Generation

Each case should expose comparable endpoint families:

- `seed_control`: standard Leiden with controlled seed changes;
- `iteration_control`: standard Leiden with iteration or convergence variants;
- `restart_control`: broader restart or multi-start baseline when available;
- `candidate_endpoint`: endpoint from a candidate perturbation or candidate
  evaluation path;
- `probe_endpoint`: endpoint from a compact intervention probe, only after
  earlier gates justify it.

For Phase 1, only existing endpoints should be indexed. New endpoint generation
is blocked unless the required evidence cannot be reconstructed from existing
artifacts.

### Basin Representation

The method distinguishes three basin evidence levels:

- `exact_membership`: final memberships are available and can be compared
  directly after label alignment.
- `strong_signature`: compact support, endpoint, or boundary signatures are
  available and stable enough to group endpoints.
- `proxy_signature`: partial signatures or trace summaries are available, but
  the result must be labeled as proxy evidence.

Basin assignment should produce a conservative status, not only an id:

- `same_observed_basin`: the endpoint is assigned to the same exact or coarse
  basin as the reference endpoint;
- `confirmed_distinct_basin`: exact membership or strong signature evidence
  assigns the endpoint to a different basin;
- `ambiguous_basin`: the distance or signature evidence is near the decision
  boundary;
- `proxy_only`: support movement, endpoint proxy distance, or trajectory evidence
  suggests a relation but does not support basin identity.

Basin assignment must avoid raw label namespace artifacts:

- exact label differences alone are not basin evidence;
- exact changed-node counts must be paired with aligned changed-support or
  best-partner alignment;
- endpoint distance and support distance should be reported separately;
- coarse basin grouping must state its distance metric and threshold.

Minimum basin outputs:

- `exact_basin_id` when full memberships allow it;
- `coarse_basin_id` when signatures are used;
- `evidence_grade`;
- `basin_assignment_method`;
- `basin_assignment_threshold`;
- `basin_assignment_status`;
- `largest_basin_size`;
- `top_basin_dominance`.

### Deferred Basin Evaluation

Basin evaluation is downstream of basin definition and wall analysis.

The first basin-index pass must not rank basins by quality, material gain, cost,
or operator usefulness. It should only record:

- endpoint identity;
- global and support-local basin assignment;
- evidence grade;
- distance metric and threshold;
- wall evidence if present;
- route evidence if present.

Quality and cost fields may be joined later, but they must not decide whether a
basin exists. A basin can be real but low quality; a high-quality endpoint can
still belong to an already-known basin.

### Wall Evidence Set

The method may use only optimizer-native wall evidence:

- immediate objective drop on a route;
- objective debt area or debt duration;
- polish reversion back to the source basin;
- support-incompatibility between basin candidates;
- endpoint-distance spread between route endpoints;
- failed direct transition between two basin candidates;
- seed, iteration, restart, candidate, or probe routes that land in different
  basin candidates.

Wall evidence must be reported before basin quality is inspected.

### Route Label Assignment

Route labels are method outputs, not free-text interpretation.

Each route label row must include:

- target case and route family;
- source basin candidate and target basin candidate;
- label name: `crosses`, `bounces`, `collapses`, `ambiguous`, or `unknown`;
- required basin-assignment evidence;
- required wall evidence;
- confidence: `supported`, `partial`, or `hypothesis`.

A route label is not accepted if it depends on final quality rather than basin
assignment and wall evidence.

### Decision Policy

The method maps each case into one basin-first decision class:

- `endpoint_only`: endpoint identities exist, but basin grouping is unresolved;
- `basin_defined`: global or support-local basin candidates are defined;
- `wall_unknown`: basin candidates exist, but wall evidence is missing;
- `wall_supported`: at least one basin pair has wall evidence;
- `route_classified`: at least one route is labeled against a wall;
- `evaluation_ready`: basin identity, wall evidence, and route taxonomy are fixed
  enough to join quality and cost fields.

This decision policy prevents every quality-positive endpoint from becoming a
basin claim.

### Probe Method, Only After Basin/Wall Readiness

A compact probe is methodologically valid only if it has:

- a predeclared source basin candidate;
- a predeclared target basin candidate;
- a declared wall or unknown-wall hypothesis;
- a predeclared route family;
- final basin assignment after the route;
- a stop rule if route evidence remains `ambiguous` or `unknown`.

Probe families remain experimental until G6. They are not the methodology
baseline and they cannot define the basin after the fact.

### Methodology Gaps Still Open

The method is not complete until these choices are fixed in the Phase 1 report
and calibration pass:

- exact metric used for endpoint distance;
- threshold or clustering rule for coarse basin assignment;
- whether the first pass needs a three-zone rule:
  `same`, `distinct`, and `ambiguous`;
- what counts as wall evidence between two observed basin candidates;
- how to separate wall height, wall duration, and wall-crossing route from
  downstream basin quality;
- required columns for `landscape_case_index.csv`;
- how to represent missing full memberships;
- which existing artifacts form the non-c0 fixture set;
- which negative-control fixture is canonical;
- how often support-distance proxies disagree with exact or strong-signature
  basin assignment;
- when quality/materiality fields are allowed to be joined after the basin-only
  index is complete.

### Current Phase 1 Decision

The Phase 1 basin-only index and review are available at:

- `research/consensus/results/adaptive_refinement/leiden_basin_phase1_index_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_phase1_review_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_definition_calibration_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_route_wall_evidence_join_20260528/`
- `research/consensus/results/adaptive_refinement/direct_pair_route_audit_field34_cc_c0_c2_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_wall_protocol_panel_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_subset_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_field12_path_resolved_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_endpoint_cache_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_replicate_schedule_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_schedule_debug_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_stable_ambiguous_relation_refinement_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_relation_taxonomy_v01_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_clean_distinct_vanilla_context_gap_fill_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_after_gap_fill_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_subset_clean_distinct_after_gap_fill_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_clean_distinct_after_gap_fill_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_route_gate_panel_combined_after_clean_distinct_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_clean_distinct_route_mechanism_review_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_polish_margin_gate_review_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_margin_validation_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_margin_validation_panel_review_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_current_results_review_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_route_label_interpretation_v0_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_route_label_blocker_triage_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_relation_boundary_rule_review_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_relation_review_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_cache_materialization_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_relation_review_after_cache_materialization_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_field34_evidence_eligibility_audit_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_remaining_wall_question_audit_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_cycle_closure_writeup_20260529/`

The index passes the first consistency checks and keeps quality, materiality,
cost, ranking, and operator-success fields out of basin-definition outputs. The
review also shows that the current support-local grouping is threshold
sensitive: `support_tau=0.5` is reproducible as a strict inventory threshold,
but it is not stable enough to become the final basin definition.

Decision: do not proceed directly to wall cartography yet. The next method step
is basin-definition calibration:

1. Accept `endpoint_identity` for clean field12, field26, and field30 rows.
2. Treat `support_tau=0.5` as same-zone inventory, not a final boundary.
3. Test a three-zone support-local relation:
   `same_support_local`, `ambiguous_support_local`, and
   `distinct_support_local`.
4. Filter field34 zero-support and duplicate endpoint rows before using field34
   as a basin-definition fixture.
5. Keep global observed basin assignment unresolved until a global
   endpoint-distance rule is accepted.
6. Promote wall evidence only for basin pairs whose source and target
   assignments are not ambiguous.

Calibration outcome: the first calibration pass produces 170 accepted endpoint
identities, 729 identity-pair relation rows, 509 ambiguous identity-pair rows,
206 distinct support-local pair rows, and 3 route-join candidate pair rows. The
3 route-join candidates are the narrow Phase 2 input. They are wall candidates,
not wall evidence.

Route-wall join outcome: the first evidence join scans existing field34/cc route
artifacts for those 3 route-join candidates and promotes no supported wall
claim. `field34_all_edges_cc_cosine_budget12:c0-c2` has route metrics on both
endpoint sides and is therefore the strongest partial pair. `c1-c2` and `c2-c4`
have only one endpoint side covered and remain ambiguous.

Direct pair-route audit outcome: the c0-c2 audit inspects the existing route
artifacts for a direct cross route between the two calibrated endpoint
identities. It finds 0 direct cross-route rows, 153 self-endpoint route rows,
and 1 direct pair-context row. The verdict is
`no_direct_pair_route_self_routes_only`, so the current wall claim status
remains `no_wall_claim`.

Decision: do not broaden to all distinct support-local pairs and do not move to
basin evaluation. The local missing object for c0-c2 is a direct pair-route
artifact, but that artifact should be treated as a diagnostic control rather
than the next primary research unit.

Scope correction: that c0-c2 replay should not become the next primary research
unit. The direct audit shows that existing artifacts are too one-sided, but it
does not justify narrowing Track C back to c0. The next object is the
representative wall-protocol panel:

- 23 calibrated panel pairs;
- 4 fields and 3 source labels;
- 8 distinct high-support representatives;
- 10 ambiguous boundary representatives;
- 2 same-zone controls;
- 3 existing-route diagnostic controls, including c0-c2.

The panel attaches the same W0-W5 wall protocol to every retained pair:
endpoint identity confirmation, direct pair-route trace, objective wall trace,
support movement trace, polish reversion check, and route label assignment.
c0-c2 is retained as a diagnostic control inside this panel, not as the center
of the research design.

Uniform subset outcome: the first W0-W5 execution subset contains four pairs:
c0-c2 as the existing-route control, one field12 distinct probe, one field12
ambiguous boundary probe, and one field30 same-zone control. All four have W0
endpoint identities. A first coarse uniform runner has now emitted W1-W5 rows
for all four pairs with 0 runner errors. This runner is a protocol milestone,
not a wall-claim milestone, because its route schedule is a one-step
support-closure jump.

Runner maturity outcome: the two field12 pairs have also been rerun with a
path-resolved 18-step route schedule. Endpoint and baseline memberships are
cached, pair progress is written to JSONL, and the `c5-c7` cache-hit smoke
confirms baseline and endpoint cache reuse. Both field12 pairs retain the same
partial label: `direct_route_reaches_target_and_polish_stays`. This improves
the runner maturity but, before the replicate pass below, still stops short of
a supported wall claim because route-order stability and route controls had not
been tested.

Route-schedule replicate outcome: the first replicate pass now records
`route_schedule` through the W1-W5 artifacts and reruns the two field12 pairs
plus the field30 same-zone control under `target_size_desc`, `target_size_asc`,
and `target_label_asc`. `field12 c3-c6` is stable across all three schedules.
`field12 c5-c7` is not stable: `target_size_asc` changes the label to
`direct_route_unassigned` with `no_wall_claim`. The same-zone control remains
`control_no_wall_claim`. A companion c0-c2 schedule debug run is also
schedule-sensitive. This makes route-order invariance a required L4 gate rather
than a minor implementation detail. The runner now writes
`uniform_route_schedule_claim_rows.csv`; its `wall_claim_gate_status` field is
the promotion gate, not the individual per-schedule route labels.

Expanded route-gate outcome: the runnable control subset now covers 7 pairs.
This adds a second field12 distinct probe and two field30 ambiguous controls.
The revised gate requires both route-order stability and a non-ambiguous basin
relation. Under that stricter gate, only `field12 c1-c6` currently receives
`passes_schedule_invariance_distinct_partial_wall_evidence`. Stable ambiguous
rows are retained as basin-definition evidence, not supported wall evidence.

Wall-panel context coverage outcome: the full 23-pair wall panel has now been
audited before wider route execution. Seventeen pairs pass runner preflight:
candidate rows, endpoint indices, vanilla rows, and graph dirs are present.
Six pairs are blocked by missing matching vanilla case rows. Among distinct
support-local pairs, zero not-yet-run rows are immediately ready for W1-W6
route-order gates after field hygiene checks; four field34 rows passed runner
preflight but required field34/tiny-support hygiene review first; four needed
context first; two were already route-order-sensitive controls; and one was the
current distinct partial-wall gate. The later field34 eligibility audit closes
those field34 rows as reference/hold/filtered evidence rather than route-gate
candidates. All 10 ambiguous support-local rows remain relation-refinement
targets before any wall promotion; three of them already have stable route
evidence and therefore deserve the first stronger basin-identity check.

Stable ambiguous relation refinement outcome: the three stable ambiguous rows
now have cached full-membership exact-support checks. None is route-promotion
eligible under the current hard same/distinct thresholds. Two rows are
near-distinct boundary cases with exact support distances just below `0.75`
(`0.749954` and `0.749216`), and one row is a near-same boundary case just
above `0.5` (`0.507791`). This shifts the bottleneck from runner maturity to
the basin-relation boundary rule.

Basin relation taxonomy v0.1 outcome: the boundary-aware relation taxonomy has
been generated at
`research/consensus/results/adaptive_refinement/leiden_basin_relation_taxonomy_v01_20260528/`.
It keeps the hard same/distinct thresholds as the current wall-promotion gate
but splits the generic ambiguous bucket into explicit review statuses. Across
the 23-pair wall panel, 11 rows remain `distinct_support_local_current_rule`, 5
ambiguous rows become boundary-review cases, 5 remain middle ambiguous holds,
and 2 remain same/control rows. The taxonomy creates 0 new wall-promotion
eligible rows.

## Claim Ladder

Track C should advance only one rung at a time.

| Level | Claim | Evidence needed | Status |
| --- | --- | --- | --- |
| L0 | Metric hygiene | aligned support, endpoint distance, source pointers, rerun commands | Phase 1 index passes first checks |
| L1 | Endpoint inventory | final endpoint identities are observed without quality ranking | available for clean field12/field26/field30 rows |
| L2 | Basin definition | endpoint identities are grouped into global or support-local basin candidates under declared metrics | pair-level calibration plus relation taxonomy v0.1 available; boundary-review rows remain blocked from wall promotion |
| L3 | Wall existence | distinct basin candidates show objective, support, or polish barriers | wall-protocol panel prepared; 0 supported wall claims |
| L4 | Wall-crossing routes | route types can be classified by whether they cross, bounce, or collapse | 11-pair route-gate panel exists; full 23-pair context coverage has 0 immediately runnable not-yet-run distinct candidates, field34 eligibility audit has 0 field34 route-gate candidates, and relation taxonomy v0.1 keeps boundary-review rows blocked |
| L5 | Basin evaluation | discovered basins can be compared for quality and cost after identity/wall analysis | deferred |
| L6 | Directed basin search | compact method reaches a chosen basin more deliberately than broad search | future work |

Do not skip from L1/L2 to L5/L6. A basin-diagnostics paper is valid if Track C
only defines basin objects and wall structure.

## Completion Gates

Gate G0: metric hygiene

- Every retained case has aligned support or best-partner alignment when exact
  changed-node counts are present.
- Endpoint distance and support distance are reported or explicitly marked
  unavailable.
- Membership evidence grade is assigned.
- Source artifacts and rerun pointers are recorded.

Gate G1: endpoint inventory

- At least two non-c0 cases have final endpoint identities recorded.
- Endpoint identity is not raw label accounting.
- Each endpoint identity records the endpoint protocol that produced it.

Gate G2: basin definition

- Endpoint identities are grouped into global observed basin candidates and,
  separately, support-local basin candidates.
- Each grouping states its distance metric, threshold, and ambiguity rule.
- At least one negative-control case is included to show that the metric does
  not label every slice as basin-rich under the chosen definition.

Gate G3: wall existence

- At least one case has two basin candidates with evidence of a wall between
  them.
- Wall evidence may include objective drop, path debt, polish reversion,
  support-incompatibility, or failed direct transition.
- Wall evidence is reported independently of final basin quality.

Gate G4: wall-crossing route taxonomy

- At least two route classes are distinguishable, such as direct transition,
  restart, candidate perturbation, local handle, or prefix growth.
- Each route is labeled as `crosses`, `bounces`, `collapses`, or `unknown`.
- The labels depend on basin assignment and wall evidence, not quality ranking.

Gate G5: basin evaluation

- Quality and cost fields are joined only after G1-G4 pass.
- Basins are compared only after their identity and wall relationships are fixed.
- This gate can say which basin is better, but it cannot redefine what a basin
  is.

Gate G6: directed basin-search claim

- A compact search or intervention reaches a predeclared basin candidate.
- It does not rely on c0-only behavior.
- It is compared against broad restart or seed/iteration variation only after
  the target basin has been defined independently.

If G0-G2 pass but G3-G6 fail, write a basin-definition paper. If G0-G4 pass but
G5-G6 fail, write a wall/cartography paper. Only G0-G6 supports a directed
basin-search method claim.

## Working Hypotheses

H1: Some slices expose multiple membership-distinct Leiden endpoints that are
not artifacts of raw label numbering.

H2: Basin ambiguity is visible in optimizer-native signals such as
seed/iteration sensitivity, support-distance spread, near-tie local moves,
objective wall/debt area, polish reversion, and endpoint signature diversity.

H3: Different basin candidates may be separated by walls. These walls can appear
as immediate objective loss, accumulated debt, polish reversion,
support-incompatibility, or failed direct moves.

H4: Wall-crossing routes can be classified before asking whether the destination
basin is better. Gate release, forced transplant, local handles, prefix-derived
perturbation, and restart are route families before any one becomes a method.

H5: Basin quality and directed-search value are downstream. A publishable
diagnostic contribution can stop at basin definition plus wall cartography if
the quality or method gates never become convincing.

## Non-Goals

- Do not run more threshold, source, or ranking sweeps without a named mechanism
  question.
- Do not treat c0 selector replay as the default next experiment.
- Do not claim dense graphs have more basins without membership-level or
  signature-level evidence.
- Do not use raw exact `changed_node_count` as basin movement evidence without
  aligned support, endpoint distance, or best-partner alignment.
- Do not use quality, cost, or materiality to decide whether two endpoints are
  in the same basin.
- Do not promote a local operator because it wins on one fixture, one seed, or
  one low-ROI positive `delta_q`.
- Do not import Track A or Track B evidence into Track C scoring, predictors, or
  claim language.

## Workstream C0: Metric Hygiene

Purpose: make sure the basin evidence is not a measurement artifact.

Required checks:

- exact changed-node counts must be paired with aligned changed-support or
  best-partner alignment;
- endpoint distance must be reported for any basin-transition claim;
- support-distance-to-vanilla and support-distance-to-candidate must be
  separated from raw label changes;
- full membership evidence is preferred; compact signatures must be labeled as
  proxies;
- every row needs a source artifact and rerun pointer when available.

Output:

- `metric_hygiene_audit.csv`
- `metric_hygiene_report.md`

Completion gate: G0.

## Workstream C1: Basin Definition

Purpose: build a basin map without asking which basin is better.

Input sources:

- existing final endpoint rows;
- final memberships when available;
- canonical endpoint signatures;
- compact support or boundary signatures when full memberships are missing.

Required rows:

- `case_id`, `field`, `graph_kind`, `seed`, `candidate_budget`;
- `endpoint_id`, `endpoint_protocol`;
- `endpoint_identity`;
- `global_distance_metric`, `global_distance_threshold`;
- `global_observed_basin_id`, `global_assignment_status`;
- `support_distance_metric`, `support_distance_threshold`;
- `support_local_basin_id`, `support_assignment_status`;
- `evidence_grade`;
- `ambiguity_flag`;
- `source_artifact`.

Forbidden in this workstream:

- final quality;
- material gain;
- cost;
- ranking regret;
- operator success.

Completion gates: G1 and G2.

## Workstream C2: Wall Cartography

Purpose: decide whether basin candidates are separated by walls.

Wall evidence families:

- objective drop or temporary objective debt along a path;
- debt duration or debt area;
- support-incompatibility between basin candidates;
- polish reversion back to the source basin;
- failed direct transition between two basin candidates.

Required rows:

- `case_id`;
- `source_basin_id`;
- `target_basin_id`;
- `basin_level`: `global_observed` or `support_local`;
- `wall_evidence_type`;
- `wall_metric`;
- `wall_value`;
- `route_id` when wall evidence comes from a path;
- `wall_assignment_status`: `supported`, `absent`, or `unknown`;
- `source_artifact`.

Quality is not a wall metric. A wall can separate two worse basins, two better
basins, or two basins whose quality is not yet inspected.

Completion gate: G3.

## Workstream C3: Route Taxonomy

Purpose: classify how paths interact with basin walls before any quality
comparison.

Route families:

- seed, node-order, and iteration variation;
- broad restart or random perturbation;
- candidate perturbation;
- forced transplant;
- gate or context release;
- local handle or label-coherent bundle;
- prefix-derived perturbation.

Route labels:

- `crosses`: starts near one basin candidate and ends in another;
- `bounces`: moves toward a wall but returns to the source basin;
- `collapses`: loses basin-relation progress during polish;
- `ambiguous`: endpoint or wall evidence is insufficient;
- `unknown`: required artifacts are missing.

Each label must cite endpoint assignment, support relation, and wall evidence.
Quality and cost are not required for route labels.

Completion gate: G4.

## Workstream C4: Deferred Basin Evaluation

Purpose: compare basins only after C1-C3 define basin identity, wall structure,
and route behavior.

Allowed fields:

- quality;
- materiality;
- cost;
- ranking regret;
- seed/restart comparison.

Rule:

- these fields may annotate a basin;
- they may not change basin ids;
- they may not turn a support relation or route trace into basin identity.

Completion gate: G5.

## Workstream C5: Directed Basin Search Lab

Purpose: test whether a predeclared basin candidate can be reached deliberately.

Requirements:

- target basin candidate is defined before the route is evaluated;
- route family is predeclared;
- seed/restart paths are comparison routes, not basin definitions;
- output is basin assignment first, quality/cost second.

Completion gate: G6.

## Completion Roadmap

### Phase 1: Basin-Only Index, No New Replay

Purpose: close G0-G2 from existing artifacts without using quality, materiality,
or operator success.

Scope:

- field30 multi-method budget12 signature artifacts;
- field26 citation-embedding signature/support artifacts;
- field34/cc c0 diagnostic artifact as a positive reference, not the main proof;
- at least one already-recovered or low-signal negative-control fixture.

Deliverables:

- `landscape_case_index.csv`
- `metric_hygiene_audit.csv`
- `basin_cartography_case_index.csv`
- `basin_cartography_summary.json`
- `leiden_landscape_diagnostics_report.md`

Decision:

- If G1 fails, close Track C as endpoint diagnostics only.
- If G1 passes but G2 fails, write an endpoint-identity note and stop before
  wall analysis.
- If G2 passes, proceed to wall cartography.

### Phase 2: Wall Cartography And Route Audit

Purpose: close G3-G4 without ranking basins by quality and without creating a
new operator.

Scope:

- use only rows indexed in Phase 1;
- attach available path, polish, support, and route evidence;
- classify candidate pairs as wall-supported, no-wall, or unknown;
- classify routes as `crosses`, `bounces`, `collapses`, or `unknown`.

Deliverables:

- `wall_evidence_rows.csv`
- `route_taxonomy_rows.csv`
- `basin_wall_cartography_report.md`
- updated `leiden_landscape_diagnostics_report.md`

Decision:

- If G3 fails, write a basin-definition paper only.
- If G3 passes but G4 fails, write a wall-existence result without route claims.
- If G4 passes, proceed to basin evaluation.

### Phase 3: Basin Evaluation, Still No New Operator

Purpose: close G5 only after basin identity and wall/route structure are fixed.

Scope:

- join quality and cost fields to the already-defined basin index;
- evaluate whether any discovered basin is better, worse, or neutral;
- do not let quality change basin assignment.

Deliverables:

- `basin_evaluation_rows.csv`
- `basin_quality_cost_summary.json`
- updated `basin_wall_cartography_report.md`

Decision:

- If G5 fails, write basin/wall cartography only.
- If G5 passes, proceed to directed basin-search tests.

### Phase 4: Directed Basin Search Or Stop

Purpose: test G6 only after a target basin has been defined independently.

Scope:

- at least two non-c0 cases;
- membership/signature basin assignment;
- predeclared target basin candidate;
- restart or seed/iteration routes as comparison routes;
- no c0-only claim.

Deliverables:

- `directed_basin_search_rows.csv`
- `route_comparison_rows.csv`
- `directed_basin_search_report.md`

Decision:

- If G6 passes, Track C can claim directed basin search.
- If G6 fails, Track C closes as a basin/wall cartography study.

## Minimum Viable Paper

The minimum viable Track C paper does not need basin quality evaluation or a
successful operator.

Core contribution:

- a metric-clean account of Leiden endpoint identities;
- a primitive definition of global observed basins versus support-local basin
  candidates;
- evidence about whether observed basin candidates are separated by walls;
- a route taxonomy showing which paths cross, bounce, collapse, or remain
  unknown;
- explicit examples showing when apparent basin movement is only label artifact,
  support-relation proxy, or route trace.

Required figures or tables:

- table of cases, evidence grades, and basin counts;
- metric-hygiene audit showing exact versus aligned support interpretation;
- global-versus-support-local basin grouping table;
- wall evidence matrix;
- route taxonomy table;
- claim-ladder table showing which gates passed and failed.

Main sentence:

Leiden basin-aware refinement must first define endpoint identities, basin
groups, and walls between them; only then can basin quality or directed search
be evaluated.

Sentence to avoid:

Track C has a validated basin-tunneling algorithm or a better-basin operator.

## First Falsification Experiment

Goal: decide whether Track C has non-c0 endpoint identities, basin definitions,
and wall evidence worth deeper route analysis.

Use existing Track C artifacts first. Do not launch new expensive replays until
the existing evidence is indexed.

Candidate fixture set:

- field30 multi-method budget12 signature artifacts;
- field26 citation-embedding signature/support artifacts;
- field34/cc c0 as the known positive diagnostic fixture;
- one already-recovered or low-signal control fixture from the failure ledger.

Procedure:

1. Build a Track C landscape case index from existing summary CSV/JSON/Markdown
   artifacts.
2. Assign endpoint identity, global observed basin candidate, and support-local
   basin candidate fields.
3. Copy or compute support distance, endpoint distance, basin-group threshold,
   ambiguity flag, and evidence grade.
4. Attach wall-evidence fields only when already available from existing
   artifacts.
5. Produce a short report that says whether C should proceed to wall
   cartography, remain endpoint-only diagnostics, or be downgraded.

This experiment is Phase 1 plus the wall-evidence inventory needed for Phase 2.
It should not add a new replay unless a required source artifact is missing and
the rerun is cheaper than hand-reconstructing the evidence.

Pass condition:

- at least two non-c0 cases show endpoint identities that are not raw label
  artifacts;
- at least one declared grouping rule creates nontrivial global or support-local
  basin candidates;
- wall evidence is present or explicitly marked missing;
- the signal is not explainable by raw label artifacts alone.

Fail condition:

- endpoint identity evidence remains c0-only;
- non-c0 cases collapse to raw label artifacts;
- all apparent basin separations disappear under declared global and
  support-local grouping rules.

## Artifact Contract

The first redesign artifact should be documentation and tables, not a new
operator run:

- `landscape_case_index.csv`
- `metric_hygiene_audit.csv`
- `basin_cartography_case_index.csv`
- `basin_cartography_summary.json`
- `wall_evidence_rows.csv`
- `route_taxonomy_rows.csv`
- `leiden_landscape_diagnostics_report.md`

Later phases may add:

- `basin_evaluation_rows.csv`
- `basin_quality_cost_summary.json`
- `directed_basin_search_rows.csv`
- `route_comparison_rows.csv`
- `directed_basin_search_report.md`

Every retained row should include a rerun or source-artifact pointer. Large raw
traces should remain where they are until the retention manifest is reviewed.

## Decision Rules

- If endpoint identities cannot be separated from raw label artifacts, stop at
  metric hygiene.
- If endpoint identities exist but basin grouping is threshold-unstable, write an
  endpoint-identity result and keep basin definition open.
- If basin groups exist but walls are not visible, write basin cartography only.
- If basin groups and walls exist but routes are unclear, write wall cartography
  and do not proceed to quality evaluation.
- Only after basin groups, walls, and routes are defined should quality/cost
  evaluation or directed basin search be considered.

## Immediate Next Step

Phase 1 basin indexing, basin-definition calibration, the narrow route-wall
evidence join, the direct c0-c2 route audit, the representative wall-protocol
panel, the 4-pair subset, the first coarse W1-W5 runner output, the
path-resolved field12 runner output, the first route-schedule replicate output,
the expanded 7-pair route-gate panel, the 23-pair context coverage audit, the
stable ambiguous relation refinement, the relation taxonomy v0.1, the clean
distinct vanilla context gap-fill, the post-gap-fill context coverage audit, the
clean distinct route-gate runner, the combined route-gate panel, the post-route
full coverage audit, the clean distinct route-mechanism review, the W4 polish
margin gate review, the Methodology v0 margin-validation decision artifact, the
held-out margin-validation review, the current results review, and the
route-label interpretation v0 freeze, and the route-label blocker triage have
been generated. The relation boundary rule review has also been generated for
the route-stable relation-blocked rows. The pending-membership relation review
has also been generated for the two relation-queue rows without cached full
membership evidence.

Current decision:

1. keep `support_tau=0.5` as the same-zone inventory threshold and
   `support_tau=0.75` as the provisional distinct-zone threshold;
2. keep ambiguous support-local pairs out of wall claims;
3. treat the 3 route-join candidates as wall-candidate pairs only;
4. treat c0-c2 as an existing-route diagnostic control, not as the main target;
5. close the existing-artifact c0-c2 audit as `no_wall_claim`;
6. use
   `research/consensus/results/adaptive_refinement/leiden_basin_wall_protocol_panel_20260528/basin_pair_wall_protocol_panel.csv`
   as the wall-evidence surface;
7. use
   `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_subset_20260528/uniform_wall_probe_subset.csv`
   as the first execution subset;
8. use
   `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_20260528/`
   as the first coarse W1-W5 runner output;
9. use
   `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_field12_path_resolved_20260528/`
   as the first path-resolved field12 W1-W5 runner output;
10. use
    `research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_expanded_controls_20260528/uniform_route_schedule_claim_panel_summary.csv`
    as the current route-order and basin-relation gate screen;
11. use
    `research/consensus/results/adaptive_refinement/leiden_basin_relation_taxonomy_v01_20260528/basin_relation_taxonomy_rows.csv`
    as the current boundary-aware relation gate;
12. use
    `research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528/runnable_distinct_pair_queue.csv`
    as the current full-panel route-gate state;
13. treat the two new `field26 bc_cosine` stable distinct partial-wall gates as
    protocol evidence, and treat the two new `field30 emb_knn` rows as
    route-order-sensitive no-wall diagnostics;
14. use
    `research/consensus/results/adaptive_refinement/leiden_basin_clean_distinct_route_mechanism_review_20260528/clean_distinct_route_mechanism_pair_summary.csv`
    before changing route-label rules. It shows the field30 failures are
    post-polish support-assignment losses rather than direct-route target-reach
    failures;
15. use
    `research/consensus/results/adaptive_refinement/leiden_basin_polish_margin_gate_review_20260528/polish_margin_pair_gate_rows.csv`
    as a diagnostic-only support-margin triage. It can distinguish
    boundary-sensitive route holds from harder support-loss holds, but it must
    not relax wall promotion yet;
16. use
    `research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_margin_validation_20260528/methodology_v0_route_gate_decision_rows.csv`
    as the current Methodology v0 route-gate decision table. Its 4-pair
    `margin_validation_panel.csv` is the only immediate validation panel, and
    it cannot promote wall claims during validation;
17. use
    `research/consensus/results/adaptive_refinement/leiden_basin_margin_validation_panel_review_20260529/margin_validation_pair_results.csv`
    as the held-out margin-validation result. The boundary-sensitive holds
    survive as an uncertainty route class; support-loss contrasts split into
    one repeated hard-loss example and one mixed example. No wall claim changes;
18. use
    `research/consensus/results/adaptive_refinement/leiden_basin_current_results_review_20260529/current_pair_state_ledger.csv`
    as the reconciled current 23-pair surface;
19. use
    `research/consensus/results/adaptive_refinement/leiden_basin_route_label_interpretation_v0_20260529/route_label_interpretation_rows.csv`
    as the frozen 11-pair route interpretation surface. It separates
    partial-wall protocol evidence, relation-blocked route evidence,
    boundary-sensitive uncertainty, hard versus mixed no-wall contrasts, and
    same-control rows; all rows remain `no_wall_promotion`;
20. use
    `research/consensus/results/adaptive_refinement/leiden_basin_route_label_blocker_triage_20260529/route_label_blocker_triage_rows.csv`
    as the current blocker triage surface. It has 10 relation-definition queue
    rows, 8 field34 hygiene queue rows, 7 wall-evidence question hold rows, and
    0 immediate route-execution rows;
21. use
    `research/consensus/results/adaptive_refinement/leiden_basin_relation_boundary_rule_review_20260529/relation_boundary_rule_review_rows.csv`
    as the current decision on route-stable relation-blocked rows. The current
    hard same/distinct gate remains accepted; threshold snapping and route
    stability override are rejected;
22. use
    `research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_cache_materialization_20260529/pending_membership_cache_relation_rows.csv`
    and
    `research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_relation_review_after_cache_materialization_20260529/pending_membership_relation_review_rows.csv`
    as the current decision on pending-membership relation checks. Full
    memberships are now cached and reproduce the source support hashes, but both
    rows remain boundary-review holds;
23. use
    `research/consensus/results/adaptive_refinement/leiden_basin_field34_evidence_eligibility_audit_20260529/field34_queue_projection_rows.csv`
    as the current field34 hygiene decision. Field34 rows are reference, hold,
    or filtered evidence under the current gates: there are 0 field34
    route-gate candidates, 0 promoted wall claims, and 0 immediate
    route-execution rows;
24. use
    `research/consensus/results/adaptive_refinement/leiden_basin_remaining_wall_question_audit_20260529/remaining_wall_question_rows.csv`
    as the current closure decision for wall-evidence execution. Under the
    fixed gates it has 0 executable non-field34 route candidates and 0
    non-field34 wall-promotion candidates;
25. use
    `research/consensus/results/adaptive_refinement/leiden_basin_cycle_closure_writeup_20260529/track_c_cycle_closure_report.md`
    as the current closure write-up. The closure is scoped to the current
    23-pair wall surface and does not close the full 206-pair calibration
    universe;
26. keep all boundary-review rows blocked from wall promotion until the boundary
    rule is fixed;
27. do not inspect quality or cost until wall evidence exists.

The next implementation is still not another c0/c2 replay and not basin
evaluation. The route-label interpretation has now been frozen after held-out
margin validation. The first mechanism review showed a W4 polish
support-assignment boundary: field30 direct routes reach a target-like
pre-polish state, but some schedules polish to target support distance above
0.5. The held-out margin review supports a boundary-sensitive uncertainty route
class, but it also shows that support-loss contrasts are not uniform: field34
c0-c2 repeats hard support loss, while field30 c6-c10 is mixed. The next method
problem is now narrower: resolve relation-definition blockers before running
more route probes. The route-stable relation review keeps those rows blocked:
exact memberships differ, but support-local distances remain inside the
predeclared middle zone. The pending-membership cache materialization closes
that cache blocker as well: full memberships reproduce the source support
hashes, but both field26 pending rows remain inside the predeclared middle zone.
The field34 eligibility audit closes the hygiene blocker as reference, hold, or
filtered evidence rather than opening a route batch: all 8 field34 queue rows
remain non-executable, with 0 route-gate candidates and 0 wall promotions. The
remaining decision was therefore whether any non-field34 wall-evidence question
still survived the fixed basin-relation gates. The remaining wall-question audit
answers no for execution: 15 non-field34 rows contain 0 executable route
candidates and 0 wall-promotion candidates. The frozen protocol, uncertainty,
and no-wall rows are reference examples or constraints, not immediate execution
targets. The current cycle should close as basin-definition and wall-protocol
evidence unless the boundary-band definition is explicitly reopened. The cycle
closure write-up freezes that as a scoped claim: it closes the current 23-pair
wall surface and blocker chain, not the full 206-pair calibration universe.
