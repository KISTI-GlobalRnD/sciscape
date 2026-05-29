# Leiden Basin Data Inventory

Status: preparation note for Track C Phase 1
Date: 2026-05-28
Scope: existing Track C artifacts only; no new replay and no basin-quality
interpretation

## Purpose

This note maps the current adaptive-refinement artifacts into basin-first data
roles. It does not decide which endpoint is better. It only asks which files can
support:

- endpoint identity;
- global or support-local basin grouping;
- wall evidence availability;
- route trace availability;
- metric hygiene checks.

Quality, materiality, cost, selector success, and operator success are ignored
here even when existing CSVs contain those columns.

## Phase 1 Output

The first basin-only consolidation pass has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_phase1_index_20260528/`

Primary outputs:

- `landscape_case_index.csv`
- `metric_hygiene_audit.csv`
- `basin_cartography_case_index.csv`
- `wall_evidence_rows.csv`
- `route_taxonomy_rows.csv`
- `basin_cartography_summary.json`
- `leiden_landscape_diagnostics_report.md`

This pass includes the combined cross-field review, strict field30 support
review, and strict field26 budget15 support review: 21 case rows, 185 endpoint
rows, 173 endpoint identities, and 162 support-local groups at the current
diagnostic threshold. These are indexed counts under declared rules, not a final
basin count.

## Phase 1 Review Output

The first basin-only review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_phase1_review_20260528/`

Primary outputs:

- `phase1_consistency_checks.csv`
- `support_threshold_sensitivity.csv`
- `support_pair_trizone_counts.csv`
- `field34_filtering_review.csv`
- `phase1_review_summary.json`
- `phase1_review_report.md`

The review passes the consistency checks for the Phase 1 index and confirms
that no quality-like columns are used in the generated basin-first outputs. It
also shows that the current `support_tau=0.5` support-local grouping is not a
final basin definition: among 829 reviewed endpoint pairs, 43 are same-zone at
`support <= 0.5`, 509 are ambiguous at `0.5 < support < 0.75`, and 277 are
distinct-zone at `support >= 0.75`.

Direction decision: do not proceed directly to wall cartography yet. The next
step is a basin-definition calibration pass that keeps endpoint identity
accepted for clean field12, field26, and field30 rows; treats
`support_tau=0.5` as an inventory threshold; introduces a same/distinct/
ambiguous support-local relation; filters field34 zero-support and duplicate
endpoints before basin counting; and promotes wall evidence only for basin
pairs whose source and target assignments are not ambiguous.

## Basin Definition Calibration Output

The first basin-definition calibration pass has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_definition_calibration_20260528/`

Primary outputs:

- `endpoint_identity_rows.csv`
- `candidate_pair_relation_rows.csv`
- `identity_pair_relation_rows.csv`
- `calibrated_basin_case_summary.csv`
- `wall_candidate_pair_rows.csv`
- `route_join_candidate_pair_rows.csv`
- `basin_definition_calibration_summary.json`
- `basin_definition_calibration_report.md`

The calibration keeps the basin definition primitive and pair-local. It filters
zero-support field34 endpoints before identity aggregation, keeps
`support_tau=0.5` as the same-zone threshold, and treats
`support_tau=0.75` as the provisional distinct-zone threshold.

Calibration result:

- 21 cases;
- 185 raw endpoint rows;
- 172 accepted endpoint rows after hygiene filtering;
- 170 accepted endpoint identities;
- 729 identity-pair relation rows;
- 14 same endpoint/support-local rows;
- 509 ambiguous identity-pair rows;
- 206 distinct support-local pair rows;
- 3 route-join candidate pair rows, all in
  `field34_all_edges_cc_cosine_budget12`.

Direction decision: the next wall step should not inspect all 206 distinct
pairs. It should join route/wall evidence only to the 3
`route_join_candidate_pair_rows.csv` rows. Those rows are wall-candidate pairs,
not wall claims.

## Route-Wall Evidence Join Output

The first narrow route-wall evidence join has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_route_wall_evidence_join_20260528/`

Primary outputs:

- `route_join_pair_context.csv`
- `route_wall_artifact_inventory.csv`
- `route_wall_evidence_rows.csv`
- `route_wall_pair_summary.csv`
- `wall_evidence_join_summary.json`
- `basin_wall_evidence_join_report.md`

This pass starts only from the 3 route-join candidate pairs. It scans existing
field34/cc route artifacts and joins candidate-level route metrics without
promoting any basin-quality or cost comparison.

Join result:

- 3 calibrated pair inputs;
- 98 route artifact directories inventoried;
- 371 joined evidence rows;
- 3 direct pair-context rows;
- 368 candidate route-trace rows;
- 1 pair with route metrics on both endpoint sides:
  `field34_all_edges_cc_cosine_budget12:c0-c2`;
- 2 pairs with only one endpoint side covered:
  `field34_all_edges_cc_cosine_budget12:c1-c2` and
  `field34_all_edges_cc_cosine_budget12:c2-c4`;
- 0 supported wall claims.

Direction decision: the next useful experiment is a direct pair-route audit for
`field34_all_edges_cc_cosine_budget12:c0-c2`. Do not broaden to the other
distinct pairs until their route evidence exists.

## Direct Pair Route Audit Output

The direct c0-c2 route audit has been generated at:

`research/consensus/results/adaptive_refinement/direct_pair_route_audit_field34_cc_c0_c2_20260528/`

Primary outputs:

- `direct_pair_route_context.csv`
- `direct_route_candidate_rows.csv`
- `direct_pair_wall_evidence_rows.csv`
- `direct_pair_route_summary.csv`
- `direct_pair_route_audit_summary.json`
- `direct_pair_route_audit_report.md`

This audit checks whether existing route artifacts directly connect the
calibrated c0-c2 endpoint pair. It does not run a new operator and does not
compare basin quality or cost.

Audit result:

- 229 candidate/context rows inspected for
  `field34_all_edges_cc_cosine_budget12:c0-c2`;
- 0 direct cross-route rows;
- 153 self-endpoint route rows;
- 1 direct pair-context row;
- 153 wall-metric rows attached to self-endpoint routes;
- verdict: `no_direct_pair_route_self_routes_only`;
- wall claim status: `no_wall_claim`.

Direction decision: existing artifacts do not support a wall claim for c0-c2.
Locally, the missing object is a direct c0-c2 route artifact. Globally, this
should not become the next primary research unit; it should be handled as one
diagnostic control inside the broader wall-protocol panel below.

## Wall Protocol Panel Output

The representative wall-protocol panel has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_wall_protocol_panel_20260528/`

Primary outputs:

- `basin_pair_wall_protocol_panel.csv`
- `uniform_wall_protocol_steps.csv`
- `wall_protocol_pair_requirements.csv`
- `wall_protocol_panel_summary.json`
- `basin_wall_protocol_panel_report.md`

This pass deliberately steps back from the c0-c2 audit. It uses the 729
calibrated identity-pair relations as the source surface and selects a
representative panel before any further route replay.

Panel result:

- 729 calibrated identity-pair rows considered;
- 206 distinct support-local rows;
- 509 ambiguous support-local rows;
- 14 same or same-identity rows;
- 23 panel pairs selected;
- 4 fields covered;
- 3 source labels covered;
- 3 existing-route diagnostic controls retained;
- c0-c2 retained only as `existing_route_diagnostic_control`;
- 21 wall-candidate panel rows still need a uniform direct pair-route trace.

Uniform wall protocol steps:

- W0: endpoint identity confirmation;
- W1: direct pair-route trace;
- W2: objective wall trace;
- W3: support movement trace;
- W4: polish reversion check;
- W5: route label assignment.

Direction decision: do not make c0-c2 the next primary replay. The next Track C
object is the panel-level wall protocol. A direct c0-c2 artifact is useful only
as one control inside this panel, not as the research center.

## Uniform Wall-Probe Subset Output

The minimal uniform wall-probe subset has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_subset_20260528/`

Primary outputs:

- `uniform_wall_probe_subset.csv`
- `uniform_wall_probe_status_matrix.csv`
- `uniform_wall_probe_execution_manifest.csv`
- `uniform_wall_probe_artifact_contract.csv`
- `uniform_wall_probe_subset_summary.json`
- `uniform_wall_probe_subset_report.md`

This artifact selects four panel pairs for the first uniform W0-W5 test:

- existing-route control:
  `field34_all_edges_cc_cosine_budget12:c0-c2`;
- non-field34 distinct probe:
  `field12_gcc_emb_full_knn30_emb_knn_budget12:c5-c7`;
- ambiguous boundary probe:
  `field12_gcc_emb_full_knn30_emb_knn_budget12:c3-c6`;
- same-zone control:
  `field30_gcc_emb_full_knn30_bc_cosine_budget12:c1-c7`.

Subset readiness:

- all four pairs have W0 endpoint identity available from calibration;
- the first coarse uniform direct pair-route runner output now exists;
- c0-c2 has both legacy field34/cc context and a coarse uniform pair-route
  control trace;
- the three non-c0-control rows have graph/vanilla context available.

Direction decision: the next implementation task is not another c0-c2 replay
and not basin evaluation. The next task is to make the uniform runner more
operational and more path-resolved.

## Uniform Direct Pair-Route Runner Output

The first minimal uniform W1-W5 runner output has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_20260528/`

Primary outputs:

- `uniform_direct_pair_route_rows.csv`
- `uniform_objective_wall_rows.csv`
- `uniform_support_movement_rows.csv`
- `uniform_polish_reversion_rows.csv`
- `uniform_route_label_rows.csv`
- `uniform_wall_probe_runner_summary.json`
- `uniform_wall_probe_runner_report.md`
- `uniform_wall_probe_runner_config.json`

Run summary:

- 4 selected pairs processed;
- 0 runner errors;
- 8 direct-route rows, 8 objective-wall rows, 8 support-movement rows, 4 polish
  reversion rows, and 4 route-label rows;
- all routes use a one-step support-closure schedule
  (`groups_per_step=100000`, `max_route_steps=1`).

Route labels:

- `field34_all_edges_cc_cosine_budget12:c0-c2`:
  `direct_route_unassigned`, `no_wall_claim`;
- `field12_gcc_emb_full_knn30_emb_knn_budget12:c5-c7`:
  `direct_route_reaches_target_and_polish_stays`,
  `wall_evidence_partial_objective_trace_present`;
- `field12_gcc_emb_full_knn30_emb_knn_budget12:c3-c6`:
  `direct_route_reaches_target_and_polish_stays`,
  `wall_evidence_partial_objective_trace_present`;
- `field30_gcc_emb_full_knn30_bc_cosine_budget12:c1-c7`:
  `same_zone_control_trace`, `control_no_wall_claim`.

Interpretation boundary:

- this output proves that W1-W5 can be emitted uniformly for the 4-pair subset;
- it is not yet a supported wall claim because the route is a coarse one-step
  support-closure jump rather than a path-resolved crossing;
- the direct-route CSV is large because edited node ids are retained exactly;
- the next runner maturity step is endpoint reconstruction caching, streaming
  pair-level progress artifacts, and a finer route schedule for the field12
  pairs.

## Uniform Runner Maturity Output

The first runner maturity pass has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_field12_path_resolved_20260528/`

Companion cache root:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_endpoint_cache_20260528/`

Cache-hit smoke output:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_cache_hit_smoke_20260528/`

Maturity changes:

- endpoint and baseline memberships are cached as `.membership.npy` files with
  JSON metadata;
- every run writes `uniform_wall_probe_runner_progress.jsonl`;
- the runner supports `--pair-ids`, `--endpoint-cache-dir`, and cache reuse;
- field12 path-resolved execution used `groups_per_step=128` and
  `max_route_steps=32`.

Field12 path-resolved result:

- 2 selected field12 pairs processed;
- 0 runner errors;
- 36 direct-route rows, 36 objective-wall rows, 36 support-movement rows, 2
  polish reversion rows, and 2 route-label rows;
- both field12 pairs have 18 direct-route rows;
- both labels remain `direct_route_reaches_target_and_polish_stays` with
  `wall_evidence_partial_objective_trace_present`;
- each field12 path has 3 objective wall-step flags under this schedule.

Cache validation:

- first field12 path-resolved run: 1 baseline cache miss and 4 endpoint cache
  misses;
- `c5-c7` cache-hit smoke: 1 baseline cache hit, 2 endpoint cache hits, 0 cache
  misses.

Interpretation boundary:

- this is a runner maturity milestone because route tracing is now path-resolved
  and replayable from cached memberships;
- before the replicate pass below, it was still not a supported wall claim
  because route-order stability and route controls had not been tested;
- basin evaluation remains deferred.

## Uniform Route-Schedule Replicate Output

The first route-schedule replicate pass has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_replicate_schedule_20260528/`

Companion schedule debug output:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_schedule_debug_20260528/`

New primary outputs:

- `uniform_route_schedule_claim_rows.csv`
- `uniform_route_schedule_stability_summary.csv`
- `uniform_route_schedule_stability_report.md`

Replicate changes:

- the uniform runner now supports `--route-schedules`;
- the first schedules are `target_size_desc`, `target_size_asc`, and
  `target_label_asc`;
- route, objective, support, polish, label, progress, config, and summary rows
  now record `route_schedule`;
- route ids include the route schedule so replicated traces remain
  distinguishable;
- the runner now emits pair-level `wall_claim_gate_status` rows so
  route-order-sensitive traces cannot be accidentally promoted from per-schedule
  labels.

Replicate result:

- `field12_gcc_emb_full_knn30_emb_knn_budget12:c3-c6` is stable across all
  three schedules: every label is
  `direct_route_reaches_target_and_polish_stays`, with
  `wall_evidence_partial_objective_trace_present`; its gate status is
  `stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim`;
- `field12_gcc_emb_full_knn30_emb_knn_budget12:c5-c7` is schedule-sensitive:
  two schedules produce the same partial wall-evidence label, but
  `target_size_asc` produces `direct_route_unassigned` and `no_wall_claim`; its
  gate status is `fails_schedule_invariance_no_supported_wall_claim`;
- `field30_gcc_emb_full_knn30_bc_cosine_budget12:c1-c7` remains a same-zone
  control under all three schedules: every label is `same_zone_control_trace`
  with `control_no_wall_claim`; its gate status is
  `stable_control_no_wall_claim`;
- the `field34_all_edges_cc_cosine_budget12:c0-c2` schedule-debug control is
  also schedule-sensitive and receives
  `fails_schedule_invariance_no_supported_wall_claim`.

Interpretation boundary:

- route schedule is now part of the W1-W6 artifact contract;
- a supported wall claim must be stable under predeclared route schedules and
  have a non-ambiguous basin relation;
- stable ambiguous rows are route evidence for basin-definition refinement, not
  supported wall evidence;
- `c5-c7` and `c0-c2` remain diagnostic probes, not wall evidence;
- basin evaluation remains deferred.

## Expanded Route-Gate Control Output

The runnable expanded control subset has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_subset_expanded_controls_20260528/`

The expanded control runner output has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_expanded_controls_20260528/`

New primary outputs:

- `uniform_route_schedule_claim_panel_summary.csv`
- `uniform_route_schedule_claim_panel_report.md`

Expansion result across the 7-pair panel:

- `field12_gcc_emb_full_knn30_bc_cosine_budget12:c1-c6` is the only current
  row with `passes_schedule_invariance_distinct_partial_wall_evidence`;
- `field34_all_edges_cc_cosine_budget12:c0-c2` and
  `field12_gcc_emb_full_knn30_emb_knn_budget12:c5-c7` are
  route-order-sensitive and receive
  `fails_schedule_invariance_no_supported_wall_claim`;
- `field12_gcc_emb_full_knn30_emb_knn_budget12:c3-c6`,
  `field30_gcc_emb_full_knn30_bc_cosine_budget12:c4-c9`, and
  `field30_gcc_emb_full_knn30_citation_embedding_budget12:c5-c6` are
  route-order stable, but their calibrated relation is
  `ambiguous_support_local`, so they receive
  `stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim`;
- `field30_gcc_emb_full_knn30_bc_cosine_budget12:c1-c7` remains
  `stable_control_no_wall_claim`.

Interpretation boundary:

- the expanded gate makes basin relation part of the promotion rule;
- route-order stability alone is insufficient for a wall claim;
- the immediate open problem is not basin quality, but whether the distinct
  partial-wall gate can repeat outside the current field12 runnable surface or
  whether more endpoint/context reconstruction is needed first.

## Wall Panel Context Coverage Output

The 23-pair wall-panel context coverage audit has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_20260528/`

Primary outputs:

- `wall_panel_context_coverage_rows.csv`
- `wall_panel_context_case_requirements.csv`
- `ambiguous_relation_refinement_queue.csv`
- `runnable_distinct_pair_queue.csv`
- `wall_panel_context_coverage_summary.json`
- `wall_panel_context_coverage_report.md`

Coverage result:

- 23 wall-panel pairs across 13 cases;
- 17 pairs pass runner preflight: candidate rows, endpoint indices, vanilla
  rows, and graph dirs are present;
- 6 pairs fail runner preflight because matching vanilla case rows are missing;
- 11 distinct support-local pairs;
- 10 ambiguous support-local pairs;
- 2 same or same-identity controls;
- 0 not-yet-run distinct pairs are immediately ready for W1-W6 route-order
  gates after field hygiene checks;
- 4 not-yet-run distinct field34 pairs pass runner preflight but require
  field34/tiny-support hygiene review before route-gate execution;
- 4 distinct pairs need context before route-gate execution;
- 2 distinct rows are already route-order-sensitive controls;
- 1 distinct row is the current partial-wall gate:
  `field12_gcc_emb_full_knn30_bc_cosine_budget12:c1-c6`;
- 10 ambiguous rows are relation-refinement targets before any wall promotion;
- 3 ambiguous rows already have stable route evidence and should be prioritized
  for stronger basin-identity evidence.

Direction decision: this context audit showed why the next action was not a
wider route batch. The field34 hygiene review below now closes the field34 side
as reference/hold/filtered evidence. Use `ambiguous_relation_refinement_queue.csv`
to decide what basin-definition evidence is missing before ambiguous rows can
become wall candidates. Basin evaluation remains deferred.

## Stable Ambiguous Relation Refinement Output

The stable ambiguous relation refinement pass has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_stable_ambiguous_relation_refinement_20260528/`

Primary outputs:

- `stable_ambiguous_relation_refinement_rows.csv`
- `stable_ambiguous_endpoint_cache_links.csv`
- `stable_ambiguous_relation_refinement_summary.json`
- `stable_ambiguous_relation_refinement_report.md`

This pass inspects cached full memberships for the three ambiguous rows that
already had stable route evidence. It does not run new routes and does not
promote wall claims.

Refinement result:

- 3 stable ambiguous input pairs;
- 3 cached full-membership comparisons available;
- 2 near-distinct boundary rows:
  `field12_gcc_emb_full_knn30_emb_knn_budget12:c3-c6` and
  `field30_gcc_emb_full_knn30_bc_cosine_budget12:c4-c9`;
- 1 near-same boundary row:
  `field30_gcc_emb_full_knn30_citation_embedding_budget12:c5-c6`;
- 0 rows eligible for route promotion under the current same/distinct hard
  thresholds.

The near-distinct rows are extremely close to the current `distinct_support_min
= 0.75`: exact cached support distances are `0.749954` and `0.749216`. The
near-same row is just above `same_support_max = 0.5`, with exact support
distance `0.507791`.

Direction decision: the current hard-threshold relation rule is now the active
methodology bottleneck. The next definition step is to decide whether Track C
keeps only `same/ambiguous/distinct`, or introduces a boundary-review relation
class backed by cached membership evidence. Do not route-promote these rows
until that rule is fixed.

## Basin Relation Taxonomy v0.1 Output

The basin relation taxonomy pass has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_relation_taxonomy_v01_20260528/`

Primary outputs:

- `basin_relation_taxonomy_rows.csv`
- `basin_relation_taxonomy_status_summary.csv`
- `basin_relation_boundary_review_queue.csv`
- `basin_relation_taxonomy_summary.json`
- `basin_relation_taxonomy_report.md`

This pass consumes the 23-pair context coverage rows plus the cached stable
ambiguous refinement rows. It does not run routes, promote wall claims, or
evaluate basin value.

Taxonomy result:

- 23 panel pairs classified;
- 11 rows remain `distinct_support_local_current_rule`;
- 5 ambiguous rows become explicit boundary-review cases;
- 5 ambiguous rows remain `middle_ambiguous_support_local_hold`;
- 2 rows remain same/control rows;
- 0 new wall-promotion eligible rows are created.

Boundary-review queue:

- cached near-distinct:
  `field12_gcc_emb_full_knn30_emb_knn_budget12:c3-c6`,
  `field30_gcc_emb_full_knn30_bc_cosine_budget12:c4-c9`;
- cached near-same:
  `field30_gcc_emb_full_knn30_citation_embedding_budget12:c5-c6`;
- pending membership near-distinct:
  `field26_gcc_emb_full_knn30_emb_knn_budget12:c4-c6`;
- pending membership near-same:
  `field26_gcc_emb_full_knn30_cc_cosine_budget12:c2-c4`.

Direction decision: use `relation_taxonomy_v0_1` as the current basin-relation
gate. Boundary-review rows are definition evidence, not wall evidence. The next
method question is whether these rows remain excluded under the hard gate or
receive a separate rule with stronger membership/signature requirements.

## Clean Distinct Vanilla Context Gap Fill

The missing vanilla runner context for clean non-field34 distinct pairs has
been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_clean_distinct_vanilla_context_gap_fill_20260528/`

Primary outputs:

- `clean_distinct_vanilla_context_gap_manifest.csv`
- `vanilla_basin_rows.csv`
- `vanilla_basin_sweep_summary.json`
- `clean_distinct_vanilla_context_gap_summary.json`
- `clean_distinct_vanilla_context_gap_report.md`

Gap-fill result:

- 2 target cases processed;
- 2 new standard Leiden vanilla rows generated with seed 11, randomness 0, and
  n-iterations 10;
- expected unlocked panel pairs: 4;
- no pair routes were run and no wall or basin-value claim is made.

The post-gap-fill wall-panel coverage audit has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_after_gap_fill_20260528/`

Post-gap-fill coverage result:

- runner context status changes from 17 runnable / 6 missing vanilla context to
  21 runnable / 2 missing vanilla context;
- the four context-required distinct rows become W1-W6 route-gate candidates;
- remaining missing vanilla context rows are not clean distinct route
  candidates under the current relation gate;
- boundary-review rows remain blocked from wall promotion by the relation
  taxonomy.

New clean route-gate queue:

- `field26_gcc_emb_full_knn30_bc_cosine_budget12:c1-c2`;
- `field26_gcc_emb_full_knn30_bc_cosine_budget12:c5-c10`;
- `field30_gcc_emb_full_knn30_emb_knn_budget12:c6-c10`;
- `field30_gcc_emb_full_knn30_emb_knn_budget12:c6-c11`.

Direction decision: the next executable Track C step is a W1-W6 route-order
gate for these four clean non-field34 distinct pairs. This should still be read
as wall-protocol evidence only; basin quality and directed-search claims remain
deferred.

## Clean Distinct Route-Gate Output

The clean distinct route-gate subset has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_subset_clean_distinct_after_gap_fill_20260528/`

The clean distinct route-gate runner output has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_uniform_wall_probe_runner_clean_distinct_after_gap_fill_20260528/`

Primary outputs:

- `uniform_wall_probe_execution_manifest.csv`
- `uniform_route_schedule_claim_rows.csv`
- `uniform_route_label_rows.csv`
- `uniform_objective_wall_rows.csv`
- `uniform_polish_reversion_rows.csv`
- `uniform_wall_probe_runner_summary.json`
- `uniform_wall_probe_runner_report.md`

Run summary:

- 4 clean non-field34 distinct pairs processed;
- 0 runner errors;
- 12 route-schedule label rows;
- 63 direct-route rows, 63 objective-wall rows, 63 support-movement rows, and
  12 polish-reversion rows;
- 2 baseline cache misses and 7 endpoint cache misses, because these were new
  field26/field30 endpoint contexts.

Gate result:

- `field26_gcc_emb_full_knn30_bc_cosine_budget12:c1-c2`:
  route-order stable and
  `passes_schedule_invariance_distinct_partial_wall_evidence`;
- `field26_gcc_emb_full_knn30_bc_cosine_budget12:c5-c10`:
  route-order stable and
  `passes_schedule_invariance_distinct_partial_wall_evidence`;
- `field30_gcc_emb_full_knn30_emb_knn_budget12:c6-c10`:
  route-order sensitive and
  `fails_schedule_invariance_no_supported_wall_claim`;
- `field30_gcc_emb_full_knn30_emb_knn_budget12:c6-c11`:
  route-order sensitive and
  `fails_schedule_invariance_no_supported_wall_claim`.

The combined 11-pair route-gate panel has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_route_gate_panel_combined_after_clean_distinct_20260528/`

The full post-route coverage audit has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_wall_panel_context_coverage_after_clean_distinct_route_gate_20260528/`

Combined gate state:

- 3 distinct partial-wall gates:
  `field12 bc c1-c6`, `field26 bc c1-c2`, and `field26 bc c5-c10`;
- 4 route-order-sensitive distinct/control rows:
  `field34 cc c0-c2`, `field12 emb c5-c7`, `field30 emb c6-c10`, and
  `field30 emb c6-c11`;
- 3 stable ambiguous route-evidence rows remain blocked by basin relation;
- 1 same-control row remains `stable_control_no_wall_claim`;
- 0 clean distinct not-yet-run route-gate candidates remain.

Direction decision: do not immediately broaden the route batch. The next
method step is to review the route mechanism behind the split: why the two
`field26 bc_cosine` distinct pairs pass schedule invariance while the two
`field30 emb_knn` distinct pairs fail it. This is still wall-protocol evidence,
not basin quality or directed-search evidence.

## Clean Distinct Route-Mechanism Review

The clean distinct route-mechanism review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_clean_distinct_route_mechanism_review_20260528/`

Primary outputs:

- `clean_distinct_route_mechanism_schedule_rows.csv`
- `clean_distinct_route_mechanism_pair_summary.csv`
- `clean_distinct_route_mechanism_field_contrast.csv`
- `clean_distinct_route_mechanism_review_summary.json`
- `clean_distinct_route_mechanism_review_report.md`

Review result:

- the two `field26 bc_cosine` pairs are schedule-invariant target-polish cases;
- the two `field30 emb_knn` pairs are schedule-dependent post-polish support
  assignment cases;
- all field30 schedules reach a target-like pre-polish route state, so the
  failure is not direct-route target reach;
- field30 fails when W4 polish leaves target support-distance above the
  `same_support_max=0.5` assignment threshold.

Concrete margins:

- field26 post-polish target support-distance max: `0.342541`, comfortably
  below the `0.5` target assignment threshold;
- field30 `c6-c10` post-polish target support-distance max: `0.660739`;
- field30 `c6-c11` post-polish target support-distance max: `0.518325`;
- field30 endpoint distances remain below the endpoint threshold, so support
  assignment is the unstable component.

Direction decision: do not broaden route execution yet and do not relax wall
promotion. The next method question is whether W4 route labels need a
predeclared polish support-margin band before a route-order-sensitive row can
be interpreted as hard no-wall versus boundary-sensitive route evidence.

## W4 Polish Margin Gate Review

The W4 polish margin gate review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_polish_margin_gate_review_20260528/`

Primary outputs:

- `polish_margin_schedule_rows.csv`
- `polish_margin_pair_gate_rows.csv`
- `polish_margin_gate_review_summary.json`
- `polish_margin_gate_review_report.md`

This review applies a diagnostic support-margin band to the current 11-pair
route-gate surface. It does not change `wall_claim_gate_status`.

Margin rule:

- target support assignment threshold: `same_support_max=0.5`;
- diagnostic support-margin band: `0.05`;
- `support_boundary_loss`: post-polish target support-distance is above `0.5`
  but within `0.05`;
- `support_hard_loss`: post-polish target support-distance is more than `0.05`
  above the threshold;
- target-stable rows remain diagnostic context, not stronger wall claims.

Review result:

- 33 route-schedule rows across 11 route-gate pairs;
- 3 distinct partial-wall gates are kept unchanged with margin context:
  `field12 bc c1-c6`, `field26 bc c1-c2`, and `field26 bc c5-c10`;
- 2 boundary-sensitive route holds:
  `field12 emb c5-c7` and `field30 emb c6-c11`;
- 2 support-loss no-wall holds:
  `field30 emb c6-c10` and `field34 cc c0-c2`;
- 3 stable ambiguous route-evidence rows remain relation-blocked definition
  evidence;
- 1 same-control row remains a no-wall control.

Direction decision: keep the existing wall-promotion rule unchanged. The margin
band is useful for triage, but it needs validation on more controls before it
can become a route-label rule. The next Track C step should be either a narrow
margin-rule validation panel or a written Methodology v0 decision that freezes
the current conservative behavior.

## Methodology v0 Margin Validation Panel

The Methodology v0 margin validation artifact has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_margin_validation_20260528/`

Primary outputs:

- `methodology_v0_route_gate_decision_rows.csv`
- `methodology_v0_state_counts.csv`
- `margin_validation_panel.csv`
- `methodology_v0_margin_validation_summary.json`
- `methodology_v0_margin_validation_report.md`

This artifact freezes the current 11 route-gate pairs into conservative
Methodology v0 states:

- 3 `partial_wall_gate_conservative` rows;
- 3 `relation_blocked_definition_evidence` rows;
- 2 `boundary_sensitive_margin_validation_candidate` rows;
- 2 `support_loss_no_wall_contrast` rows;
- 1 `same_control_no_wall` row.

The margin validation panel is deliberately narrow:

- boundary-sensitive candidates: `field12 emb c5-c7` and `field30 emb c6-c11`;
- support-loss contrasts: `field30 emb c6-c10` and `field34 cc c0-c2`.

Direction decision: use this artifact as the current route-gate Methodology v0
surface. The validation question is whether near-threshold post-polish support
losses behave differently from support-hard-loss no-wall holds under
predeclared repeat polish/schedule validation. No row in this artifact changes
`wall_claim_gate_status`, and no row should be used for basin quality, cost, or
directed-search claims.

## Held-Out Margin Validation Review

The held-out margin validation review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_margin_validation_panel_review_20260529/`

Supporting held-out runner outputs:

- `research/consensus/results/adaptive_refinement/leiden_basin_margin_validation_runner_initial_target_label_desc_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_margin_validation_runner_clean_target_label_desc_20260529/`

Primary outputs:

- `margin_validation_schedule_rows.csv`
- `margin_validation_pair_results.csv`
- `margin_validation_panel_review_summary.json`
- `margin_validation_panel_review_report.md`

The review combines the original three schedules with the held-out
`target_label_desc` schedule for the 4-pair panel.

Result:

- 16 schedule rows total, including 4 held-out schedule rows;
- 2 validated boundary-sensitive holds:
  `field12 emb c5-c7` and `field30 emb c6-c11`;
- 1 validated support-loss contrast:
  `field34 cc c0-c2`;
- 1 mixed support-loss contrast:
  `field30 emb c6-c10`, because the earlier schedules retain hard-loss
  evidence but the held-out schedule is `target_near_support_boundary`.

Direction decision: boundary-sensitive route holds can now be treated as a
separate route-label uncertainty class. This is not wall evidence. The mixed
support-loss contrast should stay no-wall, but it should not be used as a
strong repeated hard-loss example. The next method step is to freeze the
route-label interpretation, not to broaden route execution or inspect
quality/cost.

## Route-Label Interpretation v0

The route-label interpretation freeze has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_route_label_interpretation_v0_20260529/`

Primary outputs:

- `route_label_interpretation_rows.csv`
- `route_label_interpretation_rules.csv`
- `route_label_interpretation_counts.csv`
- `route_label_interpretation_summary.json`
- `route_label_interpretation_report.md`

This artifact maps the 11 Methodology v0 route-gate pairs to conservative
interpretation labels after the held-out margin-validation review. It does not
run new routes, inspect basin quality/cost, or change wall-promotion rules.

Result:

- 3 `partial_wall_protocol_evidence` rows;
- 3 `relation_blocked_route_evidence` rows;
- 2 `boundary_sensitive_route_uncertainty` rows;
- 1 `hard_support_loss_no_wall_contrast` row;
- 1 `mixed_support_loss_no_wall_hold` row;
- 1 `same_control_no_wall` row;
- 0 promoted wall claims; all 11 rows remain `no_wall_promotion`.

Direction decision: use this as the current route interpretation surface. The
next method decision should not broaden route execution. It should decide which
remaining blocker is basin-relation definition, field34 evidence eligibility,
or a genuinely new wall-evidence question; the field34 audit below resolves
that branch as non-executable reference/hold/filtered evidence.

## Route-Label Blocker Triage

The route-label blocker triage has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_route_label_blocker_triage_20260529/`

Primary outputs:

- `route_label_blocker_triage_rows.csv`
- `relation_definition_queue.csv`
- `field34_hygiene_queue.csv`
- `wall_evidence_question_hold_queue.csv`
- `route_label_blocker_triage_counts.csv`
- `route_label_blocker_triage_summary.json`
- `route_label_blocker_triage_report.md`

This artifact applies the frozen route labels to the current 23-pair surface.
It does not run routes, relax wall promotion, or inspect quality/cost.

Queue result:

- 10 relation-definition queue rows:
  3 route-evidence relation-blocked rows, 2 pending-membership relation checks,
  3 middle ambiguous relation holds, and 2 field34 rows that need hygiene before
  relation review;
- 8 field34 hygiene queue rows;
- 7 wall-evidence question hold rows, all marked no immediate execution;
- 0 immediate route-execution rows.

Direction decision: the next executable work should not be another route batch.
Prioritize relation-definition review for the three route-stable
relation-blocked rows, then pending-membership relation checks and field34
hygiene. Frozen protocol, uncertainty, and no-wall rows should be retained as
examples or constraints, not promoted to wall evidence.

## Relation Boundary Rule Review

The relation boundary rule review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_relation_boundary_rule_review_20260529/`

Primary outputs:

- `relation_boundary_rule_review_rows.csv`
- `relation_boundary_rule_counterfactuals.csv`
- `relation_boundary_rule_options.csv`
- `relation_boundary_rule_review_summary.json`
- `relation_boundary_rule_review_report.md`

This artifact reviews the 3 route-stable relation-blocked rows with cached
full-membership exact-support evidence. It compares the current hard gate
against threshold snapping and route-stability override policies.

Review result:

- reviewed rows: 3;
- near-distinct rows: 2, with exact support distances just below `0.75`
  (`0.7499544709524677` and `0.7492163009404389`);
- near-same rows: 1, with exact support distance just above `0.5`
  (`0.5077910174152154`);
- accepted policy: current hard gate;
- rejected policies: `epsilon_0p001_distinct_only`,
  `epsilon_0p01_two_sided`, and `route_stability_override`;
- promoted wall claims: 0.

Direction decision: keep the boundary-review class. Exact memberships show
different endpoint identities, but support-local distances remain inside the
predeclared middle zone. Route stability remains route evidence, not
basin-relation evidence. The next relation step is to review the two
pending-membership relation checks before any route batch.

## Pending-Membership Relation Review

The pending-membership relation review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_relation_review_20260529/`

Primary outputs:

- `pending_membership_relation_review_rows.csv`
- `pending_membership_relation_counterfactuals.csv`
- `pending_membership_relation_evidence_links.csv`
- `pending_membership_relation_review_summary.json`
- `pending_membership_relation_review_report.md`

This artifact reviews the 2 relation-queue rows marked
`pending_membership_relation_check`. It checks whether exact changed-support
node evidence and full membership cache are available. It does not run routes,
change wall promotion, or inspect basin quality/cost.

Review result:

- reviewed rows: 2;
- exact changed-support node evidence available: 2;
- full membership cache missing: 2;
- current hard-gate classification: 2
  `boundary_review_ambiguous_support_local` rows;
- proxy signatures differ on the same sketch sample: 2;
- promoted wall claims: 0;
- immediate route-execution rows: 0.

Direction decision: keep both rows as boundary-review holds. This initial review
showed that exact changed-support node sets were available but full membership
cache was missing. The follow-up materialization below closes that cache blocker
without changing the basin-relation decision.

## Pending-Membership Cache Materialization

The pending-membership cache materialization has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_cache_materialization_20260529/`

The materialized endpoint cache is stored at:

`research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_endpoint_cache_20260529/`

The after-cache relation review has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_pending_membership_relation_review_after_cache_materialization_20260529/`

Primary outputs:

- `pending_membership_cache_rows.csv`
- `pending_membership_cache_relation_rows.csv`
- `pending_membership_cache_materialization_summary.json`
- `pending_membership_cache_materialization_report.md`
- `pending_membership_relation_review_rows.csv`
- `pending_membership_relation_review_summary.json`
- `pending_membership_relation_review_report.md`

This artifact materializes full baseline and endpoint memberships for the 2
pending relation rows only. It does not run routes, change wall promotion, or
inspect basin quality/cost.

Materialization result:

- pending pairs: 2;
- materialized cache rows: 6, all `miss_materialized`;
- endpoint support hash mismatches: 0;
- full membership relation rows: 2;
- hard-gate classification after cache materialization: 2
  `boundary_review_ambiguous_support_local` rows;
- exact support distances unchanged from the source evidence:
  `0.7497612225405922` and `0.5114503816793894`;
- after-cache review status: 2 `both_endpoint_memberships_cached` rows and 2
  `cached_membership_still_boundary_review` decisions;
- promoted wall claims: 0;
- immediate route-execution rows: 0.

Direction decision: the pending-membership blocker is now closed as a cached
boundary-review hold. Full membership evidence validates the source support
hashes, but it does not move either row out of the predeclared middle zone. Do
not route-promote these rows unless the boundary-band definition is explicitly
changed.

## Field34 Evidence Eligibility Audit

The field34 evidence-eligibility audit has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_field34_evidence_eligibility_audit_20260529/`

Primary outputs:

- `field34_endpoint_universe_rows.csv`
- `field34_method_eligibility_rows.csv`
- `field34_pair_support_source_rows.csv`
- `field34_queue_projection_rows.csv`
- `field34_evidence_eligibility_summary.json`
- `field34_evidence_eligibility_report.md`

This artifact closes the field34 hygiene blocker as a source and fixture audit.
It does not run routes, change wall promotion, inspect basin quality/cost, or
turn field34 into a calibration source.

Audit result:

- endpoint rows: 39;
- method rows: 5;
- endpoint hygiene counts: 13 `zero_or_noop_filtered`, 13
  `small_support_diagnostic_reference`, 9 `tiny_support_reference_only`, 3
  `duplicate_tiny_support_reference_only`, and 1
  `moderate_support_candidate`;
- method roles: 2 `field34_duplicate_mixed_reference_only`, 1
  `field34_mostly_filtered_source`, 1
  `field34_small_support_diagnostic_reference`, and 1
  `field34_tiny_support_reference_only`;
- all 5 methods remain `not_clean_calibration_source`;
- field34 queue rows: 8, all projected to reference, hold, or filtered
  decisions;
- route-gate candidate rows: 0;
- promoted wall claims: 0;
- immediate route-execution rows: 0.

Direction decision: field34 remains useful as diagnostic/reference evidence and
as a hygiene caution surface, but it is not a clean basin-definition
calibration source under the current gates. Do not run a field34 route batch
from this audit alone. The remaining Track C decision is now whether any
non-field34 wall-evidence question is still worth testing under the fixed
basin-relation gates, or whether the current result should be closed as
basin-definition and wall-protocol evidence.

## Remaining Wall-Question Audit

The remaining wall-question audit has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_remaining_wall_question_audit_20260529/`

Primary outputs:

- `remaining_wall_question_rows.csv`
- `remaining_wall_question_decision_counts.csv`
- `remaining_wall_question_summary.json`
- `remaining_wall_question_report.md`

This artifact audits the non-field34 remainder after relation-boundary review,
pending-membership cache materialization, and field34 evidence eligibility. It
does not run routes, change wall promotion, inspect basin quality/cost, or turn
protocol references into supported wall claims.

Audit result:

- non-field34 rows: 15;
- executable non-field34 route candidates: 0;
- non-field34 wall-promotion candidates: 0;
- field34 route-gate candidates carried forward from the hygiene audit: 0;
- field34 immediate route executions: 0;
- field34 promoted wall claims: 0;
- remaining non-field34 classes: 3 `protocol_reference_only`, 2
  `route_uncertainty_reference_only`, 1 `no_wall_contrast_reference_only`, 1
  `same_or_identity_control_only`, 3 `closed_by_current_boundary_rule`, 2
  `closed_by_cached_pending_membership_boundary_review`, and 3
  `middle_ambiguous_definition_hold`.

Direction decision: the current Track C cycle should not continue with another
route batch. The defensible closure is basin-definition and wall-protocol
evidence, unless the basin-relation boundary band is explicitly reopened as a
definition problem before any route execution.

## Cycle Closure Write-up

The Track C cycle closure write-up has been generated at:

`research/consensus/results/adaptive_refinement/leiden_basin_cycle_closure_writeup_20260529/`

Primary outputs:

- `track_c_cycle_closure_summary.json`
- `track_c_cycle_closure_report.md`
- `track_c_cycle_closure_evidence_rows.csv`
- `track_c_cycle_reopen_conditions.csv`

This artifact writes the current closure as a scoped claim over the 23-pair
wall surface and blocker chain. It does not claim that the full 206-pair
calibration universe has no walls. It does not run routes, change wall
promotion, inspect basin quality/cost, or make directed-search claims.

Closure claim:

- under the fixed current gates, the current Track C wall-evidence cycle has no
  executable route candidate and no wall-promotion candidate;
- field34 is closed as reference, hold, or filtered evidence;
- the current cycle should be written as basin-definition and wall-protocol
  evidence;
- the full 206-pair universe is only future work if a new precommitted
  non-field34 panel is declared.

Reopen conditions:

- redefine the basin-relation boundary band before any route execution;
- declare a new non-field34 panel from the 206 wall-candidate universe with
  precommitted sampling, graph context, and wall-evidence requirements;
- design stronger wall-evidence requirements before upgrading protocol
  references;
- delay quality/cost evaluation until accepted basin relation and supported
  wall evidence exist.

## Primary Data Roots

| Role | Root | Current use |
| --- | --- | --- |
| strict field30 support evidence | `research/consensus/results/adaptive_refinement/leiden_multibasin_signature_field30_budget12_support_20260519/` | endpoint identities, support-local grouping, threshold sensitivity |
| strict field26 support evidence | `research/consensus/results/adaptive_refinement/leiden_multibasin_signature_field26_citation_embedding_budget15_support_20260519/` | endpoint identities, support-local grouping, threshold sensitivity |
| broad cross-field endpoint inventory | `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/` | field12/26/34 endpoint rows and candidate protocols |
| combined cross-field signature review | `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/signature_review/` | 15-case endpoint inventory and current coarse support-local grouping |
| c0/c2 route and wall traces | `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_*` | route traces, wall/debt/progress diagnostics, negative controls |
| older review bundle | `research/consensus/results/adaptive_refinement/previous_results_review_20260518/` | historical context and cost/instrumentation summaries only |

## Current Endpoint Inventory

The combined cross-field signature review contains 125 p5 candidate endpoint
rows across 15 cases. Those rows collapse to 113 endpoint identities because
some field34 candidate rows are zero-support or duplicate endpoint rows. At the
current diagnostic grouping thresholds `endpoint_tau=0.02` and
`support_tau=0.5`, it contains 106 support-local coarse groups. This is a
working support-local grouping, not a final basin count.

| field | method | endpoint rows | endpoint identities | support-local groups | zero-support rows | support-count range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| field12 | `gcc_emb_full_knn30_bc_cosine` | 12 | 12 | 12 | 0 | 590-974 |
| field12 | `gcc_emb_full_knn30_cc_cosine` | 4 | 4 | 4 | 0 | 127-162 |
| field12 | `gcc_emb_full_knn30_citation_all` | 6 | 6 | 6 | 0 | 1818-2023 |
| field12 | `gcc_emb_full_knn30_citation_embedding` | 6 | 6 | 6 | 0 | 6926-7216 |
| field12 | `gcc_emb_full_knn30_emb_knn` | 12 | 12 | 12 | 0 | 3105-4468 |
| field26 | `gcc_emb_full_knn30_bc_cosine` | 12 | 12 | 12 | 0 | 665-1828 |
| field26 | `gcc_emb_full_knn30_cc_cosine` | 5 | 5 | 5 | 0 | 454-596 |
| field26 | `gcc_emb_full_knn30_citation_all` | 5 | 5 | 5 | 0 | 2129-2485 |
| field26 | `gcc_emb_full_knn30_citation_embedding` | 12 | 12 | 12 | 0 | 3375-4206 |
| field26 | `gcc_emb_full_knn30_emb_knn` | 12 | 12 | 12 | 0 | 2040-3439 |
| field34 | `all_edges_bc_cosine` | 9 | 7 | 5 | 3 | 0-136 |
| field34 | `all_edges_cc_cosine` | 5 | 5 | 3 | 0 | 33-147 |
| field34 | `all_edges_citation_all` | 4 | 4 | 3 | 0 | 7-116 |
| field34 | `all_edges_citation_embedding` | 9 | 3 | 3 | 7 | 0-182 |
| field34 | `all_edges_emb_knn` | 12 | 8 | 6 | 3 | 0-260 |

Preparation conclusions:

- field12 and field26 are clean endpoint-inventory sources in the current
  artifact set: no zero-support rows and every listed endpoint has a distinct
  p5 signature within its case.
- field30 strict support evidence is also clean and should remain the primary
  calibration source for the existing definition discussion.
- field34 is usable, but not as a clean basin-definition source without
  filtering: the 2026-05-29 eligibility audit keeps it as reference, hold, or
  filtered evidence rather than route-gate evidence.
- Any Phase 1 index should mark zero-support rows as `no_op_or_duplicate` before
  counting support-local basin candidates.

## Endpoint And Support Distances

The strict field30/field26 support runs show exact changed-support capture:
support counts equal sketch sample sizes for all candidate endpoints in those
runs.

| root | cases | pair rows | max sampled coassignment distance | support distance min | support distance mean | support distance max | support source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| field30 strict support | 5 | 223 | 0.00132881 | 0.433333 | 0.695756 | 0.918573 | changed-node support |
| field26 strict support | 1 | 66 | 0.0000133645 | 0.522932 | 0.629140 | 0.702698 | changed-node support |

The combined 15-case review has 540 pair rows. It uses changed-node support for
most cases, but field34 has some fallback changed-pair support rows.

| field | method | pair rows | max sampled coassignment distance | support min | support mean | support max | same-coarse pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| field12 | `gcc_emb_full_knn30_bc_cosine` | 66 | 0.000036275 | 0.569042 | 0.684301 | 0.778184 | 0 |
| field12 | `gcc_emb_full_knn30_cc_cosine` | 6 | 0.00834334 | 0.571429 | 0.649100 | 0.760956 | 0 |
| field12 | `gcc_emb_full_knn30_citation_all` | 15 | 0.00215932 | 0.612346 | 0.674778 | 0.740753 | 0 |
| field12 | `gcc_emb_full_knn30_citation_embedding` | 15 | 0.000160374 | 0.575788 | 0.597094 | 0.614168 | 0 |
| field12 | `gcc_emb_full_knn30_emb_knn` | 66 | 0.0000591856 | 0.616769 | 0.730167 | 0.790999 | 0 |
| field26 | `gcc_emb_full_knn30_bc_cosine` | 66 | 0.000036275 | 0.623639 | 0.765162 | 0.884222 | 0 |
| field26 | `gcc_emb_full_knn30_cc_cosine` | 10 | 0.00555960 | 0.511450 | 0.666918 | 0.741007 | 0 |
| field26 | `gcc_emb_full_knn30_citation_all` | 10 | 0.00340489 | 0.648924 | 0.680395 | 0.725598 | 0 |
| field26 | `gcc_emb_full_knn30_citation_embedding` | 66 | 0.0000133645 | 0.522932 | 0.629140 | 0.702698 | 0 |
| field26 | `gcc_emb_full_knn30_emb_knn` | 66 | 0.0000973699 | 0.612295 | 0.726584 | 0.807919 | 0 |
| field34 | `all_edges_bc_cosine` | 36 | 0.000353204 | 0 | 0.845852 | 1 | 5 |
| field34 | `all_edges_cc_cosine` | 10 | 0.00120853 | 0.0408163 | 0.516281 | 0.800000 | 3 |
| field34 | `all_edges_citation_all` | 6 | 0.000736956 | 0.300000 | 0.766701 | 0.984127 | 1 |
| field34 | `all_edges_citation_embedding` | 36 | 0.000796142 | 0 | 0.416667 | 1 | 21 |
| field34 | `all_edges_emb_knn` | 66 | 0.00254498 | 0 | 0.877059 | 1 | 8 |

Preparation conclusions:

- The current artifacts strongly separate endpoint identity from support-local
  relation: global sampled coassignment distances can be tiny while changed
  support distances are large.
- `support_tau=0.5` is a useful diagnostic threshold for inventory, but not yet
  a final basin definition.
- `sample_coassignment_distance` is only a sampled global endpoint metric here;
  it is not enough by itself to establish a full global attraction basin.
- field34 zero support and fallback support-source rows have now been audited;
  field34 remains reference/diagnostic evidence, not a clean non-c0
  basin-definition fixture.

## Threshold Sensitivity

Existing threshold sensitivity files confirm that support-local basin counts
are threshold-dependent.

| root | rows | exact-count min | exact-count max | coarse-count min | coarse-count max | support tau values | endpoint tau values |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| field30 strict support | 150 | 6 | 12 | 1 | 12 | 0, 0.25, 0.5, 0.75, 1 | 0, 1e-06, 5e-06, 1e-05, 2e-05, 0.02 |
| field26 strict support | 30 | 12 | 12 | 1 | 12 | 0, 0.25, 0.5, 0.75, 1 | 0, 1e-06, 5e-06, 1e-05, 2e-05, 0.02 |

Preparation conclusion:

- Phase 1 should not report a final basin count. It should report counts under
  a named metric/threshold and add an ambiguity flag.

## Wall And Route Sources

The wall/route inventory should be built from existing `combined_with_field30`
transition folders, but these folders must not define basin identity by
themselves.

Useful route/wall source families:

- `basin_transition_branch_target_growth_field34_cc_c0_v0/`
- `basin_transition_branch_target_growth_field34_cc_c2_v0/`
- `basin_transition_boundary_analysis_field34_cc/`
- `basin_transition_boundary_calibration_field34_cc/`
- `basin_transition_post_gate_recovery_moves_field34_cc_*`
- `basin_transition_attachment_margin_*`
- `basin_transition_search_field34_cc_*`

Negative-control or caution families:

- `basin_transition_closure_operator_pilot_field34_cc/`
- `basin_transition_label_internal_repair_pilot_field34_cc/`
- `basin_transition_attachment_margin_stage2_recovery_field34_cc_*`

Preparation conclusions:

- These route artifacts can provide objective-debt, polish-reversion,
  support-incompatibility, failed-direct-path, or path-progress evidence.
- They should be joined only after endpoint identities and support-local basin
  candidates are fixed from the basin-definition index.
- c0 remains a known diagnostic reference. It should not be the main proof.

## Phase 1 Data Actions And Next Gate

The Phase 1 basin-only index has been built from current data without a new
replay. It contains no quality fields in the generated basin-first outputs.

Completed actions:

1. Build `landscape_case_index.csv` from the 15-case combined signature review
   plus strict field30/field26 roots.
2. Build `metric_hygiene_audit.csv` with at least these flags:
   `exact_support_capture`, `zero_support_rows`, `support_distance_source`,
   `has_pairwise_endpoint_distance`, `has_threshold_sensitivity`,
   `has_route_trace_source`.
3. Build `basin_cartography_case_index.csv` using endpoint identity counts and
   support-local group counts under the declared current diagnostic threshold.
4. Mark field34 rows with zero support or repeated signatures as
   `no_op_or_duplicate` before any basin-count interpretation.
5. Keep global observed basin assignment as `proxy_sampled` or `unresolved`
   until the endpoint-distance metric and ambiguity rule are fixed.
6. Create empty or inventory-only `wall_evidence_rows.csv` and
   `route_taxonomy_rows.csv` first, then populate them only from cases with
   accepted basin candidates.

Next gate:

1. Recompute support-local relations as a three-zone pair decision:
   `same_support_local`, `ambiguous_support_local`, and
   `distinct_support_local`.
2. Treat `support_tau=0.5` as a strict same-zone inventory threshold, not as the
   final basin boundary.
3. Test `support_tau=0.75` as a provisional distinct-zone boundary while
   preserving the large middle region as ambiguous.
4. Use the field34 evidence-eligibility audit to keep field34 no-op,
   duplicate, tiny-support, and small-support rows as filtered, hold, or
   reference evidence.
5. Keep wall and route rows inventory-only until basin-pair assignments are no
   longer ambiguous.
6. For the first Phase 2 wall-evidence join, use only
   `route_join_candidate_pair_rows.csv` from the calibration output.
7. Do not run a field34 route batch from the hygiene audit alone; the field34
   queue has 0 route-gate candidates and 0 immediate route-execution rows.
8. Use the remaining wall-question audit as the current execution closure:
   there are 0 executable non-field34 route candidates and 0 wall-promotion
   candidates under the fixed gates.
9. Use the basin-existence assumption audit to keep H1 and H2 separate. Current
   data support multiple meaningful basin candidates as candidate evidence, but
   pathway methodology remains not operational under fixed gates.
10. Before expanding wall claims, explicitly reopen the basin-relation boundary
   rule as a definition problem. Do not start with route execution.

## Current Readiness Judgment

G1 endpoint inventory is close for field12, field26, and field30. Field34 has
now been audited as reference, hold, or filtered evidence rather than a clean
calibration source.

G2 basin definition is not fixed yet. The Phase 1 review shows that
support-local grouping has a threshold cliff, so the next decision must be a
same/distinct/ambiguous calibration rule rather than a wall-cartography run.

G3 wall existence should not be evaluated as a broad case-level claim yet. The
current 23-pair surface has 3 conservative partial-wall protocol references,
but 0 supported wall claims. The remaining wall-question audit finds 0
executable non-field34 route candidates, 0 non-field34 wall-promotion
candidates, and 0 field34 route-gate candidates. The current cycle should close
as basin-definition and wall-protocol evidence unless the basin-relation
boundary band is explicitly reopened first.

The basin-existence assumption audit adds one important distinction: H1
existence has candidate evidence, while H2 pathway readiness remains closed
under the current gates. The audit reports 16 non-field34 cases with 5 strong
and 4 moderate candidate multi-basin evidence cases, plus 87 strong and 75
moderate meaningful distinct pairs under declared support thresholds. These are
not final basin counts.
