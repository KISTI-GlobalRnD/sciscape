# Leiden Basin Demo Scripts

Controlled tiny-graph demos for the Track C guiding premise. These scripts use
ordinary Leiden + CPM baselines before any custom method, route execution, wall
promotion, quality/cost comparison, or NanoClustering generality claim.

- `run_leiden_cpm_tiny_demo_seed_sweep.py`: builds four tiny graph families
  for near-tie bridge ambiguity, external-host absorption, balanced split, and
  diffuse fragmentation. It runs plain Leiden + CPM over repeated seeds and
  materializes endpoint signatures, a frozen endpoint manifest, random-restart
  discovery curves, mechanism reads, graph sidecars, gate rows, and a report.
- `run_leiden_cpm_tiny_handle_method_probe.py`: applies simple
  mechanism-aware initial memberships to the frozen tiny CPM graph families and
  compares discovery/navigation against the frozen random-restart baseline. It
  is a candidate-method probe only, not an algorithm, wall/pathway,
  quality/cost, or NanoClustering claim. Use `--candidate-set coverage_v2` to
  append replay-diagnosed coverage handles without overwriting the v1 candidate
  registry.
- `analyze_leiden_cpm_tiny_endpoint_replay.py`: replays frozen tiny CPM
  endpoint signatures as direct Leiden initial memberships. It separates
  method-handle gaps from endpoint instability and is diagnostic only, not a
  method, pathway, quality/cost, or NanoClustering claim.
- `analyze_leiden_cpm_tiny_coverage_order_robustness.py`: runs Stress 1 for
  `coverage_v2` by varying handle order policies against a reconstructed
  random-restart baseline distribution. It records ordering and early-budget
  cost caveats only; it does not add pathway, quality/cost, NanoClustering, or
  algorithm claims.
- `analyze_leiden_cpm_tiny_blind_rule_handle_probe.py`: runs Stress 2 for
  `coverage_v2` by generating graph-rule handles before reading frozen
  endpoint manifests, replay diagnoses, or endpoint signatures. It tests
  endpoint-derived dependency and keeps all claims at the tiny-demo
  candidate-method level.
- `analyze_leiden_cpm_tiny_blind_rule_ablation.py`: runs Stress 3 by ablating
  handle types from the materialized blind-rule candidate registry. It reports
  compacted and slot-preserving schedules, target-scoped attribution,
  endpoint-level first hits, dropout damage, and seed stability. It is a
  tiny-demo mechanism-localization diagnostic only.
- `analyze_leiden_cpm_tiny_mechanism_variant_panel.py`: materializes the
  Stress 4 mechanism-variant panel through P0-P4 only. It writes fixed graph
  manifests, edge/role sidecars, graph-only mechanism features, blind
  candidate registries, role/name invariance checks, and a phase-lock hash
  before any Leiden seed sweep or endpoint-evaluation artifact is read. Use
  `--panel-version v1_1` for the control-matched panel with explicit decoy
  roles for the weak absorption and diffuse controls.
- `audit_leiden_cpm_tiny_mechanism_variant_controls.py`: runs the P4.5
  control-strength audit from phase-locked P0-P4 artifacts. It checks hash
  integrity, no-endpoint-input boundaries, preserved candidate coverage, and
  whether mechanism-removed controls still contain target-like decoys before
  any P5-P8 endpoint evaluation. Use `--hard-control-decoy-gate` when auditing
  `v1_1` for promotion-oriented Stress 4 readiness.
- `run_leiden_cpm_tiny_mechanism_variant_p5_p8.py`: runs Stress 4 P5-P8 from
  phase-locked mechanism-variant inputs. It verifies the P0-P4 phase-lock and
  P4.5 audit, runs ordinary Leiden + CPM seed sweeps, freezes recurrent
  endpoint signatures, executes the phase-locked blind candidates, and compares
  target-compatible first hits against random-restart p75.
- `analyze_leiden_cpm_tiny_mechanism_variant_p8_failure_typing.py`: separates
  P5-P8 misses into structurally target-eligible misses versus endpoints that
  the frozen positive registry does not currently target. It does not rerun
  Leiden.
- `run_leiden_cpm_tiny_mechanism_variant_joint_weak_pair_probe.py`: downstream
  P8.2 diagnostic for the true `df_two_pair` eligible misses. It jointly
  applies already phase-locked single weak-pair handles without mutating the
  P0-P4 registry.
- `analyze_leiden_cpm_tiny_mechanism_variant_v1_2_schedule_robustness.py`:
  stresses the phase-locked `v1.2` registry under deterministic and random
  schedule orders. It tests whether the joint weak-pair result depends on
  canonical ordering and includes a joint-suppressed negative control.
- `run_leiden_cpm_variable_pair_synthetic_demo.py`: consumes the frozen
  NanoClustering-derived variable-pair synthetic-demo family surface and runs
  ordinary Leiden+CPM on 6 small graph families. It records graph variants,
  start-condition/seed endpoint signatures, family gates, and a report. This is
  a controlled mechanism-reproduction diagnostic only; it is not full
  NanoClustering replay, route/pathway execution, wall promotion,
  quality/cost comparison, method success, or an algorithm claim.
- `analyze_leiden_cpm_variable_pair_synthetic_endpoint_replay.py`: replays the
  variable-pair synthetic endpoint signatures as initial memberships under
  ordinary Leiden+CPM and materializes replay-stable G4 endpoint-relation
  candidates. It designs the next route gate only; it does not execute routes,
  promote walls, compare methods, evaluate quality/cost, or claim an algorithm.
- `run_leiden_cpm_variable_pair_synthetic_route_trace.py`: executes compact
  initial-membership route traces over the replay-stable variable-pair
  synthetic endpoint candidates. It classifies cross, bounce, collapse, mixed,
  and unknown outcomes, but does not promote a wall, compare methods, evaluate
  quality/cost, replay NanoClustering, or claim an algorithm.
- `analyze_leiden_cpm_variable_pair_synthetic_route_trace_g4_1_audit.py`:
  audits the variable-pair route trace for target-identical reconstruction,
  source no-ops, intervention size, reverse bridge-context release policies,
  and same-pair-state controls. It is a stricter trace diagnostic only, not
  wall promotion, method comparison, quality/cost evaluation, NanoClustering
  replay, or an algorithm claim.
- `analyze_leiden_cpm_variable_pair_synthetic_route_trace_g4_2_necessity.py`:
  decomposes the strict G4.1 synthetic crossings into source/initial/target
  CPM quality, pair state, bridge transitions, and sibling-policy context. It
  is a mechanism-necessity diagnostic only, not wall promotion, method
  comparison, quality/cost evaluation, NanoClustering replay, or an algorithm
  claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization.py`:
  tests the frozen `bridge_context_release_without_pair_merge` handle on a
  fixed independent synthetic variant/control panel without reading target
  endpoint signatures. It is a handle-generalization diagnostic only, not wall
  promotion, method comparison, quality/cost evaluation, NanoClustering replay,
  or an algorithm claim.
- `analyze_leiden_cpm_variable_pair_synthetic_g4_4_restart_comparison.py`:
  compares the frozen G4.3 source-conditioned handle against same-panel
  ordinary Leiden+CPM restart discovery of known coassigned endpoints. It is a
  fixed-panel restart diagnostic only, not wall promotion, full-method
  comparison, quality/cost evaluation, NanoClustering replay, or an algorithm
  claim.
- `analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression.py`:
  evaluates the frozen `neutral_release_with_direct_support_v1` source-local
  selector for the G4.3 bridge-release handle using graph/source-membership
  features and local CPM delta only. It is a selector/suppression diagnostic
  only, not wall promotion, full-method comparison, quality/cost evaluation,
  NanoClustering replay, or an algorithm claim.
- `analyze_leiden_cpm_variable_pair_synthetic_g4_6_schedule_accounting.py`:
  accounts for a minimal schedule that first runs ordinary Leiden+CPM, then
  applies the frozen G4.3 handle once only when the endpoint passes the frozen
  G4.5 selector. It reports observed source availability and
  restart-plus-handle unit cost. It is a schedule-accounting diagnostic only,
  not wall promotion, full-method comparison, wall-clock quality/cost
  evaluation, NanoClustering replay, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_7_independent_schedule_stress.py`:
  replays the frozen G4.3 handle, G4.5 selector, and G4.6 schedule on a
  predeclared shifted stress panel. It is an opportunity-regime boundary
  diagnostic only, not a threshold-retuning loop, wall promotion, full-method
  comparison, quality/cost evaluation, NanoClustering replay, or an algorithm
  claim.
- `analyze_leiden_cpm_variable_pair_synthetic_g4_8_opportunity_regime_design.py`:
  reads materialized G4.3, G4.6, and G4.7 outputs to classify
  opportunity-regime cells and design the next fresh predeclared regime-cell
  panel. It is a design artifact only, not a new Leiden run, threshold-retuning
  loop, wall promotion, full-method comparison, quality/cost evaluation,
  NanoClustering replay, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_8b_regime_cell_panel.py`:
  runs a fresh predeclared regime-cell panel through the frozen G4.3 handle,
  G4.5 selector, and G4.6 schedule. It tests source-handle fire and no-leak
  behavior by opportunity regime, not selector retuning, wall promotion,
  quality/cost evaluation, NanoClustering replay, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_8c_opportunity_cartography.py`:
  maps which fresh anchor and perturbation cases preserve endpoint coexistence
  and bridge-release eligible separated sources before source-discovery
  replacement. It is mechanism cartography only, not selector retuning,
  quality/cost evaluation, NanoClustering replay, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_8d_balance_cartography.py`:
  maps the predeclared pair-bridge by bridge-host 2D balance surface around the
  G4.3 ready anchor. It tests whether ready opportunity is a reproducible band
  or sparse ridge, not selector retuning, source-discovery replacement,
  quality/cost evaluation, NanoClustering replay, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_8e_diagonal_ridge_refinement.py`:
  refines the sparse G4.8D diagonal with a predeclared narrow strip of
  intermediate pair-bridge/bridge-host cells. It tests whether the ready signal
  is a continuous ridge, finite-width band, or centerline resonance lattice,
  not selector retuning, source-discovery replacement, quality/cost evaluation,
  NanoClustering replay, or an algorithm claim.
- `analyze_leiden_cpm_variable_pair_synthetic_g4_8f_centerline_signature_audit.py`:
  reads the materialized G4.8E/G4.3/G4.5/G4.6 outputs and compares centerline
  endpoint/source signatures. It audits why only the resonance cells expose a
  full source-neutral robust source set; it does not run Leiden, retune
  selectors, replace source discovery, evaluate quality/cost, replay
  NanoClustering, or claim an algorithm.
- `run_leiden_cpm_variable_pair_synthetic_g4_8g_fresh_context_signature_validation.py`:
  freezes the G4.8F construction-read signature split and reruns the fixed
  G4.3 handle, G4.5 selector, and G4.6 schedule across fresh direct/host
  contexts. It validates whether the `R/T/N` centerline signature pattern
  survives context changes; it is not selector retuning, source-discovery
  replacement, wall/pathway promotion, quality/cost evaluation,
  NanoClustering replay, or an algorithm claim.
- `analyze_leiden_cpm_variable_pair_synthetic_g4_8h_source_discovery_smoke.py`:
  reads materialized G4.8G endpoint and bridge-release initialization rows and
  applies a target-free source-discovery rule to recover release-source and
  ready-source signature sets. It separates decision columns from
  evaluation-only oracle columns; it is not selector retuning, a new Leiden run,
  wall/pathway promotion, quality/cost evaluation, NanoClustering replay, or an
  algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_8i_discovered_source_schedule_panel.py`:
  runs a fresh predeclared edge-mid direct/host panel and drives schedule
  accounting with the frozen G4.8H target-free source-discovery rule rather
  than oracle source-signature reads. It is a synthetic schedule-accounting
  diagnostic only, not wall/pathway promotion, quality/cost evaluation,
  NanoClustering replay, real-data source discovery, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_8j_off_center_failure_mode_panel.py`:
  runs the first off-center failure-mode expansion by shifting bridge-host
  support by `-0.002/+0.002` around the G4.8I centerline while using the
  frozen G4.8H target-free source-discovery rule and G4.3 handle. It tests
  whether the discovered-source schedule correctly no-ops in nonrobust and
  target-saturated failure modes; it is not selector retuning, wall/pathway
  promotion, quality/cost evaluation, NanoClustering replay, real-data source
  discovery, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_9_primitive_wall_demo.py`:
  runs a predeclared five-case synthetic primitive-wall panel inspired by the
  `local_pair_014` object-level wall evidence. It tests whether ordinary
  Leiden+CPM on a small variable-pair graph can reproduce a source-like to
  exclusive-target to source-like relation while boundary controls close. This
  is a synthetic mechanism demo only, not NanoClustering replay, wall/pathway
  generality, quality/cost evaluation, method evidence, or an algorithm claim.
- `run_leiden_cpm_variable_pair_synthetic_g4_9a_parameter_localization.py`:
  maps three predeclared two-dimensional slices around the G4.9 primitive-wall
  positive point: direct/pair-bridge, pair-bridge/bridge-host, and
  direct/bridge-host. It classifies full-ready, partial-ready, target-absent,
  target-saturated, and nonrobust boundary regimes. This is synthetic
  mechanism localization only, not selector retuning, NanoClustering replay,
  wall/pathway generality, quality/cost evaluation, method evidence, or an
  algorithm claim.
