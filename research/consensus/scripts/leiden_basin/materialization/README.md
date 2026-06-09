# Materialization

Target location for cache, membership, prepare, join, and materialization
scripts.

Current basin-surface claim language is governed by
`../../../../../docs/research/leiden_basin/core/leiden_basin_surface_claim_schema.md`.
New Track C basin audits should report `surface_level`, `object_status`,
`relation_status`, and `claim_status` before using stronger basin, wall,
pathway, method, or quality wording. Use `surface_claim_schema_adapter.py` for
the shared required-column vocabulary, value validation, and case mapping.

- `surface_claim_schema_adapter.py`: reusable surface-claim schema adapter for
  Track C basin-surface audits. It defines the required columns, allowed
  vocabulary, case-row construction, count helpers, mapping-by-case helper, and
  validation used before route execution or label-promotion logic.
- `materialize_leiden_basin_methodology_v0_panel.py`: builds the
  precommitted non-field34 methodology-v0 panel from existing basin-existence
  and calibration artifacts. It does not run routes or inspect quality/cost.
- `enrich_leiden_basin_methodology_v0_evidence.py`: joins endpoint identity,
  signature, distance, source, and cache-availability evidence onto the
  methodology-v0 panel. It does not load memberships, run routes, or inspect
  quality/cost.
- `review_leiden_basin_methodology_v0_wall_pathway_schema.py`: reviews the M2
  pair evidence against the v0 wall/pathway evidence schema using existing
  current-review and blocker ledgers. It does not execute routes, promote wall
  claims, or inspect quality/cost.
- `audit_leiden_basin_methodology_v0_partial_wall_traces.py`: audits the two
  M3 partial-wall protocol references against existing W1-W6 route trace
  artifacts. It does not execute routes, promote wall claims, or inspect
  quality/cost.
- `materialize_leiden_basin_nanoclustering_external_landscape.py`: reads
  external NanoClustering hierarchy membership artifacts and materializes a
  Track C endpoint-landscape registry, pairwise diagnostics, and seed-ensemble
  reference-cluster persistence summaries. It does not run clustering, execute
  routes, promote wall/pathway claims, or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_volatile_boundary_cases.py`: expands
  the most volatile NanoClustering seed-ensemble reference clusters into
  split/merge boundary case packets. It does not run clustering, execute routes,
  promote wall/pathway claims, or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_matched_controls.py`: matches the
  volatile NanoClustering reference clusters to stable same-branch controls with
  similar weight and unit count, then computes the same split/merge diagnostics.
  It does not run clustering, execute routes, promote wall/pathway claims, or
  inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_matched_control_deltas.py`: compares
  volatile NanoClustering boundary cases against the matched stable controls at
  pair and threshold levels. It keeps fragmentation and absorption as separate
  endpoint-boundary axes and does not run routes, promote wall/pathway claims,
  or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_fragmentation_boundary_inventory.py`:
  applies the matched-control-calibrated top-split fragmentation axis to the
  full NanoClustering seed0 reference-cluster universe. It materializes a
  primitive endpoint-boundary rule family and does not run routes, promote
  wall/pathway claims, or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_fragmentation_stratified_panel.py`:
  expands a stratified sample from the fragmentation inventory into split/merge
  archetype rows. It compares persistent, recurrent, single, moderate, and
  stable-like strata inside the same NanoClustering endpoint universe without
  running routes, promoting wall/pathway claims, or inspecting quality/cost.
- `materialize_leiden_basin_nanoclustering_recurrent_boundary_family_registry.py`:
  promotes recurrent strong fragmentation rows into endpoint-boundary family
  candidates, family-tier summaries, and a pair-construction panel. It keeps
  definition-core, stress-test, and edge-control tiers separate without running
  routes, promoting wall/pathway claims, or inspecting quality/cost.
- `materialize_leiden_basin_nanoclustering_definition_core_pair_cases.py`:
  expands the pair-construction panel's definition-core families into concrete
  seed0-reference to comparison-seed endpoint-pair cases. It reads memberships
  only and does not run routes, promote wall/pathway claims, or inspect
  quality/cost. The default scope preserves the 20-family pair panel; use
  `--family-selection-scope all_definition_core` for the full 179-family
  definition-core expansion.
- `materialize_leiden_basin_nanoclustering_basin_distinction_panel.py`:
  separates definition-core endpoint-pair cases into observed endpoint-handle
  basin candidates, source-target relation rows, and family-level distinction
  classes. It reads existing membership-derived endpoint-pair artifacts only
  and does not run clustering, execute routes, promote wall/pathway claims, or
  inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_basin_vector_panel.py`: refines the
  distinction pass by materializing split-segment vectors and dominant
  merge-host context for each definition-core endpoint-pair event. It reads
  existing membership-derived split/merge artifacts only and does not run
  clustering, execute routes, promote wall/pathway claims, or inspect
  quality/cost.
- `analyze_leiden_basin_nanoclustering_basin_vector_coherence.py`: checks
  within-family repeatability of the NanoClustering endpoint-vector families by
  comparing split-vector class, shape core, dominant host context, and dominant
  host handle. It reads the v1 basin-vector panel only and does not run
  clustering, execute routes, promote wall/pathway claims, or inspect
  quality/cost.
- `materialize_leiden_basin_nanoclustering_definition_core_v1_registry.py`:
  reads the full 179-family basin-vector coherence panel and materializes the
  current v1 primitive basin-family registry. It accepts only coherent
  vector-and-host endpoint-vector families and leaves numeric-stress,
  split-coherent host-variable, host-coherent split-mixed, and heterogeneous
  rows as definition-refinement queues.
- `materialize_leiden_basin_nanoclustering_definition_core_v1_refinement_queue_decomposition.py`:
  decomposes the v1 nonaccepted refinement queues into support-local
  subfamilies and compares candidate split axes. It records recovered coherent
  endpoint-vector subfamilies while excluding singleton/tiny partitions from
  promotion and does not run clustering, execute routes, promote wall/pathway
  claims, or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_definition_core_v2_registry.py`:
  combines accepted v1 coherent families with primary-axis recovered coherent
  refinement subfamilies into a v2 primitive registry, and keeps tiny,
  unresolved, and alternative-axis-only rows in a definition-audit queue. It
  does not run clustering, execute routes, promote wall/pathway claims, or
  inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_definition_core_v2_audit_surface.py`:
  reviews the v2 registry's support-depth sensitivity, primary-vs-best axis
  disagreements, rule-revision candidates, and residual definition-audit rows.
  It keeps this as definition-audit evidence and does not run clustering,
  execute routes, promote wall/pathway claims, or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_definition_core_v2_1_registry.py`:
  freezes the v2 primitive set as a v2.1 registry by adding support-depth
  confidence tiers, retaining marginal primary-axis caveats, and moving strong
  or weak axis exceptions into a non-promoted ledger. It does not run
  clustering, execute routes, promote wall/pathway claims, or inspect
  quality/cost.
- `analyze_leiden_basin_nanoclustering_definition_core_v2_1_detail_review.py`:
  inspects the v2.1 registry internals by confidence tier, thin-support source
  concentration, best-axis exception subfamilies, and residual definition
  priorities. It recomputes exception best-axis subfamilies from event-vector
  rows but keeps them non-promoted and does not run clustering, execute routes,
  promote wall/pathway claims, or inspect quality/cost.
- `materialize_leiden_basin_nanoclustering_definition_core_v2_1_axis_rule_candidates.py`:
  materializes second-axis, joint-axis, and strong exception-axis candidate
  rules from existing event-vector rows. It keeps every recovered candidate as
  a definition-rule candidate rather than a promoted primitive and does not run
  clustering, execute routes, promote wall/pathway claims, or inspect
  quality/cost.
- `materialize_leiden_basin_nanoclustering_definition_core_v2_2_exception_axis_registry.py`:
  inherits the v2.1 primitive registry and adds only the strong exception-axis
  recovered coherent subfamilies as v2.2 definition primitives. It keeps
  second-axis and joint-axis candidates out of promotion, carries singleton
  exception-axis rows as explicit tiny holdouts, and does not run clustering,
  execute routes, promote wall/pathway claims, or inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_definition_core_v2_2_next_step_options.py`:
  compares the two post-v2.2 choices: continuing second/joint-axis definition
  work versus freezing v2.2 as the operational definition surface. It keeps
  residual candidates as a rule-design ledger, recommends v2.2 freeze with
  residual-debt metadata under the current evidence, and does not run
  clustering, execute routes, promote wall/pathway claims, or inspect
  quality/cost.
- `analyze_leiden_basin_nanoclustering_v2_2_instrumentation_surface.py`:
  audits the frozen v2.2 definition as a downstream measurement substrate by
  joining accepted primitives and residual debt to the recurrent family
  registry, stress-test strata, edge-case controls, stratified context, and
  matched stable controls. It does not run clustering, execute routes, promote
  wall/pathway claims, or inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_v2_2_measurement_panel.py`:
  materializes the first accepted-primitive measurement panel from the frozen
  v2.2 surface. It emits primitive, event, source-family, support-summary, and
  gate tables for support depth, endpoint-vector composition, host-handle
  concentration, boundary-pattern modes, and residual caveats without running
  clustering, executing routes, promoting wall/pathway claims, or inspecting
  quality/cost.
- `analyze_leiden_basin_nanoclustering_v2_2_measurement_distribution_review.py`:
  reviews distributions inside the accepted-primitive measurement panel and
  emits descriptive bands for stable high-support primitives, thin-clean
  primitives, residual-debt caveats, and host/shape/boundary concentration
  caveats. It does not change the v2.2 definition and does not run clustering,
  execute routes, promote wall/pathway claims, or inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_v2_2_claim_tier_ladder.py`: turns the
  v2.2 distribution review into a six-tier cumulative claim ladder from the
  83-primitive headline nucleus to the full 223 accepted-primitives-with-caveats
  scope. It changes wording and accounting only; it does not change the v2.2
  definition, run clustering, execute routes, promote wall/pathway claims, or
  inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_seed_anchor_rotation.py`: rotates the
  NanoClustering reference seed across the pure Java/Rust seed ensembles and
  checks whether recurrent endpoint-boundary structure survives outside seed0.
  It is membership-only endpoint cartography; it does not build symmetric
  anchor-independent basin objects, run routes, promote wall/pathway claims, or
  inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_symmetric_endpoint_objects.py`: treats
  every `(branch, seed, cluster_id)` endpoint cluster as a node, builds
  cross-seed overlap edges, and materializes symmetric all-seed endpoint
  objects. It tests anchor dependence only; it does not run clustering,
  execute routes, promote wall/pathway claims, inspect quality/cost, or claim a
  method improvement.
- `analyze_leiden_basin_nanoclustering_symmetric_object_decomposition.py`:
  decomposes the symmetric endpoint-object audit into stable one-per-seed
  objects, multi-cluster objects, anchor-local fragments, seed0 mapping failure
  modes, and mechanism-probe candidates. It is a definition diagnostic only
  and does not run routes, promote wall/pathway claims, inspect quality/cost,
  or claim a method improvement.
- `analyze_leiden_basin_nanoclustering_dataset_contrast.py`: contrasts the
  prior Track C evidence surface against the current NanoClustering endpoint
  surface and records the density-hypothesis read. It reads existing summaries
  and membership-derived endpoint artifacts only, and does not run clustering,
  execute routes, promote wall/pathway claims, or inspect quality/cost.
- `analyze_leiden_basin_nanoclustering_joint_weak_pair_analog_screen.py`:
  screens the frozen v2.2 accepted-primitive measurement panel for real-data
  analogs of the tiny Stress 4 `v1.2` joint weak-pair mechanism. It emits
  primitive analog tiers, source-family rollups, matched control-like rows,
  gates, summary, and report artifacts. It is read-only and does not run
  clustering, construct method candidates, execute routes/pathways, promote
  walls, inspect quality/cost, or claim real-data method success.
- `design_leiden_basin_nanoclustering_joint_weak_pair_local_panel.py`:
  consumes the joint weak-pair analog screen and freezes NanoClustering local
  panel case rows, analysis tiers, candidate/control role rows, event-role
  rows, endpoint-family signature rows, core one-to-one control sensitivity
  rows, and the future endpoint-replay contract. It is a design artifact only
  and does not run clustering, execute endpoint replay, execute routes/pathways,
  promote walls, inspect quality/cost, or claim real-data method success.
- `check_leiden_basin_nanoclustering_endpoint_replay_readiness.py`: resolves
  the frozen local panel's endpoint-family handles to NanoClustering membership
  artifacts, freezes the strict-core symmetric candidate/control attempt plan,
  and checks whether the branch-specific raw graph inputs needed for endpoint
  replay are locally available. It is a readiness runner only and does not run
  clustering, execute endpoint replay, execute routes/pathways, promote walls,
  inspect quality/cost, or claim real-data method success.
- `run_leiden_basin_nanoclustering_endpoint_replay_pilot.py`: executes a
  bounded endpoint-replay pilot from the readiness artifact. It starts from the
  frozen source seed0 endpoint partition, runs the NanoClustering
  `Leiden -> min_nano postprocess -> min_docs postprocess` sequence, and scores
  terminal partitions against endpoint-family target handles. It is a pilot
  smoke only; route/pathway, wall, quality/cost, real-data method success, and
  algorithm claims remain closed.
- `materialize_leiden_basin_nanoclustering_role_local_boundary_plan.py`:
  materializes source/target node-mask objects and fixed-outside route
  contracts for the strict-core NanoClustering roles. It replaces
  full-partition warm-start replay with role-local boundary objects, but it
  does not run clustering or promote route/wall claims.
- `run_leiden_basin_nanoclustering_role_local_route_pilot.py`: executes bounded
  raw fixed-mask route smokes from the role-local boundary plan. It supports
  pair-free target seeding and source/target anchor arms, blocks empty-free-mask
  attempts before calling Rust, and deliberately excludes postprocess because
  the current postprocess wrapper is not fixed-mask aware. The bounded
  source/target-anchor expansion confirms raw local manipulability across the
  strict-core panel, but endpoint and wall claims remain closed until a
  fixed-mask-aware endpoint or postprocess readout is implemented.
- `run_leiden_basin_nanoclustering_fixed_mask_endpoint_readout_pilot.py`:
  extends the role-local route pilot with a bounded fixed-mask-aware readout.
  It reruns source/target anchor arms and applies min-nano plus min-doc
  constrained Leiden stages where only small non-fixed nodes may move. This is
  a readout pilot, not the production postprocess wrapper.
- `run_leiden_basin_nanoclustering_anchor_release_pilot.py`: reruns the
  source/target anchor terminals and releases both into the same fixed-outside
  pair mask. This is the common-feasible-set diagnostic that separates
  boundary-condition manipulability from wall-facing non-collapse evidence.
  It supports bounded geometry-stress selections such as `lowest_target_overlap`
  and `largest_pair_free`.
- `analyze_leiden_basin_nanoclustering_anchor_release_policy_comparison.py`:
  compares completed anchor-release pilot outputs across selection policies and
  de-duplicates role-target pairs. It is read-only over completed route
  artifacts and does not execute clustering, routes, wall/pathway promotion, or
  quality/cost analysis.
- `run_leiden_basin_nanoclustering_common_mask_multistart_pilot.py`: runs
  source-state, target-seeded, pair-singleton, source/target two-block, and
  random pair-block initializations under the same fixed-outside pair mask. It
  tests terminal multiplicity inside the common feasible set and does not
  promote wall/pathway, quality/cost, real-data method-success, or algorithm
  claims.
- `design_leiden_basin_nanoclustering_basin_universe_redesign.py`: materializes
  the next movable-universe candidates after local pair-mask collapse. It
  creates signature-level endpoint-family universes, candidate/control
  case-union candidates, and symmetric-object resolver rows without running
  clustering, routes/pathways, wall promotion, or quality/cost analysis.
- `run_leiden_basin_nanoclustering_signature_universe_multistart_pilot.py`:
  executes bounded multistart diagnostics over endpoint-family signature
  universes, where the movable set is dominant-host source handles plus all
  top1 endpoint handles for the signature. It tests terminal multiplicity only
  and does not promote wall/pathway, quality/cost, real-data method-success, or
  algorithm claims.
- `materialize_leiden_basin_nanoclustering_case_universe_plan.py`: turns the
  design-only candidate/control case-union rows into executable fixed-outside
  universe contracts with actual OR-union masks, overlap diagnostics, optional
  candidate/control edge-interaction metrics, route-readiness flags, and
  interaction-gated pilot priorities. It is a materialization/design artifact
  only and does not run clustering, promote wall/pathway, inspect quality/cost,
  or claim method success.
- `materialize_leiden_basin_nanoclustering_symmetric_object_universe_plan.py`:
  resolves the design-only symmetric-object rows into executable fixed-outside
  role-level object-mask contracts using the all-seed endpoint-object registry.
  It writes role universes, candidate/control case-relation diagnostics, and a
  gate matrix for the next object-level multistart pilot. It is a
  materialization/design artifact only and does not run clustering, promote
  wall/pathway, inspect quality/cost, or claim method success.
- `run_leiden_basin_nanoclustering_symmetric_object_multistart_pilot.py`:
  executes bounded multistart diagnostics over role-level symmetric-object
  universes. The movable set is the all-seed object mask, optionally expanded
  by top-k boundary-weight support-neighborhood nodes, with starts from the
  seed0 source state, seed0-object seeded state, object singleton,
  component-pattern blocks, and random object blocks. It tests terminal
  multiplicity and singleton collapse only. With
  `--save-terminal-memberships`, it also writes compact object/universe
  terminal membership slices under `terminal_memberships/` and
  `nanoclustering_symmetric_object_multistart_terminal_pair_rows.csv` for
  within-object start-pair terminal ARI. These pair rows are terminal-structure
  diagnostics only; they do not promote wall/pathway, quality/cost, real-data
  method-success, or algorithm claims.
- `analyze_leiden_basin_nanoclustering_symmetric_object_merge_viability.py`:
  audits whether selected symmetric-object/support-neighborhood universes have
  objective-positive CPM free-free merge candidates or free-to-fixed attachment
  candidates from a singleton baseline. It is a mechanism diagnostic for
  singleton collapse and support-neighborhood failure; it does not run Leiden,
  promote wall/pathway, inspect basin quality/cost as a success claim, or claim
  method success.
- `analyze_leiden_basin_nanoclustering_symmetric_object_objective_mechanisms.py`:
  audits the objective scale behind symmetric-object merge viability by
  computing critical-gamma thresholds and alternative node-weight transforms for
  the same fixed-outside universes. It identifies candidate mechanisms such as
  lower critical gamma or local weight normalization before another
  terminal-multiplicity run; it does not promote wall/pathway, basin-quality,
  cost, method-success, or algorithm claims.
- `analyze_leiden_basin_nanoclustering_symmetric_object_terminal_membership_differences.py`:
  reads saved compact terminal membership slices from the symmetric-object
  multistart runner and decomposes terminal multiplicity into start-policy
  terminal groups, start-pair co-assignment changes, variable nodes, and
  variable node-pairs. It is read-only over completed multistart artifacts and
  does not run Leiden, promote wall/pathway, inspect basin quality/cost as a
  success claim, or claim method success.
- `analyze_leiden_basin_nanoclustering_symmetric_object_variable_pair_graph_mechanisms.py`:
  reads the saved variable terminal node-pairs and scores graph-local mechanism
  evidence: direct pair edge weight, doc-weighted CPM pair delta, direct
  critical gamma, and shared-neighbor bridge mass. It is read-only over
  completed difference-review artifacts and does not run Leiden, promote
  wall/pathway, inspect basin quality/cost as a success claim, or claim method
  success.
- `design_leiden_basin_nanoclustering_symmetric_object_variable_pair_counterfactual_panel.py`:
  reads the graph-scored variable node-pairs and freezes a small predeclared
  counterfactual panel for the next local ablation or controlled demo. It
  computes direct-edge-removal pair-delta shifts, input-versus-control-gamma
  margins, bridge-to-direct ratios, and selection reasons. It is read-only over
  completed graph-mechanism artifacts and does not run Leiden, promote
  wall/pathway, inspect basin quality/cost as a success claim, or claim method
  success.
- `run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation.py`:
  consumes the frozen variable-pair counterfactual panel, builds small induced
  graphs from each pair plus recoverable top common-neighbor bridge nodes, and
  runs ordinary Leiden+CPM across direct-edge and pair-to-bridge edge-removal
  variants. It is a local mechanism diagnostic only; it does not run the full
  NanoClustering graph, execute routes/pathways, promote walls, inspect
  quality/cost as a success claim, or claim method success.
- `design_leiden_basin_nanoclustering_symmetric_object_variable_pair_synthetic_demo.py`:
  reads the frozen counterfactual panel and local ablation gate, then derives
  the controlled synthetic-demo family surface: direct-contact competition,
  coupled negative-direct bridge contact, rare start-sensitive contact, and
  explicit negative controls. It is a design artifact only and does not run
  Leiden, execute full-graph replay, promote walls/pathways, inspect
  quality/cost as a success claim, or claim method success. The corresponding
  ordinary-Leiden+CPM runner lives in `../demo/` because it is a controlled
  demo, not a NanoClustering materialization step.
- `analyze_leiden_basin_nanoclustering_g4_8_source_condition_analog_screen.py`:
  reads the frozen symmetric-object variable-pair graph-mechanism,
  local-ablation, and synthetic-demo-design artifacts to classify local
  real-data analogs of the synthetic G4.8 source-condition roles. It is
  read-only over existing artifacts and does not run Leiden, execute
  routes/pathways, promote walls, evaluate wall-clock quality/cost value,
  replay full NanoClustering, or claim method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_local_analog_validation_panel.py`:
  consumes the G4.8 source-condition analog screen and freezes all 23 local
  pairs into a stratified local validation panel. It keeps strict-ready,
  rare-ready, target-saturated no-handle, nonready, and coupled failure-control
  strata without within-stratum cherry-picking. It is a design/materialization
  script only and does not run Leiden, execute routes/pathways, promote walls,
  evaluate wall-clock quality/cost value, replay full NanoClustering, or claim
  method or algorithm success.
- `analyze_leiden_basin_nanoclustering_g4_8_local_validation_readout.py`:
  reads that frozen local analog panel plus the existing local-ablation seed
  runs, splits seeds `0-3` versus held-out seeds `4-7`, and materializes
  endpoint-derived source-signature proxies and seed/start-stratified readout
  rows. It is read-only over existing runs and does not run Leiden, execute
  routes/pathways, promote walls, evaluate wall-clock quality/cost value,
  replay full NanoClustering, or claim method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_seed_start_validation_contract.py`:
  consumes the local validation readout and freezes stable, conditional, and
  boundary execution lanes. Stable rows can feed the next local validation
  contract, conditional rows must be restricted to their listed allowed start
  conditions, and boundary rows remain diagnostic controls. It is a
  design/materialization script only and does not run Leiden, execute
  routes/pathways, promote walls, evaluate wall-clock quality/cost value,
  replay full NanoClustering, or claim method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_local_validation_execution_contract.py`:
  consumes the seed/start validation contract and freezes validation units for
  the next local validation step. Primary units are stable-lane-only, while
  conditional allowed starts and boundary allowed starts are preserved as
  separate secondary and diagnostic lanes. It is a design/materialization
  script only and does not run Leiden, execute routes/pathways, promote walls,
  evaluate wall-clock quality/cost value, replay full NanoClustering, or claim
  method or algorithm success.
- `analyze_leiden_basin_nanoclustering_g4_8_primary_stable_limit_readout.py`:
  consumes the local validation execution contract and inspects only the stable
  primary units. It materializes existing-Leiden limitation axes for ready
  partial release, target saturation, latent release without original
  coassigned source, hard no-release controls, and coupled direct/bridge
  failures. It is read-only and does not run Leiden, execute routes/pathways,
  promote walls, evaluate wall-clock quality/cost value, replay full
  NanoClustering, or claim method or algorithm success.
- `audit_leiden_basin_nanoclustering_g4_8_pathway_wall_readiness.py`:
  consumes the primary stable limitation readout and separates scoped
  pathway-probe candidates from wall-claim evidence. It allows only the two
  ready pairs to feed a predeclared Stage 2A pathway-probe design and keeps all
  wall claims closed until route traces and wall-evidence fields exist. It is
  read-only and does not run Leiden, execute routes/pathways, promote walls,
  evaluate wall-clock quality/cost value, replay full NanoClustering, or claim
  method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_contract.py`:
  consumes the pathway/wall readiness audit and freezes only the two ready
  pairs into a tiny Stage 2A route-plan contract: 10 start-conditioned probe
  units, 30 predeclared route-plan rows, and 65 retained false-positive control
  guards. It is a design/materialization script only and does not run Leiden,
  execute routes/pathways, promote walls, evaluate wall-clock quality/cost
  value, replay full NanoClustering, or claim method or algorithm success.
- `run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace.py`:
  consumes the scoped pathway-probe contract and executes only the 30
  predeclared route-plan rows on local induced graphs using fixed
  edge-weight-fraction schedules. It materializes route trace, objective,
  endpoint-assignment, support-distance, polish-reversion, and
  support-incompatibility fields. It does not run full NanoClustering replay,
  promote walls, evaluate wall-clock quality/cost value, or claim method or
  algorithm success.
- `audit_leiden_basin_nanoclustering_g4_8_scoped_pathway_wall_evidence.py`:
  consumes the executed scoped pathway-probe trace and classifies
  pathway-trace evidence versus wall readiness. It identifies primary
  bridge-release contracts as wall-audit candidates, keeps direct-dependency
  partial guards and drop-both collapse guards separate, and keeps wall claims
  closed. It does not run Leiden, broaden route execution, promote walls,
  evaluate wall-clock quality/cost value, replay full NanoClustering, or claim
  method or algorithm success.
- `audit_leiden_basin_nanoclustering_g4_8_primary_bridge_release_pathway_shape.py`:
  consumes only the already executed primary bridge-release trace rows and
  separates physical direct-edge retention, seed-level known-anchor direct-path
  candidates, intermediate unknown/support-incompatible routes, and objective
  debt/recovery shape. It keeps direct-path evidence at candidate level and
  keeps wall claims closed. It does not run Leiden, broaden route execution,
  promote walls, evaluate wall-clock quality/cost value, replay full
  NanoClustering, or claim method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_direct_path_acceptance_contract.py`:
  consumes the primary bridge-release pathway-shape audit and fixes D1-D9
  direct-path acceptance rules before any new execution. It preserves
  seed-level direct-path candidates, evaluates strict all-seed contract
  acceptance, keeps objective recovery separate, and keeps wall claims closed.
  It does not run Leiden, broaden route execution, promote walls, evaluate
  wall-clock quality/cost value, replay full NanoClustering, or claim method or
  algorithm success.
- `audit_leiden_basin_nanoclustering_g4_8_cross_seed_endpoint_atlas.py`:
  consumes the executed primary bridge-release trace rows and reclassifies
  same-seed `unknown_new_endpoint` labels against pair-level endpoint
  signatures. It separates same-seed anchor consistency from pair-level
  endpoint-atlas continuity, and keeps wall claims closed. It does not run
  Leiden, broaden route execution, promote walls, evaluate wall-clock
  quality/cost value, replay full NanoClustering, or claim method or algorithm
  success.
- `design_leiden_basin_nanoclustering_g4_8_dual_axis_direct_path_contract.py`:
  consumes the existing v1 direct-path contract, cross-seed endpoint atlas, and
  scoped route trace to materialize a two-axis contract. Axis A preserves
  strict same-seed anchor consistency; Axis B tests pair-level endpoint-atlas
  source-to-target continuity. It keeps same-seed unknown/support flags as
  diagnostics, keeps objective recovery separate, and keeps wall claims closed.
  It does not run Leiden, broaden route execution, promote walls, evaluate
  wall-clock quality/cost value, replay full NanoClustering, or claim method or
  algorithm success.
- `audit_leiden_basin_nanoclustering_g4_8_axis_b_seed_anchor_rotation.py`:
  consumes the scoped route trace and dual-axis direct-path contract, then
  rebuilds Axis B endpoint roles under full-pair, leave-start-out,
  leave-seed-out, and leave-seed-and-start-out vocabularies. It tests whether
  same-seed unknown reinterpretation and route-level endpoint continuity survive
  seed/start anchor rotation, while keeping source-start support caveats
  separate from interior endpoint evidence. It does not run Leiden, broaden
  route execution, promote walls, evaluate wall-clock quality/cost value, replay
  full NanoClustering, or claim method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_axis_b_source_start_support_contract.py`:
  consumes the Axis B seed-anchor rotation audit and materializes a split
  contract that records source-start support separately from post-start endpoint
  continuity, target-final continuity, and direct-edge retention. It preserves
  the two source-start singleton caveats and prevents interior endpoint evidence
  from repairing them. It does not run Leiden, broaden route execution, promote
  walls, evaluate wall-clock quality/cost value, replay full NanoClustering, or
  claim method or algorithm success.
- `design_leiden_basin_nanoclustering_g4_8_fresh_axis_b_panel_contract.py`:
  consumes the local validation, pathway-readiness, and source-start split
  artifacts to freeze the next fresh Axis B panel. It keeps `local_pair_009`
  and `local_pair_012` as calibration-only rows, predeclares not-yet-routed
  ready-like pairs, and bounds the first-pass fresh slice to conditional
  ready-like rows plus one control per blocked/control limitation axis. It does
  not run Leiden, execute route/pathway traces, promote walls, evaluate
  quality/cost value, replay full NanoClustering, or claim method success.
- `design_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_readout_contract.py`:
  consumes the fresh Axis B panel contract and fixes the readout protocol for
  the 36 first-pass rows. It materializes claim ladder, required readout fields,
  four control traps, aggregation rules, outcome taxonomy, readout order, and
  route/pair readout rows. It makes controls-first interpretation mandatory and
  closes stable-positive, direct-dependent, branch, basin/pathway, wall,
  quality/cost, full replay, and method claims for the first pass. It does not
  run Leiden or execute route/pathway traces.
- `run_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace.py`:
  consumes the first-pass readout contract and executes exactly the 36
  predeclared rows under the bridge-release interpolation schedule. It writes
  trace rows, seed-route summaries, route/pair/control readout results, and a
  gate matrix. The readout requires an exclusive `drop_bridge_target_anchor`
  before any ready-like screen pass, because raw target-final continuity can
  also match guard anchors. It keeps wall, quality/cost, full replay, and method
  claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_exclusive_target_contrast.py`:
  reads the executed first-pass trace and classifies each route into exclusive
  bridge-target pass, source/target signature collapse, guard-anchor collapse,
  or intermediate unknown endpoint. It materializes pair, route, signature, and
  gate rows so the next object-level audit is bounded to the clean candidate and
  one partial boundary case. It does not rerun Leiden or open wall, quality/cost,
  full replay, or method claims.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects.py`:
  reads the first-pass trace and exclusive-target contrast rows for
  `local_pair_014` and `local_pair_005` only. It materializes endpoint-object
  rows, source-to-final object relation rows, pair summaries, and gates. It
  confirms `local_pair_014` as the only positive object-level candidate and
  keeps `local_pair_005` as a source/target-collapse boundary control. It does
  not rerun Leiden or open wall, quality/cost, full replay, or method claims.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_wall_pathway_readiness.py`:
  reads the first-pass trace and symmetric endpoint-object audit rows for
  `local_pair_014` and `local_pair_005`. It classifies `local_pair_014` as the
  only pathway-probe candidate and keeps `local_pair_005` as the boundary
  control. It records direct-edge retention, bridge-fraction schedule, endpoint
  timing, objective debt/recovery shape, support incompatibility, and missing
  wall-evidence fields. It does not rerun Leiden or open wall, quality/cost,
  full replay, or method claims.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_contract.py`:
  reads the first-pass wall/pathway-readiness audit and predeclares the next
  `local_pair_014` pathway-probe contract. It creates 16 route-plan rows:
  positive `014` recovery-loop/direct-only probes and matched `005` boundary
  controls. It fixes independent direct-path availability, accepted recovery,
  boundary no-leak, and wall-claim closure rules. It does not run Leiden or open
  wall, quality/cost, full replay, or method claims.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_trace.py`:
  executes exactly the 16 first-pass `014` pathway-probe contract rows with new
  recovery-loop and direct-only schedules. It writes 88 route-step configs and
  704 trace rows over 8 seeds, accepts `014` direct-only and recovery probes at
  32/32 seed-routes each under object-level endpoint assignment, and keeps `005`
  closed with 0/64 positive leaks. It treats source/drop-direct anchor
  coincidence as source-like object evidence and does not open wall,
  quality/cost, full replay, or method claims.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence.py`:
  reads the executed first-pass `014` pathway-probe trace and pairs the
  accepted direct-only and recovery-loop routes by identical start condition and
  seed. It accepts `local_pair_014` as local primitive object-level wall evidence
  at 32/32 wall seed units and keeps the matched `local_pair_005` boundary guard
  closed at 32/32 units. It opens no generalization, exact wall-location,
  quality/cost, full replay, or method claim.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_contract.py`:
  reads the accepted `014` primitive wall-evidence audit and the synthetic
  G4.9A parameter-localization map, then freezes a fine bridge-fraction
  localization contract. It materializes 16 route-plan rows and 192 fraction
  steps for `014` descent/ascent scans plus retained `005` boundary guards. It
  also carries the G4.9A `W/w/T/N/P` boundary vocabulary into the real-data
  readout. It is design-only and does not run Leiden, retune thresholds,
  promote wall generality, evaluate quality/cost, replay full NanoClustering,
  or claim method success.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace.py`:
  executes the fixed `014` wall-localization contract without requiring a
  single expected final anchor. It materializes 1,536 fraction-level trace rows,
  128 route-scan rows, and 64 paired seed-start localization rows. The strict
  G4.9A readout finds 1/32 positive seed-start units as `W` and 31/32 as `P`;
  the retained `005` boundary has 0 positive W-like leaks. This is a trace-only
  execution and opens no generality, quality/cost, full replay, or method claim.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_transition_bands.py`:
  reads the executed wall-localization trace and separates strict wall
  intervals from transition bands. It classifies `014` as 1 strict interpretable
  wall-interval seed, 30 monotone intermediate transition-band seeds, and 1
  bounded nonmonotone transition-band seed; all 32/32 positive seed-starts are
  bounded source-target transition bands. The `005` boundary has 0
  positive-target routes and 0 positive-target steps. This is a read-only audit
  and keeps method, quality/cost, full replay, and wall-generality claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity.py`:
  reads the executed localization trace and the transition-band audit, then
  separates row-local unresolved endpoint assignments from signature-level
  unresolved intermediate objects. It finds that `014` has 152 row-local
  unresolved rows: 98 are signatures known elsewhere, while 54 remain true
  signature-level unresolved intermediate rows across two recurrent signatures.
  For `005`, all 204 row-local unresolved rows resolve to signatures known
  elsewhere, leaving 0 signature-level unresolved boundary rows. This is a
  read-only identity audit and opens no method, quality/cost, full replay, or
  wall-generality claim.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_014_intermediate_role_stability.py`:
  reads the executed `014` localization trace, the signature-identity audit, and
  local graph metadata to type transition-band signatures by `L/R/B1`-`B10`
  cluster roles. It types all 6 positive endpoint signatures: target anchor,
  two source-like anchors, hidden-known source/guard intermediate
  (`b7761471acbf`), unresolved pair-coassigned intermediate (`ca947e9fbe61`),
  and unresolved pair-separated bridge-reassignment intermediate
  (`531aa99db869`). It materializes 12 node-role rows, 6 signature-role rows,
  and 64 seed-route role sequences. It is read-only and keeps method,
  quality/cost, full replay, and wall-generality claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen.py`:
  reads the existing first-pass trace, exclusive-target contrast, `014`
  role-stability gates, and local graph metadata to screen all 9 first-pass
  pairs for the typed `014` role pattern. It materializes 38 signature-role
  rows and 288 route-role rows. `014` remains the only clean scaffold and there
  are 0 non-014 positive transfer candidates. It marks `016` as the primary
  strict-ready continuity-blocked diagnostic, `005` as the boundary guard,
  `008`/`022`/`002` as closed-control analogs, and `007`/`003` as secondary
  rare-ready blocked analogs. It is read-only and keeps method, quality/cost,
  full replay, and wall-generality claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block.py`:
  reads the existing first-pass trace, role-pattern transfer screen, and
  local-ablation outputs to localize why `016` fails post-start continuity. It
  materializes 5 pair-comparison rows, 49 step-signature rows, and 24 route
  diagnostic rows. It shows that all 24 `016` routes start source-like and end
  at the exclusive target, but all 24 pass through a single step-2
  bridge-reassignment signature (`aeb59ab537e6`) at bridge fraction 0.75 where
  `L+B1` is separated from `R`. It is read-only and keeps method, quality/cost,
  full replay, and wall-generality claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_transition_evidence_synthesis.py`:
  reads the current first-pass, exclusive-target, `014` localization,
  transition-band, signature-identity, role-stability, transfer-screen, and
  `016` continuity-block outputs. It materializes 9 pair-evidence rows, 8
  claim-evidence rows, and 5 definition-decision rows. It fixes the next design
  surface as explicit accept/reject predicates for typed transient intermediate
  pathways, tested against `014`, `016`, `005`, and closed controls before any
  new trace execution. It is read-only and keeps method, quality/cost, full
  replay, and wall-generality claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_typed_transient_predicate_screen.py`:
  reads the first-pass trace, role-pattern transfer screen, `016`
  continuity-block audit, and transition-evidence synthesis. It materializes 9
  pair-feature rows, 36 pair-predicate rows, 4 predicate rows, and 6 definition
  rows. The strict baseline accepts only `014`; the guarded single-step
  separated-transient candidate accepts only `016` with 0 guard leaks;
  endpoint-only and role-analog-only broadenings fail by leaking boundary,
  control, or rare-ready rows. It is read-only and keeps method, quality/cost,
  full replay, wall-generality, and positive-wall claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_semantic_validation.py`:
  reads the first-pass trace, role-pattern transfer screen, `016`
  continuity-block audit, and typed-transient predicate screen to classify the
  `016` step-2 transient. It materializes 24 route-semantic rows, 5
  step-semantic rows, 9 comparison rows, and 7 semantic-decision rows. The
  recurrent signature `aeb59ab537e6` appears in 24/24 routes as a typed
  separated bridge-reassignment gateway, but it is anchor-equidistant and has
  objective debt without recovery. It is read-only and keeps method,
  quality/cost, full replay, wall-generality, endpoint-basin, and positive-wall
  claims closed.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_contract.py`:
  reads the `016` transient semantic-validation artifact and predeclares a
  narrow fine bridge-fraction persistence scan for `local_pair_016` only. It
  fixes the three semantic-valid starts, eight seeds, nine bridge fractions
  around 0.75, and readout rules separating finite-band evidence from point
  saddle evidence. It is design-only and keeps method, quality/cost, full
  replay, wall-generality, endpoint-basin, and positive-wall claims closed.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace.py`:
  executes the `016` persistence contract and materializes 216 trace rows, 24
  route-persistence rows, 9 fraction-summary rows, and a gate matrix. The
  recurrent transient signature `aeb59ab537e6` appears in 24/24 routes across
  six adjacent bridge fractions, 0.625 through 0.8125, so it is a finite
  transition-gateway band rather than a 0.75 point artifact. It remains
  support-equidistant to original, drop-bridge, and drop-direct anchors and
  keeps wall/method/quality/full-replay claims closed.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_contract.py`:
  consumes the finite-band persistence trace and predeclares the same-seed
  target-anchor reverse scan. For each semantic-valid start and seed, it uses
  the same-seed `drop_bridge_edges` endpoint membership as the initial
  membership and restores bridge fraction from 0.5 to 1.0. It is design-only
  and keeps method, quality/cost, full replay, wall-generality, and positive
  wall claims closed.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace.py`:
  executes the reverse contract and materializes 216 trace rows, 24
  route-reversibility rows, 9 fraction-summary rows, and a gate matrix. The
  reverse trace starts at the target anchor in 24/24 routes, re-enters the same
  transient band at 0.625 through 0.8125 in 24/24 routes, and has mixed final
  restoration: 15/24 seed routes return to source at full bridge weight while
  9/24 do not. This is path-asymmetry/reversibility evidence only; the
  final-source-return gate fails as evidence, not as an execution failure, and
  all wall/method/quality/full-replay claims remain closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_reverse_nonreturn_stratification.py`:
  reads the executed `016` reverse trace and local-ablation anchor table to
  stratify the 9/24 strict source-return failures. It materializes 24 final
  state rows, route rows, stratum rows, seed-pattern rows, decision rows, and a
  gate matrix. The 9 non-return rows do not end in target or transient
  signatures: 8 are same-start source-family signatures that fail same-seed
  anchor reconciliation, and 1 is a same-seed drop-direct guard residual. This
  blocks target-hysteresis wording and makes source-family equivalence the next
  definition gate; it is read-only and keeps wall/method/quality/full-replay
  claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_source_family_equivalence.py`:
  reads the `016` reverse non-return stratification artifact and compares five
  candidate source-equivalence rules over the 24 reverse final states. It
  materializes route-rule rows, rule rows, decision rows, and a gate matrix.
  Strict same-seed source-anchor matching accepts 15/24 final states, while
  same-start source-family equivalence accepts 24/24 with 0 target/transient
  leaks and 1 named guard-overlap caveat. This fixes the operational reverse
  final-state vocabulary for `016`; it is read-only and keeps wall/method/
  quality/full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_pathway_shape.py`:
  reads the executed `016` persistence trace, executed reverse trace, and
  source-family equivalence artifact to reclassify the pathway shape under the
  fixed same-start source-family vocabulary. It materializes 24 route-shape
  rows, 9 fraction-ladder rows, decision rows, and a gate matrix. The forward
  and reverse traces share the same state ladder in 24/24 routes: source-family
  at 0.875 and 1.0, recurrent transient signature `aeb59ab537e6` at
  0.625-0.8125, and target anchor at 0.5. This names `016` as a
  source-family transition-band mechanism object; it is read-only and keeps
  wall/tunneling/method/quality/full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_objective_barrier_interpretation.py`:
  reads the executed `016` persistence trace, executed reverse trace, and
  pathway-shape audit to interpret objective profiles without rerunning Leiden.
  It materializes 9 objective-fraction rows, 24 route-objective rows, decision
  rows, and a gate matrix. Mean objectives are monotone with bridge fraction in
  both directions; the transient-band objective means match across
  forward/reverse traces; and 0/24 routes support fixed-landscape barrier
  language. This keeps the positive `016` claim at perturbation-relative
  source-family transition-band mechanism object; it is read-only and keeps
  wall/tunneling/method/quality/full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_mechanism_interpretation.py`:
  reads the local-ablation, semantic-validation, persistence/reverse trace,
  pathway-shape, and objective/barrier artifacts to name the `016` local
  mechanism without rerunning Leiden. It materializes local-substrate rows,
  variant-mechanism rows, fraction-mechanism rows, route-mechanism rows,
  decision rows, and a gate matrix. The mechanism class is
  direct-edge-sensitive bridge-mass competition with a single-side-bridge
  transition band: selected bridge mass is 13.676976 times the direct edge,
  edge-removal variants separate target and guard/source roles, and all 288
  transient route states are pair-separated single-side-bridge states. It is
  read-only and keeps generality/wall/tunneling/method/quality/full-replay
  claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen.py`:
  reads the existing local-ablation, local-validation, typed-predicate, and
  `016` mechanism artifacts to test fixed `016` mechanism predicates over the
  23-pair local panel without new Leiden execution. It materializes pair rows,
  class rows, next-gate rows, decision rows, and a gate matrix. The local
  `016` signature recurs in 7/23 pairs, but route-level P1 typed-transient
  acceptance remains `016`-only; the screen therefore leaves
  `G4_route_level_generality_not_yet_established` failed and narrows the next
  queue to strict nonboundary local-signature analogs `009`, `012`, and `020`,
  with `014` and `005` as controls. It is read-only and keeps
  route-generality/wall/tunneling/method/quality/full-replay claims closed.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract.py`:
  reads the mechanism-generalization screen and local-validation start rows to
  predeclare the narrow fixed-predicate route contract. It opens only
  source-family starts for strict analogs `009`, `012`, `020` and controls
  `014`, `005`, with the same 9 bridge fractions and direct fraction fixed at
  1.0. It materializes route-plan rows, fraction-step rows, readout rules, and
  a gate matrix. It is contract-only and keeps route-generality/wall/method/
  quality/full-replay claims closed.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace.py`:
  executes the fixed-predicate route contract on the existing local-ablation
  graph surface. It materializes 864 trace rows, 96 seed-route mechanism rows,
  5 pair rows, 45 fraction rows, and a gate matrix. The first strict analog
  queue is route-negative under the fixed `016` predicate: `009` and `020`
  move source-family to target-like without a single-side band, `012` has only
  single-fraction/nonfinite single-side rows, and controls `014`/`005` do not
  leak the full predicate. It keeps wall/method/quality/full-replay claims
  closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_route_negative_explanation.py`:
  reads the local-ablation, mechanism-generalization screen, `016`
  persistence, and fixed-predicate route-trace artifacts to explain why the
  strict analog queue stays route-negative. It materializes decision,
  fraction, gate, pair, substrate, summary, and report artifacts. `016` remains
  the finite single-side band reference with six adjacent all-route
  single-side fractions; `009` and `020` are abrupt source-to-target switches
  without a band, and `012` is point/seed-fragile rather than finite-band
  recurrence. It is read-only and keeps route-generality/wall/method/
  quality/full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_features.py`:
  reads the route-negative explanation, local-ablation, `016` persistence, and
  fixed-predicate route-trace artifacts to audit which feature candidates
  separate the `016` finite plateau from strict analog near misses. It
  materializes pair-feature, fraction-feature, decision, gate, summary, config,
  and report artifacts. The strongest current discriminator is an exact
  single-bridge latch (`left=1;right=0;pair=0`) with equal known-anchor support
  distance and seed/start-stable finite-band width; scalar bridge mass, direct
  delta, and bridge scope mix are rejected as sufficient explanations. It is
  read-only and keeps route-generality/wall/method/quality/full-replay claims
  closed.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_contract.py`:
  reads the plateau-stability feature audit and freezes the next readout
  vocabulary before any candidate expansion. It materializes feature-predicate,
  contrast-case, evaluation-rule, gate, summary, config, and report artifacts.
  The contract requires P1-P6 for any future `016`-like finite plateau
  recurrence and declares blockers for point-only single-side evidence, abrupt
  source-target switches, scalar-only explanations, and near-miss leakage. It
  is design-only and keeps route-generality/wall/method/quality/full-replay
  claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_application.py`:
  applies the P1-P6 plateau-stability gate contract to the current 23-pair
  first-pass panel without executing Leiden or expanding candidates. It
  materializes pair, predicate, class, decision, gate, summary, config, and
  report artifacts. Only `016` accepts all predicates; `009`, `012`, `014`,
  and `020` remain near-miss guards; at this pre-gap-fill stage, `001` and
  `007` stay non-strict diagnostics; and missing P2-P6 route/fraction readouts are marked
  not-scoreable rather than inferred. It is read-only and keeps
  route-generality/wall/method/quality/full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy.py`:
  synthesizes the existing screen, fixed route trace, route-negative
  explanation, plateau-feature audit, and P1-P6 gate application into a
  route-state morphology taxonomy. It materializes pair, class, provenance,
  decision, gate, summary, config, and report artifacts. It keeps `016` as the
  only stable finite single-side plateau reference, classifies `009`/`020` as
  abrupt source-target switches, `012`/`014` as fragmented or point
  single-side negatives, `005` as a boundary/endpoint control, and separates
  stale screen readout flags from later route evidence. It is read-only and
  keeps wall/method/quality/full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge.py`:
  reads the route-state morphology taxonomy and audits which evidence can
  bridge to basin-state assignment. It materializes pair, class, requirement,
  decision, gate, summary, config, and report artifacts. The six current
  route-scoreable pairs keep route morphology evidence, but none has accepted
  source and target basin-state assignments; wall evidence remains unknown and
  all route labels stay `unknown`. It is read-only and keeps basin identity,
  wall/pathway-label, method, quality/cost, and full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_basin_state_assignment_surface.py`:
  reads the bridge audit plus existing route-trace, route-negative, 016
  continuity, 014/005 endpoint-object, and 014 wall-evidence artifacts. It
  materializes pair, evidence, class, requirement, decision, gate, summary,
  config, and report artifacts. It accepts only `014` as a local object-level
  basin-state pair and keeps that local-only because current fixed-predicate
  route morphology treats `014` as a negative guard. `016` remains the positive
  stable-plateau route-morphology reference but lacks endpoint-object identity
  and wall evidence. It is read-only and keeps pathway-label, general-wall,
  method, quality/cost, and full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_014_016_surface_reconciliation.py`:
  reconciles the `014` object-wall evidence surface with the `016` positive
  route-morphology surface without executing new routes. It materializes pair,
  axis, schedule, decision, gate, summary, config, and report artifacts. It
  shows `014` is object-wall positive under direct-only/recovery-loop probes
  but remains a fixed-fraction morphology guard, while `016` is fixed-fraction
  stable-plateau positive but lacks endpoint-object identity and wall evidence.
  It keeps pathway-label, general-wall, method, quality/cost, and full-replay
  claims closed and recommends a design-only `016` object-wall transfer
  contract next.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract.py`:
  consumes the 014/016 reconciliation, the 014 pathway-probe contract, the 016
  continuity-block audit, and the basin-state assignment surface. It
  materializes rule, pair, route-plan, boundary-guard, decision, gate, summary,
  config, and report artifacts. It freezes 14 design-only route rows: six
  `016` positive-transfer rows over `bridges_to_left`, `pair_together`, and
  `singleton`, plus eight `005` boundary-control rows. All seven gates pass. It
  does not execute routes and keeps pathway-label, general-wall, method,
  quality/cost, and full-replay claims closed.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace.py`:
  executes exactly the 14 rows from the `016` object-wall transfer contract
  using the local fractional-edge trace kernel. It materializes execution-plan,
  trace, seed-route summary, route-contract summary, route-result,
  route-summary, pair-result, boundary-guard result, gate, summary, config, and
  report artifacts. All 10 execution/readout gates pass. `016` has 24/24
  direct-only target-available seed routes and 24/24 recovery-loop typed
  transient-block seed routes. `005` remains non-positive with zero positive
  leaks. It keeps pathway-label, general-wall, method, quality/cost, and
  full-replay claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace.py`:
  audits the executed `016` object-wall transfer trace without rerunning
  routes. It materializes evidence, route, pair, decision, gate, summary,
  config, and report artifacts. All eight audit gates pass. It keeps
  local-object-wall evidence closed because object identity remains unresolved,
  opens only a read-only object/signature identity-resolution audit over the
  existing trace, and keeps pathway-label, general-wall, method, quality/cost,
  full-replay, and route-expansion claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution.py`:
  audits the existing `016` transfer trace, transfer audit, local-ablation
  seed runs, and prior `016` semantic/pathway summaries to resolve local state
  signatures without rerunning routes. It materializes signature, step, route,
  local-ablation-provenance, evidence, decision, gate, summary, config, and
  report artifacts. All seven identity gates pass. It resolves signature-level
  state identity for target `3c9b8a190753`, transient `aeb59ab537e6`, and
  source-family `5536308f50fc` / `c475d13ca500`, but keeps endpoint-object
  identity, pathway-label, general-wall, method, quality/cost, full-replay,
  and route-expansion claims closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_identity_certificate.py`:
  audits the existing `016` signature-identity output against the first-pass
  symmetric endpoint-object audit without rerunning routes. It materializes
  scope, local-object, relation, evidence, decision, gate, summary, config, and
  report artifacts. All eight certificate gates pass. It confirms that existing
  symmetric endpoint-object membership is absent for `016`, but all four stable
  signatures can be locally certified as signature objects. The target local
  object is certified; the source family remains split; the recurrent transient
  remains a typed non-endpoint blocker; endpoint-object identity, pathway-label,
  general-wall, method, quality/cost, full-replay, and route-expansion claims
  remain closed.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_claim_schema_application.py`:
  applies the surface-qualified basin claim schema to the current `014`/`016`/
  `005` evidence split without rerunning routes. It materializes case,
  evidence, decision, gate, summary, config, and report artifacts. All six
  application gates pass. It confirms that required columns are present and
  valid: `014` is endpoint-object/certified/clean/diagnostic-only, `016` is
  signature-object/split/ladder/blocked, and `005` is
  endpoint-object/collapse/collapse/closed. It standardizes comparison
  vocabulary only and keeps wall, pathway, method, quality/cost, full-replay,
  and route-execution claims closed. It now imports
  `surface_claim_schema_adapter.py` rather than re-declaring the schema contract
  locally.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_object_surface_rule_decision.py`:
  applies `surface_claim_schema_adapter.py` to the validated `014`/`016`/`005`
  surface rows, then reads the `016` object-identity certificate, the `014`/`005`
  primitive wall-evidence audit, and G4.9/G4.9A control summaries. It
  materializes case-surface, rule, evidence, decision, gate, summary, config,
  and report artifacts. All seven decision gates pass. It accepts `016` local
  signature-objects as diagnostic basin-state surface evidence only, retains
  endpoint-object membership as the wall-wording requirement, keeps typed ladder
  wall wording closed pending a separate rule and controls, and opens no
  pathway, wall, method, quality/cost, full-replay, or route-execution claim.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness.py`:
  applies the adapter-backed object-surface rule to the full 23-pair first-pass
  panel using existing mechanism-screen, plateau, morphology, and basin-state
  assignment artifacts. It materializes pair-surface, core-readiness,
  non-scoreable, class, evidence, decision, gate, summary, config, and report
  artifacts. All nine readiness gates pass. It keeps only `016`, `014`, `009`,
  `012`, `020`, and `005` as scoreable core rows; `016` is the single
  diagnostic transition-band reference; 17 rows remain not-scoreable gaps. It
  does not claim panel generality or promote wall/pathway/method/quality/
  full-replay/route execution.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract.py`:
  reads the panel-readiness audit and local validation start rows, then
  materializes the next design-only gap-fill contract. It opens only `001` and
  `007` as diagnostic-not-scoreable local-signature gap-fill candidates, locks
  `016`/`014`/`009`/`012`/`020`/`005` as fixed reference/guard evidence, and
  excludes the other 15 screened gaps. It writes pair-role, candidate,
  route-plan, acceptance-rule, decision, gate, summary, config, and report
  artifacts. All seven design gates pass. If executed, the contract permits
  exactly 54 route rows over allowed starts and the fixed bridge-fraction
  schedule, with no panel-generality, wall, pathway, method, quality/cost,
  full-replay, or route-execution claim.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace.py`:
  executes exactly the 54 fraction-expanded `001`/`007` route rows from the
  gap-fill contract over 8 seeds. It writes trace, seed-route, pair, fraction,
  gate, summary, config, and report artifacts. All six trace gates pass. The
  run materializes 432 local fraction rows and 48 pair/start/seed readouts.
  Neither `001` nor `007` shows diagnostic transition-band recurrence.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace.py`:
  audits the gap-fill trace against the panel-readiness surface and shared
  surface-claim schema. All seven audit gates pass. It updates only `001` and
  `007`, moving them from diagnostic-not-scoreable gaps to scoreable negative
  guards. The scoreable surface is now 8 rows, the remaining screened gaps are
  15, and `016` remains the single diagnostic transition-band reference. It
  opens no panel-generality, wall, pathway, method, quality/cost, full-replay,
  or route-execution claim.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract.py`:
  designs the narrow follow-up schedule-boundary artifact check for `001` and
  `007`. It does not reopen the 15 screened gaps. It reuses the same allowed
  starts and fixes five bridge fractions, `0.5`, `0.375`, `0.25`, `0.125`, and
  `0.0`, yielding 30 route rows. All six contract gates pass.
- `run_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace.py`:
  executes the 30 low-fraction route rows over 8 seeds, producing 240 local
  fraction rows and 48 pair/start/seed readouts. All seven trace gates pass.
  Both `001` and `007` are classified as
  `low_fraction_late_target_collapse_guard`: `001` becomes target-like from
  0.375 downward, `007` from 0.25 downward, and neither pair ever shows a
  single-side band or diagnostic recurrence.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace.py`:
  audits the low-fraction trace against the surface-claim schema. All six audit
  gates pass. It updates only `001` and `007`, reclassifying them from simple
  no-recurrence negatives to late target-collapse guards. The scoreable surface
  remains 8 rows, the 15 screened gaps remain not-scoreable, and no
  panel-generality, wall, pathway, method, quality/cost, full-replay, or
  route-execution claim is opened.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_transition_type_panel_contract.py`:
  reads the existing `016` pathway-shape audit, the `001`/`007` low-fraction
  boundary audit, and the surface-rule panel to freeze the next direction as a
  transition-type panel. All six design gates pass. The contract separates
  `016` as the finite recurrent transition-band reference, `001`/`007` as late
  target-collapse controls, `009`/`012`/`020` as strict analog guards, `014` as
  a cross-surface object-wall guard, and `005` as a boundary collapse guard. It
  recommends a typed-ladder relation-rule contract before any stronger `016`
  relation wording, lists endpoint-object membership as the wall/object
  alternative, and keeps screened-gap expansion blocked.
- `design_leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract.py`:
  reads the transition-type panel, object-surface rule decision, `016`
  object-identity certificate, and `016` signature-identity resolution to
  predeclare typed-ladder relation wording. All six design gates pass. The
  contract accepts only `016` as a diagnostic typed-ladder relation over local
  signature-object states. It uses `001`/`007`, `009`/`012`/`020`, `005`, and
  `014` as false-positive and surface-separation controls, keeps
  endpoint-object wall/pathway wording blocked, and opens no screened-gap,
  route-execution, panel-generality, method, quality/cost, or full-replay
  claim.
- `audit_leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application.py`:
  applies the predeclared typed-ladder relation rule to the current eight-row
  scoreable surface from the low-fraction boundary audit. All six application
  gates pass. It changes only `016` relation wording from blocked to
  diagnostic-only typed-ladder relation wording, while keeping `014` as a
  separate object-wall diagnostic surface, `001`/`007` as target-collapse
  controls, `009`/`012`/`020` as strict analog controls, and `005` as a closed
  boundary/collapse guard. It preserves 15 not-scoreable screened gaps and
  opens no wall, pathway, panel-generality, method, quality/cost, full-replay,
  route-execution, or screened-gap expansion claim.
