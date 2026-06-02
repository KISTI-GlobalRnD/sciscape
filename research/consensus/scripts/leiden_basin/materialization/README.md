# Materialization

Target location for cache, membership, prepare, join, and materialization
scripts.

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
  multiplicity and singleton collapse only; it does not promote wall/pathway,
  quality/cost, real-data method-success, or algorithm claims.
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
