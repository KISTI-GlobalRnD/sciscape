# SciScape Research Track Map

Status: reorganization draft
Date: 2026-05-27
Scope: local checkout `/home/kimyoungjin06/Desktop/Workspace/1.4.4.Sciscape`

This document separates the current research workspace into three active tracks.
It is a claim-boundary and project-management document. It does not move files,
rename artifacts, or promote diagnostic results into paper claims.

## Reorganization Principles

- Keep paper-facing claims separate from diagnostic-only experiments.
- Keep `Dongdaemun-post`, `Dongdaemun-refinement`, and diagnostics distinct
  according to `docs/research/dongdaemun/core/dongdaemun_naming_contract.md`.
- Treat adaptive-refinement results as mechanism evidence until they beat the
  two objectives: better partition and acceptable cost.
- Stop adding policy or threshold sweeps unless the sweep answers a mechanism
  question.
- Archive raw high-volume traces only after a retained summary, manifest, and
  rerun command are recorded.
- Check `research/FAILED_DIRECTIONS.md` before running new adaptive-refinement
  experiments, so archived negative controls are not rediscovered as fresh
  ideas.

## Potential-First Research Direction

This section deliberately ignores current evidence maturity and asks what each
idea could become after stronger data, broader validation, and clearer
generalization. It should not be read as a current claim of completed results.

| Rank by upside | Track | Potential | Strongest version of the idea |
| --- | --- | --- | --- |
| 1 | C: adaptive refinement and basin tunneling | 9.5 / 10 | A general account of why graph clustering optimizers get trapped in basins, plus compact interventions that move partitions without broad restart. |
| 2 | A: multi-layer consensus boundary signal | 8.8 / 10 | A theory and empirical test of layer agreement/disagreement as a signal of stable scientific structure, boundaries, transitions, and classification uncertainty. |
| 3 | B: Dongdaemun hierarchy repair/refinement | 8.7 / 10 | A quality-audited framework for constrained, interpretable science-map hierarchy construction, extending from postprocess repair to integrated refinement. |

Upside-first priority differs from evidence-first priority. If optimizing for
maximum novelty, Track C is the first exploration target. If optimizing for
nearer paper defensibility, Track B remains the safer methods track. Track A is
high-value only if it is framed as multi-layer boundary epistemics, not merely
as another aggregation of BC, CC, and DC edges.

## Track A: Multi-Layer Consensus Boundary Signal

Current value: 8.0 / 10
Potential: 8.8 / 10
Decision: keep as a high-value empirical/theory track; avoid low-novelty
aggregation framing.

### Purpose

Show that multi-layer consensus edges provide a useful boundary signal in
scientific paper graphs, especially around local rank-shift and disagreement
cases.

### Potential-First Direction

The low-value framing is:

- BC, CC, and DC can be combined to improve a science map.

The high-value framing is:

- BC, CC, and DC encode different temporal and cognitive relations among
  papers.
- Agreement across layers marks stable scientific structure.
- Disagreement across layers marks boundaries, transitions, contested
  placement, emerging fronts, or classification uncertainty.
- The object of study is not the merged graph itself, but what layer agreement
  and disagreement reveal about the observability of scientific structure.

This makes Track A a study of multi-layer boundary epistemics in science maps,
not just a network fusion paper.

Primary research questions:

- What kinds of scientific relationships are captured by BC-only, CC-only,
  DC-only, and cross-layer consensus edges?
- Are high-disagreement edges enriched near field boundaries, rank shifts,
  emerging areas, or expert-ambiguous cases?
- Does layer agreement/disagreement remain meaningful across fields, years,
  graph scales, and resolution settings?
- Can disagreement be separated into productive boundary signal versus noisy
  graph construction artifact?

Validation target:

- cross-field and cross-time replication;
- layer ablations against single-layer and summed baselines;
- case-level evidence using text, taxonomy, expert labels, or local-review
  annotations;
- uncertainty analysis that distinguishes stable cores, boundary zones, and
  unstable placements.

### Paper Claim State

This is the most paper-ready standalone track. It already has a corrected
local-review interpretation, a manuscript package, and a manuscript-facing
figure bundle.

Defensible scope:

- consensus versus single-layer or summed baselines;
- order-balanced local review evidence;
- taxonomy and uncertainty analysis;
- boundary-signal interpretation, not Dongdaemun or adaptive-refinement claims.

### Primary Anchors

- `research/consensus/README.md`
- `research/consensus/ABSTRACT.md`
- `research/consensus/MANUSCRIPT_OUTLINE.md`
- `research/consensus/RESULTS_NOTES.md`
- `research/consensus/CAPTIONS.md`
- `research/consensus/SUBMISSION_STRATEGY.md`
- `research/consensus/figures/manuscript_joi_v1/`
- `research/consensus/results/case_banks_corrected/`
- `research/consensus/results/taxonomy_corrected/`

### Merge Into This Track

- corrected `gemini_v3` local review outputs;
- `cross_field_round2` summary outputs if they support the same consensus
  boundary-signal paper;
- small legacy pilot outputs only as appendix or audit references.

### Do Not Mix

- Dongdaemun hierarchy repair claims;
- basin-transition or local-handle selector diagnostics;
- Rust fast-path implementation notes.

## Track B: Dongdaemun-Post Hierarchy Repair Method

Current value: 8.2 / 10
Potential: 8.7 / 10
Decision: keep as the strongest evidence-first methods track; expand toward
`Dongdaemun-refinement` only after separate validation.

### Purpose

Describe and validate a quality-preserving hierarchy postprocess for science
maps: small-cluster repair, oversize split-repair probes, boundary trim, exact
CPM audit, and contraction-aware reporting.

### Potential-First Direction

The low-value framing is:

- Leiden output can be cleaned up with postprocessing rules.

The high-value framing is:

- Science-map hierarchy construction is a constrained optimization problem:
  quality, size balance, interpretability, contraction, and auditability can
  conflict.
- A hierarchy method should expose those conflicts rather than hide them behind
  a single partition score.
- `Dongdaemun-post` is the validated repair layer; `Dongdaemun-refinement` is
  the higher-upside future direction where repair signals feed back into the
  optimization process itself.

This makes Track B a method for quality-audited, constraint-aware hierarchy
construction, not just a cleanup heuristic.

Primary research questions:

- When do size and hierarchy constraints damage CPM quality, and how can those
  failures be detected?
- Which repairs preserve original-graph quality while improving interpretability
  and usable cluster size?
- Can repair diagnostics predict where an integrated refinement operator should
  act?
- Can the method generalize beyond one science-map dataset or one resolution
  regime?

Validation target:

- exact original-graph CPM audit for every repaired hierarchy;
- ablations for small-cluster repair, oversize split repair, boundary trim, and
  contraction-aware reporting;
- cross-field, cross-resolution, and size-regime stress tests;
- separate evidence for any future integrated `Dongdaemun-refinement` claim.

### Paper Claim State

This is the most defensible methods contribution. The validated claim is
`Dongdaemun-post`: a post-Leiden, quality-audited hierarchy repair layer.

Defensible scope:

- exact original-graph CPM audit remains the source of truth;
- quality-first oversize repair is the default validated method;
- hard-cap behavior is diagnostic, not the preferred claim;
- failure taxonomy and contraction-aware evidence support honest boundaries.

Not yet defensible:

- integrated-loop `Dongdaemun-refinement` as an empirical result;
- direct promotion of branch-adaptive critical-gamma or basin-transition probes
  into production policy.

### Primary Anchors

- `docs/research/dongdaemun/manuscript/dongdaemun_evidence_map.md`
- `docs/research/dongdaemun/core/dongdaemun_naming_contract.md`
- `docs/research/dongdaemun/manuscript/dongdaemun_manuscript_plan.md`
- `docs/research/dongdaemun/manuscript/dongdaemun_reproducibility_appendix.md`
- `docs/research/hierarchy/hierarchy_postprocess_research_roadmap.md`
- `docs/research/methodology/methodology_final_design.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/`
- `sciscape/clustering/hierarchy_postprocess.py`
- `sciscape/clustering/hierarchical.py`

### Merge Into This Track

- branch-adaptive critical-gamma notes as framing or future work;
- split-repair success/failure diagnostics;
- external-grain predictor evidence as a cheap screen;
- Rust fast path only as implementation support, not as a separate research
  claim.

### Keep Separate

- `Dongdaemun-refinement` design docs must remain future-work or R&D unless
  they receive separate empirical validation.

## Track C: Adaptive Refinement And Basin-Tunneling R&D

Current value: 6.7 / 10
Potential: 9.5 / 10
Decision: keep as the highest-upside research sandbox, but treat the current
cycle as closed under fixed gates. Do not resume route execution or c0-first
operator continuation unless a reopen condition below is explicitly met.

### Purpose

Investigate whether Leiden basin transitions can be induced by structured,
compact, label-coherent perturbations rather than by broad random restart or
threshold tuning.

### Potential-First Direction

The low-value framing is:

- Search over Leiden parameters, thresholds, or candidate policies until a
  better partition is found.

The high-value framing is:

- Graph clustering optimizers can get trapped in distinct partition basins.
- Those basins have membership-level and trajectory-level signatures.
- A useful refinement method should identify compact structural handles that
  move the optimizer toward a better basin with less cost than broad restart.

This makes Track C a possible general algorithmic contribution about basin
existence, basin-definition diagnostics, wall cartography, and eventually
controlled basin transition. It is not yet a validated search algorithm.

### Current State

The current Track C cycle separates two claims that were previously easy to
merge:

- `basin_existence_candidate_evidence`: multiple endpoint identities and
  support-local separations exist under declared thresholds.
- `pathway_or_wall_claim`: a route protocol can connect or cross those basin
  candidates.

The first claim has candidate evidence, not a final basin definition. The
existence audit reports 16 non-field34 cases with 5 strong and 4 moderate
candidate multi-basin evidence cases, plus 87 strong and 75 moderate meaningful
distinct pairs under declared support thresholds.

The second claim is not operational under the current gates. The current
23-pair wall surface has 0 executable route candidates and 0 wall-promotion
candidates. Field34 is closed as reference, hold, or filtered evidence after
its eligibility audit. The closure does not claim that the full 206-pair
calibration universe has no walls.

### Primary Anchors

- `docs/research/leiden_basin/README.md`
- `docs/research/leiden_basin/core/leiden_basin_cartography_redesign.md`
- `docs/research/leiden_basin/evidence/leiden_basin_existing_data_review.md`
- `docs/research/leiden_basin/core/leiden_basin_methodology_v0_design.md`
- `docs/research/leiden_basin/core/leiden_basin_generalization_demo_method_design.md`
- `docs/research/leiden_basin/evidence/leiden_basin_data_inventory.md`
- `research/consensus/TODO_SCISCI_ADAPTIVE_REFINEMENT.md`
- `research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_external_landscape_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_volatile_boundary_cases_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_matched_controls_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_matched_control_delta_analysis_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_fragmentation_boundary_inventory_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_fragmentation_stratified_panel_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_recurrent_boundary_family_registry_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_definition_core_pair_cases_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_basin_distinction_panel_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_basin_vector_panel_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_dataset_contrast_20260530/`
- `research/consensus/results/adaptive_refinement/leiden_basin_existence_assumption_audit_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_cycle_closure_writeup_20260529/`
- `sciscape/clustering/leiden_basin_profile.py`
- `sciscape/clustering/leiden_basin_search.py`
- `research/consensus/scripts/leiden_basin/`

### Active Claim Boundaries

- Endpoint identity and support-local relation are diagnostic objects, not final
  global attraction basins.
- Route-order stability is required but insufficient for wall promotion.
- Boundary-band rows must not be snapped into same or distinct merely because a
  route is stable.
- Field34 must not be used as a clean basin-definition calibration source under
  the current audit.
- No quality or cost comparison should be promoted until basin relation and
  wall evidence are fixed.

### Reopen Conditions

Reopen Track C route or operator execution only if at least one condition is met:

1. The basin-relation boundary band is redefined as a definition problem with a
   predeclared rule and audit plan.
2. A new non-field34 panel is precommitted from the 206-pair calibration
   universe rather than selected from current route success.
3. Stronger wall-evidence requirements are specified before execution, including
   objective debt/recovery, failed direct paths, support incompatibility, or
   polish reversion.
4. Quality/cost evaluation is explicitly deferred until after basin and wall
   gates pass.

### Next Methodology Target

The next Track C target is `basin_methodology_v0`, not operator replay:

- freeze primitive basin objects as endpoint identity, global observed basin,
  support-local basin candidate, basin-pair relation, and wall/pathway relation;
- select a precommitted non-field34 panel from the broader calibration universe
  before route execution;
- require wall/pathway evidence such as failed direct transition, objective
  debt/recovery, polish reversion, or support incompatibility before route
  labels are promoted;
- keep quality and cost out of the basin definition and wall-promotion gates.

The first M1 materialization now passes the panel-shape gates with 7
non-field34 cases across 3 fields and 3 method families. It creates 142
accepted distinct pair candidates for M3 schema review, but does not execute
routes or promote wall claims.

The first M2 enrichment attaches endpoint/signature/cache evidence to the same
panel: 53 endpoint evidence rows, 142 pair evidence rows, 15 endpoints with
cached full-membership metadata, and 8 pairs with both endpoint caches
available. All 142 pairs are ready for M3 schema review, not route execution.

The first M3 wall/pathway schema review keeps the route gate closed: 142 pairs
were reviewed, 5 overlap the existing 23-pair review surface, 2 are only
existing partial-wall protocol references that need trace audit before any new
probe, 3 existing route references remain blocked or insufficient, and 0 pairs
are M4 probe-ready. Support-distance distinctness remains relation evidence,
not wall evidence.

The first M4a trace audit narrows the positive result: the 2 field26
partial-wall references pass W1-W6 as
`crosses_reference_schedule_stable_target_polish`, but remain
`not_promoted_constructed_pathway_reference_only`. Objective debt/recovery and
target endpoint assignment are present across 6 schedule rows, while support
incompatibility is not observed and the post-polish support footprint remains
target-like rather than exact. No route was executed and no wall was promoted.

The first NanoClustering external landscape preparation is now materialized as
endpoint evidence only. It separates the active Paris 78,049-nano endpoint as a
current reference from two clean sidecar seed ensembles: Java has 10 seed
endpoints over 78,154 candidate nodes, and Rust has 10 seed endpoints over
78,119 candidate nodes. The seed-ensemble pairwise surface has 90 ready rows
with moderate candidate-micro movement (Java ARI mean 0.758, Rust ARI mean
0.750), and reference-cluster persistence identifies volatile micro clusters
without using basin quality/cost or wall/pathway claims.

The first NanoClustering volatile boundary case packet expands the most
unstable seed0 reference clusters into split/merge diagnostics: 24 selected
reference clusters, 48 boundary events, 310 split-segment rows, 213 merge
context rows, and 651 unit-sample rows. The dominant pattern is not simple
label reassignment: 30/48 events are `split_and_merge_boundary`, 10/48 are
`severe_split_boundary`, and only 1/48 is `mild_or_label_reassignment_boundary`.
This strengthens the endpoint-boundary interpretation but still does not create
an optimizer-native pathway or wall claim.

The matched-control pass checks whether the volatile split/merge pattern is
only a size-selection artifact. Each volatile reference cluster is matched to a
stable same-branch control with similar document weight and unit count. The
controls produce 48 boundary events, but 39/48 are
`mild_or_label_reassignment_boundary`, 9/48 are `merge_absorption_boundary`,
and 0/48 are `split_and_merge_boundary` or `severe_split_boundary`. This makes
the volatile boundary result materially stronger as endpoint-boundary evidence.
Semantic enrichment remains pending because the local seed-sweep artifacts
contain membership and size summaries, but the referenced node-manifest files
needed for label/representative-paper joins are not present locally.

The matched-control delta analysis decomposes the signal into pair-level axes.
All 24 volatile-control pairs have lower top-split retention on the volatile
side, and all 24 have more split segments than their controls. The median
fragmentation-index gap is 0.627, with Java and Rust showing the same 12/12
directional separation. Severe or split-and-merge patterns occur in 40/48
volatile events and 0/48 matched-control events. Absorption is not sufficient as
a standalone basin-boundary rule because 6 matched-control pairs show
absorption without fragmentation; the current primitive definition should keep
fragmentation and absorption as separate endpoint-boundary axes until route
traces exist.

The fragmentation-boundary inventory applies the top-split axis to the full
seed0 reference-cluster universe. Among 8,444 reference clusters, 825 have at
least one strong fragmentation event (`top_split_share_ref_weight < 0.5`), 478
have recurrent strong fragmentation across at least two comparison seeds, and
146 have persistent strong fragmentation across at least five comparison seeds.
The rule family also identifies 5,749 matched-stable-like references with no
moderate fragmentation event (`top_split_share_ref_weight < 0.8`). All 24
previous volatile cases fall into strong fragmentation categories, while all 24
matched controls fall into the stable-like category. This closes the immediate
concern that the 24-case packet was a narrow selection artifact, but it remains
endpoint cartography rather than basin-wall evidence.

The fragmentation stratified panel expands 56 selected reference clusters into
112 split/merge boundary events across persistent strong, recurrent strong,
single severe, single strong, moderate, and stable-like strata. The strongest
current primitive boundary-candidate rule is recurrent strong fragmentation:
recurrent strong events are 21/24 severe-split or split-and-merge, with 0/24
mild relabeling events. Persistent strong has 0/24 mild relabeling events but
mixes split-and-merge, severe split, merge absorption, and moderate split.
Single strong is too weak as a boundary-family rule by itself: 0/16 events are
severe-split or split-and-merge, while stable-like controls are 16/16
`mild_or_label_reassignment_boundary`. Moderate fragmentation is mostly mild
(13/16 mild). Therefore the next primitive definition should privilege
recurrent strong fragmentation over any-single-event strong fragmentation.

The recurrent boundary-family registry turns that rule into a concrete
definition worklist. It materializes 478 recurrent endpoint-boundary families
and separates them into 179 definition-core rows, 163 definition-stress-test
rows, and 136 edge-control rows. The current core is `repeat_severe_core` plus
`persistent_mixed_core`; `multi_seed_mixed_recurrent` should be used to test
definition fragility, and `pair_only_*` rows should remain controls. This is
still endpoint cartography and does not add route, wall/pathway, quality, or
cost claims.

The first definition-core endpoint-pair expansion materializes the
pair-construction panel's 20 core families into 147 strong endpoint-pair
events, 898 split-segment rows, 733 merge-context rows, and 2,160 unit-sample
rows. There are 0 mild relabeling events. The split between core tiers is
substantive: `repeat_severe_core` contributes 60 split-and-merge and 13
severe-split events, while `persistent_mixed_core` contributes mostly
merge-absorption and moderate-split events. This supplies a concrete
membership-only substrate for testing boundary-family coherence before any
route, wall/pathway, quality, or cost claim is reopened.

The first primitive basin-distinction panel converts that endpoint-pair
substrate into endpoint-handle basin candidates and relation rows. It produces
167 observed endpoint handles, 147 source-target relation rows, 20 family rows,
and 115 accepted primitive distinct relations under the current membership-only
gate. The tier separation remains meaningful: 9 of 10 `repeat_severe_core`
families are fragmentation-dominant multi-endpoint families, while
`persistent_mixed_core` separates into 6 absorption-host-dominant families, 2
mixed fragmentation/absorption families, 1 moderate-fragmentation family, and 1
family with no accepted distinct relation under the current gate. No comparison
endpoint handle is shared across multiple source families in this panel. This
is a basin-distinction artifact only: endpoint handles and source-target
relations are not final global attraction basins, wall/pathway evidence, or
quality/cost claims.

The basin-vector panel corrects the top-1 endpoint limitation by materializing
the full significant split-segment vector and dominant host context for each
definition-core endpoint-pair event. It produces 147 event-vector rows, 898
segment-handle rows, 147 host-context rows, and 20 family-vector rows. All 147
events have multi-handle split vectors under the v1 segment gate. The family
classes sharpen the basin distinction: 9 `repeat_severe_core` families become
`diffuse_multiway_fragmentation_family`, while `persistent_mixed_core` splits
into 4 external-host balanced absorption families, 3 external-host absorption
families, 1 source-host balanced two-way split family, and 2 heterogeneous
families. This rescues cases such as `java_seed0_ref886`, which v0 treated as
not accepted under a top-1 relation gate but v1 identifies as a repeated
source-host balanced two-way split. These are still endpoint-vector basin
candidates only, not final global basins or wall/pathway claims.

The follow-up basin-vector coherence diagnostic tests whether those
endpoint-vector family candidates are internally repeatable before any pathway
or quality question is reopened. It keeps the same 147 event-vector rows and 20
family rows, and classifies 8 families as coherent in both vector shape and host
context, 1 as class-coherent with numeric variation, 3 as split-coherent but
host-variable, 7 as host-coherent but split-mixed, and 1 as heterogeneous or a
rule-edge case. This strengthens the current primitive definition: the
definition-core object should be a support-local endpoint-vector family with an
explicit coherence status. The 12 non-core-coherent families are not failures;
they are subfamily or rule-edge candidates for the next basin-definition pass.

The same gate has now been expanded from the 20-family pilot to all 179
definition-core families in the recurrent registry. The full pass materializes
1026 endpoint-pair events, 5799 split-segment rows, 1026 host-context rows, and
179 family-vector rows. The v1 registry accepts 81 families as
`definition_core_v1_coherent` support-local endpoint-vector families and leaves
98 families in definition-refinement queues: 1 numeric-stress case, 17
split-coherent host-variable subfamilies, 63 host-coherent split-mixed
subfamilies, and 17 heterogeneous/rule-edge reviews. This is the current
strongest basin-definition result. It is still membership-derived endpoint
cartography and does not yet establish final global basins, wall/pathway
traversal, quality, or cost claims.

The v1 refinement-queue decomposition then tests whether the 98 nonaccepted
families are failures or overly coarse family units. It decomposes their 538
events into 215 primary subfamily rows, recovers 134 coherent endpoint-vector
subfamilies across 81 source families, and keeps 58 singleton/tiny events out
of promotion. Combined with the original 81 v1-coherent families, the current
membership-derived coherent coverage is 886 of 1026 definition-core events.
The important methodological correction is that `split_vector_class`, not the
more granular `shape_core_signature`, is the better first split for
split-mixed and heterogeneous queues; `host_context_class` is the better first
split for host-variable queues. Shape-core signatures remain a secondary
stability check. This points to a v2 primitive made from coherent v1 families
plus recovered coherent subfamilies, while the 8 no-recovery families and
unresolved residual events remain definition-audit cases.

The v2 primitive registry now makes that split explicit. It combines the 81
original v1-coherent families with 134 primary-axis recovered coherent
subfamilies, yielding 215 coherent definition-core primitives and 886 of 1026
covered endpoint-pair events. At the source-family level, 162 of 179 families
now have a v2 coherent primitive. The remaining audit surface is not collapsed
into failure: 140 primary-subfamily residual events stay out of promotion, and
17 source families have no primary-axis v2 recovery under the current rule.
Alternative-axis recoveries are recorded but not promoted until the
decomposition rule is revised. This is still a basin-definition registry only,
not wall/pathway traversal, basin-quality, cost, or directed-search evidence.

The v2 audit-surface review adds two caveats. First, the v2 registry is
inclusive: recovered coherent subfamilies are repeated, but many are thin
support rows. Sixty of the 134 recovered subfamilies have exactly 2 events, and
raising the recovered-subfamily support floor to 3, 4, 5, or 6 events would
reduce event coverage from 886/1026 to 766/1026, 658/1026, 550/1026, and
520/1026 respectively. Support depth should therefore be a confidence tier, not
a replacement definition. Second, the primary decomposition axis is broadly
adequate but not final: across the 98 queue families, primary axes recover 398
events, while each-family best axes recover 437 events. The 39-event gap is
concentrated in 12 rule-revision candidates, including 4 strong axis-exception
families where primary recovery is zero but an alternative axis recovers at
least 75% of source events. The residual audit surface remains mostly a
definition problem: 58 singleton/tiny rows stay unpromoted, and the largest
non-tiny residual class is host-coherent split-mixed rows needing a second axis
for shape or host-signature variation.

The v2.1 registry freezes that interpretation without expanding the primitive
set. It keeps the 215 v2 primitives and 886 covered events unchanged, but adds
explicit confidence tiers: 81 accepted v1-family primitives, 11 deep recovered
subfamilies, 60 moderate recovered subfamilies, 60 thin recovered subfamilies,
and 3 retained primitives with a marginal secondary-axis caveat. The axis
exception ledger is kept outside the primitive registry: 4 strong exceptions
and 5 weak exceptions are not promoted, while 3 marginal secondary-axis gains
retain their primary-axis primitive with a caveat. The residual definition
queue is likewise explicit: 58 tiny holdouts, 15 second-axis definition rows,
5 joint-axis rows, and 3 rule-edge rows. This makes v2.1 the current safest
basin-definition surface for downstream selection, but still not wall/pathway,
basin-quality, cost, or directed-search evidence.

The v2.1 detail review narrows the next definition work. Thin recovered
primitives are broad rather than dominated by one bad source family: 60 thin
primitives are spread across 51 source families, with only 7 source families
containing more than one thin primitive. The issue is therefore support-depth
confidence, not a single pathological source. The high-confidence recovered
subset is small but concrete: 11 deep recovered subfamilies cover 62 events and
mostly come from host-coherent split-mixed queues under the primary
split-vector axis. The axis-exception ledger was also recomputed at event level:
the 12 exception families decompose into 35 best-axis subfamilies over 68
events, including 17 coherent best-axis subfamilies that account for the
recorded best-axis recovery. In the 4 strong exceptions, 24 events are recovered
by best-axis coherent subfamilies, but they remain outside v2.1 until an
explicit exception-axis rule is materialized. The next definition work should
therefore prioritize second-axis rule design for host-coherent split-mixed
residuals and joint-axis design for split-coherent host-variable residuals
before any wall/pathway exploration.

The v2.1 axis-rule candidate materialization tests those next definition rules
without promoting them. Across the 15 second-axis residual targets, the best
per-target axes recover 22 of 53 events, but no single second axis resolves the
host-coherent split-mixed queue cleanly: `host_signature` is strongest in total
recovery, while `boundary_pattern` often recovers partial cases and many target
rows still have no coherent recovery. Across the 5 joint-axis targets, the
best per-target axes recover 12 of 18 events, with shape-core based axes
strongest and one host-signature case also recovering most events. The cleanest
rule signal is the strong exception-axis class: all 4 strong exception targets
recover most events through their preidentified best axis, with 24 of 29
events recovered and only 5 tiny holdouts. This suggests that exception-axis
rules are ready for a narrowly scoped definition pass, while the second-axis
queue still needs rule design rather than immediate promotion.

The v2.2 exception-axis registry executes that narrow definition pass. It
inherits all 215 v2.1 primitives and adds only the 8 strong exception-axis
coherent subfamilies, covering 24 additional events across 4 source families.
Primitive coverage rises from 886 to 910 of the 1026 definition-core events,
and the residual queue falls from 140 to 116 events. The 5 remaining
exception-axis rows are singleton/tiny holdouts. Second-axis and joint-axis
candidates remain out of primitive promotion, so v2.2 is still a
membership-derived basin-definition surface, not wall/pathway, basin-quality,
cost, or directed-search evidence.

The v2.2 next-step option review compares continuing residual second/joint-axis
definition work against freezing v2.2 as the operational surface. The remaining
second/joint queues contain 15 targets and 44 events; the current best axes
recover only 12 events, all in support-2 recovered subfamilies, and 0 targets
recover most events. Nine targets have no coherent recovery under the current
candidate axes. The more defensible next step is therefore to freeze v2.2 with
an explicit residual-debt ledger, then prepare downstream instrumentation over
accepted v2.2 primitives while carrying residual rows as caveats/exclusions.
This is not a decision to abandon the residuals; it is a decision not to
promote v2.3 from the current axis candidates.

The v2.2 instrumentation-surface audit turns that freeze decision into a
measurement gate. The 1026-event definition universe remains intact as 910
accepted primitive events plus 116 residual-debt events, with no duplicate
primitive-event rows. The accepted surface contains 166 source families: 125
are complete v2.2 primitive families and 41 carry explicit residual debt; 13
definition-core families remain residual-only. Stress-test recurrent families
and edge-case controls are kept outside promotion (0 accepted stress/control
events), and matched stable controls still show 0 severe-like split or
split-and-merge events. This makes v2.2 ready as an instrumentation substrate,
not as route/wall, basin-quality, cost, or directed-search evidence. The next
execution step should be an accepted-primitive measurement panel with residual
exclusions, not v2.3 promotion or pathway claims.

The v2.2 accepted-primitive measurement panel executes that next step. It
materializes 223 accepted primitive measurement rows and 910 accepted event
measurement rows over 166 source families, while carrying the 116 residual
definition events as source-family caveats. Accounting checks pass: 0
primitive-event mismatches, 0 duplicate accepted event rows, and 0 missing
critical endpoint-vector fields. Support depth is now visible rather than
implicit: 82 primitives are deep-support measurement units, 72 are moderate,
and 69 remain thin. Host-context modes split into 140 external-host absorption
primitives and 83 source-host preserved primitives; boundary-pattern modes
split into 100 merge-absorption, 62 moderate-split, 48 split-and-merge, and 13
severe-split primitives. This is now ready for distribution review of the
accepted primitive claims, but the wall/pathway and quality/cost gates remain
closed.

The v2.2 measurement distribution review adds that conservative read. The
accepted panel should not be narrated as 223 equally strong primitives. The
first descriptive nucleus is 83 stable high-support primitives covering 452
accepted events and 79 source families. The caveat load is still large: 42
thin-clean primitives, 52 residual-debt primitives, and 46 concentration-
caveated primitives. Persistent mixed core is the stronger stable nucleus (63
stable primitives), while repeat severe core has only 20 stable primitives and
should be worded as a harder boundary class. Five source families are
high-residual-debt review priorities (`rust_seed0_ref3868`,
`java_seed0_ref234`, `java_seed0_ref453`, `java_seed0_ref560`, and
`rust_seed0_ref1218`). This review changes the result narrative, not the v2.2
definition.

The v2.2 claim-tier ladder makes the extension from the 83-primitives nucleus
to the full 223 accepted primitives explicit. T1 is the headline nucleus: 83
primitives, 452 events, and 79 source families. T2 adds 23 thin-clean
primitives, reaching 106 primitives and 498 events. T3 adds 19 thin
concentration-caveated primitives, reaching 125 primitives and 536 events. T4
adds 46 non-residual concentration-caveated primitives, reaching 171 primitives
and 756 events. T5 adds 47 standard residual-debt primitives, reaching 218
primitives and 899 events. T6 adds the 5 high-residual-debt priority
primitives, reaching the full accepted universe: 223 primitives, 910 events,
and 166 source families. This is descriptive claim ordering only; it does not
open route, wall/pathway, quality/cost, or directed-search claims.

The assumption-inversion audit tightens the next step. The 83-primitives
headline nucleus remains useful, but it is currently a seed0-anchored
support-local endpoint-vector nucleus observed across comparison seeds 1-9, not
yet a seed-invariant basin taxonomy. The next improvement should therefore
stress the coordinate system itself: first rotate the reference seed across the
10 endpoint seeds, then build symmetric all-seed endpoint objects that do not
depend on one anchor run. This should happen before any v2.3 residual
promotion, wall/pathway probe, basin-quality evaluation, cost evaluation, or
directed-search claim.

The first-principles premise review keeps the guiding premise intact: graph
clustering may have distinct partition basins that compact interventions can
cross. What changes is the current evidence boundary. The present approach has
only reached the first measurement layer: do repeated stochastic endpoints
reveal seed-invariant, support-local structural alternatives? The 83-primitives
nucleus supports a seed0-anchored version of that question, not yet seed
invariance, wall/pathway traversability, compact intervention, quality/cost, or
semantic-boundary claims. Weakness here should diagnose the current approach,
not reject the guiding premise. The next pass/fail gate remains seed-anchor
rotation.

The generalization-demo-method design defines the forward validation ladder.
NanoClustering should now be treated as phenomenon mining, not final proof.
The next research path is: first test generality through seed-anchor rotation
and symmetric all-seed endpoint objects; then build small CPM demo graphs where
plain Leiden + CPM reproduces basin-like alternatives; then explain the
separation by local graph mechanisms such as near-tie cuts, bridge mass,
absorption, balanced split, diffuse fragmentation, or resolution interaction;
then compare any proposed method against Leiden + CPM restart baselines on
basin discovery, wall diagnosis, navigation, and cost. Quality/cost and
semantic claims remain downstream.

The first seed-anchor rotation audit executes the first generality gate. Across
the pure Java/Rust seed ensembles, all 18 non-seed0 anchors expose recurrent
strong fragmentation candidates, with 205 to 238 recurrent candidates per
anchor. This means the phenomenon is not seed0-only at coarse count level.
However, seed0 T1 family recovery under non-seed0 recurrent anchors is low
(minimum 0.121951, median 0.219512), so the current v2.2 object should not be
claimed as a seed-invariant taxonomy. The next generality gate must build
symmetric endpoint objects before wall/pathway or method claims.

The first symmetric endpoint-object audit executes that next gate. It treats
all 84,216 pure Java/Rust endpoint clusters as nodes, retains 513,667 cross-seed
overlap edges, and builds 9,470 symmetric endpoint objects. Multi-seed object
structure is abundant: 8,053 objects have at least five-seed coverage and 7,588
have at least eight-seed coverage. But the strict seed0 T1 anchor-independent
candidate share is 0.341772, below the promotion bar. The correct conclusion is
therefore sharper than the seed-rotation result: NanoClustering has stable
all-seed endpoint objects and anchor-local fragments, but the v2.2 seed0
primitive set is not yet a seed-invariant taxonomy.

The symmetric object decomposition v1 turns that caveat into named structure.
Among 9,470 symmetric objects, 7,498 are stable one-per-seed objects, 555 are
stable multi-cluster objects, 672 are partial objects, and 745 are anchor-local
fragments. Seed0 source-family mapping failures are no longer an undifferentiated
problem: 52 families are anchor-independent candidates, 42 map to multi-cluster
objects, 41 are partial, 27 are anchor-local fragments, and 4 are merged
seed0-family objects. This gives Track C a stable-control registry and 456
mechanism-probe candidates, but still does not open wall/pathway, quality/cost,
or method claims.

The first tiny CPM demo seed sweep executes the baseline-reproduction gate
outside NanoClustering. Four predeclared tiny graph families run with ordinary
Leiden + CPM over 100 seeds: near-tie bridge cliques, absorption triad,
balanced split module, and diffuse fragment star. All four produce recurrent
multi-endpoint behavior, and the near-tie bridge case produces two equal-
quality top endpoints. This gives a controlled baseline surface for the guiding
premise, but it remains baseline-only evidence: no custom method, route,
wall/pathway, quality/cost, or NanoClustering generality claim is opened.

The same tiny CPM demo now freezes the controlled baseline endpoint universe
and restart discovery curve. It records 17 baseline endpoints and 32 discovery
curve rows over budgets 1, 2, 3, 5, 10, 20, 50, and 100 with 200 random
permutations per budget. The easiest controlled case is near-tie bridge
cliques, which reaches full recurrent endpoint coverage by budget 20. The hard
case is diffuse fragment star: at budget 20, mean recurrent recall is 0.854286
and all-recurrent hit rate is 0.275. Any future method claim should beat this
frozen baseline on discovery or navigation before being tested on
NanoClustering.

The first handle-conditioned tiny method probe gives a mixed read. Simple
mechanism-aware initial memberships are directional in easy cases: overall
target-hit rate is 0.675, and near-tie bridge reaches a budget-2 recurrent
recall delta of 0.2575 over random restart. But the hard diffuse fragment star
case is worse than restart at budget 20, with recurrent-recall delta -0.282857.
Therefore the current method evidence is a candidate signal and failure
diagnostic, not an algorithmic contribution. The next controlled method work
should focus on handle ordering, hard-case diffuse coverage, and robustness
before any NanoClustering stress test.

The endpoint replay diagnostic separates that failure from endpoint
instability. Replaying all 17 frozen tiny CPM endpoint signatures as direct
Leiden initial memberships over 10 replay seeds keeps all 15 recurrent
endpoints fixed, including the six recurrent endpoints missed by method v1.
The v1 hard-case failure is therefore a handle-coverage problem on the tiny
surface, not evidence that the missed endpoint signatures are unstable under
Leiden polish.

The replay-informed `coverage_v2` handle probe then appends mechanism-readable
boundary-core and weak-pair tail-split handles. It reaches recurrent endpoint
recall 1.0 at budget 20 for all four tiny families, and changes the hard
diffuse fragment star budget-20 recurrent-recall delta from -0.282857 to
+0.145714. This opens a stronger small-demo candidate-method signal, but still
does not open algorithm, wall/pathway, quality/cost, or NanoClustering
generality claims. The next step should stress ordering robustness and compare
coverage handles under less endpoint-derived candidate generation.

The first ordering-robustness stress executes that next gate for `coverage_v2`.
It reconstructs the restart baseline from seed runs with 1,000 permutations and
tests six order policies: canonical v2, v1-first, coverage-first, handle-type
round robin, adversarial delayed coverage, and 200 random within-family orders.
At budget 20 there are 0 non-adversarial family-policy failures against the
restart mean, and `diffuse_fragment_star` remains positive under all six
policies. The important caveat is early-budget cost: if coverage handles are
delayed, diffuse falls below restart at budgets 5 and 10. This means the
budget-20 signal is not just canonical-order luck, but ordering/cost remains a
real method-design issue. The next required stress is endpoint-derived
dependency: generate handles from graph rules without frozen endpoint
templates.

The deeper ordering read makes the next gate sharper. The replay-diagnosed
misses are coverage-handle timing dependent: under adversarial delayed
coverage, absorption endpoint03 first appears at attempt 20, balanced
endpoints02/03 at attempts 19/20, and diffuse endpoints05/06/07 at attempts
18/19/20. Canonical v2 is better but still has early budget debt for balanced
split at budgets 2, 3, and 5 and diffuse at budget 5. Therefore Stress 2 must
not merely replay coverage-v2 templates. It must build a blind graph-rule
candidate registry before reading frozen endpoint manifests or endpoint
signatures, audit the graph evidence for each generated handle, and then test
whether boundary-core and weak-pair tail-split rules recover the same miss
classes under the frozen restart baseline.

The blind graph-rule handle probe executes Stress 2 and passes all independence
and discovery gates on the tiny surface. It writes an 18-candidate graph-rule
registry before reading endpoint evaluation inputs. The generator uses only
graph nodes, weighted edges, and demo-family naming conventions to create
bridge-contact, boundary-core, weak top-host, weak-pair tail-split, and control
handles. At budget 20, recurrent endpoint recall is 1.0 for all four tiny
families; `diffuse_fragment_star` has delta 0.145714 over the frozen restart
mean; and all six replay-diagnosed hard endpoints are hit earlier than their
adversarial delayed-coverage first-hit attempts. This supports tiny-demo
graph-rule handle plausibility, but not algorithmic generality. The following
ablation gate tests whether the gain is carried by the intended boundary-core
and weak-pair mechanisms rather than by a registry/order effect.

Stress 3 was therefore designed as a mechanism-attribution ablation, not
another policy sweep. It reuses the already-materialized blind-rule candidate
registry and compares named subsets: all blind rules, no controls,
boundary-core only, weak-pair tail-split only, weak top-host only, bridge-only,
controls-only, boundary-core plus weak-pair, and one-handle-type dropouts. The
pass condition is localized causality: removing boundary-core handles should
damage absorption and balanced hard endpoints, while removing weak-pair
tail-split handles should damage diffuse hard endpoints. Controls, bridge
handles, and weak top-host handles must not explain the hard-endpoint gains by
themselves. This keeps the next run inside a mechanism question and avoids
repeating the failed pattern of broad policy or threshold sweeps.

The ablation design has three extra confounds to control in execution.
First, smaller subsets can look better simply because their candidates repeat
more often within a fixed budget, so Stress 3 must report both compacted and
slot-preserving schedules. Second, subset scores must be target-scoped:
boundary-core is judged on absorption/balanced hard endpoints, weak-pair
tail-split on diffuse hard endpoints, and bridge-contact on near-tie endpoints;
global recall is only context. Third, attribution must be exact-endpoint and
seed-aware, recording which candidate ids hit each hard endpoint and whether
those hits survive handle-type dropout across method seeds.

The blind-rule handle-type ablation now executes Stress 3. It reuses the
materialized blind-rule registry and evaluates 17 named subsets under compacted
and slot-preserving schedules, yielding 2,720 scheduled attempts and 20
target-scoped attribution rows. The gate matrix has 12 pass gates and one
closed claim gate. Boundary-core-only hits all three absorption/balanced hard
endpoints before adversarial delayed coverage, weak-pair tail-split-only hits
all three diffuse hard endpoints, controls-only and weak-top-host-only hit none
of the hard endpoint targets, and responsible handle-type dropouts damage their
target classes at 0/1, 0/2, and 0/3 under both schedule modes. This supports
mechanism localization on the tiny demo surface, but still not route/pathway,
wall, quality/cost, NanoClustering generality, or algorithm claims. The next
method-design gate should be a predeclared mechanism-variant panel, not another
threshold or ordering sweep.

Stress 4 is now designed as that mechanism-variant panel. The panel is small
and fixed before execution: 12 variants across near-tie bridge, absorption,
balanced split, and diffuse fragmentation, with mechanism-preserving variants
and mechanism-removed controls. The runner should materialize a graph manifest
with role annotations and fixed gamma values, compute graph-only diagnostics,
build blind-rule candidates, run role/name invariance, and write a phase-lock
hash before any endpoint-evaluation artifacts are read. Only after that should
it run ordinary Leiden + CPM seed sweeps, freeze endpoints, replay endpoints,
and evaluate handle attempts. The central pass condition is not "all variants
improve"; it is sharper: preserved variants
should reproduce recurrent alternatives and recover target hard endpoints with
the responsible rule class, while removed-mechanism controls should not create
false positive target claims. Strong preserved-variant method evidence must
beat the target-scoped restart p75 at the same budget; mean-only gains remain
diagnostic. The required gates are fixed-panel integrity,
baseline reproduction, construction independence, graph-evidence auditability,
mechanism-preserved positive attribution, mechanism-removed specificity,
target-scope scoring, schedule normalization, endpoint-level attribution,
control non-sufficiency, seed robustness, intervention-size caveats, family
coverage, a closed claim gate, role/name invariance, endpoint replay stability,
recurrence-definition lock, matched decoy specificity, mechanism-purity audit,
baseline uncertainty, revision lock, and shadow quality/cost accounting. Even a
full pass remains tiny-demo mechanism variant robustness only.

The P0-P4 implementation slice is now materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_v1_20260531/`.
It stops before Leiden seed sweeps and endpoint evaluation. The runner writes
12 fixed variants, 513 graph-edge rows, 99 role rows, 34 blind candidate rows,
mechanism-feature diagnostics, role/name invariance rows, config, report, and a
phase-lock artifact. Role/name invariance passes across canonical node names,
opaque node ids, and permuted role labels, with phase-lock hash
`2d21ec3b65ac8859d56058fc9d84ccdc7a75fc2fc752171e89fe411e07a5805c`.
This clears only the construction-independence preparation gate. The next
execution gate is P5-P8: baseline seed sweeps, recurrent endpoint freeze,
endpoint replay, handle attempts, target-scoped attribution, restart-p75
comparison, and failure typing without editing the P0-P4 candidate rules.

The P4.5 control-strength audit is now materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_20260601/`.
It reads only phase-locked P0-P4 artifacts and does not run Leiden or endpoint
evaluation. Phase-lock integrity, no-endpoint-input boundary, role/name
invariance, preserved-candidate coverage, control false-positive lock, and
mechanism-removed feature checks all pass. The control-target-like-decoy gate
is caveated: `ab_diffuse_no_core_control` and
`df_weak_module_separate_control` each have zero target-like decoy candidates.
Therefore P5-P8 can be run only as a diagnostic on v1; promotion-oriented
Stress 4 evidence requires a revised phase-locked panel with matched decoy
controls before endpoint outcomes are read.

The improved Stress 4 design should create a new `v1.1` phase-locked panel
rather than editing `v1`. The key correction is to separate responsible
mechanism roles from decoy roles. The absorption control should remove compact
all-module boundary cores but keep scattered `boundary_core_decoy` roles that
generate `blind_small_module_boundary_core_initialization` decoy candidates with
`target_claim_allowed=false`. The diffuse control should keep the weak module
coherent or separate, while adding superficial host-contact `weak_pair_decoy`
roles that generate `blind_weak_pair_tail_split_initialization` decoy
candidates with `target_claim_allowed=false`. P4.5-G7 should become a hard gate
for promotion-oriented Stress 4: every mechanism-removed control needs at least
one target-like decoy for its responsible rule family, plus touched-node and
contact-mass matching or an explicit waiver row. Only after the revised P4.5
passes should P5-P8 be interpreted as Stress 4 evidence rather than a
diagnostic.

The `v1.1` panel is now materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_20260601/`.
It has 12 variants, 542 graph-edge rows, 107 role rows, 42 blind candidates,
role/name invariance pass, and phase-lock hash
`ee2be280a23a8aadb7c8f57e28ab654241f9e2d4d19118edbb84cc80da78177e`.
The matching P4.5 hard-gate audit is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_1_20260601/`.
All seven P4.5 gates pass with `--hard-control-decoy-gate`; readiness is
`ready_for_p5_p8_diagnostic_execution`, `weak_control_count=0`, and
`blocked_control_count=0`. The formerly weak controls now have target-like
decoys but no positive target claims: `ab_diffuse_no_core_control` has two
small-module boundary-core decoys with responsible `boundary_core_concentration=0.0`,
and `df_weak_module_separate_control` has six weak-pair tail-split decoys with
responsible `weak_pair_count=0` and `weak_pair_concentration=0.0`. This clears
only the pre-endpoint control-strength gate; P5-P8 still must be run before
claiming baseline reproduction, target attribution, restart-p75, wall/pathway,
quality/cost, NanoClustering generality, or algorithm-level evidence.

P5-P8 has now been run for `v1.1` at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_1_20260601/`.
It verifies the phase lock and P4.5 hard gate, runs 1,200 ordinary Leiden + CPM
seed runs, freezes 43 endpoints including 33 recurrent endpoints, and executes
240 phase-locked candidate attempts. The conservative result is caveated, not
promotion-ready: `p5_p8_readiness=caveated_endpoint_diagnostic_only`.
Among preserved variants, 26 recurrent endpoints are frozen; positive candidates
hit 17 recurrent endpoints, but only 15 are target-class-compatible hits. Of
those, 13 beat the random-restart p75 first-hit position. The missing coverage
is not random noise: P8-G8 reports `target_class_hits=15/26`, and
`df_one_pair` has zero target-class endpoint hits. Some `df_two_pair` recurrent
endpoints are also missed. The next action is failure typing over the missed
diffuse endpoints and candidate schedule/classes, not another broad parameter
sweep.

The first failure-typing pass is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p8_1_failure_typing_v1_1_20260601/`.
It shows that the earlier 15/26 target-class read was too broad as a failure
denominator: 6 preserved recurrent endpoints are not targeted by the current
positive registry at all (`small_module_separate` endpoints and single weak-node
split endpoints). Among the 20 structurally target-eligible endpoints, 17 are
hit and 15 beat restart p75. `df_one_pair` is not a true target-eligible miss:
its two pair-to-host-core endpoints are hit, while its two single-node split
endpoints are outside the current positive candidate class. The true eligible
miss set is now only three `df_two_pair` endpoints, each requiring two weak-pair
attachments at once: pair0-to-host0 plus pair1-to-host1, pair0-to-host0 plus
pair1-to-host3, and pair0-to-host2 plus pair1-to-host1. This points to a joint
weak-pair candidate/schedule gap, not a generic diffuse failure and not a reason
for broad threshold sweeps.

The downstream P8.2 joint weak-pair probe is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p8_2_joint_weak_pair_probe_v1_1_20260601/`.
It does not modify the phase-locked `v1.1` registry. It takes only the three
P8.1 true eligible misses and jointly applies their already phase-locked
single weak-pair handles. All three missed `df_two_pair` endpoints recover:
3 joint candidates, 30 attempts, `recovered_endpoint_count=3`, and
`all_joint_misses_recovered=true`. Each endpoint is recovered for all 10
method seeds, with first hit on attempt 1 and exact target quality 20.4. This
proves the failure mode is not that the endpoint is unreachable by
mechanism-aware initialization; it is that the current `v1.1` registry/schedule
cannot express simultaneous weak-pair moves before endpoint outcomes are read.
The next promotable design is therefore a new pre-endpoint `v1.2` panel with a
role-derived joint weak-pair candidate generator and matching controls; the
P8.2 post-hoc result itself must remain diagnostic-only.

The pre-endpoint `v1.2` Stress 4 panel is now materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_v1_2_20260601/`.
It keeps the same 12-variant graph universe and adds role-derived joint
weak-pair candidates before reading endpoint outcomes. The registry now has 53
candidates, including 4 positive joint candidates for `df_two_pair` and 7
matched joint decoy candidates for `df_weak_module_separate_control`; role/name
invariance passes and the phase-lock hash is
`293c79949d9d880ca141889604c6359a1bc6008514a0095d8a4b6b155eb8a48c`.
The `v1.2` P4.5 hard-control audit passes at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p4_5_control_audit_v1_2_20260601/`
with `phase_lock_verified=true`, `weak_control_count=0`, and
`blocked_control_count=0`.

The `v1.2` P5-P8 endpoint diagnostic is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_2_20260601/`.
It reruns 1,200 ordinary Leiden + CPM seeds, freezes the same scale of endpoint
universe (43 endpoints, 33 recurrent), and executes 240 phase-locked candidate
attempts. The conservative P5-P8 gate remains caveated because it uses all 26
preserved recurrent endpoints as the denominator and still has non-targeted
endpoint classes. The more precise P8.1 structural typing at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_p8_1_failure_typing_v1_2_20260601/`
is the stronger read: 20 structurally target-eligible preserved endpoints, 20
structural hits, 0 eligible misses, 16 beating restart p75, and 6 explicitly
not-targeted endpoints. The three `v1.1` true eligible misses are recovered by
pre-endpoint joint candidates at attempts 1, 2, and 3. This promotes the joint
weak-pair candidate rule from post-hoc diagnosis to pre-endpoint Stress 4
method evidence, but still does not open route/pathway, wall, quality/cost,
NanoClustering generality, or algorithm-level claims.

The `v1.2` schedule/order robustness diagnostic is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_tiny_cpm_mechanism_variant_panel_schedule_robustness_v1_2_20260601/`.
It keeps the `v1.2` P0-P4 registry frozen and varies schedule order over 105
candidate schedules: canonical order, positive-first, joint-first,
joint-delayed, a joint-suppressed negative control, and 100 random within-variant
permutations. All six schedule gates pass. Nonadversarial schedules have
minimum structural recall 1.0 and minimum joint recall 1.0 over the 20
structurally target-eligible endpoints. The random permutations also have
minimum structural recall 1.0 and minimum joint recall 1.0. The
joint-suppressed negative control drops joint recall to 0.0 and reduces
`df_two_pair` structural recall to 5/8, while keeping all non-joint eligible
endpoints reachable. Control positive attempts remain 0 across all schedules.
This closes the immediate "canonical order luck" concern: the joint weak-pair
rule is needed and is robust to sampled nonadversarial order variation on the
tiny Stress 4 surface. It still does not open real-data, route/pathway, wall,
quality/cost, or algorithm-level claims.

The first NanoClustering joint weak-pair analog screen is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_joint_weak_pair_analog_screen_20260601/`.
It reads the frozen v2.2 accepted-primitive measurement panel only and asks
whether the tiny-demo joint weak-pair mechanism has a real-data analog surface.
The screen finds 99 candidate primitives from 223 accepted primitives, including
17 tier-1 external multi-fragment host-competition analogs and 89 candidate
source families, with 30 matched control-like contrast rows. All five positive
screen gates pass, while the claim boundary gate remains closed by design. This
is enough to justify designing a local NanoClustering panel with frozen
pre-endpoint roles and matched controls; it is not evidence that a real-data
joint initialization works, and it does not open route/pathway, wall,
quality/cost, or algorithm-level claims.

The follow-up local panel design is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_joint_weak_pair_local_panel_design_20260601/`.
It freezes 30 candidate/control cases from the analog screen: 17 core tier-1
joint weak-pair analog cases and 13 lower-tier reserve cases. The refined
version marks 10 `strict_core_v0` cases as the primary future replay
denominator, 7 full-core caveated sensitivity cases, and 13 reserve exploratory
cases. It writes 60 role rows, 386 event-role rows, 60 endpoint-family
signature rows, 17 core-only one-to-one control sensitivity rows, and a
seven-row endpoint-replay contract. Readiness is now
`caveated_ready_for_replay_design_review` because the reusable nearest-control
main panel has only 8 unique control anchors for 30 cases, with max reuse 6.
The next valid step is endpoint-replay implementation against the frozen
`strict_core_v0` case IDs first, using endpoint-family/signature distance rather
than single endpoint-handle hits, not a route/wall, quality/cost, or algorithm
claim.

The endpoint-replay readiness check is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_endpoint_replay_readiness_20260601/`.
It resolves all 465 frozen endpoint-family target handles to local
NanoClustering membership artifacts and freezes a 200-attempt strict-core
symmetric candidate/control plan for the 10 `strict_core_v0` cases. The raw
graph gate is now `ready_for_endpoint_replay_execution`: the original
`/data/openalex_clusters/...` graph paths remain recorded as provenance, while
verified local mirrors under the NanoClustering `_current` output tree provide
the runtime `node_manifest.parquet` and `int_edges.parquet` inputs with matching
branch node counts. The active hierarchy graph is still not substituted because
its row identity does not match the sidecar candidate graphs.

The first bounded endpoint-replay pilot is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_endpoint_replay_pilot_smoke_20260601/`.
It runs the original NanoClustering sequence
`Leiden -> min_nano postprocess -> min_docs postprocess` on the first
strict-core Java case for candidate/control roles at method seed 0, starting
from the frozen source seed0 endpoint partition and scoring terminal partitions
against endpoint-family handles. It executes 2 attempts in 241.3s after a
49.8s branch graph load and scores 12 target handles. The diagnostic result is
negative for the current replay formulation: candidate and control begin from
the same full seed0 partition hash and terminate in the same partition hash
(`role_distinction_status=blocked_terminal_partition_identical_across_roles`).
Therefore full-partition warm-start replay is an execution smoke, not a valid
role-local basin/pathway probe. The next valid design must use a role-local
boundary object, fixed-mask constraint, or explicit route intervention before
any route/wall, quality/cost, real-data method-success, or algorithm claim is
opened.

The role-local boundary plan is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_boundary_plan_20260601/`.
It converts strict-core endpoint-family handles into explicit source/target
node masks and fixed-outside route contracts. All 20 strict-core roles are
`ready_role_local_fixed_mask_contract`, all 10 candidate/control case contrasts
are `distinct_role_local_objects`, and the plan freezes 139 target handles into
5,560 route rows across 4 route arms and 10 method seeds. The pair free mask is
small enough for local route probes: median 74 nodes, maximum 0.00315 of branch
nodes, and minimum fixed-outside share 0.99685.

Two raw fixed-mask route smokes then separate weak from useful interventions.
The first pair-free smoke at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_route_pilot_smoke_20260601/`
executes 4 raw route attempts but gets `route_arms_distinct_count=0`; merely
seeding the target handle while leaving the pair free is too weak. The anchor
smoke at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_route_pilot_anchor_smoke_20260601/`
executes 8 selected raw route rows over the same case, blocks 1 empty-free-mask
arm before Rust, and gets `route_arms_distinct_count=4`. This is the first
positive local manipulability signal on the NanoClustering surface.

The bounded anchor-arm expansion at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_role_local_route_pilot_anchor_expand_seed0_20260601/`
then runs the source-anchor and target-anchor arms across all 20 strict-core
roles with one target handle per role at method seed 0. It selects and records
40 route attempts over 16 target handles, blocks the same 1 empty-free-mask arm
before Rust, and gets `route_arms_distinct_count=20` for 20 role-target pairs.
Among pairs where both arms actually call Rust, the distinction rate is 19/19.
This makes the local signal stronger than the first smoke: source/target anchor
choice can steer the tiny fixed-mask boundary to different raw terminal states.
The interpretation remains narrow. This is raw fixed-mask evidence only, not
endpoint replay. Because the current postprocess wrapper is not fixed-mask
aware, wall/pathway, quality/cost, real-data method success, and algorithm
claims remain closed. The next valid gate is a fixed-mask-aware endpoint or
postprocess readout that can test whether these raw local alternatives survive
the real NanoClustering endpoint sequence.

That readout gate is now materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_fixed_mask_endpoint_readout_pilot_anchor_expand_seed0_20260601/`.
The runner keeps the same fixed-outside source/target boundary contract, then
approximates the NanoClustering min-nano and min-doc postprocess stages with
constrained Leiden rounds: small eligible nodes are free, while outside nodes
and route anchors remain fixed. Across the same 40 selected anchor-arm attempts,
it executes 78 constrained postprocess rounds, records 1 empty-free-mask block,
and preserves the arm distinction through both readout stages:
`raw_route_arms_distinct_count=20`,
`post_nano_route_arms_distinct_count=20`, and
`post_doc_route_arms_distinct_count=20`. No attempt changes its local pair hash
from raw to post-doc (`raw_to_post_doc_pair_hash_changed=0`), and no post-doc
round changes a node. This strengthens the claim that the anchor-defined local
states are readout-stable under the fixed-mask approximation. It still does not
promote endpoint replay in the production wrapper, wall/pathway claims,
quality/cost claims, real-data method success, or algorithm novelty.

A deeper audit of the readout result is materialized in
`nanoclustering_fixed_mask_endpoint_readout_pilot_deep_dive_report.md` in the
same directory. It adds two caveats. First, the 20 role-target pairs collapse
to 16 unique source/target/pair mask objects because two Rust control-side
objects are reused across case labels. Second, the zero-change postprocess
readout is partly expected: the raw route already optimized the same CPM
objective over a larger free set, while the readout reruns constrained Leiden
over a subset of those free nodes. Therefore this is a consistency/readout
stability check, not a non-tautological endpoint-survival proof. The next valid
wall-facing gate is an anchor-release test: take the source-anchor and
target-anchor terminals, release both into the same fixed-outside pair mask,
and ask whether they collapse to one terminal or remain distinct under a common
feasible set.

The anchor-release gate is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_anchor_release_pilot_anchor_expand_seed0_20260601/`.
It is negative for the current anchor-arm basin interpretation. Across the same
20 role-target pairs and 16 unique local source/target/pair mask objects, the
source-anchor and target-anchor terminals are distinct before release
(`anchor_pair_distinct_count=20`) but collapse after both are released into the
same fixed-outside pair mask (`release_pair_distinct_count=0`,
`release_pair_collapsed_count=20`). The collapsed terminal is not just one of
the anchored states: `release_equals_source_anchor_count=0` and
`release_equals_target_anchor_count=0`. It also largely destroys the anchored
target/source identity: median target doc-share after release is 0.02909, and
the median drop from the target-anchor target share is 0.97091. The correct
interpretation is therefore that these anchor arms expose boundary-condition
manipulability, not separate basins or a wall under the common feasible set.
The next search should use common-release non-collapse as the required filter
before any wall/pathway language is reopened.

That negative read is now strengthened by the policy-comparison artifact at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_anchor_release_policy_comparison_20260601/`.
It combines the default anchor expansion with two geometry-stress selections:
`lowest_target_overlap` and `largest_pair_free`. After de-duplication this covers
37 unique role-target pairs across 10 panel cases, 20 roles, and both Java/Rust
families. Every pair is anchor-distinct (`anchor_pair_distinct_count=37`), but
none survives common release (`release_pair_distinct_count=0`,
`release_pair_collapsed_count=37`). This holds even for the largest tested
common-release free set (`max=246`, median `76.0`) and the median target-anchor
target-share drop after release is `0.97125`. The current failure is therefore
not just the first-target choice; it is a broader failure of the present
source/target anchor-arm construction to produce release-stable basin
multiplicity.

The next primitive common-mask multistart gate is also negative on the first
bounded slices. The runner
`run_leiden_basin_nanoclustering_common_mask_multistart_pilot.py` tests the same
fixed-outside pair-mask universe without relying on source/target anchor
terminal comparison: for each pair it runs source-state, target-seeded,
pair-singleton, source/target two-block, and four random pair-block starts. The
comparison artifact at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_common_mask_multistart_comparison_20260601/`
combines the `largest_pair_free` top-6 and `lowest_target_overlap` top-6 slices.
Across 12 unique pairs and 96 start attempts, every pair has exactly one
terminal pair hash (`terminal_multiplicity_pair_count=0`,
`max_unique_terminal_pair_hash_count=1`). The tested common-mask free-node range
is 34-246, with median 150.5. This narrows the current failure beyond
source/target anchor release: these selected local masks behave as single
attractor regions under the present Leiden+CPM settings. It still does not
disprove basin multiplicity globally; it says the next productive search should
change the basin universe or perturbation scale rather than keep comparing
source/target anchor arms inside the same local pair masks.

The first basin-universe redesign artifact is materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_basin_universe_redesign_20260601/`.
It identifies a concrete mismatch in the failed route probes: the local route
universe used a single top1 endpoint handle, while the frozen local-panel
success unit is endpoint-family signature distance. The artifact therefore
materializes 60 signature-level universes that combine dominant-host source
handles with all top1 endpoint handles for each endpoint-family signature, plus
30 candidate/control case-level union candidates and 60 symmetric-object
resolver rows. The signature universe median is still small
(`signature_universe_node_count_median=57`, max 261) and only modestly larger
than the failed single-target pair masks
(`node_expansion_vs_baseline_pair_median=1.10169` on rows with baseline pair
stats). This makes signature-universe routing the correct next executable gate
because it aligns with the success unit, but not a likely final fix by itself.
If signature-level multistart still collapses, the next universe change should
move to candidate/control case unions or resolve symmetric-object masks rather
than add more anchor-arm selection policies.

The first executable signature-universe multistart gate is now materialized at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_signature_universe_multistart_strict_candidate_top6_seed0_20260601/`.
It selects the six largest strict-core candidate signature universes
(`universe_node_count_median=155.5`, max 261) and runs eight starts per
signature: source-state, target-union seeded, universe-singleton,
source/target two-block, and four random universe-block starts. The result is
again negative: 6 signature universes, 48 starts,
`terminal_multiplicity_signature_count=0`, and
`max_unique_terminal_universe_hash_count=1`. Median target-union doc-share in
the terminal is only 0.00247. This closes the single-top1-handle explanation on
the largest strict-core candidate slice: even when the executable mask is
aligned with the endpoint-family signature success unit, the current local
signature universes behave as single-attractor regions. The next gate should
therefore move to candidate/control case-union universes or symmetric-object
mask resolution, not more signature-level or anchor-arm selection sweeps.

The case-universe materialization at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_case_universe_plan_20260601/`
turns the design-only case rows into actual OR-union masks and route-readiness
contracts. It opens all 30 case universes, including 10 strict-core ready
cases, but also exposes a design weakness: candidate/control role-side
universes in the computed top-10 strict-core slice have zero node overlap and
very weak direct graph interaction. Median candidate/control cross-edge weight
share is `5.489829496047739e-05`, and the maximum is only
`0.0010298866522870241`. Therefore case-union is no longer the strongest next
wall-facing route by itself. If run, it should be an interaction-gated
negative-control closure over the best cross-edge cases. The stronger next
design axis is symmetric-object mask resolution, where the universe is defined
from an anchor-independent object rather than from a loose OR of two nearly
disconnected role-side regions.

The symmetric-object universe materialization at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_universe_plan_20260601/`
resolves that axis into actual fixed-outside role-level object masks. All 60
role universes are route-ready, 36 are anchor-independent, 18 are
anchor-independent P1 rows, and 5 are strict-core anchor-independent P1 rows.
The median object mask has 26 nodes, the maximum has 164, and the median object
size is 1.2105 times the seed0 component while compressing the all-seed
component-sum upper bound to 0.2117. The 30 candidate/control case relations
remain fully disjoint at the symmetric-object level, so they are diagnostic
context rather than the route universe. The next gate is role-level
symmetric-object multistart over the strict-core anchor-independent P1 slice,
widening to P2 only if that slice is too small or collapses.

The first symmetric-object multistart gate is now executed at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_strict_anchor_independent_p1_seed0_20260601/`.
It runs 40 starts over 5 strict-core anchor-independent P1 role universes
(4 unique symmetric objects) using seed0 source-state, seed0-object seeded,
object-singleton, component-pattern, and random object-block starts. Terminal
multiplicity remains zero (`max_unique_terminal_object_hash_count=1`), but the
more important failure mode is singleton collapse: all 5 object roles end with
terminal object cluster count equal to object node count, median
component-reference ARI 0.0, and median object best-cluster doc share
`0.006055013194748`. The P2 widening check at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_strict_anchor_independent_p2_unique_seed0_20260601/`
deduplicates objects and repeats the result over 2 unique P2 objects and 16
starts: zero terminal multiplicity, all 2 roles singleton-collapsed, median
terminal object cluster count 58.5 against median object size 58.5, and median
component-reference ARI 0.0. The current object-only mask is therefore not a
working basin universe. The next design should not keep widening P1/P2 object
lists; it should redefine the movable universe as a support-neighborhood or
attachment-aware object that gives the optimizer a cohesive local subproblem.

The support-neighborhood extension closes the simplest version of that
explanation. The P1 unique-object support run at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_support_top100_p1_unique_seed0_20260602/`
tests 24 starts over 4 unique strict-core P1 objects with the top 100
boundary-weight support nodes added to each free universe. It still has zero
terminal object or universe multiplicity, and every object/universe
singleton-collapses (`universe_terminal_cluster_count_median` equals the median
universe size 173.5). A representative top-1000 support smoke at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_support_top1000_smoke_one_20260602/`
also collapses a 1,076-node free universe to 1,076 terminal clusters. Therefore
the failure is not just that the object mask was too narrow.

The mechanism audit at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_merge_viability_p1_unique_top0_100_1000_20260602/`
checks CPM merge viability directly over the 4 P1 unique objects and support
top-k values 0, 100, and 1000. Across all 12 audited universes, there are zero
positive internal free-free merge candidates and zero positive external
free-to-fixed attachment nodes. The best internal merge delta is still negative
(`-43749.985507246376`), and the best external attach delta is also negative
(`-528324.9848484849`). A one-object exact-quality smoke at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_merge_viability_quality_smoke_top0_1000_20260602/`
confirms the same direction: object-only singleton is tied for best, and the
top-1000 universe singleton is the best quality variant. The next Track C gate
should therefore stop expanding support top-k and instead redesign the universe
around objective-positive CPM mechanisms, such as lower-resolution/local-weight
normalization tests or predeclared weak-pair/bridge mechanisms, before running
another terminal-multiplicity pilot.

The objective-mechanism audit at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_objective_mechanisms_p1_unique_top0_100_1000_20260602/`
turns that diagnosis into a concrete mechanism gate. Under the current
doc-weighted CPM objective, all 12 P1 unique-object universes remain closed:
zero positive internal merges and zero positive external attachments. The
largest internal critical gamma is only `4.763001162781675e-05`, so current
`gamma=0.7` is at least 14,696 times too high for even the best internal pair;
the largest external critical gamma is `2.257078917502457e-05`, at least 31,013
times below the current gamma. This means the failure is an objective-scale
mismatch, not just a missing support node problem. Among diagnostic transforms,
`sqrt_doc_weight` also remains closed, `log1p_doc_weight` opens internal
free-free merges in all 12 universes but not external attachments, and
`unit_weight`, `local_median_normalized_doc_weight`, and
`local_p90_normalized_doc_weight` open both internal and external objective
candidates in all 12 universes. These are mechanism candidates only, not method
success claims.

A bounded critical-gamma terminal check confirms that opening the objective can
change the optimizer dynamics. The representative support-top1000 smoke at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_support_top1000_gamma1e5_smoke_one_20260602/`
runs the same selected object at `gamma=1e-5`, below the observed critical-gamma
band. It no longer singleton-collapses: 4 deterministic starts produce 3
object terminal hashes and 4 universe terminal hashes, with median object
terminal cluster count 47.0 against object size 76. The bounded P1 unique
support-top100 run at
`research/consensus/results/adaptive_refinement/leiden_basin_nanoclustering_symmetric_object_multistart_support_top100_gamma1e5_p1_unique_20260602/`
extends that signal across 4 unique P1 objects and 16 deterministic starts:
all 4 objects show terminal multiplicity, all 4 show universe-level
multiplicity, and none singleton-collapses. This reopens Track C as an
objective-mechanism redesign problem. The next valid gate is not quality/cost or
wall promotion; it is a predeclared critical-gamma or weight-normalized
mechanism pilot with controls showing that multiplicity is structural rather
than a trivial low-resolution merge artifact.

The first prior-versus-current dataset contrast shows that the current
NanoClustering full/candidate graphs are dense at the nano level: about
78k-80k nodes, 105M-107M edges, average degree around 2.7k, and undirected
density around 0.033-0.035 under the available summaries. The same current
data family also contains a sparse top20-union projection with average degree
29.3 and density 0.000376. Prior Track C edge rows are not locally available,
so direct old-versus-new edge-density recomputation remains blocked. The
current read is therefore not "density explains basin fragmentation"; it is
"density is a plausible contributor, but not a sufficient explanation." The
next causal check should stay within the current data family by comparing
recurrent/volatile families against stable-like references on local weighted
degree, cut ratio, top-neighbor concentration, and full-graph versus top20
sparse behavior.

The scale-emergence question is deferred as an explicit TODO: after the basin
distinction pass, test which hierarchy scale first exposes this recurrent
fragmentation behavior, especially under the hypothesis that paper-level
fat-tailed citation edges are compressed into intermediate-cluster
inter-cluster edge mass.

### Merge Into This Track

- basin-existence and endpoint-identity diagnostics;
- wall-protocol and route-gate instrumentation;
- branch target-growth, tunneling, debt-area, attachment-margin, and
  aligned-core diagnostics as evidence sources;
- Leiden hysteresis work acceleration as cost/instrumentation evidence.

### Stop Or Archive Within This Track

- repeated c2 p6/p8 replay without a predeclared relation or wall gate;
- direct-node-only closure shrink as a main mechanism;
- label-internal repair as a main mechanism;
- stage2 local recovery or gate-only expansion as a main operator family;
- raw exact `changed_node_count` claims unless paired with aligned support,
  alignment-error, or endpoint-distance metrics.

## Deferred Track: Hybrid CPM-Critical Dendrogram And Optimal Cut

Current value: 5.8 / 10
Potential: 7 / 10
Decision: keep, but do not spend active adaptive-refinement time here now.

### Purpose

Test whether a tree-plus-cut framework recovers merge-gap information that
Leiden plus merge discards while preserving size constraints.

### Anchors

- `research/dendrogram/README.md`
- `research/dendrogram/scripts/`
- `research/dendrogram/results/pilot_field15/`

### Decision

Keep the small pilot results and README. Treat this as a separate future track,
not as part of Dongdaemun-post or basin-tunneling work unless a new cross-field
validation plan is approved.

## Deprecation And Merge Decisions

| Family | Decision | Destination |
| --- | --- | --- |
| Multi-layer consensus local review | Keep | Track A |
| Corrected taxonomy and uncertainty | Keep | Track A |
| Frozen Dongdaemun evidence bundle | Keep | Track B |
| Hierarchy postprocess validation | Keep summary, archive bulky raw runs | Track B |
| Branch-adaptive critical-gamma notes | Merge as framing/future work | Track B |
| Rust Dongdaemun fast path | Merge as implementation support | Track B |
| Branch target-growth diagnostics | Merge | Track C |
| Tunneling and debt-area diagnostics | Merge | Track C |
| Attachment-margin and local handle selector | Keep as diagnostic evidence for redesign | Track C |
| Leiden hysteresis work acceleration | Merge as cost evidence | Track C |
| Direct closure shrink and label-internal repair | Archive as negative controls | Track C archive |
| Repeated c2 selector replays | Stop unless screen-ready | Track C archive |
| Hybrid CPM-critical dendrogram | Defer | Deferred track |

## Immediate Documentation Tasks

- Keep this file as the top-level project map.
- Use `research/DATA_RETENTION_PLAN.md` for keep/archive decisions.
- Generate a read-only result manifest with
  `python3 scripts/research_retention_manifest.py` before any physical cleanup.
- Use `research/FAILED_DIRECTIONS.md` as the guardrail for dead ends,
  negative controls, and reopen conditions.
- After review, add a manifest before moving any results:
  `path,size,track,label,reason,representative_summary,rerun_command`.
- Only after the manifest is accepted should directories be moved or compressed.
