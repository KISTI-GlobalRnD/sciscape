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
  according to `docs/dongdaemun_naming_contract.md`.
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

- `docs/dongdaemun_evidence_map.md`
- `docs/dongdaemun_naming_contract.md`
- `docs/dongdaemun_manuscript_plan.md`
- `docs/dongdaemun_reproducibility_appendix.md`
- `docs/hierarchy_postprocess_research_roadmap.md`
- `docs/methodology_final_design.md`
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
- `docs/research/leiden_basin/leiden_basin_cartography_redesign.md`
- `docs/research/leiden_basin/leiden_basin_existing_data_review.md`
- `docs/research/leiden_basin/leiden_basin_methodology_v0_design.md`
- `docs/research/leiden_basin/leiden_basin_data_inventory.md`
- `research/consensus/TODO_SCISCI_ADAPTIVE_REFINEMENT.md`
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
- Use `research/FAILED_DIRECTIONS.md` as the guardrail for dead ends,
  negative controls, and reopen conditions.
- After review, add a manifest before moving any results:
  `path,size,track,label,reason,representative_summary,rerun_command`.
- Only after the manifest is accepted should directories be moved or compressed.
