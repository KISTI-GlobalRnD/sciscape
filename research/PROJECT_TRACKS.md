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

## Track A: Multi-Layer Consensus Boundary Signal

Current value: 8.0 / 10
Potential: 7 / 10
Decision: keep as a near-term paper track.

### Purpose

Show that multi-layer consensus edges provide a useful boundary signal in
scientific paper graphs, especially around local rank-shift and disagreement
cases.

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
Potential: 8 / 10
Decision: keep as the strongest methods track.

### Purpose

Describe and validate a quality-preserving hierarchy postprocess for science
maps: small-cluster repair, oversize split-repair probes, boundary trim, exact
CPM audit, and contraction-aware reporting.

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
Potential: 9 / 10
Decision: keep as a high-upside research sandbox with strict claim boundaries.

### Purpose

Investigate whether Leiden basin transitions can be induced by structured,
compact, label-coherent perturbations rather than by broad random restart or
threshold tuning.

### Current Mechanism Signal

The strongest current signal is the field34/cc c0 line:

- branch target-growth found candidate-directed support movement;
- post-gate recovery showed broad context was too expensive;
- attachment-margin and aligned-core diagnostics narrowed the mechanism;
- local handle selector evidence suggests label-coherent handle selection can
  recover a compact aligned core in the c0 slice.

This is not yet a general algorithm. It remains diagnostic until non-c0 ready
cases pass selector screening and the operator improves material or
cost-adjusted quality versus controls.

### Primary Anchors

- `research/consensus/TODO_SCISCI_ADAPTIVE_REFINEMENT.md`
- `docs/leiden_multibasin_research_guardrails.md`
- `docs/dongdaemun_basin_transition_operator_design.md`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/`
- `sciscape/clustering/leiden_basin_profile.py`
- `sciscape/clustering/leiden_basin_search.py`
- `research/consensus/scripts/*leiden_basin*`

### Merge Into This Track

- branch target-growth;
- tunneling evidence and debt-area profiles;
- post-gate recovery diagnostics;
- attachment-margin and aligned-core handle selector probes;
- Leiden hysteresis work acceleration as cost/instrumentation evidence.

### Stop Or Archive Within This Track

- repeated c2 p6/p8 replay without source-screen readiness;
- direct-node-only closure shrink as a main mechanism;
- label-internal repair as a main mechanism;
- stage2 local recovery or gate-only expansion as a main operator family;
- raw exact `changed_node_count` claims unless paired with aligned support,
  alignment-error, or endpoint-distance metrics.

### Next Valid Expansion Gate

Before running more local selector replay:

1. Generate or locate a fresh non-c0 post-gate source slice.
2. Run selector source screening in `recovery_contexts` mode.
3. Replay only rows labeled `selector_test_ready` or
   `coherent_label_completion_probe`.
4. Compare against standard seed/iteration controls and material/cost-adjusted
   quality gates.

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
| Attachment-margin and local handle selector | Keep as core R&D | Track C |
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
