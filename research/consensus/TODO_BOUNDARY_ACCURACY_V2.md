# Boundary Accuracy V2 TODO

## Current Status

- `Protocol D v1` is now running and/or completed for:
  - `field_15`: `cc-only vs citation_consensus`
  - `field_30`: `cc-only vs citation_consensus`
  - `field_12`: sample-only still running
  - `field_15`, `field_30`: `cc-only vs all_consensus` sample generation started
- Current interpretation of `Protocol D v1`:
  - It measures boundary accuracy more directly than stability metrics.
  - But the current sampler is still biased toward `both-reviewable disagreement` cases.
  - It also produces too many `NEITHER` cases, so it mixes:
    - real head-to-head boundary decisions
    - pathological cases where both methods fail

## Problem To Fix

The current boundary-accuracy protocol is still not fully fair.

Main issues:
- It can over-represent cases where `cc` is already reviewable and competitive.
- It can miss cases where `cc` fails to form a usable neighborhood at all.
- It does not cleanly separate:
  - coverage failure
  - plausible neighborhood formation
  - conditional boundary accuracy
- `NEITHER` cases are currently too common and dominate interpretation.

## Goal

Build `Protocol D v2` so that boundary evaluation answers:

1. Which method creates a plausible local neighborhood more often?
2. Conditional on both methods producing a plausible neighborhood, which one gives the more accurate boundary?
3. In which regimes does `cc` fail, and in which regimes does `consensus` fail?

## Todo

### 1. Finish Current V1 Runs

- [ ] Wait for `field_12` `cc vs citation_consensus` sample-only run to finish.
- [ ] Complete `field_15` `cc vs all_consensus` boundary-accuracy run.
- [ ] Complete `field_30` `cc vs all_consensus` boundary-accuracy run.
- [ ] Export a compact `Protocol D v1` summary table for:
  - `field_12`
  - `field_15`
  - `field_30`
  - comparisons:
    - `cc vs citation_consensus`
    - `cc vs all_consensus`

### 2. Freeze V1 As Baseline

- [ ] Save a short note in `RESULTS_NOTES.md` describing what `Protocol D v1` actually measures.
- [ ] Explicitly record that `Protocol D v1` is:
  - `intersection-heavy`
  - disagreement-focused
  - vulnerable to high `NEITHER` rates
- [ ] Do not use `Protocol D v1` alone for claims about overall boundary accuracy.

### 3. Implement Coverage-Aware Sampling (`Protocol D v2`)

- [ ] Add a new sampler that starts from the full target universe with metadata, not only disagreement cases.
- [ ] For each target, assign a `coverage_state`:
  - `A_only_reviewable`
  - `B_only_reviewable`
  - `both_reviewable`
  - `neither_reviewable`
- [ ] Define `reviewable` using the same minimum metadata + minimum neighbor/group-size constraints as current review scripts.
- [ ] Save `coverage_state` into the case payload.

Suggested file:
- `sciscape/evaluation/sampler.py`
  - add a new collector for boundary-accuracy v2

### 4. Split Population vs Diagnostic Sampling

- [ ] Add `population sample` mode:
  - random sample from the full eligible target universe
  - used for headline rate estimates
- [ ] Add `diagnostic stratified sample` mode:
  - oversample hard or asymmetric regimes
  - used for mechanism analysis
- [ ] Keep these outputs separate in saved JSON and in plots/tables.

Suggested strata for diagnostic sampling:
- [ ] `A_only_reviewable`
- [ ] `B_only_reviewable`
- [ ] `both_reviewable`
- [ ] `neither_reviewable`
- [ ] `cc sparse / consensus dense`
- [ ] `consensus sparse / cc dense`
- [ ] `high disagreement / both dense`

### 5. Add Unary Review Path

- [ ] For `A_only_reviewable` and `B_only_reviewable`, do not auto-award a win.
- [ ] Add unary plausibility review:
  - “Does this group plausibly represent the target’s immediate research neighborhood?”
- [ ] Return:
  - `PLAUSIBLE`
  - `NOT_PLAUSIBLE`
  - `UNCLEAR`

This prevents “method formed a group” from being treated as a free accuracy win.

Suggested additions:
- `sciscape/evaluation/reviewer.py`
- new script, likely:
  - `research/consensus/scripts/run_boundary_coverage_review.py`

### 6. Keep Binary Boundary Review For Both-Reviewable Cases

- [ ] Reuse the current gold-label boundary review for `both_reviewable` cases.
- [ ] Keep decisions:
  - `A_ONLY`
  - `B_ONLY`
  - `BOTH`
  - `NEITHER`
  - `UNCLEAR`
- [ ] Report:
  - full summary
  - summary excluding `NEITHER`
  - summary excluding `NEITHER` and `UNCLEAR`

### 7. Add New Metrics

- [ ] `coverage_rate`
- [ ] `reviewable_rate`
- [ ] `plausible_coverage_rate`
- [ ] `conditional_boundary_accuracy` on `both_reviewable`
- [ ] `overall_boundary_utility`
- [ ] `neither_rate`

Minimum reporting split:
- [ ] full sample
- [ ] `both_reviewable` only
- [ ] discriminative cases only (`A_ONLY`, `B_ONLY`, `BOTH`)

### 8. Add Regime-Aware Comparisons

- [ ] Do not evaluate only `cc vs citation_consensus`.
- [ ] Run, at minimum:
  - `cc-only vs citation_consensus`
  - `cc-only vs all_consensus`
  - `bc-only vs cc-only`
  - `emb-only vs cc-only`
- [ ] Add per-case metadata for:
  - edge counts around target
  - local overlap
  - cluster size asymmetry
  - degree sparsity

Goal:
- identify slices where `cc` is weak but current intersection-style review would miss it.

### 9. Pilot `Protocol D v2` On Canonical Fields

- [ ] Start with:
  - `field_12`
  - `field_15`
  - `field_30`
- [ ] For each field:
  - `population sample`: ~30 cases
  - `diagnostic stratified sample`: ~12 per major coverage stratum
- [ ] First complete `cc vs citation_consensus`
- [ ] Then complete `cc vs all_consensus`

### 10. Add `Protocol E` Micro-Gold Local Partitioning

- [ ] After `Protocol D v2`, build a smaller micro-gold set.
- [ ] For each of `field_12`, `field_15`, `field_30`, choose ~10 cases.
- [ ] Build a small local neighborhood (10–20 papers).
- [ ] Human-review the full local partition, not just a boundary decision.
- [ ] Score:
  - pairwise precision/recall/F1
  - boundary precision/recall
  - local partition similarity

This is the step needed before making strong “more accurate boundary” claims.

## Execution Order

1. Finish current `Protocol D v1` runs.
2. Summarize `Protocol D v1`.
3. Implement `coverage_state` and `population/diagnostic` sampling.
4. Implement unary plausibility review.
5. Re-run `Protocol D v2` for `field_12`, `field_15`, `field_30`.
6. Add `bc-only` and `emb-only` regime comparisons.
7. Build `Protocol E` micro-gold benchmark.

## Working Rule

Until `Protocol D v2` and at least a pilot `Protocol E` exist:

- use `stability` and `local reranking` results as supporting evidence
- avoid claiming `more accurate boundaries`
- prefer phrasing like:
  - `more stable`
  - `more scope-controlled`
  - `better local reranking`
  - `boundary-accuracy still under direct evaluation`
