# Boundary Accuracy V2 TODO

## Status (2026-04-29)

`Protocol D v2` is no longer just a plan. The coverage-aware sampler, unary
plausibility review path, binary boundary review path, and scorer are
implemented and have completed pilot runs for:

- `field_12`: `cc-only vs citation_consensus`
- `field_15`: `cc-only vs citation_consensus`
- `field_30_textfilt`: `cc-only vs citation_consensus`

Main code entry points:

- `research/consensus/scripts/run_boundary_coverage_review.py`
- `research/consensus/scripts/score_boundary_coverage.py`

Current aggregate file:

- `research/consensus/results/cross_field_round2/boundary_coverage_v2_sample_summary.json`

Current-code sample-only matrix after the Rust Leiden/postprocess optimization
pass:

- `research/consensus/results/cross_field_round2/boundary_coverage_v2_current_sample_matrix_summary.json`
- includes sample banks for:
  - `cc-only vs citation_consensus`
  - `cc-only vs all_consensus`
  - across `field_12`, `field_15`, and `field_30_textfilt`
- live review is still pending for this current-code matrix.

`Protocol D v1` is frozen as a baseline diagnostic. Its limitations are recorded
in `research/consensus/RESULTS_NOTES.md`.

## Current Pilot Summary

Positive net gaps below favor `cc_only`; negative gaps favor
`citation_consensus`.

| Slice | Universe | Population coverage `cc/citation` | Population both-reviewable | Population full net gap | Diagnostic full net gap | Diagnostic `NEITHER` rate |
|---|---:|---:|---:|---:|---:|---:|
| `field_12 ek30` | 28,808 | `0.5667 / 0.5333` | `0.3000` | `+0.3333` | `+0.1935` | `0.5161` |
| `field_15 ek30` | 38,011 | `0.8000 / 0.9333` | `0.7333` | `+0.0455` | `+0.1429` | `0.4000` |
| `field_30_textfilt ek30` | 21,491 | `0.7667 / 0.9000` | `0.6667` | `-0.3000` | `-0.1579` | `0.3684` |

Current interpretation:

- D v2 is doing what it was designed to do: separating coverage failure from
  conditional boundary accuracy.
- The evidence is field/regime dependent. It should not be summarized as
  "`cc-only` is always more accurate."
- `field_30_textfilt` is an important counter-regime where
  `citation_consensus` currently wins the pilot boundary comparison.
- High `NEITHER` rates still matter. Report full summaries, excluding-neither
  summaries, and discriminative `A_ONLY/B_ONLY` summaries separately.

## Done

- [x] Freeze `Protocol D v1` as a baseline diagnostic.
- [x] Record in `RESULTS_NOTES.md` that D v1 is intersection-heavy,
      disagreement-focused, and vulnerable to high `NEITHER` rates.
- [x] Implement coverage-aware target sampling from the full target universe.
- [x] Assign and save `coverage_state`:
  - `A_only_reviewable`
  - `B_only_reviewable`
  - `both_reviewable`
  - `neither_reviewable`
- [x] Split `population` and `diagnostic` sampling modes.
- [x] Add unary plausibility review for one-sided reviewable cases.
- [x] Keep binary boundary review for both-reviewable cases.
- [x] Score:
  - `coverage_rate`
  - `reviewable_rate`
  - `plausible_coverage_rate`
  - `conditional_boundary_accuracy`
  - `overall_boundary_utility`
  - `neither_rate`
- [x] Pilot D v2 on canonical fields for `cc-only vs citation_consensus`.

## Remaining TODO

### 1. Complete The D v2 Comparison Matrix

- [x] Generate current-code sample-only banks for `cc-only vs all_consensus`:
  - `field_12`
  - `field_15`
  - `field_30_textfilt`
- [x] Regenerate current-code sample-only banks for `cc-only vs citation_consensus`
      to avoid mixing older reviewed payloads with current clustering behavior.
- [ ] Live-review the current-code `cc-only vs all_consensus` banks.
- [ ] Live-review or selectively re-review the current-code
      `cc-only vs citation_consensus` banks.
- [ ] Run and score `bc-only vs cc-only` for the same fields.
- [ ] Run and score `emb-only vs cc-only` where the embedding layer is present
      and sufficiently populated.
- [ ] Keep `population` and `diagnostic` outputs separate in every exported
      table and figure.

### 2. Export Manuscript-Ready D v2 Tables

- [ ] Coverage table:
  - universe size
  - `A_only`, `B_only`, `both`, `neither` coverage-state counts
  - population coverage rates by method
- [ ] Conditional boundary table:
  - full net gap
  - excluding-neither net gap
  - discriminative `A_ONLY/B_ONLY` gap
  - `NEITHER` and `UNCLEAR` rates
- [ ] Regime table:
  - fields/slices where `cc-only` wins
  - fields/slices where `citation_consensus` or `all_consensus` wins
  - sparse/dense and asymmetric-reviewability regimes

### 3. Add Review Reliability Checks For D v2

- [ ] Run a small order-balanced repeat set for D v2 cases.
- [ ] Estimate agreement or winner-stability for:
  - unary plausibility decisions
  - binary `A_ONLY/B_ONLY/BOTH/NEITHER/UNCLEAR` decisions
- [ ] Decide whether the manuscript reports D v2 as pilot evidence or as a
      primary validation result.

### 4. Build Protocol E Micro-Gold Local Partitioning

- [ ] For each of `field_12`, `field_15`, and `field_30_textfilt`, choose
      approximately 10 cases from the D v2 case bank.
- [ ] Build a local neighborhood of 10-20 papers around each target.
- [ ] Human-review or tightly controlled LLM-review the full local partition,
      not just a pairwise boundary choice.
- [ ] Score:
  - pairwise precision/recall/F1
  - boundary precision/recall
  - local partition similarity

### 5. Keep The Results Ledger Current

- [ ] Add every new D v2 run to `RESULTS_NOTES.md` with:
  - source JSON path
  - model/judge version
  - field and comparison labels
  - sample sizes
  - headline coverage and boundary metrics
- [ ] Mark stale or superseded boundary review files explicitly instead of
      deleting them.

## Execution Order

1. Complete `cc-only vs all_consensus` D v2 runs.
2. Complete `bc-only vs cc-only` and `emb-only vs cc-only` regime runs.
3. Export D v2 tables and compact figures.
4. Run D v2 reliability checks.
5. Build Protocol E micro-gold cases.
6. Only then decide whether the manuscript can use "boundary accuracy" as a
   primary claim.

## Working Rule

Until D v2 is expanded beyond `cc-only vs citation_consensus` and Protocol E has
at least a pilot:

- use stability and local reranking results as supporting evidence
- describe D v2 as coverage-aware pilot evidence
- avoid blanket claims that any single method has generally more accurate
  boundaries
- prefer phrasing like:
  - `more stable`
  - `more scope-controlled`
  - `better local reranking`
  - `field-dependent boundary behavior under direct evaluation`
