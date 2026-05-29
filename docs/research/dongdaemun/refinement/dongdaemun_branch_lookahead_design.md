# Dongdaemun-BL: Branch Lookahead Design

## Summary

Dongdaemun-BL is a staged branch-selection policy for Dongdaemun-style
adaptive refinement.  It delays commitment across a small branch beam instead
of committing to the best early Leiden candidate.  The v1 integration target is
Python orchestration over existing Leiden calls; no Rust API change is required.

The core execution pattern is:

```text
candidate branches -> iter5 screening -> iter10 promotion -> convergence polish -> commit
```

The policy is motivated by iteration-budget profile evidence where final
winners can rank poorly at `iter1/2/3` but rise sharply by `iter5/10`.

## Candidate And Stage Semantics

Each branch candidate is a `(seed, randomness)` Leiden run for the same
sample/layer/graph input.  The orchestration layer runs the same candidate grid
at different iteration budgets and joins results by `(seed, randomness)`.

Default v2 candidate policy:

```text
iter5_screen_all_top3
```

Execution:

1. Run every candidate to `iter5`.
2. Rank candidates by `quality desc`, `max_doc_weight_ratio asc`,
   `n_above_max_doc_weight asc`, and `elapsed asc`.
3. Promote the top 3 candidates to `iter10`.
4. Select the best `iter10` candidate with the same ranking rule.
5. Run convergence polish only for the selected top1 candidate.
6. Commit the polished candidate if available; otherwise commit the `iter10`
   candidate.

Fallback policies for validation:

- `iter5_screen_all_top5`: same as default, but promote top5.
- `margin_polish_top2`: promote iter5 top3, then convergence-polish iter10
  top2 if their quality margin is small.
- `mixed_beam_v2`: early `iter1/2` beam with quality top5, pressure-safe top1,
  diversity rescue, and seed-family rescue.

## Python Orchestration Interface

The first implementation should be an orchestration wrapper around the existing
profile/run primitives rather than a Rust-side staged API.

Expected inputs:

- prepared graph summary path
- candidate seeds
- candidate randomness values
- resolution and target max doc weight from the summary
- policy name, defaulting to `iter5_screen_all_top3`

Expected outputs:

- selected `(seed, randomness)`
- selected budget and quality
- max-doc pressure metrics
- promoted and polished candidate IDs
- elapsed proxy or actual elapsed by stage
- policy diagnostics mirroring `branch_policy_simulation.csv`

The orchestration wrapper should persist stage rows in the same row shape used
by `leiden_random_refinement_profile_rows.csv` so the offline analyzer can
evaluate real staged runs without a separate parser.

## Commit And Safety Rules

Commit selection remains quality-first.  Pressure is a tie-breaker, not a hard
penalty, unless a later experiment proves that a pressure gate is needed.

Commit order:

```text
quality desc
max_doc_weight_ratio asc
n_above_max_doc_weight asc
elapsed asc
```

Convergence polish policy:

- default: polish selected top1 only
- margin fallback: polish top2 when absolute iter10 quality gap is `<= 25.0` or
  relative quality gap is `<= 1e-4`

If a promoted or polished run fails, the orchestrator should exclude that failed
branch, log the failure, and select from the remaining successful branches.  If
all promoted branches fail, the policy should fall back to the best successful
iter5 branch and mark the result as degraded.

## Validation Plan

Compare these policy families before enabling Dongdaemun-BL as a default:

- greedy early top1
- quality top5
- full iter10 ensemble
- full convergence ensemble
- `iter5_screen_all_top3`
- `iter5_screen_all_top5`
- `margin_polish_top2`
- `mixed_beam_v2`

Primary metric:

- best convergence quality recovery

Secondary metrics:

- elapsed saving versus full convergence
- elapsed saving versus full iter10
- best10 quality recovery
- max_doc_weight_ratio delta
- n_above_max_doc_weight delta
- late-riser capture rate
- selected candidate stability across samples/layers

Promotion to a production Dongdaemun policy requires cross-sample evidence that
the default staged policy recovers best convergence quality or a near-zero gap
while retaining meaningful elapsed savings.

## Future Rust API Option

A Rust-side API could reduce overhead once the policy is stable.  The future API
would run a candidate beam inside one backend call and expose stage-level traces
for iter5, iter10, and convergence polish.  This is an optimization path, not a
requirement for the v1 Dongdaemun-BL validation.
