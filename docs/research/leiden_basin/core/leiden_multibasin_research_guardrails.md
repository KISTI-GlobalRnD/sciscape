# Leiden Multi-Basin Research Guardrails

## Purpose

This note defines the current research direction for Leiden/Dongdaemun adaptive
refinement experiments. It is a guardrail document, not a production policy.

Naming and claim boundaries for Dongdaemun-branded artifacts are defined in
`docs/research/dongdaemun/core/dongdaemun_naming_contract.md`.

The core question is whether perturbation portfolios can expose useful local
basins that greedy Leiden or a cheap one-shot ranking would otherwise miss, and
whether those basins can be found with acceptable compute and memory cost.

## Objectives

1. Find a better partition.
   - A candidate is useful only if it reaches a materially better partition than
     the baseline or current policy.
   - A positive `delta_q` is not enough. The gain must be large enough relative
     to graph scale, baseline quality, and operating cost.

2. Find it faster.
   - A candidate-selection policy is useful only if it reaches the material
     quality target with fewer full p5 evaluations, lower wall time, or lower
     memory HWM than a full candidate sweep.
   - A faster policy that misses the material basin is not a success.

## Working Hypothesis

Large, dense, heterogeneous graphs may contain many near-tie local basins.
Greedy Leiden can lock into one basin depending on initialization, node order,
or small perturbations. A perturbation portfolio may improve basin coverage.

This is a hypothesis. It must not be stated as a result until supported by
basin evidence. Density alone is not sufficient evidence: a uniformly dense
graph can also smooth the objective landscape.

## Required Evidence

Experiments should report both quality and cost:

- `best_delta_q`
- `material_delta_q`
- `relative_delta_q_ppm`
- `regret_vs_full_sweep`
- `p5_evaluated`
- `total_elapsed_ms`
- `process_hwm_mb`
- `gain_per_second`
- `gain_per_p5_eval`
- `time_to_material_gain`

Basin-oriented experiments should additionally report:

- `distinct_basin_count`
- `distinct_material_basin_count`
- `basin_entropy`
- `top_basin_dominance`
- `best_basin_hit_rate_at_k`
- `best_quality_regret_at_k`
- `distinct_basin_coverage_at_k`

## Material Gain

Do not classify a run as successful solely because `delta_q > 0`.

Use threshold sweeps to separate real improvements from low-return positive
noise:

- Absolute thresholds: `1e-3`, `1e-2`, `1e-1`, `1.0`.
- Relative thresholds: `1 ppm`, `10 ppm`, `100 ppm` of baseline quality.
- Cost thresholds: `delta_q / elapsed_sec`, `delta_q / p5_evaluated`, and
  memory HWM when available.

Positive but small gains should be labeled as low-ROI improvements unless they
clear a material threshold.

## Basin Evidence

The strongest basin evidence is final membership comparison after p5 polish.
When full memberships are too expensive to store, use compact signatures or
sampled membership signatures, and clearly label them as proxies.

Recommended grouping signals:

- Variation of information, NMI, ARI, or sampled equivalents.
- Boundary-node assignment signatures.
- Perturbed source/target cluster signatures.
- Final quality plateaus and near-tie groups.

Do not infer multi-basin structure from score rank inversion alone. Rank
inversion is a symptom that needs membership or signature follow-up.

## Greedy Failure Conditions

When a cheap score or baseline greedy path misses the full-sweep winner, record
candidate-level diagnostics:

- `p1_delta_q`, localized score, quotient score, and optimistic score.
- `p5_delta_q` and rank inversion size.
- Source cluster, target cluster, group kind, group fraction.
- Active node count and active cluster count.
- Incident edge count, boundary weight, and near-tie count if available.
- Whether the best basin gives material gain or only low-ROI gain.

The goal is to explain when greedy or shallow polish fails, not just to rename a
policy that happens to work on one run.

## Staged Protocol

Use small and medium oracle experiments to design policies before large stress
runs:

1. Run field30-style medium graphs with `candidate_budget=12` or `20`.
2. Attach p5 labels to all candidates and compute basin diversity.
3. Compare score top-k, diversity-aware top-k, random top-k, and risk-gated
   escalation.
4. Promote only policies that keep low regret while reducing p5 evaluations or
   wall time.
5. Use field26 and larger graphs as stress validation after the medium evidence
   narrows the policy family.

For very large graphs, avoid full oracle sweeps unless explicitly justified.
Prefer cheap labels for all candidates, p5 on a shortlist, and p5 on a random
holdout to estimate miss risk.

## Anti-Patterns

- Calling any positive `delta_q` a success.
- Optimizing only recall@1 or recall@2 without cost and regret.
- Claiming dense graphs have more basins without basin evidence.
- Running more threshold sweeps without a mechanism question.
- Promoting a heuristic to default because it worked on one field or one seed.
- Treating candidate-selection changes as a new algorithm unless the transition
  rule or basin-search mechanism has changed and evidence supports it.

## Artifact Contract

New experiments should preserve enough artifacts to audit both objectives:

- Candidate rows with cheap labels and p5 labels.
- Policy comparison rows with selected candidate, p5 count, elapsed time, and
  HWM.
- Dongdaemun role metadata when the artifact is meant to support a Dongdaemun
  claim: `dongdaemun_family`, `dongdaemun_claim_level`, and `effective_output`.
- Basin summary rows.
- Coverage curves by top-k.
- Greedy failure or rank-inversion case rows.
- A short Markdown report stating which claims are supported, unsupported, or
  still hypothetical.
