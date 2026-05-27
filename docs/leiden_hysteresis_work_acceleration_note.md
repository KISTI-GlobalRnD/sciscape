# Leiden Hysteresis Work Acceleration Note

## Summary

The current Leiden hysteresis probe should be framed as a refinement-work
accelerator, not as a validated multi-basin escape method.

The observed mechanism is small:

```text
boundary group kick -> different refinement trajectory -> same qf target at lower k_work
```

The strongest current claim is:

```text
For selected boundary groups, a tiny pre-polish perturbation can reduce
refinement work needed to reach the same qf target, while sometimes preserving
non-worse long-polish quality.
```

This is not yet a claim that the final community structure changes at a
macroscopic scale. In the best-inspected field15 `bc73` case, the initial group
was 9 nodes and the final `extra5` versus `perturb5` dominant-overlap
difference was 378 nodes out of 99,439, or about 0.38%.

Two caveats are required for reading the current scorecards:

- The reported saving percentages measure refinement work only, not total
  wall-clock cost including candidate evaluation.
- The current `common qf target = min(extra_final_qf, perturb_final_qf)` is an
  exploratory matched-target metric and may be branch-biased. Paper-quality
  evidence needs branch-independent target definitions.

## Terminology

Use the following terms in this line of work:

- `i`: outer Leiden polish iteration.
- `k_phase`: cumulative count of `after_refinement` phase checkpoints.
- `k_work`: cumulative sum of `n_clusters` at `after_refinement` checkpoints.
- `qf ppm`: qf delta normalized by baseline10 qf, in parts per million.
- `common qf target`: `min(extra_final_qf, perturb_final_qf)` for a matched
  seed/layer pair. This is a current exploratory target, not the final
  acceptance target.
- `work acceleration`: lower `k_work` to reach the common qf target.

Avoid calling this an algorithmic basin escape unless later evidence shows
large, durable structural divergence and a transition-rule change.

## Evidence Snapshot

Primary artifacts:

- `research/consensus/results/adaptive_refinement/leiden_hysteresis_shatter_smoke_20260512/work_acceleration_review/work_acceleration_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_shatter_smoke_20260512/work_acceleration_review/work_acceleration_scorecard.csv`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_20260513/work_acceleration_monitor_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_20260513/long_polish_guard/long_polish_guard_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_20260513/work_acceleration_monitor_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_20260513/long_polish_guard/long_polish_guard_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_budget123_20260513/work_acceleration_monitor_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_budget123_20260513/long_polish_guard/long_polish_guard_report.md`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_smoke/work_acceleration_monitor_report.md`

### V2 Instrumentation Contract

The v2 monitor separates three quantities that must not be mixed:

- `k_work_saving_pct`: refinement-work saving to a named qf target.
- `net_elapsed_saving_pct`: operational elapsed estimate using ordinary extra
  polish time versus candidate selection plus candidate probe time.
- `analysis_trace_overhead_sec`: selected perturb branch replay used only to
  build qf and first-divergence traces.

The v2 scorecard is long by target policy. The paper-facing target policies are
branch-independent:

- `baseline_plus_25ppm`
- `extra_p5_final`

The previous `matched_min` target remains only as a diagnostic continuity row.
Rows where either branch misses a target must be interpreted through
`extra_tau_status` and `perturb_tau_status`; `did_not_reach_target` rows are not
successful acceleration rows for that target.

The v2 monitor also writes:

- `first_divergence_rows.csv`
- `first_divergence_parent_summary.csv`

These are the required inputs for deciding whether group-size-1 wins are
genuine early trajectory divergence or measurement artifacts.

`trajectory_local_merge_summaries.csv` is now optional. The default
`--local-merge-summary-mode compact` writes an empty compatibility CSV plus the
compact parent summary. Use `focused` to keep first-divergence-iteration parent
rows and `full` only for isolated debugging runs.

### Field15 Smoke

The field15 run contains 8 seed/layer rows. Interpreted as work acceleration:

| case | seed | k_work saving | same-k qf advantage | long p20 guard | readout |
| --- | ---: | ---: | ---: | ---: | --- |
| bc | 11 | 71.4% | +27.4 ppm | not run | short-run acceleration candidate |
| bc | 42 | 63.7% | +5.4 ppm | not run | work acceleration, qf near-neutral |
| bc | 73 | 42.4% | +288.8 ppm | +341.5 ppm | durable acceleration candidate |
| bc | 101 | 48.4% | +44.4 ppm | -32.5 ppm | shortcut; long guard fails |
| cc | 11 | 81.0% | +144.7 ppm | not run | short-run acceleration candidate |
| cc | 42 | 21.8% | +16.1 ppm | not run | weak/neutral acceleration |
| cc | 73 | -18.5% | -15.0 ppm | -27.8 ppm | reject |
| cc | 101 | 20.5% | +28.1 ppm | -112.1 ppm | shortcut; long guard fails |

The only field15 row that is both inspected structurally and long-guard
positive is `bc73`.

### Field30 Monitor

The field30 monitor added two graph layers and two seeds:

| case | seed | group size | k_work saving | same-k qf advantage | p20 guard |
| --- | ---: | ---: | ---: | ---: | ---: |
| bc | 11 | 1 | 56.8% | +19.9 ppm | -23.9 ppm |
| bc | 42 | 12 | 60.5% | +35.2 ppm | +32.8 ppm |
| cc | 11 | 1 | 65.8% | +63.6 ppm | +60.6 ppm |
| cc | 42 | 2 | 13.9% | +43.7 ppm | +58.9 ppm |

Field30 therefore shows a work-acceleration signal in all 4 monitored rows, and
3 of 4 rows remain non-worse after p20 long polish. Structural inspection has
not yet been applied to these 3 positive p20 rows.

### Field30 V2 Cost-Aware Monitor

The v2 monitor reran field30 `bc/cc`, seeds `11,42`, with candidate budgets
`1,3,5`. It adds branch-independent target policies and operational elapsed
cost estimates. The most useful paper-facing target in this run is
`extra_p5_final`: can the perturb branch reach the ordinary extra p5 final qf
with less work and less operational elapsed cost?

Best elapsed-positive rows by case/seed under `extra_p5_final`:

| case | seed | best budget | group size | k_work saving | net elapsed saving | same-k qf adv | p20 guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bc | 11 | 1 | 3 | 59.6% | 62.1% | +19.3 ppm | +20.2 ppm |
| bc | 42 | 1 | 12 | 63.7% | 34.4% | +41.8 ppm | +32.8 ppm |
| cc | 11 | 3 | 2 | 6.2% | 37.4% | +5.4 ppm | +115.4 ppm |
| cc | 42 | 1 | 2 | 28.1% | 66.4% | +12.7 ppm | +16.6 ppm |

Important readout:

- All 4 selected `extra_p5_final` rows pass the p20 long-polish guard.
- Budget `1` is the operational winner in 3 of 4 case/seed pairs.
- Larger budgets often preserve positive `k_work_saving_pct` but lose on
  `net_elapsed_saving_pct` because candidate probe cost grows.
- `cc11` budget `1` reaches `baseline_plus_25ppm` quickly but does not reach
  `extra_p5_final`; this is exactly the target-policy corner case the v2
  scorecard is meant to expose.
- First divergence appears at iteration 1, phase `after_aggregation_refined`,
  for all 12 budget rows in this field30 v2 run.
- The compact local-merge summary is still large
  (`trajectory_local_merge_summaries.csv`, about 196 MB for this matrix), so
  future matrix runs should either keep this artifact deliberately or add a
  tighter parent-summary extraction.

### Field30 V2 Budget 1/2/3 Monitor

The follow-up run replaced `1,3,5` with `1,2,3` and used compact local-merge
summaries by default:

- `trajectory_local_merge_summaries.csv`: 226 bytes, empty compatibility CSV.
- `first_divergence_parent_summary.csv`: about 6.5 KB for 24 branch rows.

Best elapsed-positive rows by case/seed under `extra_p5_final`:

| case | seed | best budget | group size | k_work saving | net elapsed saving | same-k qf adv | p20 guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bc | 11 | 1 | 3 | 59.6% | 62.4% | +19.3 ppm | +20.2 ppm |
| bc | 42 | 1 | 12 | 63.7% | 62.2% | +41.8 ppm | +32.8 ppm |
| cc | 11 | 3 | 2 | 6.2% | 45.0% | +5.4 ppm | +115.4 ppm |
| cc | 42 | 1 | 2 | 28.1% | 63.6% | +12.7 ppm | +16.6 ppm |

Budget interpretation:

- `bc11`, `bc42`, and `cc42` are top-1 sufficient under `extra_p5_final`.
- `cc11` is not top-2 sufficient: budgets `1` and `2` reach
  `baseline_plus_25ppm` but still report `did_not_reach_target` for
  `extra_p5_final`.
- `cc11` needs budget `3`, but the final `k_work_saving_pct` is only 6.2%;
  it remains operationally positive because the candidate pool is small.
- For this field30 slice, budget `2` does not create a new winning case.
  It mostly confirms that the useful operational regimes are top-1 and the
  specific `cc11` top-3 exception.

### Field30 Multifidelity Prescreen Readout

This line of work must keep two "p1" questions separate:

- `candidate_budget=1`: evaluate only the top external-grain candidate for the
  perturb branch.
- `p1 prescreen`: rank candidate perturbations after one cheap replay iteration,
  then run p5 only for selected finalists.

Neither meaning makes `p1` an algorithm. It is a heuristic fidelity/budget
setting inside a candidate-screening policy. Any algorithmic claim must come
from the transition rule or from a validated exception-aware evaluation policy,
not from choosing replay iteration `1` as a cheap setting.

The algorithmic object in this work is the perturbation-aware transition before
refinement: which boundary/external group is considered, whether that group is
moved or split, and how the resulting branch is handed back to Leiden
refinement. The `p1`/`p2`/`p3`/`p5` choices only control how cheaply that
candidate branch is evaluated.

The budget-1 monitor result supports the first question for most field30 rows:
`bc11`, `bc42`, and `cc42` are top-1 sufficient under `extra_p5_final`, while
`cc11` needs the budget-3 exception. Budget `2` did not create a new winning
regime in the `1,2,3` matrix.

The multifidelity label runs address the second question. With p1 prescreening:

- p1 top1 misses the full p5 winner in 3 of 4 field30 case/seed rows.
- p1 top2 misses the full p5 winner in only 1 of 4 rows.
- the sole p1-top2 miss is field30 `cc11`, where the full p5 winner is
  candidate 2 with p1 rank 3.

The corrected readout is therefore:

```text
p1 is a reasonable cheap default only as part of an exception-aware policy.
It is not a proof that one-iteration ranking always preserves the p5 winner.
```

The next research question is not "what caused one local gain inside `cc11`?"
It is:

```text
Can p1-visible features identify when budget-1 / p1-top2 is enough, and when
the run should fall back to budget-3 or p1-top3/full-p5 evaluation?
```

Candidate exception features should be computed from the cheap stage whenever
possible:

- p1 top1/top2/top3 score gaps.
- group size and group weight.
- external attachment strength to the proposed target.
- pre-polish delta and expected post-polish delta.
- source/target cut weight.
- early plateau signs in the p1 winner.

Do not use field30 `cc11` candidate 2's local `+0.636` gain as the primary
target. That value is a diagnostic observation inside the p2 replay. It is not
the decision criterion for the work-acceleration policy. Additional tracing of
that window is only useful if it feeds an exception detector or a changed
transition rule; otherwise it is another attribution loop.

## Mathematical Interpretation

The candidate perturbation is a group move under the same CPM objective. For a
source cluster `S`, target cluster `T`, and moved group `G`, the local CPM move
delta used by the probe has the form:

```text
Delta Q_move =
  W(G,T) - W(G,S\G)
  - gamma * w_G * (w_T - w_S + w_G)
```

The split-only alternative is:

```text
Delta Q_split =
  - W(G,S\G)
  + gamma * w_G * (w_S - w_G)
```

The important acceleration metric is not this immediate delta. Some useful
rows even have negative pre-polish delta. The relevant quantity is the
trajectory hitting time:

```text
K_b(i) = cumulative refinement work for branch b up to iteration i
Q_b(K) = qf reached by branch b at refinement work K
tau_b(q*) = min K such that Q_b(K) >= Q0 + q*
```

For a common target `q*`, work acceleration is:

```text
tau_perturb(q*) < tau_extra(q*)
```

Equivalently:

```text
work_saving(q*) = 1 - tau_perturb(q*) / tau_extra(q*)
```

If either branch does not reach `q*`, `tau_b(q*)` is undefined and the row must
be reported as `did_not_reach_target`, not silently treated as a tied or final
budget-limited time. The current scorecards avoid this mostly by choosing a
matched common target, but that convenience is also why the target can be
biased.

This explains why a 1 to 12 node boundary kick can matter. Leiden's local move
and refinement stages are path-dependent. A tiny perturbation can change early
boundary assignments, which changes the parent subproblems seen by refinement
and aggregation. The result can be a steeper early `dQ / dK_work`, even if the
final long-polish quality is only slightly different.

The current evidence supports this conditional explanation:

```text
small external-grain group with plausible target attachment
    -> altered early refinement trajectory
    -> lower k_work to common qf target
```

It does not prove global acceleration for arbitrary groups or arbitrary graphs.

The minimal working hypothesis to test next is:

```text
If a small group G has strong external attachment to T, and ordinary Leiden
does not commit that assignment early, then seeding G -> T can reduce the
hitting time tau(q*) by changing the first local-move/refinement decisions.
```

This is not a sufficient-condition theorem. It is a falsifiable mechanism
hypothesis for the next attribution runs.

## Claim Boundaries

Supported by current artifacts:

- Small boundary kicks reduce `k_work` to a matched qf target in the monitored
  field15/field30 rows.
- In these rows, the effect can be visible even when final qf differences are
  only tens to hundreds of ppm.
- Long-polish guard is necessary because some fast rows are later caught or
  reversed by ordinary extra polish.
- Structural inspection is currently complete for only one positive long-guard
  row, field15 `bc73`.

Not yet supported:

- A general theorem that non-monotone perturbation always accelerates Leiden.
- A claim that this finds substantially different macroscopic basins.
- A net wall-clock speedup claim after candidate evaluation cost.
- A production default policy.
- A new algorithm claim, unless the transition rule is changed and validated
  beyond candidate selection.

## Next Direction

### 1. Instrument Candidate And Total Cost

The v2 monitor now measures candidate selection/probe cost separately from
analysis-only branch replay. Candidate evaluation can be expensive because
`max_group_candidates` polish runs are evaluated sequentially. Therefore,
polish/refinement saving percentages must still not be read as operational
wall-clock speedups unless `net_elapsed_saving_pct` is also positive.

The next full report should compare:

- candidate scan time via `candidate_cluster_selection_elapsed_sec`
- candidate polish time via `candidate_probe_elapsed_sec`
- ordinary extra polish time via `extra_polish_elapsed_sec`
- operational estimate via `net_elapsed_saving_pct`
- `max_candidates = 1, 3, 5` cost/benefit curve

This is required before calling the method operationally faster.

### 2. Add First-Divergence Attribution

The v2 monitor writes first-divergence and local-merge summary artifacts.
Before a larger cross-field matrix, run a small attribution pilot on:

- field15 `bc73`
- field30 `bc42`
- field30 `cc11`
- field30 `cc42`
- one negative or reversed row, such as field30 `bc11` or field15 `bc101`

For each promoted case, capture the first iteration/depth where the perturbed
and extra branches diverge in:

- after-local-move membership hash
- after-refinement membership hash
- moved node count
- refined cluster count
- parent IDs involved in changed local merge decisions

The group-size-1 rows need special attention. A single-node kick producing
large work saving could mean extreme path sensitivity, or it could be a
measurement artifact. First-divergence attribution is the shortest path to
distinguishing those cases.

### 3. Recompute Scorecards With Branch-Independent Targets

The old common target is convenient but potentially branch-biased. The v2
scorecard includes multiple target policies:

- `matched_min`: current `min(extra_final_qf, perturb_final_qf)`, diagnostic
  only.
- `baseline_plus_25ppm`: fixed `Q0 + 25ppm`.
- `extra_p5_final`: ordinary extra-polish p5 final qf.
- `inside_min_10ppm`: `min(extra_final_qf, perturb_final_qf) - 10ppm`, to
  avoid endpoint tautology.

Rows where either branch does not reach the target must carry an explicit
`did_not_reach_target` status.

### 4. Promote The Work Metric To The Primary Acceptance Contract

Every future run should report:

- `qf ppm vs k_work` curve.
- `tau_extra(q*)` and `tau_perturb(q*)` for the common target.
- `k_work_saving_pct`.
- `same_k_work_advantage_ppm`.
- p20 or convergence-polish long guard.
- initial group size/weight and final membership-delta size.

Acceptance for a speed claim:

```text
k_work_saving_pct > 0
same_k_work_advantage_ppm >= -small_epsilon
long_guard_ppm >= -small_epsilon for promoted rows
```

This is a work-efficiency contract. It is not by itself a wall-clock contract
until candidate evaluation cost is included.

### 5. Run A Small Cross-Field Monitor Matrix

Keep this targeted:

```text
fields/layers: field15 bc/cc, field30 bc/cc, then one additional field if available
seeds: 11, 42, 73, 101
budgets: max_candidates 1, 3, 5
polish: p5 plus p20 guard for promoted rows
```

Main table:

```text
case, seed, max_candidates,
group_count, group_weight,
k_work_saving_pct,
same_k_work_advantage_ppm,
long_guard_ppm,
membership_delta_pct,
candidate_elapsed_sec,
total_elapsed_sec
```

Run this matrix only after cost instrumentation and first-divergence pilot
support the mechanism.

### 6. Improve Candidate Selection For Acceleration

The current ranking was inherited from external-grain repair probes. For work
acceleration, rank candidates by predicted trajectory impact instead of only
post-polish qf.

Candidate features to test:

- low absolute pre-polish delta but high external attachment
- small group count and high `group_to_target_weight / group_weight`
- source cluster with high boundary ambiguity
- group target that appears in near-tie local move decisions
- low candidate evaluation cost

Compare `max_candidates = 1, 3, 5` under at least two ranking strategies:

- current external-grain priority
- improved acceleration-oriented priority

This separates "top-1 ranking is already enough" from "larger candidate budgets
only help because ranking is weak."

Do not keep sweeping thresholds without attribution. If a feature matters, it
should explain earlier `tau(q*)`, not just final qf.

### 7. Only Then Consider A Rust API

If cross-field evidence remains positive, expose an opt-in API that returns
work-acceleration diagnostics, not just final accepted membership:

```text
Graph.refinement_acceleration_probe(...)
```

Minimum output:

- selected candidate
- final membership
- p5 quality
- qf curve
- `k_work_saving_pct`
- long-guard optional result
- elapsed candidate cost

Default Leiden behavior should remain unchanged.
