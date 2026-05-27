# Dongdaemun Basin Transition Operator Design

Status: diagnostic design for a closure-aware experimental basin-transition
operator.

This document is not a production policy and does not change the validated
`Dongdaemun-post` claim. It describes the next diagnostic step after the first
controlled basin-transition pilot failed to beat seed/iteration variation.

## Claim Boundary

Use this work only as `Dongdaemun diagnostics` unless a later implementation
passes exact CPM audit and beats seed controls on material gain per cost.

The current evidence supports a mechanism question, not an algorithm claim:

- endpoint-near partitions can have very different changed-support footprints;
- a compact Dongdaemun candidate support can sit inside a broader vanilla
  support footprint;
- hard freezing outside the candidate support, or transplanting candidate
  support labels, did not improve over a simple extra vanilla run.

Therefore the next operator must target the boundary explicitly. The updated
goal is stricter than the first design: decide which vanilla-extra support
nodes can be removed only after accounting for the label context they imply.
A small `S_V - S_C` edit may actually be a much larger split/merge problem.

## Current Goal

Find non-greedy basin-transition pathway prefixes that a one-step Leiden move
or naive perturbation would not select, then test whether those prefixes can be
used as a structured Dongdaemun perturbation.

The goal is not to avoid a quality barrier. Different basins should be expected
to have a wall between them. The useful question is whether the wall can be
crossed by a short, interpretable, low-barrier prefix whose quality debt is
recoverable after bounded local polish.

## Reusable Module Boundary

The reusable profiling implementation lives in:

`sciscape/clustering/leiden_basin_profile.py`

The reusable transition-search implementation lives in:

`sciscape/clustering/leiden_basin_search.py`

Research scripts under `research/consensus/scripts/` should remain thin
artifact runners. They may load graph/candidate rows and write reports, but
shared mechanics should stay in the module:

- support and endpoint distance helpers;
- `label_intersection_block` unit construction;
- fresh-label forced edits;
- ordered flip beam expansion;
- barrier-aware prefix annotation and Pareto-style selection.
- transition search states/actions;
- closure/context expansion candidates;
- polish-aware state classification and Pareto row selection.

This boundary matters for the later operator step. A prefix that survives
polish-aware evaluation should be explored by reusing the module API, not by
copying another one-off script. A transition search row is still diagnostic
unless it later beats seed controls with material quality/cost value.

## Pathway Reframing

The ordered flip v0/v1 artifacts should not be read as a failed search for
monotone QF improvement. They are evidence that single-step greedy objectives
are looking at the wrong object:

- `q_first` finds cheap or positive edits, but those edits mostly remain
  vanilla-near;
- `progress_first` moves toward the compact candidate side, but often buys that
  progress through a large raw QF debt;
- the missing object is a path-level prefix that may look unattractive at step
  one, but becomes useful after a small number of coordinated unit flips and
  bounded polish.

Raw QF debt is therefore a measured wall height, not an automatic rejection.
The rejection condition is different: the prefix is weak if polish returns it
to the original basin, if support progress disappears, or if the material gain
per cost is dominated by seed or multi-start controls.

For exploratory pathway discovery, use a reachability-first search mode before
cost-aware acceptance. In that mode, QF is accounting, not a pruning gate. A row
may be kept because it escapes the source basin or covers more target support
even when its immediate QF is poor. Only after such routes are visible should
the operator question switch back to material quality, mutable-node cost, wall
time, memory, and seed-control comparisons.

## Greedy Failure Model

Classify candidate prefixes by the greedy rule they violate:

`q_greedy_miss`

- The prefix starts with a negative or low-QF unit that `q_first` would reject.
- It is only interesting if later units or polish recover enough QF while
  preserving candidate-side support progress.

`progress_greedy_miss`

- The prefix does not maximize immediate support progress, but reaches a
  comparable support shift with a lower peak barrier or smaller closure cost.

`closure_compound_miss`

- No individual unit is good, but a label/closure compound move is coherent.
- This is the main path by which a perturbation could become more than a random
  seed change.

`polish_recovery_miss`

- The raw prefix is expensive, but bounded polish recovers the QF debt without
  fully collapsing back to vanilla support.

These labels are diagnostic until seed controls are beaten. They are meant to
explain where a perturbation should shake the partition, not to claim a new
algorithmic default.

## Barrier-Aware Pathway Objective

For each candidate pathway prefix, track at least:

- raw `delta_q_vs_start`;
- `peak_raw_barrier`;
- path-level QF debt area and wall duration, not only the maximum wall height;
- `support_distance_to_candidate` and `support_distance_to_vanilla`;
- `support_progress_per_barrier`;
- support and target progress per QF debt area;
- post-wall QF recovery slope;
- flipped node count;
- closure/context cost if available;
- polish-recovered `delta_q`;
- polish-retained support progress.

The search objective should be Pareto-style rather than one scalar winner:

- minimize peak barrier for a fixed support-progress target;
- maximize retained support progress for a fixed barrier budget;
- prefer compact prefixes when quality/support tradeoffs are tied;
- keep seed and recreated-vanilla controls in the report so low-ROI wins are
  not promoted.

This makes the future perturbation hypothesis concrete: use non-greedy pathway
prefixes as deliberate basin-wall crossing seeds, then let bounded polish decide
whether the move opens a real basin transition or collapses back to the source
basin.

## Tunneling Diagnostic Definition

Use `tunneling` as a diagnostic term, not yet as an algorithm claim.

A recoverable tunnel is a structured non-monotone basin-transition path:

- it is candidate-directed in support space;
- it pays nonzero temporary QF debt;
- its debt is short when measured by path-level debt area, not only by peak
  wall height;
- it recovers terminal QF after bounded polish or a small number of coordinated
  target-growth steps.

This separates three route families:

- `recoverable_tunnel`: wall crossing plus candidate-directed support plus
  terminal QF recovery;
- `unrecovered_detour`: lower or different wall crossing with candidate-directed
  support but terminal quality loss;
- `partial_progress_probe`: below-gate support progress that may still be an
  entrance candidate but is not a tunnel.

The operator implication is specific: do not minimize wall height alone. Prefer
short, interpretable, recoverable routes. A higher wall can be a better
operator target than a lower detour if the higher wall has lower debt area and
an immediate recovery step.

## Tunneling Operator V0 Sketch

The first operator should be a two-queue diagnostic, not a single greedy rule.

Queue A: `efficient_tunnel_seed`

- Select recoverable tunnels with high recovered shortcut score.
- Replay the entrance prefix even if the raw step is QF-negative.
- Run bounded polish on the mutable set and require QF recovery before
  accepting tail growth.
- Current best example: `c0/p4 fixed_cap`, wall and debt-area-step `0.246277`,
  final `delta_q_vs_start=+0.546858`, support-distance-to-vanilla `0.069588`,
  target progress `0.021129`.

Queue B: `wide_tunnel_seed`

- Select recoverable tunnels that buy more support movement even with higher
  debt area.
- Use these to test whether a broader target move can beat seed controls on
  material and cost-adjusted value.
- Current best example remains the `c0/p9` family, with support-distance up to
  `0.108247`, target progress `0.037785`, and debt-area-step `1.585400`.

Queue C: `post_gate_recovery_target`

- Keep unrecovered detours that cross the support gate but remain
  quality-negative.
- Do not accept them as tunnels. Instead, use them to search for an explicit
  recovery move after gate crossing: bounded label repair, boundary-shell
  release, or candidate-closure context.
- Current examples are `p6/p8/p10` side-route rows.

Acceptance remains conservative:

- exact CPM audit must pass;
- terminal QF must be material, not merely positive;
- support movement must remain candidate-directed;
- mutable-node cost, wall time, p5 evaluations, and memory HWM must be reported;
- same-case seed/iteration controls must remain in the comparison.

## Input Evidence

The first design target is the field34/cc_cosine pilot slice:

- transition landscape artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_landscape_field34_cc/`
- operator pilot artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_operator_pilot_field34_cc/`
- boundary analyzer artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_boundary_analysis_field34_cc/`
- relabel-pathway artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_minimal_pathway_field34_cc/`
- closure-context artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_context_field34_cc/`
- closure-frontier artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_frontier_field34_cc/`
- closure-operator pilot artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_operator_pilot_field34_cc/`
- closure-context release pilot artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_context_release_pilot_field34_cc/`
- label-internal repair pilot artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_label_internal_repair_pilot_field34_cc/`
- ordered flip basin profile v0 artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_ordered_flip_frontier_field34_cc_v0/`
- ordered flip basin profile v1 batch artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_ordered_flip_frontier_field34_cc_v1_cases/`
- barrier-aware pathway prefix artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_barrier_aware_prefix_field34_cc_v1/`
- polish-aware pathway prefix artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_polish_aware_prefix_field34_cc_v1/`
- basin-transition search v0 artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_v0/`
- reachability-first basin-transition search artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_reachability_v0/`
- pathway QF wall statistics artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_pathway_wall_stats_field34_cc_v0/`
- branch target-growth search artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_branch_target_growth_field34_cc_c0_v0/`
- branch candidate seed-control comparison artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_branch_candidate_controls_field34_cc_c0_v0/`
- greedy failure classifier artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_greedy_failure_classifier_field34_cc_c0_v0/`
- wall-route family profile artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_wall_route_family_profile_field34_cc_c0_v0/`
- focused side-route expansion artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_side_route_expansion_field34_cc_c0_v0/`
- pathway debt-area comparison artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_pathway_debt_area_compare_field34_cc_c0_v0/`
- tunneling evidence profile artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_tunneling_evidence_field34_cc_c0_v0/`
- tunneling path-rank artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_tunneling_path_rank_field34_cc_v0/`

Observed facts from that slice:

- all candidate-vs-vanilla pairs in the selected landscape are endpoint-near;
- support distances remain high, so endpoint closeness is too weak for basin
  identity;
- `baseline_core_transplant_polish` recovers candidate-like support but loses
  quality versus the candidate;
- `control_extra_from_baseline` has stronger quality than the candidate median,
  but it is not candidate-like in support.
- boundary calibration showed isolated group/chunk edits mostly return to the
  original basin;
- candidate support is fully contained in vanilla support for the selected
  field34/cc pairs, so the pure support-set lower bound is `S_V - S_C`;
- `S_V - S_C` ranges from `244` to `916` nodes, median `376`;
- candidate/baseline label closure around that direct support edit touches
  `2,317` to `4,367` nodes in the candidate-2 rows and `2,417` nodes in the
  candidate-0 row;
- median candidate-label closure context ratio is `5.16`, and median
  vanilla-source-label closure context ratio is `4.87`;
- a collision-safe fresh-label relabel pathway still pays a large quality
  barrier, median `64.76` QF and max `167.95` QF.
- the first closure-frontier ranker produced `1,695` label rows, `166`
  eligible candidate-label rows, and `50` selected labels at top 10 per pair;
- selected labels cover only `142` direct nodes but imply `6,422` closure nodes
  and `6,280` context-extra nodes, with median selected closure ratio `65.67`;
- selected direct nodes are collateral-like under the current boundary role
  proxy, so the first mutation pilot should be shrink-from-vanilla rather than
  expand-from-candidate.
- the direct-node-only closure operator pilot emitted `220` rows: `20`
  controls and `200` closure rows across fresh-label and candidate-nearest
  target strategies;
- candidate-nearest direct shrink is better than fresh-label shrink as a raw
  support edit: median support-burden reduction is `13` nodes versus `0` for
  fresh raw, and max reduction is `37` nodes;
- bounded direct polish can beat recreated vanilla and the
  `control_extra_from_baseline` row in some pairs, but the wins remain close to
  vanilla support: the best quality row has `delta_vs_vanilla = 7.389`,
  `delta_vs_control_extra = 4.378`, support reduction `28`, and
  support-distance-to-candidate `0.913`;
- no closure row is currently labeled `quality_win_support_shift` under the
  diagnostic gate; positive rows are `quality_win_same_basin` or dominated by
  control/quality-loss labels.
- bounded outside-support closure-context release selected `8` positive
  direct-shrink prefixes across `4` pairs and emitted `16` context rows;
- the context-release pilot again produced zero `quality_win_support_shift`
  rows. Best quality remained high (`delta_vs_vanilla = 7.389`,
  `delta_vs_control_extra = 4.378` for the best pair), but the maximum
  support-distance-to-vanilla was only `0.083`, below the `0.1` support-shift
  gate;
- releasing same-label outside-support context therefore did not reveal a
  refinement path out of the vanilla basin under the bounded default.
- the label-internal repair control selected `10` high-ratio candidate-label
  rows across `5` pairs and emitted `20` repair rows plus controls;
- the best repair row improved the compact candidate by `3.323` QF and beat
  `control_extra_from_baseline` by only `0.0296` QF, but it was still worse
  than recreated vanilla by `3.410` QF;
- no label-internal row reached `quality_win_support_shift`: raw rows were
  mostly `quality_loss`, and polish rows were `seed_control_dominates`;
- maximum support-distance-to-candidate was only `0.057`, so this control is
  candidate-near and does not explain a transition between the compact
  candidate and broad vanilla support basin.
- ordered flip profiling v0 on the `c2-s11-r0` sanity case confirms that
  QF-positive flips and candidate-progress flips can diverge at the first block:
  `q_first` gets `+0.114` QF for only `0.008` progress, while
  `progress_first` buys `0.109` progress with `-14.231` raw QF;
- after 10 raw label-intersection block flips, `q_first` still has zero raw
  barrier but is almost vanilla-near (`support-distance-to-candidate = 0.919`),
  while `progress_first` moves further (`0.882`) but pays raw barrier `25.71`.
- ordered flip profiling v1 keeps the unit definition fixed and expands only
  the target cases to the top three priority rows: `c2-s11-r0`,
  `c0-s11-r0.001`, and `c2-s42-r0`;
- all three target cases show first-step divergence between `q_first` and
  `progress_first`;
- `q_first` keeps raw barrier `0` and positive final QF, but stays close to
  the vanilla side (`support-distance-to-candidate = 0.631`, `0.919`, and
  `0.965`);
- `progress_first` buys more candidate progress but pays raw QF debt:
  final barriers are `8.22`, `25.71`, and `66.04`;
- the next target row beyond priority 3 was expensive under uncapped exact raw
  CPM recomputation, so further expansion needs a cost cap, incremental delta,
  or cached scoring before broader batch claims.
- the barrier-aware scorer does not rerun Leiden. It re-scores existing v1
  frontier rows for non-greedy prefixes with lower peak raw barriers;
- the first scorer artifact selected `150` prefixes across `3` profiles;
- best selected prefixes reach support progress `0.131`, `0.154`, and `0.062`
  with peak raw barriers `0.437`, `0.933`, and `1.253`, much lower than the
  corresponding `progress_first` end-state barriers;
- these are not operator wins. They are the first candidates for a
  polish-recovery evaluator.
- the first polish-aware evaluator tested `30` prefixes, top `10` per case,
  with only prefix nodes mutable for `3` local-polish iterations;
- all rows recovered or improved QF; there were no quality-loss rows;
- only `1/30` rows reached the diagnostic `recovered_support_shift` label:
  `c0-s11-r0.001` prefix rank `9`, with `+0.805` QF over vanilla and
  support-distance-to-vanilla `0.062`;
- all `c2-s11-r0` and `c2-s42-r0` tested prefixes were
  `recovered_vanilla_near`, so prefix-only polish mostly collapses back toward
  the source basin in those cases.
- the updated state-greedy automated transition search tested `160` state rows
  and `140` action edges over `c0-s11-r0.001` and `c2-s11-r0`, using the top
  `10` barrier-aware prefixes per case, depth `3`, beam width `5`,
  `remaining_target_topk`, `candidate_closure_topk`, and
  `boundary_shell_topk`;
- `c0-s11-r0.001` produced `8` `support_shift_q_recovered` diagnostic rows.
  The best state-greedy row applies two `remaining_target_topk` steps after
  the prefix-polish row, with `+1.188` QF versus vanilla, target progress
  `0.0332`, and support-distance-to-vanilla `0.0979`. Under the target-set
  accounting this row covers `86 / 244` target nodes, leaving `158` target
  nodes unexplained;
- `c2-s11-r0` produced `6` `support_shift_q_recovered` rows after
  `remaining_target_topk` was added, but the cost-aware best row is still the
  prefix-polish `vanilla_collapse` row. The best support-shift row covers
  `134 / 376` target nodes and reaches support-distance-to-vanilla `0.0513`,
  but its state-greedy score is slightly negative, so this is a diagnostic
  reachability signal rather than a useful operator win;
- transition-search rows now report `target_nodes`, per-state `action_nodes`,
  covered/remaining target counts, target coverage fraction, and marginal
  parent-relative target-distance/QF/cost fields. Current v0 search can add
  later target subsets, but the subset selector is still a simple pull-ranked
  `remaining_target_topk` primitive, not a finalized operator.
- reachability-first search over the same c0/c2 slice emitted `196` states and
  `176` action edges with QF removed from the pruning objective. `c0-s11-r0.001`
  produced `35 / 98` support-gate rows, and the best row covers `129 / 244`
  target nodes with support-distance-to-vanilla `0.108` and `+1.174` QF.
  `c2-s11-r0` produced only `2 / 98` support-gate rows; its best row covers
  `199 / 376` target nodes and reaches support-distance-to-vanilla `0.0538`,
  but target progress is only `0.0046`.
- reachability-first also keeps negative-QF support-gate rows on the Pareto
  frontier, so the diagnostic now answers a different question from
  state-greedy search: can a pathway be found at all before judging whether it
  is cost-effective?
- pathway QF wall statistics reconstruct every transition state as a
  root-to-terminal path and measure the maximum QF debt along that path. In
  `reachability_v0 / c0-s11-r0.001`, the lowest-wall support-gate path has wall
  `0.393`, final `delta_q=+0.276`, support-distance-to-vanilla `0.0619`, and
  target progress `0.0180`; the strongest-support path has wall `1.585`, final
  `delta_q=+1.174`, support-distance-to-vanilla `0.108`, and target progress
  `0.0378`. In `reachability_v0 / c2-s11-r0`, only `2 / 98` paths reach the
  support gate; the best one has wall `1.078`, final `delta_q=+0.746`, and
  support-distance-to-vanilla `0.0538`, but target progress is only `0.00461`.
- this makes the next criterion path-level rather than row-level: keep routes
  that improve the wall/progress curve, then test whether their QF recovery and
  mutable-node cost beat seed controls.
- branch target-growth on `c0-s11-r0.001` keeps guarded-elbow, fixed-cap, and
  fixed-tail-backfill choices alive in the same path-level beam. The best
  fixed-tail-backfill row reaches support-distance-to-vanilla `0.108247` and
  target progress `0.037785` with `66` mutable nodes, QF wall `1.5854`, and
  final `delta_q=+1.173811`. This keeps the reachability-level support movement
  while avoiding the earlier broad reachability path's `129` mutable nodes.
- same-case seed/iteration controls do not reproduce that candidate-directed
  movement in this slice: across `15` standard Leiden controls, the best
  quality control is `1.219827` QF better than the branch but has negative
  target progress (`-0.204468`), and no control row has positive target
  progress toward the compact candidate. This is a mechanism signal only, not a
  Dongdaemun-refinement claim.
- the greedy failure classifier labels `37 / 90` branch paths as both
  candidate-directed and QF-recovered at the support gate, and all `37` remain
  `branch_unique_candidate_directed_quality_lag` against the same seed controls.
  The best row carries `q_greedy_miss`, `progress_greedy_miss`,
  `closure_compound_miss`, and `polish_recovery_miss`: this narrows the next
  operator target to a non-greedy, closure-heavy prefix that crosses a raw QF
  wall and relies on bounded polish recovery.
- the wall-route profile keeps the interpretation open: the current artifact
  has one observed candidate-directed wall entry (`p9`, wall `1.585400`), but
  also has `4` lower-wall side-route candidates from `p6`, `p8`, and `p10`.
  Those side routes recover QF and reach partial target progress below the
  support gate (max support-distance-to-vanilla `0.038961`, max target progress
  `0.014847`), so the next search should expand them before assuming the `p9`
  wall is the only route.
- focused expansion of those lower-wall prefixes shows the detour is real but
  incomplete: `p6`, `p8`, and `p10` produce `57` candidate-directed gate-crossing
  rows with minimum wall `0.325642` and maximum support-distance-to-vanilla
  `0.113402`, but `0` rows recover QF at the support gate and the best
  candidate-directed `delta_q_vs_start` is `-0.292613`. The next mechanism
  question is therefore quality recovery around lower-wall detours, not merely
  finding another gate-crossing path.
- debt-area comparison clarifies that peak-wall minimization is not enough:
  the branch `p9` route has a high wall (`1.585400`) but a one-state debt area
  (`1.585400`) and recovers to `delta_q_vs_start=+1.190993`; the lowest-wall
  side route has wall and debt area `0.325642` but remains
  `delta_q_vs_start=-0.325642`, and the best side-route final quality is still
  `-0.292613`. A future operator may deliberately cross a higher wall if that
  path is short, interpretable, and recoverable.
- the tunneling evidence profile labels `37` paths as recoverable tunnels and
  `57` paths as unrecovered detours. The strongest p9 tunnel hits the wall at
  step `0` and reaches candidate-directed support plus QF recovery at step `1`.
  The p8/p10 detours cross the support gate at step `3`, but never reach a
  QF-recovered trace step; extra target growth stays under QF debt. This is
  stronger evidence that the operator should search for recovery after wall
  entry, not just lower-wall gate crossing.
- the multi-artifact tunneling ranker loads `10` existing transition artifacts
  and finds `144` recoverable tunnel seed rows plus `119` unrecovered detour
  recovery-target rows. The best efficiency seed is `c0/p4 fixed_cap` with wall
  and debt-area-step `0.246277`, final `delta_q_vs_start=+0.546858`,
  support-distance-to-vanilla `0.069588`, and target progress `0.021129`.
  The earlier p9 family remains a wider-support seed, not the only tunneling
  candidate. c2 currently acts as a negative/control slice because it reaches
  support gate without enough target progress.
- the post-gate recovery profile focuses on p6/p8/p10 side-route detours after
  they cross the candidate-directed support gate. It expands `96` rows into
  `432` trace steps and finds `13` near-miss recovery trends, `5`
  support-deepening quality tradeoffs, and `27` plateaus. p8 is the cleanest
  recovery target (`-0.325642 -> -0.292613` after gate while support reaches
  `0.088312`), p10 is the widest support-depth stress case (support
  `0.113402`, target progress `0.040106`), and p6 is mostly a plateau/control
  case. None of these rows is a finished recovered tunnel.
- the post-gate recovery move probe starts from those near-miss states and
  tests candidate closure, vanilla closure, and boundary-shell moves. Small and
  wide p8 probes remain plateau, while the p8 full-context probe finds one
  source-side recovery direction: `vanilla_closure_topk:context_only` improves
  `delta_q_vs_start` from `-0.292613` to `-0.196488` while retaining support
  (`0.088312 -> 0.090909`) and target progress (`0.035476 -> 0.036623`). This
  is not an operator win because QF is still negative and the mutable set is
  large (`502` nodes). Candidate/boundary transplant moves instead buy support
  at severe QF cost, so the next mechanism question is compact source-side
  context release, not more target transplant.
- the post-gate recovery subset probe narrows that same p8 full-context row by
  ranked partial subsets. The coarse and tail grids show a sharp all-or-nothing
  pattern: prefixes through `432/436` nodes and all standalone rank bands are
  plateau, while the full `436`-node `pull_prefix` is the only
  `q_gain_support_retained` row. This is evidence that the full probe enlarged
  the reachable search region, but the useful signal is not a simple top-k
  boundary slice. A future operator should identify coherent vanilla-label
  closure components or gate conditions before opening hundreds of context
  nodes.
- the sufficient-subset probe turns that into a narrower search scope. Greedy
  group removal over vanilla labels reduces the context from `436` nodes to one
  source-side vanilla label (`331`) with `209` nodes, while preserving the full
  QF gain (`+0.096125`) and support/progress retention. A one-round
  label-internal pull-band screen shows that removing any 32-node band from
  label `331` kills the gain, while removing bands from other labels is safe.
  The next operator should therefore target a source-side vanilla-label gate,
  not the full closure and not pull-rank top-k.
- the gate trace shows what that narrowed scope does. The 209-node gate and
  full 436-node context have identical affected/sketch endpoint structure and
  identical QF/support/progress. The semantic transition is only target node
  `2890`: the 209 gate nodes do not move, but they provide enough label context
  for node `2890` to attach, gaining `10` internal incident edges with total
  weight `2.186125` and no lost internal edge weight. This reframes the
  operator primitive from "move a closure" to "release a source-side label gate
  around high-pull target nodes".
- the gate attachment candidate scorer tests whether that single moved target
  node could be identified before another polish run. In the p8 source state,
  `2890` is only rank `10/244` by raw pull to the 209-node gate, but rank
  `1/244` by `pull_to_gate_context - pull_to_current_source_label` and by gate
  share over current source-label pull. The score says the useful signal is not
  "same vanilla label" or raw pull alone; it is a weakly anchored target node
  whose gate attachment margin is locally dominant. This should become the
  first selector for a diagnostic gate-release operator.
- the first gate-release operator probe changes that conclusion again in a
  useful direction. Opening the 209-node gate alone recovers `+0.096125` QF
  gain with `275` mutable nodes, but opening only margin top-2 target nodes
  (`2890,7325`) recovers `+0.132629` with `68` mutable nodes and no candidate
  transplant. Manual controls show `2890` alone reproduces the old gate-only
  gain and `7325` alone gives a smaller partial gain, while the pair gives the
  larger endpoint. The operator primitive should therefore be reframed as
  "compact target-mutable tunneling selected by attachment margin"; broad
  source-label gates are fallback context, not the primary action.
- a five-seed recovery control on that compact p8 probe is stable for seeds
  `21003..21007`: margin top-2 target-only has
  `q_gain_mean=min=max=+0.132629`, support `0.093506`, progress `0.037368`,
  mutable `68`, and context `2` in all seeds. The same control keeps gate-only
  at `+0.096125` with `275` mutable nodes. This is a useful first robustness
  check, but it is still one source state; the next claim boundary is
  cross-prefix/cross-source validation, not promotion to a default policy.
- the cross-prefix smoke check recomputes attachment margins on p6 wide, p8
  full-context, and p10 wide source states, then tests target-only `k=1,2,4`
  using each source row's original recovery seed. All three sources have a
  compact target-only row that beats the context-control on QF gain with lower
  mutable cost: p6 top-1 `7325` gives `+0.046504` with mutable `70` versus
  control mutable `173`; p8 top-2 `2890,7325` gives `+0.132629` with mutable
  `68` versus control `+0.096125` and mutable `502`; p10 top-1 `7325` gives
  `+0.046504` with mutable `66` versus control mutable `163`. The signal is
  therefore not unique to the p8 source, but p6/p10 only show the partial
  `7325` recovery. Treat this as evidence for a compact tunneling primitive,
  not as a default policy, until broader cases and seed/iteration controls are
  compared.
- the direct seed/iteration control comparison is mixed in the expected
  direction. Across seeds `11,42,73,101,137`, randomness `0.001,0.01`, and
  budgets `1,10,convergence`, no standard Leiden control has positive target
  progress toward the compact candidate (`candidate_directed_control_rows=0`).
  The compact rows keep positive target progress (`0.033964..0.037368`) with
  only `66..70` mutable nodes. But they do not win on QF: the best broad
  vanilla control is seed `137`, randomness `0.01`, `n=10/convergence`,
  quality `18413.798906`, target progress `-0.202961`, and the compact rows
  lag it by `-2.553622..-2.642564` QF. Against same-randomness `0.001`
  controls, the lag shrinks to `-0.159983..-0.248925`, but it is still
  negative. Label this as `operator_unique_directed_quality_lag`: a real
  directed mechanism signal, not yet a better-partition result.
- a stage2 recovery probe then tests whether the QF lag can be fixed by
  opening local context after the compact directed step. It starts from each
  best compact row and opens candidate-label, current-label, vanilla-label, and
  boundary-shell context around the selected target nodes with multipliers
  `4,16,64`, using both `context_only` and candidate-transplant variants. All
  three sources remain `stage2_no_recovery`. Context-only is an exact no-op;
  label-family candidate transplants are also no-ops for these selected nodes.
  The only nonzero proposals are p8 boundary-shell transplants, which introduce
  heavy pre-polish QF debt (`-4.511790` for 1 changed node and `-51.799403` for
  6 changed nodes) and are fully reverted by polish. Therefore the next
  operator should not be "target move then local recovery"; it should search
  joint target/context bundles that are activated together before polish.
- the first joint-bundle probe confirms that distinction. Starting from the
  source state and activating target top-k plus companion context before polish
  produces a new p8 signal: `12` p8 rows beat the same-randomness control while
  retaining positive target progress. The highest-QF row uses target top-8 plus
  `256` candidate-label context nodes with candidate-bundle transplant
  (`delta_q_vs_vanilla=+0.644965`, target progress `0.009692`, mutable `327`),
  but still lags the best broad control by `-1.748673` QF. A more compact and
  cleaner row uses target top-4 (`2050,2890,7325,9545`) plus current-label
  context top-10 with `joint_mutable` (`delta_q_vs_vanilla=+0.037027`, target
  progress `0.042027`, mutable `79`, same-randomness margin `+0.037027`), but
  it still lags the best broad control by `-2.356611`. p6/p10 improve but stay
  in `joint_directed_quality_lag`.
- the focused replay of that compact p8/current-label row changes the
  interpretation of the large changed-node count. The `candidate_bundle`
  variant reports `3799` exact changed labels, but this is mostly label
  namespace accounting: it and the `joint_mutable` variant have
  `aligned_changed_between_children=0` and endpoint distance `0`. Versus the
  source, both variants share the same aligned changed core of only `6` nodes
  (`2050,2890,5260,7325,9545,9609`). Therefore the next operator should reason
  over aligned support cores and endpoint distance, not raw exact-label
  changed-node counts.
- the basin metric audit makes the required cleanup finite rather than vague.
  Under the current combined evidence root, the refreshed audit after targeted
  recomputation, aligned-core replay, and compact boundary/handle-subset/
  stability/selector/source-screen operator probes has no rerun/backfill-required rows: `341/532` CSV
  artifacts expose label-invariant support, alignment-error, or endpoint
  metrics, `186` are not basin metric artifacts, and `5` artifacts keep compatibility
  `changed_node_count` aliases that need exact-column relabel/reinterpretation
  by readers.
  The previous signature/vanilla risk was a naming issue around
  best-partner-aligned `p5_changed_nodes_vs_baseline`; those paths now have
  explicit `alignment_error` and `aligned_changed_support` aliases. The five
  operator artifacts were also recomputed with explicit aligned/exact companion
  columns: joint-bundle `176` rows, stage2 `51` rows, gate-release v0 `61`
  rows, seed5 `45` rows, and manual `41` rows. This means
  we should not discard all QF/support/endpoint evidence, but every generic
  `changed_node_count` claim must be downgraded unless paired with aligned
  support, changed-support, alignment-error, or endpoint distance.
- the recomputed operator metric review fixes the next branch point. Stage2
  recovery should not be expanded: its `48` action rows have zero final aligned
  movement and zero QF recovery. Gate-release is a tiny local repair family:
  max final aligned movement is `2` nodes and max endpoint distance is `0`,
  even though best QF gain versus the source is `+0.132629`. Joint-bundle is
  the only remaining positive QF signal: max QF gain versus the source is
  `+0.937578`, max final aligned movement is `35`, and mean final aligned
  movement is `2.37`, while max exact-only movement is `7803`. Therefore the
  next operator should be a compact aligned-core plus boundary-context search,
  not a stage2-recovery retry or a raw changed-node expansion.
- the aligned-core frontier replay makes that next operator more concrete.
  Replaying the six positive p8 joint-bundle configs produces a stable direct
  target handle set `2050,2890,7325,9545,9609` and a separate non-target
  boundary/core node `5260`. The compact current-label/boundary-shell rows land
  on the six-node aligned core `2050,2890,5260,7325,9545,9609` with
  `+0.329640` QF gain, while the broader candidate-label row reaches
  `+0.937578` QF gain but expands to `28` aligned nodes. The operator should
  therefore price boundary-core inclusion explicitly before opening wider
  candidate-label context, and exact-only churn must stay outside the objective.
- the compact boundary operator probe then falsifies one tempting
  interpretation of that frontier. On the p8 source, target-only
  candidate-bundle transplant over `2050,2890,7325,9545,9609` already reaches
  the six-node aligned endpoint `2050,2890,5260,7325,9545,9609` with
  `+0.329640` QF gain versus the source and `+0.037027` versus vanilla. Adding
  boundary core `5260`, context-core nodes, or candidate-label context caps
  `8/32` gives no incremental QF over target-only and only increases bundle
  cost. Therefore `5260` should be treated as an induced polish response from
  the already-mutable source context, not as a node that must be explicitly
  opened in the first compact operator. The next test should minimize the
  forced direct-handle transplant and check whether the sufficient handle set
  generalizes across seeds/cases.
- the handle-subset sufficiency probe narrows the compact operator further.
  Exhausting all `31` nonempty subsets of the five direct handles shows that
  `2890,7325,9545,9609` is already sufficient: it recovers the required
  six-node aligned core `2050,2890,5260,7325,9545,9609` and matches the full
  five-handle quality (`+0.329640` QF versus source, `+0.037027` versus
  vanilla, zero QF gap to the full set). The best three-handle near miss
  `2890,9545,9609` recovers `5/6` required nodes and is only `-0.016504` QF
  behind. This suggests `2050` is an induced response rather than a necessary
  forced handle, and the first candidate mechanism is a factorized handle set:
  `9545/9609` induces the `2050/5260` block, while `2890` and `7325` supply
  independent aligned moves. This must be tested across polish seeds/cases
  before becoming an operator rule.
- the stability probe supports that factorized interpretation within the
  current c0 slice. Replaying the minimal subset, full handle set, and strongest
  near misses over `p6_wide`, `p8_fullctx`, `p10_wide` and polish seed offsets
  `2000,3000,4000` gives `9/9` stable sufficient rows for
  `2890,7325,9545,9609`, exactly matching the full five-handle set in every
  source/seed group. The near misses are also stable, but only as partial
  cores: they recover `5/6` required nodes and remain `-0.016504` or
  `-0.066125` QF behind. The next uncertainty is therefore not polish-seed
  fragility. It is whether the sufficient handle set can be selected from local
  features before knowing the endpoint, and whether this mechanism generalizes
  outside the c0/p6-p8-p10 slice.
- the handle selector replay gives the first positive answer to that local
  selection question, but only within the current frontier. Grading top-k
  selector prefixes against the exhaustive subset table shows that both
  `context_pull` and `mutable_penalized_context_pull` select
  `2890,7325,9545,9609` at `k=4`. These are marked as `local_graph_proxy`
  policies because they do not use replay-derived aligned-change counts. The
  selected subset is the minimal sufficient set, has `+0.329640` QF gain versus
  the source, zero QF gap to the full handle set, and stability fraction `1.0`
  in the existing p6/p8/p10 replay. This is not yet an operator rule: the next
  step is to recompute the same feature ranking across more cases without
  relying on the p8 aligned-core frontier as a fixed answer sheet.
- the source-local selector probe refines that rule. Direct node-level
  `gate_pull` fails because it mostly selects already source-mutable decoys.
  `non_source_gate_pull` fixes p8 but still fails p6/p10 because candidate-label
  `1184` decoys outrank the low-pull `1090` core node `2890`. Grouping first by
  candidate label and ranking groups by top-4 local margin fixes this current
  c0 slice: `candidate_label_margin_coherent` selects label `1090`, reaches
  `2890,7325,9545,9609` at `k=4` for p6, p8, and p10, recovers the six-node
  aligned core, and gains `+0.329640` QF versus the source in all three cases.
  The operator implication is now sharper: the tunneling seed should be
  label-coherent before it is node-greedy. This remains diagnostic until the
  same selector is tested on additional pair/candidate slices and compared
  against seed/iteration controls with cost.
- the first c2 selector smoke is mostly a negative/plumbing result. A
  `target_elbow_c2_top10` post-gate profile over `p1..p10` finds only one
  near-miss source, again at `p8`; that source is already QF-positive after the
  gate (`+0.982482`) and support-positive (`0.056235` from vanilla). The
  source-local selector can now run with `evaluation_core_mode=none`, avoiding
  c0 frontier leakage, but it sees only one positive coherent handle
  (`8492`, candidate label `194`) and that handle adds no material QF
  (`operator_delta_q_gain_vs_source=0.0`). Therefore c2 does not yet validate
  the multi-handle label-coherence mechanism. The next useful validation case
  must have multiple non-source positive-margin handles and an unrecovered or
  low-gain post-gate source.
- the branch-based c2 follow-up creates a broader non-c0 control but still does
  not validate the selector. The new artifact
  `basin_transition_branch_target_growth_field34_cc_c2_v0` emits `98` path rows;
  `basin_transition_post_gate_recovery_field34_cc_c2_branch_v0` finds `4`
  near-miss rows and `1` plateau. The best p6 branch row has post-gate recovery
  gain `+0.080258`, but source-move replay is already QF/support retained
  (`source_delta_q=+0.982482`, support `0.056235`). The follow-up attachment
  margin artifact `basin_transition_attachment_margin_cross_prefix_field34_cc_c2_p6_branch_v0`
  exposes only one positive non-source margin handle (`8492`, candidate label
  `194`), has zero QF gain over the source, and is classified as
  `already_recovered_control`. Because this branch bridge replays recorded
  selected-node sequences rather than loading persisted memberships, use it as
  diagnostic reconstruction, not exact source-state equality.
- the readiness profile makes that next-case filter explicit. Scanning the
  existing attachment-margin score artifacts now finds `10` source rows: `6`
  `selector_test_ready` rows, `2` `coherent_label_completion_probe` rows, and
  `2` `already_recovered_control` rows.
  The ready rows are still duplicated c0 `p6_wide`, `p8_fullctx`, and
  `p10_wide` sources across v0/v1 artifacts. The c0 p5/p7 top10 rows add a
  different mechanism shape: they have only one positive anchor (`7325`) in
  candidate label `1090`, but same-label completion to
  `2890,7325,9545,9609` at `k=4` recovers the six-node aligned core and gains
  `+0.329640` QF versus the source. This strengthens the candidate-label-group
  hypothesis, but cross-pair generality remains unproven. The next useful work
  is to screen for a fresh non-c0 post-gate near-miss or low-gain source with
  multiple positive-margin handles before running expensive selector replay.
- the selector-source screener turns that next-case filter into a reusable
  budget gate. `screen_leiden_basin_selector_sources.py` rebuilds post-gate
  source states, recomputes attachment-margin rows, and classifies the source
  before any selector-polish replay. A path-action-only screen is useful as a
  negative control but too conservative for real source selection: it screens
  out c2 branch rows as `5/5` `already_recovered_control`, while also missing
  known c0 top10 positives as `5/5` `too_few_handles`. The better diagnostic
  mode is `recovery_contexts`, which builds bounded recovery contexts without
  running polish trials. On c0 top10 it finds `5` `selector_test_ready` and
  `10` `coherent_label_completion_probe` variants from `5` selected post-gate
  sources; on c2 branch it finds `0` ready and `0` label-completion rows across
  `15` variants, all `already_recovered_control`. Therefore expensive
  selector replay should normally be gated by `selector_test_ready` or
  `coherent_label_completion_probe`, not by the mere existence of a post-gate
  near miss.
- the batch screener closes the current c2 replay branch. Applying
  `run_leiden_basin_selector_source_screen_batch.py` to all existing non-c0
  post-gate recovery artifacts discovers only c2 variants: `6` artifacts, `8`
  selected post-gate sources, and `24` source/context variants. All `24`
  variants are `already_recovered_control`; `ready_count=0` and
  `label_completion_count=0`. This means the next selector experiment should
  not rerun c2 replay. It should generate a fresh non-c0 post-gate source slice
  and pass it through the same `recovery_contexts` screen before any expensive
  local-selector replay.

This means the useful question is no longer whether the previous perturbation
beats random seed variation. It does not. The useful question is whether a
structured closure operator can expose a better quality/support/cost tradeoff
than either broad vanilla movement or compact candidate movement alone.

## Definitions

For one baseline/candidate/vanilla tuple:

- `B`: baseline membership.
- `C`: recreated compact candidate membership.
- `V`: recreated vanilla membership.
- `S_C`: changed-support nodes from `B` to `C`.
- `S_V`: changed-support nodes from `B` to `V`.
- `I = S_C intersect S_V`: shared changed support.
- `E = S_V - S_C`: vanilla-extra support.
- `R_k`: graph boundary ring within `k` hops of `S_C union E`.
- `L_C(E)`: candidate labels touched by direct edit nodes `E`.
- `L_B(E)`: baseline labels touched by direct edit nodes `E`.
- `L_V(E)`: vanilla labels touched by direct edit nodes `E`.
- `K_m(E)`: closure context nodes for mode `m`, i.e. all nodes carrying labels
  in `L_m(E)`.
- `X_m(E) = K_m(E) - E`: closure context outside the direct support edit.
- `closure_ratio_m = |X_m(E)| / |E|`.

Support sets must be computed against best-partner aligned labels, as in the
current transition pilot, so relabeling alone does not create support.

The pure support-set lower bound is `|E| + |S_C - S_V|`. In the current
field34/cc target slice, `S_C` is fully contained in `S_V`, so this reduces to
`|E|`. This lower bound is not an executable mutation. The executable operator
must respect label closure, local edges, and CPM quality debt.

### Transition Target Model

Do not use `direct_nodes` as the top-level transition object. The old wording
made the diagnostic look as if the target were one node or one fixed seed set.
The transition target is a set-level object, and each search step chooses a
small action subset from it.

For a directed transition `source -> target`, define:

- `target_nodes T`: the full support set that the transition must explain.
  For `V -> C`, this is usually the source-side excess or disagreement support
  that must be removed or reassigned. For `C -> V`, this is the missing
  source-to-target support that would need to be added back.
- `target_units U(T)`: an interpretable partition of `T`, such as
  label-intersection blocks, closure labels, or boundary-connected chunks.
  These are ranking units, not necessarily mutation units.
  The current v0 unit profiler uses three cheap definitions:
  label-intersection blocks, target-induced connected components, and
  triangle-supported components using common target-neighbor edge support.
- `action_nodes A_t`: the subset selected at step `t` for a concrete mutation
  or release. `A_t` must be a subset of `T` unless the experiment explicitly
  marks an exploratory context promotion.
- `context_nodes C_t`: nodes outside `A_t` that are not the primary target but
  may need to be released so local polish can make a coherent move.
- `mutable_nodes M_t = A_t union C_t`: the full set unfrozen for the local
  polish step.
- `covered_nodes P_t = union_{i <= t} A_i`: the part of `T` already attempted
  by the pathway.
- `remaining_nodes R_t = T - P_t`: target support not yet explained.

The greedy question is therefore not "which direct node has best QF?". It is:

1. Which subset `A_t` of the remaining target set should be attempted next?
2. Does applying `A_t` reduce label-invariant state distance to the target
   basin after a bounded polish?
3. Which small context `C_t` is necessary, and when does it become too broad to
   be a useful operator?
4. Does the marginal transition step beat seed or iteration controls on
   material gain per cost?

This framing keeps the pathway search aligned with refinement practice: each
step is a local repair/release over a target subset, not an arbitrary shake.

## Node Classes

The boundary analyzer should classify nodes or small node groups before any
operator mutates membership.

`core`

- Nodes in `S_C`.
- Candidate-side support that appears compact and reproducible.
- This is not automatically correct; it is simply the current compact endpoint
  hypothesis.

`shared`

- Nodes in `I`.
- Both candidate and vanilla changed these nodes relative to baseline.
- These nodes are the first target for bridge/collateral explanation.

`vanilla_extra`

- Nodes in `E`.
- These are the main unknowns. They may be necessary bridge nodes, collateral
  movement, or evidence that the compact candidate is missing quality.

`boundary_anchor`

- Nodes outside `S_C union S_V` but adjacent to the core or extra support.
- They provide local context and should usually be fixed in first prototypes.

`closure_context`

- Nodes in `K_m(E) - E` for a chosen closure mode.
- These nodes are outside the direct support edit but share baseline,
  candidate, or vanilla labels with direct edit nodes.
- They are the main reason the new operator should be framed as controlled
  split/merge rather than node-level revert.

`closure_label`

- A baseline, candidate, or vanilla label touched by direct edit nodes.
- The useful ranking unit for the next operator is likely a closure label, not
  an individual node or raw boundary group.

`bridge`

- Vanilla-extra nodes or groups with strong weighted links from the core to the
  vanilla destination labels, and weak pull back to baseline labels.
- These are candidates to keep or gradually release in expand-from-candidate.

`collateral`

- Vanilla-extra nodes or groups with weak core pull, strong baseline pull, or
  low estimated quality loss when reverted.
- These are candidates to shrink from the vanilla basin.

## Required Diagnostics

The boundary analyzer and closure-context analyzer are now the required dry-run
inputs before another operator run.

For each selected candidate-vs-vanilla pair, emit full-support rows, not only
the 1024-node endpoint sketch:

- per-node support class: `core`, `shared`, `vanilla_extra`,
  `boundary_anchor`;
- baseline label, candidate label, vanilla label;
- node weight;
- incident edge weight to `S_C`, to `E`, to same baseline label, to candidate
  destination labels, and to vanilla destination labels;
- strongest candidate-pull label and vanilla-pull label;
- approximate local score for three actions:
  - keep vanilla assignment;
  - revert to baseline-aligned assignment;
  - move to candidate-nearest assignment.

Also emit group-level rows aggregated by:

- baseline label;
- candidate label;
- vanilla label;
- support class;
- connected component within the induced support/boundary subgraph, if cheap.

The first analyzer output should be diagnostic only:

- `basin_transition_boundary_node_rows.csv`
- `basin_transition_boundary_group_rows.csv`
- `basin_transition_boundary_summary.json`
- `basin_transition_boundary_report.md`

The closure-context analyzer output is also diagnostic only:

- `basin_transition_closure_context_pairs.csv`
- `basin_transition_closure_context_labels.csv`
- `basin_transition_closure_context_summary.json`
- `basin_transition_closure_context_report.md`

## Scoring

Scores are screening features, not acceptance criteria.

For a node or group `g`:

- `core_pull(g)`: edge weight from `g` to `S_C`, normalized by incident weight.
- `vanilla_pull(g)`: edge weight from `g` to nodes with vanilla-compatible
  labels, normalized by incident weight.
- `candidate_pull(g)`: edge weight from `g` to candidate-compatible labels,
  normalized by incident weight.
- `baseline_pull(g)`: edge weight from `g` to baseline-compatible labels,
  normalized by incident weight.
- `bridge_score(g) = core_pull(g) * vanilla_pull(g)`.
- `collateral_score(g) = baseline_pull(g) * (1 - core_pull(g))`.
- `necessity_score(g)`: estimated quality loss when reverting `g` from vanilla
  assignment to baseline or candidate-nearest assignment.

The exact formula can change after the dry run, but all scores must be reported
beside the action that used them. Hidden ranking features make the result hard
to audit.

For closure labels, add a second screening layer:

- `direct_node_count(l)`: direct edit nodes in label `l`.
- `closure_node_count(l)`: total same-label closure nodes for `l`.
- `closure_context_extra_count(l)`: `closure_node_count - direct_node_count`.
- `closure_outside_support_count(l)`: closure nodes outside `S_C union S_V`.
- `closure_context_ratio(l)`: context extra per direct node.
- `role_mix(l)`: collateral/ambiguous/bridge composition of direct nodes.

High `closure_context_ratio` does not automatically mean "do not edit." It
means the label is a split/merge candidate whose context must be included or
explicitly protected.

## Closure-Aware Operator Family

### 0. Closure Label Frontier

Start from the closure-context rows.

Purpose: select a small, auditable set of closure labels before any membership
mutation.

Draft procedure:

1. Join closure-label rows with boundary node rows.
2. Rank labels by a two-axis rule:
   - small-to-moderate `direct_node_count`, so the pilot stays inspectable;
   - high `closure_context_extra_count` or high `closure_context_ratio`, so the
     test targets the actual closure problem.
3. Exclude labels whose direct nodes are mostly bridge-like unless the operator
   is explicitly an expand test.
4. Keep a small pilot set, for example top `5-20` labels per pair, with a hard
   cap on closure nodes.
5. Emit a label-frontier ledger before mutation:
   `closure_label_frontier_rows.csv`.

No mutation should run without this ledger.

### 1. Closure-Split Shrink From Vanilla

Start from `V`.

Purpose: test whether the broad vanilla basin contains removable collateral
movement while keeping quality close to vanilla.

Draft procedure:

1. Pick closure labels from the frontier whose direct nodes are mostly
   collateral-like or ambiguous.
2. Build a mutable set
   `M_t = A_t union C_t`, where `A_t` is the chosen target subset for the
   label and `C_t` is the selected closure or one-hop boundary context.
3. Split `A_t` away from its vanilla label into baseline-aligned or
   candidate-aligned labels. Keep closure-context nodes fixed in the first
   pilot unless they are explicitly released.
4. Run a short fixed-budget polish only over `M`.
5. Record exact CPM quality and support distances after each label release.
6. Stop when marginal quality debt or closure growth exceeds the configured
   pilot cap.

This is the direct successor to the failed hard-freeze pilot. The difference is
that the mutable region is chosen by closure evidence, not by `S_C` alone or by
isolated boundary groups.

### 2. Closure-Gated Expand From Candidate

Start from `C`.

Purpose: test whether the compact candidate is missing a small number of
necessary bridge groups from the broader vanilla support.

Draft procedure:

1. Pick closure labels whose direct nodes are bridge-like or whose
   `necessity_score` suggests reverting them is expensive.
2. Define `T` as the target support missing from the candidate-like state, then
   pick a small `A_t` from `T`; do not automatically release the entire closure
   label.
3. If quality collapses or the support path is incoherent, retry with a bounded
   portion of `C_t`, ordered by edge pull to `A_t`.
4. Initialize `A_t` nodes to vanilla labels or candidate-nearest compatible
   labels.
5. Run a short local polish after each release.
6. Stop when marginal gain per second or per changed node falls below gate.

This tests a real transition mechanism: candidate support is not merely replayed
or polished; it is allowed to expand only through evidence-backed boundary
groups.

### 3. Two-Frontier Compromise

Start from a compact candidate-like membership, but define two mutable
frontiers:

- a shrink frontier for weak vanilla-extra groups;
- an expand frontier for strong bridge groups.

The expected output is not necessarily closest to `C` or `V`. It can be an
intermediate support basin that keeps the endpoint benefit while reducing broad
collateral movement.

This should be attempted only after the first two one-sided operators produce
interpretable boundary rows.

### 4. Label-Internal Repair Control

Start from `V` or `C`, but do not target support distance first.

Purpose: isolate whether the high closure ratio is caused by a few large labels
that need internal splitting rather than external support transfer.

Draft procedure:

1. For one high-ratio closure label, induce the subgraph on
   `closure_label_nodes union one-hop direct-node boundary`.
2. Run a tiny local split/merge repair inside that induced subgraph with CPM
   unchanged.
3. Compare against direct-node-only shrink for the same label.
4. Keep the result diagnostic unless it beats the seed control on material
   gain per cost.

## Acceptance Gates

A row is not successful because `delta_q > 0`.

Compare every transition output against:

- recreated baseline;
- recreated compact candidate;
- recreated vanilla;
- `control_extra_from_baseline`;
- at least one additional vanilla seed/iteration control when available.

Required metrics:

- `delta_vs_baseline`;
- `delta_vs_candidate`;
- `delta_vs_vanilla`;
- `delta_vs_seed_control`;
- `elapsed_sec`;
- `gain_per_second`;
- `result_support_size`;
- `support_distance_to_candidate`;
- `support_distance_to_vanilla`;
- `endpoint_distance_to_candidate`;
- `endpoint_distance_to_vanilla`;
- `released_group_count`;
- `released_node_weight`;
- `p5_or_polish_iterations`;
- `process_hwm_mb` when available.

Suggested labels:

- `quality_win_support_shift`: quality beats seed control and support is
  meaningfully different.
- `quality_win_same_basin`: quality beats seed control, but support/endpoints
  are essentially the same as an existing basin.
- `support_win_low_roi`: support is cleaner or more interpretable, but quality
  gain is too small for the cost.
- `quality_loss`: quality regresses against the relevant control.
- `seed_control_dominates`: extra vanilla seed/iteration remains better on
  material gain per cost.

Only the first label can motivate a stronger Dongdaemun-refinement claim. The
others remain diagnostic.

## Implementation Sequence

1. [x] Add the boundary analyzer and emit node/group/pair rows.
2. [x] Add boundary analyzer tests.
3. [x] Add boundary group calibration and record that isolated group edits
   mostly do not persist.
4. [x] Add minimum relabel-pathway accounting over `S_V - S_C`.
5. [x] Add closure-context accounting over baseline/candidate/vanilla labels.
6. [x] Add a closure-label frontier ranker:
   `research/consensus/scripts/rank_leiden_basin_transition_closure_frontier.py`.
7. [x] Add a dry-run closure operator pilot:
   `research/consensus/scripts/run_leiden_basin_transition_closure_operator_pilot.py`.
8. [x] First pilot mode: `closure_split_shrink_from_vanilla`.
9. [x] Add bounded closure-context release after positive direct-shrink rows:
   `research/consensus/scripts/run_leiden_basin_transition_closure_context_release_pilot.py`.
10. [x] Add label-internal repair control for high-ratio closure labels:
   `research/consensus/scripts/run_leiden_basin_transition_label_internal_repair_pilot.py`.
11. [x] Add ordered flip basin profiling v0 before another operator:
   `research/consensus/scripts/profile_leiden_basin_ordered_flips.py`.
12. [x] Add ordered flip basin profiling v1 batch over the first three
    priority cases:
   `research/consensus/scripts/profile_leiden_basin_ordered_flips_batch.py`.
13. [x] Add barrier-aware pathway scorer over existing ordered-flip prefixes:
   `research/consensus/scripts/analyze_leiden_basin_barrier_aware_pathways.py`.
14. [x] Add polish-aware prefix evaluator for selected barrier-aware prefixes:
   `research/consensus/scripts/evaluate_leiden_basin_polish_prefixes.py`.
15. [x] Add automated transition search over prefix/context primitive
    combinations:
   `research/consensus/scripts/search_leiden_basin_transitions.py`.
16. [x] Split transition search semantics into `target_nodes`, `action_nodes`,
    `context_nodes`, and `mutable_nodes`; keep `direct_nodes` only as a legacy
    input alias until artifact schemas are migrated.
17. [x] Add target-set pathway rows that report `covered_target_count`,
    `remaining_target_count`, marginal target-distance reduction, marginal
    quality debt, and marginal cost per action subset.
18. [x] Extend the search beyond one prefix-derived `action_nodes` subset per
    path, so later steps can select additional subsets from
    `remaining_target_nodes` rather than only adding context around the first
    action subset.
19. [x] Profile target-node unit definitions before adding another action
    family, so node-level accretion is compared against measurable unit
    cohesion rather than intuition.
20. [x] Replace or complement node-level `remaining_target_topk` with
    unit-aware target-subset actions, so the next search can distinguish
    coherent label/block releases from broad node accretion.
21. [x] Profile pull-curve elbow candidates for node-level staged target
    growth before replacing the fixed top-k cap.
22. [x] Compare fixed-cap top-k and guarded-elbow top-k with bounded polish on
    the same c0/c2 prefixes.
23. [x] Test guarded-to-fixed escalation and fixed-tail backfill variants.
24. [x] Add a small branching target-growth search that keeps guarded, fixed,
    and backfill first-step variants alive before pruning by polished
    quality/support/cost.
25. [ ] Add cached or incremental target-unit scoring before deeper unit-aware
    beam searches. The naive full-branch unit search is too expensive for
    routine diagnostics.
26. [ ] Second pilot mode, only after search rows show an actionable block
    family:
   `closure_gated_expand_from_candidate`.
27. [x] Compare every pilot row against recreated candidate, recreated vanilla,
    and `control_extra_from_baseline`; broader seed-control replication remains
    a follow-up before any algorithm claim.
28. [ ] Promote nothing into Rust or production paths until the diagnostic
    evidence shows material quality/cost benefit.

## Non-Goals

- Do not change the standard Leiden path.
- Do not change `Dongdaemun-post`.
- Do not promote this as `Dongdaemun-refinement` default behavior.
- Do not run large graph stress tests before the field34/cc boundary analyzer
  produces interpretable rows.
- Do not treat endpoint-near alone as basin identity.
- Do not treat a low-ROI positive quality delta as a useful result.

## Latest Diagnostic Artifact

The completed direct-node and bounded context shrink pilots both said that the
shrink side was still vanilla-near. The follow-up control tested whether the
problem was instead internal splitting inside high-ratio candidate labels:

`research/consensus/scripts/run_leiden_basin_transition_label_internal_repair_pilot.py`

Default output:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_label_internal_repair_pilot_field34_cc/`

Result:

- selected `10` high-ratio candidate-label rows across `5` pairs;
- emitted `40` rows total, including `20` repair rows;
- best repair delta was `+3.323` versus compact candidate and `+0.0296`
  versus `control_extra_from_baseline`;
- the same row remained `-3.410` versus recreated vanilla;
- all repair rows were labeled `quality_loss` or `seed_control_dominates`;
- maximum support-distance-to-candidate was `0.057`, so the edits are still
  candidate-near rather than a distinct support transition.

Interpretation:

Label-internal repair does not rescue the shrink-side mechanism. It shows that
some candidate-near labels can be locally repaired, but not with material
advantage over vanilla/seed controls. The remaining one-sided mechanism worth
testing is `closure_gated_expand_from_candidate`; if that also fails, this
whole refinement family should be deprioritized in favor of a basin-graph or
multi-start selection approach.

## Basin Profiling Gate

Before another operator is implemented, run profiling as a separate diagnostic
layer. The v0 artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_ordered_flip_frontier_field34_cc_v0/`

It uses `label_intersection_block` units on `c2-s11-r0`, direction `V -> C`,
raw flips only. The key finding is that QF-first and progress-first policies
choose different first blocks.

The v1 batch artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_ordered_flip_frontier_field34_cc_v1_cases/`

It keeps the v0 definition fixed and expands only to the first three priority
target pairs. All three repeat the same mechanism split: QF-positive blocks are
low-progress and vanilla-near, while progress blocks are costly in raw QF. This
is evidence against another generic local refinement operator. The next local
operator should be attempted only if it explicitly prices this raw QF debt and
adds a bounded polish test; otherwise the honest next direction is a basin-graph
or multi-start selection approach.

The barrier-aware prefix artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_barrier_aware_prefix_field34_cc_v1/`

It re-scores the existing v1 frontier rows without recomputing memberships. The
best selected prefixes are non-greedy under one-step ranks but have much lower
peak raw barriers than `progress_first`: roughly `0.44`, `0.93`, and `1.25`
QF for support progress `0.13`, `0.15`, and `0.06`. This is the first
actionable input to a polish-aware evaluator: test whether these low-barrier
prefixes are recoverable and durable, not whether their raw score alone is
positive.

The polish-aware prefix artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_polish_aware_prefix_field34_cc_v1/`

It applies selected prefixes to vanilla, compacts the edited membership, and
runs bounded local polish with only prefix nodes mutable. The first run tested
`30` prefixes. QF recovery was strong, but durable support shift was rare:
`1/30` rows reached `recovered_support_shift`, while the remaining rows were
`recovered_vanilla_near`. This says prefix-only mutation is too narrow for most
cases; the next mechanism test should use the same module API but enlarge the
mutable set with closure/context nodes for selected prefixes, rather than
continuing threshold sweeps.

The transition-search artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_v0/`

It tests prefix/context primitive combinations over `c0-s11-r0.001` and
`c2-s11-r0`. The key update is the target/action/context/mutable split:
`target_nodes` is the full support objective, while each step chooses
`action_nodes` and optional `context_nodes`. `remaining_target_topk` gave the
first staged target-coverage signal, but c2 remains low-ROI under the current
state-greedy score. This is evidence for better unit selection, not for
promoting an operator.

The target-unit profiling artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_units_field34_cc_v0/`

It profiles `label_intersection_block`, `target_connected_component`, and
`triangle_supported_component` units over the same two transition targets. The
run produced `553` unit rows. For `c0-s11-r0.001`, a broad `81`-node connected
component is split into triangle-supported components with max size `30`. For
`c2-s11-r0`, connected components have max size `77`, triangle-supported
components max size `73`, and label-intersection blocks max size `41`.

This closes the unit-definition diagnostic for v0. The next search action
should be `remaining_target_unit_topk`: select whole target units by a
cost-aware score, then measure whether unit-level coverage improves marginal
target progress per mutable node relative to node-level `remaining_target_topk`.

The first unit-aware target-action artifacts are:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_unit_c0_label_depth1_v0/`

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_unit_c2_label_depth1_v0/`

The implementation adds `remaining_target_unit_topk`, but the result is not an
operator win. On `c0-s11-r0.001`, the best label-block unit action reached
support-distance-to-vanilla `0.057` and `delta_q=0.280`, below the prefix-only
best and far below the previous node-level staged target row. On `c2-s11-r0`,
the best label-block unit action remained `vanilla_collapse`. Mixed unit
branching was also costly enough that the full two-pair depth-three run was
stopped before writing rows.

This changes the next question: do not keep broadening naive unit beam search.
The useful next diagnostic is a cached, cost-aware unit scorer that can explain
why a unit should be bought before bounded polish is run. If that scorer cannot
beat the simple node-level staged target growth on material support shift per
mutable node, this local transition-operator family should be deprioritized.

The node-level target-elbow artifact is:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_field34_cc_v0/`

It profiles the pull-score curve used by `remaining_target_topk` without
running bounded polish. The fixed rule chooses
`ceil(anchor_size * target_action_multiplier)` up to the configured cap. The
guarded elbow rule records candidate cut points and currently uses:

- largest pull-score gap only if the drop is at least `25%` of top pull and the
  selected prefix covers at least `50%` of total positive pull;
- otherwise the first `80%` cumulative-pull point.

This avoids the first failure mode where raw max-gap often chose `k=1` while
capturing only `15-22%` of pull. On the default c0/c2 profile, the guarded rule
keeps median pull fraction around `0.81` while reducing median k: c0 from
about `16` to `11`, and c2 from roughly `31-34.5` to `18-18.5` depending on
path. This is a plausible speed/cost candidate, not yet a quality result.

The next validation should run matched fixed-cap and guarded-elbow target
actions through bounded polish. The acceptance question is whether guarded
elbow preserves material support shift and QF recovery with fewer mutable
nodes and lower wall time.

The bounded-polish target-elbow artifacts are:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_smoke_v0/`

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c0_top10_v0/`

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c2_top10_v0/`

Escalation/backfill follow-up artifacts are:

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c0_escalate_v0/`

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c2_backfill_v0/`

`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c0_backfill_v0/`

The script is:

`research/consensus/scripts/evaluate_leiden_basin_target_elbow_polish.py`

Result:

- On `c0-s11-r0.001`, fixed-cap finds the stronger transition row:
  state-greedy score `0.131`, `delta_q=1.191`,
  support-distance-to-vanilla `0.103`, target coverage `0.307`, and `75`
  mutable nodes. Guarded elbow remains a valid support-shift row with fewer
  mutable nodes: score `0.112`, `delta_q=0.929`,
  support-distance-to-vanilla `0.073`, coverage `0.189`, and `46` mutable
  nodes.
- On `c2-s11-r0`, fixed-cap again produces only a weak reachability row:
  `delta_q=0.962`, support-distance-to-vanilla `0.051`, coverage `0.356`,
  and `134` mutable nodes, but state-greedy score `-0.0077`. Guarded elbow is
  cheaper but never crosses the support gate, so it remains
  `vanilla_collapse`.

Interpretation:

Guarded elbow is not a drop-in replacement for fixed-cap target accretion. It
can reduce cost on c0 while preserving a weaker support-shift signal, but on
c2 it trims away the tail nodes needed to cross the support gate. The next
operator should therefore use guarded elbow as a first-stage scheduler with an
explicit escalation rule: if bounded polish remains support-near-vanilla,
increase toward fixed-cap or switch to a different target unit/context family.

Escalation result:

- `guarded_escalate` starts with guarded elbow and then switches to current
  fixed-cap selection when support-distance-to-vanilla remains below the gate.
- `guarded_backfill` starts with guarded elbow and, if support remains below
  gate, first adds the fixed-cap tail skipped by the previous guarded frontier.
- On `c0-s11-r0.001`, `guarded_escalate` is a better compromise than pure
  guarded elbow: score `0.130`, `delta_q=1.032`,
  support-distance-to-vanilla `0.098`, coverage `0.225`, and `55` mutable
  nodes. It is still weaker than fixed-cap on absolute movement, but much
  cheaper.
- On `c2-s11-r0`, neither escalation nor backfill crosses the support gate.
  The fixed-cap row remains the only support-shift row, and it is still a
  low-ROI reachability signal rather than a useful operator win.

Interpretation:

The first target mutable set is path-dependent. Once the guarded first step
polishes inside the wrong local basin, later fixed-cap or tail-backfill
additions do not reconstruct the fixed-cap trajectory. The next diagnostic
should therefore be branching, not a single greedy scheduler: keep guarded and
fixed first-step variants alive until bounded polish shows which branch
actually enters a useful basin wall.
