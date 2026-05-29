# SciSci Adaptive Refinement TODO

## Status

This is a design TODO, not a production implementation commitment yet.

Current local goal: replace the legacy `direct_nodes`-as-target framing with a
set-level pathway model: define the full `target_nodes` support set, select
stepwise `action_nodes` subsets from it, release only bounded `context_nodes`
when needed, and evaluate each `mutable_nodes = action_nodes union
context_nodes` step by support-progress, quality-recovery, and cost tradeoff.
The next question is not whether a wall exists; different basins should have
one. The question is which action subset crosses or reshapes it with the best
state-progress per quality and runtime cost.

Reusable module boundary: shared basin profiling mechanics live in
`sciscape/clustering/leiden_basin_profile.py`; shared transition-search
mechanics live in `sciscape/clustering/leiden_basin_search.py`. Research
scripts should stay as thin artifact runners. Do not copy ordered-flip,
support-distance, fresh-transplant, barrier-aware scorer, context expansion, or
search-state classification logic into another one-off script.

Current Leiden multi-basin work should follow
`docs/research/leiden_basin/leiden_multibasin_research_guardrails.md`: separate "find a better
partition" from "find it faster", require material and cost-adjusted quality
gains rather than any positive `delta_q`, and treat large/dense multi-basin
claims as hypotheses until basin-level evidence is available.

Dongdaemun-branded artifacts should also follow
`docs/research/dongdaemun/core/dongdaemun_naming_contract.md`: use `Dongdaemun-post` for the validated
postprocess method, `Dongdaemun-refinement` for opt-in integrated-loop research,
and `Dongdaemun diagnostics` for probes, basin signatures, and uncommitted
candidates.

The standard Rust Leiden path should stay close to CWTS/Java behavior. SciSci-
specific perturbation should be implemented as a separate experimental stage
after baseline Leiden and before small-cluster postprocess.

As of 2026-04-29, the first prerequisite is implemented: standard Leiden now
logs enough progress detail to identify late near-identity contractions and low
movement iterations (`moved_nodes`, recursion `depth`, and contraction
node/edge deltas). A large-graph recursion guard also skips recursive Leiden
calls when contraction has become nearly identity. Cluster-graph diagnostics,
boundary probes, high-gamma split probes, split-then-repair dry-runs, cached
split-repair probe/apply helpers, and the Rust Dongdaemun core are implemented.
The opt-in Rust Dongdaemun hierarchy fast path has a first field12 validation,
but those timings compare backend behavior rather than equivalent kernels. The
next algorithmic step is the integrated Dongdaemun-refinement design: use
postprocess-learned signals to allocate extra parent-internal Leiden refinement
budget before contraction, while leaving CPM unchanged.

Current Dongdaemun implementation status:

- [x] Internal cached split-repair probe/apply helper, kept `pub(crate)`.
- [x] Rust Dongdaemun core with exact CPM audit and fallback semantics.
- [x] PyO3 binding plus Python `dongdaemun_refine_rust` helper.
- [x] Receiver oversize-aware Rust candidate ranking.
- [x] Opt-in hierarchy integration via
      `use_rust_dongdaemun=True, write_artifacts=False`; artifact-writing runs
      stay on the Python backend until move-row parity is implemented.
- [x] Integrated Dongdaemun-refinement design documented in
      `docs/research/dongdaemun/refinement/dongdaemun_refinement_algorithm_design.md`.
- [x] Experimental Rust Leiden slice for size-priority parent-internal
      Dongdaemun refinement.
- [x] Adaptive local shake portfolio design documented in
      `docs/research/dongdaemun/refinement/dongdaemun_adaptive_local_shake_design.md`.
- [ ] Committed-best hard-cap state tracking (`CommittedBestCapState`).
- [ ] Python ranking parity, only if evidence reproduction needs it.
- [ ] Lower-tail interaction ablation.
- [ ] Multi-seed and hierarchical probe schedules.
- [ ] Fused Rust postprocess kernel profiling and implementation decision.
- [x] Boundary-context closure design for minimum basin-transition pathways.
- [x] Closure-label frontier ranker for field34/cc basin-transition pilot.
- [x] Closure-split shrink-from-vanilla dry-run pilot.
- [x] Bounded closure-context release pilot for labels where direct shrink is
      quality-positive but still support-near-vanilla.
- [x] Label-internal repair pilot for high-ratio closure labels.
- [x] Research-local basin profiling v0 for ordered label-intersection block
      flips on the `c2-s11-r0` pathway sanity case.
- [x] Research-local basin profiling v1 batch for ordered
      label-intersection block flips over the first three field34/cc target
      cases.
- [x] Barrier-aware pathway scorer for non-greedy prefixes:
      minimax raw barrier, support progress per barrier, compactness, and
      optional closure/context cost.
- [x] Move reusable ordered-flip and barrier-aware scorer mechanics into
      `sciscape.clustering.leiden_basin_profile`; keep research scripts as
      artifact runners.
- [x] Polish-aware prefix evaluator for top pathway prefixes:
      raw edit, bounded local polish, QF recovery, support retention, and
      vanilla/candidate control comparison.
- [x] Automated basin-transition search v0 over prefix/context primitive
      combinations.
- [x] Document the target/action/context/mutable set split for basin-transition
      pathway search.
- [x] Refactor transition-search diagnostics so `target_nodes` is the full
      support objective and `action_nodes` is the per-step selected subset.
- [x] Add pathway coverage metrics:
      `covered_target_count`, `remaining_target_count`,
      `marginal_target_distance_reduction`, `marginal_q_debt`, and
      `marginal_cost_per_target_node`.
- [x] Extend transition search so later pathway steps can choose additional
      action subsets from `remaining_target_nodes`, not only release context
      around the initial prefix action.
- [x] Profile target-node unit definitions before adding another search
      action: label-intersection blocks, target-induced connected components,
      and triangle-supported components.
- [x] Add unit-aware target-subset actions, so `remaining_target_topk` is not
      the only way to grow coverage across the target set.
- [x] Profile pull-curve elbow candidates for node-level staged target growth,
      before changing the `remaining_target_topk` cap rule.
- [ ] Add cached or incremental target-unit scoring before running deeper
      unit-aware beam searches; full branch expansion is too expensive for
      routine diagnostics.
- [x] Add a bounded-polish comparison of fixed-cap top-k versus guarded-elbow
      top-k on the same c0/c2 prefix rows.
- [x] Test guarded-to-fixed escalation and fixed-tail backfill variants for
      target-node growth.
- [x] Add a QF-independent `reachability_first` transition-search policy so
      pathway discovery can cross low-quality intermediate states while still
      reporting QF debt.
- [x] Add pathway-level QF wall statistics over transition-search state graphs:
      root-to-terminal wall height, support-gate wall distribution, wall
      frontier rows, and wall/progress/cost case summaries.
- [x] Add branching target-growth search that keeps guarded, fixed, and
      backfill variants alive at the first target step instead of committing to
      a single irreversible path.
- [x] Compare the best branch target-growth candidate against same-case
      standard Leiden seed/iteration controls on QF, support movement, target
      progress, wall, and mutable-node cost.
- [x] Greedy failure classifier:
      `q_greedy_miss`, `progress_greedy_miss`, `closure_compound_miss`, and
      `polish_recovery_miss`.
- [x] Wall-route family profile that distinguishes the observed candidate
      wall from lower-wall side-route candidates below the support gate.
- [x] Focused side-route expansion from `p6`, `p8`, and `p10` to test whether
      lower-wall partial routes can cross the support gate.
- [x] Add pathway debt-area metrics so wall routes are compared by peak height,
      duration/area under QF debt, recovery slope, and progress per debt area.
- [x] Add tunneling evidence profile:
      recoverable tunnel versus unrecovered detour labels, contrast rows, and
      per-state wall/gate/recovery traces.
- [x] Rank existing transition artifacts as tunneling operator candidates:
      recoverable tunnel seeds, unrecovered detour recovery targets, and
      partial-progress entrance probes.
- [ ] Prefix-derived perturbation pilot, only if polish-aware rows retain
      support progress with material or cost-adjusted value against seed
      controls.
- [ ] Closure-gated expand-from-candidate pilot, if one final one-sided
      mechanism test is still worth running after the negative repair control.

Branch target-growth search result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_branch_target_growth_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/basin_signatures/branch_growth/search_leiden_basin_branch_target_growth.py`
- Scope is deliberately `c0-s11-r0.001` only: top-10 barrier-aware prefixes,
  guarded-elbow, fixed-cap, and fixed-tail-backfill target-growth branches,
  beam width `8`, and `3` target stages.
- The run emitted `90` state/path rows and `80` edges. The best recovered
  support-gate row is a fixed-tail-backfill branch from prefix `p9`:
  support-distance-to-vanilla `0.108247`, target progress `0.037785`,
  `delta_q = +1.173811`, QF wall `1.5854`, and `66` mutable nodes.
- Interpretation: branch search keeps the useful reachability-level support
  movement while avoiding the earlier broad reachability path's `129` mutable
  nodes. This is a better operator-development target than c2, but it remains
  diagnostic until compared against seed controls and material cost-adjusted
  quality gates.

Branch candidate seed-control comparison result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_branch_candidate_controls_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/selector_signals/evaluate_leiden_basin_branch_candidate_controls.py`
- Scope is the same `c0-s11-r0.001` branch row, compared against `15`
  standard Leiden controls: seeds `11,42,73,101,137`, randomness `0.01`, and
  iteration settings `1`, `10`, and convergence.
- Verdict is `branch_unique_candidate_directed_quality_lag`: the branch is the
  only tested row with positive target progress toward the compact candidate
  (`+0.037785`), while every standard Leiden control has negative target
  progress.
- This is not a quality win. The best control (`seed=137`, `n=10`) is
  `1.219827` QF better than the branch, but it moves away from the candidate
  target (`target_progress = -0.204468`) and is broad/off-target in support.
- Interpretation: the branch row is a mechanism signal for candidate-directed
  movement that ordinary seed variation did not reproduce in this slice. It is
  still diagnostic, not a Dongdaemun-refinement claim, until an operator beats
  seed controls on material and cost-adjusted value.

Greedy failure classifier result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_greedy_failure_classifier_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/basin_signatures/trajectory_failure/classify_leiden_basin_greedy_failures.py`
- Reusable module additions:
  `classify_branch_greedy_failure_rows` and
  `summarize_greedy_failure_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- The classifier reads existing branch state/path rows plus the seed-control
  comparison. It does not rerun Leiden.
- On the same `c0-s11-r0.001` branch artifact, it classified `90` path rows:
  `37` rows are candidate-directed and QF-recovered at the support gate.
- All `37` candidate-directed rows are
  `branch_unique_candidate_directed_quality_lag`: they move toward the compact
  candidate while tested standard Leiden controls do not, but they still lag
  the best quality control by at least the `1.0` QF material threshold.
- The best row carries all four failure labels:
  `q_greedy_miss`, `progress_greedy_miss`, `closure_compound_miss`, and
  `polish_recovery_miss`. This gives a concrete diagnosis: the useful path is
  not top-ranked by simple greedies, starts from a closure-heavy prefix, crosses
  a raw QF wall, and only becomes usable after bounded polish.

Wall-route family profile result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_wall_route_family_profile_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/route_wall/profile_leiden_basin_wall_route_families.py`
- Reusable module additions:
  `annotate_wall_route_families` and `summarize_wall_route_families` in
  `sciscape/clustering/leiden_basin_search.py`.
- Current `c0-s11-r0.001` branch artifact has one observed candidate-directed
  wall entry: prefix `p9`, wall `1.585400`, `37` candidate-directed rows.
- This is not evidence that the wall is unique. The profile finds `4`
  lower-wall side-route candidates below the support gate, all with wall `0`,
  max support-distance-to-vanilla `0.038961`, and max target progress
  `0.014847`.
- Side-route candidates come from prefixes `p6`, `p8`, and `p10`. They do not
  cross the current support gate, but they are the first places to expand if we
  want to test whether there is a hidden detour around the `p9` wall.

Focused side-route expansion result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_side_route_expansion_field34_cc_c0_v0/`
- Expansion runner:
  `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/evaluate_leiden_basin_target_elbow_polish.py`
  with `--selected-prefix-ranks 6,8,10`, `--max-target-stages 6`, and
  `fixed_cap,guarded_elbow,guarded_backfill,guarded_escalate`.
- Profile script:
  `research/consensus/scripts/leiden_basin/transition_routes/route_wall/profile_leiden_basin_side_route_expansion.py`
- The focused run emitted `96` state/path rows over prefixes `p6`, `p8`, and
  `p10`.
- Result: lower-wall side routes do cross the support gate. There are `57`
  candidate-directed rows, `10` candidate-directed wall entries, and `7`
  distinct candidate-directed wall values. The minimum candidate-directed wall
  is `0.325642`, much lower than the earlier `p9` wall `1.585400`.
- However, none of those gate-crossing side routes recover QF:
  `support_gate_q_recovered_rows = 0`, `quality_loss_rows = 57`, and the best
  candidate-directed `delta_q_vs_start` is still `-0.292613`.
- Interpretation: the `p9` wall is not the only way to reach candidate-like
  support movement. There are lower-wall detours, but the current bounded
  target-growth operator cannot recover their quality. The next search question
  should shift from "can a side route cross the gate?" to "what recovery move
  or context release repairs the low-wall side-route quality debt?"

Pathway debt-area comparison result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_pathway_debt_area_compare_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/tunneling_pathways/profile_leiden_basin_pathway_debt_area.py`
- Reusable module addition:
  `annotate_pathway_debt_area_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope compares the branch target-growth artifact against the focused
  side-route expansion artifact on the same `c0-s11-r0.001` case.
- Result: across `186` path rows, `94` are candidate-directed. The branch
  artifact has `37` candidate-directed rows and all `37` recover QF; the
  side-route expansion has `57` candidate-directed rows and all `57` remain
  quality-loss rows.
- The branch `p9` route has a higher peak wall (`1.585400`) than the best
  side-route wall (`0.325642`), but its candidate-directed debt area is also a
  one-state shortcut (`path_q_debt_area_step = 1.585400`) and it recovers to
  `delta_q_vs_start = +1.190993`.
- The lowest-area side route also has a one-state low wall
  (`path_q_debt_area_step = 0.325642`), but it ends at
  `delta_q_vs_start = -0.325642`; the best side-route final quality is still
  `-0.292613` even after a larger debt area (`0.618255`).
- Interpretation: low peak wall alone is not a better route. The current
  evidence favors "recoverable shortcut" diagnostics: a high wall can be
  acceptable if it is short and recovers, while a low wall is only useful if a
  later recovery mechanism can turn it into a positive-quality path.

Tunneling evidence profile result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_tunneling_evidence_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/tunneling_pathways/profile_leiden_basin_tunneling_evidence.py`
- Reusable module additions:
  `annotate_tunneling_evidence_rows`,
  `trace_tunneling_path_states`, and
  `summarize_tunneling_evidence_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Definition used here: a recoverable tunnel is a candidate-directed path that
  pays nonzero QF debt, keeps debt area explicit, and ends with
  `delta_q_vs_start >= 0`. An unrecovered detour is candidate-directed but
  remains quality-negative at the terminal state.
- Result on the same `186` paths: `37` recoverable tunnel rows, all from
  `branch_target_growth`; `57` unrecovered detour rows, all from
  `side_route_expansion`.
- Best recoverable tunnel:
  `p9 fixed_cap`, wall `1.585400`, debt-area-step `1.585400`,
  final `delta_q_vs_start = +1.190993`, support-distance-to-vanilla
  `0.103093`, target progress `0.035483`.
- Lowest-area detour:
  `p8 fixed_cap`, wall `0.325642`, debt-area-step `0.325642`,
  final `delta_q_vs_start = -0.325642`, support-distance-to-vanilla
  `0.083117`, target progress `0.033200`.
- Trace-level distinction: the p9 tunnel has a wall peak at step `0`, then
  reaches candidate-directed support and QF recovery at step `1`. The p8/p10
  detours gain partial progress before the wall and cross the support gate at
  step `3`, but they never reach a QF-recovered step; additional target growth
  stays in `under_q_debt`.
- Interpretation: the next operator should not merely search for lower walls.
  It should look for an entrance prefix whose wall is followed by a fast
  recovery step. The side-route family is still useful, but only as a target
  for an explicit recovery move after the gate, not as a finished tunnel.

Tunneling path rank result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_tunneling_path_rank_field34_cc_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/tunneling_pathways/rank_leiden_basin_tunneling_paths.py`
- Reusable module additions:
  `rank_tunneling_operator_candidates` and
  `select_tunneling_operator_candidates` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope loads `10` existing field34/cc transition artifacts: branch target
  growth, side-route expansion, c0/c2 target-elbow polish variants, and
  reachability/balanced transition search states.
- Result: `1,292` path rows, `144` recoverable tunnel seed rows, and `119`
  unrecovered detour recovery-target rows. `36` top design rows are selected
  across recoverable seeds, recovery targets, and entrance probes.
- The best efficiency tunnel is no longer p9 but `c0/p4`: in
  `target_elbow_c0_top10`, `p4 fixed_cap` has wall `0.246277`,
  debt-area-step `0.246277`, final `delta_q_vs_start = +0.546858`,
  support-distance-to-vanilla `0.069588`, target progress `0.021129`, and
  operator score `2.587456`.
- The previous p9 family remains useful as a wider-support tunnel:
  support-distance-to-vanilla up to `0.108247` and target progress `0.037785`,
  but with much larger debt area (`1.585400`).
- c2 rows still do not become candidate-directed tunnels under the current
  target-progress margin; they mostly stay stalled or
  `support_gate_no_target_progress`. This keeps c0 as the current operator
  development slice and c2 as a negative/control slice.
- Algorithm implication: split the first tunneling operator into two queues:
  `efficient_tunnel_seed` from p4-like paths, and `wide_tunnel_seed` from
  p9-like paths. Keep p6/p8/p10 as `post_gate_recovery_move` targets rather
  than treating them as finished tunnels.

Post-gate recovery profile result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_field34_cc_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/profile_leiden_basin_post_gate_recovery.py`
- Reusable module additions:
  `annotate_post_gate_recovery_step_rows` and
  `summarize_post_gate_recovery_paths` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope filters the side-route tunneling candidates for `c0-s11-r0.001` and
  prefixes `p6`, `p8`, and `p10`, then expands `96` candidate rows into `432`
  trace steps and `96` path summaries.
- Result: `13` rows show a `near_miss_recovery_trend`, `5` rows show
  `support_deepening_quality_tradeoff`, and `27` rows are
  `post_gate_plateau`. None is a completed QF-recovered tunnel.
- p8 remains the cleanest recovery target: its best near-miss improves from
  gate `delta_q=-0.325642` to `-0.292613` while support rises to `0.088312`.
- p10 is the widest detour: it can reach support `0.113402` and target
  progress `0.040106`, but the widest row trades quality back from the best
  post-gate point.
- p6 is mostly a plateau/control case: several guarded variants enter the gate
  at `delta_q=-0.348459` and then add no material QF or support progress.
- Algorithm implication: the next operator experiment should not add more
  target nodes blindly after the gate. It should branch into a recovery move
  around the best post-gate near-miss state, with p8 first, p10 as the
  support-depth stress case, and p6 as a plateau failure diagnostic.

Post-gate recovery move probe result:

- Artifacts:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p8_v0/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p8_wide_v0/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p8_fullctx_v0/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p10_wide_v0/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p6_wide_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/probe_leiden_basin_post_gate_recovery_moves.py`
- Reusable module additions:
  `build_post_gate_recovery_actions` and
  `classify_post_gate_recovery_move_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope starts from the best `near_miss_recovery_trend` state for p6/p8/p10
  and tests one-step recovery moves over candidate closure, vanilla closure,
  and boundary shell. Each source state is replayed from the original
  prefix/target path before probing, so the move rows remain auditable.
- p8 narrow and wide probes are pure plateau: context-only and candidate
  transplant moves do not change QF/support/progress even after opening up to
  about `99` recovery nodes.
- p8 full-context probe reveals the first useful recovery direction:
  `vanilla_closure_topk:context_only` with `436` selected context nodes improves
  `delta_q_vs_start` from `-0.292613` to `-0.196488`, while retaining support
  (`0.088312 -> 0.090909`) and target progress (`0.035476 -> 0.036623`).
  This is still not a successful tunnel because QF remains negative and the
  mutable set is large (`502` nodes).
- Candidate/boundary transplant at full context is the wrong direction for p8:
  it increases support but collapses QF (`-4.669449` for candidate closure
  transplant, `-2.347544` for boundary transplant).
- p10 wide probe remains plateau at the same near-miss state as p8. p6 wide
  probe has two `q_gain_support_retained` rows, but they only catch up to the
  p8/p10 near-miss state (`-0.295430 -> -0.292613`) and are not a new recovery.
- Algorithm implication: post-gate recovery is not hidden in a small candidate
  closure around the target frontier. The first real recovery signal is
  source-side/vanilla-label context release, but it is currently too expensive.
  The next mechanism question is how to shrink that vanilla-closure release
  from hundreds of nodes to a compact boundary subset without losing the QF
  gain.

Post-gate recovery subset probe result:

- Artifacts:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_subsets_field34_cc_c0_p8_v0/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_subsets_field34_cc_c0_p8_tail_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/probe_leiden_basin_post_gate_recovery_subsets.py`
- Scope replays the p8 post-gate near-miss source state, takes the full
  `vanilla_closure_topk:context_only` move's `436` selected context nodes, and
  probes ranked partial subsets with the same bounded polish seed. The coarse
  run tests `20` prefix/band rows; the tail run tests `18` rows between ranks
  `384` and `436`.
- Result: only the complete `pull_prefix` of all `436` context nodes produces
  `q_gain_support_retained` (`delta_q_vs_start -0.292613 -> -0.196488`,
  support `0.088312 -> 0.090909`). Every smaller prefix through `432` nodes
  and every standalone rank band is plateau.
- Interpretation: the p8 full-context signal is real but non-local. It is not
  carried by a single high-pull boundary band or by the last zero-pull tail
  alone; it appears only when a broad vanilla-label closure is opened together.
  This supports the claim that the probe widened the search region, but it also
  warns against promoting it as an efficient operator. The next shrink attempt
  should search for coherent closure components or label-context gates, not
  merely lower the top-k cutoff.

Post-gate sufficient subset probe result:

- Artifacts:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_sufficient_subset_field34_cc_c0_p8_v1/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_sufficient_subset_field34_cc_c0_p8_labelband32_screen_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/probe_leiden_basin_post_gate_sufficient_subsets.py`
- Scope treats the 436-node full `vanilla_closure_topk:context_only` row as an
  oracle, then greedily removes structured groups while requiring at least
  `70%` of the full QF gain plus source support/progress retention.
- `vanilla_label` grouping is the first useful narrowed scope: it emits `9`
  groups, accepts `36/44` removal trials, and commits `8` removals. The final
  sufficient context is a single vanilla label (`331`) with `209` nodes
  (`47.9%` of the full 436-node context) and exactly preserves the full QF gain
  (`+0.096125`) and support/progress (`0.090909`, `0.036623`).
- A label-internal `vanilla_label_pull_band` screen with 32-node bands shows
  why this is the current stopping point: removing any band inside label `331`
  drops back to plateau (`delta_q` gain `0`), while removing bands from other
  vanilla labels preserves the full gain. Therefore the next search scope
  should be the 209-node source-side vanilla-label gate, not the original
  436-node closure and not a simple pull-rank top-k.
- Runtime note: exhaustive label-band greedy search is expensive because every
  round reruns bounded polish for many bands. Keep later band/component probes
  as one-round screens or add cached/parallel evaluation before scaling.

Post-gate gate trace result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_gate_trace_field34_cc_c0_p8_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/profile_leiden_basin_post_gate_gate_trace.py`
- Scope replays the same p8 source state and compares the narrowed 209-node
  gate against the original 436-node full context using the same recovery seed.
- Result: the 209-node gate and 436-node full context produce identical
  endpoint structure on both affected nodes and sketch nodes
  (`endpoint_distance = 0`). They also match QF gain (`+0.096125`), support
  (`0.090909`), and target progress (`0.036623`).
- The semantic source-to-child change is a single target node, `2890`. The gate
  context nodes themselves do not move; they provide the label context that lets
  node `2890` join the recovered label. The trace records `10` newly internal
  incident edges with total weight `2.186125` and no lost internal edge weight.
- Interpretation: the narrowed operator target should not be "move 209 gate
  nodes". It should be "release a source-side vanilla-label gate so a small
  target node set can attach if CPM permits". The next prototype should score
  candidate target nodes by pull to such gates and run a bounded gate-release
  polish, while keeping this diagnostic-only until seed controls and
  cost-adjusted value are checked.

Post-gate gate attachment candidate score result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_gate_attachment_candidates_field34_cc_c0_p8_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/gate_release/profile_leiden_basin_gate_attachment_candidates.py`
- Scope replays the same p8 source state, uses the 209-node sufficient gate as
  the target context, and scores all `244` target nodes from source-state graph
  features before any new recovery polish. The moved node label is overlaid
  from the existing gate trace only for evaluation.
- Result: observed moved node `2890` is not the top raw gate-pull node
  (`rank 10`, pull `2.186125`), but it is the top node by gate attachment
  margin over its current source label (`rank 1`, margin `2.186125`, current
  source-label pull `0`, source-label size `1`) and by gate share over current
  source label (`rank 1`). The mean consensus rank places it `4/244`.
- Interpretation: total pull to the gate is too broad as a selector. The more
  mechanism-like signal is "gate pull minus current source-label pull",
  especially for weakly anchored singleton/small target nodes. The next
  operator prototype should therefore gate on attachment margin and current
  source-label anchoring, not only on vanilla-label membership or raw pull.

Gate-release operator probe result:

- Artifacts:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_gate_release_operator_probe_field34_cc_c0_p8_v0/`
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_gate_release_operator_probe_field34_cc_c0_p8_manual_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/gate_release/run_leiden_basin_gate_release_operator_probe.py`
- Scope uses the attachment-score table to select target nodes, opens selected
  nodes and/or the 209-node gate as mutable context, and runs bounded polish
  without candidate transplant.
- Result: the best row is `target_only`, margin top-2 (`2890,7325`), with
  only `68` mutable nodes and `2` context nodes. It improves the p8 source
  state from `delta_q_vs_start=-0.292613` to `-0.159983`, a QF gain of
  `+0.132629`, with support `0.093506` and target progress `0.037368`. This
  beats the 209-node gate-only control (`+0.096125`, `275` mutable nodes).
- Manual controls separate the mechanism: `2890` alone reproduces the old
  gate-only gain (`+0.096125`), `7325` alone gives a smaller gain
  (`+0.046504`), and `2890,7325` together gives the larger `+0.132629` without
  opening the 209-node gate. Adding the gate to these target nodes does not
  improve the endpoint.
- Interpretation: the current best mechanism is no longer a broad gate-release
  operator. It is a compact target-mutable tunneling step selected by
  attachment margin. The 209-node gate was useful diagnostically because it
  exposed `2890`, but the cheaper operator should first test small target sets
  such as the margin top-2, then use gates only as fallback context.

Gate-release seed-control result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_gate_release_operator_probe_field34_cc_c0_p8_seed5_v0/`
- Scope reruns the compact p8 gate-release probe over recovery seeds
  `21003,21004,21005,21006,21007` for gate-only, margin top-2, and manual
  `2890`/`7325` controls.
- Result: every tested seed gives the same endpoint for each action family.
  Margin top-2 target-only (`2890,7325`) remains the best compact row in all
  `5/5` seeds (`q_gain_mean=min=max=+0.132629`, support `0.093506`,
  progress `0.037368`, mutable `68`, context `2`). Gate-only remains
  `+0.096125` with `275` mutable nodes in all seeds.
- Manual controls are also seed-stable: `2890` target-only reproduces the
  gate-only endpoint, `7325` target-only gives only `+0.046504`, and
  `2890,7325` target-only gives the stronger endpoint. Gate+target variants
  with either `2890` or `7325` also reach the stronger endpoint, showing that
  the broad gate can trigger the same pair-level transition indirectly.
- Interpretation: this clears the first seed-control check for this p8 source
  state. It does not yet prove a general operator; the next required check is
  cross-source/cross-prefix validation where the attachment-margin selector is
  recomputed before testing top-k target-only tunneling.

Attachment-margin cross-prefix smoke result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_cross_prefix_field34_cc_c0_p6_p8_p10_v1/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/attachment_margin/run_leiden_basin_attachment_margin_cross_prefix_probe.py`
- Scope replays the selected p6 wide, p8 full-context, and p10 wide
  post-gate source states, recomputes target-node attachment margin against
  each source's selected recovery context, and probes compact target-only
  mutable sets (`k=1,2,4`). The default `--recovery-seed 0` reuses each source
  row's original polish seed (`21000 + source_recovery_index`).
- Result: all three sources have a compact target-only row that beats the
  context-only control on QF gain with lower mutable cost. p6 wide selects
  node `7325` at top-1 and improves by `+0.046504` (`-0.292613 -> -0.246108`,
  support `0.090909`, progress `0.036227`, mutable `70` versus control
  mutable `173` with no QF gain). p8 full-context reproduces the stronger
  top-2 pair `2890,7325` with `+0.132629` (`-0.292613 -> -0.159983`, support
  `0.093506`, progress `0.037368`, mutable `68` versus context-control
  `+0.096125` and mutable `502`). p10 wide again selects node `7325` at
  top-1 and improves by `+0.046504` (`-0.295430 -> -0.248925`, support
  `0.085714`, progress `0.033964`, mutable `66` versus control mutable `163`
  with no QF gain).
- Interpretation: attachment-margin compact target-only tunneling is not only
  a p8 full-context artifact. It transfers to neighboring p6/p10 source states,
  but those states currently recover only the partial `7325` step rather than
  the full p8 `2890,7325` step. This keeps the mechanism alive as an operator
  candidate, but still diagnostic: the next comparison must broaden cases and
  include vanilla seed/iteration controls before promotion.

Attachment-margin seed/iteration control result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_controls_field34_cc_c0_p6_p8_p10_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/attachment_margin/evaluate_leiden_basin_attachment_margin_controls.py`
- Scope compares the best compact target-only row from each p6/p8/p10 source
  against same-case standard Leiden controls with seeds `11,42,73,101,137`,
  randomness values `0.001,0.01`, and iteration budgets `1,10,convergence`.
  The comparison uses the same attachment-margin case-context loader so the
  candidate/vanilla references match the operator probe.
- Result: no tested standard Leiden control has positive target progress
  toward the compact candidate (`candidate_directed_control_rows=0` for all
  three sources). The compact operator keeps positive target progress
  (`0.033964` to `0.037368`) with only `66` to `70` mutable nodes. However,
  it is not a QF win. The best broad control is seed `137`, randomness `0.01`,
  `n=10/convergence`, with quality `18413.798906` and negative target progress
  `-0.202961`; the compact rows lag this control by `-2.553622` to
  `-2.642564` QF. Against same-randomness `0.001` controls, the QF lag is much
  smaller (`-0.159983` to `-0.248925`) but still negative.
- Interpretation: this clears a mechanism-only test, not the two-objective
  test. Attachment-margin tunneling finds a candidate-directed local move that
  ordinary seed/iteration controls did not reproduce in this slice, but it
  still fails the "better partition" objective versus the best broad vanilla
  control. The next operator work should focus on reducing the QF lag after the
  directed step, not on declaring the current compact step successful.

Attachment-margin stage2 recovery result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_stage2_recovery_field34_cc_c0_p6_p8_p10_v2/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/post_gate_recovery/run_leiden_basin_attachment_margin_stage2_recovery.py`
- Scope starts from each best compact target-only row, then opens context
  around the selected target nodes using candidate-label, current-label,
  vanilla-label, and boundary-shell families with multipliers `4,16,64`.
  It tests both `context_only` and candidate-label transplant proposals, and
  records pre-polish changed-node/QF deltas to separate no-op proposals from
  polish reversion.
- Result: all three sources remain `stage2_no_recovery`. Context-only rows are
  exact no-ops. Candidate/current/vanilla label transplants are also no-ops for
  p6/p10 and for the p8 label families. The only nonzero proposals are p8
  boundary-shell transplants: 1 changed node at multiplier `4` with pre-polish
  QF debt `-4.511790`, and 6 changed nodes at multiplier `16` with debt
  `-51.799403`; bounded polish reverts both to the stage1 endpoint
  (`final_changed_node_count=0`, no QF/progress gain).
- Interpretation: the QF lag after compact tunneling is not recoverable by
  simply opening or candidate-transplanting local context around the selected
  target nodes. The next mechanism search should shift from sequential
  "target move then local recovery" to joint bundle selection before polish:
  find companion target/context units that must be activated together, then
  evaluate the bundle as one tunneling proposal.

Attachment-margin joint bundle result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_joint_bundle_field34_cc_c0_p6_p8_p10_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/joint_bundle/run_leiden_basin_attachment_margin_joint_bundle_probe.py`
- Scope starts from the post-gate source state, then activates target top-k
  nodes and companion context together before bounded polish. It tests target
  `k=1,2,4,8`, context families `none`, `source_context`,
  `candidate_label`, `current_label`, and `boundary_shell`, context
  multipliers `8,32`, and both `joint_mutable` and
  `candidate_bundle_transplant` moves.
- Result: this is the first nontrivial positive joint-bundle signal. p6/p10
  remain `joint_directed_quality_lag`, with best QF gains `+0.132629` over the
  source state but still QF lag versus same-randomness controls
  (`-0.159983` to `-0.162800`). p8 has `12` rows that beat the same-randomness
  control while retaining positive target progress. The highest-QF p8 row is
  target top-8 plus `256` candidate-label context nodes with
  `candidate_bundle_transplant`: `delta_q_vs_vanilla=+0.644965`, target
  progress `0.009692`, mutable `327`, but it still lags the best broad control
  by `-1.748673` QF. A more compact p8 signal is target top-4
  (`2050,2890,7325,9545`) plus current-label context top-10 with
  `joint_mutable`: `delta_q_vs_vanilla=+0.037027`, target progress
  `0.042027`, mutable `79`, and same-randomness control margin `+0.037027`;
  it still lags the best broad control by `-2.356611`.
- Interpretation: the failed sequential stage2 diagnosis was correct but
  incomplete. Local recovery after the target move is not enough; joint
  activation can open a different transition, at least for p8. This is still
  not a default policy because the broad vanilla control remains higher-QF and
  the best high-QF row is large. The next probe should focus on why p8 current
  label joint-mutability moves `3799` nodes from only `14` bundle nodes, and
  whether that transition can be made smaller or reproduced in p6/p10.

Attachment-margin joint bundle focused replay result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_joint_bundle_replay_field34_cc_c0_p8_current_label_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/joint_bundle/explain_leiden_basin_attachment_margin_joint_bundle_replay.py`
- Scope replays the compact p8/current-label bundle (`target_k=4`,
  selected targets `2050,2890,7325,9545`, context top-10, bundle size `14`)
  with the original polish seeds for both `joint_mutable` and
  `candidate_bundle_transplant`.
- Result: the earlier `3799` changed-node count is mostly exact-label
  namespace accounting, not a `3799`-node basin move. The two variants land on
  the same endpoint under label-invariant comparison:
  `aligned_changed_between_children=0`, `endpoint_distance_between_children=0`,
  while exact labels differ on `3795` nodes. Versus the source state, both
  variants have the same label-invariant changed support core of only `6`
  nodes: `2050,2890,5260,7325,9545,9609`. The replay still retains the earlier
  quality signal (`delta_q_vs_vanilla=+0.037027`,
  `delta_q_gain_vs_source=+0.329640`).
- Interpretation: this closes one false lead. The compact p8 signal is real,
  but the "14 nodes caused 3799 nodes to move" phrasing was misleading. The
  next operator-design question should focus on the 6-node aligned core and its
  two roles: four selected target nodes, one context node (`9609`), and one
  already-source-mutable off-bundle node (`5260`) pulled in by polish. Exact
  changed-node counts should be treated as implementation-level label
  accounting; basin-level claims should use aligned support changes and
  endpoint distance.

Basin evaluation metric audit:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_evaluation_metric_audit_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/evidence_panels/audits/audit_leiden_basin_evaluation_metrics.py`
- Scope scans CSV artifacts under the combined field30/field34 basin-evidence
  root and flags files whose interpretation may depend on exact label equality.
- Result over `532` CSV files after alias/backfill cleanup, the five targeted
  recomputations, the aligned-core frontier replay, and the compact boundary
  and handle-subset/stability/selector/source-screen probes: `341` expose label-invariant
  support, alignment-error, or endpoint metrics, `186` are not basin-metric artifacts,
  and `5` need exact-column relabel/reinterpretation.
  There is no remaining rerun/backfill-required group in the current audit.
  The previous high-risk signature/vanilla rows were resolved by treating
  `p5_changed_nodes_vs_baseline` as best-partner alignment error, pairing it
  with changed-support aliases, and regenerating `signature_review` with
  explicit `alignment_error` and `aligned_changed_support` columns.
- Targeted recomputations completed for the five audit-listed operator
  artifacts. New rows now carry explicit aligned/exact companions:
  joint-bundle `176` rows, stage2 recovery `51` rows, gate-release v0 `61`
  rows, gate-release seed5 `45` rows, and gate-release manual `41` rows. The
  remaining `5` audit entries are compatibility aliases, not recomputation
  blockers: generic `changed_node_count` columns are still present for old
  readers, but aligned/exact columns are now present in the same files.
- Interpretation: not every basin result must be recomputed from scratch.
  QF, support distance, endpoint distance, and candidate/vanilla progress
  claims remain usable. Any claim based on raw `changed_node_count`,
  `changed_nodes`, or exact membership equality must be downgraded unless it is
  backed by aligned support, alignment-error, changed-support, or endpoint
  metrics.

Recomputed operator metric review:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_evaluation_metric_audit_v0/recomputed_operator_metric_review/`
- Script:
  `research/consensus/scripts/leiden_basin/basin_signatures/endpoint_flips/summarize_leiden_basin_recomputed_operator_metrics.py`
- Scope summarizes the five recomputed operator artifacts using aligned-change,
  exact-only, endpoint, and QF-gain columns.
- Result: stage2 recovery remains closed. Across `48` metric rows it has
  `max_final_aligned_changed=0`, `max_quality_gain=0`, and
  `stage2_no_recovery` for all action rows. Gate-release remains a tiny local
  repair, not a basin-transition mechanism: v0/seed5/manual have
  `max_final_aligned_changed=2`, `max_endpoint_distance=0`, and identical
  best gain `+0.132629` versus the source. Joint-bundle is the only remaining
  positive QF signal: `max_quality_gain=+0.937578`,
  `max_final_aligned_changed=35`, mean final aligned change `2.37`, but
  `max_final_exact_only_changed=7803`, confirming that large exact changes are
  mostly label namespace accounting.
- Interpretation: do not expand stage2 recovery or gate-release as the next
  main operator family. The next operator should start from the joint-bundle
  compact aligned core, especially the p8 candidate-label/context row
  (`target_k=8`, `candidate_label`, `context_multiplier=32`) and the smaller
  current-label/boundary-shell rows, then search boundary context that improves
  QF without broad exact-label churn.

Joint-bundle aligned-core frontier result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/joint_bundle_aligned_core_frontier_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/joint_bundle/profile_leiden_basin_joint_bundle_aligned_core_frontier.py`
- Scope replays the six positive joint-bundle configs from the recomputed
  operator metric review and aggregates their label-invariant aligned cores.
- Result: the stable p8 core is small and structured. `9545` and `9609` change
  in `6/6` replay configs; `2050`, `2890`, and `7325` change in `5/6`; `5260`
  changes in `5/5` configs where it is present but is not a selected target.
  It is a source-mutable boundary/core node with baseline/candidate label
  `266`, vanilla label `261`, and nonzero pull to the bundle. The strongest
  QF config (`target_k=8`, `candidate_label`, `context_multiplier=32`,
  candidate-bundle transplant) expands the aligned core to `28` nodes and
  gains `+0.937578`, but many of those extra nodes appear only in that wide
  candidate-label context. The compact current-label/boundary-shell configs
  repeatedly land on the six-node core
  `2050,2890,5260,7325,9545,9609` with `+0.329640` QF gain.
- Interpretation: the next operator should treat the direct target handles
  (`2050,2890,7325,9545,9609`) and the non-target boundary core (`5260`) as
  separate classes. A useful operator should explicitly price boundary-core
  inclusion and only then expand to candidate-label context; it should not
  blindly open the full candidate label or optimize exact-only churn.

Aligned-core boundary operator probe:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_aligned_core_boundary_operator_field34_cc_c0_p8_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/basin_signatures/local_modes/run_leiden_basin_aligned_core_boundary_operator_probe.py`
- Reusable module additions:
  `select_aligned_core_boundary_nodes` and
  `build_aligned_core_boundary_plan_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope tests five compact p8 plans: target handles only, target plus boundary
  core `5260`, target plus the stable context-core nodes, and target plus
  candidate-label context caps `8` and `32`. Each plan is run with
  `joint_mutable` and `candidate_bundle_transplant`.
- Result: the best candidate-transplant row is already the target-only plan:
  direct handles `2050,2890,7325,9545,9609` produce the same six-node aligned
  endpoint `2050,2890,5260,7325,9545,9609` with
  `operator_delta_q_gain_vs_source=+0.329640`,
  `state_delta_q_vs_vanilla=+0.037027`, and
  `state_target_progress_from_vanilla=0.042027`. Adding boundary core `5260`,
  context-core nodes, or candidate-label caps `8/32` gives zero additional QF
  over target-only and only lowers QF per bundle node. `joint_mutable` rows
  reach only two aligned nodes (`2890,7325`) with `+0.132629` QF gain and stay
  below same-randomness control.
- Interpretation: for this p8 source, the useful compact action is not
  "open more boundary context". Forced candidate transplant on the direct
  target handles is sufficient, and `5260` is an induced polish response
  because it already overlaps the source mutable set. The next mechanism test
  should minimize and generalize the forced direct-handle transplant, then ask
  which handle subset is sufficient across seeds/cases.

Aligned-core handle subset sufficiency probe:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_aligned_core_handle_subset_field34_cc_c0_p8_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/aligned_core/run_leiden_basin_aligned_core_handle_subset_probe.py`
- Reusable module addition:
  `build_aligned_core_handle_subset_plan_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope exhaustively tests all `31` nonempty subsets of the five direct
  handles with context closed, using candidate-label transplant only and a
  fixed polish seed offset so subset effects are isolated.
- Result: the minimal sufficient subset is size `4`:
  `2890,7325,9545,9609`. It recovers the required six-node aligned core
  `2050,2890,5260,7325,9545,9609` and exactly matches the full five-handle
  quality within tolerance: `operator_delta_q_gain_vs_source=+0.329640`,
  `state_delta_q_vs_vanilla=+0.037027`, and
  `quality_gap_vs_full_handle_set=0`. The full five-handle row is the only
  other sufficient row. The best three-handle near miss
  `2890,9545,9609` recovers `5/6` required nodes and is only `-0.016504` QF
  behind the full handle set. The pair `9545,9609` already recovers
  `2050,5260,9545,9609` with `+0.237011` QF gain, while single-node
  `2890` and `7325` create smaller independent gains.
- Interpretation: `2050` is not required as a forced handle in this source; it
  can be induced by forcing `9545,9609` together. The compact mechanism appears
  factorized into a `9545/9609 -> 2050/5260` induced block plus the independent
  `2890` and `7325` moves. The next test should check whether this sufficient
  subset remains stable across polish seeds and nearby source cases before it
  is treated as an operator rule.

Aligned-core handle stability probe:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_aligned_core_handle_stability_field34_cc_c0_p6_p8_p10_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/aligned_core/run_leiden_basin_aligned_core_handle_stability_probe.py`
- Scope replays selected handle subsets across `p6_wide`, `p8_fullctx`, and
  `p10_wide`, with polish seed offsets `2000,3000,4000`. The selected subsets
  are the minimal sufficient set, the full five-handle set, and the strongest
  near misses from the subset sweep.
- Result: the minimal subset `2890,7325,9545,9609` is stable in all `9`
  evaluations, exactly matching the full five-handle set. Both recover the
  required aligned core `2050,2890,5260,7325,9545,9609`, have zero QF gap to
  each other within every source/seed group, and produce stable positive QF:
  `+0.329640` versus source in all rows. The near-miss subsets
  `2890,9545,9609`, `2050,2890,9545,9609`, and `7325,9545,9609` are stable
  partial cores: each recovers only `5/6` required nodes across all `9`
  evaluations, with QF gaps `-0.016504` or `-0.066125`.
- Interpretation: this is stronger than a single-seed artifact. Within the
  current c0/p6-p8-p10 slice, `2050` remains an induced response and the
  sufficient forced set is stable. The next uncertainty is not polish-seed
  fragility; it is whether the same handle-selection rule can be discovered
  from local features rather than from post-hoc knowledge of the answer, and
  whether it generalizes beyond this c0 slice.

Aligned-core handle selector replay:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_aligned_core_handle_selector_field34_cc_c0_p8_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/aligned_core/run_leiden_basin_aligned_core_handle_selector_probe.py`
- Reusable module additions:
  `score_aligned_core_handle_nodes` and
  `build_aligned_core_handle_selector_plan_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope does not rerun Leiden. It grades top-k handle selectors against the
  exhaustive subset table and joins the existing stability summary when a
  selected subset was replayed across source/seed rows.
- Result: all four selector policies reach the minimal sufficient subset at
  `k=4`, but the important signal is that two local-graph-proxy policies do so
  without replay-derived frontier counts. `context_pull` and
  `mutable_penalized_context_pull` both select `2890,7325,9545,9609` at `k=4`,
  matching the minimal sufficient subset with `+0.329640` QF gain versus the
  source, zero QF gap to the full handle set, and stability fraction `1.0`.
  The first three selected nodes form the stable near miss
  `2890,9545,9609`, recovering `5/6` required core nodes with a `-0.016504`
  QF gap.
- Interpretation: this is the first evidence that the compact sufficient
  forced set may be selectable from local edge/context features rather than
  only from post-hoc endpoint knowledge. It is still a replay diagnostic over
  one c0 frontier; the next test must rebuild the selector from local graph
  features across additional cases before promoting it to an operator rule.

Local handle selector probe:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_handle_selector_field34_cc_c0_p6_p8_p10_coherent_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/aligned_core/run_leiden_basin_local_handle_selector_probe.py`
- Reusable module additions:
  `score_local_handle_candidates` and
  `build_local_handle_selector_plan_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope uses source-local attachment/gate-pull score rows as selector input,
  not the p8 aligned-core frontier. The frontier-derived six-node core is used
  only as an evaluation target for the current c0 slice. The default run keeps
  the probe lean: source cases `p6_wide,p8_fullctx,p10_wide`, selector
  `candidate_label_margin_coherent`, and `k=1..4`.
- Failed baseline: raw `gate_pull` and `non_source_gate_pull` were not enough.
  `gate_pull` mostly selected already source-mutable decoys and made no
  material movement. `non_source_gate_pull` recovered the p8 sufficient subset
  at `k=4`, but p6/p10 selected candidate-label `1184` decoys first and only
  recovered node `7325` with `+0.046504` QF gain.
- Result: `candidate_label_margin_coherent` first scores non-source candidate
  label groups by top-4 local margin, then ranks nodes within the best group.
  It selects candidate label `1090` in all three source cases and reaches
  `2890,7325,9545,9609` at `k=4` in all three. Each source recovers the
  required aligned core `2050,2890,5260,7325,9545,9609` with
  `operator_delta_q_gain_vs_source=+0.329640`; p6/p8 have
  `state_delta_q_vs_vanilla=+0.037027`, and p10 has `+0.034210`.
- Interpretation: the selector should be label-coherent before it is
  node-greedy. The useful mechanism is not simply "pick the largest pull";
  it is "find a candidate-label group whose local top-k margin is coherent,
  then tunnel with a compact forced transplant inside that group." This is
  still same-c0 evidence and must be tested on additional pair/candidate
  slices before it can be treated as a Dongdaemun operator rule.

Local handle selector c2 smoke:

- Artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_field34_cc_c2_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c2_p8_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_cross_prefix_field34_cc_c2_p8_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_handle_selector_field34_cc_c2_p8_top10_coherent_v0/`
- Implementation update: `run_leiden_basin_local_handle_selector_probe.py`
  now accepts `--evaluation-core-mode none`, so cross-case smoke probes do not
  accidentally score c2 rows against the c0 aligned-core answer sheet.
- Scope: `c2-s11-r0`, `target_elbow_c2_top10`, prefix `p8`, selector
  `candidate_label_margin_coherent`, `k=1..4`, required-core evaluation off.
  A full `p1..p10` post-gate profile found only one near-miss row, again at
  `p8`.
- Result: this is a weak generalization test, not a positive selector
  validation. The source row is already QF-positive after the gate
  (`state_delta_q_vs_start=+0.982482`, support distance to vanilla `0.056235`).
  The source-move probe emits four `q_recovered_support_retained` rows with
  zero additional QF gain, and the local selector has only one positive
  coherent candidate-label handle (`8492`, label `194`). The evaluated handle
  adds no material gain (`operator_delta_q_gain_vs_source=0.0`) and is labeled
  `local_no_material_gain`.
- Interpretation: c2 remains a negative/control slice for the local selector
  question. It checks that the pipeline can run without frontier leakage, but
  it does not test the multi-handle label-coherence mechanism seen in c0. The
  next selector validation should choose cases with at least several
  non-source positive-margin candidates and an unrecovered or low-gain
  post-gate source; otherwise the run is only a plumbing smoke test.

Branch-based c2 non-c0 follow-up:

- Artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_branch_target_growth_field34_cc_c2_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_tunneling_path_rank_field34_cc_v1/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_field34_cc_c2_branch_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c2_p6_branch_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_cross_prefix_field34_cc_c2_p6_branch_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_selector_readiness_field34_cc_c2_p6_branch_v0/`
- Implementation update: `profile_leiden_basin_post_gate_recovery.py` accepts
  `--state-rows-filename`, so non-target-elbow state artifacts such as
  `transition_search_states.csv` and `branch_target_growth_states.csv` can reuse
  the same post-gate recovery profiler. `probe_leiden_basin_post_gate_recovery_moves.py`
  and `run_leiden_basin_attachment_margin_cross_prefix_probe.py` can now replay
  a recorded `branch_target_growth` selected-node sequence for diagnostic source
  reconstruction.
- Result: c2 branch target growth emits `98` path rows and post-gate profiling
  finds `4` `near_miss_recovery_trend` rows plus `1` plateau. The best p6
  branch source has post-gate gain `+0.080258`, but the source-move probe is
  already QF/support retained (`source_delta_q=+0.982482`, support `0.056235`)
  and all four recovery moves are `q_recovered_support_retained`.
- Selector-readiness result: the c2 p6 branch attachment probe does not expose
  a label-coherent selector test. It is classified as
  `already_recovered_control`: source `delta_q=+0.982482`, support `0.056235`,
  and the best compact attachment handle has zero QF gain over source. It still
  has only one positive non-source margin handle (`8492`, candidate label
  `194`). Branch replay reconstructs the recorded selected-node sequence rather
  than loading persisted memberships, so this remains diagnostic source
  reconstruction rather than an exact saved-membership replay.
- Interpretation: this is useful negative evidence. c2 can produce branch
  near-miss rows, but the local attachment frontier is sparse and does not
  reproduce the c0 label-competition or c0 label-completion mechanism. The next
  non-c0 search should not simply rerun c2 p6/p8; it should first screen for
  multiple positive-margin handles before spending selector replay budget.

Selector source screen:

- Artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_selector_source_screen_field34_cc_c2_branch_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_selector_source_screen_field34_cc_c2_branch_recovery_context_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_selector_source_screen_field34_cc_c0_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_selector_source_screen_field34_cc_c0_top10_recovery_context_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/selector_sources/screen_leiden_basin_selector_sources.py`
- Purpose: rebuild post-gate source states and attachment-margin rows, then
  classify whether a source is worth expensive local-selector replay. This is
  a source-screening gate, not an operator acceptance policy.
- Result: `path_action_union` is too conservative for source screening. It
  correctly screens out the c2 branch controls (`5/5` `already_recovered_control`)
  but also misses the known c0 p5/p7-style positives (`5/5` `too_few_handles`).
  The useful mode is `recovery_contexts`, which builds the same bounded recovery
  contexts used by the recovery profiler but does not run polish trials.
- Positive-control check: c0 top10 with `recovery_contexts` screens `5`
  selected post-gate sources into `15` source/context variants, finds `5`
  `selector_test_ready` rows and `10` `coherent_label_completion_probe` rows.
  This recovers the known c0 selector opportunities before replay.
- Negative-control check: c2 branch with `recovery_contexts` also screens `5`
  selected post-gate sources into `15` variants, but all `15` are
  `already_recovered_control`; `ready_count=0` and `label_completion_count=0`.
- Interpretation: use `recovery_contexts` as the default budget gate before
  running local-selector replay. Only `selector_test_ready` or
  `coherent_label_completion_probe` should normally trigger replay. A control
  row remains useful diagnostically, but it does not validate the selector.

Selector source screen batch:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_selector_source_screen_batch_field34_cc_non_c0_v0/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/selector_sources/run_leiden_basin_selector_source_screen_batch.py`
- Scope discovers existing `field34_cc` post-gate recovery artifacts, excludes
  c0 by default, and applies the `recovery_contexts` selector-source screen
  sequentially. It writes artifact-level, source-level, and readiness-level
  batch rows without aggregating the large score rows.
- Result: all currently available non-c0 post-gate artifacts are c2 variants.
  The batch screened `6` post-gate artifacts, `8` selected post-gate sources,
  and `24` source/context variants. Every variant is
  `already_recovered_control`; `ready_count=0` and
  `label_completion_count=0`.
- Interpretation: the existing c2 family is exhausted for selector replay.
  The next useful step is to generate a fresh non-c0 post-gate source slice
  and run this batch gate first. If the batch still has zero
  `selector_test_ready` or `coherent_label_completion_probe` rows, do not spend
  local-selector replay budget on that slice.

Local selector readiness profile:

- Artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_selector_readiness_field34_cc_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_selector_readiness_field34_cc_v1/`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/selector_signals/profile_leiden_basin_selector_readiness.py`
- Reusable module addition:
  `summarize_local_selector_readiness_rows` in
  `sciscape/clustering/leiden_basin_search.py`.
- Scope scans existing `attachment_margin_cross_prefix` score tables and
  classifies whether each source case can actually test a local selector. A
  `selector_test_ready` row has multiple non-source positive-margin handles
  across multiple candidate labels. A `coherent_label_completion_probe` row
  may have only one positive anchor, but its candidate-label group has enough
  nodes to test same-label completion.
- Result: the v1 inventory has `5` attachment artifacts and `10` source rows.
  It reports `6` `selector_test_ready` rows, `2`
  `coherent_label_completion_probe` rows, and `2`
  `already_recovered_control` rows. The `6` ready rows are the same c0
  `p6_wide`, `p8_fullctx`, and `p10_wide` sources duplicated across v0/v1
  attachment artifacts. The two p5/p7 top10 rows are not label-competition
  tests: each has one positive anchor in candidate label `1090`, but the full
  non-source same-label group has four nodes. The new c2 p6 branch row is the
  second `already_recovered_control` case.
- Follow-up artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_field34_cc_c0_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p5_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_post_gate_recovery_moves_field34_cc_c0_p7_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_attachment_margin_cross_prefix_field34_cc_c0_p5_p7_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_selector_readiness_field34_cc_c0_p5_p7_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_local_handle_selector_field34_cc_c0_p5_p7_top10_coherent_v0/`
- Result on p5/p7: post-gate recovery moves plateau under the existing context
  policies, but attachment margin identifies a compact top-1 anchor `7325`
  with `+0.046504` QF gain. After fixing the coherent selector to score
  positive label anchors before negative fillers, `candidate_label_margin_coherent`
  selects label `1090` and completes `2890,7325,9545,9609` at `k=4` for both
  p5 and p7. Both rows recover the required aligned core
  `2050,2890,5260,7325,9545,9609` with
  `operator_delta_q_gain_vs_source=+0.329640` and
  `state_delta_q_vs_vanilla=+0.034210`.
- Interpretation: this adds a second c0 mechanism shape. The original
  p6/p8/p10 evidence is label competition; p5/p7 is label completion from a
  single positive anchor plus same-label fillers that look locally negative.
  The c2 branch follow-up is negative: it produces a near-miss source family,
  but the selector-facing source is already recovered and has only one
  non-source positive-margin handle. Cross-pair generality is still unproven,
  but the screening criterion is now clearer: first find a non-c0 source with
  multiple positive-margin handles or a non-source same-label completion group,
  then spend replay budget.

Closure shrink pilot result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_operator_pilot_field34_cc/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/transition_operators/run_leiden_basin_transition_closure_operator_pilot.py`
- The default run emitted `220` rows: `20` controls and `200` closure rows
  over fresh-label and candidate-nearest target strategies.
- Candidate-nearest raw shrink reduced support burden more than fresh raw
  shrink: median reduction `13` nodes versus `0`, max reduction `37`.
- The best closure row reached `delta_vs_vanilla = 7.389` and
  `delta_vs_control_extra = 4.378`, but its support-distance-to-candidate
  remained `0.913`; this is a vanilla-near quality win, not a candidate-like
  basin transition.
- Diagnostic labels found no `quality_win_support_shift` rows. Positive rows
  are currently `quality_win_same_basin`, so direct-node-only shrink should not
  be promoted as a mechanism change.

Bounded closure-context release result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_context_release_pilot_field34_cc/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/closure_context/run_leiden_basin_transition_closure_context_release_pilot.py`
- The default run selected `8` positive direct-shrink prefixes across `4`
  pairs and emitted `16` context-release rows plus controls.
- Context release used the `outside_support` closure pool with
  `context_budget_per_label = 16`.
- Best quality remained positive (`delta_vs_vanilla = 7.389`,
  `delta_vs_control_extra = 4.378`), but maximum
  support-distance-to-vanilla was only `0.083`, below the `0.1`
  support-shift gate.
- Diagnostic labels again found no `quality_win_support_shift` rows. This
  makes the shrink-refinement family look vanilla-near under the current
  bounded definitions.

Label-internal repair result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_label_internal_repair_pilot_field34_cc/`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/closure_context/run_leiden_basin_transition_label_internal_repair_pilot.py`
- The default run selected `10` high-ratio candidate-label rows across `5`
  pairs and emitted `40` total rows, including `20` repair rows.
- Best repair row: `delta_vs_candidate = 3.323`,
  `delta_vs_control_extra = 0.0296`, but `delta_vs_vanilla = -3.410`.
- Maximum support-distance-to-candidate was only `0.057`, so the repair stays
  candidate-near rather than producing a distinct support transition.
- Diagnostic labels found no `quality_win_support_shift` rows. Raw repair rows
  were mostly `quality_loss`, while polish rows were all
  `seed_control_dominates`.
- Interpretation: high-ratio candidate labels can expose small candidate-near
  local gains, but label-internal repair does not provide a material
  quality/cost mechanism over vanilla or seed controls.

Basin profiling v0 result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_ordered_flip_frontier_field34_cc_v0/`
- Scripts:
  `research/consensus/scripts/leiden_basin/basin_signatures/local_modes/leiden_basin_profile.py` and
  `research/consensus/scripts/leiden_basin/basin_signatures/endpoint_flips/profile_leiden_basin_ordered_flips.py`
- Scope is deliberately narrow: `c2-s11-r0`, direction `V -> C`,
  `label_intersection_block` units, raw flips only, beam width `5`, max steps
  `10`, scoring policies `q_first`, `progress_first`, and `balanced`.
- Existing node/group minimal pathway for the same pair has
  `node_edit_lower_bound = 376`, `quality_barrier = 56.231`, and
  `final_support_distance_to_candidate = 0.824`.
- Block v0 produced `120` units and `15,885` frontier rows. The first-step
  policies diverged immediately:
  - `q_first`: `+0.114` immediate QF, but only `0.008` candidate-progress
    fraction.
  - `progress_first`: `0.109` candidate-progress fraction, but `-14.231`
    immediate QF.
  - `balanced`: near-zero progress with tiny debt (`-0.00054` QF).
- After 10 raw block flips, `q_first` keeps zero raw barrier and positive QF
  but remains almost vanilla-near (`support_distance_to_candidate = 0.919`);
  `progress_first` reaches more progress (`support_distance_to_candidate =
  0.882`) but pays a raw barrier of `25.71`.
- Interpretation: this validates the profiling question. QF-improving blocks
  and candidate-progress blocks are not the same objects in the first sanity
  case, so any transition operator must explicitly decide whether it is buying
  basin progress with QF debt or staying in a local vanilla-near improvement.

Basin profiling v1 batch result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_ordered_flip_frontier_field34_cc_v1_cases/`
- Script:
  `research/consensus/scripts/leiden_basin/basin_signatures/endpoint_flips/profile_leiden_basin_ordered_flips_batch.py`
- Scope keeps v0 fixed: direction `V -> C`, unit type
  `label_intersection_block`, raw flips only, beam width `5`, max steps `10`.
  Only the target cases expand, to priority rows `c2-s11-r0`,
  `c0-s11-r0.001`, and `c2-s42-r0`.
- The batch emitted `531` unit rows, `71,253` frontier rows, and `450` beam
  rows.
- All three cases show first-step divergence between `q_first` and
  `progress_first`.
- `q_first` keeps zero raw barrier and positive final QF, but remains
  vanilla-near: final support-distance-to-candidate is `0.631`, `0.919`, and
  `0.965`.
- `progress_first` moves more toward the candidate side, but pays raw barriers
  of `8.22`, `25.71`, and `66.04`.
- Interpretation: the v0 tradeoff is not a one-case artifact. Current evidence
  says candidate-progress blocks are expensive raw edits, while cheap
  QF-positive blocks mostly explain local vanilla-near improvements. The next
  local-operator step needs an explicit QF-debt budget plus polish recovery
  test; otherwise basin-graph or multi-start selection is the cleaner path.
- Cost note: the next priority row was expensive under uncapped exact raw CPM
  recomputation. Before expanding this diagnostic further, add a target/unit
  cap, incremental delta scoring, or cache-aware scorer.

Barrier-aware pathway reframing:

- Design doc:
  `docs/research/dongdaemun/refinement/dongdaemun_basin_transition_operator_design.md`
- The ordered flip results should be treated as evidence that naive one-step
  greedy ordering is the wrong objective, not as proof that a pathway is
  impossible.
- Different basins should have a QF wall. A useful pathway search therefore
  needs to ask which non-greedy prefix crosses that wall with the lowest peak
  raw barrier and the best polish recovery.
- The immediate implementation target is a Pareto-style scorer over pathway
  prefixes, with `peak_raw_barrier`, support progress per barrier, compactness,
  and optional closure/context cost.
- The second target is a polish-aware evaluator for selected prefixes. A prefix
  only becomes an operator candidate if bounded polish recovers enough QF while
  retaining support progress away from the original basin.
- Greedy failure labels should be carried into the output rows:
  `q_greedy_miss`, `progress_greedy_miss`, `closure_compound_miss`, and
  `polish_recovery_miss`.
- Interpretation: the perturbation hypothesis is now structural. Instead of
  random shaking or more threshold sweeps, use non-greedy pathway prefixes as
  deliberate basin-wall crossing seeds, then measure whether polish makes the
  crossing durable and cost-effective.

Barrier-aware pathway scorer result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_barrier_aware_prefix_field34_cc_v1/`
- Reusable module:
  `sciscape/clustering/leiden_basin_profile.py`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/tunneling_pathways/analyze_leiden_basin_barrier_aware_pathways.py`
- The scorer reads existing ordered-flip frontier rows. It does not rerun
  Leiden, accept a mutation, or prove recoverability.
- Default run over the first three v1 profiles selected `150` prefix rows,
  `50` per case.
- Best selected low-barrier prefixes:
  - `c0-s11-r0.001`: support progress `0.131`, peak raw barrier `0.437`.
  - `c2-s11-r0`: support progress `0.154`, peak raw barrier `0.933`.
  - `c2-s42-r0`: support progress `0.062`, peak raw barrier `1.253`.
- These prefixes are mostly `q_greedy_miss` and `progress_greedy_miss`: they
  are not the first one-step choice under either naive objective.
- Interpretation: the first useful pathway object is now visible. It is not a
  high-progress direct jump; it is a modest-progress, low-barrier non-greedy
  prefix. The next required step is a polish-aware evaluator for these selected
  prefixes.

Polish-aware prefix evaluator result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/pathway_polish_aware_prefix_field34_cc_v1/`
- Reusable module additions:
  `apply_prefix_units`, `compact_membership`, `fixed_outside`,
  `membership_metric_row`, and `classify_polish_recovery` in
  `sciscape/clustering/leiden_basin_profile.py`
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/evaluate_leiden_basin_polish_prefixes.py`
- Default run evaluated `30` prefixes: top `10` per case, prefix nodes only
  mutable, `3` local-polish iterations.
- The evaluator compacts edited memberships before calling Rust Leiden. This
  is required because fresh labels can be sparse and the Rust local-move path
  expects compact label IDs.
- QF recovery was strong: `0/30` rows were labeled `quality_loss`.
- Durable support shift was rare: only `1/30` rows reached
  `recovered_support_shift`.
  - `c0-s11-r0.001`, prefix rank `9`: raw `-1.585` QF became `+0.805` QF,
    support-distance-to-vanilla `0.062`, candidate-progress-from-vanilla
    `0.0243`.
  - `c2-s11-r0` and `c2-s42-r0`: all tested prefixes were
    `recovered_vanilla_near`.
- Interpretation: low-barrier non-greedy prefixes are recoverable in QF, but
  prefix-node-only polish usually pulls them back toward the source basin.
  The next mechanism test should expand the mutable set with closure/context
  nodes around selected prefixes, not run another threshold sweep over the same
  prefix-only operator.

Basin-transition search v0 result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_v0/`
- Reusable module:
  `sciscape/clustering/leiden_basin_search.py`
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/closure_context/search_leiden_basin_transitions.py`
- The default state-greedy run tested `160` search states and `140` action
  edges over `c0-s11-r0.001` and `c2-s11-r0`, using top `10` barrier-aware
  prefixes per case, depth `3`, beam width `5`, and the primitive actions
  `remaining_target_topk`, `candidate_closure_topk`, and
  `boundary_shell_topk`.
- `c0-s11-r0.001`: `8` rows reached the diagnostic
  `support_shift_q_recovered` label. The best state-greedy row was prefix rank
  `9` after prefix polish and two `remaining_target_topk` steps, with
  `+1.188` QF versus vanilla, target progress `0.0332`, and
  support-distance-to-vanilla `0.0979`. Target-set accounting shows this covers
  `86 / 244` target nodes and leaves `158` target nodes unexplained.
- `c2-s11-r0`: `6` rows reached `support_shift_q_recovered` only after
  `remaining_target_topk` was added, but the case-level best state-greedy row
  remains the prefix-polish `vanilla_collapse` row. The best support-shift row
  covers `134 / 376` target nodes and reaches support-distance-to-vanilla
  `0.0513`, but its state-greedy score is slightly negative, so it is a
  reachability signal, not a cost-effective operator win.
- The top10 cap is necessary for this diagnostic because top5 misses the
  previously observed `c0` support-shift prefix.
- Transition-search rows now carry the set-level accounting needed for the next
  stage: `target_node_count`, `action_node_count`, on/off-target action counts,
  covered/remaining target counts, target coverage fraction, and marginal
  parent-relative target-distance/QF/cost fields.
- Interpretation: the search harness is now useful, but the mechanism is not
  solved. `remaining_target_topk` shows that staged target coverage can produce
  stronger c0 rows and can push c2 over the support gate, but c2 remains poor
  on cost-aware score. The next step should compare node-level target accretion
  with unit-aware target actions before writing a concrete
  `closure_gated_expand_from_candidate` operator. This remains Dongdaemun
  diagnostics, not a Dongdaemun-refinement claim.

Target-unit profiling v0 result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_units_field34_cc_v0/`
- Reusable module additions:
  `target_edge_support_rows`, `components_from_edges`, and
  `build_target_unit_rows` in `sciscape/clustering/leiden_basin_search.py`.
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/profile_leiden_basin_target_units.py`
- Scope: `c0-s11-r0.001` and `c2-s11-r0`, using the full V-only target set
  from candidate-vs-vanilla support, with three diagnostic unit definitions:
  `label_intersection_block`, `target_connected_component`, and
  `triangle_supported_component`.
- The default run emitted `553` unit rows and `6` case/type summary rows.
- `c0-s11-r0.001`: target size `244`. Connected components compress the target
  to `43` units but include a broad `81`-node component. Triangle-supported
  components split that broad structure into smaller cohesive units, with max
  unit size `30`.
- `c2-s11-r0`: target size `376`. Connected components compress the target to
  `37` units with max size `77`; triangle-supported components produce `100`
  units with max size `73`; label-intersection blocks produce `120` units with
  max size `41`.
- Interpretation: unit cohesion is now measurable, but it is not yet an
  operator win. The next action should be a `remaining_target_unit_topk` family
  that grows coverage by coherent target units and reports marginal target
  progress per mutable node, instead of adding individual high-pull target
  nodes.

Unit-aware target action v0 result:

- Reusable module:
  `build_remaining_target_unit_actions` and
  `ACTION_REMAINING_TARGET_UNIT_TOPK` in
  `sciscape/clustering/leiden_basin_search.py`.
- Script option:
  `research/consensus/scripts/leiden_basin/transition_routes/closure_context/search_leiden_basin_transitions.py --action-types remaining_target_unit_topk`
- Full unit branching with mixed unit types and `max_target_unit_actions=3`
  was stopped after more than six minutes before writing rows. A smaller
  smoke run with two pairs, top five prefixes, depth two, and one unit branch
  was also stopped after more than two minutes. This makes branch cost itself a
  first-class result: deeper unit-aware searches need cached pull scores or a
  narrower candidate schedule.
- Completed depth-one artifacts:
  - `basin_transition_search_field34_cc_unit_c0_depth1_v0/`
  - `basin_transition_search_field34_cc_unit_c0_label_depth1_v0/`
  - `basin_transition_search_field34_cc_unit_c0_triangle_depth1_v0/`
  - `basin_transition_search_field34_cc_unit_c2_label_depth1_v0/`
- `c0-s11-r0.001`, label-block depth-one: best unit row was
  `support_shift_q_recovered` with score `0.0388`, `delta_q=0.2799`,
  target progress `0.0222`, and support-distance-to-vanilla `0.0571`.
  This is weaker than the existing prefix-only best (`0.0980`, `0.8052`,
  `0.0243`, `0.0623`) and much weaker than node-level staged target growth
  from the previous v0 (`0.1150`, `1.1882`, `0.0332`, `0.0979`).
- `c0-s11-r0.001`, triangle-supported depth-one: best unit row remained
  `vanilla_collapse`, not a useful support-shift improvement.
- `c2-s11-r0`, label-block depth-one: best unit row was still
  `vanilla_collapse` with score `0.0045`, `delta_q=0.8123`, target progress
  `0.0035`, and support-distance-to-vanilla `0.0416`.
- Interpretation: coherent units are useful for diagnosis, but naive
  pull-ranked unit growth is not yet a better operator than node-level staged
  growth. The next mechanism question is no longer "unit or node?" in the
  abstract; it is which cached cost-aware unit scorer can choose units that
  improve marginal support shift per mutable node before expensive polish.

Node-level target elbow profile v0 result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_field34_cc_v0/`
- Reusable module additions:
  `remaining_target_pull_frame` and `remaining_target_elbow_summary` in
  `sciscape/clustering/leiden_basin_search.py`.
- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/profile_leiden_basin_target_elbows.py`
- Scope: `c0-s11-r0.001` and `c2-s11-r0`, top `10` barrier-aware prefixes per
  case, max `3` cheap staged target-growth steps, no bounded polish.
- The profile emitted `120` stage rows and `7,680` curve rows. It compares the
  existing fixed-cap rule with a guarded elbow candidate:
  use the largest pull-score gap only when the gap is at least `25%` of the top
  pull and the selected prefix already covers at least `50%` of positive pull;
  otherwise fall back to the first `80%` cumulative-pull point.
- `c0-s11-r0.001`: median fixed k is `16`, median guarded k is `11`, and median
  guarded pull fraction is approximately `0.81`.
- `c2-s11-r0`: median fixed k is `34.5` for the fixed-cap path and `31` for
  the guarded path; median guarded k is `18-18.5`, again retaining
  approximately `0.81` pull fraction.
- The first raw max-gap attempt was too aggressive, often choosing `k=1` with
  only `15-22%` cumulative pull. The pull-fraction guard fixed that failure
  mode.
- Interpretation: there is a real elbow signal. It is not yet an algorithmic
  win because support shift and QF recovery require bounded polish. The next
  comparison should run matched fixed-cap and guarded-elbow actions through the
  same polish evaluator and judge material support shift per mutable node,
  quality recovery, and wall time.

Bounded-polish target-elbow comparison result:

- Script:
  `research/consensus/scripts/leiden_basin/operator_probes/polish_elbow/evaluate_leiden_basin_target_elbow_polish.py`
- Smoke artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_smoke_v0/`
- Per-pair top10 artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c0_top10_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c2_top10_v0/`
- Escalation/backfill artifacts:
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c0_escalate_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c2_backfill_v0/`
  - `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_target_elbow_polish_field34_cc_c0_backfill_v0/`
- Scope: matched fixed-cap and guarded-elbow target growth from the same
  barrier-aware prefixes, each with prefix polish and up to `3` staged target
  steps using `3` local-polish iterations. The combined default run is
  expensive enough that per-pair top10 runs are the more inspectable artifact
  contract for now.
- `c0-s11-r0.001`: fixed-cap best shift is stronger
  (`score=0.131`, `delta_q=1.191`, support-distance-to-vanilla `0.103`,
  coverage `0.307`, mutable nodes `75`). Guarded elbow keeps the same
  diagnostic support-shift label with lower cost (`score=0.112`,
  `delta_q=0.929`, support-distance-to-vanilla `0.073`, coverage `0.189`,
  mutable nodes `46`). This is a useful cost-reduction signal, but it buys
  less basin movement.
- `c2-s11-r0`: fixed-cap reproduces the previous weak reachability row
  (`delta_q=0.962`, support-distance-to-vanilla `0.051`, coverage `0.356`,
  mutable nodes `134`) but its state-greedy score remains negative
  (`-0.0077`). Guarded elbow reduces selected target nodes but never crosses
  the support gate; it stays `vanilla_collapse`.
- Interpretation: guarded elbow is not a general replacement for fixed-cap
  `remaining_target_topk`. It can be a cheaper early-stage policy on c0, but on
  c2 it removes exactly the tail nodes needed to cross the support gate. The
  next mechanism step should treat elbow as a cost-aware scheduler with an
  escalation rule, not as an acceptance rule: start guarded, escalate toward
  fixed-cap when support-distance stalls below the gate.
- Escalation/backfill update:
  - `guarded_escalate` uses guarded selection first, then switches to current
    fixed-cap selection when the previous polished state remains below the
    support gate.
  - `guarded_backfill` first adds the fixed-cap tail that guarded selection
    skipped, then escalates to fixed-cap if still below gate.
  - On `c0-s11-r0.001`, `guarded_escalate` is a useful compromise:
    `score=0.130`, `delta_q=1.032`, support-distance-to-vanilla `0.098`,
    coverage `0.225`, mutable nodes `55`. It keeps almost the fixed-cap
    support shift with fewer mutable nodes, but less QF than fixed-cap.
  - On `c2-s11-r0`, neither `guarded_escalate` nor `guarded_backfill` crosses
    the support gate. Fixed-cap still produces the only support-shift row, and
    that row remains low-ROI with negative state-greedy score.
  - Interpretation: the first target action is path-dependent. If the guarded
    first mutable set is too narrow, later fixed/backfill additions do not
    reconstruct the fixed-cap pathway. The next diagnostic should be a small
    branching search that keeps guarded and fixed first-step variants alive
    until bounded polish reveals which basin wall they enter.

Reachability-first transition search result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_search_field34_cc_reachability_v0/`
- Reusable module additions:
  `SEARCH_POLICY_REACHABILITY_FIRST`, `reachability_search_score`, and
  `reachability_label` in `sciscape/clustering/leiden_basin_search.py`.
- Scope: same c0/c2 target slice as the state-greedy search, top `10`
  barrier-aware prefixes per case, depth `3`, beam width `8`, and primitive
  actions `remaining_target_topk`, `candidate_closure_topk`, and
  `boundary_shell_topk`.
- The run emitted `196` states, `176` edges, `29` Pareto rows, and `2` case
  rows. QF is no longer a pruning feature for the search policy; it is still
  reported as debt/recovery for later operator judgment.
- `c0-s11-r0.001`: `35 / 98` rows reached the support gate and `63 / 98`
  showed target progress. The best reachability row applies three staged
  `remaining_target_topk` actions from prefix rank `9`, covers `129 / 244`
  target nodes, reaches support-distance-to-vanilla `0.108`, and still has
  `+1.174` QF versus vanilla. The Pareto frontier also keeps a negative-QF
  support-gate row (`delta_q = -0.293`, support-distance-to-vanilla `0.088`),
  confirming the policy is no longer silently discarding exploratory states.
- `c2-s11-r0`: `2 / 98` rows reached the support gate and `96 / 98` showed
  target progress. The best reachability row covers `199 / 376` target nodes
  and reaches support-distance-to-vanilla `0.0538` with `+0.746` QF, but
  target progress is only `0.0046`. This is a weak pathway-discovery signal,
  not a strong operator candidate.
- Interpretation: the user's criticism was correct. If pathway discovery is
  judged through QF/cost-aware greedy score too early, the search can miss
  routes that move in state space. The fix is not to ignore QF permanently:
  use `reachability_first` for exploration, then re-rank discovered paths by
  material QF, support progress, mutable-node cost, wall time, and seed
  controls. The current evidence makes c0 worth a branch-policy follow-up;
  c2 still looks fragile and needs a broader pathway search before any
  mechanism claim.

Pathway QF wall statistics result:

- Artifact:
  `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_pathway_wall_stats_field34_cc_v0/`
- Reusable module additions:
  `compute_pathway_wall_rows`, `summarize_pathway_wall_rows`, and
  `select_qf_wall_frontier` in `sciscape/clustering/leiden_basin_search.py`.
- Script:
  `research/consensus/scripts/leiden_basin/transition_routes/route_wall/summarize_leiden_basin_pathway_walls.py`
- Scope: reconstruct each `transition_search_states.csv` row as a
  root-to-terminal path, then compare `reachability_v0` with the earlier
  state-greedy `v0`. The path-level QF wall is the maximum QF debt versus the
  source basin anywhere on that parent chain.
- The default run emitted `356` pathway rows, `4` case rows, `138` frontier
  rows, and `12` wall-bucket rows.
- `reachability_v0 / c0-s11-r0.001`: `35 / 98` paths reached the support gate.
  The lowest-wall support-gate path has wall `0.393`, final `delta_q=+0.276`,
  support-distance-to-vanilla `0.0619`, target progress `0.0180`, coverage
  `107 / 244`. The strongest-support path has wall `1.585`, final
  `delta_q=+1.174`, support-distance-to-vanilla `0.108`, target progress
  `0.0378`, coverage `129 / 244`.
- `reachability_v0 / c2-s11-r0`: only `2 / 98` paths reached the support gate.
  The best and lowest-wall support-gate path has wall `1.078`, final
  `delta_q=+0.746`, support-distance-to-vanilla `0.0538`, coverage
  `199 / 376`, but target progress is only `0.00461`.
- State-greedy `v0` comparison:
  `c0` finds a lower-wall support-gate row at wall `0.328`, but it ends with
  `delta_q=-0.328`, so it is a crossing diagnostic rather than a recovered
  operator row. The stronger recovered `c0` support row still pays wall
  `1.585`. `c2` again needs wall `1.078` for support-gate paths.
- Interpretation: the basin wall is now measurable rather than rhetorical.
  For c0, there is a real tradeoff curve: low wall gives modest support
  movement, while the stronger support route pays a larger wall and still
  recovers QF. For c2, the wall is not huge, but the support-gate crossing buys
  very little target progress; this is not yet a good mechanism candidate.
  The next branch-policy search should optimize over this wall/progress curve,
  not over QF alone and not over reachability alone.

First Rust fast-path validation:

- Artifact:
  `research/consensus/results/adaptive_refinement/rust_dongdaemun_fast_path_validation/field12_seed42/`
- Input: `field12_gcc_emb_full_knn30`, source seed `42`, `gamma=0.01`,
  target max doc weight `1500`, lower-tail repaired membership as baseline.
- Python orchestration backend: accepted/committed, `Delta Q=470.27`,
  remaining oversize `1`, max doc weight `1555`, elapsed `6.57 sec`.
- Rust Dongdaemun backend: accepted/committed, `Delta Q=352.91`,
  remaining oversize `0`, max doc weight `1500`, elapsed `19.04 sec`,
  membership diff versus Python backend `106` nodes.
- Interpretation: this confirms the opt-in Rust path can satisfy the hard
  upper-tail target on the field12 seed42 fixture, but it is not ranking-parity
  equivalent to the Python orchestration path. The elapsed times are backend
  behavior timings, not kernel speed timings: the Rust path ran split-repair
  plus trim and satisfied the cap, while the Python orchestration path committed
  a different trim-centered result that left one oversize cluster. More
  phase-level runtime/quality profiles are needed before any fused kernel
  decision.

## Latest Observations

Large bcrefresh contracted-graph probes show why this should be a targeted
stage rather than another full random restart:

- `gamma=0.0005`, seed 42, convergence-guard run:
  - final clusters: `1,670,312`
  - elapsed: approximately `66.1 min`
  - high-water memory: approximately `42.9 GB`
  - final doc-size distribution remains very small
- Same input with the large-graph recursion guard:
  - final clusters: `1,670,185`
  - elapsed: approximately `59.2 min`
  - Leiden phase elapsed: approximately `58.4 min`
  - speedup over the guardless convergence run: approximately `414 sec`
    (`10.4%` total wall time)
  - high-water memory: approximately `42.9 GB`
  - CPM quality improved slightly (`+103.37`)
  - final doc-size distribution stayed effectively unchanged
- Cluster-graph dry-run diagnostics on that membership:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/bcrefresh_g0005_recguard_cluster_graph_summary.json`
  - GPU artifact directory:
    `/data/openalex_clusters/rust_profile_bcrefresh_contracted_g0005_recguard_seed42_convergence_guard_20260429/adaptive_refinement_dryrun`
  - graph reload: approximately `50.5 sec`
  - cluster-graph stats pass: approximately `39.0 sec`
  - dry-run high-water memory: approximately `26.2 GB`
  - top `50,000` macro-merge candidates all improved the `250-1500`
    doc-weight band, but none had positive CPM `delta_Q`
  - the top `10,000` exported candidates include `105` exact `delta_Q = 0`
    ties and `1,161` near-neutral candidates with `delta_Q > -1e-4`
- Medium g016 contracted-graph dry-run diagnostics on the
  `gamma=0.0085` membership:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_cluster_graph_summary.json`
  - GPU artifact directory:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_dryrun_selfloops`
  - input graph: `2,925,181` contracted nodes and `169,159,819`
    parquet edge rows
  - graph reload from raw sidecars: approximately `9.0 sec`
  - cluster-graph stats pass: approximately `7.7 sec`
  - dry-run high-water memory: approximately `7.4 GB`
  - active clusters: `1,612,895`
  - clusters in the `250-1500` doc-weight band: `51,845`
  - top `50,000` macro-merge candidates all improved the target size band,
    but none had positive CPM `delta_Q`
  - `217` top candidates had `delta_Q > -1e-4`; `2,321` had
    `delta_Q > -1e-3`
- Exploratory macro-merge policy ensemble on the exported top `10,000`
  g016 candidates:
  - artifact directory:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_macro_merge_ensemble`
  - all selected near-neutral candidates involved at least one singleton
    endpoint; no non-singleton-only policy selected a candidate
  - `epsilon=1e-4` selected `216` non-conflicting pairs with cumulative
    `Q` debt approximately `0.0113`
  - `epsilon=1e-3`, `Q` debt capped at `0.5`, selected `1,493` pairs
  - policies requiring merged weight in the `250-1500` band selected only
    `5` pairs at `epsilon=1e-4`, `23` at `epsilon=3e-4`, and `80` at
    `epsilon=1e-3`
  - no policy increased the count of clusters already in the target band;
    these merges mainly attach tiny singleton fragments to small/medium
    clusters
- Boundary-candidate dry-run on the same g016 membership:
  - artifact directory:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_boundary_candidates`
  - GPU artifact directory:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_boundary_dryrun`
  - graph reload: approximately `9.0 sec`
  - cluster-graph stats pass with second-neighbor metrics: approximately
    `7.5 sec`
  - dry-run high-water memory: approximately `7.4 GB`
  - active clusters with a second neighbor: `1,540,676`
  - neighbor-weight ratio is high (`p50=0.630`, `p90=0.930`), meaning many
    clusters are not one-sided leaves at the cluster-graph level
  - policy-filtered candidate counts:
    - `boundary_nonleaf`: `293,608`
    - `boundary_high_ambiguity`: `264,945`
    - `boundary_band_scale`: `85,204`
    - `boundary_largeish_ambiguous`: `27,453`
    - `boundary_strict`: `10,736`
- Boundary block-move probes on top `1,000` candidates:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_boundary_move_probe/README_summary.json`
  - GPU artifact directories:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_boundary_move_probe_*`
  - each run reloads graph in approximately `9-13 sec`, but the actual
    block-move probe pass takes only `0.09-0.18 sec` for `1,000` clusters
  - tested policies: `boundary_strict`, `boundary_high_ambiguity`,
    `boundary_band_scale`, and `boundary_nonleaf`
  - no policy found a positive-CPM single-block move to the top or second
    neighbor cluster
  - with a relaxed near-neutral threshold of `epsilon=0.05`,
    `boundary_high_ambiguity` found only `21` near-neutral block moves and
    `boundary_band_scale` found `22`
  - best observed single-block move still had negative `delta_Q`
    (`-0.000401`)
- Boundary grouped split/move probes on the same top `1,000` candidate sets:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_boundary_group_probe/README_summary.json`
  - GPU artifact directories:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_boundary_group_probe_*`
  - each run reloads graph in approximately `8.9-9.0 sec`, while the grouped
    probe pass itself takes only `0.08-0.11 sec` for `1,000` clusters
  - tested policies: `boundary_strict`, `boundary_high_ambiguity`,
    `boundary_band_scale`, and `boundary_nonleaf`
  - no policy found a positive-CPM grouped move or split to the top or second
    neighbor cluster
  - the grouped top/second-neighbor heuristic is therefore useful as a cheap
    negative screen, but not sufficient as the adaptive perturbation itself
- Multi-core split probes on the top `500` g016 doc-weight clusters:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_multi_core_split_probe/README_summary.json`
  - GPU artifact directories:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_multi_core_split_probe_large_top500*`
  - probe pass takes approximately `0.07 sec` for `500` clusters times
    `5-6` gamma multipliers; graph reload still dominates at approximately
    `8.9 sec`
  - no induced split is positive at the baseline `gamma=0.0085`
  - high-gamma probes reveal strong hysteresis-only splits:
    - coarse multipliers `{1.25,1.5,2,3,5}`: `2,455 / 2,500` rows are
      positive at the probe gamma but non-positive at the baseline gamma
    - fine multipliers `{1.02,1.05,1.10,1.15,1.20,1.25}`: `1,892 / 3,000`
      rows are hysteresis-only
  - split granularity rises rapidly as gamma increases:
    - at `1.02x`, median parts is `3`, median singleton weight is `2`
    - at `1.10x`, median parts is `9`, median singleton weight is `87.5`
    - at `1.25x`, median parts is `20`, median singleton weight is `306.5`
  - this supports the hysteresis model: raising gamma exposes many internal
    fragments and hub/supernode singletons, but preserving those fragments after
    lowering gamma requires an explicit utility/debt rule rather than pure CPM
    acceptance
- Split-then-repair probes on the same top `500` g016 doc-weight clusters:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/g016_gamma0p0085_split_merge_repair_probe/README_summary.json`
  - GPU artifact directory:
    `/data/openalex_clusters/sciscape_initialized_graph_weighted_probe_g016_band_250_1500_20260410/gamma_0p0085/adaptive_refinement_split_merge_repair_probe_large_top500_fine`
  - repair policy: force high-gamma split, then greedily merge at baseline
    `gamma=0.0085`; source-source and source-external merges are allowed, while
    external-external merges are blocked
  - probe pass takes approximately `1.19 sec` for `500` clusters times `6`
    gamma multipliers
  - most forced splits repair back to the original source cluster:
    `2,612 / 3,000` rows restore the source cluster
  - `374 / 3,000` rows have source mass escaping into external neighbor
    clusters after repair
  - because restored splits can produce tiny floating-point positive
    `net_delta_Q`, use `net_delta_Q > 1e-6` as the practical positive threshold
  - `144` rows are practically net-positive, `130` of them include escaped
    source mass, and `81` rows exceed `net_delta_Q > 1.0`
  - best observed repaired perturbation has `net_delta_Q = 58.91`
  - this is the first positive evidence for the hysteretic strategy: a split
    that is not acceptable by itself can become useful after baseline repair
- Cross-sample split-then-repair probes:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/split_merge_repair_cross_sample_summary.json`
  - all runs use the same fine gamma multipliers
    `{1.02, 1.05, 1.10, 1.15, 1.20, 1.25}` and the same baseline repair rule
  - g016 `gamma=0.0085`, top `500`: `144 / 3,000` rows have practical
    `net_delta_Q > 1e-6`, `81` exceed `net_delta_Q > 1.0`, and `130`
    practical positives include escaped source mass
  - g016 `gamma=0.01`, top `500`: `64 / 3,000` practical positives, `36`
    strong positives, and `64` escaped practical positives
  - g016 `gamma=0.0125`, top `500`: `23 / 3,000` practical positives, `6`
    strong positives, and `23` escaped practical positives
  - g016 `gamma=0.015`, top `500`: `36 / 3,000` practical positives, no
    strong positives, and `30` escaped practical positives
  - bcrefresh contracted `gamma=0.0005` prepared membership, top `300`:
    `91 / 1,800` practical positives, no strong positives, and `86` escaped
    practical positives
  - interpretation: the split-repair signal is not unique to the original
    g016 `0.0085` sample, but strong `net_delta_Q > 1.0` cases are
    resolution-regime dependent. Apply mode should therefore rank candidates by
    utility/cost and structural escape metrics, not by a strong-Q threshold
    alone.
- Giant-cluster split-repair probes on the g016 postprocess sweep:
  - candidate policy is still `large_doc_weight`, so these runs explicitly
    target the largest clusters first
  - g016 postprocess sweep `gamma=0.00085`, top `300`
    (`doc_weight` approximately `4,301-7,621`): `1,800 / 1,800` rows are
    strong practical positives, all with escaped source mass; median
    `net_delta_Q` is approximately `2,410`
  - g016 postprocess sweep `gamma=0.0015`, top `300`
    (`doc_weight` approximately `3,178-4,943`): `1,793 / 1,800` rows are
    strong practical positives, all with escaped source mass; median
    `net_delta_Q` is approximately `594`
  - g016 postprocess sweep `gamma=0.003`, top `300`
    (`doc_weight` approximately `1,851-2,806`): `1,739 / 1,800` rows are
    strong practical positives, all with escaped source mass; median
    `net_delta_Q` is approximately `89`
  - probe cost for these realistic large clusters is still low:
    approximately `2.2-5.3 sec` for `300` clusters times `6` gamma
    multipliers after graph load
  - an ultra-giant g016 nano-layer `gamma=0.0003` test was intentionally
    stopped: its top clusters have approximately `44k-111k` blocks and
    `1.4M-3.2M` doc weight, and even a reduced top `5` by `3` multipliers run
    exceeded several minutes before writing results
  - implication: the strategy is strongest when large clusters are under-split,
    but apply mode needs a hard cost screen by block count and estimated
    induced edges before probing ultra-giant clusters.
- Local larger-graph apply pilot on `field12_gcc_emb_full_knn30`:
  - summary artifact:
    `research/consensus/results/adaptive_refinement/field12_gcc_split_repair_apply_pilot_top50/README_summary.json`
  - sample size: `753,567` nodes and `19,144,773` undirected edges
  - prepared postprocess membership has `2,841` clusters, max cluster size
    `1,629`, and `1` cluster above the target max `1,500`
  - conservative top-50 utility/cost selection committed `20 / 20` selected
    candidates with exact `delta_Q = 2,053.90`; predicted net delta matched
    exact recomputation within approximately `0.02`, and `632` nodes changed
  - that conservative run did not reduce the max cluster because utility/cost
    ranking selected cheaper clusters before the only oversize cluster; max
    stayed `1,629`
  - oversize-only run on that single cluster, with singleton budget relaxed to
    `100`, committed exact `delta_Q = 474.44`, changed `72` nodes, created `7`
    retained clusters, and reduced max cluster size from `1,629` to `1,557`
  - implemented `oversize_first` selection mode and reran the same single
    oversize cluster: it selected a lower-`delta_Q` but stronger size-reduction
    row, committed exact `delta_Q = 249.51`, changed `83` nodes, and reduced max
    cluster size from `1,629` to `1,546`
  - large-set default-mode run over the top `50` candidates (`300` probe rows)
    omitted `--selection-mode`, used the new `oversize_first` default, selected
    the oversize source, committed exact `delta_Q = 434.34`, changed `83`
    nodes, and again reduced max cluster size from `1,629` to `1,546`; peak
    memory was approximately `734 MB`
  - implemented iterative oversize apply: each pass recomputes current oversize
    clusters from the committed membership and writes per-iteration artifacts
  - the first unrestricted iterative test found a degenerate whole-source escape
    row (`retained_source_units == 0`) that moved `1,518` nodes for only `5`
    units of max-size progress; `oversize_first` now rejects this as
    `no_retained_source_unit`
  - with that retained-source guard, iterative apply committed `2` passes,
    exact total `delta_Q = 464.96`, changed `111` nodes versus the initial
    membership, and reduced max cluster size from `1,629` to `1,518`; the third
    pass stopped with `no_selected_candidates`
  - implemented Rust-side oversize boundary trim as an optional post-polish:
    it greedily moves boundary nodes from remaining oversize clusters to existing
    neighbor clusters, preserves the target max cap for receivers, and writes
    `oversize_boundary_trim_moves.csv`
  - conservative trim (`trim_min_delta_q = 0`) after the retained iterative run
    moved `12` nodes, added exact `delta_Q = 28.62`, and reduced max cluster size
    from `1,518` to `1,506`; it stopped because the remaining moves were
    slightly negative under CPM
  - target-reaching trim (`trim_min_delta_q = -1.0`) moved `18` nodes, still
    added exact `delta_Q = 28.37`, and reduced max cluster size from `1,518` to
    exactly `1,500`; combined with the two split-repair passes this gives total
    exact `delta_Q = 493.33`, `129` changed nodes versus initial, and `0`
    clusters above the target max
  - current recommendation: keep `trim_min_delta_q = 0` as the conservative
    quality-first default, and use a small negative bound only when satisfying
    the hard max-size target is more important than per-move CPM monotonicity
- Success/failure diagnostics:
  - diagnostic artifact:
    `research/consensus/results/adaptive_refinement/split_merge_repair_success_failure_diagnostics.json`
  - successful rows have two consistent signatures:
    `restored_source_cluster == false` and `escaped_source_weight > 0`
  - weak band-focused samples mostly fail by exact restoration: repair gain
    matches forced split debt (`repair_delta_Q / split_debt ~= 1.0`), the
    largest source unit fraction returns to `1.0`, and no source mass escapes
  - realistic giant samples succeed when repair gain exceeds split debt:
    median `repair_delta_Q / split_debt` is approximately `1.44`, `1.27`, and
    `1.14` for g016 postprocess-sweep `gamma=0.00085`, `0.0015`, and `0.003`
  - many successful giant cases are not balanced multi-way splits. The dominant
    source mass often reattaches to one external neighbor
    (`escaped_source_weight / doc_weight ~= 0.97-1.00`) while smaller residual
    fragments remain.
  - failed giant rows are usually over-fragmented: singleton mass and core-part
    count are higher, repair gain falls below split debt, and `net_delta_Q`
    turns negative despite large escaped mass.
- Pre-screening diagnostics:
  - diagnostic artifact:
    `research/consensus/results/adaptive_refinement/split_repair_predictor_screening.json`
  - in the current probe set, the strongest pre-probe discriminator is simple
    cluster size. A cluster-level `doc_weight >= 1500` gate selects the giant
    successful regimes with precision `1.0` and recall approximately `0.966`
    for best-row `net_delta_Q > 1.0`.
  - this is not yet a validated universal classifier. It is partially
    confounded by sample and gamma regime, and it intentionally misses smaller
    weak-positive band cases.
  - apply mode should therefore use a cascade:
    first size/cost pre-screening, then cheap split-only diagnostics, then full
    split-repair only for candidates that pass both.
  - useful cheap split-only diagnostics are singleton fraction, core-part
    count, split debt per doc weight, and the smallest gamma multiplier that
    creates a meaningful split. These should be logged before full repair.
- External-grain pre-screen implementation:
  - Rust backend method: `RustLeidenGraph.external_grain_probes(...)`
  - CLI helper: `scripts/run_adaptive_external_grain_probe.py`
  - diagnostic comparison artifact:
    `research/consensus/results/adaptive_refinement/external_grain_vs_split_repair_comparison.json`
  - this probe groups source nodes by their strongest external neighbor
    cluster, then evaluates direct group move/split deltas without high-gamma
    induced local merge and without baseline repair
  - g016 postprocess sweep `gamma=0.003`, giant top `300`:
    external-grain pass took approximately `0.11 sec` after graph load,
    compared with approximately `2.22 sec` for full split-repair on the same
    candidates and `6` gamma multipliers
  - in that giant regime, `best_group_delta_Q > 0` selected `299 / 300`
    clusters, all of which were full split-repair strong positives; this is a
    high-recall cheap screen for under-split giant clusters
  - stricter `best_group_fraction >= 0.01` selected `64 / 300` clusters with
    perfect precision but low recall. This should be interpreted as
    "immediately visible external grain", not as the full candidate set.
  - g016 band `gamma=0.0085`, top `500`: external-grain pass took
    approximately `0.09 sec`; only `2` clusters had positive best external
    grains, and the configured recommendation rule selected none. This matches
    the earlier observation that most band candidates restore to the original
    basin.
  - implication: use external-grain as the first cheap screen. If it is
    positive, full split-repair is likely worthwhile for giant clusters. If it
    is negative but the cluster is still structurally important, fall back to
    split-only high-gamma diagnostics rather than full repair.
- Late iterations can spend several minutes while reducing cluster count by
  only hundreds of clusters.
- This suggests the valuable operations are not random leaf movements, but
  targeted macro merge/split/boundary corrections on the cluster graph.

## Motivation

Late Leiden iterations can spend several minutes on near-identity recursive
contractions while moving only a small number of aggregate nodes. Randomly
shuffling all nodes again is therefore a poor default perturbation strategy for
SciSci graphs.

SciSci clusters repeatedly show core-periphery structure. Perturbing leaf-like
nodes is usually low value, and pure high-gamma splitting is usually not
acceptable at the baseline gamma. The useful perturbation is hysteretic:

- temporarily raise gamma to expose multiple cores and hub/periphery fragments,
- force a local split only inside a small candidate cluster,
- repair that split at the baseline gamma while allowing source fragments to
  reattach to external neighbor clusters,
- accept only the repaired final state, not the forced split itself,
- and then run a limited polish around changed neighborhoods.

Macro merge and boundary movement should therefore be treated as components of
the repair stage, not as independent first-class strategies unless the current
cluster-count regime specifically calls for them.

## Cost-Effectiveness Rule

A perturbation is worth trying only if:

```text
expected_structural_gain / estimated_compute_cost
>
additional_random_leiden_iteration_gain / additional_random_leiden_iteration_cost
```

The current large bcrefresh probe should be used to estimate the denominator.
With recursion guard enabled, late iterations around the 1.6M-cluster regime
still cost approximately `316-320 sec` each while changing cluster count by
only `110-201` clusters and moving `5,040-10,104` aggregate nodes.

## Minimum Basin-Transition Pathway

For the field34/cc_cosine near-miss slice, the current diagnostic question is
not whether a proposed edit survives Leiden polish. It is the smaller
accounting question: how much support must be changed to move from the broad
vanilla footprint toward a compact Dongdaemun candidate footprint, and how much
quality debt appears along that deterministic pathway?

Relabel-pathway artifact:
`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_minimal_pathway_field34_cc/`

Implemented diagnostic:
`research/consensus/scripts/leiden_basin/transition_routes/transition_operators/analyze_leiden_basin_transition_minimal_pathway.py`

Closure-context artifact:
`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_context_field34_cc/`

Implemented diagnostic:
`research/consensus/scripts/leiden_basin/transition_routes/closure_context/analyze_leiden_basin_transition_closure_context.py`

Current result:

- The exact support-set edit target `S_V - S_C` ranges from `244` to `916`
  nodes across the selected candidate-vs-vanilla pairs; median is `376`.
- Candidate support is fully contained in vanilla support for all 5 selected
  pairs under the current support definition (`candidate_containment_ratio=1`),
  so the symmetric support-set lower bound is also `244` to `916` nodes.
- The direct edit set is not context-free. Candidate/baseline label closure
  touches `2,317` to `4,367` nodes in the candidate-2 rows and `2,417` nodes in
  the candidate-0 row. Median candidate-label context ratio is `5.16`.
- The vanilla source-label closure ratio is similarly large: median `4.87`.
- A collision-safe fresh-label relabel pathway over the same support units still
  pays a large quality barrier: median `64.76` QF and maximum `167.95` QF in
  this slice.
- `baseline_forced` and `candidate_forced` are identical here because the
  selected direct edit set is vanilla-extra support that is already unchanged
  in both baseline and candidate memberships.

Interpretation:

The candidate core is not the hard part in this fixture; it is already covered
by the vanilla support footprint. The hard part is closure: a nominal direct
support edit expands into a much larger label-context split/merge problem. The
next operator should therefore expose a boundary-context mutable set around
residual support and high-ratio closure labels, not merely replay group-level
reverts or ask whether fixed-outside polish accepts them.

Updated operator design:
`docs/research/dongdaemun/refinement/dongdaemun_basin_transition_operator_design.md`

The redesigned operator family now starts with a closure-label frontier, then
tests `closure_split_shrink_from_vanilla` before any expand-from-candidate
variant. The immediate next artifact is a frontier ranker that selects labels
by direct node count, closure context burden, outside-support count, and
boundary-role mix before any membership mutation.

Closure-frontier artifact:
`research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/basin_transition_closure_frontier_field34_cc/`

Implemented diagnostic:
`research/consensus/scripts/leiden_basin/transition_routes/closure_context/rank_leiden_basin_transition_closure_frontier.py`

Frontier result:

- The ranker emitted `1,695` label rows from closure-context plus boundary-role
  inputs.
- With default candidate-label shrink settings, `166` labels are eligible and
  `50` are selected (`top_labels_per_pair=10`).
- Selected labels cover `142` direct nodes, `6,422` closure nodes, and `6,280`
  closure-context-extra nodes.
- Median selected closure ratio is `65.67`, and maximum is `212.0`.
- Selected direct nodes are currently collateral-like under the boundary proxy,
  so the first mutation pilot should be direct-node-only
  `closure_split_shrink_from_vanilla`, not expand-from-candidate.

## Objective Terms

Use CPM as the primary accept/reject objective:

```text
Q = sum_C [ e_C - gamma * s_C * (s_C - 1) / 2 ]
```

where `s_C` is cluster doc weight and `e_C` is internal edge weight.

For merging two clusters:

```text
delta_Q_merge(A, B) = w_AB - gamma * s_A * s_B
```

For splitting one cluster into parts `P_i`:

```text
delta_Q_split = -cut(P) + gamma * sum_{i<j} s_i * s_j
```

For split-then-repair:

```text
forced_split_debt = delta_Q_split_at_baseline
repair_gain = sum accepted baseline-gamma merge gains after the forced split
net_delta_Q = forced_split_debt + repair_gain
```

The forced split may be negative. It is only a basin-transition proposal. The
candidate is evaluated after repair, using `net_delta_Q` and structural terms
such as escaped source weight, retained core count, restored-source flag,
singleton mass, and target-size-band movement.

For boundary movement of subcluster `x` from `A` to `B`:

```text
score(x -> B) = w_xB - gamma * s_x * s_B
```

Do not accept candidates solely because they increase cluster count. Penalize
singleton growth and leaf-only movement unless they clearly improve the target
distribution.

## Candidate Utility

Candidate priority should be computed from a utility/cost ratio:

```text
utility =
  net_delta_Q
  + lambda * size_band_improvement
  + eta * target_cluster_count_improvement
  + rho * escaped_source_weight_if_net_positive
  - mu * singleton_or_leaf_penalty
  - nu * restored_source_cluster_penalty

cost =
  estimated_edges_scanned
  + probe_count * induced_edges
  + repair_quotient_edges

priority = utility / cost
```

The first implementation should log these terms in dry-run mode before applying
any membership changes.

For split-repair candidates, the first acceptance policy should be conservative:

```text
accept if:
  net_delta_Q > 1e-6
  and restored_source_cluster == false
  and (escaped_source_weight > 0 or retained_source_units >= 2)
  and final_small_source_weight <= singleton_budget
  and candidate does not conflict with an already accepted source cluster
```

Candidates with `net_delta_Q > 1.0` should be treated as strong candidates.
Candidates with `0 < net_delta_Q <= 1.0` are useful for diagnosis but should
require a structural gain threshold before apply mode.

Operational selection should use two explicit modes:

- `oversize_first`: primary apply-mode policy when `target_max_doc_weight` is
  set. It only accepts candidates that reduce oversize mass, then ranks by
  remaining oversize, oversize reduction, singleton pressure, quality, and cost.
- `utility_cost`: quality/cost optimization and diagnostic policy. It can pick
  cheaper high-utility rows before the oversize source, so it should stay an
  explicit option rather than the default apply-mode behavior.

Rust promotion boundary:

- Keep experiment orchestration, CSV/JSON reports, exact quality rollback
  decisions, and validation joins in Python.
- The stable Rust promotion target is the deterministic split-repair selection
  helper: input probe rows plus a threshold policy, output per-row arrays for
  `accepted_by_policy`, `selected_for_apply`, `priority`,
  `remaining_oversize_before`, `remaining_oversize_after`,
  `oversize_reduction`, `rejection_reason_code`, and `conflict_reason_code`.
- Keep iterative apply and post-apply polish in Python until the acceptance
  policy stabilizes on larger graphs; then promote only the hot deterministic
  kernel, not the reporting layer.

## Stage 1: Cluster Graph Stats

Build reusable cluster-level stats from a baseline membership:

- cluster doc weight,
- cluster block count,
- internal edge weight,
- external edge weight,
- top neighbor weight,
- leafness proxy,
- conductance proxy,
- distance to target size band.

Suggested output:

- per-cluster parquet/csv summary,
- aggregate histogram report,
- top macro-merge and macro-split candidate tables.

Implementation status:

- [x] Rust backend method: `RustLeidenGraph.cluster_graph_stats(...)`
- [x] Python report helper:
      `sciscape.clustering.write_adaptive_refinement_report(...)`
- [x] Macro-merge candidate table ranked by predicted CPM `delta_Q`
- [ ] Macro-split candidate table

## Stage 2: Macro Merge Dry-Run

Start here because it is cheap and works on the cluster graph.

Candidate conditions:

- `delta_Q_merge >= -epsilon` rather than strictly positive; the first large
  dry-run found no positive macro-merge candidates but did find exact-tie and
  near-neutral candidates that improve the target size band,
- merged doc weight moves closer to target band,
- neither side is purely leaf-like unless the merge fixes an obvious fragment,
- no conflicting greedy merge in the same pass.

Implementation should first support:

- [x] dry-run only,
- [x] greedy non-conflicting candidate selection,
- [x] exact predicted `delta_Q_merge`,
- [x] before/after size-band simulation,
- [x] policy-ensemble comparison over epsilon/leafness/conductance filters,
- [ ] actual membership perturbation,
- [ ] post-perturb polish and rollback.

## Stage 3: Hysteretic Split-Repair

This is now the primary adaptive-refinement direction. Only probe a small
budgeted set of clusters, and judge the final repaired state rather than the
forced split itself.

Candidate conditions:

- doc weight above target maximum,
- or high doc weight within the target band when searching for hysteretic
  basin transitions,
- low internal density, high conductance proxy, or strong external attachment,
- evidence of multi-core structure after a small gamma increase,
- sufficient induced edge budget.

Budget constraints:

```text
total_induced_edges_to_probe <= 5-10% of original directed edges
top_k_split_candidates <= 500-1000
probe_seeds <= 2-3
probe_gamma_multipliers in {1.02, 1.05, 1.10, 1.15, 1.20, 1.25}
```

Candidate workflow:

1. Run high-gamma induced local merge inside the source cluster.
2. Evaluate forced-split debt at the baseline gamma.
3. Build a local quotient containing split source parts plus external neighbor
   clusters touched by those parts.
4. Repair greedily at the baseline gamma, allowing source-source and
   source-external merges while blocking external-external merges.
5. Accept only the repaired state if it passes `net_delta_Q` and structural
   filters.

Do not accept a split only because probe-gamma `delta_Q` is positive. The g016
pilot showed that probe-positive split rows are mostly hysteresis-only and
often create large singleton/hub fragments.

Implementation status:

- [x] large-cluster candidate export by doc weight
- [x] induced local-merge reclustering for candidate clusters
- [x] local split `delta_Q` evaluation at baseline and probe gamma
- [x] hysteresis-only split diagnostics
- [x] split-then-baseline-repair dry-run with external-neighbor escape metrics
- [x] cross-sample split-repair probe comparison across nearby gamma settings
      and a larger bcrefresh contracted graph
- [x] giant-cluster split-repair probe comparison on lower-gamma g016
      postprocess sweep memberships
- [x] external-grain pre-screen probe for cheap direct split/move diagnostics
- [x] split-repair candidate selection table ranked by utility/cost
- [x] split-repair apply mode for non-conflicting candidates
- [x] exact quality recomputation and rollback after apply
- [ ] limited polish over affected neighborhoods
- [x] pilot on a larger graph with clusters above the target maximum

## Stage 4: Boundary Refinement

Boundary refinement is now secondary to split-repair. It should operate on
source fragments created by high-gamma split probes or on meaningful ambiguous
regions, not arbitrary leaf nodes.

Candidate conditions:

- top-1 and top-2 destination scores are close,
- the unit connects meaningfully to more than one core cluster,
- doc weight and degree exceed minimum thresholds,
- leafness is below threshold.

Exclude:

- low-degree one-sided leaves,
- tiny doc-weight fragments,
- moves that only increase singleton count.

Implementation status:

- [x] first-neighbor / second-neighbor cluster-graph metrics
- [x] boundary ambiguity policy summaries
- [x] top boundary candidate CSV export
- [x] single-block top/second-neighbor move probe around candidate clusters
- [x] grouped top/second-neighbor split/move probe around candidate clusters
- [ ] integrate split-repair escaped fragments with boundary candidate scoring
- [ ] accepted boundary perturbation only when it improves repaired partition

## Stage 5: Local Polish

After accepted split-repair perturbations, run a limited polish:

- one standard Leiden iteration, or
- a restricted local move over changed neighborhoods.

Do not let polish turn into another full until-convergence run without a budget.

## Open Questions

- Dongdaemun versus vanilla near-endpoint paths:
  - Current field34/cc_cosine evidence shows `endpoint_near_support_far`:
    endpoint coassignment distance is tiny, but vanilla carries a much larger
    changed-support footprint.
  - Do not call this "different pathway" yet; current evidence is final
    footprint only, not move-sequence trajectory.
  - Next checks:
    1. [x] compare `baseline -> Dongdaemun candidate` and
       `baseline -> vanilla` at sketch-node/support-footprint level,
    2. [x] test whether Dongdaemun support is mostly a subset of vanilla
       support,
    3. [x] build a final-footprint transition landscape over observed
       baseline, candidate, and vanilla nodes,
    4. characterize vanilla-only extra support by baseline clusters and final
       cluster labels where those support nodes are inside the endpoint sketch,
    5. decide whether the extra vanilla footprint is boundary noise,
       collateral basin movement, or necessary to reach the near endpoint,
    6. remember that changed-support nodes are sampled independently from the
       endpoint sketch; full cluster labeling of all support nodes requires full
       memberships or recomputation,
    7. only then add trajectory tracing if footprint evidence is still
       ambiguous,
    8. [x] test a first controlled basin-transition operator against the
       cost/quality of one additional vanilla seed/iteration run before making
       any algorithmic claim,
    9. [x] design a boundary-aware shrink/expand operator because the first
       hard candidate-support freeze/transplant pilot did not beat seed
       variation
       (`docs/research/dongdaemun/refinement/dongdaemun_basin_transition_operator_design.md`),
    10. [x] add a boundary analyzer that classifies vanilla-extra support into
        bridge-like versus collateral-like groups before mutating membership
        (`research/consensus/scripts/leiden_basin/transition_routes/closure_context/analyze_leiden_basin_transition_boundaries.py`),
    11. [x] calibrate the boundary proxy scores with group-level revert/add
        dry-runs before treating collateral-like groups as removable
        (`research/consensus/scripts/leiden_basin/transition_routes/transition_diagnostics/calibrate_leiden_basin_transition_boundary_groups.py`),
    12. test whether larger collateral chunks plus their one-hop boundary ring
        can persist after polish; current fixed-outside group/chunk edits mostly
        return to their original basin,
    13. only implement shrink-from-vanilla / expand-from-candidate operators
        after the analyzer shows the boundary classes are auditable and the
        calibration edits survive polish.
- What target should be optimized directly: cluster count, size-band mass, CPM,
  or a weighted combination?
- What `net_delta_Q` and structural-gain thresholds should separate apply-mode
  candidates from diagnostic-only candidates?
- How much singleton/hub mass is acceptable after repair when `net_delta_Q` is
  positive?
- Should the split-repair gamma multiplier be selected per candidate by the
  smallest multiplier that creates an escape, or by best utility/cost?
- Should repair allow near-neutral merges (`repair_epsilon > 0`) or stay
  strictly CPM-positive at baseline gamma?
- Should macro merge happen as a pre-pass only when current cluster count is
  above target, or should it remain only inside split-repair repair?
- Should adaptive refinement consume random seeds, or should it use deterministic
  high-gamma ordering first and reserve randomness for tie-breaking?
- How should accepted adaptive changes be compared against one extra standard
  Leiden iteration on the same baseline?

## Initial Implementation Order

1. [x] Add profiling observability to standard Leiden.
2. [x] Add a large-graph recursion guard for near-identity contraction tails.
3. [x] Build cluster graph stats and dry-run report.
4. [x] Implement macro merge dry-run.
5. [x] Validate predicted `delta_Q` with Rust unit tests and dry-run summaries.
6. [x] Implement boundary move/group probes as negative screens.
7. [x] Implement high-gamma multi-core split probes.
8. [x] Implement split-then-baseline-repair dry-run.
9. [x] Build split-repair candidate selection and conflict resolution.
10. [x] Add split-repair apply mode behind an explicit experimental flag.
11. [x] Run exact quality recomputation plus rollback on accepted candidates.
12. [x] Pilot on larger graph where clusters exceed the target maximum.
13. [x] Add Rust Dongdaemun core, PyO3 binding, and Python wrapper helper.
14. [x] Add opt-in hierarchy-level Rust Dongdaemun fast path.

## Non-Goals For Now

- Do not change the default Leiden objective or local-move semantics.
- Do not mix SciSci-specific target-size heuristics into the Java/CWTS parity
  path.
- Do not accept split/merge candidates solely because they move the cluster
  count toward a target; CPM and size-distribution effects must both be logged.
- Do not chase Python/Rust ranking parity, lower-tail ablations,
  multi-seed schedules, or fused-kernel implementation in this slice.
