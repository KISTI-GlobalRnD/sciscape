# Consensus Research Script Structure

This note is the migration guardrail for `research/consensus/scripts/`.
The current directory is too large to navigate by filename alone, but many
research notes, tests, and saved artifact metadata still reference exact script
paths. Move scripts only after inventorying those references.

## Target Buckets

Use at most six top-level script buckets when this directory is split:

| Bucket | Intended content |
|---|---|
| `common/` | Shared helper modules such as `_common.py` |
| `consensus_core/` | Baseline multi-layer consensus experiments and core probes |
| `review_taxonomy/` | Boundary review, rank-shift review, taxonomy, and review scoring |
| `leiden_basin/` | Leiden basin, hysteresis, wall, route, and adaptive basin scripts |
| `dongdaemun_hierarchy/` | Dongdaemun and hierarchy-postprocess research scripts |
| `artifacts_reporting/` | Freeze, materialize, summarize, export, score, and figure utilities |

Do not create a seventh top-level bucket unless an existing bucket would become
semantically misleading. Prefer a second-level split inside the bucket instead.

## Leiden Basin Second-Level Buckets

`leiden_basin/` is the largest target bucket, so it must be split further:

| Bucket | Intended content |
|---|---|
| `leiden_basin/basin_signatures/` | Basin signatures, selector contracts, mode tradeoff, and general basin diagnostics |
| `leiden_basin/transition_routes/` | Transition, route, wall-route, pathway, tunneling, and direct-pair scripts |
| `leiden_basin/operator_probes/` | Attachment-margin, aligned-core, handle, selector, gate, recovery, and polish probes |
| `leiden_basin/evidence_panels/` | Reviews, audits, field eligibility, relation taxonomy, phase panels, and claim evidence |
| `leiden_basin/materialization/` | Cache, membership, prepare, join, and materialization scripts |
| `leiden_basin/hysteresis/` | Leiden hysteresis runs, monitors, and graph materialization |

The inventory command computes the target path for each script. Use that output
as the move manifest rather than hand-sorting files.

### Leiden Basin Third-Level Buckets

The largest Leiden buckets are split once more:

| Bucket | Intended content |
|---|---|
| `leiden_basin/operator_probes/selector_sources/` | Selector source screening scripts |
| `leiden_basin/operator_probes/selector_signals/` | Selector readiness, attainable-fast, and reduced-signal scripts |
| `leiden_basin/operator_probes/attachment_margin/` | Attachment-margin scripts |
| `leiden_basin/operator_probes/aligned_core/` | Aligned-core and local-handle scripts |
| `leiden_basin/operator_probes/joint_bundle/` | Joint-bundle scripts |
| `leiden_basin/operator_probes/gate_release/` | Gate-release and gate-attachment scripts |
| `leiden_basin/operator_probes/post_gate_recovery/` | Post-gate recovery scripts |
| `leiden_basin/operator_probes/polish_elbow/` | Polish-prefix, target-elbow, and target-unit scripts |
| `leiden_basin/transition_routes/closure_context/` | Closure-context and closure-frontier scripts |
| `leiden_basin/transition_routes/transition_operators/` | Transition operator and minimal-pathway scripts |
| `leiden_basin/transition_routes/transition_diagnostics/` | Transition-boundary and transition-landscape scripts |
| `leiden_basin/transition_routes/route_wall/` | Direct-pair, route, and wall-route scripts |
| `leiden_basin/transition_routes/route_gate_panels/` | Route-gate and route-wall panel scripts |
| `leiden_basin/transition_routes/tunneling_pathways/` | Tunneling, barrier-aware pathway, and pathway-debt scripts |
| `leiden_basin/transition_routes/route_reviews/` | Route review, route-label blocker, and route-interpretation scripts |
| `leiden_basin/basin_signatures/signature_detection/` | Multibasin signature, threshold, and decision-rule scripts |
| `leiden_basin/basin_signatures/portfolio_contracts/` | Portfolio and contract validation scripts |
| `leiden_basin/basin_signatures/trajectory_failure/` | Multifidelity, vanilla-reachability, and greedy-failure scripts |
| `leiden_basin/basin_signatures/branch_growth/` | Branch lookahead, branch-target growth, and random-refinement scripts |
| `leiden_basin/basin_signatures/local_modes/` | Local mode, p5, and quality diagnostics |
| `leiden_basin/basin_signatures/endpoint_flips/` | Endpoint, ordered-flip, and recomputed-metric diagnostics |
| `leiden_basin/evidence_panels/audits/` | Basin evidence audit scripts |
| `leiden_basin/evidence_panels/field_eligibility/` | Field eligibility and basin-definition calibration scripts |
| `leiden_basin/evidence_panels/relation_taxonomy/` | Relation taxonomy and stable/ambiguous relation scripts |
| `leiden_basin/evidence_panels/phase_panels/` | Phase index, wall protocol, and wall panel scripts |
| `leiden_basin/evidence_panels/portfolio_evidence/` | Portfolio evidence panel scripts |
| `leiden_basin/evidence_panels/review_panels/` | Current-result and margin-validation review scripts |

## Dongdaemun-Hierarchy Second-Level Buckets

`dongdaemun_hierarchy/` is also split before movement:

| Bucket | Intended content |
|---|---|
| `dongdaemun_hierarchy/trajectory_analysis/` | Dongdaemun trajectory, instability, local-candidate, online-first, and resolution analysis scripts |
| `dongdaemun_hierarchy/prototype_runs/` | Dongdaemun branch-lookahead, cyclic, and adaptive-stochastic prototype scripts |
| `dongdaemun_hierarchy/trace_summaries/` | Dongdaemun trace summary scripts |
| `dongdaemun_hierarchy/datasets/` | Dataset collection scripts |
| `dongdaemun_hierarchy/refinement_runs/` | Refinement, Rust fast-path, and safe-fast validation scripts |
| `dongdaemun_hierarchy/postprocess_evaluation/` | Hierarchy postprocess evaluation scripts |
| `dongdaemun_hierarchy/postprocess_sweeps/` | Hierarchy postprocess sweep and expansion scripts |

## Review And Consensus-Core Second-Level Buckets

The non-Leiden script families use lighter second-level splits:

| Bucket | Intended content |
|---|---|
| `review_taxonomy/boundary_reviews/` | Boundary review and boundary scoring scripts |
| `review_taxonomy/rank_shift/` | Rank-shift and null rank-shift scripts |
| `review_taxonomy/taxonomy/` | Taxonomy aggregation, classification, export, and calibration scripts |
| `review_taxonomy/review_uncertainty/` | Review uncertainty, reproducibility, and order-balance repair scripts |
| `consensus_core/baseline_comparisons/` | E1-E4 style comparison and density-matched scripts |
| `consensus_core/sweeps/` | Effective-k, cross-field, noise, and other sweep scripts |
| `consensus_core/validation/` | Semantic validation, same-gamma, common-case, and sum-noise validation scripts |

## Migration Rule

Before moving any script, run:

```bash
uv run --extra dev python scripts/inventory_research_scripts.py --fail-on-unclassified
```

Then inspect the most-referenced scripts:

```bash
uv run --extra dev python scripts/inventory_research_scripts.py --format markdown
```

Safe movement requires:

1. no `unclassified` scripts;
2. a replacement plan for every hard-coded `research/consensus/scripts/*.py`
   reference in docs, tests, and research notes;
3. compatibility wrappers or updated tests for scripts that are directly loaded
   by path;
4. no movement of uncommitted research work unless that work is intentionally
   included in the same commit.

## Recommended Phases

1. Add thin compatibility wrappers for high-reference scripts.
2. Move low-reference artifact/reporting utilities first.
3. Move domain-specific Leiden basin and Dongdaemun scripts after their active
   research notes are updated.
4. Move core E1-E7 consensus scripts last, because `research/consensus/README.md`
   documents their commands directly.
