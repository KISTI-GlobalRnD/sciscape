# Dongdaemun Manuscript Plan

Working title:

> Dongdaemun: Quality-Audited Macro-Refinement for Hierarchical CPM Science Maps

Target venue: Scientometrics. Journal of Informetrics is a stretch venue only if the
algorithmic formalism is strengthened beyond the current conservative
postprocess evidence.

Frozen evidence source:

`research/consensus/results/scientometrics_evidence_freeze_20260504`

Naming and claim boundaries:

`docs/research/dongdaemun/core/dongdaemun_naming_contract.md`

## Positioning

Dongdaemun is framed as a Scientometrics-first methodology contribution for
hierarchical science maps. The manuscript should not present Dongdaemun as a new
community-quality objective. It preserves CPM as the objective and extends the
Leiden-style search procedure with quality-audited macro-refinement moves that
are triggered by oversized communities.

The core claim is:

> Standard Leiden-style CPM optimization can leave oversized communities that
> indicate missed macro-refinement opportunities. Dongdaemun reopens those
> communities and accepts only refinements whose exact original-graph CPM
> accounting is non-regressing.

Current experiments support a conservative Dongdaemun-post implementation, not a
fully integrated Rust-native Leiden loop. The integrated loop belongs in the
method design and future-work discussion.

## Claim Register

Use these claim IDs consistently in the manuscript and in
`docs/research/dongdaemun/manuscript/dongdaemun_evidence_map.md`.

| ID | Manuscript claim | Boundary |
| --- | --- | --- |
| C1 | Leiden-style CPM hierarchy construction can leave oversized current-level communities that create hierarchy-readiness problems. | This is a failure-mode and motivation claim, not a proof that every oversized community is wrong. |
| C2 | Dongdaemun preserves the CPM objective by treating oversize as a trigger for macro-refinement proposals and auditing accepted moves with exact original-graph CPM `Delta Q >= 0`. | Oversize is not added as an objective penalty. |
| C3 | The conservative Dongdaemun-post implementation, represented by `two_stage_quality_first`, improves source-level upper-tail imbalance relative to `small_only` in the frozen six-field evidence while keeping exact CPM accounting positive on average. | Do not claim strict cap satisfaction. |
| C4 | Downstream benefits propagate partially to next-level concentration metrics such as Gini and parent max-child share. | Do not claim uniform improvement of the single largest next-level max/target ratio. |
| C5 | Tiny-cluster consolidation is an auxiliary minimum-support policy for lower-tail cleanup. | Do not present tiny merging as the main contribution or as a CPM-quality-improving method. |
| C6 | Hard-cap is useful as a diagnostic or operational variant, but `quality_first` is the defensible default because hard-cap often falls back. | Rejected hard-cap memberships are diagnostic-only unless explicitly accepted. |
| C7 | Semantic coherence results are a post-hoc sanity check over available title/abstract text. | Semantic coherence is not optimized and is not an acceptance criterion. |
| C8 | Residual oversized cases are taxonomizable into interpretable failure modes. | This is a first-pass taxonomy, not a complete theoretical classification. |
| C9 | Same-gamma extension evidence supports iterative split-repair plus conservative boundary trim as the robust postprocess mechanism. | Do not equate this with validated adaptive gamma. |
| C10 | Branch-adaptive local-gamma probing is future diagnostic work. | Current branch-adaptive evidence is negative/diagnostic and not validated as a default method. |
| C11 | Integrated loop-level Dongdaemun is a design addendum and future work package. | Current evidence is Dongdaemun-post; the integrated target is parent-internal adaptive refinement allocation, and no speedup or quality validation is claimed yet. |

## Tiny Consolidation Versus Oversized Refinement

The paper must keep two operations conceptually separate.

Tiny-cluster consolidation addresses the lower tail. It merges or resolves
clusters below a minimum support threshold so that map units are usable for
interpretation and keyword extraction. This is an operational support policy.
It should be described as fixed minimum-support cleanup, not as the
methodological contribution.

Oversized-cluster refinement addresses the upper tail. It reopens communities
that remain too large after local Leiden-style optimization and small-cluster
cleanup. Candidate macro moves are evaluated against the original graph and
accepted only when the CPM audit passes. This is Dongdaemun's main contribution.

## Manuscript Structure

### 1. Introduction

Purpose:

- Establish why hierarchical science maps need partitions that are
  interpretable, statistically usable, and stable enough for graph contraction.
- Explain why CPM Leiden is attractive: it optimizes a clear objective and is
  widely used for large-scale clustering.
- State the hierarchy-specific failure mode: contraction can amplify oversized
  current-level communities into dominant next-level supernodes.
- Introduce Dongdaemun as objective-preserving macro-refinement for oversized
  communities.

Primary claims: C1, C2, C3, C4.

Recommended wording:

> Dongdaemun leaves CPM unchanged. It expands the search neighborhood around
> oversized communities and accepts macro-refinements only when exact
> original-graph CPM quality is non-regressing.

Avoid:

- Any claim that Dongdaemun guarantees target satisfaction.
- Any claim that semantic coherence is optimized.

### 2. Background

Purpose:

- Place the method in the Louvain -> Leiden lineage: improvement of the search
  procedure while preserving the objective.
- Define CPM quality at the level needed for the paper:
  `Q_gamma(P) = sum_C [internal_weight(C) - gamma * n_C * (n_C - 1) / 2]`,
  following the project evaluator for the original graph.
- Explain why original-graph accounting matters when a hierarchy uses
  contraction: quality must be checked before passing a modified membership
  forward.

Primary claims: C2, C11.

Key references to cite from the manuscript bibliography later:

- Louvain and Leiden community detection.
- CPM and resolution-limit-free community detection.
- Science mapping and field-normalized hierarchical map construction.

### 3. Failure Mode

Purpose:

- Define oversized communities as hierarchy-readiness failures rather than
  objective failures.
- Explain why node-level/local refinement can miss collective block moves.
- Introduce metrics:
  `max/target ratio`, `oversize count`, `Gini`, `parent max-child share`, and
  exact original-graph `Delta Q`.
- State that oversize is a trigger for candidate generation, not a penalty term.

Primary claims: C1, C2, C8.

Evidence anchors:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/failure_taxonomy.csv`

### 4. Method

Purpose:

- Define Dongdaemun as a macro-refinement extension to Leiden-style CPM
  optimization.
- Explain the stages:
  raw Leiden, tiny-cluster consolidation, oversize detection, local candidate
  generation, exact original-graph CPM audit, deterministic quality-first
  selection, conflict resolution, and fallback.
- Present pseudocode for the conservative postprocess and for the future
  integrated loop. The full algorithm text belongs in
  `docs/research/dongdaemun/core/dongdaemun_algorithm_design.md`.

Primary claims: C2, C5, C6, C11.

Key principle:

> Leiden moves are CPM-improving or non-decreasing under the optimizer's move
> policy. Dongdaemun macro moves are accepted only when the exact original-graph
> CPM `Delta Q >= 0` under the study policy.

### 5. Implementation In This Study

Purpose:

- State that the evidence uses a conservative Dongdaemun-post implementation.
- Define policy names exactly as they appear in frozen artifacts:
  `raw`, `small_only`, `oversize_split_only`, `two_stage_quality_first`,
  `two_stage_hard_cap`, and `two_stage_hard_cap_aggressive` where present.
- Explain the same-gamma extension and branch-adaptive diagnostics as separate
  supplementary/future-work evidence.
- Explain why the evidence supports method feasibility and quality-audited
  structural improvement but not integrated-loop runtime claims.

Primary claims: C3, C6, C9, C10, C11.

Implementation anchor:

- `sciscape/clustering/hierarchy_postprocess.py`
- `scripts/run_adaptive_split_merge_repair_probe.py`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/source_validation_readme.md`

### 6. Experiments

Purpose:

- Describe the six-field frozen validation:
  fields `12`, `15`, `18`, `26`, `30`, and `34`.
- State source seeds: `11`, `42`, and `73`.
- State next-level seeds: `11`, `42`, and `73`.
- Define the primary source-level comparison:
  `small_only -> two_stage_quality_first`.
- Define context comparisons:
  `raw -> small_only`, `small_only -> oversize_split_only`,
  `small_only -> two_stage_hard_cap`, and hard-cap diagnostics.
- Identify semantic coherence as an available-text sanity check.

Primary claims: C3, C4, C6, C7.

Frozen evidence anchors:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/field_expansion_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/source_seed_next_level_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/semantic_coherence_report.md`

### 7. Results

Recommended result subsections:

1. Lower-tail consolidation is practical but not the contribution.
2. Dongdaemun-post improves source-level upper-tail structure with quality
   preservation.
3. Next-level propagation improves concentration metrics, but not uniformly the
   single largest parent.
4. Hard-cap diagnostic shows the fallback/quality-debt tradeoff.
5. Semantic coherence is not materially damaged in the available-text subset.
6. Branch-adaptive local-gamma probing remains diagnostic/future work.

Primary claims: C3, C4, C5, C6, C7, C8, C9, C10.

Quantitative spine:

- Six-field source-level evidence: `two_stage_quality_first` mean exact
  `Delta Q = 320.028`; lower source max/target ratio in `13/18` pairs; lower
  source oversize count in `5/18` pairs.
- Six-field next-level evidence: lower mean next-level max/target ratio by
  `0.01029`, oversize count by `0.167`, Gini by `0.00109`, and parent
  max-child share by `0.000873` across `54` source-seed/next-seed pairs.
- Strict seed sweep caveat: next-level Gini and parent concentration improve in
  `9/9` pairs, while max/target ratio improves in `0/9` pairs.
- Semantic sanity check: weighted doc-centroid coherence is non-decreasing in
  `15/18` pairs, with uneven text coverage.

### 8. Discussion

Purpose:

- Present Dongdaemun as a Leiden-compatible macro-refinement neighborhood.
- Explain the analogy to Leiden over Louvain: a search-procedure improvement
  that preserves the quality objective.
- Discuss why hierarchy construction needs upper-tail audits even when a flat
  partition quality objective is unchanged.
- Discuss limitations:
  conservative postprocess only, six fields, no integrated loop validation,
  no semantic optimization, and no strict target guarantee.

Primary claims: C2, C4, C10, C11.

### 9. Conclusion

Purpose:

- Restate that objective-consistent macro-refinement improves hierarchical CPM
  science-map readiness without changing CPM.
- Emphasize that the method reduces oversized-community pressure while keeping
  exact original-graph CPM accounting conservative.
- Close with integrated-loop Dongdaemun and branch-adaptive candidate generation
  as future work.

Primary claims: C2, C3, C4, C11.

## Claims To Avoid

- Dongdaemun guarantees strict target satisfaction.
- Tiny merging improves CPM quality.
- Adaptive gamma or dendrogram thresholding is validated as the main method.
- Semantic coherence is optimized or used as an acceptance criterion.
- Integrated Leiden-loop Dongdaemun has already been empirically validated.
- Hard-cap is the default scientific policy.

## Deliverable Dependencies

- Evidence map: `docs/research/dongdaemun/manuscript/dongdaemun_evidence_map.md`
- Figure and table plan: `docs/research/dongdaemun/manuscript/dongdaemun_figure_table_plan.md`
- Algorithm design: `docs/research/dongdaemun/core/dongdaemun_algorithm_design.md`
- Reproducibility appendix: `docs/research/dongdaemun/manuscript/dongdaemun_reproducibility_appendix.md`
