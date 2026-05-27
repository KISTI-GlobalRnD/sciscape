# Dongdaemun Evidence Map

Frozen evidence root:

`research/consensus/results/scientometrics_evidence_freeze_20260504`

This map audits the claim IDs from `docs/dongdaemun_manuscript_plan.md`.
Statuses mean:

- `supported`: the frozen bundle directly supports the claim within the stated
  boundary.
- `partially supported`: the bundle supports a directional or limited version of
  the claim, with important caveats.
- `diagnostic only`: the artifact is useful for explanation or future design,
  not for a main validated claim.
- `future work`: the claim is a design target, not validated by the frozen
  evidence.

## Main Evidence Claims

| Claim | Status | Evidence readout | Primary artifacts | Boundary |
| --- | --- | --- | --- | --- |
| C1 | supported | Oversized source communities and next-level concentration are repeatedly measured and reported as hierarchy-readiness risks. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/source_validation_readme.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/policy_comparison.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure2_size_quality_tradeoff.png` | Oversize is evidence of a missed macro-refinement opportunity or operational risk, not proof that the CPM objective is wrong. |
| C2 | supported | The study policies keep CPM as the audit objective and report exact original-graph quality deltas for accepted postprocess memberships. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/paper_evidence_brief.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/source_validation_readme.md`; `sciscape/clustering/hierarchy_postprocess.py` | This supports the Dongdaemun-post audit principle. It does not validate integrated loop-level behavior. |
| C3 | supported | In six fields, `two_stage_quality_first` has mean source-level exact `Delta Q = 320.028`, lowers source max/target ratio in `13/18` pairs, and lowers source oversize count in `5/18` pairs. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/field_expansion_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_source_seed_quality_first_vs_small_only.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure9_field_expansion_source_seed.png` | Source-level Gini is not a standalone success metric because split repair can intentionally create small clusters. |
| C4 | partially supported | Six-field next-level propagation lowers mean max/target ratio by `0.01029`, oversize count by `0.167`, Gini by `0.00109`, and parent max-child share by `0.000873`; however, strict seed sweeps show max/target is not uniformly improved. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/field_expansion_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_source_seed_next_level_quality_first_vs_small_only_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure10_field_expansion_next_level.png` | The defensible wording is reduced concentration and oversize pressure, not guaranteed max-parent improvement. |
| C5 | supported | `small_only` is the lower-tail consolidation baseline and comparison point for oversize repair. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table1_policy_comparison.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/policy_comparison.csv`; `sciscape/README.md` | The bundle supports tiny repair as an operational policy, not as the main quality-improving contribution. |
| C6 | supported | Hard-cap accepts only `1/3` initial runs and `3/18` six-field source rows; it falls back often, while diagnostics show balance tradeoffs. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table1_policy_comparison.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/field_expansion_source_seed_hard_cap_diagnostic_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/field_expansion_source_seed_next_level_hard_cap_diagnostic_summary.csv` | Hard-cap may be useful operationally, but it should not be the default manuscript method. |
| C7 | supported | Available-text TF-IDF coherence is non-decreasing in `15/18` pairs and has a small mean weighted doc-centroid delta of `0.000069`. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure11_semantic_coherence.png`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/semantic_coherence_quality_first_vs_small_only.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/semantic_coherence_report.md` | Text coverage is uneven; semantic coherence is a sanity check only. |
| C8 | supported | Residual cases are summarized with tags including `insufficient_split_repair_candidates`, `boundary_too_dense_or_receiver_cap`, `semantic_core_cluster`, `quality_floor_limited`, `fallback_required`, and `resolved`. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/failure_taxonomy_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/failure_taxonomy.csv` | The taxonomy is empirical and first-pass. |

## Supplementary Evidence Claims

| Claim | Status | Evidence readout | Supplementary artifacts | Boundary |
| --- | --- | --- | --- | --- |
| C3 | supported | Initial source-seed pilot: quality-first lowers source max/target ratio in `9/9` pairs and has mean exact `Delta Q = 523.941`. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/source_seed_sweep_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/source_seed_sweep_quality_first_vs_small_only.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure7_source_seed_pilot.png` | This is a pilot over fields 12, 15, and 34; the six-field table is the main source-level evidence. |
| C4 | partially supported | Strict target and seed sweeps show consistent concentration improvements but mixed max/target behavior. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/next_level_target_sweep_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/next_level_seed_sweep_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure5_next_level_target_sweep.png`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure6_seed_stability_deltas.png` | These artifacts justify caveated propagation language. |
| C4 | partially supported | Initial source-seed next-level propagation lowers Gini and parent max-child share in `24/27` pairs, max/target ratio in `12/27` pairs. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/source_seed_next_level_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/source_seed_next_level_quality_first_vs_small_only_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure8_source_seed_next_level_propagation.png` | Supplementary robustness for the propagation claim. |
| C9 | supported | Same-gamma extension shows `iterative_quality_first_plus_trim` matches current `two_stage_quality_first`, while trim-free ablation leaves more oversize pressure. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/iterative_quality_first_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_policy_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_vs_current.csv` | This supports iterative repair plus conservative trim, not adaptive gamma validation. |

## Diagnostic Evidence

| Claim | Status | Evidence readout | Diagnostic artifacts | Boundary |
| --- | --- | --- | --- | --- |
| C6 | diagnostic only | Hard-cap diagnostic rows sometimes improve strict balance but often depend on rejected or fallback-prone memberships. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/next_level_target_sweep_hard_cap_diagnostics.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/next_level_seed_sweep_hard_cap_diagnostic_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/source_seed_sweep_hard_cap_diagnostic_summary.csv` | Use for tradeoff explanation only. |
| C7 | diagnostic only | Semantic metrics provide an available-text post-hoc sanity check. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/semantic_coherence_cluster_metrics.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/semantic_coherence_field_breakdown.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/metadata/semantic_coherence_summary.json` | Not an acceptance criterion. |
| C10 | diagnostic only | Branch-adaptive pilot generated `72` candidate rows over fields `26` and `30`, but no tau setting selected accepted candidates. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/branch_adaptive_diagnostics_report.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/figure12_branch_adaptive_tau_sensitivity.png`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_tau_sensitivity.csv` | Negative/diagnostic result; not a validated method. |
| C8 | diagnostic only | Failure-mode tags guide local resolution probing and boundary diagnostics. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/original_artifact_index.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/unselected_validation_artifacts.csv` | Archive files index unselected artifacts; they are not main paper evidence by themselves. |

## Future-Work Evidence

| Claim | Status | Evidence readout | Artifacts or code anchors | Boundary |
| --- | --- | --- | --- | --- |
| C10 | future work | Branch-adaptive gamma is motivated by failure modes and diagnostics, but current selected-candidate rate is zero in the frozen pilot. | `docs/branch_adaptive_quality_first_research_note.md`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/branch_adaptive_diagnostics_report.md` | Present as future candidate generation work only. |
| C11 | future work | Integrated loop-level Dongdaemun is now specified as parent-internal adaptive refinement allocation, not direct postprocess commits inside Leiden. | `docs/dongdaemun_refinement_algorithm_design.md`; `docs/dongdaemun_algorithm_design.md`; `sciscape/clustering/hierarchy_postprocess.py`; `sciscape/clustering/leiden_rust.py` | No integrated-loop empirical claim should appear in Results. |

## Artifact Registry

Main reports:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/paper_evidence_brief.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/source_validation_readme.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/field_expansion_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/actual_next_level_report.md`

Main tables:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/policy_comparison.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table1_policy_comparison.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_source_seed_quality_first_vs_small_only.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_source_seed_next_level_quality_first_vs_small_only_summary.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_field_breakdown.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/semantic_coherence_quality_first_vs_small_only.csv`

Main figures:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure1_two_stage_pipeline.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure2_size_quality_tradeoff.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure4_actual_next_level_propagation.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure9_field_expansion_source_seed.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure10_field_expansion_next_level.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure11_semantic_coherence.png`

Reproducibility metadata:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/MANIFEST.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/bundle_summary.json`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/README.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/original_artifact_index.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/run_directory_index.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/unselected_validation_artifacts.csv`
