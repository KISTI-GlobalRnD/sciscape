# Dongdaemun Figure And Table Plan

Frozen evidence root:

`research/consensus/results/scientometrics_evidence_freeze_20260504`

The main paper should prioritize the algorithm, source-level upper-tail repair,
and next-level propagation. Semantic and branch-adaptive evidence should stay
supplementary unless reviewers ask for more interpretability validation.

## Main Figures

| Figure | Placement | Draft caption | Source artifact |
| --- | --- | --- | --- |
| Figure 1. Dongdaemun-post pipeline | Main | Conservative Dongdaemun-post workflow for hierarchical CPM maps. Raw Leiden memberships are first passed through fixed lower-tail consolidation, then oversized communities trigger split-repair and boundary-polish proposals. Accepted macro moves must pass exact original-graph CPM accounting before projection and contraction. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure1_two_stage_pipeline.png` |
| Figure 2. Source-level size-quality tradeoff | Main | Source-level effect of oversize refinement policies. `two_stage_quality_first` reduces upper-tail size imbalance relative to `small_only` while preserving positive exact CPM accounting in the observed runs; strict target satisfaction is not guaranteed. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure2_size_quality_tradeoff.png` |
| Figure 3. Six-field source-seed expansion | Main | Six-field source-level validation over fields 12, 15, 18, 26, 30, and 34 with source seeds 11, 42, and 73. The quality-first policy lowers source max/target ratio in most pairs while keeping exact CPM deltas positive on average. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure9_field_expansion_source_seed.png` |
| Figure 4. Six-field next-level propagation | Main | Propagation of source-level Dongdaemun-post memberships into next-level contraction and Leiden reruns across source and next-level seeds. Improvements are strongest for concentration metrics and are not a uniform guarantee on the single largest next-level parent. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure10_field_expansion_next_level.png` |

## Conditional Main Or Supplementary Figures

| Figure | Recommended placement | Draft caption | Source artifact |
| --- | --- | --- | --- |
| Actual next-level propagation | Supplementary, or Main if a shorter three-field propagation figure is needed | Actual next-level contraction, deterministic gamma sweep, Leiden rerun, and small-cluster repair for the initial evidence set. The adaptive next-level target removes oversize for all effective policies, so the figure is best used as context rather than as the strongest propagation claim. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure4_actual_next_level_propagation.png` |
| Semantic coherence sanity check | Supplementary | Available-title/abstract TF-IDF coherence comparison between `small_only` and `two_stage_quality_first`. The result checks that quality-first does not materially damage available-text coherence; it is not an optimization criterion. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/figures/figure11_semantic_coherence.png` |

## Supplementary Figures

| Figure | Placement | Draft caption | Source artifact |
| --- | --- | --- | --- |
| Contraction precondition | Supplementary | Supernode-weight distribution before a fresh next-level Leiden rerun. This figure supports the hierarchy-readiness motivation for upper-tail repair. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure3_contraction_precondition.png` |
| Next-level target sweep | Supplementary | Strict target-multiplier stress test. Quality-first reduces concentration and oversize pressure under the strictest target but does not uniformly reduce the largest next-level parent. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure5_next_level_target_sweep.png` |
| Seed stability deltas | Supplementary | Strict 1x next-level seed sweep over seeds 11, 42, and 73. Gini and parent concentration improve consistently, while max/target ratio does not. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure6_seed_stability_deltas.png` |
| Source-seed pilot | Supplementary | Initial source-seed pilot over fields 12, 15, and 34. Quality-first lowers source max/target ratio in all nine source-seed pairs and keeps positive exact CPM accounting. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure7_source_seed_pilot.png` |
| Source-seed next-level propagation pilot | Supplementary | Initial 27-pair source-seed/next-seed propagation pilot. Concentration metrics improve more consistently than the single largest next-level parent. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure8_source_seed_next_level_propagation.png` |
| Branch-adaptive tau sensitivity | Supplementary/future-work appendix | Branch-adaptive local-gamma diagnostic sweep. No tau setting selected accepted candidates in the frozen pilot, so this figure documents a negative diagnostic result rather than a validated method. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/figure12_branch_adaptive_tau_sensitivity.png` |

## Main Tables

| Table | Placement | Draft caption | Source artifact |
| --- | --- | --- | --- |
| Table 1. Policy comparison | Main | Policy-level comparison for the initial validation set. `two_stage_quality_first` accepts all observed runs and improves the effective max/target ratio relative to `small_only`; hard-cap accepts fewer runs and falls back more often. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table1_policy_comparison.md` |
| Table 2. Six-field source and propagation summary | Main | Six-field summary over fields 12, 15, 18, 26, 30, and 34. Report source-level exact `Delta Q`, source max/target deltas, next-level Gini deltas, and parent max-child-share deltas with counts of improving pairs. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_source_seed_quality_first_vs_small_only.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_source_seed_next_level_quality_first_vs_small_only_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/field_expansion_field_breakdown.csv` |
| Table 3. Failure taxonomy | Main or Supplementary depending on page budget | Empirical taxonomy of residual oversized cases and fallback modes. The table makes residual oversize auditable rather than treating it as an implementation failure. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md` |

## Supplementary Tables

| Table | Placement | Draft caption | Source artifact |
| --- | --- | --- | --- |
| Full policy comparison CSV | Supplementary data | Machine-readable version of Table 1 with accepted rates, fallback rates, target satisfaction rates, and reported/effective max ratios. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/policy_comparison.csv` |
| Next-level target sweep | Supplementary | Stress-test comparison across source-level target multipliers. Supports the caveat that concentration improves more reliably than max/target ratio. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/next_level_target_sweep_quality_first_vs_small_only.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/next_level_target_sweep_policy_comparison.csv` |
| Next-level seed stability | Supplementary | Strict 1x next-level seed stability over next-level seeds 11, 42, and 73. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/next_level_seed_sweep_quality_first_vs_small_only_summary.csv` |
| Source-seed propagation pilot | Supplementary | Initial source-seed and next-seed propagation evidence across fields 12, 15, and 34. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/source_seed_next_level_quality_first_vs_small_only_summary.csv` |
| Hard-cap diagnostics | Supplementary | Diagnostic hard-cap balance changes and fallback tradeoffs. These rows should be labeled rejected/diagnostic where applicable. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/field_expansion_source_seed_hard_cap_diagnostic_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/field_expansion_source_seed_next_level_hard_cap_diagnostic_summary.csv` |
| Semantic coherence | Supplementary | Available-text semantic coherence sanity check by field and policy. | `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/semantic_coherence_quality_first_vs_small_only.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/tables/semantic_coherence_field_breakdown.csv` |
| Same-gamma extension | Supplementary/future-work appendix | Same-gamma iterative repair plus conservative boundary trim. Shows that the current quality-first behavior matches `iterative_quality_first_plus_trim` in the pilot. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_policy_summary.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_vs_current.csv` |
| Branch-adaptive diagnostics | Supplementary/future-work appendix | Candidate and tau diagnostics for branch-adaptive local-gamma probing. Use only as future-work evidence. | `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_tau_sensitivity.csv`; `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_quality_first_vs_current.csv` |

## Main Versus Supplementary Decision

Keep main:

- Algorithm/pipeline figure.
- Source-level size-quality figure.
- Six-field source expansion figure.
- Six-field next-level propagation figure.
- Policy comparison table.
- Six-field source/propagation summary table.

Move to supplementary by default:

- Actual next-level figure for the initial evidence set.
- Target and seed sweeps.
- Source-seed pilot figures.
- Semantic coherence figure and tables.
- Hard-cap diagnostic details.
- Same-gamma extension.
- Branch-adaptive diagnostics.

This keeps the main manuscript focused on Dongdaemun as quality-audited
macro-refinement, while preserving diagnostics that explain caveats and future
work.
