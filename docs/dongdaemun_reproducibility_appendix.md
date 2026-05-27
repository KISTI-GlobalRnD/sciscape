# Dongdaemun Reproducibility Appendix

Frozen evidence root:

`research/consensus/results/scientometrics_evidence_freeze_20260504`

This appendix describes what is frozen for the Dongdaemun paper pack and how to
audit references from the manuscript-facing documents.

## Bundle Summary

The freeze is a curated evidence bundle for the Scientometrics manuscript track.
It contains:

- `155` copied files.
- `0` missing required files.
- `24` main evidence files.
- `124` supplementary files.
- `7` copied manuscript/research-note documents.
- `2394` unselected validation files indexed in the archive.

Machine-readable metadata:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/bundle_summary.json`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/MANIFEST.csv`

The bundle was copied from validation results under:

`research/consensus/results/adaptive_refinement/hierarchy_postprocess_validation`

## Manifest And Checksums

Use `MANIFEST.csv` as the source of truth for copied file integrity. Each row
contains:

- `source_root`
- `source_relative_path`
- `output_relative_path`
- `category`
- `role`
- `status`
- `size_bytes`
- `sha256`

Manifest path:

`research/consensus/results/scientometrics_evidence_freeze_20260504/MANIFEST.csv`

The manuscript should cite frozen bundle paths, not transient run-directory
paths, unless a method appendix explicitly discusses archived unselected
artifacts.

## Frozen Main Evidence

Main reports:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/paper_evidence_brief.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/source_validation_readme.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/field_expansion_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/reports/actual_next_level_report.md`

Main tables:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/policy_comparison.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/policy_comparison.parquet`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table1_policy_comparison.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/table2_failure_taxonomy.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/main/tables/failure_taxonomy_summary.csv`
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

## Frozen Supplementary Evidence

Stress tests and robustness reports:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/next_level_target_sweep_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/next_level_seed_sweep_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/source_seed_sweep_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/source_seed_next_level_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/reports/semantic_coherence_report.md`

Supplementary figures:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure3_contraction_precondition.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure5_next_level_target_sweep.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure6_seed_stability_deltas.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure7_source_seed_pilot.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/figures/figure8_source_seed_next_level_propagation.png`

Same-gamma extension:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/iterative_quality_first_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/iterative_quality_first_compute_summary.json`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_effects.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_passes.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_candidates.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_vs_current.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/same_gamma_extension/tables/iterative_quality_first_policy_summary.csv`

Branch-adaptive diagnostics:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/branch_adaptive_diagnostics_report.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/branch_adaptive_compute_summary.json`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/figure12_branch_adaptive_tau_sensitivity.png`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_split_candidates.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_parent_summary.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_tau_sensitivity.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_tau_candidate_selection.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_candidate_stability.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_policy_effects.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/supplementary/branch_adaptive/tables/branch_adaptive_quality_first_vs_current.csv`

## Fields And Seeds

Frozen six-field source-level evidence:

- Field IDs: `12`, `15`, `18`, `26`, `30`, `34`.
- Samples:
  - `field12_gcc_emb_full_knn30`
  - `field15_gcc_emb_full_knn30`
  - `field18_gcc_emb_full_knn30`
  - `field26_gcc_emb_full_knn30`
  - `field30_gcc_emb_full_knn30`
  - `field34_combo_dc_bc_cc_sum`
- Source seeds: `11`, `42`, `73`.
- Source-level quality-first pairs: `18`.

Frozen next-level propagation evidence:

- Next-level seeds: `11`, `42`, `73`.
- Six-field source-seed/next-seed pairs: `54`.
- Initial source-seed/next-seed propagation pilot pairs: `27`.

Branch-adaptive diagnostic scope:

- Fields: `26`, `30`.
- Source seeds: `11`, `42`, `73`.
- Candidate rows: `72`.
- Parent rows: `8`.

Same-gamma extension scope:

- Fields: `12`, `26`, `30`.
- Source seeds: `11`, `42`, `73`.
- Effect rows: `36`.
- Candidate-selection rows: `522`.

## Policy Variants

Manuscript-facing policies:

- `raw`: initial Leiden output before hierarchy postprocess.
- `small_only`: fixed tiny-cluster consolidation only.
- `oversize_split_only`: split-repair candidate application without the full
  two-stage policy.
- `two_stage_quality_first`: Dongdaemun-post default evidence policy.
- `two_stage_hard_cap`: strict cap variant that falls back unless quality and
  target constraints both pass.
- `two_stage_hard_cap_aggressive`: stricter diagnostic variant in the initial
  policy table.

Supplementary/future policies:

- `iterative_quality_first`
- `iterative_quality_first_plus_trim`
- `target_reaching_trim_diagnostic`
- `branch_adaptive_quality_first`

## Frozen Results

Results treated as frozen for manuscript planning:

- Initial policy comparison: accepted rates, fallback rates, target satisfaction
  rates, and exact `Delta Q` summaries in `table1_policy_comparison.md` and
  `policy_comparison.csv`.
- Six-field source-level evidence in
  `field_expansion_source_seed_quality_first_vs_small_only.csv`.
- Six-field next-level propagation evidence in
  `field_expansion_source_seed_next_level_quality_first_vs_small_only_summary.csv`.
- Field heterogeneity in `field_expansion_field_breakdown.csv`.
- Target and seed stress tests in supplementary target/seed sweep tables.
- Semantic coherence sanity check in semantic tables and reports.
- Failure taxonomy in `table2_failure_taxonomy.md` and
  `failure_taxonomy_summary.csv`.
- Hard-cap diagnostic summaries in supplementary hard-cap tables.
- Same-gamma and branch-adaptive diagnostics in their supplementary folders.

Results intentionally not treated as main validated claims:

- Strict cap satisfaction by `two_stage_quality_first`.
- Quality improvement from tiny-cluster consolidation.
- Integrated loop-level Dongdaemun runtime or quality improvements.
- Validated adaptive gamma or branch-adaptive threshold selection.
- Semantic optimization.

## Archive Exclusions

The archive directory indexes validation artifacts that were not copied into the
curated main/supplementary set. These artifacts are intentionally excluded from
the paper-facing evidence bundle to keep the freeze compact.

Archive paths:

- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/README.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/original_artifact_index.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/run_directory_index.csv`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/archive/unselected_validation_artifacts.csv`

Use archive indexes only to trace provenance or recover a run directory for
follow-up analysis. Do not cite unselected artifacts as main manuscript evidence
unless they are promoted into a new dated freeze.

## Reproduction Checklist

Before submitting or sharing a manuscript draft:

1. Verify every path cited in manuscript-facing docs exists relative to repo
   root.
2. Verify every cited frozen artifact has a row in `MANIFEST.csv` unless it is a
   code anchor such as `sciscape/clustering/hierarchy_postprocess.py`.
3. Verify `bundle_summary.json` reports `missing_required_count = 0`.
4. Confirm the six-field source table includes fields `12`, `15`, `18`, `26`,
   `30`, and `34`.
5. Confirm source and next-level seed descriptions use `11`, `42`, and `73`.
6. Confirm rejected hard-cap memberships are labeled diagnostic where cited.
7. Confirm semantic coherence text is described as available-text sanity checking
   only.
8. Confirm branch-adaptive and integrated-loop Dongdaemun are labeled future
   work.

## Suggested Validation Commands

List frozen files:

```bash
find research/consensus/results/scientometrics_evidence_freeze_20260504 -type f | sort
```

Check bundle summary:

```bash
cat research/consensus/results/scientometrics_evidence_freeze_20260504/bundle_summary.json
```

Check manifest rows:

```bash
head -n 5 research/consensus/results/scientometrics_evidence_freeze_20260504/MANIFEST.csv
```

Run tests only if code changes are made after this documentation pack:

```bash
pytest -q
```
