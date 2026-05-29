# SciScape Data Retention And Archive Plan

Status: reorganization draft
Date: 2026-05-27
Scope: documentation-only. No files have been moved or deleted.

This plan classifies local research outputs into keep, consolidate, archive,
and drop-candidate groups. It is meant to be reviewed before any physical file
move. The companion track map is `research/PROJECT_TRACKS.md`.
The companion failure ledger is `research/FAILED_DIRECTIONS.md`.

## Retention Labels

| Label | Meaning | Action now |
| --- | --- | --- |
| `KEEP-LIVE` | Active paper or R&D evidence that should stay in place. | Keep path unchanged. |
| `KEEP-SUMMARY` | Keep reports, summaries, manifest, and selected tables; raw traces can move later. | Write manifest before moving raw files. |
| `CONSOLIDATE` | Multiple runs tell the same story; retain the representative artifact and index the rest. | Pick canonical artifact first. |
| `ARCHIVE` | Useful audit or negative evidence, but not needed in the active working set. | Move/compress only after manifest approval. |
| `DROP-CANDIDATE` | Empty, accidental, or fully superseded local artifact. | Delete only after explicit approval. |

## Archive Workflow

Before any file move:

1. Create a manifest with these columns:
   `path,size,track,label,reason,representative_summary,rerun_command`.
2. Confirm each `KEEP-SUMMARY` directory has at least one retained report,
   summary JSON, or CSV table.
3. Confirm large raw traces are either reproducible or not needed for the next
   claim.
4. Move only reviewed `ARCHIVE` paths into an archive root such as
   `research/archive/2026-05-track-reorg/`.
5. Never archive the only copy of a paper-facing frozen bundle.

## Size Snapshot

Current result sizes observed on 2026-05-27:

- `research/consensus/results/adaptive_refinement/`: about `8.4G`.
- `research/consensus/results/scientometrics_evidence_freeze_20260504/`:
  about `8.5M`.
- `research/dendrogram/results/`: about `856K`.

Large adaptive-refinement directories:

- `leiden_hysteresis_exception_detector_graphs_20260514`: about `3.4G`.
- `hierarchy_postprocess_validation`: about `805M`.
- `leiden_hysteresis_multifidelity_candidate_trajectory_cc11_20260513`:
  about `421M`.
- `leiden_hysteresis_multifidelity_candidate_trajectory_cc11_footprint_20260514`:
  about `421M`.
- `dongdaemun_adaptive_local_shake_pilot_20260512`: about `383M`.
- `dongdaemun_trajectory_divergence_pilot_*`: about `348M` each.
- `dongdaemun_adaptive_local_shake_up_only_ablation_20260512`: about `287M`.
- `dongdaemun_safe_fast_layer_comparison`: about `260M`.
- `dongdaemun_refinement_qs_profile`: about `237M`.
- `leiden_hysteresis_work_acceleration_monitor_v2_20260513`: about `189M`.

## Track A: Multi-Layer Consensus Boundary Signal

Decision: mostly keep. This track is small and paper-facing.

### KEEP-LIVE

- `research/consensus/README.md`
- `research/consensus/ABSTRACT.md`
- `research/consensus/MANUSCRIPT_OUTLINE.md`
- `research/consensus/RESULTS_NOTES.md`
- `research/consensus/CAPTIONS.md`
- `research/consensus/SUBMISSION_STRATEGY.md`
- `research/consensus/figures/manuscript_joi_v1/`
- `research/consensus/results/case_banks_corrected/`
- `research/consensus/results/taxonomy_corrected/`
- `research/consensus/results/cross_field_round2/`

### CONSOLIDATE

- `research/consensus/results/case_banks/`
- `research/consensus/results/taxonomy/`
- `research/consensus/results/local_review_field12/`
- `research/consensus/results/local_review_field15/`
- `research/consensus/results/ab_review_field15/`
- `research/consensus/results/ablation_compare_field15/`

Reason: these are audit or predecessor outputs for the corrected local-review
story. Keep them indexed, but treat corrected case banks and corrected taxonomy
as the active reference.

### ARCHIVE

- `research/consensus/results/k_sweep_first_pass/`
- `research/consensus/results/pilot_field15/`
- `research/consensus/results/controls/`
- root-level early JSONs under `research/consensus/results/` after their
  values are copied into an audit index.

Reason: useful history, but not the current manuscript-facing evidence unless
explicitly cited.

## Track B: Dongdaemun-Post Hierarchy Repair Method

Decision: keep paper-facing bundle and compact summaries; archive bulky raw
validation runs after summary coverage is confirmed.

### KEEP-LIVE

- `docs/dongdaemun_evidence_map.md`
- `docs/dongdaemun_naming_contract.md`
- `docs/dongdaemun_manuscript_plan.md`
- `docs/dongdaemun_reproducibility_appendix.md`
- `docs/hierarchy_postprocess_research_roadmap.md`
- `docs/methodology_final_design.md`
- `research/consensus/results/scientometrics_evidence_freeze_20260504/`
- `research/consensus/results/adaptive_refinement/split_merge_repair_cross_sample_summary.json`
- `research/consensus/results/adaptive_refinement/split_merge_repair_success_failure_diagnostics.json`
- `research/consensus/results/adaptive_refinement/split_repair_predictor_screening.json`
- `research/consensus/results/adaptive_refinement/external_grain_vs_split_repair_comparison.json`

### KEEP-SUMMARY

- `research/consensus/results/adaptive_refinement/hierarchy_postprocess_validation/`
- `research/consensus/results/adaptive_refinement/field12_gcc_split_repair_apply_pilot_top50/`
- `research/consensus/results/adaptive_refinement/rust_dongdaemun_fast_path_validation/`
- `research/consensus/results/adaptive_refinement/g016_post_g00085_split_merge_repair_probe_giant_top300/`
- `research/consensus/results/adaptive_refinement/g016_post_g0015_split_merge_repair_probe_giant_top300/`
- `research/consensus/results/adaptive_refinement/g016_post_g003_split_merge_repair_probe_giant_top300/`
- `research/consensus/results/adaptive_refinement/g016_post_g003_external_grain_probe_giant_top300/`
- `research/consensus/results/adaptive_refinement/bcrefresh_g0005_split_merge_repair_probe_prepared_top300/`

Reason: these are important validation or scaling evidence, but many raw
artifacts are not needed in the active working set once summary reports and
tables are indexed.

### CONSOLIDATE

- `research/consensus/results/adaptive_refinement/field12_postprocess_policy_matrix/`
- `research/consensus/results/adaptive_refinement/postprocess_policy_matrix_cross_sample/`
- `research/consensus/results/adaptive_refinement/field15_auto_fast_*`
- `research/consensus/results/adaptive_refinement/field30_auto_fast_*`
- `research/consensus/results/adaptive_refinement/field34_auto_fast_*`

Reason: these represent policy matrix exploration. Keep representative
quality-first and hard-cap diagnostic summaries, then archive redundant policy
variants.

### ARCHIVE

- early `g016_gamma0p0085_boundary_*` macro/boundary dry-runs after summaries
  are retained;
- weak band-focused split-repair probes that are superseded by cross-sample
  summaries;
- Rust timing-only artifacts that do not support a separate method claim.

## Track C: Adaptive Refinement And Basin-Tunneling R&D

Decision: keep the central c0/c2 basin evidence root and active scripts; archive
large raw trace families and negative operator branches after summaries are
indexed.

### KEEP-LIVE

Core docs and code:

- `research/consensus/TODO_SCISCI_ADAPTIVE_REFINEMENT.md`
- `docs/research/leiden_basin/README.md`
- `docs/research/leiden_basin/leiden_basin_cartography_redesign.md`
- `docs/research/leiden_basin/leiden_basin_data_inventory.md`
- `docs/research/leiden_basin/leiden_basin_existing_data_review.md`
- `docs/research/leiden_basin/leiden_basin_methodology_v0_design.md`
- `docs/leiden_multibasin_research_guardrails.md`
- `docs/dongdaemun_basin_transition_operator_design.md`
- `sciscape/clustering/leiden_basin_profile.py`
- `sciscape/clustering/leiden_basin_search.py`
- `research/consensus/scripts/leiden_basin/`

Current terminal evidence:

- `research/consensus/results/adaptive_refinement/leiden_basin_methodology_v0_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_existence_assumption_audit_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_remaining_wall_question_audit_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_cycle_closure_writeup_20260529/`

Representative basin-definition and route-gate chain:

- `research/consensus/results/adaptive_refinement/leiden_basin_phase1_index_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_phase1_review_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_definition_calibration_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_wall_protocol_panel_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_route_gate_panel_combined_after_clean_distinct_20260528/`
- `research/consensus/results/adaptive_refinement/leiden_basin_route_label_interpretation_v0_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_relation_boundary_rule_review_20260529/`
- `research/consensus/results/adaptive_refinement/leiden_basin_field34_evidence_eligibility_audit_20260529/`

Legacy support/signature roots that remain active references:

- `research/consensus/results/adaptive_refinement/leiden_multibasin_crossfield_budget12_support_20260519/`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_signature_field30_budget12_support_20260519/`
- `research/consensus/results/adaptive_refinement/leiden_multibasin_signature_field26_citation_embedding_budget15_support_20260519/`

Detailed intermediate Track C artifacts remain documented in
`docs/research/leiden_basin/leiden_basin_data_inventory.md`; promote individual
intermediate result directories back into `KEEP-LIVE` only if a current claim,
reopen gate, or reproducibility bundle cites them directly.

### KEEP-SUMMARY

- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_budget123_20260513/`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_parallel_memory_benchmark_memopt_summary_20260514/`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_parallel_memory_benchmark_summary_20260514/`
- `research/consensus/results/adaptive_refinement/previous_results_review_20260518/`

Reason: these support cost and instrumentation interpretation, but should not
stay as independent research tracks.

### CONSOLIDATE

- `research/consensus/results/adaptive_refinement/leiden_multibasin_signature_*`
- `research/consensus/results/adaptive_refinement/leiden_approx_polish_label_*`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_multifidelity_label_*`

Reason: keep the best support/signature evidence and index the rest as
predecessor signal or stress validation.

### ARCHIVE

Archive after representative summaries and rerun commands are captured:

- `research/consensus/results/adaptive_refinement/leiden_hysteresis_exception_detector_graphs_20260514/`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_multifidelity_candidate_trajectory_cc11_20260513/`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_multifidelity_candidate_trajectory_cc11_footprint_20260514/`
- `research/consensus/results/adaptive_refinement/dongdaemun_adaptive_local_shake_pilot_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_adaptive_local_shake_structure_compare_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_adaptive_local_shake_up_only_ablation_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_adaptive_stochastic_greedy_prototype_20260511/`
- `research/consensus/results/adaptive_refinement/dongdaemun_adaptive_stochastic_greedy_prototype_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_trajectory_divergence_pilot_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_trajectory_divergence_pilot_v2_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_trajectory_divergence_pilot_v3_20260512/`
- `research/consensus/results/adaptive_refinement/dongdaemun_refinement_qs_profile/`
- `research/consensus/results/adaptive_refinement/dongdaemun_safe_fast_layer_comparison/`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_20260513/`
- `research/consensus/results/adaptive_refinement/leiden_hysteresis_work_acceleration_monitor_v2_smoke/`

Reason: these are bulky exploratory traces or superseded mechanism probes. They
remain useful as audit history, but they should not define the active research
surface.

### NEGATIVE CONTROLS TO KEEP INDEXED

- direct-node closure shrink;
- label-internal repair;
- stage2 local recovery;
- c2 branch rows labeled already-recovered control;
- gate-only recovery rows that require broad context without material quality
  success.

Reason: these prevent re-running dead ends. Retain summary rows even if raw
artifacts are archived.

## Deferred Track: Hybrid CPM-Critical Dendrogram

Decision: keep small pilot outputs in place.

### KEEP-LIVE

- `research/dendrogram/README.md`
- `research/dendrogram/scripts/`
- `research/dendrogram/results/pilot_field15/`

Reason: the whole track is small, separate, and not currently blocking active
Dongdaemun/adaptive work.

## DROP-CANDIDATE

These should not be removed without explicit approval:

- `C`

Reason: empty file at repo root, likely accidental.

## Next Physical Cleanup Step

Do not move data yet. The next safe step is to generate a machine-readable
manifest:

```bash
python3 scripts/research_retention_manifest.py
```

That script does not exist yet. If created, it should only read file sizes and
write a CSV manifest; it should not move or delete anything.

The manifest should include a `failure_id` column for archived negative-control
artifacts when they correspond to entries in `research/FAILED_DIRECTIONS.md`.
