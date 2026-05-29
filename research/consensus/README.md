# Research 1: Consensus as Boundary Signal in Multi-Layer Scientific Paper Graphs

## Hypotheses

- **H1**: At the same rank budget (`top_k`), consensus level predicts cluster-boundary precision better than single-layer edge strength.
- **H2**: Embedding layers substitute for CC in citation-poor fields.
- **H3**: The best combination separates roles: consensus captures boundaries, exclusive edges capture density/periphery.

## Current Scope

This folder is organized as an executable research package rather than a note dump.

- `scripts/` contains experiment runners and figure generation.
- `results/` stores JSON outputs from each experiment.
- `figures/` stores paper-ready charts generated from the result JSON files.
- The current code supports E1-E7 directly.

Script reorganization should follow
[`SCRIPT_STRUCTURE.md`](./SCRIPT_STRUCTURE.md). The script directory has many
path references in docs, tests, and artifact metadata, so run the inventory
check before moving files:

```bash
uv run --extra dev python scripts/inventory_research_scripts.py --fail-on-unclassified
```

## Manuscript Package

The current writing package is aligned to `Journal of Informetrics`.

- manuscript outline:
  - [MANUSCRIPT_OUTLINE.md](./MANUSCRIPT_OUTLINE.md)
- abstract drafts:
  - [ABSTRACT.md](./ABSTRACT.md)
- submission and journal-fit notes:
  - [SUBMISSION_STRATEGY.md](./SUBMISSION_STRATEGY.md)
- figure and table captions:
  - [CAPTIONS.md](./CAPTIONS.md)
- result-to-text notes:
  - [RESULTS_NOTES.md](./RESULTS_NOTES.md)
- GPU embedding filter runbook:
  - [GPU_EMB_FILTER_RUNBOOK.md](./GPU_EMB_FILTER_RUNBOOK.md)
- manuscript-ready figure bundle:
  - [figures/manuscript_joi_v1/README.md](./figures/manuscript_joi_v1/README.md)

## Current Corrected Status

As of 2026-04-22, the local-review pipeline should be interpreted using the
order-balanced `gemini_v3` outputs under `results/case_banks_corrected/`.

Main methodological corrections now reflected in code:
- `reviewer.py`
  - preserves explicit `A/B/TIE` winners from the judge instead of forcing
    `TIE` whenever `score_a == score_b`
  - records presented method order more explicitly in swapped A/B reviews
  - supports order-balanced dual-pass rerank review and checkpoint-safe retry
- `cluster_naming.py`
  - respects the client/model configuration in `summarise_cluster()`
  - uses a more robust Gemini fallback rule for env-based configuration
- `evaluation/__init__.py`
  - exports the newer review/sampling APIs used by the research scripts
- `sampler.py`
  - `n_cross_edges` now means literal cross-cluster edge count, not truncated
    cross-edge weight
- `_common.py`
  - `layer_top_k` metadata is now stable even when `top_k` is represented as a
    string such as `none`

Current order-balanced local-review reference files:
- `field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json`
- `field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json`
- `field_12_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json`

Current corrected win counts (`baseline / consensus / tie`):
- `field_15 k=6`: `12 / 19 / 17`
- `field_15 k=30`: `10 / 31 / 7`
- `field_12 k=6`: `3 / 33 / 12`

Working interpretation after correction:
- `field_15 k=6`: low-separation regime with substantial ambiguity, but stable
  non-tie cases still lean toward `consensus`
- `field_15 k=30`: clear `consensus` advantage after order balancing
- `field_12 k=6`: very strong `consensus` advantage after order balancing

Older single-pass and pre-repair review outputs are still kept for audit, but
they should not be treated as the current reference results.

## Data Assumptions

- Field layout: `data/linktype_edges_gcc/field_XX/*.parquet`
- Standard layers expected by the scripts:
  - `bc_cosine.parquet`
  - `cc_cosine.parquet`
  - `dc_fractional.parquet`
  - `emb_knn.parquet` or `emb_full_knn30.parquet`
- Metadata table for blind review:
  - `uid`, `title`, `abstract`, `pubyear`
  - `work_id` is also accepted and renamed to `uid`

## Experiment Design

### Protocols

- `Protocol A: practical_top_k`
  - use `top_k=30` on each participating layer
  - answers the practical deployment question
- `Protocol B: candidate_budget_matched`
  - use `effective_k` to split the pre-fusion candidate budget across layers
  - answers the reranking-fairness question
- `Protocol C: edge_count_matched`
  - anchor on the best Protocol A single-layer baseline
  - uniformly scale the consensus per-layer `top_k` vector until the final
    undirected edge count is matched within `±5%`
  - answers the first-order clustering-load question
- Interpretation rule:
  - `effective_k` is not a clustering-fairness guarantee
  - edge-count matching is only a first-order density normalization and does
    not remove higher-order topology differences

### E1: Same effective-k comparison

- Fix a global `effective_k=30` budget by default.
- Single-layer runs use the full budget.
- Multi-layer runs split that budget across participating layers, so the
  summed per-layer `top_k` matches the same effective-k target.
- Compare:
  - BC-only
  - CC-only
  - DC-only
  - Emb-only when present
  - citation-only consensus (`BC+CC+DC`)
  - all-layer consensus (`BC+CC+DC+Emb`)
- Outputs:
  - `*_comparison.json`
  - cluster count, max cluster %, AMI/ARI stability, singleton %, consensus edge mix
  - actual per-layer `top_k` allocation used by each method
  - `protocol` metadata recording `candidate_budget_matched` or `practical_top_k`

### E1C: Density-matched comparison

- Fix the Protocol A best single-layer method as the canonical baseline.
- Match the consensus graph to that baseline's final undirected edge count
  using uniform scaling of the Protocol A per-layer `top_k` vector.
- Acceptance criterion:
  - `abs(achieved_edge_count - target_edge_count) / target_edge_count <= 0.05`
- Outputs:
  - `*_density_matched_comparison.json`
  - target / achieved edge count and relative edge error
  - original Protocol A per-layer `top_k`
  - matched per-layer `top_k`
  - Protocol A anchor runs used to select the baseline

### E2: Leave-one-out ablation

- Start from the all-layer consensus run.
- Remove each layer one at a time.
- Outputs:
  - `*_leave_one_out.json`
  - Δ AMI, Δ cluster count, Δ max cluster %

### E3: Consensus level × cluster-structure analysis

- Stratify edges by agreement count across layers.
- Measure intra-cluster vs cross-cluster ratio at each consensus level.
- Outputs:
  - `*_consensus_tiers.json`
  - `*_consensus_tiers.txt`

### E4: Cross-field generalization

- Runs E1 across fields:
  - `field_15`, `field_12`, `field_18`, `field_26`, `field_29`, `field_30`, `field_34`
- Outputs:
  - per-field `*_comparison.json`
  - `cross_field_summary.json`
  - summary uses the best single-layer method in each field as the baseline,
    not BC-only by assumption

### E6: Effective-k sweep

- Sweep `effective_k` up to 30 for one field or across many fields.
- Purpose:
  - find where consensus starts to help or hurt
  - compare peak `k` by field
  - measure whether field-specific optima differ
- Sweep runners resume from any existing per-`k` JSON by default.
- Pass `--overwrite` to recompute completed points.
- Outputs:
  - `*_k_sweep.json`
  - `cross_field_k_sweep_summary.json`

### E7: Local rank-shift review

- Build two graph combinations on the same field and same `top_k` budget.
- Default comparison is `sum` vs `consensus`.
- Find target nodes whose top-ranked local neighbors change strongly between the two methods.
- Ask the LLM which ranked local neighborhood better captures the target's immediate research context.
- This is a mechanism-level check:
  - not "which whole cluster is better?"
  - but "which local reranking around this target looks more semantically correct?"
- Two modes:
  - `--sample-only`: builds the rank-shift review set without calling an LLM
  - default mode: runs blind reranking comparison
- Outputs:
  - `*_rank_shift_review.json`

### E5: Blind A/B review on disagreement boundary nodes

- Build two clusterings on the same field and same `top_k` budget.
- Default comparison is `sum` vs `consensus`.
- Sample hard boundary nodes where the two methods produce meaningfully
  different local same-cluster groups.
- The script falls back to lower boundary quantiles and looser overlap
  thresholds when the initial disagreement pool is empty.
- Two modes:
  - `--sample-only`: builds the disagreement review set without calling an LLM
  - default mode: runs blind `comparison` and `belonging` prompts
  - `--secondary-checks`: additionally scores group cohesion and outliers per method
- Outputs:
  - `*_boundary_review.json`
  - `boundary_review_summary.json` from the grid runner

## Scripts

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `scripts/consensus_core/baseline_comparisons/run_comparison.py` | E1 single-layer vs consensus comparison | `*_comparison.json` |
| `scripts/consensus_core/baseline_comparisons/run_leave_one_out.py` | E2 ablation study | `*_leave_one_out.json` |
| `scripts/consensus_core/baseline_comparisons/run_consensus_tiers.py` | E3 per-tier intra/cross analysis | `*_consensus_tiers.json`, `.txt` |
| `scripts/consensus_core/sweeps/run_cross_field.py` | E4 multi-field sweep | `cross_field_summary.json` |
| `scripts/review_taxonomy/boundary_reviews/run_boundary_review.py` | E5 A/B disagreement review for one field/k | `*_boundary_review.json` |
| `scripts/review_taxonomy/boundary_reviews/run_boundary_review_grid.py` | E5 grid across fields and k values | `boundary_review_summary.json` |
| `scripts/review_taxonomy/boundary_reviews/run_boundary_accuracy_review.py` | Protocol D v1 binary boundary adjudication on disagreement cases | `*_boundary_accuracy_review.json` |
| `scripts/review_taxonomy/boundary_reviews/run_boundary_coverage_review.py` | Protocol D v2 coverage-aware population/diagnostic boundary review | `*_boundary_coverage_v2_review.json` |
| `scripts/review_taxonomy/boundary_reviews/score_boundary_coverage.py` | Aggregate Protocol D v2 coverage-aware review outputs | `boundary_coverage_v2_summary.json` |
| `scripts/consensus_core/sweeps/run_effective_k_sweep.py` | E6 one-field effective-k sweep | `*_k_sweep.json` |
| `scripts/consensus_core/sweeps/run_cross_field_k_sweep.py` | E6 multi-field effective-k sweep | `cross_field_k_sweep_summary.json` |
| `scripts/consensus_core/baseline_comparisons/run_density_matched_comparison.py` | Protocol C density-matched clustering comparison | `*_density_matched_comparison.json` |
| `scripts/review_taxonomy/rank_shift/run_rank_shift_review.py` | E7 local reranking review for one field/k | `*_rank_shift_review.json` |
| `scripts/artifacts_reporting/summarize_consensus_bridge.py` | Bridge summary between `citation_consensus` and `all_consensus` result packages | bridge JSON |
| `scripts/review_taxonomy/review_uncertainty/repair_order_balanced_reviews.py` | Repair saved dual-pass review outputs after reviewer winner-logic changes | repaired `*_rank_shift_review.json` |
| `scripts/artifacts_reporting/prepare_gamma_cache.py` | Precompute reusable gamma values for large fields | `*_gamma_cache_prep.json`, gamma cache JSON |
| `scripts/consensus_core/validation/build_common_case_bank.py` | Build a shared rank-shift target bank across multiple pairwise comparisons | `*_common_case_bank.json` |
| `scripts/review_taxonomy/taxonomy/classify_case_taxonomy.py` | Classify reviewed bank cases into a primary winner/loser taxonomy | `*_taxonomy.json`, `taxonomy_combined.json/.csv` |
| `scripts/review_taxonomy/taxonomy/aggregate_taxonomy_results.py` | Aggregate existing taxonomy JSON files into a combined summary | `taxonomy_combined.json/.csv` |
| `scripts/review_taxonomy/taxonomy/export_taxonomy_report.py` | Export a combined taxonomy summary as Markdown and LaTeX snippets | `taxonomy_report.md`, `taxonomy_report.tex` |
| `scripts/review_taxonomy/taxonomy/run_taxonomy_calibration.py` | Build a stratified taxonomy calibration set and optionally re-label it with a live judge | `*_sample.json`, `*_llm_comparison.json` |
| `scripts/review_taxonomy/review_uncertainty/estimate_review_uncertainty.py` | Add Wilson and bootstrap uncertainty to local review win rates | `review_uncertainty.json` |
| `scripts/artifacts_reporting/fit_consensus_regime_model.py` | Fit simple predictive models for when consensus wins a local review case | `consensus_regime_model.json` |
| `scripts/artifacts_reporting/generate_figures.py` | Render result JSON files to PNG figures | `figures/*.png` |

## Typical Commands

### E1

```bash
python research/consensus/scripts/consensus_core/baseline_comparisons/run_comparison.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  --effective-k 30 \
  --min-size 10 \
  -o research/consensus/results
```

### E2

```bash
python research/consensus/scripts/consensus_core/baseline_comparisons/run_leave_one_out.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  -o research/consensus/results
```

### E3

```bash
python research/consensus/scripts/consensus_core/baseline_comparisons/run_consensus_tiers.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  -o research/consensus/results
```

### E4

```bash
python research/consensus/scripts/consensus_core/sweeps/run_cross_field.py \
  data/linktype_edges_gcc \
  --effective-k 30 \
  -o research/consensus/results
```

### E5 sample-only

```bash
python research/consensus/scripts/review_taxonomy/boundary_reviews/run_boundary_review.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15 \
  --method-a sum \
  --method-b consensus \
  --top-k 6 \
  --sample-only \
  -o research/consensus/results
```

### E5 field/k grid

```bash
python research/consensus/scripts/review_taxonomy/boundary_reviews/run_boundary_review_grid.py \
  data/linktype_edges_gcc \
  data/openalex_metadata \
  --fields field_15,field_12 \
  --k-values 6,30 \
  -o research/consensus/results
```

### E6 one-field sweep

```bash
python research/consensus/scripts/consensus_core/sweeps/run_effective_k_sweep.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  --k-values 1-30 \
  -o research/consensus/results
```

### E6 cross-field sweep

```bash
python research/consensus/scripts/consensus_core/sweeps/run_cross_field_k_sweep.py \
  data/linktype_edges_gcc \
  --k-values 1-30 \
  -o research/consensus/results
```

### E7 local rank-shift review

```bash
python research/consensus/scripts/review_taxonomy/rank_shift/run_rank_shift_review.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15_k06_sum_vs_consensus_local \
  --method-a sum \
  --method-b consensus \
  --top-k 6 \
  -o research/consensus/results
```

### Gamma cache precompute for large fields

```bash
python research/consensus/scripts/artifacts_reporting/prepare_gamma_cache.py \
  data/linktype_edges_gcc/field_12 \
  --field field_12 \
  --config 'sum_minus_cc|sum|*|cc_cosine' \
  --config 'sum_minus_emb|sum|*|emb_knn' \
  --config 'consensus_all|consensus|*|-' \
  --top-k 30 \
  --gamma-cache research/consensus/results/gamma_cache/field_12_gamma_cache.json \
  -o research/consensus/results/gamma_cache
```

### Common case bank for fair pairwise local review

```bash
python research/consensus/scripts/consensus_core/validation/build_common_case_bank.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15 \
  --reference consensus_all \
  --config 'consensus_all|consensus|*|-' \
  --config 'sum_all|sum|*|-' \
  --config 'sum_minus_cc|sum|*|cc_cosine' \
  --config 'sum_minus_emb|sum|*|emb_knn' \
  --top-k 30 \
  --n-targets 128 \
  --gamma-cache research/consensus/results/gamma_cache/field_15_gamma_cache.json \
  -o research/consensus/results/case_banks
```

Then reuse the bank in pairwise review:

```bash
python research/consensus/scripts/review_taxonomy/rank_shift/run_rank_shift_review.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15_k30_sum_minus_cc_vs_consensus_bank \
  --method-a sum \
  --method-b consensus \
  --label-a sum_minus_cc \
  --label-b consensus_all \
  --exclude-layers-a cc_cosine \
  --top-k 30 \
  --n-cases 48 \
  --case-bank research/consensus/results/case_banks/field_15_k30_consensus_all_common_case_bank.json \
  --strict-case-bank \
  --gamma-cache research/consensus/results/gamma_cache/field_15_gamma_cache.json \
  -o research/consensus/results
```

For the current preferred protocol, add order balancing and write into the
corrected results directory:

```bash
python research/consensus/scripts/review_taxonomy/rank_shift/run_rank_shift_review.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15_k30_sum_minus_cc_vs_consensus_bank \
  --method-a sum \
  --method-b consensus \
  --label-a sum_minus_cc \
  --label-b consensus_all \
  --exclude-layers-a cc_cosine \
  --top-k 30 \
  --n-cases 48 \
  --case-bank research/consensus/results/case_banks/field_15_k30_consensus_all_common_case_bank.json \
  --strict-case-bank \
  --order-balanced \
  --model gemini-2.5-pro \
  --gamma-cache research/consensus/results/gamma_cache/field_15_gamma_cache.json \
  -o research/consensus/results/case_banks_corrected
```

If you need to repair an existing dual-pass review after a winner-logic change,
reuse the saved `balanced_passes` without calling the judge again:

```bash
python research/consensus/scripts/review_taxonomy/review_uncertainty/repair_order_balanced_reviews.py \
  research/consensus/results/case_banks_corrected/field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v2_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v2_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_12_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v2_rank_shift_review.json \
  --suffix _gemini_v3
```

### Case taxonomy on reviewed bank outputs

```bash
python research/consensus/scripts/review_taxonomy/taxonomy/classify_case_taxonomy.py \
  research/consensus/results/case_banks_corrected/field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  --model gemini-2.5-pro \
  -o research/consensus/results/taxonomy
```

This produces:
- per-file `*_taxonomy.json`
- combined `taxonomy_combined.json`
- combined `taxonomy_combined.csv`
- `taxonomy_combined.json` also includes:
  - `representative_examples_by_winner`
  - `representative_by_label`

If you already have per-file taxonomy JSON outputs and just want a merged summary:

```bash
python research/consensus/scripts/review_taxonomy/taxonomy/aggregate_taxonomy_results.py \
  research/consensus/results/taxonomy/field_15_k06_sum_minus_emb_vs_consensus_bank_n48_taxonomy.json \
  research/consensus/results/taxonomy/field_15_k30_sum_minus_cc_vs_consensus_bank_n48_taxonomy.json \
  -o research/consensus/results/taxonomy
```

The taxonomy is single-label by design and currently uses:
- `single_cue_specificity`
- `broad_context_noise`
- `method_family_coherence`
- `material_family_coherence`
- `application_umbrella_noise`
- `semantic_drift`
- `coherent_refinement`
- `over_regularized_consensus`

If live LLM classification is slow or unavailable, use the deterministic fallback:

```bash
python research/consensus/scripts/review_taxonomy/taxonomy/classify_case_taxonomy.py \
  research/consensus/results/case_banks_corrected/field_12_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  --classifier heuristic \
  -o research/consensus/results/taxonomy
```

### Paper-ready taxonomy export

```bash
python research/consensus/scripts/review_taxonomy/taxonomy/export_taxonomy_report.py \
  research/consensus/results/taxonomy/taxonomy_aggregated.json \
  -o research/consensus/results/taxonomy \
  --stem taxonomy_report
```

This produces:
- `taxonomy_report.md`
- `taxonomy_report.tex`

### Taxonomy calibration set

```bash
python research/consensus/scripts/review_taxonomy/taxonomy/run_taxonomy_calibration.py \
  research/consensus/results/taxonomy/taxonomy_aggregated.json \
  --n-cases 24 \
  -o research/consensus/results/taxonomy \
  --stem taxonomy_calibration
```

Optional live re-labeling:

```bash
python research/consensus/scripts/review_taxonomy/taxonomy/run_taxonomy_calibration.py \
  research/consensus/results/taxonomy/taxonomy_aggregated.json \
  --n-cases 12 \
  --run-llm \
  --model gpt-oss:20b \
  -o research/consensus/results/taxonomy \
  --stem taxonomy_calibration
```

### Review uncertainty

```bash
python research/consensus/scripts/review_taxonomy/review_uncertainty/estimate_review_uncertainty.py \
  research/consensus/results/case_banks_corrected/field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_12_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  -o research/consensus/results/taxonomy \
  --stem review_uncertainty
```

### Consensus regime model

```bash
python research/consensus/scripts/artifacts_reporting/fit_consensus_regime_model.py \
  research/consensus/results/case_banks_corrected/field_15_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_15_k30_sum_minus_cc_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  research/consensus/results/case_banks_corrected/field_12_k06_sum_minus_emb_vs_consensus_bank_n48_order_balanced_gemini_v3_rank_shift_review.json \
  -o research/consensus/results/taxonomy \
  --stem consensus_regime_model
```

### Null-control review

Sample low-shift, near-tie neighborhoods where the compared methods almost agree:

```bash
python research/consensus/scripts/review_taxonomy/rank_shift/run_null_rank_shift_control.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15_k30_sum_minus_cc_vs_consensus_null_control \
  --method-a sum \
  --method-b consensus \
  --label-a sum_minus_cc \
  --label-b consensus_all \
  --exclude-layers-a cc_cosine \
  --top-k 30 \
  --n-cases 12 \
  --gamma-cache research/consensus/results/gamma_cache/field_15_gamma_cache.json \
  --sample-only \
  -o research/consensus/results/controls
```

To keep the sampled control bank fixed during live review, reuse the saved JSON:

```bash
python research/consensus/scripts/review_taxonomy/rank_shift/run_null_rank_shift_control.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15_k30_sum_minus_cc_vs_consensus_null_control \
  --method-a sum \
  --method-b consensus \
  --label-a sum_minus_cc \
  --label-b consensus_all \
  --exclude-layers-a cc_cosine \
  --top-k 30 \
  --n-cases 12 \
  --gamma-cache research/consensus/results/gamma_cache/field_15_gamma_cache.json \
  --input-json research/consensus/results/controls/field_15_k30_sum_minus_cc_vs_consensus_null_control_null_control.json \
  --model gemini-2.5-pro \
  -o research/consensus/results/controls
```

### Figure generation

```bash
python research/consensus/scripts/artifacts_reporting/generate_figures.py \
  research/consensus/results \
  -o research/consensus/figures
```

## Notes

- Stability-heavy runs are expensive on `field_15`; use smaller `--n-seeds` for smoke tests.
- `run_cross_field_k_sweep.py` accepts either `data/linktype_edges_gcc/field_XX/*.parquet`
  or nested layouts like `data/.../field_XX/edges/*.parquet`.
- `run_boundary_review_grid.py` currently targets GCC-ready fields first.
- Blind review uses the configured `SCISCAPE_LLM_*` environment via `create_client()`.
- If `SCISCAPE_LLM_MODEL` contains `gemini`, reviewer prompts apply an extra
  calibration block tuned for stricter A/B scientific judgment.
- `generate_figures.py` only consumes result files that already exist; it does not run experiments itself.
