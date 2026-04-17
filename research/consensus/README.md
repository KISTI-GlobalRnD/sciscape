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
- The current code supports E1-E5 directly.

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

### E5: Boundary-node blind review

- Samples hard boundary nodes from the consensus clustering.
- Stratifies by node-level average consensus.
- If the requested boundary quantile yields no reviewable cases, the script
  falls back to lower quantiles automatically.
- Two modes:
  - `--sample-only`: builds the review set without calling an LLM
  - default mode: runs belonging, cohesion, and outlier prompts
- Outputs:
  - `*_boundary_review.json`

## Scripts

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `scripts/run_comparison.py` | E1 single-layer vs consensus comparison | `*_comparison.json` |
| `scripts/run_leave_one_out.py` | E2 ablation study | `*_leave_one_out.json` |
| `scripts/run_consensus_tiers.py` | E3 per-tier intra/cross analysis | `*_consensus_tiers.json`, `.txt` |
| `scripts/run_cross_field.py` | E4 multi-field sweep | `cross_field_summary.json` |
| `scripts/run_boundary_review.py` | E5 sample generation or LLM review | `*_boundary_review.json` |
| `scripts/run_effective_k_sweep.py` | E6 one-field effective-k sweep | `*_k_sweep.json` |
| `scripts/run_cross_field_k_sweep.py` | E6 multi-field effective-k sweep | `cross_field_k_sweep_summary.json` |
| `scripts/generate_figures.py` | Render result JSON files to PNG figures | `figures/*.png` |

## Typical Commands

### E1

```bash
python research/consensus/scripts/run_comparison.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  --effective-k 30 \
  --min-size 10 \
  -o research/consensus/results
```

### E2

```bash
python research/consensus/scripts/run_leave_one_out.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  -o research/consensus/results
```

### E3

```bash
python research/consensus/scripts/run_consensus_tiers.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  -o research/consensus/results
```

### E4

```bash
python research/consensus/scripts/run_cross_field.py \
  data/linktype_edges_gcc \
  --effective-k 30 \
  -o research/consensus/results
```

### E5 sample-only

```bash
python research/consensus/scripts/run_boundary_review.py \
  data/linktype_edges_gcc/field_15 \
  data/openalex_metadata/field_15/works_text.parquet \
  --field field_15 \
  --sample-only \
  -o research/consensus/results
```

### E6 one-field sweep

```bash
python research/consensus/scripts/run_effective_k_sweep.py \
  data/linktype_edges_gcc/field_15 \
  --field field_15 \
  --k-values 1-30 \
  -o research/consensus/results
```

### E6 cross-field sweep

```bash
python research/consensus/scripts/run_cross_field_k_sweep.py \
  data/linktype_edges_gcc \
  --k-values 1-30 \
  -o research/consensus/results
```

### Figure generation

```bash
python research/consensus/scripts/generate_figures.py \
  research/consensus/results \
  -o research/consensus/figures
```

## Notes

- Stability-heavy runs are expensive on `field_15`; use smaller `--n-seeds` for smoke tests.
- `run_cross_field_k_sweep.py` accepts either `data/linktype_edges_gcc/field_XX/*.parquet`
  or nested layouts like `data/.../field_XX/edges/*.parquet`.
- Blind review uses the configured `SCISCAPE_LLM_*` environment via `create_client()`.
- `generate_figures.py` only consumes result files that already exist; it does not run experiments itself.
