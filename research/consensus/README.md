# Research 1: Consensus as Boundary Signal in Multi-Layer Scientific Paper Graphs

## Hypotheses

- **H1**: At same rank budget (top-k), consensus level (n_layers agreeing) predicts cluster-boundary precision better than single-layer edge strength
- **H2**: Embedding layers substitute for CC in citation-poor fields
- **H3**: Optimal combination follows role separation — consensus = boundary signal, exclusive edges = density/periphery

## Experiment Design

### E1: Same effective-k comparison
- Fix total edges per node (k_eff = 30)
- Compare: single-layer (BC-only, CC-only, DC-only) vs multi-layer consensus
- Metric: AMI stability (5 seeds), boundary-node precision (LLM blind review)

### E2: Leave-one-out
- 4-layer (BC+CC+DC+Emb) → remove each layer one at a time
- Measure: cluster count change, AMI vs full, quality report

### E3: Consensus level × rank tier analysis
- Stratify edges by consensus level (1L, 2L, 3L, 4L)
- For each tier: intra-cluster vs cross-cluster ratio
- Expected: higher consensus → higher intra-cluster %

### E4: Cross-field generalization
- Fields: 15 (Chem Eng), 12 (Arts & Hum), 18, 26, 29, 30, 34
- Same pipeline, same parameters → field-dependent patterns

### E5: Boundary-node blind review (formal)
- Sample: consensus-level stratified (N=50 per level)
- Review: belonging + cohesion + outlier (3 criteria)
- Confusion matrix per consensus level

## Data

- oa26 fields: 15, 12, 18, 26, 29, 30, 34
- Edge types: DC (fractional), BC (cosine), CC, Emb (SPECTER2 k-NN)
- Preprocessing: top-30 per node per layer → 1/rank → consensus combination → GCC

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_comparison.py` | E1: single vs multi-layer comparison |
| `scripts/run_leave_one_out.py` | E2: ablation study |
| `scripts/run_consensus_tiers.py` | E3: per-tier intra/cross analysis |
| `scripts/run_cross_field.py` | E4: 7-field generalization |
| `scripts/run_boundary_review.py` | E5: LLM boundary evaluation |
| `scripts/generate_figures.py` | All figures for paper |

## Expected Output

- `results/`: per-field JSON results
- `figures/`: publication-ready plots
- `paper/`: LaTeX manuscript
