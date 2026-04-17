# Research 2: Hybrid CPM-Critical Hierarchy with Optimal Size-Constrained Cut

## Research Question

Can a tree+cut framework recover merge-gap information that Leiden+merge discards,
yielding more valid clusters under the same minimum-size constraint?

## Hypotheses

- **H1**: Under same min-size constraint, tree+cut recovers more valid clusters than Leiden+merge
- **H2**: Separating the cut stage from the tree stage enables explicit control of stochasticity and size bias
- **H3**: A common cut algorithm applied to different tree baselines isolates tree quality for fair comparison

## Method: Hybrid CPM-Critical Hierarchy

```
Step 1: Nano Leiden (auto-γ, postprocess)
Step 2: Contract graph (inter-cluster edges → super-node edges)
Step 3: 1/rank re-normalize contracted edges
Step 4: Repeat from Step 1 on contracted graph (micro, meso, macro levels)
Step 5: Size-constrained optimal cut on the hierarchy tree
```

Key innovation: contraction + 1/rank re-normalization enables CPM to split
at upper levels where the contracted graph would otherwise be near-complete.

## Baselines

| Method | Implementation | Notes |
|--------|---------------|-------|
| **Leiden + merge** | `sciscape.clustering.runner` | Standard: Leiden → merge small clusters greedily |
| **Recursive split** | Custom | Top-down: split largest cluster recursively |
| **Paris + DP cut** | `sknetwork.hierarchy.Paris` | Agglomerative hierarchy + same DP cut |
| **Nested SBM** | `graph-tool` (optional) | Bayesian nonparametric baseline |

## Metrics

| Metric | What it measures |
|--------|------------------|
| Cluster count under min-size | How many valid clusters survive the constraint |
| Merge-gap recovery | # clusters with quality > threshold that Leiden+merge merges away |
| Text coherence (TF-IDF) | Intra-cluster keyword overlap |
| AMI stability (5 seeds) | Reproducibility |
| Expert boundary judgment | LLM blind review at cluster boundaries |
| Determinism | AMI between identical runs with different seeds |

## Experiments

### E1: Leiden+merge vs hybrid (same γ)
- Fix γ at optimal value from auto-gamma
- Compare: Leiden+merge (standard postprocess) vs hybrid (contract+rerun)
- Output: cluster count, sizes, quality

### E2: Hierarchy depth analysis
- 1 level (nano only) vs 2 levels vs 3 levels vs 4 levels
- How many additional valid clusters does each level recover?

### E3: Baseline comparison on 100K network
- Same network, same min-size, 4 methods
- Paris: `pip install scikit-network`
- Table: method × metric matrix

### E4: Cut algorithm ablation
- Same hierarchy tree, different cut strategies:
  - Greedy (current postprocess)
  - Dynamic programming (optimal)
  - Threshold-based

## Data

- Primary: field_15 (100K nodes, citation-rich)
- Secondary: field_12 (citation-poor, different structure)
- Reproducibility: fixed seed=42, cached intermediate results

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_leiden_merge.py` | Baseline: standard Leiden+merge |
| `scripts/run_hybrid.py` | Our method: hybrid hierarchy |
| `scripts/run_baselines.py` | Paris, recursive split, (SBM) |
| `scripts/compare_methods.py` | Generate comparison table + figures |

## Scope Separation

**This paper (hybrid applied):**
- Contracted graph hierarchy with 1/rank normalization
- Size-constrained optimal cut
- Practical evaluation on 2-7 fields

**Future paper (exact theory):**
- Singleton-start exact CPM dendrogram (GBBS/SeqHAC)
- Theoretical guarantees on merge-gap recovery
- Scalability analysis
