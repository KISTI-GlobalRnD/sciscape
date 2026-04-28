# Research 2: Hybrid CPM-Critical Hierarchy with Optimal Size-Constrained Cut

## Research Question

Can a tree+cut framework recover merge-gap information that Leiden+merge discards, while preserving the same minimum-size constraint?

## Hypotheses

- **H1**: Under the same minimum-size constraint, tree+cut recovers more valid clusters than Leiden+merge.
- **H2**: Separating tree construction from cut selection makes stochasticity and size bias easier to control.
- **H3**: Applying a common cut rule to the same hierarchy isolates tree quality more cleanly than comparing postprocess heuristics alone.

## Current Scope

This folder now distinguishes three executable pieces:

- `run_leiden_merge.py`
  - standard flat baseline
- `run_hybrid.py`
  - hierarchy runner that measures depth-wise recovery
- `run_optimal_cut.py`
  - explicit nano → contract → dendrogram → `constrained_cut` pipeline
- `run_cut_ablation.py`
  - same contracted dendrogram, but compares `optimal cut` against threshold-based cuts

This means the current codebase supports:

- flat baseline comparison
- hierarchy depth tracking
- explicit optimal cut on the contracted tree
- threshold-vs-optimal cut ablation on the same tree
- result aggregation and figure generation

Optional baselines such as Paris and nested SBM are still tracked, but only executed when their external dependencies are installed.

## Method

### A. Hybrid hierarchy runner

```text
Step 1: Auto-γ Leiden on the original graph
Step 2: Contract graph by the recovered communities
Step 3: Re-run Leiden on the contracted graph
Step 4: Repeat for nano / micro / meso / macro
```

This is what `scripts/run_hybrid.py` executes.

### B. Hybrid + optimal cut runner

```text
Step 1: Auto-γ Leiden on the original graph (nano partition)
Step 2: Contract graph by nano communities
Step 3: Build CPM dendrogram on the contracted graph
Step 4: Run size-constrained optimal cut with original-node leaf sizes
Step 5: Map cut labels back to original papers
```

This is what `scripts/run_optimal_cut.py` executes.

The current implementation matches the `landscape.py` micro-cut path, but is now exposed as an independent research script.

## Baselines

| Method | Status | Notes |
|--------|--------|-------|
| `Leiden + merge` | implemented | `scripts/run_leiden_merge.py` |
| `Hybrid hierarchy` | implemented | `scripts/run_hybrid.py` |
| `Hybrid + optimal cut` | implemented | `scripts/run_optimal_cut.py` |
| `Paris + DP cut` | optional | requires `scikit-network` |
| `Nested SBM` | optional | requires `graph-tool` |
| `Recursive split` | planned | manifest entry only for now |

## Metrics

| Metric | Meaning |
|--------|---------|
| Cluster count | Number of valid clusters recovered |
| Max cluster % | Largest cluster as % of total nodes |
| Singleton % | Residual tiny-cluster signature |
| AMI stability | Reproducibility of the flat or hierarchy entry point |
| Total cut stability | Sum of persistence over cut nodes |
| Hybrid depth trajectory | How many clusters survive at each hierarchy level |

## Scripts

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `scripts/run_leiden_merge.py` | Flat baseline | `*_leiden_merge.json` |
| `scripts/run_hybrid.py` | Hierarchy depth analysis | `*_hybrid.json` |
| `scripts/run_optimal_cut.py` | Explicit contracted-tree optimal cut | `*_hybrid_cut.json`, `*_hybrid_cut_membership.parquet` |
| `scripts/run_cut_ablation.py` | Compare optimal cut vs threshold sweep on the same dendrogram | `*_cut_ablation.json` |
| `scripts/run_baselines.py` | Run the implemented methods and record skipped optional baselines | `*_baseline_manifest.json` |
| `scripts/compare_methods.py` | Merge result files into markdown, JSON, and PNG summaries | `*_comparison.md`, `.json`, `.png` |

## Typical Commands

### Flat baseline

```bash
python research/dendrogram/scripts/run_leiden_merge.py \
  data/linktype_edges_gcc/field_15/dc_fractional.parquet \
  --field field_15 \
  --min-size 30 \
  --target-pct 3.0 \
  -o research/dendrogram/results
```

### Hybrid hierarchy

```bash
python research/dendrogram/scripts/run_hybrid.py \
  data/linktype_edges_gcc/field_15/dc_fractional.parquet \
  --field field_15 \
  --n-levels 4 \
  --min-size 30 \
  --target-pct 3.0 \
  -o research/dendrogram/results
```

### Hybrid + optimal cut

```bash
python research/dendrogram/scripts/run_optimal_cut.py \
  data/linktype_edges_gcc/field_15/dc_fractional.parquet \
  --field field_15 \
  --nano-min-size 30 \
  --target-pct 3.0 \
  -o research/dendrogram/results
```

### Cut ablation

```bash
python research/dendrogram/scripts/run_cut_ablation.py \
  data/linktype_edges_gcc/field_15/dc_fractional.parquet \
  --field field_15 \
  --nano-min-size 30 \
  --target-pct 3.0 \
  --n-thresholds 24 \
  -o research/dendrogram/results
```

### Combined run + aggregation

```bash
python research/dendrogram/scripts/run_baselines.py \
  data/linktype_edges_gcc/field_15/dc_fractional.parquet \
  --field field_15 \
  -o research/dendrogram/results

python research/dendrogram/scripts/compare_methods.py \
  research/dendrogram/results \
  --field field_15
```

## Notes

- `run_hybrid.py` and `run_optimal_cut.py` answer different questions:
  - hierarchy runner: how much structure survives across levels
  - cut runner: what happens when the contracted tree is cut explicitly
- `run_cut_ablation.py` answers the next question:
  - on the same tree, how much better is the dynamic-programming cut than a threshold cut
- `run_baselines.py` records optional methods as skipped when `scikit-network` or `graph-tool` are unavailable.

## Scope Separation

**Current applied paper**

- contracted graph hierarchy
- contracted-tree optimal cut
- baseline comparison on field-level networks

**Future theory paper**

- singleton-start exact CPM dendrogram
- guarantees on merge-gap recovery
- scalability and approximation analysis
