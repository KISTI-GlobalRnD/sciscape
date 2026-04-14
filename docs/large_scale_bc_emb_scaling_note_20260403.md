# Large-Scale BC+Emb Scaling Note (2026-04-03)

## Problem
Full `BC + Emb` weighted-sum graph construction succeeded, but the original SciScape block-init path did not scale to the full graph.

Observed full graph size:
- nodes: about `56.6M`
- edges: about `4.136B`

The failure was not primarily host RAM exhaustion. The remote host still had ample available memory. The main bottleneck was the graph construction path:
- `Polars concat(uid1, uid2).unique()` on the full edge table
- followed by string-UID to dense-index remapping
- followed by full `igraph` object construction

In short: the blocking issue was the **graph-build strategy**, not the underlying `BC + Emb` graph itself.

## Working strategy
Use a two-stage sparse-native path.

1. Build the full generalized `BC + Emb` graph once.
2. Sparsify it before clustering.
3. Run high-gamma block-init on the sparsified graph using an integer-edge backend.

## What was done
### 1. Full fusion
A full generalized `weighted_sum` `BC + Emb` graph was materialized.

### 2. Symmetric top-k sparsification
The full graph was reduced with symmetric per-node top-k.

Result for `top-k = 30`:
- nodes: `56,623,524`
- edges: `1,495,632,735`

This reduced the edge count substantially while preserving the node universe.

### 3. Integer remap
The sparsified graph was converted to:
- `node_manifest.parquet`: `node_idx <-> uid`
- `int_edges.parquet`: `src_idx, dst_idx, weight`

This avoids repeated string-UID handling during clustering.

### 4. Sparse-native block-init backend
A sparse-native `NetworKit` path was introduced.

Why:
- `NetworKit.Graph.addEdges` accepts COO-style sparse input directly
- `ParallelLeiden` is available in `NetworKit`
- this avoids the expensive `Polars -> string UID mapping -> full igraph build` path

## Practical lesson
For full-scale `BC + Emb` graphs, do **not** use the default string-UID `build_graph()` path directly.

Preferred large-scale pattern:
1. full fusion
2. top-k or threshold sparsification
3. integer remap
4. sparse-native community detection / block-init
5. only join back to string UIDs at the output stage

## Recommended default
For future large runs, treat this as the default block-init pattern:
- full multilayer fusion graph
- symmetric top-k sparsification
- integer edge list
- sparse-native backend (`NetworKit` or equivalent)
- UID join only for final outputs

## Current implication for SciScape
SciScape should keep the current graph/object path for small to medium graphs.
For very large graphs, it needs a dedicated large-scale backend path that bypasses:
- full-string UID concat/unique
- full in-memory `igraph` construction from raw edge tables

A practical next step is to formalize this as an optional large-scale execution path inside SciScape.
