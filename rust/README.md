# sciscape-leiden (Rust backend)

High-performance CPM Leiden clustering with fixed-node support.

## Build

```bash
cd rust
maturin develop --release   # install into active venv
cargo test                  # run Rust unit + integration tests
```

## Why Rust is faster than Java

| Factor | Java backend | Rust backend |
|---|---|---|
| **Startup** | JVM cold start ~2s per invocation | Zero startup (shared library loaded once) |
| **Data transfer** | Write parquet → spawn JVM → read parquet → write result → read result (5 I/O round-trips) | PyO3 passes numpy arrays directly into Rust (zero-copy via memory mapping) |
| **Memory layout** | JVM heap with GC pauses; boxed objects scattered in memory | Contiguous `Vec<f64>` / `Vec<u32>` in CSR format; cache-line friendly |
| **Hot-path optimization** | JIT compilation after warm-up; branch prediction limited by virtual dispatch | `unsafe` unchecked indexing on guaranteed-bounds arrays; LLVM auto-vectorization at compile time |
| **Contraction** | File-based: write contracted graph → re-read | In-place `Workspace` arrays reused across recursion levels (zero allocation after first iteration) |
| **Parallelism** | Single-threaded process spawning | Rayon work-stealing for multi-start and subnetwork extraction |
| **Postprocess** | Not available (Python fallback via igraph) | Native cascading-γ postprocess with greedy fallback, runs in compiled code |

### Empirical results (field_34, ~15k nodes)

- Java: 16-22% slower than Rust per Leiden invocation
- Rust postprocess: 4-10ms for 500-5000 node graphs
- Total pipeline speedup: ~2x (dominated by I/O elimination)

## Architecture

```
src/
  graph.rs          CSR graph (undirected, weighted, node_weights, self_loop_weights)
  clustering.rs     Cluster assignment with optional fixed-node mask
  leiden.rs         Main algorithm: move → refine → aggregate → recurse
  fast_local_move.rs  Phase 1: greedy node moving (CWTS port)
  local_merge.rs    Phase 2: refinement via local merging (CWTS port)
  contraction.rs    O(n+m) graph contraction with scatter-gather
  postprocess.rs    Cascading-γ reassignment + greedy fallback
  quality.rs        CPM quality function
  workspace.rs      Pre-allocated reusable arrays (zero allocation in hot path)
  python.rs         PyO3 bindings (run_leiden, run_postprocess, cpm_quality)
```

## API

From Python (via `sciscape.clustering.leiden_rust`):

```python
from sciscape.clustering.leiden_rust import run_leiden_rust, postprocess_small_clusters_rust

result = run_leiden_rust(
    edges_src=src, edges_dst=dst, edges_weight=weights,
    resolution=0.01, n_nodes=n, seed=42,
)
# result.membership, result.quality, result.n_clusters

post = postprocess_small_clusters_rust(
    resolution=0.01, min_size=10,
    membership=result.membership,
    edges_src=src, edges_dst=dst, edges_weight=weights, n_nodes=n,
)
# post.membership, post.n_clusters, post.changed_at_round, post.rounds
```
