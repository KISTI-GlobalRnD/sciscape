# Implementation Plan: CPM-Critical Dendrogram

## Implementation Strategy

### GBBS 대신 Python 직접 구현

GBBS/ParHAC는 C++ CLI 전용이고 Python HAC API가 미문서화.
Subprocess wrapping은 I/O 파싱 취약점이 크고 디버깅 어려움.

**Python sparse HAC가 현실적인 이유:**
- 100K 노드에서 O(n²) = 10B → 불가
- 하지만 sparse graph에서 실제 후보 = 3.28M 쌍 (edge-connected만)
- Lazy priority queue + neighbor dict로 O(m log n) 수준 달성
- 예상 런타임: 10-30분 (연구용 충분)
- 향후 GBBS 통합은 scalability extension으로 별도 진행

### Fallback

만약 Python이 100K에서 너무 느리면:
1. Leiden(γ_high) → contract → Python HAC on ~5K nodes (확실히 빠름)
2. Cython/Numba 가속
3. GBBS subprocess wrapping

---

## Module Structure

```
sciscape/clustering/
├── dendrogram.py          # [NEW] CPM-critical dendrogram construction
├── constrained_cut.py     # [NEW] Size-constrained optimal cut DP
├── triadic_preprocess.py  # [NEW] Edge reweighting via triadic closure
├── hierarchy_builder.py   # [EXISTING] Leiden-based hierarchy (untouched)
├── runner.py              # [EXISTING] LeidenRunner (reuse graph interface)
├── postprocess.py         # [EXISTING] merge_small_clusters (comparison baseline)
└── ...
```

---

## File 1: `dendrogram.py` — CPM-Critical Dendrogram

### Data Structures

```python
@dataclass
class DendrogramNode:
    """A node in the binary merge tree."""
    id: int                    # Unique node ID
    size: int                  # Number of original nodes in subtree
    merge_height: float        # ρ at which this merge occurred (γ* critical resolution)
    left: int | None           # Left child ID (None for leaves)
    right: int | None          # Right child ID (None for leaves)

@dataclass
class Dendrogram:
    """Complete binary dendrogram from agglomerative CPM-density clustering."""
    nodes: Dict[int, DendrogramNode]  # id → node
    root: int                          # Root node ID
    n_leaves: int                      # Number of original nodes
    leaf_ids: List[int]                # Original node IDs (leaves)

    def to_linkage_matrix(self) -> np.ndarray:
        """Convert to scipy-compatible (n-1, 4) linkage matrix."""
        ...

    def subtree_sizes(self) -> Dict[int, int]:
        """Size of each internal node's subtree."""
        ...

    def persistence(self, node_id: int) -> float:
        """γ_birth - γ_death for stability measurement."""
        ...
```

### Core Algorithm: Sparse Average-Linkage HAC

```python
def build_cpm_dendrogram(
    graph: ig.Graph,
    *,
    weights: str = "weight",
    progress: Callable | None = None,
) -> Dendrogram:
    """Build CPM-critical dendrogram via exact average-linkage on sparse graph.

    Equivalent to greedy agglomerative clustering with merge criterion:
        ρ(A, B) = e_AB / (|A| · |B|)

    Uses lazy max-heap with neighbor dictionaries for sparse efficiency.
    Time: O(m · log²n) amortized for sparse graphs.
    Space: O(n + m)
    """
```

### Implementation Detail: Lazy Max-Heap Approach

```
Data structures:
  - cluster_neighbors: Dict[int, Dict[int, float]]
    cluster_neighbors[a][b] = sum of edge weights between clusters a and b
  - cluster_size: Dict[int, int]
    cluster_size[a] = number of original nodes in cluster a
  - alive: Set[int]
    Set of active (unmerged) cluster IDs
  - heap: max-heap of (ρ, cluster_a, cluster_b)
    Lazy: entries may reference dead clusters

Algorithm:
  1. Initialize: each node is a singleton cluster
     For each edge (u,v,w): cluster_neighbors[u][v] += w
     Push (ρ=w/(1·1), u, v) for each edge

  2. Repeat n-1 times:
     a. Pop max (ρ, a, b) from heap
        - If a or b not in alive: skip (lazy deletion)
        - If ρ ≠ current_ρ(a,b): skip (stale entry)
     b. Create new internal node c = merge(a, b)
        - merge_height = ρ
        - size[c] = size[a] + size[b]
     c. Update neighbors:
        For each neighbor d of a or b (d ≠ a, d ≠ b):
          e_cd = e_ad + e_bd  (sum edge weights to both a and b)
          ρ_new = e_cd / (size[c] · size[d])
          cluster_neighbors[c][d] = e_cd
          cluster_neighbors[d][c] = e_cd
          Push (ρ_new, c, d) to heap
        Remove a, b from alive, add c
     d. Record DendrogramNode(id=c, left=a, right=b, height=ρ, size=size[c])

  3. Return Dendrogram(nodes, root=c, n_leaves=n)
```

### Complexity

- Heap operations: Each edge processed O(log n) times worst case
- Neighbor updates: Each merge touches O(degree) neighbors
- Total: O(m · log n) amortized with lazy deletion
- For 100K nodes, 3.28M edges: ~10-30 min in Python

### Triadic Closure Preprocessing (separate function)

```python
def reweight_triadic(graph: ig.Graph, *, weights: str = "weight") -> ig.Graph:
    """Reweight edges by triadic closure: w'(i,j) = w(i,j) · (1 + |CN(i,j)|).

    Creates a copy of the graph with modified edge weights.
    Time: O(m · √m) for triangle enumeration.
    """
```

---

## File 2: `constrained_cut.py` — Size-Constrained Optimal Cut

### Algorithm

```python
@dataclass
class CutResult:
    """Result of size-constrained optimal cut on a dendrogram."""
    partition: List[Set[int]]        # List of clusters (sets of original node IDs)
    membership: List[int]            # Node ID → cluster ID mapping
    n_clusters: int                  # Number of clusters
    total_stability: float           # Sum of persistence values
    cut_nodes: List[int]             # Dendrogram node IDs at the cut

def constrained_cut(
    dendrogram: Dendrogram,
    min_size: int,
) -> CutResult:
    """Find partition maximizing #clusters s.t. all clusters have ≥ min_size nodes.

    Lexicographic objective: max(#clusters, total_stability).

    Algorithm: Bottom-up DP on binary tree, O(n).
    At each node v:
      split_count = opt(left) + opt(right)   if both feasible
      keep_count  = 1                        if size(v) ≥ min_size
      Choose: split if split_count ≥ 2, else keep if feasible
      Tie-break: higher total stability wins

    Returns optimal partition.
    """
```

### DP State

```python
@dataclass
class _DPState:
    count: int          # Number of clusters achievable in this subtree
    stability: float    # Total persistence of clusters
    feasible: bool      # Whether this subtree can form valid partition
```

### Traceback

After computing opt[v] for all nodes, trace back from root:
- If opt[v] chose "split": recurse into children
- If opt[v] chose "keep": v's entire subtree is one cluster
- Collect leaf nodes of each "keep" subtree → partition

---

## File 3: `triadic_preprocess.py` — Edge Reweighting

```python
def count_triangles_per_edge(graph: ig.Graph) -> List[int]:
    """Count common neighbors for each edge. O(m · α) where α = arboricity."""

def reweight_triadic(graph: ig.Graph, *, weights: str = "weight") -> ig.Graph:
    """Return new graph with w'(i,j) = w(i,j) · (1 + |CN(i,j)|)."""
```

igraph has `graph.count_triangles()` but not per-edge. Implement via:
```python
for edge in graph.es:
    u, v = edge.source, edge.target
    cn = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))
    edge["weight"] = edge["weight"] * (1 + cn)
```

For 100K nodes, avg degree 65: each edge checks ~65 neighbors.
Total: 3.28M × 65 = 213M set operations → minutes in Python.

---

## Integration into Landscape Pipeline

### New function in `landscape.py`:

```python
def run_landscape_dendrogram(
    edge_path: Path,
    abstract_path: Path,
    output_dir: Path,
    *,
    min_docs: int = 1000,
    triadic: bool = True,
    refine: bool = False,
    ...
) -> dict:
    """Landscape pipeline using CPM-critical dendrogram instead of Leiden hierarchy.

    Step 0: Load graph, giant component, (optional) triadic reweighting
    Step 1: Build CPM-critical dendrogram
    Step 2: Size-constrained optimal cut
    Step 3: (Optional) Leiden refinement
    Step 4: Keyword extraction
    Step 5: Report generation
    """
```

---

## Testing Plan

### Unit Tests

```
tests/
├── test_dendrogram.py
│   ├── test_singleton_graph          # 1 node → trivial dendrogram
│   ├── test_two_nodes                # 2 nodes → 1 merge
│   ├── test_triangle                 # 3 nodes, 3 edges → verify merge order
│   ├── test_four_node_counterexample # The known CPM ≠ greedy example
│   ├── test_disconnected_components  # Components merge last (ρ=0)
│   ├── test_weighted_edges           # Non-unit weights
│   ├── test_merge_heights_monotonic  # Dendrogram validity
│   ├── test_to_linkage_matrix        # scipy compatibility
│   └── test_deterministic            # Same graph → same tree always
│
├── test_constrained_cut.py
│   ├── test_trivial_cut              # k=1 → all singletons
│   ├── test_k_equals_n              # k=n → single cluster
│   ├── test_balanced_binary_tree     # Known optimal solution
│   ├── test_stability_tiebreak       # Equal count → higher stability wins
│   ├── test_infeasible               # k > n → error or single cluster
│   └── test_varying_k_same_tree      # Multiple k values, verify monotonicity
│
├── test_triadic_preprocess.py
│   ├── test_triangle_boost           # Triangle edge gets weight * 2
│   ├── test_no_triangles             # Weight unchanged
│   └── test_preserves_structure      # Same nodes/edges, different weights
│
└── test_integration.py
    ├── test_dendrogram_pipeline      # End-to-end: graph → dendrogram → cut → partition
    └── test_ceiling_breaking         # On sample network: more clusters than Leiden+merge?
```

### Benchmark Test (Separate Script)

```python
# scripts/benchmark_dendrogram.py
# Run on KRISS 100K network, compare:
# 1. Leiden-CPM (γ=1e-4) + merge → 38 clusters
# 2. CPM-critical tree + DP cut → ? clusters
# 3. Paris + DP cut → ? clusters
# Report: cluster count, density, keyword coherence
```

---

## Implementation Order

### Phase 1: Core (1-2 weeks)
1. `constrained_cut.py` — simplest, pure tree DP, easy to test
2. `dendrogram.py` — core HAC algorithm, extensive testing needed
3. Unit tests for both

### Phase 2: Integration (1 week)
4. `triadic_preprocess.py` — straightforward
5. Integration into landscape pipeline
6. Integration tests

### Phase 3: Validation (1 week)
7. Run on KRISS 100K network
8. Compare with Leiden-CPM baseline (38 clusters)
9. **Milestone: Does it break the ceiling?**

### Phase 4: Comparison (1 week)
10. Paris baseline (scikit-network)
11. Mass-weighted θ variants (ablation)
12. Full experimental matrix

---

## Dependencies

### Required (already in project)
- igraph (graph operations)
- numpy (arrays, linkage matrix)
- scipy (sparse matrices, optional dendrogram visualization)

### Optional (for comparisons)
- scikit-network (Paris algorithm baseline)
- graph-tool (Peixoto nested SBM — heavy dependency, optional)

### NOT needed
- GBBS/ParHAC (Python implementation suffices for 100K scale)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-:|:-:|-----------|
| Python too slow for 100K | Medium | High | Fallback: Leiden contract → HAC on 5K nodes |
| Memory exceeded (neighbor dicts) | Low | High | Sparse: only edge-connected pairs stored |
| Dendrogram doesn't break 38 ceiling | Medium | Critical | Reframe: "dendrogram + cut" still useful for reusability |
| GBBS needed for larger networks | Future | Medium | Documented as scalability extension |

## Performance Optimization Notes

### Python-specific optimizations
1. Use `heapq` (C-implemented) for max-heap (negate values for max)
2. Neighbor dicts: `defaultdict(float)` for auto-initialization
3. `alive` set: O(1) membership check
4. Batch neighbor updates: merge smaller into larger cluster's dict (weighted union)
5. Consider `sortedcontainers.SortedList` if heap becomes bottleneck

### Memory estimate
- cluster_neighbors: at most 2m entries (each edge stored twice) → ~50MB for 3.28M edges
- heap: at most m·log(n) entries with lazy deletion → ~100MB
- dendrogram nodes: 2n-1 nodes → ~10MB
- **Total: ~200MB — well within modern machine capacity**
