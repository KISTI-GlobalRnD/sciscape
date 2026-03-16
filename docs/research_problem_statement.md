# Research Problem: Hierarchical Community Detection with Size Constraints

## 1. Problem Definition

### Input
- Weighted undirected graph G = (V, E, w) representing a citation network
- Minimum cluster size constraint k (e.g., k = 1,000 documents)

### Goal
- Find a partition of V into clusters where:
  - Every cluster has at least k nodes
  - The number of clusters is maximized
  - Each cluster represents a cohesive research community

### Desired Framework
1. **Step 1**: Build a complete dendrogram (individual nodes → single root) using a single optimal criterion
2. **Step 2**: Find a size-constrained optimal cut on the dendrogram

Step 1 and Step 2 are cleanly separated:
- The dendrogram encodes the network's hierarchical structure
- The cut encodes the user's constraint
- Different constraints yield different partitions from the same dendrogram

---

## 2. Why Existing Methods Fail

### 2.1 Current Pipeline: Leiden-CPM + Post-hoc Merge

```
Leiden(γ) → raw clusters → merge_small(k) → final partition
```

**Empirical evidence** (100,000 node citation network, k=1000):

| γ (CPM) | Raw clusters | After merge (≥1000) | Information loss |
|---------|-------------|-------------------|-----------------|
| 1e-5    | 606         | 14                | 592 clusters absorbed |
| 5e-5    | 1,011       | 27                | 984 clusters absorbed |
| 1e-4    | 1,248       | 38 (best)         | 1,210 clusters absorbed |
| 2e-4    | 1,548       | 30                | 1,518 clusters absorbed |
| 1e-3    | 2,950       | 2                 | 2,948 clusters absorbed |

At the optimal γ=1e-4: **97% of raw clusters are destroyed** by post-hoc merge.
The "merge gap" — clusters with 100-999 nodes — is where information is lost.
These are potentially meaningful research sub-topics that get absorbed into larger clusters.

Additional experiments:
- Multi-seed (5 seeds × 8 γ values = 40 combinations): best = 38 clusters
- Recursive splitting of large clusters: 38 → 45 (+7 only)
- Single-γ approach has a hard ceiling around 38-45 clusters for this network

### 2.2 Partition Function Limitations

#### Modularity
- **Resolution limit** (Fortunato & Barthélemy, 2007): Cannot detect communities with internal edges < √(2m). In our network (m=3.28M), this means communities with < ~2,560 internal edges are invisible.
- γ's meaning depends on graph size → not comparable across datasets
- Result: 49 clusters at best, but many are tiny (min=3 nodes)

#### CPM (Constant Potts Model)
- **Resolution-limit-free**: γ = absolute edge density threshold
- **Subset property**: γ₁ < γ₂ → communities at γ₂ are subsets of communities at γ₁
- **Fragmentation**: Slightly increasing γ shatters sparse regions into dust clusters
- **Size bias**: ρ = e_ij / (n_i · n_j) penalizes large clusters disproportionately
  - 5000×5000 cluster pair with 1000 edges: ρ = 0.00004
  - 10×10 cluster pair with 5 edges: ρ = 0.05
  - Large clusters almost never merge, even when clearly related
- **Heterogeneous density**: Single γ cannot simultaneously handle dense and sparse regions
- **External connections ignored**: Only considers internal density, not separation quality

### 2.3 Leiden Algorithm Limitations

Leiden is a fast, high-quality optimizer with connectivity guarantees. But:

- **Single resolution per run**: One γ → one flat partition
- **No size constraints**: min_size is purely post-processing
- **Greedy single-node moves**: Cannot move groups of nodes simultaneously
- **Stochastic**: Different seeds → different results (±2 clusters observed)
- **Intermediate hierarchy discarded**: Internal aggregation passes create implicit levels but are not exposed
- **Not designed for dendrograms**: Optimizes flat partition, not hierarchical structure

### 2.4 Existing Hierarchical Methods

#### Paris Algorithm (Bonald et al., 2018)
- Bottom-up agglomerative, produces complete dendrogram
- Uses modularity-based distance → **inherits resolution limit**
- No CPM variant exists

#### Recursive Partitioning (Li et al., 2022)
- Top-down: recursively bisect via spectral methods
- **Error lock-in**: Initial mistakes propagate and cannot be corrected
- **Dendrogram inversions**: Structural distortions proven in 2025 (JASA)

#### Bottom-up vs Top-down (2025, JASA)
- Proven: bottom-up achieves exact recovery at intermediate depths
- Top-down fails at information-theoretic thresholds
- **Bottom-up is theoretically superior for hierarchy construction**

---

## 3. Identified Research Gap

| What exists | What is missing |
|-------------|-----------------|
| Paris: complete dendrogram, efficient | CPM-based (resolution-limit-free) dendrogram |
| CPM: resolution-limit-free, subset property | Efficient bottom-up dendrogram using CPM criterion |
| Constrained optimal cut on trees (Mauduit 2024) | Applied to large-scale community detection |
| Bottom-up > Top-down (proven 2025) | CPM density-based bottom-up agglomerative |
| Size-constrained graph partitioning | Size-constrained community detection WITH dendrogram |

**Core gap**: No algorithm builds a resolution-limit-free complete dendrogram via bottom-up agglomeration, with size-constrained optimal cut, at scale.

---

## 4. Proposed Approach: CPM-based Agglomerative Dendrogram

### 4.1 Core Idea

Build dendrogram by merging cluster pairs in order of inter-cluster edge density:

```
ρ(C_i, C_j) = e_ij / (n_i · n_j)
```

- Start from singleton clusters (each node = 1 cluster)
- Merge the pair with highest ρ
- Record ρ at merge as dendrogram height
- Repeat until single root

**Properties**:
- Cutting at height γ yields CPM partition at resolution γ
- Subset property automatically satisfied
- Single run, all resolutions embedded in dendrogram
- Resolution-limit-free (no global graph size in formula)

### 4.2 Potential Improvements to CPM Criterion

Three candidate merge criteria under investigation:

#### (a) Pure CPM density
```
s(C_i, C_j) = e_ij / (n_i · n_j)
```
- Simple, subset property guaranteed, γ has clear physical meaning
- But: size bias (fragmentation), ignores external connections

#### (b) Density Ratio
```
s(C_i, C_j) = ρ_between(C_i, C_j) / max(ρ_within(C_i), ρ_within(C_j))
```
- Removes size bias: compares inter-cluster to intra-cluster density
- Auto-adapts to local network density
- But: subset property unproven, γ interpretation unclear

#### (c) Local Null Model
```
expected_ij = Σ_{u∈C_i, v∈C_j} f(d_u, d_v)   [local, not global]
s(C_i, C_j) = e_ij / expected_ij
```
- Resolution-limit-free AND degree-aware
- But: theoretical properties need verification

### 4.3 Size-Constrained Optimal Cut (Step 2)

Given the dendrogram, find a (possibly non-uniform) cut that:
- Maximizes the number of leaf clusters
- Subject to: every leaf cluster has ≥ k nodes

This is a tree DP problem, solvable in O(n).

Non-uniform cut: dense regions are cut deeper (more clusters), sparse regions cut higher (fewer, larger clusters). This naturally handles density heterogeneity.

---

## 5. Open Problems

### 5.1 Theoretical
- Reducibility of CPM density: Does ρ satisfy the nearest-neighbor chain condition for efficient computation?
- Subset property under Density Ratio criterion
- Formal relationship between dendrogram cut and CPM partition optimality

### 5.2 Computational
- Naive complexity: O(n²) for all-pair distances. Need O(m log n) or better.
- Nearest-neighbor chain applicability: requires reducibility proof
- Practical acceleration for sparse graphs (only edge-connected pairs have ρ > 0)

### 5.3 Empirical
- Does the dendrogram produce more clusters (≥k) than single-γ Leiden + merge?
- How does the merge criterion affect cluster quality (coherence, keyword separability)?
- Comparison with Paris, Leiden hierarchy, recursive partitioning on benchmark networks

### 5.4 Design Decisions
- Handling of fusion/interdisciplinary research: merger order when two parent fields have similar ρ
- Tie-breaking strategy
- Whether to incorporate Leiden's refinement concept at merge steps

---

## 6. Context and Motivation

This research is motivated by the SciScape project — a toolkit for scientific landscape analysis. The practical pipeline is:

```
Citation network → Hierarchical clustering → Keyword extraction → Landscape report
```

Current limitation: Leiden-CPM can only produce ~38 meaningful clusters (≥1000 docs each) from a 100,000 paper network. Theoretical maximum is ~100. The new method aims to close this gap by eliminating the information loss in post-hoc merge.

The dendrogram-based approach also enables:
- Reusable structure: same dendrogram, different k values → different granularities
- Temporal analysis: cluster birth/death analysis AFTER structural clustering (separation of concerns)
- Interactive exploration: users can navigate the hierarchy at any level

---

## References

1. Traag, Waltman, van Eck (2019). "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports.
2. Traag et al. (2011). "Narrow scope for resolution-limit-free community detection." Physical Review E.
3. Fortunato & Barthélemy (2007). "Resolution limit in community detection." PNAS.
4. Bonald et al. (2018). "Hierarchical Graph Clustering using Node Pair Sampling." MLG/KDD.
5. Li et al. (2022). "Hierarchical Community Detection by Recursive Partitioning." JASA.
6. (2025). "When Does Bottom-up Beat Top-down in Hierarchical Community Detection?" JASA.
7. Mauduit & Simonetto (2024). "Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts." Asilomar.
8. Lancichinetti & Fortunato (2012). "Consensus clustering in complex networks." Scientific Reports.
9. Meyerhenke et al. (2016). "Partitioning (hierarchically clustered) complex networks via size-constrained graph clustering." J. Heuristics.
