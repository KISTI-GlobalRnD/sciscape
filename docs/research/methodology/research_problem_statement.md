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

### Framework: Tree + Cut Separation
1. **Step 1**: Build a complete dendrogram (individual nodes → single root) using a single criterion
2. **Step 2**: Find a size-constrained optimal cut on the dendrogram

Step 1 and Step 2 are cleanly separated:
- The dendrogram encodes the network's hierarchical structure
- The cut encodes the user's constraint
- Different constraints yield different partitions from the same dendrogram

**Key framing**: The novelty is the *framework* (tree-cut separation applied to resolution-limit-free community detection), not a single new score.

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

Additional experiments:
- Multi-seed (5 seeds × 8 γ values = 40 combinations): best = 38 clusters
- Recursive splitting of large clusters: 38 → 45 (+7 only)
- Single-γ approach has a hard ceiling around 38-45 clusters for this network

### 2.2 Partition Function Limitations

#### Modularity
- **Resolution limit** (Fortunato & Barthélemy, 2007): Cannot detect communities with internal edges < √(2m)
- γ's meaning depends on graph size → not comparable across datasets

#### CPM (Constant Potts Model)
- **Resolution-limit-free**: γ = absolute edge density threshold
- **Subset property**: γ₁ < γ₂ → communities at γ₂ are subsets of communities at γ₁
- **Size bias**: ρ = e_ij / (n_i · n_j) penalizes large clusters disproportionately
- **Fundamental constraint (Traag et al. 2011)**: CPM is essentially the ONLY resolution-limit-free method. Size bias CANNOT be removed while maintaining resolution-limit-freeness. This must be addressed in the cut stage, not the merge criterion.

### 2.3 Leiden Algorithm Limitations

- **Single resolution per run**: One γ → one flat partition
- **No size constraints**: min_size is purely post-processing
- **Stochastic**: Different seeds → different results (±2 clusters observed)
- **Intermediate hierarchy discarded**: Internal aggregation creates implicit levels but are not exposed
- **Not designed for dendrograms**: Optimizes flat partition, not hierarchical structure

### 2.4 Existing Hierarchical Methods

#### Paris Algorithm (Bonald et al., 2018)
- Bottom-up agglomerative, produces complete dendrogram
- Uses modularity-based distance → **inherits resolution limit**
- **Key insight**: Paris's merge rule = average linkage on modularity-based similarity. Our CPM-density rule shares the same Lance-Williams algebraic structure.

#### Recursive Partitioning (Li et al., 2022)
- Top-down: recursively bisect via spectral methods
- **Error lock-in**: Initial mistakes propagate and cannot be corrected
- **Dendrogram inversions**: Structural distortions proven in 2025 (JASA)

#### Bottom-up vs Top-down (2025, JASA)
- Proven: bottom-up achieves exact recovery at intermediate depths
- **Bottom-up is theoretically superior for hierarchy construction**

---

## 3. Key Theoretical Discovery: CPM Density = Average Linkage

### 3.1 The Equivalence

The inter-cluster CPM density:
```
ρ(A, B) = e_AB / (|A| · |B|)
```
is **mathematically identical** to graph-based average-linkage similarity (sum of edge weights / product of cluster sizes). This is not merely an analogy — it is a mathematical identity.

### 3.2 Consequences

**Lance-Williams recurrence (UPGMA form)**:
```
ρ(A∪B, C) = [|A| · ρ(A,C) + |B| · ρ(B,C)] / (|A| + |B|)
```

This convex combination guarantees:
1. **Reducibility**: ρ(A∪B, C) ≤ max(ρ(A,C), ρ(B,C)) — unconditionally
2. **Valid ultrametric dendrogram**: Monotonically non-increasing merge heights, no inversions
3. **NN-chain applicable**: All existing near-linear average-linkage algorithms work directly
4. **Deterministic**: No seeds, no stochasticity → perfect reproducibility
5. **CPM interpretation**: Merge height γ*(A,B) = e_AB/(|A||B|) is the critical resolution where ΔQ_CPM changes sign

### 3.3 No Published Work Exploits This

Bonald et al. (2018) proved reducibility for modularity-based Paris distance d(a,b) = w(a,b)/(w(a)·w(b)). Replacing degree-based weights with uniform (size-based) weights yields CPM density, and the proof carries over identically. **No published paper appears to have explicitly noted this CPM-density reducibility result or the CPM = average-linkage equivalence.** This is the genuine gap.

### 3.4 Caveat: Greedy ≠ CPM Global Optimum

The dendrogram cut at height γ is NOT the globally optimal CPM partition at γ.

**4-node counterexample**:
```
Edges: (0,1), (0,2), (1,2), (1,3)

Greedy ρ-tree possible cuts: {0},{1},{2},{3} | {0,2},{1,3} | {0,1,2,3}
CPM optimum at γ=0.4:        {0,1,2},{3}   ← NOT in the tree!
```

The tree is a **CPM-critical laminar approximation**. This is stated honestly as a feature: the greedy tree provides a tractable, deterministic, reproducible structure that preserves the merge gap information lost by flat methods.

---

## 4. Algorithmic Breakthrough: Near-Linear Time via Dhulipala et al.

### 4.1 The CPM = Average-Linkage Equivalence Unlocks Existing Algorithms

The Dhulipala et al. research program (ICML 2021, NeurIPS 2022, SIGMOD 2024) provides near-linear-time algorithms for average-linkage graph HAC. Since CPM density IS average linkage, these apply directly:

| Algorithm | Complexity | Type | Scale demonstrated |
|-----------|-----------|------|--------------------|
| SeqHAC exact (ICML 2021) | Õ(n√m) | Sequential, exact | Millions of nodes |
| SeqHAC approx (ICML 2021) | Õ(m) | Sequential, (1+ε)-approx | Millions of nodes |
| ParHAC (NeurIPS 2022) | Õ(m) work, polylog depth | Shared-memory parallel | ~100B edges |
| TeraHAC (SIGMOD 2024) | ~dozens of MapReduce rounds | Distributed | **8 trillion edges** |

### 4.2 Practical Implications for Our Network

For 100K nodes, 3.28M edges:
- **Exact**: Õ(100K × √3.28M) ≈ Õ(170M) operations → **minutes on single machine**
- **Approximate (1+ε)**: Õ(3.28M) → **seconds**
- (1+ε=1.1) approximation incurs ~3% loss in ARI/NMI empirically

### 4.3 Open Source Implementation

**GBBS** (github.com/ParAlg/gbbs) and **ParHAC** (github.com/ParAlg/ParHAC) implement average-linkage as the primary supported linkage type.

### 4.4 Complexity Lower Bound

Bateni et al. (ICALP 2024): Exact average-linkage graph HAC requires **Ω(n^{3/2−ε})** time. SeqHAC's Õ(n√m) is essentially tight. Implication: use (1+ε)-approximate Õ(m) for production, exact for validation.

### 4.5 Key Data Structure: Neighbor-Heap

Per-cluster heaps of neighboring clusters sorted by similarity. For reducible linkages like CPM density, cached similarity values are upper bounds → **lazy evaluation**: stale entries discarded when popped. Uses dynamic low-outdegree graph orientations bounded by arboricity α. For sparse citation networks (α ≈ O(√n)), major practical speedups.

---

## 5. Size-Constrained Optimal Cut: O(n) Greedy DP

### 5.1 Algorithm

At each internal node v with children l, r:
- **Split** if both subtrees can independently form valid partitions (opt(l) ≥ 1 AND opt(r) ≥ 1), yielding opt(l) + opt(r) clusters
- **Keep** the entire subtree as one cluster if size(v) ≥ k, yielding 1 cluster
- Splitting always dominates keeping when valid (gives ≥2 vs. 1)

**O(n) time**, single bottom-up pass, O(1) work per node. Provably optimal. No approximation.

### 5.2 Stability Tie-Break

Among cuts with equal cluster count, maximize total stability:
```
F(v) = max_lexicographic(#clusters, total_stability)

F(v) = max(
    (1, p(v))                    if |v| ≥ k
    F(left(v)) + F(right(v))
)
```
where p(v) = branch-length-based stability (persistence in dendrogram).

### 5.3 Extensions

For richer objectives (e.g., maximizing total within-cluster density + count):
```
opt(v) = max{quality_keep(v), opt(l) + opt(r)}
```
Remains O(n) for additive quality functions on binary trees.

For (l, u)-partition (both min and max cluster sizes): solvable in O(n⁵) via DP with interval compression (Buchin & Selbach, 2022; Ito et al., 2012).

---

## 6. Size Bias: A Fundamental Constraint

### 6.1 Traag's Theorem (2011)

CPM is essentially the ONLY resolution-limit-free community detection method. Any criterion that corrects size bias (e.g., density ratio ρ_between/max(ρ_within)) necessarily introduces non-local information → **reintroduces resolution limits**.

### 6.2 Criterion Comparison

| Criterion | Resolution-free | Size bias | Reducible | Ultrametric | Lance-Williams |
|-----------|:-:|:-:|:-:|:-:|:-:|
| CPM density e_ij/(n_i·n_j) | ✅ | Has bias | ✅ | ✅ | ✅ (UPGMA) |
| Density ratio ρ_between/max(ρ_within) | Likely ❌ | Mitigated | Likely ❌ | Unknown | ❌ |
| Paris distance e_ij/(w_i·w_j) | ❌ | Less bias | ✅ | ✅ | ✅ |

### 6.3 Strategy: Address Size Bias in Cut, Not Merge

- Build dendrogram using pure CPM density (all theoretical/algorithmic advantages preserved)
- Use quality-weighted constrained cut that optimizes a size-aware objective
- The dendrogram faithfully encodes multi-resolution CPM structure
- The cutting algorithm selects the scale

### 6.4 Mass-Weighted Family (Experimental Extension)

```
s_θ(A, B) = e_AB / (Θ(A) · Θ(B)),    Θ(C) = Σ_{u∈C} θ_u,  θ_u > 0
```

Preserves reducibility and Lance-Williams form. Candidate masses:
| θ_u | Effect |
|-----|--------|
| 1 | Pure CPM (baseline, guaranteed resolution-limit-free) |
| √d_u | Moderate degree correction |
| d_u^α | Tunable (α=0 → CPM, α=1 → configuration-model-like) |

**WARNING**: θ_u ≠ 1 may break resolution-limit-freeness. Must verify theoretically before claiming. Report as experimental comparison, not as main method.

---

## 7. Hybrid Pipeline: Theoretical Justification

### 7.1 Architecture

```
Phase 1: Leiden(γ_high) → micro-clusters (~5K-10K clusters)     [seconds]
Phase 2: Contract graph (micro-clusters → super-nodes)           [seconds]
Phase 3: CPM-density HAC on contracted graph (SeqHAC/GBBS)      [minutes]
Phase 4: Size-constrained DP cut                                 [milliseconds]
Phase 5: (Optional) Leiden refinement with initial_membership    [minutes]
```

### 7.2 Why This Works

- **Leiden is already a hybrid pipeline**: local moves → refinement → graph aggregation → repeat. Contracted-graph HAC is a natural extension.
- **CPM subset property**: Fine partition at γ_high refines all coarser optimal partitions. Use `resolution_profile()` to verify nesting.
- **Loukas (2019, JMLR)**: Spectral coarsening theory proves coarsened graph ε-approximates original graph's first k eigenvectors. For moderate coarsening (100K → 5K-10K), spectral preservation is strong.
- **Ghoroghchian et al. (2021)**: Coarsening preserves community structure if micro-clusters don't cross true community boundaries.
- **Practical validation**: Scanpy (bioinformatics) runs Leiden at multiple resolutions then builds dendrogram of clusters — exactly this approach, validated on single-cell datasets.

### 7.3 Open Theoretical Question

No published work proves that starting agglomerative CPM-density HAC from Leiden micro-clusters yields the same dendrogram as starting from singletons. However: if the micro-partition at γ_high is a strict refinement of all optimal partitions at lower γ, the dendrogram from micro-clusters is a "pruned" version of the singleton dendrogram, missing only finest-scale merges within micro-clusters. The coarse structure is preserved exactly.

### 7.4 Pure Singleton vs. Hybrid

| Aspect | Singleton start | Hybrid (Leiden warm-start) |
|--------|----------------|---------------------------|
| Reproducibility | Perfect (deterministic) | Near-perfect (Leiden seed, but micro-level only) |
| Completeness | Full dendrogram | Pruned (missing within-micro merges) |
| Computation | Õ(n√m) exact, Õ(m) approx | Leiden O(m) + HAC on contracted |
| Theory | Clean | Needs subset-property verification |

**Both are valid.** For the paper, present singleton-start as the theoretically clean baseline and hybrid as the practical pipeline.

---

## 8. Experimental Design

### 8.1 Fair Comparison: Same Step 2, Different Step 1

Apply the **same** size-constrained DP cut to all dendrogram methods:

| Method | Step 1 (Tree) | Step 2 (Cut) |
|--------|--------------|-------------|
| Leiden-CPM + merge | Leiden(γ) + post-hoc merge | Standard merge |
| Paris + DP cut | Paris dendrogram | **Our DP cut** |
| Recursive partitioning + DP cut | Top-down bisection tree | **Our DP cut** |
| CPM-critical tree + DP cut | ρ-agglomeration (θ=1) | **Our DP cut** |
| CPM-critical tree (hybrid) + DP cut | Leiden micro → HAC | **Our DP cut** |

This isolates whether performance differences come from the tree or the cut.

### 8.2 Validation Baseline

**Peixoto's nested SBM** (graph-tool): Bayesian model selection producing statistically rigorous hierarchical descriptions. For 100K nodes: 2-12 hours MCMC. Different philosophy (Bayesian vs. optimization) but provides gold-standard reference.

### 8.3 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| k-feasible cluster count | Primary goal: more clusters ≥ k |
| Internal citation density | Cluster cohesion |
| External separation | Inter-cluster sparsity |
| Keyword coherence | Semantic quality (title/abstract-based) |
| Cluster connectedness | Structural integrity |
| Seed/tie-break stability | Reproducibility |

### 8.4 First Milestone

> **Does the CPM-critical tree + DP cut break the 38-cluster ceiling?**

If yes, the framework already has a paper. Everything else is depth.

---

## 9. Theoretical Results to Prove

### Four propositions/theorems for the paper

1. **Proposition 1**: CPM-density agglomeration is equivalent to average linkage on weighted adjacency matrix (with zeros for non-edges). Proof: direct from Lance-Williams coefficients.

2. **Proposition 2**: The merge height γ*(A,B) = ρ(A,B) is the CPM critical resolution — the exact γ at which ΔQ_CPM(A,B) = e_AB - γ|A||B| changes sign.

3. **Theorem 3**: Size-constrained optimal cut DP — correctness (splitting always dominates keeping when valid), O(n) complexity, uniqueness under lexicographic stability tie-break.

4. **Theorem 4 (optional)**: Under HSBM with level separation condition, the CPM-critical tree achieves exact recovery at intermediate depths. (Use Dreveton et al. 2025 as proof template.)

---

## 10. Implementation Roadmap

### Phase 1: CPM-Critical Tree Baseline
- Option A: Implement ρ-agglomeration directly (Python + priority queue)
- Option B: Use GBBS/ParHAC average-linkage implementation
- Implement O(n) size-constrained DP cut
- Test on 100k-node network
- **Success criterion**: > 38 clusters with min_size ≥ 1000

### Phase 2: Hybrid Pipeline
- Leiden(γ_high) → contract → HAC on contracted graph
- Compare dendrogram with Phase 1 singleton-start
- Verify CPM subset property holds in practice

### Phase 3: Fair Baseline Comparisons
- Paris + same DP cut
- Recursive partitioning + same DP cut
- Leiden-CPM at multiple γ + standard merge
- (Optional) Peixoto nested SBM as reference

### Phase 4: Mass-Weighted Exploration (Experimental)
- s_θ family (θ = 1, √d, d^α)
- Compare cluster quality across θ choices
- Check resolution-limit-freeness theoretically

### Phase 5: Paper Writing
- Prove Propositions 1-2 (straightforward)
- Prove Theorem 3 (DP correctness)
- Attempt Theorem 4 (HSBM recovery)
- Write up with emphasis on framework, not single score

---

## 11. Context and Motivation

This research is motivated by the SciScape project — a toolkit for scientific landscape analysis:

```
Citation network → Hierarchical clustering → Keyword extraction → Landscape report
```

Current limitation: Leiden-CPM produces ~38 meaningful clusters (≥1000 docs each) from 100,000 papers. The new framework aims to break this ceiling by:
1. Building a proper dendrogram (preserving merge gap information)
2. Applying optimal size-constrained cut (non-uniform depth)

The dendrogram also enables:
- Reusable structure: same tree, different k → different granularities
- Temporal analysis: cluster birth/death AFTER structural clustering
- Interactive exploration: navigate hierarchy at any level

---

## References

1. Traag, Waltman, van Eck (2019). "From Louvain to Leiden." Scientific Reports.
2. Traag et al. (2011). "Narrow scope for resolution-limit-free community detection." Physical Review E.
3. Fortunato & Barthélemy (2007). "Resolution limit in community detection." PNAS.
4. Bonald et al. (2018). "Hierarchical Graph Clustering using Node Pair Sampling." MLG/KDD.
5. Li et al. (2022). "Hierarchical Community Detection by Recursive Partitioning." JASA.
6. Dreveton et al. (2025). "When Does Bottom-up Beat Top-down in Hierarchical Community Detection?" JASA.
7. Mauduit & Simonetto (2024). "Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts." Asilomar.
8. Lancichinetti & Fortunato (2012). "Consensus clustering in complex networks." Scientific Reports.
9. Meyerhenke et al. (2016). "Partitioning (hierarchically clustered) complex networks via size-constrained graph clustering." J. Heuristics.
10. Lance & Williams (1967). "A general theory of classificatory sorting strategies." The Computer Journal.
11. Dhulipala et al. (2021). "Hierarchical Agglomerative Graph Clustering in Nearly-Linear Time." ICML.
12. Dhulipala et al. (2022). "Parallel Hierarchical Agglomerative Graph Clustering." NeurIPS.
13. Dhulipala et al. (2024). "TeraHAC: Hierarchical Agglomerative Clustering of Trillion-Edge Graphs." SIGMOD.
14. Bateni et al. (2024). "Optimal Bounds for Approximate Average-Linkage Clustering." ICALP.
15. Loukas (2019). "Graph Reduction with Spectral and Cut Guarantees." JMLR.
16. Ghoroghchian et al. (2021). "Graph Community Detection from Coarse Measurements." D&A.
17. Peixoto (2014). "Hierarchical Block Structures and High-Resolution Model Selection in Large Networks." Physical Review X.
18. Buchin & Selbach (2022). "Constrained Dendrogram Cuts." (via Ito et al., Algorithmica 2012).
19. Jeub, Sporns, Fortunato (2018). "Multiresolution Consensus Clustering." Scientific Reports.
20. Arnaudon et al. (2024). "PyGenStability: Multiscale Community Detection with Generalized Markov Stability." arXiv.
