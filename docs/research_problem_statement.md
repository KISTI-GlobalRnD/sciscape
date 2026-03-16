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

**Key framing**: The novelty is the *framework* (tree-cut separation for community detection), not a single new score.

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
- **Key observation**: Paris's merge rule is equivalent to average linkage on a particular similarity matrix; our CPM-density rule shares the same algebraic structure

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
| Paris: complete dendrogram, efficient | Resolution-limit-free (CPM-based) dendrogram |
| CPM: resolution-limit-free, subset property | Bottom-up dendrogram using CPM criterion |
| Constrained optimal cut on trees (Mauduit 2024) | Applied to large-scale community detection |
| Bottom-up > Top-down (proven 2025) | CPM density-based bottom-up agglomerative |
| Size-constrained graph partitioning | Size-constrained community detection WITH dendrogram |

**Core gap**: No algorithm builds a resolution-limit-free complete dendrogram via bottom-up agglomeration, with size-constrained optimal cut, at scale. The novelty is not a single merge criterion but the *framework* that separates structure discovery from constraint satisfaction.

---

## 4. Proposed Approach

### 4.1 Naming Convention

**NOT** "CPM-optimal dendrogram" — this implies global optimality which is false.

Recommended names:
- **CPM-critical dendrogram** (the merge height is the critical γ at which merging becomes beneficial)
- **Greedy CPM merge tree**

### 4.2 Step 1 Baseline: Pure CPM-Critical Tree

Build dendrogram by merging cluster pairs in order of inter-cluster edge density:

```
ρ(C_i, C_j) = e_ij / (n_i · n_j)
```

- Start from singleton clusters (each node = 1 cluster)
- Merge the pair with highest ρ
- Record ρ at merge as dendrogram height (= critical resolution γ*)
- Repeat until single root

**Proven properties**:
- **Reducibility**: After merging A, B into AB:
  ```
  ρ(AB, C) = (n_A · ρ(A,C) + n_B · ρ(B,C)) / (n_A + n_B)
  ```
  This is a size-weighted average → always ≤ max(ρ(A,C), ρ(B,C))
  → NN-chain applicable → O(m log²n) complexity
- **Valid dendrogram**: Merge heights are monotonically non-increasing
- **Deterministic**: No seeds, no stochasticity → perfect reproducibility
- **Resolution-limit-free**: No global graph size in formula
- **CPM interpretation**: Merge height γ*(A,B) = e_AB/(|A||B|) is the exact breakpoint where ΔQ_CPM(A,B) = e_AB - γ·|A||B| changes sign

**Important caveat — Greedy ≠ Global optimum**:

The dendrogram cut at height γ does NOT necessarily equal the globally optimal CPM partition at γ. Counterexample (4-node graph):
```
Edges: (0,1), (0,2), (1,2), (1,3)

ρ-agglomeration possible cuts:
  {0}, {1}, {2}, {3}
  {0,2}, {1,3}
  {0,1,2,3}

But CPM optimum at γ=0.4:
  {0,1,2}, {3}           ← NOT in the dendrogram!
```

This tree is a **CPM-inspired laminar approximation**, not an exact CPM hierarchy. This is honestly stated and turned into a strength: the greedy tree provides a tractable, deterministic structure that preserves the merge gap information lost by flat methods.

**Known weaknesses of pure ρ**:
- Size bias: large clusters penalized by n_i · n_j denominator
- Singleton noise: early merges driven by single edges

### 4.3 Step 1 Extension: Mass-Weighted CPM Family (Main Novelty)

Instead of pure density, define a **mass-weighted similarity**:

```
s_θ(A, B) = e_AB / (Θ(A) · Θ(B)),    where Θ(C) = Σ_{u∈C} θ_u,  θ_u > 0
```

**This preserves ALL theoretical properties**:

| Property | Status |
|----------|--------|
| CPM-like quality function | ✅ ΔQ_{γ,θ}(A,B) = e_AB - γ · Θ(A) · Θ(B) |
| Reducibility (NN-chain) | ✅ s_θ(A∪B, C) = [Θ(A)·s_θ(A,C) + Θ(B)·s_θ(B,C)] / [Θ(A)+Θ(B)] |
| Valid dendrogram | ✅ Monotonic merge heights |
| Deterministic | ✅ |
| Resolution-limit-free | ✅ (when θ_u is local, not global) |

**Candidate mass functions**:

| θ_u | Interpretation | Effect |
|-----|---------------|--------|
| 1 | Pure CPM density (baseline) | Size bias, but simplest |
| √d_u | Geometric degree correction | Moderate degree-awareness |
| d_u^α (α ∈ [0,1]) | Tunable degree correction | α=0 → CPM, α=1 → configuration-model-like |

The α-family provides a smooth interpolation between:
- α=0: Pure CPM (resolution-limit-free, size-biased)
- α=1: Degree-normalized (similar to configuration model null, but local)

**Why this is the right extension**:
- Density Ratio (ρ_between/ρ_within) breaks additive structure → no reducibility
- Arbitrary local null models break Lance-Williams form → no NN-chain
- Mass-weighted family preserves ALL algebra while adding degree-awareness
- Connection to Traag et al.'s resolution-limit-free class provides theoretical grounding

**Open question**: Does the mass-weighted family satisfy locality in the sense required for resolution-limit-freeness? This needs separate proof and is honestly left open.

### 4.4 Step 2: Size-Constrained Optimal Cut with Stability Tie-Break

Given the dendrogram, find a (possibly non-uniform) cut that:
- **Primary objective**: Maximize the number of leaf clusters
- **Constraint**: Every leaf cluster has ≥ k nodes
- **Tie-break**: Among cuts with equal cluster count, maximize total stability

**DP formulation with lexicographic objective**:

For each tree node v with subtree size |v|:
```
F(v) = max_lexicographic(#clusters, total_stability)

F(v) = max(
    (1, p(v))                    if |v| ≥ k    (v is a leaf cluster)
    F(left(v)) + F(right(v))     (recurse into children)
)
```

where p(v) = branch-length-based stability (persistence of node v in the dendrogram).

- Solvable in O(n) time
- Non-uniform cut: dense regions cut deeper (more clusters), sparse regions cut higher (fewer, larger clusters)
- Stability tie-break prevents choosing cuts at unstable (short-lived) merge points

---

## 5. Theoretical Results to Prove

### Target: Four propositions/theorems for the paper

1. **Proposition 1**: Pure ρ-agglomeration is equivalent to average linkage on weighted adjacency matrix (with zeros for non-edges).

2. **Proposition 2**: Both ρ and s_θ are CPM-family pairwise critical merge thresholds:
   γ*(A,B) = s_θ(A,B) is the resolution at which ΔQ_{γ,θ}(A,B) = 0.

3. **Theorem 3**: Size-constrained optimal cut DP — correctness, O(n) complexity, uniqueness under lexicographic tie-break.

4. **Theorem 4 (optional)**: Under HSBM with level separation condition, the CPM-critical tree achieves exact recovery at intermediate depths. (Use recent bottom-up recovery theory as proof template.)

---

## 6. Experimental Design

### 6.1 Fair Comparison: Same Step 2, Different Step 1

Apply the **same** size-constrained DP cut to all dendrogram methods:

| Method | Step 1 (Tree) | Step 2 (Cut) |
|--------|--------------|-------------|
| Leiden-CPM + merge | Leiden(γ) + post-hoc merge | Standard merge |
| Paris + DP cut | Paris dendrogram | **Our DP cut** |
| Recursive partitioning + DP cut | Top-down bisection tree | **Our DP cut** |
| Our ρ-tree + DP cut | CPM-critical dendrogram (θ=1) | **Our DP cut** |
| Our s_θ-tree + DP cut | Mass-weighted dendrogram (θ=d^α) | **Our DP cut** |

This isolates whether performance differences come from the tree or the cut.

### 6.2 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| k-feasible cluster count | Primary goal: more clusters ≥ k |
| Internal citation density | Cluster cohesion |
| External separation | Inter-cluster sparsity |
| Keyword coherence | Semantic quality (title/abstract-based) |
| Cluster connectedness | Structural integrity |
| Seed/tie-break stability | Reproducibility |

### 6.3 First Milestone

> **Does the pure ρ-tree + DP cut break the 38-cluster ceiling?**

If yes, the framework already has a paper. The mass-weighted extension then provides the theoretical depth and further improvement.

---

## 7. Implementation Roadmap

### Phase 1: Pure CPM-Critical Tree (Baseline)
- Implement ρ-agglomeration with NN-chain acceleration
- Implement size-constrained DP cut
- Test on 100k-node KRISS network
- **Success criterion**: > 38 clusters with min_size ≥ 1000

### Phase 2: Mass-Weighted Extension
- Implement s_θ family (θ = 1, √d, d^α)
- Compare dendrogram structures across θ choices
- Evaluate cluster quality metrics

### Phase 3: Fair Baseline Comparisons
- Run Paris on same network with same DP cut
- Run recursive partitioning with same DP cut
- Run Leiden-CPM at multiple γ with standard merge

### Phase 4: Theoretical Analysis
- Prove Propositions 1-2 (straightforward)
- Prove Theorem 3 (DP correctness)
- Attempt Theorem 4 (HSBM recovery — stretch goal)

---

## 8. Preprocessing Considerations

### Structural Edge Weighting (Singleton Noise Mitigation)
```
w'(i,j) = w(i,j) · (1 + |common_neighbors(i,j)|)
```
- Strengthens edges within triangles (triadic closure)
- Weakens random cross-community citations
- Graph preprocessing only — does not modify the algorithm
- Optional enhancement, reported as variant in experiments

---

## 9. Context and Motivation

This research is motivated by the SciScape project — a toolkit for scientific landscape analysis. The practical pipeline is:

```
Citation network → Hierarchical clustering → Keyword extraction → Landscape report
```

Current limitation: Leiden-CPM can only produce ~38 meaningful clusters (≥1000 docs each) from a 100,000 paper network. Theoretical maximum is ~100. The new method aims to close this gap by preserving the merge gap information in a laminar hierarchy.

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
10. Lance & Williams (1967). "A general theory of classificatory sorting strategies: 1. Hierarchical systems." The Computer Journal.
