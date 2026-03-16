# Literature Review: Hierarchical Community Detection for Scientific Networks

## Problem Statement

Build a complete hierarchical tree (dendrogram) from individual papers to a single root
using a single criterion, then find a size-constrained optimal cut.

**Two-step framework:**
- Step 1: Build dendrogram (structure discovery)
- Step 2: Size-constrained optimal cut (constraint satisfaction)

The novelty is the *framework* — separating structure from constraints — not a single new score.

---

## 1. Paris Algorithm (Bonald et al., 2018)

**Paper:** "Hierarchical Graph Clustering using Node Pair Sampling"
**Link:** https://arxiv.org/abs/1806.01664

- Bottom-up agglomerative using node pair sampling probability as distance
- Nearest-neighbor chain (NNC) technique for O(m log²n) acceleration
- Produces complete dendrogram — run once, all resolutions embedded
- Distance: modularity-based → inherits resolution limit
- Implemented in scikit-network
- **Key insight for our work:** Paris's merge rule is equivalent to average linkage on a particular similarity matrix. Our CPM-density merge rule shares the same Lance-Williams algebraic structure, meaning NNC directly applies.
- **Gap:** No CPM-based variant exists

## 2. When Does Bottom-up Beat Top-down? (2025, JASA)

**Paper:** "When does bottom-up beat top-down in hierarchical community detection?"
**Link:** https://arxiv.org/abs/2306.00833

- Bottom-up achieves exact recovery at intermediate depths up to info-theoretic threshold
- Top-down (recursive partitioning) fails due to error lock-in and propagation
- Top-down dendrograms suffer from inversions (structural distortions)
- Bottom-up uses Bethe-Hessian spectral method + agglomerative hierarchy
- **Takeaway:** Bottom-up is theoretically superior for tree construction
- **For our work:** Provides proof template for Theorem 4 (HSBM recovery under CPM-critical tree)

## 3. Constrained Hierarchical Clustering via Graph Coarsening (Mauduit & Simonetto, 2024)

**Paper:** "Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts"
**Link:** https://arxiv.org/abs/2312.04209

- Two-step: (1) graph coarsening → hierarchy, (2) optimal cut with constraints
- Horizontal constraints (cannot-link, must-link) + vertical (level precedence)
- Natural extension of Loukas's local-variation graph coarsening
- Constraint violation < 5%, Dasgupta's cost improved 19%
- Applied to NLP word clustering, not large-scale network community detection
- **Takeaway:** "Tree + optimal cut" paradigm is established
- **For our work:** Validates the Step 1 / Step 2 separation framework

## 4. Consensus Clustering (Lancichinetti & Fortunato, 2012)

**Paper:** "Consensus clustering in complex networks"
**Link:** https://www.nature.com/articles/srep00336

- Run community detection at multiple resolutions/seeds
- Build co-clustering matrix → re-cluster for stable partition
- Significantly improves stability and accuracy
- Does NOT produce a dendrogram — flat partition only
- Multiresolution extension (2018): https://www.nature.com/articles/s41598-018-21352-7
- **For our work:** Considered as alternative (Path C) but rejected — bottom-level instability propagates upward through hierarchy. Pure agglomerative (deterministic) avoids this entirely.

## 5. Size-Constrained Graph Clustering (Meyerhenke et al., 2016)

**Paper:** "Partitioning (hierarchically clustered) complex networks via size-constrained graph clustering"
**Link:** https://link.springer.com/article/10.1007/s10732-016-9315-8

- Size-constrained label propagation (SCLaP) for coarsening + refinement
- Targets networks with hierarchically clustered structure
- Quality comparable to hMetis, 10x faster
- **Limitation:** Graph partitioning (balanced cut) objective, not community detection

## 6. Persistent Community Detection via TDA (AAAI 2024)

**Paper:** "Learning Persistent Community Structures in Dynamic Networks via Topological Data Analysis"
**Link:** https://arxiv.org/abs/2401.03194

- Applies persistent homology to dynamic networks
- Constructs probabilistic community networks → compute persistence
- Tracks birth/death of communities over time
- Persistence barcodes encode structural similarity
- **For our work:** Stability tie-break in Step 2 DP is inspired by persistence — prefer clusters with longer "lifespan" in the dendrogram

## 7. Core Leiden/CPM References

- **Leiden:** Traag, Waltman, van Eck (2019) "From Louvain to Leiden"
  https://www.nature.com/articles/s41598-019-41695-z
- **CPM theory:** Traag et al. (2011) "Narrow scope for resolution-limit-free community detection"
  https://arxiv.org/abs/1104.3083
- **CPM as hedonic game:** (2025) "From Leiden to Pleasure Island"
  https://arxiv.org/html/2509.03834v1
- **Recursive partitioning:** Li et al. (2022, JASA)
  https://arxiv.org/abs/1810.01509

## 8. Lance-Williams / Linkage Theory

- **Lance & Williams (1967):** "A general theory of classificatory sorting strategies"
  - Unified framework for agglomerative clustering update formulas
  - Our mass-weighted CPM family fits this framework exactly
- **Nearest-neighbor chain:** Applicable when dissimilarity is reducible
  - CPM density ρ and mass-weighted s_θ both satisfy reducibility
  - Enables O(n²) worst-case, O(m log²n) for sparse graphs

---

## Identified Gap (Revised)

| Exists | Missing |
|--------|---------|
| Paris: complete dendrogram, efficient | Resolution-limit-free (CPM-based) dendrogram |
| CPM: resolution-limit-free, subset property | Bottom-up dendrogram using CPM criterion |
| Constrained optimal cut on tree | Applied to large-scale community detection |
| Bottom-up > Top-down (proven 2025) | CPM density-based bottom-up agglomerative |
| Size-constrained partitioning | Size-constrained community detection WITH dendrogram |
| Average linkage theory (Lance-Williams) | Mass-weighted CPM linkage family |

**Core gap:** No algorithm builds a resolution-limit-free complete dendrogram via bottom-up agglomeration, with size-constrained optimal cut, at scale. Existing methods either use modularity (resolution limit) or produce flat partitions (no hierarchy).

**Our contribution is a framework, not a single score:**
1. CPM-critical tree as analyzable baseline
2. Mass-weighted extension (s_θ) as the novel, degree-aware family
3. Size-constrained DP cut with stability tie-break
4. Fair experimental comparison using same Step 2 across all baselines

---

## Key Theoretical Insight: ρ-Agglomeration = Average Linkage

The update formula for CPM density after merging A, B:
```
ρ(A∪B, C) = [n_A · ρ(A,C) + n_B · ρ(B,C)] / (n_A + n_B)
```

This is exactly the UPGMA (size-weighted average linkage) update.

More generally, for mass-weighted s_θ:
```
s_θ(A∪B, C) = [Θ(A) · s_θ(A,C) + Θ(B) · s_θ(B,C)] / [Θ(A) + Θ(B)]
```

This is average linkage with mass Θ as weights. The Lance-Williams coefficients are:
- α_A = Θ(A) / (Θ(A) + Θ(B))
- α_B = Θ(B) / (Θ(A) + Θ(B))
- β = 0, γ = 0

Reducibility follows immediately: the merged similarity is a convex combination of the original similarities, hence bounded above by the maximum.

---

## Critical Caveat: Greedy Tree ≠ CPM Global Optimum

The dendrogram cut at height γ is NOT the globally optimal CPM partition at γ.

**4-node counterexample:**
```
Graph: edges (0,1), (0,2), (1,2), (1,3)

Greedy ρ-tree cuts: {0,1,2,3} or {0,2},{1,3} or {0,1,2,3}
CPM optimum at γ=0.4: {0,1,2},{3}  ← not in the tree!
```

The tree is a **CPM-inspired laminar approximation**. This is stated honestly and positioned as a feature: the greedy tree provides a tractable, deterministic, reproducible structure that preserves merge gap information lost by flat methods.

---

## Open Research Directions (Updated)

1. **Mass-weighted CPM family (s_θ):** Evaluate θ_u = 1, √d_u, d_u^α on real networks
2. **Locality proof:** Does s_θ satisfy resolution-limit-freeness for all θ choices?
3. **HSBM recovery:** Exact/approximate recovery conditions under CPM-critical tree
4. **Structural preprocessing:** Triadic-closure edge weighting to mitigate singleton noise
5. **Scalability:** Practical NN-chain implementation for graphs with millions of edges
