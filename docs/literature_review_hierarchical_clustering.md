# Literature Review: Hierarchical Community Detection for Scientific Networks

## Problem Statement

Build a complete hierarchical tree (dendrogram) from individual papers to a single root
using a single criterion, then find a size-constrained optimal cut.

**Two-step framework:**
- Step 1: Build dendrogram (structure discovery)
- Step 2: Size-constrained optimal cut (constraint satisfaction)

**Key equivalence discovered**: CPM density = average linkage on graphs. This unlocks
near-linear-time algorithms from the graph HAC literature.

---

## Core References

### 1. Paris Algorithm (Bonald et al., 2018)

**Paper:** "Hierarchical Graph Clustering using Node Pair Sampling"
**Link:** https://arxiv.org/abs/1806.01664

- Bottom-up agglomerative using node pair sampling probability as distance
- Distance d(a,b) = w(a,b)/(w(a)·w(b)) where w(·) = degree-based weights
- Nearest-neighbor chain (NNC) for O(m log²n)
- Proves reducibility for modularity-based distance
- Produces complete dendrogram — run once, all resolutions embedded
- Implemented in scikit-network
- **Critical for our work**: Replacing degree-based weights with size-based weights yields CPM density. The reducibility proof carries over identically. **No published paper noted this.**

### 2. Bottom-up vs Top-down (Dreveton et al., 2025, JASA)

**Paper:** "When does bottom-up beat top-down in hierarchical community detection?"
**Link:** https://arxiv.org/abs/2306.00833

- Bottom-up achieves exact recovery at intermediate depths up to info-theoretic threshold
- Top-down (recursive partitioning) fails due to error lock-in and inversions
- Uses Bethe-Hessian spectral method + agglomerative hierarchy
- **Takeaway:** Bottom-up is theoretically superior
- **For our work:** Proof template for HSBM recovery under CPM-critical tree

### 3. CPM Theory (Traag et al., 2011)

**Paper:** "Narrow scope for resolution-limit-free community detection"
**Link:** https://arxiv.org/abs/1104.3083

- **Fundamental theorem:** CPM is essentially the ONLY resolution-limit-free method
- Resolution-limit-free = subpartition restricted to any induced subgraph equals restriction of global partition
- Only quality functions with purely local pairwise contributions qualify
- CPM: H = Σ_c [e_c − γ·C(n_c,2)] qualifies; modularity does not
- **Critical implication:** Size bias is INHERENT to CPM. Cannot be removed while staying resolution-limit-free. Any correction (density ratio, degree normalization) reintroduces non-local information → resolution limits.

### 4. Leiden Algorithm (Traag et al., 2019)

**Paper:** "From Louvain to Leiden: guaranteeing well-connected communities"
**Link:** https://www.nature.com/articles/s41598-019-41695-z

- 3-phase: local moves → refinement → graph aggregation → repeat
- Connectivity guarantee (unlike Louvain)
- `initial_membership` and `node_sizes` parameters allow warm-starting from non-singleton partition
- `resolution_profile()` efficiently scans all CPM resolution values, identifying transition points
- **For our work:** Leiden micro-clustering as warm-start for HAC (hybrid pipeline)

---

## Near-Linear Average-Linkage HAC (= CPM Density HAC)

### 5. SeqHAC (Dhulipala et al., ICML 2021)

**Paper:** "Hierarchical Agglomerative Graph Clustering in Nearly-Linear Time"

- Exact: Õ(n√m) time
- Approximate: Õ(m) time with (1+ε) approximation
- Core: neighbor-heap data structure, lazy evaluation for reducible linkages
- Dynamic low-outdegree graph orientations (bounded by arboricity α)
- For sparse citation networks (α ≈ O(√n)): major practical speedups
- **Our network (100K V, 3.28M E):** exact ≈ minutes, approx ≈ seconds

### 6. ParHAC (Dhulipala et al., NeurIPS 2022)

**Paper:** "Parallel Hierarchical Agglomerative Graph Clustering"

- Shared-memory parallel version
- Õ(m) work, polylog depth
- Demonstrated on ~100B edges

### 7. TeraHAC (Dhulipala et al., SIGMOD 2024)

**Paper:** "TeraHAC: Hierarchical Agglomerative Clustering of Trillion-Edge Graphs"

- Distributed (MapReduce)
- "(1+ε)-good merge" innovation: merge locally checkable → independent merges across partitions
- Demonstrated on **8 trillion edges**
- 100× fewer communication rounds than prior distributed HAC

### 8. Lower Bound (Bateni et al., ICALP 2024)

**Paper:** "Optimal Bounds for Approximate Average-Linkage Clustering"

- Exact average-linkage graph HAC requires **Ω(n^{3/2−ε})** time
- Confirms SeqHAC exact Õ(n√m) is essentially tight
- **Implication:** Use (1+ε)-approximate Õ(m) for production, exact for validation

**Open source:** GBBS (github.com/ParAlg/gbbs), ParHAC (github.com/ParAlg/ParHAC)

---

## Constrained Cutting and Tree Optimization

### 9. Constrained Hierarchical Clustering (Mauduit & Simonetto, 2024)

**Paper:** "Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts"
**Link:** https://arxiv.org/abs/2312.04209

- Two-step: (1) graph coarsening → hierarchy, (2) optimal cut with constraints
- Horizontal constraints (cannot-link, must-link) + vertical (level precedence)
- **Takeaway:** "Tree + optimal cut" paradigm is established

### 10. Size-Constrained Optimal Cut

**Simple O(n) greedy DP** for maximizing clusters with min-size k:
- Split if both children independently form valid partitions (always dominates keeping)
- Single bottom-up pass, provably optimal
- No approximation, no heuristics

For (l,u)-partition (both min and max sizes): O(n⁵) via interval compression DP
(Buchin & Selbach 2022, Ito et al. Algorithmica 2012)

### 11. Size-Constrained Graph Clustering (Meyerhenke et al., 2016)

**Paper:** "Partitioning (hierarchically clustered) complex networks via size-constrained graph clustering"
**Link:** https://link.springer.com/article/10.1007/s10732-016-9315-8

- Size-constrained label propagation (SCLaP)
- Quality comparable to hMetis, 10x faster
- **Limitation:** Graph partitioning objective, not community detection

---

## Graph Coarsening Theory

### 12. Spectral Coarsening (Loukas, 2019, JMLR)

**Paper:** "Graph Reduction with Spectral and Cut Guarantees"

- Restricted spectral approximation: coarsened graph ε-approximates first k eigenvectors/eigenvalues
- Cut values approximately preserved
- For moderate coarsening (100K → 5K-10K): strong spectral preservation
- **Justifies contracted-graph approach:** HAC on Leiden-contracted graph faithfully represents coarse structure

### 13. Community Detection from Coarse Measurements (Ghoroghchian et al., 2021)

- Coarsening preserves community structure **if and only if** micro-clusters don't cross true community boundaries
- CPM subset property (nesting of optimal partitions) provides this guarantee in theory
- Leiden's heuristic introduces some boundary imprecision in practice

---

## Bayesian Hierarchical Approach

### 14. Peixoto's Nested SBM (Phys. Rev. X, 2014)

**Paper:** "Hierarchical Block Structures and High-Resolution Model Selection in Large Networks"

- Overcomes resolution limit via recursive hierarchical priors
- Achieves B_max = O(N/log N) detectable blocks
- MCMC inference: 2-12 hours for 100K nodes
- Implemented in graph-tool
- Different philosophy: Bayesian model selection vs. optimization
- **For our work:** Gold-standard validation benchmark, not competitor

---

## Multi-Scale Frameworks

### 15. Multiresolution Consensus Clustering (Jeub et al., 2018)

**Paper:** "Multiresolution Consensus Clustering in Complex Networks"
**Link:** https://www.nature.com/articles/s41598-018-21352-7

- Samples resolution parameters, combines into hierarchical consensus
- Closest published method to hybrid pipeline approach

### 16. PyGenStability (Arnaudon et al., 2024)

- Markov Stability framework using Leiden/Louvain at each scale
- Identifies robust partitions as those persistent across scales AND reproducible across optimizer runs

### 17. Scanpy's Bioinformatics Pipeline

- Runs Leiden at multiple resolutions, builds dendrogram of clusters
- Exactly the micro-cluster → hierarchy approach
- Widely validated on single-cell datasets
- **Practical precedent for our hybrid pipeline**

### 18. Persistent Community Detection via TDA (AAAI 2024)

**Paper:** "Learning Persistent Community Structures in Dynamic Networks via Topological Data Analysis"
**Link:** https://arxiv.org/abs/2401.03194

- Persistent homology on dynamic networks
- Persistence barcodes for community birth/death
- **For our work:** Stability concept in DP cut tie-breaking

---

## Additional References

- **Recursive partitioning:** Li et al. (2022, JASA) https://arxiv.org/abs/1810.01509
- **Consensus clustering:** Lancichinetti & Fortunato (2012) https://www.nature.com/articles/srep00336
- **CPM as hedonic game:** (2025) "From Leiden to Pleasure Island" https://arxiv.org/html/2509.03834v1
- **Resolution limit:** Fortunato & Barthélemy (2007) PNAS
- **Lance-Williams:** Lance & Williams (1967) "A general theory of classificatory sorting strategies"
- **PASCO:** Lasalle et al. (2025) — parallel structured coarsening framework

---

## Gap Summary (Final)

| Exists | Missing | Our Contribution |
|--------|---------|-----------------|
| Paris: modularity-based dendrogram | CPM-based (resolution-limit-free) dendrogram | CPM-critical tree via average-linkage equivalence |
| Near-linear average-linkage HAC | Applied to community detection | Direct application of Dhulipala et al. |
| Constrained optimal cut on trees | Applied to community detection dendrograms | O(n) DP with stability tie-break |
| Bottom-up > Top-down (proven) | CPM density bottom-up dendrogram | Theoretical analysis under HSBM |
| Leiden micro-clustering | Combined with proper HAC dendrogram | Hybrid pipeline with subset-property guarantee |

**The key insight no one has published:** CPM density = average linkage, therefore all near-linear graph HAC algorithms produce resolution-limit-free dendrograms directly. Combined with O(n) optimal cutting under size constraints, this solves the 97% information loss problem in Leiden-CPM + post-hoc merge pipelines.
