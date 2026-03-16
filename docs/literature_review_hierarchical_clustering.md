# Literature Review: Hierarchical Community Detection for Scientific Networks

## Problem Statement

Build a complete hierarchical tree (dendrogram) from individual papers to a single root
using a single optimal criterion, then find a size-constrained optimal cut.

**Two-step framework:**
- Step 1: Build dendrogram (the hard problem)
- Step 2: Size-constrained optimal cut on the tree (well-defined optimization)

---

## 1. Paris Algorithm (Bonald et al., 2018)

**Paper:** "Hierarchical Graph Clustering using Node Pair Sampling"
**Link:** https://arxiv.org/pdf/1806.01664

- Bottom-up agglomerative using node pair sampling probability as distance
- Nearest-neighbor chain technique for acceleration
- Produces complete dendrogram — run once, all resolutions embedded
- Distance: modularity-based → inherits resolution limit
- Implemented in scikit-network
- **Gap:** No CPM-based variant exists

## 2. When Does Bottom-up Beat Top-down? (2025, JASA)

**Paper:** "When does bottom-up beat top-down in hierarchical community detection?"
**Link:** https://arxiv.org/abs/2306.00833

- Bottom-up achieves exact recovery at intermediate depths up to info-theoretic threshold
- Top-down (recursive partitioning) fails due to error lock-in and propagation
- Top-down dendrograms suffer from inversions (structural distortions)
- Bottom-up uses Bethe-Hessian spectral method + agglomerative hierarchy
- **Takeaway:** Bottom-up is theoretically superior for tree construction

## 3. Constrained Hierarchical Clustering via Graph Coarsening (Mauduit & Simonetto, 2024)

**Paper:** "Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts"
**Link:** https://arxiv.org/abs/2312.04209

- Two-step: (1) graph coarsening → hierarchy, (2) optimal cut with constraints
- Horizontal constraints (cannot-link, must-link) + vertical (level precedence)
- Natural extension of Loukas's local-variation graph coarsening
- Constraint violation < 5%, Dasgupta's cost improved 19%
- Applied to NLP word clustering, not large-scale network community detection
- **Takeaway:** "Tree + optimal cut" paradigm is established but not yet applied to our setting

## 4. Consensus Clustering (Lancichinetti & Fortunato, 2012)

**Paper:** "Consensus clustering in complex networks"
**Link:** https://www.nature.com/articles/srep00336

- Run community detection at multiple resolutions/seeds
- Build co-clustering matrix → re-cluster for stable partition
- Significantly improves stability and accuracy
- Does NOT produce a dendrogram — flat partition only
- Multiresolution extension (2018): https://www.nature.com/articles/s41598-018-21352-7

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
- **Potential:** Apply persistence concept to γ-sweep or temporal evolution

## 7. Core Leiden/CPM References

- **Leiden:** Traag, Waltman, van Eck (2019) "From Louvain to Leiden"
  https://www.nature.com/articles/s41598-019-41695-z
- **CPM theory:** Traag et al. (2011) "Narrow scope for resolution-limit-free community detection"
  https://arxiv.org/abs/1104.3083
- **CPM as hedonic game:** (2025) "From Leiden to Pleasure Island"
  https://arxiv.org/html/2509.03834v1
- **Recursive partitioning:** Li et al. (2022, JASA)
  https://arxiv.org/abs/1810.01509

---

## Identified Gap

| Exists | Missing |
|--------|---------|
| Paris: complete dendrogram, efficient | Paris + CPM (resolution-limit-free dendrogram) |
| CPM: resolution-limit-free, subset property | CPM-based complete dendrogram algorithm |
| Constrained optimal cut on tree | Applied to large-scale network community detection |
| Bottom-up > Top-down (proven 2025) | CPM distance-based bottom-up agglomerative |
| Size-constrained partitioning | Size-constrained community detection WITH dendrogram |
| TDA persistent communities (temporal) | Spatio-temporal dendrogram for scientific networks |

**Core gap:** A bottom-up dendrogram algorithm using CPM's density criterion,
combined with size-constrained optimal cut, does not yet exist.

---

## Open Research Directions

1. **CPM-Paris hybrid:** Replace Paris's modularity-based distance with CPM density criterion
2. **γ-sweep dendrogram:** Use CPM subset property to construct implicit dendrogram from resolution sweep
3. **Temporal integration:** Incorporate publication year into dendrogram construction for scientific networks
4. **Constrained cut on CPM dendrogram:** Apply Mauduit & Simonetto's optimal cut framework
