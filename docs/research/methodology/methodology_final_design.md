# Final Methodology Design: CPM-Critical Hierarchical Community Detection

## Design Philosophy

> 힘들더라도, 최종 완성품이 좋게.

Every design choice prioritizes the **quality and rigor of the final result** over
implementation convenience. Where theory and practice conflict, choose the path
that is theoretically defensible AND empirically verifiable.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT                                       │
│  Weighted undirected graph G = (V, E, w)                        │
│  Minimum cluster size k                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 0: Preprocessing (Optional)                               │
│  Structural edge reweighting: triadic closure boost             │
│  w'(i,j) = w(i,j) · (1 + |common_neighbors(i,j)|)             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Dendrogram Construction                                │
│                                                                 │
│  Primary: Exact CPM-density agglomeration from singletons       │
│           = Average linkage HAC on graph                        │
│           Implementation: SeqHAC exact [Õ(n√m)]                 │
│                                                                 │
│  Merge criterion: ρ(A,B) = e_AB / (|A| · |B|)                  │
│  Merge height: γ*(A,B) = ρ(A,B)  (CPM critical resolution)     │
│  Tie-breaking: lexicographic (min_node_id(A), min_node_id(B))   │
│                                                                 │
│  Output: Complete binary dendrogram T                           │
│          Each internal node stores: merge height, subtree size  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Size-Constrained Optimal Cut                           │
│                                                                 │
│  Input: Dendrogram T, minimum size k                            │
│  Objective: max_lex(#clusters, total_stability)                 │
│  Constraint: every leaf cluster has ≥ k nodes                   │
│                                                                 │
│  Algorithm: Bottom-up greedy DP, O(n)                           │
│  Stability: p(v) = γ_birth(v) − γ_death(v)                     │
│             (resolution range where cluster v exists)           │
│                                                                 │
│  Output: Partition P = {C_1, ..., C_m} where |C_i| ≥ k ∀i     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Refinement (Optional)                                  │
│                                                                 │
│  Run Leiden-CPM at γ = median(cut heights)                      │
│  with initial_membership = Step 2 partition                     │
│  Accept only if: quality improves AND cluster count ≥ Step 2    │
│                                                                 │
│  Purpose: Correct greedy errors at cut boundary                 │
│  Note: Makes result seed-dependent — report both versions       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                         │
│  - Complete dendrogram T (reusable for any k)                   │
│  - Partition P at requested k                                   │
│  - Per-cluster metadata: size, internal density, stability      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decision 1: Singleton Start (Not Hybrid)

**Choice: Start from singletons, NOT Leiden warm-start.**

| Factor | Singleton | Hybrid (Leiden warm-start) |
|--------|-----------|---------------------------|
| Reproducibility | Perfect (deterministic) | Seed-dependent at micro level |
| Theoretical purity | One algorithm, one criterion | Two algorithms mixed |
| Dendrogram completeness | Full (all scales) | Pruned (missing within-micro) |
| Paper defensibility | Clean story | "Why not just use Leiden?" attack |
| Computation (our network) | Exact: ~minutes (SeqHAC) | Faster, but not needed |

**Rationale**: For 100K nodes / 3.28M edges, SeqHAC exact runs in minutes.
We don't NEED the hybrid shortcut. The singleton approach gives us:
- A cleaner paper (single principle, no mixing)
- Perfect reproducibility (zero stochasticity)
- Complete dendrogram (all resolution scales)
- No open theoretical question about equivalence

The hybrid pipeline is documented as a **scalability extension** for larger networks
(>1M nodes) where singleton-start becomes expensive. It is NOT the primary method.

---

## Decision 2: Exact Algorithm (Not Approximate)

**Choice: SeqHAC exact [Õ(n√m)], NOT (1+ε)-approximate.**

| Factor | Exact | (1+ε)-Approximate |
|--------|-------|-------------------|
| Dendrogram quality | Guaranteed optimal greedy | ~3% quality loss |
| Reproducibility | Bit-for-bit identical | ε-dependent variation |
| Our network cost | ~minutes | ~seconds |
| Paper claims | "Exact CPM-critical tree" | "Approximate" (weaker) |

**Rationale**: The time difference (minutes vs seconds) is negligible for our scale.
Exact gives us stronger claims and simpler analysis. No ε parameter to justify.

The (1+ε)-approximate version is documented for **production deployment** on larger
networks, but the paper results use exact.

---

## Decision 3: Implementation via GBBS Wrapping

**Choice: Wrap GBBS C++ implementation, NOT reimplement in Python.**

| Factor | GBBS wrap | Python reimplementation |
|--------|-----------|----------------------|
| Correctness | Battle-tested, published | Must verify from scratch |
| Performance | C++, optimized | 10-100x slower |
| Maintenance | Community-maintained | Our burden |
| Paper credibility | "Using published implementation" | "Our implementation" (scrutiny) |
| Integration effort | C++ build + Python binding | Pure Python |

**Rationale**: GBBS implements exactly what we need (average-linkage graph HAC).
Reimplementing introduces bug risk and adds no scientific value. The contribution
is the framework (CPM = average linkage + optimal cut), not a new HAC implementation.

**Fallback**: If GBBS integration proves too difficult, implement in Python with
scipy's priority queue for the contracted-graph hybrid approach (5K-10K nodes).
This is the backup plan, not the primary path.

---

## Decision 4: Pure CPM Density (No Mass-Weighting in Primary Method)

**Choice: θ_u = 1 (pure CPM density) as the primary method.**

| Factor | θ = 1 (pure CPM) | θ = d^α (mass-weighted) |
|--------|-------------------|------------------------|
| Resolution-limit-free | Guaranteed (Traag 2011) | Unknown — may break |
| Theoretical grounding | CPM theory, 15+ years | Novel, unproven |
| GBBS compatibility | Direct (= average linkage) | Requires modification |
| Paper risk | Safe baseline | Risky if RLF breaks |

**Rationale**: Traag's theorem says size bias is fundamental — you can't fix it
without breaking resolution-limit-freeness. Fighting this theorem is bad strategy.

Mass-weighted family (θ = √d, d^α) is reported as **experimental comparison** only:
- Run as ablation study
- If it works better empirically, note it as "empirically useful but
  resolution-limit-free status is open"
- Do NOT claim it as the main method

---

## Decision 5: Stability = Persistence in Dendrogram

**Choice: p(v) = γ_birth(v) − γ_death(v)**

For each cluster v in the dendrogram:
- **γ_birth(v)**: The merge height at which v was formed (its two children merged)
- **γ_death(v)**: The merge height at which v was consumed (merged into its parent)
- **p(v) = γ_birth − γ_death**: The resolution range where v exists as a distinct cluster

Why this definition:
- Directly interpretable: "This cluster is stable across γ ∈ [γ_death, γ_birth]"
- Connects to persistent homology / TDA literature
- Computable in O(1) per node from dendrogram heights
- Longer persistence = more robust cluster

**DP tie-break**: Among cuts with the same number of clusters, prefer the one
with higher total persistence Σ p(v) across all leaf clusters.

---

## Decision 6: Triadic Closure Preprocessing (Enabled by Default)

**Choice: Apply structural edge reweighting.**

```
w'(i,j) = w(i,j) · (1 + |common_neighbors(i,j)|)
```

| Factor | With preprocessing | Without |
|--------|-------------------|---------|
| Singleton noise | Mitigated | High |
| Cross-community citations | Dampened | Full weight |
| Triangle-rich communities | Strengthened | Same as noise |
| Computation cost | O(m·√m) for triangle counting | None |
| Reversibility | Report both variants | — |

**Rationale**: Citation networks have significant cross-field citation noise.
A single interdisciplinary citation should not have the same weight as a citation
embedded in a dense research community. Triadic closure naturally distinguishes these.

**In paper**: Report results with and without preprocessing as ablation.

---

## Decision 7: Optional Refinement (Report Both)

**Choice: Run Leiden refinement but report BOTH versions.**

- Version A: Pure dendrogram cut (deterministic, reproducible)
- Version B: Dendrogram cut + Leiden refinement (potentially higher quality)

Refinement details:
```python
# γ = median merge height at the cut level
γ_refine = median([γ_birth(v) for v in cut_clusters])

# Leiden with warm start
refined = leiden(G, resolution=γ_refine,
                 initial_membership=dp_cut_partition,
                 objective="cpm")

# Accept only if quality improves
if refined.quality > original_quality and n_clusters(refined) >= n_clusters(dp_cut):
    use refined
else:
    keep dp_cut_partition
```

**Rationale**: The paper's primary result is Version A (deterministic).
Version B is an "enhancement" section showing that Leiden refinement can
further improve quality while preserving the dendrogram's structural advantages.

---

## Complete Algorithm Pseudocode

```
Algorithm: CPM-Critical Hierarchical Community Detection

Input:  G = (V, E, w), minimum cluster size k
Output: Partition P where |C_i| ≥ k ∀i, |P| maximized

─── Step 0: Preprocessing ───
for each edge (i,j) ∈ E:
    t_ij ← |{v : (i,v) ∈ E and (j,v) ∈ E}|   // common neighbors
    w'(i,j) ← w(i,j) · (1 + t_ij)

─── Step 1: CPM-Critical Dendrogram ───
// Equivalent to exact average-linkage HAC on G' = (V, E, w')
T ← SeqHAC_exact(G', linkage=average)

// Each merge records:
//   height[v] = e_AB / (|A| · |B|)     (CPM critical resolution)
//   size[v]   = |A| + |B|

─── Step 2: Size-Constrained Optimal Cut ───
// Bottom-up DP on binary tree T
for each node v in T (bottom-up order):
    if v is a leaf (singleton):
        if k ≤ 1:
            opt[v] ← (1, 0)              // (count=1, stability=0)
        else:
            opt[v] ← (0, −∞)             // infeasible
    else:
        l, r ← children(v)
        split ← opt[l] + opt[r]          // component-wise addition
        keep  ← (1, persistence(v))      if size[v] ≥ k else (0, −∞)

        opt[v] ← lexmax(split, keep)     // prefer more clusters, then more stability

// Trace back to extract partition
P ← traceback(T, opt)

─── Step 3: Optional Refinement ───
γ* ← median({height[v] : v ∈ cut_nodes(P)})
P_refined ← Leiden(G, γ*, initial_membership=P, objective="cpm")
if quality(P_refined) > quality(P) and |P_refined| ≥ |P|:
    P ← P_refined

return P
```

---

## Complexity Analysis

| Step | Time | Space | Our network |
|------|------|-------|-------------|
| Step 0: Triangle counting | O(m · α) where α = arboricity | O(m) | ~seconds |
| Step 0: Edge reweighting | O(m) | O(m) | ~seconds |
| Step 1: SeqHAC exact | Õ(n√m) | O(n + m) | ~minutes |
| Step 2: DP cut | O(n) | O(n) | ~milliseconds |
| Step 3: Leiden refinement | O(m) per iteration | O(n + m) | ~minutes |
| **Total** | **Õ(n√m)** dominated by Step 1 | **O(n + m)** | **~minutes** |

For 100K nodes, 3.28M edges: **end-to-end in under 10 minutes on a single machine.**

---

## Experimental Plan

### Primary experiment: Break the ceiling

| Method | Expected clusters (k=1000) |
|--------|---------------------------|
| Leiden-CPM (γ=1e-4) + merge | 38 (current ceiling) |
| CPM-critical tree + DP cut | **? (target: >38)** |

If >38: the framework works. Proceed to full comparison.

### Full comparison matrix

| Method | Tree | Cut | Deterministic |
|--------|------|-----|:---:|
| Leiden-CPM + merge | Leiden single-γ | Post-hoc merge | ❌ |
| Paris + DP cut | Paris dendrogram | Our DP cut | ✅ |
| CPM-critical + DP cut | Our ρ-tree | Our DP cut | ✅ |
| CPM-critical + DP cut + refine | Our ρ-tree → Leiden | Our DP cut → Leiden | ❌ |
| Nested SBM (reference) | Bayesian MCMC | Model selection | ❌ |

### Ablation studies

1. **With/without triadic closure preprocessing**
2. **Exact vs (1+ε)-approximate dendrogram** (quality difference)
3. **With/without Leiden refinement** (Step 3)
4. **Mass-weighted θ = 1, √d, d^α** (experimental, note RLF caveat)
5. **Varying k** (500, 1000, 2000, 5000) on same dendrogram

### Quality metrics

| Metric | Definition | Priority |
|--------|-----------|:---:|
| Cluster count (≥k) | Number of clusters with ≥k nodes | 1st |
| Internal density | Σ e_internal / Σ n(n-1)/2 per cluster | 2nd |
| Conductance | External edges / min(vol(C), vol(V\C)) | 2nd |
| Keyword coherence | Top-50 keyword overlap with ground truth topics | 3rd |
| Connectedness | Are all clusters connected subgraphs? | Check |
| Stability | Variance across tie-breaking rules | Check |

---

## Deliverables

### Code
1. `sciscape/clustering/cpm_dendrogram.py` — GBBS wrapper + fallback Python HAC
2. `sciscape/clustering/constrained_cut.py` — O(n) DP with stability tie-break
3. `sciscape/clustering/triadic_preprocess.py` — Edge reweighting
4. Integration into `sciscape.landscape` pipeline

### Paper
1. Proposition 1: CPM density = average linkage (with proof)
2. Proposition 2: Merge height = CPM critical resolution (with proof)
3. Theorem 3: DP cut optimality and O(n) complexity (with proof)
4. Theorem 4 (stretch): HSBM recovery under CPM-critical tree
5. Experiments: ceiling-breaking, full comparison, ablations

### Reusable artifact
- Complete dendrogram for KRISS 100K network
- Queryable: any k → optimal partition in O(n)
- Publishable as supplementary data
