# Expert Feedback Log

Chronological record of external expert feedback on the hierarchical community detection research.

---

## Feedback 1 (2026-03-16): Framework framing and mass-weighted family

### Key points

1. **"New framework" not "new score"**: The novelty is the tree-cut separation
   framework for resolution-limit-free community detection, not the merge criterion itself.

2. **Pure ρ = average linkage**: CPM density e_AB/(|A||B|) is structurally identical
   to average linkage on the weighted adjacency matrix. The reducibility proof from
   Paris (Bonald 2018) carries over identically. This reduces algorithmic novelty but
   makes the baseline "exactly analyzable."

3. **4-node counterexample**: Greedy ρ-tree ≠ CPM global optimum.
   Graph: edges (0,1), (0,2), (1,2), (1,3).
   Tree cuts: {0,1,2,3}, {0,2},{1,3}, {0,1,2,3}.
   CPM optimum at γ=0.4: {0,1,2},{3} — not in the tree.
   Recommend naming: "CPM-critical dendrogram" not "CPM-optimal."

4. **Mass-weighted CPM family** as the real novelty:
   s_θ(A,B) = e_AB / (Θ(A)·Θ(B)), Θ(C) = Σ θ_u.
   Preserves Lance-Williams form, reducibility, NNC applicability.
   Candidates: θ_u = 1, √d_u, d_u^α.
   Density ratio and local null model both break additive structure — avoid.

5. **Step 2 stability tie-break**: Lexicographic optimization
   max(#clusters, total_stability) to avoid unstable cuts.

6. **Four theorems to prove**: (1) ρ = average linkage, (2) s_θ = CPM critical
   merge threshold, (3) DP cut correctness/complexity, (4) HSBM recovery.

7. **Experimental design**: Apply same Step 2 (DP cut) to all baselines (Paris,
   recursive partitioning, Leiden+merge, our ρ-tree, our s_θ-tree) to isolate
   tree quality from cut quality.

### Action taken
- Updated research_problem_statement.md and literature_review.md
- Committed as `1170ac8`

---

## Feedback 2 (2026-03-16): Near-linear algorithms and fundamental constraints

### Key points

1. **CPM density = average linkage unlocks Dhulipala et al. stack**:
   - SeqHAC exact: Õ(n√m) — ICML 2021
   - SeqHAC approx: Õ(m) with (1+ε) — ICML 2021
   - ParHAC: Õ(m) parallel — NeurIPS 2022
   - TeraHAC: distributed, 8T edges — SIGMOD 2024
   - Open source: GBBS (github.com/ParAlg/gbbs)
   - Our network (100K V, 3.28M E): exact = minutes, approx = seconds

2. **Bateni et al. (ICALP 2024) lower bound**: Exact average-linkage requires
   Ω(n^{3/2−ε}). SeqHAC is essentially tight. Use approx for production.

3. **Size bias is FUNDAMENTAL (Traag 2011)**: CPM is essentially the ONLY
   resolution-limit-free method. Size bias cannot be removed while maintaining
   resolution-limit-freeness. Any correction (density ratio, degree normalization)
   reintroduces non-local information → resolution limits. Address in cut, not merge.

4. **Density ratio is dangerous**: max(ρ_within) denominator is non-monotonic
   under merging → likely breaks reducibility → dendrogram inversions. Avoid.

5. **O(n) greedy DP is provably optimal** for size-constrained cut:
   Split always dominates keeping when valid (≥2 vs 1). Single bottom-up pass.

6. **Hybrid pipeline is theoretically justified**:
   - CPM subset property → nesting guarantee
   - Loukas (2019 JMLR) → spectral coarsening preserves structure
   - Ghoroghchian (2021) → coarsening preserves communities iff micro-clusters
     don't cross boundaries
   - Scanpy → practical precedent in bioinformatics

7. **Peixoto's nested SBM** as validation benchmark (Bayesian, not competitor).

8. **Concrete 5-phase pipeline recommendation**:
   Phase 1: Leiden micro-clustering (seconds)
   Phase 2: Contract graph (seconds)
   Phase 3: CPM-density HAC via SeqHAC/GBBS (minutes)
   Phase 4: Size-constrained DP cut (milliseconds)
   Phase 5: Optional Leiden refinement (minutes)

9. **Mass-weighted family (s_θ) downgraded**: May break resolution-limit-freeness.
   Report as experimental comparison, not main method.

10. **Open theoretical question**: No proof that Leiden-warm-start dendrogram =
    singleton-start dendrogram. Ingredients (CPM subset property, nesting) exist
    but formal proof is missing.

### Action taken
- Major revision of both research documents
- Committed as `d805bee`

---

## Summary of direction changes

| Before feedback | After feedback |
|----------------|---------------|
| Mass-weighted s_θ is main novelty | s_θ is experimental extension only |
| Need to implement HAC from scratch | GBBS/ParHAC available as open-source |
| Size bias fixable in merge criterion | Size bias fundamental (Traag theorem) |
| Density ratio as promising alternative | Density ratio dangerous (breaks reducibility) |
| Unsure about computational feasibility | Near-linear algorithms solve it completely |
| Framework framing was secondary | Framework IS the primary contribution |
