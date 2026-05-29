# Research Framework: Citation Network Clustering Methodology

## 1. Two Premises (대전제)

This research operates under two foundational premises. All sub-experiments
derive from and serve these premises.

### Premise 1: Network Structure → Community Structure

**Assumption**: Citation networks (direct citation, bibliographic coupling,
co-citation) encode meaningful research community relationships. By clustering
the network, we can discover coherent research topics/communities.

**Implication**: The choice of which citation signal to use (DC, BC, CC, or
combinations thereof) directly determines what "community" means. Different
link types capture different facets:

| Link Type | What it captures | Characteristics |
|-----------|-----------------|-----------------|
| DC (Direct Citation) | Intellectual lineage | Sparse, directional, high stability |
| BC (Bibliographic Coupling) | Shared knowledge base | Dense, symmetric, rich structure |
| CC (Co-citation) | Community recognition | Medium density, limited coverage (~70%) |
| ExtDC (Extended DC) | 2-hop intellectual proximity | Denser than DC, highest run-to-run NMI |

**Open question — Ground Truth Problem** (see Section 3).

### Premise 2: Overcome Limitations of Leiden + CPM + Modularity

**Goal**: Develop a methodology that addresses the known limitations of
current community detection methods for scientific paper networks.

**7 Limitations identified:**

| # | Limitation | Affects | Severity |
|---|-----------|---------|----------|
| L1 | **Modularity resolution limit** — cannot detect communities smaller than √(2m) internal edges (Fortunato & Barthélemy 2007) | Modularity-based methods | Fundamental |
| L2 | **CPM size bias** — ρ = e_ij/(n_i·n_j) penalizes large clusters disproportionately. Traag (2011): cannot be removed while preserving resolution-limit-freeness | CPM-based methods | Fundamental, must address in cut stage |
| L3 | **Single resolution per run** — one γ produces one flat partition, missing multi-scale structure | Leiden | Architectural |
| L4 | **No size constraints** — min_size is post-processing. At optimal γ=1e-4: 1,248 raw clusters → 38 after merge (97% destroyed). The "merge gap" (100-999 nodes) is where information is lost | Leiden + post-hoc merge | Critical |
| L5 | **Stochasticity / Hub instability** — different seeds → different results. Hub nodes (high-degree) are most affected. Ensemble NMI < 1.0 across runs | Leiden | Significant |
| L6 | **No proper dendrogram** — internal aggregation creates implicit hierarchy but discards it | Leiden | Architectural |
| L7 | **Existing hierarchical alternatives have own problems** — Paris inherits resolution limit (modularity-based); recursive partitioning has error lock-in and inversions | Paris, top-down methods | Known |

---

## 2. Three-Phase Research Pipeline

Each phase addresses specific limitations, and connects back to the premises.

```
Phase 1: Network Preprocessing     → Premise 1 + L5
Phase 2: Multi-resolution Stability → Premise 2 + L3, L5
Phase 3: Hierarchy + Constrained Cut → Premise 2 + L2, L4, L6
```

### Phase 1: Network Noise & Hub Handling

**Connects to**: Premise 1 (which signal?), L5 (hub instability)

| Sub-experiment | Question | Status |
|----------------|----------|--------|
| DC/BC/CC/ExtDC comparison | Which link type best captures communities? | Done (fields 34, 12) |
| Normalization (fractional, cosine, assoc_strength) | Does normalization reduce degree bias / hub instability? | Done |
| Backbone extraction (top-k, disparity, MDL) | Does edge filtering improve stability without losing quality? | Partially done (top-k) |
| Combination methods (sum, max, noisy-OR) | Does multi-signal fusion improve over single link types? | Done (field 34) |

**Key findings so far (field 34, 14,790 nodes):**
- DC: most stable (NodeStab 0.911), but sparsest
- BC: richest structure (1.1M edges), but hub-unstable without backbone
- Backbone top-k=30 on BC: hub stability 0.875 → 0.944, edges reduced 73%
- Best overall: dc+bc+cc_sum+topk30 (NodeStab 0.906, HubStab 0.940)
- Stability ≠ Quality: these are orthogonal axes (see Section 3)

**Existing scripts:**
- `scripts/build_linktype_edges.py` — builds DC/BC/CC/ExtDC from raw citations
- `scripts/eval_linktype_stability.py` — Leiden ensemble stability evaluation
- `scripts/combine_linktype_edges.py` — combination methods (sum/max/noisy-OR)
- `scripts/eval_quality_vs_stability.py` — quality vs stability comparison

### Phase 2: Multi-resolution + Stability

**Connects to**: L3 (single resolution), L5 (stochasticity)

- Leiden ensemble across resolution range
- Consensus clustering / resolution profile
- Identifying stable resolution bands

**Status**: Not yet started.

### Phase 3: Hierarchy + Size-Constrained Cut

**Connects to**: L2 (size bias), L4 (merge gap), L6 (no dendrogram)

**Core idea**: Separate structure discovery (dendrogram) from constraint
satisfaction (cut). CPM-density = average linkage → near-linear algorithms
exist (Dhulipala et al., ICML 2021 / NeurIPS 2022 / SIGMOD 2024).

**Status**: Theoretical framework complete. See:
- `docs/research/methodology/research_problem_statement.md` — full problem statement + theory
- `docs/research/methodology/methodology_final_design.md` — algorithm design
- `docs/research/methodology/literature_review_hierarchical_clustering.md` — literature survey

---

## 3. The Ground Truth Problem (미해결)

### Why subfield labels are invalid ground truth

Subfield labels (e.g., OpenAlex subfields) are themselves products of
clustering algorithms applied to citation networks. Using them to validate
a new clustering method creates **circular reasoning**:

```
Citation network → [some clustering] → Subfield labels
Citation network → [our clustering] → Evaluate against subfield labels
                                       ↑ circular!
```

This was identified as a fundamental methodological flaw during experiments
on field 12 (13 subfields). DC and BC showed essentially identical SF_NMI
(0.570 vs 0.567), which may reflect agreement with the original clustering
method rather than true quality.

### What constitutes valid ground truth?

This remains an open research question. Candidate independent signals:

| Signal | Independence | Scalability | Strength |
|--------|-------------|-------------|----------|
| **Text similarity** (title/abstract embeddings) | High — linguistic, not citation-based | High | Moderate (noisy) |
| **Author/affiliation overlap** | High — social, not citation-based | Medium | Moderate (sparse) |
| **Journal distribution** | Partial — journals reflect but don't define communities | High | Weak (too coarse) |
| **Expert judgment** | Highest — gold standard | Very low | Strong (but subjective) |
| **Downstream task performance** | High — pragmatic evaluation | Task-dependent | Task-dependent |
| **Keyword coherence** | High — content-based | High | Moderate |

**Key insight**: The field has not rigorously solved this problem. Even
Boyack, Klavans, Waltman, and Van Eck relied on subfield-level comparisons
or domain expert spot-checks. A proper answer to "what is correct clustering?"
may require multi-modal validation combining several independent signals.

### Stability vs Quality: Orthogonal Axes

Our experiments reveal that stability and quality are **not the same thing**:

- **DC dominates stability** but produces sparse, potentially less informative clusters
- **BC captures richer structure** but is less stable due to hub effects
- Boyack et al. claim BC > DC for quality; our data shows DC ≥ BC for stability
- These are not contradictory — they measure different things

A complete evaluation framework must measure BOTH axes and recognize the
tradeoff.

---

## 4. Evaluation Framework: Beyond Waltman's BM25 Accuracy

### 4.1. Waltman et al. (2020) Methodology — Review and Critique

Waltman, Boyack, Colavizza & Van Eck (2020) proposed the most principled
existing framework for evaluating citation-based clustering quality.
[QSS 1(2), 691-713. arXiv:1901.06815]

**Core idea**: Cluster by citation-based relatedness (DC, BC, CC), evaluate
by an independent text-based relatedness (BM25). Cross-modal evaluation
avoids self-reinforcing bias.

**Key formulas:**

Accuracy of clustering X evaluated by relatedness C:
```
A^{X|C} = (1/N) Σ_{i,j} I(c_i^X = c_j^X) · r_ij^C          (4)
```

Granularity (for fair comparison across methods):
```
G = N / Σ_k (s_k^X)²                                          (16)
```

BM25 relatedness between papers i and j:
```
r_ij^{BM25} = Σ_l I(o_il>0) · IDF_l · o_jl(k1+1) / [o_jl + k1(1-b+b·d_j/d̄)]
IDF_l = log[(N-n_l+0.5)/(n_l+0.5)]
k1=2.0, b=0.75, terms=noun+adj phrases from title+abstract
```

Consistency property: `A^{X|X} ≥ A^{Y|X}` for any Y (clustering by X
always wins when evaluated by X → evaluation criterion must be independent).

GA (Granularity-Accuracy) plots: log-log plot of G vs A for each
relatedness measure across 10 γ values (0.00001 to 0.01).

**Result**: BC ≈ EDC > DC-BC-CC > DC >> CC (evaluated by BM25).

### 4.2. Critical Review — Six Weaknesses

**W1. BM25 asymmetry** — `r_ij ≠ r_ji`. BM25 was designed for query→document
retrieval, not document↔document similarity. The accuracy formula implicitly
symmetrizes by summing both directions, but this specific symmetrization
(sum) has no justification over alternatives (max, geometric mean).

**W2. Text processing arbitrariness** — POS tagger choice (OpenNLP 1.5.2),
noun+adj only, longest-match-only counting, fixed parameters k1=2.0 b=0.75
borrowed from Boyack 2011 (biomedical). No sensitivity analysis shows these
choices don't affect the BC > DC conclusion.

**W3. Purely lexical matching** — BM25 has zero semantic understanding.
"Deep learning" and "neural network" share no terms → BM25=0. In 2025,
learned embeddings (SPECTER2, SciBERT) capture semantics and are strictly
superior for measuring topical relatedness.

**W4. Independence assumption violation** — The paper claims citation and
text are independent "noisy proxies of true relatedness." In reality, BC
high → same field → similar terminology → correlated BM25 signal. The noise
is NOT uncorrelated. This potentially biases the evaluation in favor of BC.

**W5. Cohesion-only metric, no separation** — Accuracy (4) only measures
within-cluster relatedness. It does not penalize splitting truly related
papers into different clusters (false negatives). A complete quality metric
requires both cohesion (precision) and separation (recall).

**W6. Size bias in accuracy** — Large clusters contribute quadratically
(s_k² pairs) to the accuracy sum. A few large clusters dominate the metric;
quality of small specialized clusters is effectively invisible.

### 4.3. Proposed Improved Evaluation Framework

Our framework addresses all six weaknesses through a two-phase design:

**Phase 0: Methodology Validation (Synthetic Data)**

Validate that our evaluation metrics correctly identify planted ground truth.

```
LFR network (planted communities with known structure)
  + Topic assignment per community
  + Text generation per node (topic-based vocabulary)
  → Citation structure and text correlated but with independent noise
  → Planted truth available for exact recovery measurement
```

Metric validation:
- Our metrics should rank methods correctly against planted truth
- Compare sensitivity: our metrics vs Waltman's BM25 accuracy
- Demonstrate that our framework detects failures that BM25 accuracy misses

**Phase 1: Real Data Evaluation (Multi-modal)**

Three independent signals, each capturing different facets of relatedness:

```
Signal 1: Text (semantic)
├── Method: SPECTER2 or text-only embedding (all-MiniLM-L6)
├── Metric: cosine similarity between paper embeddings
├── Independence: high (linguistic, not citation-based)
├── Advantage over BM25: symmetric, semantic, no arbitrary parameters
└── Caveat: SPECTER2 is partially citation-informed;
    text-only embedding as purity check

Signal 2: Social (author/affiliation overlap)
├── Method: Jaccard(authors_i, authors_j) or co-authorship network
├── Independence: very high (social signal, orthogonal to both citation and text)
├── Advantage: truly independent modality
└── Caveat: sparse (most paper pairs share no authors)

Signal 3: Structural (information-theoretic)
├── Method: description length / network compression quality
├── Independence: N/A (no external signal needed)
├── Advantage: avoids ground truth problem entirely
└── Caveat: measures structural compression, not semantic meaningfulness
```

**Improved accuracy metrics:**

M1. Silhouette-based accuracy (addresses W5 cohesion-only + W6 size bias):
```
s_i = (b_i - a_i) / max(a_i, b_i)
a_i = mean similarity to same-cluster papers
b_i = mean similarity to nearest-cluster papers
Accuracy = mean(s_i)  — equal weight per paper, not per pair
```

M2. Precision-Recall framework (addresses W5):
```
"True positive pair": embedding_sim(i,j) > threshold
Precision = |same_cluster ∩ text_similar| / |same_cluster|
Recall    = |same_cluster ∩ text_similar| / |text_similar|
F1 = 2·P·R / (P+R)
```

M3. Cluster-level unweighted accuracy (addresses W6):
```
A_unweighted = (1/K) Σ_k mean_{i,j ∈ k} sim(i,j)
```

**Improved granularity comparison:**

Effective number of clusters (more interpretable than N/Σs²):
```
K_eff = exp(H),  H = -Σ_k p_k log(p_k),  p_k = s_k / N
```

Pareto front visualization across multiple objectives:
```
Axes: (Stability, Text coherence, Structural quality, Coverage)
Best method = on the Pareto front (not dominated on any axis)
```

### 4.4. Multi-modal Consensus as Quality Definition

Instead of privileging any single evaluation signal, define quality as
the degree of agreement across independent modalities:

```
Citation clustering ←→ Text clustering:   NMI_ct
Citation clustering ←→ Author clustering: NMI_ca
Text clustering     ←→ Author clustering: NMI_ta
```

The citation-based method (DC, BC, CC, etc.) whose clustering achieves
the highest agreement with BOTH text and author modalities simultaneously
is the most robustly accurate. No single modality is treated as ground truth.

This addresses the fundamental criticism that any single evaluation criterion
(including BM25) is itself a noisy, biased proxy.

### 4.5. Connection to Research Phases

```
Evaluation framework serves ALL three phases:

Phase 0 (synthetic validation)
  └─→ Confirms our metrics work before applying to real data

Phase 1 (link type comparison)
  └─→ Which DC/BC/CC/combination + backbone is best?
       Evaluated by multi-modal framework, not just BM25 or subfield NMI

Phase 2 (multi-resolution)
  └─→ GA plot with improved metrics across resolution range

Phase 3 (dendrogram + cut)
  └─→ Does our dendrogram approach beat flat Leiden?
       Evaluated by the same framework → fair comparison
```

---

## 5. Connection Map (updated): Sub-experiments → Premises → Limitations

```
Premise 1: Network → Communities
├── Which signal? ──────────── DC/BC/CC/ExtDC comparison (Phase 1)
├── How to combine? ─────────── Combination methods (Phase 1)
└── How to validate? ─────────── Ground truth problem (Section 3, OPEN)

Premise 2: Overcome Leiden+CPM limitations
├── L1 (resolution limit) ────── Use CPM, not modularity (baseline decision)
├── L2 (size bias) ────────────── Address in cut stage, not merge (Phase 3)
├── L3 (single resolution) ───── Multi-resolution profiling (Phase 2)
├── L4 (merge gap) ────────────── Dendrogram + DP cut (Phase 3)
├── L5 (hub instability) ─────── Normalization + Backbone (Phase 1)
│                                  Ensemble consensus (Phase 2)
├── L6 (no dendrogram) ────────── CPM-density HAC (Phase 3)
└── L7 (alt. method flaws) ────── Our method avoids Paris resolution limit
                                   and top-down error lock-in (Phase 3)
```

---

## 6. Experimental Datasets

4 OpenAlex oa26 fields (full population, GCC + k-core(30)):

| Field | Nodes | Status |
|-------|------:|--------|
| 34 | 14,790 | Phase 1 complete |
| 12 | 29,007 | Phase 1 partially complete |
| 30 | 43,284 | Not started |
| 29 | 52,450 | Not started |

Data locations: see `scripts/build_linktype_edges.py` header and
memory file `project_sample_datasets.md`.

---

## 7. Related Documents

| Document | Content |
|----------|---------|
| `docs/research/methodology/research_problem_statement.md` | Phase 3 theory: CPM-density = avg linkage, DP cut, proofs |
| `docs/research/methodology/methodology_final_design.md` | Phase 3 algorithm design |
| `docs/research/methodology/literature_review_hierarchical_clustering.md` | HAC + community detection literature |
| `docs/research/dendrogram/implementation_plan.md` | Phase 3 implementation roadmap |
| `docs/developer/decisions.md` | Keyword pipeline design decisions (separate stream) |

### Key References for Evaluation Framework

| Paper | Key contribution | Our use |
|-------|-----------------|---------|
| Waltman et al. (2020) QSS 1(2) [arXiv:1901.06815] | BM25 accuracy + GA plot + consistency property | Baseline to improve upon |
| Boyack et al. (2011) PLoS ONE 6(3) [PMC3060097] | 9 text-based similarity comparison; BM25 identified as best | Why BM25 was chosen |
| Boyack & Klavans (2010) JASIST 61(12) | JSD textual coherence + grant concentration | Earlier evaluation approach |
| Lancichinetti et al. (2008) PRE 78(4) | LFR benchmark networks | Synthetic validation |

---

## 8. What This Document Does NOT Cover

- **Keyword extraction pipeline**: Separate stream (`sciscape/keyword_extraction/`).
  See `docs/developer/decisions.md` and memory `keyword_pipeline_design.md`.
- **Implementation details**: See individual script headers and `docs/research/dendrogram/implementation_plan.md`.
- **Literature review**: See `docs/research/methodology/literature_review_hierarchical_clustering.md` and
  memory `reference_citation_normalization_literature.md`.
