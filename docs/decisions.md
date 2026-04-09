# Decision Log

## Format
Each entry: `[DATE] [PHASE] DECISION: ... REASON: ... ALTERNATIVES: ...`
Items marked **REVIEW NEEDED** require user confirmation.

---

## [2026-03-11] [Phase A] DECISION: Use mixin pattern for code extraction
**REASON**: LLM canonicalization code (~1,500 lines) is deeply coupled to Pipeline class
(accesses self.config, self.C_all, self.DF_all, self.feature_names_all, etc.).
Mixin pattern allows moving code to separate file with ZERO logic changes.
Pipeline inherits from mixin → methods available via MRO.
**ALTERNATIVES**: (a) Standalone functions with explicit params — too many params, risky.
(b) Helper class with pipeline reference — adds indirection.
Mixin chosen as safest for Phase A (pure restructuring).

## [2026-03-11] [Phase A] DECISION: temporal.py also uses mixin pattern
**REASON**: Consistency with llm_canonicalize.py approach. _compute_year_series accesses
self.config, self.vec_uni, self._data, self.cluster_year_token_denoms.
Standalone function would need 6+ parameters.

## [2026-03-11] [Phase A.2] DECISION: extraction.py contains utilities + _DataSource
**REASON**: Standalone functions and _DataSource have no self-coupling to Pipeline.
Clean separation: extraction.py = data loading + math utils, pipeline.py = orchestration.
llm_canonicalize.py keeps its own local _normalize_text_basic copy (avoids circular import).

## [2026-03-12] [Phase B] DECISION: Two-level normalization strategy
**REASON**: Vocabulary merge (Stage 2, pre-scoring) and keyword normalization (Stage 5,
post-scoring) serve different purposes. Vocab merge operates on sparse matrix columns
(safe, reversible, no semantic judgment). Keyword normalization operates on scored terms
(Greek letters, abbreviation expansion, edit-distance merge).
**KEY DESIGN**: Edit-distance merge only fires when minor-form frequency is very low
(< 1% of major form) to avoid merging distinct concepts like "model"/"modal".

## [2026-03-12] [Phase C] DECISION: Blocking strategy for term network
**REASON**: O(n^2) pairwise comparison is infeasible for 100K+ terms. Token-based
blocking groups terms sharing any word. Special "_abbrev" block pairs short terms
(<=5 chars) with multi-word terms for abbreviation detection.
**BUG FIX**: Abbreviation initials must use original word ORDER (list), not set iteration
which has non-deterministic order in Python.

## [2026-03-12] [Phase C] DECISION: PMI normalization for co-occurrence layer
**REASON**: Raw co-occurrence counts are biased by term frequency. Symmetric normalization
D^{-1/2} * C * D^{-1/2} produces values in [0,1] comparable across term pairs.

## [2026-03-12] [Phase D] DECISION: Quantile-based depth levels
**REASON**: Fixed thresholds would break across datasets with different score distributions.
Quantile-based assignment (np.digitize on quantile boundaries) is robust and adaptive.
**SIGNALS**: doc_coverage (inverted), cross_cluster_count (inverted), ngram_length,
co-occurrence asymmetry P(A|B) vs P(B|A). All min-max normalized before combination.

## [2026-03-11] [Phase A.5+A.7] DECISION: Combine pipeline.py creation and keyword_extraction.py thinning
**REASON**: After extracting utilities (A.2), llm_canonicalize (A.3), temporal (A.4),
keyword_extraction.py contained ONLY the Pipeline class. No intermediate step needed.
keyword_extraction.py becomes 13-line re-export shim for backward compatibility.
**RESULT**: Original 2,918-line monolith → 7 focused modules:
  config.py (160), extraction.py (326), pipeline.py (597),
  llm_canonicalize.py (1712), temporal.py (218),
  diagnostics.py (397), keyword_extraction.py (13 shim)

## [2026-03-12] [Algorithm] DECISION: Stage 7→8 bridge injects candidates column
**REASON**: term_network (Stage 7) produces group-based merge_candidates, but
llm_canonicalize (Stage 8) expects per-row "candidates" column. Bridge method
`_bridge_merge_candidates` converts between formats without modifying either stage's
internal logic. Harmless when no merge groups exist.

## [2026-03-12] [Algorithm] DECISION: vocab_merge frequency ratio gate
**REASON**: `build_merge_map` now accepts the count matrix and checks
`minor_freq / major_freq <= merge_frequency_ratio`. Prevents false merges like
"aids" (HIV/AIDS) → "aid" (assistance) when both forms have comparable frequency.
Backward-compatible: without count matrix, all merges pass.

## [2026-03-12] [Algorithm] DECISION: Normalization blocking strategy
**REASON**: O(n²) pairwise edit-distance was infeasible at scale. Token-based blocking
groups terms by shared words (multi-word) or prefix (single-word). Length filter
provides additional O(1) pruning. Preserves correctness since edit distance ≥ length diff.

## [2026-03-12] [Phase A.8] DECISION: Generalized checkpoint system
**REASON**: Existing save/load_stage2_snapshot only supported one stage boundary.
New system supports arbitrary stage names (scoring, normalization, cooccurrence,
term_network, canonicalize). Dict/list columns serialized as JSON for parquet compat.
run_from_checkpoint resumes from the next stage. Legacy API preserved as thin wrappers.

## [2026-03-12] [Output] DECISION: Expose cross_cluster_count in depth output
**REASON**: cross_cluster_count was computed internally in depth.py but not exposed.
Tier 3 output schema now includes it. Useful for downstream analysis (bridge terms
between research communities).

## [2026-03-12] [Quality] DECISION: 6 quality filters (P1–P6)
**REASON**: pilot (10K docs, 5 clusters) revealed 17% of keywords were academic
boilerplate ("based", "using", "results"), 17 plural-duplicate pairs, and artifacts.
**P1** Academic stopword filter: domain-agnostic single+multi-word removal. Multi-word
terms filtered only when ALL tokens are academic stopwords ("proposed method" → filtered,
"fault diagnosis" → kept). ~40 common academic verbs/nouns in builtin set.
**P2** Plural merge in Stage 5: `_phrase_singular` singularizes last word of phrases
("point clouds" → "point cloud"). Reuses `_simple_singular` from vocab_merge.
**P3** Auto-merge high-confidence term network groups without LLM: merges when all pair
similarities exceed `auto_merge_min_similarity` (default 0.85).
**P4** Short-term abbreviation expansion via cooccurrence: for len≤2 terms, finds best
cooccurrence partner (substring match preferred, fallback to any long partner).
Threshold lowered to 0.05 because 2-char terms have dispersed cooccurrence.
**P5** Artifact filter: regex patterns for LaTeX noise, pure numbers, single chars.
**P6** Cross-cluster score penalty: `score /= cross_cluster_count` when term appears
in ≥ min_count clusters. Applied pre-ranking in Stage 4.
**ALTERNATIVES**: Topic-model-based stopword detection (too heavy). TF-IDF Z-score
filtering (already partly handled by c-TF-IDF). Chose lightweight config-driven approach.

## [2026-03-12] [Quality] DECISION: British/American spelling variant normalization
**REASON**: "disk"/"disc" occupied 4 slots in cluster 26 (disk, disc, disks, discs).
Generic same-length edit-distance-1 rule was too aggressive (incorrectly merged
"model"/"modal"). Replaced with explicit dictionary of ~35 British→American mappings
applied per-word in Stage 5. Combined with plural merge: disc→disk, discs→disks→disk.
**ALTERNATIVES**: (a) Heuristic same-length-dist-1 rule — rejected, false positives.
(b) LLM-based canonicalization — overkill for known spelling patterns.

## [2026-03-12] [Quality] DECISION: max_group_size limit for term network
**REASON**: Connected components in term network grew unbounded via sub-phrase chains
("point cloud" → "point cloud data" → "point cloud classification" → 12 members).
Added max_group_size (default 5) that splits oversized groups by iteratively removing
weakest edges. Reduces candidate noise for downstream LLM canonicalization.
**ALTERNATIVES**: (a) Max-depth BFS — more complex, similar result.
(b) Minimum spanning tree pruning — overkill for this use case.

## [2026-04-09] [Clustering] DECISION: Pass node_sizes through hierarchy_builder contraction
**REASON**: `hierarchy_builder.build_level()` contracts the graph after each level but did
not pass `node_sizes` to the next level's `runner.run()`. CPM's resolution term is
`γ × Σ (n_c choose 2)` where `n_c` must reflect the original number of nodes each
super-node represents. Without `node_sizes`, each super-node counts as 1, breaking γ
preservation across levels. `block_init.cascade_search()` already handled this correctly.
**FIX**: After contraction, compute per-super-node original node counts and store in
`self._node_sizes`. Pass to `runner.run(node_sizes=self._node_sizes)`. Multi-level
contraction accumulates sizes correctly.
**CWTS REFERENCE**: CWTS `publicationclassification` (Java) also uses contraction-based
hierarchy. CPM monotonicity (γ₀ > γ₁ ⟹ clusters at γ₀ are subsets of clusters at γ₁)
theoretically justifies this approach, though Leiden's stochastic nature means the
guarantee holds only at the global optimum.
**ALTERNATIVES**: (a) Independent Leiden at each level on full graph + post-hoc nesting
— preserves refinement but O(n) per level on large graphs.
(b) Block init with extreme γ_block — reduces contraction error but diminishes speed gain.

