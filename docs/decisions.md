# Decision Log — Keyword Pipeline Redesign

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

## [2026-03-11] [Phase A.5+A.7] DECISION: Combine pipeline.py creation and keyword_extraction.py thinning
**REASON**: After extracting utilities (A.2), llm_canonicalize (A.3), temporal (A.4),
keyword_extraction.py contained ONLY the Pipeline class. No intermediate step needed.
keyword_extraction.py becomes 13-line re-export shim for backward compatibility.
**RESULT**: Original 2,918-line monolith → 7 focused modules:
  config.py (160), extraction.py (326), pipeline.py (597),
  llm_canonicalize.py (1712), temporal.py (218),
  diagnostics.py (397), keyword_extraction.py (13 shim)

