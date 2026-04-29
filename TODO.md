# SciScape TODO (2026-04-29)

## Snapshot
- Python package `sciscape` (Leiden clustering + 10-stage keyword extraction) with `sos` shim for legacy imports.
- **Rust backends**: `rust/` (Leiden clustering), `rust-text/` (keyword extraction hot paths)
- **Rust Leiden status**: memory-optimized remap/postprocess paths, convergence guard, profiling logs, Python handle API, Java cross-check tests
- Input adapters: WoS, Scopus, OpenAlex, BibTeX (`sciscape.adapters`)
- **Multi-layer combination**: boosted consensus (top-30 → 1/rank → ×n_layers → GCC → auto-γ)
- **OpenAlex integration**: `sciscape query` (CLI) + `sciscape web` (FastAPI)
- Landscape pipeline: `sciscape.landscape.run_landscape()` (multi-layer edges → clustering → keywords → report)
- Visualization: D3.js network (cluster + term co-occurrence), dashboard, hierarchy
- CPM dendrogram research path: `cpm-dendro/`, `sciscape.clustering.dendrogram`, `constrained_cut`, and `research/dendrogram/`
- Consensus validation research path: `research/consensus/` with corrected local-review, taxonomy, regime, and coverage-aware boundary protocols
- CLI: `sciscape cluster | keywords | convert | landscape | query | web`
- Tests: 800+ Python tests plus Rust crate tests (`pytest -q`, `cargo test --manifest-path rust/Cargo.toml`)
- Install: `pip install ./rust ./rust-text .` or `make install`

## Current Priorities

### Core Module Polish

- "oil immersed" orphan edge case (no 3-gram in vocabulary, no bridging overlap match).
- Multi-level hierarchy visualization (nano → micro → meso) using membership across levels.
- Changelog/release policy, PyPI publish.
- Top-k and field-generalization defaults: current experiments cover `field_12`, `field_15`, `field_18`, `field_26`, `field_30`, with `field_29/34` still useful for broader release confidence.
- Dendrogram path validation: implementation exists; remaining work is larger comparison runs, optional Paris/nested-SBM baselines, and release-facing documentation.

### Rust Leiden / Large-Scale Clustering

- Keep the standard Rust Leiden path close to CWTS/Java semantics; experimental SciSci-specific refinement stays separate.
- Use profiling logs from large bcrefresh contracted probes to guide any further optimization.
- Next experimental path: cluster-graph stats and macro-merge dry-run, documented in `research/consensus/TODO_SCISCI_ADAPTIVE_REFINEMENT.md`.
- Re-run large-scale validation after profiling/convergence changes on the target GPU datasets before making final speed/robustness claims.

### Consensus Methodology

- `Protocol D v2` coverage-aware boundary review is implemented and piloted for `cc-only vs citation_consensus`; see `research/consensus/TODO_BOUNDARY_ACCURACY_V2.md`.
- Fill the remaining D v2 comparison matrix:
  - `cc-only vs all_consensus`
  - `bc-only vs cc-only`
  - `emb-only vs cc-only` where embeddings are available
- Convert D v2 pilot outputs into manuscript tables/figures.
- Build `Protocol E` micro-gold local partitioning before making strong absolute boundary-accuracy claims.

## Recently Done (2026-04-16~29)

- Rust Leiden memory/performance pass: streaming remap CSR fill, workspace reuse, postprocess workspace reuse, local-merge compaction, convergence guard, and profiling observability.
- GPU bcrefresh contracted graph probes with doc-count node weights and high-water memory tracking.
- Java/Rust cross-check path for Leiden behavior and warm-start/debug comparisons.
- Protocol D v2 coverage-aware boundary sampler/reviewer/scorer.
- Corrected order-balanced `gemini_v3` local-review reference set and taxonomy/regime-support artifacts.
- CPM dendrogram crate/wrapper/cut scripts are implemented enough for research comparison runs.

## Done (2026-04-10~15)
- Rust Leiden backend (faster than Java, PyO3 bindings, postprocess with cascade+Dijkstra)
- Rust text backend (edit distance, similarity layers, cooccurrence, vocab merge)
- Multi-layer edge combination (boosted consensus, validated on 2 fields)
- Auto-gamma selection (binary search, target max cluster < 3%)
- OpenAlex API integration (query → fetch → DC+BC+CC edges → landscape)
- Web interface (FastAPI + D3.js network + SSE progress)
- Network visualization (layer toggle, hierarchy, overlay, density, temporal, bridge, search)
- Cluster auto-labeling (keyword-based + LLM)
- Component-level Dijkstra postprocess (multi-hop small cluster assignment)
- 55-case LLM evaluation framework (belonging, cohesion, outliers)
- LaTeX report: Boosted Multi-Layer Edge Combination

## Done (2026-03-31)
- Dashboard: Global views for all tabs (Keywords, Temporal, Hierarchy, Network, Dictionary)
- Dashboard: "All Clusters" mode with cluster/keyword network subtabs
- Dashboard: Sciscape logo, header/footer redesign
- Bug fixes: dead code, edge vs vertex attribute, infinite loop guard, term_network safety
- Input validation: config min_df/max_df, CLI parsing, column alias warning, node_sizes guard
- Config: gamma_block_margin, gamma_log_step, bayesian_alpha, bayesian_prior extracted from hardcoded values
- Tests: +105 new tests (CLI 52, temporal 16, clustering graph/pipeline/partitioning 37)
