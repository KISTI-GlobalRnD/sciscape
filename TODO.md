# SciScape TODO (2026-04-15)

## Snapshot
- Python package `sciscape` (Leiden clustering + 10-stage keyword extraction) with `sos` shim for legacy imports.
- **Rust backends**: `rust/` (Leiden clustering), `rust-text/` (keyword extraction hot paths)
- Input adapters: WoS, Scopus, OpenAlex, BibTeX (`sciscape.adapters`)
- **Multi-layer combination**: boosted consensus (top-30 → 1/rank → ×n_layers → GCC → auto-γ)
- **OpenAlex integration**: `sciscape query` (CLI) + `sciscape web` (FastAPI)
- Landscape pipeline: `sciscape.landscape.run_landscape()` (multi-layer edges → clustering → keywords → report)
- Visualization: D3.js network (cluster + term co-occurrence), dashboard, hierarchy
- CLI: `sciscape cluster | keywords | convert | landscape | query | web`
- Tests: 739+ passing (`pytest -q`)
- Install: `pip install ./rust ./rust-text .` or `make install`

## Remaining
- "oil immersed" orphan edge case (no 3-gram in vocabulary, no bridging overlap match).
- Multi-level hierarchy visualization (nano → micro → meso) using membership across levels.
- Changelog/release policy, PyPI publish.
- Optimal top-k value (20? 30? 50?) — more field experiments needed.
- More fields generalization test (field 18/26/29/30/34).

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
