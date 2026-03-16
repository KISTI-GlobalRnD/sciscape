# SciScape TODO (2026-03-14)

## Snapshot
- Python package `sciscape` (Leiden clustering + 10-stage keyword extraction) with `sos` shim for legacy imports.
- Input adapters: WoS, Scopus, OpenAlex, BibTeX (`sciscape.adapters`)
- Landscape pipeline: `sciscape.landscape.run_landscape()` (edges → hierarchical clustering → keywords → report)
- CLI: `sciscape cluster | keywords | convert | landscape`
- Visualization: dashboard, network map, hierarchy treemap/sunburst, temporal comparison
- Tests: 418+ passing (`pytest -q`)

## Remaining
- Contracted graph resolution 자동 스케일링: contraction 후 γ 값 보정 (현재 수동 지정)
- Keyword extraction at OpenAlex scale: `membership_map()` RAM optimization, end-to-end streaming.
- Changelog/release policy, PyPI publish.
- "oil immersed" orphan edge case (no 3-gram in vocabulary, no bridging overlap match).
- Multi-level hierarchy visualization (nano → micro → meso) using membership across levels.

## Local Map (Integration)
- Provide an adapter layer that produces the clustering edge list from OpenAlex-derived tables.
- Generate `uid1`, `uid2`, `rel_sum2` from DC/BC/CC pair construction outputs.
- Emit artifacts in the exact format `run_pipeline` expects, or extend `run_pipeline` to accept a Parquet edge table directly.
