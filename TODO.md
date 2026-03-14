# SciScape TODO (2026-03-14)

## Snapshot
- Python package `sciscape` (Leiden clustering + 9-stage keyword extraction) with `sos` shim for legacy imports.
- Input adapters: WoS, Scopus, OpenAlex (`sciscape.adapters`)
- CLI: `sciscape cluster | keywords | convert`
- Visualization: dashboard, network map, hierarchy treemap/sunburst, temporal comparison
- Tests: 420+ passing (`pytest -q`)

## Done (since last update)
- ~~Document stable I/O schema contracts~~ → `docs/io_schema.md`
- ~~Add minimal CLI~~ → `sciscape/cli.py` with cluster, keywords, convert subcommands
- ~~Fix root README~~ → Updated with architecture diagram, CLI docs, adapter docs
- ~~Fix sciscape/tests imports~~ → All 9 tests pass with `sciscape.*` paths
- ~~Network map visualization~~ → `_network_map.py` (MDS/spring layout)
- ~~Hierarchy visualization~~ → `_hierarchy.py` (treemap/sunburst)
- ~~Temporal comparison~~ → `_temporal.py` (heatmap/trend lines)
- ~~Input adapters~~ → `sciscape/adapters/` (WoS, Scopus, OpenAlex, BibTeX)
- ~~.env.example~~ → LLM + DB config documented
- ~~Normalize LLM config~~ → `SCISCAPE_LLM_*` unified env vars with legacy fallback
- ~~Optimize graph build~~ → vectorized Polars join, min_weight filtering
- ~~Parquet/CSV edge input~~ → `load_edge_table` supports .parquet, .csv, .tsv, .zip
- ~~CI workflow~~ → `.github/workflows/ci.yml` (pytest + ruff, Python 3.10-3.12)
- ~~Report export~~ → `export_report()` multi-page HTML (dashboard + charts)

## Remaining
- Keyword extraction at OpenAlex scale: `membership_map()` RAM optimization, end-to-end streaming.
- Changelog/release policy, PyPI publish.
- "oil immersed" orphan edge case (no 3-gram in vocabulary, no bridging overlap match).
- Multi-level hierarchy visualization (nano → micro → meso) using membership across levels.

## Local Map (Integration)
- Provide an adapter layer that produces the clustering edge list from OpenAlex-derived tables.
- Generate `uid1`, `uid2`, `rel_sum2` from DC/BC/CC pair construction outputs.
- Emit artifacts in the exact format `run_pipeline` expects, or extend `run_pipeline` to accept a Parquet edge table directly.
