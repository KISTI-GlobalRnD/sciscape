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
- ~~Input adapters~~ → `sciscape/adapters/` (WoS, Scopus, OpenAlex)

## P1 (Operational Usability)
- Add `.env.example` for LLM config (cluster naming): `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`.
- Add `.env.example` for DB fetcher config (core documents): `CLUSTER_DB_*` settings and table/column mappings.
- Normalize LLM configuration across modules (cluster naming uses `OLLAMA_*`, Stage 7 uses `OPENAI_API_KEY`/`alias_*`).
- Add `sciscape convert` support for BibTeX (.bib) input.

## P2 (Performance / Scale)
- Clustering graph build scalability.
  - Avoid `to_list()` + Python loops in `build_graph` for large edge tables (vectorize with numpy/arrow where possible).
  - Add optional edge filtering (min weight threshold, top-k per node, or sampling) before graph build.
  - Accept Parquet edges (Arrow/Polars) in addition to zipped TSV to remove zip+CSV overhead.
- Keyword extraction at OpenAlex scale.
  - `membership_map()` currently loads the full mapping into RAM; add an alternative path for very large corpora.
  - Keep row-group streaming end-to-end, or clearly document memory expectations.

## P3 (Packaging / Hygiene)
- Add CI workflow (at least `pytest`) via GitHub Actions.
- Add a changelog/release policy.
- Publish to PyPI (or internal registry).
- Add `[project.optional-dependencies] viz = ["plotly"]` to pyproject.toml.

## P4 (Quality / Completeness)
- "oil immersed" orphan edge case (no 3-gram in vocabulary, no bridging overlap match).
- Multi-level hierarchy visualization (nano → micro → meso) using membership data across levels.
- Dashboard integration: embed network map and hierarchy charts into the HTML dashboard.

## Local Map (Integration)
- Provide an adapter layer that produces the clustering edge list from OpenAlex-derived tables.
- Generate `uid1`, `uid2`, `rel_sum2` from DC/BC/CC pair construction outputs.
- Emit artifacts in the exact format `run_pipeline` expects, or extend `run_pipeline` to accept a Parquet edge table directly.
