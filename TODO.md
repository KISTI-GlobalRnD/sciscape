# SciScape TODO (2026-02-27)

## Snapshot
- Repo provides Python package `sciscape` (Leiden clustering + keyword extraction) with `sos` shim for legacy imports.
- Legacy `leiden_module/` directory has been removed (superseded by `sciscape/`).
- Clustering input contract today: zipped TSV inside a zip (expects columns `uid1`, `uid2`, `rel_sum2`) via `sciscape.clustering.load_edge_table`.
- Keyword extraction input contract today: Parquet abstracts + Parquet membership mapping via `sciscape.keyword_extraction`.
- Default tests (`pytest -q`) run only `tests/` (2 tests pass). `sciscape/tests` currently fails collection due to outdated imports (`clustering.*`, `keyword_extraction.*`).
- No CI/workflows are configured in this repo.
- Root `README.md` references `final_report_site/...` paths that are not present in this repo.

## P0 (Blockers / Correctness)
- Fix `sciscape/tests/*` imports to use `sciscape.clustering` / `sciscape.keyword_extraction` (or delete/migrate these tests if obsolete).
- Update `pytest.ini` so default `pytest` covers the intended test set (at least `tests/` + fixed `sciscape/tests/`).
- ~~Decide source of truth~~ **DONE**: `sciscape/` is canonical; `leiden_module/` removed.
- Document stable I/O schema contracts (inputs + outputs) in one place.
- Clustering contract: required columns, allowed dtypes, edge weight meaning, membership output schema (`uid`, `cluster_*`, indices).
- Keyword contract: required columns (`uid`, `abstract`, `pubyear`, optional `title`), membership schema (`uid`, cluster id column), output columns and types.
- Fix root `README.md` to avoid pointing at non-existent paths (either remove the KRISS report section or re-home those docs into this repo).

## P1 (Operational Usability)
- Add a minimal CLI (console scripts) for the common flows.
- Run clustering from a zip edge list and emit membership/description parquet.
- Run keyword extraction from parquet inputs and emit keyword parquet.
- Run Stage 2.5 canonicalisation from a saved Stage 2 snapshot.
- Add `.env.example` for LLM config (cluster naming): `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`.
- Add `.env.example` for DB fetcher config (core documents): `CLUSTER_DB_*` settings and table/column mappings.
- Normalize LLM configuration across modules (cluster naming uses `OLLAMA_*`, Stage 2.5 uses `OPENAI_API_KEY`/`alias_*`).

## P2 (Performance / Scale)
- Clustering graph build scalability.
- Avoid `to_list()` + Python loops in `build_graph` for large edge tables (vectorize with numpy/arrow where possible).
- Add optional edge filtering (min weight threshold, top-k per node, or sampling) before graph build.
- Accept Parquet edges (Arrow/Polars) in addition to zipped TSV to remove zip+CSV overhead.
- Keyword extraction at OpenAlex scale.
- `membership_map()` currently loads the full mapping into RAM; add an alternative path (polars join, sqlite temp table, or chunked uid mapping) for very large corpora.
- Keep row-group streaming end-to-end (avoid converting to pandas when possible), or clearly document memory expectations and required filtering.

## P3 (Packaging / Hygiene)
- Fix `MANIFEST.in` referencing missing `AGENTS.md` (either add the file or remove the include line).
- Add CI workflow (at least `pytest`) and pin minimal supported Python versions (already `>=3.10` in `pyproject.toml`).
- Add a changelog/release policy so downstream projects can pin versions confidently.

## Local Map (Integration)
- Provide an adapter layer (script or module) that produces the clustering edge list from OpenAlex-derived tables.
- Generate `uid1`, `uid2`, `rel_sum2` from DC/BC/CC pair construction outputs.
- Emit artifacts in the exact format `run_pipeline` expects (zip + inner TSV), or extend `run_pipeline` to accept a Parquet edge table directly.
