# Changelog

## [Unreleased]

### Added
- Local release gate script (`scripts/release_check.sh`) covering whitespace checks,
  Rust tests, PyO3 rebuilds, Python tests, and CLI import/help smoke.
- Release readiness checklist (`docs/developer/release_readiness.md`) documenting supported
  toolkit surface, internal research surface, fresh-clone smoke checks, and
  artifact policy.
- Public workflow and module map docs (`docs/user/workflows.md`, `docs/user/modules.md`)
  for the full-cycle SciSci analysis/visualization package surface.
- Live OpenAlex example presets for perovskite solar cells and graph neural
  networks (`examples/openalex_live_demo.py`).
- Domain-agnostic keyword quality refinement with audit flags, display labels,
  and optional quality-score reranking for landscape/`--enable-all` outputs.
- **Hierarchical clustering** (`sciscape.clustering.hierarchical`): 4-level nano/micro/meso/macro hierarchy with auto-gamma per level, graph contraction + 1/rank re-normalization
- **Automatic gamma selection** (`sciscape.clustering.auto_gamma`): density-aware range estimation, parallel coarse scan (ThreadPool), binary refinement, skip-postprocess optimization
- **Adaptive top-k** (`sciscape.linkage.filters.compute_adaptive_k`): sqrt(n)-based dynamic top-k replacing fixed k=30
- **Rust graph utilities** (`rust/src/graph_utils.rs`): filter_top_k, find_gcc (Union-Find), contract_edges with PyO3 bindings
- **In-memory integer remap** (`integer_remap_memory`): 4x faster than disk-based version
- **Consensus visualization** (`sciscape.visualization.consensus`): multi-layer edge consensus stats, overlap matrix, backbone identification
- **Edge landscape visualization** (`sciscape.visualization.edge_landscape`): year x year heatmaps, multilayer comparison
- **OpenAlex pipeline** (`sciscape.openalex`): query -> fetch -> DC/BC/CC edges -> landscape
- **Web interface** (`sciscape.web`): FastAPI + D3.js + Plotly with 19 API endpoints
- **Evaluation framework** (`sciscape.evaluation`): LLM blind review, stability profiling
- **Temporal tracking** (`sciscape.visualization.temporal_tracking`): rolling window cluster evolution

### Changed
- README and package README now frame the supported SciScape toolkit surface
  as a full-cycle SciSci analysis/visualization package across ingestion,
  network construction, clustering, keyword extraction, visualization/reporting,
  and evaluation while separating internal Dongdaemun/adaptive-refinement
  research surfaces.
- Makefile development targets default to `uv` so they work in checkouts where
  only `python3` is available and the unversioned `python`/`pip` commands are
  absent.
- New research outputs under `research/**/results/**` are ignored by default;
  curated evidence remains tracked and new result artifacts require explicit
  retention review before force-adding.
- `combine_edge_layers` consensus strategy: single group_by (2.1x faster)
- DC/BC/CC edge construction: array-list (8.3x/3.3x faster) replacing dict-list
- Edge landscape: numpy vectorized (np.add.at) replacing iter_rows
- Cluster edge aggregation: Polars join+group_by replacing iter_rows
- Bridge paper detection: vectorized Polars filtering
- Temporal snapshots: pre-joined edge table (3.1x faster)
- Rust threshold lowered: filter_top_k >200, GCC >100 (was >1000/>500)
- Float32 -> Float64 in integer_remap for precision

### Fixed
- Auto-gamma binary refinement ratio check (was silently skipping refinement)
- BC_count sparse matrix moved outside per-node loop (was O(n^2) inside)
- Empty Counter guards preventing NameError/ValueError crashes
- Cached level membership reconstruction (scatter mapping was inverted)
- Self-citation exclusion in DC edge building
- uint64 -> int64 safe cast in contract_edges and sweep_gamma
- File handle leaks (context managers for JSON I/O)
- Tempdir resource leak (shutil.rmtree in finally, then replaced with in-memory)
- Web app: unsafe [0] indexing on empty cluster column lists
- Web app: LLM labeling error handler returning dict instead of setting job state

## [0.1.0] - 2026-02-13

### Added
- Initial release: Leiden clustering, keyword extraction, landscape pipeline
- CLI with cluster/keywords/convert/landscape commands
- igraph + leidenalg backend
- Polars-based edge processing
