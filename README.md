# SciScape

SciScape is a full-cycle SciSci analysis and visualization package for research
paper networks: data ingestion, multi-layer clustering, keyword extraction,
network visualization, reporting, and evaluation.

## Full-Cycle Modules

- **Data ingestion**: WoS, Scopus, OpenAlex, and BibTeX conversion into SciScape tables
- **Network construction**: DC, BC, CC, and embedding edge builders for paper networks
- **Multi-layer consensus**: Combine layers with top-k filtering, 1/rank normalization, and consensus weighting (edges confirmed by multiple layers get multiplicative boost)
- **Hierarchical clustering core**: nano → micro → meso → macro with auto-gamma per level
- **Rust CPM/Leiden backend**: High-performance Leiden clustering, contraction, projection, and small-cluster postprocess for the main pipeline
- **10-stage keyword extraction**: TF-IDF, quality reranking/display labels, cooccurrence, term network, temporal/depth signals, and optional LLM canonicalization
- **Network visualization and reports**: FastAPI + D3.js network visualization, Plotly treemap/sunburst, temporal charts, and standalone viewer export
- **Evaluation framework**: LLM blind review (3 criteria), stability analysis (AMI/ARI), and quality reports
- **SciSci research modules**: Optional diagnostics and hierarchy postprocess helpers for auditable development runs; Dongdaemun surfaces stay development-only
- **Rust acceleration**: Clustering + keyword extraction hot paths (50-200x speedup)
- **OpenAlex integration**: Query → fetch → build citation edges → landscape report

## Current Scope

The supported package surface is the end-to-end SciSci analysis and
visualization workflow: source-data conversion, OpenAlex ingestion, multi-layer
edge construction and combination, Rust CPM/Leiden clustering, hierarchical
landscape construction, keyword extraction, network visualization/report
generation, standalone viewer export, and evaluation utilities. Clustering is
the core engine, but the package is branded around the full analysis and
visualization lifecycle.

Dongdaemun is a development/research family name, not the default product
surface. Use the specific terms in `docs/dongdaemun_naming_contract.md`:
`Dongdaemun-post` for the manuscript-backed post-Leiden hierarchy repair
method, `Dongdaemun-refinement` for opt-in integrated refinement R&D, and
diagnostic-only names for basin/adaptive-refinement probes.

## Workflow Docs

- `docs/workflows.md`: supported end-to-end workflows from source data to
  landscapes, keywords, reports, and viewer artifacts.
- `docs/modules.md`: lifecycle module map across ingest, network, landscape,
  clustering, keywords, visualization, and evaluation.
- `docs/io_schema.md`: table-level input/output schemas.
- `docs/release_readiness.md`: local validation gate and artifact policy.

## Install

```bash
# Full install from a checkout (Rust backends + Python)
pip install ./rust ./rust-text .

# Or use Makefile
make install

# Development (editable)
make install-dev
```

**Requirements**: Python >=3.10, Rust toolchain (`rustup`), maturin.
Without Rust: `pip install .` (Python fallback, slower).

For a pre-push or local release gate, run:

```bash
./scripts/release_check.sh
```

## Quick Start

### 1. OpenAlex Query (all-in-one)

```bash
sciscape query "machine learning" --years 2020-2024 --email you@univ.edu -o results/
```

### 2. Landscape Pipeline

```bash
sciscape landscape abstracts.parquet \
    --layers bc=data/bc.parquet,cc=data/cc.parquet,dc=data/dc.parquet \
    --combine-strategy consensus \
    --auto-gamma \
    -o output/
```

This produces hierarchy membership, keyword tables, network/report artifacts,
and viewer-ready data under the output directory.

### 3. Web Interface

```bash
sciscape web
# Open http://localhost:8000
```

This launches the FastAPI-backed query workflow: paste an OpenAlex query,
run the analysis, stream progress, and inspect network, hierarchy, temporal,
keyword, quality, and download tabs in the browser. This is separate from
`sciscape viewer`, which is a static data viewer for existing `data.json` or
keyword CSV/TSV files.

### 4. Python API

```python
from sciscape.clustering.hierarchical import build_hierarchy

result = build_hierarchy(
    layer_paths={
        "bc": "data/bc_cosine.parquet",
        "cc": "data/cc_cosine.parquet",
        "dc": "data/dc_fractional.parquet",
        "emb": "data/emb_bg_knn30.parquet",
    },
    cache_dir="output/field_15",
    n_levels=4,
    progress=print,
)
# nano(~1450) -> micro(~414) -> meso(~152) -> macro(~45)

# Optional internal development mode: after normal small-cluster repair,
# run auditable oversize diagnostics before projecting/contracting the level.
from sciscape.clustering.hierarchy_oversize_postprocess import HierarchyPostprocessConfig

result = build_hierarchy(
    layer_paths={...},
    cache_dir="output/field_15",
    n_levels=4,
    hierarchy_postprocess=HierarchyPostprocessConfig(
        enabled=True,
        oversize_policy="quality_first",  # default; preserves CPM quality first
        # use_rust_dongdaemun=True,     # development-only fast path
        # write_artifacts=False,        # required for the Rust fast path today
    ),
)

# Label clusters
from sciscape.clustering.label_pipeline import label_hierarchy
labels = label_hierarchy(abstracts_df, result.to_dataframe(uids))
```

See `docs/workflows.md` for full CLI and Python examples covering OpenAlex,
local edge layers, existing membership, and clustering-only runs.

Curated live OpenAlex demos are available in `examples/openalex_live_demo.py`
for perovskite solar cells and graph neural networks.

## Pipeline

```
Each layer (BC, CC, DC, Emb)
  -> top-30 per node (noise removal)
  -> 1/rank normalization (scale unification)
  -> consensus weighting (sum_w_rank * n_layers)
  -> GCC filter
  -> auto-gamma (binary search, target max < 3%)
  -> Rust Leiden -> postprocess (cascade + greedy + Dijkstra)
  -> optional hierarchy oversize postprocess (split-repair + boundary trim)
  -> contraction -> repeat for next hierarchy level
```

## Architecture

```
sciscape/
  clustering/
    hierarchical.py        4-level hierarchy (nano->micro->meso->macro)
    hierarchy_oversize_postprocess.py
                            Internal opt-in oversize postprocess automation
    hierarchy_postprocess.py
                            Compatibility import path for older code
    auto_gamma.py          Automatic gamma selection (binary search)
    prepartition.py        Pre-partition (Lego block assembly)
    leiden_rust.py          Rust Leiden backend wrapper
    label_pipeline.py      TF-IDF labels + string_grouper dedup
    abbreviation_dict.py   Compatibility wrapper for conservative abbreviation evidence
    postprocess.py         Cascade gamma + greedy + component Dijkstra
    runner.py              LeidenRunner + RustLeidenRunner
  keyword_extraction/       10-stage pipeline + keyword quality/abbreviation evidence
  linkage/
    combine.py             Multi-layer consensus combination
    filters.py             top-k, GCC, weight normalization
    builders.py            DC, BC, CC edge construction
  openalex/                 OpenAlex API client + edge builder
  evaluation/               LLM reviewer + worst-case sampler
  visualization/
    consensus.py            Consensus distribution analysis
    edge_landscape.py       Year x year heatmaps per layer
    hierarchy_treemap.py    Plotly treemap/sunburst
  web/                      FastAPI + D3.js + SSE progress
  landscape.py              End-to-end pipeline orchestration
  cli.py                    CLI entry point

rust/                       Rust Leiden backend (sciscape-leiden)
rust-text/                  Rust keyword extraction (sciscape-text)
experiments/                Research experiment scripts
docs/                       LaTeX report + figures
```

## CLI Commands

| Command | Description |
|---|---|
| `sciscape query` | OpenAlex -> fetch -> edges -> landscape (all-in-one) |
| `sciscape landscape` | Edges -> clustering -> keywords -> report |
| `sciscape cluster` | Leiden clustering only |
| `sciscape keywords` | Keyword extraction only |
| `sciscape visualize` | Turn a keyword table into dashboard/report HTML |
| `sciscape convert` | Convert WoS/Scopus/OpenAlex/BibTeX -> parquet |
| `sciscape viewer` | Generate a static viewer shell for existing data files |
| `sciscape export` | Export network files for Gephi/Cytoscape |
| `sciscape web` | Launch query-to-analysis FastAPI web interface |

## Research Status

The active research map is maintained in `research/PROJECT_TRACKS.md`.
Use that file, `research/FAILED_DIRECTIONS.md`, and
`research/DATA_RETENTION_PLAN.md` as the current source for claim boundaries,
negative controls, and artifact retention.

Historical reports remain available, but should not be treated as the current
claim source without checking the active research notes:

- `docs/multilayer_combination_report.pdf`
- `docs/hierarchy_two_stage_postprocess_report.tex`
- `docs/dongdaemun_naming_contract.md`
- `docs/leiden_multibasin_research_guardrails.md`

## Tests

```bash
# Use the repo-local uv environment; system python may not have pytest/pip.
uv run --extra dev python -m pytest -q

# After changing Rust PyO3 bindings, rebuild the editable native extension.
uv run --extra dev maturin develop --manifest-path rust/Cargo.toml
uv run --extra dev python -m pytest -q

# Full local pre-push/release gate.
./scripts/release_check.sh
```

See `docs/release_readiness.md` for the release checklist and research artifact
policy.

## License

LicenseRef-KRISS-Internal
