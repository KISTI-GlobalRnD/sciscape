# SciScape

Scientific landscape analysis toolkit: multi-layer consensus clustering + hierarchical keyword extraction for research paper networks.

## Features

- **Multi-layer consensus**: Combine DC, BC, CC, and embedding edges with top-k filtering, 1/rank normalization, and consensus weighting (edges confirmed by multiple layers get multiplicative boost)
- **Hierarchical clustering**: nano → micro → meso → macro with auto-gamma per level
- **Rust acceleration**: Leiden clustering + keyword extraction hot paths (50-200x speedup)
- **OpenAlex integration**: Query → fetch → build citation edges → landscape report
- **Web interface**: FastAPI + D3.js network visualization + Plotly treemap
- **10-stage keyword extraction**: TF-IDF, cooccurrence, term network, LLM canonicalization
- **Evaluation framework**: LLM blind review (3 criteria), stability analysis (AMI/ARI)

## Install

```bash
# Full install (Rust backends + Python)
pip install ./rust ./rust-text .

# Or use Makefile
make install

# Development (editable)
make install-dev
```

**Requirements**: Python >=3.10, Rust toolchain (`rustup`), maturin.
Without Rust: `pip install .` (Python fallback, slower).

## Quick Start

### 1. OpenAlex Query (all-in-one)

```bash
sciscape query "machine learning" --years 2020-2024 --email you@univ.edu -o results/
```

### 2. Multi-layer Clustering

```bash
sciscape landscape abstracts.parquet \
    --layers bc=data/bc.parquet,cc=data/cc.parquet,dc=data/dc.parquet \
    --combine-strategy consensus \
    --auto-gamma \
    -o output/
```

### 3. Web Interface

```bash
sciscape web
# Open http://localhost:8000
```

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

# Label clusters
from sciscape.clustering.label_pipeline import label_hierarchy
labels = label_hierarchy(abstracts_df, result.to_dataframe(uids))
```

## Pipeline

```
Each layer (BC, CC, DC, Emb)
  -> top-30 per node (noise removal)
  -> 1/rank normalization (scale unification)
  -> consensus weighting (sum_w_rank * n_layers)
  -> GCC filter
  -> auto-gamma (binary search, target max < 3%)
  -> Rust Leiden -> postprocess (cascade + greedy + Dijkstra)
  -> contraction -> repeat for next hierarchy level
```

## Architecture

```
sciscape/
  clustering/
    hierarchical.py        4-level hierarchy (nano->micro->meso->macro)
    auto_gamma.py          Automatic gamma selection (binary search)
    prepartition.py        Pre-partition (Lego block assembly)
    leiden_rust.py          Rust Leiden backend wrapper
    label_pipeline.py      TF-IDF labels + string_grouper dedup
    abbreviation_dict.py   Auto abbreviation extraction from abstracts
    postprocess.py         Cascade gamma + greedy + component Dijkstra
    runner.py              LeidenRunner + RustLeidenRunner
  keyword_extraction/       10-stage pipeline
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
| `sciscape convert` | Convert WoS/Scopus/OpenAlex/BibTeX -> parquet |
| `sciscape web` | Launch web interface |

## Key Results

Validated on 2 OpenAlex fields (Chemical Engineering 115K, Arts and Humanities 754K):

- **Consensus > Sum**: Boundary node assignment (29:26 in 55 LLM blind reviews)
- **4-layer > 3-layer**: +Emb reduces clusters by 62%, doubles avg size (field_12)
- **Auto-gamma essential**: gamma shifts 24-32x between 3L/4L
- **Stability**: AMI = 0.902 +/- 0.009 (5 seeds)
- **Consensus backbone**: 6.5% of 3-layer edges (3x boost) form cluster cores

See `docs/multilayer_combination_report.pdf` for the full 15-page analysis.

## Tests

```bash
pytest -q   # 818+ tests
```

## License

LicenseRef-KRISS-Internal
