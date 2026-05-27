# SciScape Module Map

Status: public module orientation
Date: 2026-05-27

SciScape is branded as a full-cycle SciSci analysis and visualization package.
The modules below are user-facing functional areas, not necessarily one-to-one
with package directories.

## Lifecycle Summary

| Stage | User-facing purpose | CLI surface | Primary Python modules | Main outputs |
| --- | --- | --- | --- | --- |
| Ingest | Convert or fetch research-paper records. | `convert`, `query` | `sciscape.adapters`, `sciscape.openalex` | `abstracts.parquet`, raw/fetched records |
| Network | Build and combine paper-network edge layers. | `query`, `landscape --layers` | `sciscape.linkage`, `sciscape.openalex.edges` | DC/BC/CC/embedding edge tables, `combined_edges.parquet` |
| Landscape | Build the hierarchy over the fused paper network. | `landscape` | `sciscape.landscape`, `sciscape.clustering` | `membership.parquet`, level caches, hierarchy metadata |
| Clustering Engine | Run CPM/Leiden and postprocess lower-level partitions. | `cluster` | `sciscape.clustering`, `sciscape_leiden` | membership and description tables |
| Keywords | Extract, normalize, and score cluster-level terms. | `keywords`, `landscape` | `sciscape.keyword_extraction` | `keywords.parquet` |
| Visualization | Produce network, hierarchy, temporal, and viewer artifacts. | `web`, `viewer`, `export` | `sciscape.visualization`, `sciscape.keyword_extraction.visualization`, `sciscape.web`, `sciscape.export` | `report/`, HTML views, GEXF/GraphML |
| Evaluation | Check stability, quality, and review evidence. | `landscape --evaluate` | `sciscape.evaluation`, `sciscape.visualization.consensus` | stability summaries, quality reports, review outputs |

## Ingest

Purpose: turn external bibliometric formats into the standard SciScape abstract
table.

Supported sources:

- Web of Science exports.
- Scopus CSV exports.
- OpenAlex JSON/CSV/API records.
- BibTeX files.

CLI:

```bash
sciscape convert wos savedrecs.txt -o abstracts.parquet
sciscape convert scopus scopus_export.csv -o abstracts.parquet
sciscape convert openalex works.jsonl -o abstracts.parquet
sciscape convert bibtex references.bib -o abstracts.parquet
```

Python:

```python
from sciscape.adapters import read_bibtex, read_openalex, read_scopus, read_wos
```

## Network

Purpose: build paper-network edge layers and combine them into a graph suitable
for clustering.

Core edge families:

- DC: direct citation.
- BC: bibliographic coupling.
- CC: co-citation.
- Embedding or semantic KNN edges when available.

Python:

```python
from sciscape.linkage import build_bc, build_cc, build_dc, combine_edge_layers
```

Typical outputs:

- layer edge tables.
- filtered/normalized edge tables.
- `combined_edges.parquet`.

## Landscape

Purpose: run the end-to-end landscape workflow from prepared abstracts and edge
tables to hierarchy, keywords, and reports.

CLI:

```bash
sciscape landscape abstracts.parquet \
  --layers bc=bc.parquet,cc=cc.parquet,dc=dc.parquet \
  --combine-strategy consensus \
  --auto-gamma \
  -o output
```

Python:

```python
from sciscape.landscape import LandscapeConfig, run_landscape
```

Outputs:

- `membership.parquet`.
- `abstracts_subset.parquet`.
- `keywords.parquet`.
- `report/`.
- hierarchy cache directories.

## Clustering Engine

Purpose: provide the lower-level CPM/Leiden clustering and hierarchy machinery
used by the landscape workflow.

CLI:

```bash
sciscape cluster edges.zip edges.txt --levels 5,100 80,500 -o membership.parquet
```

Python:

```python
from sciscape.clustering import LeidenConfig, run_pipeline
from sciscape.clustering.hierarchical import build_hierarchy
```

Public defaults focus on Rust CPM/Leiden, projection, contraction, and
small-cluster postprocess. Dongdaemun-related integrated refinement remains
development/research-only unless explicitly enabled and documented.

## Keywords

Purpose: turn membership and abstracts into interpretable cluster-level terms.

CLI:

```bash
sciscape keywords abstracts.parquet membership.parquet \
  --include-title \
  --enable-all \
  -o keywords.parquet
```

Python:

```python
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
```

Capabilities:

- c-TF-IDF and frequency scoring.
- vocabulary cleanup and synonym-style merge support.
- cooccurrence and term-network signals.
- temporal and depth features.
- optional LLM canonicalization.

## Visualization And Export

Purpose: make landscape outputs inspectable by people and by downstream graph
tools.

CLI:

```bash
sciscape web
sciscape viewer -o viewer.html
sciscape export edges.parquet membership.parquet --format gexf -o network.gexf
```

Python:

```python
from sciscape.keyword_extraction.visualization import export_dashboard, export_report, export_viewer
from sciscape.visualization import compute_consensus_stats, save_treemap_html
```

Outputs:

- report HTML files.
- dashboard data.
- standalone viewer HTML.
- GEXF or GraphML exports.

## Evaluation

Purpose: inspect stability, quality, local review evidence, and edge/cluster
diagnostics.

Primary surfaces:

- `sciscape landscape --evaluate` for stability and quality summaries.
- `sciscape.evaluation` for review and sampler helpers.
- `sciscape.visualization.consensus` and `sciscape.linkage.diagnostics` for
  edge/consensus diagnostics.

Evaluation outputs should report both quality and cost where relevant. For
Dongdaemun or adaptive-refinement claims, follow
`docs/leiden_multibasin_research_guardrails.md`.

## Research And Development Surfaces

The following are useful for internal experiments but should not be presented
as default product surfaces without supporting evidence:

- `Dongdaemun-refinement` integrated refinement paths.
- basin-transition and adaptive-refinement diagnostics.
- policy or threshold sweeps that do not answer a mechanism question.

Use `docs/dongdaemun_naming_contract.md`,
`research/PROJECT_TRACKS.md`, and `research/FAILED_DIRECTIONS.md` before
promoting a research helper into a supported workflow.
