# SciScape Full-Cycle Workflows

Status: public workflow orientation
Date: 2026-05-27

SciScape is organized around full-cycle SciSci analysis and visualization of
research paper networks. The workflows below describe the supported public
paths from input data to landscape, keywords, reports, and viewer artifacts.

For table-level schemas, see `docs/user/io_schema.md`. For module boundaries, see
`docs/user/modules.md`.

## Workflow 1: OpenAlex To Landscape Report

Use this when the starting point is a topic query and SciScape should fetch
records, build citation-derived edges, cluster the network, extract keywords,
and produce a report.

### Inputs

- Search string such as `"machine learning"`.
- Optional publication-year filter.
- Optional OpenAlex polite-pool email.

### CLI

```bash
sciscape query "machine learning" \
  --years 2020-2024 \
  --email you@example.org \
  --max-works 5000 \
  -o workspace/openalex_output/machine_learning
```

### Python API

```python
from pathlib import Path
from sciscape.openalex import OpenAlexPipelineConfig, run_openalex_pipeline

cfg = OpenAlexPipelineConfig(
    query="machine learning",
    filters={"publication_year": "2020-2024"},
    max_works=5000,
    email="you@example.org",
    edge_types=["dc", "bc"],
    output_dir=Path("workspace/openalex_output/machine_learning"),
    run_landscape=True,
    progress=print,
)
result = run_openalex_pipeline(cfg)
```

### Main Outputs

- `abstracts.parquet`: standard SciScape abstract table.
- edge tables for the requested link types.
- landscape output directory when `run_landscape=True`.
- `membership.parquet`, `keywords.parquet`, and `report/`.

### Modules Involved

- `sciscape.openalex`
- `sciscape.linkage`
- `sciscape.landscape`
- `sciscape.clustering`
- `sciscape.keyword_extraction`
- `sciscape.keyword_extraction.visualization`

### Curated Live Examples

Two live OpenAlex presets are available in `examples/openalex_live_demo.py`.
They are intended for a realistic demo dataset rather than CI:

```bash
uv run --extra dev python examples/openalex_live_demo.py \
  --preset perovskite \
  --email you@example.org

uv run --extra dev python examples/openalex_live_demo.py \
  --preset gnn \
  --email you@example.org
```

The `perovskite` preset uses the query `perovskite solar cells`. The `gnn`
preset uses `title_and_abstract.search=graph neural networks` to reduce generic
search noise.

### Web UI

The same OpenAlex-to-visualization path is available from the local web UI:

```bash
sciscape web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, paste a query, and click **Analyze**. The web
surface streams progress while OpenAlex fetch, edge construction, landscape
clustering, keyword extraction, and report generation run in the background.
When the job completes, the results tabs show interactive network, hierarchy,
temporal, cluster, and keyword views, and the Download tab links to the
generated report/dashboard HTML artifacts.

This is the query-analysis app. The standalone `sciscape viewer` artifact below
is intentionally static: it visualizes an existing `data.json` or keyword
CSV/TSV file and does not run OpenAlex or clustering itself.

## Workflow 2: Local Abstracts And Edges To Viewer

Use this when abstracts and one or more edge layers already exist locally.
This is the normal end-to-end workflow for a prepared field dataset.

### Inputs

- `abstracts.parquet` with at least `uid`, `title`, `abstract`, and `pubyear`.
- A single edge table, or layer-specific edge tables such as BC, CC, DC, and
  embedding KNN edges.

### CLI: Multi-Layer Landscape

```bash
sciscape landscape abstracts.parquet \
  --layers bc=workspace/data/bc.parquet,cc=workspace/data/cc.parquet,dc=workspace/data/dc.parquet,emb=workspace/data/emb.parquet \
  --combine-strategy consensus \
  --combine-top-k auto \
  --auto-gamma \
  -o workspace/output/landscape
```

### CLI: Single Edge Table

```bash
sciscape landscape abstracts.parquet workspace/data/edges.parquet \
  --gamma-range 1e-6,1e-3 \
  -o workspace/output/landscape
```

### Python API

```python
from pathlib import Path
from sciscape.landscape import LandscapeConfig, run_landscape

cfg = LandscapeConfig(
    layer_paths={
        "bc": Path("workspace/data/bc.parquet"),
        "cc": Path("workspace/data/cc.parquet"),
        "dc": Path("workspace/data/dc.parquet"),
        "emb": Path("workspace/data/emb.parquet"),
    },
    combine_strategy="consensus",
    combine_top_k="auto",
    auto_gamma=True,
    report_title="My SciScape Landscape",
)

result = run_landscape(
    Path("workspace/output/landscape/combined_edges.parquet"),
    Path("abstracts.parquet"),
    Path("workspace/output/landscape"),
    config=cfg,
)
```

### Main Outputs

- `combined_edges.parquet` when layer paths are provided.
- `membership.parquet`.
- level-specific hierarchy caches.
- `abstracts_subset.parquet`.
- `keywords.parquet`.
- `report/` with dashboard/report artifacts.

### Modules Involved

- `sciscape.linkage`
- `sciscape.landscape`
- `sciscape.clustering`
- `sciscape.keyword_extraction`
- `sciscape.keyword_extraction.visualization`
- `sciscape.web` when serving the report or web UI.

## Workflow 3: Existing Membership To Keywords And Visuals

Use this when clustering was already done and the next step is naming,
keyword extraction, temporal term summaries, or dashboard artifacts.

### Inputs

- `abstracts.parquet` with `uid`, `title`, `abstract`, and `pubyear`.
- `membership.parquet` with `uid` and at least one `cluster_*` column.

### CLI

```bash
sciscape keywords abstracts.parquet membership.parquet \
  --include-title \
  --enable-all \
  --top-n 100 \
  -o keywords.parquet
```

Generate dashboard/report artifacts from an existing keyword table:

```bash
sciscape visualize keywords.parquet -o report/
```

For a quick one-file sample preview from a minimal CSV:

```bash
sciscape visualize sample_keywords.csv \
  --dashboard-only \
  -o sample_dashboard.html
```

Generate a blank standalone viewer shell:

```bash
sciscape viewer -o viewer.html
```

For GitHub Pages, put the viewer and data file in the same static directory:

```bash
sciscape viewer -o public/viewer.html
cp report/data.json public/data.json
```

The hosted viewer auto-loads same-directory `data.json`. To keep data elsewhere,
open `viewer.html?data=path/to/data.json` instead.

Launch the local web interface:

```bash
sciscape web --host 127.0.0.1 --port 8000
```

### Python API

```python
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline

cfg = KeywordExtractionConfig(
    abstract_path="abstracts.parquet",
    membership_path="membership.parquet",
    include_title=True,
    top_n_keywords=100,
)
keywords = run_keyword_pipeline(cfg)
keywords.to_parquet("keywords.parquet", index=False)
```

### Main Outputs

- `keywords.parquet`.
  - `term` preserves the extracted raw keyword surface used by downstream code.
  - `display_label`, `quality_score`, and `quality_flags` expose the
    domain-agnostic refinement layer for human-facing reports.
- dashboard/report artifacts from `sciscape visualize`.
- optional blank viewer artifacts from `sciscape viewer`.
- web UI session artifacts when using `sciscape web`.

### Modules Involved

- `sciscape.keyword_extraction`
- `sciscape.keyword_extraction.visualization`
- `sciscape.visualization`
- `sciscape.web`

## Workflow 4: Clustering Engine Only

Use this for lower-level clustering experiments where keyword extraction and
visualization are intentionally out of scope.

### Inputs

- Edge table as Parquet, or a ZIP/TSV edge table accepted by the legacy
  clustering pipeline.

### CLI

```bash
sciscape cluster edges.zip edges.txt \
  --levels 5,100 80,500 \
  -o membership.parquet
```

### Python API

```python
from sciscape.clustering import LeidenConfig, run_pipeline

tables = run_pipeline(
    zip_path="edges.zip",
    inner_name="edges.txt",
    config=LeidenConfig(
        level_constraints=[(5, 100), (80, 500)],
        resolution_bounds=(1e-3, 5.0),
    ),
)
tables.membership.write_parquet("membership.parquet")
```

### Main Outputs

- `membership.parquet`.
- description table.
- per-level resolution and quality metadata.

### Modules Involved

- `sciscape.clustering`
- `rust/` through `sciscape_leiden` when the Rust backend is available.

## Notes On Research-Only Surfaces

Dongdaemun and adaptive-refinement names are research/development surfaces, not
the default public workflow. Use `docs/research/dongdaemun/core/dongdaemun_naming_contract.md` to
distinguish `Dongdaemun-post`, `Dongdaemun-refinement`, and diagnostic-only
artifacts before making method claims.
