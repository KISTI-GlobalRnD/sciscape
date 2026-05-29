# What Data Goes Into SciScape?

Status: public entrypoint
Date: 2026-05-29

Use this page to choose the right SciScape entrypoint from the files or query
you already have.

## Choose Your Starting Point

| What you have | Use this command or surface | Required input | Main output |
| --- | --- | --- | --- |
| A topic query | `sciscape query` or `sciscape web` | Query text, optional year range and OpenAlex email | `abstracts.parquet`, edge tables, `landscape/`, `report/` |
| Existing abstracts and edges | `sciscape landscape` | `abstracts.parquet` plus one edge table or `--layers` | `membership.parquet`, `keywords.parquet`, `report/data.json` |
| Existing membership | `sciscape keywords` then `sciscape visualize` | `abstracts.parquet`, `membership.parquet` | `keywords.parquet`, dashboard/report HTML |
| Existing keyword table | `sciscape visualize` or `sciscape viewer` | `keywords.parquet`, `.csv`, or `.tsv` | standalone dashboard or report |
| Existing report data | `sciscape viewer` or GitHub Pages | `report/data.json` | browser-only static viewer |
| Graph-tool export target | `sciscape export` | edge table and membership table | `.gexf` or `.graphml` |

## File Checklist

### `abstracts.parquet`

Required for landscape and keyword extraction:

- `uid`: stable paper ID.
- `title`: paper title.
- `abstract`: paper abstract text.
- `pubyear`: publication year.

### Edge Table

Required for clustering and landscape when you already have a network:

- `uid1`: source paper ID.
- `uid2`: target paper ID.
- `rel_sum2` or another numeric weight column accepted by the pipeline.

Layer-specific files are usually named `edges_dc.parquet`, `edges_bc.parquet`,
`edges_cc.parquet`, or `edges_emb.parquet`.

### `membership.parquet`

Required when clustering is already done and you only want keywords or visuals:

- `uid`: paper ID.
- at least one cluster column, usually `cluster`, `cluster_nano`,
  `cluster_micro`, or another `cluster_*` column.

### `keywords.parquet` / `keywords.csv` / `keywords.tsv`

Required for direct visualization:

- `cluster_id`: cluster ID.
- `term`: keyword text.
- optional but recommended: `score`, `frequency`, `doc_coverage`,
  `display_label`, `quality_score`, `quality_flags`.

### `report/data.json`

Required for static web viewing:

- produced by `sciscape landscape` or `sciscape visualize`.
- place it next to `viewer.html` as `data.json`, or open
  `viewer.html?data=path/to/data.json`.

## Curated Demo Outputs

The canonical live demo presets are defined in
`examples/demo_presets.json` and run through `examples/openalex_live_demo.py`.

```bash
uv run --extra dev python examples/openalex_live_demo.py --list-presets
uv run --extra dev python examples/openalex_live_demo.py --preset both --email you@example.org
```

By default, generated demo outputs are written under:

```text
workspace/examples_output/openalex_live/
├── perovskite_solar_cells_2020_2024/
│   ├── abstracts.parquet
│   ├── edges.parquet
│   └── landscape/report/data.json
└── graph_neural_networks_2020_2024/
    ├── abstracts.parquet
    ├── edges.parquet
    └── landscape/report/data.json
```

Validate generated demo outputs:

```bash
uv run --extra dev python scripts/sciscape_quality_gate.py \
  --demo-root workspace/examples_output/openalex_live
```
