# SciScape Project Structure

This file is the repo map for deciding where code, data, demos, and generated
artifacts belong.

## Main Surfaces

| Path | Role | Edit policy |
|---|---|---|
| `sciscape/` | Primary Python package implementation | Main development target |
| `sciscape/web/` | FastAPI web app engine for query-to-analysis workflows | Main web app target |
| `sciscape/keyword_extraction/visualization/` | Static dashboard/viewer generation | Main viewer target |
| `rust/` | Rust CPM/Leiden backend | Main clustering backend |
| `rust-text/` | Rust text/keyword acceleration backend | Main text backend |
| `sos/` | Compatibility shim to `sciscape.*` | Keep minimal |
| `tests/` | Package and workflow tests | Keep current with behavior |
| `docs/` | User, research, and development documentation | Keep navigable |
| `examples/` | Reproducible example scripts/notebooks | Keep small and source-like |
| `viewer/` | Curated GitHub Pages/static viewer demo | Keep directly openable |
| `workspace/` | Ignored local inputs, caches, runs, and web jobs | Do not commit generated contents |

## Current Data Entry Points

Use these as the user-facing mental model.

| User wants to... | Input | Command or location | Output to inspect/share |
|---|---|---|---|
| Run from a search query | OpenAlex query string | `sciscape web` or `sciscape query` | `workspace/openalex_output/`, `workspace/web_output/`, `data.json` |
| Visualize an existing SciScape result | `data.json` | `viewer/index.html` beside `viewer/data.json` | Browser viewer |
| Visualize a keyword table | `keywords.parquet`, `.csv`, or `.tsv` | `sciscape visualize keywords.parquet -o report/` | `report/data.json`, `report/index.html` |
| Run the local pipeline | `abstracts.parquet` plus `edges.parquet` or layer edges | `sciscape landscape ...` | membership, keywords, report bundle |
| Export to graph tools | `edges.parquet` plus `membership.parquet` | `sciscape export ...` | GEXF or GraphML |

For GitHub Pages, the minimum shareable bundle is:

```text
viewer/
├── index.html
└── data.json
```

`index.html` auto-loads same-directory `data.json` over HTTP(S).

## Local Workspace Outputs

These paths are useful locally, but should not be treated as source:

| Path | Meaning |
|---|---|
| `workspace/data/` | Local input/cache data |
| `workspace/output/` | Ad hoc local pipeline outputs |
| `workspace/examples_output/` | Example run outputs |
| `workspace/reports/` | Local generated reports |
| `workspace/web_output/` | FastAPI web job outputs |
| `build/`, `dist/`, `*.egg-info/` | Packaging artifacts |
| `docs/api/` | Generated API docs |

Most of these are ignored by `.gitignore`. Promote an artifact into the repo
only when it is a small curated demo or documented release artifact.
The former top-level `data/`, `output/`, `examples_output/`, `reports/`, and
`sciscape_web_output/` paths are legacy local paths and remain ignored during
migration.

## Research And Auxiliary Areas

| Path | Role | Rule |
|---|---|---|
| `research/` | Development and manuscript research artifacts | Do not mix with package cleanup unless explicitly scoped |
| `research/experiments/` | Experiment scripts and exploratory evaluations | Keep separate from package APIs |
| `research/auxiliary/` | Historical or auxiliary backend workspaces | Check before changing |

## Removed Top-Level Legacy Packages

The former top-level `clustering/`, `keyword_extraction/`, and root
`__init__.py` entries were removed from the source tree. Use
`sciscape.clustering`, `sciscape.keyword_extraction`, or the `sos.*`
compatibility shim where compatibility is required.

## Cleanup Rule

When adding a new file, decide which bucket it belongs to before writing it:

1. Package code goes under `sciscape/`.
2. Public examples go under `examples/`.
3. Static viewer demos go under `viewer/`.
4. Generated run outputs stay under `workspace/`.
5. Research-only material stays under `research/`.
6. Top-level package-like directories should not be added outside `sciscape/`
   or `sos/`.
