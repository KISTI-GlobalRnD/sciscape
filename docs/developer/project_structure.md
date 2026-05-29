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

## Documentation Map

The `docs/` tree is split by audience and artifact type. Keep each
human-facing parent small enough to scan before adding a new document.

| Path | Role |
|---|---|
| `docs/user/` | Public workflows, module map, and IO schemas |
| `docs/developer/` | Repo structure, release readiness, and maintenance decisions |
| `docs/research/` | Research design notes and claim boundaries |
| `docs/research/dongdaemun/` | Dongdaemun core, refinement, and manuscript notes |
| `docs/research/leiden_basin/` | Leiden basin and branch-adaptive research notes |
| `docs/research/hierarchy/` | Hierarchy postprocess research notes |
| `docs/research/dendrogram/` | CPM dendrogram implementation notes |
| `docs/research/methodology/` | General methodology, literature, and problem framing |
| `docs/papers/` | Manuscript drafts, reports, and paper figures |
| `docs/archive/` | Historical notes kept for traceability |
| `docs/assets/` | Images and static documentation assets |
| `docs/api/` | Generated API docs; ignored by source-navigation fanout checks |

## Documentation Fanout Rule

Human-facing documentation folders should stay at six visible entries or fewer.
Visible entries are direct child folders plus direct document files, excluding
`README.md`. When a folder would exceed that limit, split it into topic
subfolders and add or update a local `README.md` index.

Generated outputs, API docs, caches, and local workspace artifacts are excluded
from this rule. Run the check before committing documentation layout changes:

```bash
uv run --extra dev python scripts/check_doc_fanout.py
```

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
7. Documentation parents should pass the six-entry fanout check or be split
   into topic subfolders with a local index.
8. Large research script directories should get an inventory and reference
   migration plan before files are moved.
