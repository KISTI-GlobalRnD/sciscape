# SciScape Examples

This directory contains runnable examples for the full-cycle SciSci
analysis/visualization workflow.

## Live OpenAlex Demos

`openalex_live_demo.py` provides two curated live OpenAlex presets:

| Preset | Dataset | Why it is useful |
| --- | --- | --- |
| `perovskite` | Perovskite solar cells, 2020-2024 | Dense citation structure, coherent materials-science subtopics, good default demo. |
| `gnn` | Graph neural networks, 2020-2024 | Broader CS/AI topic spread using `title_and_abstract.search` to reduce generic search noise. |

List presets without network access:

```bash
uv run --extra dev python examples/openalex_live_demo.py --list-presets
```

Run the default perovskite demo:

```bash
uv run --extra dev python examples/openalex_live_demo.py \
  --preset perovskite \
  --email you@example.org
```

Run the graph neural networks demo:

```bash
uv run --extra dev python examples/openalex_live_demo.py \
  --preset gnn \
  --email you@example.org
```

Run both presets:

```bash
uv run --extra dev python examples/openalex_live_demo.py \
  --preset both \
  --email you@example.org
```

For a faster fetch/edge-build-only pass:

```bash
uv run --extra dev python examples/openalex_live_demo.py \
  --preset both \
  --skip-landscape \
  --email you@example.org
```

Outputs are written under `examples_output/openalex_live/` by default. This
directory is intentionally ignored by git as local generated output.
