# SciScape Release Readiness Checklist

Status: local pre-release gate
Date: 2026-05-27

This checklist separates the supported SciSci full-cycle analysis/visualization
package surface from internal research tracks and defines the local validation
gate to run before a push or release.

## Supported Package Surface

- Python package `sciscape`.
- CLI entry point `sciscape`.
- Data conversion and OpenAlex ingestion.
- Multi-layer edge construction and combination.
- Rust CPM/Leiden clustering backend in `rust/`.
- Hierarchical landscape construction.
- Keyword extraction and Rust text acceleration backend in `rust-text/`.
- Network visualization, report generation, standalone viewer export, and web
  commands.
- Evaluation utilities for stability, quality, and review workflows.

## Internal Research Surface

- `Dongdaemun-post` is the current manuscript-backed hierarchy repair method.
- `Dongdaemun-refinement` is opt-in R&D until cross-sample quality and runtime
  evidence is strong enough to promote.
- Adaptive-refinement and basin-transition artifacts are diagnostics unless a
  track document explicitly promotes them.
- Follow `docs/research/dongdaemun/core/dongdaemun_naming_contract.md` and
  `docs/research/leiden_basin/core/leiden_multibasin_research_guardrails.md` for claim boundaries.

## Local Release Gate

Run:

```bash
./scripts/release_check.sh
```

The script uses `uv run --extra dev --extra web` for Python checks when `uv` is
available and falls back to `python`/`maturin` commands otherwise.

The gate performs:

- whitespace/conflict-marker check with `git diff --check`;
- staged diff whitespace/conflict-marker check with `git diff --cached --check`;
- Rust tests for `rust/` and `rust-text/`;
- editable rebuild of both PyO3 extensions with `maturin develop`;
- `scripts/sciscape_quality_gate.py --smoke --web-demo-smoke`, covering
  artifact filtering, term co-occurrence, dashboard generation, the web demo
  launcher, local demo opening, and key visualization/download endpoints without
  external data;
- full Python test suite with `pytest -q`;
- CLI import/help smoke check.

## Optional Demo Output Gate

Generated live demo outputs can be checked against the canonical preset
manifest:

```bash
uv run --extra dev python scripts/sciscape_quality_gate.py \
  --demo-root workspace/examples_output/openalex_live
```

Use `--allow-missing` when documenting a machine that has not generated the
live examples yet.

## Optional Artifact Contract Gate

Any existing SciScape result root, report directory, or `report/data.json` can
be checked for feature availability and blocking schema issues:

```bash
uv run --extra dev python scripts/sciscape_quality_gate.py \
  --artifact-root workspace/examples_output/openalex_live/perovskite_solar_cells_2020_2024 \
  --write-artifact-contract
```

This writes `landscape/qa/artifact_contract.json` when a landscape directory is
detected. The same validator powers feature inference for local/static result
loading. The contract also scans keyword tables and embedded report terms for
HTML, LaTeX preamble, and publisher metadata fragments. Top-ranked
contamination is treated as a blocking release issue.

## Keyword Extraction Scale TODO

The current keyword engine is validated for capped Atlas-scale extraction with
several thousand clusters. Do not claim unrestricted full-corpus scale support
until `docs/developer/keyword_extraction_scaling.md` has a completed benchmark
matrix covering runtime, memory high-water mark, shard reuse, and QC failures
across capped and uncapped runs.

## Fresh Clone Smoke

Before a public release, verify from a new clone or clean virtual environment:

```bash
python -m pip install --upgrade pip
python -m pip install ./rust ./rust-text .
sciscape --help
python -m pytest -q
```

Use `python -m pip install -e ./rust -e ./rust-text -e ".[dev,viz,arrow,openalex,web]"`
for editable development installs.

## Artifact Policy

Generated result directories under `research/**/results/**` are ignored by
default. Existing tracked curated evidence remains tracked, but new artifacts
should be committed only after review against `research/DATA_RETENTION_PLAN.md`.

Prefer committing compact summaries, manifests, and paper-facing bundles over
raw trace directories. Use `git add -f` only when the retention plan justifies a
new result artifact.

## Before External Release

- Audit README research claims against the current tracked evidence.
- Update `CHANGELOG.md` and version metadata.
- Confirm CI green on the target branch.
- Confirm license/distribution intent, because the root package is currently
  `LicenseRef-KRISS-Internal`.
