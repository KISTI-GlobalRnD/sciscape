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

The script uses `uv run --extra dev` when `uv` is available and falls back to
`python`/`maturin` commands otherwise.

The gate performs:

- whitespace/conflict-marker check with `git diff --check`;
- staged diff whitespace/conflict-marker check with `git diff --cached --check`;
- Rust tests for `rust/` and `rust-text/`;
- editable rebuild of both PyO3 extensions with `maturin develop`;
- full Python test suite with `pytest -q`;
- CLI import/help smoke check.

## Planned Workflow Smoke

The next release-hardening step is a small full-cycle smoke test that exercises
the package identity directly:

- build a synthetic abstract table and edge table;
- run the landscape or equivalent module sequence;
- confirm membership, keywords, report, and viewer-facing artifacts exist;
- keep the test small enough to run inside the normal `pytest -q` gate.

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
