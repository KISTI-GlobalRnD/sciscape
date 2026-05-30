# Atlas App Benchmark For SciScape

Status: planning benchmark
Date: 2026-05-30

This note benchmarks the NanoClustering Science Atlas Explorer as a reference
for SciScape's non-UI stabilization work. The goal is not to copy the Atlas UI.
It is to copy the package and data-contract habits that make the Atlas app
auditable, replaceable, and smoke-testable.

## Reference Surfaces

Benchmarked from:

- `/home/kimyoungjin06/Desktop/Workspace/1.1.4.KISTI_NanoClustering/apps/science-atlas-explorer/`
- `scripts/serve_science_atlas_api.py`
- `outputs/specter2/_current/science_atlas_explorer_paris_datapack_domain8_micro3773_desc_work_20260528/`

Useful Atlas files and roles:

| Atlas surface | Role |
| --- | --- |
| `MANIFEST.json` | top-level datapack identity, module map, counts, source paths, API attachment roles |
| `atlas_versions.json` | stable version block exposed by `/versions` and echoed in API payloads |
| `core/` | cluster spine: nodes, terms, representatives, neighbor edges, search docs |
| `layout/`, `terrain_layout/`, `graph_layout/` | replaceable layout modules with their own manifests and QA |
| `dashboard/` | derived mart for review, source health, search documents, progress |
| `work/` | document-level provenance and title search package |
| `descriptions/`, `excellence/` | optional enrichment modules with independent manifests |
| `qa/` | package QA, count reconciliation, and readiness proof |

## Atlas Patterns Worth Copying

1. Module separation is explicit.

   Atlas separates the package into stable modules: `core`, `layout`,
   `dashboard`, `work`, enrichment modules, and `qa`. Sciscape does not need the
   same scale, but it should stop treating `report/data.json` as the only
   contract. A small result root should have explicit artifacts and a manifest.

2. Versions are payload-level facts.

   Atlas has an explicit version block:

   - `datapack_version`
   - `atlas_version`
   - `hierarchy_version`
   - `layout_version`
   - module-specific versions
   - `created_at_utc`

   Sciscape should introduce a small `sciscape_versions.json` or embed the same
   block inside `report/data.json`. Web/API/viewer code should be able to show
   and validate this without guessing from path names.

3. API responses echo data provenance.

   Atlas response types include `versions`, evidence source names, feature
   availability, and typed optional sections. Sciscape should do the same in a
   lighter form: `data.json`, local web job results, and term network responses
   should expose the artifact version and available features.

4. Optional modules are advertised, not assumed.

   Atlas `/versions` returns a `features` block. This prevents docs and UI from
   claiming that a sidecar is live when it is not attached. Sciscape needs the
   same idea for:

   - `has_keywords`
   - `has_membership`
   - `has_edges`
   - `has_report_data`
   - `has_term_network`
   - `has_cooccurrence_evidence`
   - `has_temporal`
   - `has_hierarchy`

5. QA is a first-class artifact.

   Atlas packages keep `qa/package_qa.json`, count reconciliation, and smoke
   records with the datapack. Sciscape should generate a compact
   `qa/artifact_contract.json` for report roots and demo outputs. The release
   gate can then validate a real artifact rather than only checking functions.

6. Document-level provenance is preserved.

   Atlas treats work membership as a provenance anchor. Sciscape's equivalent is
   the trio `abstracts.parquet`, `membership.parquet`, and `edges.parquet`.
   Demo/report bundles should not be considered complete if they only preserve
   labels and omit the data needed to interpret them.

7. Smoke tests target the contract, not the visuals.

   Atlas has Playwright smoke for UI behavior, but the stronger package pattern
   is that it tests stable payloads and route contracts. Since Sciscape UI will
   be rewritten, current work should focus on contract-level smoke:

   - manifest loads
   - expected files exist
   - schemas contain required columns
   - counts reconcile
   - local demo can be opened
   - cluster network and term network endpoints return non-empty payloads

## What Not To Copy Directly

- Do not copy Atlas' large sidecar architecture into Sciscape now.
- Do not add DuckDB/SQLite package marts unless a specific workflow needs them.
- Do not hard-code domain-specific levels like Domain/Macro/Meso/Micro/Nano into
  Sciscape's public contract. Sciscape should accept generic `cluster_*` levels
  and describe whichever levels exist.
- Do not depend on the current Sciscape web UI. The UI is expected to be
  replaced.
- Do not add a large checked-in full demo datapack. Keep full OpenAlex demos
  reproducible and ignored; add only a small offline demo if needed.

## Sciscape Adaptation

### Minimal Result Root

For a complete local SciScape result:

```text
<result_root>/
├── MANIFEST.json
├── abstracts.parquet
├── edges.parquet
└── landscape/
    ├── membership.parquet
    ├── keywords.parquet
    ├── sciscape_versions.json
    ├── qa/
    │   └── artifact_contract.json
    └── report/
        ├── data.json
        ├── index.html
        └── report.html
```

The manifest can be optional for old outputs but should be generated by new
`landscape`, `visualize`, and demo workflows.

### Version Block

Recommended minimal version block:

```json
{
  "schema_version": "sciscape_result_manifest_v1",
  "sciscape_version": "0.2.0",
  "result_version": "sciscape_result_<timestamp_or_slug>",
  "created_at_utc": "2026-05-30T00:00:00+00:00",
  "pipeline": {
    "source": "sciscape landscape",
    "cluster_backend": "rust_leiden",
    "keyword_backend": "python_or_rust_text"
  }
}
```

### Feature Block

Recommended feature block:

```json
{
  "features": {
    "keywords": true,
    "membership": true,
    "edges": true,
    "report_data": true,
    "term_network": true,
    "cooccurrence_evidence": true,
    "hierarchy": false,
    "temporal": false
  }
}
```

### Contract Validator

Add one validator entrypoint that can validate either a result root or a report
data file:

```bash
uv run --extra dev python scripts/sciscape_quality_gate.py \
  --artifact-root workspace/examples_output/openalex_live/perovskite_solar_cells_2020_2024
```

Initial validator checks:

- `abstracts.parquet`: `uid`, `title`, `abstract`, `pubyear`
- edge table: `uid1`, `uid2`, one numeric weight column
- `membership.parquet`: `uid`, at least one `cluster` or `cluster_*` column
- `keywords.parquet`: `cluster_id`, `term`, score/frequency columns when present
- `report/data.json`: parseable JSON, non-empty cluster list, schema/version block
- feature flags match available files and columns

### Release Gate Mapping

Current release gate already has:

- keyword artifact filtering smoke
- term co-occurrence smoke
- dashboard generation smoke
- web demo launcher smoke

Next release gate layer should add:

- synthetic result-root manifest generation
- artifact contract validation
- optional validation of existing live demo outputs with strict artifact schemas

## Recommended Next Implementation Slice

1. Add `docs/developer/artifact_contract.md`.
2. Add `sciscape/artifacts.py` or `sciscape/io_contract.py` with:
   - required column definitions
   - feature inference
   - result-root validation
   - compact manifest writer
3. Add `--artifact-root` to `scripts/sciscape_quality_gate.py`.
4. Update `export_report` to embed/write a version block in `data.json`.
5. Add tests for valid and invalid synthetic result roots.
6. Keep UI changes out of scope.

This gives Sciscape the Atlas-style durability without inheriting the Atlas
app's scale or UI assumptions.
