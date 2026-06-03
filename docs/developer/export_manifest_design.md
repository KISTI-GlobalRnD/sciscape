# Export Manifest Design

This document defines the export manifest contract for reports, static viewers,
graph files, matrix/table exports, maps, VOSviewer-style outputs, and result
bundles.

The purpose is to make every exported file traceable to its source result,
input artifacts, filters, feature states, schema versions, and QA checks. Export
files should be shareable, but they must not become detached from the analysis
state that produced them.

## Goal

SciScape should treat exports as first-class reproducible artifacts.

An export manifest must answer:

- what was exported;
- which result and source artifacts produced it;
- which filters, selections, view settings, and transforms were applied;
- which files were written and how they should be opened;
- whether stable IDs, weights, labels, and coordinates were preserved;
- whether the export is safe for public sharing or downstream tools;
- whether the export is stable, beta, blocked, or unavailable.

## Non-Goals

- Do not turn SciScape into a full graph editor.
- Do not claim full VOSviewer, Biblioshiny, Gephi, or Cytoscape replacement
  behavior from basic exports.
- Do not require every export to be a self-contained bundle.
- Do not include private absolute paths, credentials, local usernames, or
  unintended raw text in public export manifests.
- Do not treat a downloadable file as validated just because it exists.

## Export Manifest Versus Result Manifest

`result_manifest.json` lists available exports at the result-root level.

An export manifest is the detailed contract for one generated export or export
bundle. Result manifests should point to export manifests when they exist.

| Contract | Scope | Purpose |
| --- | --- | --- |
| `result_manifest.json` | whole result root | navigation, features, artifacts, run state, export list |
| `exports/<export_id>/export_manifest.json` | one export or bundle | inputs, transforms, files, QA, sharing status |
| `export_qa.json` | one export or bundle | validation summary for export safety and interoperability |

## Canonical Directory Shape

Reusable export artifacts should live under:

```text
<result_root>/exports/<export_id>/
  export_manifest.json
  export_files.parquet
  export_inputs.parquet
  export_transforms.parquet
  export_qa.json
  files/
    ...
```

Existing export files may keep their current locations, such as:

```text
<result_root>/landscape/report/data.json
<result_root>/landscape/report/report.html
<result_root>/landscape/report/index.html
<result_root>/network.gexf
<result_root>/network.graphml
```

In that case, the export manifest should still live under
`exports/<export_id>/` and reference those existing files by result-root
relative path.

For a landscape-scoped result, the same export directory may live under:

```text
<result_root>/landscape/exports/<export_id>/
```

Writers should prefer result-root `exports/` for files intended to be reused or
shared beyond a single landscape lens.

## Schema Versions

Use explicit schema names:

- `sciscape_export_manifest_v1`
- `sciscape_export_files_v1`
- `sciscape_export_inputs_v1`
- `sciscape_export_transforms_v1`
- `sciscape_export_qa_v1`

## Export Families

The contract should support these export families:

| Family | Examples | Primary consumer |
| --- | --- | --- |
| `static_viewer` | `index.html`, compact data bundle | browser/static hosting |
| `report` | HTML report, JSON report data | human review and sharing |
| `graph` | GEXF, GraphML, future Pajek | Gephi, Cytoscape, graph tools |
| `matrix` | sparse triplets, CSV/Parquet matrix tables | analyst workbench, external tools |
| `table` | keyword, membership, co-occurrence, QA tables | spreadsheet and scripts |
| `map` | atlas map payload, layout coordinates | static maps and downstream viewers |
| `vosviewer` | network/map/thesaurus-compatible tables | VOSviewer-style workflows |
| `package` | result bundle, demo bundle, release archive | handoff and reproducibility |

Family-specific exporters may add extra files, but they must still satisfy the
common export manifest and QA contract.

## Export Manifest

`export_manifest.json` is the source of truth for one export.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_export_manifest_v1` |
| `export_id` | string | stable local identifier |
| `title` | string | human-readable export title |
| `result_id` | string or null | parent result when available |
| `export_family` | string | one of the supported families |
| `export_kind` | string | specific kind, such as `gexf_graph` or `html_report` |
| `format` | string | primary file format |
| `status` | string | `passed`, `warning`, `blocked`, or `missing` |
| `feature_refs` | array | result features represented by the export |
| `source_artifacts` | array | artifact refs and roles used as inputs |
| `selection` | object | included entities, filters, subsets, and visible lens state |
| `transform_summary` | object | weighting, normalization, layout, rendering, and field mapping summary |
| `compatibility` | object | target tools, format version, and limitations |
| `outputs` | object | paths to files, inputs, transforms, QA, and optional bundle root |
| `created_at_utc` | string | creation timestamp |
| `warnings` | array | non-blocking caveats |

Example:

```json
{
  "schema_version": "sciscape_export_manifest_v1",
  "export_id": "network_graphml_default",
  "title": "GraphML network export",
  "result_id": "openalex_gnn_20260603",
  "export_family": "graph",
  "export_kind": "graphml_graph",
  "format": "graphml",
  "status": "passed",
  "feature_refs": ["cluster_map", "evidence", "export"],
  "source_artifacts": [
    {"role": "edges", "artifact_ref": "edges", "path": "edges.parquet"},
    {"role": "membership", "artifact_ref": "membership", "path": "landscape/membership.parquet"}
  ],
  "selection": {
    "scope": "full_result",
    "included_entity_types": ["work", "cluster"],
    "filters": [],
    "view_state_ref": null
  },
  "transform_summary": {
    "edge_weight_field": "rel_sum2",
    "cluster_field": "cluster",
    "layout_source": null,
    "label_source": "uid",
    "dropped_fields": []
  },
  "compatibility": {
    "target_tools": ["Cytoscape"],
    "format_version": "GraphML",
    "stable_ids_preserved": true,
    "limitations": []
  },
  "outputs": {
    "files": "export_files.parquet",
    "inputs": "export_inputs.parquet",
    "transforms": "export_transforms.parquet",
    "qa": "export_qa.json",
    "primary_file": "network.graphml"
  },
  "created_at_utc": "2026-06-03T00:00:00+00:00",
  "warnings": []
}
```

## Export Files Table

`export_files.parquet` inventories files produced by the export.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_export_files_v1` |
| `export_id` | string | parent export id |
| `file_id` | string | stable local file id |
| `path` | string | result-root relative path |
| `role` | string | `primary`, `sidecar`, `asset`, `manifest`, `qa`, or `index` |
| `format` | string | file format |
| `media_type` | string or null | MIME type when useful |
| `size_bytes` | int or null | file size |
| `checksum` | string or null | checksum when available |
| `public_share_state` | string | `safe`, `review`, `blocked`, or `internal` |

Optional columns:

- `row_count`, `column_count`;
- `schema_version_ref`;
- `target_tool`;
- `open_hint`;
- `warning_flags`.

Rules:

- `path` must be relative to the result root.
- Absolute local paths are invalid for public export manifests.
- The primary file must exist unless the export is marked `missing` or
  `blocked`.
- Files marked `public_share_state=safe` must pass private-path and raw-text
  leakage checks.

## Export Inputs Table

`export_inputs.parquet` records source artifacts and source feature states.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_export_inputs_v1` |
| `export_id` | string | parent export id |
| `input_id` | string | stable local input id |
| `artifact_ref` | string | result manifest artifact ref |
| `artifact_role` | string | semantic role |
| `artifact_path` | string | result-root relative path |
| `feature_state` | string | `stable`, `beta`, `hidden`, or `blocked` |
| `required` | bool | whether the export is invalid without this input |

Optional columns:

- `schema_version_ref`;
- `row_count`, `column_count`;
- `checksum`;
- `filter_ref`;
- `warning_flags`.

Rules:

- Required inputs must exist and resolve.
- Blocked source features cannot produce stable exports.
- Beta source features may produce beta exports if warnings are preserved in
  `export_qa.json`.

## Export Transforms Table

`export_transforms.parquet` records transformations applied before writing
files.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_export_transforms_v1` |
| `export_id` | string | parent export id |
| `transform_id` | string | stable transform id |
| `step_index` | int | ordered transform step |
| `transform_type` | string | `filter`, `field_mapping`, `normalization`, `layout`, `format_conversion`, `packaging`, or future type |
| `description` | string | short human-readable description |
| `parameters` | string or object | deterministic parameters |

Optional columns:

- `input_refs`;
- `output_refs`;
- `rule_set_ref`;
- `view_state_ref`;
- `warning_flags`.

Rules:

- Transform steps must be ordered and contiguous.
- Field mappings must preserve stable IDs unless the export kind explicitly
  declares a tool-specific renaming rule.
- Selected or visible-subset exports must record the selection transform.

## QA Contract

`export_qa.json` should summarize export validity and sharing safety.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_export_qa_v1` |
| `export_id` | parent export id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | named checks with status and counts |
| `counts` | files, inputs, transforms, missing files, private paths |
| `compatibility` | target tool checks and limitations |
| `warnings` | non-blocking warnings |
| `blocking_issues` | release-blocking issues |

Minimum checks:

- manifest schema is supported;
- primary export file exists;
- files, inputs, transforms, and QA paths resolve;
- required source artifacts exist;
- required source features are not blocked;
- result-root relative paths are used;
- private absolute paths are absent from public files and manifests;
- stable IDs are preserved or declared as mapped;
- graph exports include source, target, weight, and relation direction when
  applicable;
- matrix exports include row/column entity metadata;
- table exports record row count and field mapping;
- static viewer/report exports embed or reference compatible result data;
- package exports include artifact inventory and checksums when available;
- VOSviewer-style exports record counting method, field mapping, and
  compatibility limitations.

## Validation States

Export validation should feed the normal result contract:

| Condition | Result |
| --- | --- |
| at least one export manifest and QA pass, with primary files present | `export=stable` |
| exports exist but have missing optional files, beta inputs, or compatibility warnings | `export=beta` |
| an advertised export is missing primary files or has blocked QA | result `blocked` |
| enough data exists to write exports but no export manifest exists | `export=beta` |
| no exportable data exists | `export=hidden` |

This preserves the current behavior where basic report or graph exports can be
available while making stable export claims require manifest-backed QA.

## Family-Specific Rules

### Static Viewer And Report

Required refs:

- `report_data`;
- report/viewer HTML primary file;
- feature state snapshot;
- data schema version and app/viewer version when available.

Rules:

- Static viewer exports must record whether they are self-contained or require
  adjacent data files.
- Report exports must record generated timestamp, source result, and QA state.
- Public viewer bundles must not include private absolute paths.

### Graph Exports

Required refs:

- edge artifact;
- membership or node metadata;
- weight field and graph direction;
- node ID field and cluster field.

Rules:

- Node IDs must remain stable source keys.
- Missing membership may allow beta graph export only when node attributes
  clearly say cluster is unavailable.
- GEXF/GraphML compatibility should be checked at least for parseability and
  required node/edge fields.

### Matrix And Table Exports

Required refs:

- matrix or source table artifact;
- row/column entity metadata for matrices;
- normalization, weighting, and threshold metadata.

Rules:

- Matrix exports must not drop row/column entity keys.
- Table exports must record columns, row count, and filters.
- Co-occurrence table exports should point back to the stable co-occurrence or
  matrix artifact that produced them.

### VOSviewer-Style Exports

Required refs:

- network or co-occurrence matrix;
- label field mapping;
- weight/counting method;
- optional thesaurus/rule-set ref.

Rules:

- Do not mark VOSviewer-style exports as stable until the required file formats
  and field mappings are validated against the declared target workflow.
- VOSviewer compatibility limitations should be explicit, not implied by file
  names.

### Package Exports

Required refs:

- included artifacts and export manifests;
- package root;
- file inventory;
- checksums when available.

Rules:

- Package manifests must use relative paths.
- A package can include sidecars only when manifest-relative references still
  resolve after extraction.
- Release bundles should include QA summaries and feature states.

## Writer Utility Design

The first writer utility should wrap existing exports:

```python
write_export_manifest(
    result_root,
    *,
    export_id,
    export_family,
    export_kind,
    primary_file,
    source_artifacts,
    feature_refs,
    files=None,
    selection=None,
    transforms=None,
    compatibility=None,
)
```

The writer should:

1. validate source artifacts and feature states;
2. inventory output files;
3. write `export_files.parquet`;
4. write `export_inputs.parquet`;
5. write `export_transforms.parquet`;
6. generate `export_manifest.json`;
7. generate `export_qa.json`;
8. update the result manifest export list when requested;
9. return paths, counts, warnings, and QA status.

The first adapters should wrap existing report, static viewer, GEXF, and
GraphML exports. Matrix and VOSviewer-style adapters can follow after matrix
artifacts are implemented.

## Validator Utility Design

The validator should be reusable outside full result validation:

```python
validate_export_manifest(export_dir) -> ExportManifestValidationResult
```

It should return:

- schema version;
- export id, family, kind, and status;
- artifact paths;
- file, input, and transform counts;
- missing-file, private-path, and compatibility diagnostics;
- warnings and blocking issues;
- feature exposure suggestion.

Full result validation can then expose export states without rerunning export
commands.

## Implementation Order

1. Add schema constants and dataclasses for export manifests, files, inputs,
   transforms, and QA.
2. Add `validate_export_manifest`.
3. Add `write_export_manifest`.
4. Wrap existing report, static viewer, GEXF, and GraphML export paths.
5. Extend `validate_result_root` to identify export manifests and expose
   stable/beta/blocked export states.
6. Add a tiny synthetic export smoke that writes a table export and validates
   file inventory, source refs, and private-path checks.
7. Add matrix and VOSviewer-style export manifests only after their source
   artifact contracts are stable.

## Acceptance Criteria

- An export can be validated without opening the web app or external tool.
- Every export records source artifacts, feature states, transforms, output
  files, and QA.
- Exported files use stable IDs or explicitly declare field mappings.
- Public export manifests do not contain private absolute paths.
- Report, static viewer, GEXF, and GraphML outputs can be wrapped by the export
  manifest contract.
- `export=stable` requires manifest-backed QA, not only file existence.
- The full result contract can tell whether `export` is hidden, beta, stable,
  or blocked from artifacts alone.

## Open Questions

- Should export manifests require checksums for all files or only release
  bundles?
- Should static viewer exports embed result data or reference adjacent
  `data.json` by default?
- Should GEXF/GraphML parseability be checked in the writer, validator, or both?
- Should VOSviewer-style export validation include an external smoke fixture or
  only schema-level field checks?
- Should selected-subset exports preserve the original result manifest as a
  source ref or create a derived result manifest?
