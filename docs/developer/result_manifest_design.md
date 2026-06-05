# SciScape Result Manifest Design

Date: 2026-06-01
Status: P0 design draft

This document defines the first `result_manifest.json` contract. The manifest is
the result-root navigation and provenance layer above `artifact_contract.json`.
It should answer a user's practical question: "what is this result, what can I
open, what is safe to show, and where are the files?"

## Design Goals

- Make every result root self-describing.
- Let the local web app open a result without folder-specific heuristics.
- Keep feature exposure states artifact-backed: `hidden`, `beta`, or `stable`.
- Connect long-running jobs to visible progress, partial outputs, checkpoints,
  failure state, and resume metadata.
- Preserve backward compatibility with older result roots that only have
  inferable artifacts.
- Keep the manifest small enough to read in a text editor.

## Non-Goals

- It is not a full workspace database.
- It is not a replacement for `artifact_contract.json`.
- It is not a substitute for report `data.json`.
- It must not let the UI advertise a feature that the validator would reject.
- It should not duplicate large tables, records, matrices, or graph edges.

## Relationship To Existing Contracts

| Contract | Scope | Source Of Truth For |
|---|---|---|
| `result_manifest.json` | result root | navigation, provenance, artifact paths, exposure states, run state, export list |
| `artifact_contract.json` | validator snapshot | artifact existence, inferred features, warnings, blocking errors, counts |
| `report/data.json` | static viewer payload | bounded viewer data, atlas nodes/edges, embedded report feature block |
| `demo_presets.json` | curated examples | demo list, expected files, suggested command |
| `workspace.json` | multi-result workspace | projects, recent results, rule sets, reusable views; see `workspace_manifest_design.md` |

The manifest can point to the artifact contract, but the artifact contract
remains stricter. If the manifest says a feature is available and the validator
disagrees, the validator wins.

## Canonical Location

New outputs should write:

```text
<result_root>/result_manifest.json
```

For compatibility, readers may also accept legacy:

```text
<result_root>/MANIFEST.json
```

Writers should use `result_manifest.json`. Existing documentation that mentions
`MANIFEST.json` should be treated as a legacy alias.

## Minimal Result Root

The preferred result root becomes:

```text
<result_root>/
+-- result_manifest.json
+-- abstracts.parquet
+-- edges.parquet
`-- landscape/
    +-- membership.parquet
    +-- keywords.parquet
    +-- edge_evidence_samples.json
    +-- sciscape_versions.json
    +-- qa/
    |   `-- artifact_contract.json
    `-- report/
        +-- data.json
        +-- index.html
        `-- report.html
```

Older outputs may omit `result_manifest.json`. In that case the validator may
infer a temporary manifest view from existing artifacts, but should mark the
manifest state as `missing`.

## Schema Overview

Required top-level fields:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `schema_version` | string | yes | `sciscape_result_manifest_v1` |
| `result_id` | string | yes | stable ID for the result root |
| `title` | string | yes | user-facing display title |
| `result_kind` | string | yes | `query_result`, `file_pipeline`, `static_bundle`, `demo_result`, or `imported_result` |
| `created_at_utc` | string | yes | ISO-8601 timestamp |
| `updated_at_utc` | string | no | last manifest update time |
| `sciscape_version` | string | yes | package version that wrote the manifest |
| `result_root` | string | yes | relative root marker, usually `"."` |
| `source` | object | yes | query/file/source metadata |
| `run_state` | object | yes | status and progress metadata |
| `artifacts` | object | yes | relative artifact registry |
| `features` | object | yes | feature exposure states |
| `quality` | object | yes | validator and gate summary |
| `exports` | array | yes | generated export list |
| `provenance` | object | yes | command/config/runtime metadata |

Optional top-level fields:

| Field | Meaning |
|---|---|
| `description` | free-form short result description |
| `tags` | local labels for browsing |
| `ui` | display hints that are allowed to be ignored |
| `notes` | human-readable operator notes |

## Feature Exposure State

Each feature entry should be an object, not a boolean.

```json
{
  "cluster_map": {
    "state": "stable",
    "reason": "membership and atlas payload validated",
    "artifact_refs": ["membership", "report_data", "artifact_contract"],
    "warnings": []
  }
}
```

Allowed states:

| State | Meaning |
|---|---|
| `hidden` | feature should not appear in the app |
| `beta` | feature may appear with caveats or warnings |
| `stable` | feature can be treated as a normal product surface |

Initial feature keys should match `sciscape.artifacts.FEATURE_KEYS` and add
`cooccurrence` as a more specific P1/P1.5 display surface:

- `overview`
- `cluster_map`
- `keyword`
- `term_network`
- `cooccurrence`
- `matrix`
- `evidence`
- `temporal`
- `evolution`
- `narrative`
- `quality`
- `export`

Compatibility rule: the existing boolean feature block in
`artifact_contract.json` can be mapped to `stable` when the validator passes and
to `hidden` when false. Features with warnings but no blocking errors may be
mapped to `beta`.

## Artifact Registry

`artifacts` should be a dictionary keyed by stable artifact refs. Each artifact
record should be relative to the result root.

```json
{
  "keywords": {
    "role": "keywords",
    "path": "landscape/keywords.parquet",
    "format": "parquet",
    "schema_version": null,
    "status": "present",
    "required_for": ["keyword", "term_network", "cooccurrence"],
    "rows": 240,
    "columns": ["cluster_id", "term", "score"],
    "size_bytes": 17342,
    "checksum": null
  }
}
```

Required artifact record fields:

| Field | Required | Meaning |
|---|---:|---|
| `role` | yes | semantic role, such as `records`, `edges`, `membership`, `keywords`, `report_data`, `export`, or `qa` |
| `path` | yes | path relative to result root |
| `format` | yes | `parquet`, `json`, `html`, `csv`, `graphml`, `gexf`, or `directory` |
| `status` | yes | `present`, `missing`, `partial`, `failed`, or `generated` |
| `required_for` | yes | feature keys this artifact supports |

Optional artifact record fields:

- `schema_version`
- `rows`
- `columns`
- `size_bytes`
- `checksum`
- `created_at_utc`
- `description`
- `warnings`

Initial stable artifact refs:

| Ref | Typical Path |
|---|---|
| `records` | `abstracts.parquet` |
| `edges` | `edges.parquet` or `combined_edges.parquet` |
| `membership` | `landscape/membership.parquet` |
| `keywords` | `landscape/keywords.parquet` |
| `edge_evidence` | `landscape/edge_evidence_samples.json` |
| `report_data` | `landscape/report/data.json` |
| `report_html` | `landscape/report/report.html` |
| `viewer_html` | `landscape/report/index.html` |
| `artifact_contract` | `landscape/qa/artifact_contract.json` |
| `term_network` | term-network payload or future stable term-network artifact |
| `cooccurrence` | `landscape/term_cooccurrence.parquet` and `landscape/term_cooccurrence_map.json` |
| `matrix` | `matrices/<matrix_id>/matrix_manifest.json` plus values and entity tables |
| `temporal` | `temporal/temporal_manifest.json` plus periods, activity, entity series, events, and QA |
| `evolution` | `evolution/evolution_manifest.json` plus slices, states, transitions, lineages, events, and QA |
| `narrative` | `narrative/narrative_manifest.json` plus targets, claims, evidence refs, links, sections, reviews, and QA |
| `export_manifest` | `exports/<export_id>/export_manifest.json` plus files, inputs, transforms, and QA |
| `job_status` | live query status JSON |
| `keyword_progress` / `pipeline_progress` | keyword or pipeline progress JSON |
| `scoring_shard_manifest` | keyword scoring shard manifest |

## Run State

`run_state` connects result roots to live jobs and long-running analysis.

```json
{
  "status": "complete",
  "started_at_utc": "2026-06-01T00:00:00+00:00",
  "finished_at_utc": "2026-06-01T00:12:10+00:00",
  "heartbeat_at_utc": "2026-06-01T00:12:10+00:00",
  "progress": {
    "current": 100,
    "total": 100,
    "unit": "percent"
  },
  "shards": {
    "total": 1,
    "complete": 1,
    "failed": 0,
    "running": 0
  },
  "checkpoints": [],
  "partial_outputs": [],
  "failure": null,
  "resume": {
    "supported": false,
    "command": null
  }
}
```

Allowed run statuses:

- `planned`
- `queued`
- `running`
- `partial`
- `complete`
- `failed`
- `cancelled`
- `stopped_by_qc`
- `imported`

Live OpenAlex query jobs write `job_status.json` in the result root and mirror
that state into `run_state`. Progress callbacks refresh the lightweight status
sidecar, while `result_manifest.json` is refreshed at lifecycle boundaries such
as job start, completion, and failure. Manifest generation also detects existing
`keyword_progress.json`, `progress.json`, and `scoring_shards/manifest.json`
sidecars so reopened result folders retain progress, shard, checkpoint, partial
output, and resume metadata.

## Quality Block

The quality block summarizes validation without replacing detailed QA files.

```json
{
  "validation_state": "passed",
  "artifact_contract_path": "landscape/qa/artifact_contract.json",
  "warning_count": 0,
  "blocking_count": 0,
  "gate_paths": ["landscape/qa/artifact_contract.json"],
  "last_validated_at_utc": "2026-06-01T00:12:15+00:00"
}
```

Allowed validation states:

- `not_run`
- `passed`
- `passed_with_warnings`
- `blocked`
- `missing_manifest`

## Export List

`exports` should list generated user-facing outputs and external files.

```json
[
  {
    "export_id": "report_html",
    "kind": "html_report",
    "path": "landscape/report/report.html",
    "format": "html",
    "feature_refs": ["overview", "keyword", "cluster_map"],
    "source_artifact_refs": ["report_data"],
    "export_manifest_ref": "exports/report_html/export_manifest.json",
    "status": "present",
    "files": [
      {
        "file_id": "report",
        "role": "primary",
        "path": "landscape/report/report.html",
        "format": "html",
        "public_share_state": "local",
        "bytes": 84012,
        "exists": true
      }
    ]
  }
]
```

Export kinds should start small:

- `static_viewer`
- `html_report`
- `json_report_data`
- `gexf_graph`
- `graphml_graph`
- `keyword_table`
- `cooccurrence_table`
- `matrix_table`
- `vosviewer_map_network`
- `package_bundle`

When an export manifest exists, the list entry should include
`export_manifest_ref`. Legacy entries may omit it and should be treated as beta
until export QA is available.

For manifest-backed exports, the list entry should expose a compact `files`
array copied from `export_files.parquet`. This is intentionally a small
inventory, not a duplicate of the exported files. It lets the web app and static
viewer resolve bundle outputs from `result_manifest.json` without opening every
export sidecar first.

## Source Block

The source block should describe what created the result without forcing every
pipeline to share the same ingestion model.

```json
{
  "source_type": "openalex_query",
  "query": "graph neural networks bibliometrics",
  "source_files": [],
  "record_count": 1000,
  "retrieved_at_utc": "2026-06-01T00:00:00+00:00",
  "filters": {
    "from_year": 2018,
    "to_year": 2026
  }
}
```

Allowed source types:

- `openalex_query`
- `wos_file`
- `scopus_file`
- `bibtex_file`
- `parquet_records`
- `static_bundle`
- `demo_fixture`
- `unknown`

## Provenance Block

The provenance block should capture enough information to rerun or audit the
analysis.

```json
{
  "commands": [
    "sciscape query --query 'graph neural networks bibliometrics' --limit 1000"
  ],
  "config_paths": [],
  "config_hash": null,
  "git_commit": null,
  "git_dirty": null,
  "random_seed": null,
  "environment": {
    "python": null,
    "platform": null
  }
}
```

## Minimal Example

```json
{
  "schema_version": "sciscape_result_manifest_v1",
  "result_id": "openalex_gnn_bibliometrics_20260601",
  "title": "OpenAlex: graph neural networks bibliometrics",
  "result_kind": "query_result",
  "created_at_utc": "2026-06-01T00:00:00+00:00",
  "sciscape_version": "0.2.0",
  "result_root": ".",
  "source": {
    "source_type": "openalex_query",
    "query": "graph neural networks bibliometrics",
    "source_files": [],
    "record_count": 1000,
    "retrieved_at_utc": "2026-06-01T00:00:00+00:00",
    "filters": {}
  },
  "run_state": {
    "status": "complete",
    "started_at_utc": "2026-06-01T00:00:00+00:00",
    "finished_at_utc": "2026-06-01T00:12:10+00:00",
    "heartbeat_at_utc": "2026-06-01T00:12:10+00:00",
    "progress": {"current": 100, "total": 100, "unit": "percent"},
    "shards": {"total": 1, "complete": 1, "failed": 0, "running": 0},
    "checkpoints": [],
    "partial_outputs": [],
    "failure": null,
    "resume": {"supported": false, "command": null}
  },
  "artifacts": {
    "records": {
      "role": "records",
      "path": "abstracts.parquet",
      "format": "parquet",
      "status": "present",
      "required_for": ["overview", "evidence", "temporal"]
    },
    "membership": {
      "role": "membership",
      "path": "landscape/membership.parquet",
      "format": "parquet",
      "status": "present",
      "required_for": ["cluster_map", "evidence"]
    },
    "keywords": {
      "role": "keywords",
      "path": "landscape/keywords.parquet",
      "format": "parquet",
      "status": "present",
      "required_for": ["keyword", "term_network", "cooccurrence"]
    },
    "artifact_contract": {
      "role": "qa",
      "path": "landscape/qa/artifact_contract.json",
      "format": "json",
      "status": "present",
      "required_for": ["quality"]
    }
  },
  "features": {
    "overview": {"state": "stable", "reason": "records validated", "artifact_refs": ["records"], "warnings": []},
    "cluster_map": {"state": "stable", "reason": "membership validated", "artifact_refs": ["membership"], "warnings": []},
    "keyword": {"state": "stable", "reason": "keywords validated", "artifact_refs": ["keywords"], "warnings": []},
    "term_network": {"state": "stable", "reason": "feature validated", "artifact_refs": ["cooccurrence"], "warnings": []},
    "cooccurrence": {"state": "stable", "reason": "feature validated", "artifact_refs": ["cooccurrence"], "warnings": []},
    "matrix": {"state": "hidden", "reason": "no matrix artifact", "artifact_refs": [], "warnings": []},
    "evidence": {"state": "stable", "reason": "records and membership joinable", "artifact_refs": ["records", "membership"], "warnings": []},
    "temporal": {"state": "beta", "reason": "pubyear exists but no temporal artifact", "artifact_refs": ["records"], "warnings": []},
    "evolution": {"state": "hidden", "reason": "no evolution artifact", "artifact_refs": [], "warnings": []},
    "narrative": {"state": "hidden", "reason": "no narrative artifact", "artifact_refs": [], "warnings": []},
    "quality": {"state": "stable", "reason": "artifact contract passed", "artifact_refs": ["artifact_contract"], "warnings": []},
    "export": {"state": "beta", "reason": "report export present but export manifest incomplete", "artifact_refs": ["report_data"], "warnings": []}
  },
  "quality": {
    "validation_state": "passed",
    "artifact_contract_path": "landscape/qa/artifact_contract.json",
    "warning_count": 0,
    "blocking_count": 0,
    "gate_paths": ["landscape/qa/artifact_contract.json"],
    "last_validated_at_utc": "2026-06-01T00:12:15+00:00"
  },
  "exports": [],
  "provenance": {
    "commands": [],
    "config_paths": [],
    "config_hash": null,
    "git_commit": null,
    "git_dirty": null,
    "random_seed": null,
    "environment": {}
  }
}
```

## Validator Responsibilities

The first implementation should add a small validator/writer without changing
all pipelines at once.

1. Read `result_manifest.json` when present.
2. Fall back to `MANIFEST.json` for legacy roots.
3. If no manifest exists, infer a temporary manifest view from
   `validate_result_root()`.
4. Resolve all artifact paths relative to the result root.
5. Reject absolute paths unless they are explicitly marked external source
   files.
6. Recompute file existence, size, and basic table metadata.
7. Reconcile manifest feature states against `artifact_contract.json`.
8. Downgrade unsupported or warning-heavy features to `hidden` or `beta`.
9. Preserve validator warnings as user-visible manifest quality notes.

## Implementation Plan

1. Add constants and dataclasses in `sciscape/artifacts.py`:
   `RESULT_MANIFEST_SCHEMA_VERSION`, `ArtifactRecord`, `FeatureExposure`,
   `RunState`, and `ResultManifest`.
2. Add `build_result_manifest(path, mode=...)` that wraps
   `validate_result_root()`.
3. Add `write_result_manifest(path, output_path=None, mode=...)`.
4. Update `scripts/sciscape_quality_gate.py` with `--write-result-manifest`.
5. Update web local result loading to prefer `result_manifest.json` before
   falling back to inference.
6. Update demo/static/query outputs to write the manifest.
7. Add tests for valid, missing, legacy, and contradictory manifests.
8. Update release readiness docs after the writer exists.
9. `[x]` Summarize manifest-backed export file inventories in
   `result_manifest.exports`.

## Open Design Decisions

- Whether `term_network` should receive its own stable sidecar separate from the
  P1.5 co-occurrence table/map artifacts.
- Whether checksums should be required for release bundles only or for every
  local result.
- Whether `result_id` should be generated from a slug, UUID, or content hash.
- Whether the first writer should update in place during long-running jobs or
  write a separate `job_status.json` and reference it from the manifest.
