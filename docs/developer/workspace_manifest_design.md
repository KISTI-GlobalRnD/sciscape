# Workspace Manifest Design

This document defines the workspace, project, dataset, run, result, rule-set,
view, and export registry contract.

The purpose is to make SciScape start from analysis state rather than from
internal folder guessing. A user should be able to open a workspace and see
projects, datasets, runs, validated results, reusable rules, views, and exports
without knowing which generated subfolder contains the right files.

## Goal

SciScape should treat a workspace as a lightweight registry over reproducible
artifacts.

A workspace contract must answer:

- what projects exist;
- which datasets and result roots belong to each project;
- which runs are active, complete, failed, or recoverable;
- which result roots are validated and what features they expose;
- which rule sets, views, and exports can be reused;
- which objects are missing, stale, archived, or blocked;
- how the local web app, CLI, and static report tooling should find the same
  analysis state.

## Non-Goals

- Do not build a full database in this milestone.
- Do not duplicate large records, graph edges, matrices, or report payloads in
  `workspace.json`.
- Do not replace `result_manifest.json`, `artifact_contract.json`, or
  `export_manifest.json`.
- Do not make the UI infer product features from workspace entries alone. The
  result validator and artifact contracts remain the authority for lenses.
- Do not require every legacy result root to be rewritten before it can be
  opened. Legacy results can be registered with warnings.

## Contract Layers

| Contract | Scope | Source of truth for |
| --- | --- | --- |
| `workspace.json` | workspace root | registry, projects, object refs, recent activity, defaults |
| `projects/<project_id>/project_manifest.json` | one project | topic, datasets, runs, results, rules, views, exports |
| `datasets/<dataset_id>/dataset_manifest.json` | one dataset | source, records path, schema, counts, ingest provenance |
| `runs/<run_id>/run_manifest.json` | one run/job | live or completed execution state, config, progress, outputs |
| `results/<result_id>/result_manifest.json` or external result root | one validated result | artifact paths, feature states, run state, exports |
| `rules/<rule_set_id>/rule_set_manifest.json` | one reusable rule set | cleaning, filter, thesaurus, alias, or view rules |
| `views/<view_id>/view_manifest.json` | one saved view | lens state, selected result, filters, pinned entities |
| `exports/<export_id>/export_manifest.json` | one export | exported files, inputs, transforms, QA |

The workspace registry points to object manifests. Object manifests point to
files. Artifact contracts validate files.

## Canonical Directory Shape

The preferred workspace root is:

```text
<workspace_root>/
  workspace.json
  workspace_qa.json
  projects/
    <project_id>/
      project_manifest.json
  datasets/
    <dataset_id>/
      dataset_manifest.json
      records.parquet
  runs/
    <run_id>/
      run_manifest.json
      status.json
      logs/
      partial/
  results/
    <result_id>/
      result_manifest.json
      ...
  rules/
    <rule_set_id>/
      rule_set_manifest.json
      rules.*
  views/
    <view_id>/
      view_manifest.json
  exports/
    <export_id>/
      export_manifest.json
      ...
```

Existing generated folders may remain under the current ignored locations such
as `workspace/output/`, `workspace/examples_output/`, `workspace/web_output/`,
or `viewer/`. The workspace manifest can register those result roots by
relative path without moving them.

## Schema Versions

Use explicit schema names:

- `sciscape_workspace_manifest_v1`
- `sciscape_workspace_project_manifest_v1`
- `sciscape_workspace_dataset_manifest_v1`
- `sciscape_workspace_run_manifest_v1`
- `sciscape_workspace_rule_set_manifest_v1`
- `sciscape_workspace_view_manifest_v1`
- `sciscape_workspace_qa_v1`

Result and export manifests keep their own schema versions:

- `sciscape_result_manifest_v1`
- `sciscape_export_manifest_v1`

## Workspace Manifest

`workspace.json` is the top-level registry.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_workspace_manifest_v1` |
| `workspace_id` | string | stable local identifier |
| `name` | string | human-readable workspace name |
| `root` | string | workspace-root relative marker, usually `.` |
| `created_at_utc` | string | creation timestamp |
| `updated_at_utc` | string | last registry update timestamp |
| `objects` | object | refs to projects, datasets, runs, results, rules, views, exports |
| `recent` | object | recent projects, runs, results, views, exports |
| `defaults` | object | default project, result, mode, and local output roots |
| `settings` | object | local preferences that are safe to persist |
| `warnings` | array | non-blocking caveats |

Example:

```json
{
  "schema_version": "sciscape_workspace_manifest_v1",
  "workspace_id": "workspace_local_default",
  "name": "SciScape Local Workspace",
  "root": ".",
  "created_at_utc": "2026-06-03T00:00:00+00:00",
  "updated_at_utc": "2026-06-03T00:00:00+00:00",
  "objects": {
    "projects": [
      {"project_id": "openalex_perovskite", "path": "projects/openalex_perovskite/project_manifest.json", "state": "active"}
    ],
    "datasets": [],
    "runs": [],
    "results": [
      {"result_id": "perovskite_2020_2024", "path": "workspace/examples_output/openalex_live/perovskite_solar_cells_2020_2024/result_manifest.json", "state": "validated"}
    ],
    "rule_sets": [],
    "views": [],
    "exports": []
  },
  "recent": {
    "projects": ["openalex_perovskite"],
    "results": ["perovskite_2020_2024"],
    "runs": [],
    "views": []
  },
  "defaults": {
    "project_id": "openalex_perovskite",
    "mode": "local_result",
    "output_roots": ["workspace/output", "workspace/examples_output", "workspace/web_output"]
  },
  "settings": {
    "auto_register_completed_runs": true,
    "show_legacy_results": true
  },
  "warnings": []
}
```

Rules:

- Paths must be relative to the workspace root unless explicitly marked as
  external.
- Absolute local paths are invalid in public or shareable workspace manifests.
- `workspace.json` must stay small and text-readable.
- A workspace object may be registered as `missing`, `stale`, or `legacy`, but
  the UI must show that state.
- Feature availability is read from result and artifact validation, not from
  workspace object presence.

## Project Manifest

`project_manifest.json` groups datasets, runs, results, rules, views, and
exports under a research question or review topic.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_workspace_project_manifest_v1` |
| `project_id` | string | stable project id |
| `title` | string | project title |
| `description` | string | short description |
| `status` | string | `active`, `archived`, `draft`, or `blocked` |
| `tags` | array | user or system tags |
| `object_refs` | object | dataset, run, result, rule, view, and export ids |
| `created_at_utc` | string | creation timestamp |
| `updated_at_utc` | string | last update timestamp |

Optional fields:

- `owner`;
- `notes`;
- `default_result_id`;
- `default_view_id`;
- `quality_state`;
- `warnings`.

Rules:

- Project ids are stable local keys, not display labels.
- Archiving a project hides it from the default Home list but does not delete
  datasets, runs, results, or exports.
- A project may reference external result roots only if those paths are marked
  external and validation state is visible.

## Dataset Manifest

`dataset_manifest.json` describes imported or fetched records before analysis.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_workspace_dataset_manifest_v1` |
| `dataset_id` | string | stable dataset id |
| `source_type` | string | `openalex_query`, `wos_file`, `scopus_file`, `bibtex_file`, `local_parquet`, or future type |
| `records_path` | string | path to normalized records |
| `record_schema` | string or null | expected record schema |
| `record_count` | int | normalized record count |
| `source_refs` | array | source files, query refs, or API params |
| `normalization` | object | adapter and normalization summary |
| `created_at_utc` | string | creation timestamp |

Optional fields:

- `title`;
- `year_min`, `year_max`;
- `field_summary`;
- `checksum`;
- `privacy_state`;
- `warnings`.

Rules:

- Normalized records must preserve stable `uid` values.
- Source files should be referenced by relative path when inside the workspace.
- Queries should record query string, filters, max records, API source, and
  retrieval timestamp.
- Raw source data may be absent, but absence must be visible.

## Run Manifest

`run_manifest.json` describes a pipeline execution or recoverable job.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_workspace_run_manifest_v1` |
| `run_id` | string | stable run id |
| `project_id` | string or null | parent project |
| `mode` | string | `live_query`, `file_pipeline`, `landscape`, `keywords`, `validation`, etc. |
| `state` | string | `queued`, `running`, `paused`, `complete`, `failed`, `cancelled`, `partial`, or `blocked` |
| `config_ref` | string or object | replayable config or config path |
| `input_refs` | array | dataset, result, rule, or artifact refs used as input |
| `output_refs` | array | result, artifact, export, or partial-output refs |
| `progress` | object | current progress, heartbeat, shards, checkpoints |
| `started_at_utc` | string or null | start timestamp |
| `finished_at_utc` | string or null | finish timestamp |

Optional fields:

- `command`;
- `environment`;
- `logs`;
- `error`;
- `resume`;
- `resource_usage`;
- `warnings`.

Rules:

- Long-running jobs must expose heartbeat and partial-output state.
- A failed or cancelled run may still register recoverable partial artifacts.
- Completed runs should point to a result manifest or explain why no result was
  produced.
- Runtime configs should be replayable or explicitly marked as non-replayable.

## Rule Set Manifest

`rule_set_manifest.json` describes reusable cleaning, filter, alias,
thesaurus, abbreviation, view, or export rules.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_workspace_rule_set_manifest_v1` |
| `rule_set_id` | string | stable rule-set id |
| `rule_type` | string | `keyword_cleaning`, `entity_alias`, `thesaurus`, `filter`, `view`, `export`, or future type |
| `rules_path` | string | rule file path |
| `source` | string | `system`, `user`, `imported`, or `generated` |
| `version` | string | rule-set version |
| `created_at_utc` | string | creation timestamp |

Optional fields:

- `description`;
- `applies_to`;
- `parent_rule_set_id`;
- `impact_summary_ref`;
- `qa_ref`;
- `warnings`.

Rules:

- User rule edits create new versions or review decisions; they do not silently
  mutate source data.
- Rule sets used by results should be referenced by result, matrix, temporal,
  narrative, or export manifests.

## View Manifest

`view_manifest.json` stores saved UI/lens state that can be reopened.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_workspace_view_manifest_v1` |
| `view_id` | string | stable view id |
| `result_id` | string | result being viewed |
| `lens` | string | `atlas`, `matrix`, `temporal`, `evolution`, `narrative`, `quality`, or `export` |
| `state` | object | selected nodes, filters, layout, thresholds, pins, or tabs |
| `created_at_utc` | string | creation timestamp |
| `updated_at_utc` | string | last update timestamp |

Optional fields:

- `project_id`;
- `title`;
- `description`;
- `share_state`;
- `warnings`.

Rules:

- View manifests are display state, not analysis truth.
- A saved view must not enable a lens that the current result feature state
  hides or blocks.
- Browser-local pins can remain local and do not need to be written unless the
  user saves a view.

## Workspace QA Contract

`workspace_qa.json` should summarize registry validity.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_workspace_qa_v1` |
| `workspace_id` | parent workspace id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | named checks with status and counts |
| `counts` | projects, datasets, runs, results, rules, views, exports, missing refs |
| `warnings` | non-blocking warnings |
| `blocking_issues` | release-blocking issues |

Minimum checks:

- workspace schema is supported;
- registered object manifests exist or are explicitly marked missing/external;
- object ids are unique within each object family;
- relative paths resolve;
- default project/result/view refs resolve;
- recent refs resolve or are marked stale;
- result refs either validate or expose warnings;
- active runs have heartbeat, status, or recoverable failure metadata;
- public/shareable workspace manifests do not include private absolute paths;
- archived objects are not selected as defaults unless explicitly requested.

## Workspace States

Workspace validation should feed the app shell:

| Condition | Result |
| --- | --- |
| workspace manifest and object refs validate | `workspace=stable` |
| workspace opens with stale, missing, external, or legacy refs | `workspace=beta` |
| workspace manifest is malformed or defaults cannot resolve | `workspace=blocked` |
| no workspace exists, but local result roots can be discovered | `workspace=inferred` |
| no workspace and no local results exist | `workspace=hidden` |

This state is separate from result feature states. A workspace can be stable
while a selected result is beta or blocked.

## Discovery And Legacy Registration

SciScape should support a low-friction path for existing outputs:

1. scan configured output roots such as `workspace/output`,
   `workspace/examples_output`, `workspace/web_output`, and `viewer`;
2. identify result roots, report directories, and `data.json` files;
3. validate each candidate with the artifact contract;
4. register valid or partially valid candidates as legacy results;
5. write warnings for stale or missing expected artifacts;
6. let the user promote a legacy result into a project.

Legacy registration must not move data by default.

## Writer Utility Design

The first writer utility should be narrow:

```python
write_workspace_manifest(
    workspace_root,
    *,
    workspace_id,
    name,
    projects=None,
    datasets=None,
    runs=None,
    results=None,
    rule_sets=None,
    views=None,
    exports=None,
    defaults=None,
    settings=None,
)
```

The writer should:

1. validate object ids and relative paths;
2. write or update `workspace.json`;
3. write `workspace_qa.json`;
4. avoid moving existing result roots unless explicitly requested;
5. return paths, counts, warnings, and QA status.

The first registration helper should be:

```python
register_result_in_workspace(workspace_root, result_root, *, project_id=None)
```

It should validate the result root, add a result ref, and update recent refs.

## Validator Utility Design

The validator should be reusable outside the web app:

```python
validate_workspace(workspace_root) -> WorkspaceValidationResult
```

It should return:

- schema version;
- workspace id and state;
- object counts;
- missing/stale/external refs;
- default object resolution;
- recent object resolution;
- result validation summaries;
- warnings and blocking issues.

The web app can use this result to render Home without scanning arbitrary
folders on every load.

## Implementation Order

1. Add schema constants and dataclasses for workspace, project, dataset, run,
   rule-set, view, and workspace QA manifests.
2. Add `validate_workspace`.
3. Add `write_workspace_manifest`.
4. Add `register_result_in_workspace` for existing local result roots.
5. Update local web result discovery to prefer `workspace.json` when present,
   with legacy scan fallback.
6. Add a workspace smoke fixture with one project, one legacy result, one
   completed run, one stale ref, and one export ref.
7. Only then redesign Home around workspace objects rather than folders.

## Acceptance Criteria

- A workspace can be validated without loading the web app.
- Users can find recent results without knowing internal folder names.
- Legacy result roots can be registered without moving files.
- Active and failed runs expose progress, heartbeat, partial outputs, and
  recovery state when available.
- Workspace entries do not override artifact validation or feature states.
- Public/shareable manifests avoid private absolute paths.
- The web app can distinguish `no_workspace`, `workspace_loaded`,
  `result_loaded`, `run_active`, and `publish_ready` from manifests alone.

## Open Questions

- Should the default workspace live at repo-local `workspace/workspace.json` or
  user-home configuration?
- Should projects be optional for v1, with results allowed directly under the
  workspace root?
- Should external result roots be allowed in public manifests, or only in local
  private workspaces?
- Should archived projects remain in the same manifest or move to a separate
  archive index?
- Should workspace view manifests be saved automatically or only on explicit
  user action?
