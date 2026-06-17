# Matrix Artifact Design

This document defines the general matrix artifact contract that should follow
the stable P1.5 term co-occurrence artifacts.

The purpose is to make Matrix Builder outputs replayable, inspectable, and
validated before any generic matrix UI is exposed as a product surface.

## Goal

SciScape should treat matrices as first-class analysis artifacts, not as hidden
intermediate arrays.

A matrix artifact must answer:

- what entities define the rows and columns;
- what value each cell means;
- how the cell values were produced, normalized, thresholded, and filtered;
- which source artifacts and rule sets were used;
- whether the matrix is safe to inspect, export, or compare.

## Non-Goals

- Do not expose a generic Matrix Builder UI in this milestone.
- Do not replace `term_cooccurrence.parquet` and
  `term_cooccurrence_map.json`; those remain the stable P1.5 term-specific
  contract.
- Do not require dense matrix materialization.
- Do not merge every network or clustering artifact into this schema.
- Do not use matrix artifacts to imply causal relationships or narrative claims.

## Matrix Families

The contract should support these matrix families:

| Family | Example | Shape | Primary use |
| --- | --- | --- | --- |
| `occurrence` | paper x term, cluster x term, source x field | rectangular | feature counts and analyst tables |
| `cooccurrence` | term x term, author x author, organization x organization | usually square | term maps and relationship evidence |
| `proximity` | paper x paper, cluster x cluster | square | bibliographic coupling, co-citation, citation proximity |
| `similarity` | term x term, cluster x cluster, paper x paper | square | normalized comparison and map layouts |
| `projection` | projected author network from paper-author bipartite table | square | derived entity networks |
| `temporal` | term x year, cluster x period | rectangular | trend and evolution views |

The first implementation should support sparse edge-list/triplet output. Dense
JSON arrays should be allowed only for tiny QA examples.

## Canonical Directory Shape

General matrix artifacts should live under:

```text
<result_root>/matrices/<matrix_id>/
  matrix_manifest.json
  matrix_values.parquet
  row_entities.parquet
  column_entities.parquet
  matrix_qa.json
```

For a landscape-scoped result, the same directory may live under:

```text
<result_root>/landscape/matrices/<matrix_id>/
```

Writers should prefer the result-root `matrices/` directory for reusable
workbench outputs and the landscape-local directory for lens-specific outputs.

## Schema Versions

Use explicit schema names:

- `sciscape_matrix_manifest_v1`
- `sciscape_matrix_values_sparse_triplet_v1`
- `sciscape_matrix_entities_v1`
- `sciscape_matrix_qa_v1`

The existing `sciscape_cooccurrence_artifact_v1` remains a specialized term
co-occurrence schema. A future adapter may wrap it as a `cooccurrence` matrix,
but it should not be retroactively redefined.

## Matrix Manifest

`matrix_manifest.json` is the source of truth for a matrix artifact.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_matrix_manifest_v1` |
| `matrix_id` | string | stable local identifier |
| `title` | string | human-readable title |
| `matrix_family` | string | one of the supported families |
| `format` | string | `sparse_triplet` for v1 |
| `result_id` | string or null | parent result when available |
| `row_entity_type` | string | `work`, `cluster`, `term`, `author`, `organization`, `source`, `year`, etc. |
| `column_entity_type` | string | same vocabulary as row entity type |
| `shape` | object | `rows`, `columns`, `nnz` |
| `value` | object | value name, type, range, and interpretation |
| `weighting` | object | raw metric, normalization, threshold, top-k, direction |
| `source_artifacts` | array | input artifact refs and roles |
| `rule_sets` | array | cleaning/thesaurus/filter rules used |
| `transforms` | array | ordered transform steps |
| `outputs` | object | paths to values, row entities, column entities, QA |
| `created_at_utc` | string | creation timestamp |
| `warnings` | array | non-blocking caveats |

Example:

```json
{
  "schema_version": "sciscape_matrix_manifest_v1",
  "matrix_id": "term_cooccurrence_default",
  "title": "Term co-occurrence",
  "matrix_family": "cooccurrence",
  "format": "sparse_triplet",
  "result_id": "openalex_gnn_20260603",
  "row_entity_type": "term",
  "column_entity_type": "term",
  "shape": {"rows": 128, "columns": 128, "nnz": 1240},
  "value": {
    "name": "cooccurrence_weight",
    "type": "float",
    "range": [0.0, 1.0],
    "interpretation": "normalized within-cluster term co-occurrence strength"
  },
  "weighting": {
    "raw_metric": "within_cluster_keyword_pair",
    "normalization": "max",
    "threshold": 0.0,
    "top_k_per_group": 10,
    "symmetric": true
  },
  "source_artifacts": [
    {"role": "keywords", "path": "landscape/keywords.parquet"},
    {"role": "cooccurrence", "path": "landscape/term_cooccurrence.parquet"}
  ],
  "rule_sets": [],
  "transforms": [
    {"step": "load_keywords"},
    {"step": "select_top_terms", "top_k": 10},
    {"step": "build_sparse_triplets"}
  ],
  "outputs": {
    "values": "matrix_values.parquet",
    "rows": "row_entities.parquet",
    "columns": "column_entities.parquet",
    "qa": "matrix_qa.json"
  },
  "created_at_utc": "2026-06-03T00:00:00+00:00",
  "warnings": []
}
```

## Sparse Values Table

`matrix_values.parquet` stores non-zero or retained cells.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_matrix_values_sparse_triplet_v1` |
| `matrix_id` | string | parent matrix id |
| `row_key` | string | stable row entity key |
| `column_key` | string | stable column entity key |
| `row_index` | int | zero-based row index |
| `column_index` | int | zero-based column index |
| `value` | float | normalized or final matrix value |
| `raw_value` | float or null | raw pre-normalization value |
| `support_count` | int or null | count of records/events supporting the cell |
| `rank` | int or null | optional row-local rank |
| `relation` | string | metric or relationship label |

Optional columns:

- `row_group`, `column_group`;
- `period`;
- `source_artifact`;
- `evidence_ref`;
- `warning_flags`.

Rules:

- `row_key` and `column_key` must exist in the entity tables.
- `value` must be numeric.
- Duplicate `(row_key, column_key, period)` rows are invalid unless the manifest
  declares a multi-layer matrix.
- Symmetric matrices must either store only upper-triangle rows with
  `storage="upper_triangle"` or store both directions with matching values.

## Entity Tables

`row_entities.parquet` and `column_entities.parquet` identify the matrix axes.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_matrix_entities_v1` |
| `matrix_id` | string | parent matrix id |
| `entity_key` | string | stable key used by `matrix_values.parquet` |
| `entity_index` | int | zero-based index |
| `entity_type` | string | row or column entity type |
| `label` | string | display label |

Optional columns:

- `cluster_uid`, `uid`, `term`, `year`, `source`;
- `doc_count`, `frequency`, `score`;
- `parent_uid`, `level`;
- `normalization_key`;
- `qa_flags`.

Rules:

- `entity_key` must be unique within each entity table.
- `entity_index` must be unique and contiguous unless the manifest declares a
  fixed external index.
- Entity labels are display metadata, not join keys.

## QA Contract

`matrix_qa.json` should summarize validation and comparability.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_matrix_qa_v1` |
| `matrix_id` | parent matrix id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | named checks with status and counts |
| `counts` | rows, columns, nnz, duplicate cells, missing refs |
| `warnings` | non-blocking warnings |
| `blocking_issues` | release-blocking issues |

Minimum checks:

- manifest schema is supported;
- values table exists and has required columns;
- row and column entity tables exist and have required columns;
- all value row/column keys resolve;
- numeric values are finite;
- shape in manifest matches entity and values counts;
- symmetry rule is respected when declared;
- source artifact refs exist;
- threshold/top-k settings are recorded.

## Validation States

Matrix validation should feed the normal result contract:

| Condition | Result |
| --- | --- |
| manifest, values, row entities, column entities, and QA all pass | `matrix=stable` |
| artifact exists but has non-blocking warnings | `matrix=beta` |
| matrix artifact is advertised but missing required tables | result `blocked` |
| no matrix artifact exists | `matrix=hidden` unless co-occurrence/report edges support a narrower lens |

This does not change the P1.5 `cooccurrence` exposure. A stable co-occurrence
artifact can keep `cooccurrence=stable` even when the generic `matrix` workbench
is not yet implemented.

## Writer Utility Design

The first writer utility should be narrow:

```python
write_matrix_artifact(
    result_root,
    matrix_id,
    matrix_family,
    values_df,
    row_entities_df,
    column_entities_df,
    *,
    value_spec,
    weighting,
    source_artifacts,
    rule_sets=None,
    transforms=None,
)
```

The writer should:

1. validate incoming DataFrames before writing;
2. write all files atomically where practical;
3. generate `matrix_manifest.json`;
4. generate `matrix_qa.json`;
5. return paths and QA status;
6. avoid dense conversion.

The first adapter should be:

```python
write_matrix_from_term_cooccurrence(result_root)
```

It should wrap `landscape/term_cooccurrence.parquet` into a general
`cooccurrence` matrix artifact without changing the existing P1.5 files.

## Validator Utility Design

The validator should be reusable outside full result validation:

```python
validate_matrix_artifact(matrix_dir) -> MatrixArtifactValidationResult
```

It should return:

- schema version;
- matrix id and family;
- status;
- artifact paths;
- counts;
- warnings and blocking issues;
- feature exposure suggestion.

Full result validation can then include matrix summaries without parsing every
large cell table in detail when only a quick artifact contract is needed.

## Export Contract

Matrix exports should be explicit artifacts, not ad hoc downloads.

Supported v1 export targets:

- `csv_triplets`: row, column, value rows;
- `parquet_triplets`: canonical values table;
- `vosviewer_network`: only when entity type and symmetry are compatible;
- `json_summary`: manifest plus QA summary.

Exports should reference the source `matrix_id` and include the same
normalization and threshold metadata.

## Implementation Order

1. Add schema constants and dataclasses for matrix manifests, values, entities,
   and QA.
2. Add `write_matrix_artifact` and `validate_matrix_artifact`.
3. Add a term-co-occurrence adapter that wraps the P1.5 artifacts into
   `matrices/term_cooccurrence_default/`.
4. Extend `validate_result_root` to identify general matrix manifests and expose
   stable/beta/blocked matrix states.
5. Add a synthetic matrix quality gate that writes and validates a tiny matrix.
6. Only then add a Matrix Builder mode or UI panel.

Initial implementation note:

- `write_matrix_artifact`, `validate_matrix_artifact`, and
  `write_matrix_from_term_cooccurrence` are available in `sciscape.artifacts`.
- CLI `sciscape matrix wrap-term-cooccurrence <result_root>` materializes the
  term co-occurrence wrapper under `matrices/term_cooccurrence_default/` by
  default.
- The current writer supports sparse-triplet Parquet values, row/column entity
  tables, `matrix_manifest.json`, and `matrix_qa.json`.
- Validation checks required columns, entity refs, finite numeric values,
  duplicate cells, manifest shape, declared symmetry, and source artifact refs.
- `validate_result_root` identifies `matrices/*/matrix_manifest.json` and uses
  that manifest-backed artifact for stable general `matrix` exposure. Existing
  P1.5 term co-occurrence artifacts remain separate and continue to expose the
  narrower `cooccurrence` feature.
- Matrix Builder UI, richer matrix commands/exports, and partitioned
  large-matrix output remain future work.

## Acceptance Criteria

- A matrix artifact can be validated without loading the web app.
- A matrix artifact records row and column entity metadata.
- A matrix artifact records weighting, normalization, threshold, and source
  artifacts.
- A malformed matrix blocks only when it is advertised or selected as a required
  artifact.
- The existing P1.5 co-occurrence artifacts remain valid and can be wrapped into
  the general matrix contract.
- The full result contract can tell whether `matrix` is hidden, beta, stable, or
  blocked from artifacts alone.

## Open Questions

- Should large matrix values support partitioned Parquet under
  `matrix_values/` before the first UI mode?
- Should checksums be required for every matrix file or only release bundles?
- Should `matrix_family=network` be added, or should networks remain separate
  edge artifacts?
- Should row and column entity tables support external IDs as mandatory fields
  for VOSviewer and KnowledgeMatrix interoperability?
