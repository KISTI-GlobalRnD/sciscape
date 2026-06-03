# Temporal Trend Artifact Design

This document defines the temporal trend artifact contract that should come
before cluster evolution maps and narrative claims about change.

The purpose is to make publication activity, keyword trends, cluster activity,
term-family trends, growth signals, and burst-like summaries replayable and
validated from files rather than inferred only from `pubyear` at display time.

## Goal

SciScape should treat temporal analysis as an artifact-backed lens.

A temporal artifact must answer:

- which periods were analyzed;
- what entity each time series describes;
- what each metric means and how it was normalized;
- which source artifacts and rule sets were used;
- whether missing years, sparse periods, or weak denominators make the result
  unsafe to compare;
- which trend or burst events are supported by reproducible rows.

## Non-Goals

- Do not call yearly keyword charts "cluster evolution".
- Do not infer lineage, split, merge, emergence, decline, or stability events
  from this contract alone.
- Do not require a UI redesign in this milestone.
- Do not require advanced CiteSpace or SciMAT-style burst/thematic evolution
  methods before simple temporal trend artifacts are stable.
- Do not generate narrative claims about field change without evidence-backed
  narrative artifacts.

## Temporal Versus Evolution

The temporal artifact describes activity and trend signals within fixed
entities and periods.

The evolution artifact, defined separately, describes how cluster identities
change across slices. Evolution requires lineage evidence, transition rows, and
split/merge logic. Temporal charts may support an evolution view, but they do
not prove evolution by themselves.

| Lens | Question | Required artifact |
| --- | --- | --- |
| `temporal` | How did documents, terms, clusters, or term families vary by period? | `temporal/temporal_manifest.json` plus temporal tables |
| `evolution` | How did clusters continue, split, merge, emerge, or decline? | evolution lineage and transition artifacts |
| `narrative` | What evidence-backed story can be told about change? | narrative claims with evidence refs |

## Canonical Directory Shape

Temporal artifacts should live under:

```text
<result_root>/temporal/
  temporal_manifest.json
  periods.parquet
  activity.parquet
  entity_series.parquet
  temporal_events.parquet
  temporal_qa.json
```

For a landscape-scoped result, the same directory may live under:

```text
<result_root>/landscape/temporal/
```

Writers should prefer the result-root `temporal/` directory for reusable
workbench outputs and the landscape-local directory for lens-specific outputs.

`temporal_events.parquet` is optional for a minimal trend artifact. If the
manifest advertises burst, growth, or decline events, the file is required.

## Schema Versions

Use explicit schema names:

- `sciscape_temporal_manifest_v1`
- `sciscape_temporal_periods_v1`
- `sciscape_temporal_activity_v1`
- `sciscape_temporal_entity_series_v1`
- `sciscape_temporal_events_v1`
- `sciscape_temporal_qa_v1`

## Temporal Manifest

`temporal_manifest.json` is the source of truth for a temporal artifact.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_temporal_manifest_v1` |
| `temporal_id` | string | stable local identifier |
| `title` | string | human-readable title |
| `result_id` | string or null | parent result when available |
| `periodization` | object | unit, window, step, min/max year, and inclusion rule |
| `entity_types` | array | supported entities such as `result`, `cluster`, `term`, `term_family` |
| `metrics` | array | metric definitions and comparability rules |
| `event_types` | array | advertised event types, if any |
| `source_artifacts` | array | input artifact refs and roles |
| `rule_sets` | array | cleaning, thesaurus, abbreviation, and filter rules used |
| `transforms` | array | ordered transform steps |
| `outputs` | object | paths to periods, activity, series, events, and QA |
| `created_at_utc` | string | creation timestamp |
| `warnings` | array | non-blocking caveats |

Example:

```json
{
  "schema_version": "sciscape_temporal_manifest_v1",
  "temporal_id": "yearly_cluster_keyword_trends",
  "title": "Yearly cluster and keyword trends",
  "result_id": "openalex_gnn_20260603",
  "periodization": {
    "unit": "year",
    "window_years": 1,
    "step_years": 1,
    "start_year": 2014,
    "end_year": 2025,
    "closed": "left",
    "include_unknown_year": false
  },
  "entity_types": ["result", "cluster", "term", "term_family"],
  "metrics": [
    {
      "name": "doc_count",
      "value_type": "integer",
      "denominator": null,
      "normalization": "none",
      "interpretation": "documents assigned to the entity during the period"
    },
    {
      "name": "ppm",
      "value_type": "float",
      "denominator": "cluster_year_token_count",
      "normalization": "per_million_tokens",
      "interpretation": "term frequency per million cluster tokens in the period"
    }
  ],
  "event_types": ["growth", "decline", "burst"],
  "source_artifacts": [
    {"role": "records", "path": "abstracts.parquet"},
    {"role": "membership", "path": "landscape/membership.parquet"},
    {"role": "keywords", "path": "landscape/keywords.parquet"}
  ],
  "rule_sets": [],
  "transforms": [
    {"step": "parse_publication_years"},
    {"step": "build_periods"},
    {"step": "aggregate_activity"},
    {"step": "aggregate_entity_series"},
    {"step": "detect_temporal_events", "methods": ["growth_rate"]}
  ],
  "outputs": {
    "periods": "periods.parquet",
    "activity": "activity.parquet",
    "series": "entity_series.parquet",
    "events": "temporal_events.parquet",
    "qa": "temporal_qa.json"
  },
  "created_at_utc": "2026-06-03T00:00:00+00:00",
  "warnings": []
}
```

## Periods Table

`periods.parquet` defines the timeline used by all temporal rows.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_temporal_periods_v1` |
| `temporal_id` | string | parent temporal id |
| `period_id` | string | stable period key, such as `year:2024` |
| `period_index` | int | zero-based ordering |
| `period_label` | string | display label |
| `start_year` | int | inclusive start year |
| `end_year` | int | exclusive end year for windows, same as start for point years |
| `unit` | string | `year`, `rolling_window`, or future unit |

Rules:

- `period_id` must be unique.
- `period_index` must be contiguous unless the manifest declares an external
  fixed period index.
- Period ordering is the source of truth for trend calculations.

## Activity Table

`activity.parquet` stores result-level and period-level activity.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_temporal_activity_v1` |
| `temporal_id` | string | parent temporal id |
| `period_id` | string | key from `periods.parquet` |
| `start_year` | int | period start year |
| `end_year` | int | period end year |
| `doc_count` | int | documents in the period |
| `edge_count` | int or null | edges available within the period |
| `active_cluster_count` | int or null | clusters with at least one document in the period |
| `unknown_year_count` | int | records excluded or grouped because year is missing |

Optional columns:

- `term_count`;
- `term_family_count`;
- `source_count`;
- `mean_docs_per_cluster`;
- `warning_flags`.

Rules:

- Activity rows must cover every period in `periods.parquet`.
- Counts must be non-negative.
- `unknown_year_count` should be reported even when unknown-year records are
  excluded from all period rows.

## Entity Series Table

`entity_series.parquet` stores long-format series rows for documents, clusters,
terms, and term families.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_temporal_entity_series_v1` |
| `temporal_id` | string | parent temporal id |
| `entity_type` | string | `result`, `cluster`, `term`, `term_family`, or future type |
| `entity_key` | string | stable entity key |
| `entity_label` | string | display label |
| `period_id` | string | key from `periods.parquet` |
| `metric` | string | metric name from manifest |
| `value` | float | final metric value |
| `raw_value` | float or null | pre-normalization value |
| `denominator` | float or null | period/entity denominator |
| `support_count` | int or null | records or events supporting the row |

Optional columns:

- `cluster_uid`, `cluster_id`, `parent_uid`, `level`;
- `term`, `term_family`, `normalization_key`;
- `rank_in_period`, `rank_in_entity`;
- `baseline_value`, `delta`, `growth_rate`, `z_score`;
- `evidence_ref`;
- `warning_flags`.

Rules:

- `(entity_type, entity_key, period_id, metric)` must be unique.
- `value`, `raw_value`, and `denominator` must be finite when present.
- A denominator of zero must be represented as null, not as a finite normalized
  value.
- Term and term-family rows should preserve the canonical cleaning rule or
  normalization key that produced the entity.
- Cluster rows are activity for the existing cluster assignment. They are not
  lineage claims.

## Temporal Events Table

`temporal_events.parquet` stores burst-like and trend events derived from the
series table.

Required columns when events are advertised:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_temporal_events_v1` |
| `temporal_id` | string | parent temporal id |
| `event_id` | string | stable local event id |
| `event_type` | string | `growth`, `decline`, `burst`, `peak`, or future type |
| `entity_type` | string | entity described by the event |
| `entity_key` | string | stable entity key |
| `entity_label` | string | display label |
| `start_period_id` | string | first period in event |
| `end_period_id` | string | last period in event |
| `metric` | string | source series metric |
| `score` | float | method-specific event strength |
| `method` | string | event detection method |
| `support_count` | int | records or period rows supporting the event |

Optional columns:

- `baseline_value`, `peak_value`, `final_value`;
- `growth_rate`, `duration_periods`;
- `cluster_uid`, `term`, `term_family`;
- `evidence_ref`;
- `warning_flags`.

Rules:

- Event rows must reference periods and entity-series rows that exist.
- Event methods must be declared in the manifest transforms.
- Burst-like events are signals, not narrative claims. UI text should say
  "burst signal" unless a stronger method contract is added.

## QA Contract

`temporal_qa.json` should summarize validation and comparability.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_temporal_qa_v1` |
| `temporal_id` | parent temporal id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | named checks with status and counts |
| `counts` | periods, activity rows, series rows, event rows, missing years |
| `warnings` | non-blocking warnings |
| `blocking_issues` | release-blocking issues |

Minimum checks:

- manifest schema is supported;
- periods table exists and has required columns;
- activity table exists and covers all periods;
- entity series table exists and has required columns;
- event table exists when events are advertised;
- period refs in activity, series, and events resolve;
- series metrics are declared in the manifest;
- numeric values are finite;
- duplicate series rows are absent;
- source artifact refs exist;
- missing-year and sparse-period counts are reported;
- event methods and threshold parameters are recorded.

## Validation States

Temporal validation should feed the normal result contract:

| Condition | Result |
| --- | --- |
| manifest, periods, activity, entity series, and QA all pass | `temporal=stable` |
| artifact exists but has non-blocking warnings | `temporal=beta` |
| artifact is advertised but missing required tables | result `blocked` |
| no temporal artifact exists, but `pubyear` is available | `temporal=beta` with reason `pubyear exists but no temporal artifact` |
| no temporal artifact or usable `pubyear` exists | `temporal=hidden` |

This preserves the current lightweight temporal feature inference while giving
future result roots a stable artifact-backed path.

## Writer Utility Design

The first writer utility should be narrow:

```python
write_temporal_artifacts(
    result_root,
    *,
    temporal_id,
    records_df,
    membership_df=None,
    keywords_df=None,
    edge_df=None,
    periodization=None,
    metrics=None,
    event_methods=None,
    source_artifacts=None,
    rule_sets=None,
)
```

The writer should:

1. validate input year columns before aggregation;
2. build `periods.parquet`;
3. write `activity.parquet`;
4. write long-format `entity_series.parquet`;
5. write `temporal_events.parquet` only when event methods are requested;
6. generate `temporal_manifest.json`;
7. generate `temporal_qa.json`;
8. return paths, counts, warnings, and QA status.

The first implementation should support yearly periods. Rolling windows can be
added after yearly validation is stable.

## Validator Utility Design

The validator should be reusable outside full result validation:

```python
validate_temporal_artifact(temporal_dir) -> TemporalArtifactValidationResult
```

It should return:

- schema version;
- temporal id;
- status;
- artifact paths;
- period and row counts;
- event counts by type;
- missing-year and sparse-period diagnostics;
- warnings and blocking issues;
- feature exposure suggestion.

Full result validation can then expose temporal states without recomputing
series from raw records.

## Implementation Order

1. Add schema constants and dataclasses for temporal manifests, tables, events,
   and QA.
2. Add `write_temporal_artifacts` for yearly result, cluster, term, and
   term-family trends.
3. Add `validate_temporal_artifact`.
4. Extend `validate_result_root` to identify temporal manifests and expose
   stable/beta/blocked temporal states.
5. Add a synthetic temporal quality gate that writes and validates a tiny
   yearly trend artifact.
6. Add optional burst/growth event rows using existing temporal and burst
   helpers.
7. Only then build temporal controls or connect temporal signals to the Atlas
   evidence inspector.

Initial implementation note:

- `write_temporal_artifacts` and `validate_temporal_artifact` are available in
  `sciscape.artifacts`.
- The current writer supports yearly periods, result-level document activity,
  cluster document activity, keyword year-series rows when temporal keyword
  columns are present, and optional `growth_rate` signal rows.
- Validation checks required columns, period refs, declared metrics, finite
  numeric values, duplicate series rows, event refs, source artifact refs,
  missing-year counts, and the QA sidecar.
- `validate_result_root` identifies `temporal/temporal_manifest.json` and uses
  artifact-backed rows for stable `temporal` exposure. Pubyear-only results
  remain available as beta temporal views.
- This contract still does not imply cluster lineage, split, merge, or
  evolution claims; those remain under `evolution_artifact_design.md`.

## Acceptance Criteria

- A temporal artifact can be validated without loading the web app.
- Temporal rows record period definitions, metric definitions, normalization,
  denominators, and source artifacts.
- `pubyear`-only results remain usable as beta temporal views, but stable
  temporal views require artifact-backed rows.
- Term-family and abbreviation-aware rows can be linked back to replayable
  cleaning rules.
- Burst and growth signals are labeled as signals and include method metadata.
- Temporal artifacts do not imply cluster lineage, split, merge, or evolution
  claims.
- The full result contract can tell whether `temporal` is hidden, beta, stable,
  or blocked from artifacts alone.

## Open Questions

- Should sparse periods be warnings by default or configurable blockers for
  release bundles?
- Should rolling-window periods be represented as `start_year`/`end_year` only,
  or should they include exact date boundaries when source data supports dates?
- Should term-family rows require a separate thesaurus artifact before they can
  be stable?
- Should burst methods beyond simple growth-rate signals receive separate
  schema versions?
