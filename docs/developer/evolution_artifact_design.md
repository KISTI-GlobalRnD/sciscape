# Cluster Evolution Artifact Design

This document defines the cluster evolution artifact contract that should come
after temporal trend artifacts and before any cluster evolution map UI.

The purpose is to make cluster continuity, split, merge, emergence, decline,
and stability claims replayable and validated from transition evidence rather
than inferred from yearly charts or labels alone.

## Goal

SciScape should treat cluster evolution as a lineage-backed analysis artifact.

An evolution artifact must answer:

- which time slices were compared;
- which cluster states exist in each slice;
- how cluster states were matched across slices;
- what evidence supports each transition;
- whether a transition is a continuation, split, merge, emergence, decline, or
  weak/ambiguous match;
- how stable each lineage is across time;
- whether the result is safe to inspect as a cluster evolution map.

## Non-Goals

- Do not infer evolution from temporal trend rows alone.
- Do not call a yearly keyword chart or activity chart a cluster evolution map.
- Do not require a UI implementation in this milestone.
- Do not require reclustering from scratch for every time slice in the first
  implementation. A membership-slice projection can be the first supported
  method if its limitations are recorded.
- Do not generate narrative claims about why a cluster changed without
  narrative evidence references.

## Temporal Versus Evolution

Temporal artifacts describe activity and trend signals for fixed entities.
Evolution artifacts describe identity links between cluster states across time
slices.

| Lens | Required claim | Evidence type |
| --- | --- | --- |
| `temporal` | Entity activity changed over periods | period rows and metric rows |
| `evolution` | Cluster identity continued, split, merged, emerged, or declined | slice states, transition rows, lineage rows, event QA |
| `narrative` | A human-readable interpretation of change | evidence-backed narrative claims |

Temporal signals may decorate an evolution map, but they are not sufficient to
enable the evolution lens.

## Canonical Directory Shape

Evolution artifacts should live under:

```text
<result_root>/evolution/
  evolution_manifest.json
  time_slices.parquet
  cluster_states.parquet
  transitions.parquet
  lineages.parquet
  evolution_events.parquet
  evolution_qa.json
  synthetic_smoke_example.json
```

For a landscape-scoped result, the same directory may live under:

```text
<result_root>/landscape/evolution/
```

Writers should prefer the result-root `evolution/` directory for reusable
workbench outputs and the landscape-local directory for lens-specific outputs.

`synthetic_smoke_example.json` is not required for user outputs. It is a small
fixture contract for implementation and release gates.

## Schema Versions

Use explicit schema names:

- `sciscape_evolution_manifest_v1`
- `sciscape_evolution_time_slices_v1`
- `sciscape_evolution_cluster_states_v1`
- `sciscape_evolution_transitions_v1`
- `sciscape_evolution_lineages_v1`
- `sciscape_evolution_events_v1`
- `sciscape_evolution_qa_v1`
- `sciscape_evolution_synthetic_smoke_v1`

## Evolution Manifest

`evolution_manifest.json` is the source of truth for an evolution artifact.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_evolution_manifest_v1` |
| `evolution_id` | string | stable local identifier |
| `title` | string | human-readable title |
| `result_id` | string or null | parent result when available |
| `slice_method` | object | how time slices and cluster states were produced |
| `matching_method` | object | transition metric, thresholds, and tie policy |
| `event_rules` | object | split, merge, emergence, decline, and continuation rules |
| `entity_scope` | object | levels, cluster universe, document universe, and filters |
| `metrics` | array | transition and stability metric definitions |
| `source_artifacts` | array | input artifact refs and roles |
| `rule_sets` | array | cleaning, filtering, and label rules used |
| `transforms` | array | ordered transform steps |
| `outputs` | object | paths to slices, states, transitions, lineages, events, and QA |
| `created_at_utc` | string | creation timestamp |
| `warnings` | array | non-blocking caveats |

Example:

```json
{
  "schema_version": "sciscape_evolution_manifest_v1",
  "evolution_id": "yearly_cluster_evolution_default",
  "title": "Yearly cluster evolution",
  "result_id": "openalex_gnn_20260603",
  "slice_method": {
    "unit": "year",
    "window_years": 1,
    "step_years": 1,
    "start_year": 2020,
    "end_year": 2024,
    "state_method": "membership_projection",
    "include_unknown_year": false
  },
  "matching_method": {
    "metric": "jaccard_doc_overlap",
    "min_transition_score": 0.2,
    "min_support_count": 3,
    "tie_policy": "keep_all_above_threshold",
    "normalization": "source_and_target_overlap"
  },
  "event_rules": {
    "continuation_min_score": 0.5,
    "split_min_children": 2,
    "merge_min_parents": 2,
    "emergence_max_incoming_score": 0.2,
    "decline_max_outgoing_score": 0.2,
    "ambiguous_score_margin": 0.05
  },
  "entity_scope": {
    "cluster_level": "default",
    "cluster_id_namespace": "slice_local",
    "document_universe": "records_with_valid_pubyear",
    "filter_refs": []
  },
  "metrics": [
    {
      "name": "transition_score",
      "value_type": "float",
      "range": [0.0, 1.0],
      "interpretation": "document-overlap continuity score between source and target states"
    },
    {
      "name": "lineage_stability",
      "value_type": "float",
      "range": [0.0, 1.0],
      "interpretation": "aggregate continuity strength across a lineage"
    }
  ],
  "source_artifacts": [
    {"role": "records", "path": "abstracts.parquet"},
    {"role": "membership", "path": "landscape/membership.parquet"},
    {"role": "temporal", "path": "temporal/temporal_manifest.json"}
  ],
  "rule_sets": [],
  "transforms": [
    {"step": "build_time_slices"},
    {"step": "derive_cluster_states"},
    {"step": "score_adjacent_slice_transitions"},
    {"step": "assign_evolution_events"},
    {"step": "build_lineages"}
  ],
  "outputs": {
    "time_slices": "time_slices.parquet",
    "cluster_states": "cluster_states.parquet",
    "transitions": "transitions.parquet",
    "lineages": "lineages.parquet",
    "events": "evolution_events.parquet",
    "qa": "evolution_qa.json",
    "synthetic_smoke": "synthetic_smoke_example.json"
  },
  "created_at_utc": "2026-06-03T00:00:00+00:00",
  "warnings": []
}
```

## Time Slices Table

`time_slices.parquet` defines the slice axis used by states and transitions.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_evolution_time_slices_v1` |
| `evolution_id` | string | parent evolution id |
| `slice_id` | string | stable slice key, such as `year:2022` |
| `slice_index` | int | zero-based ordering |
| `slice_label` | string | display label |
| `start_year` | int | inclusive start year |
| `end_year` | int | exclusive end year for windows, same as start for point years |
| `unit` | string | `year`, `rolling_window`, or future unit |
| `doc_count` | int | documents available in the slice |

Optional columns:

- `edge_count`;
- `active_cluster_count`;
- `unknown_year_count`;
- `warning_flags`.

Rules:

- `slice_id` must be unique.
- `slice_index` must be contiguous unless the manifest declares an external
  fixed slice index.
- Adjacent transitions are defined by `slice_index`, not by display label.

## Cluster States Table

`cluster_states.parquet` stores the state of each cluster in each slice.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_evolution_cluster_states_v1` |
| `evolution_id` | string | parent evolution id |
| `state_id` | string | stable slice-specific cluster state key |
| `slice_id` | string | key from `time_slices.parquet` |
| `slice_index` | int | copied ordering for fast reads |
| `cluster_key` | string | original or projected cluster key |
| `cluster_label` | string | display label |
| `doc_count` | int | documents assigned to this state in the slice |
| `term_count` | int or null | representative term count when available |
| `top_terms` | string or list | bounded representative terms |

Optional columns:

- `cluster_uid`, `cluster_id`, `parent_uid`, `level`;
- `centroid_x`, `centroid_y`;
- `representative_work_ids`;
- `activity_score`, `growth_score`;
- `source_cluster_key`;
- `warning_flags`.

Rules:

- `state_id` must be unique.
- Every `slice_id` must resolve to `time_slices.parquet`.
- `doc_count` must be positive for visible states.
- Cluster states describe slice-local identity. Cross-slice identity exists
  only through `transitions.parquet` and `lineages.parquet`.

## Transitions Table

`transitions.parquet` stores directed evidence rows between adjacent slice
cluster states.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_evolution_transitions_v1` |
| `evolution_id` | string | parent evolution id |
| `transition_id` | string | stable local transition id |
| `source_state_id` | string | state in the earlier slice |
| `target_state_id` | string | state in the later slice |
| `source_slice_id` | string | earlier slice key |
| `target_slice_id` | string | later slice key |
| `metric` | string | transition metric from the manifest |
| `score` | float | final transition score |
| `support_count` | int | shared records, edges, terms, or other support count |
| `source_doc_count` | int | source state document count |
| `target_doc_count` | int | target state document count |
| `relation` | string | `candidate`, `continuation`, `split_child`, `merge_parent`, or `ambiguous` |

Optional columns:

- `shared_doc_count`;
- `shared_term_count`;
- `jaccard`, `overlap_source`, `overlap_target`;
- `rank_from_source`, `rank_to_target`;
- `evidence_ref`;
- `warning_flags`.

Rules:

- Source and target states must exist.
- Source and target slices must be adjacent unless the manifest allows
  skip-slice transitions.
- `score` must be finite and within the range declared by the manifest metric.
- Duplicate `(source_state_id, target_state_id, metric)` rows are invalid.
- Transitions are evidence rows. Event labels are assigned in
  `evolution_events.parquet`.

## Lineages Table

`lineages.parquet` stores derived identity paths across slices.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_evolution_lineages_v1` |
| `evolution_id` | string | parent evolution id |
| `lineage_id` | string | stable lineage key |
| `state_id` | string | state belonging to the lineage |
| `slice_id` | string | state slice key |
| `slice_index` | int | state slice order |
| `role` | string | `root`, `continuation`, `split_branch`, `merge_branch`, `terminal`, or `singleton` |
| `stability_score` | float | lineage-local continuity score |

Optional columns:

- `parent_lineage_id`;
- `root_state_id`, `previous_state_id`, `next_state_id`;
- `lineage_label`;
- `branch_index`;
- `event_refs`;
- `warning_flags`.

Rules:

- Every `state_id` must resolve to `cluster_states.parquet`.
- A state may belong to multiple lineages only when the manifest declares
  multi-parent merge handling.
- `stability_score` must be finite and in `[0, 1]`.

## Evolution Events Table

`evolution_events.parquet` stores event labels derived from transition and
lineage evidence.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_evolution_events_v1` |
| `evolution_id` | string | parent evolution id |
| `event_id` | string | stable local event id |
| `event_type` | string | `continuation`, `split`, `merge`, `emergence`, `decline`, or `ambiguous` |
| `slice_id` | string | slice where the event is observed |
| `state_id` | string | primary state for the event |
| `lineage_id` | string or null | related lineage when assigned |
| `transition_refs` | string or list | supporting transition IDs |
| `score` | float | event strength or confidence score |
| `support_count` | int | records or transition rows supporting the event |
| `method` | string | event assignment method |

Optional columns:

- `source_state_ids`;
- `target_state_ids`;
- `source_lineage_ids`;
- `target_lineage_ids`;
- `event_label`;
- `evidence_ref`;
- `warning_flags`.

Rules:

- `event_type` must be one of the manifest-supported event rules.
- Split events must have at least two target states or branch lineages.
- Merge events must have at least two source states or parent lineages.
- Emergence events must have no incoming transition above the configured
  threshold.
- Decline events must have no outgoing transition above the configured
  threshold.
- Ambiguous events must expose the competing transition refs.

## QA Contract

`evolution_qa.json` should summarize validation and lineage reliability.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_evolution_qa_v1` |
| `evolution_id` | parent evolution id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | named checks with status and counts |
| `counts` | slices, states, transitions, lineages, events, missing refs |
| `event_counts` | events by type |
| `warnings` | non-blocking warnings |
| `blocking_issues` | release-blocking issues |

Minimum checks:

- manifest schema is supported;
- time slices table exists and has required columns;
- cluster states table exists and has required columns;
- transitions table exists and has required columns;
- lineages table exists and has required columns;
- evolution events table exists and has required columns;
- all state, slice, lineage, and transition refs resolve;
- transitions connect adjacent slices unless configured otherwise;
- numeric scores are finite and within metric ranges;
- split and merge events satisfy minimum parent/child counts;
- emergence and decline events respect threshold rules;
- ambiguous transitions are exposed rather than silently discarded;
- source artifact refs exist;
- method thresholds and tie policies are recorded.

## Validation States

Evolution validation should feed the normal result contract:

| Condition | Result |
| --- | --- |
| manifest, slices, states, transitions, lineages, events, and QA all pass | `evolution=stable` |
| artifact exists but has sparse slices, weak matches, or non-blocking warnings | `evolution=beta` |
| artifact is advertised but missing required tables or refs | result `blocked` |
| only temporal artifacts or `pubyear` exist | `evolution=hidden` |
| no evolution artifact exists | `evolution=hidden` |

This keeps temporal availability separate from cluster evolution availability.

## Synthetic Smoke Example

The first implementation should include a tiny deterministic smoke fixture that
exercises every required event type.

Recommended shape:

```text
Slices:
  year:2020
  year:2021
  year:2022

States:
  A20 -> A21 -> A22       continuation
  B20 -> B21a, B21b       split
  C20a, C20b -> C21       merge
  D21 -> D22              emergence followed by continuation
  E20                     decline
  X20 -> Y21, Z21         ambiguous near-tie
```

The fixture should encode the expected outputs as JSON:

```json
{
  "schema_version": "sciscape_evolution_synthetic_smoke_v1",
  "evolution_id": "synthetic_all_events",
  "time_slices": [
    {"slice_id": "year:2020", "slice_index": 0, "start_year": 2020, "end_year": 2020},
    {"slice_id": "year:2021", "slice_index": 1, "start_year": 2021, "end_year": 2021},
    {"slice_id": "year:2022", "slice_index": 2, "start_year": 2022, "end_year": 2022}
  ],
  "expected_event_counts": {
    "continuation": 3,
    "split": 1,
    "merge": 1,
    "emergence": 1,
    "decline": 1,
    "ambiguous": 1
  },
  "expected_blocking_issues": []
}
```

The fixture should be small enough to validate without clustering. It should
write cluster states and transitions directly, then run the same validator used
for real outputs.

## Writer Utility Design

The first writer utility should be narrow:

```python
write_evolution_artifacts(
    result_root,
    *,
    evolution_id,
    records_df,
    membership_df,
    temporal_manifest=None,
    periodization=None,
    matching_method=None,
    event_rules=None,
    source_artifacts=None,
    rule_sets=None,
)
```

The writer should:

1. validate records, years, and membership columns;
2. build `time_slices.parquet`;
3. derive `cluster_states.parquet`;
4. score adjacent-slice transitions;
5. assign evolution events from transition evidence;
6. build `lineages.parquet`;
7. generate `evolution_manifest.json`;
8. generate `evolution_qa.json`;
9. return paths, counts, warnings, and QA status.

The first implementation can use membership projection and document-overlap
matching. Reclustered time-slice states and topic/term matching can be added
after the v1 validator is stable.

## Validator Utility Design

The validator should be reusable outside full result validation:

```python
validate_evolution_artifact(evolution_dir) -> EvolutionArtifactValidationResult
```

It should return:

- schema version;
- evolution id;
- status;
- artifact paths;
- slice, state, transition, lineage, and event counts;
- event counts by type;
- sparse-slice and ambiguous-transition diagnostics;
- warnings and blocking issues;
- feature exposure suggestion.

Full result validation can then expose evolution states without recomputing
lineage from raw records.

## Implementation Order

1. Add schema constants and dataclasses for evolution manifests, tables, events,
   lineages, and QA.
2. Add `validate_evolution_artifact`.
3. Add the synthetic smoke fixture writer and validator test.
4. Add `write_evolution_artifacts` for membership-projection evolution.
5. Extend `validate_result_root` to identify evolution manifests and expose
   stable/beta/blocked evolution states.
6. Add a quality-gate flag that requires the synthetic evolution smoke to pass.
7. Only then build an evolution map UI or connect evolution signals to
   evidence-backed narratives.

## Acceptance Criteria

- An evolution artifact can be validated without loading the web app.
- Time slices, cluster states, transitions, lineages, events, and QA are stored
  as separate inspectable artifacts.
- Split, merge, emergence, decline, continuation, and ambiguous events have
  reproducible supporting transition rows.
- `temporal=stable` does not imply `evolution=stable`.
- Evolution UI can be enabled or disabled from feature states alone.
- Synthetic smoke covers at least one continuation, split, merge, emergence,
  decline, and ambiguous near-tie case.
- The full result contract can tell whether `evolution` is hidden, beta,
  stable, or blocked from artifacts alone.

## Initial Implementation Note

`sciscape.artifacts` now implements the v1 artifact contract:

- `write_evolution_artifacts` writes yearly membership-projection evolution
  artifacts under `<result_root>/evolution/`.
- `sciscape.evolution.build_membership_projection_evolution` builds the
  in-memory slice, state, transition, lineage, and event tables before artifact
  serialization; this keeps analysis logic separate from artifact validation and
  lets richer matching strategies evolve in the analysis module first.
- `validate_evolution_artifact` validates manifests, slices, states,
  transitions, lineages, events, QA, references, adjacent-slice transitions,
  event support, and score ranges.
- `write_evolution_synthetic_smoke_artifact` writes a deterministic fixture
  that covers continuation, split, merge, emergence, decline, and ambiguous
  events.
- `validate_result_root` identifies `evolution/evolution_manifest.json`, blocks
  malformed evolution artifacts, and exposes stable/beta/hidden evolution
  feature state through the result manifest.

The real-data writer intentionally uses static membership projection in v1. It
can safely emit continuation, emergence, and decline from projected cluster
presence, but it should not fabricate split or merge claims without richer
time-slice-specific clustering or matching evidence. Split, merge, and
ambiguous validation are covered by the synthetic smoke fixture until a richer
matching writer is added.

## Open Questions

- Should v1 support only adjacent-slice transitions, or allow skip-slice
  continuity for sparse fields?
- Should lineages prefer single best paths, all above-threshold paths, or a
  bounded multi-parent graph?
- Should document overlap be the default matching metric, or should term overlap
  be required when membership is projected from a full-period clustering?
- Should split and merge events require asymmetric overlap thresholds to avoid
  over-labeling small noisy clusters?
- Should evolution artifacts support hierarchical levels in one artifact or one
  artifact per level?
