# SciScape Artifact Contract

Status: implementation contract
Date: 2026-06-01

This document defines the minimum result-root contract used to decide which
SciScape lenses can be shown in the CLI, local web app, static viewer, and
release gates. The contract exists so Atlas Map-style features are enabled from
real artifacts rather than from UI assumptions.

Implementation lives in `sciscape.artifacts`. The release gate entrypoint is:

```bash
uv run --extra dev python scripts/sciscape_quality_gate.py \
  --artifact-root <result-root-or-report-data-json> \
  --write-artifact-contract
```

## Supported Inputs

The validator accepts any of these paths:

- a complete SciScape result root;
- a `landscape/` directory;
- a `landscape/report/` directory;
- a `landscape/report/data.json` file;
- a legacy standalone `data.json`.

The validator must be conservative. Missing optional artifacts should disable
the matching lens. Inconsistent or contaminated artifacts should produce
warnings or blocking errors.

## Minimal Result Root

The preferred result root is:

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

Older outputs may omit `result_manifest.json`, `sciscape_versions.json`, and
`qa/artifact_contract.json`. They should still validate if enough artifacts are
present to infer at least one lens.

`result_manifest.json` is the navigation, provenance, feature-state, export,
and run-status layer above this artifact contract. Its design is documented in
`result_manifest_design.md`. Readers may accept legacy `MANIFEST.json`, but new
writers should emit `result_manifest.json`.

## Required Tables

| Artifact | Required fields | Notes |
| --- | --- | --- |
| `abstracts.parquet` | `uid`, `title`, `abstract`, `pubyear` | Powers overview, evidence, and temporal availability. |
| `edges.parquet` or `combined_edges.parquet` | `uid1`, `uid2`, numeric weight column | Weight candidates include `rel_sum2`, `weight`, `score`, `similarity`, `cosine`, `edge_weight`, and `*_weight`. |
| `landscape/membership.parquet` | `uid`, `cluster` or `cluster_*` | Powers cluster map, evidence joins, and level inference. |
| `landscape/keywords.parquet` | `cluster_id`, `term` | Powers keyword, term network, and keyword QA checks. |
| `landscape/edge_evidence_samples.json` | `source_uid`, `target_uid`, bounded `samples` | Powers raw work-pair evidence for Atlas neighbor relations when generated. |
| `landscape/term_cooccurrence.parquet` | `schema_version`, `cluster_uid`, `cluster_level`, `cluster_id`, `source`, `target`, `weight`, `relation` | Stable P1.5 term co-occurrence table artifact. |
| `landscape/term_cooccurrence_map.json` | `schema_version`, `edge_count`, `term_count`, `cluster_count`, `terms` | Term lookup map paired with `term_cooccurrence.parquet`. |
| `rules/<rule_set_id>/rule_set_manifest.json` | `schema_version`, `rule_set_id`, `rule_type`, `rules_path`, `applications_path`, `before_after_path`, `impact_summary_ref`, `qa_ref` | Replayable keyword cleaning rule artifact contract defined in `keyword_rule_artifact_design.md`. |
| `rules/<rule_set_id>/rules.parquet` | `schema_version`, `rule_set_id`, `rule_id`, `rule_family`, `match_type`, `pattern`, `action`, `destructive`, `enabled` | Rule definitions for keyword cleaning, aliasing, acronym expansion, grouping, and flags. |
| `rules/<rule_set_id>/rule_applications.parquet` | `schema_version`, `rule_set_id`, `application_id`, `rule_id`, `cluster_id`, `raw_term`, `action`, `decision`, `evidence_type` | Replay audit log for rule-term applications. |
| `rules/<rule_set_id>/term_before_after.parquet` | `schema_version`, `rule_set_id`, `cluster_id`, `raw_term`, `term_before`, `term_after`, `display_label`, `rule_ids`, `blocked` | Compact before/after table for Cleaning mode and QA review. |
| `rules/<rule_set_id>/impact_summary.json` | `schema_version`, `rule_set_id`, `counts`, `rule_family_counts`, `action_counts`, `contamination_summary`, `examples` | Bounded summary of cleaning impact. |
| `rules/<rule_set_id>/rule_set_qa.json` | `schema_version`, `rule_set_id`, `status`, `checks`, `counts`, `contamination_counts`, `warnings`, `blocking_issues` | Keyword rule artifact QA summary. |
| `matrices/<matrix_id>/matrix_manifest.json` | `schema_version`, `matrix_id`, `matrix_family`, `format`, `shape`, `value`, `weighting`, `outputs` | General matrix artifact contract defined in `matrix_artifact_design.md`. |
| `matrices/<matrix_id>/matrix_values.parquet` | `schema_version`, `matrix_id`, `row_key`, `column_key`, `row_index`, `column_index`, `value`, `relation` | Sparse triplet matrix values. |
| `matrices/<matrix_id>/row_entities.parquet` and `column_entities.parquet` | `schema_version`, `matrix_id`, `entity_key`, `entity_index`, `entity_type`, `label` | Axis metadata for matrix rows and columns. |
| `matrices/<matrix_id>/matrix_qa.json` | `schema_version`, `matrix_id`, `status`, `counts`, `checks`, `warnings`, `blocking_issues` | Matrix artifact validation report exposed for Download-tab review. |
| `temporal/temporal_manifest.json` | `schema_version`, `temporal_id`, `periodization`, `entity_types`, `metrics`, `outputs` | Temporal trend artifact contract defined in `temporal_artifact_design.md`. |
| `temporal/periods.parquet` | `schema_version`, `temporal_id`, `period_id`, `period_index`, `start_year`, `end_year`, `unit` | Period axis for temporal rows. |
| `temporal/activity.parquet` | `schema_version`, `temporal_id`, `period_id`, `doc_count`, `unknown_year_count` | Result-level activity per period. |
| `temporal/entity_series.parquet` | `schema_version`, `temporal_id`, `entity_type`, `entity_key`, `period_id`, `metric`, `value` | Long-format temporal series for result, cluster, term, and term-family entities. |
| `temporal/temporal_events.parquet` | `schema_version`, `temporal_id`, `event_id`, `event_type`, `entity_type`, `entity_key`, `start_period_id`, `end_period_id`, `metric`, `score`, `method` | Optional growth, decline, burst, or peak signal rows. |
| `temporal/temporal_qa.json` | `schema_version`, `temporal_id`, `status`, `checks`, `counts`, `warnings`, `blocking_issues` | Temporal artifact QA summary. |
| `evolution/evolution_manifest.json` | `schema_version`, `evolution_id`, `slice_method`, `matching_method`, `event_rules`, `outputs` | Cluster evolution artifact contract defined in `evolution_artifact_design.md`. |
| `evolution/time_slices.parquet` | `schema_version`, `evolution_id`, `slice_id`, `slice_index`, `start_year`, `end_year`, `doc_count` | Slice axis for cluster-state evolution. |
| `evolution/cluster_states.parquet` | `schema_version`, `evolution_id`, `state_id`, `slice_id`, `cluster_key`, `cluster_label`, `doc_count` | Slice-specific cluster states. |
| `evolution/transitions.parquet` | `schema_version`, `evolution_id`, `transition_id`, `source_state_id`, `target_state_id`, `score`, `support_count`, `relation` | Directed transition evidence between adjacent slice states. |
| `evolution/lineages.parquet` | `schema_version`, `evolution_id`, `lineage_id`, `state_id`, `slice_id`, `role`, `stability_score` | Derived identity paths across slices. |
| `evolution/evolution_events.parquet` | `schema_version`, `evolution_id`, `event_id`, `event_type`, `slice_id`, `state_id`, `transition_refs`, `score`, `method` | Continuation, split, merge, emergence, decline, or ambiguous event rows. |
| `evolution/evolution_qa.json` | `schema_version`, `evolution_id`, `status`, `checks`, `counts`, `event_counts`, `warnings`, `blocking_issues` | Evolution artifact QA summary. |
| `narrative/narrative_manifest.json` | `schema_version`, `narrative_id`, `narrative_scope`, `claim_policy`, `evidence_policy`, `outputs` | Narrative evidence-reference artifact contract defined in `narrative_artifact_design.md`. |
| `narrative/narrative_targets.parquet` | `schema_version`, `narrative_id`, `target_id`, `target_type`, `target_key`, `target_label`, `feature_state` | Entities that may receive narrative claims. |
| `narrative/claims.parquet` | `schema_version`, `narrative_id`, `claim_id`, `target_id`, `section_id`, `claim_type`, `claim_text`, `support_state`, `confidence`, `evidence_ref_count` | Claim rows with support state and text origin. |
| `narrative/evidence_sources.parquet` | `schema_version`, `narrative_id`, `evidence_source_id`, `artifact_ref`, `artifact_role`, `artifact_path`, `resolver`, `source_state` | Artifact sources that evidence refs can point to. |
| `narrative/evidence_refs.parquet` | `schema_version`, `narrative_id`, `evidence_ref_id`, `evidence_source_id`, `evidence_type`, `entity_type`, `entity_key`, `locator_type`, `locator` | Row-level or aggregate evidence pointers. |
| `narrative/claim_evidence_links.parquet` | `schema_version`, `narrative_id`, `claim_id`, `evidence_ref_id`, `evidence_role`, `link_strength`, `required` | Claim-to-evidence links. |
| `narrative/narrative_sections.parquet` | `schema_version`, `narrative_id`, `section_id`, `target_id`, `section_type`, `section_title`, `section_state`, `claim_count` | Narrative display grouping and section state. |
| `narrative/review_decisions.parquet` | `schema_version`, `narrative_id`, `decision_id`, `claim_id`, `decision_type`, `reviewer`, `decided_at_utc`, `reason`; optional `target_id`, `cluster_uid` | Optional review decisions when review state is advertised. |
| `narrative/narrative_qa.json` | `schema_version`, `narrative_id`, `status`, `checks`, `counts`, `claim_counts`, `unsupported_claims`, `warnings`, `blocking_issues` | Narrative artifact QA summary. |
| `narrative/generation_metadata.json` | `schema_version`, `narrative_id`, `generation_mode`, `text_origins`, `llm_generation_used`, `model_generation`, `parameters`, `transforms` | Provenance for deterministic or model-assisted narrative text generation. |
| `narrative/publication_summary.json` | `schema_version`, `narrative_id`, `publication_state`, `counts`, `cluster_index`, `clusters`, `warnings` | Review-state-aware publication summary that renders only accepted or not-required claims and exposes a cluster-level report index. |
| `narrative/publication_summary.md` | Markdown headings, review summary, cluster index, rendered claims, omitted-claim list | Human-readable reviewed narrative publication summary. |
| `narrative/publication_summary.html` | Static HTML report, review summary cards, cluster index, rendered claims, omitted-claim list | Browser-readable reviewed narrative publication report generated from the same JSON payload. |
| `narrative/publication_bundle.zip` | Result-relative narrative publication files | Shareable reviewed narrative bundle containing the publication summaries plus manifest, QA, generation metadata, review decisions, and claim/evidence tables. |
| `exports/<export_id>/export_manifest.json` | `schema_version`, `export_id`, `export_family`, `export_kind`, `format`, `status`, `feature_refs`, `source_artifacts`, `outputs` | Export manifest contract defined in `export_manifest_design.md`. |
| `exports/<export_id>/export_files.parquet` | `schema_version`, `export_id`, `file_id`, `path`, `role`, `format`, `public_share_state` | Output file inventory. |
| `exports/<export_id>/export_inputs.parquet` | `schema_version`, `export_id`, `input_id`, `artifact_ref`, `artifact_role`, `artifact_path`, `feature_state`, `required` | Source artifacts and feature states used by the export. |
| `exports/<export_id>/export_transforms.parquet` | `schema_version`, `export_id`, `transform_id`, `step_index`, `transform_type`, `description`, `parameters` | Filters, field mappings, layout, packaging, and format conversion steps. |
| `exports/<export_id>/export_qa.json` | `schema_version`, `export_id`, `status`, `checks`, `counts`, `compatibility`, `warnings`, `blocking_issues` | Export QA summary. |
| matrix-like legacy artifacts | matrix rows plus row/column metadata when available | Legacy `*matrix*` artifacts may support beta matrix exposure. Stable matrix exposure requires `matrices/<matrix_id>/matrix_manifest.json`. Co-occurrence artifacts support the narrower co-occurrence/term-network lens unless wrapped as a general matrix. |

## Feature Block

Every validation result and embedded report contract should include:

```json
{
  "schema_version": "sciscape_artifact_contract_v1",
  "mode": "local_result",
  "result_state": "loaded",
  "features": {
    "overview": true,
    "cluster_map": true,
    "keyword": true,
    "term_network": true,
    "matrix": false,
    "evidence": true,
    "temporal": true,
    "evolution": false,
    "narrative": false,
    "quality": true,
    "export": true
  },
  "warnings": [],
  "versions": {},
  "artifacts": {},
  "counts": {}
}
```

Feature inference rules:

| Feature | Enable when |
| --- | --- |
| `overview` | abstracts exist or report clusters exist |
| `cluster_map` | membership exists or report clusters exist |
| `keyword` | keyword table exists or report clusters carry keywords; replayable keyword rule artifacts may strengthen stable exposure only when QA passes |
| `term_network` | stable co-occurrence artifact rows, report co-occurrence/network edges, or keyword rows can form within-cluster pairs |
| `matrix` | stable general matrix manifest exists, or a legacy matrix-like artifact exists for beta exposure |
| `evidence` | abstracts and membership both exist |
| `temporal` | stable temporal artifacts exist, or abstracts include `pubyear` for beta temporal views |
| `evolution` | stable evolution artifacts exist; embedded legacy evolution payloads may support beta views only |
| `narrative` | stable narrative artifacts exist; embedded legacy narrative payloads may support beta views only |
| `quality` | validation can run; keyword rule QA may add warnings or blocking issues |
| `export` | stable export manifests exist, or keyword/cluster map/report data exists for beta legacy exports |

`cooccurrence` is a result-manifest feature state rather than a legacy boolean in
the lightweight report contract. Stable co-occurrence artifacts can enable term
co-occurrence and term-network views without implying the generic Matrix Builder
is ready.

## Report Data Contract

Standalone report/viewer data should embed a lightweight `_sciscape` block:

```json
{
  "_sciscape": {
    "schema_version": "sciscape_report_data_contract_v1",
    "mode": "static_viewer",
    "result_state": "loaded",
    "features": {},
    "warnings": [],
    "atlas": {
      "schema_version": "sciscape_atlas_payload_v1",
      "levels": ["cluster"],
      "nodes": [],
      "node_count": 0,
      "edges": [],
      "edge_count": 0,
      "warnings": []
    },
    "versions": {
      "sciscape_version": "0.2.0",
      "report_data_contract_schema_version": "sciscape_report_data_contract_v1",
      "atlas_payload_schema_version": "sciscape_atlas_payload_v1"
    },
    "created_at_utc": "2026-06-01T00:00:00+00:00"
  }
}
```

The static viewer may use this block to disable tabs and explain missing
features, but the block is not allowed to override the stricter result-root
validator.

## Atlas Map Node Contract

The first Atlas Map shell should be able to derive or consume nodes with these
fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `cluster_uid` | yes | Stable UI key, independent of display label. |
| `level` | yes | Generic ordered level name; Domain/Macro/Meso/Micro are defaults, not hard-coded requirements. |
| `cluster_id` | yes | Source cluster identifier within the level. |
| `label` | yes | Human-readable label. |
| `short_label` | no | Compact label for dense map surfaces. |
| `parent_uid` | no | Parent cluster UID for lineage and child overlays. |
| `doc_count` | yes, nullable | Works/documents assigned to the cluster. Null means this report only has keyword-level cluster data. |
| `doc_count_source` | yes | Source column for `doc_count`, or `unavailable`. |
| `keyword_count` | yes | Number of representative keyword rows attached to the cluster. |
| `child_count` | no | Direct child count when hierarchy exists. |
| `x`, `y` | no | Optional layout coordinates. |
| `keywords` | no | Representative terms with rank/score/count when available. |
| `representative_works` | no | Compact supporting works with `uid`, `title`, optional `year`, and optional `cited_by_count`. |
| `representative_work_count` | no | Count of joined records behind the bounded representative work list. |
| `badges` | no | QC or review warnings with severity. |

If required identity fields are not available, the UI may still show
keyword/report tables, but it should not advertise an Atlas Map-quality cluster
map. If `doc_count` is null, the map can still show identity and keyword
structure, but Scale and related metric lenses should be disabled or caveated.

When `membership.parquet` is available, Atlas payload generation should enrich
nodes with `doc_count`, `doc_count_source`, `parent_uid`, `child_count`, and a
root-to-node `lineage` path. Generic single-level outputs may map the lone
membership column to the report `cluster` level. Multi-level outputs should use
the `cluster_<level>` column names, such as `cluster_macro`, `cluster_micro`,
and `cluster_nano`, to infer hierarchy.

When both `membership.parquet` and `abstracts.parquet` are available, local
Atlas payload generation may also add bounded `representative_works` for each
cluster. This is intentionally display-scale evidence: it proves that cluster
identity can be traced back to records, but it is not a substitute for the full
abstract table or future sampled edge-evidence artifacts.

Legacy report data often stores leaf clusters as anonymous `cluster:*` entries
even when `membership.parquet` contains `cluster_micro` and `cluster_nano`.
When the report cluster IDs match the finest membership column, local Atlas
payload enrichment may promote those report nodes to the finest level and add
missing parent nodes from membership. These inferred parent nodes should carry
`node_source: "membership_parent"` so the UI can distinguish them from
report-authored clusters.

## Atlas Map Edge Contract

When `edges.parquet` and `membership.parquet` are both available, the local web
app may enrich the static report payload with cluster-level edges:

| Field | Required | Meaning |
| --- | --- | --- |
| `source_uid` | yes | Source cluster UID. |
| `target_uid` | yes | Target cluster UID. |
| `level` | yes | Level at which the edge was aggregated. |
| `weight` | yes | Sum of paper-edge weights between the two clusters. |
| `edge_count` | yes | Number of paper edges contributing to the aggregate. |
| `shared_terms` | no | Representative keyword overlap between the two endpoint clusters. |
| `same_parent` | no | Whether both endpoints share the same inferred parent UID. |
| `relation_label` | no | Compact display label such as `same-parent` or `cross-cluster`. |

Each node may also carry a bounded `neighbors` list and `neighbor_count`, derived
from these aggregate edges. This is an evidence surface for navigation, not a
replacement for the raw paper-edge table.

Optional edge-evidence sidecars may attach bounded raw samples to those
aggregate relations. A sidecar may be JSON or Parquet and should use filenames
matching `*edge*evidence*`, `*neighbor*evidence*`, or `*relation*evidence*`.
`run_landscape()` writes `edge_evidence_samples.json` by default when
`edges.parquet`, `membership.parquet`, and abstracts are available; this writer
is bounded by relation count and sample count so it remains a review artifact,
not a copy of the full edge table.
Each relation row should identify the aggregate endpoints with
`source_uid`/`target_uid` or compatible aliases such as
`source_cluster_uid`/`target_cluster_uid` and `cluster_uid`/`neighbor_uid`.
Rows may either carry a `samples` array or represent one sample per row.
Recognized sample fields include `source_work_uid`, `target_work_uid`,
`source_title`, `target_title`, `edge_type`, and `weight`; common aliases such
as `uid1`, `uid2`, `title1`, `title2`, `layer`, and `rel_sum2` are normalized
for display.

The web Atlas inspector may summarize selected-cluster evidence from these node
fields without adding a new payload schema: record join (`doc_count_source`),
representative terms, representative works, lineage, aggregate neighbors,
clickable child clusters, and node-level QA badges. Rows with missing inputs
should be displayed as missing evidence rather than silently hidden when the
absence changes interpretation.

Client-side Atlas search may also be derived from the same node fields. Search
hits should remain result-local and level-grouped unless a future artifact adds
server-side search scores or record-level snippets.

Client-side Atlas lenses may be derived from node fields as display state only.
The initial supported lenses are identity (`cluster`), document-count scale
(`scale`), and evidence-readiness (`evidence`). These lenses may alter sorting,
accent intensity, and compact card metrics, but they should not be treated as
new analysis artifacts unless a future result root emits explicit lens metrics.
The lens scale strip should also stay client-derived: scope, coverage,
min/median/max range, and normalization text are computed from visible
`atlas.nodes`, not persisted as result data.

Client-side Atlas view modes are separate from metric lenses. `Map`,
`Hierarchy`, and `Evidence` views may be restored with `atlas_view`, but they
must remain alternate renderings of the loaded `atlas.nodes` and related result
metadata.

Client-side Atlas focus controls are also display state. `Global`, `Family`,
`Neighbors`, and `Pinned` may be restored with `atlas_focus`; they filter the
currently visible node list from `parent_uid`, `neighbors`, and browser-local
pins without changing the result payload.

Client-side neighbor relation evidence is also display state unless future
payloads attach raw samples. `atlas_neighbor` may restore the selected neighbor
relation. With the current v1 payload it must show only aggregate relation
facts from `neighbors`: relation label, edge count, weight, same-parent status,
and shared terms. Raw work-pair or paper-pair evidence remains unavailable
unless a neighbor object explicitly carries normalized `samples` from an
edge-evidence sidecar.

The web app treats `artifact_contract.features` as a first-class feature block
when a job result does not also expose top-level `features`. The Atlas module
readiness strip is therefore display-only state derived from the result feature
contract, not a separate payload module. Missing modules should be described as
deferred until backing artifacts attach, not rendered as empty analysis panels.

The web Atlas orientation strip is also payload-derived. It should summarize
source, level path, visible node coverage, evidence readiness, warning count,
and selected lineage from `atlas`, `artifact_contract`, and result metadata
without requiring a new sidecar.

Pinned clusters and recent cluster selections are browser-local session state.
They may reference stable `cluster_uid` values from the loaded Atlas payload,
but they are not analysis artifacts and should not be written back into the
result root.

## Blocking Conditions

Validation should block release-quality status when:

- required fields are missing from advertised artifacts;
- membership UIDs are absent from abstracts;
- keyword cluster IDs are absent from membership;
- `data.json` advertises a feature that artifacts do not support;
- top-ranked keyword rows contain HTML, LaTeX preamble, publisher metadata, or
  other hard artifact fragments.

Non-blocking warnings are allowed for incomplete optional modules, missing
report contracts on legacy data, absent term networks, and review-only keyword
artifact flags.

## Output Location

`write_artifact_contract()` writes to:

```text
<result_root>/landscape/qa/artifact_contract.json
```

when a landscape directory is detected. Otherwise it writes to:

```text
<result_root>/qa/artifact_contract.json
```

This file is the source of truth for local result loading, release readiness,
and future Atlas Map lens availability.
