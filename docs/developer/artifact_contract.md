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
| matrix/co-occurrence artifacts | matrix rows plus row/column metadata when available | Any `*matrix*` or `*cooccurrence*` artifact is treated as matrix evidence. |
| evolution artifacts | `*evolution*` or `*trajectory*` JSON/parquet | Powers the evolution lens only when present or embedded in `data.json`. |
| narrative artifacts | `*narrative*` JSON/parquet | Powers narrative only when present or embedded in `data.json`. |

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
    "matrix": true,
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
| `keyword` | keyword table exists or report clusters carry keywords |
| `term_network` | report co-occurrence/network edges exist or keyword rows can form within-cluster pairs |
| `matrix` | matrix/co-occurrence artifact exists or report term edges exist |
| `evidence` | abstracts and membership both exist |
| `temporal` | abstracts include `pubyear` |
| `evolution` | evolution/trajectory artifact exists or report data embeds evolution payloads |
| `narrative` | narrative artifact exists or report data embeds narrative payloads |
| `quality` | validation can run |
| `export` | keyword, cluster map, or report data exists |

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
