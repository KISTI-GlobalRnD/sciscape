# Atlas Evidence Inspector Design

This document defines the next Atlas absorption step after the result manifest,
P1 query-to-Atlas smoke, and stable co-occurrence artifacts are in place.

## Goal

The Atlas evidence inspector should help a user read a selected cluster without
inventing evidence that the result root did not validate.

The inspector must consume only:

- `result.feature_states`, derived from `result_manifest.json`;
- `result.artifact_contract`, derived from `validate_result_root`;
- `result.atlas`, built from validated report, membership, edge, abstract, and
  edge-evidence artifacts;
- result-local endpoint payloads that are already backed by those artifacts.

The inspector is a display view model. It is not a new analysis artifact and
must not write state back into the result root.

## Non-Goals

- Do not redesign the whole web app UI in this step.
- Do not add LLM narrative generation.
- Do not claim record-level neighbor evidence unless `atlas.neighbors[].samples`
  or `atlas.edges[].samples` is present.
- Do not expose generic Matrix Builder behavior through the inspector. The
  stable `term_cooccurrence.parquet` and `term_cooccurrence_map.json` artifacts
  are enough for this milestone.

## Source Contract

The inspector should build one client-side `InspectorEvidenceModel` from the
loaded job result.

Required inputs:

| Input | Required fields | Use |
| --- | --- | --- |
| `result.result_state` | `loaded`, `partial`, `blocked`, `empty` | global rendering gate |
| `result.feature_states` | feature -> `hidden`, `beta`, `stable` | section availability |
| `result.artifact_contract.warnings` | `code`, `severity`, `artifact`, `message` | QA and blocking rows |
| `result.artifact_contract.counts` | artifact and row counts | readiness summaries |
| `result.result_manifest.artifacts` | role, path, rows, schema_version | artifact-backed source labels |
| `result.atlas.nodes[]` | `cluster_uid`, `level`, `cluster_id`, `label`, `doc_count`, `doc_count_source`, `keywords`, `representative_works`, `neighbors`, `lineage`, `badges` | selected cluster evidence |
| `result.atlas.edges[]` | `source_uid`, `target_uid`, `weight`, `edge_count`, `relation_label`, `samples` | relation evidence |
| `result.atlas.warnings[]` | warning rows | Atlas-specific caveats |

Optional inputs:

- `/api/jobs/{job_id}/term-network`, only when `feature_states.term_network` is
  `stable` or `beta`.
- `/api/jobs/{job_id}/labels`, only as a label suggestion surface, not evidence
  provenance.

## Exposure Rules

Every inspector section has a feature gate.

| State | UI behavior |
| --- | --- |
| `stable` | show section normally and cite the artifact role/path when useful |
| `beta` | show section with a compact caveat and the warning count |
| `hidden` | collapse section to an unavailable row with the manifest reason |
| `blocked` result | hide analysis sections and show validation failures first |

No section may silently fall back from missing record-level evidence to an
unsupported interpretation. Missing samples should render as `Aggregate only`.

## Inspector Sections

The selected-cluster inspector should be ordered for cluster reading:

1. Identity and readiness
2. Meaning layer
3. Relation evidence
4. Hierarchy and children
5. Representative works
6. QA and limitations

### 1. Identity And Readiness

Gate: `cluster_map`.

Content:

- cluster label, `cluster_uid`, level, cluster id;
- document count and `doc_count_source`;
- feature badges for keyword, co-occurrence, evidence, temporal, evolution, and
  narrative;
- artifact source summary: records, membership, keywords, co-occurrence, edge
  evidence, report data.

Rules:

- If `doc_count_source` is unavailable, show `doc count unavailable`; do not
  infer it from keyword counts.
- If `result_state` is `partial`, keep the map visible but mark missing sections.

### 2. Meaning Layer

Gate: `keyword`.

Content:

- representative terms from `node.keywords`;
- keyword rank, score, frequency, tier, scope, and artifact badges when present;
- common/shared/cluster-specific wording only if those fields are already in
  the keyword payload.

Rules:

- Do not re-rank terms in the inspector.
- If keyword state is `beta`, show the reason and relevant QA warnings before
  the term list.

### 3. Relation Evidence

Gates: `term_network`, `cooccurrence`, and `evidence`.

Content:

- selected neighbor rows from `node.neighbors`;
- relation label, same-parent status, edge count, normalized weight, shared
  terms, and sample count;
- co-occurrence readiness from the `cooccurrence` artifact record;
- raw sample rows only when normalized `samples` are attached.

Rules:

- If `feature_states.cooccurrence == stable`, the relation panel may say
  `Co-occurrence artifact available`.
- If neighbor samples are missing, show aggregate relation facts only.
- If `feature_states.evidence == hidden`, do not show representative work-pair
  rows even when aggregate edge counts exist.

### 4. Hierarchy And Children

Gate: `cluster_map`.

Content:

- lineage path from `node.lineage`;
- parent cluster link when present;
- child rows derived from `parent_uid` and visible `atlas.nodes`;
- child count and child preview.

Rules:

- Child overlays are display state; they do not create a new hierarchy artifact.
- `Show children` should reveal the hidden child level when the payload has it.

### 5. Representative Works

Gate: `evidence`.

Content:

- `node.representative_works` with title, year, citation count, DOI/source when
  present;
- `representative_work_count`;
- source label such as `membership:cluster+abstracts`.

Rules:

- If representative works are unavailable but abstracts exist, show `not joined`
  rather than showing empty cards.
- Do not use representative works as proof of neighbor relations unless relation
  samples point to those records.

### 6. QA And Limitations

Gate: `quality`.

Content:

- blocking issues first;
- beta warnings grouped by artifact role;
- Atlas warnings from `result.atlas.warnings`;
- missing module list from `feature_states.hidden`.

Rules:

- This section is always available when validation can run.
- Warnings should be actionable: artifact role, issue code, and short message.

## View Model Shape

The client may derive this non-persisted structure:

```json
{
  "schema_version": "sciscape_inspector_evidence_view_v1",
  "node_uid": "cluster:7",
  "result_state": "loaded",
  "sections": {
    "identity": {"state": "stable", "rows": []},
    "meaning": {"state": "stable", "rows": []},
    "relations": {"state": "beta", "rows": [], "warnings": []},
    "hierarchy": {"state": "stable", "rows": []},
    "works": {"state": "hidden", "reason": "feature is not backed by available artifacts"},
    "qa": {"state": "stable", "warnings": []}
  }
}
```

This structure should not be saved as a result artifact. It is only a testable
client-side normalization boundary.

## URL And Session State

The inspector may use existing URL/session keys:

- `atlas_node`: selected node;
- `atlas_neighbor`: selected neighbor relation;
- `atlas_view=evidence`: evidence-oriented rendering;
- `atlas_focus`: global, family, neighbors, or pinned;
- browser-local pins and recent selections.

These values must reference stable `cluster_uid` values, but they must not alter
artifact validation or result state.

## Acceptance Criteria

- A blocked result shows validation failures before any cluster-reading panel.
- A loaded result with `keyword=stable` shows representative terms from the
  selected node without re-ranking.
- A loaded result with `cooccurrence=stable` shows that the co-occurrence
  artifact is available and uses relation rows only from validated payloads.
- Neighbor rows without `samples` are labeled aggregate-only.
- Representative work rows appear only when `evidence` is stable or beta and
  `node.representative_works` is non-empty.
- Hidden modules are displayed as unavailable with manifest reasons, not as
  empty panels.
- The P1 Atlas smoke gate remains the contract-level regression path for this
  inspector.

## Implementation Order

Current status:

- `sciscape/web/static/index.html` builds a client-side
  `sciscape_inspector_evidence_view_v1` model for the selected Atlas node.
- The inspector now uses feature states to mark identity, meaning, relation,
  hierarchy, works, and QA sections as stable, beta, hidden, or blocked.
- Representative works and neighbor rows remain payload-backed only; no
  inspector state is written back to the result root.
- Homepage smoke tests assert that the builder and section renderers are exposed.
- The P1 Atlas smoke gate asserts stable co-occurrence and evidence states, plus
  payload-backed neighbor rows with aggregate relation fields, shared-term
  fields, and sampled edge evidence.
- The browser-level Atlas inspector smoke opens the static web app in headless
  Chrome, selects a node, checks sample-backed neighbor evidence, switches to an
  aggregate-only neighbor fallback, and verifies node selection resets neighbor
  state.
- The inspector includes a compact review checklist that summarizes whether the
  selected cluster has enough terms, representative works, relation evidence,
  and QA signal for interpretation.
- The inspector also renders a compact review packet that combines selected
  cluster state, top terms, representative works, and the active neighbor
  evidence mode before any narrative-generation step.
- The Evidence view includes a filterable current-level review queue that counts
  ready/review/blocked clusters, syncs the `atlas_review` URL state, and
  provides direct and next-target navigation to clusters that need analyst
  review.
- The review checklist and queue expose explicit readiness reasons, such as
  missing terms, missing works, hidden sections, QA warnings, or blocking states.
- The Narrative review block writes claim decisions, displays pending/saved/failed
  feedback, and reads back the latest persisted decision for the selected claim.
- When reviewed publication artifacts exist, the Narrative review block links to
  the HTML report, Markdown view, JSON/Markdown downloads, and reviewed
  publication bundle from the same result manifest artifact records used by the
  Download tab. The inspector can also fetch
  `/api/jobs/{job_id}/narrative/publication` to preview the reviewed JSON summary
  without leaving the selected cluster context.

Remaining implementation order:

1. Keep the UI compact: detail belongs in the inspector, drawer, or QA section,
   not on the primary map.
2. Expand the checklist into a fuller review workflow once the UI/UX rewrite
   starts.

Completed implementation order:

1. Add a small client-side evidence model builder in the web app script.
2. Add tests that assert the homepage exposes the builder and section states.
3. Add UI rendering only after the model builder is test-covered.
4. Extend the P1 Atlas smoke to assert stable co-occurrence, stable evidence,
   and the aggregate/sample/shared-term-field neighbor contract in the opened
   result payload.
5. Add `scripts/sciscape_quality_gate.py --atlas-inspector-smoke` for
   browser-level node selection, sample-backed neighbor evidence, and
   aggregate-only neighbor fallback checks.
6. Add the inspector review checklist and include ready/review state checks in
   the browser inspector smoke.
7. Add the selected-cluster review packet and smoke-test term, work,
   sample-backed relation, and aggregate-only relation summaries.
8. Add a filterable current-level review queue for review-target navigation and
   smoke-test that it renders actionable review rows and preserves URL state.
