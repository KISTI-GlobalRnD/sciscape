# Narrative Evidence Reference Artifact Design

This document defines the narrative evidence-reference artifact contract that
should come before any narrative generation or polished cluster-story UI.

The purpose is to make every cluster interpretation claim traceable to validated
artifacts, rows, metrics, and QA caveats. Narrative is allowed only as an
evidence-backed review surface, not as the source of truth for the analysis.

## Goal

SciScape should treat narratives as claim graphs backed by artifacts.

A narrative artifact must answer:

- which entity the narrative describes;
- which claims are being made;
- what evidence supports each claim;
- which evidence sources and artifact rows were used;
- whether the claim is supported, weak, contradicted, or unsupported;
- whether the text was deterministic, manually reviewed, or model-generated;
- whether the result is safe to display as a narrative lens.

## Non-Goals

- Do not add narrative generation in this milestone.
- Do not ship free-form LLM cluster descriptions as product output.
- Do not present narrative text without resolvable evidence references.
- Do not use representative works, temporal charts, or neighboring clusters as
  proof unless the corresponding evidence rows are attached.
- Do not turn missing evidence into confident prose. Missing evidence is a
  caveat, not a claim.
- Do not describe institutional strategy, policy value, or field causality from
  paper-cluster artifacts alone.

## Narrative Versus Evidence Inspector

The Atlas evidence inspector is a display view model derived from validated
payload fields. It does not write result-root state.

The narrative artifact is a persisted claim/evidence contract. It may feed a
future narrative panel, report, or review workflow, but it must remain
auditable without the web app.

| Surface | Purpose | Persisted |
| --- | --- | --- |
| Evidence inspector | Let a user inspect selected cluster evidence | no |
| Narrative artifact | Store evidence-backed claims and review state | yes |
| Narrative generator | Produce or update claim text | not in this milestone |

## Canonical Directory Shape

Narrative artifacts should live under:

```text
<result_root>/narrative/
  narrative_manifest.json
  narrative_targets.parquet
  claims.parquet
  evidence_sources.parquet
  evidence_refs.parquet
  claim_evidence_links.parquet
  narrative_sections.parquet
  review_decisions.parquet
  narrative_qa.json
  generation_metadata.json
  publication_summary.json
  publication_summary.md
  publication_summary.html
  publication_bundle.zip
```

For a landscape-scoped result, the same directory may live under:

```text
<result_root>/landscape/narrative/
```

`review_decisions.parquet` is optional until the app has an explicit review
workflow. If review state is advertised in the manifest, the file is required.

Reviewed publication summaries should include `cluster_index` alongside
`clusters`. The index is the report navigation surface: each row carries the
cluster anchor, target id, cluster uid, display label, claim counts, rendered
claim count, omitted claim count, pending review count, and cluster-level
publication state. Markdown and HTML reports should render this index before
the full claim sections.

## Schema Versions

Use explicit schema names:

- `sciscape_narrative_manifest_v1`
- `sciscape_narrative_targets_v1`
- `sciscape_narrative_claims_v1`
- `sciscape_narrative_evidence_sources_v1`
- `sciscape_narrative_evidence_refs_v1`
- `sciscape_narrative_claim_evidence_links_v1`
- `sciscape_narrative_sections_v1`
- `sciscape_narrative_review_decisions_v1`
- `sciscape_narrative_qa_v1`
- `sciscape_narrative_generation_metadata_v1`

## Narrative Manifest

`narrative_manifest.json` is the source of truth for a narrative artifact.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_manifest_v1` |
| `narrative_id` | string | stable local identifier |
| `title` | string | human-readable title |
| `result_id` | string or null | parent result when available |
| `narrative_scope` | object | target types, levels, cluster universe, and filters |
| `claim_policy` | object | allowed claim types, minimum evidence, and unsupported-claim behavior |
| `evidence_policy` | object | evidence source requirements and resolver rules |
| `text_policy` | object | deterministic, reviewed, and model-generated text rules |
| `source_artifacts` | array | input artifact refs and roles |
| `rule_sets` | array | cleaning, labeling, filter, or review rules used |
| `transforms` | array | ordered transform or review steps |
| `outputs` | object | paths to targets, claims, evidence, sections, reviews, and QA |
| `created_at_utc` | string | creation timestamp |
| `warnings` | array | non-blocking caveats |

Example:

```json
{
  "schema_version": "sciscape_narrative_manifest_v1",
  "narrative_id": "cluster_narrative_evidence_default",
  "title": "Cluster narrative evidence references",
  "result_id": "openalex_gnn_20260603",
  "narrative_scope": {
    "target_types": ["cluster"],
    "cluster_level": "default",
    "include_hidden_features": false,
    "filter_refs": []
  },
  "claim_policy": {
    "allowed_claim_types": [
      "identity",
      "keyword_meaning",
      "representative_work",
      "relation",
      "temporal_signal",
      "evolution_signal",
      "quality_caveat",
      "limitation"
    ],
    "minimum_evidence_refs": 1,
    "unsupported_claim_action": "block",
    "contradiction_action": "block",
    "weak_evidence_action": "mark_beta"
  },
  "evidence_policy": {
    "allowed_source_roles": [
      "keywords",
      "records",
      "membership",
      "cooccurrence",
      "matrix",
      "temporal",
      "evolution",
      "edge_evidence",
      "quality"
    ],
    "require_resolvable_refs": true,
    "allow_aggregate_only": true,
    "allow_quotes": false
  },
  "text_policy": {
    "allowed_origins": ["deterministic_template", "human_review"],
    "llm_generation_allowed": false,
    "require_generation_metadata_when_model_generated": true
  },
  "source_artifacts": [
    {"role": "keywords", "path": "landscape/keywords.parquet"},
    {"role": "cooccurrence", "path": "landscape/term_cooccurrence.parquet"},
    {"role": "quality", "path": "landscape/qa/artifact_contract.json"}
  ],
  "rule_sets": [],
  "transforms": [
    {"step": "collect_targets"},
    {"step": "collect_evidence_sources"},
    {"step": "build_deterministic_claim_scaffold"},
    {"step": "link_claim_evidence"},
    {"step": "validate_claim_support"}
  ],
  "outputs": {
    "targets": "narrative_targets.parquet",
    "claims": "claims.parquet",
    "evidence_sources": "evidence_sources.parquet",
    "evidence_refs": "evidence_refs.parquet",
    "claim_evidence_links": "claim_evidence_links.parquet",
    "sections": "narrative_sections.parquet",
    "reviews": "review_decisions.parquet",
    "qa": "narrative_qa.json",
    "generation_metadata": "generation_metadata.json",
    "publication_json": "publication_summary.json",
    "publication_markdown": "publication_summary.md",
    "publication_html": "publication_summary.html",
    "publication_bundle": "publication_bundle.zip"
  },
  "created_at_utc": "2026-06-03T00:00:00+00:00",
  "warnings": []
}
```

## Generation Metadata

`generation_metadata.json` records how claim text was produced. It is required
for newly written SciScape narrative artifacts, and model-generated text cannot
be promoted without model and prompt metadata.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_generation_metadata_v1` |
| `narrative_id` | string | parent narrative id |
| `generation_mode` | string | `deterministic_scaffold`, `model_assisted`, `imported`, or future mode |
| `text_origins` | array | text origins present in the claim graph |
| `llm_generation_used` | boolean | true when any claim text was generated by an LLM |
| `model_generation` | object or null | provider, model, run id, parameters, and timestamps when LLM text is used |
| `prompt_ref` | string or null | stable prompt template or artifact reference |
| `prompt_digest` | string or null | digest of the prompt payload when applicable |
| `source_artifacts` | array | artifact roles and paths used for generation |
| `parameters` | object | deterministic scaffold or generation parameters |
| `transforms` | array | ordered generation steps |
| `sciscape_version` | string | package version |
| `created_at_utc` | string | creation timestamp |

Model-assisted claim text must pass through
`apply_narrative_generation_updates(...)` or an equivalent writer that preserves
the existing claim/evidence link table. The writer may update `claim_text` and
`text_origin` for declared claim ids, but must not rewrite evidence refs or
claim-evidence links. Updated model-generated claims reset to `not_reviewed`
or `needs_revision`; publication renderers should continue to omit them until a
review decision accepts or marks them not required. The writer must record
provider, model, model run id, prompt reference, optional prompt digest, updated
claim ids, and transform steps in `generation_metadata.json`.

The CLI batch path is `sciscape narrative apply-generated <result_root>
<updates_file>`. The update file may be a JSON object with `claim_updates`, a
JSON array of claim updates, or JSONL rows. This path imports already generated
claim text and routes it through the same evidence-preserving writer; it does
not execute prompts or call model providers.

The prompt rendering path is `sciscape narrative render-prompts <result_root>`.
It writes `narrative/generation_prompts/prompt_batch_manifest.json` and
`narrative/generation_prompts/prompt_jobs.jsonl` from existing claim/evidence
tables. Each JSONL row carries a stable `claim_id`, prompt reference, prompt
digest, current scaffold claim, target context, and evidence refs. This is an
auditable provider-neutral input package; it does not call a model and does not
modify `claims.parquet`.

## Narrative Targets Table

`narrative_targets.parquet` stores the entities that may receive narrative
claims.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_targets_v1` |
| `narrative_id` | string | parent narrative id |
| `target_id` | string | stable local target key |
| `target_type` | string | `cluster`, `relation`, `result`, or future type |
| `target_key` | string | source entity key, such as `cluster_uid` |
| `target_label` | string | display label |
| `feature_state` | string | target feature state when applicable |

Optional columns:

- `cluster_uid`, `cluster_id`, `parent_uid`, `level`;
- `source_state_id`, `target_state_id`, `lineage_id`;
- `doc_count`, `keyword_count`, `evidence_count`;
- `warning_flags`.

Rules:

- `target_id` must be unique.
- `target_key` must resolve to a validated cluster, relation, result, lineage,
  or other declared source entity.
- A hidden feature state must create limitation or caveat claims, not confident
  interpretation claims.

## Claims Table

`claims.parquet` stores one row per narrative claim.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_claims_v1` |
| `narrative_id` | string | parent narrative id |
| `claim_id` | string | stable local claim id |
| `target_id` | string | target from `narrative_targets.parquet` |
| `section_id` | string | section from `narrative_sections.parquet` |
| `claim_type` | string | controlled claim type |
| `claim_text` | string | user-facing claim text or deterministic scaffold text |
| `support_state` | string | `supported`, `weak`, `contradicted`, `unsupported`, or `caveat` |
| `confidence` | float | bounded confidence or support score in `[0, 1]` |
| `evidence_ref_count` | int | number of linked evidence refs |
| `text_origin` | string | `deterministic_template`, `human_review`, `model_generated`, or `imported` |
| `review_state` | string | `not_reviewed`, `accepted`, `needs_revision`, `rejected`, or `not_required` |

Optional columns:

- `claim_template_id`;
- `source_claim_id`;
- `model_run_id`;
- `prompt_ref`;
- `language`;
- `sort_order`;
- `warning_flags`;
- `unsupported_reason`.

Rules:

- `claim_id` must be unique.
- `target_id` and `section_id` must resolve.
- `confidence` must be finite and in `[0, 1]`.
- `support_state=supported` requires at least one resolvable evidence ref.
- `support_state=unsupported` is allowed only as an explicit blocked or caveat
  row; it must not be rendered as normal narrative text.
- `text_origin=model_generated` requires model, prompt, and generation metadata
  in the manifest or review decisions.

## Evidence Sources Table

`evidence_sources.parquet` stores the artifact sources that evidence refs may
point to.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_evidence_sources_v1` |
| `narrative_id` | string | parent narrative id |
| `evidence_source_id` | string | stable evidence source id |
| `artifact_ref` | string | result manifest artifact ref |
| `artifact_role` | string | role such as `keywords`, `records`, `cooccurrence`, `temporal`, `evolution`, `quality` |
| `artifact_path` | string | path relative to result root |
| `schema_version_ref` | string or null | source artifact schema version |
| `resolver` | string | row/entity resolver strategy |
| `source_state` | string | `stable`, `beta`, `hidden`, or `blocked` |

Optional columns:

- `row_count`, `column_count`;
- `checksum`;
- `warning_flags`;
- `qa_ref`.

Rules:

- `artifact_ref` must exist in the result manifest or artifact contract.
- `source_state=hidden` may support caveat claims only.
- `source_state=blocked` cannot support normal claims.

## Evidence Refs Table

`evidence_refs.parquet` stores row-level or aggregate evidence pointers.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_evidence_refs_v1` |
| `narrative_id` | string | parent narrative id |
| `evidence_ref_id` | string | stable evidence ref id |
| `evidence_source_id` | string | source from `evidence_sources.parquet` |
| `evidence_type` | string | `term`, `work`, `relation`, `metric`, `temporal_event`, `evolution_event`, `qa_caveat`, or future type |
| `entity_type` | string | entity described by the evidence |
| `entity_key` | string | stable entity key |
| `locator_type` | string | `row_id`, `row_filter`, `entity_key`, `metric_key`, `aggregate`, or `json_path` |
| `locator` | string | resolver value |
| `evidence_label` | string | compact display label |
| `support_count` | int or null | support count when available |

Optional columns:

- `metric`, `value`, `rank`, `score`;
- `cluster_uid`, `term`, `work_uid`, `transition_id`, `event_id`;
- `period_id`, `slice_id`, `lineage_id`;
- `quote_text`, `quote_start`, `quote_end`;
- `privacy_policy`, `excerpt_policy`;
- `warning_flags`.

Rules:

- `evidence_ref_id` must be unique.
- `evidence_source_id` must resolve.
- Row-level locators must be resolvable by the declared resolver.
- Aggregate refs must explicitly use `locator_type=aggregate`.
- Quotes or snippets may be included only when source policy allows them.
- Evidence refs are pointers and metrics, not narrative claims.

## Claim Evidence Links Table

`claim_evidence_links.parquet` links claims to evidence refs.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_claim_evidence_links_v1` |
| `narrative_id` | string | parent narrative id |
| `claim_id` | string | claim from `claims.parquet` |
| `evidence_ref_id` | string | evidence ref from `evidence_refs.parquet` |
| `evidence_role` | string | `primary`, `supporting`, `caveat`, `contradicting`, or `context` |
| `link_strength` | float | bounded strength in `[0, 1]` |
| `required` | bool | whether the claim is invalid without this ref |

Optional columns:

- `sort_order`;
- `link_reason`;
- `warning_flags`.

Rules:

- `(claim_id, evidence_ref_id, evidence_role)` must be unique.
- `claim_id` and `evidence_ref_id` must resolve.
- `link_strength` must be finite and in `[0, 1]`.
- Supported claims need at least one `primary` or required `supporting` link.
- Contradicting links must downgrade the claim to `contradicted`, `weak`, or
  `caveat`.

## Narrative Sections Table

`narrative_sections.parquet` stores display grouping and section-level state.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_sections_v1` |
| `narrative_id` | string | parent narrative id |
| `section_id` | string | stable section id |
| `target_id` | string | target from `narrative_targets.parquet` |
| `section_type` | string | `identity`, `meaning`, `relations`, `temporal`, `evolution`, `limitations`, or future type |
| `section_title` | string | display title |
| `section_state` | string | `stable`, `beta`, `hidden`, or `blocked` |
| `claim_count` | int | claims in the section |

Optional columns:

- `sort_order`;
- `artifact_refs`;
- `warning_flags`;
- `hidden_reason`.

Rules:

- `section_id` must be unique.
- `section_state=stable` requires all non-caveat claims to be supported.
- `section_state=hidden` must have a reason and should not contain confident
  claims.

## Review Decisions Table

`review_decisions.parquet` records human or automated review changes.

Required columns when review is advertised:

| Column | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_narrative_review_decisions_v1` |
| `narrative_id` | string | parent narrative id |
| `decision_id` | string | stable decision id |
| `claim_id` | string | claim being reviewed |
| `decision_type` | string | `accepted`, `needs_revision`, `not_required`, or `rejected` |
| `reviewer` | string | reviewer identifier or `system` |
| `decided_at_utc` | string | decision timestamp |
| `reason` | string | short reason |

Optional columns:

- `old_claim_text`;
- `new_claim_text`;
- `old_support_state`;
- `new_support_state`;
- `evidence_ref_ids`;
- `target_id`;
- `cluster_uid`;
- `review_batch_id`;
- `warning_flags`.

Rules:

- `decision_id` must be unique.
- `claim_id` must resolve.
- New review writeback should include `target_id` and `cluster_uid`; readers must
  tolerate older review rows that only identify the resolved `claim_id`.
- Edits that change claim meaning must keep or update evidence refs.
- Rejected claims must not render as normal narrative text.

## QA Contract

`narrative_qa.json` should summarize claim support and display safety.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_narrative_qa_v1` |
| `narrative_id` | parent narrative id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | named checks with status and counts |
| `counts` | targets, sections, claims, evidence refs, links, reviews |
| `claim_counts` | claims by support state and type |
| `unsupported_claims` | unsupported claim IDs and reasons |
| `warnings` | non-blocking warnings |
| `blocking_issues` | release-blocking issues |

Minimum checks:

- manifest schema is supported;
- target, claim, evidence source, evidence ref, link, section, and QA files
  exist when advertised;
- all target, section, source, evidence, and claim refs resolve;
- every supported claim has at least one primary or required supporting
  evidence link;
- evidence source artifacts exist and are not blocked;
- aggregate-only evidence is labeled as aggregate-only;
- contradicted claims are not rendered as supported claims;
- unsupported claims are blocked or rendered only as explicit caveats;
- model-generated text has model, prompt, and run metadata;
- QA caveats and missing feature states are not contradicted by claims;
- review decisions preserve evidence refs for claims marked as accepted,
  revision-needed, not-required, or rejected.

## Validation States

Narrative validation should feed the normal result contract:

| Condition | Result |
| --- | --- |
| manifest, targets, claims, evidence sources, evidence refs, links, sections, and QA all pass | `narrative=stable` |
| artifact exists but has weak evidence, aggregate-only evidence, or `not_reviewed` optional claims | `narrative=beta` |
| artifact is advertised but has unresolved evidence refs or unsupported normal claims | result `blocked` |
| deterministic scaffold exists but no narrative artifact is written | `narrative=hidden` |
| no narrative artifact exists | `narrative=hidden` |

This keeps narrative availability separate from evidence, temporal, and
evolution availability. A result can have rich evidence surfaces without a
stable narrative artifact.

## Claim Type Rules

The first narrative contract should support these claim types:

| Claim type | Required evidence |
| --- | --- |
| `identity` | cluster target, doc count, label source, and keyword refs |
| `keyword_meaning` | representative term rows, term-family rows, or keyword QA refs |
| `representative_work` | work refs with title/year and cluster membership refs |
| `relation` | neighbor edge, co-occurrence, matrix, or edge-evidence refs |
| `temporal_signal` | temporal entity-series or temporal-event refs |
| `evolution_signal` | evolution transition, lineage, or event refs |
| `quality_caveat` | QA warning or blocking issue refs |
| `limitation` | missing feature state, sparse artifact, or unsupported evidence refs |

Rules:

- Relation claims need relation evidence, not only shared keywords.
- Temporal signal claims need temporal artifacts or explicit beta caveats.
- Evolution signal claims need evolution artifacts; temporal artifacts are not
  sufficient.
- Quality caveats and limitations may be rendered even when other claims are
  hidden.

## Writer Utility Design

The first implementation step is the lighter
`cluster_review_packet_artifact` defined in
`cluster_review_packet_design.md`. It collects deterministic cluster evidence
rows and validates local evidence refs, but it does not create narrative
claims. Narrative claim artifacts should consume that packet rather than
re-reading unrelated UI state.

The first writer utility should create deterministic scaffolds only:

```python
write_narrative_evidence_artifacts(
    result_root,
    *,
    narrative_id,
    targets,
    evidence_sources,
    claims=None,
    sections=None,
    review_decisions=None,
    source_artifacts=None,
    rule_sets=None,
)
```

The writer should:

1. validate declared source artifacts and feature states;
2. write `narrative_targets.parquet`;
3. write `evidence_sources.parquet`;
4. write `evidence_refs.parquet`;
5. write deterministic claim scaffolds only when evidence refs are resolvable;
6. write `claim_evidence_links.parquet`;
7. write `narrative_sections.parquet`;
8. write `review_decisions.parquet` only when review state exists;
9. generate `narrative_manifest.json`;
10. generate `narrative_qa.json`;
11. write publication summary JSON/Markdown/HTML with a cluster-level index,
    then write the publication bundle only from accepted or not-required
    reviewed claims;
12. return paths, counts, warnings, and QA status.

The first implementation should not call an LLM. Later generators may update
claim text only if they preserve evidence refs and write generation metadata.

## Validator Utility Design

The validator should be reusable outside full result validation:

```python
validate_narrative_artifact(narrative_dir) -> NarrativeArtifactValidationResult
```

It should return:

- schema version;
- narrative id;
- status;
- artifact paths;
- target, section, claim, evidence, and review counts;
- unsupported, contradicted, weak, and aggregate-only claim counts;
- unresolved evidence refs;
- warnings and blocking issues;
- feature exposure suggestion.

Full result validation can then expose narrative states without generating or
rewriting narrative text.

## Implementation Order

0. `[x]` Add a deterministic cluster review packet writer and validator as the
   evidence packet feeding future narrative artifacts.
1. `[x]` Add schema constants and dataclasses for narrative manifests, targets,
   claims, evidence refs, links, sections, review decisions, and QA.
2. `[x]` Add `validate_narrative_artifact`.
3. `[x]` Add a tiny deterministic narrative evidence smoke fixture.
4. `[x]` Add `write_narrative_evidence_artifacts` for claim scaffolds.
5. `[x]` Extend `validate_result_root` to identify narrative manifests and expose
   stable/beta/blocked narrative states.
6. `[~]` Add a quality-gate flag that rejects unsupported normal claims.
   Result-root validation now blocks unsupported normal claims; a dedicated CLI
   flag can still make this explicit in release checks.
7. `[x]` Add `apply_narrative_generation_updates` as the safe model-assisted
   claim-text update hook that preserves evidence links and refreshes generation
   metadata plus QA.
8. `[~]` Add narrative review UI and stable generation runner integration.
   Atlas now exposes claim-level review actions and save/failure feedback; the
   CLI can render JSONL prompt batches and apply JSON/JSONL generated claim
   batches through the safe model-assisted update hook. Provider execution
   remains pending.

## Acceptance Criteria

- A narrative artifact can be validated without loading the web app.
- Every supported claim links to resolvable evidence refs.
- Unsupported or contradicted claims are blocked or shown only as caveats.
- Model-generated text cannot be marked stable without generation metadata.
- Narrative rows can cite keywords, works, co-occurrence, matrices, temporal
  events, evolution events, and QA caveats.
- `evidence=stable`, `temporal=stable`, or `evolution=stable` does not imply
  `narrative=stable`.
- The full result contract can tell whether `narrative` is hidden, beta,
  stable, or blocked from artifacts alone.

## Open Questions

- Should v1 allow narrative text at all, or only claim scaffolds and evidence
  refs?
- Should quotes/snippets be prohibited by default until source licensing and
  privacy policy are explicit?
- Should human review be required before any model-generated claim can become
  stable?
- Should claim confidence be computed by source quality, link strength, or a
  separate narrative scoring rule?
- Should relation and evolution claims require multiple independent evidence
  refs before they can be stable?
