# Keyword Rule Artifact Design

Status: v1 implemented for writer, validator, result-manifest exposure, and
keyword-pipeline auto-emission
Date: 2026-06-06

This document defines the replayable keyword cleaning rule artifact contract for
SciScape. It should come before editable cleaning UI, VOSviewer thesaurus export,
matrix comparisons by rule set, and narrative generation that depends on term
labels.

## Purpose

Keyword cleaning must be inspectable, replayable, and conservative. A result
should be able to explain:

- which raw terms were extracted;
- which deterministic rules touched each term;
- whether a rule changed, grouped, flagged, or blocked the term;
- what evidence justified the action;
- how top labels, families, and quality flags changed before and after cleaning;
- whether the cleaned display is safe for release-quality interpretation.

The artifact is a rule-and-application contract, not a free-form cleanup log.
It can feed a future Cleaning mode, workspace rule registry, VOSviewer-style
thesaurus export, matrix builder, temporal term-family tracking, and
evidence-backed narrative review.

## Safety Rules

- Preserve raw terms. A rule may hide a term from display, but must not remove
  the original raw term from the audit trail.
- Use `block` only for structurally certain artifacts such as encoded HTML,
  LaTeX preamble fragments, publisher boilerplate, or known metadata rows.
- Use `flag`, `tier_down`, or `keep_with_flag` for ambiguous stop terms, short
  tokens, acronyms, roman numerals, and domain-sensitive words.
- Treat alias, acronym, spelling, plural, and subphrase operations as
  parent-child or normalization evidence, not as silent deletion.
- Never auto-merge ambiguous acronyms without parenthetical or rule evidence.
- Never apply a domain-specific dictionary as a default global rule.
- User edits create a new rule-set version or review decision; they do not
  mutate source records or older rule artifacts.
- Rule replay must be deterministic under the same input keyword table and rule
  set.

## Artifact Location

Keyword rule artifacts should live under the result root:

```text
<result_root>/rules/<rule_set_id>/
  rule_set_manifest.json
  rules.parquet
  rule_applications.parquet
  term_before_after.parquet
  impact_summary.json
  rule_set_qa.json
```

Workspace-level reusable rule sets may later register the same manifest under:

```text
<workspace_root>/rules/<rule_set_id>/rule_set_manifest.json
```

The result-local artifact remains authoritative for a result because it records
the exact keyword table, source artifacts, and rule applications used by that
result.

## Schema Versions

- `sciscape_keyword_rule_set_manifest_v1`
- `sciscape_keyword_rules_v1`
- `sciscape_keyword_rule_applications_v1`
- `sciscape_keyword_term_before_after_v1`
- `sciscape_keyword_rule_impact_summary_v1`
- `sciscape_keyword_rule_qa_v1`

## Rule Set Manifest

`rule_set_manifest.json` is the source of truth for one replayable rule set.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_keyword_rule_set_manifest_v1` |
| `rule_set_id` | string | stable local identifier |
| `rule_type` | string | `keyword_cleaning` |
| `title` | string | human-readable title |
| `source` | string | `system`, `user`, `imported`, or `generated` |
| `version` | string | rule-set version |
| `created_at_utc` | string | creation timestamp |
| `applies_to` | object | result, cluster level, and keyword table scope |
| `source_artifacts` | array | input artifact refs and roles |
| `rules_path` | string | relative path to `rules.parquet` |
| `applications_path` | string | relative path to `rule_applications.parquet` |
| `before_after_path` | string | relative path to `term_before_after.parquet` |
| `impact_summary_ref` | string | relative path to `impact_summary.json` |
| `qa_ref` | string | relative path to `rule_set_qa.json` |
| `policy` | object | allowed actions and destructive-action limits |
| `outputs` | object | output file map |

Optional fields:

- `description`
- `parent_rule_set_id`
- `review_state`
- `warnings`
- `provenance`

`source_artifacts[].path` should be relative to the rule artifact result root
when the source file lives under that root. When the source file is produced in
a separate engine output directory, the path must be absolute so provenance
remains unambiguous.

Example:

```json
{
  "schema_version": "sciscape_keyword_rule_set_manifest_v1",
  "rule_set_id": "keyword_cleaning_default_v1",
  "rule_type": "keyword_cleaning",
  "title": "Default keyword cleaning rules",
  "source": "system",
  "version": "1.0.0",
  "created_at_utc": "2026-06-06T00:00:00+00:00",
  "applies_to": {
    "result_id": "example_result",
    "cluster_level": "cluster",
    "keyword_table": "landscape/keywords.parquet"
  },
  "source_artifacts": [
    {"artifact_ref": "keywords", "artifact_role": "keyword_table", "artifact_path": "landscape/keywords.parquet"},
    {"artifact_ref": "records", "artifact_role": "records", "artifact_path": "abstracts.parquet"}
  ],
  "rules_path": "rules.parquet",
  "applications_path": "rule_applications.parquet",
  "before_after_path": "term_before_after.parquet",
  "impact_summary_ref": "impact_summary.json",
  "qa_ref": "rule_set_qa.json",
  "policy": {
    "default_ambiguous_action": "flag",
    "allowed_destructive_families": ["artifact_block", "metadata_block", "latex_fragment", "html_fragment"],
    "preserve_raw_terms": true,
    "require_evidence_for_merge": true
  },
  "outputs": {
    "rules": "rules.parquet",
    "applications": "rule_applications.parquet",
    "before_after": "term_before_after.parquet",
    "impact_summary": "impact_summary.json",
    "qa": "rule_set_qa.json"
  }
}
```

## Rules Table

`rules.parquet` stores one row per rule.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_keyword_rules_v1` |
| `rule_set_id` | string | parent rule-set id |
| `rule_id` | string | stable rule id |
| `rule_family` | string | family listed below |
| `match_type` | string | `exact`, `normalized_exact`, `regex`, `token_pattern`, `evidence_link`, or `manual` |
| `pattern` | string | match expression or source key |
| `replacement` | string | canonical term, parent term, or empty string |
| `action` | string | allowed action listed below |
| `confidence_policy` | string | `strict`, `conservative`, or `review_required` |
| `destructive` | bool | true only when display blocking is allowed |
| `enabled` | bool | whether the rule is active |
| `created_by` | string | `system`, `user`, `imported`, or `generated` |
| `reason` | string | short explanation |

Optional fields:

- `priority`
- `scope`
- `evidence_ref`
- `source_rule_id`
- `created_at_utc`
- `updated_at_utc`
- `review_status`

Allowed `rule_family` values:

| Family | Default action | Notes |
| --- | --- | --- |
| `artifact_block` | `block` | structurally certain broken text |
| `metadata_block` | `block` | publisher/page metadata fragments |
| `latex_fragment` | `block` | LaTeX preamble or formula residue |
| `html_fragment` | `block` | tags, entities, or encoded markup |
| `stop_term` | `tier_down` | generic terms; destructive by default is not allowed |
| `alias` | `alias_to` | variant maps to canonical label |
| `acronym_expand` | `expand_to` | short form maps to long form with evidence |
| `subphrase_group` | `group_under` | shorter or derivative phrase grouped under parent |
| `spelling_normalize` | `normalize` | notation or spelling variants |
| `plural_singular` | `normalize` | plural to singular when safe |
| `tier_adjust` | `tier_down` | label tier or display priority changes |
| `review_flag` | `flag` | non-destructive review marker |

Allowed `action` values:

- `block`
- `flag`
- `normalize`
- `alias_to`
- `expand_to`
- `group_under`
- `tier_down`
- `keep_with_flag`

## Rule Applications Table

`rule_applications.parquet` stores one row for each rule-term application or
skipped decision. This is the replay audit trail.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_keyword_rule_applications_v1` |
| `rule_set_id` | string | parent rule-set id |
| `application_id` | string | stable row id |
| `rule_id` | string | applied or evaluated rule |
| `cluster_id` | int/string | cluster key |
| `raw_term` | string | original extracted term |
| `normalized_term_before` | string | term before the rule action |
| `display_label_before` | string | display label before the rule action |
| `normalized_term_after` | string | term after the rule action |
| `display_label_after` | string | display label after the rule action |
| `action` | string | rule action |
| `decision` | string | `applied`, `flagged`, `skipped`, or `blocked` |
| `evidence_type` | string | `pattern`, `parenthetical_abbreviation`, `frequency`, `subphrase`, `manual`, or `none` |
| `evidence_value` | string | compact evidence payload or pointer |
| `score_before` | float | pre-action score |
| `score_after` | float | post-action score |
| `frequency` | float/int | term frequency/count |
| `rank_before` | int/null | pre-action rank |
| `rank_after` | int/null | post-action rank |

Optional fields:

- `term_uid`
- `doc_support`
- `cluster_support`
- `review_status`
- `warning`

## Term Before/After Table

`term_before_after.parquet` is the compact table for UI, reports, and impact
review. It can be regenerated from the keyword table and application log.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | `sciscape_keyword_term_before_after_v1` |
| `rule_set_id` | string | parent rule-set id |
| `cluster_id` | int/string | cluster key |
| `raw_term` | string | original term |
| `term_before` | string | normalized/display term before rules |
| `term_after` | string | normalized/display term after rules |
| `display_label` | string | final display label |
| `family_id` | string/null | term family id |
| `parent_term` | string/null | parent term when grouped |
| `variant_count` | int | represented raw/variant terms |
| `rule_ids` | string | pipe-delimited or JSON-encoded rule ids |
| `quality_flags` | string | existing and rule-derived quality flags |
| `review_status` | string | `auto`, `needs_review`, `approved`, or `rejected` |
| `tier_before` | string | pre-rule display tier |
| `tier_after` | string | post-rule display tier |
| `blocked` | bool | hidden from display |
| `block_reason` | string/null | reason when blocked |

## Impact Summary

`impact_summary.json` should summarize rule-set effects without requiring users
to scan full parquet tables.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_keyword_rule_impact_summary_v1` |
| `rule_set_id` | parent rule-set id |
| `counts` | total terms, changed terms, blocked terms, flagged terms, grouped terms |
| `rule_family_counts` | counts by rule family |
| `action_counts` | counts by action |
| `rank_change_summary` | top-label and rank movement summary |
| `contamination_summary` | before/after artifact counts |
| `examples` | bounded examples for blocked, flagged, grouped, and changed terms |

Useful example buckets:

- `top_blocked_artifacts`
- `top_flagged_ambiguous_terms`
- `largest_alias_groups`
- `largest_subphrase_groups`
- `top10_changed_clusters`
- `top10_before_after_labels`

## Rule Set QA

`rule_set_qa.json` determines whether the rule artifact is safe to expose.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `sciscape_keyword_rule_qa_v1` |
| `rule_set_id` | parent rule-set id |
| `status` | `passed`, `warning`, or `blocked` |
| `checks` | per-check results |
| `counts` | rows, rules, applications, warnings, blocking issues |
| `contamination_counts` | top-label and all-row artifact counts before/after |
| `destructive_action_counts` | destructive actions by rule family |
| `unresolved_rule_ids` | rule ids referenced by outputs but missing from rules |
| `warnings` | non-blocking warnings |
| `blocking_issues` | blocking issues |

Minimum checks:

- manifest exists and uses the expected schema version;
- all declared output files exist;
- tables have required columns;
- `rule_ids` referenced by before/after rows resolve to `rules.parquet`;
- raw terms are recoverable for every before/after row;
- destructive actions appear only in allowed destructive families;
- `block` is not used for ambiguous stop terms, acronyms, or subphrase groups;
- evidence-bearing actions have evidence values or explicit review status;
- top-N display labels contain no HTML, LaTeX preamble, or publisher metadata
  artifacts after cleaning;
- before/after row counts can be explained by application decisions.

Status mapping:

| Condition | Feature state effect |
| --- | --- |
| QA passed and keyword table exists | `keyword=stable`, `quality=stable` |
| QA warning only | `keyword=beta` or `stable` with warnings, depending on top-label contamination |
| QA blocked | `keyword` must not be promoted by the rule artifact; `quality` reports blocking issue |
| Rule manifest exists but no applications yet | rule artifact is inspectable, but does not prove cleaning was applied |

## Keyword Table Trace Fields

When rule artifacts are applied, `landscape/keywords.parquet` or the cleaned
keyword table should preserve these optional trace fields:

| Field | Meaning |
| --- | --- |
| `raw_term` | original extracted term |
| `normalized_term` | deterministic normalized term |
| `display_label` | final display label |
| `family_id` | term-family id |
| `parent_term` | parent phrase or expansion target |
| `variant_count` | number of raw aliases/children represented |
| `rule_ids` | rule ids that affected or flagged the term |
| `cleaning_action` | final dominant action |
| `review_status` | review state |
| `blocked` | hidden from display |
| `block_reason` | reason when blocked |

Outputs without these columns remain readable as legacy keyword artifacts, but
they do not prove replayable cleaning.

## Result Manifest Integration

Validated rule artifacts should appear in `result_manifest.json` as artifact
refs:

- `keyword_rules`: `rules/<rule_set_id>/rule_set_manifest.json`
- `keyword_rule_applications`: `rules/<rule_set_id>/rule_applications.parquet`
- `keyword_rule_before_after`: `rules/<rule_set_id>/term_before_after.parquet`
- `keyword_rule_qa`: `rules/<rule_set_id>/rule_set_qa.json`

Feature refs should be updated conservatively:

- `features.keyword.artifact_refs` may include `keywords` and `keyword_rules`;
- `features.quality.artifact_refs` may include `artifact_contract`,
  `keyword_rule_qa`, and other validated QA artifacts such as narrative or
  review-packet QA;
- `features.export.artifact_refs` may include `keyword_rules` when exporting
  thesaurus or rule-set files.

The rule artifact must not override the keyword feature state if the keyword
table itself is missing, malformed, or contaminated.

## Workspace Integration

Workspace rule-set manifests may register reusable rule sets, but result-local
rule artifacts should record the actual applications and impact for a specific
run. A workspace rule set can be referenced by:

- matrix artifacts, when matrices are built after cleaning;
- temporal artifacts, when term families are tracked over time;
- narrative artifacts, when claims depend on cleaned labels;
- export artifacts, when VOSviewer thesaurus or rule-set files are produced.

## Writer Contract

The first implementation should expose a deterministic writer:

```python
write_keyword_rule_artifacts(
    result_root,
    *,
    rule_set_id,
    keywords,
    rules,
    applications=None,
    source_artifacts=None,
    output_dir=None,
) -> dict
```

The writer should:

1. validate source keyword rows;
2. normalize rule ids and paths;
3. write `rules.parquet`;
4. write or derive `rule_applications.parquet`;
5. write `term_before_after.parquet`;
6. write `impact_summary.json`;
7. write `rule_set_manifest.json`;
8. validate the full rule artifact;
9. write `rule_set_qa.json`;
10. return manifest and QA paths.

## Validator Contract

The first validator should expose:

```python
validate_keyword_rule_artifact(rule_dir_or_manifest) -> KeywordRuleValidationResult
```

The validation result should report:

- rule set id;
- manifest path;
- status;
- row counts;
- rule family counts;
- action counts;
- contamination counts;
- missing files;
- missing columns;
- unresolved rule ids;
- warnings;
- blocking issues.

Full result-root validation can then expose keyword rule artifacts without
rewriting keywords.

## Implementation Order

Completed:

- `sciscape.artifacts` defines schema constants, `KeywordRuleValidationResult`,
  `write_keyword_rule_artifacts`, and `validate_keyword_rule_artifact`.
- `validate_result_root` and `build_result_manifest` expose `keyword_rules`,
  `keyword_rule_qa`, contamination counts, and quality gate paths.
- `sciscape.keyword_extraction.rule_artifact` converts keyword quality columns
  into rule, application, and before/after tables.
- Review-only quality flags are emitted as separate non-destructive rules, for
  example `quality_review_short_form` and
  `quality_review_review_short_form`, so applications keep compact evidence
  pointers without increasing destructive filtering.
- Legacy keyword extraction writes the artifact when
  `keyword_rule_result_root` is configured.
- The cluster-sharded keyword engine writes the artifact under its output root
  by default and records absolute source paths when artifacts live outside the
  selected result root.

Remaining:

1. Add optional source trace columns to keyword outputs where quality decisions
   still need document-level evidence pointers.
2. Add VOSviewer thesaurus/rule-set export adapters after the rule artifact is
   stable.
3. Add editable Cleaning mode and workspace-level reusable rule registry.

Historical order:

1. Add schema constants and lightweight validation result classes in
   `sciscape/artifacts.py`.
2. Add `write_keyword_rule_artifacts`.
3. Add `validate_keyword_rule_artifact`.
4. Add a tiny contaminated fixture that exercises HTML, LaTeX, metadata,
   ambiguous acronym, and subphrase cases.
5. Extend `validate_result_root` and `build_result_manifest` to expose
   `keyword_rules`, `keyword_rule_qa`, and related feature states.
6. Keep existing keyword extraction behavior unchanged until rule artifacts can
   be written and validated.
7. Add optional trace columns to keyword outputs.
8. Add VOSviewer thesaurus/rule-set export adapters after the rule artifact is
   stable.

## Acceptance Criteria

- A rule artifact can be validated without loading the web app.
- The writer preserves raw terms and records every automatic display-changing
  action.
- Ambiguous terms are flagged or tiered down rather than blocked.
- Destructive actions are limited to structurally certain artifact families.
- Top-ranked HTML, LaTeX, and publisher metadata artifacts are blocked in the
  contaminated fixture.
- `result_manifest.json` can tell whether replayable keyword cleaning is
  hidden, beta, stable, or blocked from artifacts alone.
- Web Download can expose the manifest, rule table, applications, before/after
  table, impact summary, and QA via manifest-backed file inventory.

## Open Questions

- Should imported VOSviewer thesaurus files be represented as `rules.parquet`
  rows directly, or as a separate imported source sidecar?
- Should user review decisions live inside this rule artifact or in a future
  workspace review-decision artifact?
- Should `stop_term` rules be allowed to block within a specific user-created
  rule set, or should they always remain `tier_down` plus review in v1?
