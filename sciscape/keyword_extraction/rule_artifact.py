"""Build replayable keyword-cleaning rule artifacts from keyword tables."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from sciscape.artifacts import write_keyword_rule_artifacts

from .utils import _looks_like_metadata_artifact_term


_FLAG_SEPARATOR = "|"

_REVIEW_FLAGS = frozenset(
    {
        "artifact_formula",
        "compact_formula_fragment",
        "dimension_fragment",
        "mixed_formula_fragment",
        "unresolved_compact_short_form",
        "shape_only",
        "short_form",
        "ambiguous_short_form",
    }
)


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _flag_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    text = str(value).strip().lower()
    if not text:
        return set()
    return {part.strip() for part in text.replace(",", _FLAG_SEPARATOR).split(_FLAG_SEPARATOR) if part.strip()}


def _jsonish_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [_clean_text(item) for item in parsed if _clean_text(item)]
    return [text]


def _add_rule(
    rules: OrderedDict[str, dict[str, Any]],
    *,
    rule_id: str,
    rule_family: str,
    match_type: str,
    pattern: str,
    replacement: str = "",
    action: str,
    confidence_policy: str,
    destructive: bool,
    reason: str,
) -> str:
    rules.setdefault(
        rule_id,
        {
            "rule_id": rule_id,
            "rule_family": rule_family,
            "match_type": match_type,
            "pattern": pattern,
            "replacement": replacement,
            "action": action,
            "confidence_policy": confidence_policy,
            "destructive": bool(destructive),
            "enabled": True,
            "created_by": "sciscape",
            "reason": reason,
        },
    )
    return rule_id


def _artifact_rule_family(term: str, flags: set[str]) -> str:
    lowered = term.lower()
    if any(token in lowered for token in ("htmlview", "<div", "</div", "lt div", "gt lt", "class html")):
        return "html_fragment"
    if any(token in lowered for token in ("usepackage", "begin equation", "end equation", "documentclass")):
        return "latex_fragment"
    if "metadata_fragment" in flags:
        return "metadata_block"
    return "artifact_block"


def _artifact_rule_id(family: str) -> str:
    if family == "html_fragment":
        return "html_fragment_block"
    if family == "latex_fragment":
        return "latex_fragment_block"
    if family == "metadata_block":
        return "metadata_fragment_block"
    return "artifact_shape_block"


def _rule_slug(value: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


def _score_value(row: Mapping[str, Any], *columns: str) -> Any:
    for column in columns:
        value = row.get(column)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return None


def _rank_value(row: Mapping[str, Any]) -> Any:
    return _score_value(row, "representative_rank", "rank")


def build_keyword_rule_artifact_inputs(
    keywords: pd.DataFrame,
    *,
    rule_set_id: str = "keyword_cleaning_default_v1",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert a keyword table into rule, application, and before/after tables.

    The adapter is intentionally conservative: only structurally certain
    metadata/HTML/LaTeX artifacts become destructive ``block`` applications.
    Ambiguous quality risks remain visible through non-destructive flags.
    """

    if keywords is None:
        keywords = pd.DataFrame()
    df = keywords.copy()
    rules: OrderedDict[str, dict[str, Any]] = OrderedDict()
    applications: list[dict[str, Any]] = []
    before_after: list[dict[str, Any]] = []

    for index, row_series in df.iterrows():
        row = row_series.to_dict()
        cluster_id = row.get("cluster_id", "")
        term = _clean_text(row.get("term"))
        raw_term = _clean_text(row.get("raw_term")) or term
        display_label = _clean_text(row.get("display_label")) or term
        normalized_term = _clean_text(row.get("normalized_term")) or term
        flags = _flag_set(row.get("quality_flags"))
        tier_before = _clean_text(row.get("keyword_label_tier")) or _clean_text(row.get("tier")) or "candidate"
        tier_after = tier_before
        review_status = "accepted"
        blocked = False
        block_reason = ""
        applied_rules: list[tuple[str, str]] = []

        if term and (_looks_like_metadata_artifact_term(term) or "metadata_fragment" in flags):
            family = _artifact_rule_family(term, flags)
            rule_id = _add_rule(
                rules,
                rule_id=_artifact_rule_id(family),
                rule_family=family,
                match_type="detector",
                pattern=family,
                action="block",
                confidence_policy="strict",
                destructive=True,
                reason="structurally certain keyword artifact detected",
            )
            blocked = True
            block_reason = family
            applied_rules.append((rule_id, block_reason))
            review_status = "blocked"
            tier_after = "drop"

        if not blocked and (flags & _REVIEW_FLAGS or tier_before.startswith("review_")):
            review_evidence = [flag for flag in sorted(flags & _REVIEW_FLAGS)]
            if tier_before.startswith("review_"):
                review_evidence.append(f"tier:{tier_before}")
            for evidence in review_evidence:
                evidence_kind, _, evidence_value = evidence.partition(":")
                if not evidence_value:
                    evidence_kind = "quality_flag"
                    evidence_value = evidence
                rule_id = _add_rule(
                    rules,
                    rule_id=f"quality_review_{_rule_slug(evidence_value)}",
                    rule_family="review_flag",
                    match_type=evidence_kind,
                    pattern=evidence_value,
                    action="keep_with_flag",
                    confidence_policy="review_required",
                    destructive=False,
                    reason=f"ambiguous quality risk kept for review: {evidence_value}",
                )
                applied_rules.append((rule_id, evidence_value))
            review_status = "needs_review"

        merged_from = [item for item in _jsonish_list(row.get("norm_merged_from")) if item and item != term]
        if merged_from:
            rule_id = _add_rule(
                rules,
                rule_id="normalization_group_members",
                rule_family="subphrase_group",
                match_type="evidence_link",
                pattern="norm_merged_from",
                replacement=term,
                action="group_under",
                confidence_policy="conservative",
                destructive=False,
                reason="term has explicit normalization/grouping evidence",
            )
            applied_rules.append((rule_id, _FLAG_SEPARATOR.join(merged_from)))

        abbreviation_target = _clean_text(row.get("abbreviation_target"))
        abbreviation_status = _clean_text(row.get("abbreviation_status"))
        if abbreviation_target and abbreviation_target != term:
            rule_id = _add_rule(
                rules,
                rule_id="abbreviation_evidence_expand",
                rule_family="acronym_expand",
                match_type="evidence_link",
                pattern="abbreviation_target",
                replacement=abbreviation_target,
                action="expand_to",
                confidence_policy="review_required" if abbreviation_status.startswith("ambiguous") else "conservative",
                destructive=False,
                reason="parenthetical or corpus abbreviation evidence is available",
            )
            evidence = f"{term}->{abbreviation_target}"
            if abbreviation_status:
                evidence = f"{evidence};status={abbreviation_status}"
            applied_rules.append((rule_id, evidence))

        if raw_term and raw_term != term and not blocked:
            rule_id = _add_rule(
                rules,
                rule_id="surface_normalization",
                rule_family="spelling_normalize",
                match_type="evidence_link",
                pattern="raw_term_to_term",
                replacement=term,
                action="normalize",
                confidence_policy="conservative",
                destructive=False,
                reason="raw term differs from normalized keyword display term",
            )
            applied_rules.append((rule_id, f"{raw_term}->{term}"))

        term_after = "" if blocked else term
        display_after = "" if blocked else display_label
        applied_rule_ids = [rule_id for rule_id, _ in applied_rules]
        before_after.append(
            {
                "rule_set_id": rule_set_id,
                "cluster_id": cluster_id,
                "raw_term": raw_term,
                "term_before": raw_term,
                "term_after": term_after,
                "display_label": display_after,
                "family_id": normalized_term or term_after,
                "parent_term": abbreviation_target if abbreviation_target and abbreviation_target != term else "",
                "variant_count": max(1, len(merged_from) + 1),
                "rule_ids": _FLAG_SEPARATOR.join(applied_rule_ids),
                "quality_flags": _FLAG_SEPARATOR.join(sorted(flags)),
                "review_status": review_status,
                "tier_before": tier_before,
                "tier_after": tier_after,
                "blocked": blocked,
                "block_reason": block_reason,
            }
        )

        for rule_id, evidence_value in applied_rules:
            rule = rules[rule_id]
            applications.append(
                {
                    "rule_set_id": rule_set_id,
                    "application_id": f"app_{index + 1:08d}_{len(applications) + 1:04d}",
                    "rule_id": rule_id,
                    "cluster_id": cluster_id,
                    "raw_term": raw_term,
                    "normalized_term_before": raw_term,
                    "display_label_before": raw_term,
                    "normalized_term_after": term_after,
                    "display_label_after": display_after,
                    "action": rule["action"],
                    "decision": "blocked" if blocked and rule["action"] == "block" else "applied",
                    "evidence_type": rule["match_type"],
                    "evidence_value": evidence_value or block_reason or rule["pattern"],
                    "score_before": _score_value(row, "score", "quality_score"),
                    "score_after": _score_value(row, "quality_score", "score"),
                    "frequency": _score_value(row, "frequency", "doc_coverage"),
                    "rank_before": _rank_value(row),
                    "rank_after": None if blocked else _rank_value(row),
                }
            )

    rules_df = pd.DataFrame(list(rules.values()))
    applications_df = pd.DataFrame(applications)
    before_after_df = pd.DataFrame(before_after)
    return rules_df, applications_df, before_after_df


def write_keyword_cleaning_rule_artifacts(
    result_root: str | Path,
    *,
    keywords: pd.DataFrame | None = None,
    rule_set_id: str = "keyword_cleaning_default_v1",
    output_dir: str | Path | None = None,
    source_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write keyword cleaning rule artifacts from a keyword table or result root."""

    if keywords is None:
        root = Path(result_root)
        candidates = [root / "landscape" / "keywords.parquet", root / "keywords.parquet"]
        for candidate in candidates:
            if candidate.exists():
                keywords = pd.read_parquet(candidate)
                break
    if keywords is None:
        keywords = pd.DataFrame()

    rules, applications, before_after = build_keyword_rule_artifact_inputs(
        keywords,
        rule_set_id=rule_set_id,
    )
    return write_keyword_rule_artifacts(
        result_root,
        rule_set_id=rule_set_id,
        rules=rules,
        applications=applications,
        before_after=before_after,
        keywords=keywords,
        output_dir=output_dir,
        source_artifacts=source_artifacts,
        transforms=[{"step": "build_keyword_rule_artifact_inputs", "source": "keyword_table_quality_columns"}],
    )


__all__ = [
    "build_keyword_rule_artifact_inputs",
    "write_keyword_cleaning_rule_artifacts",
]
