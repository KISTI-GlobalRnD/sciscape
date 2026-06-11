"""Cluster evolution analysis utilities.

This module computes evolution tables from records, membership, and optional
keyword evidence. Artifact writing and validation stay in ``sciscape.artifacts``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


EVOLUTION_TIME_SLICES_SCHEMA_VERSION = "sciscape_evolution_time_slices_v1"
EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION = "sciscape_evolution_cluster_states_v1"
EVOLUTION_TRANSITIONS_SCHEMA_VERSION = "sciscape_evolution_transitions_v1"
EVOLUTION_LINEAGES_SCHEMA_VERSION = "sciscape_evolution_lineages_v1"
EVOLUTION_EVENTS_SCHEMA_VERSION = "sciscape_evolution_events_v1"

REQUIRED_EVOLUTION_STATE_COLUMNS = {
    "schema_version",
    "evolution_id",
    "state_id",
    "slice_id",
    "slice_index",
    "cluster_key",
    "cluster_label",
    "doc_count",
    "term_count",
    "top_terms",
}
REQUIRED_EVOLUTION_TRANSITION_COLUMNS = {
    "schema_version",
    "evolution_id",
    "transition_id",
    "source_state_id",
    "target_state_id",
    "source_slice_id",
    "target_slice_id",
    "metric",
    "score",
    "support_count",
    "source_doc_count",
    "target_doc_count",
    "relation",
}
REQUIRED_EVOLUTION_LINEAGE_COLUMNS = {
    "schema_version",
    "evolution_id",
    "lineage_id",
    "state_id",
    "slice_id",
    "slice_index",
    "role",
    "stability_score",
}
REQUIRED_EVOLUTION_EVENT_COLUMNS = {
    "schema_version",
    "evolution_id",
    "event_id",
    "event_type",
    "slice_id",
    "state_id",
    "lineage_id",
    "transition_refs",
    "score",
    "support_count",
    "method",
}


@dataclass(frozen=True)
class EvolutionAnalysisResult:
    """In-memory evolution analysis output ready for artifact serialization."""

    evolution_id: str
    slices: pd.DataFrame
    states: pd.DataFrame
    transitions: pd.DataFrame
    lineages: pd.DataFrame
    events: pd.DataFrame
    matching_method: dict[str, Any]
    event_rules: dict[str, Any]
    periodization: dict[str, Any]
    entity_scope: dict[str, Any]
    metrics: list[dict[str, Any]]
    transforms: list[dict[str, Any]]


def _safe_id(value: object, *, fallback: str = "result") -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_.:-")
    return safe or fallback


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _config_int(config: Mapping[str, Any] | None, key: str, default: int, *, label: str) -> int:
    raw = config.get(key, default) if config else default
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} {key} must be an integer") from exc
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{label} {key} must be an integer")
    return int(number)


def _validate_yearly_periodization(periodization: Mapping[str, Any] | None) -> None:
    if not periodization:
        return
    unit = str(periodization.get("unit", "year")).strip().lower()
    if unit != "year":
        raise ValueError("evolution analysis currently supports only unit='year'")
    window_years = _config_int(periodization, "window_years", 1, label="evolution periodization")
    step_years = _config_int(periodization, "step_years", 1, label="evolution periodization")
    if window_years != 1:
        raise ValueError("evolution analysis currently supports only window_years=1")
    if step_years != 1:
        raise ValueError("evolution analysis currently supports only step_years=1")
    if bool(periodization.get("include_unknown_year", False)):
        raise ValueError("evolution analysis currently supports only include_unknown_year=False")


def _normalize_matching_method(matching_method: Mapping[str, Any]) -> dict[str, Any]:
    metric = str(matching_method.get("metric") or "projected_cluster_identity").strip()
    if metric != "projected_cluster_identity":
        raise ValueError(
            "unsupported evolution matching metric for membership projection: "
            f"{metric}. Use projected_cluster_identity."
        )
    try:
        threshold = float(matching_method.get("min_transition_score", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("evolution matching min_transition_score must be a number") from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("evolution matching min_transition_score must be between 0 and 1")
    min_support = _config_int(matching_method, "min_support_count", 1, label="evolution matching")
    if min_support < 1:
        raise ValueError("evolution matching min_support_count must be at least 1")
    normalized = dict(matching_method)
    normalized.update(
        {
            "metric": metric,
            "min_transition_score": threshold,
            "min_support_count": min_support,
        }
    )
    return normalized


def _year_column(records: pd.DataFrame) -> str | None:
    for column in ("pubyear", "year", "publication_year"):
        if column in records.columns:
            return column
    return None


def _uid_column(records: pd.DataFrame) -> str | None:
    for column in ("uid", "work_id", "paper_id", "id"):
        if column in records.columns:
            return column
    return None


def _cluster_columns(membership: pd.DataFrame | None) -> list[str]:
    if membership is None or membership.empty:
        return []
    return [column for column in membership.columns if column == "cluster" or column.startswith("cluster_")]


def _cluster_column(membership: pd.DataFrame) -> str | None:
    columns = _cluster_columns(membership)
    if "cluster" in columns:
        return "cluster"
    return sorted(columns)[0] if columns else None


def _level_from_cluster_column(column: str) -> str:
    if column == "cluster":
        return "cluster"
    if column.startswith("cluster_"):
        return column.removeprefix("cluster_")
    return column


def _keyword_label_column(columns: list[str]) -> str:
    for name in ("term", "label", "keyword", "term_label"):
        if name in columns:
            return name
    return columns[0] if columns else "term"


def _build_time_slices(evolution_id: str, years: list[int], periodization: Mapping[str, Any] | None) -> pd.DataFrame:
    if not years:
        raise ValueError("evolution analysis requires at least one valid publication year")
    _validate_yearly_periodization(periodization)
    start_year = _config_int(periodization, "start_year", min(years), label="evolution periodization")
    end_year = _config_int(periodization, "end_year", max(years), label="evolution periodization")
    if end_year < start_year:
        raise ValueError("evolution end_year must be greater than or equal to start_year")
    rows = []
    for index, year in enumerate(range(start_year, end_year + 1)):
        rows.append(
            {
                "schema_version": EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "slice_id": f"year:{year}",
                "slice_index": int(index),
                "slice_label": str(year),
                "start_year": int(year),
                "end_year": int(year),
                "unit": "year",
                "doc_count": 0,
                "edge_count": None,
                "active_cluster_count": 0,
                "unknown_year_count": 0,
                "warning_flags": "",
            }
        )
    return pd.DataFrame(rows)


def _state_id(slice_id: str, cluster_key: str) -> str:
    return _safe_id(f"{slice_id}_{cluster_key}", fallback="state")


def _lineage_id(cluster_key: str) -> str:
    return _safe_id(f"lineage_{cluster_key}", fallback="lineage")


def _cluster_label_map(keywords: pd.DataFrame | None, cluster_column: str) -> dict[str, dict[str, Any]]:
    if keywords is None or keywords.empty or "cluster_id" not in keywords.columns:
        return {}
    label_col = _keyword_label_column(list(keywords.columns))
    rows: dict[str, dict[str, Any]] = {}
    for cluster_id, group in keywords.groupby("cluster_id", dropna=False, sort=True):
        terms = [str(value).strip() for value in group[label_col].dropna().head(5).tolist() if str(value).strip()]
        key = f"{_level_from_cluster_column(cluster_column)}:{cluster_id}"
        rows[key] = {
            "label": terms[0] if terms else key,
            "top_terms": terms,
            "term_count": int(len(terms)),
        }
    return rows


def _state_rows(
    evolution_id: str,
    records: pd.DataFrame,
    membership: pd.DataFrame,
    slices: pd.DataFrame,
    *,
    year_column: str,
    uid_column: str,
    cluster_column: str,
    keywords: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    level = _level_from_cluster_column(cluster_column)
    label_map = _cluster_label_map(keywords, cluster_column)
    work = records[[uid_column, year_column]].dropna(subset=[uid_column]).copy()
    work[uid_column] = work[uid_column].map(str)
    work["_evolution_year"] = pd.to_numeric(work[year_column], errors="coerce")
    work = work.dropna(subset=["_evolution_year"]).drop_duplicates(subset=[uid_column, "_evolution_year"])
    mem = membership[["uid", cluster_column]].dropna(subset=["uid", cluster_column]).copy()
    mem["uid"] = mem["uid"].map(str)
    mem = mem.drop_duplicates(subset=["uid", cluster_column])
    joined = work.merge(mem, left_on=uid_column, right_on="uid", how="inner")
    year_to_slice = {
        int(row.start_year): (str(row.slice_id), int(row.slice_index))
        for row in slices.itertuples(index=False)
    }
    rows = []
    state_docs: dict[str, set[str]] = {}
    grouped = (
        joined.dropna(subset=["_evolution_year", cluster_column])
        .groupby(["_evolution_year", cluster_column], sort=True)
        .agg(work_ids=(uid_column, lambda values: sorted(set(map(str, values)))))
        .reset_index()
    )
    for _, item in grouped.iterrows():
        year = _coerce_int(item["_evolution_year"])
        if year is None or year not in year_to_slice:
            continue
        cluster_id = str(item[cluster_column])
        cluster_key = f"{level}:{cluster_id}"
        slice_id, slice_index = year_to_slice[year]
        state_id = _state_id(slice_id, cluster_key)
        label_info = label_map.get(cluster_key, {})
        top_terms = label_info.get("top_terms") or []
        doc_ids = set(item["work_ids"])
        state_docs[state_id] = doc_ids
        rows.append(
            {
                "schema_version": EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "state_id": state_id,
                "slice_id": slice_id,
                "slice_index": int(slice_index),
                "cluster_key": cluster_key,
                "cluster_label": str(label_info.get("label") or cluster_key),
                "doc_count": int(len(doc_ids)),
                "term_count": int(label_info.get("term_count") or len(top_terms)),
                "top_terms": json.dumps(top_terms, ensure_ascii=True),
                "cluster_uid": cluster_key,
                "cluster_id": cluster_id,
                "level": level,
                "representative_work_ids": json.dumps(sorted(doc_ids), ensure_ascii=True),
                "source_cluster_key": cluster_key,
                "warning_flags": "",
            }
        )
    states = pd.DataFrame(rows)
    if states.empty:
        states = pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_STATE_COLUMNS))
    return states, state_docs


def _update_slice_counts(slices: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    out = slices.copy()
    if states.empty:
        return out
    counts = states.groupby("slice_id").agg(doc_count=("doc_count", "sum"), active_cluster_count=("state_id", "count"))
    for index, row in out.iterrows():
        slice_id = row["slice_id"]
        if slice_id in counts.index:
            out.at[index, "doc_count"] = int(counts.loc[slice_id, "doc_count"])
            out.at[index, "active_cluster_count"] = int(counts.loc[slice_id, "active_cluster_count"])
    return out


def _transition_rows(
    evolution_id: str,
    slices: pd.DataFrame,
    states: pd.DataFrame,
    state_docs: Mapping[str, set[str]],
    *,
    matching_method: Mapping[str, Any],
) -> pd.DataFrame:
    if states.empty or len(slices) < 2:
        return pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_TRANSITION_COLUMNS))
    matching = _normalize_matching_method(matching_method)
    metric = str(matching["metric"])
    threshold = float(matching["min_transition_score"])
    min_support = int(matching["min_support_count"])
    states_by_slice = {slice_id: group.copy() for slice_id, group in states.groupby("slice_id", sort=False)}
    ordered_slices = slices.sort_values("slice_index", kind="stable")
    rows = []
    for source_slice, target_slice in zip(ordered_slices.iloc[:-1].itertuples(index=False), ordered_slices.iloc[1:].itertuples(index=False)):
        source_states = states_by_slice.get(str(source_slice.slice_id), pd.DataFrame())
        target_states = states_by_slice.get(str(target_slice.slice_id), pd.DataFrame())
        if source_states.empty or target_states.empty:
            continue
        for source in source_states.itertuples(index=False):
            for target in target_states.itertuples(index=False):
                same_cluster = str(source.cluster_key) == str(target.cluster_key)
                score = 1.0 if same_cluster else 0.0
                support = min(int(source.doc_count), int(target.doc_count)) if same_cluster else 0
                if score < threshold or support < min_support:
                    continue
                transition_id = _safe_id(f"{source.state_id}_to_{target.state_id}_{metric}", fallback="transition")
                rows.append(
                    {
                        "schema_version": EVOLUTION_TRANSITIONS_SCHEMA_VERSION,
                        "evolution_id": evolution_id,
                        "transition_id": transition_id,
                        "source_state_id": str(source.state_id),
                        "target_state_id": str(target.state_id),
                        "source_slice_id": str(source.slice_id),
                        "target_slice_id": str(target.slice_id),
                        "metric": metric,
                        "score": float(score),
                        "support_count": int(support),
                        "source_doc_count": int(source.doc_count),
                        "target_doc_count": int(target.doc_count),
                        "relation": "continuation",
                        "shared_doc_count": int(support),
                        "rank_from_source": 1,
                        "rank_to_target": 1,
                        "warning_flags": "",
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_TRANSITION_COLUMNS))


def _lineage_rows(evolution_id: str, states: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_LINEAGE_COLUMNS))
    incoming_scores: dict[str, list[float]] = {}
    outgoing_scores: dict[str, list[float]] = {}
    if not transitions.empty:
        for row in transitions.itertuples(index=False):
            incoming_scores.setdefault(str(row.target_state_id), []).append(float(row.score))
            outgoing_scores.setdefault(str(row.source_state_id), []).append(float(row.score))
    rows = []
    for cluster_key, group in states.sort_values(["cluster_key", "slice_index"], kind="stable").groupby("cluster_key", sort=True):
        ordered = group.sort_values("slice_index", kind="stable")
        lineage_id = _lineage_id(str(cluster_key))
        for offset, row in enumerate(ordered.itertuples(index=False)):
            if len(ordered) == 1:
                role = "singleton"
            elif offset == 0:
                role = "root"
            elif offset == len(ordered) - 1:
                role = "terminal"
            else:
                role = "continuation"
            scores = incoming_scores.get(str(row.state_id), []) + outgoing_scores.get(str(row.state_id), [])
            stability = float(sum(scores) / len(scores)) if scores else 1.0
            rows.append(
                {
                    "schema_version": EVOLUTION_LINEAGES_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "lineage_id": lineage_id,
                    "state_id": str(row.state_id),
                    "slice_id": str(row.slice_id),
                    "slice_index": int(row.slice_index),
                    "role": role,
                    "stability_score": max(0.0, min(1.0, stability)),
                    "root_state_id": str(ordered.iloc[0]["state_id"]),
                    "lineage_label": str(row.cluster_label),
                    "event_refs": "[]",
                    "warning_flags": "",
                }
            )
    return pd.DataFrame(rows)


def _lineage_by_state(lineages: pd.DataFrame) -> dict[str, str]:
    if lineages.empty or "state_id" not in lineages.columns:
        return {}
    return {str(row.state_id): str(row.lineage_id) for row in lineages.itertuples(index=False)}


def _event_rows_from_graph(
    evolution_id: str,
    slices: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    lineages: pd.DataFrame,
    *,
    event_rules: Mapping[str, Any],
) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_EVENT_COLUMNS))
    continuation_min = float(event_rules.get("continuation_min_score", 0.5))
    ambiguous_margin = float(event_rules.get("ambiguous_score_margin", 0.05))
    lineage_lookup = _lineage_by_state(lineages)
    last_slice_index = int(slices["slice_index"].max()) if not slices.empty else 0
    incoming: dict[str, list[Any]] = {}
    outgoing: dict[str, list[Any]] = {}
    if not transitions.empty:
        for row in transitions.itertuples(index=False):
            if float(row.score) < continuation_min:
                continue
            outgoing.setdefault(str(row.source_state_id), []).append(row)
            incoming.setdefault(str(row.target_state_id), []).append(row)
    rows = []
    for row in transitions.itertuples(index=False) if not transitions.empty else []:
        if str(row.relation) != "continuation" or float(row.score) < continuation_min:
            continue
        event_id = _safe_id(f"continuation_{row.transition_id}", fallback="continuation_event")
        rows.append(
            {
                "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "event_id": event_id,
                "event_type": "continuation",
                "slice_id": str(row.target_slice_id),
                "state_id": str(row.target_state_id),
                "lineage_id": lineage_lookup.get(str(row.target_state_id)),
                "transition_refs": json.dumps([str(row.transition_id)], ensure_ascii=True),
                "score": float(row.score),
                "support_count": int(row.support_count),
                "method": "projected_membership_transition",
                "source_state_ids": json.dumps([str(row.source_state_id)], ensure_ascii=True),
                "target_state_ids": json.dumps([str(row.target_state_id)], ensure_ascii=True),
                "event_label": "Continuation",
                "warning_flags": "",
            }
        )
    for state in states.itertuples(index=False):
        state_id = str(state.state_id)
        slice_index = int(state.slice_index)
        in_rows = incoming.get(state_id, [])
        out_rows = outgoing.get(state_id, [])
        if slice_index > 0 and not in_rows:
            rows.append(
                {
                    "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "event_id": _safe_id(f"emergence_{state_id}", fallback="emergence_event"),
                    "event_type": "emergence",
                    "slice_id": str(state.slice_id),
                    "state_id": state_id,
                    "lineage_id": lineage_lookup.get(state_id),
                    "transition_refs": "[]",
                    "score": 1.0,
                    "support_count": int(state.doc_count),
                    "method": "no_incoming_transition_above_threshold",
                    "source_state_ids": "[]",
                    "target_state_ids": json.dumps([state_id], ensure_ascii=True),
                    "event_label": "Emergence",
                    "warning_flags": "",
                }
            )
        if slice_index < last_slice_index and not out_rows:
            rows.append(
                {
                    "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "event_id": _safe_id(f"decline_{state_id}", fallback="decline_event"),
                    "event_type": "decline",
                    "slice_id": str(state.slice_id),
                    "state_id": state_id,
                    "lineage_id": lineage_lookup.get(state_id),
                    "transition_refs": "[]",
                    "score": 1.0,
                    "support_count": int(state.doc_count),
                    "method": "no_outgoing_transition_above_threshold",
                    "source_state_ids": json.dumps([state_id], ensure_ascii=True),
                    "target_state_ids": "[]",
                    "event_label": "Decline",
                    "warning_flags": "",
                }
            )
        if len(out_rows) >= 2:
            scores = sorted((float(item.score) for item in out_rows), reverse=True)
            if len(scores) >= 2 and abs(scores[0] - scores[1]) <= ambiguous_margin:
                rows.append(
                    {
                        "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
                        "evolution_id": evolution_id,
                        "event_id": _safe_id(f"ambiguous_{state_id}", fallback="ambiguous_event"),
                        "event_type": "ambiguous",
                        "slice_id": str(state.slice_id),
                        "state_id": state_id,
                        "lineage_id": lineage_lookup.get(state_id),
                        "transition_refs": json.dumps([str(item.transition_id) for item in out_rows], ensure_ascii=True),
                        "score": float(scores[0]),
                        "support_count": int(sum(int(item.support_count) for item in out_rows)),
                        "method": "near_tie_transition_scores",
                        "source_state_ids": json.dumps([state_id], ensure_ascii=True),
                        "target_state_ids": json.dumps([str(item.target_state_id) for item in out_rows], ensure_ascii=True),
                        "event_label": "Ambiguous transition",
                        "warning_flags": "",
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_EVENT_COLUMNS))


def build_membership_projection_evolution(
    *,
    evolution_id: str,
    records_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    keywords_df: pd.DataFrame | None = None,
    periodization: Mapping[str, Any] | None = None,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
) -> EvolutionAnalysisResult:
    """Build yearly cluster evolution tables from static membership projection."""

    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    if records_df.empty:
        raise ValueError("records_df must not be empty")
    if membership_df.empty:
        raise ValueError("membership_df must not be empty")
    year_column = _year_column(records_df)
    uid_column = _uid_column(records_df)
    cluster_column = _cluster_column(membership_df)
    if year_column is None:
        raise ValueError("records_df must include pubyear, year, or publication_year")
    if uid_column is None:
        raise ValueError("records_df must include uid, work_id, paper_id, or id")
    if "uid" not in membership_df.columns:
        raise ValueError("membership_df must include uid")
    if cluster_column is None:
        raise ValueError("membership_df must include cluster or cluster_* column")
    years_raw = pd.to_numeric(records_df[year_column], errors="coerce")
    valid_years = sorted({int(year) for year in years_raw.dropna().tolist() if int(year) > 0})
    if not valid_years:
        raise ValueError("records_df has no valid publication years")

    matching = {
        "metric": "projected_cluster_identity",
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "tie_policy": "keep_all_above_threshold",
        "normalization": "static_membership_projection",
    }
    if matching_method:
        matching.update(dict(matching_method))
    matching = _normalize_matching_method(matching)
    rules = {
        "continuation_min_score": 0.5,
        "split_min_children": 2,
        "merge_min_parents": 2,
        "emergence_max_incoming_score": 0.0,
        "decline_max_outgoing_score": 0.0,
        "ambiguous_score_margin": 0.05,
    }
    if event_rules:
        rules.update(dict(event_rules))

    slices = _build_time_slices(evolution_id, valid_years, periodization)
    states, state_docs = _state_rows(
        evolution_id,
        records_df,
        membership_df,
        slices,
        year_column=year_column,
        uid_column=uid_column,
        cluster_column=cluster_column,
        keywords=keywords_df,
    )
    slices = _update_slice_counts(slices, states)
    transitions = _transition_rows(
        evolution_id,
        slices,
        states,
        state_docs,
        matching_method=matching,
    )
    lineages = _lineage_rows(evolution_id, states, transitions)
    events = _event_rows_from_graph(
        evolution_id,
        slices,
        states,
        transitions,
        lineages,
        event_rules=rules,
    )
    periodization_payload = {
        "unit": "year",
        "window_years": 1,
        "step_years": 1,
        "start_year": int(slices["start_year"].min()),
        "end_year": int(slices["start_year"].max()),
        "state_method": "membership_projection",
        "include_unknown_year": False,
    }
    if periodization:
        periodization_payload.update(dict(periodization))
        periodization_payload["unit"] = "year"
    metrics = [
        {
            "name": str(matching["metric"]),
            "value_type": "float",
            "range": [0.0, 1.0],
            "interpretation": "continuity score between adjacent slice-local cluster states",
        },
        {
            "name": "lineage_stability",
            "value_type": "float",
            "range": [0.0, 1.0],
            "interpretation": "aggregate continuity strength across a lineage",
        },
    ]
    analysis_transforms = [
        {"step": "parse_publication_years"},
        {"step": "build_time_slices"},
        {"step": "project_static_membership_to_slices", "cluster_column": cluster_column},
        {"step": "score_adjacent_slice_transitions"},
        {"step": "build_lineages"},
        {"step": "assign_evolution_events"},
        *[dict(item) for item in (transforms or [])],
    ]
    return EvolutionAnalysisResult(
        evolution_id=evolution_id,
        slices=slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
        events=events,
        matching_method=matching,
        event_rules=rules,
        periodization=periodization_payload,
        entity_scope={
            "cluster_level": _level_from_cluster_column(cluster_column),
            "cluster_id_namespace": "projected_static_membership",
            "document_universe": "records_with_valid_pubyear_and_membership",
            "filter_refs": [],
        },
        metrics=metrics,
        transforms=analysis_transforms,
    )


__all__ = [
    "EvolutionAnalysisResult",
    "build_membership_projection_evolution",
]
