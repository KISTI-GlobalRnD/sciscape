"""Cluster evolution analysis utilities.

This module computes evolution tables from records, membership, and optional
keyword evidence. Artifact writing and validation stay in ``sciscape.artifacts``.
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


EVOLUTION_TIME_SLICES_SCHEMA_VERSION = "sciscape_evolution_time_slices_v1"
EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION = "sciscape_evolution_cluster_states_v1"
EVOLUTION_TRANSITIONS_SCHEMA_VERSION = "sciscape_evolution_transitions_v1"
EVOLUTION_LINEAGES_SCHEMA_VERSION = "sciscape_evolution_lineages_v1"
EVOLUTION_EVENTS_SCHEMA_VERSION = "sciscape_evolution_events_v1"
EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION = "sciscape_evolution_state_membership_v1"

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
REQUIRED_EVOLUTION_TIME_SLICE_COLUMNS = {
    "schema_version",
    "evolution_id",
    "slice_id",
    "slice_index",
    "slice_label",
    "start_year",
    "end_year",
    "unit",
    "doc_count",
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
    state_membership: pd.DataFrame | None = None


@dataclass(frozen=True)
class EvolutionEvidenceTables:
    """Schema-ready time-slice evidence tables for evolution writers."""

    evolution_id: str
    slices: pd.DataFrame
    state_evidence: pd.DataFrame
    state_membership: pd.DataFrame
    periodization: dict[str, Any]
    entity_scope: dict[str, Any]
    transforms: list[dict[str, Any]]


def _safe_id(value: object, *, fallback: str = "result") -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_.:-")
    return safe or fallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_progress_json(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    progress_path = Path(path).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp_path.replace(progress_path)


def _safe_filename_id(value: object, *, fallback: str = "item") -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")
    return safe or fallback


def _write_membership_part(
    parts_dir: str | Path | None,
    rows: list[dict[str, Any]],
    *,
    slice_id: object,
    slice_index: object,
) -> Path | None:
    if parts_dir is None:
        return None
    try:
        index = int(slice_index)
    except (TypeError, ValueError):
        index = 0
    part_dir = Path(parts_dir).expanduser()
    part_dir.mkdir(parents=True, exist_ok=True)
    safe_slice = _safe_filename_id(slice_id, fallback=f"slice_{index}")
    part_path = part_dir / f"slice_{index:06d}_{safe_slice}.parquet"
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
    pd.DataFrame(rows).sort_values(["slice_index", "cluster_id", "uid"], kind="stable").to_parquet(tmp_path, index=False)
    tmp_path.replace(part_path)
    return part_path


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _normalize_json_list(value: Any, *, field: str) -> tuple[str, int]:
    if _is_missing(value):
        return "[]", 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "[]", 0
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be a JSON list when encoded as a string") from exc
            if not isinstance(parsed, list):
                raise ValueError(f"{field} must be a JSON list")
            values = parsed
        else:
            values = [text]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    normalized = [str(item).strip() for item in values if not _is_missing(item) and str(item).strip()]
    return json.dumps(normalized, ensure_ascii=True), len(normalized)


def _text_or_default(value: Any, default: str) -> str:
    if _is_missing(value):
        return default
    text = str(value).strip()
    return text or default


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
    threshold, min_support = _score_support_thresholds(matching_method)
    normalized = dict(matching_method)
    normalized.update(
        {
            "metric": metric,
            "min_transition_score": threshold,
            "min_support_count": min_support,
        }
    )
    return normalized


def _score_support_thresholds(matching_method: Mapping[str, Any] | None) -> tuple[float, int]:
    try:
        threshold = float((matching_method or {}).get("min_transition_score", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("evolution matching min_transition_score must be a number") from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("evolution matching min_transition_score must be between 0 and 1")
    min_support = _config_int(matching_method, "min_support_count", 1, label="evolution matching")
    if min_support < 1:
        raise ValueError("evolution matching min_support_count must be at least 1")
    return threshold, min_support


def _default_event_rules() -> dict[str, Any]:
    return {
        "continuation_min_score": 0.5,
        "split_min_children": 2,
        "merge_min_parents": 2,
        "emergence_max_incoming_score": 0.0,
        "decline_max_outgoing_score": 0.0,
        "ambiguous_score_margin": 0.05,
    }


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


def _resolve_uid_column(df: pd.DataFrame, requested: str | None, *, label: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{label} missing requested uid column: {requested}")
        return requested
    column = _uid_column(df)
    if column is None:
        raise ValueError(f"{label} must include uid, work_id, paper_id, or id")
    return column


def _state_document_column(state_membership: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in state_membership.columns:
            raise ValueError(f"state_membership missing requested uid_column: {requested}")
        return requested
    for column in ("uid", "work_id", "paper_id", "document_id", "id"):
        if column in state_membership.columns:
            return column
    raise ValueError("state_membership must include uid, work_id, paper_id, document_id, or id")


def _cluster_columns(membership: pd.DataFrame | None) -> list[str]:
    if membership is None or membership.empty:
        return []
    return [column for column in membership.columns if column == "cluster" or column.startswith("cluster_")]


def _cluster_column(membership: pd.DataFrame) -> str | None:
    columns = _cluster_columns(membership)
    if "cluster" in columns:
        return "cluster"
    return sorted(columns)[0] if columns else None


def _slice_membership_cluster_column(membership: pd.DataFrame) -> str | None:
    for column in ("cluster", "cluster_id", "cluster_key"):
        if column in membership.columns:
            return column
    return _cluster_column(membership)


def _level_from_cluster_column(column: str) -> str:
    if column == "cluster":
        return "cluster"
    if column in {"cluster_id", "cluster_key"}:
        return "cluster"
    if column.startswith("cluster_"):
        return column.removeprefix("cluster_")
    return column


def _keyword_label_column(columns: list[str]) -> str:
    for name in ("term", "label", "keyword", "term_label"):
        if name in columns:
            return name
    return columns[0] if columns else "term"


def _cluster_key_from_value(value: Any, *, column: str, level: str) -> tuple[str, str, str]:
    text = str(value).strip()
    if column == "cluster_key" and ":" in text:
        raw_level, raw_cluster_id = text.split(":", 1)
        state_level = raw_level.strip() or level
        cluster_id = raw_cluster_id.strip() or text
        return f"{state_level}:{cluster_id}", cluster_id, state_level
    return f"{level}:{text}", text, level


def _parse_slice_years(slice_id: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{4})(?:\D+(\d{4}))?", str(slice_id))
    if not match:
        return None, None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return start, end


def _single_group_value(group: pd.DataFrame, column: str, *, label: str) -> Any:
    values = [value for value in group[column].tolist() if not _is_missing(value) and str(value).strip() != ""]
    unique = {str(value).strip() for value in values}
    if len(unique) > 1:
        raise ValueError(f"slice-local membership has conflicting {label} for slice_id={group.name}")
    return values[0] if values else None


def _infer_slice_table_from_membership(
    evolution_id: str,
    slice_membership: pd.DataFrame,
    *,
    slice_id_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_id, group in slice_membership.groupby(slice_id_column, sort=False):
        slice_text = str(slice_id).strip()
        if not slice_text:
            continue
        raw_index = _single_group_value(group, "slice_index", label="slice_index") if "slice_index" in group.columns else None
        raw_start = _single_group_value(group, "start_year", label="start_year") if "start_year" in group.columns else None
        raw_end = _single_group_value(group, "end_year", label="end_year") if "end_year" in group.columns else None
        parsed_start, parsed_end = _parse_slice_years(slice_text)
        start_year = _coerce_int(raw_start) if raw_start is not None else parsed_start
        end_year = _coerce_int(raw_end) if raw_end is not None else parsed_end
        if start_year is None or end_year is None:
            raise ValueError(
                "slice-local membership needs start_year/end_year columns "
                f"or parseable year values in slice_id: {slice_text}"
            )
        unit = (
            _text_or_default(_single_group_value(group, "unit", label="unit"), "year")
            if "unit" in group.columns
            else "year"
        )
        label = (
            _text_or_default(_single_group_value(group, "slice_label", label="slice_label"), slice_text)
            if "slice_label" in group.columns
            else slice_text
        )
        rows.append(
            {
                "slice_id": slice_text,
                "_raw_slice_index": _coerce_int(raw_index) if raw_index is not None else None,
                "slice_label": label,
                "start_year": int(start_year),
                "end_year": int(end_year),
                "unit": unit,
            }
        )
    if not rows:
        raise ValueError("slice-local membership produced no slices")
    rows.sort(
        key=lambda row: (
            row["_raw_slice_index"] if row["_raw_slice_index"] is not None else 10**9,
            row["start_year"],
            row["end_year"],
            row["slice_id"],
        )
    )
    if any(row["_raw_slice_index"] is not None for row in rows):
        raw_indexes = [row["_raw_slice_index"] for row in rows]
        if any(value is None for value in raw_indexes):
            raise ValueError("slice-local membership must provide slice_index for every slice or none")
        if sorted(raw_indexes) != list(range(len(rows))):
            raise ValueError("slice-local membership slice_index must be contiguous from zero")
    for index, row in enumerate(rows):
        row["schema_version"] = EVOLUTION_TIME_SLICES_SCHEMA_VERSION
        row["evolution_id"] = evolution_id
        row["slice_index"] = int(row["_raw_slice_index"] if row["_raw_slice_index"] is not None else index)
        row["doc_count"] = 0
        row["edge_count"] = None
        row["active_cluster_count"] = 0
        row["unknown_year_count"] = 0
        row["warning_flags"] = ""
        del row["_raw_slice_index"]
    return _normalize_evolution_slices(evolution_id, pd.DataFrame(rows))


def _slice_local_keyword_label_map(
    keywords: pd.DataFrame | None,
    *,
    default_level: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if keywords is None or keywords.empty:
        return {}
    label_col = _keyword_label_column(list(keywords.columns))
    slice_col = "slice_id" if "slice_id" in keywords.columns else None
    state_col = "state_id" if "state_id" in keywords.columns else None
    cluster_key_col = "cluster_key" if "cluster_key" in keywords.columns else None
    cluster_id_col = None
    for name in ("cluster_id", "cluster", "source_cluster_id"):
        if name in keywords.columns:
            cluster_id_col = name
            break
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}

    keys: list[str] = []
    if state_col:
        keys.append(state_col)
    if slice_col:
        keys.append(slice_col)
    if cluster_key_col:
        keys.append(cluster_key_col)
    elif cluster_id_col:
        keys.append(cluster_id_col)
    if not keys:
        return rows

    for group_key, group in keywords.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        item = dict(zip(keys, group_key))
        terms = [str(value).strip() for value in group[label_col].dropna().head(5).tolist() if str(value).strip()]
        if not terms:
            continue
        label = {"label": terms[0], "top_terms": terms, "term_count": int(len(terms))}
        if state_col:
            state_id = str(item[state_col]).strip()
            if state_id:
                rows[("state_id", "", state_id)] = label
        slice_id = str(item.get(slice_col, "")).strip() if slice_col else ""
        if cluster_key_col:
            cluster_key = str(item[cluster_key_col]).strip()
        elif cluster_id_col:
            cluster_key = f"{default_level}:{str(item[cluster_id_col]).strip()}"
        else:
            cluster_key = ""
        if cluster_key:
            if slice_id:
                rows[("slice_cluster_key", slice_id, cluster_key)] = label
            rows[("cluster_key", "", cluster_key)] = label
            if ":" in cluster_key:
                rows[("cluster_id", "", cluster_key.split(":", 1)[1])] = label
        elif cluster_id_col:
            cluster_id = str(item[cluster_id_col]).strip()
            if slice_id:
                rows[("slice_cluster_id", slice_id, cluster_id)] = label
            rows[("cluster_id", "", cluster_id)] = label
    return rows


def _edge_column(edges: pd.DataFrame, requested: str | None, candidates: tuple[str, ...], *, label: str) -> str:
    if requested:
        if requested not in edges.columns:
            raise ValueError(f"edges_df missing requested {label} column: {requested}")
        return requested
    for column in candidates:
        if column in edges.columns:
            return column
    raise ValueError(f"edges_df must include one of these {label} columns: {', '.join(candidates)}")


def _edge_weight_column(edges: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in edges.columns:
            raise ValueError(f"edges_df missing requested weight column: {requested}")
        return requested
    for column in ("rel_sum2", "weight", "w", "value"):
        if column in edges.columns:
            return column
    return None


def _contiguous_labels(labels: list[int] | list[str] | Any) -> list[int]:
    mapping: dict[str, int] = {}
    out: list[int] = []
    for value in labels:
        key = str(int(value)) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        out.append(mapping[key])
    return out


def _run_slice_leiden(
    *,
    uids: list[str],
    edges: pd.DataFrame,
    source_column: str,
    target_column: str,
    weight_column: str | None,
    resolution: float,
    objective: str,
    seed: int,
    n_iterations: int,
    backend: str,
) -> tuple[list[int], float | None, int, str]:
    uid_to_index = {uid: index for index, uid in enumerate(uids)}
    if len(uids) == 0:
        return [], None, 0, "none"
    if len(uids) == 1 or edges.empty:
        return list(range(len(uids))), None, len(uids), "singleton"

    work = edges[[source_column, target_column] + ([weight_column] if weight_column else [])].copy()
    work["_src"] = work[source_column].map(str).map(uid_to_index)
    work["_dst"] = work[target_column].map(str).map(uid_to_index)
    work = work.dropna(subset=["_src", "_dst"])
    work = work[work["_src"] != work["_dst"]]
    if work.empty:
        return list(range(len(uids))), None, len(uids), "singleton"
    src = work["_src"].astype(int).to_numpy()
    dst = work["_dst"].astype(int).to_numpy()
    weights = (
        pd.to_numeric(work[weight_column], errors="coerce").fillna(1.0).astype(float).to_numpy()
        if weight_column
        else pd.Series([1.0] * len(work), dtype=float).to_numpy()
    )
    objective = str(objective or "cpm").strip().lower()
    if objective not in {"cpm", "modularity"}:
        raise ValueError("slice-local reclustering objective must be cpm or modularity")
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "rust", "igraph"}:
        raise ValueError("slice-local reclustering backend must be auto, rust, or igraph")

    if backend in {"auto", "rust"}:
        try:
            from .clustering.runner import RustLeidenRunner

            runner = RustLeidenRunner(
                src.astype("uint32"),
                dst.astype("uint32"),
                weights.astype("float64"),
                len(uids),
                objective=objective,
                default_iterations=n_iterations,
                default_seed=seed,
            )
            result = runner.run(resolution)
            membership = _contiguous_labels(result.membership.tolist())
            return membership, float(result.quality), len(set(membership)), "rust"
        except Exception:
            if backend == "rust":
                raise

    import igraph as ig

    from .clustering.runner import LeidenRunner

    graph = ig.Graph(n=len(uids), edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    graph.es["weight"] = weights.tolist()
    graph.vs["uid"] = uids
    graph.simplify(combine_edges={"weight": "sum"}, multiple=True, loops=False)
    runner = LeidenRunner(
        graph,
        objective=objective,
        default_iterations=n_iterations,
        default_seed=seed,
    )
    result = runner.run(resolution)
    membership = _contiguous_labels(list(result.membership))
    return membership, float(result.quality), len(set(membership)), "igraph"


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


def _build_periodized_time_slices(
    evolution_id: str,
    years: list[int],
    periodization: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not years:
        raise ValueError("evolution evidence requires at least one valid publication year")
    unit = str((periodization or {}).get("unit", "year")).strip().lower()
    if unit != "year":
        raise ValueError("evolution evidence currently supports only unit='year'")
    window_years = _config_int(periodization, "window_years", 1, label="evolution evidence periodization")
    step_years = _config_int(periodization, "step_years", 1, label="evolution evidence periodization")
    if window_years < 1:
        raise ValueError("evolution evidence window_years must be at least 1")
    if step_years < 1:
        raise ValueError("evolution evidence step_years must be at least 1")
    if bool((periodization or {}).get("include_unknown_year", False)):
        raise ValueError("evolution evidence currently supports only include_unknown_year=False")
    include_partial_windows = bool((periodization or {}).get("include_partial_windows", False))
    start_year = _config_int(periodization, "start_year", min(years), label="evolution evidence periodization")
    end_year = _config_int(periodization, "end_year", max(years), label="evolution evidence periodization")
    if end_year < start_year:
        raise ValueError("evolution evidence end_year must be greater than or equal to start_year")

    rows = []
    for index, start in enumerate(range(start_year, end_year + 1, step_years)):
        raw_end = start + window_years - 1
        if raw_end > end_year and rows and not include_partial_windows:
            break
        end = min(end_year, raw_end)
        if end < start:
            continue
        if window_years == 1:
            slice_id = f"year:{start}"
            label = str(start)
        else:
            slice_id = f"year:{start}-{end}"
            label = f"{start}-{end}"
        rows.append(
            {
                "schema_version": EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "slice_id": slice_id,
                "slice_index": int(index),
                "slice_label": label,
                "start_year": int(start),
                "end_year": int(end),
                "unit": "year",
                "doc_count": 0,
                "edge_count": None,
                "active_cluster_count": 0,
                "unknown_year_count": 0,
                "warning_flags": "",
            }
        )
    if not rows:
        raise ValueError("evolution evidence periodization produced no slices")
    payload = {
        "unit": "year",
        "window_years": int(window_years),
        "step_years": int(step_years),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "state_method": "slice_membership_projection",
        "include_unknown_year": False,
        "include_partial_windows": bool(include_partial_windows),
    }
    return pd.DataFrame(rows), payload


def _normalize_evolution_slices(evolution_id: str, slices: pd.DataFrame) -> pd.DataFrame:
    required = {"slice_id", "slice_index", "start_year", "end_year"}
    missing = sorted(required - set(slices.columns))
    if missing:
        raise ValueError(f"slices missing required columns for evolution analysis: {', '.join(missing)}")
    if slices.empty:
        raise ValueError("slices must not be empty")
    rows = []
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    for item in slices.to_dict("records"):
        slice_id = _text_or_default(item.get("slice_id"), "")
        if not slice_id:
            raise ValueError("slices slice_id must not be empty")
        if slice_id in seen_ids:
            raise ValueError(f"slices contains duplicate slice_id: {slice_id}")
        seen_ids.add(slice_id)
        slice_index = _coerce_int(item.get("slice_index"))
        if slice_index is None or slice_index < 0:
            raise ValueError("slices slice_index must be a non-negative integer")
        if slice_index in seen_indexes:
            raise ValueError(f"slices contains duplicate slice_index: {slice_index}")
        seen_indexes.add(slice_index)
        start_year = _coerce_int(item.get("start_year"))
        end_year = _coerce_int(item.get("end_year"))
        if start_year is None or end_year is None or start_year <= 0 or end_year <= 0:
            raise ValueError("slices start_year and end_year must be positive integers")
        if end_year < start_year:
            raise ValueError("slices end_year must be greater than or equal to start_year")
        doc_count = _coerce_int(item.get("doc_count"))
        if doc_count is None or doc_count < 0:
            doc_count = 0
        rows.append(
            {
                "schema_version": EVOLUTION_TIME_SLICES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "slice_id": slice_id,
                "slice_index": int(slice_index),
                "slice_label": _text_or_default(item.get("slice_label"), str(start_year)),
                "start_year": int(start_year),
                "end_year": int(end_year),
                "unit": _text_or_default(item.get("unit"), "year"),
                "doc_count": int(doc_count),
                "edge_count": item.get("edge_count") if "edge_count" in item and not _is_missing(item.get("edge_count")) else None,
                "active_cluster_count": _coerce_int(item.get("active_cluster_count")) or 0,
                "unknown_year_count": _coerce_int(item.get("unknown_year_count")) or 0,
                "warning_flags": _text_or_default(item.get("warning_flags"), ""),
            }
        )
    expected = list(range(len(rows)))
    actual = sorted(seen_indexes)
    if actual != expected:
        raise ValueError("slices slice_index must be contiguous from zero")
    return pd.DataFrame(rows).sort_values("slice_index", kind="stable").reset_index(drop=True)


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


def build_slice_membership_evidence(
    *,
    evolution_id: str,
    records_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    keywords_df: pd.DataFrame | None = None,
    periodization: Mapping[str, Any] | None = None,
    cluster_column: str | None = None,
    uid_column: str | None = None,
    membership_uid_column: str | None = None,
    representative_work_limit: int = 50,
) -> EvolutionEvidenceTables:
    """Build schema-ready slice/state/membership evidence tables.

    The builder projects an existing membership table into yearly or rolling
    time windows. It does not relabel transitions itself; the resulting tables
    are intended for ``build_document_overlap_evolution`` or the corresponding
    artifact writer.
    """

    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    if records_df.empty:
        raise ValueError("records_df must not be empty")
    if membership_df.empty:
        raise ValueError("membership_df must not be empty")
    year_column = _year_column(records_df)
    if year_column is None:
        raise ValueError("records_df must include pubyear, year, or publication_year")
    record_uid_column = _resolve_uid_column(records_df, uid_column, label="records_df")
    member_uid_column = _resolve_uid_column(membership_df, membership_uid_column, label="membership_df")
    if cluster_column is not None:
        if cluster_column not in membership_df.columns:
            raise ValueError(f"membership_df missing requested cluster_column: {cluster_column}")
        selected_cluster_column = cluster_column
    else:
        selected_cluster_column = _cluster_column(membership_df)
    if selected_cluster_column is None:
        raise ValueError("membership_df must include cluster or cluster_* column")
    if representative_work_limit < 0:
        raise ValueError("representative_work_limit must be non-negative")

    years_raw = pd.to_numeric(records_df[year_column], errors="coerce")
    valid_years = sorted({int(year) for year in years_raw.dropna().tolist() if int(year) > 0})
    if not valid_years:
        raise ValueError("records_df has no valid publication years")
    slices, periodization_payload = _build_periodized_time_slices(evolution_id, valid_years, periodization)
    level = _level_from_cluster_column(selected_cluster_column)
    label_map = _cluster_label_map(keywords_df, selected_cluster_column)

    records = records_df[[record_uid_column, year_column]].dropna(subset=[record_uid_column]).copy()
    records["_uid"] = records[record_uid_column].map(str).str.strip()
    records["_evolution_year"] = pd.to_numeric(records[year_column], errors="coerce")
    records = records[(records["_uid"] != "") & records["_evolution_year"].notna()]
    records["_evolution_year"] = records["_evolution_year"].astype(int)
    records = records[records["_evolution_year"] > 0]
    records = records.drop_duplicates(subset=["_uid", "_evolution_year"])
    membership = membership_df[[member_uid_column, selected_cluster_column]].dropna(subset=[member_uid_column, selected_cluster_column]).copy()
    membership["_uid"] = membership[member_uid_column].map(str).str.strip()
    membership["_cluster_id"] = membership[selected_cluster_column].map(str).str.strip()
    membership = membership[(membership["_uid"] != "") & (membership["_cluster_id"] != "")]
    membership = membership.drop_duplicates(subset=["_uid", "_cluster_id"])
    joined = records.merge(membership[["_uid", "_cluster_id"]], on="_uid", how="inner")
    if joined.empty:
        raise ValueError("records_df and membership_df have no overlapping document ids")

    rows: list[dict[str, Any]] = []
    state_docs: dict[str, set[str]] = {}
    for slice_row in slices.sort_values("slice_index", kind="stable").itertuples(index=False):
        start_year = int(slice_row.start_year)
        end_year = int(slice_row.end_year)
        in_slice = joined[(joined["_evolution_year"] >= start_year) & (joined["_evolution_year"] <= end_year)]
        if in_slice.empty:
            continue
        grouped = (
            in_slice.groupby("_cluster_id", sort=True)
            .agg(work_ids=("_uid", lambda values: sorted(set(map(str, values)))))
            .reset_index()
        )
        for item in grouped.to_dict("records"):
            cluster_id = str(item["_cluster_id"])
            cluster_key = f"{level}:{cluster_id}"
            state_id = _state_id(str(slice_row.slice_id), cluster_key)
            doc_ids = set(item["work_ids"])
            if not doc_ids:
                continue
            label_info = label_map.get(cluster_key, {})
            top_terms = list(label_info.get("top_terms") or [])
            state_docs[state_id] = doc_ids
            rows.append(
                {
                    "schema_version": EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "state_id": state_id,
                    "slice_id": str(slice_row.slice_id),
                    "slice_index": int(slice_row.slice_index),
                    "cluster_key": cluster_key,
                    "cluster_label": str(label_info.get("label") or cluster_key),
                    "doc_count": int(len(doc_ids)),
                    "term_count": int(label_info.get("term_count") or len(top_terms)),
                    "top_terms": json.dumps(top_terms, ensure_ascii=True),
                    "cluster_uid": cluster_key,
                    "cluster_id": cluster_id,
                    "level": level,
                    "representative_work_ids": json.dumps(sorted(doc_ids)[:representative_work_limit], ensure_ascii=True),
                    "source_cluster_key": cluster_key,
                    "warning_flags": "",
                }
            )
    if not rows:
        raise ValueError("slice membership evidence produced no active cluster states")

    state_evidence = pd.DataFrame(rows).sort_values(["slice_index", "cluster_key"], kind="stable").reset_index(drop=True)
    slices = _update_slice_counts(slices, state_evidence)
    state_membership = _state_membership_rows(evolution_id, state_evidence, state_docs)
    entity_scope = {
        "cluster_level": level,
        "cluster_id_namespace": "projected_membership_evidence",
        "document_universe": "records_with_valid_pubyear_and_membership",
        "filter_refs": [],
    }
    transforms = [
        {"step": "parse_publication_years"},
        {
            "step": "build_periodized_time_slices",
            "window_years": int(periodization_payload["window_years"]),
            "step_years": int(periodization_payload["step_years"]),
        },
        {"step": "project_membership_to_time_slices", "cluster_column": selected_cluster_column},
        {"step": "write_state_document_membership"},
    ]
    return EvolutionEvidenceTables(
        evolution_id=evolution_id,
        slices=slices.reset_index(drop=True),
        state_evidence=state_evidence,
        state_membership=state_membership,
        periodization=periodization_payload,
        entity_scope=entity_scope,
        transforms=transforms,
    )


def build_slice_local_membership_evidence(
    *,
    evolution_id: str,
    slice_membership_df: pd.DataFrame,
    slices_df: pd.DataFrame | None = None,
    keywords_df: pd.DataFrame | None = None,
    cluster_column: str | None = None,
    uid_column: str | None = None,
    slice_id_column: str = "slice_id",
    representative_work_limit: int = 50,
    default_level: str = "cluster",
) -> EvolutionEvidenceTables:
    """Build evolution evidence from slice-local clustering membership.

    The input membership is already scoped by ``slice_id``. Cluster ids are not
    assumed to be stable across slices; downstream continuity should be derived
    from document-overlap evidence.
    """

    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    if slice_membership_df.empty:
        raise ValueError("slice_membership_df must not be empty")
    if slice_id_column not in slice_membership_df.columns:
        raise ValueError(f"slice_membership_df missing slice_id column: {slice_id_column}")
    member_uid_column = _resolve_uid_column(slice_membership_df, uid_column, label="slice_membership_df")
    if cluster_column is not None:
        if cluster_column not in slice_membership_df.columns:
            raise ValueError(f"slice_membership_df missing requested cluster_column: {cluster_column}")
        selected_cluster_column = cluster_column
    else:
        selected_cluster_column = _slice_membership_cluster_column(slice_membership_df)
    if selected_cluster_column is None:
        raise ValueError("slice_membership_df must include cluster, cluster_id, cluster_key, or cluster_* column")
    if representative_work_limit < 0:
        raise ValueError("representative_work_limit must be non-negative")

    level = str(default_level or "").strip() or _level_from_cluster_column(selected_cluster_column)
    if selected_cluster_column not in {"cluster_id", "cluster_key"}:
        level = _level_from_cluster_column(selected_cluster_column)
    passthrough = [column for column in ("slice_index", "start_year", "end_year", "slice_label", "unit") if column in slice_membership_df.columns]
    membership = slice_membership_df[[slice_id_column, member_uid_column, selected_cluster_column, *passthrough]].copy()
    membership["_slice_id"] = membership[slice_id_column].map(str).str.strip()
    membership["_uid"] = membership[member_uid_column].map(str).str.strip()
    membership["_cluster_raw"] = membership[selected_cluster_column].map(str).str.strip()
    membership = membership[(membership["_slice_id"] != "") & (membership["_uid"] != "") & (membership["_cluster_raw"] != "")]
    membership = membership.drop_duplicates(subset=["_slice_id", "_uid", "_cluster_raw"])
    if membership.empty:
        raise ValueError("slice-local membership has no valid slice/document/cluster rows")
    duplicate_docs = (
        membership.groupby(["_slice_id", "_uid"], sort=True)["_cluster_raw"]
        .nunique()
        .reset_index(name="cluster_count")
    )
    duplicate_docs = duplicate_docs[duplicate_docs["cluster_count"] > 1]
    if not duplicate_docs.empty:
        first = duplicate_docs.iloc[0]
        raise ValueError(
            "slice-local membership must assign each document to one cluster per slice; "
            f"duplicate uid={first['_uid']} slice_id={first['_slice_id']}"
        )

    if slices_df is None:
        slices = _infer_slice_table_from_membership(evolution_id, membership, slice_id_column="_slice_id")
    else:
        slices = _normalize_evolution_slices(evolution_id, slices_df)
    slice_ids = set(slices["slice_id"].map(str))
    unknown_slices = sorted(set(membership["_slice_id"]) - slice_ids)
    if unknown_slices:
        preview = ", ".join(unknown_slices[:5])
        suffix = "..." if len(unknown_slices) > 5 else ""
        raise ValueError(f"slice_membership_df references unknown slice_id: {preview}{suffix}")

    label_map = _slice_local_keyword_label_map(keywords_df, default_level=level)
    slice_index = {str(row.slice_id): int(row.slice_index) for row in slices.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    state_docs: dict[str, set[str]] = {}
    grouped = (
        membership.groupby(["_slice_id", "_cluster_raw"], sort=True)
        .agg(work_ids=("_uid", lambda values: sorted(set(map(str, values)))))
        .reset_index()
    )
    for item in grouped.to_dict("records"):
        slice_id = str(item["_slice_id"])
        cluster_key, cluster_id, state_level = _cluster_key_from_value(
            item["_cluster_raw"],
            column=selected_cluster_column,
            level=level,
        )
        state_id = _state_id(slice_id, cluster_key)
        doc_ids = set(item["work_ids"])
        if not doc_ids:
            continue
        label_info = (
            label_map.get(("state_id", "", state_id))
            or label_map.get(("slice_cluster_key", slice_id, cluster_key))
            or label_map.get(("slice_cluster_id", slice_id, cluster_id))
            or label_map.get(("cluster_key", "", cluster_key))
            or label_map.get(("cluster_id", "", cluster_id))
            or {}
        )
        top_terms = list(label_info.get("top_terms") or [])
        state_docs[state_id] = doc_ids
        rows.append(
            {
                "schema_version": EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "state_id": state_id,
                "slice_id": slice_id,
                "slice_index": int(slice_index[slice_id]),
                "cluster_key": cluster_key,
                "cluster_label": str(label_info.get("label") or cluster_key),
                "doc_count": int(len(doc_ids)),
                "term_count": int(label_info.get("term_count") or len(top_terms)),
                "top_terms": json.dumps(top_terms, ensure_ascii=True),
                "cluster_uid": cluster_key,
                "cluster_id": cluster_id,
                "level": state_level,
                "representative_work_ids": json.dumps(sorted(doc_ids)[:representative_work_limit], ensure_ascii=True),
                "source_cluster_key": cluster_key,
                "warning_flags": "",
            }
        )
    if not rows:
        raise ValueError("slice-local membership evidence produced no active cluster states")

    state_evidence = pd.DataFrame(rows).sort_values(["slice_index", "cluster_key"], kind="stable").reset_index(drop=True)
    slices = _update_slice_counts(slices, state_evidence)
    ordered_slices = slices.sort_values("slice_index", kind="stable")
    durations = (
        pd.to_numeric(ordered_slices["end_year"], errors="coerce")
        - pd.to_numeric(ordered_slices["start_year"], errors="coerce")
        + 1
    ).dropna()
    start_diffs = pd.to_numeric(ordered_slices["start_year"], errors="coerce").diff().dropna()
    positive_steps = [int(value) for value in start_diffs.tolist() if int(value) > 0]
    periodization = {
        "unit": "year",
        "window_years": int(durations.max()) if not durations.empty else 1,
        "step_years": int(min(positive_steps)) if positive_steps else 1,
        "start_year": int(slices["start_year"].min()),
        "end_year": int(slices["end_year"].max()),
        "state_method": "slice_local_membership",
        "include_unknown_year": False,
    }
    entity_scope = {
        "cluster_level": level,
        "cluster_id_namespace": "slice_local_membership",
        "document_universe": "slice_local_membership_rows",
        "filter_refs": [],
    }
    transforms = [
        {"step": "normalize_slice_local_membership", "cluster_column": selected_cluster_column},
        {"step": "derive_slice_local_cluster_states"},
        {"step": "write_state_document_membership"},
    ]
    return EvolutionEvidenceTables(
        evolution_id=evolution_id,
        slices=slices.reset_index(drop=True),
        state_evidence=state_evidence,
        state_membership=_state_membership_rows(evolution_id, state_evidence, state_docs),
        periodization=periodization,
        entity_scope=entity_scope,
        transforms=transforms,
    )


def build_slice_reclustering_membership(
    *,
    evolution_id: str,
    records_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    periodization: Mapping[str, Any] | None = None,
    uid_column: str | None = None,
    edge_source_column: str | None = None,
    edge_target_column: str | None = None,
    edge_weight_column: str | None = None,
    resolution: float = 1.0,
    objective: str = "cpm",
    seed: int = 0,
    n_iterations: int = 10,
    backend: str = "auto",
    min_docs_per_slice: int = 1,
    progress_path: str | Path | None = None,
    max_workers: int = 1,
    membership_parts_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run one-level slice-local Leiden clustering and return membership rows.

    This is a bridge for cluster evolution. It intentionally produces only
    slice-scoped membership; keyword extraction, hierarchy construction, and
    report generation remain separate pipeline stages.
    """

    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    if records_df.empty:
        raise ValueError("records_df must not be empty")
    if edges_df.empty:
        raise ValueError("edges_df must not be empty")
    year_column = _year_column(records_df)
    if year_column is None:
        raise ValueError("records_df must include pubyear, year, or publication_year")
    record_uid_column = _resolve_uid_column(records_df, uid_column, label="records_df")
    source_column = _edge_column(edges_df, edge_source_column, ("uid1", "source", "src", "from", "node1"), label="source")
    target_column = _edge_column(edges_df, edge_target_column, ("uid2", "target", "dst", "to", "node2"), label="target")
    weight_column = _edge_weight_column(edges_df, edge_weight_column)
    try:
        resolution_float = float(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError("slice-local reclustering resolution must be a number") from exc
    if not math.isfinite(resolution_float) or resolution_float < 0.0:
        raise ValueError("slice-local reclustering resolution must be non-negative")
    objective_norm = str(objective or "cpm").strip().lower()
    if objective_norm not in {"cpm", "modularity"}:
        raise ValueError("slice-local reclustering objective must be cpm or modularity")
    iterations = _config_int({"n_iterations": n_iterations}, "n_iterations", 10, label="slice-local reclustering")
    if iterations < 0:
        raise ValueError("slice-local reclustering n_iterations must be non-negative")
    min_docs = _config_int({"min_docs_per_slice": min_docs_per_slice}, "min_docs_per_slice", 1, label="slice-local reclustering")
    if min_docs < 1:
        raise ValueError("slice-local reclustering min_docs_per_slice must be at least 1")
    workers = _config_int({"max_workers": max_workers}, "max_workers", 1, label="slice-local reclustering")
    if workers < 1:
        raise ValueError("slice-local reclustering max_workers must be at least 1")

    records = records_df[[record_uid_column, year_column]].dropna(subset=[record_uid_column]).copy()
    records["_uid"] = records[record_uid_column].map(str).str.strip()
    records["_year"] = pd.to_numeric(records[year_column], errors="coerce")
    records = records[(records["_uid"] != "") & records["_year"].notna()]
    records["_year"] = records["_year"].astype(int)
    records = records[records["_year"] > 0].drop_duplicates(subset=["_uid", "_year"])
    if records.empty:
        raise ValueError("records_df has no valid publication years")

    valid_years = sorted(set(records["_year"].astype(int).tolist()))
    slices, _ = _build_periodized_time_slices(evolution_id, valid_years, periodization)
    edge_work = edges_df[[source_column, target_column] + ([weight_column] if weight_column else [])].copy()
    edge_work["_source_uid"] = edge_work[source_column].map(str).str.strip()
    edge_work["_target_uid"] = edge_work[target_column].map(str).str.strip()
    edge_work = edge_work[(edge_work["_source_uid"] != "") & (edge_work["_target_uid"] != "")]
    if weight_column:
        edge_work["_weight"] = pd.to_numeric(edge_work[weight_column], errors="coerce").fillna(1.0)
    else:
        edge_work["_weight"] = 1.0

    rows: list[dict[str, Any]] = []
    ordered_slices = slices.sort_values("slice_index", kind="stable").reset_index(drop=True)
    progress: dict[str, Any] = {
        "schema_version": "sciscape_slice_reclustering_progress_v1",
        "evolution_id": evolution_id,
        "status": "running",
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "total_slices": int(len(ordered_slices)),
        "processed_slices": 0,
        "completed_slices": 0,
        "skipped_slices": 0,
        "membership_rows": 0,
        "membership_part_count": 0,
        "membership_part_rows": 0,
        "params": {
            "backend": str(backend),
            "objective": objective_norm,
            "resolution": resolution_float,
            "seed": int(seed),
            "n_iterations": int(iterations),
            "min_docs_per_slice": int(min_docs),
            "max_workers": int(workers),
            "membership_parts_dir": str(Path(membership_parts_dir).expanduser()) if membership_parts_dir is not None else None,
        },
        "last_slice": None,
    }
    _write_progress_json(progress_path, progress)

    def _slice_job(slice_row: Mapping[str, Any], uids: list[str], slice_edges: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        membership, quality, n_clusters, backend_used = _run_slice_leiden(
            uids=uids,
            edges=slice_edges,
            source_column="_source_uid",
            target_column="_target_uid",
            weight_column="_weight",
            resolution=resolution_float,
            objective=objective_norm,
            seed=int(seed),
            n_iterations=iterations,
            backend=backend,
        )
        job_rows = [
            {
                "evolution_id": evolution_id,
                "slice_id": str(slice_row["slice_id"]),
                "slice_index": int(slice_row["slice_index"]),
                "slice_label": str(slice_row["slice_label"]),
                "start_year": int(slice_row["start_year"]),
                "end_year": int(slice_row["end_year"]),
                "unit": str(slice_row["unit"]),
                "uid": uid,
                "cluster_id": int(cluster_id),
                "resolution": resolution_float,
                "objective": objective_norm,
                "seed": int(seed),
                "n_iterations": int(iterations),
                "backend": backend_used,
                "quality": quality,
                "slice_doc_count": int(len(uids)),
                "slice_edge_count": int(len(slice_edges)),
                "slice_cluster_count": int(n_clusters),
            }
            for uid, cluster_id in zip(uids, membership)
        ]
        last_slice = {
            "slice_id": str(slice_row["slice_id"]),
            "slice_index": int(slice_row["slice_index"]),
            "status": "completed",
            "doc_count": int(len(uids)),
            "edge_count": int(len(slice_edges)),
            "cluster_count": int(n_clusters),
            "backend": backend_used,
            "quality": quality,
        }
        return job_rows, last_slice

    try:
        jobs: list[tuple[dict[str, Any], list[str], pd.DataFrame]] = []
        for slice_row in ordered_slices.to_dict("records"):
            in_slice = records[
                (records["_year"] >= int(slice_row["start_year"]))
                & (records["_year"] <= int(slice_row["end_year"]))
            ]
            uids = sorted(set(in_slice["_uid"].tolist()))
            if len(uids) < min_docs:
                progress["processed_slices"] = int(progress["processed_slices"]) + 1
                progress["skipped_slices"] = int(progress["skipped_slices"]) + 1
                progress["updated_at_utc"] = _utc_now()
                progress["last_slice"] = {
                    "slice_id": str(slice_row["slice_id"]),
                    "slice_index": int(slice_row["slice_index"]),
                    "status": "skipped",
                    "doc_count": int(len(uids)),
                    "edge_count": 0,
                    "reason": "below_min_docs_per_slice",
                }
                _write_progress_json(progress_path, progress)
                continue
            uid_set = set(uids)
            slice_edges = edge_work[
                edge_work["_source_uid"].isin(uid_set)
                & edge_work["_target_uid"].isin(uid_set)
            ].copy()
            jobs.append((slice_row, uids, slice_edges))

        def _record_completed(job_rows: list[dict[str, Any]], last_slice: dict[str, Any]) -> None:
            part_path = _write_membership_part(
                membership_parts_dir,
                job_rows,
                slice_id=last_slice.get("slice_id"),
                slice_index=last_slice.get("slice_index"),
            )
            if part_path is not None:
                last_slice = dict(last_slice)
                last_slice["membership_part_path"] = str(part_path)
                progress["membership_part_count"] = int(progress["membership_part_count"]) + 1
                progress["membership_part_rows"] = int(progress["membership_part_rows"]) + int(len(job_rows))
            rows.extend(job_rows)
            progress["processed_slices"] = int(progress["processed_slices"]) + 1
            progress["completed_slices"] = int(progress["completed_slices"]) + 1
            progress["membership_rows"] = int(len(rows))
            progress["updated_at_utc"] = _utc_now()
            progress["last_slice"] = last_slice
            _write_progress_json(progress_path, progress)

        if workers == 1 or len(jobs) <= 1:
            for job in jobs:
                _record_completed(*_slice_job(*job))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_slice_job, *job) for job in jobs]
                for future in as_completed(futures):
                    _record_completed(*future.result())
    except Exception as exc:
        progress["status"] = "failed"
        progress["updated_at_utc"] = _utc_now()
        progress["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_progress_json(progress_path, progress)
        raise
    if not rows:
        progress["status"] = "failed"
        progress["updated_at_utc"] = _utc_now()
        progress["error"] = {"type": "ValueError", "message": "slice-local reclustering produced no membership rows"}
        _write_progress_json(progress_path, progress)
        raise ValueError("slice-local reclustering produced no membership rows")
    progress["status"] = "completed"
    progress["updated_at_utc"] = _utc_now()
    progress["membership_rows"] = int(len(rows))
    _write_progress_json(progress_path, progress)
    return pd.DataFrame(rows).sort_values(["slice_index", "cluster_id", "uid"], kind="stable").reset_index(drop=True)


def _state_membership_rows(evolution_id: str, states: pd.DataFrame, state_docs: Mapping[str, set[str]]) -> pd.DataFrame:
    columns = [
        "schema_version",
        "evolution_id",
        "state_id",
        "slice_id",
        "slice_index",
        "cluster_key",
        "cluster_id",
        "level",
        "uid",
    ]
    if states.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    state_lookup = {str(row.state_id): row for row in states.itertuples(index=False)}
    for state_id in sorted(state_docs):
        state = state_lookup.get(str(state_id))
        if state is None:
            continue
        for uid in sorted(state_docs[state_id]):
            rows.append(
                {
                    "schema_version": EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "state_id": str(state_id),
                    "slice_id": str(state.slice_id),
                    "slice_index": int(state.slice_index),
                    "cluster_key": str(state.cluster_key),
                    "cluster_id": str(getattr(state, "cluster_id", "")),
                    "level": str(getattr(state, "level", "")),
                    "uid": str(uid),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _state_membership_sidecar_rows(
    evolution_id: str,
    states: pd.DataFrame,
    state_membership: pd.DataFrame,
    *,
    state_id_column: str,
    uid_column: str | None,
) -> pd.DataFrame:
    columns = [
        "schema_version",
        "evolution_id",
        "state_id",
        "slice_id",
        "slice_index",
        "cluster_key",
        "cluster_id",
        "level",
        "uid",
    ]
    if states.empty or state_membership.empty:
        return pd.DataFrame(columns=columns)
    if state_id_column not in state_membership.columns:
        raise ValueError(f"state_membership missing state id column for sidecar: {state_id_column}")
    doc_col = _state_document_column(state_membership, uid_column)
    state_lookup = {str(row.state_id): row for row in states.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    work = state_membership[[state_id_column, doc_col]].dropna(subset=[state_id_column, doc_col]).copy()
    work["_state_id"] = work[state_id_column].map(str).str.strip()
    work["_uid"] = work[doc_col].map(str).str.strip()
    work = work[(work["_state_id"] != "") & (work["_uid"] != "")]
    work = work.drop_duplicates(subset=["_state_id", "_uid"])
    for item in work.sort_values(["_state_id", "_uid"], kind="stable").to_dict("records"):
        state_id = str(item["_state_id"])
        state = state_lookup.get(state_id)
        if state is None:
            continue
        rows.append(
            {
                "schema_version": EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION,
                "evolution_id": evolution_id,
                "state_id": state_id,
                "slice_id": str(state.slice_id),
                "slice_index": int(state.slice_index),
                "cluster_key": str(state.cluster_key),
                "cluster_id": str(getattr(state, "cluster_id", "")),
                "level": str(getattr(state, "level", "")),
                "uid": str(item["_uid"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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


def build_evolution_state_table(
    *,
    evolution_id: str,
    slices: pd.DataFrame,
    state_evidence: pd.DataFrame,
    default_level: str = "cluster",
) -> pd.DataFrame:
    """Normalize raw slice-local state evidence into the evolution state schema."""

    slice_required = {"slice_id", "slice_index"}
    slice_missing = sorted(slice_required - set(slices.columns))
    if slice_missing:
        raise ValueError(f"slices missing required columns for state table: {', '.join(slice_missing)}")
    evidence_required = {"slice_id", "doc_count"}
    evidence_missing = sorted(evidence_required - set(state_evidence.columns))
    if evidence_missing:
        raise ValueError(f"state_evidence missing required columns: {', '.join(evidence_missing)}")
    if "cluster_key" not in state_evidence.columns and "cluster_id" not in state_evidence.columns:
        raise ValueError("state_evidence must include cluster_key or cluster_id")
    if state_evidence.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_STATE_COLUMNS))

    slice_index: dict[str, Any] = {}
    for row in slices.itertuples(index=False):
        slice_id = str(row.slice_id)
        if slice_id in slice_index:
            raise ValueError(f"slices contains duplicate slice_id: {slice_id}")
        slice_index[slice_id] = row

    rows = []
    seen_state_ids: set[str] = set()
    seen_slice_keys: set[tuple[str, str]] = set()
    optional_columns = [
        "cluster_uid",
        "parent_uid",
        "centroid_x",
        "centroid_y",
        "activity_score",
        "growth_score",
    ]
    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    for evidence in state_evidence.to_dict("records"):
        slice_id = str(evidence.get("slice_id", "")).strip()
        if slice_id not in slice_index:
            raise ValueError(f"state_evidence references unknown slice_id: {slice_id}")
        raw_level = evidence.get("level", default_level)
        level = str(raw_level if not _is_missing(raw_level) else default_level).strip() or default_level
        raw_cluster_key = evidence.get("cluster_key")
        raw_cluster_id = evidence.get("cluster_id")
        if _is_missing(raw_cluster_key) or str(raw_cluster_key).strip() == "":
            if _is_missing(raw_cluster_id) or str(raw_cluster_id).strip() == "":
                raise ValueError("state_evidence must include non-empty cluster_key or cluster_id")
            cluster_id = str(raw_cluster_id).strip()
            cluster_key = f"{level}:{cluster_id}"
        else:
            cluster_key = str(raw_cluster_key).strip()
            cluster_id = str(raw_cluster_id).strip() if not _is_missing(raw_cluster_id) else cluster_key.split(":")[-1]
        slice_key = (slice_id, cluster_key)
        if slice_key in seen_slice_keys:
            raise ValueError(f"state_evidence contains duplicate slice_id/cluster_key: {slice_id} {cluster_key}")
        seen_slice_keys.add(slice_key)
        raw_state_id = evidence.get("state_id")
        state_id = str(raw_state_id).strip() if not _is_missing(raw_state_id) and str(raw_state_id).strip() else _state_id(slice_id, cluster_key)
        if state_id in seen_state_ids:
            raise ValueError(f"state_evidence contains duplicate state_id: {state_id}")
        seen_state_ids.add(state_id)
        doc_count = _coerce_int(evidence.get("doc_count"))
        if doc_count is None or doc_count <= 0:
            raise ValueError("state_evidence doc_count must be a positive integer")
        top_terms, inferred_term_count = _normalize_json_list(evidence.get("top_terms", []), field="top_terms")
        raw_term_count = evidence.get("term_count")
        term_count = _coerce_int(raw_term_count) if not _is_missing(raw_term_count) else inferred_term_count
        if term_count is None or term_count < 0:
            raise ValueError("state_evidence term_count must be a non-negative integer")
        representative_work_ids, _ = _normalize_json_list(
            evidence.get("representative_work_ids", []),
            field="representative_work_ids",
        )
        raw_label = evidence.get("cluster_label")
        if _is_missing(raw_label):
            raw_label = evidence.get("label")
        cluster_label = str(raw_label if not _is_missing(raw_label) else cluster_key).strip() or cluster_key
        slice_row = slice_index[slice_id]
        row: dict[str, Any] = {
            "schema_version": EVOLUTION_CLUSTER_STATES_SCHEMA_VERSION,
            "evolution_id": evolution_id,
            "state_id": state_id,
            "slice_id": slice_id,
            "slice_index": int(slice_row.slice_index),
            "cluster_key": cluster_key,
            "cluster_label": cluster_label,
            "doc_count": int(doc_count),
            "term_count": int(term_count),
            "top_terms": top_terms,
            "cluster_uid": _text_or_default(evidence.get("cluster_uid"), cluster_key),
            "cluster_id": cluster_id,
            "level": level,
            "representative_work_ids": representative_work_ids,
            "source_cluster_key": _text_or_default(evidence.get("source_cluster_key"), cluster_key),
            "warning_flags": _text_or_default(evidence.get("warning_flags"), ""),
        }
        for column in optional_columns:
            if column in evidence and column not in row:
                row[column] = evidence[column]
        rows.append(row)

    return pd.DataFrame(rows)


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
    transitions = pd.DataFrame(rows) if rows else pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_TRANSITION_COLUMNS))
    return label_evolution_transition_relations(transitions)


def rank_evolution_transitions(transitions: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic source-side and target-side transition ranks."""

    if transitions.empty:
        out = transitions.copy()
        if "rank_from_source" not in out.columns:
            out["rank_from_source"] = pd.Series(dtype="int64")
        if "rank_to_target" not in out.columns:
            out["rank_to_target"] = pd.Series(dtype="int64")
        return out
    required = {"source_state_id", "target_state_id", "score"}
    missing = sorted(required - set(transitions.columns))
    if missing:
        raise ValueError(f"transitions missing required columns for ranking: {', '.join(missing)}")
    out = transitions.copy()
    if "support_count" not in out.columns:
        out["support_count"] = 0
    out["_rank_score"] = pd.to_numeric(out["score"], errors="coerce").fillna(float("-inf"))
    out["_rank_support"] = pd.to_numeric(out["support_count"], errors="coerce").fillna(0)
    out["_rank_target"] = out["target_state_id"].map(str)
    out["_rank_source"] = out["source_state_id"].map(str)
    source_order = out.sort_values(
        ["source_state_id", "_rank_score", "_rank_support", "_rank_target"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    source_ranks = source_order.groupby("source_state_id", sort=False).cumcount() + 1
    out.loc[source_order.index, "rank_from_source"] = source_ranks.astype(int).to_numpy()
    target_order = out.sort_values(
        ["target_state_id", "_rank_score", "_rank_support", "_rank_source"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    target_ranks = target_order.groupby("target_state_id", sort=False).cumcount() + 1
    out.loc[target_order.index, "rank_to_target"] = target_ranks.astype(int).to_numpy()
    return out.drop(columns=["_rank_score", "_rank_support", "_rank_target", "_rank_source"])


def label_evolution_transition_relations(
    transitions: pd.DataFrame,
    *,
    event_rules: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Assign deterministic relation labels to ranked transition evidence."""

    rules = _default_event_rules()
    if event_rules:
        rules.update(dict(event_rules))
    continuation_min = float(rules.get("continuation_min_score", 0.5))
    split_min_children = _config_int(rules, "split_min_children", 2, label="evolution event_rules")
    merge_min_parents = _config_int(rules, "merge_min_parents", 2, label="evolution event_rules")
    ambiguous_margin = float(rules.get("ambiguous_score_margin", 0.05))
    if split_min_children < 2:
        raise ValueError("evolution event_rules split_min_children must be at least 2")
    if merge_min_parents < 2:
        raise ValueError("evolution event_rules merge_min_parents must be at least 2")

    out = rank_evolution_transitions(transitions)
    if out.empty:
        if "relation" not in out.columns:
            out["relation"] = pd.Series(dtype="object")
        return out
    if "relation" not in out.columns:
        out["relation"] = "candidate"
    existing = out["relation"].fillna("").map(lambda value: str(value).strip())
    mutable = existing.isin({"", "candidate"})
    out.loc[mutable, "relation"] = "candidate"

    score = pd.to_numeric(out["score"], errors="coerce").fillna(float("-inf"))
    strong = score >= continuation_min
    strong_source_counts = out.loc[strong].groupby("source_state_id")["target_state_id"].transform("count")
    strong_target_counts = out.loc[strong].groupby("target_state_id")["source_state_id"].transform("count")
    out["_strong_source_count"] = 0
    out["_strong_target_count"] = 0
    out.loc[strong, "_strong_source_count"] = strong_source_counts.to_numpy()
    out.loc[strong, "_strong_target_count"] = strong_target_counts.to_numpy()

    continuation_mask = mutable & strong & (out["_strong_source_count"] == 1) & (out["_strong_target_count"] == 1)
    split_mask = mutable & strong & (out["_strong_source_count"] >= split_min_children)
    merge_mask = mutable & strong & (out["_strong_target_count"] >= merge_min_parents)
    out.loc[continuation_mask, "relation"] = "continuation"
    out.loc[split_mask, "relation"] = "split_child"
    out.loc[merge_mask, "relation"] = "merge_parent"
    out.loc[split_mask & merge_mask, "relation"] = "ambiguous"

    for source_id, group in out.loc[mutable & strong].groupby("source_state_id", sort=False):
        if len(group) < 2 or int(group["_strong_source_count"].iloc[0]) >= split_min_children:
            continue
        ordered = group.sort_values(["score", "support_count", "target_state_id"], ascending=[False, False, True], kind="stable")
        top_score = float(ordered.iloc[0]["score"])
        second_score = float(ordered.iloc[1]["score"])
        if abs(top_score - second_score) <= ambiguous_margin:
            out.loc[ordered.index, "relation"] = "ambiguous"

    return out.drop(columns=["_strong_source_count", "_strong_target_count"])


def build_evolution_transition_table(
    *,
    evolution_id: str,
    states: pd.DataFrame,
    transition_evidence: pd.DataFrame,
    metric: str,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    allow_skip_slices: bool = False,
) -> pd.DataFrame:
    """Normalize raw state-transition evidence into the evolution transition schema."""

    metric = str(metric or "").strip()
    if not metric:
        raise ValueError("evolution transition metric must not be empty")
    state_required = {"state_id", "slice_id", "slice_index", "doc_count"}
    state_missing = sorted(state_required - set(states.columns))
    if state_missing:
        raise ValueError(f"states missing required columns for transition table: {', '.join(state_missing)}")
    evidence_required = {"source_state_id", "target_state_id", "score", "support_count"}
    evidence_missing = sorted(evidence_required - set(transition_evidence.columns))
    if evidence_missing:
        raise ValueError(f"transition_evidence missing required columns: {', '.join(evidence_missing)}")
    if states.empty or transition_evidence.empty:
        return label_evolution_transition_relations(pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_TRANSITION_COLUMNS)), event_rules=event_rules)

    threshold, min_support = _score_support_thresholds(matching_method)
    state_index: dict[str, Any] = {}
    for state in states.itertuples(index=False):
        state_id = str(state.state_id)
        if state_id in state_index:
            raise ValueError(f"states contains duplicate state_id: {state_id}")
        state_index[state_id] = state

    rows = []
    optional_columns = [
        "shared_doc_count",
        "shared_term_count",
        "jaccard",
        "overlap_source",
        "overlap_target",
        "overlap_min",
        "evidence_ref",
        "warning_flags",
    ]
    for evidence in transition_evidence.itertuples(index=False):
        source_state_id = str(evidence.source_state_id)
        target_state_id = str(evidence.target_state_id)
        if source_state_id not in state_index:
            raise ValueError(f"transition_evidence references unknown source_state_id: {source_state_id}")
        if target_state_id not in state_index:
            raise ValueError(f"transition_evidence references unknown target_state_id: {target_state_id}")
        source = state_index[source_state_id]
        target = state_index[target_state_id]
        source_slice_index = int(source.slice_index)
        target_slice_index = int(target.slice_index)
        if target_slice_index <= source_slice_index:
            raise ValueError("transition_evidence target slices must follow source slices")
        if not allow_skip_slices and target_slice_index != source_slice_index + 1:
            raise ValueError("transition_evidence must connect adjacent slices unless allow_skip_slices=True")
        try:
            score = float(evidence.score)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition_evidence score must be a number") from exc
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("transition_evidence score must be finite and between 0 and 1")
        support_count = _coerce_int(evidence.support_count)
        if support_count is None or support_count < 0:
            raise ValueError("transition_evidence support_count must be a non-negative integer")
        if score < threshold or support_count < min_support:
            continue
        transition_id = getattr(evidence, "transition_id", None)
        if transition_id is None or str(transition_id).strip() == "":
            transition_id = _safe_id(f"{source_state_id}_to_{target_state_id}_{metric}", fallback="transition")
        row: dict[str, Any] = {
            "schema_version": EVOLUTION_TRANSITIONS_SCHEMA_VERSION,
            "evolution_id": _safe_id(evolution_id, fallback="cluster_evolution"),
            "transition_id": str(transition_id),
            "source_state_id": source_state_id,
            "target_state_id": target_state_id,
            "source_slice_id": str(source.slice_id),
            "target_slice_id": str(target.slice_id),
            "metric": metric,
            "score": float(score),
            "support_count": int(support_count),
            "source_doc_count": int(source.doc_count),
            "target_doc_count": int(target.doc_count),
            "relation": str(getattr(evidence, "relation", "candidate") or "candidate"),
        }
        for column in optional_columns:
            if hasattr(evidence, column):
                row[column] = getattr(evidence, column)
        if "warning_flags" not in row:
            row["warning_flags"] = ""
        rows.append(row)

    transitions = pd.DataFrame(rows) if rows else pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_TRANSITION_COLUMNS))
    return label_evolution_transition_relations(transitions, event_rules=event_rules)


def build_document_overlap_transition_evidence(
    *,
    states: pd.DataFrame,
    state_membership: pd.DataFrame,
    uid_column: str | None = None,
    state_id_column: str = "state_id",
    metric: str = "jaccard_doc_overlap",
    min_shared_docs: int = 1,
    min_score: float = 0.0,
    require_complete_membership: bool = True,
) -> pd.DataFrame:
    """Derive adjacent-slice transition evidence from state-document membership.

    The membership table must contain complete document membership for each
    state by default. Representative-work samples are not sufficient evidence
    for split, merge, or continuation claims.
    """

    supported_metrics = {"jaccard_doc_overlap", "overlap_source", "overlap_target", "overlap_min"}
    metric = str(metric or "").strip()
    if metric not in supported_metrics:
        raise ValueError(f"unsupported document-overlap metric: {metric}")
    min_shared_docs_int = _config_int(
        {"min_shared_docs": min_shared_docs},
        "min_shared_docs",
        1,
        label="document-overlap transition evidence",
    )
    if min_shared_docs_int < 1:
        raise ValueError("document-overlap transition evidence min_shared_docs must be at least 1")
    try:
        min_score_float = float(min_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("document-overlap transition evidence min_score must be a number") from exc
    if not math.isfinite(min_score_float) or min_score_float < 0.0 or min_score_float > 1.0:
        raise ValueError("document-overlap transition evidence min_score must be between 0 and 1")
    if not state_id_column:
        raise ValueError("state_id_column must not be empty")

    state_required = {"state_id", "slice_id", "slice_index", "doc_count"}
    state_missing = sorted(state_required - set(states.columns))
    if state_missing:
        raise ValueError(f"states missing required columns for document-overlap evidence: {', '.join(state_missing)}")
    doc_col = _state_document_column(state_membership, uid_column)
    membership_required = {state_id_column, doc_col}
    membership_missing = sorted(membership_required - set(state_membership.columns))
    if membership_missing:
        raise ValueError(f"state_membership missing required columns: {', '.join(membership_missing)}")

    evidence_columns = [
        "transition_id",
        "source_state_id",
        "target_state_id",
        "score",
        "support_count",
        "shared_doc_count",
        "jaccard",
        "overlap_source",
        "overlap_target",
        "overlap_min",
        "metric",
        "evidence_ref",
        "warning_flags",
    ]
    if states.empty or state_membership.empty:
        return pd.DataFrame(columns=evidence_columns)

    state_rows = states[["state_id", "slice_id", "slice_index", "doc_count"]].copy()
    state_rows["state_id"] = state_rows["state_id"].map(str)
    if state_rows["state_id"].duplicated().any():
        duplicate = str(state_rows.loc[state_rows["state_id"].duplicated(), "state_id"].iloc[0])
        raise ValueError(f"states contains duplicate state_id: {duplicate}")
    state_rows["_doc_count"] = pd.to_numeric(state_rows["doc_count"], errors="coerce")
    if state_rows["_doc_count"].isna().any() or (state_rows["_doc_count"] < 0).any():
        raise ValueError("states doc_count must be non-negative for document-overlap evidence")
    state_info = {
        str(row["state_id"]): {
            "slice_id": str(row["slice_id"]),
            "slice_index": int(row["slice_index"]),
            "doc_count": int(row["_doc_count"]),
        }
        for row in state_rows.to_dict("records")
    }

    membership = state_membership[[state_id_column, doc_col]].dropna(subset=[state_id_column, doc_col]).copy()
    membership["_state_id"] = membership[state_id_column].map(str).str.strip()
    membership["_uid"] = membership[doc_col].map(str).str.strip()
    membership = membership[(membership["_state_id"] != "") & (membership["_uid"] != "")]
    membership = membership.drop_duplicates(subset=["_state_id", "_uid"])
    if membership.empty:
        return pd.DataFrame(columns=evidence_columns)
    unknown = sorted(set(membership["_state_id"]) - set(state_info))
    if unknown:
        preview = ", ".join(unknown[:5])
        suffix = "..." if len(unknown) > 5 else ""
        raise ValueError(f"state_membership references unknown state_id: {preview}{suffix}")

    membership_counts = membership.groupby("_state_id", sort=True)["_uid"].nunique().to_dict()
    mismatch_states: list[str] = []
    for state_id, info in state_info.items():
        observed = int(membership_counts.get(state_id, 0))
        expected = int(info["doc_count"])
        if observed != expected:
            mismatch_states.append(f"{state_id} expected={expected} observed={observed}")
    if mismatch_states and require_complete_membership:
        preview = "; ".join(mismatch_states[:5])
        suffix = "; ..." if len(mismatch_states) > 5 else ""
        raise ValueError(
            "state_membership must contain complete state-document rows; "
            f"mismatched doc_count for {preview}{suffix}"
        )
    mismatch_ids = {item.split(" expected=", 1)[0] for item in mismatch_states}

    membership["_slice_id"] = membership["_state_id"].map(lambda value: state_info[value]["slice_id"])
    membership["_slice_index"] = membership["_state_id"].map(lambda value: state_info[value]["slice_index"])
    doc_sets = {
        str(state_id): set(group["_uid"].tolist())
        for state_id, group in membership.groupby("_state_id", sort=True)
    }
    state_order = state_rows.sort_values(["slice_index", "state_id"], kind="stable")
    states_by_slice = {
        (str(slice_id), int(slice_index)): group["state_id"].map(str).tolist()
        for (slice_id, slice_index), group in state_order.groupby(["slice_id", "slice_index"], sort=True)
    }
    ordered_slices = (
        state_order[["slice_id", "slice_index"]]
        .drop_duplicates()
        .sort_values(["slice_index", "slice_id"], kind="stable")
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for source_slice, target_slice in zip(ordered_slices.iloc[:-1].itertuples(index=False), ordered_slices.iloc[1:].itertuples(index=False)):
        source_slice_index = int(source_slice.slice_index)
        target_slice_index = int(target_slice.slice_index)
        if target_slice_index != source_slice_index + 1:
            continue
        source_state_ids = states_by_slice.get((str(source_slice.slice_id), source_slice_index), [])
        target_state_ids = states_by_slice.get((str(target_slice.slice_id), target_slice_index), [])
        if not source_state_ids or not target_state_ids:
            continue
        source_by_doc: dict[str, list[str]] = {}
        target_by_doc: dict[str, list[str]] = {}
        for source_state_id in source_state_ids:
            for uid in doc_sets.get(source_state_id, set()):
                source_by_doc.setdefault(uid, []).append(source_state_id)
        for target_state_id in target_state_ids:
            for uid in doc_sets.get(target_state_id, set()):
                target_by_doc.setdefault(uid, []).append(target_state_id)
        pair_counts: dict[tuple[str, str], int] = {}
        for uid in sorted(set(source_by_doc) & set(target_by_doc)):
            for source_state_id in sorted(source_by_doc[uid]):
                for target_state_id in sorted(target_by_doc[uid]):
                    pair = (source_state_id, target_state_id)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for (source_state_id, target_state_id), shared in sorted(pair_counts.items()):
            if shared < min_shared_docs_int:
                continue
            source_membership_count = len(doc_sets.get(source_state_id, set()))
            target_membership_count = len(doc_sets.get(target_state_id, set()))
            source_doc_count = max(int(state_info[source_state_id]["doc_count"]), source_membership_count)
            target_doc_count = max(int(state_info[target_state_id]["doc_count"]), target_membership_count)
            union_count = source_doc_count + target_doc_count - shared
            jaccard = float(shared / union_count) if union_count > 0 else 0.0
            overlap_source = float(shared / source_doc_count) if source_doc_count > 0 else 0.0
            overlap_target = float(shared / target_doc_count) if target_doc_count > 0 else 0.0
            min_denominator = min(source_doc_count, target_doc_count)
            overlap_min = float(shared / min_denominator) if min_denominator > 0 else 0.0
            score_by_metric = {
                "jaccard_doc_overlap": jaccard,
                "overlap_source": overlap_source,
                "overlap_target": overlap_target,
                "overlap_min": overlap_min,
            }
            score = float(score_by_metric[metric])
            if score < min_score_float:
                continue
            warnings = []
            if source_state_id in mismatch_ids or target_state_id in mismatch_ids:
                warnings.append("membership_doc_count_mismatch")
            rows.append(
                {
                    "transition_id": _safe_id(f"{source_state_id}_to_{target_state_id}_{metric}", fallback="transition"),
                    "source_state_id": source_state_id,
                    "target_state_id": target_state_id,
                    "score": score,
                    "support_count": int(shared),
                    "shared_doc_count": int(shared),
                    "jaccard": jaccard,
                    "overlap_source": overlap_source,
                    "overlap_target": overlap_target,
                    "overlap_min": overlap_min,
                    "metric": metric,
                    "evidence_ref": "state_document_membership",
                    "warning_flags": ",".join(warnings),
                }
            )

    if not rows:
        return pd.DataFrame(columns=evidence_columns)
    out = pd.DataFrame(rows)
    out["_source_slice_index"] = out["source_state_id"].map(lambda value: state_info[str(value)]["slice_index"])
    out["_target_sort"] = out["target_state_id"].map(str)
    out = out.sort_values(
        ["_source_slice_index", "source_state_id", "score", "support_count", "_target_sort"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    return out.drop(columns=["_source_slice_index", "_target_sort"]).reset_index(drop=True)


def _state_membership_with_state_ids(
    *,
    state_membership: pd.DataFrame,
    states: pd.DataFrame,
    state_id_column: str,
    default_level: str,
) -> pd.DataFrame:
    if state_id_column in state_membership.columns:
        return state_membership
    required = {"slice_id"}
    missing = sorted(required - set(state_membership.columns))
    if missing:
        raise ValueError(
            "state_membership must include state_id or slice_id plus cluster_key/cluster_id; "
            f"missing: {', '.join(missing)}"
        )
    if "cluster_key" not in state_membership.columns and "cluster_id" not in state_membership.columns:
        raise ValueError("state_membership must include state_id, cluster_key, or cluster_id")
    state_required = {"slice_id", "cluster_key", "state_id"}
    state_missing = sorted(state_required - set(states.columns))
    if state_missing:
        raise ValueError(f"states missing required columns for state membership lookup: {', '.join(state_missing)}")
    lookup: dict[tuple[str, str], str] = {}
    for row in states.itertuples(index=False):
        key = (str(row.slice_id), str(row.cluster_key))
        if key in lookup:
            raise ValueError(f"states contains duplicate slice_id/cluster_key: {key[0]} {key[1]}")
        lookup[key] = str(row.state_id)

    work = state_membership.copy()
    derived_state_ids: list[str] = []
    for item in work.to_dict("records"):
        slice_id = str(item.get("slice_id", "")).strip()
        if not slice_id:
            raise ValueError("state_membership slice_id must not be empty when deriving state ids")
        raw_cluster_key = item.get("cluster_key")
        if _is_missing(raw_cluster_key) or str(raw_cluster_key).strip() == "":
            raw_cluster_id = item.get("cluster_id")
            if _is_missing(raw_cluster_id) or str(raw_cluster_id).strip() == "":
                raise ValueError("state_membership must include non-empty cluster_key or cluster_id when deriving state ids")
            raw_level = item.get("level", default_level)
            level = str(raw_level if not _is_missing(raw_level) else default_level).strip() or default_level
            cluster_key = f"{level}:{str(raw_cluster_id).strip()}"
        else:
            cluster_key = str(raw_cluster_key).strip()
        state_id = lookup.get((slice_id, cluster_key))
        if state_id is None:
            raise ValueError(f"state_membership references unknown slice_id/cluster_key: {slice_id} {cluster_key}")
        derived_state_ids.append(state_id)
    work[state_id_column] = derived_state_ids
    return work


def build_document_overlap_evolution(
    *,
    evolution_id: str,
    slices: pd.DataFrame,
    state_evidence: pd.DataFrame,
    state_membership: pd.DataFrame,
    metric: str = "jaccard_doc_overlap",
    uid_column: str | None = None,
    state_id_column: str = "state_id",
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    periodization: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    default_level: str = "cluster",
    require_complete_membership: bool = True,
) -> EvolutionAnalysisResult:
    """Build evolution tables by deriving transitions from state-document overlap."""

    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    metric = str(metric or "").strip() or "jaccard_doc_overlap"
    matching = {
        "metric": metric,
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "tie_policy": "keep_all_above_threshold",
        "normalization": "state_document_membership_overlap",
        "require_complete_membership": bool(require_complete_membership),
    }
    if matching_method:
        matching.update(dict(matching_method))
    matching["metric"] = metric
    threshold, min_support = _score_support_thresholds(matching)
    matching["min_transition_score"] = threshold
    matching["min_support_count"] = min_support
    rules = _default_event_rules()
    if event_rules:
        rules.update(dict(event_rules))

    normalized_slices = _normalize_evolution_slices(evolution_id, slices)
    states = build_evolution_state_table(
        evolution_id=evolution_id,
        slices=normalized_slices,
        state_evidence=state_evidence,
        default_level=default_level,
    )
    normalized_slices = _update_slice_counts(normalized_slices, states)
    membership_with_ids = _state_membership_with_state_ids(
        state_membership=state_membership,
        states=states,
        state_id_column=state_id_column,
        default_level=default_level,
    )
    state_membership_sidecar = _state_membership_sidecar_rows(
        evolution_id,
        states,
        membership_with_ids,
        state_id_column=state_id_column,
        uid_column=uid_column,
    )
    transition_evidence = build_document_overlap_transition_evidence(
        states=states,
        state_membership=membership_with_ids,
        uid_column=uid_column,
        state_id_column=state_id_column,
        metric=metric,
        min_shared_docs=min_support,
        min_score=threshold,
        require_complete_membership=require_complete_membership,
    )
    transitions = build_evolution_transition_table(
        evolution_id=evolution_id,
        states=states,
        transition_evidence=transition_evidence,
        metric=metric,
        matching_method=matching,
        event_rules=rules,
    )
    lineages = _lineage_rows(evolution_id, states, transitions)
    events = classify_evolution_events(
        evolution_id=evolution_id,
        slices=normalized_slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
        event_rules=rules,
    )
    periodization_payload = {
        "unit": _text_or_default(normalized_slices.iloc[0].get("unit"), "year"),
        "start_year": int(normalized_slices["start_year"].min()),
        "end_year": int(normalized_slices["end_year"].max()),
        "state_method": "explicit_state_evidence",
        "transition_method": "state_document_membership_overlap",
        "include_unknown_year": False,
    }
    if periodization:
        periodization_payload.update(dict(periodization))
    entity_scope_payload = {
        "cluster_level": default_level,
        "cluster_id_namespace": "explicit_state_evidence",
        "document_universe": "state_document_membership",
        "filter_refs": [],
    }
    if entity_scope:
        entity_scope_payload.update(dict(entity_scope))
    metrics = [
        {
            "name": metric,
            "value_type": "float",
            "range": [0.0, 1.0],
            "interpretation": "document-overlap continuity score between adjacent slice-local cluster states",
        },
        {
            "name": "lineage_stability",
            "value_type": "float",
            "range": [0.0, 1.0],
            "interpretation": "aggregate continuity strength across a lineage",
        },
    ]
    analysis_transforms = [
        {"step": "normalize_time_slices"},
        {"step": "normalize_state_evidence", "default_level": default_level},
        {
            "step": "derive_transition_evidence_from_state_document_membership",
            "metric": metric,
            "require_complete_membership": bool(require_complete_membership),
        },
        {"step": "normalize_transition_evidence", "metric": metric},
        {"step": "build_lineages"},
        {"step": "assign_evolution_events"},
        *[dict(item) for item in (transforms or [])],
    ]
    return EvolutionAnalysisResult(
        evolution_id=evolution_id,
        slices=normalized_slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
        events=events,
        matching_method=matching,
        event_rules=rules,
        periodization=periodization_payload,
        entity_scope=entity_scope_payload,
        metrics=metrics,
        transforms=analysis_transforms,
        state_membership=state_membership_sidecar,
    )


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
    split_min_children = _config_int(event_rules, "split_min_children", 2, label="evolution event_rules")
    merge_min_parents = _config_int(event_rules, "merge_min_parents", 2, label="evolution event_rules")
    ambiguous_margin = float(event_rules.get("ambiguous_score_margin", 0.05))
    if split_min_children < 2:
        raise ValueError("evolution event_rules split_min_children must be at least 2")
    if merge_min_parents < 2:
        raise ValueError("evolution event_rules merge_min_parents must be at least 2")
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
        split_rows = [item for item in out_rows if str(item.relation) != "ambiguous"]
        merge_rows = [item for item in in_rows if str(item.relation) != "ambiguous"]
        if len(split_rows) >= split_min_children:
            rows.append(
                {
                    "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "event_id": _safe_id(f"split_{state_id}", fallback="split_event"),
                    "event_type": "split",
                    "slice_id": str(state.slice_id),
                    "state_id": state_id,
                    "lineage_id": lineage_lookup.get(state_id),
                    "transition_refs": json.dumps([str(item.transition_id) for item in split_rows], ensure_ascii=True),
                    "score": float(max(float(item.score) for item in split_rows)),
                    "support_count": int(sum(int(item.support_count) for item in split_rows)),
                    "method": "multi_outgoing_transition_above_threshold",
                    "source_state_ids": json.dumps([state_id], ensure_ascii=True),
                    "target_state_ids": json.dumps([str(item.target_state_id) for item in split_rows], ensure_ascii=True),
                    "event_label": "Split",
                    "warning_flags": "",
                }
            )
        if len(merge_rows) >= merge_min_parents:
            rows.append(
                {
                    "schema_version": EVOLUTION_EVENTS_SCHEMA_VERSION,
                    "evolution_id": evolution_id,
                    "event_id": _safe_id(f"merge_{state_id}", fallback="merge_event"),
                    "event_type": "merge",
                    "slice_id": str(state.slice_id),
                    "state_id": state_id,
                    "lineage_id": lineage_lookup.get(state_id),
                    "transition_refs": json.dumps([str(item.transition_id) for item in merge_rows], ensure_ascii=True),
                    "score": float(max(float(item.score) for item in merge_rows)),
                    "support_count": int(sum(int(item.support_count) for item in merge_rows)),
                    "method": "multi_incoming_transition_above_threshold",
                    "source_state_ids": json.dumps([str(item.source_state_id) for item in merge_rows], ensure_ascii=True),
                    "target_state_ids": json.dumps([state_id], ensure_ascii=True),
                    "event_label": "Merge",
                    "warning_flags": "",
                }
            )
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
        ambiguous_rows = [item for item in out_rows if str(item.relation) == "ambiguous"]
        if not ambiguous_rows and len(split_rows) < split_min_children:
            ambiguous_rows = out_rows
        if len(ambiguous_rows) >= 2:
            scores = sorted((float(item.score) for item in ambiguous_rows), reverse=True)
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
                        "transition_refs": json.dumps([str(item.transition_id) for item in ambiguous_rows], ensure_ascii=True),
                        "score": float(scores[0]),
                        "support_count": int(sum(int(item.support_count) for item in ambiguous_rows)),
                        "method": "near_tie_transition_scores",
                        "source_state_ids": json.dumps([state_id], ensure_ascii=True),
                        "target_state_ids": json.dumps([str(item.target_state_id) for item in ambiguous_rows], ensure_ascii=True),
                        "event_label": "Ambiguous transition",
                        "warning_flags": "",
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=sorted(REQUIRED_EVOLUTION_EVENT_COLUMNS))


def classify_evolution_events(
    *,
    evolution_id: str,
    slices: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    lineages: pd.DataFrame,
    event_rules: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Classify evolution events from explicit state-transition evidence."""

    rules = _default_event_rules()
    if event_rules:
        rules.update(dict(event_rules))
    return _event_rows_from_graph(
        _safe_id(evolution_id, fallback="cluster_evolution"),
        slices,
        states,
        transitions,
        lineages,
        event_rules=rules,
    )


def build_evidence_backed_evolution(
    *,
    evolution_id: str,
    slices: pd.DataFrame,
    state_evidence: pd.DataFrame,
    transition_evidence: pd.DataFrame,
    metric: str,
    matching_method: Mapping[str, Any] | None = None,
    event_rules: Mapping[str, Any] | None = None,
    periodization: Mapping[str, Any] | None = None,
    entity_scope: Mapping[str, Any] | None = None,
    transforms: list[Mapping[str, Any]] | None = None,
    default_level: str = "cluster",
    allow_skip_slices: bool = False,
) -> EvolutionAnalysisResult:
    """Build a complete evolution analysis from explicit state and transition evidence."""

    evolution_id = _safe_id(evolution_id, fallback="cluster_evolution")
    metric = str(metric or "").strip()
    if not metric:
        raise ValueError("evolution transition metric must not be empty")
    matching = {
        "metric": metric,
        "min_transition_score": 0.5,
        "min_support_count": 1,
        "tie_policy": "keep_all_above_threshold",
        "normalization": "explicit_state_transition_evidence",
        "allow_skip_slices": bool(allow_skip_slices),
    }
    if matching_method:
        matching.update(dict(matching_method))
    threshold, min_support = _score_support_thresholds(matching)
    matching["min_transition_score"] = threshold
    matching["min_support_count"] = min_support
    rules = _default_event_rules()
    if event_rules:
        rules.update(dict(event_rules))

    normalized_slices = _normalize_evolution_slices(evolution_id, slices)
    states = build_evolution_state_table(
        evolution_id=evolution_id,
        slices=normalized_slices,
        state_evidence=state_evidence,
        default_level=default_level,
    )
    normalized_slices = _update_slice_counts(normalized_slices, states)
    transitions = build_evolution_transition_table(
        evolution_id=evolution_id,
        states=states,
        transition_evidence=transition_evidence,
        metric=metric,
        matching_method=matching,
        event_rules=rules,
        allow_skip_slices=allow_skip_slices,
    )
    lineages = _lineage_rows(evolution_id, states, transitions)
    events = classify_evolution_events(
        evolution_id=evolution_id,
        slices=normalized_slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
        event_rules=rules,
    )
    periodization_payload = {
        "unit": _text_or_default(normalized_slices.iloc[0].get("unit"), "year"),
        "start_year": int(normalized_slices["start_year"].min()),
        "end_year": int(normalized_slices["end_year"].max()),
        "state_method": "explicit_state_evidence",
        "include_unknown_year": False,
    }
    if periodization:
        periodization_payload.update(dict(periodization))
    entity_scope_payload = {
        "cluster_level": default_level,
        "cluster_id_namespace": "explicit_state_evidence",
        "document_universe": "state_evidence_doc_counts",
        "filter_refs": [],
    }
    if entity_scope:
        entity_scope_payload.update(dict(entity_scope))
    metrics = [
        {
            "name": metric,
            "value_type": "float",
            "range": [0.0, 1.0],
            "interpretation": "continuity score between explicit adjacent slice-local cluster states",
        },
        {
            "name": "lineage_stability",
            "value_type": "float",
            "range": [0.0, 1.0],
            "interpretation": "aggregate continuity strength across a lineage",
        },
    ]
    analysis_transforms = [
        {"step": "normalize_time_slices"},
        {"step": "normalize_state_evidence", "default_level": default_level},
        {"step": "normalize_transition_evidence", "metric": metric},
        {"step": "build_lineages"},
        {"step": "assign_evolution_events"},
        *[dict(item) for item in (transforms or [])],
    ]
    return EvolutionAnalysisResult(
        evolution_id=evolution_id,
        slices=normalized_slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
        events=events,
        matching_method=matching,
        event_rules=rules,
        periodization=periodization_payload,
        entity_scope=entity_scope_payload,
        metrics=metrics,
        transforms=analysis_transforms,
    )


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
    rules = _default_event_rules()
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
    state_membership = _state_membership_rows(evolution_id, states, state_docs)
    transitions = _transition_rows(
        evolution_id,
        slices,
        states,
        state_docs,
        matching_method=matching,
    )
    lineages = _lineage_rows(evolution_id, states, transitions)
    events = classify_evolution_events(
        evolution_id=evolution_id,
        slices=slices,
        states=states,
        transitions=transitions,
        lineages=lineages,
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
        state_membership=state_membership,
    )


__all__ = [
    "EvolutionAnalysisResult",
    "EvolutionEvidenceTables",
    "build_document_overlap_evolution",
    "build_document_overlap_transition_evidence",
    "build_evidence_backed_evolution",
    "build_evolution_state_table",
    "build_membership_projection_evolution",
    "build_slice_reclustering_membership",
    "build_slice_local_membership_evidence",
    "build_slice_membership_evidence",
    "EVOLUTION_STATE_MEMBERSHIP_SCHEMA_VERSION",
    "build_evolution_transition_table",
    "classify_evolution_events",
    "label_evolution_transition_relations",
    "rank_evolution_transitions",
]
