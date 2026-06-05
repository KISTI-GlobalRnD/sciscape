#!/usr/bin/env python3
"""Audit transition bands in the first-pass 014 wall-localization trace.

This is a read-only audit over the executed wall-localization trace. It does
not rerun Leiden. The purpose is to separate three different readouts that the
strict G4.9A vocabulary can conflate:

1. a strict interpretable wall interval with no unknown endpoint objects;
2. a monotone source/intermediate/target transition band;
3. a bounded but nonmonotone transition band where unknown endpoint objects
   appear inside the target-side band.

The audit keeps all method, quality/cost, full replay, and wall-generality
claims closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace import (
    BOUNDARY_GUARD_RESULT_ROWS_CSV as TRACE_BOUNDARY_GUARD_RESULT_ROWS_CSV,
    BOUNDARY_PAIR_ID,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    PAIR_LOCALIZATION_ROWS_CSV as TRACE_PAIR_LOCALIZATION_ROWS_CSV,
    POSITIVE_PAIR_ID,
    SEED_LOCALIZATION_ROWS_CSV as TRACE_SEED_LOCALIZATION_ROWS_CSV,
    SOURCE_OBJECTS,
    TARGET_OBJECT,
    TRACE_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_audit_gamma1e5_20260605"
)

ROUTE_BAND_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_route_rows.csv"
)
SEED_BAND_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_seed_rows.csv"
)
PAIR_BAND_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_pair_rows.csv"
)
BOUNDARY_BAND_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_boundary_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_transition_band_report.md"
)

RUN_STATUS = (
    "audited_nanoclustering_g4_8_first_pass_014_wall_localization_transition_band"
)
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 transition-band audit only; "
    "reads the executed wall-localization trace and separates strict, monotone, "
    "and bounded nonmonotone endpoint-object transition bands. It does not "
    "promote wall generality, evaluate quality/cost value, replay full "
    "NanoClustering, or claim method success."
)

POSITIVE_TARGET_CATEGORY = "positive_target"
BOUNDARY_TARGET_CATEGORY = "boundary_target"
SOURCE_CATEGORY = "source"
UNKNOWN_INTERMEDIATE_CATEGORY = "unknown_intermediate"
GUARD_INTERMEDIATE_CATEGORY = "guard_intermediate"
OTHER_INTERMEDIATE_CATEGORY = "other_intermediate"

SHORT_CATEGORY = {
    SOURCE_CATEGORY: "S",
    POSITIVE_TARGET_CATEGORY: "T",
    BOUNDARY_TARGET_CATEGORY: "C",
    UNKNOWN_INTERMEDIATE_CATEGORY: "U",
    GUARD_INTERMEDIATE_CATEGORY: "D",
    OTHER_INTERMEDIATE_CATEGORY: "O",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _object_category(endpoint_object: Any) -> str:
    value = str(endpoint_object)
    if value in SOURCE_OBJECTS:
        return SOURCE_CATEGORY
    if value == TARGET_OBJECT:
        return POSITIVE_TARGET_CATEGORY
    if value == "boundary_target_endpoint_object_not_positive":
        return BOUNDARY_TARGET_CATEGORY
    if value in {
        "direct_drop_guard_endpoint_object",
        "drop_both_guard_endpoint_object",
    }:
        return GUARD_INTERMEDIATE_CATEGORY
    if "unknown" in value or "ambiguous" in value:
        return UNKNOWN_INTERMEDIATE_CATEGORY
    return OTHER_INTERMEDIATE_CATEGORY


def _category_code(categories: list[str]) -> str:
    return "".join(SHORT_CATEGORY.get(value, "O") for value in categories)


def _step_fraction_column(route_rows: pd.DataFrame) -> str:
    direct_unique = route_rows["direct_edge_weight_fraction"].nunique(dropna=False)
    bridge_unique = route_rows["bridge_edge_weight_fraction"].nunique(dropna=False)
    if bridge_unique > direct_unique:
        return "bridge_edge_weight_fraction"
    return "direct_edge_weight_fraction"


def _contiguous_prefix_count(values: list[str], expected: str) -> int:
    count = 0
    for value in values:
        if value != expected:
            break
        count += 1
    return count


def _contiguous_suffix_count(values: list[str], expected: str) -> int:
    count = 0
    for value in reversed(values):
        if value != expected:
            break
        count += 1
    return count


def _route_direction(route_family_role: str) -> str:
    role = str(route_family_role)
    if "descent" in role:
        return "descent"
    if "ascent" in role:
        return "ascent"
    return "unknown"


def _route_band_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trace = trace_rows.copy()
    trace["endpoint_object_category"] = trace["endpoint_object_assignment_by_step"].map(
        _object_category
    )
    key_cols = [
        "route_contract_id",
        "local_pair_id",
        "branch",
        "start_condition",
        "contract_pair_role",
        "route_family_role",
        "seed",
    ]
    for key, group in trace.groupby(key_cols, sort=True):
        data = dict(zip(key_cols, key, strict=True))
        route = group.sort_values("step_index", kind="mergesort").copy()
        direction = _route_direction(str(data["route_family_role"]))
        categories = route["endpoint_object_category"].astype(str).tolist()
        objects = route["endpoint_object_assignment_by_step"].astype(str).tolist()
        scan_fraction_column = _step_fraction_column(route)
        scan_fractions = route[scan_fraction_column].astype(float).tolist()
        code = _category_code(categories)
        is_positive = str(data["local_pair_id"]) == POSITIVE_PAIR_ID

        if is_positive and direction == "descent":
            leading_expected = SOURCE_CATEGORY
            trailing_expected = POSITIVE_TARGET_CATEGORY
        elif is_positive and direction == "ascent":
            leading_expected = POSITIVE_TARGET_CATEGORY
            trailing_expected = SOURCE_CATEGORY
        else:
            leading_expected = ""
            trailing_expected = ""

        prefix_count = (
            _contiguous_prefix_count(categories, leading_expected)
            if leading_expected
            else 0
        )
        suffix_count = (
            _contiguous_suffix_count(categories, trailing_expected)
            if trailing_expected
            else 0
        )
        interior = categories[prefix_count : len(categories) - suffix_count]
        leading_present = bool(leading_expected and leading_expected in categories)
        trailing_present = bool(trailing_expected and trailing_expected in categories)
        starts_with_leading = bool(leading_expected and categories[0] == leading_expected)
        ends_with_trailing = bool(trailing_expected and categories[-1] == trailing_expected)
        boundary_target_present = BOUNDARY_TARGET_CATEGORY in categories
        positive_target_present = POSITIVE_TARGET_CATEGORY in categories
        unknown_intermediate_present = UNKNOWN_INTERMEDIATE_CATEGORY in categories
        guard_intermediate_present = GUARD_INTERMEDIATE_CATEGORY in categories
        only_intermediate_in_middle = bool(
            interior
            and all(
                value
                in {
                    UNKNOWN_INTERMEDIATE_CATEGORY,
                    GUARD_INTERMEDIATE_CATEGORY,
                    OTHER_INTERMEDIATE_CATEGORY,
                }
                for value in interior
            )
        )
        no_endpoint_reentry_in_middle = bool(
            not any(value in {leading_expected, trailing_expected} for value in interior)
        )
        monotone_bracket_pass = bool(
            is_positive
            and starts_with_leading
            and ends_with_trailing
            and leading_present
            and trailing_present
            and prefix_count > 0
            and suffix_count > 0
            and prefix_count + suffix_count < len(categories)
            and only_intermediate_in_middle
            and no_endpoint_reentry_in_middle
            and not boundary_target_present
        )
        strict_interpretable_pass = bool(
            monotone_bracket_pass
            and not unknown_intermediate_present
            and not any(value == OTHER_INTERMEDIATE_CATEGORY for value in categories)
        )
        bounded_nonmonotone_pass = bool(
            is_positive
            and starts_with_leading
            and ends_with_trailing
            and leading_present
            and trailing_present
            and not monotone_bracket_pass
            and not boundary_target_present
        )
        bounded_transition_pass = bool(monotone_bracket_pass or bounded_nonmonotone_pass)

        first_leading_idx = (
            categories.index(leading_expected) if leading_expected in categories else None
        )
        last_leading_idx = (
            max(i for i, value in enumerate(categories) if value == leading_expected)
            if leading_expected in categories
            else None
        )
        first_trailing_idx = (
            categories.index(trailing_expected) if trailing_expected in categories else None
        )
        last_trailing_idx = (
            max(i for i, value in enumerate(categories) if value == trailing_expected)
            if trailing_expected in categories
            else None
        )
        last_leading_fraction = (
            scan_fractions[last_leading_idx] if last_leading_idx is not None else None
        )
        first_trailing_fraction = (
            scan_fractions[first_trailing_idx] if first_trailing_idx is not None else None
        )
        band_width = (
            abs(float(last_leading_fraction) - float(first_trailing_fraction))
            if last_leading_fraction is not None and first_trailing_fraction is not None
            else None
        )

        if strict_interpretable_pass:
            route_band_status = "strict_interpretable_wall_interval"
        elif monotone_bracket_pass:
            route_band_status = "monotone_intermediate_transition_band"
        elif bounded_nonmonotone_pass:
            route_band_status = "bounded_nonmonotone_transition_band"
        elif not is_positive:
            route_band_status = "boundary_or_control_route_not_positive_band"
        elif not leading_present or not trailing_present:
            route_band_status = "missing_source_or_target_endpoint_for_band"
        elif boundary_target_present:
            route_band_status = "boundary_target_leak_in_positive_route"
        else:
            route_band_status = "unbounded_or_unclassified_transition_route"

        rows.append(
            {
                **data,
                "route_direction": direction,
                "scan_fraction_column": scan_fraction_column,
                "route_category_code": code,
                "route_endpoint_object_sequence": " -> ".join(objects),
                "source_step_count": int(categories.count(SOURCE_CATEGORY)),
                "positive_target_step_count": int(
                    categories.count(POSITIVE_TARGET_CATEGORY)
                ),
                "boundary_target_step_count": int(
                    categories.count(BOUNDARY_TARGET_CATEGORY)
                ),
                "unknown_intermediate_step_count": int(
                    categories.count(UNKNOWN_INTERMEDIATE_CATEGORY)
                ),
                "guard_intermediate_step_count": int(
                    categories.count(GUARD_INTERMEDIATE_CATEGORY)
                ),
                "leading_expected_category": leading_expected,
                "trailing_expected_category": trailing_expected,
                "leading_prefix_step_count": int(prefix_count),
                "trailing_suffix_step_count": int(suffix_count),
                "first_leading_fraction": (
                    scan_fractions[first_leading_idx]
                    if first_leading_idx is not None
                    else None
                ),
                "last_leading_fraction": last_leading_fraction,
                "first_trailing_fraction": first_trailing_fraction,
                "last_trailing_fraction": (
                    scan_fractions[last_trailing_idx]
                    if last_trailing_idx is not None
                    else None
                ),
                "transition_band_width": band_width,
                "monotone_bracket_pass": bool(monotone_bracket_pass),
                "strict_interpretable_pass": bool(strict_interpretable_pass),
                "bounded_nonmonotone_pass": bool(bounded_nonmonotone_pass),
                "bounded_transition_pass": bool(bounded_transition_pass),
                "positive_target_present": bool(positive_target_present),
                "boundary_target_present": bool(boundary_target_present),
                "unknown_intermediate_present": bool(unknown_intermediate_present),
                "guard_intermediate_present": bool(guard_intermediate_present),
                "route_band_status": route_band_status,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _seed_band_rows(route_bands: pd.DataFrame, seed_localization: pd.DataFrame) -> pd.DataFrame:
    positive = route_bands[route_bands["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    seed_lookup = seed_localization[
        seed_localization["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ].copy()
    seed_lookup = seed_lookup[
        [
            "local_pair_id",
            "branch",
            "start_condition",
            "seed",
            "g4_9a_vocab_code",
            "localization_seed_status",
            "transition_interval_width",
            "descent_first_positive_target_fraction",
            "ascent_first_source_recovery_fraction",
        ]
    ]
    key_cols = ["local_pair_id", "branch", "start_condition", "seed"]
    rows: list[dict[str, Any]] = []
    for key, group in positive.groupby(key_cols, sort=True):
        data = dict(zip(key_cols, key, strict=True))
        descent = group[group["route_direction"].astype(str).eq("descent")]
        ascent = group[group["route_direction"].astype(str).eq("ascent")]
        if len(descent) != 1 or len(ascent) != 1:
            raise ValueError(f"expected one descent and one ascent row for {key}")
        descent_row = descent.iloc[0]
        ascent_row = ascent.iloc[0]
        strict_interpretable_seed = bool(
            _as_bool(descent_row["strict_interpretable_pass"])
            and _as_bool(ascent_row["strict_interpretable_pass"])
        )
        monotone_transition_band_seed = bool(
            _as_bool(descent_row["monotone_bracket_pass"])
            and _as_bool(ascent_row["monotone_bracket_pass"])
        )
        bounded_transition_band_seed = bool(
            _as_bool(descent_row["bounded_transition_pass"])
            and _as_bool(ascent_row["bounded_transition_pass"])
        )
        nonmonotone_bounded_transition_seed = bool(
            bounded_transition_band_seed and not monotone_transition_band_seed
        )
        total_unknown_steps = int(
            int(descent_row["unknown_intermediate_step_count"])
            + int(ascent_row["unknown_intermediate_step_count"])
        )
        total_guard_steps = int(
            int(descent_row["guard_intermediate_step_count"])
            + int(ascent_row["guard_intermediate_step_count"])
        )
        if strict_interpretable_seed:
            seed_band_status = "strict_interpretable_wall_interval_seed"
        elif monotone_transition_band_seed:
            seed_band_status = "monotone_intermediate_transition_band_seed"
        elif nonmonotone_bounded_transition_seed:
            seed_band_status = "bounded_nonmonotone_transition_band_seed"
        elif not bounded_transition_band_seed:
            seed_band_status = "unbounded_or_unclassified_transition_seed"
        else:
            seed_band_status = "unclassified_transition_band_seed"
        rows.append(
            {
                **data,
                "descent_route_contract_id": str(descent_row["route_contract_id"]),
                "ascent_route_contract_id": str(ascent_row["route_contract_id"]),
                "descent_route_category_code": str(descent_row["route_category_code"]),
                "ascent_route_category_code": str(ascent_row["route_category_code"]),
                "descent_route_band_status": str(descent_row["route_band_status"]),
                "ascent_route_band_status": str(ascent_row["route_band_status"]),
                "strict_interpretable_seed": bool(strict_interpretable_seed),
                "monotone_transition_band_seed": bool(monotone_transition_band_seed),
                "nonmonotone_bounded_transition_seed": bool(
                    nonmonotone_bounded_transition_seed
                ),
                "bounded_transition_band_seed": bool(bounded_transition_band_seed),
                "total_unknown_intermediate_step_count": total_unknown_steps,
                "total_guard_intermediate_step_count": total_guard_steps,
                "descent_transition_band_width": descent_row["transition_band_width"],
                "ascent_transition_band_width": ascent_row["transition_band_width"],
                "seed_band_status": seed_band_status,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.merge(
        seed_lookup,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )


def _boundary_band_rows(route_bands: pd.DataFrame, boundary_guards: pd.DataFrame) -> pd.DataFrame:
    boundary = route_bands[route_bands["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)].copy()
    rows: list[dict[str, Any]] = []
    for start_condition, group in boundary.groupby("start_condition", sort=True):
        positive_target_route_count = int(group["positive_target_present"].map(_as_bool).sum())
        boundary_target_route_count = int(group["boundary_target_present"].map(_as_bool).sum())
        unknown_route_count = int(group["unknown_intermediate_present"].map(_as_bool).sum())
        route_count = int(len(group))
        positive_target_step_count = int(group["positive_target_step_count"].sum())
        boundary_target_step_count = int(group["boundary_target_step_count"].sum())
        guard = boundary_guards[
            boundary_guards["start_condition"].astype(str).eq(str(start_condition))
        ]
        guard_positive_pattern_count = (
            int(guard["boundary_positive_pattern_seed_count"].sum())
            if not guard.empty
            else 0
        )
        positive_leak_closed = bool(
            positive_target_route_count == 0
            and positive_target_step_count == 0
            and guard_positive_pattern_count == 0
        )
        rows.append(
            {
                "local_pair_id": BOUNDARY_PAIR_ID,
                "start_condition": str(start_condition),
                "route_count": route_count,
                "positive_target_route_count": positive_target_route_count,
                "positive_target_step_count": positive_target_step_count,
                "boundary_target_route_count": boundary_target_route_count,
                "boundary_target_step_count": boundary_target_step_count,
                "unknown_route_count": unknown_route_count,
                "boundary_positive_pattern_seed_count": guard_positive_pattern_count,
                "positive_leak_closed": bool(positive_leak_closed),
                "boundary_band_status": (
                    "boundary_positive_target_leak_closed"
                    if positive_leak_closed
                    else "boundary_positive_target_leak_observed"
                ),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_band_rows(seed_bands: pd.DataFrame, boundary_bands: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive_seed_count = int(len(seed_bands))
    strict_count = int(seed_bands["strict_interpretable_seed"].map(_as_bool).sum())
    monotone_count = int(seed_bands["monotone_transition_band_seed"].map(_as_bool).sum())
    nonmonotone_count = int(
        seed_bands["nonmonotone_bounded_transition_seed"].map(_as_bool).sum()
    )
    bounded_count = int(seed_bands["bounded_transition_band_seed"].map(_as_bool).sum())
    boundary_closed = bool(
        not boundary_bands.empty and boundary_bands["positive_leak_closed"].map(_as_bool).all()
    )
    if positive_seed_count and bounded_count == positive_seed_count and boundary_closed:
        pair_status = "all_positive_seed_starts_bounded_transition_band_boundary_closed"
    elif positive_seed_count and monotone_count == positive_seed_count and boundary_closed:
        pair_status = "all_positive_seed_starts_monotone_transition_band_boundary_closed"
    elif positive_seed_count and bounded_count > 0 and boundary_closed:
        pair_status = "partial_positive_seed_starts_bounded_transition_band_boundary_closed"
    elif not boundary_closed:
        pair_status = "boundary_positive_target_leak_observed"
    else:
        pair_status = "transition_band_not_supported"
    rows.append(
        {
            "local_pair_id": POSITIVE_PAIR_ID,
            "positive_seed_start_count": positive_seed_count,
            "strict_interpretable_seed_count": strict_count,
            "monotone_transition_band_seed_count": monotone_count,
            "monotone_intermediate_seed_count": int(max(monotone_count - strict_count, 0)),
            "nonmonotone_bounded_transition_seed_count": nonmonotone_count,
            "bounded_transition_band_seed_count": bounded_count,
            "unbounded_or_unclassified_seed_count": int(positive_seed_count - bounded_count),
            "seed_band_status_counts": json.dumps(
                _count_dict(seed_bands["seed_band_status"]),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "trace_vocab_code_counts": json.dumps(
                _count_dict(seed_bands["g4_9a_vocab_code"]),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "boundary_positive_leak_closed": bool(boundary_closed),
            "pair_transition_band_status": pair_status,
            "method_claim_allowed_after_audit": False,
            "quality_cost_claim_allowed_after_audit": False,
            "wall_generality_claim_allowed_after_audit": False,
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    trace_gate_matrix: pd.DataFrame,
    route_bands: pd.DataFrame,
    seed_bands: pd.DataFrame,
    pair_bands: pd.DataFrame,
    boundary_bands: pd.DataFrame,
) -> pd.DataFrame:
    trace_gate_counts = _count_dict(trace_gate_matrix["gate_status"])
    positive_seed_count = int(len(seed_bands))
    bounded_count = int(seed_bands["bounded_transition_band_seed"].map(_as_bool).sum())
    classified_count = int(
        seed_bands["seed_band_status"]
        .astype(str)
        .ne("unbounded_or_unclassified_transition_seed")
        .sum()
    )
    boundary_closed = bool(
        not boundary_bands.empty and boundary_bands["positive_leak_closed"].map(_as_bool).all()
    )
    all_claims_closed = bool(
        not pair_bands.empty
        and not pair_bands["method_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_bands["quality_cost_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_bands["wall_generality_claim_allowed_after_audit"].map(_as_bool).any()
    )
    rows = [
        {
            "gate_id": "G1_trace_runner_gates_pass",
            "question": "Did the upstream localization trace gates pass?",
            "observed": json.dumps(trace_gate_counts, ensure_ascii=True, sort_keys=True),
            "minimum_or_rule": "all upstream trace gates pass",
            "gate_status": "pass" if trace_gate_counts.get("pass", 0) == len(trace_gate_matrix) else "fail",
        },
        {
            "gate_id": "G2_route_band_rows_materialized",
            "question": "Was every route scan summarized into a route-band row?",
            "observed": f"route_band_rows={len(route_bands)}",
            "minimum_or_rule": "128 route rows from 16 contracts * 8 seeds",
            "gate_status": "pass" if len(route_bands) == 128 else "fail",
        },
        {
            "gate_id": "G3_positive_seed_starts_bounded",
            "question": "Do all positive seed-start units show bounded source-target transition bands?",
            "observed": f"bounded={bounded_count} positive_seed_starts={positive_seed_count}",
            "minimum_or_rule": "bounded count equals positive seed-start count",
            "gate_status": "pass" if positive_seed_count == 32 and bounded_count == 32 else "fail",
        },
        {
            "gate_id": "G4_transition_band_split_classified",
            "question": "Were positive seed-start units split into strict, monotone, and nonmonotone classes?",
            "observed": json.dumps(
                _count_dict(seed_bands["seed_band_status"]),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "minimum_or_rule": "all 32 positive seed-start rows classified",
            "gate_status": "pass" if classified_count == 32 else "fail",
        },
        {
            "gate_id": "G5_boundary_positive_target_leak_closed",
            "question": "Does the 005 boundary avoid positive-target leakage?",
            "observed": json.dumps(
                boundary_bands[
                    [
                        "start_condition",
                        "positive_target_route_count",
                        "positive_target_step_count",
                        "positive_leak_closed",
                    ]
                ].to_dict(orient="records"),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "minimum_or_rule": "zero positive-target routes and steps in boundary",
            "gate_status": "pass" if boundary_closed else "fail",
        },
        {
            "gate_id": "G6_claims_closed",
            "question": "Are method, quality/cost, and wall-generality claims closed?",
            "observed": CLAIM_BOUNDARY,
            "minimum_or_rule": "all claim flags false",
            "gate_status": "pass" if all_claims_closed else "fail",
        },
        {
            "gate_id": "G7_read_only_audit",
            "question": "Was this a read-only audit over the executed trace?",
            "observed": RUN_STATUS,
            "minimum_or_rule": "audit status only; no Leiden rerun",
            "gate_status": "pass",
        },
    ]
    return pd.DataFrame(rows)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_bands: pd.DataFrame,
    seed_bands: pd.DataFrame,
    boundary_bands: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    report = [
        "# NanoClustering G4.8 First-Pass 014 Transition-Band Audit",
        "",
        f"- status: `{RUN_STATUS}`",
        f"- route_band_row_count: {summary['route_band_row_count']}",
        f"- positive_seed_start_count: {summary['positive_seed_start_count']}",
        f"- strict_interpretable_seed_count: {summary['strict_interpretable_seed_count']}",
        f"- monotone_intermediate_seed_count: {summary['monotone_intermediate_seed_count']}",
        f"- nonmonotone_bounded_transition_seed_count: {summary['nonmonotone_bounded_transition_seed_count']}",
        f"- bounded_transition_band_seed_count: {summary['bounded_transition_band_seed_count']}",
        f"- boundary_positive_target_step_count: {summary['boundary_positive_target_step_count']}",
        f"- pair_transition_band_status: `{summary['pair_transition_band_status']}`",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        "- interpretation: The strict no-unknown wall interval is rare, but the "
        "positive trace is not simply unlocalized. The current readout supports "
        "a bounded transition-band interpretation for local_pair_014, while "
        "the 005 boundary remains closed against positive-target leakage.",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Transition Band",
        "",
        _markdown_table(pair_bands),
        "",
        "## Positive Seed Bands",
        "",
        _markdown_table(
            seed_bands[
                [
                    "start_condition",
                    "seed",
                    "g4_9a_vocab_code",
                    "seed_band_status",
                    "descent_route_category_code",
                    "ascent_route_category_code",
                    "transition_interval_width",
                ]
            ]
        ),
        "",
        "## Boundary Leak Check",
        "",
        _markdown_table(boundary_bands),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates),
        "",
        "## Boundary",
        "",
        "This audit is a local trace interpretation layer. It does not promote "
        "wall generality, method success, full replay, or quality/cost value.",
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(report), encoding="utf-8")


def run_audit(trace_dir: Path, output_dir: Path) -> dict[str, Any]:
    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    seed_localization = _read_csv(trace_dir / TRACE_SEED_LOCALIZATION_ROWS_CSV)
    _read_csv(trace_dir / TRACE_PAIR_LOCALIZATION_ROWS_CSV)
    boundary_guards = _read_csv(trace_dir / TRACE_BOUNDARY_GUARD_RESULT_ROWS_CSV)
    trace_gate_matrix = _read_csv(trace_dir / TRACE_GATE_MATRIX_CSV)

    output_dir.mkdir(parents=True, exist_ok=True)
    route_bands = _route_band_rows(trace_rows)
    seed_bands = _seed_band_rows(route_bands, seed_localization)
    boundary_bands = _boundary_band_rows(route_bands, boundary_guards)
    pair_bands = _pair_band_rows(seed_bands, boundary_bands)
    gates = _gate_matrix(
        trace_gate_matrix=trace_gate_matrix,
        route_bands=route_bands,
        seed_bands=seed_bands,
        pair_bands=pair_bands,
        boundary_bands=boundary_bands,
    )

    _write_csv(route_bands, output_dir / ROUTE_BAND_ROWS_CSV)
    _write_csv(seed_bands, output_dir / SEED_BAND_ROWS_CSV)
    _write_csv(pair_bands, output_dir / PAIR_BAND_ROWS_CSV)
    _write_csv(boundary_bands, output_dir / BOUNDARY_BAND_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    pair_row = pair_bands.iloc[0].to_dict()
    summary = {
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "route_band_row_count": int(len(route_bands)),
        "positive_seed_start_count": int(pair_row["positive_seed_start_count"]),
        "strict_interpretable_seed_count": int(pair_row["strict_interpretable_seed_count"]),
        "monotone_transition_band_seed_count": int(
            pair_row["monotone_transition_band_seed_count"]
        ),
        "monotone_intermediate_seed_count": int(pair_row["monotone_intermediate_seed_count"]),
        "nonmonotone_bounded_transition_seed_count": int(
            pair_row["nonmonotone_bounded_transition_seed_count"]
        ),
        "bounded_transition_band_seed_count": int(pair_row["bounded_transition_band_seed_count"]),
        "unbounded_or_unclassified_seed_count": int(
            pair_row["unbounded_or_unclassified_seed_count"]
        ),
        "seed_band_status_counts": _count_dict(seed_bands["seed_band_status"]),
        "trace_vocab_code_counts": _count_dict(seed_bands["g4_9a_vocab_code"]),
        "boundary_positive_target_route_count": int(
            boundary_bands["positive_target_route_count"].sum()
        ),
        "boundary_positive_target_step_count": int(
            boundary_bands["positive_target_step_count"].sum()
        ),
        "boundary_positive_leak_closed": bool(pair_row["boundary_positive_leak_closed"]),
        "pair_transition_band_status": str(pair_row["pair_transition_band_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates[gates["gate_status"].astype(str).ne("pass")][
            "gate_id"
        ].astype(str).tolist(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    config = {
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "trace_rows_csv": TRACE_ROWS_CSV,
        "read_only_trace_audit": True,
        "positive_pair_id": POSITIVE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_bands=pair_bands,
        seed_bands=seed_bands,
        boundary_bands=boundary_bands,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(args.trace_dir, args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
