#!/usr/bin/env python3
"""Audit whether useful Dongdaemun basins are reachable by vanilla seed sweeps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import numpy as np
import pandas as pd

from analyze_leiden_accurate_basin_portfolio_contract import (
    build_accurate_contract_rows,
    build_accurate_portfolio_member_rows,
)
from analyze_leiden_contract_tiered_subset import build_pair_tier_rows
from analyze_leiden_multibasin_decision_rules import _case_field_method
from analyze_leiden_multibasin_signatures import (
    CHANGED_SUPPORT_COLUMN,
    SKETCH_HASH_COLUMN,
    SKETCH_MEMBERSHIP_COLUMN,
    _coassignment_bits,
    _finite_float,
    _group_columns,
    _jaccard_distance,
    _parse_sketch,
    _read_csvs,
    _signature_frame,
)

VANILLA_FILENAMES = (
    "vanilla_basin_rows.csv",
    "standard_leiden_basin_rows.csv",
    "leiden_vanilla_basin_rows.csv",
    "leiden_random_refinement_profile_rows.csv",
)
TARGET_ORDER = {
    "material_winner": 0,
    "core_alternative": 1,
    "diagnostic_alternative": 2,
}

def _base_mask(frame: pd.DataFrame, base: dict[str, Any]) -> pd.Series:
    mask = pd.Series([True] * len(frame), index=frame.index)
    for column, value in base.items():
        if column in frame.columns:
            mask &= frame[column] == value
    return mask

def _candidate_set_from_pair_tiers(pair_tiers: pd.DataFrame, tier: str) -> set[int]:
    out: set[int] = set()
    if pair_tiers.empty or "pair_obligation_tier" not in pair_tiers.columns:
        return out
    group = pair_tiers[pair_tiers["pair_obligation_tier"] == tier]
    for _, row in group.iterrows():
        for column in ("left_candidate_index", "right_candidate_index"):
            value = row.get(column)
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                continue
            if candidate >= 0:
                out.add(candidate)
    return out

def _target_class(
    *,
    role: str,
    candidate: int,
    core_candidates: set[int],
    diagnostic_candidates: set[int],
    material_contract: bool,
) -> str:
    roles = set(str(role).split(";"))
    if "quality_first_best" in roles and material_contract:
        return "material_winner"
    if candidate in core_candidates or "near_qf_alternative" in roles:
        return "core_alternative"
    if candidate in diagnostic_candidates or "support_distinct_lookalike" in roles:
        return "diagnostic_alternative"
    return "not_target"

def _candidate_lookup(signature_rows: pd.DataFrame) -> dict[tuple[Any, ...], pd.Series]:
    group_cols = _group_columns(signature_rows)
    lookup: dict[tuple[Any, ...], pd.Series] = {}
    for _, row in signature_rows.iterrows():
        key = tuple(row.get(column) for column in group_cols) + (
            int(row.get("candidate_index", -1)),
        )
        lookup[key] = row
    return lookup

def _row_case_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        row.get("candidate_eval_mode"),
        row.get("case"),
        row.get("seed"),
        row.get("candidate_budget"),
        row.get("max_group_candidates"),
    )

def build_target_basin_rows(
    candidates: pd.DataFrame,
    *,
    material_regret_q: float = 10.0,
    near_best_delta_q: float = 10.0,
    support_distinct_tau: float = 0.5,
) -> pd.DataFrame:
    signature_rows = _signature_frame(candidates)
    if signature_rows.empty:
        return pd.DataFrame()
    contract_rows = build_accurate_contract_rows(
        candidates,
        material_regret_q=material_regret_q,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
    )
    member_rows = build_accurate_portfolio_member_rows(
        candidates,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
    )
    pair_tiers = build_pair_tier_rows(
        candidates,
        material_regret_q=material_regret_q,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
    )
    group_cols = _group_columns(signature_rows)
    candidate_lookup = _candidate_lookup(signature_rows)
    rows: list[dict[str, Any]] = []
    for group_key, members in member_rows.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        contract = contract_rows[_base_mask(contract_rows, base)]
        if contract.empty:
            continue
        pair_group = pair_tiers[_base_mask(pair_tiers, base)]
        core_candidates = _candidate_set_from_pair_tiers(pair_group, "core")
        diagnostic_candidates = _candidate_set_from_pair_tiers(pair_group, "diagnostic")
        material_contract = bool(
            contract.iloc[0].get("quality_first_material_premium", False)
        )
        field, method = _case_field_method(members.iloc[0])
        for _, member in members.iterrows():
            candidate = int(member.get("candidate_index", -1))
            target_class = _target_class(
                role=str(member.get("portfolio_role", "")),
                candidate=candidate,
                core_candidates=core_candidates,
                diagnostic_candidates=diagnostic_candidates,
                material_contract=material_contract,
            )
            if target_class == "not_target":
                continue
            lookup_key = tuple(base.get(column) for column in group_cols) + (candidate,)
            source = candidate_lookup.get(lookup_key)
            rows.append(
                {
                    **base,
                    "field": field,
                    "method": method,
                    "target_class": target_class,
                    "candidate_index": candidate,
                    "portfolio_role": member.get("portfolio_role"),
                    "p1_rank": member.get("p1_rank"),
                    "p5_delta_q": _finite_float(member.get("p5_delta_q")),
                    "q_gap_to_best": _finite_float(member.get("q_gap_to_best")),
                    "distance_to_best_support": _finite_float(
                        member.get("distance_to_best_support")
                    ),
                    "same_coarse_as_best": bool(member.get("same_coarse_as_best")),
                    "p5_basin_signature": (
                        "" if source is None else str(source.get("p5_basin_signature", ""))
                    ),
                    SKETCH_HASH_COLUMN: (
                        "" if source is None else str(source.get(SKETCH_HASH_COLUMN, ""))
                    ),
                    SKETCH_MEMBERSHIP_COLUMN: (
                        "" if source is None else str(source.get(SKETCH_MEMBERSHIP_COLUMN, ""))
                    ),
                    CHANGED_SUPPORT_COLUMN: (
                        "" if source is None else str(source.get(CHANGED_SUPPORT_COLUMN, ""))
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "target_class", "q_gap_to_best", "candidate_index"],
        key=lambda values: values.map(lambda value: TARGET_ORDER.get(value, value)),
    )

def _read_vanilla_rows(vanilla_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in vanilla_dirs:
        root = directory.expanduser().resolve()
        for filename in VANILLA_FILENAMES:
            frame = _read_csvs(root, filename)
            if not frame.empty:
                frame["vanilla_filename"] = filename
                frame["vanilla_root"] = str(root)
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def _normalize_vanilla_rows(vanilla: pd.DataFrame) -> pd.DataFrame:
    if vanilla.empty:
        return vanilla
    out = vanilla.copy()
    if "p5_basin_signature" not in out.columns:
        for alias in ("basin_signature", "signature", "membership_signature"):
            if alias in out.columns:
                out["p5_basin_signature"] = out[alias]
                break
    if SKETCH_MEMBERSHIP_COLUMN not in out.columns:
        for alias in ("basin_sketch_membership", "sketch_membership"):
            if alias in out.columns:
                out[SKETCH_MEMBERSHIP_COLUMN] = out[alias]
                break
    if SKETCH_HASH_COLUMN not in out.columns:
        for alias in ("basin_sketch_node_hash", "sketch_node_hash"):
            if alias in out.columns:
                out[SKETCH_HASH_COLUMN] = out[alias]
                break
    if CHANGED_SUPPORT_COLUMN not in out.columns:
        for alias in ("basin_changed_support_nodes", "changed_support_nodes"):
            if alias in out.columns:
                out[CHANGED_SUPPORT_COLUMN] = out[alias]
                break
    for column in ("field", "method"):
        if column not in out.columns:
            out[column] = out.apply(lambda row: _case_field_method(row)[0 if column == "field" else 1], axis=1)
    return out

def _same_case_vanilla(vanilla: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    if vanilla.empty:
        return vanilla
    if "case" in vanilla.columns and pd.notna(target.get("case")):
        exact = vanilla[vanilla["case"] == target.get("case")]
        if not exact.empty:
            return exact
    mask = pd.Series([True] * len(vanilla), index=vanilla.index)
    if "field" in vanilla.columns and pd.notna(target.get("field")):
        mask &= vanilla["field"] == target.get("field")
    if "method" in vanilla.columns and pd.notna(target.get("method")):
        mask &= vanilla["method"] == target.get("method")
    return vanilla[mask]

def _has_comparable_evidence(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    if "p5_basin_signature" in frame.columns:
        signature = frame["p5_basin_signature"].fillna("").astype(str)
        if signature.str.len().gt(0).any():
            return True
    if SKETCH_MEMBERSHIP_COLUMN in frame.columns:
        sketches = frame[SKETCH_MEMBERSHIP_COLUMN].fillna("").astype(str)
        if sketches.str.len().gt(0).any():
            return True
    return False

def _sketch_metrics(target: pd.Series, candidate: pd.Series) -> dict[str, Any]:
    target_membership = _parse_sketch(target.get(SKETCH_MEMBERSHIP_COLUMN))
    candidate_membership = _parse_sketch(candidate.get(SKETCH_MEMBERSHIP_COLUMN))
    if target_membership.size == 0 or candidate_membership.size == 0:
        return {
            "endpoint_distance": math.nan,
            "support_distance": math.nan,
            "sketch_node_count": 0,
        }
    if target_membership.size != candidate_membership.size:
        return {
            "endpoint_distance": math.nan,
            "support_distance": math.nan,
            "sketch_node_count": int(target_membership.size),
        }
    target_bits = _coassignment_bits(target_membership)
    candidate_bits = _coassignment_bits(candidate_membership)
    if target_bits.size == 0 or target_bits.size != candidate_bits.size:
        endpoint_distance = math.nan
    else:
        endpoint_distance = float(np.mean(target_bits != candidate_bits))
    target_support = _parse_sketch(target.get(CHANGED_SUPPORT_COLUMN))
    candidate_support = _parse_sketch(candidate.get(CHANGED_SUPPORT_COLUMN))
    support_distance, support_intersection, support_union = _jaccard_distance(
        target_support,
        candidate_support,
    )
    if support_union == 0:
        support_distance = math.nan
    target_support_size = len(set(int(value) for value in target_support))
    candidate_support_size = len(set(int(value) for value in candidate_support))
    sketch_node_count = int(target_membership.size)
    support_similarity = (
        math.nan if not math.isfinite(support_distance) else 1.0 - support_distance
    )
    return {
        "endpoint_distance": endpoint_distance,
        "support_distance": support_distance,
        "support_similarity": support_similarity,
        "support_intersection_size": int(support_intersection),
        "support_union_size": int(support_union),
        "target_support_size": int(target_support_size),
        "candidate_support_size": int(candidate_support_size),
        "sketch_node_count": sketch_node_count,
        "support_union_fraction_of_sketch": (
            float(support_union) / float(sketch_node_count)
            if sketch_node_count
            else math.nan
        ),
        "target_support_fraction_of_sketch": (
            float(target_support_size) / float(sketch_node_count)
            if sketch_node_count
            else math.nan
        ),
        "candidate_support_fraction_of_sketch": (
            float(candidate_support_size) / float(sketch_node_count)
            if sketch_node_count
            else math.nan
        ),
    }

def _sketch_distances(target: pd.Series, candidate: pd.Series) -> tuple[float, float]:
    metrics = _sketch_metrics(target, candidate)
    return (
        _finite_float(metrics.get("endpoint_distance")),
        _finite_float(metrics.get("support_distance")),
    )

def _matching_rows(
    target: pd.Series,
    vanilla_case: pd.DataFrame,
    *,
    endpoint_tau: float,
    support_tau: float,
) -> tuple[pd.DataFrame, str]:
    if vanilla_case.empty:
        return pd.DataFrame(), ""
    rows: list[dict[str, Any]] = []
    target_signature = str(target.get("p5_basin_signature", ""))
    for _, vanilla in vanilla_case.iterrows():
        match_type = ""
        endpoint_distance = math.nan
        support_distance = math.nan
        metrics: dict[str, Any] = {}
        signature = str(vanilla.get("p5_basin_signature", ""))
        if target_signature and signature and target_signature == signature:
            match_type = "exact_signature"
            endpoint_distance = 0.0
            support_distance = 0.0
        else:
            target_hash = str(target.get(SKETCH_HASH_COLUMN, ""))
            vanilla_hash = str(vanilla.get(SKETCH_HASH_COLUMN, ""))
            if target_hash and vanilla_hash and target_hash == vanilla_hash:
                metrics = _sketch_metrics(target, vanilla)
                endpoint_distance = _finite_float(metrics.get("endpoint_distance"))
                support_distance = _finite_float(metrics.get("support_distance"))
                endpoint_ok = (
                    math.isfinite(endpoint_distance)
                    and endpoint_distance <= endpoint_tau
                )
                support_ok = (
                    not math.isfinite(support_distance)
                    or support_distance <= support_tau
                )
                if endpoint_ok and support_ok:
                    match_type = "sketch_near"
        if not match_type:
            continue
        rows.append(
            {
                "match_type": match_type,
                "vanilla_seed": vanilla.get("seed"),
                "vanilla_randomness": vanilla.get("randomness"),
                "vanilla_requested_n_iterations": vanilla.get(
                    "requested_n_iterations",
                    vanilla.get("n_iterations"),
                ),
                "vanilla_quality": _finite_float(
                    vanilla.get("quality", vanilla.get("p5_delta_q"))
                ),
                "endpoint_distance": endpoint_distance,
                "support_distance": support_distance,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key not in {"endpoint_distance", "support_distance"}
                },
                "vanilla_source_path": vanilla.get("source_path"),
                "vanilla_filename": vanilla.get("vanilla_filename"),
            }
        )
    if not rows:
        return pd.DataFrame(), ""
    out = pd.DataFrame(rows)
    if (out["match_type"] == "exact_signature").any():
        return out, "exact_signature"
    return out, "sketch_near"

def _nearest_sketch_evidence(
    target: pd.Series,
    vanilla_case: pd.DataFrame,
    *,
    endpoint_tau: float,
    support_tau: float,
) -> dict[str, Any]:
    target_hash = str(target.get(SKETCH_HASH_COLUMN, ""))
    if (
        not target_hash
        or vanilla_case.empty
        or SKETCH_HASH_COLUMN not in vanilla_case.columns
    ):
        return {"same_case_sketch_comparable_count": 0}
    rows: list[dict[str, Any]] = []
    for _, vanilla in vanilla_case.iterrows():
        vanilla_hash = str(vanilla.get(SKETCH_HASH_COLUMN, ""))
        if not vanilla_hash or vanilla_hash != target_hash:
            continue
        metrics = _sketch_metrics(target, vanilla)
        rows.append(
            {
                "best_endpoint_distance": metrics.get("endpoint_distance"),
                "best_support_distance": metrics.get("support_distance"),
                "best_support_similarity": metrics.get("support_similarity"),
                "best_support_intersection_size": metrics.get(
                    "support_intersection_size"
                ),
                "best_support_union_size": metrics.get("support_union_size"),
                "best_target_support_size": metrics.get("target_support_size"),
                "best_vanilla_support_size": metrics.get("candidate_support_size"),
                "best_sketch_node_count": metrics.get("sketch_node_count"),
                "best_support_union_fraction_of_sketch": metrics.get(
                    "support_union_fraction_of_sketch"
                ),
                "best_target_support_fraction_of_sketch": metrics.get(
                    "target_support_fraction_of_sketch"
                ),
                "best_vanilla_support_fraction_of_sketch": metrics.get(
                    "candidate_support_fraction_of_sketch"
                ),
                "best_sketch_seed": vanilla.get("seed"),
                "best_sketch_randomness": vanilla.get("randomness"),
                "best_sketch_requested_n_iterations": vanilla.get(
                    "requested_n_iterations",
                    vanilla.get("n_iterations"),
                ),
            }
        )
    if not rows:
        return {"same_case_sketch_comparable_count": 0}
    evidence = pd.DataFrame(rows)
    orderable = evidence.assign(
        _endpoint=evidence["best_endpoint_distance"].map(
            lambda value: _finite_float(value, math.inf)
        ),
        _support=evidence["best_support_distance"].map(
            lambda value: _finite_float(value, math.inf)
        ),
    ).sort_values(["_endpoint", "_support"], na_position="last")
    best = orderable.iloc[0].to_dict()
    endpoint = _finite_float(best.get("best_endpoint_distance"))
    support = _finite_float(best.get("best_support_distance"))
    endpoint_near = math.isfinite(endpoint) and endpoint <= endpoint_tau
    support_passes = not math.isfinite(support) or support <= support_tau
    return {
        "same_case_sketch_comparable_count": int(len(evidence)),
        "best_endpoint_distance": endpoint,
        "best_support_distance": support,
        "best_support_similarity": best.get("best_support_similarity"),
        "best_support_intersection_size": best.get("best_support_intersection_size"),
        "best_support_union_size": best.get("best_support_union_size"),
        "best_target_support_size": best.get("best_target_support_size"),
        "best_vanilla_support_size": best.get("best_vanilla_support_size"),
        "best_sketch_node_count": best.get("best_sketch_node_count"),
        "best_support_union_fraction_of_sketch": best.get(
            "best_support_union_fraction_of_sketch"
        ),
        "best_target_support_fraction_of_sketch": best.get(
            "best_target_support_fraction_of_sketch"
        ),
        "best_vanilla_support_fraction_of_sketch": best.get(
            "best_vanilla_support_fraction_of_sketch"
        ),
        "best_sketch_seed": best.get("best_sketch_seed"),
        "best_sketch_randomness": best.get("best_sketch_randomness"),
        "best_sketch_requested_n_iterations": best.get(
            "best_sketch_requested_n_iterations"
        ),
        "best_endpoint_within_tau": bool(endpoint_near),
        "best_support_within_tau": bool(support_passes),
        "endpoint_near_support_far": bool(endpoint_near and not support_passes),
    }

def _reachability_label(matches: pd.DataFrame) -> str:
    if matches.empty:
        return "not_reached_in_available_sweep"
    comparable_iterations = (
        matches["vanilla_requested_n_iterations"].fillna("").astype(str).str.lower()
    )
    only_convergence = comparable_iterations.ne("").all() and comparable_iterations.isin(
        {"0", "convergence"}
    ).all()
    if only_convergence:
        return "iteration_reachable"
    if len(matches) == 1:
        return "rare_seed_reachable"
    return "seed_reachable"

def build_reachability_rows(
    target_rows: pd.DataFrame,
    vanilla_rows: pd.DataFrame,
    *,
    endpoint_tau: float = 0.02,
    support_tau: float = 0.5,
) -> pd.DataFrame:
    if target_rows.empty:
        return pd.DataFrame()
    vanilla = _normalize_vanilla_rows(vanilla_rows)
    rows: list[dict[str, Any]] = []
    for _, target in target_rows.iterrows():
        vanilla_case = _same_case_vanilla(vanilla, target)
        sketch_evidence: dict[str, Any] = {"same_case_sketch_comparable_count": 0}
        if vanilla.empty:
            label = "unresolved_no_vanilla_rows"
            match_count = 0
            evidence = False
            match_type = ""
            matches = pd.DataFrame()
        elif vanilla_case.empty:
            label = "unresolved_no_vanilla_case"
            match_count = 0
            evidence = False
            match_type = ""
            matches = pd.DataFrame()
        elif not _has_comparable_evidence(vanilla_case):
            label = "unresolved_no_vanilla_signature_evidence"
            match_count = 0
            evidence = False
            match_type = ""
            matches = pd.DataFrame()
        else:
            sketch_evidence = _nearest_sketch_evidence(
                target,
                vanilla_case,
                endpoint_tau=endpoint_tau,
                support_tau=support_tau,
            )
            matches, match_type = _matching_rows(
                target,
                vanilla_case,
                endpoint_tau=endpoint_tau,
                support_tau=support_tau,
            )
            label = _reachability_label(matches)
            match_count = len(matches)
            evidence = True
        rows.append(
            {
                "field": target.get("field"),
                "method": target.get("method"),
                "case": target.get("case"),
                "target_class": target.get("target_class"),
                "candidate_index": target.get("candidate_index"),
                "portfolio_role": target.get("portfolio_role"),
                "p1_rank": target.get("p1_rank"),
                "p5_delta_q": target.get("p5_delta_q"),
                "q_gap_to_best": target.get("q_gap_to_best"),
                "p5_basin_signature": target.get("p5_basin_signature"),
                "reachability_label": label,
                "match_type": match_type,
                "match_count": match_count,
                "same_case_vanilla_row_count": int(len(vanilla_case)),
                "same_case_vanilla_has_signature_evidence": evidence,
                "matched_seed_values": _join_unique(matches, "vanilla_seed"),
                "matched_randomness_values": _join_unique(matches, "vanilla_randomness"),
                "matched_iteration_values": _join_unique(
                    matches,
                    "vanilla_requested_n_iterations",
                ),
                **sketch_evidence,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "target_class", "reachability_label", "candidate_index"],
        key=lambda values: values.map(lambda value: TARGET_ORDER.get(value, value)),
    )

def _join_unique(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    values = [
        str(value)
        for value in frame[column].dropna().tolist()
        if str(value) and str(value).lower() != "nan"
    ]
    return ";".join(sorted(set(values)))

def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return math.nan
    return float(values.median())

def build_reachability_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for target_class, group in rows.groupby("target_class", dropna=False):
        labels = group["reachability_label"].fillna("").astype(str)
        if "same_case_sketch_comparable_count" in group.columns:
            sketch_comparable = pd.to_numeric(
                group["same_case_sketch_comparable_count"],
                errors="coerce",
            ).fillna(0)
        else:
            sketch_comparable = pd.Series(0, index=group.index)
        if "endpoint_near_support_far" in group.columns:
            endpoint_near_support_far = group["endpoint_near_support_far"].fillna(False)
        else:
            endpoint_near_support_far = pd.Series(False, index=group.index)
        near_misses = group[endpoint_near_support_far.map(bool)]
        summary_rows.append(
            {
                "target_class": str(target_class),
                "target_basin_count": int(len(group)),
                "seed_reachable_count": int((labels == "seed_reachable").sum()),
                "rare_seed_reachable_count": int(
                    (labels == "rare_seed_reachable").sum()
                ),
                "iteration_reachable_count": int(
                    (labels == "iteration_reachable").sum()
                ),
                "not_reached_in_available_sweep_count": int(
                    (labels == "not_reached_in_available_sweep").sum()
                ),
                "unresolved_count": int(labels.str.startswith("unresolved").sum()),
                "same_case_vanilla_row_count_sum": int(
                    pd.to_numeric(
                        group["same_case_vanilla_row_count"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "same_case_with_signature_evidence_count": int(
                    group["same_case_vanilla_has_signature_evidence"].map(bool).sum()
                ),
                "same_case_sketch_comparable_count": int(
                    sketch_comparable.gt(0).sum()
                ),
                "endpoint_near_support_far_count": int(
                    endpoint_near_support_far.map(bool).sum()
                ),
                "endpoint_near_support_far_union_median": _numeric_median(
                    near_misses,
                    "best_support_union_size",
                ),
                "endpoint_near_support_far_union_fraction_median": _numeric_median(
                    near_misses,
                    "best_support_union_fraction_of_sketch",
                ),
                "endpoint_near_support_far_target_support_median": _numeric_median(
                    near_misses,
                    "best_target_support_size",
                ),
                "endpoint_near_support_far_vanilla_support_median": _numeric_median(
                    near_misses,
                    "best_vanilla_support_size",
                ),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        "target_class",
        key=lambda values: values.map(lambda value: TARGET_ORDER.get(value, 99)),
    )

def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)

def write_report(
    output_dir: Path,
    target_rows: pd.DataFrame,
    reachability_rows: pd.DataFrame,
    summary: pd.DataFrame,
    vanilla_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Basin Reachability Audit",
        "",
        "This audit asks whether useful Dongdaemun basins are already reachable by vanilla seed/randomness/iteration sweeps. A basin is reachable only when a same-case vanilla row has comparable signature or sketch evidence.",
        "",
        "## Audit Coverage",
        "",
        f"- target basin rows: {len(target_rows)}",
        f"- vanilla evidence rows loaded: {len(vanilla_rows)}",
        "",
    ]
    if summary.empty:
        lines.append("- No reachability summary was available.")
    else:
        lines.extend(["## Reachability Summary", ""])
        lines.extend(_markdown_table(summary).splitlines())
        lines.extend(["", "## Unresolved Or Unreached Targets", ""])
        unresolved = reachability_rows[
            reachability_rows["reachability_label"].astype(str).str.startswith(
                "unresolved"
            )
            | (reachability_rows["reachability_label"] == "not_reached_in_available_sweep")
        ].copy()
        display = unresolved[
            [
                column
                for column in [
                    "field",
                    "method",
                    "target_class",
                    "candidate_index",
                    "portfolio_role",
                    "p1_rank",
                    "q_gap_to_best",
                    "reachability_label",
                    "same_case_vanilla_row_count",
                    "same_case_vanilla_has_signature_evidence",
                    "same_case_sketch_comparable_count",
                    "best_endpoint_distance",
                    "best_support_distance",
                    "best_support_intersection_size",
                    "best_support_union_size",
                    "best_support_union_fraction_of_sketch",
                    "endpoint_near_support_far",
                ]
                if column in unresolved.columns
            ]
        ].head(30)
        lines.extend(_markdown_table(display).splitlines())
        if "endpoint_near_support_far" in reachability_rows.columns:
            near_misses = reachability_rows[
                reachability_rows["endpoint_near_support_far"]
                .fillna(False)
                .map(bool)
            ].copy()
        else:
            near_misses = pd.DataFrame()
        if not near_misses.empty:
            near_display = near_misses[
                [
                    column
                    for column in [
                        "field",
                        "method",
                        "target_class",
                        "candidate_index",
                        "p5_delta_q",
                        "q_gap_to_best",
                        "best_endpoint_distance",
                        "best_support_distance",
                        "best_support_similarity",
                        "best_support_intersection_size",
                        "best_support_union_size",
                        "best_target_support_size",
                        "best_vanilla_support_size",
                        "best_support_union_fraction_of_sketch",
                        "best_sketch_seed",
                        "best_sketch_requested_n_iterations",
                    ]
                    if column in near_misses.columns
                ]
            ].head(30)
            lines.extend(["", "## Endpoint-Near Support-Far Examples", ""])
            lines.extend(_markdown_table(near_display).splitlines())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- `seed_reachable` means the same useful basin appears multiple times in comparable vanilla evidence.",
                "- `rare_seed_reachable` means the basin appears once; this weakens a novelty claim but may still justify a cheaper targeted route.",
                "- `not_reached_in_available_sweep` is only meaningful when same-case vanilla signature/sketch evidence exists.",
                "- `endpoint_near_support_far` marks sketch-near endpoints whose changed support is too different to call the same basin under the current definition.",
                "- Support-far rows should be interpreted with support union/intersection and union fraction; a large distance on a tiny union is weaker evidence than a large distance over a large support footprint.",
                "- `unresolved_*` means the current evidence cannot adjudicate reachability. It must not be interpreted as perturbation-only.",
            ]
        )
    (output_dir / "dongdaemun_basin_reachability_audit_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory to scan recursively for candidate_level_rows.csv",
    )
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory to scan for vanilla basin/reachability evidence rows",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--material-regret-q", type=float, default=10.0)
    parser.add_argument("--near-best-delta-q", type=float, default=10.0)
    parser.add_argument("--support-distinct-tau", type=float, default=0.5)
    parser.add_argument("--endpoint-tau", type=float, default=0.02)
    parser.add_argument("--support-tau", type=float, default=0.5)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    candidate_frames = [
        _read_csvs(directory.expanduser().resolve(), "candidate_level_rows.csv")
        for directory in args.candidate_dir
    ]
    candidate_frames = [frame for frame in candidate_frames if not frame.empty]
    candidates = (
        pd.concat(candidate_frames, ignore_index=True, sort=False)
        if candidate_frames
        else pd.DataFrame()
    )
    vanilla_rows = _read_vanilla_rows(args.vanilla_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rows = build_target_basin_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    reachability_rows = build_reachability_rows(
        target_rows,
        vanilla_rows,
        endpoint_tau=args.endpoint_tau,
        support_tau=args.support_tau,
    )
    summary = build_reachability_summary(reachability_rows)
    target_rows.to_csv(
        output_dir / "dongdaemun_basin_reachability_target_rows.csv",
        index=False,
    )
    reachability_rows.to_csv(
        output_dir / "dongdaemun_basin_reachability_rows.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "dongdaemun_basin_reachability_summary.csv",
        index=False,
    )
    write_report(output_dir, target_rows, reachability_rows, summary, vanilla_rows)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "target_rows": int(len(target_rows)),
            "vanilla_rows": int(len(vanilla_rows)),
            "reachability_rows": int(len(reachability_rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
