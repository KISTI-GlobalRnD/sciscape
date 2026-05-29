#!/usr/bin/env python3
"""Attribute which cheap/pre-p5 signals recover contract-required candidates."""

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


import pandas as pd

from analyze_leiden_attainable_fast_selector import (
    SELECTOR_SPECS,
    _parse_candidate_set,
    _selected_best,
    _select_candidates,
    _subset_elapsed,
)
from analyze_leiden_contract_tiered_subset import (
    TIER_ORDER,
    build_tiered_subset_rows,
)
from analyze_leiden_multibasin_decision_rules import _case_field_method
from analyze_leiden_multibasin_signatures import (
    _finite_float,
    _group_columns,
    _read_csvs,
    _signature_frame,
)
from analyze_leiden_pre_p5_mode_trigger import CHEAP_METRIC_COLUMNS
from analyze_leiden_quality_first_choice import _ranked_by_p1

MAX_METRIC_RANK = 2
SELECTOR_DELTA_SPECS: tuple[tuple[str, str], ...] = (
    ("p1_top1", "cheap_metric_top1_union"),
    ("p1_top3", "p1_top3_plus_metric_top1"),
    ("p1_top5", "p1_top5_plus_metric_top1"),
    ("cheap_metric_top1_union", "cheap_metric_top2_union"),
)

def _base_mask(frame: pd.DataFrame, base: dict[str, Any]) -> pd.Series:
    mask = pd.Series([True] * len(frame), index=frame.index)
    for column, value in base.items():
        if column in frame.columns:
            mask &= frame[column] == value
    return mask

def _candidate_set(frame: pd.DataFrame, column: str = "candidate_index") -> set[int]:
    if frame.empty or column not in frame.columns:
        return set()
    return {
        int(value)
        for value in pd.to_numeric(frame[column], errors="coerce").dropna().tolist()
    }

def _join_candidates(candidates: set[int]) -> str:
    return ";".join(str(candidate) for candidate in sorted(candidates))

def _case_key(base: dict[str, Any], field: str, method: str) -> str:
    pieces = [f"{column}={base[column]}" for column in sorted(base)]
    pieces.extend([f"field={field}", f"method={method}"])
    return "|".join(str(piece) for piece in pieces)

def _rank_by_metric(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in group.columns:
        return pd.DataFrame()
    values = pd.to_numeric(group[metric], errors="coerce")
    if values.notna().sum() == 0:
        return pd.DataFrame()
    return group.assign(
        _selector_metric=values,
        _candidate_index=pd.to_numeric(
            group.get("candidate_index"),
            errors="coerce",
        ).fillna(0),
    ).sort_values(
        ["_selector_metric", "_candidate_index"],
        ascending=[False, True],
        na_position="last",
    )

def _metric_sources(
    ranked: pd.DataFrame,
    *,
    metric_top_n: int,
) -> dict[int, set[str]]:
    sources: dict[int, set[str]] = {}
    for metric in CHEAP_METRIC_COLUMNS:
        metric_ranked = _rank_by_metric(ranked, metric)
        if metric_ranked.empty:
            continue
        for rank, (_, row) in enumerate(metric_ranked.head(metric_top_n).iterrows(), start=1):
            candidate = int(row.get("candidate_index", -1))
            if candidate < 0:
                continue
            sources.setdefault(candidate, set()).add(f"{metric}@rank{rank}")
    return sources

def _prefix_sources(ranked: pd.DataFrame, top_k: int) -> dict[int, set[str]]:
    sources: dict[int, set[str]] = {}
    for _, row in ranked.head(min(top_k, len(ranked))).iterrows():
        candidate = int(row.get("candidate_index", -1))
        if candidate >= 0:
            sources.setdefault(candidate, set()).add(f"p1_top{top_k}")
    return sources

def _merge_source_maps(*maps: dict[int, set[str]]) -> dict[int, set[str]]:
    merged: dict[int, set[str]] = {}
    for source_map in maps:
        for candidate, sources in source_map.items():
            merged.setdefault(candidate, set()).update(sources)
    return merged

def _selector_source_map(ranked: pd.DataFrame, selector_name: str) -> dict[int, set[str]]:
    specs = {name: (family, params) for name, family, params in SELECTOR_SPECS}
    if selector_name not in specs:
        raise ValueError(f"Unknown selector: {selector_name}")
    selector_family, selector_params = specs[selector_name]
    if selector_family == "prefix":
        return _prefix_sources(ranked, int(selector_params["top_k"]))
    if selector_family == "stratified":
        sources: dict[int, set[str]] = {}
        for position in selector_params["positions"]:
            index = int(position) - 1
            if 0 <= index < len(ranked):
                candidate = int(ranked.iloc[index].get("candidate_index", -1))
                if candidate >= 0:
                    sources.setdefault(candidate, set()).add(f"p1_rank{position}")
        return sources
    if selector_family == "cheap_union":
        return _metric_sources(
            ranked,
            metric_top_n=int(selector_params["metric_top_n"]),
        )
    if selector_family == "hybrid":
        return _merge_source_maps(
            _prefix_sources(ranked, int(selector_params["top_k"])),
            _metric_sources(
                ranked,
                metric_top_n=int(selector_params["metric_top_n"]),
            ),
        )
    raise ValueError(f"Unknown selector family: {selector_family}")

def _format_candidate_sources(
    sources: dict[int, set[str]],
    candidates: set[int],
) -> str:
    pieces = []
    for candidate in sorted(candidates):
        candidate_sources = sorted(sources.get(candidate, set()))
        if candidate_sources:
            pieces.append(f"{candidate}:{','.join(candidate_sources)}")
    return ";".join(pieces)

def _source_names(source_string: str) -> set[str]:
    names: set[str] = set()
    for candidate_piece in str(source_string).split(";"):
        if ":" not in candidate_piece:
            continue
        _, source_piece = candidate_piece.split(":", 1)
        for source in source_piece.split(","):
            source = source.strip()
            if not source:
                continue
            names.add(source.split("@", 1)[0])
    return names

def _candidate_requirement_tier(
    candidate: int,
    required_by_tier: dict[str, set[int]],
) -> str:
    if candidate in required_by_tier.get("hard", set()):
        return "hard"
    if candidate in required_by_tier.get("core", set()):
        return "core"
    if candidate in required_by_tier.get("diagnostic", set()):
        return "diagnostic"
    return "not_required"

def _candidate_delta_map(ranked: pd.DataFrame) -> dict[int, float]:
    return {
        int(row.get("candidate_index", -1)): _finite_float(row.get("p5_delta_q"))
        for _, row in ranked.iterrows()
    }

def _rank_map(ranked: pd.DataFrame) -> dict[int, int]:
    return {
        int(row.get("candidate_index", -1)): rank
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1)
    }

def _elapsed_series(ranked: pd.DataFrame) -> pd.Series:
    if "p5_elapsed_ms" in ranked.columns:
        return pd.to_numeric(ranked["p5_elapsed_ms"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(ranked), index=ranked.index, dtype=float)

def _selector_specs_by_name() -> dict[str, tuple[str, dict[str, Any]]]:
    return {name: (family, params) for name, family, params in SELECTOR_SPECS}

def build_signal_candidate_rows(
    candidates: pd.DataFrame,
    *,
    material_regret_q: float = 10.0,
    near_best_delta_q: float = 10.0,
    support_distinct_tau: float = 0.5,
    max_metric_rank: int = MAX_METRIC_RANK,
) -> pd.DataFrame:
    signature_rows = _signature_frame(candidates)
    if signature_rows.empty:
        return pd.DataFrame()
    tier_rows = build_tiered_subset_rows(
        candidates,
        material_regret_q=material_regret_q,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
    )
    group_cols = _group_columns(signature_rows)
    if not group_cols:
        signature_rows = signature_rows.copy()
        signature_rows["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in signature_rows.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        ranked = _ranked_by_p1(group)
        ranked = ranked[pd.to_numeric(ranked.get("p5_delta_q"), errors="coerce").notna()]
        if ranked.empty:
            continue
        case_tiers = tier_rows[_base_mask(tier_rows, base)]
        if case_tiers.empty:
            continue
        required_by_tier = {
            tier: _parse_candidate_set(
                case_tiers[case_tiers["contract_tier"] == tier].iloc[0].get(
                    "required_candidate_indices"
                )
            )
            for tier in TIER_ORDER
            if not case_tiers[case_tiers["contract_tier"] == tier].empty
        }
        first_tier = case_tiers.iloc[0]
        best_candidate = int(first_tier.get("quality_first_candidate_index", -1))
        p1_candidate = int(first_tier.get("p1_candidate_index", -1))
        rank_by_candidate = _rank_map(ranked)
        p5_by_candidate = _candidate_delta_map(ranked)
        best_delta = p5_by_candidate.get(best_candidate, math.nan)
        p1_top1 = _candidate_set(ranked.head(1))
        p1_top3 = _candidate_set(ranked.head(min(3, len(ranked))))
        p1_top5 = _candidate_set(ranked.head(min(5, len(ranked))))
        p1_top10 = _candidate_set(ranked.head(min(10, len(ranked))))
        field, method = _case_field_method(ranked.iloc[0])
        case_key = _case_key(base, field, method)
        for metric in CHEAP_METRIC_COLUMNS:
            metric_ranked = _rank_by_metric(ranked, metric)
            if metric_ranked.empty:
                continue
            for metric_rank, (_, row) in enumerate(
                metric_ranked.head(max_metric_rank).iterrows(),
                start=1,
            ):
                candidate = int(row.get("candidate_index", -1))
                if candidate < 0:
                    continue
                p5_delta = p5_by_candidate.get(candidate, math.nan)
                rows.append(
                    {
                        **base,
                        "case_key": case_key,
                        "field": field,
                        "method": method,
                        "metric_name": metric,
                        "metric_rank": metric_rank,
                        "candidate_index": candidate,
                        "metric_value": _finite_float(row.get(metric)),
                        "candidate_count": int(len(ranked)),
                        "p1_rank": int(rank_by_candidate.get(candidate, 0)),
                        "p1_delta_q": _finite_float(row.get("p1_delta_q")),
                        "p5_delta_q": p5_delta,
                        "p5_gap_to_quality_first_q": (
                            best_delta - p5_delta
                            if math.isfinite(best_delta) and math.isfinite(p5_delta)
                            else math.nan
                        ),
                        "quality_first_candidate_index": best_candidate,
                        "p1_candidate_index": p1_candidate,
                        "is_quality_first_candidate": candidate == best_candidate,
                        "is_p1_candidate": candidate == p1_candidate,
                        "requirement_tier": _candidate_requirement_tier(
                            candidate,
                            required_by_tier,
                        ),
                        "required_hard": candidate in required_by_tier.get("hard", set()),
                        "required_core": candidate in required_by_tier.get("core", set()),
                        "required_diagnostic": candidate
                        in required_by_tier.get("diagnostic", set()),
                        "required_hard_candidate_indices": _join_candidates(
                            required_by_tier.get("hard", set())
                        ),
                        "required_core_candidate_indices": _join_candidates(
                            required_by_tier.get("core", set())
                        ),
                        "required_diagnostic_candidate_indices": _join_candidates(
                            required_by_tier.get("diagnostic", set())
                        ),
                        "selected_by_p1_top1": candidate in p1_top1,
                        "selected_by_p1_top3": candidate in p1_top3,
                        "selected_by_p1_top5": candidate in p1_top5,
                        "selected_by_p1_top10": candidate in p1_top10,
                        "metric_recovers_hard_vs_p1_top1": (
                            candidate in required_by_tier.get("hard", set())
                            and candidate not in p1_top1
                        ),
                        "metric_recovers_core_vs_p1_top3": (
                            candidate in required_by_tier.get("core", set())
                            and candidate not in p1_top3
                        ),
                        "metric_recovers_core_vs_p1_top5": (
                            candidate in required_by_tier.get("core", set())
                            and candidate not in p1_top5
                        ),
                        "metric_recovers_diagnostic_vs_p1_top10": (
                            candidate in required_by_tier.get("diagnostic", set())
                            and candidate not in p1_top10
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "metric_name", "metric_rank", "candidate_index"],
        na_position="last",
    )

def build_metric_signal_summary(
    candidate_rows: pd.DataFrame,
    *,
    max_metric_rank: int = MAX_METRIC_RANK,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for metric, metric_rows in candidate_rows.groupby("metric_name", dropna=False):
        for top_n in range(1, max_metric_rank + 1):
            top_rows = metric_rows[
                pd.to_numeric(metric_rows["metric_rank"], errors="coerce") <= top_n
            ]
            case_rows: list[dict[str, Any]] = []
            for case_key, case in top_rows.groupby("case_key", dropna=False):
                selected = _candidate_set(case)
                first = case.iloc[0]
                hard = _parse_candidate_set(first.get("required_hard_candidate_indices"))
                core = _parse_candidate_set(first.get("required_core_candidate_indices"))
                diagnostic = _parse_candidate_set(
                    first.get("required_diagnostic_candidate_indices")
                )
                selected_hard = selected & hard
                selected_core = selected & core
                selected_diagnostic = selected & diagnostic
                case_rows.append(
                    {
                        "case_key": case_key,
                        "selected_candidate_count": len(selected),
                        "hard_hit": len(selected_hard) > 0,
                        "hard_full_cover": bool(hard and hard <= selected),
                        "hard_recover_vs_p1_top1": bool(
                            case[
                                case["candidate_index"].isin(selected_hard)
                            ]["metric_recovers_hard_vs_p1_top1"].map(bool).any()
                        ),
                        "core_hit": len(selected_core) > 0,
                        "core_full_cover": bool(core and core <= selected),
                        "core_recover_vs_p1_top3": bool(
                            case[
                                case["candidate_index"].isin(selected_core)
                            ]["metric_recovers_core_vs_p1_top3"].map(bool).any()
                        ),
                        "core_recover_vs_p1_top5": bool(
                            case[
                                case["candidate_index"].isin(selected_core)
                            ]["metric_recovers_core_vs_p1_top5"].map(bool).any()
                        ),
                        "diagnostic_hit": len(selected_diagnostic) > 0,
                        "diagnostic_recover_vs_p1_top10": bool(
                            case[
                                case["candidate_index"].isin(selected_diagnostic)
                            ]["metric_recovers_diagnostic_vs_p1_top10"].map(bool).any()
                        ),
                        "endpoint_hit": bool(
                            case["is_quality_first_candidate"].map(bool).any()
                        ),
                        "endpoint_recover_vs_p1_top1": bool(
                            case[
                                case["is_quality_first_candidate"].map(bool)
                            ]["selected_by_p1_top1"].map(lambda value: not bool(value)).any()
                        ),
                    }
                )
            if not case_rows:
                continue
            case_frame = pd.DataFrame(case_rows)
            selected_count = pd.to_numeric(
                case_frame["selected_candidate_count"],
                errors="coerce",
            )
            rows.append(
                {
                    "metric_name": str(metric),
                    "metric_top_n": top_n,
                    "case_count": int(len(case_frame)),
                    "selected_candidate_count_mean": _finite_float(
                        selected_count.mean()
                    ),
                    "hard_candidate_hit_case_count": int(
                        case_frame["hard_hit"].map(bool).sum()
                    ),
                    "hard_full_cover_case_count": int(
                        case_frame["hard_full_cover"].map(bool).sum()
                    ),
                    "hard_recover_vs_p1_top1_case_count": int(
                        case_frame["hard_recover_vs_p1_top1"].map(bool).sum()
                    ),
                    "core_candidate_hit_case_count": int(
                        case_frame["core_hit"].map(bool).sum()
                    ),
                    "core_full_cover_case_count": int(
                        case_frame["core_full_cover"].map(bool).sum()
                    ),
                    "core_recover_vs_p1_top3_case_count": int(
                        case_frame["core_recover_vs_p1_top3"].map(bool).sum()
                    ),
                    "core_recover_vs_p1_top5_case_count": int(
                        case_frame["core_recover_vs_p1_top5"].map(bool).sum()
                    ),
                    "diagnostic_candidate_hit_case_count": int(
                        case_frame["diagnostic_hit"].map(bool).sum()
                    ),
                    "diagnostic_recover_vs_p1_top10_case_count": int(
                        case_frame["diagnostic_recover_vs_p1_top10"].map(bool).sum()
                    ),
                    "endpoint_hit_case_count": int(
                        case_frame["endpoint_hit"].map(bool).sum()
                    ),
                    "endpoint_recover_vs_p1_top1_case_count": int(
                        case_frame["endpoint_recover_vs_p1_top1"].map(bool).sum()
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "hard_recover_vs_p1_top1_case_count",
            "core_recover_vs_p1_top3_case_count",
            "endpoint_recover_vs_p1_top1_case_count",
            "metric_name",
            "metric_top_n",
        ],
        ascending=[False, False, False, True, True],
    )

def build_selector_delta_rows(
    candidates: pd.DataFrame,
    *,
    material_regret_q: float = 10.0,
    near_best_delta_q: float = 10.0,
    support_distinct_tau: float = 0.5,
) -> pd.DataFrame:
    signature_rows = _signature_frame(candidates)
    if signature_rows.empty:
        return pd.DataFrame()
    tier_rows = build_tiered_subset_rows(
        candidates,
        material_regret_q=material_regret_q,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
    )
    specs = _selector_specs_by_name()
    group_cols = _group_columns(signature_rows)
    if not group_cols:
        signature_rows = signature_rows.copy()
        signature_rows["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in signature_rows.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        ranked = _ranked_by_p1(group)
        ranked = ranked[pd.to_numeric(ranked.get("p5_delta_q"), errors="coerce").notna()]
        if ranked.empty:
            continue
        case_tiers = tier_rows[_base_mask(tier_rows, base)]
        if case_tiers.empty:
            continue
        elapsed = _elapsed_series(ranked)
        accurate_elapsed = _finite_float(elapsed.sum())
        p5_by_candidate = _candidate_delta_map(ranked)
        field, method = _case_field_method(ranked.iloc[0])
        case_key = _case_key(base, field, method)
        for base_selector, expanded_selector in SELECTOR_DELTA_SPECS:
            base_family, base_params = specs[base_selector]
            expanded_family, expanded_params = specs[expanded_selector]
            base_selected = _select_candidates(ranked, base_family, base_params)
            expanded_selected = _select_candidates(
                ranked,
                expanded_family,
                expanded_params,
            )
            base_sources = _selector_source_map(ranked, base_selector)
            expanded_sources = _selector_source_map(ranked, expanded_selector)
            added_candidates = expanded_selected - base_selected
            base_best_candidate, base_best_delta = _selected_best(ranked, base_selected)
            expanded_best_candidate, expanded_best_delta = _selected_best(
                ranked,
                expanded_selected,
            )
            for _, tier_row in case_tiers.iterrows():
                tier = str(tier_row.get("contract_tier"))
                required = _parse_candidate_set(tier_row.get("required_candidate_indices"))
                base_missing = required - base_selected
                expanded_missing = required - expanded_selected
                newly_covered = (required & expanded_selected) - (
                    required & base_selected
                )
                best_candidate = int(tier_row.get("quality_first_candidate_index", -1))
                best_delta = p5_by_candidate.get(best_candidate, math.nan)
                base_regret = (
                    best_delta - base_best_delta
                    if math.isfinite(best_delta) and math.isfinite(base_best_delta)
                    else math.nan
                )
                expanded_regret = (
                    best_delta - expanded_best_delta
                    if math.isfinite(best_delta) and math.isfinite(expanded_best_delta)
                    else math.nan
                )
                base_elapsed = _subset_elapsed(ranked, elapsed, base_selected)
                expanded_elapsed = _subset_elapsed(ranked, elapsed, expanded_selected)
                rows.append(
                    {
                        **base,
                        "case_key": case_key,
                        "field": field,
                        "method": method,
                        "contract_tier": tier,
                        "base_selector": base_selector,
                        "expanded_selector": expanded_selector,
                        "candidate_count": int(len(ranked)),
                        "base_selected_candidate_indices": _join_candidates(
                            base_selected
                        ),
                        "expanded_selected_candidate_indices": _join_candidates(
                            expanded_selected
                        ),
                        "added_candidate_indices": _join_candidates(added_candidates),
                        "required_candidate_indices": _join_candidates(required),
                        "base_missing_required_candidate_indices": _join_candidates(
                            base_missing
                        ),
                        "expanded_missing_required_candidate_indices": _join_candidates(
                            expanded_missing
                        ),
                        "newly_covered_required_candidate_indices": _join_candidates(
                            newly_covered
                        ),
                        "newly_covered_required_candidate_count": int(
                            len(newly_covered)
                        ),
                        "base_tier_covered": len(base_missing) == 0,
                        "expanded_tier_covered": len(expanded_missing) == 0,
                        "tier_cover_flip": len(base_missing) > 0
                        and len(expanded_missing) == 0,
                        "base_required_candidate_coverage_fraction": (
                            (len(required) - len(base_missing)) / len(required)
                            if required
                            else 1.0
                        ),
                        "expanded_required_candidate_coverage_fraction": (
                            (len(required) - len(expanded_missing)) / len(required)
                            if required
                            else 1.0
                        ),
                        "required_candidate_coverage_fraction_gain": (
                            len(base_missing) - len(expanded_missing)
                        )
                        / len(required)
                        if required
                        else 0.0,
                        "responsible_signal_sources": _format_candidate_sources(
                            expanded_sources,
                            newly_covered,
                        ),
                        "base_signal_sources_for_required": _format_candidate_sources(
                            base_sources,
                            required & base_selected,
                        ),
                        "base_selected_best_candidate_index": base_best_candidate,
                        "expanded_selected_best_candidate_index": expanded_best_candidate,
                        "quality_first_candidate_index": best_candidate,
                        "base_quality_regret_q": base_regret,
                        "expanded_quality_regret_q": expanded_regret,
                        "quality_regret_q_reduction": (
                            base_regret - expanded_regret
                            if math.isfinite(base_regret)
                            and math.isfinite(expanded_regret)
                            else math.nan
                        ),
                        "base_material_regret": bool(
                            math.isfinite(base_regret)
                            and base_regret >= material_regret_q
                        ),
                        "expanded_material_regret": bool(
                            math.isfinite(expanded_regret)
                            and expanded_regret >= material_regret_q
                        ),
                        "material_regret_fixed": bool(
                            math.isfinite(base_regret)
                            and math.isfinite(expanded_regret)
                            and base_regret >= material_regret_q
                            and expanded_regret < material_regret_q
                        ),
                        "base_elapsed_ratio_vs_accurate": (
                            base_elapsed / accurate_elapsed
                            if accurate_elapsed > 0.0
                            else math.nan
                        ),
                        "expanded_elapsed_ratio_vs_accurate": (
                            expanded_elapsed / accurate_elapsed
                            if accurate_elapsed > 0.0
                            else math.nan
                        ),
                        "elapsed_ratio_increase": (
                            (expanded_elapsed - base_elapsed) / accurate_elapsed
                            if accurate_elapsed > 0.0
                            else math.nan
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "field",
            "method",
            "base_selector",
            "expanded_selector",
            "contract_tier",
        ],
        na_position="last",
    )

def build_selector_gain_summary(delta_rows: pd.DataFrame) -> pd.DataFrame:
    if delta_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (base_selector, expanded_selector, tier), group in delta_rows.groupby(
        ["base_selector", "expanded_selector", "contract_tier"],
        dropna=False,
    ):
        gain = pd.to_numeric(
            group["required_candidate_coverage_fraction_gain"],
            errors="coerce",
        )
        elapsed = pd.to_numeric(group["elapsed_ratio_increase"], errors="coerce")
        regret_reduction = pd.to_numeric(
            group["quality_regret_q_reduction"],
            errors="coerce",
        )
        rows.append(
            {
                "base_selector": str(base_selector),
                "expanded_selector": str(expanded_selector),
                "contract_tier": str(tier),
                "case_count": int(len(group)),
                "newly_covered_case_count": int(
                    (
                        pd.to_numeric(
                            group["newly_covered_required_candidate_count"],
                            errors="coerce",
                        ).fillna(0)
                        > 0
                    ).sum()
                ),
                "tier_cover_flip_count": int(group["tier_cover_flip"].map(bool).sum()),
                "material_regret_fixed_count": int(
                    group["material_regret_fixed"].map(bool).sum()
                ),
                "coverage_fraction_gain_mean": _finite_float(gain.mean()),
                "quality_regret_q_reduction_sum": _finite_float(
                    regret_reduction.fillna(0.0).sum()
                ),
                "elapsed_ratio_increase_mean": _finite_float(elapsed.mean()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "tier_cover_flip_count",
            "newly_covered_case_count",
            "quality_regret_q_reduction_sum",
            "base_selector",
            "contract_tier",
        ],
        ascending=[False, False, False, True, True],
    )

def build_realized_signal_summary(delta_rows: pd.DataFrame) -> pd.DataFrame:
    if delta_rows.empty:
        return pd.DataFrame()
    source_rows: list[dict[str, Any]] = []
    gain_rows = delta_rows[
        pd.to_numeric(
            delta_rows["newly_covered_required_candidate_count"],
            errors="coerce",
        ).fillna(0)
        > 0
    ]
    for _, row in gain_rows.iterrows():
        for source in _source_names(row.get("responsible_signal_sources", "")):
            source_rows.append(
                {
                    "signal_name": source,
                    "base_selector": row.get("base_selector"),
                    "expanded_selector": row.get("expanded_selector"),
                    "contract_tier": row.get("contract_tier"),
                    "case_key": row.get("case_key"),
                    "tier_cover_flip": bool(row.get("tier_cover_flip")),
                    "material_regret_fixed": bool(row.get("material_regret_fixed")),
                    "quality_regret_q_reduction": _finite_float(
                        row.get("quality_regret_q_reduction")
                    ),
                }
            )
    if not source_rows:
        return pd.DataFrame()
    source_frame = pd.DataFrame(source_rows)
    rows: list[dict[str, Any]] = []
    for (signal, expanded_selector, tier), group in source_frame.groupby(
        ["signal_name", "expanded_selector", "contract_tier"],
        dropna=False,
    ):
        regret_reduction = pd.to_numeric(
            group["quality_regret_q_reduction"],
            errors="coerce",
        )
        rows.append(
            {
                "signal_name": str(signal),
                "expanded_selector": str(expanded_selector),
                "contract_tier": str(tier),
                "newly_covered_case_count": int(group["case_key"].nunique()),
                "tier_cover_flip_count": int(group["tier_cover_flip"].map(bool).sum()),
                "material_regret_fixed_count": int(
                    group["material_regret_fixed"].map(bool).sum()
                ),
                "quality_regret_q_reduction_sum": _finite_float(
                    regret_reduction.fillna(0.0).sum()
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "tier_cover_flip_count",
            "newly_covered_case_count",
            "quality_regret_q_reduction_sum",
            "signal_name",
        ],
        ascending=[False, False, False, True],
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
    candidate_rows: pd.DataFrame,
    metric_summary: pd.DataFrame,
    delta_rows: pd.DataFrame,
    gain_summary: pd.DataFrame,
    realized_signal_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Selector Signal Attribution",
        "",
        "This diagnostic attributes which cheap/pre-p5 signals recover hard/core/diagnostic contract-required candidates. The selector still does not use p5 labels; p5-derived tiers are used only as evaluation labels.",
        "",
    ]
    if candidate_rows.empty:
        lines.append("- No signal attribution rows were available.")
    else:
        lines.extend(["## Metric Recovery Potential", ""])
        display_metric = metric_summary[
            [
                column
                for column in [
                    "metric_name",
                    "metric_top_n",
                    "case_count",
                    "hard_recover_vs_p1_top1_case_count",
                    "core_recover_vs_p1_top3_case_count",
                    "core_recover_vs_p1_top5_case_count",
                    "endpoint_recover_vs_p1_top1_case_count",
                    "diagnostic_recover_vs_p1_top10_case_count",
                ]
                if column in metric_summary.columns
            ]
        ].head(16)
        lines.extend(_markdown_table(display_metric).splitlines())
        lines.extend(["", "## Realized Selector Gains", ""])
        display_gain = gain_summary[
            [
                column
                for column in [
                    "base_selector",
                    "expanded_selector",
                    "contract_tier",
                    "newly_covered_case_count",
                    "tier_cover_flip_count",
                    "material_regret_fixed_count",
                    "quality_regret_q_reduction_sum",
                    "elapsed_ratio_increase_mean",
                ]
                if column in gain_summary.columns
            ]
        ].head(16)
        lines.extend(_markdown_table(display_gain).splitlines())
        lines.extend(["", "## Signal Sources Behind Gains", ""])
        if realized_signal_summary.empty:
            lines.append("- No selector gains had attributable signal sources.")
        else:
            display_sources = realized_signal_summary[
                [
                    column
                    for column in [
                        "signal_name",
                        "expanded_selector",
                        "contract_tier",
                        "newly_covered_case_count",
                        "tier_cover_flip_count",
                        "material_regret_fixed_count",
                        "quality_regret_q_reduction_sum",
                    ]
                    if column in realized_signal_summary.columns
                ]
            ].head(20)
            lines.extend(_markdown_table(display_sources).splitlines())
        lines.extend(["", "## Largest Remaining Core Gaps", ""])
        core_gaps = delta_rows[
            (delta_rows["contract_tier"] == "core")
            & (~delta_rows["expanded_tier_covered"].map(bool))
        ].copy()
        if core_gaps.empty:
            lines.append("- No expanded-selector core gaps were available.")
        else:
            core_gaps["_gap_score"] = (
                pd.to_numeric(
                    core_gaps["required_candidate_coverage_fraction_gain"],
                    errors="coerce",
                )
                .fillna(0.0)
                .mul(-1.0)
                + pd.to_numeric(core_gaps["expanded_quality_regret_q"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            display_gaps = core_gaps.sort_values("_gap_score", ascending=False).head(16)
            display_gaps = display_gaps[
                [
                    column
                    for column in [
                        "field",
                        "method",
                        "base_selector",
                        "expanded_selector",
                        "required_candidate_indices",
                        "expanded_selected_candidate_indices",
                        "expanded_missing_required_candidate_indices",
                        "expanded_quality_regret_q",
                        "expanded_elapsed_ratio_vs_accurate",
                    ]
                    if column in display_gaps.columns
                ]
            ]
            lines.extend(_markdown_table(display_gaps).splitlines())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- A useful fast-mode signal should recover hard/core obligations or remove material regret at modest added p5 cost.",
                "- A signal that only recovers diagnostic obligations can still be valuable for accurate-mode explanation, but it is weaker evidence for production fast mode.",
                "- If the same gains come from many redundant cheap metrics, the next step is to reduce the signal family rather than add more thresholds.",
            ]
        )
    (output_dir / "dongdaemun_selector_signal_attribution_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory to scan recursively for candidate_level_rows.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--material-regret-q", type=float, default=10.0)
    parser.add_argument("--near-best-delta-q", type=float, default=10.0)
    parser.add_argument("--support-distinct-tau", type=float, default=0.5)
    parser.add_argument("--max-metric-rank", type=int, default=MAX_METRIC_RANK)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    frames = [
        _read_csvs(input_dir.expanduser().resolve(), "candidate_level_rows.csv")
        for input_dir in args.input_dir
    ]
    frames = [frame for frame in frames if not frame.empty]
    candidates = (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = build_signal_candidate_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
        max_metric_rank=args.max_metric_rank,
    )
    metric_summary = build_metric_signal_summary(
        candidate_rows,
        max_metric_rank=args.max_metric_rank,
    )
    delta_rows = build_selector_delta_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    gain_summary = build_selector_gain_summary(delta_rows)
    realized_signal_summary = build_realized_signal_summary(delta_rows)
    candidate_rows.to_csv(
        output_dir / "dongdaemun_selector_signal_attribution_candidate_rows.csv",
        index=False,
    )
    metric_summary.to_csv(
        output_dir / "dongdaemun_selector_signal_attribution_metric_summary.csv",
        index=False,
    )
    delta_rows.to_csv(
        output_dir / "dongdaemun_selector_signal_attribution_delta_rows.csv",
        index=False,
    )
    gain_summary.to_csv(
        output_dir / "dongdaemun_selector_signal_attribution_gain_summary.csv",
        index=False,
    )
    realized_signal_summary.to_csv(
        output_dir / "dongdaemun_selector_signal_attribution_realized_signal_summary.csv",
        index=False,
    )
    write_report(
        output_dir,
        candidate_rows,
        metric_summary,
        delta_rows,
        gain_summary,
        realized_signal_summary,
    )
    print(
        {
            "candidate_rows": int(len(candidates)),
            "signal_candidate_rows": int(len(candidate_rows)),
            "metric_summary_rows": int(len(metric_summary)),
            "selector_delta_rows": int(len(delta_rows)),
            "selector_gain_summary_rows": int(len(gain_summary)),
            "realized_signal_summary_rows": int(len(realized_signal_summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
