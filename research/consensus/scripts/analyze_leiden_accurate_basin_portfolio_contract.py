#!/usr/bin/env python3
"""Define the Dongdaemun-accurate basin portfolio output contract."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_basin_portfolio_evidence import (
    build_basin_portfolio_rows,
    build_lookalike_pair_rows,
)
from analyze_leiden_multibasin_decision_rules import (
    _base_mask,
    _case_field_method,
    _support_sketch_exact,
)
from analyze_leiden_multibasin_signatures import (
    _finite_float,
    _group_columns,
    _mark_material_gain,
    _read_csvs,
    _signature_frame,
    build_coarse_basin_rows,
    build_pairwise_basin_matrix,
)
from analyze_leiden_quality_first_choice import _coarse_by_candidate, _ranked_by_p1


CONTRACT_VERSION = "dongdaemun_accurate_basin_portfolio.v1"


def _key_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "candidate_eval_mode",
            "case",
            "seed",
            "candidate_budget",
            "max_group_candidates",
        ]
        if column in frame.columns
    ]


def _row_key(row: pd.Series, columns: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def _contract_level(
    *,
    material_premium: bool,
    support_distinct_pairs: int,
    near_qf_alternatives: int,
) -> str:
    if material_premium and support_distinct_pairs > 0:
        return "best_plus_near_qf_support_distinct_portfolio"
    if material_premium:
        return "best_plus_quality_premium_evidence"
    if support_distinct_pairs > 0:
        return "best_plus_support_distinct_portfolio"
    if near_qf_alternatives > 0:
        return "best_plus_near_qf_alternatives"
    return "best_endpoint_only"


def _winner_only_risk(
    *,
    material_premium: bool,
    support_distinct_pairs: int,
    near_qf_alternatives: int,
) -> str:
    if material_premium and support_distinct_pairs > 0:
        return "high_quality_and_interpretation_risk"
    if material_premium:
        return "quality_risk"
    if support_distinct_pairs > 0:
        return "interpretation_risk"
    if near_qf_alternatives > 0:
        return "near_tie_review_risk"
    return "low"


def _pairwise_distance_to_best(
    pairwise: pd.DataFrame,
    base: dict[str, Any],
    best_candidate: int,
    candidate: int,
) -> dict[str, Any]:
    if candidate == best_candidate:
        return {
            "distance_to_best_endpoint": 0.0,
            "distance_to_best_support": 0.0,
            "same_coarse_as_best": True,
        }
    if pairwise.empty:
        return {
            "distance_to_best_endpoint": math.nan,
            "distance_to_best_support": math.nan,
            "same_coarse_as_best": None,
        }
    group_pairwise = pairwise[_base_mask(pairwise, base)]
    if group_pairwise.empty:
        return {
            "distance_to_best_endpoint": math.nan,
            "distance_to_best_support": math.nan,
            "same_coarse_as_best": None,
        }
    mask = (
        (group_pairwise["left_candidate_index"] == best_candidate)
        & (group_pairwise["right_candidate_index"] == candidate)
    ) | (
        (group_pairwise["left_candidate_index"] == candidate)
        & (group_pairwise["right_candidate_index"] == best_candidate)
    )
    if not mask.any():
        return {
            "distance_to_best_endpoint": math.nan,
            "distance_to_best_support": math.nan,
            "same_coarse_as_best": None,
        }
    row = group_pairwise[mask].iloc[0]
    return {
        "distance_to_best_endpoint": _finite_float(row.get("sample_coassignment_distance")),
        "distance_to_best_support": _finite_float(row.get("coarse_support_distance")),
        "same_coarse_as_best": bool(row.get("same_coarse_basin", False)),
    }


def _candidate_roles(
    *,
    candidate: int,
    best_candidate: int,
    p1_candidate: int,
    near_best_candidates: set[int],
    lookalike_candidates: set[int],
) -> str:
    roles: list[str] = []
    if candidate == best_candidate:
        roles.append("quality_first_best")
    if candidate == p1_candidate:
        roles.append("p1_choice")
    if candidate in near_best_candidates and candidate != best_candidate:
        roles.append("near_qf_alternative")
    if candidate in lookalike_candidates and candidate != best_candidate:
        roles.append("support_distinct_lookalike")
    return ";".join(roles) if roles else "context_candidate"


def _contract_output_obligations(
    *,
    near_qf_alternatives: int,
    support_distinct_pairs: int,
    material_premium: bool,
) -> str:
    obligations = ["return_best_endpoint"]
    if material_premium:
        obligations.append("report_p1_quality_premium")
    if near_qf_alternatives > 0:
        obligations.append("return_near_qf_alternatives")
    if support_distinct_pairs > 0:
        obligations.append("return_support_distinct_iso_q_pairs")
    obligations.append("report_support_exactness")
    return ";".join(obligations)


def build_accurate_contract_rows(
    candidates: pd.DataFrame,
    *,
    material_regret_q: float = 10.0,
    near_best_delta_q: float = 10.0,
    support_distinct_tau: float = 0.5,
    material_delta_q: float = 1.0,
    material_relative_ppm: float = 10.0,
    coarse_endpoint_tau: float = 0.02,
    coarse_support_tau: float = 0.5,
    iso_q_delta: float = 10.0,
    iso_q_relative_ppm: float = 10.0,
) -> pd.DataFrame:
    signature_rows = _signature_frame(candidates)
    signature_rows = _mark_material_gain(
        signature_rows,
        material_delta_q=material_delta_q,
        material_relative_ppm=material_relative_ppm,
    )
    if signature_rows.empty:
        return pd.DataFrame()
    pairwise = build_pairwise_basin_matrix(
        signature_rows,
        coarse_endpoint_tau=coarse_endpoint_tau,
        coarse_support_tau=coarse_support_tau,
        iso_q_delta=iso_q_delta,
        iso_q_relative_ppm=iso_q_relative_ppm,
    )
    portfolio = build_basin_portfolio_rows(
        candidates,
        material_regret_q=material_regret_q,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
        material_delta_q=material_delta_q,
        material_relative_ppm=material_relative_ppm,
        coarse_endpoint_tau=coarse_endpoint_tau,
        coarse_support_tau=coarse_support_tau,
        iso_q_delta=iso_q_delta,
        iso_q_relative_ppm=iso_q_relative_ppm,
    )
    key_cols = _key_columns(signature_rows)
    portfolio_by_key = {
        _row_key(row, key_cols): row for _, row in portfolio.iterrows()
    }
    group_cols = _group_columns(signature_rows)
    if not group_cols:
        signature_rows = signature_rows.copy()
        signature_rows["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in signature_rows.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        key = tuple(base.get(column) for column in key_cols)
        portfolio_row = portfolio_by_key.get(key, pd.Series(dtype=object))
        ranked = _ranked_by_p1(group)
        if ranked.empty:
            continue
        best_idx = ranked["_p5_delta_q"].idxmax()
        best = ranked.loc[best_idx]
        p1_choice = ranked.iloc[0]
        best_candidate = int(best.get("candidate_index", -1))
        p1_candidate = int(p1_choice.get("candidate_index", -1))
        best_rank = int(list(ranked.index).index(best_idx) + 1)
        best_delta = _finite_float(best.get("p5_delta_q"))
        p1_delta = _finite_float(p1_choice.get("p5_delta_q"))
        premium = best_delta - p1_delta if math.isfinite(best_delta) and math.isfinite(p1_delta) else math.nan
        group_pairwise = pairwise[_base_mask(pairwise, base)] if not pairwise.empty else pairwise
        support_distinct_pairs = int(
            _finite_float(portfolio_row.get("support_distinct_iso_q_pair_count"), 0.0)
        )
        near_qf_alternatives = int(
            _finite_float(portfolio_row.get("near_best_alternative_count"), 0.0)
        )
        material_premium = bool(math.isfinite(premium) and premium >= material_regret_q)
        field, method = _case_field_method(group.iloc[0])
        rows.append(
            {
                **base,
                "field": field,
                "method": method,
                "contract_version": CONTRACT_VERSION,
                "accurate_mode_output_contract": _contract_level(
                    material_premium=material_premium,
                    support_distinct_pairs=support_distinct_pairs,
                    near_qf_alternatives=near_qf_alternatives,
                ),
                "selection_principle": "choose_max_p5_then_attach_basin_portfolio",
                "winner_only_risk": _winner_only_risk(
                    material_premium=material_premium,
                    support_distinct_pairs=support_distinct_pairs,
                    near_qf_alternatives=near_qf_alternatives,
                ),
                "output_obligations": _contract_output_obligations(
                    near_qf_alternatives=near_qf_alternatives,
                    support_distinct_pairs=support_distinct_pairs,
                    material_premium=material_premium,
                ),
                "candidate_count": int(len(group)),
                "coarse_basin_count": int(
                    _finite_float(portfolio_row.get("coarse_basin_count"), 0.0)
                ),
                "support_sketch_exact": _support_sketch_exact(group),
                "quality_first_candidate_index": best_candidate,
                "quality_first_p1_rank": best_rank,
                "quality_first_p5_delta_q": best_delta,
                "quality_first_relative_delta_q_ppm": _finite_float(
                    best.get("p5_relative_delta_q_ppm")
                ),
                "p1_candidate_index": p1_candidate,
                "p1_choice_p5_delta_q": p1_delta,
                "quality_first_premium_over_p1_q": premium,
                "quality_first_material_premium": material_premium,
                "near_best_delta_q": near_best_delta_q,
                "near_qf_candidate_count": int(
                    _finite_float(portfolio_row.get("near_best_candidate_count"), 0.0)
                ),
                "near_qf_alternative_count": near_qf_alternatives,
                "near_qf_coarse_basin_count": int(
                    _finite_float(portfolio_row.get("near_best_coarse_basin_count"), 0.0)
                ),
                "support_distinct_iso_q_pair_count": support_distinct_pairs,
                "partition_distinct_iso_q_pair_count": int(
                    _finite_float(portfolio_row.get("partition_distinct_iso_q_pair_count"), 0.0)
                ),
                "max_lookalike_support_distance": _finite_float(
                    portfolio_row.get("max_lookalike_support_distance")
                ),
                "min_lookalike_q_gap": _finite_float(
                    portfolio_row.get("min_lookalike_q_gap")
                ),
                "pairwise_rows": int(len(group_pairwise)),
                "operator_summary": (
                    "return best endpoint plus portfolio evidence"
                    if support_distinct_pairs > 0 or near_qf_alternatives > 0
                    else "return best endpoint with support exactness"
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "quality_first_material_premium",
            "support_distinct_iso_q_pair_count",
            "quality_first_premium_over_p1_q",
            "field",
            "method",
        ],
        ascending=[False, False, False, True, True],
        na_position="last",
    )


def build_accurate_portfolio_member_rows(
    candidates: pd.DataFrame,
    *,
    near_best_delta_q: float = 10.0,
    support_distinct_tau: float = 0.5,
    material_delta_q: float = 1.0,
    material_relative_ppm: float = 10.0,
    coarse_endpoint_tau: float = 0.02,
    coarse_support_tau: float = 0.5,
    iso_q_delta: float = 10.0,
    iso_q_relative_ppm: float = 10.0,
) -> pd.DataFrame:
    signature_rows = _signature_frame(candidates)
    signature_rows = _mark_material_gain(
        signature_rows,
        material_delta_q=material_delta_q,
        material_relative_ppm=material_relative_ppm,
    )
    if signature_rows.empty:
        return pd.DataFrame()
    pairwise = build_pairwise_basin_matrix(
        signature_rows,
        coarse_endpoint_tau=coarse_endpoint_tau,
        coarse_support_tau=coarse_support_tau,
        iso_q_delta=iso_q_delta,
        iso_q_relative_ppm=iso_q_relative_ppm,
    )
    coarse = build_coarse_basin_rows(signature_rows, pairwise)
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
        if ranked.empty:
            continue
        best_idx = ranked["_p5_delta_q"].idxmax()
        best = ranked.loc[best_idx]
        p1_choice = ranked.iloc[0]
        best_candidate = int(best.get("candidate_index", -1))
        p1_candidate = int(p1_choice.get("candidate_index", -1))
        best_delta = _finite_float(best.get("p5_delta_q"))
        group_pairwise = pairwise[_base_mask(pairwise, base)] if not pairwise.empty else pairwise
        group_coarse = coarse[_base_mask(coarse, base)] if not coarse.empty else coarse
        coarse_map = _coarse_by_candidate(group_coarse)
        near_best = ranked[
            best_delta
            - pd.to_numeric(ranked.get("p5_delta_q"), errors="coerce")
            <= near_best_delta_q
        ].copy()
        near_best_candidates = {
            int(value)
            for value in pd.to_numeric(near_best.get("candidate_index"), errors="coerce")
            .dropna()
            .tolist()
        }
        lookalike_candidates: set[int] = set()
        if not group_pairwise.empty:
            support_distinct = group_pairwise[
                group_pairwise.get("partition_distinct_iso_q_pair", False).map(bool)
                & (
                    pd.to_numeric(
                        group_pairwise.get("coarse_support_distance"),
                        errors="coerce",
                    )
                    >= support_distinct_tau
                )
            ]
            for _, pair in support_distinct.iterrows():
                lookalike_candidates.add(int(pair.get("left_candidate_index", -1)))
                lookalike_candidates.add(int(pair.get("right_candidate_index", -1)))
        selected_candidates = sorted(
            {best_candidate, p1_candidate} | near_best_candidates | lookalike_candidates
        )
        p1_rank_by_candidate = {
            int(row.get("candidate_index", -1)): pos
            for pos, (_, row) in enumerate(ranked.iterrows(), start=1)
        }
        field, method = _case_field_method(group.iloc[0])
        for candidate in selected_candidates:
            candidate_rows = group[
                pd.to_numeric(group.get("candidate_index"), errors="coerce") == candidate
            ]
            if candidate_rows.empty:
                continue
            row = candidate_rows.iloc[0]
            p5_delta = _finite_float(row.get("p5_delta_q"))
            distance = _pairwise_distance_to_best(
                pairwise,
                base,
                best_candidate,
                candidate,
            )
            rows.append(
                {
                    **base,
                    "field": field,
                    "method": method,
                    "contract_version": CONTRACT_VERSION,
                    "candidate_index": candidate,
                    "portfolio_role": _candidate_roles(
                        candidate=candidate,
                        best_candidate=best_candidate,
                        p1_candidate=p1_candidate,
                        near_best_candidates=near_best_candidates,
                        lookalike_candidates=lookalike_candidates,
                    ),
                    "p1_rank": p1_rank_by_candidate.get(candidate),
                    "p5_delta_q": p5_delta,
                    "q_gap_to_best": (
                        best_delta - p5_delta
                        if math.isfinite(best_delta) and math.isfinite(p5_delta)
                        else math.nan
                    ),
                    "p5_relative_delta_q_ppm": _finite_float(
                        row.get("p5_relative_delta_q_ppm")
                    ),
                    "coarse_basin_id": coarse_map.get(candidate),
                    "material_gain": bool(row.get("material_gain", False)),
                    **distance,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "q_gap_to_best", "candidate_index"],
        na_position="last",
    )


def build_contract_summary(contract_rows: pd.DataFrame) -> pd.DataFrame:
    if contract_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_name, group in [("all", contract_rows)]:
        rows.append(_summarize_contract_group(group_name, group))
    for field, group in contract_rows.groupby("field", dropna=False):
        rows.append(_summarize_contract_group(f"field={field}", group))
    for method, group in contract_rows.groupby("method", dropna=False):
        rows.append(_summarize_contract_group(f"method={method}", group))
    return pd.DataFrame(rows)


def _summarize_contract_group(name: str, group: pd.DataFrame) -> dict[str, Any]:
    levels = group["accurate_mode_output_contract"].value_counts()
    risks = group["winner_only_risk"].value_counts()
    return {
        "group": name,
        "case_count": int(len(group)),
        "best_endpoint_only_count": int(levels.get("best_endpoint_only", 0)),
        "best_plus_support_distinct_portfolio_count": int(
            levels.get("best_plus_support_distinct_portfolio", 0)
        ),
        "best_plus_quality_premium_evidence_count": int(
            levels.get("best_plus_quality_premium_evidence", 0)
        ),
        "best_plus_near_qf_alternatives_count": int(
            levels.get("best_plus_near_qf_alternatives", 0)
        ),
        "best_plus_near_qf_support_distinct_portfolio_count": int(
            levels.get("best_plus_near_qf_support_distinct_portfolio", 0)
        ),
        "high_quality_and_interpretation_risk_count": int(
            risks.get("high_quality_and_interpretation_risk", 0)
        ),
        "quality_risk_count": int(risks.get("quality_risk", 0)),
        "interpretation_risk_count": int(risks.get("interpretation_risk", 0)),
        "near_tie_review_risk_count": int(risks.get("near_tie_review_risk", 0)),
        "quality_first_premium_q_sum": _finite_float(
            pd.to_numeric(
                group.get("quality_first_premium_over_p1_q"),
                errors="coerce",
            ).sum()
        ),
        "support_distinct_iso_q_pair_count": int(
            pd.to_numeric(
                group.get("support_distinct_iso_q_pair_count"),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
    }


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
    contract_rows: pd.DataFrame,
    member_rows: pd.DataFrame,
    lookalike_rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Accurate Basin Portfolio Contract",
        "",
        "This artifact defines the accurate-mode output contract: choose the best p5 endpoint, then attach the basin portfolio evidence needed to explain quality premium and near-QF structural alternatives.",
        "",
    ]
    if contract_rows.empty:
        lines.append("- No accurate contract rows were available.")
    else:
        total = len(contract_rows)
        winner_only = int((contract_rows["accurate_mode_output_contract"] == "best_endpoint_only").sum())
        high_risk = int(
            (contract_rows["winner_only_risk"] == "high_quality_and_interpretation_risk").sum()
        )
        support_cases = int(
            (
                pd.to_numeric(
                    contract_rows["support_distinct_iso_q_pair_count"],
                    errors="coerce",
                ).fillna(0)
                > 0
            ).sum()
        )
        material_cases = int(contract_rows["quality_first_material_premium"].map(bool).sum())
        lines.extend(
            [
                "## Headline",
                "",
                f"- contract version: {CONTRACT_VERSION}",
                f"- cases: {total}",
                f"- winner-only sufficient cases: {winner_only}/{total}",
                f"- material quality-premium cases: {material_cases}/{total}",
                f"- support-distinct portfolio cases: {support_cases}/{total}",
                f"- high quality+interpretation risk cases: {high_risk}/{total}",
                "",
                "## Contract Summary",
                "",
            ]
        )
        display_summary = summary[
            [
                column
                for column in [
                    "group",
                    "case_count",
                    "best_endpoint_only_count",
                    "best_plus_support_distinct_portfolio_count",
                    "best_plus_quality_premium_evidence_count",
                    "best_plus_near_qf_alternatives_count",
                    "best_plus_near_qf_support_distinct_portfolio_count",
                    "high_quality_and_interpretation_risk_count",
                    "quality_first_premium_q_sum",
                    "support_distinct_iso_q_pair_count",
                ]
                if column in summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Highest Obligation Cases", ""])
        display_contract = contract_rows[
            [
                column
                for column in [
                    "field",
                    "method",
                    "accurate_mode_output_contract",
                    "winner_only_risk",
                    "quality_first_candidate_index",
                    "quality_first_p1_rank",
                    "quality_first_premium_over_p1_q",
                    "near_qf_alternative_count",
                    "support_distinct_iso_q_pair_count",
                    "output_obligations",
                ]
                if column in contract_rows.columns
            ]
        ]
        lines.extend(
            _markdown_table(
                display_contract.sort_values(
                    [
                        "support_distinct_iso_q_pair_count",
                        "quality_first_premium_over_p1_q",
                    ],
                    ascending=[False, False],
                ).head(12)
            ).splitlines()
        )
        lines.extend(["", "## Portfolio Member Examples", ""])
        if member_rows.empty or "portfolio_role" not in member_rows.columns:
            lines.append("- No portfolio member rows were available.")
        else:
            member_cols = [
                column
                for column in [
                    "field",
                    "method",
                    "candidate_index",
                    "portfolio_role",
                    "p1_rank",
                    "p5_delta_q",
                    "q_gap_to_best",
                    "coarse_basin_id",
                    "distance_to_best_support",
                    "same_coarse_as_best",
                ]
                if column in member_rows.columns
            ]
            lines.extend(
                _markdown_table(
                    member_rows[
                        member_rows["portfolio_role"].str.contains(
                            "quality_first_best|support_distinct_lookalike",
                            na=False,
                        )
                    ].head(30)[member_cols]
                ).splitlines()
            )
        lines.extend(["", "## Lookalike Pair Count", ""])
        lines.append(f"- support-distinct iso-Q pair rows: {len(lookalike_rows)}")
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Accurate mode must return a structured portfolio, not only the winning membership.",
                "- Winner-only output is sufficient only when there is neither material p1 premium nor support-distinct near-QF evidence.",
                "- Support exactness is part of the output contract because truncated support sketches change how strongly the portfolio can be interpreted.",
                "- This contract should be the reference before designing a fast-mode API or production trigger.",
            ]
        )
    (output_dir / "dongdaemun_accurate_basin_portfolio_contract_report.md").write_text(
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [
        _read_csvs(input_dir.expanduser().resolve(), "candidate_level_rows.csv")
        for input_dir in args.input_dir
    ]
    frames = [frame for frame in frames if not frame.empty]
    candidates = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_rows = build_accurate_contract_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    member_rows = build_accurate_portfolio_member_rows(
        candidates,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    lookalike_rows = build_lookalike_pair_rows(
        candidates,
        support_distinct_tau=args.support_distinct_tau,
    )
    summary = build_contract_summary(contract_rows)
    contract_rows.to_csv(
        output_dir / "dongdaemun_accurate_contract_case_rows.csv",
        index=False,
    )
    member_rows.to_csv(
        output_dir / "dongdaemun_accurate_portfolio_member_rows.csv",
        index=False,
    )
    lookalike_rows.to_csv(
        output_dir / "dongdaemun_accurate_support_distinct_pairs.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "dongdaemun_accurate_contract_summary.csv",
        index=False,
    )
    write_report(output_dir, contract_rows, member_rows, lookalike_rows, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "contract_rows": int(len(contract_rows)),
            "member_rows": int(len(member_rows)),
            "lookalike_rows": int(len(lookalike_rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
