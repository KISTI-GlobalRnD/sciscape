#!/usr/bin/env python3
"""Explain why Dongdaemun should search better and similar-looking basins."""

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

def _near_best_support_stats(
    pairwise: pd.DataFrame,
    base: dict[str, Any],
    best_candidate: int,
    near_candidates: set[int],
) -> tuple[float, float]:
    if pairwise.empty or not near_candidates:
        return math.nan, math.nan
    group_pairwise = pairwise[_base_mask(pairwise, base)]
    distances: list[float] = []
    for _, row in group_pairwise.iterrows():
        left = int(row.get("left_candidate_index", -1))
        right = int(row.get("right_candidate_index", -1))
        if best_candidate not in {left, right}:
            continue
        other = right if left == best_candidate else left
        if other not in near_candidates:
            continue
        distance = _finite_float(row.get("coarse_support_distance"))
        if math.isfinite(distance):
            distances.append(distance)
    if not distances:
        return math.nan, math.nan
    return _finite_float(pd.Series(distances).mean()), max(distances)

def _need_label(
    *,
    material_premium: bool,
    premium: float,
    support_distinct_lookalikes: int,
    near_best_coarse_count: int,
) -> str:
    if material_premium and support_distinct_lookalikes > 0:
        return "search_better_and_review_similar"
    if material_premium:
        return "search_better_basin"
    if support_distinct_lookalikes > 0 or near_best_coarse_count > 1:
        return "review_similar_basins"
    if math.isfinite(premium) and premium > 0.0:
        return "low_margin_best_choice"
    return "p1_already_best_no_portfolio_pressure"

def build_basin_portfolio_rows(
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
        labeled = group[pd.to_numeric(group.get("p5_delta_q"), errors="coerce").notna()].copy()
        if labeled.empty:
            continue
        ranked = _ranked_by_p1(labeled)
        best_idx = ranked["_p5_delta_q"].idxmax()
        best = ranked.loc[best_idx]
        p1_choice = ranked.iloc[0]
        best_candidate = int(best.get("candidate_index", -1))
        p1_candidate = int(p1_choice.get("candidate_index", -1))
        best_rank = int(list(ranked.index).index(best_idx) + 1)
        best_delta = _finite_float(best.get("p5_delta_q"))
        p1_delta = _finite_float(p1_choice.get("p5_delta_q"))
        premium = (
            best_delta - p1_delta
            if math.isfinite(best_delta) and math.isfinite(p1_delta)
            else math.nan
        )
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
            for value in pd.to_numeric(
                near_best.get("candidate_index"),
                errors="coerce",
            )
            .dropna()
            .tolist()
        }
        near_best_alternatives = near_best_candidates - {best_candidate}
        near_best_coarse = {
            coarse_map[candidate]
            for candidate in near_best_candidates
            if candidate in coarse_map
        }
        support_mean, support_max = _near_best_support_stats(
            pairwise,
            base,
            best_candidate,
            near_best_alternatives,
        )
        if group_pairwise.empty:
            partition_distinct_iso = 0
            support_distinct_iso = 0
            max_lookalike_support = math.nan
            min_lookalike_q_gap = math.nan
        else:
            lookalikes = group_pairwise[
                group_pairwise.get("partition_distinct_iso_q_pair", False).map(bool)
            ].copy()
            support_distinct = lookalikes[
                pd.to_numeric(
                    lookalikes.get("coarse_support_distance"),
                    errors="coerce",
                )
                >= support_distinct_tau
            ]
            partition_distinct_iso = int(len(lookalikes))
            support_distinct_iso = int(len(support_distinct))
            max_lookalike_support = _finite_float(
                pd.to_numeric(
                    support_distinct.get("coarse_support_distance"),
                    errors="coerce",
                ).max()
            )
            min_lookalike_q_gap = _finite_float(
                pd.to_numeric(support_distinct.get("q_delta_abs"), errors="coerce").min()
            )
        material_premium = bool(math.isfinite(premium) and premium >= material_regret_q)
        field, method = _case_field_method(labeled.iloc[0])
        rows.append(
            {
                **base,
                "field": field,
                "method": method,
                "portfolio_need_label": _need_label(
                    material_premium=material_premium,
                    premium=premium,
                    support_distinct_lookalikes=support_distinct_iso,
                    near_best_coarse_count=len(near_best_coarse),
                ),
                "candidate_count": int(len(ranked)),
                "coarse_basin_count": int(len(group_coarse)),
                "support_sketch_exact": _support_sketch_exact(labeled),
                "p1_candidate_index": p1_candidate,
                "quality_first_candidate_index": best_candidate,
                "quality_first_p1_rank": best_rank,
                "p1_choice_p5_delta_q": p1_delta,
                "quality_first_p5_delta_q": best_delta,
                "better_basin_premium_q": premium,
                "better_basin_material_premium": material_premium,
                "near_best_delta_q": near_best_delta_q,
                "near_best_candidate_count": int(len(near_best)),
                "near_best_alternative_count": int(len(near_best_alternatives)),
                "near_best_coarse_basin_count": int(len(near_best_coarse)),
                "near_best_best_support_distance_mean": support_mean,
                "near_best_best_support_distance_max": support_max,
                "partition_distinct_iso_q_pair_count": partition_distinct_iso,
                "support_distinct_iso_q_pair_count": support_distinct_iso,
                "support_distinct_tau": support_distinct_tau,
                "max_lookalike_support_distance": max_lookalike_support,
                "min_lookalike_q_gap": min_lookalike_q_gap,
                "why_search_better_basin": (
                    "material QF premium over the p1 choice"
                    if material_premium
                    else (
                        "low-margin QF premium over p1"
                        if math.isfinite(premium) and premium > 0.0
                        else "p1 already reaches the current best endpoint"
                    )
                ),
                "why_review_similar_basins": (
                    "iso-Q pairs can be partition/support-distinct"
                    if support_distinct_iso > 0
                    else (
                        "near-best alternatives span multiple coarse basins"
                        if len(near_best_coarse) > 1
                        else "no strong similar-basin pressure in this slice"
                    )
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "better_basin_material_premium",
            "support_distinct_iso_q_pair_count",
            "better_basin_premium_q",
        ],
        ascending=[False, False, False],
        na_position="last",
    )

def build_lookalike_pair_rows(
    candidates: pd.DataFrame,
    *,
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
    if pairwise.empty:
        return pd.DataFrame()
    rows = pairwise[
        pairwise.get("partition_distinct_iso_q_pair", False).map(bool)
    ].copy()
    rows = rows[
        pd.to_numeric(rows.get("coarse_support_distance"), errors="coerce")
        >= support_distinct_tau
    ].copy()
    if rows.empty:
        return rows
    rows["field"] = rows.apply(lambda row: _case_field_method(row)[0], axis=1)
    rows["method"] = rows.apply(lambda row: _case_field_method(row)[1], axis=1)
    rows["lookalike_reason"] = "similar QF but partition/support-distinct"
    display_cols = [
        column
        for column in [
            "field",
            "method",
            "case",
            "seed",
            "candidate_budget",
            "left_candidate_index",
            "right_candidate_index",
            "q_delta_abs",
            "q_relative_ppm_abs",
            "left_p5_delta_q",
            "right_p5_delta_q",
            "sample_coassignment_distance",
            "coarse_support_distance",
            "changed_node_support_union",
            "lookalike_reason",
        ]
        if column in rows.columns
    ]
    return rows.sort_values(
        ["q_delta_abs", "coarse_support_distance"],
        ascending=[True, False],
    )[display_cols]

def build_portfolio_summary(portfolio_rows: pd.DataFrame) -> pd.DataFrame:
    if portfolio_rows.empty:
        return pd.DataFrame()
    rows = [_summarize_portfolio_group("all", portfolio_rows)]
    if "field" in portfolio_rows.columns:
        for field, group in portfolio_rows.groupby("field", dropna=False):
            rows.append(_summarize_portfolio_group(f"field={field}", group))
    if "method" in portfolio_rows.columns:
        for method, group in portfolio_rows.groupby("method", dropna=False):
            rows.append(_summarize_portfolio_group(f"method={method}", group))
    return pd.DataFrame(rows)

def _summarize_portfolio_group(name: str, group: pd.DataFrame) -> dict[str, Any]:
    premium = pd.to_numeric(group.get("better_basin_premium_q"), errors="coerce")
    lookalike_count = pd.to_numeric(
        group.get("support_distinct_iso_q_pair_count"),
        errors="coerce",
    ).fillna(0)
    labels = group["portfolio_need_label"].value_counts()
    return {
        "group": name,
        "case_count": int(len(group)),
        "search_better_and_review_similar_count": int(
            labels.get("search_better_and_review_similar", 0)
        ),
        "search_better_basin_count": int(labels.get("search_better_basin", 0)),
        "review_similar_basins_count": int(labels.get("review_similar_basins", 0)),
        "low_margin_best_choice_count": int(labels.get("low_margin_best_choice", 0)),
        "p1_already_best_no_portfolio_pressure_count": int(
            labels.get("p1_already_best_no_portfolio_pressure", 0)
        ),
        "material_better_basin_count": int(
            group.get("better_basin_material_premium", False).map(bool).sum()
        ),
        "better_basin_premium_q_sum": _finite_float(premium.sum()),
        "better_basin_premium_q_mean": _finite_float(premium.mean()),
        "support_distinct_iso_q_pair_count": int(lookalike_count.sum()),
        "cases_with_support_distinct_iso_q_pairs": int((lookalike_count > 0).sum()),
        "near_best_candidate_count_mean": _finite_float(
            pd.to_numeric(
                group.get("near_best_candidate_count"),
                errors="coerce",
            ).mean()
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
    portfolio_rows: pd.DataFrame,
    lookalike_rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Basin Portfolio Evidence",
        "",
        "This diagnostic separates two claims: better basins matter because they recover QF premium, and similar-looking basins matter because iso-Q endpoints can be partition/support-distinct.",
        "",
    ]
    if portfolio_rows.empty:
        lines.append("- No basin portfolio rows were available.")
    else:
        total = len(portfolio_rows)
        material = int(portfolio_rows["better_basin_material_premium"].map(bool).sum())
        similar = int(
            (
                pd.to_numeric(
                    portfolio_rows["support_distinct_iso_q_pair_count"],
                    errors="coerce",
                ).fillna(0)
                > 0
            ).sum()
        )
        both = int(
            (
                portfolio_rows["portfolio_need_label"]
                == "search_better_and_review_similar"
            ).sum()
        )
        premium_sum = _finite_float(
            pd.to_numeric(
                portfolio_rows.get("better_basin_premium_q"),
                errors="coerce",
            ).sum()
        )
        lookalike_pairs = len(lookalike_rows)
        lines.extend(
            [
                "## Headline",
                "",
                f"- cases: {total}",
                f"- material better-basin cases: {material}/{total}",
                f"- cases with support-distinct iso-Q lookalikes: {similar}/{total}",
                f"- cases requiring both better search and similar-basin review: {both}/{total}",
                f"- total better-basin premium over p1: {premium_sum:.6g}",
                f"- support-distinct iso-Q lookalike pairs: {lookalike_pairs}",
                "",
                "## Portfolio Summary",
                "",
            ]
        )
        display_summary = summary[
            [
                column
                for column in [
                    "group",
                    "case_count",
                    "search_better_and_review_similar_count",
                    "search_better_basin_count",
                    "review_similar_basins_count",
                    "low_margin_best_choice_count",
                    "material_better_basin_count",
                    "better_basin_premium_q_sum",
                    "support_distinct_iso_q_pair_count",
                    "cases_with_support_distinct_iso_q_pairs",
                ]
                if column in summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Strongest Better-Basin Evidence", ""])
        better_cols = [
            column
            for column in [
                "field",
                "method",
                "portfolio_need_label",
                "p1_candidate_index",
                "quality_first_candidate_index",
                "quality_first_p1_rank",
                "better_basin_premium_q",
                "near_best_candidate_count",
                "support_distinct_iso_q_pair_count",
                "why_search_better_basin",
                "why_review_similar_basins",
            ]
            if column in portfolio_rows.columns
        ]
        lines.extend(
            _markdown_table(
                portfolio_rows.sort_values(
                    "better_basin_premium_q",
                    ascending=False,
                )
                .head(10)[better_cols]
            ).splitlines()
        )
        lines.extend(["", "## Closest Similar-Q Distinct-Basin Pairs", ""])
        if lookalike_rows.empty:
            lines.append("- No support-distinct iso-Q lookalike pairs were available.")
        else:
            lookalike_cols = [
                column
                for column in [
                    "field",
                    "method",
                    "left_candidate_index",
                    "right_candidate_index",
                    "q_delta_abs",
                    "coarse_support_distance",
                    "sample_coassignment_distance",
                    "changed_node_support_union",
                    "lookalike_reason",
                ]
                if column in lookalike_rows.columns
            ]
            lines.extend(_markdown_table(lookalike_rows.head(20)[lookalike_cols]).splitlines())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Better-basin search is justified when the quality-first endpoint has material QF premium over the p1 endpoint.",
                "- Similar-basin review is justified when small-QF-gap pairs are support/partition-distinct; those pairs can lead to different downstream interpretations even when QF is almost tied.",
                "- Fast mode should not erase similar-basin evidence; it can hide both delayed winners and near-tie structural alternatives.",
                "- The portfolio view therefore treats basin diversity as an object of measurement, not only as noise around a single winner.",
            ]
        )
    (output_dir / "dongdaemun_basin_portfolio_evidence_report.md").write_text(
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
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--material-relative-ppm", type=float, default=10.0)
    parser.add_argument("--coarse-endpoint-tau", type=float, default=0.02)
    parser.add_argument("--coarse-support-tau", type=float, default=0.5)
    parser.add_argument("--iso-q-delta", type=float, default=10.0)
    parser.add_argument("--iso-q-relative-ppm", type=float, default=10.0)
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
    kwargs = {
        "material_regret_q": args.material_regret_q,
        "near_best_delta_q": args.near_best_delta_q,
        "support_distinct_tau": args.support_distinct_tau,
        "material_delta_q": args.material_delta_q,
        "material_relative_ppm": args.material_relative_ppm,
        "coarse_endpoint_tau": args.coarse_endpoint_tau,
        "coarse_support_tau": args.coarse_support_tau,
        "iso_q_delta": args.iso_q_delta,
        "iso_q_relative_ppm": args.iso_q_relative_ppm,
    }
    portfolio_rows = build_basin_portfolio_rows(candidates, **kwargs)
    lookalike_rows = build_lookalike_pair_rows(
        candidates,
        support_distinct_tau=args.support_distinct_tau,
        material_delta_q=args.material_delta_q,
        material_relative_ppm=args.material_relative_ppm,
        coarse_endpoint_tau=args.coarse_endpoint_tau,
        coarse_support_tau=args.coarse_support_tau,
        iso_q_delta=args.iso_q_delta,
        iso_q_relative_ppm=args.iso_q_relative_ppm,
    )
    summary = build_portfolio_summary(portfolio_rows)
    portfolio_rows.to_csv(output_dir / "dongdaemun_basin_portfolio_case_rows.csv", index=False)
    lookalike_rows.to_csv(output_dir / "dongdaemun_basin_portfolio_lookalike_pairs.csv", index=False)
    summary.to_csv(output_dir / "dongdaemun_basin_portfolio_summary.csv", index=False)
    write_report(output_dir, portfolio_rows, lookalike_rows, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "portfolio_rows": int(len(portfolio_rows)),
            "lookalike_rows": int(len(lookalike_rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
