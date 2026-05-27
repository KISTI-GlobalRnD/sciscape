#!/usr/bin/env python3
"""Measure how fast Dongdaemun modes cover the accurate basin contract."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_accurate_basin_portfolio_contract import (
    CONTRACT_VERSION,
    build_accurate_contract_rows,
    build_accurate_portfolio_member_rows,
)
from analyze_leiden_basin_portfolio_evidence import build_lookalike_pair_rows
from analyze_leiden_multibasin_decision_rules import _case_field_method
from analyze_leiden_multibasin_signatures import (
    _finite_float,
    _group_columns,
    _read_csvs,
    _signature_frame,
)
from analyze_leiden_quality_first_choice import _ranked_by_p1


FAST_MODE_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("fast_p1", "fast", 1),
    ("fast_top3", "fast", 3),
    ("fast_top5", "fast", 5),
    ("fast_top10", "fast", 10),
    ("accurate_full_budget", "accurate", None),
)


def _candidate_elapsed_ms(frame: pd.DataFrame) -> pd.Series:
    if "p5_elapsed_ms" in frame.columns:
        return pd.to_numeric(frame["p5_elapsed_ms"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)


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


def _role_candidate_set(member_rows: pd.DataFrame, role: str) -> set[int]:
    if member_rows.empty or "portfolio_role" not in member_rows.columns:
        return set()
    return _candidate_set(
        member_rows[member_rows["portfolio_role"].str.contains(role, na=False)]
    )


def _covered_pair_count(pairs: pd.DataFrame, evaluated_candidates: set[int]) -> int:
    if pairs.empty:
        return 0
    count = 0
    for _, row in pairs.iterrows():
        left = int(row.get("left_candidate_index", -1))
        right = int(row.get("right_candidate_index", -1))
        if left in evaluated_candidates and right in evaluated_candidates:
            count += 1
    return count


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def _mode_label(top_k: int | None, candidate_count: int) -> str:
    if top_k is None:
        return "full"
    return f"top{min(top_k, candidate_count)}"


def _coverage_class(
    *,
    contract_fully_covered: bool,
    endpoint_covered: bool,
    near_missed: int,
    support_pair_missed: int,
) -> str:
    if contract_fully_covered:
        return "contract_covered"
    portfolio_missed = near_missed > 0 or support_pair_missed > 0
    if not endpoint_covered and portfolio_missed:
        return "quality_and_portfolio_missed"
    if not endpoint_covered:
        return "quality_missed"
    if portfolio_missed:
        return "portfolio_missed"
    return "metadata_or_threshold_missed"


def _operator_interpretation(
    *,
    contract_fully_covered: bool,
    endpoint_covered: bool,
    near_missed: int,
    support_pair_missed: int,
) -> str:
    if contract_fully_covered:
        return "fast prefix matches the accurate contract for this case"
    if not endpoint_covered:
        return "fast prefix cannot support a final quality claim"
    if near_missed > 0 or support_pair_missed > 0:
        return "fast prefix can name the endpoint but not the full basin evidence"
    return "fast prefix needs contract metadata review"


def build_fast_contract_coverage_rows(
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
    pair_rows = build_lookalike_pair_rows(
        candidates,
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
        group_contract = contract_rows[_base_mask(contract_rows, base)]
        if group_contract.empty:
            continue
        contract = group_contract.iloc[0]
        group_members = member_rows[_base_mask(member_rows, base)]
        group_pairs = pair_rows[_base_mask(pair_rows, base)]
        elapsed = _candidate_elapsed_ms(ranked)
        accurate_elapsed = _finite_float(elapsed.sum())
        best_idx = ranked["_p5_delta_q"].idxmax()
        best = ranked.loc[best_idx]
        best_candidate = int(best.get("candidate_index", -1))
        best_delta = _finite_float(best.get("p5_delta_q"))
        best_rank = int(list(ranked.index).index(best_idx) + 1)
        p1_candidate = int(ranked.iloc[0].get("candidate_index", -1))
        material_required = bool(contract.get("quality_first_material_premium", False))
        near_candidates = _role_candidate_set(group_members, "near_qf_alternative")
        support_candidates = _role_candidate_set(
            group_members,
            "support_distinct_lookalike",
        )
        pair_count = int(len(group_pairs))
        field, method = _case_field_method(ranked.iloc[0])
        for mode_name, mode_family, top_k in FAST_MODE_SPECS:
            effective_top_k = len(ranked) if top_k is None else min(top_k, len(ranked))
            evaluated = ranked.head(effective_top_k)
            evaluated_candidates = _candidate_set(evaluated)
            selected_idx = evaluated["_p5_delta_q"].idxmax()
            selected = evaluated.loc[selected_idx]
            selected_candidate = int(selected.get("candidate_index", -1))
            selected_delta = _finite_float(selected.get("p5_delta_q"))
            regret = (
                best_delta - selected_delta
                if math.isfinite(best_delta) and math.isfinite(selected_delta)
                else math.nan
            )
            endpoint_covered = bool(selected_candidate == best_candidate)
            quality_premium_covered = bool((not material_required) or endpoint_covered)
            near_covered = len(near_candidates & evaluated_candidates)
            near_missed = len(near_candidates - evaluated_candidates)
            support_candidate_covered = len(support_candidates & evaluated_candidates)
            support_candidate_missed = len(support_candidates - evaluated_candidates)
            pair_covered = _covered_pair_count(group_pairs, evaluated_candidates)
            pair_missed = pair_count - pair_covered
            contract_fully_covered = bool(
                endpoint_covered
                and quality_premium_covered
                and near_missed == 0
                and pair_missed == 0
            )
            selected_elapsed = _finite_float(elapsed.loc[evaluated.index].sum())
            rows.append(
                {
                    **base,
                    "field": field,
                    "method": method,
                    "contract_version": CONTRACT_VERSION,
                    "mode_name": mode_name,
                    "mode_family": mode_family,
                    "mode_prefix": _mode_label(top_k, len(ranked)),
                    "top_k": effective_top_k,
                    "candidate_count": int(len(ranked)),
                    "p5_evaluated": int(len(evaluated)),
                    "evaluated_candidate_indices": ";".join(
                        str(candidate) for candidate in sorted(evaluated_candidates)
                    ),
                    "accurate_mode_output_contract": contract.get(
                        "accurate_mode_output_contract"
                    ),
                    "winner_only_risk": contract.get("winner_only_risk"),
                    "output_obligations": contract.get("output_obligations"),
                    "p1_candidate_index": p1_candidate,
                    "selected_candidate_index": selected_candidate,
                    "quality_first_candidate_index": best_candidate,
                    "quality_first_p1_rank": best_rank,
                    "endpoint_obligation_covered": endpoint_covered,
                    "selected_p5_delta_q": selected_delta,
                    "quality_first_p5_delta_q": best_delta,
                    "quality_regret_q": regret,
                    "material_regret": bool(
                        math.isfinite(regret) and regret >= material_regret_q
                    ),
                    "material_quality_premium_required": material_required,
                    "material_quality_premium_covered": quality_premium_covered,
                    "quality_first_premium_over_p1_q": _finite_float(
                        contract.get("quality_first_premium_over_p1_q")
                    ),
                    "near_qf_candidate_count": int(len(near_candidates)),
                    "near_qf_candidate_covered_count": near_covered,
                    "near_qf_candidate_missed_count": near_missed,
                    "near_qf_candidate_coverage_fraction": _fraction(
                        near_covered,
                        len(near_candidates),
                    ),
                    "support_distinct_candidate_count": int(len(support_candidates)),
                    "support_distinct_candidate_covered_count": (
                        support_candidate_covered
                    ),
                    "support_distinct_candidate_missed_count": support_candidate_missed,
                    "support_distinct_candidate_coverage_fraction": _fraction(
                        support_candidate_covered,
                        len(support_candidates),
                    ),
                    "support_distinct_iso_q_pair_count": pair_count,
                    "support_distinct_iso_q_pair_covered_count": pair_covered,
                    "support_distinct_iso_q_pair_missed_count": pair_missed,
                    "support_distinct_iso_q_pair_coverage_fraction": _fraction(
                        pair_covered,
                        pair_count,
                    ),
                    "support_exactness_reported_for_evaluated_subset": True,
                    "contract_fully_covered": contract_fully_covered,
                    "coverage_class": _coverage_class(
                        contract_fully_covered=contract_fully_covered,
                        endpoint_covered=endpoint_covered,
                        near_missed=near_missed,
                        support_pair_missed=pair_missed,
                    ),
                    "estimated_p5_elapsed_ms": selected_elapsed,
                    "accurate_full_budget_p5_elapsed_ms": accurate_elapsed,
                    "elapsed_ratio_vs_accurate": (
                        selected_elapsed / accurate_elapsed
                        if accurate_elapsed > 0.0
                        else math.nan
                    ),
                    "speedup_vs_accurate": (
                        accurate_elapsed / selected_elapsed
                        if selected_elapsed > 0.0
                        else math.nan
                    ),
                    "operator_interpretation": _operator_interpretation(
                        contract_fully_covered=contract_fully_covered,
                        endpoint_covered=endpoint_covered,
                        near_missed=near_missed,
                        support_pair_missed=pair_missed,
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["field", "method", "mode_name"])


def build_fast_contract_summary(coverage_rows: pd.DataFrame) -> pd.DataFrame:
    if coverage_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for mode_name, group in coverage_rows.groupby("mode_name", dropna=False):
        rows.append(_summarize_mode(str(mode_name), group))
    order = {name: pos for pos, (name, _, _) in enumerate(FAST_MODE_SPECS)}
    return pd.DataFrame(rows).sort_values(
        "mode_name",
        key=lambda values: values.map(lambda value: order.get(value, len(order))),
    )


def _summarize_mode(name: str, group: pd.DataFrame) -> dict[str, Any]:
    regret = pd.to_numeric(group.get("quality_regret_q"), errors="coerce")
    support_missed = pd.to_numeric(
        group.get("support_distinct_iso_q_pair_missed_count"),
        errors="coerce",
    )
    near_missed = pd.to_numeric(
        group.get("near_qf_candidate_missed_count"),
        errors="coerce",
    )
    elapsed_ratio = pd.to_numeric(group.get("elapsed_ratio_vs_accurate"), errors="coerce")
    support_fraction = pd.to_numeric(
        group.get("support_distinct_iso_q_pair_coverage_fraction"),
        errors="coerce",
    )
    near_fraction = pd.to_numeric(
        group.get("near_qf_candidate_coverage_fraction"),
        errors="coerce",
    )
    return {
        "mode_name": name,
        "mode_family": str(group["mode_family"].iloc[0]),
        "case_count": int(len(group)),
        "endpoint_covered_count": int(group["endpoint_obligation_covered"].map(bool).sum()),
        "contract_fully_covered_count": int(group["contract_fully_covered"].map(bool).sum()),
        "material_regret_count": int(group["material_regret"].map(bool).sum()),
        "quality_regret_q_sum": _finite_float(regret.sum()),
        "quality_regret_q_max": _finite_float(regret.max()),
        "near_qf_candidate_missed_sum": int(near_missed.fillna(0).sum()),
        "support_distinct_iso_q_pair_missed_sum": int(support_missed.fillna(0).sum()),
        "near_qf_candidate_coverage_mean": _finite_float(near_fraction.mean()),
        "support_distinct_iso_q_pair_coverage_mean": _finite_float(
            support_fraction.mean()
        ),
        "elapsed_ratio_vs_accurate_mean": _finite_float(elapsed_ratio.mean()),
        "speedup_vs_accurate_harmonic_proxy": (
            1.0 / _finite_float(elapsed_ratio.mean())
            if _finite_float(elapsed_ratio.mean()) > 0.0
            else math.nan
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
    coverage_rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Fast Contract Coverage",
        "",
        "This diagnostic treats the accurate basin portfolio contract as the reference and measures which obligations each fast prefix can cover.",
        "",
    ]
    if coverage_rows.empty:
        lines.append("- No fast contract coverage rows were available.")
    else:
        lines.extend(["## Headline", ""])
        for mode_name in ["fast_p1", "fast_top3", "fast_top5", "fast_top10"]:
            mode = summary[summary["mode_name"] == mode_name]
            if mode.empty:
                continue
            row = mode.iloc[0]
            lines.append(
                f"- {mode_name}: full contract coverage "
                f"{int(row.get('contract_fully_covered_count'))}/"
                f"{int(row.get('case_count'))}; "
                f"missed support pairs "
                f"{int(row.get('support_distinct_iso_q_pair_missed_sum'))}; "
                f"mean elapsed ratio "
                f"{_finite_float(row.get('elapsed_ratio_vs_accurate_mean')):.6g}"
            )
        lines.extend(["", "## Mode Summary", ""])
        display_summary = summary[
            [
                column
                for column in [
                    "mode_name",
                    "case_count",
                    "endpoint_covered_count",
                    "contract_fully_covered_count",
                    "material_regret_count",
                    "quality_regret_q_sum",
                    "near_qf_candidate_missed_sum",
                    "support_distinct_iso_q_pair_missed_sum",
                    "near_qf_candidate_coverage_mean",
                    "support_distinct_iso_q_pair_coverage_mean",
                    "elapsed_ratio_vs_accurate_mean",
                ]
                if column in summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Largest Remaining Fast Misses", ""])
        fast_rows = coverage_rows[
            coverage_rows["mode_name"].isin(["fast_p1", "fast_top3", "fast_top5", "fast_top10"])
        ].copy()
        display_cols = [
            column
            for column in [
                "field",
                "method",
                "mode_name",
                "coverage_class",
                "quality_first_p1_rank",
                "quality_regret_q",
                "near_qf_candidate_missed_count",
                "support_distinct_iso_q_pair_missed_count",
                "elapsed_ratio_vs_accurate",
                "operator_interpretation",
            ]
            if column in fast_rows.columns
        ]
        fast_rows["_miss_score"] = (
            pd.to_numeric(
                fast_rows.get("support_distinct_iso_q_pair_missed_count"),
                errors="coerce",
            ).fillna(0)
            + pd.to_numeric(
                fast_rows.get("near_qf_candidate_missed_count"),
                errors="coerce",
            ).fillna(0)
            + pd.to_numeric(fast_rows.get("quality_regret_q"), errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )
        lines.extend(
            _markdown_table(
                fast_rows.sort_values("_miss_score", ascending=False)
                .head(16)[display_cols]
            ).splitlines()
        )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Fast mode should be evaluated against explicit output obligations, not only against QF uplift.",
                "- Endpoint hit is necessary for a final quality claim, but it is not sufficient when near-QF support-distinct pairs are part of the accurate output.",
                "- A prefix can be useful operationally even when it is not contract-equivalent; that should be reported as exploratory or approximate mode.",
                "- The accurate contract remains the reference for final claims until a fast trigger can prove obligation coverage without using p5-derived oracle labels.",
            ]
        )
    (output_dir / "dongdaemun_fast_contract_coverage_report.md").write_text(
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
    coverage_rows = build_fast_contract_coverage_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    summary = build_fast_contract_summary(coverage_rows)
    coverage_rows.to_csv(
        output_dir / "dongdaemun_fast_contract_coverage_case_rows.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "dongdaemun_fast_contract_coverage_summary.csv",
        index=False,
    )
    write_report(output_dir, coverage_rows, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "coverage_rows": int(len(coverage_rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
