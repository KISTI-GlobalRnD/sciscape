#!/usr/bin/env python3
"""Evaluate pre-p5 attainable fast selectors against contract tiers."""

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

SELECTOR_SPECS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("p1_top1", "prefix", {"top_k": 1}),
    ("p1_top3", "prefix", {"top_k": 3}),
    ("p1_top5", "prefix", {"top_k": 5}),
    ("p1_top10", "prefix", {"top_k": 10}),
    ("rank_stratified_p1", "stratified", {"positions": (1, 3, 5, 10)}),
    ("cheap_metric_top1_union", "cheap_union", {"metric_top_n": 1}),
    ("cheap_metric_top2_union", "cheap_union", {"metric_top_n": 2}),
    ("p1_top3_plus_metric_top1", "hybrid", {"top_k": 3, "metric_top_n": 1}),
    ("p1_top5_plus_metric_top1", "hybrid", {"top_k": 5, "metric_top_n": 1}),
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

def _parse_candidate_set(value: Any) -> set[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    out: set[int] = set()
    for part in str(value).split(";"):
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out

def _rank_by_metric(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    values = pd.to_numeric(group.get(metric), errors="coerce")
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

def _metric_top_candidates(
    group: pd.DataFrame,
    *,
    metric_top_n: int,
) -> set[int]:
    selected: set[int] = set()
    for metric in CHEAP_METRIC_COLUMNS:
        if metric not in group.columns:
            continue
        ranked = _rank_by_metric(group, metric)
        if ranked.empty:
            continue
        selected |= _candidate_set(ranked.head(metric_top_n))
    return selected

def _select_candidates(
    ranked_by_p1: pd.DataFrame,
    selector_family: str,
    selector_params: dict[str, Any],
) -> set[int]:
    if selector_family == "prefix":
        top_k = int(selector_params["top_k"])
        return _candidate_set(ranked_by_p1.head(min(top_k, len(ranked_by_p1))))
    if selector_family == "stratified":
        selected_rows = []
        for position in selector_params["positions"]:
            index = int(position) - 1
            if 0 <= index < len(ranked_by_p1):
                selected_rows.append(ranked_by_p1.iloc[index])
        return {
            int(row.get("candidate_index", -1))
            for row in selected_rows
            if int(row.get("candidate_index", -1)) >= 0
        }
    if selector_family == "cheap_union":
        return _metric_top_candidates(
            ranked_by_p1,
            metric_top_n=int(selector_params["metric_top_n"]),
        )
    if selector_family == "hybrid":
        top_k = int(selector_params["top_k"])
        selected = _candidate_set(ranked_by_p1.head(min(top_k, len(ranked_by_p1))))
        selected |= _metric_top_candidates(
            ranked_by_p1,
            metric_top_n=int(selector_params["metric_top_n"]),
        )
        return selected
    raise ValueError(f"Unknown selector family: {selector_family}")

def _subset_elapsed(
    ranked: pd.DataFrame,
    elapsed: pd.Series,
    selected_candidates: set[int],
) -> float:
    if not selected_candidates:
        return 0.0
    candidates = pd.to_numeric(ranked.get("candidate_index"), errors="coerce")
    indices = ranked[candidates.isin(selected_candidates)].index
    return _finite_float(elapsed.loc[indices].sum())

def _candidate_delta_map(ranked: pd.DataFrame) -> dict[int, float]:
    return {
        int(row.get("candidate_index", -1)): _finite_float(row.get("p5_delta_q"))
        for _, row in ranked.iterrows()
    }

def _selected_best(
    ranked: pd.DataFrame,
    selected_candidates: set[int],
) -> tuple[int, float]:
    if not selected_candidates:
        return -1, math.nan
    selected = ranked[
        pd.to_numeric(ranked.get("candidate_index"), errors="coerce").isin(
            selected_candidates
        )
    ].copy()
    if selected.empty:
        return -1, math.nan
    selected["_selector_p5"] = pd.to_numeric(selected.get("p5_delta_q"), errors="coerce")
    best_idx = selected["_selector_p5"].idxmax()
    row = selected.loc[best_idx]
    return int(row.get("candidate_index", -1)), _finite_float(row.get("p5_delta_q"))

def build_attainable_selector_rows(
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
        elapsed = (
            pd.to_numeric(ranked.get("p5_elapsed_ms"), errors="coerce").fillna(0.0)
            if "p5_elapsed_ms" in ranked.columns
            else pd.Series([0.0] * len(ranked), index=ranked.index, dtype=float)
        )
        accurate_elapsed = _finite_float(elapsed.sum())
        delta_by_candidate = _candidate_delta_map(ranked)
        best_row = case_tiers.iloc[0]
        best_candidate = int(best_row.get("quality_first_candidate_index", -1))
        best_delta = delta_by_candidate.get(best_candidate, math.nan)
        field, method = _case_field_method(ranked.iloc[0])
        for selector_name, selector_family, selector_params in SELECTOR_SPECS:
            selected_candidates = _select_candidates(
                ranked,
                selector_family,
                selector_params,
            )
            selected_count = len(selected_candidates)
            selected_elapsed = _subset_elapsed(ranked, elapsed, selected_candidates)
            selected_best_candidate, selected_best_delta = _selected_best(
                ranked,
                selected_candidates,
            )
            quality_regret = (
                best_delta - selected_best_delta
                if math.isfinite(best_delta) and math.isfinite(selected_best_delta)
                else math.nan
            )
            for _, tier in case_tiers.iterrows():
                required = _parse_candidate_set(tier.get("required_candidate_indices"))
                missing = required - selected_candidates
                tier_covered = len(missing) == 0
                endpoint_covered = best_candidate in selected_candidates
                rows.append(
                    {
                        **base,
                        "field": field,
                        "method": method,
                        "selector_name": selector_name,
                        "selector_family": selector_family,
                        "contract_tier": tier.get("contract_tier"),
                        "candidate_count": int(len(ranked)),
                        "selected_candidate_indices": ";".join(
                            str(candidate) for candidate in sorted(selected_candidates)
                        ),
                        "selected_candidate_count": selected_count,
                        "selected_candidate_fraction": (
                            selected_count / len(ranked) if len(ranked) > 0 else math.nan
                        ),
                        "required_candidate_indices": tier.get(
                            "required_candidate_indices"
                        ),
                        "oracle_required_candidate_count": int(
                            tier.get("oracle_required_candidate_count", 0)
                        ),
                        "required_candidate_covered_count": (
                            len(required) - len(missing)
                        ),
                        "required_candidate_missed_count": len(missing),
                        "required_candidate_coverage_fraction": (
                            (len(required) - len(missing)) / len(required)
                            if required
                            else 1.0
                        ),
                        "tier_covered": tier_covered,
                        "endpoint_obligation_covered": endpoint_covered,
                        "selected_best_candidate_index": selected_best_candidate,
                        "quality_first_candidate_index": best_candidate,
                        "quality_first_p1_rank": int(
                            tier.get("quality_first_p1_rank", -1)
                        ),
                        "selected_best_p5_delta_q": selected_best_delta,
                        "quality_first_p5_delta_q": best_delta,
                        "quality_regret_q": quality_regret,
                        "material_regret": bool(
                            math.isfinite(quality_regret)
                            and quality_regret >= material_regret_q
                        ),
                        "estimated_p5_elapsed_ms": selected_elapsed,
                        "accurate_full_budget_p5_elapsed_ms": accurate_elapsed,
                        "elapsed_ratio_vs_accurate": (
                            selected_elapsed / accurate_elapsed
                            if accurate_elapsed > 0.0
                            else math.nan
                        ),
                        "oracle_required_candidate_count": int(
                            tier.get("oracle_required_candidate_count", 0)
                        ),
                        "oracle_required_elapsed_ratio_vs_accurate": _finite_float(
                            tier.get("oracle_elapsed_ratio_vs_accurate")
                        ),
                        "selector_over_oracle_candidate_overhead": (
                            selected_count
                            - int(tier.get("oracle_required_candidate_count", 0))
                        ),
                        "selector_over_oracle_elapsed_ratio_gap": (
                            (
                                selected_elapsed / accurate_elapsed
                                if accurate_elapsed > 0.0
                                else math.nan
                            )
                            - _finite_float(tier.get("oracle_elapsed_ratio_vs_accurate"))
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "selector_name", "contract_tier"],
        na_position="last",
    )

def build_attainable_selector_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (selector_name, tier), group in rows.groupby(
        ["selector_name", "contract_tier"],
        dropna=False,
    ):
        summary_rows.append(_summarize_selector_tier(str(selector_name), str(tier), group))
    order = {name: pos for pos, (name, _, _) in enumerate(SELECTOR_SPECS)}
    tier_order = {name: pos for pos, name in enumerate(TIER_ORDER)}
    return pd.DataFrame(summary_rows).sort_values(
        ["selector_name", "contract_tier"],
        key=lambda values: values.map(
            lambda value: order.get(value, tier_order.get(value, 99))
        ),
    )

def _summarize_selector_tier(
    selector_name: str,
    tier: str,
    group: pd.DataFrame,
) -> dict[str, Any]:
    selected_count = pd.to_numeric(group.get("selected_candidate_count"), errors="coerce")
    coverage = pd.to_numeric(
        group.get("required_candidate_coverage_fraction"),
        errors="coerce",
    )
    regret = pd.to_numeric(group.get("quality_regret_q"), errors="coerce")
    elapsed_ratio = pd.to_numeric(group.get("elapsed_ratio_vs_accurate"), errors="coerce")
    overhead = pd.to_numeric(
        group.get("selector_over_oracle_candidate_overhead"),
        errors="coerce",
    )
    elapsed_gap = pd.to_numeric(
        group.get("selector_over_oracle_elapsed_ratio_gap"),
        errors="coerce",
    )
    return {
        "selector_name": selector_name,
        "selector_family": str(group["selector_family"].iloc[0]),
        "contract_tier": tier,
        "case_count": int(len(group)),
        "tier_covered_count": int(group["tier_covered"].map(bool).sum()),
        "endpoint_covered_count": int(
            group["endpoint_obligation_covered"].map(bool).sum()
        ),
        "material_regret_count": int(group["material_regret"].map(bool).sum()),
        "selected_candidate_count_mean": _finite_float(selected_count.mean()),
        "required_candidate_coverage_fraction_mean": _finite_float(coverage.mean()),
        "quality_regret_q_sum": _finite_float(regret.sum()),
        "quality_regret_q_max": _finite_float(regret.max()),
        "elapsed_ratio_vs_accurate_mean": _finite_float(elapsed_ratio.mean()),
        "selector_over_oracle_candidate_overhead_sum": int(overhead.fillna(0).sum()),
        "selector_over_oracle_elapsed_ratio_gap_mean": _finite_float(
            elapsed_gap.mean()
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
    rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Attainable Fast Selector",
        "",
        "This diagnostic evaluates cheap/pre-p5 selector families against hard/core/diagnostic contract tiers. Selector choices use no p5 labels; p5 is used only for oracle evaluation.",
        "",
    ]
    if rows.empty:
        lines.append("- No attainable selector rows were available.")
    else:
        lines.extend(["## Headline", ""])
        headline = summary[summary["contract_tier"].isin(["hard", "core"])].copy()
        headline = headline.sort_values(
            [
                "contract_tier",
                "tier_covered_count",
                "elapsed_ratio_vs_accurate_mean",
            ],
            ascending=[True, False, True],
        )
        display_headline = headline[
            [
                column
                for column in [
                    "selector_name",
                    "contract_tier",
                    "tier_covered_count",
                    "case_count",
                    "material_regret_count",
                    "selected_candidate_count_mean",
                    "elapsed_ratio_vs_accurate_mean",
                    "selector_over_oracle_elapsed_ratio_gap_mean",
                ]
                if column in headline.columns
            ]
        ].head(12)
        lines.extend(_markdown_table(display_headline).splitlines())
        lines.extend(["", "## Selector Summary", ""])
        display_summary = summary[
            [
                column
                for column in [
                    "selector_name",
                    "contract_tier",
                    "tier_covered_count",
                    "case_count",
                    "endpoint_covered_count",
                    "material_regret_count",
                    "selected_candidate_count_mean",
                    "required_candidate_coverage_fraction_mean",
                    "quality_regret_q_sum",
                    "elapsed_ratio_vs_accurate_mean",
                    "selector_over_oracle_candidate_overhead_sum",
                ]
                if column in summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Largest Core Misses", ""])
        core_misses = rows[
            (rows["contract_tier"] == "core") & (~rows["tier_covered"].map(bool))
        ].copy()
        if core_misses.empty:
            lines.append("- No core misses were available.")
        else:
            core_misses["_miss_score"] = (
                pd.to_numeric(
                    core_misses.get("required_candidate_missed_count"),
                    errors="coerce",
                ).fillna(0)
                + pd.to_numeric(core_misses.get("quality_regret_q"), errors="coerce")
                .fillna(0)
                .clip(lower=0)
            )
            display_cols = [
                column
                for column in [
                    "field",
                    "method",
                    "selector_name",
                    "required_candidate_indices",
                    "selected_candidate_indices",
                    "required_candidate_missed_count",
                    "quality_first_p1_rank",
                    "quality_regret_q",
                    "elapsed_ratio_vs_accurate",
                ]
                if column in core_misses.columns
            ]
            lines.extend(
                _markdown_table(
                    core_misses.sort_values("_miss_score", ascending=False)
                    .head(16)[display_cols]
                ).splitlines()
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- A selector that improves hard/core coverage without approaching full diagnostic cost is a plausible fast-mode mechanism candidate.",
                "- Cheap-metric union selectors are diagnostic prototypes, not production policies; their value is whether non-p1 signals can recover oracle-required candidates.",
                "- Diagnostic-tier misses are not automatically failures for fast mode if the result is labeled approximate and the final claim is reserved for accurate mode.",
            ]
        )
    (output_dir / "dongdaemun_attainable_fast_selector_report.md").write_text(
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
    rows = build_attainable_selector_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    summary = build_attainable_selector_summary(rows)
    rows.to_csv(
        output_dir / "dongdaemun_attainable_fast_selector_case_rows.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "dongdaemun_attainable_fast_selector_summary.csv",
        index=False,
    )
    write_report(output_dir, rows, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "selector_rows": int(len(rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
