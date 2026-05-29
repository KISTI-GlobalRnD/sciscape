#!/usr/bin/env python3
"""Evaluate reduced non-p1 cheap-signal selectors against contract tiers."""

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
    _candidate_delta_map,
    _candidate_set,
    _parse_candidate_set,
    _rank_by_metric,
    _selected_best,
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

REDUCED_SIGNAL_METRICS: tuple[str, ...] = (
    "group_to_target_weight",
    "group_weight",
    "group_split_delta_q",
    "priority",
    "incident_directed_edges",
)
WEIGHT_FLOW_METRICS: tuple[str, ...] = (
    "group_to_target_weight",
    "group_weight",
    "incident_directed_edges",
)
DELTA_PRIORITY_METRICS: tuple[str, ...] = (
    "group_to_target_weight",
    "group_split_delta_q",
    "priority",
)

SELECTOR_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "p1_top3", "scope": "reference", "kind": "prefix", "top_k": 3},
    {"name": "p1_top5", "scope": "reference", "kind": "prefix", "top_k": 5},
    {
        "name": "cheap_metric_top1_union",
        "scope": "reference",
        "kind": "metric_union",
        "metrics": tuple(CHEAP_METRIC_COLUMNS),
        "metric_top_n": 1,
    },
    {
        "name": "cheap_metric_top2_union",
        "scope": "reference",
        "kind": "metric_union",
        "metrics": tuple(CHEAP_METRIC_COLUMNS),
        "metric_top_n": 2,
    },
    {
        "name": "p1_top3_plus_metric_top1",
        "scope": "reference",
        "kind": "hybrid",
        "top_k": 3,
        "metrics": tuple(CHEAP_METRIC_COLUMNS),
        "metric_top_n": 1,
    },
    {
        "name": "reduced_nonp1_top1_union",
        "scope": "reduced",
        "kind": "metric_union",
        "metrics": REDUCED_SIGNAL_METRICS,
        "metric_top_n": 1,
    },
    {
        "name": "reduced_nonp1_top2_union",
        "scope": "reduced",
        "kind": "metric_union",
        "metrics": REDUCED_SIGNAL_METRICS,
        "metric_top_n": 2,
    },
    {
        "name": "p1_top3_plus_reduced_nonp1_top1",
        "scope": "reduced",
        "kind": "hybrid",
        "top_k": 3,
        "metrics": REDUCED_SIGNAL_METRICS,
        "metric_top_n": 1,
    },
    {
        "name": "p1_top5_plus_reduced_nonp1_top1",
        "scope": "reduced",
        "kind": "hybrid",
        "top_k": 5,
        "metrics": REDUCED_SIGNAL_METRICS,
        "metric_top_n": 1,
    },
    {
        "name": "p1_top3_plus_weight_flow_top1",
        "scope": "reduced",
        "kind": "hybrid",
        "top_k": 3,
        "metrics": WEIGHT_FLOW_METRICS,
        "metric_top_n": 1,
    },
    {
        "name": "p1_top3_plus_delta_priority_top1",
        "scope": "reduced",
        "kind": "hybrid",
        "top_k": 3,
        "metrics": DELTA_PRIORITY_METRICS,
        "metric_top_n": 1,
    },
)
REFERENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("p1_top3", "p1_top3_plus_reduced_nonp1_top1"),
    ("p1_top5", "p1_top5_plus_reduced_nonp1_top1"),
    ("cheap_metric_top1_union", "reduced_nonp1_top1_union"),
    ("cheap_metric_top2_union", "reduced_nonp1_top2_union"),
    ("p1_top3_plus_metric_top1", "p1_top3_plus_reduced_nonp1_top1"),
    ("p1_top3_plus_reduced_nonp1_top1", "p1_top3_plus_weight_flow_top1"),
    ("p1_top3_plus_reduced_nonp1_top1", "p1_top3_plus_delta_priority_top1"),
)

def _base_mask(frame: pd.DataFrame, base: dict[str, Any]) -> pd.Series:
    mask = pd.Series([True] * len(frame), index=frame.index)
    for column, value in base.items():
        if column in frame.columns:
            mask &= frame[column] == value
    return mask

def _join_candidates(candidates: set[int]) -> str:
    return ";".join(str(candidate) for candidate in sorted(candidates))

def _metric_top_candidates(
    group: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    metric_top_n: int,
) -> set[int]:
    selected: set[int] = set()
    for metric in metrics:
        if metric not in group.columns:
            continue
        ranked = _rank_by_metric(group, metric)
        if ranked.empty:
            continue
        selected |= _candidate_set(ranked.head(metric_top_n))
    return selected

def _select_candidates(ranked: pd.DataFrame, spec: dict[str, Any]) -> set[int]:
    kind = str(spec["kind"])
    if kind == "prefix":
        top_k = int(spec["top_k"])
        return _candidate_set(ranked.head(min(top_k, len(ranked))))
    if kind == "metric_union":
        return _metric_top_candidates(
            ranked,
            metrics=tuple(spec["metrics"]),
            metric_top_n=int(spec["metric_top_n"]),
        )
    if kind == "hybrid":
        top_k = int(spec["top_k"])
        selected = _candidate_set(ranked.head(min(top_k, len(ranked))))
        selected |= _metric_top_candidates(
            ranked,
            metrics=tuple(spec["metrics"]),
            metric_top_n=int(spec["metric_top_n"]),
        )
        return selected
    raise ValueError(f"Unknown selector kind: {kind}")

def _elapsed_series(ranked: pd.DataFrame) -> pd.Series:
    if "p5_elapsed_ms" in ranked.columns:
        return pd.to_numeric(ranked["p5_elapsed_ms"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(ranked), index=ranked.index, dtype=float)

def build_reduced_selector_rows(
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
        elapsed = _elapsed_series(ranked)
        accurate_elapsed = _finite_float(elapsed.sum())
        delta_by_candidate = _candidate_delta_map(ranked)
        best_row = case_tiers.iloc[0]
        best_candidate = int(best_row.get("quality_first_candidate_index", -1))
        best_delta = delta_by_candidate.get(best_candidate, math.nan)
        field, method = _case_field_method(ranked.iloc[0])
        for spec in SELECTOR_SPECS:
            selected = _select_candidates(ranked, spec)
            selected_count = len(selected)
            selected_elapsed = _subset_elapsed(ranked, elapsed, selected)
            selected_best_candidate, selected_best_delta = _selected_best(
                ranked,
                selected,
            )
            quality_regret = (
                best_delta - selected_best_delta
                if math.isfinite(best_delta) and math.isfinite(selected_best_delta)
                else math.nan
            )
            for _, tier in case_tiers.iterrows():
                required = _parse_candidate_set(tier.get("required_candidate_indices"))
                missing = required - selected
                rows.append(
                    {
                        **base,
                        "field": field,
                        "method": method,
                        "selector_name": spec["name"],
                        "selector_scope": spec["scope"],
                        "selector_kind": spec["kind"],
                        "selector_metrics": ";".join(spec.get("metrics", ())),
                        "contract_tier": tier.get("contract_tier"),
                        "candidate_count": int(len(ranked)),
                        "selected_candidate_indices": _join_candidates(selected),
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
                        "required_candidate_covered_count": len(required)
                        - len(missing),
                        "required_candidate_missed_count": len(missing),
                        "required_candidate_coverage_fraction": (
                            (len(required) - len(missing)) / len(required)
                            if required
                            else 1.0
                        ),
                        "tier_covered": len(missing) == 0,
                        "endpoint_obligation_covered": best_candidate in selected,
                        "selected_best_candidate_index": selected_best_candidate,
                        "quality_first_candidate_index": best_candidate,
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
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "selector_scope", "selector_name", "contract_tier"],
        na_position="last",
    )

def build_reduced_selector_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (selector_name, tier), group in rows.groupby(
        ["selector_name", "contract_tier"],
        dropna=False,
    ):
        selected_count = pd.to_numeric(
            group["selected_candidate_count"],
            errors="coerce",
        )
        coverage = pd.to_numeric(
            group["required_candidate_coverage_fraction"],
            errors="coerce",
        )
        regret = pd.to_numeric(group["quality_regret_q"], errors="coerce")
        elapsed_ratio = pd.to_numeric(
            group["elapsed_ratio_vs_accurate"],
            errors="coerce",
        )
        summary_rows.append(
            {
                "selector_name": str(selector_name),
                "selector_scope": str(group["selector_scope"].iloc[0]),
                "selector_kind": str(group["selector_kind"].iloc[0]),
                "selector_metrics": str(group["selector_metrics"].iloc[0]),
                "contract_tier": str(tier),
                "case_count": int(len(group)),
                "tier_covered_count": int(group["tier_covered"].map(bool).sum()),
                "endpoint_covered_count": int(
                    group["endpoint_obligation_covered"].map(bool).sum()
                ),
                "material_regret_count": int(
                    group["material_regret"].map(bool).sum()
                ),
                "selected_candidate_count_mean": _finite_float(
                    selected_count.mean()
                ),
                "required_candidate_coverage_fraction_mean": _finite_float(
                    coverage.mean()
                ),
                "quality_regret_q_sum": _finite_float(regret.sum()),
                "quality_regret_q_max": _finite_float(regret.max()),
                "elapsed_ratio_vs_accurate_mean": _finite_float(
                    elapsed_ratio.mean()
                ),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    order = {spec["name"]: pos for pos, spec in enumerate(SELECTOR_SPECS)}
    tier_order = {tier: pos for pos, tier in enumerate(TIER_ORDER)}
    return pd.DataFrame(summary_rows).sort_values(
        ["selector_name", "contract_tier"],
        key=lambda values: values.map(
            lambda value: order.get(value, tier_order.get(value, 99))
        ),
    )

def build_reference_comparison_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for reference, reduced in REFERENCE_PAIRS:
        for tier in TIER_ORDER:
            ref_rows = summary[
                (summary["selector_name"] == reference)
                & (summary["contract_tier"] == tier)
            ]
            red_rows = summary[
                (summary["selector_name"] == reduced)
                & (summary["contract_tier"] == tier)
            ]
            if ref_rows.empty or red_rows.empty:
                continue
            ref = ref_rows.iloc[0]
            red = red_rows.iloc[0]
            rows.append(
                {
                    "reference_selector": reference,
                    "reduced_selector": reduced,
                    "contract_tier": tier,
                    "case_count": int(red.get("case_count", 0)),
                    "tier_covered_count_delta": int(
                        red.get("tier_covered_count", 0)
                    )
                    - int(ref.get("tier_covered_count", 0)),
                    "material_regret_count_delta": int(
                        red.get("material_regret_count", 0)
                    )
                    - int(ref.get("material_regret_count", 0)),
                    "quality_regret_q_sum_delta": _finite_float(
                        red.get("quality_regret_q_sum")
                    )
                    - _finite_float(ref.get("quality_regret_q_sum")),
                    "selected_candidate_count_mean_delta": _finite_float(
                        red.get("selected_candidate_count_mean")
                    )
                    - _finite_float(ref.get("selected_candidate_count_mean")),
                    "elapsed_ratio_vs_accurate_mean_delta": _finite_float(
                        red.get("elapsed_ratio_vs_accurate_mean")
                    )
                    - _finite_float(ref.get("elapsed_ratio_vs_accurate_mean")),
                    "reference_tier_covered_count": int(
                        ref.get("tier_covered_count", 0)
                    ),
                    "reduced_tier_covered_count": int(
                        red.get("tier_covered_count", 0)
                    ),
                    "reference_material_regret_count": int(
                        ref.get("material_regret_count", 0)
                    ),
                    "reduced_material_regret_count": int(
                        red.get("material_regret_count", 0)
                    ),
                    "reference_elapsed_ratio_vs_accurate_mean": _finite_float(
                        ref.get("elapsed_ratio_vs_accurate_mean")
                    ),
                    "reduced_elapsed_ratio_vs_accurate_mean": _finite_float(
                        red.get("elapsed_ratio_vs_accurate_mean")
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [
            "contract_tier",
            "tier_covered_count_delta",
            "elapsed_ratio_vs_accurate_mean_delta",
            "reduced_selector",
        ],
        ascending=[True, False, True, True],
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
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Reduced Signal Selector",
        "",
        "This diagnostic tests whether attribution-derived non-p1 cheap signals preserve hard/core recovery at lower cost than broad cheap-metric unions. Candidate selection uses only pre-p5 signals; p5 labels are used only for contract evaluation.",
        "",
    ]
    if rows.empty:
        lines.append("- No reduced selector rows were available.")
    else:
        lines.extend(["## Reduced Selector Summary", ""])
        display_summary = summary[
            summary["selector_scope"].isin(["reduced"])
            & summary["contract_tier"].isin(["hard", "core"])
        ].copy()
        display_summary = display_summary[
            [
                column
                for column in [
                    "selector_name",
                    "contract_tier",
                    "tier_covered_count",
                    "case_count",
                    "material_regret_count",
                    "selected_candidate_count_mean",
                    "quality_regret_q_sum",
                    "elapsed_ratio_vs_accurate_mean",
                ]
                if column in display_summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Reference Comparison", ""])
        display_comparison = comparison[
            comparison["contract_tier"].isin(["hard", "core"])
        ].copy()
        display_comparison = display_comparison[
            [
                column
                for column in [
                    "reference_selector",
                    "reduced_selector",
                    "contract_tier",
                    "tier_covered_count_delta",
                    "material_regret_count_delta",
                    "quality_regret_q_sum_delta",
                    "selected_candidate_count_mean_delta",
                    "elapsed_ratio_vs_accurate_mean_delta",
                    "reference_tier_covered_count",
                    "reduced_tier_covered_count",
                ]
                if column in display_comparison.columns
            ]
        ]
        lines.extend(_markdown_table(display_comparison).splitlines())
        lines.extend(["", "## Remaining Core Misses", ""])
        core_misses = rows[
            (rows["selector_scope"] == "reduced")
            & (rows["contract_tier"] == "core")
            & (~rows["tier_covered"].map(bool))
        ].copy()
        if core_misses.empty:
            lines.append("- No reduced-selector core misses were available.")
        else:
            core_misses["_miss_score"] = (
                pd.to_numeric(
                    core_misses["required_candidate_missed_count"],
                    errors="coerce",
                ).fillna(0.0)
                + pd.to_numeric(core_misses["quality_regret_q"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
            display_misses = core_misses.sort_values(
                "_miss_score",
                ascending=False,
            ).head(20)
            display_misses = display_misses[
                [
                    column
                    for column in [
                        "field",
                        "method",
                        "selector_name",
                        "required_candidate_indices",
                        "selected_candidate_indices",
                        "required_candidate_missed_count",
                        "quality_regret_q",
                        "elapsed_ratio_vs_accurate",
                    ]
                    if column in display_misses.columns
                ]
            ]
            lines.extend(_markdown_table(display_misses).splitlines())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- If a reduced selector keeps zero material regret while lowering elapsed ratio, it is a stronger fast-mode candidate than the broad union.",
                "- If top2 reduced signals mainly improve diagnostic/core inventory with small QF-regret reduction, that is evidence for accurate-mode explanation rather than production fast mode.",
                "- If p1 hybrids outperform pure reduced-signal unions, the mechanism is best described as p1 prefix plus targeted non-p1 recovery, not a standalone cheap metric selector.",
            ]
        )
    (output_dir / "dongdaemun_reduced_signal_selector_report.md").write_text(
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
    candidates = (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_reduced_selector_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    summary = build_reduced_selector_summary(rows)
    comparison = build_reference_comparison_rows(summary)
    rows.to_csv(
        output_dir / "dongdaemun_reduced_signal_selector_case_rows.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "dongdaemun_reduced_signal_selector_summary.csv",
        index=False,
    )
    comparison.to_csv(
        output_dir / "dongdaemun_reduced_signal_selector_reference_comparison.csv",
        index=False,
    )
    write_report(output_dir, rows, summary, comparison)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "selector_rows": int(len(rows)),
            "summary_rows": int(len(summary)),
            "comparison_rows": int(len(comparison)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
