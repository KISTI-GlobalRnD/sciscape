#!/usr/bin/env python3
"""Explain why Dongdaemun needs both fast and accurate Leiden modes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

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
from analyze_leiden_quality_first_choice import _ranked_by_p1


MODE_SPECS = (
    ("fast_p1", "fast", 1, "lowest-latency p1-ranked single candidate"),
    ("fast_top3", "fast", 3, "small guarded fast mode"),
    ("fast_top5", "fast", 5, "wider guarded fast mode"),
    ("accurate_full_budget", "accurate", None, "quality-first full candidate budget"),
)


def _candidate_elapsed_ms(frame: pd.DataFrame) -> pd.Series:
    if "p5_elapsed_ms" in frame.columns:
        return pd.to_numeric(frame["p5_elapsed_ms"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)


def _coarse_count_for_group(coarse: pd.DataFrame, base: dict[str, Any]) -> int:
    if coarse.empty:
        return 0
    return int(len(coarse[_base_mask(coarse, base)]))


def build_two_mode_rows(
    candidates: pd.DataFrame,
    *,
    material_regret_q: float = 10.0,
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
        elapsed = _candidate_elapsed_ms(ranked)
        accurate_elapsed = _finite_float(elapsed.sum())
        best_idx = ranked["_p5_delta_q"].idxmax()
        best = ranked.loc[best_idx]
        best_delta = _finite_float(best.get("p5_delta_q"))
        best_candidate = int(best.get("candidate_index", -1))
        best_rank = int(list(ranked.index).index(best_idx) + 1)
        field, method = _case_field_method(labeled.iloc[0])
        for mode_name, mode_family, top_k, mode_role in MODE_SPECS:
            effective_top_k = len(ranked) if top_k is None else min(top_k, len(ranked))
            selected = ranked.head(effective_top_k)
            selected_elapsed = _finite_float(elapsed.loc[selected.index].sum())
            selected_best_idx = selected["_p5_delta_q"].idxmax()
            selected_best = selected.loc[selected_best_idx]
            selected_delta = _finite_float(selected_best.get("p5_delta_q"))
            selected_candidate = int(selected_best.get("candidate_index", -1))
            regret = (
                best_delta - selected_delta
                if math.isfinite(best_delta) and math.isfinite(selected_delta)
                else math.nan
            )
            rows.append(
                {
                    **base,
                    "field": field,
                    "method": method,
                    "mode_name": mode_name,
                    "mode_family": mode_family,
                    "mode_role": mode_role,
                    "top_k": effective_top_k,
                    "candidate_count": int(len(ranked)),
                    "p5_evaluated": int(len(selected)),
                    "selected_candidate_index": selected_candidate,
                    "quality_first_candidate_index": best_candidate,
                    "quality_first_p1_rank": best_rank,
                    "quality_first_hit": bool(selected_candidate == best_candidate),
                    "selected_p5_delta_q": selected_delta,
                    "quality_first_p5_delta_q": best_delta,
                    "quality_regret_q": regret,
                    "material_regret": bool(
                        math.isfinite(regret) and regret >= material_regret_q
                    ),
                    "quality_regret_fraction": (
                        regret / abs(best_delta)
                        if math.isfinite(regret) and abs(best_delta) > 0.0
                        else math.nan
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
                    "coarse_basin_count": _coarse_count_for_group(coarse, base),
                    "support_sketch_exact": _support_sketch_exact(labeled),
                    "why_fast_mode_matters": (
                        "lower p5 evaluation cost for exploratory or interactive runs"
                    ),
                    "why_accurate_mode_matters": (
                        "prevents hidden quality regret when the best endpoint is delayed"
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["field", "method", "mode_name"], na_position="last")


def build_two_mode_summary(mode_rows: pd.DataFrame) -> pd.DataFrame:
    if mode_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for mode_name, group in mode_rows.groupby("mode_name", dropna=False):
        rows.append(_summarize_mode(str(mode_name), group))
    return pd.DataFrame(rows).sort_values("estimated_p5_elapsed_ms_sum")


def _summarize_mode(name: str, group: pd.DataFrame) -> dict[str, Any]:
    regret = pd.to_numeric(group.get("quality_regret_q"), errors="coerce")
    elapsed = pd.to_numeric(group.get("estimated_p5_elapsed_ms"), errors="coerce")
    elapsed_ratio = pd.to_numeric(group.get("elapsed_ratio_vs_accurate"), errors="coerce")
    return {
        "mode_name": name,
        "mode_family": str(group["mode_family"].iloc[0]),
        "case_count": int(len(group)),
        "quality_first_hit_count": int(group["quality_first_hit"].map(bool).sum()),
        "material_regret_count": int(group["material_regret"].map(bool).sum()),
        "quality_regret_q_sum": _finite_float(regret.sum()),
        "quality_regret_q_mean": _finite_float(regret.mean()),
        "quality_regret_q_max": _finite_float(regret.max()),
        "estimated_p5_elapsed_ms_sum": _finite_float(elapsed.sum()),
        "elapsed_ratio_vs_accurate_mean": _finite_float(elapsed_ratio.mean()),
        "speedup_vs_accurate_harmonic_proxy": (
            1.0 / _finite_float(elapsed_ratio.mean())
            if _finite_float(elapsed_ratio.mean()) > 0.0
            else math.nan
        ),
    }


def build_mode_need_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    accurate = summary[summary["mode_name"] == "accurate_full_budget"]
    fast = summary[summary["mode_name"] == "fast_p1"]
    top5 = summary[summary["mode_name"] == "fast_top5"]
    rows: list[dict[str, Any]] = []
    if not fast.empty and not accurate.empty:
        f = fast.iloc[0]
        a = accurate.iloc[0]
        rows.append(
            {
                "need": "fast_mode",
                "evidence": "accurate mode costs more p5 evaluation time",
                "supporting_metric": "fast_p1_elapsed_ratio_vs_accurate_mean",
                "value": _finite_float(f.get("elapsed_ratio_vs_accurate_mean")),
                "interpretation": (
                    "fast mode is useful for exploratory, budget-bound, or interactive passes"
                ),
            }
        )
        rows.append(
            {
                "need": "accurate_mode",
                "evidence": "fast mode misses delayed best endpoints",
                "supporting_metric": "fast_p1_quality_regret_q_sum",
                "value": _finite_float(f.get("quality_regret_q_sum")),
                "interpretation": (
                    "accurate mode is needed when the output must choose the best available endpoint"
                ),
            }
        )
        rows.append(
            {
                "need": "accurate_mode",
                "evidence": "fast mode has material regret cases",
                "supporting_metric": "fast_p1_material_regret_count",
                "value": _finite_float(f.get("material_regret_count")),
                "interpretation": (
                    "a single greedy candidate is not enough for final quality claims"
                ),
            }
        )
        rows.append(
            {
                "need": "mode_pair",
                "evidence": "accurate mode eliminates regret by definition within the candidate budget",
                "supporting_metric": "accurate_quality_regret_q_sum",
                "value": _finite_float(a.get("quality_regret_q_sum")),
                "interpretation": (
                    "the two-mode design separates exploratory speed from final selection quality"
                ),
            }
        )
    if not top5.empty:
        t = top5.iloc[0]
        rows.append(
            {
                "need": "accurate_mode",
                "evidence": "guarded fast mode can still miss material delayed-best cases",
                "supporting_metric": "fast_top5_material_regret_count",
                "value": _finite_float(t.get("material_regret_count")),
                "interpretation": (
                    "even a wider fast guard is not a substitute for the accurate mode"
                ),
            }
        )
    return pd.DataFrame(rows)


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
    mode_rows: pd.DataFrame,
    summary: pd.DataFrame,
    need_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Two-Mode Tradeoff Review",
        "",
        "This diagnostic separates two product/research modes: a faster mode for bounded exploration and an accurate mode for final quality-first selection.",
        "",
    ]
    if mode_rows.empty:
        lines.append("- No mode rows were available.")
    else:
        fast = summary[summary["mode_name"] == "fast_p1"].iloc[0]
        accurate = summary[summary["mode_name"] == "accurate_full_budget"].iloc[0]
        top5 = summary[summary["mode_name"] == "fast_top5"].iloc[0]
        lines.extend(
            [
                "## Headline",
                "",
                f"- fast_p1 mean elapsed ratio vs accurate: {_finite_float(fast.get('elapsed_ratio_vs_accurate_mean')):.6g}",
                f"- fast_p1 total quality regret: {_finite_float(fast.get('quality_regret_q_sum')):.6g}",
                f"- fast_p1 material regret cases: {int(fast.get('material_regret_count'))}/{int(fast.get('case_count'))}",
                f"- fast_top5 material regret cases: {int(top5.get('material_regret_count'))}/{int(top5.get('case_count'))}",
                f"- accurate_full_budget quality regret: {_finite_float(accurate.get('quality_regret_q_sum')):.6g}",
                "",
                "## Mode Summary",
                "",
            ]
        )
        lines.extend(_markdown_table(summary).splitlines())
        lines.extend(["", "## Why Both Modes Are Needed", ""])
        lines.extend(_markdown_table(need_rows).splitlines())
        lines.extend(["", "## Largest Fast-Mode Misses", ""])
        fast_rows = mode_rows[mode_rows["mode_name"] == "fast_p1"].copy()
        display_cols = [
            column
            for column in [
                "field",
                "method",
                "mode_name",
                "selected_candidate_index",
                "quality_first_candidate_index",
                "quality_first_p1_rank",
                "quality_regret_q",
                "material_regret",
                "elapsed_ratio_vs_accurate",
                "speedup_vs_accurate",
                "coarse_basin_count",
            ]
            if column in fast_rows.columns
        ]
        lines.extend(
            _markdown_table(
                fast_rows.sort_values("quality_regret_q", ascending=False)
                .head(10)[display_cols]
            ).splitlines()
        )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Fast mode is justified by cost and responsiveness, not by being universally correct.",
                "- Accurate mode is justified by delayed-best cases and material regret under fast selection.",
                "- The accurate mode is the quality-first reference within the current candidate budget.",
                "- The research goal is not to collapse these into one mode, but to learn when fast mode is safe and when accurate mode must be used.",
            ]
        )
    (output_dir / "dongdaemun_two_mode_tradeoff_report.md").write_text(
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
    mode_rows = build_two_mode_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        material_delta_q=args.material_delta_q,
        material_relative_ppm=args.material_relative_ppm,
        coarse_endpoint_tau=args.coarse_endpoint_tau,
        coarse_support_tau=args.coarse_support_tau,
        iso_q_delta=args.iso_q_delta,
        iso_q_relative_ppm=args.iso_q_relative_ppm,
    )
    summary = build_two_mode_summary(mode_rows)
    need_rows = build_mode_need_rows(summary)
    mode_rows.to_csv(output_dir / "dongdaemun_two_mode_case_tradeoff.csv", index=False)
    summary.to_csv(output_dir / "dongdaemun_two_mode_summary.csv", index=False)
    need_rows.to_csv(output_dir / "dongdaemun_two_mode_need_evidence.csv", index=False)
    write_report(output_dir, mode_rows, summary, need_rows)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "mode_rows": int(len(mode_rows)),
            "summary_rows": int(len(summary)),
            "need_rows": int(len(need_rows)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
