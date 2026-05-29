#!/usr/bin/env python3
"""Reframe Leiden p5 candidates as a quality-first choice ledger."""

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
    _best_p1_rank,
    _case_field_method,
    _support_sketch_exact,
)
from analyze_leiden_multibasin_signatures import (
    TOP_K_VALUES,
    _finite_float,
    _group_columns,
    _mark_material_gain,
    _read_csvs,
    _signature_frame,
    build_coarse_basin_rows,
    build_pairwise_basin_matrix,
)

def _ranked_by_p1(group: pd.DataFrame) -> pd.DataFrame:
    return group.assign(
        _rank_metric=pd.to_numeric(group.get("p1_delta_q"), errors="coerce"),
        _candidate_index=pd.to_numeric(group.get("candidate_index"), errors="coerce").fillna(0),
        _p5_delta_q=pd.to_numeric(group.get("p5_delta_q"), errors="coerce"),
    ).sort_values(
        ["_rank_metric", "_candidate_index"],
        ascending=[False, True],
        na_position="last",
    )

def _coarse_by_candidate(coarse: pd.DataFrame) -> dict[int, int]:
    out: dict[int, int] = {}
    if coarse.empty:
        return out
    for _, row in coarse.iterrows():
        try:
            coarse_id = int(row.get("coarse_basin_id"))
        except (TypeError, ValueError):
            continue
        for part in str(row.get("member_candidate_indices", "")).split(";"):
            if not part:
                continue
            try:
                out[int(part)] = coarse_id
            except ValueError:
                continue
    return out

def _pairwise_choice_distance(
    pairwise: pd.DataFrame,
    base: dict[str, Any],
    left_candidate: int,
    right_candidate: int,
) -> dict[str, Any]:
    if left_candidate == right_candidate:
        return {
            "p1_and_best_same_coarse_basin": True,
            "p1_to_best_endpoint_distance": 0.0,
            "p1_to_best_support_distance": 0.0,
        }
    if pairwise.empty:
        return {
            "p1_and_best_same_coarse_basin": None,
            "p1_to_best_endpoint_distance": math.nan,
            "p1_to_best_support_distance": math.nan,
        }
    group_pairwise = pairwise[_base_mask(pairwise, base)]
    if group_pairwise.empty:
        return {
            "p1_and_best_same_coarse_basin": None,
            "p1_to_best_endpoint_distance": math.nan,
            "p1_to_best_support_distance": math.nan,
        }
    mask = (
        (group_pairwise["left_candidate_index"] == left_candidate)
        & (group_pairwise["right_candidate_index"] == right_candidate)
    ) | (
        (group_pairwise["left_candidate_index"] == right_candidate)
        & (group_pairwise["right_candidate_index"] == left_candidate)
    )
    if not mask.any():
        return {
            "p1_and_best_same_coarse_basin": None,
            "p1_to_best_endpoint_distance": math.nan,
            "p1_to_best_support_distance": math.nan,
        }
    row = group_pairwise[mask].iloc[0]
    return {
        "p1_and_best_same_coarse_basin": bool(row.get("same_coarse_basin", False)),
        "p1_to_best_endpoint_distance": _finite_float(
            row.get("sample_coassignment_distance")
        ),
        "p1_to_best_support_distance": _finite_float(
            row.get("coarse_support_distance")
        ),
    }

def _quality_first_frame(
    *,
    best_rank: int | None,
    top1_regret: float,
    acceptable_regret_q: float,
    material_regret_q: float,
    best_gain_material: bool,
) -> tuple[str, str]:
    if best_rank == 1:
        return (
            "p1_already_best",
            "the old greedy top1 and the quality-first choice are identical",
        )
    if not best_gain_material:
        return (
            "technical_best_only",
            "select the best endpoint, but do not claim a material improvement",
        )
    if math.isfinite(top1_regret) and top1_regret <= acceptable_regret_q:
        return (
            "near_tie_choose_best",
            "p1 is near the best endpoint, but quality-first still picks the max p5 row",
        )
    if math.isfinite(top1_regret) and top1_regret < material_regret_q:
        return (
            "low_margin_upgrade",
            "quality-first improves over p1, but the premium is below the material regret gate",
        )
    if best_rank is not None and best_rank <= 5:
        return (
            "delayed_best_shallow",
            "the best endpoint is delayed but still appears within the first five p1-ranked candidates",
        )
    return (
        "delayed_best_deep",
        "the best endpoint appears deep enough that top-k thrift would hide a material choice",
    )

def build_quality_first_choice_rows(
    candidates: pd.DataFrame,
    *,
    acceptable_regret_q: float = 1.0,
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
        best_idx = ranked["_p5_delta_q"].idxmax()
        best = ranked.loc[best_idx]
        p1_choice = ranked.iloc[0]
        best_rank = _best_p1_rank(labeled)
        best_delta = _finite_float(best.get("p5_delta_q"))
        p1_delta = _finite_float(p1_choice.get("p5_delta_q"))
        top1_regret = (
            best_delta - p1_delta
            if math.isfinite(best_delta) and math.isfinite(p1_delta)
            else math.nan
        )
        frame, frame_reason = _quality_first_frame(
            best_rank=best_rank,
            top1_regret=top1_regret,
            acceptable_regret_q=acceptable_regret_q,
            material_regret_q=material_regret_q,
            best_gain_material=bool(best.get("material_gain", False)),
        )
        group_coarse = coarse[_base_mask(coarse, base)] if not coarse.empty else coarse
        distance = _pairwise_choice_distance(
            pairwise,
            base,
            int(p1_choice.get("candidate_index", -1)),
            int(best.get("candidate_index", -1)),
        )
        field, method = _case_field_method(labeled.iloc[0])
        p5_elapsed = pd.to_numeric(labeled.get("p5_elapsed_ms"), errors="coerce")
        rows.append(
            {
                **base,
                "field": field,
                "method": method,
                "quality_first_frame": frame,
                "quality_first_reason": frame_reason,
                "selection_principle": "evaluate_candidate_budget_then_choose_max_p5_delta_q",
                "claim_strength": (
                    "material_best_choice"
                    if bool(best.get("material_gain", False))
                    else "technical_best_only"
                ),
                "candidate_count": int(len(labeled)),
                "coarse_basin_count": int(len(group_coarse)),
                "support_sketch_exact": _support_sketch_exact(labeled),
                "p1_candidate_index": int(p1_choice.get("candidate_index", -1)),
                "quality_first_candidate_index": int(best.get("candidate_index", -1)),
                "quality_first_p1_rank": best_rank,
                "extra_candidates_before_best": (
                    int(best_rank - 1) if best_rank is not None else None
                ),
                "p1_choice_p5_delta_q": p1_delta,
                "quality_first_p5_delta_q": best_delta,
                "quality_first_relative_delta_q_ppm": _finite_float(
                    best.get("p5_relative_delta_q_ppm")
                ),
                "quality_first_premium_over_p1_q": top1_regret,
                "quality_first_premium_fraction": (
                    top1_regret / abs(best_delta)
                    if math.isfinite(top1_regret) and abs(best_delta) > 0.0
                    else math.nan
                ),
                "quality_first_material_premium": bool(
                    math.isfinite(top1_regret) and top1_regret >= material_regret_q
                ),
                "quality_first_gain_material": bool(best.get("material_gain", False)),
                "p5_elapsed_ms_sum": _finite_float(p5_elapsed.sum()),
                "p5_elapsed_ms_until_best_by_p1_rank": _finite_float(
                    pd.to_numeric(ranked.head(best_rank or len(ranked)).get("p5_elapsed_ms"), errors="coerce").sum()
                )
                if "p5_elapsed_ms" in ranked.columns
                else math.nan,
                **distance,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["quality_first_material_premium", "quality_first_premium_over_p1_q", "field", "method"],
        ascending=[False, False, True, True],
        na_position="last",
    )

def build_quality_frontier_rows(
    candidates: pd.DataFrame,
    *,
    acceptable_regret_q: float = 1.0,
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
        best_idx = ranked["_p5_delta_q"].idxmax()
        full_best = ranked.loc[best_idx]
        full_best_delta = _finite_float(full_best.get("p5_delta_q"))
        group_coarse = coarse[_base_mask(coarse, base)] if not coarse.empty else coarse
        coarse_map = _coarse_by_candidate(group_coarse)
        field, method = _case_field_method(labeled.iloc[0])
        top_k_values = sorted(set(k for k in TOP_K_VALUES if k <= len(ranked)) | {len(ranked)})
        for top_k in top_k_values:
            selected = ranked.head(top_k)
            selected_best_idx = selected["_p5_delta_q"].idxmax()
            selected_best = selected.loc[selected_best_idx]
            selected_best_delta = _finite_float(selected_best.get("p5_delta_q"))
            regret = (
                full_best_delta - selected_best_delta
                if math.isfinite(full_best_delta) and math.isfinite(selected_best_delta)
                else math.nan
            )
            selected_indices = [
                int(value)
                for value in pd.to_numeric(
                    selected.get("candidate_index"),
                    errors="coerce",
                )
                .dropna()
                .tolist()
            ]
            selected_coarse = sorted(
                {coarse_map[index] for index in selected_indices if index in coarse_map}
            )
            rows.append(
                {
                    **base,
                    "field": field,
                    "method": method,
                    "top_k": int(top_k),
                    "p5_evaluated": int(len(selected)),
                    "selected_candidate_indices": ";".join(
                        str(index) for index in selected_indices
                    ),
                    "selected_coarse_basins": ";".join(
                        str(index) for index in selected_coarse
                    ),
                    "best_seen_candidate_index": int(
                        selected_best.get("candidate_index", -1)
                    ),
                    "quality_first_candidate_index": int(
                        full_best.get("candidate_index", -1)
                    ),
                    "quality_first_hit": bool(
                        int(selected_best.get("candidate_index", -1))
                        == int(full_best.get("candidate_index", -1))
                    ),
                    "best_so_far_p5_delta_q": selected_best_delta,
                    "quality_first_p5_delta_q": full_best_delta,
                    "quality_regret_q": regret,
                    "frontier_state": (
                        "oracle_recovered"
                        if math.isfinite(regret) and regret <= acceptable_regret_q
                        else (
                            "materially_short"
                            if math.isfinite(regret) and regret >= material_regret_q
                            else "near_best"
                        )
                    ),
                    "coarse_coverage_count": len(selected_coarse),
                    "coarse_total_count": int(len(group_coarse)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["field", "method", "top_k"], na_position="last")

def build_quality_first_summary(choice_rows: pd.DataFrame) -> pd.DataFrame:
    if choice_rows.empty:
        return pd.DataFrame()
    rows = [_summarize_choice_group("all", choice_rows)]
    if "field" in choice_rows.columns:
        for field, group in choice_rows.groupby("field", dropna=False):
            rows.append(_summarize_choice_group(f"field={field}", group))
    if "method" in choice_rows.columns:
        for method, group in choice_rows.groupby("method", dropna=False):
            rows.append(_summarize_choice_group(f"method={method}", group))
    return pd.DataFrame(rows)

def _summarize_choice_group(name: str, group: pd.DataFrame) -> dict[str, Any]:
    frames = group["quality_first_frame"].value_counts()
    premium = pd.to_numeric(
        group.get("quality_first_premium_over_p1_q"),
        errors="coerce",
    )
    return {
        "group": name,
        "case_count": int(len(group)),
        "p1_already_best_count": int(frames.get("p1_already_best", 0)),
        "near_tie_choose_best_count": int(frames.get("near_tie_choose_best", 0)),
        "low_margin_upgrade_count": int(frames.get("low_margin_upgrade", 0)),
        "delayed_best_shallow_count": int(frames.get("delayed_best_shallow", 0)),
        "delayed_best_deep_count": int(frames.get("delayed_best_deep", 0)),
        "technical_best_only_count": int(frames.get("technical_best_only", 0)),
        "material_premium_count": int(
            group.get("quality_first_material_premium", False).map(bool).sum()
        ),
        "premium_q_sum": _finite_float(premium.sum()),
        "premium_q_mean": _finite_float(premium.mean()),
        "max_quality_first_p1_rank": _finite_float(
            pd.to_numeric(group.get("quality_first_p1_rank"), errors="coerce").max()
        ),
        "support_exact_count": int(
            group.get("support_sketch_exact", False).map(lambda value: value is True).sum()
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
    choice_rows: pd.DataFrame,
    frontier_rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Quality-First Choice Review",
        "",
        "This diagnostic changes the framing from `how little can we evaluate` to `which endpoint is the best available choice under the current candidate budget`.",
        "",
    ]
    if choice_rows.empty:
        lines.append("- No quality-first choice rows were available.")
    else:
        total = len(choice_rows)
        p1_best = int((choice_rows["quality_first_frame"] == "p1_already_best").sum())
        material_premium = int(
            choice_rows["quality_first_material_premium"].map(bool).sum()
        )
        deep = int((choice_rows["quality_first_frame"] == "delayed_best_deep").sum())
        premium_sum = _finite_float(
            pd.to_numeric(
                choice_rows.get("quality_first_premium_over_p1_q"),
                errors="coerce",
            ).sum()
        )
        lines.extend(
            [
                "## Headline",
                "",
                f"- cases: {total}",
                f"- p1 already chooses the quality-first endpoint: {p1_best}/{total}",
                f"- material quality-first premium over p1: {material_premium}/{total}",
                f"- deep delayed-best cases: {deep}/{total}",
                f"- total p5 premium recovered by quality-first selection: {premium_sum:.6g}",
                "",
                "## Frame Summary",
                "",
            ]
        )
        display_summary = summary[
            [
                column
                for column in [
                    "group",
                    "case_count",
                    "p1_already_best_count",
                    "near_tie_choose_best_count",
                    "low_margin_upgrade_count",
                    "delayed_best_shallow_count",
                    "delayed_best_deep_count",
                    "material_premium_count",
                    "premium_q_sum",
                    "premium_q_mean",
                    "max_quality_first_p1_rank",
                ]
                if column in summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Largest Quality-First Premiums", ""])
        display_cols = [
            column
            for column in [
                "field",
                "method",
                "quality_first_frame",
                "p1_candidate_index",
                "quality_first_candidate_index",
                "quality_first_p1_rank",
                "quality_first_p5_delta_q",
                "quality_first_premium_over_p1_q",
                "p1_to_best_support_distance",
                "p1_and_best_same_coarse_basin",
            ]
            if column in choice_rows.columns
        ]
        lines.extend(
            _markdown_table(
                choice_rows.sort_values(
                    "quality_first_premium_over_p1_q",
                    ascending=False,
                )
                .head(10)[display_cols]
            ).splitlines()
        )
        if not frontier_rows.empty:
            lines.extend(["", "## Frontier Reading", ""])
            top_frontier = frontier_rows[
                frontier_rows["top_k"].isin([1, 3, 5, 10])
            ].copy()
            display_frontier_cols = [
                column
                for column in [
                    "field",
                    "method",
                    "top_k",
                    "best_seen_candidate_index",
                    "quality_first_candidate_index",
                    "quality_first_hit",
                    "quality_regret_q",
                    "frontier_state",
                    "coarse_coverage_count",
                    "coarse_total_count",
                ]
                if column in top_frontier.columns
            ]
            lines.extend(
                _markdown_table(
                    top_frontier.sort_values(
                        ["quality_regret_q", "field", "method", "top_k"],
                        ascending=[False, True, True, True],
                    ).head(30)[display_frontier_cols]
                ).splitlines()
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- The primary output is the quality-first endpoint, not a cheaper proxy policy.",
                "- `quality_first_p1_rank` is a deliberation-depth measurement: how far an old p1 ordering had to go before exposing the best available endpoint.",
                "- Cost columns are retained as accounting evidence, but they do not decide the endpoint in this frame.",
                "- Low-margin and near-tie rows still choose the best p5 endpoint; they simply should not be overclaimed as strong algorithmic wins.",
            ]
        )
    (output_dir / "dongdaemun_quality_first_choice_report.md").write_text(
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
    parser.add_argument("--acceptable-regret-q", type=float, default=1.0)
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
    kwargs = {
        "acceptable_regret_q": args.acceptable_regret_q,
        "material_regret_q": args.material_regret_q,
        "material_delta_q": args.material_delta_q,
        "material_relative_ppm": args.material_relative_ppm,
        "coarse_endpoint_tau": args.coarse_endpoint_tau,
        "coarse_support_tau": args.coarse_support_tau,
        "iso_q_delta": args.iso_q_delta,
        "iso_q_relative_ppm": args.iso_q_relative_ppm,
    }
    choice_rows = build_quality_first_choice_rows(candidates, **kwargs)
    frontier_rows = build_quality_frontier_rows(candidates, **kwargs)
    summary = build_quality_first_summary(choice_rows)
    choice_rows.to_csv(output_dir / "dongdaemun_quality_first_choice_ledger.csv", index=False)
    frontier_rows.to_csv(output_dir / "dongdaemun_quality_first_frontier_by_k.csv", index=False)
    summary.to_csv(output_dir / "dongdaemun_quality_first_summary.csv", index=False)
    write_report(output_dir, choice_rows, frontier_rows, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "choice_rows": int(len(choice_rows)),
            "frontier_rows": int(len(frontier_rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
