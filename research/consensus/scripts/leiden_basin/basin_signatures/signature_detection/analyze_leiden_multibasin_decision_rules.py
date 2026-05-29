#!/usr/bin/env python3
"""Calibrate greedy-failure decision rules from Leiden p5 basin candidates."""

from __future__ import annotations

import argparse
import math
import re
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

from analyze_leiden_multibasin_signatures import (
    TOP_K_VALUES,
    _finite_float,
    _group_columns,
    _mark_material_gain,
    _read_csvs,
    _signature_frame,
    build_basin_summary,
    build_coarse_basin_rows,
    build_coverage_rows,
    build_pairwise_basin_matrix,
)

KNOWN_METHODS = (
    "citation_embedding",
    "citation_all",
    "bc_cosine",
    "cc_cosine",
    "emb_knn",
)

def _case_field_method(row: pd.Series) -> tuple[int | None, str]:
    text = " ".join(
        str(row.get(column, ""))
        for column in ("case", "source_path", "batch_case_slug")
        if column in row.index
    )
    field_match = re.search(r"field(\d+)", text)
    field = int(field_match.group(1)) if field_match else None
    method = ""
    for candidate in KNOWN_METHODS:
        if candidate in text:
            method = candidate
            break
    return field, method

def _base_mask(frame: pd.DataFrame, base: dict[str, Any]) -> pd.Series:
    mask = pd.Series([True] * len(frame), index=frame.index)
    for column, value in base.items():
        if column in frame.columns:
            mask &= frame[column] == value
    return mask

def _topk_lookup(coverage: pd.DataFrame, base: dict[str, Any]) -> dict[int, pd.Series]:
    if coverage.empty:
        return {}
    group = coverage[_base_mask(coverage, base)]
    out: dict[int, pd.Series] = {}
    for _, row in group.iterrows():
        try:
            top_k = int(row.get("top_k"))
        except (TypeError, ValueError):
            continue
        out[top_k] = row
    return out

def _minimal_k_to_regret(group: pd.DataFrame, *, acceptable_regret_q: float) -> int | None:
    ranked = group.assign(
        _rank_metric=pd.to_numeric(group.get("p1_delta_q"), errors="coerce"),
        _candidate_index=pd.to_numeric(group.get("candidate_index"), errors="coerce").fillna(0),
        _p5_delta_q=pd.to_numeric(group.get("p5_delta_q"), errors="coerce"),
    ).sort_values(
        ["_rank_metric", "_candidate_index"],
        ascending=[False, True],
        na_position="last",
    )
    best = _finite_float(ranked["_p5_delta_q"].max())
    if not math.isfinite(best):
        return None
    best_seen = -math.inf
    for pos, value in enumerate(ranked["_p5_delta_q"], start=1):
        if math.isfinite(float(value)):
            best_seen = max(best_seen, float(value))
        if math.isfinite(best_seen) and best - best_seen <= acceptable_regret_q:
            return pos
    return None

def _best_p1_rank(group: pd.DataFrame) -> int | None:
    ranked = group.assign(
        _rank_metric=pd.to_numeric(group.get("p1_delta_q"), errors="coerce"),
        _candidate_index=pd.to_numeric(group.get("candidate_index"), errors="coerce").fillna(0),
        _p5_delta_q=pd.to_numeric(group.get("p5_delta_q"), errors="coerce"),
    ).sort_values(
        ["_rank_metric", "_candidate_index"],
        ascending=[False, True],
        na_position="last",
    )
    if ranked.empty or ranked["_p5_delta_q"].notna().sum() == 0:
        return None
    best_idx = ranked["_p5_delta_q"].idxmax()
    positions = {idx: pos for pos, idx in enumerate(ranked.index, start=1)}
    return positions.get(best_idx)

def _decision_label(
    *,
    best_gain_material: bool,
    top1_regret: float,
    top5_regret: float,
    min_k_to_acceptable: int | None,
    acceptable_regret_q: float,
    material_regret_q: float,
) -> tuple[str, str]:
    if not best_gain_material:
        return (
            "low_roi_skip_expansion",
            "best p5 gain does not clear the material-gain gate",
        )
    if math.isfinite(top1_regret) and top1_regret <= acceptable_regret_q:
        return ("p1_sufficient", "p1 top1 is within the acceptable regret band")
    if math.isfinite(top1_regret) and top1_regret < material_regret_q:
        return (
            "optional_topk_low_margin",
            "p1 misses, but the top1 regret is below the material regret gate",
        )
    if min_k_to_acceptable is None:
        return (
            "full_budget_or_new_perturbation",
            "no p1-ranked prefix reaches the acceptable regret band",
        )
    if min_k_to_acceptable <= 3:
        return ("top3_guard", "a shallow top3 guard recovers the p5 endpoint")
    if min_k_to_acceptable <= 5:
        return ("top5_guard", "top5 is needed to reach the acceptable regret band")
    if min_k_to_acceptable <= 10:
        return ("top10_guard", "top10 is needed to reach the acceptable regret band")
    return (
        "full_budget_or_new_perturbation",
        "the acceptable endpoint appears beyond top10",
    )

def _support_sketch_exact(group: pd.DataFrame) -> bool | None:
    count_col = "p5_basin_changed_support_node_count"
    sample_col = "p5_basin_changed_support_sketch_sample_size"
    if count_col not in group.columns or sample_col not in group.columns:
        return None
    counts = pd.to_numeric(group[count_col], errors="coerce")
    samples = pd.to_numeric(group[sample_col], errors="coerce")
    valid = counts.notna() & samples.notna()
    if not valid.any():
        return None
    return bool((counts[valid] <= samples[valid]).all())

def build_case_decision_rows(
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
    coverage = build_coverage_rows(signature_rows)
    basin_summary = build_basin_summary(signature_rows)
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
        p5 = pd.to_numeric(group.get("p5_delta_q"), errors="coerce")
        labeled = group[p5.notna()].copy()
        if labeled.empty:
            continue
        best_idx = p5.idxmax()
        best = group.loc[best_idx]
        topk = _topk_lookup(coverage, base)
        top1_regret = _finite_float(topk.get(1, pd.Series(dtype=object)).get("best_quality_regret_at_k"))
        top2_regret = _finite_float(topk.get(2, pd.Series(dtype=object)).get("best_quality_regret_at_k"))
        top3_regret = _finite_float(topk.get(3, pd.Series(dtype=object)).get("best_quality_regret_at_k"))
        top5_regret = _finite_float(topk.get(5, pd.Series(dtype=object)).get("best_quality_regret_at_k"))
        top10_regret = _finite_float(topk.get(10, pd.Series(dtype=object)).get("best_quality_regret_at_k"))
        min_k_to_acceptable = _minimal_k_to_regret(
            labeled,
            acceptable_regret_q=acceptable_regret_q,
        )
        best_gain_material = bool(best.get("material_gain", False))
        label, reason = _decision_label(
            best_gain_material=best_gain_material,
            top1_regret=top1_regret,
            top5_regret=top5_regret,
            min_k_to_acceptable=min_k_to_acceptable,
            acceptable_regret_q=acceptable_regret_q,
            material_regret_q=material_regret_q,
        )

        group_pairwise = pairwise[_base_mask(pairwise, base)] if not pairwise.empty else pairwise
        group_coarse = coarse[_base_mask(coarse, base)] if not coarse.empty else coarse
        group_summary = (
            basin_summary[_base_mask(basin_summary, base)].iloc[0]
            if not basin_summary.empty and _base_mask(basin_summary, base).any()
            else pd.Series(dtype=object)
        )
        first_field, first_method = _case_field_method(labeled.iloc[0])
        rows.append(
            {
                **base,
                "field": first_field,
                "method": first_method,
                "decision_label": label,
                "decision_reason": reason,
                "acceptable_regret_q": acceptable_regret_q,
                "material_regret_q": material_regret_q,
                "candidate_count": int(len(labeled)),
                "exact_basin_count": int(
                    group_summary.get(
                        "distinct_basin_count",
                        labeled["p5_basin_signature"].nunique(),
                    )
                ),
                "coarse_basin_count": int(len(group_coarse)),
                "coarse_basin_ratio": (
                    float(len(group_coarse)) / float(len(labeled))
                    if len(labeled) > 0
                    else math.nan
                ),
                "partition_distinct_iso_q_pair_count": int(
                    group_pairwise.get("partition_distinct_iso_q_pair", False).map(bool).sum()
                )
                if not group_pairwise.empty
                else 0,
                "iso_q_pair_count": int(group_pairwise.get("iso_q_pair", False).map(bool).sum())
                if not group_pairwise.empty
                else 0,
                "mean_support_distance": _finite_float(
                    pd.to_numeric(
                        group_pairwise.get("coarse_support_distance"),
                        errors="coerce",
                    ).mean()
                )
                if not group_pairwise.empty
                else math.nan,
                "max_support_distance": _finite_float(
                    pd.to_numeric(
                        group_pairwise.get("coarse_support_distance"),
                        errors="coerce",
                    ).max()
                )
                if not group_pairwise.empty
                else math.nan,
                "support_sketch_exact": _support_sketch_exact(labeled),
                "best_candidate_index": int(best.get("candidate_index", -1)),
                "best_p1_rank": _best_p1_rank(labeled),
                "best_p5_delta_q": _finite_float(best.get("p5_delta_q")),
                "best_relative_delta_q_ppm": _finite_float(
                    best.get("p5_relative_delta_q_ppm")
                ),
                "best_gain_material": best_gain_material,
                "top1_regret_q": top1_regret,
                "top2_regret_q": top2_regret,
                "top3_regret_q": top3_regret,
                "top5_regret_q": top5_regret,
                "top10_regret_q": top10_regret,
                "top1_regret_fraction_of_best_gain": (
                    top1_regret / abs(_finite_float(best.get("p5_delta_q")))
                    if math.isfinite(top1_regret)
                    and abs(_finite_float(best.get("p5_delta_q"))) > 0.0
                    else math.nan
                ),
                "min_k_to_acceptable_regret": min_k_to_acceptable,
                "p1_top1_material_miss": bool(
                    math.isfinite(top1_regret) and top1_regret >= material_regret_q
                ),
                "p1_top5_material_miss": bool(
                    math.isfinite(top5_regret) and top5_regret >= material_regret_q
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["p1_top1_material_miss", "top1_regret_q", "field", "method"],
        ascending=[False, False, True, True],
        na_position="last",
    )

def build_decision_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    group_cols = [column for column in ("field", "method") if column in decisions.columns]
    rows: list[dict[str, Any]] = []
    for key_name, group in [("all", decisions)]:
        rows.append(_summarize_group(key_name, group))
    if "field" in group_cols:
        for field, group in decisions.groupby("field", dropna=False):
            rows.append(_summarize_group(f"field={field}", group))
    if "method" in group_cols:
        for method, group in decisions.groupby("method", dropna=False):
            rows.append(_summarize_group(f"method={method}", group))
    return pd.DataFrame(rows)

def _summarize_group(name: str, group: pd.DataFrame) -> dict[str, Any]:
    labels = group["decision_label"].value_counts()
    return {
        "group": name,
        "case_count": int(len(group)),
        "p1_sufficient_count": int(labels.get("p1_sufficient", 0)),
        "optional_low_margin_count": int(labels.get("optional_topk_low_margin", 0)),
        "top3_guard_count": int(labels.get("top3_guard", 0)),
        "top5_guard_count": int(labels.get("top5_guard", 0)),
        "top10_guard_count": int(labels.get("top10_guard", 0)),
        "full_budget_or_new_perturbation_count": int(
            labels.get("full_budget_or_new_perturbation", 0)
        ),
        "low_roi_skip_expansion_count": int(labels.get("low_roi_skip_expansion", 0)),
        "top1_material_miss_count": int(group["p1_top1_material_miss"].map(bool).sum()),
        "top5_material_miss_count": int(group["p1_top5_material_miss"].map(bool).sum()),
        "top1_regret_q_mean": _finite_float(
            pd.to_numeric(group.get("top1_regret_q"), errors="coerce").mean()
        ),
        "top5_regret_q_mean": _finite_float(
            pd.to_numeric(group.get("top5_regret_q"), errors="coerce").mean()
        ),
        "coarse_basin_ratio_mean": _finite_float(
            pd.to_numeric(group.get("coarse_basin_ratio"), errors="coerce").mean()
        ),
        "partition_distinct_iso_q_pair_count": int(
            pd.to_numeric(
                group.get("partition_distinct_iso_q_pair_count"),
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

def write_report(output_dir: Path, decisions: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Dongdaemun Greedy Failure Decision Review",
        "",
        "This diagnostic converts p1-vs-p5 basin evidence into a provisional top-k decision rule. It is calibration evidence, not an accepted production policy.",
        "",
    ]
    if decisions.empty:
        lines.append("- No decision rows were available.")
    else:
        total = len(decisions)
        p1_ok = int((decisions["decision_label"] == "p1_sufficient").sum())
        top1_miss = int(decisions["p1_top1_material_miss"].map(bool).sum())
        top5_miss = int(decisions["p1_top5_material_miss"].map(bool).sum())
        exact_support = int(decisions["support_sketch_exact"].map(lambda value: value is True).sum())
        lines.extend(
            [
                "## Headline",
                "",
                f"- cases: {total}",
                f"- p1 sufficient cases: {p1_ok}/{total}",
                f"- material top1 misses: {top1_miss}/{total}",
                f"- material top5 residual misses: {top5_miss}/{total}",
                f"- exact support sketches: {exact_support}/{total}",
                "",
                "## Decision Summary",
                "",
            ]
        )
        display_summary = summary[
            [
                column
                for column in [
                    "group",
                    "case_count",
                    "p1_sufficient_count",
                    "optional_low_margin_count",
                    "top3_guard_count",
                    "top5_guard_count",
                    "top10_guard_count",
                    "full_budget_or_new_perturbation_count",
                    "top1_material_miss_count",
                    "top5_material_miss_count",
                    "top1_regret_q_mean",
                    "top5_regret_q_mean",
                ]
                if column in summary.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Largest Greedy Misses", ""])
        display_cols = [
            column
            for column in [
                "field",
                "method",
                "decision_label",
                "best_candidate_index",
                "best_p1_rank",
                "best_p5_delta_q",
                "top1_regret_q",
                "top5_regret_q",
                "min_k_to_acceptable_regret",
                "coarse_basin_count",
                "partition_distinct_iso_q_pair_count",
                "mean_support_distance",
            ]
            if column in decisions.columns
        ]
        lines.extend(
            _markdown_table(
                decisions.sort_values("top1_regret_q", ascending=False)
                .head(10)[display_cols]
            ).splitlines()
        )
        lines.extend(
            [
                "",
                "## Reading",
                "",
                "- Use `p1_sufficient` as evidence that top1 is enough under this regret gate.",
                "- Use `top3_guard`, `top5_guard`, and `top10_guard` as candidate budget lower bounds, not as proof that the same k will generalize.",
                "- Treat `full_budget_or_new_perturbation` as the region where rank-only expansion may be inefficient and a different perturbation family should be tested.",
                "- Cases below the material-gain gate should not be counted as meaningful wins even if p5 technically improves QF.",
            ]
        )
    (output_dir / "dongdaemun_greedy_failure_decision_report.md").write_text(
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
    decisions = build_case_decision_rows(
        candidates,
        acceptable_regret_q=args.acceptable_regret_q,
        material_regret_q=args.material_regret_q,
        material_delta_q=args.material_delta_q,
        material_relative_ppm=args.material_relative_ppm,
        coarse_endpoint_tau=args.coarse_endpoint_tau,
        coarse_support_tau=args.coarse_support_tau,
        iso_q_delta=args.iso_q_delta,
        iso_q_relative_ppm=args.iso_q_relative_ppm,
    )
    summary = build_decision_summary(decisions)
    decisions.to_csv(output_dir / "dongdaemun_greedy_failure_case_decisions.csv", index=False)
    summary.to_csv(output_dir / "dongdaemun_greedy_failure_decision_summary.csv", index=False)
    write_report(output_dir, decisions, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "decision_rows": int(len(decisions)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
