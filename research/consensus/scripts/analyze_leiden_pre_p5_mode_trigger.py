#!/usr/bin/env python3
"""Evaluate pre-p5 signals for Dongdaemun fast-to-accurate escalation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_basin_portfolio_evidence import build_basin_portfolio_rows
from analyze_leiden_multibasin_decision_rules import _case_field_method
from analyze_leiden_multibasin_signatures import (
    _finite_float,
    _group_columns,
    _read_csvs,
)
from analyze_leiden_quality_first_choice import _ranked_by_p1
from analyze_leiden_two_mode_tradeoff import build_two_mode_rows


CHEAP_METRIC_COLUMNS = [
    "priority",
    "p1_delta_q",
    "group_delta_q",
    "group_move_delta_q",
    "group_split_delta_q",
    "group_weight",
    "group_fraction",
    "group_cut_weight",
    "group_to_target_weight",
    "localized_delta_q",
    "quotient_delta_q",
    "pre_delta_q",
    "ub_delta_q",
    "incident_directed_edges",
]
TOP1_VALUE_COLUMNS = [
    "priority",
    "group_delta_q",
    "group_weight",
    "group_fraction",
    "group_cut_weight",
    "group_to_target_weight",
    "localized_delta_q",
    "quotient_delta_q",
    "pre_delta_q",
    "ub_delta_q",
    "incident_directed_edges",
    "assigned_fraction",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([math.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_value(series: pd.Series, position: int) -> float:
    if len(series) <= position:
        return math.nan
    return _finite_float(series.iloc[position])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / abs(denominator)


def _metric_top_candidate(group: pd.DataFrame, metric: str) -> int | None:
    if metric not in group.columns or "candidate_index" not in group.columns:
        return None
    values = pd.to_numeric(group[metric], errors="coerce")
    if values.notna().sum() == 0:
        return None
    ranked = group.assign(
        _metric=values,
        _candidate_index=pd.to_numeric(group.get("candidate_index"), errors="coerce").fillna(0),
    ).sort_values(["_metric", "_candidate_index"], ascending=[False, True], na_position="last")
    try:
        return int(ranked.iloc[0].get("candidate_index"))
    except (TypeError, ValueError):
        return None


def _mean_abs_spearman(group: pd.DataFrame) -> float:
    if "p1_delta_q" not in group.columns:
        return math.nan
    p1 = pd.to_numeric(group["p1_delta_q"], errors="coerce")
    values: list[float] = []
    for column in CHEAP_METRIC_COLUMNS:
        if column == "p1_delta_q" or column not in group.columns:
            continue
        other = pd.to_numeric(group[column], errors="coerce")
        valid = p1.notna() & other.notna()
        if valid.sum() < 2:
            continue
        if p1[valid].nunique() < 2 or other[valid].nunique() < 2:
            continue
        corr = p1[valid].rank().corr(other[valid].rank())
        if math.isfinite(float(corr)):
            values.append(abs(float(corr)))
    if not values:
        return math.nan
    return float(pd.Series(values).mean())


def _risk_score(row: dict[str, Any]) -> float:
    pieces: list[float] = []
    rel_gap = _finite_float(row.get("p1_gap_1_2_rel"))
    if math.isfinite(rel_gap):
        pieces.append(1.0 - max(0.0, min(rel_gap, 1.0)))
    agreement = _finite_float(row.get("metric_top1_agreement_rate"))
    if math.isfinite(agreement):
        pieces.append(1.0 - max(0.0, min(agreement, 1.0)))
    unique_count = _finite_float(row.get("metric_unique_top1_count"))
    metric_count = _finite_float(row.get("cheap_metric_count"))
    if math.isfinite(unique_count) and math.isfinite(metric_count) and metric_count > 1.0:
        pieces.append(max(0.0, min((unique_count - 1.0) / (metric_count - 1.0), 1.0)))
    mean_abs_corr = _finite_float(row.get("mean_abs_spearman_with_p1"))
    if math.isfinite(mean_abs_corr):
        pieces.append(1.0 - max(0.0, min(mean_abs_corr, 1.0)))
    near_top = _finite_float(row.get("p1_near_top_10pct_count"))
    candidate_count = _finite_float(row.get("candidate_count"))
    if math.isfinite(near_top) and math.isfinite(candidate_count) and candidate_count > 0.0:
        pieces.append(max(0.0, min(near_top / candidate_count, 1.0)))
    if not pieces:
        return math.nan
    return float(pd.Series(pieces).mean())


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


def _key(row: pd.Series, columns: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def build_pre_p5_feature_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        if group.empty:
            continue
        ranked = _ranked_by_p1(group)
        p1 = pd.to_numeric(ranked.get("p1_delta_q"), errors="coerce")
        top1 = _safe_value(p1, 0)
        top2 = _safe_value(p1, 1)
        top3 = _safe_value(p1, 2)
        top5 = _safe_value(p1, min(4, len(p1) - 1)) if len(p1) else math.nan
        gap12 = top1 - top2 if math.isfinite(top1) and math.isfinite(top2) else math.nan
        gap13 = top1 - top3 if math.isfinite(top1) and math.isfinite(top3) else math.nan
        gap15 = top1 - top5 if math.isfinite(top1) and math.isfinite(top5) else math.nan
        p1_candidate = int(ranked.iloc[0].get("candidate_index", -1))
        metric_tops = {
            metric: _metric_top_candidate(group, metric)
            for metric in CHEAP_METRIC_COLUMNS
            if metric in group.columns
        }
        metric_values = [value for value in metric_tops.values() if value is not None]
        agreement = sum(1 for value in metric_values if value == p1_candidate)
        unique_tops = len(set(metric_values))
        row: dict[str, Any] = {
            **base,
            "field": _case_field_method(group.iloc[0])[0],
            "method": _case_field_method(group.iloc[0])[1],
            "candidate_count": int(len(group)),
            "p1_candidate_index": p1_candidate,
            "p1_top1_delta_q": top1,
            "p1_top2_delta_q": top2,
            "p1_top3_delta_q": top3,
            "p1_top5_floor_delta_q": top5,
            "p1_gap_1_2_abs": gap12,
            "p1_gap_1_3_abs": gap13,
            "p1_gap_1_5_abs": gap15,
            "p1_gap_1_2_rel": _safe_ratio(gap12, top1),
            "p1_gap_1_3_rel": _safe_ratio(gap13, top1),
            "p1_gap_1_5_rel": _safe_ratio(gap15, top1),
            "p1_delta_mean": _finite_float(p1.mean()),
            "p1_delta_std": _finite_float(p1.std(ddof=0)),
            "p1_delta_range": _finite_float(p1.max() - p1.min()),
            "p1_positive_count": int((p1 > 0).sum()),
            "p1_near_top_10pct_count": int(
                (
                    p1
                    >= (
                        top1
                        - max(abs(top1) * 0.10, 1.0)
                        if math.isfinite(top1)
                        else math.inf
                    )
                ).sum()
            ),
            "cheap_metric_count": int(len(metric_values)),
            "metric_top1_agreement_count": int(agreement),
            "metric_top1_agreement_rate": (
                agreement / len(metric_values) if metric_values else math.nan
            ),
            "metric_unique_top1_count": int(unique_tops),
            "mean_abs_spearman_with_p1": _mean_abs_spearman(group),
        }
        for column in TOP1_VALUE_COLUMNS:
            if column in ranked.columns:
                row[f"top1_{column}"] = _finite_float(ranked.iloc[0].get(column))
                values = _num(group, column)
                row[f"{column}_mean"] = _finite_float(values.mean())
                row[f"{column}_std"] = _finite_float(values.std(ddof=0))
        row["pre_p5_risk_score"] = _risk_score(row)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["field", "method"], na_position="last")


def build_pre_p5_oracle_rows(
    candidates: pd.DataFrame,
    *,
    material_regret_q: float = 10.0,
    near_best_delta_q: float = 10.0,
    support_distinct_tau: float = 0.5,
) -> pd.DataFrame:
    features = build_pre_p5_feature_rows(candidates)
    if features.empty:
        return features
    mode_rows = build_two_mode_rows(candidates, material_regret_q=material_regret_q)
    fast_p1 = mode_rows[mode_rows["mode_name"] == "fast_p1"].copy()
    portfolio = build_basin_portfolio_rows(
        candidates,
        material_regret_q=material_regret_q,
        near_best_delta_q=near_best_delta_q,
        support_distinct_tau=support_distinct_tau,
    )
    key_cols = _key_columns(features)
    fast_by_key = {
        _key(row, key_cols): row
        for _, row in fast_p1.iterrows()
    }
    portfolio_by_key = {
        _key(row, key_cols): row
        for _, row in portfolio.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, feature in features.iterrows():
        key = _key(feature, key_cols)
        fast = fast_by_key.get(key, pd.Series(dtype=object))
        port = portfolio_by_key.get(key, pd.Series(dtype=object))
        material_quality = bool(fast.get("material_regret", False))
        similar_review = int(_finite_float(port.get("support_distinct_iso_q_pair_count"), 0.0)) > 0
        rows.append(
            {
                **feature.to_dict(),
                "oracle_accurate_for_quality": material_quality,
                "oracle_accurate_for_portfolio": similar_review,
                "oracle_accurate_for_final": bool(material_quality or similar_review),
                "oracle_fast_p1_quality_regret_q": _finite_float(
                    fast.get("quality_regret_q")
                ),
                "oracle_quality_first_p1_rank": _finite_float(
                    fast.get("quality_first_p1_rank")
                ),
                "oracle_support_distinct_iso_q_pair_count": int(
                    _finite_float(port.get("support_distinct_iso_q_pair_count"), 0.0)
                ),
                "oracle_better_basin_premium_q": _finite_float(
                    port.get("better_basin_premium_q")
                ),
                "oracle_portfolio_need_label": str(
                    port.get("portfolio_need_label", "")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oracle_accurate_for_final", "pre_p5_risk_score", "field", "method"],
        ascending=[False, False, True, True],
        na_position="last",
    )


def _evaluate_predictions(
    rows: pd.DataFrame,
    *,
    target_column: str,
    predicted: pd.Series,
    policy_name: str,
    heldout_field: Any,
    threshold: float | None,
) -> dict[str, Any]:
    target = rows[target_column].map(bool)
    pred = predicted.map(bool)
    false_negative = target & ~pred
    false_positive = ~target & pred
    return {
        "policy_name": policy_name,
        "target": target_column,
        "heldout_field": heldout_field,
        "threshold": threshold,
        "case_count": int(len(rows)),
        "required_count": int(target.sum()),
        "predicted_accurate_count": int(pred.sum()),
        "false_negative_count": int(false_negative.sum()),
        "false_positive_count": int(false_positive.sum()),
        "missed_quality_regret_q": _finite_float(
            pd.to_numeric(
                rows.loc[false_negative, "oracle_fast_p1_quality_regret_q"],
                errors="coerce",
            ).sum()
        ),
        "missed_support_distinct_iso_q_pairs": int(
            pd.to_numeric(
                rows.loc[false_negative, "oracle_support_distinct_iso_q_pair_count"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
    }


def build_leave_field_out_trigger_rows(
    oracle_rows: pd.DataFrame,
    *,
    target_columns: tuple[str, ...] = (
        "oracle_accurate_for_quality",
        "oracle_accurate_for_portfolio",
        "oracle_accurate_for_final",
    ),
) -> pd.DataFrame:
    if oracle_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    fields = sorted(value for value in oracle_rows["field"].dropna().unique())
    for target_column in target_columns:
        for field in fields:
            train = oracle_rows[oracle_rows["field"] != field].copy()
            test = oracle_rows[oracle_rows["field"] == field].copy()
            if train.empty or test.empty:
                continue
            required_scores = pd.to_numeric(
                train.loc[train[target_column].map(bool), "pre_p5_risk_score"],
                errors="coerce",
            ).dropna()
            threshold = float(required_scores.min()) if not required_scores.empty else math.inf
            predicted = pd.to_numeric(
                test["pre_p5_risk_score"],
                errors="coerce",
            ) >= threshold
            rows.append(
                _evaluate_predictions(
                    test,
                    target_column=target_column,
                    predicted=predicted,
                    policy_name="risk_score_lfo_conservative",
                    heldout_field=field,
                    threshold=threshold,
                )
            )
    return pd.DataFrame(rows)


def build_baseline_policy_rows(
    oracle_rows: pd.DataFrame,
    *,
    target_columns: tuple[str, ...] = (
        "oracle_accurate_for_quality",
        "oracle_accurate_for_portfolio",
        "oracle_accurate_for_final",
    ),
) -> pd.DataFrame:
    if oracle_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for target_column in target_columns:
        rows.append(
            _evaluate_predictions(
                oracle_rows,
                target_column=target_column,
                predicted=pd.Series([False] * len(oracle_rows), index=oracle_rows.index),
                policy_name="always_fast",
                heldout_field="all",
                threshold=None,
            )
        )
        rows.append(
            _evaluate_predictions(
                oracle_rows,
                target_column=target_column,
                predicted=pd.Series([True] * len(oracle_rows), index=oracle_rows.index),
                policy_name="always_accurate",
                heldout_field="all",
                threshold=None,
            )
        )
    return pd.DataFrame(rows)


def build_trigger_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary: list[dict[str, Any]] = []
    for (policy, target), group in rows.groupby(["policy_name", "target"], dropna=False):
        summary.append(
            {
                "policy_name": policy,
                "target": target,
                "fold_count": int(len(group)),
                "case_count": int(pd.to_numeric(group["case_count"], errors="coerce").sum()),
                "required_count": int(pd.to_numeric(group["required_count"], errors="coerce").sum()),
                "predicted_accurate_count": int(
                    pd.to_numeric(group["predicted_accurate_count"], errors="coerce").sum()
                ),
                "false_negative_count": int(
                    pd.to_numeric(group["false_negative_count"], errors="coerce").sum()
                ),
                "false_positive_count": int(
                    pd.to_numeric(group["false_positive_count"], errors="coerce").sum()
                ),
                "missed_quality_regret_q": _finite_float(
                    pd.to_numeric(group["missed_quality_regret_q"], errors="coerce").sum()
                ),
                "missed_support_distinct_iso_q_pairs": int(
                    pd.to_numeric(
                        group["missed_support_distinct_iso_q_pairs"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(summary)


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
    oracle_rows: pd.DataFrame,
    lfo_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Pre-p5 Mode Trigger Review",
        "",
        "This diagnostic uses only pre-p5 candidate features for escalation signals. p5-derived values appear only as oracle labels and missed-regret accounting.",
        "",
    ]
    if oracle_rows.empty:
        lines.append("- No pre-p5 trigger rows were available.")
    else:
        total = len(oracle_rows)
        quality_required = int(oracle_rows["oracle_accurate_for_quality"].map(bool).sum())
        portfolio_required = int(oracle_rows["oracle_accurate_for_portfolio"].map(bool).sum())
        final_required = int(oracle_rows["oracle_accurate_for_final"].map(bool).sum())
        lines.extend(
            [
                "## Headline",
                "",
                f"- cases: {total}",
                f"- quality-accurate required: {quality_required}/{total}",
                f"- portfolio-accurate required: {portfolio_required}/{total}",
                f"- final accurate required: {final_required}/{total}",
                "",
                "## Policy Summary",
                "",
            ]
        )
        display_summary = summary_rows[
            [
                column
                for column in [
                    "policy_name",
                    "target",
                    "fold_count",
                    "case_count",
                    "required_count",
                    "predicted_accurate_count",
                    "false_negative_count",
                    "false_positive_count",
                    "missed_quality_regret_q",
                    "missed_support_distinct_iso_q_pairs",
                ]
                if column in summary_rows.columns
            ]
        ]
        lines.extend(_markdown_table(display_summary).splitlines())
        lines.extend(["", "## Leave-Field-Out Conservative Trigger", ""])
        if lfo_rows.empty:
            lines.append("- Leave-field-out rows require at least two fields.")
        else:
            lfo_display = lfo_rows.sort_values(["target", "heldout_field"])
            lines.extend(
                _markdown_table(
                    lfo_display[
                        [
                            column
                            for column in [
                                "target",
                                "heldout_field",
                                "threshold",
                                "case_count",
                                "required_count",
                                "predicted_accurate_count",
                                "false_negative_count",
                                "missed_quality_regret_q",
                                "missed_support_distinct_iso_q_pairs",
                            ]
                            if column in lfo_display.columns
                        ]
                    ]
                ).splitlines()
            )
        lines.extend(["", "## Highest Pre-p5 Risk Cases", ""])
        risk_cols = [
            column
            for column in [
                "field",
                "method",
                "pre_p5_risk_score",
                "metric_top1_agreement_rate",
                "metric_unique_top1_count",
                "p1_gap_1_2_rel",
                "oracle_accurate_for_quality",
                "oracle_accurate_for_portfolio",
                "oracle_fast_p1_quality_regret_q",
                "oracle_support_distinct_iso_q_pair_count",
            ]
            if column in oracle_rows.columns
        ]
        lines.extend(
            _markdown_table(
                oracle_rows.sort_values("pre_p5_risk_score", ascending=False)
                .head(12)[risk_cols]
            ).splitlines()
        )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- The target is not to maximize fast-mode usage. The target is to avoid missing material better basins and support-distinct near-QF alternatives.",
                "- If portfolio review is part of the final output contract, most cases are accurate-required in this 20-case slice.",
                "- A cheap trigger should be accepted only if its false negatives carry negligible QF regret and negligible hidden lookalike support.",
                "- Leave-field-out rows are calibration evidence, not a production trigger guarantee.",
            ]
        )
    (output_dir / "dongdaemun_pre_p5_mode_trigger_report.md").write_text(
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
    oracle_rows = build_pre_p5_oracle_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    lfo_rows = build_leave_field_out_trigger_rows(oracle_rows)
    baseline_rows = build_baseline_policy_rows(oracle_rows)
    summary_rows = build_trigger_summary(pd.concat([baseline_rows, lfo_rows], ignore_index=True))
    oracle_rows.to_csv(output_dir / "dongdaemun_pre_p5_feature_oracle_rows.csv", index=False)
    lfo_rows.to_csv(output_dir / "dongdaemun_pre_p5_lfo_trigger_rows.csv", index=False)
    baseline_rows.to_csv(output_dir / "dongdaemun_pre_p5_baseline_policy_rows.csv", index=False)
    summary_rows.to_csv(output_dir / "dongdaemun_pre_p5_trigger_summary.csv", index=False)
    write_report(output_dir, oracle_rows, lfo_rows, baseline_rows, summary_rows)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "oracle_rows": int(len(oracle_rows)),
            "lfo_rows": int(len(lfo_rows)),
            "baseline_rows": int(len(baseline_rows)),
            "summary_rows": int(len(summary_rows)),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
