#!/usr/bin/env python3
"""Score p1-visible exception detectors for Leiden perturbation candidates.

This postprocessor treats p1 as a cheap fidelity/budget setting, not as the
algorithm.  It asks whether cheap p1-stage features can flag rows where a
perturbation-aware candidate branch needs top3/full-p5 fallback to preserve the
full p5 decision.
"""

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

SCRIPT_DIR = Path(__file__).resolve().parent
from analyze_leiden_multifidelity_candidate_misses import (  # noqa: E402
    _finite_float,
    _rank_order,
    _select_structural_rescue_row,
    _truthy,
    augment_policy_rows,
    build_candidate_rank_diagnostics,
    build_policy_decision_summary,
)

DEFAULT_INPUT_DIRS = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_multifidelity_label_field30_20260513",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_p1_exception_detector_20260514"
)

GROUP_COLUMNS = ("dataset", "case", "seed", "candidate_budget", "candidate_eval_mode")
TOP_FEATURE_COLUMNS = (
    "candidate_index",
    "p1_delta_q",
    "pre_delta_q",
    "best_group_delta_q",
    "group_count",
    "group_weight",
    "group_fraction",
    "group_to_target_weight",
    "group_cut_weight",
    "target_weight_per_node",
    "group_kind",
    "priority",
    "recommended_for_split_repair",
)

def _parse_input_dirs(value: str | None) -> list[Path]:
    if value is None:
        return [path.resolve() for path in DEFAULT_INPUT_DIRS]
    out = [Path(part).expanduser().resolve() for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one input dir")
    return out

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def _dataset_label(path: Path) -> str:
    return path.expanduser().resolve().name

def _matching_rows(frame: pd.DataFrame, **key: Any) -> pd.DataFrame:
    rows = frame
    for column, value in key.items():
        if column in rows.columns:
            rows = rows[rows[column] == value]
    return rows

def _safe_bool(value: Any) -> bool | str:
    if value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return _truthy(value)

def _target_changed(summary: pd.DataFrame, key: dict[str, Any]) -> bool | str:
    if summary.empty:
        return ""
    rows = _matching_rows(
        summary,
        case=key.get("case"),
        seed=key.get("seed"),
        candidate_budget=key.get("candidate_budget"),
        candidate_eval_mode=key.get("candidate_eval_mode"),
    )
    if rows.empty:
        return ""
    return _safe_bool(rows.iloc[0].get("extra_p5_final_target_conclusion_changed", ""))

def _budget_sufficient(row: pd.Series | None) -> bool | str:
    if row is None:
        return ""
    target_changed = _safe_bool(row.get("extra_p5_final_target_conclusion_changed", ""))
    if target_changed == "":
        extra = str(row.get("extra_tau_status", row.get("extra_p5_final_extra_tau_status", "")))
        perturb = str(row.get("perturb_tau_status", row.get("extra_p5_final_perturb_tau_status", "")))
        if extra and perturb:
            return extra == "reached" and perturb == "reached"
        return ""
    return not bool(target_changed)

def _budget_labels(scorecard: pd.DataFrame, key: dict[str, Any]) -> dict[str, Any]:
    if scorecard.empty or "target_policy" not in scorecard:
        return {
            "budget1_sufficient": "",
            "budget3_sufficient": "",
            "budget3_exception": "",
        }
    rows = scorecard[
        (scorecard["case"] == key.get("case"))
        & (scorecard["seed"].astype(int) == int(key.get("seed", 0)))
        & (scorecard["target_policy"].astype(str) == "extra_p5_final")
    ]
    if rows.empty:
        return {
            "budget1_sufficient": "",
            "budget3_sufficient": "",
            "budget3_exception": "",
        }
    by_budget = {
        int(row["candidate_budget"]): row
        for _, row in rows.iterrows()
        if math.isfinite(_finite_float(row.get("candidate_budget"), math.nan))
    }
    b1 = _budget_sufficient(by_budget.get(1))
    b3 = _budget_sufficient(by_budget.get(3))
    exception = "" if b1 == "" or b3 == "" else (not bool(b1) and bool(b3))
    return {
        "budget1_sufficient": b1,
        "budget3_sufficient": b3,
        "budget3_exception": exception,
    }

def _top_row(group: pd.DataFrame, rank: int) -> pd.Series | None:
    rows = group[group["p1_rank"].map(lambda value: _finite_float(value, math.inf) == float(rank))]
    if rows.empty:
        return None
    return rows.sort_values("candidate_index").iloc[0]

def _prefixed_top_features(rank: int, row: pd.Series | None) -> dict[str, Any]:
    prefix = f"p1_top{rank}_"
    if row is None:
        return {f"{prefix}{column}": math.nan for column in TOP_FEATURE_COLUMNS}
    return {f"{prefix}{column}": row.get(column, math.nan) for column in TOP_FEATURE_COLUMNS}

def build_exception_feature_rows(
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [column for column in GROUP_COLUMNS if column in diagnostics.columns]
    for key, group in diagnostics.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key_values, strict=False))
        ordered = [_top_row(group, rank) for rank in (1, 2, 3)]
        winner_rows = group[group["is_full_p5_winner"].map(_truthy)]
        winner = None if winner_rows.empty else winner_rows.iloc[0]
        top1 = ordered[0]
        top2 = ordered[1]
        top3 = ordered[2]
        top1_delta = math.nan if top1 is None else _finite_float(top1.get("p1_delta_q"), math.nan)
        top2_delta = math.nan if top2 is None else _finite_float(top2.get("p1_delta_q"), math.nan)
        top3_delta = math.nan if top3 is None else _finite_float(top3.get("p1_delta_q"), math.nan)
        rescue = _select_structural_rescue_row(group)
        rescue_index = -1 if rescue is None else int(rescue.get("candidate_index", -1))
        base: dict[str, Any] = {
            **key_dict,
            "candidate_count": int(len(group)),
            "p1_top1_top2_gap": top1_delta - top2_delta
            if math.isfinite(top1_delta) and math.isfinite(top2_delta)
            else math.nan,
            "p1_top2_top3_gap": top2_delta - top3_delta
            if math.isfinite(top2_delta) and math.isfinite(top3_delta)
            else math.nan,
            "full_p5_winner_candidate_index": -1
            if winner is None
            else int(winner.get("candidate_index", -1)),
            "full_p5_winner_p1_rank": math.nan
            if winner is None
            else _finite_float(winner.get("p1_rank"), math.nan),
            "full_p5_winner_p5_rank": math.nan
            if winner is None
            else _finite_float(winner.get("p5_rank"), math.nan),
            "full_p5_winner_p1_delta_q": math.nan
            if winner is None
            else _finite_float(winner.get("p1_delta_q"), math.nan),
            "full_p5_winner_p5_delta_q": math.nan
            if winner is None
            else _finite_float(winner.get("p5_delta_q"), math.nan),
            "missed_by_p1_top1": False
            if winner is None
            else _truthy(winner.get("missed_by_p1_top1")),
            "missed_by_p1_top2": False
            if winner is None
            else _truthy(winner.get("missed_by_p1_top2")),
            "needs_p1_top3": False
            if winner is None
            else _finite_float(winner.get("p1_rank"), math.inf) > 2.0,
            "structural_rescue_triggered": rescue is not None,
            "structural_rescue_candidate_index": rescue_index,
            "structural_rescue_candidate_is_full_p5_winner": False
            if rescue is None
            else _truthy(rescue.get("is_full_p5_winner")),
            "extra_p5_final_target_conclusion_changed": _target_changed(summary, key_dict),
        }
        for rank, row in enumerate(ordered, start=1):
            base.update(_prefixed_top_features(rank, row))
        base.update(_budget_labels(scorecard, key_dict))
        rows.append(base)
    return pd.DataFrame(rows).sort_values(group_cols)

def _selected_indices(group: pd.DataFrame, top_n: int) -> list[int]:
    ordered = _rank_order(group, "p1_delta_q")
    out: list[int] = []
    for index in ordered[:top_n]:
        out.append(int(group.loc[index, "candidate_index"]))
    return out

def _indices_for_policy(group: pd.DataFrame, feature_row: pd.Series, policy: str) -> tuple[list[int], bool]:
    if policy == "always_p1_top1":
        return _selected_indices(group, 1), False
    if policy == "always_p1_top2":
        return _selected_indices(group, 2), False
    if policy == "always_p1_top3":
        return _selected_indices(group, 3), True
    if policy == "p1_top2_with_gap_fallback_0p25":
        fallback = _finite_float(feature_row.get("p1_top2_top3_gap"), math.inf) <= 0.25
        return _selected_indices(group, 3 if fallback else 2), fallback
    if policy == "p1_top2_with_gap_fallback_0p50":
        fallback = _finite_float(feature_row.get("p1_top2_top3_gap"), math.inf) <= 0.50
        return _selected_indices(group, 3 if fallback else 2), fallback
    if policy == "p1_top2_with_structural_rescue":
        indices = _selected_indices(group, 2)
        rescue = int(_finite_float(feature_row.get("structural_rescue_candidate_index"), -1.0))
        fallback = rescue >= 0 and rescue not in indices
        if fallback:
            indices.append(rescue)
        return indices, fallback
    if policy == "p1_top2_with_budget3_exception":
        structural = _truthy(feature_row.get("structural_rescue_triggered"))
        small_gap = _finite_float(feature_row.get("p1_top2_top3_gap"), math.inf) <= 0.50
        fallback = bool(structural or small_gap)
        return _selected_indices(group, 3 if fallback else 2), fallback
    raise ValueError(f"unknown policy: {policy}")

def _elapsed_for_indices(group: pd.DataFrame, indices: list[int]) -> tuple[int, float, bool]:
    p1_elapsed = sum(
        _finite_float(value, 0.0)
        for value in group.get("p1_elapsed_ms", pd.Series(dtype=float))
        if math.isfinite(_finite_float(value, math.nan))
    )
    selected = group[group["candidate_index"].astype(int).isin(indices)]
    p5_values = [
        _finite_float(value, math.nan)
        for value in selected.get("p5_elapsed_ms", pd.Series(dtype=float))
    ]
    available = bool(p5_values) and all(math.isfinite(value) for value in p5_values)
    p5_elapsed = sum(value for value in p5_values if math.isfinite(value))
    return len(p5_values), p1_elapsed + p5_elapsed, available

def _full_elapsed(group: pd.DataFrame) -> float:
    values = [
        _finite_float(value, math.nan)
        for value in group.get("p5_elapsed_ms", pd.Series(dtype=float))
    ]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) if finite else math.nan

def build_policy_evaluation_rows(
    diagnostics: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    if diagnostics.empty or feature_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [column for column in GROUP_COLUMNS if column in diagnostics.columns]
    policies = (
        "always_p1_top1",
        "always_p1_top2",
        "always_p1_top3",
        "p1_top2_with_gap_fallback_0p25",
        "p1_top2_with_gap_fallback_0p50",
        "p1_top2_with_structural_rescue",
        "p1_top2_with_budget3_exception",
    )
    feature_by_key = {
        tuple(row[column] for column in group_cols): row
        for _, row in feature_rows.iterrows()
    }
    for key, group in diagnostics.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_tuple = tuple(key_values)
        key_dict = dict(zip(group_cols, key_values, strict=False))
        feature_row = feature_by_key[key_tuple]
        winner = int(_finite_float(feature_row.get("full_p5_winner_candidate_index"), -1.0))
        full_elapsed = _full_elapsed(group)
        p1_top2_miss = _truthy(feature_row.get("missed_by_p1_top2"))
        for policy in policies:
            indices, fallback = _indices_for_policy(group, feature_row, policy)
            p5_count, elapsed_ms, available = _elapsed_for_indices(group, indices)
            evaluates_winner = winner >= 0 and winner in set(indices)
            saving_pct = (
                (full_elapsed - elapsed_ms) / full_elapsed * 100.0
                if math.isfinite(full_elapsed) and full_elapsed > 0.0 and math.isfinite(elapsed_ms)
                else math.nan
            )
            rows.append(
                {
                    **key_dict,
                    "policy": policy,
                    "evaluated_candidate_indices": ",".join(str(index) for index in indices),
                    "p5_evaluated": int(p5_count),
                    "available": bool(available),
                    "fallback_triggered": bool(fallback),
                    "full_p5_winner_candidate_index": winner,
                    "evaluates_full_p5_winner": bool(evaluates_winner),
                    "policy_missed_full_p5_winner": not bool(evaluates_winner),
                    "p1_top2_miss": bool(p1_top2_miss),
                    "rescued_p1_top2_miss": bool(p1_top2_miss and evaluates_winner),
                    "false_fallback": bool((not p1_top2_miss) and fallback),
                    "total_elapsed_ms": elapsed_ms,
                    "full_top3_p5_total_elapsed_ms": full_elapsed,
                    "elapsed_saving_pct_vs_full_top3_p5": saving_pct,
                    "extra_p5_final_target_conclusion_changed": feature_row.get(
                        "extra_p5_final_target_conclusion_changed",
                        "",
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values([*group_cols, "policy"])

def build_policy_scorecard(policy_rows: pd.DataFrame) -> pd.DataFrame:
    if policy_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    base_miss_count = int(
        policy_rows.drop_duplicates([*GROUP_COLUMNS])["p1_top2_miss"].map(_truthy).sum()
    )
    for policy, group in policy_rows.groupby("policy", sort=True):
        n_rows = len(group)
        misses = int(group["policy_missed_full_p5_winner"].map(_truthy).sum())
        rescued = int(group["rescued_p1_top2_miss"].map(_truthy).sum())
        non_miss = group[~group["p1_top2_miss"].map(_truthy)]
        false_fallback = int(group["false_fallback"].map(_truthy).sum())
        target_changed = int(
            group["extra_p5_final_target_conclusion_changed"].map(_truthy).sum()
        )
        rows.append(
            {
                "policy": policy,
                "n_rows": int(n_rows),
                "n_p1_top2_miss_rows": int(base_miss_count),
                "n_policy_misses": misses,
                "n_p1_top2_misses_rescued": rescued,
                "miss_recall": rescued / base_miss_count if base_miss_count else math.nan,
                "false_fallback_rate": false_fallback / len(non_miss) if len(non_miss) else 0.0,
                "mean_p5_evaluated": float(group["p5_evaluated"].mean()),
                "mean_elapsed_saving_pct": float(
                    group["elapsed_saving_pct_vs_full_top3_p5"].mean()
                ),
                "n_target_conclusion_changed_source": target_changed,
                "n_extra_p5_final_regressions_proxy": int(
                    (
                        group["policy_missed_full_p5_winner"].map(_truthy)
                        & group["extra_p5_final_target_conclusion_changed"].map(_truthy)
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_policy_misses", "mean_p5_evaluated", "policy"],
        na_position="last",
    )

def _format_float(value: Any, digits: int = 2) -> str:
    out = _finite_float(value, math.nan)
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}"

def write_report(path: Path, features: pd.DataFrame, scorecard: pd.DataFrame) -> None:
    lines = [
        "# Leiden p1 Exception Detector Report",
        "",
        "This report evaluates p1-visible fallback policies for perturbation-aware Leiden candidate screening.",
        "",
        "## Overview",
        "",
        f"- Case rows: {len(features)}",
        f"- p1 top2 miss rows: {int(features['missed_by_p1_top2'].map(_truthy).sum()) if not features.empty else 0}",
        "- `p1` is treated as a cheap fidelity/budget setting, not as the algorithm.",
        "- The algorithmic object remains the perturbation-aware transition before refinement.",
        "",
        "## Policy Scorecard",
        "",
        "| policy | misses | rescued top2 misses | recall | false fallback | mean p5 eval | mean saving % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in scorecard.iterrows():
        lines.append(
            "| {policy} | {misses} | {rescued} | {recall} | {false} | {p5} | {saving} |".format(
                policy=row["policy"],
                misses=int(row["n_policy_misses"]),
                rescued=int(row["n_p1_top2_misses_rescued"]),
                recall=_format_float(row["miss_recall"], 2),
                false=_format_float(row["false_fallback_rate"], 2),
                p5=_format_float(row["mean_p5_evaluated"], 2),
                saving=_format_float(row["mean_elapsed_saving_pct"], 1),
            )
        )
    miss_rows = features[features["missed_by_p1_top2"].map(_truthy)] if not features.empty else pd.DataFrame()
    lines.extend(
        [
            "",
            "## p1 Top2 Miss Rows",
            "",
        ]
    )
    if miss_rows.empty:
        lines.append("No p1 top2 miss rows were present.")
    else:
        lines.extend(
            [
                "| dataset | case | seed | winner | winner p1 rank | top2-top3 gap | structural rescue | budget3 exception |",
                "|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for _, row in miss_rows.iterrows():
            lines.append(
                "| {dataset} | {case} | {seed} | {winner} | {rank} | {gap} | {rescue} | {budget} |".format(
                    dataset=row.get("dataset", ""),
                    case=row.get("case", ""),
                    seed=int(row.get("seed", 0)),
                    winner=int(row.get("full_p5_winner_candidate_index", -1)),
                    rank=_format_float(row.get("full_p5_winner_p1_rank"), 0),
                    gap=_format_float(row.get("p1_top2_top3_gap"), 3),
                    rescue="yes" if _truthy(row.get("structural_rescue_triggered")) else "no",
                    budget="" if row.get("budget3_exception", "") == "" else (
                        "yes" if _truthy(row.get("budget3_exception")) else "no"
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- A low miss count is necessary but not sufficient; the policy must also keep fallback cost below `always_p1_top3`.",
            "- The source target-conclusion column describes the traced monitor branch, not a re-trace of every simulated policy.",
            "- If the detector cannot recover p1-top2 misses without broad fallback, the correct conclusion is that p1 is only a limited cheap default.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def analyze_input_dirs(input_dirs: list[Path], output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_diagnostics: list[pd.DataFrame] = []
    all_features: list[pd.DataFrame] = []
    for input_dir in input_dirs:
        input_dir = input_dir.expanduser().resolve()
        dataset = _dataset_label(input_dir)
        candidates = _read_csv(input_dir / "candidate_level_rows.csv")
        policies = _optional_csv(input_dir / "policy_comparison_rows.csv")
        scorecard = _optional_csv(input_dir / "work_acceleration_monitor_scorecard.csv")
        diagnostics = build_candidate_rank_diagnostics(candidates)
        augmented = augment_policy_rows(policies, diagnostics)
        summary = build_policy_decision_summary(diagnostics, augmented, scorecard)
        diagnostics.insert(0, "dataset", dataset)
        diagnostics.insert(1, "input_dir", str(input_dir))
        summary.insert(0, "dataset", dataset)
        summary.insert(1, "input_dir", str(input_dir))
        scorecard = scorecard.copy()
        if not scorecard.empty:
            scorecard.insert(0, "dataset", dataset)
        features = build_exception_feature_rows(diagnostics, summary, scorecard)
        features.insert(1, "input_dir", str(input_dir))
        all_diagnostics.append(diagnostics)
        all_features.append(features)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True, sort=False) if all_diagnostics else pd.DataFrame()
    features = pd.concat(all_features, ignore_index=True, sort=False) if all_features else pd.DataFrame()
    policy_rows = build_policy_evaluation_rows(diagnostics, features)
    scorecard = build_policy_scorecard(policy_rows)

    feature_path = output_dir / "p1_exception_feature_rows.csv"
    policy_path = output_dir / "p1_exception_policy_rows.csv"
    scorecard_path = output_dir / "p1_exception_policy_scorecard.csv"
    report_path = output_dir / "p1_exception_detector_report.md"
    features.to_csv(feature_path, index=False)
    policy_rows.to_csv(policy_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    write_report(report_path, features, scorecard)
    return {
        "feature_rows": feature_path,
        "policy_rows": policy_path,
        "policy_scorecard": scorecard_path,
        "report": report_path,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    paths = analyze_input_dirs(_parse_input_dirs(args.input_dirs), args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")

if __name__ == "__main__":
    main()
