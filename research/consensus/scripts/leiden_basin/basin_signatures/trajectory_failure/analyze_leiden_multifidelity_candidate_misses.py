#!/usr/bin/env python3
"""Attribute Leiden multi-fidelity candidate ranking misses.

This is an analysis-only postprocessor for
``run_leiden_hysteresis_work_acceleration_monitor.py`` outputs. It keeps the
production Leiden path untouched and answers whether p1 prescreen rankings
missed the full p5 winner, whether top2/top3 retry would recover it, and what
candidate-level structure is visible in miss cases.
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

DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_multifidelity_label_field30_20260513"
)

CANDIDATE_GROUP_COLUMNS = ("case", "seed", "candidate_budget", "candidate_eval_mode")
TARGET_POLICIES = ("extra_p5_final", "baseline_plus_25ppm")
P1_POLICIES = ("p1_top1_then_p5", "p1_top2_then_p5", "p1_top3_then_p5")
STRUCTURAL_RESCUE_POLICY = "p1_top2_plus_best_low_target_rescue1"
STRUCTURAL_RESCUE_TARGET_WEIGHT_PER_NODE_MAX = 0.5
STRUCTURE_COLUMNS = (
    "group_count",
    "group_weight",
    "group_fraction",
    "group_to_target_weight",
    "group_cut_weight",
    "target_weight_per_node",
    "priority",
    "best_group_delta_q",
    "group_kind",
)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)

def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CANDIDATE_GROUP_COLUMNS if column in frame.columns]

def _rank_order(group: pd.DataFrame, metric: str) -> list[Any]:
    def key(item: tuple[Any, pd.Series]) -> tuple[int, float, int]:
        index, row = item
        value = _finite_float(row.get(metric), math.nan)
        candidate_index = int(row.get("candidate_index", 0))
        if math.isfinite(value):
            return (0, -value, candidate_index)
        return (1, 0.0, candidate_index)

    return [index for index, _ in sorted(group.iterrows(), key=key)]

def _assign_rank(frame: pd.DataFrame, metric: str, output_column: str) -> None:
    frame[output_column] = math.nan
    for _, group in frame.groupby(_group_columns(frame), dropna=False, sort=False):
        rank = 1
        for index in _rank_order(group, metric):
            value = _finite_float(frame.at[index, metric], math.nan)
            if not math.isfinite(value):
                continue
            frame.at[index, output_column] = rank
            rank += 1

def _target_weight_per_node(row: pd.Series | dict[str, Any]) -> float:
    group_count = _finite_float(row.get("group_count"), math.nan)
    target_weight = _finite_float(row.get("group_to_target_weight"), math.nan)
    if not math.isfinite(group_count) or group_count <= 0.0:
        return math.nan
    if not math.isfinite(target_weight):
        return math.nan
    return target_weight / group_count

def build_candidate_rank_diagnostics(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    diagnostics = candidate_rows.copy()
    if diagnostics.empty:
        return diagnostics
    required = {"candidate_index", "p1_delta_q", "p5_delta_q"}
    missing = required - set(diagnostics.columns)
    if missing:
        raise ValueError(f"candidate rows missing required columns: {sorted(missing)}")

    _assign_rank(diagnostics, "p1_delta_q", "p1_rank")
    _assign_rank(diagnostics, "p5_delta_q", "p5_rank")
    diagnostics["rank_gap"] = diagnostics["p1_rank"] - diagnostics["p5_rank"]
    diagnostics["p1_p5_delta_gap"] = diagnostics["p5_delta_q"] - diagnostics["p1_delta_q"]
    diagnostics["target_weight_per_node"] = diagnostics.apply(
        _target_weight_per_node,
        axis=1,
    )
    diagnostics["is_full_p5_winner"] = diagnostics["p5_rank"].eq(1)
    for top_n in (1, 2, 3):
        diagnostics[f"is_p1_top{top_n}"] = diagnostics["p1_rank"].le(top_n)
    diagnostics["missed_by_p1_top1"] = (
        diagnostics["is_full_p5_winner"] & ~diagnostics["is_p1_top1"]
    )
    diagnostics["missed_by_p1_top2"] = (
        diagnostics["is_full_p5_winner"] & ~diagnostics["is_p1_top2"]
    )

    preferred = [
        *CANDIDATE_GROUP_COLUMNS,
        "selected_policy",
        "candidate_index",
        "source_cluster",
        "target_cluster",
        "p1_rank",
        "p5_rank",
        "rank_gap",
        "p1_delta_q",
        "p5_delta_q",
        "p1_p5_delta_gap",
        "is_full_p5_winner",
        "is_p1_top1",
        "is_p1_top2",
        "is_p1_top3",
        "missed_by_p1_top1",
        "missed_by_p1_top2",
        *STRUCTURE_COLUMNS,
    ]
    ordered = [column for column in preferred if column in diagnostics.columns]
    remainder = [column for column in diagnostics.columns if column not in ordered]
    return diagnostics[ordered + remainder].sort_values(
        [column for column in (*CANDIDATE_GROUP_COLUMNS, "p5_rank", "p1_rank") if column in diagnostics.columns],
        na_position="last",
    )

def _candidate_baseline_quality(group: pd.DataFrame) -> float:
    for _, row in group.iterrows():
        p5_quality = _finite_float(row.get("p5_quality"), math.nan)
        p5_delta = _finite_float(row.get("p5_delta_q"), math.nan)
        if math.isfinite(p5_quality) and math.isfinite(p5_delta):
            return p5_quality - p5_delta
        p1_quality = _finite_float(row.get("p1_quality"), math.nan)
        p1_delta = _finite_float(row.get("p1_delta_q"), math.nan)
        if math.isfinite(p1_quality) and math.isfinite(p1_delta):
            return p1_quality - p1_delta
    return math.nan

def _synthesize_policy_row(
    *,
    group: pd.DataFrame,
    policy: str,
    top_n: int,
) -> dict[str, Any]:
    p1_order = _rank_order(group, "p1_delta_q")
    selected_indices = p1_order[:top_n]
    selected_rows = group.loc[selected_indices]
    p5_rows = selected_rows[selected_rows["p5_delta_q"].map(lambda value: math.isfinite(_finite_float(value)))]
    available = not selected_rows.empty and len(p5_rows) == len(selected_rows)
    p1_elapsed_ms = sum(
        _finite_float(value, 0.0)
        for value in group.get("p1_elapsed_ms", pd.Series(dtype=float))
        if math.isfinite(_finite_float(value, math.nan))
    )
    p5_elapsed_ms = sum(
        _finite_float(value, 0.0)
        for value in p5_rows.get("p5_elapsed_ms", pd.Series(dtype=float))
        if math.isfinite(_finite_float(value, math.nan))
    )
    selected_candidate_index = -1
    final_delta_q = math.nan
    quality = math.nan
    accepted = False
    matches_full_p5 = False
    if available:
        winner = max(
            p5_rows.to_dict("records"),
            key=lambda row: (
                _finite_float(row.get("p5_delta_q"), -math.inf),
                -int(row.get("candidate_index", 0)),
            ),
        )
        selected_candidate_index = int(winner.get("candidate_index", -1))
        final_delta_q = _finite_float(winner.get("p5_delta_q"), math.nan)
        quality = _finite_float(winner.get("p5_quality"), math.nan)
        baseline_quality = _candidate_baseline_quality(group)
        if math.isfinite(quality) and math.isfinite(baseline_quality):
            accepted = quality >= baseline_quality
        elif math.isfinite(final_delta_q):
            accepted = final_delta_q >= 0.0
        full_winners = group[group["is_full_p5_winner"]]
        if not full_winners.empty:
            matches_full_p5 = selected_candidate_index == int(
                full_winners.iloc[0]["candidate_index"]
            )
    base: dict[str, Any] = {}
    for column in (*CANDIDATE_GROUP_COLUMNS, "max_group_candidates", "selected_policy", "label_full_p5"):
        if column in group.columns:
            base[column] = group.iloc[0][column]
    return {
        **base,
        "policy": policy,
        "selected_candidate_index": selected_candidate_index,
        "candidate_count": len(group),
        "p1_evaluated": len(group),
        "p5_evaluated": len(p5_rows),
        "p1_elapsed_ms": p1_elapsed_ms,
        "p5_elapsed_ms": p5_elapsed_ms,
        "total_elapsed_ms": p1_elapsed_ms + p5_elapsed_ms,
        "final_delta_q": final_delta_q,
        "quality": quality,
        "accepted": accepted,
        "available": available,
        "matches_full_p5": matches_full_p5,
        "synthesized": True,
    }

def augment_policy_rows(policy_rows: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    policy = policy_rows.copy()
    if not policy.empty:
        policy["synthesized"] = False
    synthesized: list[dict[str, Any]] = []
    group_cols = _group_columns(diagnostics)
    if diagnostics.empty:
        return policy
    for key, group in diagnostics.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        existing = policy
        for column, value in zip(group_cols, key_values, strict=False):
            if column in existing.columns:
                existing = existing[existing[column] == value]
        present = set(existing["policy"].astype(str)) if "policy" in existing else set()
        for top_n, policy_name in enumerate(P1_POLICIES, start=1):
            if policy_name not in present:
                synthesized.append(
                    _synthesize_policy_row(group=group, policy=policy_name, top_n=top_n)
                )
    if synthesized:
        policy = pd.concat([policy, pd.DataFrame(synthesized)], ignore_index=True, sort=False)
    return policy

def _policy_row_from_candidate_indices(
    *,
    group: pd.DataFrame,
    policy: str,
    selected_indices: list[Any],
    rescue_row: pd.Series | None = None,
) -> dict[str, Any]:
    selected_rows = group.loc[selected_indices]
    p5_rows = selected_rows[
        selected_rows["p5_delta_q"].map(lambda value: math.isfinite(_finite_float(value)))
    ]
    available = not selected_rows.empty and len(p5_rows) == len(selected_rows)
    p1_elapsed_ms = sum(
        _finite_float(value, 0.0)
        for value in group.get("p1_elapsed_ms", pd.Series(dtype=float))
        if math.isfinite(_finite_float(value, math.nan))
    )
    p5_elapsed_ms = sum(
        _finite_float(value, 0.0)
        for value in p5_rows.get("p5_elapsed_ms", pd.Series(dtype=float))
        if math.isfinite(_finite_float(value, math.nan))
    )
    selected_candidate_index = -1
    final_delta_q = math.nan
    quality = math.nan
    accepted = False
    matches_full_p5 = False
    if available:
        winner = max(
            p5_rows.to_dict("records"),
            key=lambda row: (
                _finite_float(row.get("p5_delta_q"), -math.inf),
                -int(row.get("candidate_index", 0)),
            ),
        )
        selected_candidate_index = int(winner.get("candidate_index", -1))
        final_delta_q = _finite_float(winner.get("p5_delta_q"), math.nan)
        quality = _finite_float(winner.get("p5_quality"), math.nan)
        baseline_quality = _candidate_baseline_quality(group)
        if math.isfinite(quality) and math.isfinite(baseline_quality):
            accepted = quality >= baseline_quality
        elif math.isfinite(final_delta_q):
            accepted = final_delta_q >= 0.0
        full_winners = group[group["is_full_p5_winner"]]
        if not full_winners.empty:
            matches_full_p5 = selected_candidate_index == int(
                full_winners.iloc[0]["candidate_index"]
            )

    base: dict[str, Any] = {}
    for column in (*CANDIDATE_GROUP_COLUMNS, "max_group_candidates", "selected_policy", "label_full_p5"):
        if column in group.columns:
            base[column] = group.iloc[0][column]
    rescue_candidate_index = -1 if rescue_row is None else int(rescue_row.get("candidate_index", -1))
    rescue_target_weight_per_node = (
        math.nan
        if rescue_row is None
        else _finite_float(rescue_row.get("target_weight_per_node"), math.nan)
    )
    return {
        **base,
        "policy": policy,
        "selected_candidate_index": selected_candidate_index,
        "candidate_count": len(group),
        "p1_evaluated": len(group),
        "p5_evaluated": len(p5_rows),
        "p1_elapsed_ms": p1_elapsed_ms,
        "p5_elapsed_ms": p5_elapsed_ms,
        "total_elapsed_ms": p1_elapsed_ms + p5_elapsed_ms,
        "final_delta_q": final_delta_q,
        "quality": quality,
        "accepted": accepted,
        "available": available,
        "matches_full_p5": matches_full_p5,
        "synthesized": True,
        "rescue_selected": rescue_row is not None,
        "rescue_candidate_index": rescue_candidate_index,
        "rescue_candidate_p1_rank": math.nan
        if rescue_row is None
        else _finite_float(rescue_row.get("p1_rank"), math.nan),
        "rescue_candidate_p5_rank": math.nan
        if rescue_row is None
        else _finite_float(rescue_row.get("p5_rank"), math.nan),
        "rescue_candidate_is_full_p5_winner": False
        if rescue_row is None
        else _truthy(rescue_row.get("is_full_p5_winner")),
        "rescue_candidate_group_kind": "" if rescue_row is None else rescue_row.get("group_kind", ""),
        "rescue_candidate_group_count": math.nan
        if rescue_row is None
        else _finite_float(rescue_row.get("group_count"), math.nan),
        "rescue_candidate_target_weight_per_node": rescue_target_weight_per_node,
        "rescue_candidate_group_to_target_weight": math.nan
        if rescue_row is None
        else _finite_float(rescue_row.get("group_to_target_weight"), math.nan),
        "rescue_candidate_group_cut_weight": math.nan
        if rescue_row is None
        else _finite_float(rescue_row.get("group_cut_weight"), math.nan),
        "finalist_candidate_indices": ",".join(
            str(int(row.get("candidate_index", -1))) for _, row in selected_rows.iterrows()
        ),
    }

def _eligible_structural_rescue_rows(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return group.copy()
    rows = group.copy()
    rows["target_weight_per_node"] = rows.apply(_target_weight_per_node, axis=1)
    return rows[
        rows["p1_rank"].gt(2)
        & rows["group_kind"].astype(str).eq("best")
        & rows["group_count"].map(lambda value: _finite_float(value, math.nan) >= 2.0)
        & rows["target_weight_per_node"].map(
            lambda value: _finite_float(value, math.nan)
            <= STRUCTURAL_RESCUE_TARGET_WEIGHT_PER_NODE_MAX
        )
    ].copy()

def _select_structural_rescue_row(group: pd.DataFrame) -> pd.Series | None:
    eligible = _eligible_structural_rescue_rows(group)
    if eligible.empty:
        return None
    eligible = eligible.assign(
        _target_weight_per_node=eligible["target_weight_per_node"].map(
            lambda value: _finite_float(value, math.inf)
        ),
        _group_cut_weight=eligible["group_cut_weight"].map(
            lambda value: _finite_float(value, math.inf)
        ),
        _p1_rank=eligible["p1_rank"].map(lambda value: _finite_float(value, math.inf)),
        _candidate_index=eligible["candidate_index"].map(
            lambda value: int(_finite_float(value, 0.0))
        ),
    )
    return eligible.sort_values(
        ["_target_weight_per_node", "_group_cut_weight", "_p1_rank", "_candidate_index"],
        kind="mergesort",
    ).iloc[0]

def build_structural_rescue_policy_rows(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = _group_columns(diagnostics)
    for _, group in diagnostics.groupby(group_cols, dropna=False, sort=False):
        p1_order = _rank_order(group, "p1_delta_q")
        selected_indices = list(p1_order[:2])
        rescue_row = _select_structural_rescue_row(group)
        if rescue_row is not None and rescue_row.name not in selected_indices:
            selected_indices.append(rescue_row.name)
        rows.append(
            _policy_row_from_candidate_indices(
                group=group,
                policy=STRUCTURAL_RESCUE_POLICY,
                selected_indices=selected_indices,
                rescue_row=rescue_row,
            )
        )
    return pd.DataFrame(rows)

def _matching_rows(frame: pd.DataFrame, key: dict[str, Any]) -> pd.DataFrame:
    rows = frame
    for column, value in key.items():
        if column in rows.columns:
            rows = rows[rows[column] == value]
    return rows

def _policy_row(policy_rows: pd.DataFrame, policy: str) -> pd.Series | None:
    rows = policy_rows[policy_rows["policy"].astype(str) == policy] if "policy" in policy_rows else pd.DataFrame()
    if rows.empty:
        return None
    return rows.iloc[0]

def _selected_candidate(row: pd.Series | None) -> int | None:
    if row is None:
        return None
    value = _finite_float(row.get("selected_candidate_index"), math.nan)
    if not math.isfinite(value) or value < 0:
        return None
    return int(value)

def _elapsed_saving(full_row: pd.Series | None, policy_row: pd.Series | None) -> tuple[float, float]:
    if full_row is None or policy_row is None:
        return math.nan, math.nan
    full_elapsed = _finite_float(full_row.get("total_elapsed_ms"), math.nan)
    policy_elapsed = _finite_float(policy_row.get("total_elapsed_ms"), math.nan)
    if not math.isfinite(full_elapsed) or not math.isfinite(policy_elapsed) or full_elapsed <= 0.0:
        return math.nan, math.nan
    saving = full_elapsed - policy_elapsed
    return saving, saving / full_elapsed * 100.0

def _target_summary(scorecard: pd.DataFrame, key: dict[str, Any], target_policy: str) -> dict[str, Any]:
    if scorecard.empty or "target_policy" not in scorecard:
        return {
            f"{target_policy}_extra_tau_status": "",
            f"{target_policy}_perturb_tau_status": "",
            f"{target_policy}_target_conclusion_changed": "",
            f"{target_policy}_k_work_saving_pct": math.nan,
            f"{target_policy}_net_elapsed_saving_pct": math.nan,
        }
    rows = _matching_rows(scorecard, {k: v for k, v in key.items() if k != "candidate_eval_mode"})
    rows = rows[rows["target_policy"].astype(str) == target_policy]
    if rows.empty:
        return {
            f"{target_policy}_extra_tau_status": "",
            f"{target_policy}_perturb_tau_status": "",
            f"{target_policy}_target_conclusion_changed": "",
            f"{target_policy}_k_work_saving_pct": math.nan,
            f"{target_policy}_net_elapsed_saving_pct": math.nan,
        }
    row = rows.iloc[0]
    extra_status = str(row.get("extra_tau_status", ""))
    perturb_status = str(row.get("perturb_tau_status", ""))
    return {
        f"{target_policy}_extra_tau_status": extra_status,
        f"{target_policy}_perturb_tau_status": perturb_status,
        f"{target_policy}_target_conclusion_changed": extra_status != perturb_status,
        f"{target_policy}_k_work_saving_pct": _finite_float(
            row.get("k_work_saving_pct"), math.nan
        ),
        f"{target_policy}_net_elapsed_saving_pct": _finite_float(
            row.get("net_elapsed_saving_pct"), math.nan
        ),
    }

def build_policy_decision_summary(
    diagnostics: pd.DataFrame,
    policy_rows: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = _group_columns(diagnostics)
    for key, group in diagnostics.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key_values, strict=False))
        winner_rows = group[group["is_full_p5_winner"]]
        winner = winner_rows.iloc[0] if not winner_rows.empty else None
        policies = _matching_rows(policy_rows, key_dict)
        full_row = _policy_row(policies, "full_top3_p5")
        base: dict[str, Any] = {
            **key_dict,
            "candidate_count": len(group),
            "full_p5_winner_candidate_index": _selected_candidate(full_row)
            if full_row is not None
            else (None if winner is None else int(winner["candidate_index"])),
            "full_p5_winner_p1_rank": math.nan if winner is None else winner["p1_rank"],
            "full_p5_winner_p5_rank": math.nan if winner is None else winner["p5_rank"],
            "full_p5_winner_p1_delta_q": math.nan if winner is None else winner["p1_delta_q"],
            "full_p5_winner_p5_delta_q": math.nan if winner is None else winner["p5_delta_q"],
            "full_p5_winner_p1_p5_delta_gap": math.nan
            if winner is None
            else winner["p1_p5_delta_gap"],
            "full_p5_winner_in_p1_top1": False if winner is None else bool(winner["is_p1_top1"]),
            "full_p5_winner_in_p1_top2": False if winner is None else bool(winner["is_p1_top2"]),
            "full_p5_winner_in_p1_top3": False if winner is None else bool(winner["is_p1_top3"]),
            "missed_by_p1_top1": False if winner is None else bool(winner["missed_by_p1_top1"]),
            "missed_by_p1_top2": False if winner is None else bool(winner["missed_by_p1_top2"]),
            "full_top3_p5_selected_candidate_index": _selected_candidate(full_row),
            "full_top3_p5_accepted": "" if full_row is None else _truthy(full_row.get("accepted")),
            "full_top3_p5_available": "" if full_row is None else _truthy(full_row.get("available")),
            "full_top3_p5_total_elapsed_ms": math.nan
            if full_row is None
            else _finite_float(full_row.get("total_elapsed_ms"), math.nan),
        }
        if winner is not None:
            for column in STRUCTURE_COLUMNS:
                if column in winner:
                    base[f"full_p5_winner_{column}"] = winner[column]
        full_selected = _selected_candidate(full_row)
        full_accepted = None if full_row is None else _truthy(full_row.get("accepted"))
        for policy_name in P1_POLICIES:
            row = _policy_row(policies, policy_name)
            selected = _selected_candidate(row)
            accepted = None if row is None else _truthy(row.get("accepted"))
            saving_ms, saving_pct = _elapsed_saving(full_row, row)
            same_selected = selected is not None and full_selected is not None and selected == full_selected
            same_accepted = accepted is not None and full_accepted is not None and accepted == full_accepted
            base.update(
                {
                    f"{policy_name}_available": "" if row is None else _truthy(row.get("available")),
                    f"{policy_name}_synthesized": "" if row is None else _truthy(row.get("synthesized")),
                    f"{policy_name}_selected_candidate_index": selected,
                    f"{policy_name}_accepted": "" if accepted is None else accepted,
                    f"{policy_name}_matches_full_p5": ""
                    if row is None
                    else _truthy(row.get("matches_full_p5")),
                    f"{policy_name}_same_selected_candidate": same_selected,
                    f"{policy_name}_same_pass_conclusion": same_accepted,
                    f"{policy_name}_same_decision_as_full_top3_p5": same_selected and same_accepted,
                    f"{policy_name}_total_elapsed_ms": math.nan
                    if row is None
                    else _finite_float(row.get("total_elapsed_ms"), math.nan),
                    f"{policy_name}_elapsed_saving_ms_vs_full_top3_p5": saving_ms,
                    f"{policy_name}_elapsed_saving_pct_vs_full_top3_p5": saving_pct,
                }
            )
        for target_policy in TARGET_POLICIES:
            base.update(_target_summary(scorecard, key_dict, target_policy))
        rows.append(base)
    return pd.DataFrame(rows).sort_values(group_cols)

def build_structural_rescue_summary(
    diagnostics: pd.DataFrame,
    augmented_policy_rows: pd.DataFrame,
    structural_policy_rows: pd.DataFrame,
) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = _group_columns(diagnostics)
    for key, group in diagnostics.groupby(group_cols, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key_values, strict=False))
        winner_rows = group[group["is_full_p5_winner"]]
        winner = winner_rows.iloc[0] if not winner_rows.empty else None
        policies = _matching_rows(augmented_policy_rows, key_dict)
        rescue_policies = _matching_rows(structural_policy_rows, key_dict)
        full_row = _policy_row(policies, "full_top3_p5")
        top2_row = _policy_row(policies, "p1_top2_then_p5")
        rescue_row = _policy_row(rescue_policies, STRUCTURAL_RESCUE_POLICY)
        full_selected = _selected_candidate(full_row)
        rescue_selected = _selected_candidate(rescue_row)
        full_accepted = None if full_row is None else _truthy(full_row.get("accepted"))
        rescue_accepted = None if rescue_row is None else _truthy(rescue_row.get("accepted"))
        top2_saving_ms, top2_saving_pct = _elapsed_saving(full_row, top2_row)
        rescue_saving_ms, rescue_saving_pct = _elapsed_saving(full_row, rescue_row)
        same_selected = (
            rescue_selected is not None
            and full_selected is not None
            and rescue_selected == full_selected
        )
        same_accepted = (
            rescue_accepted is not None
            and full_accepted is not None
            and rescue_accepted == full_accepted
        )
        base: dict[str, Any] = {
            **key_dict,
            "policy": STRUCTURAL_RESCUE_POLICY,
            "candidate_count": len(group),
            "full_p5_winner_candidate_index": _selected_candidate(full_row)
            if full_row is not None
            else (None if winner is None else int(winner["candidate_index"])),
            "full_p5_winner_p1_rank": math.nan if winner is None else winner["p1_rank"],
            "full_p5_winner_p5_rank": math.nan if winner is None else winner["p5_rank"],
            "missed_by_p1_top2": False if winner is None else bool(winner["missed_by_p1_top2"]),
            "p1_top2_selected_candidate_index": _selected_candidate(top2_row),
            "p1_top2_same_decision_as_full_top3_p5": False
            if top2_row is None
            else _truthy(top2_row.get("matches_full_p5"))
            and (
                full_accepted is None
                or _truthy(top2_row.get("accepted")) == full_accepted
            ),
            "p1_top2_elapsed_saving_ms_vs_full_top3_p5": top2_saving_ms,
            "p1_top2_elapsed_saving_pct_vs_full_top3_p5": top2_saving_pct,
            "structural_rescue_available": ""
            if rescue_row is None
            else _truthy(rescue_row.get("available")),
            "structural_rescue_selected": ""
            if rescue_row is None
            else _truthy(rescue_row.get("rescue_selected")),
            "structural_rescue_candidate_index": None
            if rescue_row is None
            else _selected_candidate(pd.Series({"selected_candidate_index": rescue_row.get("rescue_candidate_index")})),
            "structural_rescue_candidate_is_full_p5_winner": ""
            if rescue_row is None
            else _truthy(rescue_row.get("rescue_candidate_is_full_p5_winner")),
            "structural_rescue_candidate_p1_rank": math.nan
            if rescue_row is None
            else _finite_float(rescue_row.get("rescue_candidate_p1_rank"), math.nan),
            "structural_rescue_candidate_p5_rank": math.nan
            if rescue_row is None
            else _finite_float(rescue_row.get("rescue_candidate_p5_rank"), math.nan),
            "structural_rescue_candidate_group_kind": ""
            if rescue_row is None
            else rescue_row.get("rescue_candidate_group_kind", ""),
            "structural_rescue_candidate_group_count": math.nan
            if rescue_row is None
            else _finite_float(rescue_row.get("rescue_candidate_group_count"), math.nan),
            "structural_rescue_candidate_target_weight_per_node": math.nan
            if rescue_row is None
            else _finite_float(
                rescue_row.get("rescue_candidate_target_weight_per_node"), math.nan
            ),
            "structural_rescue_candidate_group_to_target_weight": math.nan
            if rescue_row is None
            else _finite_float(
                rescue_row.get("rescue_candidate_group_to_target_weight"), math.nan
            ),
            "structural_rescue_candidate_group_cut_weight": math.nan
            if rescue_row is None
            else _finite_float(rescue_row.get("rescue_candidate_group_cut_weight"), math.nan),
            "structural_rescue_finalist_candidate_indices": ""
            if rescue_row is None
            else rescue_row.get("finalist_candidate_indices", ""),
            "structural_rescue_selected_candidate_index": rescue_selected,
            "structural_rescue_accepted": "" if rescue_accepted is None else rescue_accepted,
            "structural_rescue_matches_full_p5": ""
            if rescue_row is None
            else _truthy(rescue_row.get("matches_full_p5")),
            "structural_rescue_same_selected_candidate": same_selected,
            "structural_rescue_same_pass_conclusion": same_accepted,
            "structural_rescue_same_decision_as_full_top3_p5": same_selected
            and same_accepted,
            "structural_rescue_total_elapsed_ms": math.nan
            if rescue_row is None
            else _finite_float(rescue_row.get("total_elapsed_ms"), math.nan),
            "full_top3_p5_total_elapsed_ms": math.nan
            if full_row is None
            else _finite_float(full_row.get("total_elapsed_ms"), math.nan),
            "structural_rescue_elapsed_saving_ms_vs_full_top3_p5": rescue_saving_ms,
            "structural_rescue_elapsed_saving_pct_vs_full_top3_p5": rescue_saving_pct,
        }
        rows.append(base)
    return pd.DataFrame(rows).sort_values(group_cols)

def _format_bool(value: Any) -> str:
    if value == "":
        return ""
    return "yes" if _truthy(value) else "no"

def _format_float(value: Any, digits: int = 2) -> str:
    out = _finite_float(value, math.nan)
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}"

def write_report(path: Path, diagnostics: pd.DataFrame, summary: pd.DataFrame) -> None:
    miss_rows = diagnostics[diagnostics["missed_by_p1_top2"]].copy()
    lines = [
        "# Leiden Multi-Fidelity Candidate Miss Report",
        "",
        "This report attributes full p5 winners that were missed by the p1 prescreen ranking.",
        "",
        "## Overview",
        "",
        f"- Candidate rows: {len(diagnostics)}",
        f"- Case rows: {len(summary)}",
        f"- Full p5 winners missed by p1 top1: {int(diagnostics['missed_by_p1_top1'].sum()) if 'missed_by_p1_top1' in diagnostics else 0}",
        f"- Full p5 winners missed by p1 top2: {int(diagnostics['missed_by_p1_top2'].sum()) if 'missed_by_p1_top2' in diagnostics else 0}",
        "",
        "## p1 Top2 Miss Cases",
        "",
    ]
    if miss_rows.empty:
        lines.append("No full p5 winner was missed by p1 top2.")
    else:
        lines.extend(
            [
                "| case | seed | budget | winner | p1 rank | p5 delta | p1 delta | p5-p1 gap | group | kind | target weight | cut weight |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        for _, row in miss_rows.iterrows():
            lines.append(
                "| {case} | {seed} | {budget} | {winner} | {p1_rank} | {p5_delta} | {p1_delta} | {gap} | {group} | {kind} | {target} | {cut} |".format(
                    case=row.get("case", ""),
                    seed=int(row.get("seed", 0)),
                    budget=int(row.get("candidate_budget", 0)),
                    winner=int(row.get("candidate_index", -1)),
                    p1_rank=int(row.get("p1_rank", 0)),
                    p5_delta=_format_float(row.get("p5_delta_q"), 3),
                    p1_delta=_format_float(row.get("p1_delta_q"), 3),
                    gap=_format_float(row.get("p1_p5_delta_gap"), 3),
                    group=int(row.get("group_count", 0)) if pd.notna(row.get("group_count", math.nan)) else "",
                    kind=row.get("group_kind", ""),
                    target=_format_float(row.get("group_to_target_weight"), 3),
                    cut=_format_float(row.get("group_cut_weight"), 3),
                )
            )
    lines.extend(
        [
            "",
            "## Policy Decision Summary",
            "",
            "| case | seed | budget | winner p1 rank | top1 same | top2 same | top3 same | top1 saving % | top2 saving % | top3 saving % | extra target changed | 25ppm target changed |",
            "|---|---:|---:|---:|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            "| {case} | {seed} | {budget} | {rank} | {top1} | {top2} | {top3} | {save1} | {save2} | {save3} | {extra} | {fixed} |".format(
                case=row.get("case", ""),
                seed=int(row.get("seed", 0)),
                budget=int(row.get("candidate_budget", 0)),
                rank=_format_float(row.get("full_p5_winner_p1_rank"), 0),
                top1=_format_bool(row.get("p1_top1_then_p5_same_decision_as_full_top3_p5")),
                top2=_format_bool(row.get("p1_top2_then_p5_same_decision_as_full_top3_p5")),
                top3=_format_bool(row.get("p1_top3_then_p5_same_decision_as_full_top3_p5")),
                save1=_format_float(row.get("p1_top1_then_p5_elapsed_saving_pct_vs_full_top3_p5"), 1),
                save2=_format_float(row.get("p1_top2_then_p5_elapsed_saving_pct_vs_full_top3_p5"), 1),
                save3=_format_float(row.get("p1_top3_then_p5_elapsed_saving_pct_vs_full_top3_p5"), 1),
                extra=_format_bool(row.get("extra_p5_final_target_conclusion_changed")),
                fixed=_format_bool(row.get("baseline_plus_25ppm_target_conclusion_changed")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `p1_top3_then_p5` is synthesized when older monitor outputs omit it; the synthesized cost includes all p1 prescreen work plus p5 work for the p1 top3 finalists.",
            "- Target conclusion columns come from `work_acceleration_monitor_scorecard.csv` and describe the actually traced perturb branch, not a re-trace of every synthetic policy.",
            "- A p1 top3 policy can have negative elapsed saving when the candidate budget is already three, because it pays the p1 prescreen cost and then still evaluates all p5 labels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_structural_rescue_report(path: Path, rescue_summary: pd.DataFrame) -> None:
    if rescue_summary.empty:
        path.write_text(
            "# Leiden Multi-Fidelity Structural Rescue Report\n\nNo rescue rows were generated.\n",
            encoding="utf-8",
        )
        return
    missed = rescue_summary[rescue_summary["missed_by_p1_top2"].map(_truthy)]
    regressions = rescue_summary[
        ~rescue_summary["missed_by_p1_top2"].map(_truthy)
        & ~rescue_summary["structural_rescue_same_decision_as_full_top3_p5"].map(_truthy)
    ]
    rescued_misses = missed[
        missed["structural_rescue_same_decision_as_full_top3_p5"].map(_truthy)
    ]
    lines = [
        "# Leiden Multi-Fidelity Structural Rescue Report",
        "",
        "Analysis-only simulation of `p1_top2 + structural rescue 1` using full p5 label artifacts.",
        "",
        "## Overview",
        "",
        f"- Policy: `{STRUCTURAL_RESCUE_POLICY}`",
        "- Eligibility: p1 rank outside top2, `group_kind == best`, `group_count >= 2`, and `group_to_target_weight / group_count <= 0.5`.",
        f"- Case rows: {len(rescue_summary)}",
        f"- p1 top2 miss rows: {len(missed)}",
        f"- p1 top2 miss rows rescued: {len(rescued_misses)}",
        f"- Non-miss rows regressed: {len(regressions)}",
        "",
        "## Case Summary",
        "",
        "| case | seed | budget | winner p1 rank | rescue candidate | rescued full winner | same full decision | saving % | top2 same full | top2 saving % | finalists |",
        "|---|---:|---:|---:|---:|---|---|---:|---|---:|---|",
    ]
    for _, row in rescue_summary.iterrows():
        rescue_candidate = row.get("structural_rescue_candidate_index")
        rescue_candidate_text = (
            ""
            if pd.isna(rescue_candidate)
            else str(int(_finite_float(rescue_candidate, -1.0)))
        )
        lines.append(
            "| {case} | {seed} | {budget} | {rank} | {rescue_candidate} | {rescued_winner} | {same} | {saving} | {top2_same} | {top2_saving} | {finalists} |".format(
                case=row.get("case", ""),
                seed=int(row.get("seed", 0)),
                budget=int(row.get("candidate_budget", 0)),
                rank=_format_float(row.get("full_p5_winner_p1_rank"), 0),
                rescue_candidate=rescue_candidate_text,
                rescued_winner=_format_bool(
                    row.get("structural_rescue_candidate_is_full_p5_winner")
                ),
                same=_format_bool(
                    row.get("structural_rescue_same_decision_as_full_top3_p5")
                ),
                saving=_format_float(
                    row.get("structural_rescue_elapsed_saving_pct_vs_full_top3_p5"),
                    1,
                ),
                top2_same=_format_bool(row.get("p1_top2_same_decision_as_full_top3_p5")),
                top2_saving=_format_float(
                    row.get("p1_top2_elapsed_saving_pct_vs_full_top3_p5"),
                    1,
                ),
                finalists=row.get("structural_rescue_finalist_candidate_indices", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is not an operational policy yet; it relies on label artifacts to score whether the rescued finalist would have matched full p5.",
            "- Positive saving means the simulated p1 top2 plus one rescue finalist was cheaper than full top3 p5 for that case.",
            "- A rescue policy that fixes miss rows but regresses non-miss rows should be rejected or narrowed before field expansion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def analyze_input_dir(input_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = (output_dir or input_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _read_csv(input_dir / "candidate_level_rows.csv")
    policies = _optional_csv(input_dir / "policy_comparison_rows.csv")
    scorecard = _optional_csv(input_dir / "work_acceleration_monitor_scorecard.csv")

    diagnostics = build_candidate_rank_diagnostics(candidates)
    augmented_policies = augment_policy_rows(policies, diagnostics)
    summary = build_policy_decision_summary(diagnostics, augmented_policies, scorecard)
    structural_policy_rows = build_structural_rescue_policy_rows(diagnostics)
    structural_summary = build_structural_rescue_summary(
        diagnostics,
        augmented_policies,
        structural_policy_rows,
    )

    diagnostics_path = output_dir / "multifidelity_candidate_rank_diagnostics.csv"
    summary_path = output_dir / "multifidelity_policy_decision_summary.csv"
    report_path = output_dir / "multifidelity_miss_case_report.md"
    structural_policy_path = output_dir / "multifidelity_structural_rescue_policy_rows.csv"
    structural_summary_path = output_dir / "multifidelity_structural_rescue_summary.csv"
    structural_report_path = output_dir / "multifidelity_structural_rescue_report.md"

    diagnostics.to_csv(diagnostics_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, diagnostics, summary)
    structural_policy_rows.to_csv(structural_policy_path, index=False)
    structural_summary.to_csv(structural_summary_path, index=False)
    write_structural_rescue_report(structural_report_path, structural_summary)
    return {
        "candidate_rank_diagnostics": diagnostics_path,
        "policy_decision_summary": summary_path,
        "miss_case_report": report_path,
        "structural_rescue_policy_rows": structural_policy_path,
        "structural_rescue_summary": structural_summary_path,
        "structural_rescue_report": structural_report_path,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --input-dir.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    paths = analyze_input_dir(args.input_dir, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")

if __name__ == "__main__":
    main()
