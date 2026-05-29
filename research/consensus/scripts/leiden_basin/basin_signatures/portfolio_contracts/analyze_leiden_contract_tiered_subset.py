#!/usr/bin/env python3
"""Tier Dongdaemun accurate-contract obligations and estimate oracle subset size."""

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

TIER_ORDER = ("hard", "core", "diagnostic")
PREFIX_K_VALUES = (1, 3, 5, 10)

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

def _candidate_elapsed_ms(frame: pd.DataFrame) -> pd.Series:
    if "p5_elapsed_ms" in frame.columns:
        return pd.to_numeric(frame["p5_elapsed_ms"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)

def _join_candidates(candidates: set[int]) -> str:
    return ";".join(str(candidate) for candidate in sorted(candidates))

def _pair_key(row: pd.Series) -> tuple[int, int]:
    left = int(row.get("left_candidate_index", -1))
    right = int(row.get("right_candidate_index", -1))
    return tuple(sorted((left, right)))

def _pair_endpoint_set(pairs: pd.DataFrame) -> set[int]:
    endpoints: set[int] = set()
    if pairs.empty:
        return endpoints
    for _, row in pairs.iterrows():
        left, right = _pair_key(row)
        if left >= 0:
            endpoints.add(left)
        if right >= 0:
            endpoints.add(right)
    return endpoints

def _pair_tier(
    pair: pd.Series,
    *,
    best_candidate: int,
    near_candidates: set[int],
) -> tuple[str, str]:
    left, right = _pair_key(pair)
    if best_candidate in {left, right}:
        return "core", "touches_quality_first_best"
    if left in near_candidates or right in near_candidates:
        return "core", "touches_near_qf_candidate"
    return "diagnostic", "support_distinct_iso_q_inventory"

def _prefix_elapsed(
    ranked: pd.DataFrame,
    elapsed: pd.Series,
    prefix_k: int,
) -> float:
    if prefix_k <= 0:
        return 0.0
    return _finite_float(elapsed.loc[ranked.head(prefix_k).index].sum())

def _subset_elapsed(
    ranked: pd.DataFrame,
    elapsed: pd.Series,
    required_candidates: set[int],
) -> float:
    if not required_candidates:
        return 0.0
    candidate_numbers = pd.to_numeric(ranked.get("candidate_index"), errors="coerce")
    indices = ranked[candidate_numbers.isin(required_candidates)].index
    return _finite_float(elapsed.loc[indices].sum())

def _prefix_k_for_required(
    required_candidates: set[int],
    p1_rank_by_candidate: dict[int, int],
) -> int:
    ranks = [
        p1_rank_by_candidate[candidate]
        for candidate in required_candidates
        if candidate in p1_rank_by_candidate
    ]
    return max(ranks) if ranks else 0

def build_pair_tier_rows(
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
        best_candidate = int(contract.get("quality_first_candidate_index", -1))
        best_delta = _finite_float(contract.get("quality_first_p5_delta_q"))
        group_members = member_rows[_base_mask(member_rows, base)]
        near_candidates = _role_candidate_set(group_members, "near_qf_alternative")
        group_pairs = pair_rows[_base_mask(pair_rows, base)]
        if group_pairs.empty:
            continue
        p5_by_candidate = {
            int(row.get("candidate_index", -1)): _finite_float(row.get("p5_delta_q"))
            for _, row in ranked.iterrows()
        }
        rank_by_candidate = {
            int(row.get("candidate_index", -1)): pos
            for pos, (_, row) in enumerate(ranked.iterrows(), start=1)
        }
        field, method = _case_field_method(ranked.iloc[0])
        for _, pair in group_pairs.iterrows():
            left, right = _pair_key(pair)
            tier, reason = _pair_tier(
                pair,
                best_candidate=best_candidate,
                near_candidates=near_candidates,
            )
            left_delta = p5_by_candidate.get(left, math.nan)
            right_delta = p5_by_candidate.get(right, math.nan)
            rows.append(
                {
                    **base,
                    "field": field,
                    "method": method,
                    "contract_version": CONTRACT_VERSION,
                    "pair_obligation_tier": tier,
                    "pair_tier_reason": reason,
                    "left_candidate_index": left,
                    "right_candidate_index": right,
                    "left_p1_rank": rank_by_candidate.get(left),
                    "right_p1_rank": rank_by_candidate.get(right),
                    "left_q_gap_to_best": (
                        best_delta - left_delta
                        if math.isfinite(best_delta) and math.isfinite(left_delta)
                        else math.nan
                    ),
                    "right_q_gap_to_best": (
                        best_delta - right_delta
                        if math.isfinite(best_delta) and math.isfinite(right_delta)
                        else math.nan
                    ),
                    "q_delta_abs": _finite_float(pair.get("q_delta_abs")),
                    "coarse_support_distance": _finite_float(
                        pair.get("coarse_support_distance")
                    ),
                    "changed_node_support_union": int(
                        _finite_float(pair.get("changed_node_support_union"), 0.0)
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["field", "method", "pair_obligation_tier", "q_delta_abs"],
        na_position="last",
    )

def build_tiered_subset_rows(
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
    pair_tiers = build_pair_tier_rows(
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
        group_contract = contract_rows[_base_mask(contract_rows, base)]
        if group_contract.empty:
            continue
        contract = group_contract.iloc[0]
        group_members = member_rows[_base_mask(member_rows, base)]
        group_pair_tiers = pair_tiers[_base_mask(pair_tiers, base)]
        best_candidate = int(contract.get("quality_first_candidate_index", -1))
        p1_candidate = int(contract.get("p1_candidate_index", -1))
        material_required = bool(contract.get("quality_first_material_premium", False))
        near_candidates = _role_candidate_set(group_members, "near_qf_alternative")
        if group_pair_tiers.empty or "pair_obligation_tier" not in group_pair_tiers.columns:
            core_pairs = group_pair_tiers
        else:
            core_pairs = group_pair_tiers[
                group_pair_tiers["pair_obligation_tier"] == "core"
            ]
        all_pairs = group_pair_tiers
        hard_required = {best_candidate}
        if material_required and p1_candidate >= 0:
            hard_required.add(p1_candidate)
        core_required = hard_required | near_candidates | _pair_endpoint_set(core_pairs)
        diagnostic_required = core_required | _pair_endpoint_set(all_pairs)
        tier_specs = {
            "hard": {
                "required": hard_required,
                "pair_count": 0,
                "description": "best endpoint and material p1 premium accounting",
            },
            "core": {
                "required": core_required,
                "pair_count": int(len(core_pairs)),
                "description": "hard obligations plus near-QF and best/near-linked support pairs",
            },
            "diagnostic": {
                "required": diagnostic_required,
                "pair_count": int(len(all_pairs)),
                "description": "core obligations plus full support-distinct iso-Q inventory",
            },
        }
        rank_by_candidate = {
            int(row.get("candidate_index", -1)): pos
            for pos, (_, row) in enumerate(ranked.iterrows(), start=1)
        }
        elapsed = _candidate_elapsed_ms(ranked)
        accurate_elapsed = _finite_float(elapsed.sum())
        field, method = _case_field_method(ranked.iloc[0])
        for tier in TIER_ORDER:
            spec = tier_specs[tier]
            required = {candidate for candidate in spec["required"] if candidate >= 0}
            required_count = int(len(required))
            prefix_k = _prefix_k_for_required(required, rank_by_candidate)
            oracle_elapsed = _subset_elapsed(ranked, elapsed, required)
            prefix_elapsed = _prefix_elapsed(ranked, elapsed, prefix_k)
            row: dict[str, Any] = {
                **base,
                "field": field,
                "method": method,
                "contract_version": CONTRACT_VERSION,
                "contract_tier": tier,
                "tier_description": spec["description"],
                "accurate_mode_output_contract": contract.get(
                    "accurate_mode_output_contract"
                ),
                "winner_only_risk": contract.get("winner_only_risk"),
                "candidate_count": int(len(ranked)),
                "required_candidate_indices": _join_candidates(required),
                "oracle_required_candidate_count": required_count,
                "oracle_required_candidate_fraction": (
                    required_count / len(ranked) if len(ranked) > 0 else math.nan
                ),
                "p1_prefix_k_required": int(prefix_k),
                "p1_prefix_fraction_required": (
                    prefix_k / len(ranked) if len(ranked) > 0 else math.nan
                ),
                "p1_prefix_overhead_count": int(prefix_k - required_count),
                "p1_prefix_efficiency": (
                    required_count / prefix_k if prefix_k > 0 else math.nan
                ),
                "quality_first_candidate_index": best_candidate,
                "quality_first_p1_rank": int(
                    contract.get("quality_first_p1_rank", -1)
                ),
                "p1_candidate_index": p1_candidate,
                "material_quality_premium_required": material_required,
                "quality_first_premium_over_p1_q": _finite_float(
                    contract.get("quality_first_premium_over_p1_q")
                ),
                "near_qf_candidate_required_count": int(
                    len(near_candidates) if tier in {"core", "diagnostic"} else 0
                ),
                "support_distinct_pair_required_count": int(spec["pair_count"]),
                "oracle_required_elapsed_ms": oracle_elapsed,
                "p1_prefix_required_elapsed_ms": prefix_elapsed,
                "accurate_full_budget_p5_elapsed_ms": accurate_elapsed,
                "oracle_elapsed_ratio_vs_accurate": (
                    oracle_elapsed / accurate_elapsed
                    if accurate_elapsed > 0.0
                    else math.nan
                ),
                "p1_prefix_elapsed_ratio_vs_accurate": (
                    prefix_elapsed / accurate_elapsed
                    if accurate_elapsed > 0.0
                    else math.nan
                ),
            }
            for k in PREFIX_K_VALUES:
                row[f"p1_top{k}_covers_tier"] = bool(prefix_k <= k)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["field", "method", "contract_tier"])

def build_tiered_subset_summary(tier_rows: pd.DataFrame) -> pd.DataFrame:
    if tier_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tier in TIER_ORDER:
        group = tier_rows[tier_rows["contract_tier"] == tier]
        if group.empty:
            continue
        rows.append(_summarize_tier(tier, group))
    return pd.DataFrame(rows)

def _summarize_tier(tier: str, group: pd.DataFrame) -> dict[str, Any]:
    oracle_count = pd.to_numeric(
        group.get("oracle_required_candidate_count"),
        errors="coerce",
    )
    prefix_k = pd.to_numeric(group.get("p1_prefix_k_required"), errors="coerce")
    overhead = pd.to_numeric(group.get("p1_prefix_overhead_count"), errors="coerce")
    oracle_elapsed_ratio = pd.to_numeric(
        group.get("oracle_elapsed_ratio_vs_accurate"),
        errors="coerce",
    )
    prefix_elapsed_ratio = pd.to_numeric(
        group.get("p1_prefix_elapsed_ratio_vs_accurate"),
        errors="coerce",
    )
    pairs = pd.to_numeric(
        group.get("support_distinct_pair_required_count"),
        errors="coerce",
    )
    row: dict[str, Any] = {
        "contract_tier": tier,
        "case_count": int(len(group)),
        "oracle_required_candidate_count_mean": _finite_float(oracle_count.mean()),
        "oracle_required_candidate_count_max": _finite_float(oracle_count.max()),
        "p1_prefix_k_required_mean": _finite_float(prefix_k.mean()),
        "p1_prefix_k_required_max": _finite_float(prefix_k.max()),
        "p1_prefix_overhead_count_sum": int(overhead.fillna(0).sum()),
        "support_distinct_pair_required_sum": int(pairs.fillna(0).sum()),
        "oracle_elapsed_ratio_vs_accurate_mean": _finite_float(
            oracle_elapsed_ratio.mean()
        ),
        "p1_prefix_elapsed_ratio_vs_accurate_mean": _finite_float(
            prefix_elapsed_ratio.mean()
        ),
    }
    for k in PREFIX_K_VALUES:
        column = f"p1_top{k}_covers_tier"
        row[f"p1_top{k}_covers_tier_count"] = int(group[column].map(bool).sum())
    return row

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
    tier_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# Dongdaemun Contract Tiered Oracle Subset",
        "",
        "This diagnostic splits the accurate contract into hard/core/diagnostic obligations and compares oracle subset size with p1-prefix cost.",
        "",
    ]
    if tier_rows.empty:
        lines.append("- No tiered subset rows were available.")
    else:
        lines.extend(["## Headline", ""])
        for tier in TIER_ORDER:
            row = summary[summary["contract_tier"] == tier]
            if row.empty:
                continue
            summary_row = row.iloc[0]
            lines.append(
                f"- {tier}: oracle mean candidates "
                f"{_finite_float(summary_row.get('oracle_required_candidate_count_mean')):.6g}; "
                f"p1-prefix mean k "
                f"{_finite_float(summary_row.get('p1_prefix_k_required_mean')):.6g}; "
                f"top10 covers "
                f"{int(summary_row.get('p1_top10_covers_tier_count'))}/"
                f"{int(summary_row.get('case_count'))}"
            )
        lines.extend(["", "## Tier Summary", ""])
        lines.extend(_markdown_table(summary).splitlines())
        lines.extend(["", "## Largest P1 Prefix Gaps", ""])
        display_cols = [
            column
            for column in [
                "field",
                "method",
                "contract_tier",
                "oracle_required_candidate_count",
                "p1_prefix_k_required",
                "p1_prefix_overhead_count",
                "support_distinct_pair_required_count",
                "p1_prefix_elapsed_ratio_vs_accurate",
                "required_candidate_indices",
            ]
            if column in tier_rows.columns
        ]
        lines.extend(
            _markdown_table(
                tier_rows.sort_values(
                    ["p1_prefix_overhead_count", "p1_prefix_k_required"],
                    ascending=[False, False],
                ).head(16)[display_cols]
            ).splitlines()
        )
        lines.extend(["", "## Pair Tier Counts", ""])
        if pair_rows.empty:
            lines.append("- No support-distinct pair rows were available.")
        else:
            pair_summary = (
                pair_rows.groupby("pair_obligation_tier", dropna=False)
                .size()
                .reset_index(name="pair_count")
            )
            lines.extend(_markdown_table(pair_summary).splitlines())
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Hard obligations estimate the minimum candidate set needed for the final endpoint and material p1-premium accounting.",
                "- Core obligations add near-QF alternatives and support-distinct pairs touching the best or near-QF candidates.",
                "- Diagnostic obligations add the full support-distinct iso-Q inventory; this is the strictest evidence-preservation tier.",
                "- The gap between oracle subset size and p1-prefix k shows whether smarter candidate selection could help, while the diagnostic tier indicates when full accurate mode remains necessary.",
            ]
        )
    (output_dir / "dongdaemun_contract_tiered_subset_report.md").write_text(
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
    tier_rows = build_tiered_subset_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    pair_rows = build_pair_tier_rows(
        candidates,
        material_regret_q=args.material_regret_q,
        near_best_delta_q=args.near_best_delta_q,
        support_distinct_tau=args.support_distinct_tau,
    )
    summary = build_tiered_subset_summary(tier_rows)
    tier_rows.to_csv(
        output_dir / "dongdaemun_contract_tiered_subset_case_rows.csv",
        index=False,
    )
    pair_rows.to_csv(
        output_dir / "dongdaemun_contract_tiered_pair_rows.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "dongdaemun_contract_tiered_subset_summary.csv",
        index=False,
    )
    write_report(output_dir, tier_rows, pair_rows, summary)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "tier_rows": int(len(tier_rows)),
            "pair_rows": int(len(pair_rows)),
            "summary_rows": int(len(summary)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
