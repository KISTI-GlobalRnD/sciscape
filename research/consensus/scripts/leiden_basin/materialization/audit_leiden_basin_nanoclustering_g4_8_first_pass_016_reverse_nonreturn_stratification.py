#!/usr/bin/env python3
"""Stratify the local_pair_016 reverse non-return routes.

This read-only audit consumes the executed same-seed target-anchor reverse
trace for ``local_pair_016``. It asks whether the 9/24 routes that fail strict
same-seed source return are true target/transient persistence, source-family
anchor ambiguity, or guard-anchor behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_ABLATION_DIR,
    SEED_RUNS_CSV as LOCAL_ABLATION_SEED_RUNS_CSV,
)


PRIMARY_PAIR_ID = "local_pair_016"
TRANSIENT_SIGNATURE_ID = "aeb59ab537e6"
TARGET_SIGNATURE_ID = "3c9b8a190753"

DEFAULT_REVERSE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_reverse_nonreturn_stratification_audit_gamma1e5_20260605"
)

ROUTE_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_route_rows.csv"
FINAL_STATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_final_state_rows.csv"
)
STRATUM_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_stratum_rows.csv"
SEED_PATTERN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_seed_pattern_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_reverse_nonreturn_stratification"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_reverse_nonreturn_stratification"
WALL_PROMOTION_STATUS = "not_promoted_reverse_nonreturn_stratification_only"
METHOD_STATUS = "reverse_nonreturn_stratification_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 reverse non-return "
    "stratification only; reads the executed reverse trace and local ablation "
    "anchor table to classify strict source-return failures. It does not rerun "
    "Leiden, execute new routes, promote basin walls, replay full "
    "NanoClustering, evaluate quality/cost value, or claim method success."
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _gate_row(
    gate_id: str,
    question: str,
    observed: Any,
    minimum_or_rule: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "observed": observed,
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(int(max_rows))
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(_json_safe(value), sort_keys=True).replace("|", "\\|")
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in visible.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    reverse_dir = Path(args.reverse_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    return {
        "reverse_summary": _read_json(
            reverse_dir / "nanoclustering_g4_8_first_pass_016_transient_reverse_summary.json"
        ),
        "reverse_gates": _read_csv(
            reverse_dir / "nanoclustering_g4_8_first_pass_016_transient_reverse_gate_matrix.csv"
        ),
        "reverse_route_rows": _read_csv(
            reverse_dir / "nanoclustering_g4_8_first_pass_016_transient_reverse_route_rows.csv"
        ),
        "reverse_trace_rows": _read_csv(
            reverse_dir / "nanoclustering_g4_8_first_pass_016_transient_reverse_trace_rows.csv"
        ),
        "seed_runs": _read_csv(local_ablation_dir / LOCAL_ABLATION_SEED_RUNS_CSV),
    }


def _anchor_lookup(seed_runs: pd.DataFrame) -> dict[tuple[str, int, str], str]:
    primary = seed_runs[seed_runs["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    lookup: dict[tuple[str, int, str], str] = {}
    for row in primary.itertuples(index=False):
        lookup[(str(row.start_condition), int(row.seed), str(row.graph_variant))] = str(
            row.endpoint_signature_id
        )
    return lookup


def _signature_sets(seed_runs: pd.DataFrame) -> dict[str, Any]:
    primary = seed_runs[seed_runs["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    original = primary[primary["graph_variant"].astype(str).eq("original")]
    drop_direct = primary[primary["graph_variant"].astype(str).eq("drop_direct_edge")]
    drop_bridge = primary[primary["graph_variant"].astype(str).eq("drop_bridge_edges")]
    return {
        "global_original": set(original["endpoint_signature_id"].astype(str)),
        "global_drop_direct": set(drop_direct["endpoint_signature_id"].astype(str)),
        "global_drop_bridge": set(drop_bridge["endpoint_signature_id"].astype(str)),
        "start_original": {
            start: set(group["endpoint_signature_id"].astype(str))
            for start, group in original.groupby("start_condition", sort=False)
        },
        "start_drop_direct": {
            start: set(group["endpoint_signature_id"].astype(str))
            for start, group in drop_direct.groupby("start_condition", sort=False)
        },
        "start_drop_bridge": {
            start: set(group["endpoint_signature_id"].astype(str))
            for start, group in drop_bridge.groupby("start_condition", sort=False)
        },
    }


def _final_state_stratum(row: pd.Series) -> str:
    if _as_bool(row["same_seed_source_return"]):
        return "same_seed_source_return"
    if _as_bool(row["same_seed_drop_direct_guard_match"]):
        return "same_seed_drop_direct_guard_nonreturn"
    if _as_bool(row["same_start_source_family_signature"]):
        return "same_start_source_family_not_same_seed_anchor"
    if _as_bool(row["global_source_family_signature"]):
        return "cross_start_source_family_not_same_seed_anchor"
    if _as_bool(row["same_seed_drop_bridge_match"]):
        return "same_seed_drop_bridge_residual_nonreturn"
    if str(row["final_signature_id"]) == TRANSIENT_SIGNATURE_ID:
        return "transient_persisted_to_final"
    if str(row["final_signature_id"]) == TARGET_SIGNATURE_ID:
        return "target_persisted_to_final"
    if float(row["final_support_distance_to_drop_bridge_edges"]) < float(
        row["final_support_distance_to_original"]
    ):
        return "drop_bridge_nearest_unknown_nonreturn"
    return "unclassified_nonreturn"


def _final_state_rows(context: dict[str, Any]) -> pd.DataFrame:
    trace_rows = context["reverse_trace_rows"].copy()
    route_rows = context["reverse_route_rows"].copy()
    seed_runs = context["seed_runs"]
    max_step = int(trace_rows["step_index"].astype(int).max())
    final_rows = trace_rows[trace_rows["step_index"].astype(int).eq(max_step)].copy()
    anchor_by_key = _anchor_lookup(seed_runs)
    sig_sets = _signature_sets(seed_runs)

    output: list[dict[str, Any]] = []
    route_class_by_key = {
        (str(row.start_condition), int(row.seed)): str(row.reverse_trace_class)
        for row in route_rows.itertuples(index=False)
    }
    source_fractions_by_key = {
        (str(row.start_condition), int(row.seed)): str(row.source_fractions)
        for row in route_rows.itertuples(index=False)
    }
    for row in final_rows.sort_values(["start_condition", "seed"], kind="mergesort").itertuples(
        index=False
    ):
        start = str(row.start_condition)
        seed = int(row.seed)
        final_signature = str(row.result_endpoint_signature_id)
        same_seed_original = anchor_by_key.get((start, seed, "original"), "")
        same_seed_drop_direct = anchor_by_key.get((start, seed, "drop_direct_edge"), "")
        same_seed_drop_bridge = anchor_by_key.get((start, seed, "drop_bridge_edges"), "")
        same_seed_source_return = "original_source_anchor" in str(
            row.endpoint_assignment_by_step
        ) or final_signature == same_seed_original
        out_row = {
            "local_pair_id": PRIMARY_PAIR_ID,
            "route_key": f"{start}|seed={seed}",
            "start_condition": start,
            "seed": seed,
            "reverse_trace_class": route_class_by_key.get((start, seed), ""),
            "source_fractions": source_fractions_by_key.get((start, seed), ""),
            "final_bridge_edge_weight_fraction": float(row.bridge_edge_weight_fraction),
            "final_signature_id": final_signature,
            "final_assignment_by_step": str(row.endpoint_assignment_by_step),
            "final_mechanism_read": str(row.mechanism_read),
            "final_pair_coassigned": bool(_as_bool(row.pair_coassigned)),
            "final_left_bridge_same_cluster_count": int(
                row.left_bridge_same_cluster_count
            ),
            "final_right_bridge_same_cluster_count": int(
                row.right_bridge_same_cluster_count
            ),
            "final_pair_bridge_same_cluster_count": int(
                row.pair_bridge_same_cluster_count
            ),
            "same_seed_original_signature_id": same_seed_original,
            "same_seed_drop_direct_signature_id": same_seed_drop_direct,
            "same_seed_drop_bridge_signature_id": same_seed_drop_bridge,
            "same_seed_source_return": bool(same_seed_source_return),
            "same_seed_drop_direct_guard_match": bool(
                final_signature == same_seed_drop_direct
                or "drop_direct_guard_anchor" in str(row.endpoint_assignment_by_step)
            ),
            "same_seed_drop_bridge_match": bool(final_signature == same_seed_drop_bridge),
            "same_start_source_family_signature": bool(
                final_signature in sig_sets["start_original"].get(start, set())
            ),
            "global_source_family_signature": bool(
                final_signature in sig_sets["global_original"]
            ),
            "same_start_drop_direct_family_signature": bool(
                final_signature in sig_sets["start_drop_direct"].get(start, set())
            ),
            "global_drop_direct_family_signature": bool(
                final_signature in sig_sets["global_drop_direct"]
            ),
            "global_drop_bridge_family_signature": bool(
                final_signature in sig_sets["global_drop_bridge"]
            ),
            "final_support_distance_to_original": float(row.support_distance_to_original),
            "final_support_distance_to_drop_bridge_edges": float(
                row.support_distance_to_drop_bridge_edges
            ),
            "final_support_distance_to_drop_direct_edge": float(
                row.support_distance_to_drop_direct_edge
            ),
            "final_objective_delta_from_start": float(row.objective_delta_from_start),
            "final_objective_recovery_from_min": float(row.objective_recovery_from_min),
            "route_execution_status": ROUTE_EXECUTION_STATUS,
            "wall_promotion_status": WALL_PROMOTION_STATUS,
            "method_status": METHOD_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "run_status": RUN_STATUS,
        }
        out_row["final_state_stratum"] = _final_state_stratum(pd.Series(out_row))
        output.append(out_row)
    return pd.DataFrame(output)


def _route_rows(final_rows: pd.DataFrame) -> pd.DataFrame:
    rows = final_rows.copy()
    rows["strict_source_return_pass"] = rows["same_seed_source_return"].astype(bool)
    rows["nonreturn_explanation"] = rows["final_state_stratum"]
    return rows[
        [
            "route_key",
            "start_condition",
            "seed",
            "reverse_trace_class",
            "strict_source_return_pass",
            "nonreturn_explanation",
            "source_fractions",
            "final_signature_id",
            "final_assignment_by_step",
            "final_mechanism_read",
            "final_pair_coassigned",
            "same_seed_original_signature_id",
            "same_seed_drop_direct_signature_id",
            "same_seed_drop_bridge_signature_id",
            "same_start_source_family_signature",
            "global_source_family_signature",
            "final_support_distance_to_original",
            "final_support_distance_to_drop_bridge_edges",
            "final_support_distance_to_drop_direct_edge",
            "final_objective_recovery_from_min",
            "route_execution_status",
            "wall_promotion_status",
            "method_status",
            "claim_boundary",
            "run_status",
        ]
    ].copy()


def _stratum_rows(final_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, group in final_rows.groupby("final_state_stratum", sort=False):
        nonreturn = group[~group["same_seed_source_return"].astype(bool)]
        rows.append(
            {
                "final_state_stratum": str(stratum),
                "route_count": int(len(group)),
                "nonreturn_route_count": int(len(nonreturn)),
                "start_conditions": ";".join(sorted(group["start_condition"].astype(str).unique())),
                "seeds": ";".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
                "signature_ids": ";".join(
                    sorted(group["final_signature_id"].astype(str).unique())
                ),
                "mechanism_reads": ";".join(
                    sorted(group["final_mechanism_read"].astype(str).unique())
                ),
                "pair_coassigned_count": int(group["final_pair_coassigned"].astype(bool).sum()),
                "same_start_source_family_count": int(
                    group["same_start_source_family_signature"].astype(bool).sum()
                ),
                "global_source_family_count": int(
                    group["global_source_family_signature"].astype(bool).sum()
                ),
                "mean_distance_to_original": float(
                    group["final_support_distance_to_original"].mean()
                ),
                "mean_distance_to_drop_bridge": float(
                    group["final_support_distance_to_drop_bridge_edges"].mean()
                ),
                "mean_distance_to_drop_direct": float(
                    group["final_support_distance_to_drop_direct_edge"].mean()
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["nonreturn_route_count", "route_count"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def _seed_pattern_rows(final_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, group in final_rows.groupby("seed", sort=True):
        rows.append(
            {
                "seed": int(seed),
                "route_count": int(len(group)),
                "strict_source_return_count": int(
                    group["same_seed_source_return"].astype(bool).sum()
                ),
                "nonreturn_route_count": int(
                    (~group["same_seed_source_return"].astype(bool)).sum()
                ),
                "nonreturn_starts": ";".join(
                    sorted(
                        group.loc[
                            ~group["same_seed_source_return"].astype(bool),
                            "start_condition",
                        ].astype(str)
                    )
                ),
                "final_state_strata": ";".join(
                    sorted(group["final_state_stratum"].astype(str).unique())
                ),
                "final_signature_ids": ";".join(
                    sorted(group["final_signature_id"].astype(str).unique())
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _decision_rows(final_rows: pd.DataFrame, stratum_rows: pd.DataFrame) -> pd.DataFrame:
    nonreturn = final_rows[~final_rows["same_seed_source_return"].astype(bool)]
    final_target_or_transient = nonreturn[
        nonreturn["final_signature_id"].astype(str).isin(
            {TARGET_SIGNATURE_ID, TRANSIENT_SIGNATURE_ID}
        )
    ]
    source_family_nonreturn = nonreturn[
        nonreturn["global_source_family_signature"].astype(bool)
    ]
    guard_nonreturn = nonreturn[
        nonreturn["final_state_stratum"].astype(str).str.contains("guard")
    ]
    start_counts = (
        final_rows.groupby("start_condition")["same_seed_source_return"]
        .agg(["count", "sum"])
        .reset_index()
    )
    start_counts["nonreturn"] = start_counts["count"] - start_counts["sum"]
    decisions = [
        {
            "decision_id": "D1_nonreturn_not_target_or_transient_persistence",
            "axis": "final_endpoint_identity",
            "observed": (
                f"{len(final_target_or_transient)}/{len(nonreturn)} non-return "
                "rows end as target/transient signatures"
            ),
            "decision": "nonreturn_exits_target_and_transient_band",
            "passes": len(final_target_or_transient) == 0,
            "claim_effect": "blocks target-hysteresis wording for final state",
        },
        {
            "decision_id": "D2_source_family_explains_most_nonreturn",
            "axis": "source_anchor_vocabulary",
            "observed": (
                f"{len(source_family_nonreturn)}/{len(nonreturn)} non-return "
                "rows are global source-family signatures"
            ),
            "decision": "strict_same_seed_source_anchor_is_too_brittle",
            "passes": len(source_family_nonreturn) >= max(1, len(nonreturn) - 1),
            "claim_effect": "next gate should refine source-family equivalence before localization",
        },
        {
            "decision_id": "D3_guard_residual_named",
            "axis": "guard_anchor",
            "observed": (
                f"{len(guard_nonreturn)}/{len(nonreturn)} non-return rows match "
                "drop-direct guard behavior"
            ),
            "decision": "guard_residual_is_named_not_promoted",
            "passes": len(guard_nonreturn) <= 1,
            "claim_effect": "keeps guard residual separate from source-family rows",
        },
        {
            "decision_id": "D4_start_seed_dependence_visible",
            "axis": "stratification",
            "observed": start_counts.to_dict("records"),
            "decision": "reverse_return_is_start_seed_dependent",
            "passes": bool(start_counts["nonreturn"].gt(0).any())
            and bool(start_counts["sum"].gt(0).any()),
            "claim_effect": "blocks all-seed reversible pathway claim",
        },
        {
            "decision_id": "D5_claim_boundary",
            "axis": "claim_boundary",
            "observed": CLAIM_BOUNDARY,
            "decision": "read_only_stratification_only",
            "passes": True,
            "claim_effect": "keeps wall, method, full replay, and quality claims closed",
        },
    ]
    return pd.DataFrame(decisions)


def _gate_matrix(
    *,
    reverse_gates: pd.DataFrame,
    final_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    stratum_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
) -> pd.DataFrame:
    pass_gates = set(
        reverse_gates.loc[reverse_gates["gate_status"].astype(str).eq("pass"), "gate_id"]
    )
    expected_upstream = {
        "G1_contract_gates_pass",
        "G2_exact_reverse_trace_scope",
        "G3_target_anchor_initialization_materialized",
        "G4_reverse_sequence_classified",
        "G6_claim_boundaries_closed",
    }
    nonreturn = final_rows[~final_rows["same_seed_source_return"].astype(bool)]
    final_target_or_transient = nonreturn[
        nonreturn["final_signature_id"].astype(str).isin(
            {TARGET_SIGNATURE_ID, TRANSIENT_SIGNATURE_ID}
        )
    ]
    source_or_guard_nonreturn = nonreturn[
        nonreturn["final_state_stratum"].astype(str).isin(
            {
                "same_start_source_family_not_same_seed_anchor",
                "cross_start_source_family_not_same_seed_anchor",
                "same_seed_drop_direct_guard_nonreturn",
            }
        )
    ]
    rows = [
        _gate_row(
            "G1_upstream_reverse_trace_usable",
            "Are the reverse trace scope/init/classification gates usable despite final-source failure?",
            {
                "reverse_gate_status_counts": reverse_gates[
                    "gate_status"
                ].value_counts().to_dict(),
                "required_pass_gates_present": sorted(expected_upstream & pass_gates),
            },
            "G1-G4 and G6 of reverse trace pass; G5 may fail as evidence",
            expected_upstream.issubset(pass_gates),
        ),
        _gate_row(
            "G2_final_rows_and_nonreturn_scope",
            "Did the audit classify all 24 final rows and the expected 9 non-return rows?",
            {
                "final_rows": len(final_rows),
                "route_rows": len(route_rows),
                "nonreturn_rows": len(nonreturn),
                "strata": stratum_rows["final_state_stratum"].tolist(),
            },
            "24 final rows, 9 strict source non-return rows, all classified",
            len(final_rows) == 24 and len(route_rows) == 24 and len(nonreturn) == 9,
        ),
        _gate_row(
            "G3_nonreturn_not_final_target_or_transient",
            "Do strict non-return rows avoid target/transient signatures at the final step?",
            {
                "nonreturn_rows": len(nonreturn),
                "final_target_or_transient_count": len(final_target_or_transient),
            },
            "0 non-return rows end as target or transient signature",
            len(nonreturn) > 0 and len(final_target_or_transient) == 0,
        ),
        _gate_row(
            "G4_source_family_or_guard_explains_nonreturn",
            "Are strict non-return rows explained by source-family or guard-anchor strata?",
            {
                "source_or_guard_nonreturn": len(source_or_guard_nonreturn),
                "stratum_counts": stratum_rows[
                    ["final_state_stratum", "route_count", "nonreturn_route_count"]
                ].to_dict("records"),
            },
            "all 9 non-return rows are source-family or guard-anchor strata",
            len(source_or_guard_nonreturn) == len(nonreturn),
        ),
        _gate_row(
            "G5_seed_and_start_patterns_materialized",
            "Are seed and start dependence visible for the mixed reverse class?",
            {
                "seed_rows": seed_rows[
                    ["seed", "strict_source_return_count", "nonreturn_route_count"]
                ].to_dict("records"),
                "start_counts": final_rows.groupby("start_condition")[
                    "same_seed_source_return"
                ].agg(["count", "sum"]).reset_index().to_dict("records"),
            },
            "seed-pattern table exists and at least one start has mixed outcomes",
            not seed_rows.empty
            and bool(
                final_rows.groupby("start_condition")["same_seed_source_return"]
                .agg(["sum", "count"])
                .assign(nonreturn=lambda frame: frame["count"] - frame["sum"])
                .eval("sum > 0 and nonreturn > 0")
                .any()
            ),
        ),
        _gate_row(
            "G6_claim_boundaries_closed",
            "Are wall, method, full replay, and quality/cost claims closed?",
            {
                "decision_passes": int(decision_rows["passes"].map(_as_bool).sum()),
                "claim_boundary": CLAIM_BOUNDARY,
            },
            "all decisions pass and claim boundary is read-only",
            bool(decision_rows["passes"].map(_as_bool).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    reverse_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    final_rows: pd.DataFrame,
    stratum_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    nonreturn = final_rows[~final_rows["same_seed_source_return"].astype(bool)]
    source_family_nonreturn = int(
        nonreturn["global_source_family_signature"].astype(bool).sum()
    )
    guard_nonreturn = int(
        nonreturn["final_state_stratum"].astype(str).str.contains("guard").sum()
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_summary.v1",
        "status": RUN_STATUS,
        "reverse_dir": str(reverse_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "final_row_count": int(len(final_rows)),
        "strict_source_return_count": int(
            final_rows["same_seed_source_return"].astype(bool).sum()
        ),
        "strict_source_nonreturn_count": int(len(nonreturn)),
        "source_family_nonreturn_count": source_family_nonreturn,
        "guard_nonreturn_count": guard_nonreturn,
        "stratum_counts": stratum_rows[
            ["final_state_stratum", "route_count", "nonreturn_route_count"]
        ].to_dict("records"),
        "seed_pattern_counts": seed_rows[
            ["seed", "strict_source_return_count", "nonreturn_route_count"]
        ].to_dict("records"),
        "decision_status_counts": decision_rows["passes"].map(_as_bool).value_counts().to_dict(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The strict reverse non-return rows do not end in the target or "
            "transient signatures. Most are source-family signatures that fail "
            "same-seed anchor reconciliation, with one drop-direct guard "
            "residual. The mixed reverse result is therefore primarily a source "
            "anchor-vocabulary/seed-dependence issue, not final target "
            "hysteresis."
        ),
        "recommended_next_gate": (
            "Define and audit source-family equivalence for 016 reverse final "
            "states before threshold localization or broader control expansion."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    final_rows: pd.DataFrame,
    stratum_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Reverse Non-Return Stratification",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- strict_source_return_count: {summary['strict_source_return_count']}",
        f"- strict_source_nonreturn_count: {summary['strict_source_nonreturn_count']}",
        f"- source_family_nonreturn_count: {summary['source_family_nonreturn_count']}",
        f"- guard_nonreturn_count: {summary['guard_nonreturn_count']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Strata",
        "",
        _markdown_table(
            stratum_rows,
            [
                "final_state_stratum",
                "route_count",
                "nonreturn_route_count",
                "start_conditions",
                "seeds",
                "signature_ids",
                "mechanism_reads",
                "mean_distance_to_original",
                "mean_distance_to_drop_bridge",
                "mean_distance_to_drop_direct",
            ],
            max_rows=20,
        ),
        "",
        "## Seed Patterns",
        "",
        _markdown_table(
            seed_rows,
            [
                "seed",
                "strict_source_return_count",
                "nonreturn_route_count",
                "nonreturn_starts",
                "final_state_strata",
                "final_signature_ids",
            ],
            max_rows=20,
        ),
        "",
        "## Final Rows",
        "",
        _markdown_table(
            final_rows,
            [
                "route_key",
                "reverse_trace_class",
                "final_state_stratum",
                "final_signature_id",
                "final_assignment_by_step",
                "final_mechanism_read",
                "same_seed_original_signature_id",
                "final_support_distance_to_original",
                "final_support_distance_to_drop_bridge_edges",
                "final_support_distance_to_drop_direct_edge",
            ],
            max_rows=30,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(
            decision_rows,
            ["decision_id", "axis", "observed", "decision", "passes", "claim_effect"],
            max_rows=20,
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gates,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Recommended Next Gate",
        "",
        summary["recommended_next_gate"],
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverse-dir", type=Path, default=DEFAULT_REVERSE_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    reverse_dir = Path(args.reverse_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _load_context(args)
    final_rows = _final_state_rows(context)
    route_rows = _route_rows(final_rows)
    stratum_rows = _stratum_rows(final_rows)
    seed_rows = _seed_pattern_rows(final_rows)
    decision_rows = _decision_rows(final_rows, stratum_rows)
    gates = _gate_matrix(
        reverse_gates=context["reverse_gates"],
        final_rows=final_rows,
        route_rows=route_rows,
        stratum_rows=stratum_rows,
        seed_rows=seed_rows,
        decision_rows=decision_rows,
    )
    summary = _summary(
        reverse_dir=reverse_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        final_rows=final_rows,
        stratum_rows=stratum_rows,
        seed_rows=seed_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_reverse_nonreturn_config.v1",
        "reverse_dir": str(reverse_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(route_rows, output_dir / ROUTE_ROWS_CSV)
    _write_csv(final_rows, output_dir / FINAL_STATE_ROWS_CSV)
    _write_csv(stratum_rows, output_dir / STRATUM_ROWS_CSV)
    _write_csv(seed_rows, output_dir / SEED_PATTERN_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        path=output_dir / REPORT_MD,
        summary=summary,
        final_rows=final_rows,
        stratum_rows=stratum_rows,
        seed_rows=seed_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
