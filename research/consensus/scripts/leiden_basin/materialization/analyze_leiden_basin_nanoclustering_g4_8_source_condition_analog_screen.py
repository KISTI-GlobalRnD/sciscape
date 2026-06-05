#!/usr/bin/env python3
"""Screen NanoClustering variable-pair rows for G4.8 source-condition analogs.

This is a read-only analog screen after the synthetic G4.8F-J chain. It reads
the frozen symmetric-object variable-pair graph mechanism, counterfactual local
ablation, and synthetic-demo-design artifacts. It does not run Leiden, execute
routes/pathways, promote walls, evaluate quality/cost value, or claim a method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_GRAPH_MECHANISM_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_graph_mechanisms_gamma1e5_20260603"
)
DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_DEMO_DESIGN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_gamma1e5_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_source_condition_analog_screen_gamma1e5_20260604"
)

GRAPH_ROWS_CSV = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_rows.csv"
LOCAL_PAIR_GATE_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_pair_gate_rows.csv"
)
DEMO_PAIR_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_pair_rows.csv"
)

ANALOG_ROWS_CSV = "nanoclustering_g4_8_source_condition_analog_rows.csv"
ROLE_SUMMARY_CSV = "nanoclustering_g4_8_source_condition_analog_role_summary.csv"
FAMILY_SUMMARY_CSV = "nanoclustering_g4_8_source_condition_analog_family_summary.csv"
OBJECT_SUMMARY_CSV = "nanoclustering_g4_8_source_condition_analog_object_summary.csv"
SUMMARY_JSON = "nanoclustering_g4_8_source_condition_analog_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_source_condition_analog_config.json"
REPORT_MD = "nanoclustering_g4_8_source_condition_analog_report.md"

RUN_STATUS = "read_only_nanoclustering_g4_8_source_condition_analog_screen"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 source-condition analog screen only; reads frozen "
    "symmetric-object variable-pair graph-mechanism, local-ablation, and "
    "synthetic-demo-design artifacts to classify source-local analog roles. "
    "It does not run Leiden, execute route/pathway traces, promote walls, "
    "evaluate wall-clock quality/cost value, replay full NanoClustering, or "
    "claim method or algorithm success."
)


def _prefix_stats(prefix: str, values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
    }


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    rows = frame[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    body: list[str] = []
    for row in rows.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _classify_source_condition(row: pd.Series, *, thresholds: dict[str, float]) -> tuple[str, str]:
    original = float(row["original_pair_coassigned_share"])
    drop_direct = float(row["drop_direct_pair_coassigned_share"])
    drop_bridge = float(row["drop_bridge_pair_coassigned_share"])
    drop_both = float(row["drop_direct_and_bridge_pair_coassigned_share"])
    suppressed_max = float(thresholds["suppressed_share_max"])
    released_min = float(thresholds["released_share_min"])
    partial_min = float(thresholds["partial_source_min"])
    partial_max = float(thresholds["partial_source_max"])
    target_min = float(thresholds["target_saturated_min"])

    direct_suppressed = drop_direct <= suppressed_max
    both_suppressed = drop_both <= suppressed_max
    bridge_released = drop_bridge >= released_min

    if partial_min <= original <= partial_max and direct_suppressed and bridge_released and both_suppressed:
        return "R_candidate", "strict_partial_release_ready_analog"
    if 0.0 < original < partial_min and direct_suppressed and bridge_released and both_suppressed:
        return "R_weak", "rare_start_release_ready_analog"
    if original >= target_min and direct_suppressed and bridge_released and both_suppressed:
        return "T_like", "target_saturated_direct_contact_no_handle_analog"
    if original >= target_min and direct_suppressed and drop_bridge <= suppressed_max and both_suppressed:
        return "T_or_failure", "coupled_direct_bridge_context_failure_control"
    if original <= suppressed_max and direct_suppressed and bridge_released and both_suppressed:
        return "N_like", "latent_release_without_original_source_control"
    if original <= suppressed_max and direct_suppressed and drop_bridge <= suppressed_max and both_suppressed:
        return "N_like", "no_local_source_or_release_control"
    return "mixed", "mixed_or_unclassified_source_condition"


def _load_inputs(
    *,
    graph_mechanism_dir: Path,
    local_ablation_dir: Path,
    demo_design_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph_rows = _read_csv(graph_mechanism_dir / GRAPH_ROWS_CSV)
    local_rows = _read_csv(local_ablation_dir / LOCAL_PAIR_GATE_ROWS_CSV)
    demo_rows = _read_csv(demo_design_dir / DEMO_PAIR_ROWS_CSV)
    return graph_rows, local_rows, demo_rows


def _merge_inputs(
    *,
    graph_rows: pd.DataFrame,
    local_rows: pd.DataFrame,
    demo_rows: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["object_role_universe_id", "branch", "left_node_id", "right_node_id"]
    base_cols = [
        "local_pair_id",
        *key_cols,
        "pair_scope",
        "counterfactual_class",
        "selection_reason",
        "original_pair_coassigned_share",
        "drop_direct_pair_coassigned_share",
        "drop_bridge_pair_coassigned_share",
        "drop_direct_and_bridge_pair_coassigned_share",
        "drop_direct_pair_coassigned_delta",
        "drop_bridge_pair_coassigned_delta",
        "drop_direct_and_bridge_pair_coassigned_delta",
        "original_distinct_endpoint_count",
        "original_recurrent_endpoint_count",
        "original_has_local_switch_signal",
        "gate_class",
        "gate_status",
    ]
    rows = local_rows[[col for col in base_cols if col in local_rows.columns]].copy()

    demo_cols = [
        "local_pair_id",
        "design_family",
        "synthetic_demo_role",
        "design_rationale",
        "next_demo_requirements",
        "direct_dependency_score",
        "bridge_competition_score",
        "direct_bridge_coupling_score",
        "local_reproducibility_score",
        "selected_bridge_count",
        "excluded_outside_bridge_count",
        "local_node_count",
        "local_doc_missing_node_count",
        "bridge_to_direct_weight_ratio",
        "original_top_endpoint_share",
        "original_pair_bridge_same_cluster_median",
        "original_mechanism_read_counts",
    ]
    rows = rows.merge(
        demo_rows[[col for col in demo_cols if col in demo_rows.columns]],
        on="local_pair_id",
        how="left",
        validate="one_to_one",
    )

    graph_cols = [
        *key_cols,
        "left_node_scope",
        "right_node_scope",
        "left_doc_count",
        "right_doc_count",
        "pair_doc_count_sum",
        "same_terminal_start_count",
        "apart_terminal_start_count",
        "together_start_policies",
        "apart_start_policies",
        "direct_edge_weight",
        "direct_cpm_delta_q",
        "direct_critical_gamma",
        "direct_positive_at_gamma",
        "direct_positive_at_gamma3e5",
        "direct_positive_at_gamma1e4",
        "common_neighbor_count",
        "common_neighbor_min_weight_sum",
        "common_neighbor_object_min_weight_sum",
        "common_neighbor_support_min_weight_sum",
        "common_neighbor_outside_min_weight_sum",
        "mechanism_label",
    ]
    graph_subset = graph_rows[[col for col in graph_cols if col in graph_rows.columns]].copy()
    return rows.merge(graph_subset, on=key_cols, how="left", validate="one_to_one")


def _classify_rows(frame: pd.DataFrame, *, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = frame.copy()
    role_pairs = rows.apply(
        lambda row: _classify_source_condition(row, thresholds=thresholds),
        axis=1,
    )
    rows["analog_macro_role"] = [role for role, _ in role_pairs]
    rows["analog_source_condition"] = [condition for _, condition in role_pairs]
    rows["source_availability_proxy"] = rows["original_pair_coassigned_share"].astype(float)
    rows["bridge_release_lift_proxy"] = (
        rows["drop_bridge_pair_coassigned_share"].astype(float)
        - rows["original_pair_coassigned_share"].astype(float)
    )
    rows["direct_dependency_proxy"] = (
        rows["original_pair_coassigned_share"].astype(float)
        - rows["drop_direct_pair_coassigned_share"].astype(float)
    )
    rows["bridge_competition_proxy"] = rows["bridge_release_lift_proxy"].astype(float)
    rows["ready_source_proxy"] = rows["analog_macro_role"].isin({"R_candidate", "R_weak"})
    rows["strict_ready_source_proxy"] = rows["analog_macro_role"].eq("R_candidate")
    rows["weak_ready_source_proxy"] = rows["analog_macro_role"].eq("R_weak")
    rows["target_saturated_proxy"] = rows["analog_macro_role"].isin({"T_like", "T_or_failure"})
    rows["nonready_control_proxy"] = rows["analog_macro_role"].isin({"N_like", "T_or_failure"})
    rows["exact_g4_8f_signature_available"] = False
    rows["signature_limit_note"] = (
        "local-ablation proxy only; no NanoClustering endpoint source-signature "
        "set equivalent to synthetic G4.8F is materialized in this screen"
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _summary_table(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        data: dict[str, Any] = dict(zip(group_cols, key, strict=True))
        data.update(
            {
                "pair_count": int(len(group)),
                "ready_source_proxy_count": _bool_count(group, "ready_source_proxy"),
                "strict_ready_source_proxy_count": _bool_count(group, "strict_ready_source_proxy"),
                "weak_ready_source_proxy_count": _bool_count(group, "weak_ready_source_proxy"),
                "target_saturated_proxy_count": _bool_count(group, "target_saturated_proxy"),
                "nonready_control_proxy_count": _bool_count(group, "nonready_control_proxy"),
                "source_condition_counts": json.dumps(
                    _count_dict(group["analog_source_condition"]),
                    sort_keys=True,
                ),
                "gate_class_counts": json.dumps(_count_dict(group["gate_class"]), sort_keys=True),
                "counterfactual_class_counts": json.dumps(
                    _count_dict(group["counterfactual_class"]),
                    sort_keys=True,
                ),
                "design_family_counts": json.dumps(
                    _count_dict(group.get("design_family", pd.Series(dtype=str))),
                    sort_keys=True,
                ),
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
        for col in [
            "original_pair_coassigned_share",
            "drop_direct_pair_coassigned_share",
            "drop_bridge_pair_coassigned_share",
            "drop_direct_and_bridge_pair_coassigned_share",
            "bridge_release_lift_proxy",
            "direct_dependency_proxy",
            "direct_cpm_delta_q",
            "direct_critical_gamma",
            "common_neighbor_min_weight_sum",
        ]:
            if col in group.columns:
                data.update(_prefix_stats(col, group[col]))
        rows.append(data)
    return pd.DataFrame(rows)


def _build_summary(
    *,
    rows: pd.DataFrame,
    role_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    object_summary: pd.DataFrame,
    output_dir: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    strict_count = _bool_count(rows, "strict_ready_source_proxy")
    weak_count = _bool_count(rows, "weak_ready_source_proxy")
    ready_count = _bool_count(rows, "ready_source_proxy")
    target_count = _bool_count(rows, "target_saturated_proxy")
    nonready_count = _bool_count(rows, "nonready_control_proxy")
    if strict_count > 0 and target_count > 0 and nonready_count > 0:
        screen_status = "real_data_analog_surface_has_ready_and_controls"
        recommended_next_gate = (
            "Design a frozen local analog validation panel over strict-ready, "
            "rare-ready, target-saturated, and nonready controls; keep it read-only "
            "or local-graph only before any full NanoClustering replay."
        )
    elif ready_count > 0:
        screen_status = "real_data_analog_surface_has_weak_ready_only"
        recommended_next_gate = (
            "Inspect weak-ready rows and controls before any local validation panel."
        )
    else:
        screen_status = "real_data_analog_surface_not_found_under_current_proxy"
        recommended_next_gate = (
            "Treat the G4.8 synthetic source-condition rule as toy-family-specific "
            "until a different real-data proxy is materialized."
        )
    return {
        "schema": "nanoclustering_g4_8_source_condition_analog_summary.v1",
        "status": screen_status,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(output_dir),
        "pair_count": int(len(rows)),
        "object_count": int(rows["object_role_universe_id"].nunique()) if not rows.empty else 0,
        "branch_counts": _count_dict(rows["branch"]) if "branch" in rows else {},
        "analog_macro_role_counts": _count_dict(rows["analog_macro_role"]),
        "analog_source_condition_counts": _count_dict(rows["analog_source_condition"]),
        "design_family_counts": _count_dict(rows["design_family"]) if "design_family" in rows else {},
        "gate_class_counts": _count_dict(rows["gate_class"]) if "gate_class" in rows else {},
        "counterfactual_class_counts": (
            _count_dict(rows["counterfactual_class"]) if "counterfactual_class" in rows else {}
        ),
        "ready_source_proxy_count": int(ready_count),
        "strict_ready_source_proxy_count": int(strict_count),
        "weak_ready_source_proxy_count": int(weak_count),
        "target_saturated_proxy_count": int(target_count),
        "nonready_control_proxy_count": int(nonready_count),
        "exact_g4_8f_signature_available": False,
        "thresholds": thresholds,
        "recommended_next_gate": recommended_next_gate,
        "written_artifacts": [
            ANALOG_ROWS_CSV,
            ROLE_SUMMARY_CSV,
            FAMILY_SUMMARY_CSV,
            OBJECT_SUMMARY_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
        "role_summary_rows": int(len(role_summary)),
        "family_summary_rows": int(len(family_summary)),
        "object_summary_rows": int(len(object_summary)),
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    role_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Source-Condition Analog Screen",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- object_count: {summary['object_count']}",
        f"- ready_source_proxy_count: {summary['ready_source_proxy_count']}",
        f"- strict_ready_source_proxy_count: {summary['strict_ready_source_proxy_count']}",
        f"- weak_ready_source_proxy_count: {summary['weak_ready_source_proxy_count']}",
        f"- target_saturated_proxy_count: {summary['target_saturated_proxy_count']}",
        f"- nonready_control_proxy_count: {summary['nonready_control_proxy_count']}",
        f"- exact_g4_8f_signature_available: {summary['exact_g4_8f_signature_available']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Role Summary",
        "",
    ]
    if role_summary.empty:
        lines.append("No rows.")
    else:
        cols = [
            "analog_macro_role",
            "pair_count",
            "strict_ready_source_proxy_count",
            "weak_ready_source_proxy_count",
            "target_saturated_proxy_count",
            "nonready_control_proxy_count",
            "original_pair_coassigned_share_median",
            "drop_bridge_pair_coassigned_share_median",
            "bridge_release_lift_proxy_median",
            "direct_dependency_proxy_median",
            "source_condition_counts",
        ]
        lines.append(_markdown_table(role_summary, cols))
    lines.extend(["", "## Design Family Summary", ""])
    if family_summary.empty:
        lines.append("No rows.")
    else:
        cols = [
            "design_family",
            "pair_count",
            "ready_source_proxy_count",
            "target_saturated_proxy_count",
            "nonready_control_proxy_count",
            "original_pair_coassigned_share_median",
            "drop_bridge_pair_coassigned_share_median",
            "source_condition_counts",
        ]
        lines.append(_markdown_table(family_summary, cols))
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This screen uses local-ablation coassignment proxies, not the exact "
            "synthetic G4.8F endpoint/source signature set. Positive analog rows "
            "justify a frozen local validation panel, not full NanoClustering replay, "
            "wall/pathway promotion, quality/cost evaluation, or method claims.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    graph_mechanism_dir = Path(args.graph_mechanism_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    demo_design_dir = Path(args.demo_design_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "partial_source_min": float(args.partial_source_min),
        "partial_source_max": float(args.partial_source_max),
        "target_saturated_min": float(args.target_saturated_min),
        "released_share_min": float(args.released_share_min),
        "suppressed_share_max": float(args.suppressed_share_max),
    }
    graph_rows, local_rows, demo_rows = _load_inputs(
        graph_mechanism_dir=graph_mechanism_dir,
        local_ablation_dir=local_ablation_dir,
        demo_design_dir=demo_design_dir,
    )
    merged = _merge_inputs(graph_rows=graph_rows, local_rows=local_rows, demo_rows=demo_rows)
    rows = _classify_rows(merged, thresholds=thresholds)
    rows = rows.sort_values(
        ["analog_macro_role", "analog_source_condition", "design_family", "local_pair_id"],
        kind="mergesort",
    )

    role_summary = _summary_table(rows, ["analog_macro_role"])
    family_summary = _summary_table(rows, ["design_family"])
    object_summary = _summary_table(rows, ["object_role_universe_id", "branch"])
    summary = _build_summary(
        rows=rows,
        role_summary=role_summary,
        family_summary=family_summary,
        object_summary=object_summary,
        output_dir=output_dir,
        thresholds=thresholds,
    )
    config = {
        "schema": "nanoclustering_g4_8_source_condition_analog_config.v1",
        "graph_mechanism_dir": str(graph_mechanism_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "demo_design_dir": str(demo_design_dir),
        "output_dir": str(output_dir),
        "thresholds": thresholds,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rows, output_dir / ANALOG_ROWS_CSV)
    _write_csv(role_summary, output_dir / ROLE_SUMMARY_CSV)
    _write_csv(family_summary, output_dir / FAMILY_SUMMARY_CSV)
    _write_csv(object_summary, output_dir / OBJECT_SUMMARY_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, role_summary=role_summary, family_summary=family_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-mechanism-dir", type=Path, default=DEFAULT_GRAPH_MECHANISM_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--demo-design-dir", type=Path, default=DEFAULT_DEMO_DESIGN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--partial-source-min", type=float, default=0.20)
    parser.add_argument("--partial-source-max", type=float, default=0.80)
    parser.add_argument("--target-saturated-min", type=float, default=0.80)
    parser.add_argument("--released-share-min", type=float, default=0.95)
    parser.add_argument("--suppressed-share-max", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    summary = run_screen(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
