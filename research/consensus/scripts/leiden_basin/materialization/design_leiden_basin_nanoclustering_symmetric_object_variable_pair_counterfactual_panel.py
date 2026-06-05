#!/usr/bin/env python3
"""Design a small counterfactual panel from variable-pair graph mechanisms.

This reads the graph-local variable-pair mechanism rows emitted by
``analyze_leiden_basin_nanoclustering_symmetric_object_variable_pair_graph_mechanisms.py``
and freezes a compact candidate panel for the next mechanism test.

The scoring is intentionally counterfactual and read-only:

- what happens to the local pair CPM delta if the direct edge is removed;
- whether the pair is an input-gamma-only phase-boundary case;
- whether the shared-neighbor bridge mass is large relative to the direct edge;
- which object/pair scopes are covered by the predeclared selection.

It does not run Leiden, promote wall/pathway, inspect basin quality/cost as a
success claim, or claim method success.
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
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_counterfactual_panel_gamma1e5_20260603"
)

INPUT_PAIR_ROWS_CSV = "nanoclustering_symmetric_object_variable_pair_graph_mechanism_rows.csv"
CANDIDATE_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_counterfactual_candidate_rows.csv"
)
PANEL_ROWS_CSV = "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_rows.csv"
OBJECT_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_object_rows.csv"
)
SUMMARY_JSON = "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_summary.json"
REPORT_MD = "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_report.md"
CONFIG_JSON = "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_config.json"

RUN_STATUS = "designed_symmetric_object_variable_pair_counterfactual_panel"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object variable-pair counterfactual panel design "
    "only; re-scores saved variable terminal node-pairs and selects a small "
    "predeclared mechanism panel. It does not run Leiden, promote wall/pathway, "
    "basin-quality, cost, real-data method-success, or algorithm claims."
)


def _prefix_stats(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
    }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    numerator = numerator.astype(float)
    return numerator.divide(denominator.where(denominator.ne(0.0), np.nan))


def _policy_contrast_class(row: pd.Series) -> str:
    together = _policy_set(row.get("together_start_policies", ""))
    apart = _policy_set(row.get("apart_start_policies", ""))
    if not together or not apart:
        return "missing_policy_contrast"
    component = "object_seed_component_blocks"
    singleton = "object_singleton"
    seeded = {"seed0_source_state", "seed0_object_seeded"}
    if component in together and component not in apart:
        return "component_blocks_together"
    if component in apart and component not in together:
        return "component_blocks_apart"
    if singleton in together and singleton not in apart:
        return "singleton_together"
    if singleton in apart and singleton not in together:
        return "singleton_apart"
    if together <= seeded and not apart <= seeded:
        return "seeded_only_together"
    if apart <= seeded and not together <= seeded:
        return "seeded_only_apart"
    return "mixed_policy_contrast"


def _policy_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {part for part in str(value).split(";") if part}


def _counterfactual_class(row: pd.Series) -> str:
    direct_positive = bool(row["direct_positive_at_gamma"])
    persistent_3e5 = bool(row["direct_positive_at_gamma3e5"])
    direct_delta = float(row["direct_cpm_delta_q"])
    bridge_mass = float(row["common_neighbor_min_weight_sum"])
    if persistent_3e5:
        return "persistent_direct_positive_control"
    if direct_positive:
        return "direct_phase_boundary_pair"
    if direct_delta <= 0.0 and bridge_mass > 0.0:
        return "bridge_mediated_negative_direct_pair"
    return "weak_or_unclassified_pair"


def _score_candidates(
    *,
    pair_frame: pd.DataFrame,
    input_gamma: float,
    control_gamma: float,
    high_gamma: float,
) -> pd.DataFrame:
    frame = pair_frame.copy()
    frame["direct_cpm_delta_after_direct_edge_removal"] = (
        0.0 - frame["direct_cpm_penalty_at_gamma"].astype(float)
    )
    frame["direct_edge_removal_delta_q_shift"] = (
        frame["direct_cpm_delta_after_direct_edge_removal"].astype(float)
        - frame["direct_cpm_delta_q"].astype(float)
    )
    frame["direct_positive_after_direct_edge_removal"] = (
        frame["direct_cpm_delta_after_direct_edge_removal"].astype(float) > 0.0
    )
    frame["direct_edge_needed_for_input_gamma_positive"] = (
        frame["direct_positive_at_gamma"].astype(bool)
        & ~frame["direct_positive_after_direct_edge_removal"].astype(bool)
    )
    frame["direct_cpm_delta_at_control_gamma"] = (
        frame["direct_edge_weight"].astype(float)
        - (float(control_gamma) * frame["penalty_factor_doc_product"].astype(float))
    )
    frame["direct_cpm_delta_at_high_gamma"] = (
        frame["direct_edge_weight"].astype(float)
        - (float(high_gamma) * frame["penalty_factor_doc_product"].astype(float))
    )
    frame["direct_margin_to_control_gamma"] = frame["direct_cpm_delta_at_control_gamma"]
    frame["direct_margin_to_high_gamma"] = frame["direct_cpm_delta_at_high_gamma"]
    frame["direct_critical_gamma_gap_to_control"] = (
        frame["direct_critical_gamma"].astype(float) - float(control_gamma)
    )
    frame["direct_critical_gamma_gap_to_high"] = (
        frame["direct_critical_gamma"].astype(float) - float(high_gamma)
    )
    frame["bridge_to_direct_weight_ratio"] = _safe_ratio(
        frame["common_neighbor_min_weight_sum"],
        frame["direct_edge_weight"],
    )
    frame["bridge_to_input_penalty_ratio"] = _safe_ratio(
        frame["common_neighbor_min_weight_sum"],
        frame["direct_cpm_penalty_at_gamma"],
    )
    frame["bridge_to_weighted_degree_floor_ratio"] = _safe_ratio(
        frame["common_neighbor_min_weight_sum"],
        pd.concat(
            [
                frame["left_weighted_degree"].astype(float),
                frame["right_weighted_degree"].astype(float),
            ],
            axis=1,
        ).min(axis=1),
    )
    frame["counterfactual_class"] = frame.apply(_counterfactual_class, axis=1)
    frame["policy_contrast_class"] = frame.apply(_policy_contrast_class, axis=1)
    frame["selection_reason"] = ""
    frame["selected_for_panel"] = False
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    frame["input_gamma"] = float(input_gamma)
    frame["control_gamma"] = float(control_gamma)
    frame["high_gamma"] = float(high_gamma)
    return frame


def _mark_selection(
    frame: pd.DataFrame,
    row_ids: pd.Index,
    reason: str,
) -> None:
    if row_ids.empty:
        return
    existing = frame.loc[row_ids, "selection_reason"].astype(str)
    frame.loc[row_ids, "selection_reason"] = [
        ";".join(part for part in [old, reason] if part)
        for old in existing
    ]
    frame.loc[row_ids, "selected_for_panel"] = True


def _select_panel(
    *,
    candidate_frame: pd.DataFrame,
    global_top_direct: int,
    global_top_bridge: int,
    per_object_top_direct: int,
    per_object_top_bridge: int,
    per_object_top_negative_bridge: int,
) -> pd.DataFrame:
    frame = candidate_frame.copy()
    direct_positive = frame["direct_positive_at_gamma"].astype(bool)
    negative_direct = frame["direct_cpm_delta_q"].astype(float) <= 0.0
    persistent = frame["direct_positive_at_gamma3e5"].astype(bool)

    direct_order = ["direct_cpm_delta_q", "common_neighbor_min_weight_sum"]
    bridge_order = ["common_neighbor_min_weight_sum", "direct_cpm_delta_q"]

    _mark_selection(
        frame,
        frame[direct_positive]
        .sort_values(direct_order, ascending=[False, False], kind="mergesort")
        .head(int(global_top_direct))
        .index,
        "global_top_direct_positive",
    )
    _mark_selection(
        frame,
        frame.sort_values(bridge_order, ascending=[False, False], kind="mergesort")
        .head(int(global_top_bridge))
        .index,
        "global_top_bridge_mass",
    )
    _mark_selection(
        frame,
        frame[persistent]
        .sort_values("direct_cpm_delta_q", ascending=False, kind="mergesort")
        .index,
        "persistent_direct_positive_control",
    )

    for object_role_id, group in frame.groupby("object_role_universe_id", sort=False):
        del object_role_id
        _mark_selection(
            frame,
            group[group["direct_positive_at_gamma"].astype(bool)]
            .sort_values(direct_order, ascending=[False, False], kind="mergesort")
            .head(int(per_object_top_direct))
            .index,
            "per_object_top_direct_positive",
        )
        _mark_selection(
            frame,
            group.sort_values(bridge_order, ascending=[False, False], kind="mergesort")
            .head(int(per_object_top_bridge))
            .index,
            "per_object_top_bridge_mass",
        )
        _mark_selection(
            frame,
            group[group["direct_cpm_delta_q"].astype(float) <= 0.0]
            .sort_values(bridge_order, ascending=[False, False], kind="mergesort")
            .head(int(per_object_top_negative_bridge))
            .index,
            "per_object_negative_direct_bridge_control",
        )

    panel = frame[frame["selected_for_panel"].astype(bool)].copy()
    if panel.empty:
        return panel
    panel["panel_priority"] = (
        panel["selection_reason"].str.contains("persistent_direct_positive_control").astype(int)
        * 10_000
        + panel["selection_reason"].str.contains("global_top_direct_positive").astype(int)
        * 1_000
        + panel["selection_reason"].str.contains("global_top_bridge_mass").astype(int)
        * 500
        + panel["selection_reason"].str.contains("per_object_top_direct_positive").astype(int)
        * 100
        + panel["selection_reason"].str.contains("per_object_top_bridge_mass").astype(int)
        * 50
        + panel["selection_reason"]
        .str.contains("per_object_negative_direct_bridge_control")
        .astype(int)
        * 25
    )
    return panel.sort_values(
        [
            "panel_priority",
            "object_role_universe_id",
            "direct_cpm_delta_q",
            "common_neighbor_min_weight_sum",
        ],
        ascending=[False, True, False, False],
        kind="mergesort",
    )


def _object_summary(
    *,
    candidate_frame: pd.DataFrame,
    panel_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    panel_ids = set(panel_frame.index)
    for object_role_id, group in candidate_frame.groupby("object_role_universe_id", sort=False):
        selected = group.loc[group.index.intersection(panel_ids)]
        row = {
            "object_role_universe_id": object_role_id,
            "branch": group["branch"].iloc[0],
            "candidate_pair_count": int(len(group)),
            "panel_pair_count": int(len(selected)),
            "candidate_direct_positive_pair_count": int(
                group["direct_positive_at_gamma"].astype(bool).sum()
            ),
            "candidate_persistent_direct_control_pair_count": int(
                group["direct_positive_at_gamma3e5"].astype(bool).sum()
            ),
            "candidate_negative_direct_pair_count": int(
                (group["direct_cpm_delta_q"].astype(float) <= 0.0).sum()
            ),
            "candidate_counterfactual_class_counts": json.dumps(
                group["counterfactual_class"].value_counts().to_dict(),
                sort_keys=True,
            ),
            "panel_counterfactual_class_counts": json.dumps(
                selected["counterfactual_class"].value_counts().to_dict(),
                sort_keys=True,
            )
            if not selected.empty
            else "{}",
            "panel_selection_reason_counts": _reason_counts(selected),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for prefix, column in [
            ("candidate_direct_cpm_delta_q", "direct_cpm_delta_q"),
            ("candidate_direct_critical_gamma", "direct_critical_gamma"),
            ("candidate_bridge_min_weight", "common_neighbor_min_weight_sum"),
            ("candidate_bridge_to_direct_ratio", "bridge_to_direct_weight_ratio"),
        ]:
            row.update(_prefix_stats(prefix, group[column].to_numpy(dtype=np.float64)))
        if selected.empty:
            for prefix in [
                "panel_direct_cpm_delta_q",
                "panel_direct_critical_gamma",
                "panel_bridge_min_weight",
                "panel_bridge_to_direct_ratio",
            ]:
                row.update(_prefix_stats(prefix, np.asarray([], dtype=np.float64)))
        else:
            for prefix, column in [
                ("panel_direct_cpm_delta_q", "direct_cpm_delta_q"),
                ("panel_direct_critical_gamma", "direct_critical_gamma"),
                ("panel_bridge_min_weight", "common_neighbor_min_weight_sum"),
                ("panel_bridge_to_direct_ratio", "bridge_to_direct_weight_ratio"),
            ]:
                row.update(_prefix_stats(prefix, selected[column].to_numpy(dtype=np.float64)))
        rows.append(row)
    return pd.DataFrame(rows)


def _reason_counts(frame: pd.DataFrame) -> str:
    counts: dict[str, int] = {}
    if frame.empty:
        return "{}"
    for value in frame["selection_reason"].astype(str):
        for reason in value.split(";"):
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return json.dumps(counts, sort_keys=True)


def _build_summary(
    *,
    graph_mechanism_dir: Path,
    output_dir: Path,
    candidate_frame: pd.DataFrame,
    panel_frame: pd.DataFrame,
    object_frame: pd.DataFrame,
) -> dict[str, Any]:
    selected = candidate_frame["selected_for_panel"].astype(bool)
    summary: dict[str, Any] = {
        "schema": "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_summary.v1",
        "status": RUN_STATUS,
        "graph_mechanism_dir": str(graph_mechanism_dir),
        "output_dir": str(output_dir),
        "candidate_pair_count": int(len(candidate_frame)),
        "panel_pair_count": int(len(panel_frame)),
        "object_count": int(candidate_frame["object_role_universe_id"].nunique()),
        "panel_object_count": int(panel_frame["object_role_universe_id"].nunique())
        if not panel_frame.empty
        else 0,
        "selected_pair_share": float(selected.mean()) if len(candidate_frame) else None,
        "candidate_counterfactual_class_counts": candidate_frame[
            "counterfactual_class"
        ].value_counts().to_dict(),
        "panel_counterfactual_class_counts": panel_frame[
            "counterfactual_class"
        ].value_counts().to_dict()
        if not panel_frame.empty
        else {},
        "candidate_pair_scope_counts": candidate_frame["pair_scope"].value_counts().to_dict(),
        "panel_pair_scope_counts": panel_frame["pair_scope"].value_counts().to_dict()
        if not panel_frame.empty
        else {},
        "panel_selection_reason_counts": _reason_counts(panel_frame),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for prefix, column in [
        ("candidate_direct_cpm_delta_q", "direct_cpm_delta_q"),
        ("candidate_direct_critical_gamma", "direct_critical_gamma"),
        ("candidate_bridge_min_weight", "common_neighbor_min_weight_sum"),
        ("candidate_bridge_to_direct_ratio", "bridge_to_direct_weight_ratio"),
        ("candidate_direct_edge_removal_shift", "direct_edge_removal_delta_q_shift"),
    ]:
        summary.update(_prefix_stats(prefix, candidate_frame[column].to_numpy(dtype=np.float64)))
    if panel_frame.empty:
        for prefix in [
            "panel_direct_cpm_delta_q",
            "panel_direct_critical_gamma",
            "panel_bridge_min_weight",
            "panel_bridge_to_direct_ratio",
            "panel_direct_edge_removal_shift",
        ]:
            summary.update(_prefix_stats(prefix, np.asarray([], dtype=np.float64)))
    else:
        for prefix, column in [
            ("panel_direct_cpm_delta_q", "direct_cpm_delta_q"),
            ("panel_direct_critical_gamma", "direct_critical_gamma"),
            ("panel_bridge_min_weight", "common_neighbor_min_weight_sum"),
            ("panel_bridge_to_direct_ratio", "bridge_to_direct_weight_ratio"),
            ("panel_direct_edge_removal_shift", "direct_edge_removal_delta_q_shift"),
        ]:
            summary.update(_prefix_stats(prefix, panel_frame[column].to_numpy(dtype=np.float64)))
    summary["object_rows"] = object_frame.to_dict("records")
    return summary


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    object_frame: pd.DataFrame,
    panel_frame: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Variable-Pair Counterfactual Panel",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_pair_count: {summary['candidate_pair_count']}",
        f"- panel_pair_count: {summary['panel_pair_count']}",
        f"- panel_object_count: {summary['panel_object_count']}",
        f"- candidate_counterfactual_class_counts: {summary['candidate_counterfactual_class_counts']}",
        f"- panel_counterfactual_class_counts: {summary['panel_counterfactual_class_counts']}",
        f"- panel_pair_scope_counts: {summary['panel_pair_scope_counts']}",
        f"- panel_selection_reason_counts: {summary['panel_selection_reason_counts']}",
        f"- panel_direct_critical_gamma_max: {summary.get('panel_direct_critical_gamma_max')}",
        f"- panel_bridge_min_weight_max: {summary.get('panel_bridge_min_weight_max')}",
        f"- panel_bridge_to_direct_ratio_median: {summary.get('panel_bridge_to_direct_ratio_median')}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Objects",
    ]
    if object_frame.empty:
        lines.append("- no objects")
    else:
        for row in object_frame.sort_values(
            ["panel_pair_count", "candidate_pair_count"],
            ascending=[False, False],
            kind="mergesort",
        ).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['object_role_universe_id']}: "
                f"candidates={data['candidate_pair_count']}, "
                f"panel={data['panel_pair_count']}, "
                f"candidate_classes={data['candidate_counterfactual_class_counts']}, "
                f"panel_classes={data['panel_counterfactual_class_counts']}, "
                f"selection_reasons={data['panel_selection_reason_counts']}"
            )
    lines.extend(["", "## Panel Rows"])
    if panel_frame.empty:
        lines.append("- no selected rows")
    else:
        report_columns = [
            "object_role_universe_id",
            "branch",
            "left_node_id",
            "right_node_id",
            "pair_scope",
            "counterfactual_class",
            "selection_reason",
            "direct_cpm_delta_q",
            "direct_critical_gamma",
            "common_neighbor_min_weight_sum",
            "bridge_to_direct_weight_ratio",
            "policy_contrast_class",
        ]
        for row in panel_frame[report_columns].itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['object_role_universe_id']} "
                f"{data['left_node_id']}-{data['right_node_id']} "
                f"scope={data['pair_scope']} "
                f"class={data['counterfactual_class']} "
                f"reasons={data['selection_reason']} "
                f"delta={data['direct_cpm_delta_q']} "
                f"critical_gamma={data['direct_critical_gamma']} "
                f"bridge_min_weight={data['common_neighbor_min_weight_sum']} "
                f"bridge_direct_ratio={data['bridge_to_direct_weight_ratio']} "
                f"policy={data['policy_contrast_class']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "These rows freeze a small mechanism panel for a later local "
                "ablation/counterfactual or controlled demo. They are not a "
                "Leiden rerun and do not establish wall/pathway, quality/cost, "
                "method-success, or algorithm claims."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    graph_mechanism_dir = Path(args.graph_mechanism_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_frame = _read_csv(graph_mechanism_dir / INPUT_PAIR_ROWS_CSV)
    candidate_frame = _score_candidates(
        pair_frame=pair_frame,
        input_gamma=float(args.input_gamma),
        control_gamma=float(args.control_gamma),
        high_gamma=float(args.high_gamma),
    )
    panel_frame = _select_panel(
        candidate_frame=candidate_frame,
        global_top_direct=int(args.global_top_direct),
        global_top_bridge=int(args.global_top_bridge),
        per_object_top_direct=int(args.per_object_top_direct),
        per_object_top_bridge=int(args.per_object_top_bridge),
        per_object_top_negative_bridge=int(args.per_object_top_negative_bridge),
    )
    if not panel_frame.empty:
        candidate_frame.loc[panel_frame.index, "selected_for_panel"] = True
        candidate_frame.loc[panel_frame.index, "selection_reason"] = panel_frame[
            "selection_reason"
        ]
    object_frame = _object_summary(
        candidate_frame=candidate_frame,
        panel_frame=panel_frame,
    )

    _write_csv(candidate_frame, output_dir / CANDIDATE_ROWS_CSV)
    _write_csv(panel_frame, output_dir / PANEL_ROWS_CSV)
    _write_csv(object_frame, output_dir / OBJECT_ROWS_CSV)
    summary = _build_summary(
        graph_mechanism_dir=graph_mechanism_dir,
        output_dir=output_dir,
        candidate_frame=candidate_frame,
        panel_frame=panel_frame,
        object_frame=object_frame,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_variable_pair_counterfactual_panel.v1",
        "graph_mechanism_dir": str(graph_mechanism_dir),
        "output_dir": str(output_dir),
        "input_gamma": float(args.input_gamma),
        "control_gamma": float(args.control_gamma),
        "high_gamma": float(args.high_gamma),
        "global_top_direct": int(args.global_top_direct),
        "global_top_bridge": int(args.global_top_bridge),
        "per_object_top_direct": int(args.per_object_top_direct),
        "per_object_top_bridge": int(args.per_object_top_bridge),
        "per_object_top_negative_bridge": int(args.per_object_top_negative_bridge),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        object_frame=object_frame,
        panel_frame=panel_frame,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-mechanism-dir", type=Path, default=DEFAULT_GRAPH_MECHANISM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-gamma", type=float, default=1.0e-5)
    parser.add_argument("--control-gamma", type=float, default=3.0e-5)
    parser.add_argument("--high-gamma", type=float, default=1.0e-4)
    parser.add_argument("--global-top-direct", type=int, default=6)
    parser.add_argument("--global-top-bridge", type=int, default=6)
    parser.add_argument("--per-object-top-direct", type=int, default=2)
    parser.add_argument("--per-object-top-bridge", type=int, default=2)
    parser.add_argument("--per-object-top-negative-bridge", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
