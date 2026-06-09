#!/usr/bin/env python3
"""Run the 001/007 low-fraction schedule-boundary audit trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract import (
    CANDIDATE_IDS,
    CLAIM_BOUNDARY as CONTRACT_CLAIM_BOUNDARY,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    LOW_FRACTIONS,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace import (
    DEFAULT_LOCAL_ABLATION_DIR,
    _fraction_readout_rows,
    _pair_readout_rows,
    _seed_route_rows,
    _trace_rows,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from surface_claim_schema_adapter import (
    surface_claim_count_dict as _count_dict,
    surface_claim_gate_row as _gate_row,
    surface_claim_json_dump as _json_dump,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_gamma1e5_20260609"
)

TRACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_rows.csv"
)
SEED_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_seed_route_rows.csv"
)
PAIR_READOUT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_pair_rows.csv"
)
FRACTION_READOUT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_fraction_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_report.md"
)

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace"
ROUTE_EXECUTION_STATUS = "executed_surface_rule_low_fraction_boundary_local_fraction_trace"
WALL_PROMOTION_STATUS = "not_promoted_low_fraction_boundary_trace_only"
METHOD_STATUS = "surface_rule_low_fraction_boundary_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 001/007 low-fraction schedule-boundary "
    "trace only; executes the predeclared 0.5-to-0.0 local readout. It does "
    "not run full NanoClustering, promote wall/pathway labels, evaluate "
    "quality/cost value, replay full NanoClustering, or claim method success."
)
EXPECTED_ROUTE_PLAN_ROWS = 30


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return _json_dump(value).replace("|", "\\|")
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


def _retag(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    for column, value in [
        ("route_execution_status", ROUTE_EXECUTION_STATUS),
        ("wall_promotion_status", WALL_PROMOTION_STATUS),
        ("method_status", METHOD_STATUS),
        ("claim_boundary", CLAIM_BOUNDARY),
        ("run_status", RUN_STATUS),
    ]:
        if column in rows.columns:
            rows[column] = value
    if "wall_claim_ready_after_trace" in rows.columns:
        rows["wall_claim_ready_after_trace"] = False
    return rows


def _low_fraction_pair_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    pair_rows = _retag(_pair_readout_rows(seed_rows))
    details = []
    for pair_id, group in seed_rows.groupby("local_pair_id", sort=False):
        single_side_total = int(group["single_side_fraction_count"].astype(int).sum())
        target_like_total = int(group["target_like_fraction_count"].astype(int).sum())
        finite_band_count = int(
            group["single_side_adjacent_fraction_band"].astype(bool).sum()
        )
        recurrence_count = int(group["diagnostic_recurrence_pass"].astype(bool).sum())
        final_target_count = int(group["final_target_like"].astype(bool).sum())
        if recurrence_count > 0:
            boundary_class = "low_fraction_diagnostic_recurrence_candidate"
            artifact_status = "negative_guard_schedule_artifact_candidate"
        elif single_side_total > 0 or finite_band_count > 0:
            boundary_class = "low_fraction_single_side_signal_without_full_recurrence"
            artifact_status = "negative_guard_schedule_artifact_candidate"
        elif target_like_total > 0 or final_target_count > 0:
            boundary_class = "low_fraction_late_target_collapse_guard"
            artifact_status = "negative_guard_schedule_boundary_qualified"
        else:
            boundary_class = "low_fraction_no_recurrence_negative_guard_reinforced"
            artifact_status = "negative_guard_not_boundary_artifact_under_tested_schedule"
        details.append(
            {
                "local_pair_id": str(pair_id),
                "low_fraction_single_side_fraction_total": single_side_total,
                "low_fraction_target_like_fraction_total": target_like_total,
                "low_fraction_finite_band_sequence_count": finite_band_count,
                "low_fraction_recurrence_sequence_count": recurrence_count,
                "low_fraction_final_target_like_sequence_count": final_target_count,
                "low_fraction_boundary_class": boundary_class,
                "schedule_artifact_status": artifact_status,
            }
        )
    detail_rows = pd.DataFrame(details)
    return pair_rows.merge(detail_rows, on="local_pair_id", how="left", validate="one_to_one")


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    seeds: int,
) -> pd.DataFrame:
    expected_trace_rows = int(len(route_plan)) * int(seeds)
    expected_seed_rows = (
        int(route_plan[["local_pair_id", "start_condition"]].drop_duplicates().shape[0])
        * int(seeds)
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_contract_gates_pass",
                "Did every low-fraction contract gate pass?",
                _count_dict(contract_gates["gate_status"]),
                "all contract gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_exact_route_scope",
                "Was execution restricted to the 30 predeclared low-fraction rows?",
                {
                    "route_plan_rows": int(len(route_plan)),
                    "executed_route_contracts": int(trace_rows["route_contract_id"].nunique()),
                    "executed_pairs": sorted(trace_rows["local_pair_id"].astype(str).unique()),
                    "fractions": sorted(trace_rows["bridge_edge_weight_fraction"].astype(float).unique()),
                },
                "30 route rows, only 001/007, fixed low fractions",
                int(len(route_plan)) == EXPECTED_ROUTE_PLAN_ROWS
                and int(trace_rows["route_contract_id"].nunique()) == EXPECTED_ROUTE_PLAN_ROWS
                and set(trace_rows["local_pair_id"].astype(str)) == set(CANDIDATE_IDS)
                and sorted(trace_rows["bridge_edge_weight_fraction"].astype(float).unique())
                == sorted(LOW_FRACTIONS),
            ),
            _gate_row(
                "G3_seed_replicates_complete",
                "Was every route row executed for every seed?",
                {
                    "trace_rows": int(len(trace_rows)),
                    "expected_trace_rows": expected_trace_rows,
                    "seed_count": int(seeds),
                },
                "route_plan_rows * seeds trace rows",
                int(len(trace_rows)) == expected_trace_rows,
            ),
            _gate_row(
                "G4_sequence_readouts_materialized",
                "Were pair/start/seed low-fraction sequences materialized?",
                {
                    "seed_route_rows": int(len(seed_rows)),
                    "expected_seed_route_rows": expected_seed_rows,
                    "route_class_counts": _count_dict(seed_rows["gap_fill_route_class"]),
                },
                "six pair/start routes times seed count",
                int(len(seed_rows)) == expected_seed_rows,
            ),
            _gate_row(
                "G5_anchor_source_family_valid",
                "Does the 0.5 anchor start in the same source-family state?",
                {
                    "source_family_start_counts": _count_dict(seed_rows["source_family_start"]),
                    "route_count": int(len(seed_rows)),
                },
                "every pair/start/seed sequence starts in source-family at 0.5",
                bool(seed_rows["source_family_start"].astype(bool).all()),
            ),
            _gate_row(
                "G6_low_fraction_classes_materialized",
                "Were low-fraction boundary classes assigned for both pairs?",
                pair_rows[["local_pair_id", "low_fraction_boundary_class"]].to_dict("records"),
                "001/007 have boundary classes",
                len(pair_rows) == len(CANDIDATE_IDS)
                and set(pair_rows["local_pair_id"].astype(str)) == set(CANDIDATE_IDS)
                and bool(pair_rows["low_fraction_boundary_class"].astype(str).str.len().gt(0).all()),
            ),
            _gate_row(
                "G7_no_claim_promotion",
                "Are wall, pathway, method, quality, replay, and generality claims closed?",
                {
                    "wall_claim_ready_after_trace": _count_dict(
                        seed_rows["wall_claim_ready_after_trace"]
                    ),
                    "wall_promotion_status": _count_dict(seed_rows["wall_promotion_status"]),
                    "method_status": _count_dict(seed_rows["method_status"]),
                },
                "all claim flags remain false/diagnostic-only",
                bool(seed_rows["wall_claim_ready_after_trace"].eq(False).all())
                and bool(seed_rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all())
                and bool(seed_rows["method_status"].eq(METHOD_STATUS).all()),
            ),
        ]
    )


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
    seeds: int,
) -> dict[str, Any]:
    class_counts = _count_dict(pair_rows["low_fraction_boundary_class"])
    recurrence_ids = sorted(
        pair_rows.loc[
            pair_rows["low_fraction_boundary_class"].astype(str).eq(
                "low_fraction_diagnostic_recurrence_candidate"
            ),
            "local_pair_id",
        ].astype(str)
    )
    single_side_ids = sorted(
        pair_rows.loc[
            pair_rows["low_fraction_boundary_class"].astype(str).eq(
                "low_fraction_single_side_signal_without_full_recurrence"
            ),
            "local_pair_id",
        ].astype(str)
    )
    late_collapse_ids = sorted(
        pair_rows.loc[
            pair_rows["low_fraction_boundary_class"].astype(str).eq(
                "low_fraction_late_target_collapse_guard"
            ),
            "local_pair_id",
        ].astype(str)
    )
    reinforced_ids = sorted(
        pair_rows.loc[
            pair_rows["low_fraction_boundary_class"].astype(str).eq(
                "low_fraction_no_recurrence_negative_guard_reinforced"
            ),
            "local_pair_id",
        ].astype(str)
    )
    if recurrence_ids or single_side_ids:
        recommended_next = (
            "Audit low-fraction single-side signal rows before using 001/007 as "
            "negative guards; keep wall/pathway claims closed."
        )
    elif late_collapse_ids:
        recommended_next = (
            "Audit 001/007 as late target-collapse guards rather than reinforced "
            "no-recurrence negatives; keep them separate from the 016 transition band."
        )
    else:
        recommended_next = (
            "Audit 001/007 as reinforced scoreable negative guards and freeze the "
            "8-row guard panel for the next 016 mechanism step."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "route_plan_row_count": int(len(route_plan)),
        "trace_row_count": int(len(trace_rows)),
        "seed_route_row_count": int(len(seed_rows)),
        "pair_readout_row_count": int(len(pair_rows)),
        "fraction_readout_row_count": int(len(fraction_rows)),
        "seed_count": int(seeds),
        "candidate_pair_ids": sorted(pair_rows["local_pair_id"].astype(str).tolist()),
        "low_fraction_boundary_class_counts": class_counts,
        "low_fraction_diagnostic_recurrence_pair_ids": recurrence_ids,
        "low_fraction_single_side_signal_pair_ids": single_side_ids,
        "low_fraction_late_target_collapse_pair_ids": late_collapse_ids,
        "low_fraction_reinforced_negative_pair_ids": reinforced_ids,
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].astype(str).tolist(),
        "route_execution_opened": True,
        "panel_generality_claim_ready": False,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "interpretation": (
            "The trace is an independent-fraction local schedule-boundary readout. "
            "It can qualify whether the 001/007 negative guard was 0.5-bound, but "
            "it cannot establish wall, pathway, method, quality/cost, replay, or "
            "panel-generality claims."
        ),
        "recommended_next_gate": recommended_next,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 001/007 Low-Fraction Boundary Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- seed_route_row_count: {summary['seed_route_row_count']}",
        f"- candidate_pair_ids: {summary['candidate_pair_ids']}",
        f"- low_fraction_boundary_class_counts: {summary['low_fraction_boundary_class_counts']}",
        f"- low_fraction_diagnostic_recurrence_pair_ids: {summary['low_fraction_diagnostic_recurrence_pair_ids']}",
        f"- low_fraction_single_side_signal_pair_ids: {summary['low_fraction_single_side_signal_pair_ids']}",
        f"- low_fraction_late_target_collapse_pair_ids: {summary['low_fraction_late_target_collapse_pair_ids']}",
        f"- low_fraction_reinforced_negative_pair_ids: {summary['low_fraction_reinforced_negative_pair_ids']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Readout Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "route_sequence_count",
                "diagnostic_recurrence_pass_count",
                "low_fraction_single_side_fraction_total",
                "low_fraction_target_like_fraction_total",
                "low_fraction_final_target_like_sequence_count",
                "low_fraction_boundary_class",
                "schedule_artifact_status",
            ],
        ),
        "",
        "## Seed Route Rows",
        "",
        _markdown_table(
            seed_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "source_family_start",
                "single_side_fraction_count",
                "target_like_fraction_count",
                "final_target_like",
                "diagnostic_recurrence_pass",
                "gap_fill_route_class",
                "mechanism_read_sequence",
            ],
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        ),
        "",
        "## Boundary",
        "",
        "This is a local independent-fraction readout, not a pathway trace.",
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    trace_rows, candidate_pair_count = _trace_rows(
        route_plan=route_plan,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    trace_rows = _retag(trace_rows)
    seed_rows = _retag(_seed_route_rows(trace_rows))
    pair_rows = _low_fraction_pair_rows(seed_rows)
    fraction_rows = _retag(_fraction_readout_rows(trace_rows))
    gates = _gate_matrix(
        contract_gates=contract_gates,
        route_plan=route_plan,
        trace_rows=trace_rows,
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        seeds=int(args.seeds),
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        trace_rows=trace_rows,
        seed_rows=seed_rows,
        pair_rows=pair_rows,
        fraction_rows=fraction_rows,
        gates=gates,
        seeds=int(args.seeds),
    )
    summary["candidate_pair_count"] = int(candidate_pair_count)
    summary["contract_claim_boundary"] = CONTRACT_CLAIM_BOUNDARY

    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(seed_rows, output_dir / SEED_ROUTE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_READOUT_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_READOUT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, pair_rows=pair_rows, seed_rows=seed_rows, gates=gates)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
