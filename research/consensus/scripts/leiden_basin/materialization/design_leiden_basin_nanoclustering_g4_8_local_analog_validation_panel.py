#!/usr/bin/env python3
"""Freeze a local validation panel from the G4.8 source-condition analog screen.

This consumes the read-only NanoClustering G4.8 source-condition analog screen
and freezes all available local pairs into a stratified validation panel. It is
a design/materialization artifact only: it does not run Leiden, execute
routes/pathways, promote walls, evaluate quality/cost value, replay full
NanoClustering, or claim method success.
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


DEFAULT_ANALOG_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_source_condition_analog_screen_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_analog_validation_panel_gamma1e5_20260604"
)

INPUT_ANALOG_ROWS_CSV = "nanoclustering_g4_8_source_condition_analog_rows.csv"
PANEL_ROWS_CSV = "nanoclustering_g4_8_local_analog_validation_panel_rows.csv"
STRATUM_SUMMARY_CSV = "nanoclustering_g4_8_local_analog_validation_panel_stratum_summary.csv"
OBJECT_SUMMARY_CSV = "nanoclustering_g4_8_local_analog_validation_panel_object_summary.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_local_analog_validation_panel_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_local_analog_validation_panel_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_local_analog_validation_panel_config.json"
REPORT_MD = "nanoclustering_g4_8_local_analog_validation_panel_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_local_analog_validation_panel"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 local analog validation panel design only; freezes "
    "stratified local pairs from the source-condition analog screen for a next "
    "local validation step. It does not run Leiden, execute route/pathway "
    "traces, promote walls, evaluate wall-clock quality/cost value, replay full "
    "NanoClustering, or claim method or algorithm success."
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


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


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in frame[cols].itertuples(index=False):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _validation_stratum(row: pd.Series) -> tuple[str, str, str, str]:
    macro_role = str(row["analog_macro_role"])
    condition = str(row["analog_source_condition"])
    if macro_role == "R_candidate":
        return (
            "strict_ready",
            "ready",
            "partial source exists; direct edge is needed; bridge removal releases full pair coassignment",
            "validate exact source-signature proxy and held-out local coassignment behavior",
        )
    if macro_role == "R_weak":
        return (
            "rare_ready",
            "ready",
            "source exists only in rare seed/start endpoints; bridge removal releases pair coassignment",
            "validate seed/start sensitivity without promoting it as robust ready",
        )
    if macro_role == "T_like":
        return (
            "target_saturated_no_handle",
            "target_saturated",
            "original local graph is already target coassigned; handle should not run",
            "validate no-op schedule behavior and avoid counting saturation as discovery",
        )
    if macro_role == "T_or_failure":
        return (
            "coupled_direct_bridge_failure_control",
            "failure_control",
            "original pair is coassigned but bridge removal collapses it; context is not a ready source",
            "validate false-positive suppression for direct-bridge coupled cases",
        )
    if condition == "latent_release_without_original_source_control":
        return (
            "latent_release_no_source_control",
            "nonready_control",
            "bridge removal can coassign the pair but the original graph has no source availability",
            "validate no-handle behavior for latent release without original source",
        )
    if condition == "no_local_source_or_release_control":
        return (
            "no_release_control",
            "nonready_control",
            "no original source availability and no bridge-release coassignment",
            "validate hard negative behavior",
        )
    return (
        "mixed_unclassified",
        "mixed",
        "source-condition proxy is mixed or outside the frozen strata",
        "inspect before use; do not include in validation execution without review",
    )


def _priority_for_stratum(stratum: str) -> int:
    order = {
        "strict_ready": 1,
        "rare_ready": 2,
        "target_saturated_no_handle": 3,
        "latent_release_no_source_control": 4,
        "no_release_control": 5,
        "coupled_direct_bridge_failure_control": 6,
        "mixed_unclassified": 99,
    }
    return int(order.get(str(stratum), 99))


def _build_panel_rows(analog_rows: pd.DataFrame) -> pd.DataFrame:
    rows = analog_rows.copy()
    assignments = rows.apply(_validation_stratum, axis=1)
    rows["validation_stratum"] = [item[0] for item in assignments]
    rows["validation_family"] = [item[1] for item in assignments]
    rows["expected_validation_signal"] = [item[2] for item in assignments]
    rows["next_validation_probe"] = [item[3] for item in assignments]
    rows["panel_priority"] = rows["validation_stratum"].map(_priority_for_stratum)
    rows["include_in_validation_panel"] = ~rows["validation_stratum"].eq("mixed_unclassified")
    rows["panel_selection_reason"] = (
        "frozen_full_23_pair_analog_screen_panel; no within-stratum cherry-picking"
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    sort_cols = [
        "panel_priority",
        "validation_stratum",
        "object_role_universe_id",
        "local_pair_id",
    ]
    return rows.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _summary_table(panel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in panel.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        data: dict[str, Any] = dict(zip(group_cols, key, strict=True))
        data.update(
            {
                "pair_count": int(len(group)),
                "included_pair_count": int(group["include_in_validation_panel"].sum()),
                "object_count": int(group["object_role_universe_id"].nunique()),
                "branch_count": int(group["branch"].nunique()),
                "java_pair_count": int(group["branch"].astype(str).eq("java").sum()),
                "rust_pair_count": int(group["branch"].astype(str).eq("rust").sum()),
                "design_family_count": int(group["design_family"].nunique()),
                "design_family_counts": json.dumps(
                    _count_dict(group["design_family"]),
                    sort_keys=True,
                ),
                "analog_macro_role_counts": json.dumps(
                    _count_dict(group["analog_macro_role"]),
                    sort_keys=True,
                ),
                "source_condition_counts": json.dumps(
                    _count_dict(group["analog_source_condition"]),
                    sort_keys=True,
                ),
                "gate_class_counts": json.dumps(
                    _count_dict(group["gate_class"]),
                    sort_keys=True,
                ),
                "counterfactual_class_counts": json.dumps(
                    _count_dict(group["counterfactual_class"]),
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
            "bridge_to_direct_weight_ratio",
        ]:
            if col in group.columns:
                data.update(_prefix_stats(col, group[col]))
        rows.append(data)
    return pd.DataFrame(rows)


def _gate_row(gate_id: str, question: str, passed: bool, observed: Any, minimum: Any) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "gate_status": "pass" if bool(passed) else "fail",
        "observed": observed,
        "minimum_or_rule": minimum,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_gate_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    included = panel[panel["include_in_validation_panel"].astype(bool)].copy()
    counts = included["validation_stratum"].value_counts().to_dict()
    object_counts = included.groupby("validation_stratum")["object_role_universe_id"].nunique()
    branch_counts = included.groupby("validation_stratum")["branch"].nunique()
    rows = [
        _gate_row(
            "G1_all_rows_included",
            "Does the panel preserve the full analog screen rather than cherry-pick rows?",
            int(len(included)) == int(len(panel)) and int(len(panel)) == 23,
            f"included={len(included)} total={len(panel)}",
            "all 23 analog-screen rows included",
        ),
        _gate_row(
            "G2_ready_strata_present",
            "Are both strict-ready and rare-ready strata present?",
            counts.get("strict_ready", 0) >= 5 and counts.get("rare_ready", 0) >= 2,
            json.dumps({k: int(v) for k, v in counts.items()}, sort_keys=True),
            "strict_ready>=5 and rare_ready>=2",
        ),
        _gate_row(
            "G3_control_strata_present",
            "Are target-saturated and nonready controls both present?",
            counts.get("target_saturated_no_handle", 0) >= 5
            and (
                counts.get("latent_release_no_source_control", 0)
                + counts.get("no_release_control", 0)
            )
            >= 4
            and counts.get("coupled_direct_bridge_failure_control", 0) >= 2,
            json.dumps({k: int(v) for k, v in counts.items()}, sort_keys=True),
            "target_saturated>=5, nonready>=4, coupled_failure>=2",
        ),
        _gate_row(
            "G4_branch_coverage",
            "Does the panel cover both Java and Rust branches?",
            set(included["branch"].astype(str)) >= {"java", "rust"},
            json.dumps(_count_dict(included["branch"]), sort_keys=True),
            "java and rust both present",
        ),
        _gate_row(
            "G5_object_coverage",
            "Does the panel cover at least four source objects overall?",
            int(included["object_role_universe_id"].nunique()) >= 4,
            int(included["object_role_universe_id"].nunique()),
            "object_count>=4",
        ),
        _gate_row(
            "G6_strict_ready_not_single_object",
            "Are strict-ready rows spread beyond one object?",
            int(object_counts.get("strict_ready", 0)) >= 3
            and int(branch_counts.get("strict_ready", 0)) >= 2,
            f"objects={int(object_counts.get('strict_ready', 0))}; branches={int(branch_counts.get('strict_ready', 0))}",
            "strict_ready object_count>=3 and branch_count>=2",
        ),
        _gate_row(
            "G7_rare_ready_not_single_branch",
            "Are rare-ready rows represented across both branches?",
            int(branch_counts.get("rare_ready", 0)) >= 2,
            f"objects={int(object_counts.get('rare_ready', 0))}; branches={int(branch_counts.get('rare_ready', 0))}",
            "rare_ready branch_count>=2",
        ),
        _gate_row(
            "G8_design_family_diversity",
            "Does the panel preserve the six design-family surface?",
            int(included["design_family"].nunique()) >= 6,
            json.dumps(_count_dict(included["design_family"]), sort_keys=True),
            "design_family_count>=6",
        ),
        _gate_row(
            "G9_signature_gap_recorded",
            "Is the exact G4.8F source-signature gap explicitly kept closed?",
            "exact_g4_8f_signature_available" in included.columns
            and not bool(included["exact_g4_8f_signature_available"].fillna(False).any()),
            "exact_g4_8f_signature_available=false for all rows",
            "no exact source-signature claim",
        ),
        _gate_row(
            "G10_no_execution_claim",
            "Is this artifact design-only rather than execution evidence?",
            True,
            RUN_STATUS,
            "design/materialization only",
        ),
    ]
    return pd.DataFrame(rows)


def _panel_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty:
        return "validation_panel_gate_failed"
    if bool(gate_matrix["gate_status"].astype(str).eq("pass").all()):
        return "frozen_local_analog_validation_panel_ready"
    return "validation_panel_gate_failed"


def _build_summary(
    *,
    panel: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    object_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    analog_screen_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    included = panel[panel["include_in_validation_panel"].astype(bool)]
    status = _panel_status(gate_matrix)
    return {
        "schema": "nanoclustering_g4_8_local_analog_validation_panel_summary.v1",
        "status": status,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "analog_screen_dir": str(analog_screen_dir),
        "output_dir": str(output_dir),
        "pair_count": int(len(panel)),
        "included_pair_count": int(len(included)),
        "object_count": int(included["object_role_universe_id"].nunique()),
        "branch_counts": _count_dict(included["branch"]),
        "validation_stratum_counts": _count_dict(included["validation_stratum"]),
        "validation_family_counts": _count_dict(included["validation_family"]),
        "design_family_counts": _count_dict(included["design_family"]),
        "ready_pair_count": int(included["validation_family"].eq("ready").sum()),
        "target_control_pair_count": int(
            included["validation_family"].eq("target_saturated").sum()
        ),
        "nonready_control_pair_count": int(
            included["validation_family"].eq("nonready_control").sum()
        ),
        "failure_control_pair_count": int(
            included["validation_family"].eq("failure_control").sum()
        ),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "exact_g4_8f_signature_available": False,
        "recommended_next_gate": (
            "Run a local validation readout over this frozen panel only: inspect "
            "held-out local seed/start behavior and materialize source-signature "
            "proxies before any route/pathway, quality/cost, or full "
            "NanoClustering replay claim."
        ),
        "written_artifacts": [
            PANEL_ROWS_CSV,
            STRATUM_SUMMARY_CSV,
            OBJECT_SUMMARY_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
        "stratum_summary_rows": int(len(stratum_summary)),
        "object_summary_rows": int(len(object_summary)),
        "gate_matrix_rows": int(len(gate_matrix)),
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    stratum_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Local Analog Validation Panel",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- included_pair_count: {summary['included_pair_count']}",
        f"- object_count: {summary['object_count']}",
        f"- validation_stratum_counts: {summary['validation_stratum_counts']}",
        f"- validation_family_counts: {summary['validation_family_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- exact_g4_8f_signature_available: {summary['exact_g4_8f_signature_available']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Validation Strata",
        "",
    ]
    if stratum_summary.empty:
        lines.append("No strata.")
    else:
        lines.append(
            _markdown_table(
                stratum_summary,
                [
                    "validation_stratum",
                    "pair_count",
                    "object_count",
                    "branch_count",
                    "java_pair_count",
                    "rust_pair_count",
                    "design_family_counts",
                    "original_pair_coassigned_share_median",
                    "drop_bridge_pair_coassigned_share_median",
                    "bridge_release_lift_proxy_median",
                    "direct_dependency_proxy_median",
                ],
            )
        )
    lines.extend(["", "## Gate Matrix", ""])
    if gate_matrix.empty:
        lines.append("No gates.")
    else:
        lines.append(
            _markdown_table(
                gate_matrix,
                ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This artifact freezes the panel for the next local validation step. It "
            "does not execute that validation and does not open route/pathway, wall, "
            "quality/cost, full NanoClustering replay, or method claims.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_design(args: argparse.Namespace) -> dict[str, Any]:
    analog_screen_dir = Path(args.analog_screen_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analog_rows = _read_csv(analog_screen_dir / INPUT_ANALOG_ROWS_CSV)
    panel = _build_panel_rows(analog_rows)
    stratum_summary = _summary_table(panel, ["validation_stratum", "validation_family"])
    object_summary = _summary_table(panel, ["object_role_universe_id", "branch"])
    gate_matrix = _build_gate_matrix(panel)
    summary = _build_summary(
        panel=panel,
        stratum_summary=stratum_summary,
        object_summary=object_summary,
        gate_matrix=gate_matrix,
        analog_screen_dir=analog_screen_dir,
        output_dir=output_dir,
    )
    config = {
        "schema": "nanoclustering_g4_8_local_analog_validation_panel_config.v1",
        "analog_screen_dir": str(analog_screen_dir),
        "output_dir": str(output_dir),
        "selection_policy": "include_all_23_pairs_from_source_condition_analog_screen",
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(panel, output_dir / PANEL_ROWS_CSV)
    _write_csv(stratum_summary, output_dir / STRATUM_SUMMARY_CSV)
    _write_csv(object_summary, output_dir / OBJECT_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, stratum_summary=stratum_summary, gate_matrix=gate_matrix)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analog-screen-dir", type=Path, default=DEFAULT_ANALOG_SCREEN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_design(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
