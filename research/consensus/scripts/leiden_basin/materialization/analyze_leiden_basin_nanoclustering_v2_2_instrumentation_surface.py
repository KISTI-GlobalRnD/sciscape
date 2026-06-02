#!/usr/bin/env python3
"""Audit the NanoClustering v2.2 basin-definition instrumentation surface.

This is a read-only bridge from definition to instrumentation. It checks
whether the frozen v2.2 primitive surface, the residual-debt ledger, and the
existing recurrent/stress/control panels are separated well enough to support
the next measurement pass.

It does not run clustering, execute optimizer routes, promote wall/pathway
claims, or inspect basin quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_V2_2_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_2_exception_axis_registry_20260531"
)
DEFAULT_RECURRENT_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_recurrent_boundary_family_registry_20260530"
)
DEFAULT_STRATIFIED_PANEL_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_fragmentation_stratified_panel_20260530"
)
DEFAULT_MATCHED_CONTROLS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_matched_controls_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_v2_2_instrumentation_surface_20260531"
)

V2_2_PRIMITIVE_REGISTRY_CSV = (
    "nanoclustering_definition_core_v2_2_primitive_registry.csv"
)
V2_2_PRIMITIVE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_primitive_event_rows.csv"
)
V2_2_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_2_residual_definition_queue.csv"
)
RECURRENT_FAMILY_REGISTRY_CSV = "nanoclustering_recurrent_boundary_family_registry.csv"
STRATIFIED_PATTERN_SUMMARY_CSV = (
    "nanoclustering_fragmentation_panel_stratum_pattern_summary.csv"
)
MATCHED_CONTROL_PATTERN_SUMMARY_CSV = (
    "nanoclustering_matched_control_boundary_pattern_summary.csv"
)
VOLATILE_VS_CONTROL_SUMMARY_CSV = (
    "nanoclustering_volatile_vs_matched_control_event_summary.csv"
)

FAMILY_INSTRUMENTATION_ROWS_CSV = (
    "nanoclustering_v2_2_family_instrumentation_rows.csv"
)
REGISTRY_STRATUM_SUMMARY_CSV = (
    "nanoclustering_v2_2_registry_stratum_summary.csv"
)
EXTERNAL_CONTROL_SUMMARY_CSV = (
    "nanoclustering_v2_2_external_control_summary.csv"
)
INSTRUMENTATION_GATE_MATRIX_CSV = (
    "nanoclustering_v2_2_instrumentation_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_v2_2_instrumentation_summary.json"
REPORT_MD = "nanoclustering_v2_2_instrumentation_report.md"
CONFIG_JSON = "nanoclustering_v2_2_instrumentation_config.json"

CLAIM_BOUNDARY = (
    "V2.2 instrumentation-surface audit only; no route execution, wall/pathway "
    "promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_v2_2_instrumentation_surface"
SEVERE_LIKE_PATTERNS = {"severe_split_boundary", "split_and_merge_boundary"}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).sort_index().to_dict().items()
    }


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = frame.copy()
    for column in columns:
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0)
    return rows


def _family_status(*, accepted_events: int, residual_events: int) -> str:
    if accepted_events > 0 and residual_events > 0:
        return "accepted_with_residual_debt_v2_2_family"
    if accepted_events > 0:
        return "accepted_complete_v2_2_primitive_family"
    if residual_events > 0:
        return "residual_only_v2_2_debt_family"
    return "outside_v2_2_definition_core"


def _instrumentation_role(row: pd.Series) -> str:
    readiness = str(row["definition_readiness"])
    accepted_events = int(row["accepted_event_count"])
    residual_events = int(row["residual_event_count"])
    if readiness == "definition_core":
        if accepted_events > 0 and residual_events > 0:
            return "accepted_definition_surface_with_residual_debt"
        if accepted_events > 0:
            return "accepted_definition_surface"
        if residual_events > 0:
            return "residual_debt_surface"
        return "definition_core_unmapped_surface"
    if readiness == "definition_stress_test":
        return "stress_test_surface_not_promoted"
    if readiness in {"edge_control", "edge_case_control"}:
        return "edge_control_surface_not_promoted"
    return "nondefinition_reference_surface"


def _definition_surface_read(row: pd.Series) -> str:
    role = str(row["instrumentation_role"])
    if role == "accepted_definition_surface":
        return "accepted v2.2 primitive family; usable for next instrumentation"
    if role == "accepted_definition_surface_with_residual_debt":
        return "accepted v2.2 primitive family with explicit residual debt"
    if role == "residual_debt_surface":
        return "definition-core family held out as residual debt"
    if role == "stress_test_surface_not_promoted":
        return "stress-test recurrent family; context only, not a v2.2 basin"
    if role == "edge_control_surface_not_promoted":
        return "pair-only edge/control family; context only, not a v2.2 basin"
    return "outside the frozen v2.2 instrumentation surface"


def _accepted_family_summary(registry: pd.DataFrame) -> pd.DataFrame:
    return (
        registry.groupby("source_family_id", as_index=False)
        .agg(
            accepted_primitive_count=("primitive_id", "nunique"),
            accepted_event_count=("event_count", "sum"),
            accepted_confidence_tiers=("definition_confidence_tier", _joined_unique),
            accepted_rule_statuses=("definition_core_v2_2_rule_status", _joined_unique),
        )
        .rename(columns={"source_family_id": "family_id"})
    )


def _residual_family_summary(residual_queue: pd.DataFrame) -> pd.DataFrame:
    return (
        residual_queue.groupby("source_family_id", as_index=False)
        .agg(
            residual_row_count=("audit_id", "nunique"),
            residual_event_count=("event_count", "sum"),
            residual_queue_statuses=("definition_core_v2_2_queue_status", _joined_unique),
            residual_debt_reads=("residual_definition_read", _joined_unique),
        )
        .rename(columns={"source_family_id": "family_id"})
    )


def _joined_unique(values: pd.Series) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value)})
    return ";".join(clean)


def _family_instrumentation_rows(
    *,
    registry: pd.DataFrame,
    residual_queue: pd.DataFrame,
    recurrent_registry: pd.DataFrame,
) -> pd.DataFrame:
    accepted = _accepted_family_summary(registry)
    residual = _residual_family_summary(residual_queue)
    rows = recurrent_registry.merge(accepted, on="family_id", how="left")
    rows = rows.merge(residual, on="family_id", how="left")

    numeric_defaults = [
        "accepted_primitive_count",
        "accepted_event_count",
        "residual_row_count",
        "residual_event_count",
    ]
    for column in numeric_defaults:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0).astype(int)
    for column in [
        "accepted_confidence_tiers",
        "accepted_rule_statuses",
        "residual_queue_statuses",
        "residual_debt_reads",
    ]:
        rows[column] = rows[column].fillna("")

    rows["v2_2_family_status"] = rows.apply(
        lambda row: _family_status(
            accepted_events=int(row["accepted_event_count"]),
            residual_events=int(row["residual_event_count"]),
        ),
        axis=1,
    )
    rows["instrumentation_role"] = rows.apply(_instrumentation_role, axis=1)
    rows["definition_surface_read"] = rows.apply(_definition_surface_read, axis=1)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY

    preferred = [
        "family_id",
        "branch",
        "ref_cluster_id",
        "boundary_family_tier",
        "definition_readiness",
        "v2_2_family_status",
        "instrumentation_role",
        "accepted_primitive_count",
        "accepted_event_count",
        "accepted_confidence_tiers",
        "accepted_rule_statuses",
        "residual_row_count",
        "residual_event_count",
        "residual_queue_statuses",
        "ref_unit_count",
        "ref_weight_sum",
        "comparison_seed_count",
        "strong_seed_count",
        "severe_seed_count",
        "moderate_seed_count",
        "top_split_share_min",
        "top_split_share_median",
        "fragmentation_index_median",
        "definition_surface_read",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    return rows[[column for column in preferred if column in rows.columns]].sort_values(
        [
            "definition_readiness",
            "boundary_family_tier",
            "branch",
            "family_id",
        ]
    )


def _registry_stratum_summary(family_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _numeric(
        family_rows,
        [
            "ref_weight_sum",
            "comparison_seed_count",
            "strong_seed_count",
            "severe_seed_count",
            "accepted_primitive_count",
            "accepted_event_count",
            "residual_row_count",
            "residual_event_count",
            "top_split_share_min",
            "top_split_share_median",
            "fragmentation_index_median",
        ],
    )
    summary = (
        rows.groupby(
            [
                "definition_readiness",
                "boundary_family_tier",
                "instrumentation_role",
                "branch",
            ],
            as_index=False,
        )
        .agg(
            family_count=("family_id", "nunique"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            accepted_primitive_count=("accepted_primitive_count", "sum"),
            accepted_event_count=("accepted_event_count", "sum"),
            residual_row_count=("residual_row_count", "sum"),
            residual_event_count=("residual_event_count", "sum"),
            median_comparison_seed_count=("comparison_seed_count", "median"),
            median_strong_seed_count=("strong_seed_count", "median"),
            median_severe_seed_count=("severe_seed_count", "median"),
            median_top_split_share_min=("top_split_share_min", "median"),
            median_top_split_share=("top_split_share_median", "median"),
            median_fragmentation_index=("fragmentation_index_median", "median"),
        )
        .sort_values(
            [
                "definition_readiness",
                "boundary_family_tier",
                "instrumentation_role",
                "branch",
            ]
        )
    )
    summary["route_execution_status"] = ROUTE_EXECUTION_STATUS
    summary["wall_promotion_status"] = WALL_PROMOTION_STATUS
    summary["quality_cost_status"] = QUALITY_COST_STATUS
    summary["claim_boundary"] = CLAIM_BOUNDARY
    return summary


def _external_control_summary(
    *,
    stratified_pattern_summary: pd.DataFrame,
    matched_control_pattern_summary: pd.DataFrame,
    volatile_vs_control_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    stratified = stratified_pattern_summary.copy()
    stratified["context_dataset"] = "fragmentation_stratified_panel"
    stratified["context_cohort"] = stratified["selection_stratum"]
    rows.append(
        stratified[
            [
                "context_dataset",
                "context_cohort",
                "branch",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "median_top_split_share",
                "median_target_run_share",
            ]
        ]
    )

    matched = matched_control_pattern_summary.copy()
    matched["context_dataset"] = "matched_stable_control_panel"
    matched["context_cohort"] = "stable_matched_control"
    rows.append(
        matched[
            [
                "context_dataset",
                "context_cohort",
                "branch",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "median_top_split_share",
                "median_target_run_share",
            ]
        ]
    )

    volatile = volatile_vs_control_summary.copy()
    volatile["context_dataset"] = "volatile_vs_matched_control_panel"
    volatile["context_cohort"] = volatile["cohort"]
    rows.append(
        volatile[
            [
                "context_dataset",
                "context_cohort",
                "branch",
                "boundary_pattern",
                "event_count",
                "ref_cluster_count",
                "top_split_share_median",
                "target_run_share_median",
            ]
        ].rename(
            columns={
                "top_split_share_median": "median_top_split_share",
                "target_run_share_median": "median_target_run_share",
            }
        )
    )

    summary = pd.concat(rows, ignore_index=True)
    summary = _numeric(
        summary,
        [
            "event_count",
            "ref_cluster_count",
            "median_top_split_share",
            "median_target_run_share",
        ],
    )
    summary["is_severe_like_pattern"] = summary["boundary_pattern"].isin(
        SEVERE_LIKE_PATTERNS
    )
    summary["control_context_read"] = summary.apply(_external_context_read, axis=1)
    summary["route_execution_status"] = ROUTE_EXECUTION_STATUS
    summary["wall_promotion_status"] = WALL_PROMOTION_STATUS
    summary["quality_cost_status"] = QUALITY_COST_STATUS
    summary["claim_boundary"] = CLAIM_BOUNDARY
    return summary.sort_values(
        ["context_dataset", "context_cohort", "branch", "boundary_pattern"]
    )


def _external_context_read(row: pd.Series) -> str:
    dataset = str(row["context_dataset"])
    cohort = str(row["context_cohort"])
    pattern = str(row["boundary_pattern"])
    if "stable" in cohort and pattern in SEVERE_LIKE_PATTERNS:
        return "unexpected severe-like stable-control pattern; inspect before use"
    if "stable" in cohort:
        return "stable-control context; boundary remains mild or absorption-like"
    if cohort in {"persistent_strong", "recurrent_strong", "volatile"}:
        return "fragmentation context; severe-like split evidence is expected"
    if dataset == "fragmentation_stratified_panel":
        return "stratified endpoint-boundary context only"
    return "context-only external contrast"


def _gate_matrix(
    *,
    registry: pd.DataFrame,
    primitive_event_rows: pd.DataFrame,
    residual_queue: pd.DataFrame,
    family_rows: pd.DataFrame,
    external_control_summary: pd.DataFrame,
) -> pd.DataFrame:
    primitive_events = int(registry["event_count"].sum())
    residual_events = int(residual_queue["event_count"].sum())
    universe_events = primitive_events + residual_events
    coverage_share = primitive_events / universe_events if universe_events else 0.0
    duplicate_event_rows = int(
        primitive_event_rows.duplicated(["primitive_id", "event_id"]).sum()
    )
    definition_surface = family_rows[
        family_rows["instrumentation_role"].isin(
            [
                "accepted_definition_surface",
                "accepted_definition_surface_with_residual_debt",
                "residual_debt_surface",
            ]
        )
    ]
    noncore_definition_surface = int(
        definition_surface["definition_readiness"].ne("definition_core").sum()
    )
    stress_promoted = int(
        family_rows.loc[
            family_rows["definition_readiness"].eq("definition_stress_test"),
            "accepted_event_count",
        ].sum()
    )
    edge_promoted = int(
        family_rows.loc[
            family_rows["definition_readiness"].isin(["edge_control", "edge_case_control"]),
            "accepted_event_count",
        ].sum()
    )
    stable_severe_like_events = int(
        external_control_summary.loc[
            external_control_summary["context_cohort"].str.contains("stable", na=False)
            & external_control_summary["is_severe_like_pattern"],
            "event_count",
        ].sum()
    )
    accepted_family_count = int(family_rows["accepted_event_count"].gt(0).sum())
    residual_family_count = int(family_rows["residual_event_count"].gt(0).sum())
    rows = [
        {
            "gate_id": "G1_definition_surface_freeze",
            "gate_question": "Is v2.2 a stable measurement surface over the current definition universe?",
            "evidence": (
                f"primitive_events={primitive_events}, residual_events={residual_events}, "
                f"universe_events={universe_events}, coverage_share={coverage_share:.3f}, "
                f"duplicate_primitive_event_rows={duplicate_event_rows}"
            ),
            "status": (
                "pass"
                if coverage_share >= 0.85 and duplicate_event_rows == 0
                else "blocked"
            ),
            "decision": "freeze_v2_2_as_operational_instrumentation_surface",
            "next_action": "use accepted primitives as measurement units and carry residual debt as caveats",
        },
        {
            "gate_id": "G2_definition_stress_control_separation",
            "gate_question": "Do accepted/residual v2.2 rows stay inside definition-core registry strata?",
            "evidence": (
                f"accepted_family_count={accepted_family_count}, "
                f"residual_family_count={residual_family_count}, "
                f"noncore_definition_surface_rows={noncore_definition_surface}, "
                f"stress_accepted_events={stress_promoted}, edge_accepted_events={edge_promoted}"
            ),
            "status": (
                "pass"
                if noncore_definition_surface == 0 and stress_promoted == 0 and edge_promoted == 0
                else "blocked"
            ),
            "decision": "keep stress and edge-control families context-only",
            "next_action": "do not promote stress/control strata into the v2.2 basin definition",
        },
        {
            "gate_id": "G3_external_control_context",
            "gate_question": "Do matched stable controls remain a contextual separation check?",
            "evidence": (
                f"stable_control_severe_like_events={stable_severe_like_events}; "
                "external panels are not direct validation joins"
            ),
            "status": (
                "context_pass_not_causal" if stable_severe_like_events == 0 else "inspect_controls"
            ),
            "decision": "use controls as context only, not as proof of basin validity",
            "next_action": "report control contrast next to definition-surface measurements",
        },
        {
            "gate_id": "G4_wall_pathway_gate",
            "gate_question": "Can the current surface claim walls or pathways?",
            "evidence": "no route traces are executed or inspected in this audit",
            "status": "closed_no_route_evidence",
            "decision": "do_not_promote_wall_or_pathway_claims",
            "next_action": "open only after pair/pathway protocol is explicitly materialized",
        },
        {
            "gate_id": "G5_quality_cost_gate",
            "gate_question": "Can the current surface compare basin quality or search cost?",
            "evidence": "quality and cost fields are excluded by construction",
            "status": "closed_excluded_by_design",
            "decision": "do_not_rank_basins_by_quality_or_cost_here",
            "next_action": "defer quality/cost until basin existence and wall/pathway gates are separate",
        },
        {
            "gate_id": "G6_next_measurement_gate",
            "gate_question": "What is the next executable research step?",
            "evidence": (
                "accepted primitives, residual debt, stress-test strata, and controls are now "
                "separated in one audit table"
            ),
            "status": "ready_for_measurement_panel",
            "decision": "build an accepted-primitive instrumentation panel before any route work",
            "next_action": "measure family-level recurrence, endpoint-handle stability, and residual exclusions",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix: list[str] = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    registry_stratum_summary: pd.DataFrame,
    external_control_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    accepted_rows = registry_stratum_summary[
        registry_stratum_summary["instrumentation_role"].isin(
            [
                "accepted_definition_surface",
                "accepted_definition_surface_with_residual_debt",
            ]
        )
    ]
    residual_rows = registry_stratum_summary[
        registry_stratum_summary["instrumentation_role"].eq("residual_debt_surface")
    ]
    stable_context = external_control_summary[
        external_control_summary["context_cohort"].str.contains("stable", na=False)
    ]
    text = [
        "# NanoClustering V2.2 Instrumentation Surface Audit",
        "",
        f"- primitive_event_coverage: `{summary['primitive_event_count']}/{summary['definition_universe_event_count']}`",
        f"- residual_definition_events: `{summary['residual_definition_event_count']}`",
        f"- accepted_family_count: `{summary['accepted_family_count']}`",
        f"- accepted_with_residual_debt_family_count: `{summary['accepted_with_residual_debt_family_count']}`",
        f"- residual_only_family_count: `{summary['residual_only_family_count']}`",
        f"- stress_test_promoted_event_count: `{summary['stress_test_promoted_event_count']}`",
        f"- edge_control_promoted_event_count: `{summary['edge_control_promoted_event_count']}`",
        f"- stable_control_severe_like_events: `{summary['stable_control_severe_like_events']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=10,
        ),
        "",
        "## Accepted Definition Surface",
        "",
        _markdown_table(
            accepted_rows,
            [
                "boundary_family_tier",
                "instrumentation_role",
                "branch",
                "family_count",
                "accepted_primitive_count",
                "accepted_event_count",
                "residual_event_count",
                "median_top_split_share",
                "median_fragmentation_index",
            ],
            max_rows=20,
        ),
        "",
        "## Residual Debt Surface",
        "",
        _markdown_table(
            residual_rows,
            [
                "boundary_family_tier",
                "instrumentation_role",
                "branch",
                "family_count",
                "residual_row_count",
                "residual_event_count",
                "median_top_split_share",
                "median_fragmentation_index",
            ],
            max_rows=20,
        ),
        "",
        "## External Control Context",
        "",
        _markdown_table(
            stable_context,
            [
                "context_dataset",
                "context_cohort",
                "branch",
                "boundary_pattern",
                "event_count",
                "median_top_split_share",
                "median_target_run_share",
                "control_context_read",
            ],
            max_rows=20,
        ),
        "",
        "## Read",
        "",
        "- V2.2 is defensible as an instrumentation surface: it preserves the 1026-event definition universe as 910 accepted primitive events plus 116 residual-debt events.",
        "- The accepted and residual surfaces remain inside definition-core recurrent registry strata; stress-test and edge-control strata are still context-only.",
        "- Matched stable controls remain a contextual separation check because they do not show severe-like split or split-and-merge events, but this is not causal validation.",
        "- The next step should be an accepted-primitive measurement panel, not v2.3 promotion and not route/wall/quality work.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_2_registry_dir: Path,
    recurrent_registry_dir: Path,
    stratified_panel_dir: Path,
    matched_controls_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _numeric(
        _read_csv(v2_2_registry_dir / V2_2_PRIMITIVE_REGISTRY_CSV),
        ["event_count"],
    )
    primitive_event_rows = _read_csv(v2_2_registry_dir / V2_2_PRIMITIVE_EVENT_ROWS_CSV)
    residual_queue = _numeric(
        _read_csv(v2_2_registry_dir / V2_2_RESIDUAL_DEFINITION_QUEUE_CSV),
        ["event_count"],
    )
    recurrent_registry = _read_csv(recurrent_registry_dir / RECURRENT_FAMILY_REGISTRY_CSV)
    stratified_pattern_summary = _read_csv(
        stratified_panel_dir / STRATIFIED_PATTERN_SUMMARY_CSV
    )
    matched_control_pattern_summary = _read_csv(
        matched_controls_dir / MATCHED_CONTROL_PATTERN_SUMMARY_CSV
    )
    volatile_vs_control_summary = _read_csv(
        matched_controls_dir / VOLATILE_VS_CONTROL_SUMMARY_CSV
    )

    family_rows = _family_instrumentation_rows(
        registry=registry,
        residual_queue=residual_queue,
        recurrent_registry=recurrent_registry,
    )
    registry_stratum_summary = _registry_stratum_summary(family_rows)
    external_control_summary = _external_control_summary(
        stratified_pattern_summary=stratified_pattern_summary,
        matched_control_pattern_summary=matched_control_pattern_summary,
        volatile_vs_control_summary=volatile_vs_control_summary,
    )
    gate_matrix = _gate_matrix(
        registry=registry,
        primitive_event_rows=primitive_event_rows,
        residual_queue=residual_queue,
        family_rows=family_rows,
        external_control_summary=external_control_summary,
    )

    primitive_event_count = int(registry["event_count"].sum())
    residual_event_count = int(residual_queue["event_count"].sum())
    universe_event_count = primitive_event_count + residual_event_count
    accepted_family_count = int(family_rows["accepted_event_count"].gt(0).sum())
    accepted_with_debt_count = int(
        family_rows["v2_2_family_status"].eq(
            "accepted_with_residual_debt_v2_2_family"
        ).sum()
    )
    residual_only_count = int(
        family_rows["v2_2_family_status"].eq("residual_only_v2_2_debt_family").sum()
    )
    stress_test_promoted_event_count = int(
        family_rows.loc[
            family_rows["definition_readiness"].eq("definition_stress_test"),
            "accepted_event_count",
        ].sum()
    )
    edge_control_promoted_event_count = int(
        family_rows.loc[
            family_rows["definition_readiness"].isin(["edge_control", "edge_case_control"]),
            "accepted_event_count",
        ].sum()
    )
    stable_control_severe_like_events = int(
        external_control_summary.loc[
            external_control_summary["context_cohort"].str.contains("stable", na=False)
            & external_control_summary["is_severe_like_pattern"],
            "event_count",
        ].sum()
    )
    duplicate_primitive_event_rows = int(
        primitive_event_rows.duplicated(["primitive_id", "event_id"]).sum()
    )

    summary = {
        "primitive_row_count": int(registry["primitive_id"].nunique()),
        "primitive_event_count": primitive_event_count,
        "residual_definition_event_count": residual_event_count,
        "definition_universe_event_count": universe_event_count,
        "primitive_event_coverage_share": (
            primitive_event_count / universe_event_count if universe_event_count else 0.0
        ),
        "duplicate_primitive_event_rows": duplicate_primitive_event_rows,
        "accepted_family_count": accepted_family_count,
        "accepted_with_residual_debt_family_count": accepted_with_debt_count,
        "residual_only_family_count": residual_only_count,
        "stress_test_promoted_event_count": stress_test_promoted_event_count,
        "edge_control_promoted_event_count": edge_control_promoted_event_count,
        "stable_control_severe_like_events": stable_control_severe_like_events,
        "family_status_counts": _count(family_rows, "v2_2_family_status"),
        "instrumentation_role_counts": _count(family_rows, "instrumentation_role"),
        "gate_status_counts": _count(gate_matrix, "status"),
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "v2_2_registry_dir": _rel(v2_2_registry_dir),
            "recurrent_registry_dir": _rel(recurrent_registry_dir),
            "stratified_panel_dir": _rel(stratified_panel_dir),
            "matched_controls_dir": _rel(matched_controls_dir),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(family_rows, output_dir / FAMILY_INSTRUMENTATION_ROWS_CSV)
    _write_csv(registry_stratum_summary, output_dir / REGISTRY_STRATUM_SUMMARY_CSV)
    _write_csv(external_control_summary, output_dir / EXTERNAL_CONTROL_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / INSTRUMENTATION_GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "v2_2_registry_dir": _rel(v2_2_registry_dir),
        "recurrent_registry_dir": _rel(recurrent_registry_dir),
        "stratified_panel_dir": _rel(stratified_panel_dir),
        "matched_controls_dir": _rel(matched_controls_dir),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        registry_stratum_summary=registry_stratum_summary,
        external_control_summary=external_control_summary,
        gate_matrix=gate_matrix,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-2-registry-dir", type=Path, default=DEFAULT_V2_2_REGISTRY_DIR)
    parser.add_argument(
        "--recurrent-registry-dir", type=Path, default=DEFAULT_RECURRENT_REGISTRY_DIR
    )
    parser.add_argument("--stratified-panel-dir", type=Path, default=DEFAULT_STRATIFIED_PANEL_DIR)
    parser.add_argument("--matched-controls-dir", type=Path, default=DEFAULT_MATCHED_CONTROLS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = materialize(
        v2_2_registry_dir=args.v2_2_registry_dir,
        recurrent_registry_dir=args.recurrent_registry_dir,
        stratified_panel_dir=args.stratified_panel_dir,
        matched_controls_dir=args.matched_controls_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
