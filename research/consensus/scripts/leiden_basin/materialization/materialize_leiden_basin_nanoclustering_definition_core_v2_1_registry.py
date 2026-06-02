#!/usr/bin/env python3
"""Materialize the NanoClustering definition-core v2.1 registry.

V2.1 keeps the v2 primitive definition fixed, adds support-depth confidence
tiers, and separates axis-exception candidates into an audit ledger. It does
not run clustering, execute optimizer routes, promote wall/pathway claims, or
inspect basin quality/cost.
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
DEFAULT_V2_REGISTRY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_registry_20260530"
)
DEFAULT_AUDIT_SURFACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_audit_surface_review_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_1_registry_20260531"
)

V2_PRIMITIVE_REGISTRY_CSV = "nanoclustering_definition_core_v2_primitive_registry.csv"
V2_PRIMITIVE_EVENT_ROWS_CSV = "nanoclustering_definition_core_v2_primitive_event_rows.csv"
PRIMARY_VS_BEST_AXIS_ROWS_CSV = (
    "nanoclustering_definition_core_v2_primary_vs_best_axis_rows.csv"
)
RULE_REVISION_CANDIDATES_CSV = (
    "nanoclustering_definition_core_v2_rule_revision_candidates.csv"
)
RESIDUAL_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_residual_subfamily_rows.csv"
)

V2_1_PRIMITIVE_REGISTRY_CSV = (
    "nanoclustering_definition_core_v2_1_primitive_registry.csv"
)
V2_1_PRIMITIVE_EVENT_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_primitive_event_rows.csv"
)
V2_1_AXIS_EXCEPTION_LEDGER_CSV = (
    "nanoclustering_definition_core_v2_1_axis_exception_ledger.csv"
)
V2_1_RESIDUAL_DEFINITION_QUEUE_CSV = (
    "nanoclustering_definition_core_v2_1_residual_definition_queue.csv"
)
V2_1_CONFIDENCE_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_1_confidence_summary.csv"
)
V2_1_AXIS_RULE_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_summary.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v2_1_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_1_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_1_config.json"

CLAIM_BOUNDARY = (
    "Definition-core v2.1 registry only; no route execution, wall/pathway "
    "promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_definition_core_v2_1_registry"


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


def _support_depth_tier(row: pd.Series) -> str:
    event_count = int(row["event_count"])
    primitive_type = str(row["primitive_type"])
    if primitive_type == "v1_coherent_family":
        if event_count >= 6:
            return "accepted_v1_family_support_ge6"
        if event_count >= 4:
            return "accepted_v1_family_support_4_to_5"
        return "accepted_v1_family_support_2_to_3"
    if event_count >= 5:
        return "recovered_deep_repeat_ge5"
    if event_count >= 3:
        return "recovered_moderate_repeat_3_to_4"
    return "recovered_thin_repeat_2"


def _support_confidence_tier(row: pd.Series) -> str:
    primitive_type = str(row["primitive_type"])
    event_count = int(row["event_count"])
    if primitive_type == "v1_coherent_family":
        return "v2_1_family_coherent_confidence"
    if event_count >= 5:
        return "v2_1_deep_recovered_confidence"
    if event_count >= 3:
        return "v2_1_moderate_recovered_confidence"
    return "v2_1_thin_recovered_confidence"


def _support_depth_read(row: pd.Series) -> str:
    primitive_type = str(row["primitive_type"])
    event_count = int(row["event_count"])
    if primitive_type == "v1_coherent_family":
        return "accepted v1 coherent family; support depth recorded as confidence metadata"
    if event_count >= 5:
        return "recovered coherent subfamily with deeper repeated support"
    if event_count >= 3:
        return "recovered coherent subfamily with moderate repeated support"
    return "recovered coherent subfamily with minimal repeated support; keep but treat as thin"


def _axis_rule_status(row: pd.Series, axis_lookup: dict[str, str]) -> str:
    if str(row["primitive_type"]) == "v1_coherent_family":
        return "not_applicable_v1_coherent_family"
    read = axis_lookup.get(str(row["source_family_id"]), "primary_axis_sufficient_under_current_rule")
    if read == "marginal_best_axis_gain":
        return "primary_axis_retained_with_secondary_axis_gain"
    return read


def _definition_confidence_tier(row: pd.Series) -> str:
    support = str(row["support_confidence_tier"])
    axis = str(row["axis_rule_status"])
    if axis == "primary_axis_retained_with_secondary_axis_gain":
        return "v2_1_axis_caveat_primitive"
    return support


def _definition_read(row: pd.Series) -> str:
    confidence = str(row["definition_confidence_tier"])
    if confidence == "v2_1_axis_caveat_primitive":
        return "retained v2 primitive, but source family has a secondary-axis gain caveat"
    if confidence == "v2_1_thin_recovered_confidence":
        return "retained v2 primitive with thin repeated support; confidence tier is low"
    if confidence == "v2_1_moderate_recovered_confidence":
        return "retained v2 primitive with moderate repeated support"
    if confidence == "v2_1_deep_recovered_confidence":
        return "retained v2 primitive with deeper repeated support"
    return "retained accepted v1 coherent family"


def _primitive_registry(
    *,
    primitive_registry: pd.DataFrame,
    primary_vs_best: pd.DataFrame,
) -> pd.DataFrame:
    axis_lookup = {
        str(row["family_id"]): str(row["axis_decision_read"])
        for _, row in primary_vs_best.iterrows()
    }
    rows = primitive_registry.copy()
    rows["definition_core_v2_1_status"] = "definition_core_v2_1_retained_primitive"
    rows["support_depth_tier"] = rows.apply(_support_depth_tier, axis=1)
    rows["support_confidence_tier"] = rows.apply(_support_confidence_tier, axis=1)
    rows["support_depth_read"] = rows.apply(_support_depth_read, axis=1)
    rows["axis_rule_status"] = rows.apply(lambda row: _axis_rule_status(row, axis_lookup), axis=1)
    rows["definition_confidence_tier"] = rows.apply(_definition_confidence_tier, axis=1)
    rows["definition_core_v2_1_read"] = rows.apply(_definition_read, axis=1)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "primitive_id",
        "primitive_type",
        "definition_core_v2_1_status",
        "definition_confidence_tier",
        "support_depth_tier",
        "support_confidence_tier",
        "axis_rule_status",
        "definition_core_v2_1_read",
        "support_depth_read",
        "source_family_id",
        "source_definition_core_v1_status",
        "primitive_vector_class",
        "primitive_coherence_status",
        "branch",
        "boundary_family_tier",
        "event_count",
        "source_event_count",
        "event_count_share_of_source_family",
        "decomposition_axis",
        "decomposition_key",
        "dominant_split_vector_class",
        "dominant_host_context_class",
        "dominant_shape_core_signature",
        "dominant_host_handle_id",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder].sort_values(
        [
            "definition_confidence_tier",
            "primitive_type",
            "boundary_family_tier",
            "event_count",
            "primitive_id",
        ],
        ascending=[True, True, True, False, True],
    )


def _primitive_event_rows(
    *,
    primitive_event_rows: pd.DataFrame,
    v2_1_registry: pd.DataFrame,
) -> pd.DataFrame:
    event_rows = primitive_event_rows.merge(
        v2_1_registry[
            [
                "primitive_id",
                "definition_core_v2_1_status",
                "definition_confidence_tier",
                "support_depth_tier",
                "support_confidence_tier",
                "axis_rule_status",
                "definition_core_v2_1_read",
            ]
        ],
        on="primitive_id",
        how="left",
        validate="many_to_one",
    )
    if event_rows["definition_core_v2_1_status"].isna().any():
        raise ValueError("missing v2.1 primitive metadata for event rows")
    event_rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    event_rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    event_rows["quality_cost_status"] = QUALITY_COST_STATUS
    event_rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "primitive_id",
        "primitive_type",
        "definition_core_v2_1_status",
        "definition_confidence_tier",
        "support_depth_tier",
        "axis_rule_status",
        "source_family_id",
        "event_id",
        "branch",
        "boundary_family_tier",
        "split_vector_class",
        "host_context_class",
        "shape_core_signature",
        "comparison_seed",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in event_rows.columns if column not in preferred]
    return event_rows.loc[:, preferred + remainder]


def _axis_exception_status(row: pd.Series) -> str:
    read = str(row["axis_decision_read"])
    if read == "strong_axis_exception_candidate":
        return "strong_axis_exception_candidate_not_promoted"
    if read == "weak_axis_exception_candidate":
        return "weak_axis_exception_diagnostic_not_promoted"
    return "marginal_secondary_axis_gain_primary_retained"


def _axis_exception_next_action(row: pd.Series) -> str:
    read = str(row["axis_decision_read"])
    if read == "strong_axis_exception_candidate":
        return "materialize_event_level_exception_axis_before_any_promotion"
    if read == "weak_axis_exception_candidate":
        return "hold_as_diagnostic_until_more_support_or_rule_edge_review"
    return "retain_primary_axis_and_record_best_axis_as_secondary_check"


def _axis_exception_ledger(rule_candidates: pd.DataFrame) -> pd.DataFrame:
    rows = rule_candidates.copy()
    rows["definition_core_v2_1_exception_status"] = rows.apply(
        _axis_exception_status,
        axis=1,
    )
    rows["definition_core_v2_1_registry_effect"] = rows[
        "definition_core_v2_1_exception_status"
    ].map(
        {
            "strong_axis_exception_candidate_not_promoted": "outside_v2_1_primitive_registry",
            "weak_axis_exception_diagnostic_not_promoted": "outside_v2_1_primitive_registry",
            "marginal_secondary_axis_gain_primary_retained": (
                "existing_primary_primitive_retained_with_axis_caveat"
            ),
        }
    )
    rows["next_definition_action"] = rows.apply(_axis_exception_next_action, axis=1)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "family_id",
        "definition_core_v1_status",
        "definition_core_v2_1_exception_status",
        "definition_core_v2_1_registry_effect",
        "next_definition_action",
        "source_event_count",
        "primary_axis",
        "best_axis",
        "primary_recovered_event_count",
        "best_recovered_event_count",
        "best_recovered_event_share",
        "best_gain_event_count",
        "axis_decision_read",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder].sort_values(
        [
            "definition_core_v2_1_exception_status",
            "best_gain_event_count",
            "family_id",
        ],
        ascending=[True, False, True],
    )


def _residual_queue(residual_rows: pd.DataFrame) -> pd.DataFrame:
    rows = residual_rows.copy()
    rows["definition_core_v2_1_queue_status"] = rows["residual_definition_read"].map(
        {
            "single_event_or_tiny_support_do_not_promote": "support_depth_tiny_holdout",
            "needs_second_axis_for_shape_or_host_signature_variation": (
                "second_axis_definition_queue"
            ),
            "needs_joint_shape_host_or_host_signature_review": "joint_axis_definition_queue",
            "rule_edge_signature_review_before_any_promotion": "rule_edge_definition_queue",
        }
    ).fillna("definition_refinement_queue")
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "audit_id",
        "source_family_id",
        "source_definition_core_v1_status",
        "definition_core_v2_1_queue_status",
        "definition_core_v2_audit_status",
        "residual_definition_read",
        "event_count",
        "decomposition_axis",
        "decomposition_key",
        "subfamily_coherence_status",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows.loc[:, preferred + remainder].sort_values(
        [
            "definition_core_v2_1_queue_status",
            "source_definition_core_v1_status",
            "event_count",
            "audit_id",
        ],
        ascending=[True, True, False, True],
    )


def _confidence_summary(v2_1_registry: pd.DataFrame) -> pd.DataFrame:
    rows = (
        v2_1_registry.groupby(
            [
                "primitive_type",
                "definition_confidence_tier",
                "support_depth_tier",
                "axis_rule_status",
            ],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
            median_event_count=("event_count", "median"),
        )
        .sort_values(
            ["primitive_type", "definition_confidence_tier", "event_count_sum"],
            ascending=[True, True, False],
        )
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _axis_rule_summary(
    *,
    v2_1_registry: pd.DataFrame,
    axis_exception_ledger: pd.DataFrame,
    residual_queue: pd.DataFrame,
) -> pd.DataFrame:
    primitive_rows = (
        v2_1_registry.groupby(["axis_rule_status"], as_index=False)
        .agg(
            row_count=("primitive_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
        )
        .rename(columns={"axis_rule_status": "axis_rule_or_queue_status"})
    )
    primitive_rows["summary_scope"] = "retained_primitive_axis_rule_status"
    exception_rows = (
        axis_exception_ledger.groupby(["definition_core_v2_1_exception_status"], as_index=False)
        .agg(
            row_count=("family_id", "size"),
            event_count_sum=("source_event_count", "sum"),
            source_family_count=("family_id", "nunique"),
        )
        .rename(columns={"definition_core_v2_1_exception_status": "axis_rule_or_queue_status"})
    )
    exception_rows["summary_scope"] = "axis_exception_ledger_status"
    residual_rows = (
        residual_queue.groupby(["definition_core_v2_1_queue_status"], as_index=False)
        .agg(
            row_count=("audit_id", "size"),
            event_count_sum=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
        )
        .rename(columns={"definition_core_v2_1_queue_status": "axis_rule_or_queue_status"})
    )
    residual_rows["summary_scope"] = "residual_definition_queue_status"
    rows = pd.concat([primitive_rows, exception_rows, residual_rows], ignore_index=True)
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["summary_scope", "event_count_sum"], ascending=[True, False])


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
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    suffix: list[str] = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    v2_1_registry: pd.DataFrame,
    axis_exception_ledger: pd.DataFrame,
    residual_queue: pd.DataFrame,
    confidence_summary: pd.DataFrame,
    axis_rule_summary: pd.DataFrame,
) -> None:
    text = [
        "# NanoClustering Definition-Core V2.1 Registry",
        "",
        f"- primitive_rows: `{len(v2_1_registry)}`",
        f"- primitive_event_count: `{int(v2_1_registry['event_count'].sum())}`",
        f"- source_family_count: `{v2_1_registry['source_family_id'].nunique()}`",
        f"- axis_exception_ledger_rows: `{len(axis_exception_ledger)}`",
        f"- residual_definition_queue_rows: `{len(residual_queue)}`",
        f"- residual_definition_queue_events: `{int(residual_queue['event_count'].sum())}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Confidence Summary",
        "",
        _markdown_table(
            confidence_summary,
            [
                "primitive_type",
                "definition_confidence_tier",
                "support_depth_tier",
                "axis_rule_status",
                "primitive_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=50,
        ),
        "",
        "## Axis And Queue Summary",
        "",
        _markdown_table(
            axis_rule_summary,
            [
                "summary_scope",
                "axis_rule_or_queue_status",
                "row_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=50,
        ),
        "",
        "## Axis Exception Ledger",
        "",
        _markdown_table(
            axis_exception_ledger,
            [
                "family_id",
                "definition_core_v1_status",
                "definition_core_v2_1_exception_status",
                "definition_core_v2_1_registry_effect",
                "source_event_count",
                "primary_axis",
                "best_axis",
                "primary_recovered_event_count",
                "best_recovered_event_count",
                "best_recovered_event_share",
            ],
            max_rows=30,
        ),
        "",
        "## Residual Definition Queue",
        "",
        _markdown_table(
            residual_queue.groupby(
                [
                    "definition_core_v2_1_queue_status",
                    "source_definition_core_v1_status",
                    "residual_definition_read",
                ],
                as_index=False,
            )
            .agg(
                queue_row_count=("audit_id", "size"),
                event_count_sum=("event_count", "sum"),
                source_family_count=("source_family_id", "nunique"),
            )
            .sort_values(["definition_core_v2_1_queue_status", "event_count_sum"], ascending=[True, False]),
            [
                "definition_core_v2_1_queue_status",
                "source_definition_core_v1_status",
                "residual_definition_read",
                "queue_row_count",
                "event_count_sum",
                "source_family_count",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- V2.1 does not add new primitives beyond v2; it makes confidence and exceptions explicit.",
        "- Thin recovered primitives are retained as definition primitives, but downstream use should treat their confidence tier separately.",
        "- Strong axis exceptions remain outside the primitive registry until event-level exception-axis materialization is performed.",
        "- Residual definition queues remain definition work, not wall/pathway, quality, cost, or directed-search evidence.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_registry_dir: Path,
    audit_surface_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    primitive_registry = _read_csv(v2_registry_dir / V2_PRIMITIVE_REGISTRY_CSV)
    primitive_event_rows = _read_csv(v2_registry_dir / V2_PRIMITIVE_EVENT_ROWS_CSV)
    primary_vs_best = _read_csv(audit_surface_dir / PRIMARY_VS_BEST_AXIS_ROWS_CSV)
    rule_candidates = _read_csv(audit_surface_dir / RULE_REVISION_CANDIDATES_CSV)
    residual_rows = _read_csv(audit_surface_dir / RESIDUAL_SUBFAMILY_ROWS_CSV)

    v2_1_registry = _primitive_registry(
        primitive_registry=primitive_registry,
        primary_vs_best=primary_vs_best,
    )
    v2_1_event_rows = _primitive_event_rows(
        primitive_event_rows=primitive_event_rows,
        v2_1_registry=v2_1_registry,
    )
    axis_exception_ledger = _axis_exception_ledger(rule_candidates)
    residual_queue = _residual_queue(residual_rows)
    confidence_summary = _confidence_summary(v2_1_registry)
    axis_rule_summary = _axis_rule_summary(
        v2_1_registry=v2_1_registry,
        axis_exception_ledger=axis_exception_ledger,
        residual_queue=residual_queue,
    )

    if len(v2_1_registry) != len(primitive_registry):
        raise ValueError("v2.1 must not add or remove primitives")
    if len(v2_1_event_rows) != len(primitive_event_rows):
        raise ValueError("v2.1 event row count must match v2 primitive events")
    if int(v2_1_registry["event_count"].sum()) != len(v2_1_event_rows):
        raise ValueError("v2.1 registry event_count sum must match event rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(v2_1_registry, output_dir / V2_1_PRIMITIVE_REGISTRY_CSV)
    _write_csv(v2_1_event_rows, output_dir / V2_1_PRIMITIVE_EVENT_ROWS_CSV)
    _write_csv(axis_exception_ledger, output_dir / V2_1_AXIS_EXCEPTION_LEDGER_CSV)
    _write_csv(residual_queue, output_dir / V2_1_RESIDUAL_DEFINITION_QUEUE_CSV)
    _write_csv(confidence_summary, output_dir / V2_1_CONFIDENCE_SUMMARY_CSV)
    _write_csv(axis_rule_summary, output_dir / V2_1_AXIS_RULE_SUMMARY_CSV)
    _write_report(
        output_dir=output_dir,
        v2_1_registry=v2_1_registry,
        axis_exception_ledger=axis_exception_ledger,
        residual_queue=residual_queue,
        confidence_summary=confidence_summary,
        axis_rule_summary=axis_rule_summary,
    )

    summary = {
        "ok": True,
        "v2_registry_dir": _rel(v2_registry_dir),
        "audit_surface_dir": _rel(audit_surface_dir),
        "output_dir": _rel(output_dir),
        "primitive_row_count": int(len(v2_1_registry)),
        "primitive_event_row_count": int(len(v2_1_event_rows)),
        "primitive_event_count_sum": int(v2_1_registry["event_count"].sum()),
        "primitive_source_family_count": int(v2_1_registry["source_family_id"].nunique()),
        "definition_confidence_tier_counts": _count(
            v2_1_registry,
            "definition_confidence_tier",
        ),
        "support_depth_tier_counts": _count(v2_1_registry, "support_depth_tier"),
        "axis_rule_status_counts": _count(v2_1_registry, "axis_rule_status"),
        "axis_exception_ledger_count": int(len(axis_exception_ledger)),
        "axis_exception_status_counts": _count(
            axis_exception_ledger,
            "definition_core_v2_1_exception_status",
        ),
        "residual_definition_queue_row_count": int(len(residual_queue)),
        "residual_definition_queue_event_count": int(residual_queue["event_count"].sum()),
        "residual_definition_queue_status_counts": _count(
            residual_queue,
            "definition_core_v2_1_queue_status",
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "primitive_registry_csv": _rel(output_dir / V2_1_PRIMITIVE_REGISTRY_CSV),
            "primitive_event_rows_csv": _rel(output_dir / V2_1_PRIMITIVE_EVENT_ROWS_CSV),
            "axis_exception_ledger_csv": _rel(output_dir / V2_1_AXIS_EXCEPTION_LEDGER_CSV),
            "residual_definition_queue_csv": _rel(
                output_dir / V2_1_RESIDUAL_DEFINITION_QUEUE_CSV
            ),
            "confidence_summary_csv": _rel(output_dir / V2_1_CONFIDENCE_SUMMARY_CSV),
            "axis_rule_summary_csv": _rel(output_dir / V2_1_AXIS_RULE_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "v2_registry_dir": _rel(v2_registry_dir),
        "audit_surface_dir": _rel(audit_surface_dir),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
        "v2_1_rule": (
            "Retain v2 primitives, annotate support-depth confidence, and keep "
            "axis exceptions in a non-promoted ledger."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-registry-dir", type=Path, default=DEFAULT_V2_REGISTRY_DIR)
    parser.add_argument("--audit-surface-dir", type=Path, default=DEFAULT_AUDIT_SURFACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        v2_registry_dir=args.v2_registry_dir.resolve(),
        audit_surface_dir=args.audit_surface_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
