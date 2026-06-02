#!/usr/bin/env python3
"""Review the two post-v2.2 NanoClustering basin-definition options.

The review compares:

1. continue refining the remaining second-axis and joint-axis residuals;
2. freeze v2.2 as the operational basin-definition surface and move to the
   next instrumentation/design step.

This is a read-only analysis over existing membership-derived artifacts. It
does not run clustering, execute optimizer routes, promote wall/pathway claims,
or inspect basin quality/cost.
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
DEFAULT_AXIS_RULE_CANDIDATES_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_1_axis_rule_candidates_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_2_next_step_options_20260531"
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
AXIS_RULE_TARGET_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_target_axis_rows.csv"
)
AXIS_RULE_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_1_axis_rule_candidate_subfamily_rows.csv"
)

REMAINING_AXIS_TARGET_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_remaining_axis_target_rows.csv"
)
REMAINING_BEST_AXIS_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_remaining_best_axis_rows.csv"
)
REMAINING_BEST_RECOVERED_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_2_remaining_best_recovered_subfamily_rows.csv"
)
RESIDUAL_QUEUE_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_2_residual_queue_summary.csv"
)
OPTION_DECISION_MATRIX_CSV = (
    "nanoclustering_definition_core_v2_2_option_decision_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v2_2_next_step_options_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_2_next_step_options_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_2_next_step_options_config.json"

DEFINITION_QUEUE_STATUSES = {
    "second_axis_definition_queue",
    "joint_axis_definition_queue",
}
RECOVERED_RESULT = "candidate_recovered_coherent_endpoint_vector_subfamily"
CLAIM_BOUNDARY = (
    "Definition-core v2.2 next-step option review only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search "
    "claim."
)


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


def _coverage_read(coverage_share: float) -> str:
    if coverage_share >= 0.88:
        return "broad_operational_definition_surface"
    if coverage_share >= 0.80:
        return "usable_but_definition_debt_visible"
    return "definition_surface_still_sparse"


def _candidate_read(row: pd.Series) -> str:
    axis_read = str(row["candidate_axis_read"])
    if axis_read == "candidate_axis_recovers_most_events":
        return "strict_candidate_for_future_promotion_review"
    if axis_read == "candidate_axis_recovers_partial_events":
        return "partial_thin_definition_signal_not_ready"
    return "no_current_axis_recovery"


def _best_axis_rows(remaining_targets: pd.DataFrame) -> pd.DataFrame:
    if remaining_targets.empty:
        return pd.DataFrame()
    rows = (
        remaining_targets.sort_values(
            [
                "target_unit_id",
                "candidate_recovered_event_count",
                "candidate_recovered_subfamily_count",
                "candidate_tiny_event_count",
                "candidate_axis",
            ],
            ascending=[True, False, False, True, True],
        )
        .drop_duplicates("target_unit_id")
        .copy()
    )
    rows["next_step_candidate_read"] = rows.apply(_candidate_read, axis=1)
    rows["promotion_gate_status"] = rows["candidate_axis_read"].map(
        {
            "candidate_axis_recovers_most_events": "review_only_not_auto_promoted",
            "candidate_axis_recovers_partial_events": "blocked_partial_axis_recovery",
            "candidate_axis_no_coherent_recovery": "blocked_no_axis_recovery",
        }
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _remaining_axis_targets(
    *,
    residual_queue: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    definition_residual = residual_queue[
        residual_queue["definition_core_v2_2_queue_status"].isin(DEFINITION_QUEUE_STATUSES)
    ].copy()
    remaining_ids = set(definition_residual["audit_id"].astype(str))
    rows = target_rows[target_rows["target_unit_id"].astype(str).isin(remaining_ids)].copy()
    rows = rows.merge(
        definition_residual[
            [
                "audit_id",
                "definition_core_v2_2_queue_status",
                "definition_core_v2_2_queue_read",
            ]
        ],
        left_on="target_unit_id",
        right_on="audit_id",
        how="left",
        validate="many_to_one",
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.drop(columns=["audit_id"])


def _best_recovered_subfamilies(
    *,
    best_axis_rows: pd.DataFrame,
    subfamily_rows: pd.DataFrame,
) -> pd.DataFrame:
    if best_axis_rows.empty:
        return pd.DataFrame()
    keys = best_axis_rows[["target_unit_id", "candidate_axis"]].copy()
    rows = subfamily_rows.merge(
        keys,
        on=["target_unit_id", "candidate_axis"],
        how="inner",
        validate="many_to_one",
    )
    rows = rows[rows["candidate_definition_result"].eq(RECOVERED_RESULT)].copy()
    if rows.empty:
        return rows
    rows["support_depth_bucket"] = pd.cut(
        rows["event_count"],
        bins=[0, 1, 2, 4, 10**9],
        labels=["singleton", "thin_2", "moderate_3_to_4", "deep_ge5"],
        right=True,
    ).astype(str)
    rows["promotion_gate_status"] = "blocked_partial_or_no_target_level_recovery"
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["target_unit_id", "candidate_axis", "event_count"], ascending=[True, True, False])


def _residual_summary(residual_queue: pd.DataFrame) -> pd.DataFrame:
    rows = (
        residual_queue.groupby(["definition_core_v2_2_queue_status"], as_index=False)
        .agg(
            residual_row_count=("audit_id", "size"),
            residual_event_count=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
        )
        .sort_values(["residual_event_count", "definition_core_v2_2_queue_status"], ascending=[False, True])
    )
    rows["definition_debt_class"] = rows["definition_core_v2_2_queue_status"].map(
        {
            "support_depth_tiny_holdout": "support_floor_debt",
            "exception_axis_tiny_holdout": "support_floor_debt",
            "second_axis_definition_queue": "rule_design_debt",
            "joint_axis_definition_queue": "rule_design_debt",
            "rule_edge_definition_queue": "rule_edge_debt",
        }
    ).fillna("definition_debt")
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _option_decision_matrix(
    *,
    registry: pd.DataFrame,
    residual_queue: pd.DataFrame,
    best_axis_rows: pd.DataFrame,
    recovered_subfamilies: pd.DataFrame,
) -> pd.DataFrame:
    primitive_events = int(registry["event_count"].sum())
    residual_events = int(residual_queue["event_count"].sum())
    universe_events = primitive_events + residual_events
    coverage_share = primitive_events / universe_events if universe_events else 0.0
    definition_queue = residual_queue[
        residual_queue["definition_core_v2_2_queue_status"].isin(DEFINITION_QUEUE_STATUSES)
    ]
    best_recovered_events = (
        int(best_axis_rows["candidate_recovered_event_count"].sum())
        if not best_axis_rows.empty
        else 0
    )
    best_source_events = (
        int(best_axis_rows["source_event_count"].sum()) if not best_axis_rows.empty else 0
    )
    most_target_count = (
        int(best_axis_rows["candidate_axis_read"].eq("candidate_axis_recovers_most_events").sum())
        if not best_axis_rows.empty
        else 0
    )
    partial_target_count = (
        int(best_axis_rows["candidate_axis_read"].eq("candidate_axis_recovers_partial_events").sum())
        if not best_axis_rows.empty
        else 0
    )
    no_recovery_target_count = (
        int(best_axis_rows["candidate_axis_read"].eq("candidate_axis_no_coherent_recovery").sum())
        if not best_axis_rows.empty
        else 0
    )
    thin_recovered_events = (
        int(
            recovered_subfamilies.loc[
                recovered_subfamilies["event_count"].le(2),
                "event_count",
            ].sum()
        )
        if not recovered_subfamilies.empty
        else 0
    )
    non_tiny_residual_events = int(
        residual_queue[
            residual_queue["definition_core_v2_2_queue_status"].isin(
                [
                    "second_axis_definition_queue",
                    "joint_axis_definition_queue",
                    "rule_edge_definition_queue",
                ]
            )
        ]["event_count"].sum()
    )
    tiny_residual_events = residual_events - non_tiny_residual_events
    rows = [
        {
            "option_id": "A_continue_second_joint_definition_now",
            "option_read": "continue definition refinement before freezing",
            "evidence_summary": (
                f"{len(best_axis_rows)} second/joint targets cover {best_source_events} "
                f"events; current best axes recover {best_recovered_events}; "
                f"most-event target count is {most_target_count}."
            ),
            "positive_signal": (
                f"{partial_target_count} targets have partial recovery; recovered "
                f"subfamily event count is {best_recovered_events}."
            ),
            "blocking_signal": (
                f"{no_recovery_target_count} targets have no coherent recovery and "
                f"{thin_recovered_events} recovered events are in support-2 subfamilies."
            ),
            "decision": "do_not_promote_v2_3_from_current_axis_candidates",
            "next_action": (
                "keep as residual rule-design ledger; inspect failure modes before "
                "any new decomposition rule"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "option_id": "B_freeze_v2_2_operational_definition",
            "option_read": "freeze v2.2 as the current operational basin-definition surface",
            "evidence_summary": (
                f"v2.2 covers {primitive_events}/{universe_events} events "
                f"({coverage_share:.3f}) with {registry['primitive_id'].nunique()} "
                "non-overlapping primitives."
            ),
            "positive_signal": (
                "event universe is preserved; confidence tiers and residual debt are "
                "explicit; no route, wall, quality, or cost fields are promoted"
            ),
            "blocking_signal": (
                f"{residual_events} events remain residual, including "
                f"{non_tiny_residual_events} non-tiny rule-design/rule-edge events "
                f"and {tiny_residual_events} tiny holdout events."
            ),
            "decision": "recommended_current_freeze_with_residual_debt_ledger",
            "next_action": (
                "prepare downstream instrumentation over accepted v2.2 primitives, "
                "while carrying residual queue as exclusion/caveat metadata"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["coverage_read"] = _coverage_read(coverage_share)
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
    registry: pd.DataFrame,
    residual_summary: pd.DataFrame,
    remaining_targets: pd.DataFrame,
    best_axis_rows: pd.DataFrame,
    recovered_subfamilies: pd.DataFrame,
    option_matrix: pd.DataFrame,
) -> None:
    primitive_events = int(registry["event_count"].sum())
    residual_events = int(residual_summary["residual_event_count"].sum())
    universe_events = primitive_events + residual_events
    best_source_events = int(best_axis_rows["source_event_count"].sum()) if not best_axis_rows.empty else 0
    best_recovered_events = (
        int(best_axis_rows["candidate_recovered_event_count"].sum())
        if not best_axis_rows.empty
        else 0
    )
    text = [
        "# NanoClustering Definition-Core V2.2 Next-Step Option Review",
        "",
        f"- primitive_event_coverage: `{primitive_events}/{universe_events}`",
        f"- residual_definition_events: `{residual_events}`",
        f"- remaining_second_joint_targets: `{best_axis_rows['target_unit_id'].nunique() if not best_axis_rows.empty else 0}`",
        f"- remaining_second_joint_events: `{best_source_events}`",
        f"- current_best_axis_recovered_events: `{best_recovered_events}`",
        f"- current_best_axis_most_event_targets: `{int(best_axis_rows['candidate_axis_read'].eq('candidate_axis_recovers_most_events').sum()) if not best_axis_rows.empty else 0}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Option Matrix",
        "",
        _markdown_table(
            option_matrix,
            [
                "option_id",
                "option_read",
                "evidence_summary",
                "blocking_signal",
                "decision",
                "next_action",
            ],
            max_rows=10,
        ),
        "",
        "## Residual Queue Summary",
        "",
        _markdown_table(
            residual_summary,
            [
                "definition_core_v2_2_queue_status",
                "definition_debt_class",
                "residual_row_count",
                "residual_event_count",
                "source_family_count",
            ],
            max_rows=20,
        ),
        "",
        "## Remaining Best Axis Rows",
        "",
        _markdown_table(
            best_axis_rows,
            [
                "target_unit_id",
                "rule_scope",
                "source_event_count",
                "candidate_axis",
                "candidate_axis_read",
                "candidate_recovered_event_count",
                "candidate_tiny_event_count",
                "candidate_unresolved_event_count",
                "promotion_gate_status",
            ],
            max_rows=40,
        ),
        "",
        "## Best-Axis Recovered Subfamilies",
        "",
        _markdown_table(
            recovered_subfamilies,
            [
                "target_unit_id",
                "candidate_subfamily_id",
                "candidate_axis",
                "event_count",
                "support_depth_bucket",
                "candidate_subfamily_vector_class",
                "candidate_subfamily_coherence_status",
            ],
            max_rows=40,
        ),
        "",
        "## Read",
        "",
        "- Continuing definition work is not ready for immediate v2.3 promotion: no remaining second/joint target has a best axis that recovers most events.",
        "- The positive residual signal is thin and partial: current best axes recover 12 of 44 remaining second/joint events.",
        "- Freezing v2.2 is the cleaner next move if paired with an explicit residual-debt ledger and downstream instrumentation gates.",
        "- This review does not change the basin registry and does not supply wall/pathway, quality, cost, or directed-search evidence.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_2_registry_dir: Path,
    axis_rule_candidates_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _read_csv(v2_2_registry_dir / V2_2_PRIMITIVE_REGISTRY_CSV)
    event_rows = _read_csv(v2_2_registry_dir / V2_2_PRIMITIVE_EVENT_ROWS_CSV)
    residual_queue = _read_csv(v2_2_registry_dir / V2_2_RESIDUAL_DEFINITION_QUEUE_CSV)
    target_rows = _read_csv(axis_rule_candidates_dir / AXIS_RULE_TARGET_ROWS_CSV)
    subfamily_rows = _read_csv(axis_rule_candidates_dir / AXIS_RULE_SUBFAMILY_ROWS_CSV)

    remaining_targets = _remaining_axis_targets(
        residual_queue=residual_queue,
        target_rows=target_rows,
    )
    best_axis_rows = _best_axis_rows(remaining_targets)
    recovered_subfamilies = _best_recovered_subfamilies(
        best_axis_rows=best_axis_rows,
        subfamily_rows=subfamily_rows,
    )
    residual_summary = _residual_summary(residual_queue)
    option_matrix = _option_decision_matrix(
        registry=registry,
        residual_queue=residual_queue,
        best_axis_rows=best_axis_rows,
        recovered_subfamilies=recovered_subfamilies,
    )

    primitive_events = int(registry["event_count"].sum())
    residual_events = int(residual_queue["event_count"].sum())
    if len(event_rows) != primitive_events:
        raise ValueError("v2.2 event row count must match registry event sum")
    if event_rows["event_id"].duplicated().any():
        raise ValueError("v2.2 primitive event rows must be non-overlapping")
    if primitive_events + residual_events != 1026:
        raise ValueError("expected v2.2 primitive plus residual universe to equal 1026")
    if not best_axis_rows.empty and best_axis_rows[
        "candidate_axis_read"
    ].eq("candidate_axis_recovers_most_events").any():
        raise ValueError("unexpected most-event remaining target; review promotion gate")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(remaining_targets, output_dir / REMAINING_AXIS_TARGET_ROWS_CSV)
    _write_csv(best_axis_rows, output_dir / REMAINING_BEST_AXIS_ROWS_CSV)
    _write_csv(
        recovered_subfamilies,
        output_dir / REMAINING_BEST_RECOVERED_SUBFAMILY_ROWS_CSV,
    )
    _write_csv(residual_summary, output_dir / RESIDUAL_QUEUE_SUMMARY_CSV)
    _write_csv(option_matrix, output_dir / OPTION_DECISION_MATRIX_CSV)
    _write_report(
        output_dir=output_dir,
        registry=registry,
        residual_summary=residual_summary,
        remaining_targets=remaining_targets,
        best_axis_rows=best_axis_rows,
        recovered_subfamilies=recovered_subfamilies,
        option_matrix=option_matrix,
    )

    best_recovered_events = (
        int(best_axis_rows["candidate_recovered_event_count"].sum())
        if not best_axis_rows.empty
        else 0
    )
    summary = {
        "ok": True,
        "v2_2_registry_dir": _rel(v2_2_registry_dir),
        "axis_rule_candidates_dir": _rel(axis_rule_candidates_dir),
        "output_dir": _rel(output_dir),
        "primitive_event_count": primitive_events,
        "residual_event_count": residual_events,
        "definition_universe_event_count": primitive_events + residual_events,
        "primitive_coverage_share": primitive_events / (primitive_events + residual_events),
        "remaining_second_joint_target_count": int(best_axis_rows["target_unit_id"].nunique())
        if not best_axis_rows.empty
        else 0,
        "remaining_second_joint_event_count": int(best_axis_rows["source_event_count"].sum())
        if not best_axis_rows.empty
        else 0,
        "remaining_best_axis_recovered_event_count": best_recovered_events,
        "remaining_best_axis_recovered_event_share": best_recovered_events
        / float(best_axis_rows["source_event_count"].sum())
        if not best_axis_rows.empty
        else 0.0,
        "remaining_best_axis_read_counts": _count(best_axis_rows, "candidate_axis_read"),
        "residual_queue_status_counts": _count(
            residual_queue,
            "definition_core_v2_2_queue_status",
        ),
        "option_decisions": {
            row["option_id"]: row["decision"] for _, row in option_matrix.iterrows()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "remaining_axis_target_rows_csv": _rel(
                output_dir / REMAINING_AXIS_TARGET_ROWS_CSV
            ),
            "remaining_best_axis_rows_csv": _rel(
                output_dir / REMAINING_BEST_AXIS_ROWS_CSV
            ),
            "remaining_best_recovered_subfamily_rows_csv": _rel(
                output_dir / REMAINING_BEST_RECOVERED_SUBFAMILY_ROWS_CSV
            ),
            "residual_queue_summary_csv": _rel(output_dir / RESIDUAL_QUEUE_SUMMARY_CSV),
            "option_decision_matrix_csv": _rel(output_dir / OPTION_DECISION_MATRIX_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "v2_2_registry_dir": _rel(v2_2_registry_dir),
        "axis_rule_candidates_dir": _rel(axis_rule_candidates_dir),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
        "option_review_rule": (
            "Compare current residual definition-recovery evidence against freezing "
            "v2.2 as the operational definition surface. Do not promote residual "
            "axis candidates without a most-event target-level rule signal."
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
    parser.add_argument("--v2-2-registry-dir", type=Path, default=DEFAULT_V2_2_REGISTRY_DIR)
    parser.add_argument(
        "--axis-rule-candidates-dir",
        type=Path,
        default=DEFAULT_AXIS_RULE_CANDIDATES_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        v2_2_registry_dir=args.v2_2_registry_dir.resolve(),
        axis_rule_candidates_dir=args.axis_rule_candidates_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
