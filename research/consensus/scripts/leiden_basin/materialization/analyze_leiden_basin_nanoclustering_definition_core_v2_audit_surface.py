#!/usr/bin/env python3
"""Analyze the NanoClustering definition-core v2 audit surface.

This reads the v2 primitive registry and the v1 refinement decomposition to
separate support-depth fragility, primary-vs-best axis disagreements, and
residual definition-audit rows. It does not run clustering, execute routes,
promote wall/pathway claims, or inspect basin quality/cost.
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
DEFAULT_REFINEMENT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v1_refinement_queue_decomposition_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_definition_core_v2_audit_surface_review_20260531"
)

PRIMITIVE_REGISTRY_CSV = "nanoclustering_definition_core_v2_primitive_registry.csv"
AUDIT_QUEUE_ROWS_CSV = "nanoclustering_definition_core_v2_audit_queue_rows.csv"
SUBFAMILY_ROWS_CSV = "nanoclustering_definition_core_v1_refinement_subfamily_rows.csv"
AXIS_COMPARISON_ROWS_CSV = (
    "nanoclustering_definition_core_v1_refinement_axis_comparison_rows.csv"
)

SUPPORT_DEPTH_SUMMARY_CSV = (
    "nanoclustering_definition_core_v2_support_depth_summary.csv"
)
PRIMARY_VS_BEST_AXIS_ROWS_CSV = (
    "nanoclustering_definition_core_v2_primary_vs_best_axis_rows.csv"
)
RULE_REVISION_CANDIDATES_CSV = (
    "nanoclustering_definition_core_v2_rule_revision_candidates.csv"
)
RESIDUAL_SUBFAMILY_ROWS_CSV = (
    "nanoclustering_definition_core_v2_residual_subfamily_rows.csv"
)
SUMMARY_JSON = "nanoclustering_definition_core_v2_audit_surface_summary.json"
REPORT_MD = "nanoclustering_definition_core_v2_audit_surface_report.md"
CONFIG_JSON = "nanoclustering_definition_core_v2_audit_surface_config.json"

FULL_EVENT_COUNT = 1026
CLAIM_BOUNDARY = (
    "Definition-core v2 audit-surface review only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
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


def _support_depth_summary(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for min_recovered_events in [2, 3, 4, 5, 6]:
        retained = registry[
            registry["primitive_type"].eq("v1_coherent_family")
            | registry["event_count"].ge(min_recovered_events)
        ].copy()
        recovered = retained[retained["primitive_type"].eq("recovered_coherent_subfamily")]
        rows.append(
            {
                "min_recovered_events": min_recovered_events,
                "retained_primitive_count": int(len(retained)),
                "retained_v1_family_count": int(
                    retained["primitive_type"].eq("v1_coherent_family").sum()
                ),
                "retained_recovered_subfamily_count": int(len(recovered)),
                "retained_event_count": int(retained["event_count"].sum()),
                "retained_source_family_count": int(retained["source_family_id"].nunique()),
                "event_coverage_share": float(retained["event_count"].sum() / FULL_EVENT_COUNT),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    exact = registry[registry["primitive_type"].eq("recovered_coherent_subfamily")].copy()
    exact_rows = (
        exact.groupby("event_count", as_index=False)
        .agg(
            recovered_subfamily_count=("primitive_id", "size"),
            recovered_event_count_sum=("event_count", "sum"),
            recovered_source_family_count=("source_family_id", "nunique"),
        )
        .rename(columns={"event_count": "exact_recovered_event_count"})
    )
    summary = pd.DataFrame(rows)
    summary["summary_scope"] = "retained_under_min_recovered_event_threshold"
    exact_rows["summary_scope"] = "exact_recovered_event_count_distribution"
    for column in summary.columns:
        if column not in exact_rows:
            exact_rows[column] = ""
    for column in exact_rows.columns:
        if column not in summary:
            summary[column] = ""
    return pd.concat([summary[exact_rows.columns], exact_rows], ignore_index=True, sort=False)


def _primary_vs_best_axis(axis_rows: pd.DataFrame) -> pd.DataFrame:
    primary = axis_rows[axis_rows["is_primary_axis"]].copy()
    best = (
        axis_rows.sort_values(
            [
                "family_id",
                "recovered_coherent_event_count",
                "recovered_coherent_subfamily_count",
                "tiny_event_count",
            ],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("family_id")
        .copy()
    )
    rows = primary[
        [
            "family_id",
            "definition_core_v1_status",
            "boundary_family_tier",
            "family_vector_class",
            "source_event_count",
            "axis",
            "recovered_coherent_event_count",
            "recovered_coherent_event_share",
            "recovered_coherent_subfamily_count",
            "tiny_event_count",
        ]
    ].rename(
        columns={
            "axis": "primary_axis",
            "recovered_coherent_event_count": "primary_recovered_event_count",
            "recovered_coherent_event_share": "primary_recovered_event_share",
            "recovered_coherent_subfamily_count": "primary_recovered_subfamily_count",
            "tiny_event_count": "primary_tiny_event_count",
        }
    )
    rows = rows.merge(
        best[
            [
                "family_id",
                "axis",
                "recovered_coherent_event_count",
                "recovered_coherent_event_share",
                "recovered_coherent_subfamily_count",
                "tiny_event_count",
            ]
        ].rename(
            columns={
                "axis": "best_axis",
                "recovered_coherent_event_count": "best_recovered_event_count",
                "recovered_coherent_event_share": "best_recovered_event_share",
                "recovered_coherent_subfamily_count": "best_recovered_subfamily_count",
                "tiny_event_count": "best_tiny_event_count",
            }
        ),
        on="family_id",
        how="left",
        validate="one_to_one",
    )
    rows["best_gain_event_count"] = (
        rows["best_recovered_event_count"] - rows["primary_recovered_event_count"]
    )
    rows["best_gain_event_share"] = (
        rows["best_recovered_event_share"] - rows["primary_recovered_event_share"]
    )
    rows["axis_decision_read"] = rows.apply(_axis_decision_read, axis=1)
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["best_gain_event_count", "source_event_count", "family_id"],
        ascending=[False, False, True],
    )


def _axis_decision_read(row: pd.Series) -> str:
    if float(row["best_gain_event_count"]) <= 0:
        return "primary_axis_sufficient_under_current_rule"
    if float(row["primary_recovered_event_count"]) == 0 and float(
        row["best_recovered_event_share"]
    ) >= 0.75:
        return "strong_axis_exception_candidate"
    if float(row["primary_recovered_event_count"]) == 0:
        return "weak_axis_exception_candidate"
    return "marginal_best_axis_gain"


def _rule_revision_candidates(primary_vs_best: pd.DataFrame) -> pd.DataFrame:
    candidates = primary_vs_best[primary_vs_best["best_gain_event_count"].gt(0)].copy()
    candidates["rule_revision_read"] = candidates["axis_decision_read"].map(
        {
            "strong_axis_exception_candidate": (
                "review as possible explicit exception; do not promote without rule revision"
            ),
            "weak_axis_exception_candidate": (
                "keep as diagnostic; alternative axis recovers too little for direct promotion"
            ),
            "marginal_best_axis_gain": (
                "primary axis still works; note best-axis gain as secondary refinement"
            ),
        }
    )
    candidates["claim_boundary"] = CLAIM_BOUNDARY
    return candidates


def _residual_subfamily_rows(audit_rows: pd.DataFrame) -> pd.DataFrame:
    residual = audit_rows[
        audit_rows["event_count_basis"].eq("primary_subfamily_residual_additive")
    ].copy()
    residual["residual_definition_read"] = residual.apply(_residual_read, axis=1)
    residual["claim_boundary"] = CLAIM_BOUNDARY
    return residual.sort_values(
        [
            "definition_core_v2_audit_status",
            "source_definition_core_v1_status",
            "event_count",
            "audit_id",
        ],
        ascending=[True, True, False, True],
    )


def _residual_read(row: pd.Series) -> str:
    status = str(row["definition_core_v2_audit_status"])
    source_status = str(row["source_definition_core_v1_status"])
    if status == "diagnostic_tiny_subfamily_not_promoted":
        return "single_event_or_tiny_support_do_not_promote"
    if source_status == "host_coherent_split_mixed_subfamily":
        return "needs_second_axis_for_shape_or_host_signature_variation"
    if source_status == "split_coherent_host_variable_subfamily":
        return "needs_joint_shape_host_or_host_signature_review"
    if source_status == "heterogeneous_rule_edge_review":
        return "rule_edge_signature_review_before_any_promotion"
    return "definition_refinement_required"


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
    support_depth: pd.DataFrame,
    primary_vs_best: pd.DataFrame,
    rule_candidates: pd.DataFrame,
    residual_rows: pd.DataFrame,
) -> None:
    threshold_summary = support_depth[
        support_depth["summary_scope"].eq("retained_under_min_recovered_event_threshold")
    ]
    exact_summary = support_depth[
        support_depth["summary_scope"].eq("exact_recovered_event_count_distribution")
    ]
    residual_rollup = (
        residual_rows.groupby(
            [
                "source_definition_core_v1_status",
                "definition_core_v2_audit_status",
                "residual_definition_read",
            ],
            as_index=False,
        )
        .agg(
            residual_row_count=("audit_id", "size"),
            source_family_count=("source_family_id", "nunique"),
            event_count_sum=("event_count", "sum"),
            median_event_count=("event_count", "median"),
        )
        .sort_values(
            ["definition_core_v2_audit_status", "event_count_sum"],
            ascending=[True, False],
        )
    )
    gain_rollup = (
        primary_vs_best.groupby("axis_decision_read", as_index=False)
        .agg(
            family_count=("family_id", "size"),
            source_event_count_sum=("source_event_count", "sum"),
            primary_recovered_event_sum=("primary_recovered_event_count", "sum"),
            best_recovered_event_sum=("best_recovered_event_count", "sum"),
            best_gain_event_sum=("best_gain_event_count", "sum"),
        )
        .sort_values("best_gain_event_sum", ascending=False)
    )
    text = [
        "# NanoClustering Definition-Core V2 Audit Surface Review",
        "",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        f"- primary_vs_best_families: `{len(primary_vs_best)}`",
        f"- primary_recovered_events: `{int(primary_vs_best['primary_recovered_event_count'].sum())}`",
        f"- best_axis_recovered_events: `{int(primary_vs_best['best_recovered_event_count'].sum())}`",
        f"- best_axis_gain_events: `{int(primary_vs_best['best_gain_event_count'].sum())}`",
        f"- rule_revision_candidate_families: `{len(rule_candidates)}`",
        f"- residual_primary_subfamily_rows: `{len(residual_rows)}`",
        f"- residual_primary_subfamily_events: `{int(residual_rows['event_count'].sum())}`",
        "",
        "## Support Depth Sensitivity",
        "",
        _markdown_table(
            threshold_summary,
            [
                "min_recovered_events",
                "retained_primitive_count",
                "retained_recovered_subfamily_count",
                "retained_event_count",
                "retained_source_family_count",
                "event_coverage_share",
            ],
            max_rows=10,
        ),
        "",
        "## Exact Recovered Support Distribution",
        "",
        _markdown_table(
            exact_summary,
            [
                "exact_recovered_event_count",
                "recovered_subfamily_count",
                "recovered_event_count_sum",
                "recovered_source_family_count",
            ],
            max_rows=10,
        ),
        "",
        "## Primary vs Best Axis Rollup",
        "",
        _markdown_table(
            gain_rollup,
            [
                "axis_decision_read",
                "family_count",
                "source_event_count_sum",
                "primary_recovered_event_sum",
                "best_recovered_event_sum",
                "best_gain_event_sum",
            ],
            max_rows=10,
        ),
        "",
        "## Rule Revision Candidates",
        "",
        _markdown_table(
            rule_candidates,
            [
                "family_id",
                "definition_core_v1_status",
                "source_event_count",
                "primary_axis",
                "best_axis",
                "primary_recovered_event_count",
                "best_recovered_event_count",
                "best_recovered_event_share",
                "axis_decision_read",
            ],
            max_rows=30,
        ),
        "",
        "## Residual Definition Rows",
        "",
        _markdown_table(
            residual_rollup,
            [
                "source_definition_core_v1_status",
                "definition_core_v2_audit_status",
                "residual_definition_read",
                "residual_row_count",
                "source_family_count",
                "event_count_sum",
                "median_event_count",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- The v2 registry is inclusive: many recovered primitives are repeated but thin.",
        "- Raising the recovered-subfamily support floor rapidly reduces coverage, so support depth should be a confidence tier, not an immediate definition replacement.",
        "- Primary axes recover most queue evidence, but a small exception class has primary zero and nonzero alternative-axis recovery.",
        "- The largest residual definition issue is host-coherent split-mixed rows that still need a second axis, not a quality or pathway question.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    v2_registry_dir: Path,
    refinement_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    primitive_registry = _read_csv(v2_registry_dir / PRIMITIVE_REGISTRY_CSV)
    audit_rows = _read_csv(v2_registry_dir / AUDIT_QUEUE_ROWS_CSV)
    axis_rows = _read_csv(refinement_dir / AXIS_COMPARISON_ROWS_CSV)

    support_depth = _support_depth_summary(primitive_registry)
    primary_vs_best = _primary_vs_best_axis(axis_rows)
    rule_candidates = _rule_revision_candidates(primary_vs_best)
    residual_rows = _residual_subfamily_rows(audit_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(support_depth, output_dir / SUPPORT_DEPTH_SUMMARY_CSV)
    _write_csv(primary_vs_best, output_dir / PRIMARY_VS_BEST_AXIS_ROWS_CSV)
    _write_csv(rule_candidates, output_dir / RULE_REVISION_CANDIDATES_CSV)
    _write_csv(residual_rows, output_dir / RESIDUAL_SUBFAMILY_ROWS_CSV)
    _write_report(
        output_dir=output_dir,
        support_depth=support_depth,
        primary_vs_best=primary_vs_best,
        rule_candidates=rule_candidates,
        residual_rows=residual_rows,
    )

    threshold_summary = support_depth[
        support_depth["summary_scope"].eq("retained_under_min_recovered_event_threshold")
    ]
    exact_summary = support_depth[
        support_depth["summary_scope"].eq("exact_recovered_event_count_distribution")
    ]
    summary = {
        "ok": True,
        "v2_registry_dir": _rel(v2_registry_dir),
        "refinement_dir": _rel(refinement_dir),
        "output_dir": _rel(output_dir),
        "primary_vs_best_family_count": int(len(primary_vs_best)),
        "primary_recovered_event_count": int(
            primary_vs_best["primary_recovered_event_count"].sum()
        ),
        "best_axis_recovered_event_count": int(
            primary_vs_best["best_recovered_event_count"].sum()
        ),
        "best_axis_gain_event_count": int(primary_vs_best["best_gain_event_count"].sum()),
        "family_count_with_best_axis_gain": int(
            primary_vs_best["best_gain_event_count"].gt(0).sum()
        ),
        "rule_revision_candidate_count": int(len(rule_candidates)),
        "axis_decision_read_counts": _count(primary_vs_best, "axis_decision_read"),
        "residual_primary_subfamily_row_count": int(len(residual_rows)),
        "residual_primary_subfamily_event_count": int(residual_rows["event_count"].sum()),
        "residual_definition_read_counts": _count(residual_rows, "residual_definition_read"),
        "recovered_exact_event_count_distribution": {
            int(row["exact_recovered_event_count"]): int(row["recovered_subfamily_count"])
            for _, row in exact_summary.iterrows()
        },
        "support_depth_thresholds": [
            {
                "min_recovered_events": int(row["min_recovered_events"]),
                "retained_primitive_count": int(row["retained_primitive_count"]),
                "retained_recovered_subfamily_count": int(
                    row["retained_recovered_subfamily_count"]
                ),
                "retained_event_count": int(row["retained_event_count"]),
                "event_coverage_share": float(row["event_coverage_share"]),
            }
            for _, row in threshold_summary.iterrows()
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "support_depth_summary_csv": _rel(output_dir / SUPPORT_DEPTH_SUMMARY_CSV),
            "primary_vs_best_axis_rows_csv": _rel(output_dir / PRIMARY_VS_BEST_AXIS_ROWS_CSV),
            "rule_revision_candidates_csv": _rel(output_dir / RULE_REVISION_CANDIDATES_CSV),
            "residual_subfamily_rows_csv": _rel(output_dir / RESIDUAL_SUBFAMILY_ROWS_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "v2_registry_dir": _rel(v2_registry_dir),
        "refinement_dir": _rel(refinement_dir),
        "output_dir": _rel(output_dir),
        "full_event_count": FULL_EVENT_COUNT,
        "claim_boundary": CLAIM_BOUNDARY,
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
    parser.add_argument("--refinement-dir", type=Path, default=DEFAULT_REFINEMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        v2_registry_dir=args.v2_registry_dir.resolve(),
        refinement_dir=args.refinement_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
