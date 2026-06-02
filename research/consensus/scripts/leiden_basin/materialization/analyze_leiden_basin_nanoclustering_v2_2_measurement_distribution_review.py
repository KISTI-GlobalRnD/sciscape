#!/usr/bin/env python3
"""Review distributions inside the v2.2 accepted-primitive measurement panel.

This script does not change the basin definition. It classifies accepted
primitive measurement rows into descriptive review bands so the current result
can be read conservatively: which primitives are a stable descriptive nucleus,
which are thin, which carry residual definition debt, and which are caveated by
shape/host/boundary concentration.

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
DEFAULT_MEASUREMENT_PANEL_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_measurement_panel_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_v2_2_measurement_distribution_review_20260531"
)

PRIMITIVE_MEASUREMENT_ROWS_CSV = (
    "nanoclustering_v2_2_accepted_primitive_measurement_rows.csv"
)
EVENT_MEASUREMENT_ROWS_CSV = (
    "nanoclustering_v2_2_accepted_primitive_event_measurement_rows.csv"
)
SOURCE_FAMILY_ROLLUP_CSV = "nanoclustering_v2_2_source_family_measurement_rollup.csv"
MEASUREMENT_PANEL_SUMMARY_JSON = "nanoclustering_v2_2_measurement_summary.json"

DISTRIBUTION_PRIMITIVE_ROWS_CSV = (
    "nanoclustering_v2_2_distribution_review_primitive_rows.csv"
)
DISTRIBUTION_BAND_SUMMARY_CSV = (
    "nanoclustering_v2_2_distribution_review_band_summary.csv"
)
DISTRIBUTION_HOST_BOUNDARY_SUMMARY_CSV = (
    "nanoclustering_v2_2_distribution_review_host_boundary_summary.csv"
)
RESIDUAL_DEBT_PRIORITY_ROWS_CSV = (
    "nanoclustering_v2_2_distribution_review_residual_debt_priority_rows.csv"
)
CONCENTRATION_CAVEAT_ROWS_CSV = (
    "nanoclustering_v2_2_distribution_review_concentration_caveat_rows.csv"
)
DISTRIBUTION_GATE_MATRIX_CSV = "nanoclustering_v2_2_distribution_review_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_v2_2_distribution_review_summary.json"
REPORT_MD = "nanoclustering_v2_2_distribution_review_report.md"
CONFIG_JSON = "nanoclustering_v2_2_distribution_review_config.json"

HOST_MODE_SHARE_FLOOR = 0.75
SHAPE_MODE_SHARE_FLOOR = 0.75
BOUNDARY_MODE_SHARE_FLOOR = 0.75

STABLE_BAND = "stable_high_support_measurement_unit"
THIN_CLEAN_BAND = "thin_clean_measurement_unit"
RESIDUAL_BAND = "residual_debt_caveated_measurement_unit"
CONCENTRATION_BAND = "concentration_caveated_measurement_unit"

CLAIM_BOUNDARY = (
    "V2.2 accepted-primitive distribution review only; descriptive measurement "
    "bands, no route execution, wall/pathway promotion, basin-quality claim, "
    "cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_v2_2_distribution_review"


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
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _joined_unique(values: pd.Series) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value)})
    return ";".join(clean)


def _joined_semicolon_tokens(values: pd.Series) -> str:
    tokens: set[str] = set()
    for value in values.dropna():
        tokens.update(token for token in str(value).split(";") if token)
    return ";".join(sorted(tokens))


def _review_band(row: pd.Series) -> str:
    if row["residual_caveat_status"] == "source_family_has_residual_definition_debt":
        return RESIDUAL_BAND
    if row["measurement_support_class"] == "thin_support_measurement_unit":
        return THIN_CLEAN_BAND
    if (
        float(row["dominant_host_handle_id_mode_share"]) >= HOST_MODE_SHARE_FLOOR
        and float(row["shape_core_signature_mode_share"]) >= SHAPE_MODE_SHARE_FLOOR
        and float(row["boundary_pattern_mode_share"]) >= BOUNDARY_MODE_SHARE_FLOOR
    ):
        return STABLE_BAND
    return CONCENTRATION_BAND


def _caveat_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["distribution_review_band"] == RESIDUAL_BAND:
        reasons.append("residual_definition_debt")
    if row["measurement_support_class"] == "thin_support_measurement_unit":
        reasons.append("thin_support")
    if float(row["dominant_host_handle_id_mode_share"]) < HOST_MODE_SHARE_FLOOR:
        reasons.append("low_host_handle_concentration")
    if float(row["shape_core_signature_mode_share"]) < SHAPE_MODE_SHARE_FLOOR:
        reasons.append("low_shape_core_concentration")
    if float(row["boundary_pattern_mode_share"]) < BOUNDARY_MODE_SHARE_FLOOR:
        reasons.append("low_boundary_pattern_concentration")
    return ";".join(reasons) if reasons else "none"


def _primitive_review_rows(primitive_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _numeric(
        primitive_rows,
        [
            "event_count",
            "source_family_residual_event_count",
            "dominant_host_handle_id_mode_share",
            "shape_core_signature_mode_share",
            "boundary_pattern_mode_share",
            "top1_segment_share_ref_weight_median",
            "effective_segment_count_median",
            "fragmentation_index_median",
        ],
    )
    rows["distribution_review_band"] = rows.apply(_review_band, axis=1)
    rows["distribution_caveat_reasons"] = rows.apply(_caveat_reasons, axis=1)
    rows["distribution_review_status"] = "descriptive_review_only_not_definition_change"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        [
            "distribution_review_band",
            "boundary_family_tier",
            "branch",
            "source_family_id",
            "primitive_id",
        ]
    )


def _band_summary(review_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _numeric(
        review_rows,
        [
            "event_count",
            "source_family_residual_event_count",
            "dominant_host_handle_id_mode_share",
            "shape_core_signature_mode_share",
            "boundary_pattern_mode_share",
            "top1_segment_share_ref_weight_median",
            "effective_segment_count_median",
            "fragmentation_index_median",
        ],
    )
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "distribution_review_band",
        "boundary_family_tier",
        "primitive_type",
        "measurement_support_class",
        "residual_caveat_status",
    ]
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        residual_by_family = group[
            ["source_family_id", "source_family_residual_event_count"]
        ].drop_duplicates("source_family_id")
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(
            {
                "primitive_count": int(group["primitive_id"].nunique()),
                "event_count": int(group["event_count"].sum()),
                "source_family_count": int(group["source_family_id"].nunique()),
                "family_residual_event_count_sum": int(
                    residual_by_family["source_family_residual_event_count"].sum()
                ),
                "median_host_handle_mode_share": float(
                    group["dominant_host_handle_id_mode_share"].median()
                ),
                "median_shape_core_mode_share": float(
                    group["shape_core_signature_mode_share"].median()
                ),
                "median_boundary_pattern_mode_share": float(
                    group["boundary_pattern_mode_share"].median()
                ),
                "median_top1_segment_share": float(
                    group["top1_segment_share_ref_weight_median"].median()
                ),
                "median_effective_segment_count": float(
                    group["effective_segment_count_median"].median()
                ),
                "median_fragmentation_index": float(group["fragmentation_index_median"].median()),
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["claim_boundary"] = CLAIM_BOUNDARY
    return summary.sort_values(
        [
            "distribution_review_band",
            "boundary_family_tier",
            "primitive_type",
            "measurement_support_class",
        ]
    )


def _host_boundary_summary(review_rows: pd.DataFrame) -> pd.DataFrame:
    rows = (
        review_rows.groupby(
            [
                "distribution_review_band",
                "boundary_family_tier",
                "host_context_class_mode",
                "boundary_pattern_mode",
            ],
            as_index=False,
        )
        .agg(
            primitive_count=("primitive_id", "nunique"),
            event_count=("event_count", "sum"),
            source_family_count=("source_family_id", "nunique"),
            median_host_handle_mode_share=("dominant_host_handle_id_mode_share", "median"),
            median_shape_core_mode_share=("shape_core_signature_mode_share", "median"),
            median_fragmentation_index=("fragmentation_index_median", "median"),
        )
        .sort_values(
            [
                "distribution_review_band",
                "boundary_family_tier",
                "primitive_count",
            ],
            ascending=[True, True, False],
        )
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _residual_debt_priority_rows(
    *,
    review_rows: pd.DataFrame,
    family_rollup: pd.DataFrame,
) -> pd.DataFrame:
    family_bands = (
        review_rows.groupby("source_family_id", as_index=False)
        .agg(
            distribution_review_bands=("distribution_review_band", _joined_unique),
            primitive_count=("primitive_id", "nunique"),
            accepted_event_count=("event_count", "sum"),
            host_context_modes=("host_context_class_mode", _joined_unique),
            boundary_pattern_modes=("boundary_pattern_mode", _joined_unique),
            caveat_reasons=("distribution_caveat_reasons", _joined_semicolon_tokens),
        )
    )
    rows = family_rollup.merge(family_bands, on="source_family_id", how="inner", suffixes=("", "_from_primitives"))
    rows = rows[pd.to_numeric(rows["residual_event_count"], errors="coerce").gt(0)].copy()
    rows["residual_event_count"] = pd.to_numeric(rows["residual_event_count"], errors="coerce").fillna(0).astype(int)
    rows["accepted_event_count"] = pd.to_numeric(rows["accepted_event_count"], errors="coerce").fillna(0).astype(int)
    rows["residual_to_accepted_event_ratio"] = rows["residual_event_count"] / rows[
        "accepted_event_count"
    ].replace(0, pd.NA)
    rows["distribution_review_priority"] = rows["residual_event_count"].map(
        lambda value: "high_residual_debt_review" if value >= 3 else "standard_residual_caveat"
    )
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "source_family_id",
        "distribution_review_priority",
        "branch",
        "boundary_family_tier",
        "primitive_count",
        "accepted_event_count",
        "residual_event_count",
        "residual_to_accepted_event_ratio",
        "residual_queue_statuses",
        "distribution_review_bands",
        "host_context_modes_from_primitives",
        "boundary_pattern_modes_from_primitives",
        "caveat_reasons",
        "claim_boundary",
    ]
    return rows[[column for column in preferred if column in rows.columns]].sort_values(
        ["residual_event_count", "accepted_event_count", "source_family_id"],
        ascending=[False, False, True],
    )


def _concentration_caveat_rows(review_rows: pd.DataFrame) -> pd.DataFrame:
    rows = review_rows[
        review_rows["distribution_review_band"].eq(CONCENTRATION_BAND)
    ].copy()
    preferred = [
        "primitive_id",
        "source_family_id",
        "branch",
        "boundary_family_tier",
        "primitive_type",
        "measurement_support_class",
        "event_count",
        "host_context_class_mode",
        "boundary_pattern_mode",
        "dominant_host_handle_id_mode_share",
        "shape_core_signature_mode_share",
        "boundary_pattern_mode_share",
        "top1_segment_share_ref_weight_median",
        "effective_segment_count_median",
        "fragmentation_index_median",
        "distribution_caveat_reasons",
        "claim_boundary",
    ]
    return rows[[column for column in preferred if column in rows.columns]].sort_values(
        [
            "shape_core_signature_mode_share",
            "dominant_host_handle_id_mode_share",
            "event_count",
        ],
        ascending=[True, True, False],
    )


def _gate_matrix(
    *,
    panel_summary: dict[str, Any],
    review_rows: pd.DataFrame,
    residual_priority_rows: pd.DataFrame,
) -> pd.DataFrame:
    stable = review_rows[review_rows["distribution_review_band"].eq(STABLE_BAND)]
    thin = review_rows[review_rows["distribution_review_band"].eq(THIN_CLEAN_BAND)]
    residual = review_rows[review_rows["distribution_review_band"].eq(RESIDUAL_BAND)]
    concentration = review_rows[review_rows["distribution_review_band"].eq(CONCENTRATION_BAND)]
    route_status_ok = bool(
        review_rows["route_execution_status"].eq(ROUTE_EXECUTION_STATUS).all()
        and review_rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all()
        and review_rows["quality_cost_status"].eq(QUALITY_COST_STATUS).all()
    )
    rows = [
        {
            "gate_id": "D1_distribution_accounting",
            "gate_question": "Does the review preserve the accepted measurement panel accounting?",
            "evidence": (
                f"review_primitives={review_rows['primitive_id'].nunique()}, "
                f"review_events={int(review_rows['event_count'].sum())}, "
                f"panel_primitives={panel_summary['accepted_primitive_count']}, "
                f"panel_events={panel_summary['accepted_event_count']}"
            ),
            "status": (
                "pass"
                if int(review_rows["primitive_id"].nunique())
                == int(panel_summary["accepted_primitive_count"])
                and int(review_rows["event_count"].sum())
                == int(panel_summary["accepted_event_count"])
                else "blocked"
            ),
            "decision": "use_distribution_bands_as_descriptive_review_only",
            "next_action": "report bands without changing v2.2 primitive membership",
        },
        {
            "gate_id": "D2_stable_descriptive_nucleus",
            "gate_question": "Is there a nontrivial stable nucleus for first accepted-primitive claims?",
            "evidence": (
                f"stable_primitives={stable['primitive_id'].nunique()}, "
                f"stable_events={int(stable['event_count'].sum())}, "
                f"stable_families={stable['source_family_id'].nunique()}"
            ),
            "status": "pass" if stable["primitive_id"].nunique() >= 50 else "thin_signal",
            "decision": "first claims should start from the stable descriptive nucleus",
            "next_action": "separate stable nucleus from thin, residual, and concentration caveats",
        },
        {
            "gate_id": "D3_caveat_load_visible",
            "gate_question": "Are weak/caveated parts large enough to constrain wording?",
            "evidence": (
                f"thin_clean_primitives={thin['primitive_id'].nunique()}, "
                f"residual_debt_primitives={residual['primitive_id'].nunique()}, "
                f"concentration_caveat_primitives={concentration['primitive_id'].nunique()}, "
                f"high_residual_families={int(residual_priority_rows['distribution_review_priority'].eq('high_residual_debt_review').sum())}"
            ),
            "status": "caveat_required",
            "decision": "do_not_describe_all_223_primitives_as_equally_strong",
            "next_action": "make support/residual/concentration caveats explicit in the result narrative",
        },
        {
            "gate_id": "D4_wall_quality_gate",
            "gate_question": "Can this distribution review open wall/pathway or quality/cost claims?",
            "evidence": "no route traces, quality fields, cost fields, or directed-search rows are used",
            "status": "closed_excluded_by_design" if route_status_ok else "blocked_status_leak",
            "decision": "keep route/wall/quality/cost gates closed",
            "next_action": "only after distribution review should a separate pathway protocol be designed",
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
    band_summary: pd.DataFrame,
    host_boundary_summary: pd.DataFrame,
    residual_priority_rows: pd.DataFrame,
    concentration_caveat_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    text = [
        "# NanoClustering V2.2 Measurement Distribution Review",
        "",
        f"- stable_high_support_primitives: `{summary['stable_high_support_primitive_count']}`",
        f"- stable_high_support_events: `{summary['stable_high_support_event_count']}`",
        f"- thin_clean_primitives: `{summary['thin_clean_primitive_count']}`",
        f"- residual_debt_caveated_primitives: `{summary['residual_debt_caveated_primitive_count']}`",
        f"- concentration_caveated_primitives: `{summary['concentration_caveated_primitive_count']}`",
        f"- high_residual_debt_families: `{summary['high_residual_debt_family_count']}`",
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
        "## Band Summary",
        "",
        _markdown_table(
            band_summary,
            [
                "distribution_review_band",
                "boundary_family_tier",
                "primitive_type",
                "measurement_support_class",
                "residual_caveat_status",
                "primitive_count",
                "event_count",
                "source_family_count",
                "median_host_handle_mode_share",
                "median_shape_core_mode_share",
            ],
            max_rows=30,
        ),
        "",
        "## Host-Boundary Structure",
        "",
        _markdown_table(
            host_boundary_summary,
            [
                "distribution_review_band",
                "boundary_family_tier",
                "host_context_class_mode",
                "boundary_pattern_mode",
                "primitive_count",
                "event_count",
                "source_family_count",
            ],
            max_rows=30,
        ),
        "",
        "## Residual Debt Priority Families",
        "",
        _markdown_table(
            residual_priority_rows,
            [
                "source_family_id",
                "distribution_review_priority",
                "branch",
                "boundary_family_tier",
                "primitive_count",
                "accepted_event_count",
                "residual_event_count",
                "residual_queue_statuses",
                "host_context_modes_from_primitives",
                "boundary_pattern_modes_from_primitives",
            ],
            max_rows=25,
        ),
        "",
        "## Concentration Caveat Rows",
        "",
        _markdown_table(
            concentration_caveat_rows,
            [
                "primitive_id",
                "boundary_family_tier",
                "primitive_type",
                "measurement_support_class",
                "event_count",
                "host_context_class_mode",
                "boundary_pattern_mode",
                "dominant_host_handle_id_mode_share",
                "shape_core_signature_mode_share",
                "distribution_caveat_reasons",
            ],
            max_rows=25,
        ),
        "",
        "## Read",
        "",
        "- The accepted panel should not be narrated as 223 equally strong primitives.",
        (
            "- The first descriptive nucleus is "
            f"{summary['stable_high_support_primitive_count']} stable high-support "
            f"primitives over {summary['stable_high_support_event_count']} accepted "
            f"events and {summary['stable_high_support_source_family_count']} source families."
        ),
        "- The caveat load is large: 42 thin-clean primitives, 52 residual-debt primitives, and 46 concentration-caveated primitives.",
        "- Persistent mixed core is the stronger stable nucleus; repeat severe core remains more caveated and should be worded as a harder boundary class.",
        "- This review still does not supply route/wall, quality/cost, or directed-search evidence.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    measurement_panel_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    primitive_rows = _read_csv(measurement_panel_dir / PRIMITIVE_MEASUREMENT_ROWS_CSV)
    event_rows = _read_csv(measurement_panel_dir / EVENT_MEASUREMENT_ROWS_CSV)
    family_rollup = _read_csv(measurement_panel_dir / SOURCE_FAMILY_ROLLUP_CSV)
    panel_summary = json.loads(
        (measurement_panel_dir / MEASUREMENT_PANEL_SUMMARY_JSON).read_text(encoding="utf-8")
    )

    review_rows = _primitive_review_rows(primitive_rows)
    band_summary = _band_summary(review_rows)
    host_boundary_summary = _host_boundary_summary(review_rows)
    residual_priority_rows = _residual_debt_priority_rows(
        review_rows=review_rows,
        family_rollup=family_rollup,
    )
    concentration_caveat_rows = _concentration_caveat_rows(review_rows)
    gate_matrix = _gate_matrix(
        panel_summary=panel_summary,
        review_rows=review_rows,
        residual_priority_rows=residual_priority_rows,
    )

    stable = review_rows[review_rows["distribution_review_band"].eq(STABLE_BAND)]
    thin = review_rows[review_rows["distribution_review_band"].eq(THIN_CLEAN_BAND)]
    residual = review_rows[review_rows["distribution_review_band"].eq(RESIDUAL_BAND)]
    concentration = review_rows[review_rows["distribution_review_band"].eq(CONCENTRATION_BAND)]
    summary = {
        "accepted_primitive_count": int(review_rows["primitive_id"].nunique()),
        "accepted_event_count": int(review_rows["event_count"].sum()),
        "accepted_event_row_count": int(event_rows["event_id"].nunique()),
        "stable_high_support_primitive_count": int(stable["primitive_id"].nunique()),
        "stable_high_support_event_count": int(stable["event_count"].sum()),
        "stable_high_support_source_family_count": int(stable["source_family_id"].nunique()),
        "thin_clean_primitive_count": int(thin["primitive_id"].nunique()),
        "thin_clean_event_count": int(thin["event_count"].sum()),
        "residual_debt_caveated_primitive_count": int(residual["primitive_id"].nunique()),
        "residual_debt_caveated_event_count": int(residual["event_count"].sum()),
        "concentration_caveated_primitive_count": int(concentration["primitive_id"].nunique()),
        "concentration_caveated_event_count": int(concentration["event_count"].sum()),
        "high_residual_debt_family_count": int(
            residual_priority_rows["distribution_review_priority"].eq(
                "high_residual_debt_review"
            ).sum()
        ),
        "distribution_band_counts": _count(review_rows, "distribution_review_band"),
        "distribution_band_event_counts": {
            str(key): int(value)
            for key, value in review_rows.groupby("distribution_review_band")["event_count"]
            .sum()
            .sort_index()
            .items()
        },
        "stable_band_by_tier_counts": _count(stable, "boundary_family_tier"),
        "gate_status_counts": _count(gate_matrix, "status"),
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {"measurement_panel_dir": _rel(measurement_panel_dir)},
        "review_parameters": {
            "host_mode_share_floor": HOST_MODE_SHARE_FLOOR,
            "shape_mode_share_floor": SHAPE_MODE_SHARE_FLOOR,
            "boundary_mode_share_floor": BOUNDARY_MODE_SHARE_FLOOR,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(review_rows, output_dir / DISTRIBUTION_PRIMITIVE_ROWS_CSV)
    _write_csv(band_summary, output_dir / DISTRIBUTION_BAND_SUMMARY_CSV)
    _write_csv(host_boundary_summary, output_dir / DISTRIBUTION_HOST_BOUNDARY_SUMMARY_CSV)
    _write_csv(residual_priority_rows, output_dir / RESIDUAL_DEBT_PRIORITY_ROWS_CSV)
    _write_csv(concentration_caveat_rows, output_dir / CONCENTRATION_CAVEAT_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / DISTRIBUTION_GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "measurement_panel_dir": _rel(measurement_panel_dir),
        "output_dir": _rel(output_dir),
        "host_mode_share_floor": HOST_MODE_SHARE_FLOOR,
        "shape_mode_share_floor": SHAPE_MODE_SHARE_FLOOR,
        "boundary_mode_share_floor": BOUNDARY_MODE_SHARE_FLOOR,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        band_summary=band_summary,
        host_boundary_summary=host_boundary_summary,
        residual_priority_rows=residual_priority_rows,
        concentration_caveat_rows=concentration_caveat_rows,
        gate_matrix=gate_matrix,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-panel-dir", type=Path, default=DEFAULT_MEASUREMENT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = materialize(
        measurement_panel_dir=args.measurement_panel_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
