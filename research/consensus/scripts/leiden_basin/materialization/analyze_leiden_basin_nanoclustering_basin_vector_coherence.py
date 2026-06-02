#!/usr/bin/env python3
"""Analyze within-family coherence of NanoClustering basin vectors.

This reads the basin-vector panel and checks whether split-vector shape,
split-vector class, host-context class, and dominant host handles repeat within
each definition-core family. It does not run clustering, execute optimizer
routes, promote wall/pathway claims, or inspect basin quality/cost.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_VECTOR_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_vector_panel_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_vector_coherence_20260530"
)

EVENT_VECTOR_ROWS_CSV = "nanoclustering_basin_vector_event_rows.csv"
FAMILY_VECTOR_ROWS_CSV = "nanoclustering_basin_vector_family_rows.csv"

COHERENCE_EVENT_ROWS_CSV = "nanoclustering_basin_vector_coherence_event_rows.csv"
COHERENCE_FAMILY_ROWS_CSV = "nanoclustering_basin_vector_coherence_family_rows.csv"
COHERENCE_CLASS_SUMMARY_CSV = "nanoclustering_basin_vector_coherence_class_summary.csv"
COHERENCE_EXCEPTION_ROWS_CSV = "nanoclustering_basin_vector_coherence_exception_rows.csv"
SUMMARY_JSON = "nanoclustering_basin_vector_coherence_summary.json"
REPORT_MD = "nanoclustering_basin_vector_coherence_report.md"
CONFIG_JSON = "nanoclustering_basin_vector_coherence_config.json"

CLAIM_BOUNDARY = (
    "Basin-vector coherence diagnostic only; no route execution, wall/pathway "
    "promotion, basin-quality claim, cost claim, or directed-search claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_basin_vector_coherence"


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


def _count_string(values: pd.Series) -> str:
    counts = Counter(values.dropna().astype(str))
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _dominant(values: pd.Series) -> tuple[str, int, float]:
    counts = Counter(values.dropna().astype(str))
    if not counts:
        return "", 0, 0.0
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    total = len(values.dropna())
    return label, int(count), count / total if total else 0.0


def _iqr(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.quantile(0.75) - values.quantile(0.25))


def _bin(value: Any, cuts: list[float], labels: list[str]) -> str:
    if value is None or pd.isna(value):
        return "na"
    x = float(value)
    for cut, label in zip(cuts, labels):
        if x < cut:
            return label
    return labels[-1]


def _segment_count_bin(value: Any) -> str:
    if value is None or pd.isna(value):
        return "na"
    count = int(value)
    if count <= 3:
        return "seg_le3"
    if count <= 5:
        return "seg_4_5"
    if count <= 7:
        return "seg_6_7"
    return "seg_ge8"


def _shape_bins(row: pd.Series) -> tuple[str, str, str, str]:
    top1_bin = _bin(
        row["top1_segment_share_ref_weight"],
        [0.25, 0.35, 0.45, 0.5],
        ["top1_lt25", "top1_25_35", "top1_35_45", "top1_45_50", "top1_ge50"],
    )
    top2_bin = _bin(
        row["top2_segment_share_ref_weight"],
        [0.2, 0.3, 0.4],
        ["top2_lt20", "top2_20_30", "top2_30_40", "top2_ge40"],
    )
    effective_bin = _bin(
        row["effective_segment_count"],
        [3.0, 4.0, 5.0],
        ["eff_lt3", "eff_3_4", "eff_4_5", "eff_ge5"],
    )
    segment_bin = _segment_count_bin(row["significant_segment_count"])
    return top1_bin, top2_bin, effective_bin, segment_bin


def _shape_core_signature(row: pd.Series) -> str:
    top1_bin, top2_bin, effective_bin, _ = _shape_bins(row)
    return "|".join([str(row["split_vector_class"]), top1_bin, top2_bin, effective_bin])


def _shape_signature(row: pd.Series) -> str:
    top1_bin, top2_bin, effective_bin, segment_bin = _shape_bins(row)
    return "|".join([str(row["split_vector_class"]), top1_bin, top2_bin, effective_bin, segment_bin])


def _event_rows(event_vectors: pd.DataFrame) -> pd.DataFrame:
    rows = event_vectors.copy()
    rows["shape_core_signature"] = rows.apply(_shape_core_signature, axis=1)
    rows["shape_signature"] = rows.apply(_shape_signature, axis=1)
    rows["host_signature"] = rows.apply(
        lambda row: f"{row['host_context_class']}|{row['dominant_host_handle_id']}",
        axis=1,
    )
    rows["coherence_event_scope"] = "split_shape_signature_and_dominant_host_context"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "event_id",
        "family_id",
        "branch",
        "boundary_family_tier",
        "family_vector_class",
        "split_vector_class",
        "host_context_class",
        "shape_core_signature",
        "shape_signature",
        "host_signature",
        "comparison_seed",
        "boundary_pattern",
        "significant_segment_count",
        "top1_segment_share_ref_weight",
        "top2_segment_share_ref_weight",
        "top2_segment_share_sum",
        "effective_segment_count",
        "dominant_host_handle_id",
        "dominant_host_is_source_ref",
        "target_share_of_best_run_cluster_weight",
        "coherence_event_scope",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in rows.columns if column not in preferred]
    return rows[preferred + remainder].sort_values(["branch", "family_id", "comparison_seed"])


def _coherence_status(row: pd.Series) -> str:
    split_purity = float(row["dominant_split_vector_class_share"])
    host_purity = float(row["dominant_host_context_class_share"])
    shape_core_share = float(row["dominant_shape_core_signature_share"])
    host_share = float(row["dominant_host_handle_share"])
    top1_iqr = float(row["top1_segment_share_iqr"])
    top2_iqr = float(row["top2_segment_share_iqr"])

    split_coherent = split_purity >= 0.75 and shape_core_share >= 0.5 and top1_iqr <= 0.12
    host_coherent = host_purity >= 0.75 and host_share >= 0.5
    numeric_stable = top1_iqr <= 0.08 and top2_iqr <= 0.10

    if split_coherent and host_coherent and numeric_stable:
        return "coherent_vector_and_host_family"
    if split_coherent and host_coherent:
        return "coherent_class_with_numeric_variation"
    if split_coherent and not host_coherent:
        return "split_coherent_host_variable_family"
    if host_coherent and not split_coherent:
        return "host_coherent_split_mixed_family"
    return "heterogeneous_or_rule_edge_family"


def _family_rows(
    *,
    event_rows: pd.DataFrame,
    family_vectors: pd.DataFrame,
) -> pd.DataFrame:
    family_class = family_vectors[
        [
            "family_id",
            "family_vector_class",
            "event_count",
            "dominant_host_event_share",
            "distinct_dominant_host_count",
            "top1_endpoint_handle_count",
            "top_segment_handle_signature_count",
        ]
    ]
    rows: list[dict[str, Any]] = []
    for family_id, group in event_rows.groupby("family_id", sort=False):
        first = group.iloc[0]
        split_label, split_count, split_share = _dominant(group["split_vector_class"])
        host_label, host_count, host_share = _dominant(group["host_context_class"])
        shape_label, shape_count, shape_share = _dominant(group["shape_signature"])
        shape_core_label, shape_core_count, shape_core_share = _dominant(
            group["shape_core_signature"]
        )
        host_handle, host_handle_count, host_handle_share = _dominant(
            group["dominant_host_handle_id"]
        )
        top1 = group["top1_segment_share_ref_weight"].astype(float)
        top2 = group["top2_segment_share_ref_weight"].astype(float)
        effective = group["effective_segment_count"].astype(float)
        segment_count = group["significant_segment_count"].astype(float)
        rows.append(
            {
                "family_id": family_id,
                "branch": first["branch"],
                "boundary_family_tier": first["boundary_family_tier"],
                "family_vector_class": first["family_vector_class"],
                "event_count": int(len(group)),
                "dominant_split_vector_class": split_label,
                "dominant_split_vector_class_count": split_count,
                "dominant_split_vector_class_share": split_share,
                "split_vector_class_counts": _count_string(group["split_vector_class"]),
                "dominant_host_context_class": host_label,
                "dominant_host_context_class_count": host_count,
                "dominant_host_context_class_share": host_share,
                "host_context_class_counts": _count_string(group["host_context_class"]),
                "dominant_shape_signature": shape_label,
                "dominant_shape_signature_count": shape_count,
                "dominant_shape_signature_share": shape_share,
                "shape_signature_count": int(group["shape_signature"].nunique()),
                "dominant_shape_core_signature": shape_core_label,
                "dominant_shape_core_signature_count": shape_core_count,
                "dominant_shape_core_signature_share": shape_core_share,
                "shape_core_signature_count": int(group["shape_core_signature"].nunique()),
                "dominant_host_handle_id": host_handle,
                "dominant_host_handle_count": host_handle_count,
                "dominant_host_handle_share": host_handle_share,
                "distinct_dominant_host_handle_count": int(
                    group["dominant_host_handle_id"].nunique()
                ),
                "top1_segment_share_median": float(top1.median()),
                "top1_segment_share_iqr": _iqr(top1),
                "top2_segment_share_median": float(top2.median()),
                "top2_segment_share_iqr": _iqr(top2),
                "top2_segment_share_sum_median": float(
                    group["top2_segment_share_sum"].astype(float).median()
                ),
                "effective_segment_count_median": float(effective.median()),
                "effective_segment_count_iqr": _iqr(effective),
                "significant_segment_count_median": float(segment_count.median()),
                "significant_segment_count_iqr": _iqr(segment_count),
                "boundary_pattern_counts": _count_string(group["boundary_pattern"]),
                "coherence_scope": "within_family_split_shape_and_host_context",
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    frame = pd.DataFrame(rows).merge(
        family_class.drop(columns=["family_vector_class", "event_count"]),
        on="family_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_vector_panel"),
    )
    frame["coherence_status"] = frame.apply(_coherence_status, axis=1)
    preferred = [
        "family_id",
        "branch",
        "boundary_family_tier",
        "family_vector_class",
        "coherence_status",
        "event_count",
        "dominant_split_vector_class",
        "dominant_split_vector_class_share",
        "dominant_host_context_class",
        "dominant_host_context_class_share",
        "dominant_shape_signature_share",
        "dominant_shape_core_signature_share",
        "dominant_host_handle_share",
        "shape_signature_count",
        "shape_core_signature_count",
        "distinct_dominant_host_handle_count",
        "top1_segment_share_median",
        "top1_segment_share_iqr",
        "top2_segment_share_median",
        "top2_segment_share_iqr",
        "effective_segment_count_median",
        "effective_segment_count_iqr",
        "significant_segment_count_median",
        "significant_segment_count_iqr",
        "split_vector_class_counts",
        "host_context_class_counts",
        "boundary_pattern_counts",
        "dominant_shape_signature",
        "dominant_shape_core_signature",
        "dominant_host_handle_id",
        "coherence_scope",
        "route_execution_status",
        "wall_promotion_status",
        "quality_cost_status",
        "claim_boundary",
    ]
    remainder = [column for column in frame.columns if column not in preferred]
    return frame[preferred + remainder].sort_values(
        ["boundary_family_tier", "coherence_status", "family_vector_class", "family_id"]
    )


def _class_summary(family_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family_class, coherence), group in family_rows.groupby(
        ["family_vector_class", "coherence_status"], sort=True
    ):
        rows.append(
            {
                "family_vector_class": family_class,
                "coherence_status": coherence,
                "family_count": int(len(group)),
                "event_count_sum": int(group["event_count"].sum()),
                "dominant_split_vector_class_share_median": float(
                    group["dominant_split_vector_class_share"].median()
                ),
                "dominant_host_context_class_share_median": float(
                    group["dominant_host_context_class_share"].median()
                ),
                "dominant_shape_signature_share_median": float(
                    group["dominant_shape_signature_share"].median()
                ),
                "dominant_shape_core_signature_share_median": float(
                    group["dominant_shape_core_signature_share"].median()
                ),
                "dominant_host_handle_share_median": float(
                    group["dominant_host_handle_share"].median()
                ),
                "top1_segment_share_iqr_median": float(
                    group["top1_segment_share_iqr"].median()
                ),
                "effective_segment_count_iqr_median": float(
                    group["effective_segment_count_iqr"].median()
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(["family_vector_class", "coherence_status"])


def _exception_rows(family_rows: pd.DataFrame) -> pd.DataFrame:
    rows = family_rows[
        family_rows["coherence_status"].isin(
            {
                "host_coherent_split_mixed_family",
                "split_coherent_host_variable_family",
                "heterogeneous_or_rule_edge_family",
                "coherent_class_with_numeric_variation",
            }
        )
    ].copy()
    rows["exception_read"] = rows.apply(
        lambda row: (
            "split shape is stable but host context varies"
            if row["coherence_status"] == "split_coherent_host_variable_family"
            else "host context is stable but split class or shape varies"
            if row["coherence_status"] == "host_coherent_split_mixed_family"
            else "class is stable but numeric split shares vary beyond strict band"
            if row["coherence_status"] == "coherent_class_with_numeric_variation"
            else "mixed family or current vector rules need refinement"
        ),
        axis=1,
    )
    return rows


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
    suffix = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    event_rows: pd.DataFrame,
    family_rows: pd.DataFrame,
    class_summary: pd.DataFrame,
    exception_rows: pd.DataFrame,
) -> None:
    text = [
        "# NanoClustering Basin Vector Coherence",
        "",
        f"- event_rows: `{len(event_rows)}`",
        f"- family_rows: `{len(family_rows)}`",
        f"- class_summary_rows: `{len(class_summary)}`",
        f"- exception_rows: `{len(exception_rows)}`",
        f"- coherent_vector_and_host_family: `{int(family_rows['coherence_status'].eq('coherent_vector_and_host_family').sum())}`",
        f"- host_coherent_split_mixed_family: `{int(family_rows['coherence_status'].eq('host_coherent_split_mixed_family').sum())}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Coherence Summary",
        "",
        _markdown_table(
            class_summary,
            [
                "family_vector_class",
                "coherence_status",
                "family_count",
                "event_count_sum",
                "dominant_split_vector_class_share_median",
                "dominant_host_context_class_share_median",
                "dominant_shape_signature_share_median",
                "dominant_shape_core_signature_share_median",
                "dominant_host_handle_share_median",
            ],
            max_rows=30,
        ),
        "",
        "## Family Rows",
        "",
        _markdown_table(
            family_rows,
            [
                "family_id",
                "family_vector_class",
                "coherence_status",
                "event_count",
                "dominant_split_vector_class_share",
                "dominant_host_context_class_share",
                "dominant_shape_signature_share",
                "dominant_shape_core_signature_share",
                "dominant_host_handle_share",
                "top1_segment_share_iqr",
                "effective_segment_count_iqr",
            ],
            max_rows=25,
        ),
        "",
        "## Exceptions",
        "",
        _markdown_table(
            exception_rows,
            [
                "family_id",
                "family_vector_class",
                "coherence_status",
                "exception_read",
                "split_vector_class_counts",
                "host_context_class_counts",
                "boundary_pattern_counts",
            ],
            max_rows=25,
        ),
        "",
        "## Read",
        "",
        "- The coherent rows support endpoint-vector basin families rather than single endpoint-pair handles.",
        "- Host-coherent but split-mixed rows should be treated as rule-edge or subfamily candidates, not failures.",
        "- This remains membership-derived endpoint cartography; no wall/pathway, quality, or cost claim is added.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(*, vector_dir: Path, output_dir: Path) -> dict[str, Any]:
    event_vectors = _read_csv(vector_dir / EVENT_VECTOR_ROWS_CSV)
    family_vectors = _read_csv(vector_dir / FAMILY_VECTOR_ROWS_CSV)
    event_rows = _event_rows(event_vectors.merge(
        family_vectors[["family_id", "family_vector_class"]],
        on="family_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_family"),
    ))
    family_rows = _family_rows(event_rows=event_rows, family_vectors=family_vectors)
    class_summary = _class_summary(family_rows)
    exception_rows = _exception_rows(family_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(event_rows, output_dir / COHERENCE_EVENT_ROWS_CSV)
    _write_csv(family_rows, output_dir / COHERENCE_FAMILY_ROWS_CSV)
    _write_csv(class_summary, output_dir / COHERENCE_CLASS_SUMMARY_CSV)
    _write_csv(exception_rows, output_dir / COHERENCE_EXCEPTION_ROWS_CSV)
    _write_report(
        output_dir=output_dir,
        event_rows=event_rows,
        family_rows=family_rows,
        class_summary=class_summary,
        exception_rows=exception_rows,
    )

    summary = {
        "ok": True,
        "vector_dir": _rel(vector_dir),
        "output_dir": _rel(output_dir),
        "event_row_count": int(len(event_rows)),
        "family_row_count": int(len(family_rows)),
        "class_summary_row_count": int(len(class_summary)),
        "exception_row_count": int(len(exception_rows)),
        "coherence_status_counts": {
            str(key): int(value)
            for key, value in family_rows["coherence_status"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "family_vector_class_by_coherence": {
            f"{family_class}|{status}": int(count)
            for (family_class, status), count in family_rows.groupby(
                ["family_vector_class", "coherence_status"]
            ).size().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "event_rows_csv": _rel(output_dir / COHERENCE_EVENT_ROWS_CSV),
            "family_rows_csv": _rel(output_dir / COHERENCE_FAMILY_ROWS_CSV),
            "class_summary_csv": _rel(output_dir / COHERENCE_CLASS_SUMMARY_CSV),
            "exception_rows_csv": _rel(output_dir / COHERENCE_EXCEPTION_ROWS_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "vector_dir": _rel(vector_dir),
        "output_dir": _rel(output_dir),
        "shape_core_signature": "split_vector_class plus binned top1, top2, and effective segment count",
        "shape_signature": "shape_core_signature plus binned significant segment count",
        "coherence_rules": {
            "split_coherent": "dominant split class share >= 0.75, dominant shape-core signature share >= 0.5, top1 IQR <= 0.12",
            "host_coherent": "dominant host context share >= 0.75 and dominant host handle share >= 0.5",
            "numeric_stable": "top1 IQR <= 0.08 and top2 IQR <= 0.10",
        },
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
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        vector_dir=args.vector_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
