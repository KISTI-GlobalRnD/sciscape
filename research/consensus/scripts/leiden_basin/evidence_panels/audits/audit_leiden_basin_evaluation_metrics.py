#!/usr/bin/env python3
"""Audit Leiden basin artifacts for label-invariant evaluation coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import pandas as pd

COMBINED_DIR = Path(
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_evaluation_metric_audit_v0"

ROWS_FILENAME = "basin_evaluation_metric_audit_rows.csv"
SUMMARY_FILENAME = "basin_evaluation_metric_audit_summary.json"
REPORT_FILENAME = "basin_evaluation_metric_audit_report.md"

EXACT_PATTERNS = (
    "changed_node_count",
    "changed_nodes",
)
KNOWN_LABEL_INVARIANT_CHANGED_PATTERNS = (
    "changed_nodes_vs_baseline",
    "changed_support",
    "changed_node_support",
    "changed_pair_support",
    "aligned_changed",
    "alignment_error",
)
SAFE_EXACT_PATTERNS = (
    "exact_changed",
    "exact_label_changed",
)
ALIGNED_PATTERNS = (
    "aligned_changed",
    "aligned_partition",
)
ALIGNMENT_ERROR_PATTERNS = (
    "alignment_error",
    "fragmentation",
    "mixing",
)
ENDPOINT_PATTERNS = (
    "endpoint_distance",
)
SUPPORT_PATTERNS = (
    "support_distance",
    "support_progress",
)
CHANGED_SUPPORT_PATTERNS = (
    "changed_support",
    "changed_node_support",
    "changed_pair_support",
)

def _contains_any(values: list[str], patterns: tuple[str, ...]) -> bool:
    return any(any(pattern in value for pattern in patterns) for value in values)

def _matching_columns(values: list[str], patterns: tuple[str, ...]) -> list[str]:
    return [value for value in values if any(pattern in value for pattern in patterns)]

def _all_known_label_invariant_changed(columns: list[str]) -> bool:
    return bool(columns) and all(
        any(pattern in column for pattern in KNOWN_LABEL_INVARIANT_CHANGED_PATTERNS)
        for column in columns
    )

def _read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return [str(value).strip() for value in next(reader, [])]
    except (OSError, UnicodeDecodeError, StopIteration):
        return []

def _risk_label(columns: list[str]) -> str:
    lowered = [column.lower() for column in columns]
    raw_changed_columns = _matching_columns(lowered, EXACT_PATTERNS)
    has_raw_changed = bool(raw_changed_columns)
    has_safe_exact = _contains_any(lowered, SAFE_EXACT_PATTERNS)
    has_aligned = _contains_any(lowered, ALIGNED_PATTERNS)
    has_alignment_error = _contains_any(lowered, ALIGNMENT_ERROR_PATTERNS)
    has_endpoint = _contains_any(lowered, ENDPOINT_PATTERNS)
    has_support = _contains_any(lowered, SUPPORT_PATTERNS)
    has_changed_support = _contains_any(lowered, CHANGED_SUPPORT_PATTERNS)
    has_label_invariant = (
        has_aligned
        or has_alignment_error
        or has_endpoint
        or has_support
        or has_changed_support
    )
    if has_raw_changed and not has_label_invariant:
        return "rerun_or_backfill_required"
    if has_raw_changed and _all_known_label_invariant_changed(raw_changed_columns):
        return "label_invariant_metrics_present"
    if has_raw_changed and has_label_invariant:
        return "relabel_exact_columns_and_reinterpret"
    if has_safe_exact and not has_label_invariant:
        return "aligned_metric_missing"
    if has_label_invariant:
        return "label_invariant_metrics_present"
    return "not_basin_metric_artifact"

def _artifact_family(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""

def audit_root(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.csv")):
        artifact_family = _artifact_family(path, root)
        if artifact_family.startswith("basin_evaluation_metric_audit"):
            continue
        columns = _read_csv_header(path)
        lowered = [column.lower() for column in columns]
        if not columns:
            continue
        rows.append(
            {
                "path": str(path),
                "artifact_family": artifact_family,
                "file_name": path.name,
                "column_count": int(len(columns)),
                "risk_label": _risk_label(columns),
                "has_changed_node_count": bool(
                    _contains_any(lowered, EXACT_PATTERNS)
                ),
                "has_explicit_exact_changed": bool(
                    _contains_any(lowered, SAFE_EXACT_PATTERNS)
                ),
                "has_aligned_changed": bool(
                    _contains_any(lowered, ALIGNED_PATTERNS)
                ),
                "has_alignment_error": bool(
                    _contains_any(lowered, ALIGNMENT_ERROR_PATTERNS)
                ),
                "has_endpoint_distance": bool(
                    _contains_any(lowered, ENDPOINT_PATTERNS)
                ),
                "has_support_metric": bool(
                    _contains_any(lowered, SUPPORT_PATTERNS)
                ),
                "has_changed_support_metric": bool(
                    _contains_any(lowered, CHANGED_SUPPORT_PATTERNS)
                ),
                "changed_columns": ",".join(
                    column
                    for column in columns
                    if "changed" in column.lower()
                    or "alignment_error" in column.lower()
                    or "endpoint_distance" in column.lower()
                    or "support_distance" in column.lower()
                    or "support_progress" in column.lower()
                ),
            }
        )
    return pd.DataFrame(rows)

def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| "
            + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns)
            + " |"
        )
    return lines

def _write_report(path: Path, rows: pd.DataFrame) -> None:
    counts = (
        rows["risk_label"].value_counts().rename_axis("risk_label").reset_index(name="count")
        if not rows.empty
        else pd.DataFrame(columns=["risk_label", "count"])
    )
    risky = rows[
        rows["risk_label"].isin(
            {
                "rerun_or_backfill_required",
                "relabel_exact_columns_and_reinterpret",
                "aligned_metric_missing",
            }
        )
    ].copy()
    risky = risky.sort_values(["risk_label", "artifact_family", "file_name"])
    lines = [
        "# Basin Evaluation Metric Audit",
        "",
        "This audit flags artifacts whose basin interpretation may depend on",
        "exact label equality rather than label-invariant support/endpoint metrics.",
        "",
        "## Risk Counts",
        "",
    ]
    lines.extend(_markdown_table(counts, max_rows=20))
    lines.extend(["", "## Risky Artifacts", ""])
    display_cols = [
        "risk_label",
        "artifact_family",
        "file_name",
        "has_changed_node_count",
        "has_aligned_changed",
        "has_alignment_error",
        "has_endpoint_distance",
        "has_support_metric",
        "has_changed_support_metric",
        "changed_columns",
    ]
    lines.extend(_markdown_table(risky[[c for c in display_cols if c in risky]], max_rows=120))
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "- `changed_node_count` without aligned, endpoint, or changed-support metrics is not basin evidence.",
            "- Exact-label counts are implementation diagnostics only.",
            "- `changed_nodes_vs_baseline` is accepted only when it is paired with alignment-error or changed-support aliases.",
            "- Basin-level claims need label-invariant aligned support, endpoint distance, QF, and cost.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_audit(*, root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_root(root)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    result = {
        "schema": "leiden_basin_evaluation_metric_audit.v0",
        "root": str(root),
        "output_dir": str(output_dir),
        "csv_file_count": int(len(rows)),
        "risk_counts": rows["risk_label"].value_counts().to_dict()
        if not rows.empty
        else {},
        "paths": {
            "rows": str(output_dir / ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_FILENAME, rows)
    return result

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=COMBINED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    result = run_audit(root=args.root, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
