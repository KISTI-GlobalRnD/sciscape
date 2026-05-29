#!/usr/bin/env python3
"""Build a diagnostic basin-transition landscape from vanilla and p5 rows.

This is intentionally diagnostic: it compares observed final basin nodes and
candidate transition hypotheses, but it does not execute a transition operator.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
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

SCRIPT_DIR = Path(__file__).resolve().parent

from analyze_leiden_basin_reachability_audit import (  # noqa: E402
    _read_vanilla_rows,
    _sketch_metrics,
)
from analyze_leiden_multibasin_decision_rules import (  # noqa: E402
    _case_field_method as _infer_field_method,
)
from analyze_leiden_multibasin_signatures import (  # noqa: E402
    CHANGED_SUPPORT_COLUMN,
    SKETCH_HASH_COLUMN,
    SKETCH_MEMBERSHIP_COLUMN,
    _parse_sketch,
)
from collect_leiden_vanilla_reachability_sweep import _read_candidate_rows  # noqa: E402

DEFAULT_CANDIDATE_DIRS = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519",
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_signature_field30_budget12_support_20260519",
)
DEFAULT_VANILLA_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "vanilla_reachability_sweep_field34_cc_n10_compatible_sketch"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_landscape_field34_cc"
)
NODE_ROWS_FILENAME = "basin_transition_landscape_nodes.csv"
EDGE_ROWS_FILENAME = "basin_transition_landscape_edges.csv"
HYPOTHESIS_ROWS_FILENAME = "basin_transition_landscape_hypotheses.csv"
SUMMARY_FILENAME = "basin_transition_landscape_summary.json"
REPORT_FILENAME = "basin_transition_landscape_report.md"

def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _node_support(row: pd.Series) -> set[int]:
    return {int(value) for value in _parse_sketch(row.get(CHANGED_SUPPORT_COLUMN))}

def _support_overlap(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    left_support = _node_support(left)
    right_support = _node_support(right)
    intersection = left_support & right_support
    union = left_support | right_support
    return {
        "left_support_size": len(left_support),
        "right_support_size": len(right_support),
        "support_intersection_size": len(intersection),
        "left_only_support_size": len(left_support - right_support),
        "right_only_support_size": len(right_support - left_support),
        "support_union_size": len(union),
        "left_overlap_ratio": (
            float(len(intersection)) / float(len(left_support))
            if left_support
            else math.nan
        ),
        "right_overlap_ratio": (
            float(len(intersection)) / float(len(right_support))
            if right_support
            else math.nan
        ),
        "left_support_subset_of_right": bool(left_support <= right_support)
        if left_support
        else False,
        "right_support_subset_of_left": bool(right_support <= left_support)
        if right_support
        else False,
    }

def _baseline_quality(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["case", "field", "method", "baseline_quality"])
    frame = candidates.copy()
    frame["baseline_quality"] = pd.to_numeric(
        frame.get("p5_quality"),
        errors="coerce",
    ) - pd.to_numeric(frame.get("p5_delta_q"), errors="coerce")
    return (
        frame.groupby(["case", "field", "method"], as_index=False)["baseline_quality"]
        .median()
        .dropna()
    )

def _ensure_field_method(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    inferred = out.apply(_infer_field_method, axis=1)
    inferred_field = inferred.map(lambda item: item[0])
    inferred_method = inferred.map(lambda item: item[1])
    if "field" not in out.columns:
        out["field"] = inferred_field
    else:
        out["field"] = out["field"].where(out["field"].notna(), inferred_field)
    if "method" not in out.columns:
        out["method"] = inferred_method
    else:
        method = out["method"].astype("string")
        out["method"] = method.where(method.notna() & method.ne(""), inferred_method)
    return out

def _case_metadata(row: pd.Series) -> tuple[Any, Any, Any]:
    return row.get("case"), row.get("field"), row.get("method")

def _candidate_nodes(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        sketch = str(row.get(SKETCH_MEMBERSHIP_COLUMN, ""))
        if not sketch or sketch.lower() == "nan":
            continue
        baseline_quality = _safe_float(row.get("p5_quality")) - _safe_float(
            row.get("p5_delta_q")
        )
        candidate_index = _safe_int(row.get("candidate_index"), -1)
        rows.append(
            {
                "node_id": f"candidate:{row.get('case')}:{candidate_index}",
                "node_kind": "dongdaemun_candidate",
                "case": row.get("case"),
                "field": row.get("field"),
                "method": row.get("method"),
                "candidate_index": candidate_index,
                "seed": row.get("seed"),
                "randomness": "",
                "requested_n_iterations": "",
                "quality": row.get("p5_quality"),
                "quality_delta_vs_baseline": row.get("p5_delta_q"),
                "baseline_quality": baseline_quality,
                "selected_by_full_p5": row.get("selected_by_full_p5", ""),
                "p5_relative_delta_q_ppm": row.get("p5_relative_delta_q_ppm", ""),
                "p5_basin_signature": row.get("p5_basin_signature", ""),
                SKETCH_HASH_COLUMN: row.get(SKETCH_HASH_COLUMN, ""),
                "p5_basin_sketch_baseline_membership": row.get(
                    "p5_basin_sketch_baseline_membership",
                    "",
                ),
                SKETCH_MEMBERSHIP_COLUMN: row.get(SKETCH_MEMBERSHIP_COLUMN, ""),
                CHANGED_SUPPORT_COLUMN: row.get(CHANGED_SUPPORT_COLUMN, ""),
                "changed_support_size": len(_node_support(row)),
            }
        )
    return pd.DataFrame(rows)

def _vanilla_nodes(vanilla: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if vanilla.empty:
        return pd.DataFrame()
    frame = vanilla.copy()
    frame = frame.merge(baseline, on=["case", "field", "method"], how="left")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        sketch = str(row.get(SKETCH_MEMBERSHIP_COLUMN, ""))
        if not sketch or sketch.lower() == "nan":
            continue
        seed = _safe_int(row.get("seed"), -1)
        randomness = _safe_float(row.get("randomness"))
        iterations = str(row.get("requested_n_iterations", row.get("n_iterations", "")))
        quality = _safe_float(row.get("quality"))
        baseline_quality = _safe_float(row.get("baseline_quality"))
        rows.append(
            {
                "node_id": (
                    f"vanilla:{row.get('case')}:seed={seed}:"
                    f"r={randomness:g}:n={iterations}"
                ),
                "node_kind": "vanilla_seed_basin",
                "case": row.get("case"),
                "field": row.get("field"),
                "method": row.get("method"),
                "candidate_index": "",
                "seed": seed,
                "randomness": randomness,
                "requested_n_iterations": iterations,
                "quality": quality,
                "quality_delta_vs_baseline": quality - baseline_quality
                if math.isfinite(quality) and math.isfinite(baseline_quality)
                else math.nan,
                "baseline_quality": baseline_quality,
                "selected_by_full_p5": "",
                "p5_relative_delta_q_ppm": "",
                "p5_basin_signature": row.get("p5_basin_signature", ""),
                SKETCH_HASH_COLUMN: row.get(SKETCH_HASH_COLUMN, ""),
                "p5_basin_sketch_baseline_membership": row.get(
                    "p5_basin_sketch_baseline_membership",
                    "",
                ),
                SKETCH_MEMBERSHIP_COLUMN: row.get(SKETCH_MEMBERSHIP_COLUMN, ""),
                CHANGED_SUPPORT_COLUMN: row.get(CHANGED_SUPPORT_COLUMN, ""),
                "changed_support_size": len(_node_support(row)),
            }
        )
    return pd.DataFrame(rows)

def _baseline_nodes(candidates: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidates.empty:
        return pd.DataFrame()
    for _, base in baseline.iterrows():
        case, field, method = _case_metadata(base)
        group = candidates[
            (candidates["case"] == case)
            & (candidates["field"] == field)
            & (candidates["method"] == method)
        ]
        if group.empty:
            continue
        source = group.iloc[0]
        baseline_membership = str(source.get("p5_basin_sketch_baseline_membership", ""))
        if not baseline_membership or baseline_membership.lower() == "nan":
            continue
        quality = _safe_float(base.get("baseline_quality"))
        rows.append(
            {
                "node_id": f"baseline:{case}",
                "node_kind": "baseline",
                "case": case,
                "field": field,
                "method": method,
                "candidate_index": "",
                "seed": "",
                "randomness": "",
                "requested_n_iterations": "",
                "quality": quality,
                "quality_delta_vs_baseline": 0.0,
                "baseline_quality": quality,
                "selected_by_full_p5": "",
                "p5_relative_delta_q_ppm": "",
                "p5_basin_signature": "baseline",
                SKETCH_HASH_COLUMN: source.get(SKETCH_HASH_COLUMN, ""),
                "p5_basin_sketch_baseline_membership": baseline_membership,
                SKETCH_MEMBERSHIP_COLUMN: baseline_membership,
                CHANGED_SUPPORT_COLUMN: "",
                "changed_support_size": 0,
            }
        )
    return pd.DataFrame(rows)

def build_node_rows(candidates: pd.DataFrame, vanilla: pd.DataFrame) -> pd.DataFrame:
    baseline = _baseline_quality(candidates)
    nodes = [
        _baseline_nodes(candidates, baseline),
        _candidate_nodes(candidates),
        _vanilla_nodes(vanilla, baseline),
    ]
    nodes = [node for node in nodes if not node.empty]
    if not nodes:
        return pd.DataFrame()
    return pd.concat(nodes, ignore_index=True, sort=False)

def _edge_scope(left_kind: str, right_kind: str) -> str:
    kinds = {left_kind, right_kind}
    if kinds == {"vanilla_seed_basin"}:
        return "vanilla_to_vanilla"
    if kinds == {"dongdaemun_candidate"}:
        return "candidate_to_candidate"
    if "baseline" in kinds:
        other = right_kind if left_kind == "baseline" else left_kind
        return f"baseline_to_{other}"
    if kinds == {"vanilla_seed_basin", "dongdaemun_candidate"}:
        return "vanilla_to_candidate"
    return "other"

def build_edge_rows(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (_, left), (_, right) in combinations(nodes.iterrows(), 2):
        if left.get("case") != right.get("case"):
            continue
        left_hash = str(left.get(SKETCH_HASH_COLUMN, ""))
        right_hash = str(right.get(SKETCH_HASH_COLUMN, ""))
        if not left_hash or left_hash != right_hash:
            continue
        metrics = _sketch_metrics(left, right)
        support = _support_overlap(left, right)
        quality_left = _safe_float(left.get("quality"))
        quality_right = _safe_float(right.get("quality"))
        delta_left = _safe_float(left.get("quality_delta_vs_baseline"))
        delta_right = _safe_float(right.get("quality_delta_vs_baseline"))
        rows.append(
            {
                "case": left.get("case"),
                "field": left.get("field"),
                "method": left.get("method"),
                "left_node_id": left.get("node_id"),
                "right_node_id": right.get("node_id"),
                "left_node_kind": left.get("node_kind"),
                "right_node_kind": right.get("node_kind"),
                "edge_scope": _edge_scope(left.get("node_kind"), right.get("node_kind")),
                "left_quality": quality_left,
                "right_quality": quality_right,
                "right_minus_left_quality": quality_right - quality_left
                if math.isfinite(quality_left) and math.isfinite(quality_right)
                else math.nan,
                "left_delta_vs_baseline": delta_left,
                "right_delta_vs_baseline": delta_right,
                "right_minus_left_delta": delta_right - delta_left
                if math.isfinite(delta_left) and math.isfinite(delta_right)
                else math.nan,
                **metrics,
                **support,
            }
        )
    return pd.DataFrame(rows)

def build_transition_hypotheses(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    pairs = edges[edges["edge_scope"] == "vanilla_to_candidate"].copy()
    for _, edge in pairs.iterrows():
        left_kind = str(edge.get("left_node_kind"))
        candidate_is_left = left_kind == "dongdaemun_candidate"
        candidate_prefix = "left" if candidate_is_left else "right"
        vanilla_prefix = "right" if candidate_is_left else "left"
        candidate_delta = _safe_float(edge.get(f"{candidate_prefix}_delta_vs_baseline"))
        vanilla_delta = _safe_float(edge.get(f"{vanilla_prefix}_delta_vs_baseline"))
        candidate_support = _safe_int(edge.get(f"{candidate_prefix}_support_size"), 0) or 0
        vanilla_support = _safe_int(edge.get(f"{vanilla_prefix}_support_size"), 0) or 0
        candidate_overlap = _safe_float(edge.get(f"{candidate_prefix}_overlap_ratio"))
        vanilla_overlap = _safe_float(edge.get(f"{vanilla_prefix}_overlap_ratio"))
        candidate_subset = bool(edge.get(f"{candidate_prefix}_support_subset_of_{vanilla_prefix}"))
        vanilla_extra = max(vanilla_support - _safe_int(edge.get("support_intersection_size"), 0), 0)
        endpoint_distance = _safe_float(edge.get("endpoint_distance"))
        rows.append(
            {
                "case": edge.get("case"),
                "field": edge.get("field"),
                "method": edge.get("method"),
                "candidate_node_id": edge.get(f"{candidate_prefix}_node_id"),
                "vanilla_node_id": edge.get(f"{vanilla_prefix}_node_id"),
                "candidate_delta_vs_baseline": candidate_delta,
                "vanilla_delta_vs_baseline": vanilla_delta,
                "vanilla_minus_candidate_delta": vanilla_delta - candidate_delta
                if math.isfinite(vanilla_delta) and math.isfinite(candidate_delta)
                else math.nan,
                "endpoint_distance": endpoint_distance,
                "support_distance": edge.get("support_distance"),
                "candidate_support_size": candidate_support,
                "vanilla_support_size": vanilla_support,
                "support_intersection_size": edge.get("support_intersection_size"),
                "candidate_overlap_ratio": candidate_overlap,
                "vanilla_overlap_ratio": vanilla_overlap,
                "candidate_support_subset_of_vanilla": candidate_subset,
                "vanilla_extra_support_size": vanilla_extra,
                "endpoint_near": bool(
                    math.isfinite(endpoint_distance) and endpoint_distance <= 0.02
                ),
                "hypothesis": (
                    "candidate_local_core_inside_broader_vanilla"
                    if candidate_subset and vanilla_extra > 0
                    else "candidate_partially_overlaps_vanilla"
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["endpoint_distance", "vanilla_minus_candidate_delta"],
        ascending=[True, False],
        na_position="last",
    )

def _filter_frame(
    frame: pd.DataFrame,
    *,
    field: int | None,
    method: str | None,
    n_iterations: int | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = _ensure_field_method(frame)
    if field is not None and "field" in out.columns:
        out = out[pd.to_numeric(out["field"], errors="coerce") == int(field)]
    if method and "method" in out.columns:
        out = out[out["method"].astype(str) == str(method)]
    if n_iterations is not None and "requested_n_iterations" in out.columns:
        out = out[
            pd.to_numeric(out["requested_n_iterations"], errors="coerce")
            == int(n_iterations)
        ]
    return out

def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines

def write_report(
    path: Path,
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> None:
    lines = [
        "# Basin Transition Landscape Diagnostic",
        "",
        "This diagnostic treats vanilla seed outputs and Dongdaemun p5 candidates as observed basin nodes. It does not prove an executable transition operator yet.",
        "",
        "## Node Counts",
        "",
    ]
    if nodes.empty:
        lines.append("- no nodes")
    else:
        counts = nodes.groupby("node_kind").size().reset_index(name="count")
        lines.extend(_markdown_table(counts))
    lines.extend(["", "## Quality Distribution", ""])
    quality = (
        nodes.groupby("node_kind")["quality_delta_vs_baseline"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        if not nodes.empty
        else pd.DataFrame()
    )
    lines.extend(_markdown_table(quality))
    lines.extend(["", "## Candidate-Vanilla Transition Hypotheses", ""])
    display_cols = [
        "candidate_node_id",
        "vanilla_node_id",
        "candidate_delta_vs_baseline",
        "vanilla_delta_vs_baseline",
        "vanilla_minus_candidate_delta",
        "endpoint_distance",
        "support_distance",
        "candidate_support_size",
        "vanilla_support_size",
        "support_intersection_size",
        "candidate_overlap_ratio",
        "vanilla_overlap_ratio",
        "candidate_support_subset_of_vanilla",
        "vanilla_extra_support_size",
        "hypothesis",
    ]
    display = hypotheses[[c for c in display_cols if c in hypotheses.columns]].head(20)
    lines.extend(_markdown_table(display))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- A hypothesis row is a final-footprint relation, not a demonstrated transition.",
            "- Promotion requires executing a controlled basin-transition operator and comparing quality/cost against another vanilla seed run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_analysis(
    *,
    candidate_dirs: tuple[Path, ...],
    vanilla_dirs: tuple[Path, ...],
    output_dir: Path,
    field: int | None,
    method: str | None,
    n_iterations: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _filter_frame(
        _read_candidate_rows(candidate_dirs),
        field=field,
        method=method,
        n_iterations=None,
    )
    vanilla = _filter_frame(
        _read_vanilla_rows(list(vanilla_dirs)),
        field=field,
        method=method,
        n_iterations=n_iterations,
    )
    nodes = build_node_rows(candidates, vanilla)
    edges = build_edge_rows(nodes)
    hypotheses = build_transition_hypotheses(edges)
    nodes.to_csv(output_dir / NODE_ROWS_FILENAME, index=False)
    edges.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    hypotheses.to_csv(output_dir / HYPOTHESIS_ROWS_FILENAME, index=False)
    summary = {
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dirs": [str(path) for path in vanilla_dirs],
        "field": field,
        "method": method,
        "n_iterations": n_iterations,
        "node_rows": int(len(nodes)),
        "edge_rows": int(len(edges)),
        "hypothesis_rows": int(len(hypotheses)),
        "output_dir": str(output_dir),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, nodes=nodes, edges=edges, hypotheses=hypotheses)
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field", type=int, default=34)
    parser.add_argument("--method", default="cc_cosine")
    parser.add_argument("--n-iterations", type=int, default=10)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dirs=tuple(args.vanilla_dir or (DEFAULT_VANILLA_DIR,)),
        output_dir=args.output_dir,
        field=args.field,
        method=args.method,
        n_iterations=args.n_iterations,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
