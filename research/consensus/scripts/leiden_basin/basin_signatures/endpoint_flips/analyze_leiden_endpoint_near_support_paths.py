#!/usr/bin/env python3
"""Decompose endpoint-near/support-far Dongdaemun versus vanilla footprints."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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


import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from analyze_leiden_basin_reachability_audit import _read_vanilla_rows  # noqa: E402
from analyze_leiden_multibasin_signatures import (  # noqa: E402
    CHANGED_SUPPORT_COLUMN,
    SKETCH_HASH_COLUMN,
    SKETCH_MEMBERSHIP_COLUMN,
    _parse_sketch,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _case_candidate_rows,
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
    hash_u32_sequence,
)

DEFAULT_AUDIT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_reachability_audit_with_field34_cc_compatible_sketch"
)
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
    "vanilla_reachability_sweep_field34_cc_compatible_sketch"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "endpoint_near_support_path_field34_cc"
)
SUMMARY_FILENAME = "endpoint_near_support_path_summary.csv"
CLUSTER_ROWS_FILENAME = "endpoint_near_support_path_cluster_rows.csv"
REPORT_FILENAME = "endpoint_near_support_path_report.md"

def parse_node_set(value: Any) -> set[int]:
    values = _parse_sketch(value)
    return {int(node) for node in values}

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

def _row_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        row.get("candidate_eval_mode"),
        row.get("case"),
        _safe_int(row.get("seed")),
        _safe_int(row.get("candidate_budget")),
        _safe_int(row.get("max_group_candidates")),
        _safe_int(row.get("candidate_index")),
    )

def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[Any, ...], pd.Series]:
    lookup: dict[tuple[Any, ...], pd.Series] = {}
    if candidates.empty:
        return lookup
    for _, row in candidates.iterrows():
        lookup[_row_key(row)] = row
    return lookup

def _match_candidate(target: pd.Series, candidates: pd.DataFrame) -> pd.Series | None:
    lookup = _candidate_lookup(candidates)
    exact = lookup.get(_row_key(target))
    if exact is not None:
        return exact
    case_rows = _case_candidate_rows(candidates, str(target.get("case", "")))
    if case_rows.empty:
        return None
    candidate_index = _safe_int(target.get("candidate_index"), -1)
    rows = case_rows[
        pd.to_numeric(case_rows.get("candidate_index"), errors="coerce")
        == candidate_index
    ]
    if rows.empty:
        return None
    return rows.iloc[0]

def _match_vanilla(target: pd.Series, vanilla_case: pd.DataFrame) -> pd.Series | None:
    if vanilla_case.empty:
        return None
    out = vanilla_case.copy()
    seed = _safe_int(target.get("best_sketch_seed"))
    if seed is not None and "seed" in out.columns:
        out = out[pd.to_numeric(out["seed"], errors="coerce") == seed]
    randomness = _safe_float(target.get("best_sketch_randomness"))
    if math.isfinite(randomness) and "randomness" in out.columns:
        values = pd.to_numeric(out["randomness"], errors="coerce")
        out = out[np.isclose(values, randomness)]
    requested_int = _safe_int(target.get("best_sketch_requested_n_iterations"))
    requested = str(target.get("best_sketch_requested_n_iterations", ""))
    if requested and requested.lower() != "nan" and "requested_n_iterations" in out.columns:
        requested_values = out["requested_n_iterations"].astype(str)
        if requested_int is None:
            out = out[requested_values == requested]
        else:
            numeric_values = pd.to_numeric(out["requested_n_iterations"], errors="coerce")
            out = out[numeric_values == requested_int]
    if out.empty:
        return None
    return out.iloc[0]

def _top_counts(values: list[int], *, limit: int = 8) -> str:
    if not values:
        return ""
    counts = Counter(values).most_common(limit)
    return ";".join(f"{label}:{count}" for label, count in counts)

def _segment_cluster_rows(
    *,
    target: pd.Series,
    segment: str,
    nodes: set[int],
    node_to_index: dict[int, int],
    baseline_labels: np.ndarray,
    dongdaemun_labels: np.ndarray,
    vanilla_labels: np.ndarray,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, int, int]] = Counter()
    missing = 0
    for node in nodes:
        index = node_to_index.get(int(node))
        if index is None:
            missing += 1
            continue
        counts[
            (
                int(baseline_labels[index]),
                int(dongdaemun_labels[index]),
                int(vanilla_labels[index]),
            )
        ] += 1
    rows = [
        {
            "case": target.get("case"),
            "field": target.get("field"),
            "method": target.get("method"),
            "candidate_index": target.get("candidate_index"),
            "target_class": target.get("target_class"),
            "segment": segment,
            "baseline_cluster": baseline_cluster,
            "dongdaemun_cluster": dongdaemun_cluster,
            "vanilla_cluster": vanilla_cluster,
            "node_count": count,
            "missing_from_sketch": 0,
        }
        for (baseline_cluster, dongdaemun_cluster, vanilla_cluster), count in counts.most_common()
    ]
    if missing:
        rows.append(
            {
                "case": target.get("case"),
                "field": target.get("field"),
                "method": target.get("method"),
                "candidate_index": target.get("candidate_index"),
                "target_class": target.get("target_class"),
                "segment": segment,
                "baseline_cluster": "",
                "dongdaemun_cluster": "",
                "vanilla_cluster": "",
                "node_count": 0,
                "missing_from_sketch": int(missing),
            }
        )
    return rows

def _segment_summary(
    *,
    nodes: set[int],
    node_to_index: dict[int, int],
    baseline_labels: np.ndarray,
    dongdaemun_labels: np.ndarray,
    vanilla_labels: np.ndarray,
) -> dict[str, Any]:
    baseline: list[int] = []
    dongdaemun: list[int] = []
    vanilla: list[int] = []
    for node in nodes:
        index = node_to_index.get(int(node))
        if index is None:
            continue
        baseline.append(int(baseline_labels[index]))
        dongdaemun.append(int(dongdaemun_labels[index]))
        vanilla.append(int(vanilla_labels[index]))
    return {
        "baseline_cluster_top": _top_counts(baseline),
        "dongdaemun_cluster_top": _top_counts(dongdaemun),
        "vanilla_cluster_top": _top_counts(vanilla),
        "distinct_baseline_clusters": len(set(baseline)),
        "distinct_dongdaemun_clusters": len(set(dongdaemun)),
        "distinct_vanilla_clusters": len(set(vanilla)),
    }

def _same_case_vanilla(vanilla: pd.DataFrame, case: str) -> pd.DataFrame:
    if vanilla.empty or "case" not in vanilla.columns:
        return pd.DataFrame()
    return vanilla[vanilla["case"].astype(str) == str(case)].copy()

def _build_sketch_context(
    *,
    graph_dir: Path,
    case_candidates: pd.DataFrame,
    resolution: float,
    baseline_seed: int,
    baseline_iterations: int,
    baseline_randomness: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    graph, node_weights, arrays = _load_graph(graph_dir)
    baseline = graph.run_leiden(
        resolution=float(resolution),
        seed=int(baseline_seed),
        n_iterations=int(baseline_iterations),
        randomness=float(baseline_randomness),
        membership_dtype=np.uint32,
    )
    return compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=np.asarray(baseline.membership, dtype=np.uint64),
        node_weights=node_weights,
        candidate_rows=case_candidates,
    )

def build_path_rows(
    *,
    audit_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dirs: tuple[Path, ...],
    field: int | None,
    method: str | None,
    resolution: float,
    baseline_seed: int,
    baseline_iterations: int,
    baseline_randomness: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reachability = pd.read_csv(audit_dir / "dongdaemun_basin_reachability_rows.csv")
    targets = pd.read_csv(audit_dir / "dongdaemun_basin_reachability_target_rows.csv")
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla = _read_vanilla_rows(list(vanilla_dirs))
    near = reachability[
        reachability.get("endpoint_near_support_far", False).fillna(False).map(bool)
    ].copy()
    if field is not None and "field" in near.columns:
        near = near[pd.to_numeric(near["field"], errors="coerce") == int(field)]
    if method and "method" in near.columns:
        near = near[near["method"].astype(str) == str(method)]
    target_lookup = {
        (
            row.get("case"),
            _safe_int(row.get("candidate_index")),
        ): row
        for _, row in targets.iterrows()
    }
    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    sketch_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    candidate_case_cache: dict[str, pd.DataFrame] = {}
    for _, reachability_row in near.iterrows():
        case = str(reachability_row.get("case", ""))
        candidate_index = _safe_int(reachability_row.get("candidate_index"), -1)
        target = target_lookup.get((case, candidate_index))
        if target is None:
            continue
        case_candidates = candidate_case_cache.get(case)
        if case_candidates is None:
            case_candidates = _case_candidate_rows(candidates, case)
            candidate_case_cache[case] = case_candidates
        candidate = _match_candidate(target, case_candidates)
        vanilla_case = _same_case_vanilla(vanilla, case)
        vanilla_row = _match_vanilla(reachability_row, vanilla_case)
        if candidate is None or vanilla_row is None:
            continue
        graph_dir = Path(str(vanilla_row.get("graph_dir", ""))).expanduser()
        if not graph_dir.is_absolute():
            graph_dir = (REPO_ROOT / graph_dir).resolve()
        if case not in sketch_cache:
            sketch_cache[case] = _build_sketch_context(
                graph_dir=graph_dir,
                case_candidates=case_candidates,
                resolution=resolution,
                baseline_seed=baseline_seed,
                baseline_iterations=baseline_iterations,
                baseline_randomness=baseline_randomness,
            )
        sketch_nodes, sketch_context = sketch_cache[case]
        sketch_hash = hash_u32_sequence(sketch_nodes)
        node_to_index = {int(node): index for index, node in enumerate(sketch_nodes)}
        baseline_labels = _parse_sketch(candidate.get("p5_basin_sketch_baseline_membership"))
        dongdaemun_labels = _parse_sketch(target.get(SKETCH_MEMBERSHIP_COLUMN))
        vanilla_labels = _parse_sketch(vanilla_row.get(SKETCH_MEMBERSHIP_COLUMN))
        if (
            baseline_labels.size != sketch_nodes.size
            or dongdaemun_labels.size != sketch_nodes.size
            or vanilla_labels.size != sketch_nodes.size
        ):
            continue
        target_support = parse_node_set(target.get(CHANGED_SUPPORT_COLUMN))
        vanilla_support = parse_node_set(vanilla_row.get(CHANGED_SUPPORT_COLUMN))
        intersection = target_support & vanilla_support
        target_only = target_support - vanilla_support
        vanilla_only = vanilla_support - target_support
        union = target_support | vanilla_support
        sketch_count = int(sketch_nodes.size)
        sketch_node_set = set(int(node) for node in sketch_nodes)
        target_support_in_sketch = target_support & sketch_node_set
        vanilla_support_in_sketch = vanilla_support & sketch_node_set
        intersection_in_sketch = intersection & sketch_node_set
        target_only_in_sketch = target_only & sketch_node_set
        vanilla_only_in_sketch = vanilla_only & sketch_node_set
        union_in_sketch = union & sketch_node_set
        subset_target_in_vanilla = bool(target_support <= vanilla_support)
        subset_vanilla_in_target = bool(vanilla_support <= target_support)
        overlap_ratio_target = (
            float(len(intersection)) / float(len(target_support))
            if target_support
            else math.nan
        )
        overlap_ratio_vanilla = (
            float(len(intersection)) / float(len(vanilla_support))
            if vanilla_support
            else math.nan
        )
        row = {
            "case": case,
            "field": target.get("field"),
            "method": target.get("method"),
            "target_class": target.get("target_class"),
            "candidate_index": candidate_index,
            "p5_delta_q": target.get("p5_delta_q"),
            "q_gap_to_best": target.get("q_gap_to_best"),
            "vanilla_seed": vanilla_row.get("seed"),
            "vanilla_randomness": vanilla_row.get("randomness"),
            "vanilla_requested_n_iterations": vanilla_row.get(
                "requested_n_iterations",
                vanilla_row.get("n_iterations"),
            ),
            "endpoint_distance": reachability_row.get("best_endpoint_distance"),
            "support_distance": reachability_row.get("best_support_distance"),
            "support_similarity": reachability_row.get("best_support_similarity"),
            "sketch_node_count": sketch_count,
            "reconstructed_sketch_hash": sketch_hash,
            "candidate_sketch_hash": target.get(SKETCH_HASH_COLUMN),
            "vanilla_sketch_hash": vanilla_row.get(SKETCH_HASH_COLUMN),
            "sketch_hash_matches": bool(
                sketch_hash == str(target.get(SKETCH_HASH_COLUMN))
                == str(vanilla_row.get(SKETCH_HASH_COLUMN))
            ),
            "target_support_size": len(target_support),
            "vanilla_support_size": len(vanilla_support),
            "support_intersection_size": len(intersection),
            "target_only_support_size": len(target_only),
            "vanilla_only_support_size": len(vanilla_only),
            "support_union_size": len(union),
            "endpoint_sketch_node_count": sketch_count,
            "support_union_to_endpoint_sketch_size_ratio": (
                float(len(union)) / float(sketch_count) if sketch_count else math.nan
            ),
            "target_support_to_endpoint_sketch_size_ratio": (
                float(len(target_support)) / float(sketch_count)
                if sketch_count
                else math.nan
            ),
            "vanilla_support_to_endpoint_sketch_size_ratio": (
                float(len(vanilla_support)) / float(sketch_count)
                if sketch_count
                else math.nan
            ),
            "support_union_in_endpoint_sketch_size": len(union_in_sketch),
            "support_union_outside_endpoint_sketch_size": len(union - sketch_node_set),
            "target_support_in_endpoint_sketch_size": len(target_support_in_sketch),
            "target_support_outside_endpoint_sketch_size": len(
                target_support - sketch_node_set
            ),
            "vanilla_support_in_endpoint_sketch_size": len(vanilla_support_in_sketch),
            "vanilla_support_outside_endpoint_sketch_size": len(
                vanilla_support - sketch_node_set
            ),
            "intersection_in_endpoint_sketch_size": len(intersection_in_sketch),
            "target_only_in_endpoint_sketch_size": len(target_only_in_sketch),
            "vanilla_only_in_endpoint_sketch_size": len(vanilla_only_in_sketch),
            "target_support_overlap_ratio": overlap_ratio_target,
            "vanilla_support_overlap_ratio": overlap_ratio_vanilla,
            "target_support_subset_of_vanilla": subset_target_in_vanilla,
            "vanilla_support_subset_of_target": subset_vanilla_in_target,
            **{
                f"target_support_{key}": value
                for key, value in _segment_summary(
                    nodes=target_support,
                    node_to_index=node_to_index,
                    baseline_labels=baseline_labels,
                    dongdaemun_labels=dongdaemun_labels,
                    vanilla_labels=vanilla_labels,
                ).items()
            },
            **{
                f"vanilla_only_{key}": value
                for key, value in _segment_summary(
                    nodes=vanilla_only,
                    node_to_index=node_to_index,
                    baseline_labels=baseline_labels,
                    dongdaemun_labels=dongdaemun_labels,
                    vanilla_labels=vanilla_labels,
                ).items()
            },
        }
        summary_rows.append(row)
        segments = {
            "target_support": target_support,
            "vanilla_support": vanilla_support,
            "intersection": intersection,
            "target_only": target_only,
            "vanilla_only": vanilla_only,
        }
        for segment, nodes in segments.items():
            cluster_rows.extend(
                _segment_cluster_rows(
                    target=target,
                    segment=segment,
                    nodes=nodes,
                    node_to_index=node_to_index,
                    baseline_labels=baseline_labels,
                    dongdaemun_labels=dongdaemun_labels,
                    vanilla_labels=vanilla_labels,
                )
            )
    metadata = {
        "audit_dir": str(audit_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dirs": [str(path) for path in vanilla_dirs],
        "near_row_count": int(len(near)),
        "summary_row_count": int(len(summary_rows)),
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(cluster_rows), metadata

def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
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
    return "\n".join(lines)

def write_outputs(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    cluster_rows: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / SUMMARY_FILENAME, index=False)
    cluster_rows.to_csv(output_dir / CLUSTER_ROWS_FILENAME, index=False)
    (output_dir / "endpoint_near_support_path_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_columns = [
        "field",
        "method",
        "candidate_index",
        "endpoint_distance",
        "support_distance",
        "target_support_size",
        "vanilla_support_size",
        "support_intersection_size",
        "target_only_support_size",
        "vanilla_only_support_size",
        "target_support_overlap_ratio",
        "vanilla_support_overlap_ratio",
        "target_support_subset_of_vanilla",
        "support_union_in_endpoint_sketch_size",
        "support_union_outside_endpoint_sketch_size",
        "vanilla_only_in_endpoint_sketch_size",
        "vanilla_support_outside_endpoint_sketch_size",
    ]
    report_frame = summary[[column for column in report_columns if column in summary.columns]]
    lines = [
        "# Endpoint-Near Support-Far Path Footprint Review",
        "",
        "This artifact compares final-footprint evidence for `baseline -> Dongdaemun candidate` and `baseline -> vanilla` on the same sketch nodes. It does not record move-sequence trajectories.",
        "",
        "## Summary",
        "",
        _markdown_table(report_frame),
        "",
        "## Interpretation Guardrail",
        "",
        "- A larger vanilla support footprint means vanilla changed more sampled support nodes relative to the same baseline.",
        "- Changed-support nodes are sampled independently from the endpoint sketch; cluster-label breakdowns only cover support nodes that also fall inside the endpoint sketch.",
        "- It does not by itself prove vanilla took a longer trajectory; that requires move-sequence tracing.",
        "- If Dongdaemun support is mostly contained in vanilla support, the working hypothesis is a local shortcut inside a broader vanilla footprint.",
    ]
    (output_dir / REPORT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=list(DEFAULT_CANDIDATE_DIRS),
    )
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        action="append",
        default=[DEFAULT_VANILLA_DIR],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field", type=int, default=34)
    parser.add_argument("--method", default="cc_cosine")
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--baseline-seed", type=int, default=11)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--baseline-randomness", type=float, default=0.01)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary, cluster_rows, metadata = build_path_rows(
        audit_dir=args.audit_dir,
        candidate_dirs=tuple(args.candidate_dir),
        vanilla_dirs=tuple(args.vanilla_dir),
        field=args.field,
        method=args.method,
        resolution=float(args.resolution),
        baseline_seed=int(args.baseline_seed),
        baseline_iterations=int(args.baseline_iterations),
        baseline_randomness=float(args.baseline_randomness),
    )
    write_outputs(
        output_dir=args.output_dir,
        summary=summary,
        cluster_rows=cluster_rows,
        metadata=metadata,
    )
    print(
        {
            "summary_rows": int(len(summary)),
            "cluster_rows": int(len(cluster_rows)),
            "output_dir": str(args.output_dir),
        }
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
