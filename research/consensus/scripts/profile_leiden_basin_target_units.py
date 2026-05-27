#!/usr/bin/env python3
"""Profile simple target-node units for Leiden basin-transition diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from profile_leiden_basin_ordered_flips_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PROFILE_BATCH_DIR,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_VANILLA_DIR,
    VANILLA_ROWS_FILENAME,
    _find_candidate_row,
    _find_vanilla_row,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    v_only_support_nodes,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    TARGET_UNIT_TYPES,
    build_target_unit_rows,
)


DEFAULT_OUTPUT_DIR = DEFAULT_PROFILE_BATCH_DIR.parent / "basin_transition_target_units_field34_cc_v0"
UNIT_ROWS_FILENAME = "target_unit_rows.csv"
CASE_ROWS_FILENAME = "target_unit_case_rows.csv"
SUMMARY_FILENAME = "target_unit_summary.json"
CONFIG_FILENAME = "target_unit_config.json"
REPORT_FILENAME = "target_unit_report.md"


def _parse_csv_tuple(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


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
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _selected_pair_rows(prefix_dir: Path, pair_ids: tuple[str, ...]) -> pd.DataFrame:
    rows = pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME)
    if pair_ids:
        rows = rows[rows["pair_id"].astype(str).isin(pair_ids)].copy()
    if rows.empty:
        raise ValueError("No pair rows selected for target-unit profiling")
    return (
        rows.sort_values(["pair_id", "barrier_aware_score"], ascending=[True, False])
        .drop_duplicates("pair_id", keep="first")
    )


def _profile_case(
    *,
    pair_row: pd.Series,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    unit_types: tuple[str, ...],
    triangle_support_min: int,
) -> pd.DataFrame:
    case = str(pair_row["case"])
    pair_id = str(pair_row["pair_id"])
    candidate_index = int(pair_row["candidate_index"])
    vanilla_seed = int(pair_row["vanilla_seed"])
    vanilla_randomness = float(pair_row["vanilla_randomness"])
    vanilla_n = str(pair_row["vanilla_requested_n_iterations"])
    candidate_rows = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)
    candidate_row = _find_candidate_row(
        candidate_rows,
        case=case,
        candidate_index=candidate_index,
    )
    vanilla_row = _find_vanilla_row(
        vanilla_rows,
        case=case,
        seed=vanilla_seed,
        randomness=vanilla_randomness,
        n_iterations=vanilla_n,
    )
    graph_dir = Path(str(vanilla_row["graph_dir"]))
    graph, node_weights, arrays = _load_graph(graph_dir)
    baseline = _run_leiden(
        graph,
        resolution=resolution,
        seed=int(candidate_row.get("seed", 0)),
        n_iterations=baseline_iterations,
        randomness=randomness,
    )
    candidate = _recreate_candidate(
        graph=graph,
        arrays=arrays,
        node_weights=node_weights,
        baseline_membership=baseline.membership,
        baseline_quality=baseline.quality,
        row=candidate_row,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
        polish_iterations=candidate_polish_iterations,
    )
    vanilla = _run_leiden(
        graph,
        resolution=resolution,
        seed=vanilla_seed,
        n_iterations=int(_safe_int(vanilla_n, baseline_iterations) or baseline_iterations),
        randomness=vanilla_randomness,
    )
    candidate_support, vanilla_support, target_nodes = v_only_support_nodes(
        baseline.membership,
        candidate.recreated.membership,
        vanilla.membership,
    )
    rows = build_target_unit_rows(
        target_nodes=target_nodes,
        candidate_support_nodes=candidate_support,
        baseline_membership=baseline.membership,
        candidate_membership=candidate.recreated.membership,
        vanilla_membership=vanilla.membership,
        src=np.asarray(arrays.src, dtype=np.uint32),
        dst=np.asarray(arrays.dst, dtype=np.uint32),
        weight=np.asarray(arrays.weight, dtype=np.float64),
        node_count=int(baseline.membership.size),
        unit_types=unit_types,
        triangle_support_min=triangle_support_min,
    )
    if rows.empty:
        return rows
    rows.insert(0, "pair_id", pair_id)
    rows.insert(0, "method", pair_row.get("method", ""))
    rows.insert(0, "field", pair_row.get("field", ""))
    rows.insert(0, "case", case)
    rows["candidate_index"] = candidate_index
    rows["vanilla_seed"] = vanilla_seed
    rows["vanilla_randomness"] = vanilla_randomness
    rows["vanilla_requested_n_iterations"] = vanilla_n
    rows["candidate_support_size"] = int(candidate_support.size)
    rows["vanilla_support_size"] = int(vanilla_support.size)
    rows["v_only_support_size"] = int(target_nodes.size)
    rows["triangle_support_min"] = int(triangle_support_min)
    return rows


def _case_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(["pair_id", "unit_type"], sort=True)
    out = grouped.agg(
        unit_rows=("unit_id", "size"),
        target_node_count=("target_node_count", "max"),
        covered_node_count=("unit_node_count", "sum"),
        max_unit_node_count=("unit_node_count", "max"),
        median_unit_node_count=("unit_node_count", "median"),
        median_density=("unit_density", "median"),
        median_conductance=("unit_conductance", "median"),
        max_triangle_edge_fraction=("triangle_edge_fraction", "max"),
        median_triangle_edge_fraction=("triangle_edge_fraction", "median"),
        max_mean_edge_support=("mean_edge_support", "max"),
        max_pull_to_candidate_support_weight=("pull_to_candidate_support_weight", "max"),
        median_candidate_closure_ratio=("candidate_closure_ratio", "median"),
        max_candidate_closure_ratio=("candidate_closure_ratio", "max"),
    ).reset_index()
    out["coverage_ratio"] = out["covered_node_count"] / out["target_node_count"].clip(lower=1)
    return out


def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Basin Target Unit Profile v0",
        "",
        "This artifact profiles simple unit definitions over basin-transition target nodes.",
        "",
        "It is diagnostic-only. Exact clique search is intentionally avoided; `triangle_supported_component` uses target-induced common-neighbor support as a cheap clique-like cohesion proxy.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "prefix_dir",
        "unit_rows",
        "case_rows",
        "pair_ids",
        "unit_types",
        "triangle_support_min",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Unit-Type Case Rows", ""])
    case_cols = [
        "pair_id",
        "unit_type",
        "unit_rows",
        "target_node_count",
        "covered_node_count",
        "coverage_ratio",
        "max_unit_node_count",
        "median_unit_node_count",
        "median_density",
        "median_conductance",
        "max_triangle_edge_fraction",
        "median_triangle_edge_fraction",
        "max_mean_edge_support",
        "max_pull_to_candidate_support_weight",
        "median_candidate_closure_ratio",
        "max_candidate_closure_ratio",
    ]
    lines.extend(
        _markdown_table(
            case_rows[[c for c in case_cols if c in case_rows.columns]],
            max_rows=80,
        )
    )
    lines.extend(["", "## Top Dense Units", ""])
    top_cols = [
        "pair_id",
        "unit_type",
        "unit_id",
        "unit_node_count",
        "unit_density",
        "triangle_edge_fraction",
        "mean_edge_support",
        "unit_conductance",
        "pull_to_candidate_support_weight",
        "candidate_closure_ratio",
        "node_ids",
    ]
    if rows.empty:
        dense = rows
    else:
        dense = rows.sort_values(
            [
                "triangle_edge_fraction",
                "unit_density",
                "unit_node_count",
                "pull_to_candidate_support_weight",
            ],
            ascending=[False, False, False, False],
        )
    lines.extend(
        _markdown_table(dense[[c for c in top_cols if c in dense.columns]], max_rows=40)
    )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- Unit cohesion is not an operator win.",
            "- Prefer units that improve target progress per mutable node in a later action test.",
            "- Treat very large closure ratios as a warning that the unit may expand into a broad split/merge problem.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(
    *,
    prefix_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_ids: tuple[str, ...],
    baseline_iterations: int,
    candidate_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    unit_types: tuple[str, ...],
    triangle_support_min: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_pair_rows(prefix_dir, pair_ids)
    frames: list[pd.DataFrame] = []
    for _, pair_row in selected.iterrows():
        frames.append(
            _profile_case(
                pair_row=pair_row,
                candidate_dirs=candidate_dirs,
                vanilla_dir=vanilla_dir,
                baseline_iterations=baseline_iterations,
                candidate_polish_iterations=candidate_polish_iterations,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
                unit_types=unit_types,
                triangle_support_min=triangle_support_min,
            )
        )
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    case_rows = _case_rows(rows)
    if not rows.empty:
        rows = rows.sort_values(
            ["pair_id", "unit_type", "unit_node_count", "unit_id"],
            ascending=[True, True, False, True],
        )
    rows.to_csv(output_dir / UNIT_ROWS_FILENAME, index=False)
    case_rows.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    config = {
        "prefix_dir": str(prefix_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_ids": list(pair_ids),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "unit_types": list(unit_types),
        "triangle_support_min": int(triangle_support_min),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_target_units.v0",
        "output_dir": str(output_dir),
        "unit_rows": int(len(rows)),
        "case_rows": int(len(case_rows)),
        **config,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        case_rows=case_rows,
        summary=summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-ids", default="c0-s11-r0.001,c2-s11-r0")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--unit-types", default=",".join(TARGET_UNIT_TYPES))
    parser.add_argument("--triangle-support-min", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(
        prefix_dir=args.prefix_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        unit_types=_parse_csv_tuple(args.unit_types, TARGET_UNIT_TYPES),
        triangle_support_min=args.triangle_support_min,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
