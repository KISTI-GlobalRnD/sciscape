#!/usr/bin/env python3
"""Pilot controlled basin-transition operators for endpoint-near basins.

This is diagnostic-only. It recreates full baseline/candidate/vanilla
memberships for a small case, then tests restricted polish operators around the
candidate changed-support core.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from analyze_leiden_basin_reachability_audit import (  # noqa: E402
    _coassignment_bits,
    _jaccard_distance,
)
from analyze_leiden_basin_transition_landscape import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
)
from analyze_leiden_multibasin_signatures import _parse_sketch  # noqa: E402
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _best_partner_maps,
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
    encode_membership_sketch,
    hash_u32_sequence,
)
from run_leiden_hysteresis_work_acceleration_monitor import (  # noqa: E402
    _compact_membership,
    _reconstruct_external_group,
)


DEFAULT_LANDSCAPE_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_landscape_field34_cc"
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
    "basin_transition_operator_pilot_field34_cc"
)
HYPOTHESES_FILENAME = "basin_transition_landscape_hypotheses.csv"
VANILLA_ROWS_FILENAME = "vanilla_basin_rows.csv"
ROWS_FILENAME = "basin_transition_operator_rows.csv"
SUMMARY_FILENAME = "basin_transition_operator_summary.json"
REPORT_FILENAME = "basin_transition_operator_report.md"


@dataclass(frozen=True)
class RecreatedMembership:
    membership: np.ndarray
    quality: float
    elapsed_sec: float


@dataclass(frozen=True)
class CandidateMembership:
    recreated: RecreatedMembership
    row: pd.Series
    group_nodes: np.ndarray
    support_nodes: np.ndarray


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


def _parse_candidate_index(node_id: Any) -> int:
    text = str(node_id)
    match = re.search(r":(\d+)$", text)
    if not match:
        raise ValueError(f"Could not parse candidate index from {text!r}")
    return int(match.group(1))


def _parse_vanilla_config(node_id: Any) -> tuple[int, float, str]:
    text = str(node_id)
    match = re.search(r":seed=(\d+):r=([^:]+):n=(.+)$", text)
    if not match:
        raise ValueError(f"Could not parse vanilla config from {text!r}")
    return int(match.group(1)), float(match.group(2)), match.group(3)


def changed_support_nodes(
    baseline: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.uint64)
    membership = np.asarray(membership, dtype=np.uint64)
    if baseline.shape != membership.shape:
        raise ValueError("baseline and membership must have the same shape")
    baseline_best, membership_best = _best_partner_maps(baseline, membership)
    changed: list[int] = []
    for node, (base_label, label) in enumerate(zip(baseline, membership, strict=False)):
        base = int(base_label)
        current = int(label)
        baseline_aligned = baseline_best.get(base) == current
        membership_aligned = membership_best.get(current) == base
        if not (baseline_aligned and membership_aligned):
            changed.append(node)
    return np.asarray(changed, dtype=np.uint32)


def support_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    distance, intersection, union = _jaccard_distance(
        np.asarray(left, dtype=np.uint32),
        np.asarray(right, dtype=np.uint32),
    )
    return float(distance), int(intersection), int(union)


def endpoint_distance(
    left_membership: np.ndarray,
    right_membership: np.ndarray,
    sketch_nodes: np.ndarray,
) -> float:
    nodes = np.asarray(sketch_nodes, dtype=np.int64)
    if nodes.size == 0:
        return math.nan
    left = np.asarray(left_membership, dtype=np.uint64)[nodes]
    right = np.asarray(right_membership, dtype=np.uint64)[nodes]
    left_bits = _coassignment_bits(left)
    right_bits = _coassignment_bits(right)
    if left_bits.size == 0 or left_bits.size != right_bits.size:
        return math.nan
    return float(np.mean(left_bits != right_bits))


def transplant_support_groups(
    base_membership: np.ndarray,
    donor_membership: np.ndarray,
    support_nodes: np.ndarray,
) -> np.ndarray:
    out = np.asarray(base_membership, dtype=np.uint64).copy()
    support = np.asarray(support_nodes, dtype=np.int64)
    if support.size == 0:
        return out
    offset = int(out.max(initial=0)) + 1
    donor_labels = np.asarray(donor_membership, dtype=np.uint64)[support]
    label_map: dict[int, int] = {}
    next_label = offset
    for node, donor_label in zip(support, donor_labels, strict=False):
        donor = int(donor_label)
        if donor not in label_map:
            label_map[donor] = next_label
            next_label += 1
        out[int(node)] = np.uint64(label_map[donor])
    return out


def fixed_outside(nodes: int, support_nodes: np.ndarray) -> np.ndarray:
    fixed = np.ones(int(nodes), dtype=np.bool_)
    fixed[np.asarray(support_nodes, dtype=np.int64)] = False
    return fixed


def _candidate_key(candidate_index: int) -> str:
    return f"candidate:{candidate_index}"


def _vanilla_key(seed: int, randomness: float, n_iterations: str) -> str:
    return f"vanilla:seed={seed}:r={randomness:g}:n={n_iterations}"


def _find_candidate_row(
    candidates: pd.DataFrame,
    *,
    case: str,
    candidate_index: int,
) -> pd.Series:
    rows = candidates[
        (candidates["case"].astype(str) == str(case))
        & (
            pd.to_numeric(candidates["candidate_index"], errors="coerce")
            == int(candidate_index)
        )
    ]
    if rows.empty:
        raise ValueError(f"Missing candidate row for {case} index={candidate_index}")
    return rows.iloc[0]


def _find_vanilla_row(
    vanilla: pd.DataFrame,
    *,
    case: str,
    seed: int,
    randomness: float,
    n_iterations: str,
) -> pd.Series:
    rows = vanilla[vanilla["case"].astype(str) == str(case)].copy()
    rows = rows[pd.to_numeric(rows["seed"], errors="coerce") == int(seed)]
    rows = rows[np.isclose(pd.to_numeric(rows["randomness"], errors="coerce"), randomness)]
    numeric_n = _safe_int(n_iterations)
    if numeric_n is None:
        rows = rows[rows["requested_n_iterations"].astype(str) == str(n_iterations)]
    else:
        rows = rows[
            pd.to_numeric(rows["requested_n_iterations"], errors="coerce")
            == int(numeric_n)
        ]
    if rows.empty:
        raise ValueError(
            f"Missing vanilla row for {case} seed={seed} r={randomness:g} n={n_iterations}"
        )
    return rows.iloc[0]


def _run_leiden(
    graph: Any,
    *,
    resolution: float,
    seed: int,
    n_iterations: int,
    randomness: float,
    initial_membership: np.ndarray | None = None,
    fixed_nodes: np.ndarray | None = None,
) -> RecreatedMembership:
    start = time.perf_counter()
    result = graph.run_leiden(
        resolution=float(resolution),
        seed=int(seed),
        n_iterations=int(n_iterations),
        randomness=float(randomness),
        initial_membership=initial_membership,
        fixed_nodes=fixed_nodes,
    )
    return RecreatedMembership(
        membership=np.asarray(result.membership, dtype=np.uint64),
        quality=float(result.quality),
        elapsed_sec=float(time.perf_counter() - start),
    )


def _recreate_candidate(
    *,
    graph: Any,
    arrays: Any,
    node_weights: np.ndarray,
    baseline_membership: np.ndarray,
    baseline_quality: float,
    row: pd.Series,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_iterations: int,
) -> CandidateMembership:
    source = int(row["source_cluster"])
    target = int(row["target_cluster"])
    group_nodes, _reconstruction = _reconstruct_external_group(
        src=arrays.src,
        dst=arrays.dst,
        weight=arrays.weight,
        membership=baseline_membership,
        node_weights=node_weights,
        source_cluster=source,
        target_cluster=target,
    )
    perturbed = np.asarray(baseline_membership, dtype=np.uint64).copy()
    perturbed[np.asarray(group_nodes, dtype=np.int64)] = np.uint64(target)
    perturbed = _compact_membership(perturbed)
    candidate_index = int(row["candidate_index"])
    recreated = _run_leiden(
        graph,
        resolution=resolution,
        seed=int(row.get("seed", 0)) + int(perturb_seed_offset) + candidate_index,
        n_iterations=polish_iterations,
        randomness=randomness,
        initial_membership=perturbed,
    )
    support = changed_support_nodes(baseline_membership, recreated.membership)
    if not math.isfinite(recreated.quality - baseline_quality):
        raise RuntimeError("candidate quality is not finite")
    return CandidateMembership(
        recreated=recreated,
        row=row,
        group_nodes=np.asarray(group_nodes, dtype=np.uint32),
        support_nodes=support,
    )


def _operator_rows_for_pair(
    *,
    graph: Any,
    baseline: RecreatedMembership,
    candidate: CandidateMembership,
    vanilla: RecreatedMembership,
    candidate_row: pd.Series,
    vanilla_row: pd.Series,
    hypothesis_row: pd.Series,
    sketch_nodes: np.ndarray,
    resolution: float,
    randomness: float,
    transition_iterations: int,
    perturb_seed_offset: int,
) -> list[dict[str, Any]]:
    candidate_support = candidate.support_nodes
    vanilla_support = changed_support_nodes(baseline.membership, vanilla.membership)
    fixed = fixed_outside(baseline.membership.shape[0], candidate_support)
    candidate_index = int(candidate_row["candidate_index"])
    operator_seed = int(candidate_row.get("seed", 0)) + int(perturb_seed_offset) + candidate_index

    initial_memberships = {
        "candidate_recreated": candidate.recreated.membership,
        "vanilla_recreated": vanilla.membership,
        "control_extra_from_baseline": baseline.membership,
        "vanilla_core_free_polish": vanilla.membership,
        "vanilla_core_transplant_polish": transplant_support_groups(
            vanilla.membership,
            candidate.recreated.membership,
            candidate_support,
        ),
        "baseline_core_transplant_polish": transplant_support_groups(
            baseline.membership,
            candidate.recreated.membership,
            candidate_support,
        ),
    }
    fixed_by_operator = {
        "candidate_recreated": None,
        "vanilla_recreated": None,
        "control_extra_from_baseline": None,
        "vanilla_core_free_polish": fixed,
        "vanilla_core_transplant_polish": fixed,
        "baseline_core_transplant_polish": fixed,
    }

    rows: list[dict[str, Any]] = []
    for operator, initial in initial_memberships.items():
        if operator == "candidate_recreated":
            result = candidate.recreated
        elif operator == "vanilla_recreated":
            result = vanilla
        else:
            result = _run_leiden(
                graph,
                resolution=resolution,
                seed=operator_seed,
                n_iterations=transition_iterations,
                randomness=randomness,
                initial_membership=np.asarray(initial, dtype=np.uint64),
                fixed_nodes=fixed_by_operator[operator],
            )
        result_support = changed_support_nodes(baseline.membership, result.membership)
        dist_candidate, inter_candidate, union_candidate = support_distance(
            result_support,
            candidate_support,
        )
        dist_vanilla, inter_vanilla, union_vanilla = support_distance(
            result_support,
            vanilla_support,
        )
        rows.append(
            {
                "case": hypothesis_row.get("case"),
                "field": hypothesis_row.get("field"),
                "method": hypothesis_row.get("method"),
                "candidate_index": candidate_index,
                "vanilla_seed": int(vanilla_row["seed"]),
                "vanilla_randomness": float(vanilla_row["randomness"]),
                "vanilla_requested_n_iterations": vanilla_row["requested_n_iterations"],
                "operator": operator,
                "transition_iterations": int(transition_iterations),
                "fixed_outside_candidate_support": fixed_by_operator[operator] is not None,
                "baseline_quality": baseline.quality,
                "candidate_quality": candidate.recreated.quality,
                "vanilla_quality": vanilla.quality,
                "quality": result.quality,
                "delta_vs_baseline": result.quality - baseline.quality,
                "delta_vs_candidate": result.quality - candidate.recreated.quality,
                "delta_vs_vanilla": result.quality - vanilla.quality,
                "elapsed_sec": result.elapsed_sec,
                "candidate_expected_quality": _safe_float(candidate_row.get("p5_quality")),
                "candidate_reproduction_quality_error": (
                    candidate.recreated.quality - _safe_float(candidate_row.get("p5_quality"))
                ),
                "landscape_endpoint_distance": hypothesis_row.get("endpoint_distance"),
                "landscape_support_distance": hypothesis_row.get("support_distance"),
                "candidate_support_size": int(candidate_support.size),
                "vanilla_support_size": int(vanilla_support.size),
                "result_support_size": int(result_support.size),
                "result_support_distance_to_candidate": dist_candidate,
                "result_support_intersection_with_candidate": inter_candidate,
                "result_support_union_with_candidate": union_candidate,
                "result_support_distance_to_vanilla": dist_vanilla,
                "result_support_intersection_with_vanilla": inter_vanilla,
                "result_support_union_with_vanilla": union_vanilla,
                "result_endpoint_distance_to_candidate": endpoint_distance(
                    result.membership,
                    candidate.recreated.membership,
                    sketch_nodes,
                ),
                "result_endpoint_distance_to_vanilla": endpoint_distance(
                    result.membership,
                    vanilla.membership,
                    sketch_nodes,
                ),
                "sketch_node_hash": hash_u32_sequence(sketch_nodes),
                "sketch_membership": encode_membership_sketch(
                    result.membership,
                    sketch_nodes,
                ),
            }
        )
    return rows


def _select_hypotheses(
    hypotheses: pd.DataFrame,
    *,
    max_pairs: int,
    candidate_indices: set[int],
) -> pd.DataFrame:
    rows = hypotheses[hypotheses["hypothesis"].eq("candidate_local_core_inside_broader_vanilla")].copy()
    if candidate_indices:
        rows = rows[
            rows["candidate_node_id"]
            .map(_parse_candidate_index)
            .isin(candidate_indices)
        ]
    rows = rows.sort_values(
        ["endpoint_distance", "vanilla_minus_candidate_delta"],
        ascending=[True, False],
        na_position="last",
    )
    return rows.head(max_pairs)


def run_pilot(
    *,
    candidate_dirs: tuple[Path, ...],
    landscape_dir: Path,
    vanilla_dir: Path,
    output_dir: Path,
    max_pairs: int,
    candidate_indices: set[int],
    baseline_iterations: int,
    transition_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = pd.read_csv(landscape_dir / HYPOTHESES_FILENAME)
    selected = _select_hypotheses(
        hypotheses,
        max_pairs=max_pairs,
        candidate_indices=candidate_indices,
    )
    if selected.empty:
        raise ValueError("No transition hypotheses selected")
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, RecreatedMembership] = {}
    candidate_cache: dict[tuple[str, int], CandidateMembership] = {}
    vanilla_cache: dict[tuple[str, int, float, str], RecreatedMembership] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    out_rows: list[dict[str, Any]] = []

    for _, hypothesis in selected.iterrows():
        case = str(hypothesis["case"])
        candidate_index = _parse_candidate_index(hypothesis["candidate_node_id"])
        seed, vanilla_randomness, vanilla_n = _parse_vanilla_config(
            hypothesis["vanilla_node_id"]
        )
        candidate_row = _find_candidate_row(
            candidates,
            case=case,
            candidate_index=candidate_index,
        )
        vanilla_row = _find_vanilla_row(
            vanilla_rows,
            case=case,
            seed=seed,
            randomness=vanilla_randomness,
            n_iterations=vanilla_n,
        )
        graph_dir = Path(str(vanilla_row["graph_dir"]))
        graph_key = str(graph_dir)
        if graph_key not in graph_cache:
            graph_cache[graph_key] = _load_graph(graph_dir)
        graph, node_weights, arrays = graph_cache[graph_key]
        if case not in baseline_cache:
            baseline_cache[case] = _run_leiden(
                graph,
                resolution=resolution,
                seed=int(candidate_row.get("seed", 0)),
                n_iterations=baseline_iterations,
                randomness=randomness,
            )
        baseline = baseline_cache[case]
        ckey = (case, candidate_index)
        if ckey not in candidate_cache:
            candidate_cache[ckey] = _recreate_candidate(
                graph=graph,
                arrays=arrays,
                node_weights=node_weights,
                baseline_membership=baseline.membership,
                baseline_quality=baseline.quality,
                row=candidate_row,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
                polish_iterations=polish_iterations,
            )
        candidate = candidate_cache[ckey]
        vkey = (case, seed, vanilla_randomness, vanilla_n)
        if vkey not in vanilla_cache:
            vanilla_cache[vkey] = _run_leiden(
                graph,
                resolution=resolution,
                seed=seed,
                n_iterations=int(_safe_int(vanilla_n, baseline_iterations) or baseline_iterations),
                randomness=vanilla_randomness,
            )
        vanilla = vanilla_cache[vkey]
        sketch_nodes, sketch_context = compatible_sketch_nodes(
            arrays=arrays,
            baseline_membership=baseline.membership,
            node_weights=node_weights,
            candidate_rows=candidates[candidates["case"].astype(str) == case],
        )
        if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
            raise RuntimeError(f"sketch context mismatch for {case}")
        out_rows.extend(
            _operator_rows_for_pair(
                graph=graph,
                baseline=baseline,
                candidate=candidate,
                vanilla=vanilla,
                candidate_row=candidate_row,
                vanilla_row=vanilla_row,
                hypothesis_row=hypothesis,
                sketch_nodes=sketch_nodes,
                resolution=resolution,
                randomness=randomness,
                transition_iterations=transition_iterations,
                perturb_seed_offset=perturb_seed_offset,
            )
        )

    rows = pd.DataFrame(out_rows)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_transition_operator_pilot.v1",
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "selected_hypothesis_rows": int(len(selected)),
        "operator_rows": int(len(rows)),
        "baseline_iterations": int(baseline_iterations),
        "transition_iterations": int(transition_iterations),
        "polish_iterations": int(polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "output_dir": str(output_dir),
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, rows)
    return summary


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


def write_report(path: Path, rows: pd.DataFrame) -> None:
    lines = [
        "# Basin Transition Operator Pilot",
        "",
        "This is a diagnostic pilot. It tests restricted polish operators but does not establish a production policy.",
        "",
        "## Operator Summary",
        "",
    ]
    summary = (
        rows.groupby("operator", as_index=False)
        .agg(
            rows=("operator", "size"),
            delta_vs_baseline_median=("delta_vs_baseline", "median"),
            delta_vs_vanilla_median=("delta_vs_vanilla", "median"),
            delta_vs_candidate_median=("delta_vs_candidate", "median"),
            elapsed_sec_median=("elapsed_sec", "median"),
            support_distance_to_candidate_median=(
                "result_support_distance_to_candidate",
                "median",
            ),
            endpoint_distance_to_candidate_median=(
                "result_endpoint_distance_to_candidate",
                "median",
            ),
        )
        if not rows.empty
        else pd.DataFrame()
    )
    lines.extend(_markdown_table(summary))
    lines.extend(["", "## Pair Rows", ""])
    display_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "operator",
        "delta_vs_baseline",
        "delta_vs_vanilla",
        "delta_vs_candidate",
        "result_support_size",
        "result_support_distance_to_candidate",
        "result_endpoint_distance_to_candidate",
        "elapsed_sec",
    ]
    lines.extend(_markdown_table(rows[[c for c in display_cols if c in rows.columns]]))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- A restricted polish win is meaningful only if it beats the quality/cost of an additional vanilla run.",
            "- Endpoint-near results remain insufficient when support distance stays high.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_candidate_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument(
        "--candidate-indices",
        default="",
        help="Optional comma-separated candidate indices to include.",
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--transition-iterations", type=int, default=5)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_pilot(
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        candidate_indices=_parse_candidate_indices(args.candidate_indices),
        baseline_iterations=args.baseline_iterations,
        transition_iterations=args.transition_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
