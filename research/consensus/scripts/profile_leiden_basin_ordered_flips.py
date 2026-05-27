#!/usr/bin/env python3
"""Profile ordered block flips between selected Leiden basin endpoints.

v0 is intentionally narrow: field34/cc, candidate 2 vs vanilla seed 11 r=0,
direction V -> C, label-intersection blocks, raw flips only.
"""

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

from analyze_leiden_basin_transition_minimal_pathway import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_MINIMAL_PATHWAY_DIR,
    PAIR_ROWS_FILENAME as MINIMAL_PAIR_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    SCORING_POLICIES,
    UNIT_TYPE_LABEL_INTERSECTION,
    build_label_intersection_units,
    run_ordered_flip_beam,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_LANDSCAPE_DIR,
    DEFAULT_VANILLA_DIR,
    HYPOTHESES_FILENAME,
    VANILLA_ROWS_FILENAME,
    _find_candidate_row,
    _find_vanilla_row,
    _parse_candidate_index,
    _parse_vanilla_config,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "pathway_ordered_flip_frontier_field34_cc_v0"
)
UNIT_ROWS_FILENAME = "ordered_flip_unit_rows.csv"
FRONTIER_ROWS_FILENAME = "ordered_flip_frontier_rows.csv"
BEAM_ROWS_FILENAME = "ordered_flip_beam_rows.csv"
SUMMARY_FILENAME = "ordered_flip_summary.json"
REPORT_FILENAME = "ordered_flip_report.md"


def _parse_csv_tuple(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _target_hypothesis(
    hypotheses: pd.DataFrame,
    *,
    candidate_index: int,
    vanilla_seed: int,
    vanilla_randomness: float,
    vanilla_n: str,
) -> pd.Series:
    rows: list[pd.Series] = []
    for _, row in hypotheses.iterrows():
        try:
            row_candidate_index = _parse_candidate_index(row["candidate_node_id"])
            seed, randomness, n_iterations = _parse_vanilla_config(row["vanilla_node_id"])
        except ValueError:
            continue
        if (
            int(row_candidate_index) == int(candidate_index)
            and int(seed) == int(vanilla_seed)
            and math.isclose(float(randomness), float(vanilla_randomness))
            and str(n_iterations) == str(vanilla_n)
        ):
            rows.append(row)
    if not rows:
        raise ValueError(
            "Missing target hypothesis for "
            f"candidate={candidate_index} seed={vanilla_seed} "
            f"r={vanilla_randomness:g} n={vanilla_n}"
        )
    return rows[0]


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 24) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _best_beam_rows(beam_rows: pd.DataFrame) -> pd.DataFrame:
    if beam_rows.empty:
        return beam_rows
    return (
        beam_rows.sort_values(
            [
                "scoring_policy",
                "step_index",
                "raw_barrier_so_far",
                "result_support_distance_to_candidate",
                "delta_q_vs_start",
            ],
            ascending=[True, False, True, True, False],
        )
        .groupby("scoring_policy", as_index=False)
        .head(1)
        .sort_values("scoring_policy")
    )


def write_report(
    path: Path,
    *,
    unit_rows: pd.DataFrame,
    frontier_rows: pd.DataFrame,
    beam_rows: pd.DataFrame,
    minimal_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Ordered Flip Basin Profile v0",
        "",
        "This is a basin-profiling artifact, not a transition operator.",
        "",
        "Scope: one field34/cc target pair, direction `V -> C`, unit type `label_intersection_block`, raw flips only.",
        "",
        "Raw negative QF is an upper-bound barrier: a later local polish may recover part of it, but this profile asks whether the first raw blocks already align QF and candidate progress.",
        "",
        "## Target",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "pair_id",
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "vanilla_requested_n_iterations",
        "candidate_quality",
        "vanilla_quality",
        "vanilla_minus_candidate_quality",
        "candidate_support_size",
        "vanilla_support_size",
        "v_only_support_size",
        "unit_count",
        "beam_width",
        "max_steps",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Existing Node-Level Minimal Pathway", ""])
    if not minimal_rows.empty:
        minimal_cols = [
            "action",
            "ordering_policy",
            "node_edit_lower_bound",
            "quality_barrier",
            "quality_peak_gain",
            "final_delta_vs_start",
            "final_support_distance_to_candidate",
            "final_support_distance_to_vanilla",
        ]
        lines.extend(
            _markdown_table(
                minimal_rows[[c for c in minimal_cols if c in minimal_rows.columns]],
                max_rows=8,
            )
        )

    lines.extend(["", "## Block-Beam End States", ""])
    if not beam_rows.empty:
        beam_cols = [
            "scoring_policy",
            "step_index",
            "selected_unit_count",
            "flipped_node_count",
            "delta_q_vs_start",
            "raw_barrier_so_far",
            "result_support_distance_to_candidate",
            "result_support_distance_to_vanilla",
            "chosen_unit_id",
            "chosen_delta_q_immediate",
            "chosen_incremental_progress_fraction",
        ]
        lines.extend(
            _markdown_table(
                _best_beam_rows(beam_rows)[
                    [c for c in beam_cols if c in beam_rows.columns]
                ],
                max_rows=12,
            )
        )

    lines.extend(["", "## First-Step Policy Choices", ""])
    if not frontier_rows.empty:
        first = frontier_rows[frontier_rows["step_index"].eq(1)].copy()
        first_rows = []
        for policy in SCORING_POLICIES:
            score_col = f"{policy}_score"
            policy_first = first[first["scoring_policy"].eq(policy)]
            chosen = policy_first.sort_values(
                [score_col, "delta_q_immediate", "incremental_progress_fraction"],
                ascending=[False, False, False],
            ).head(1)
            if chosen.empty:
                continue
            row = chosen.iloc[0]
            first_rows.append(
                {
                    "policy": policy,
                    "unit_id": row["unit_id"],
                    "unit_node_count": int(row["unit_node_count"]),
                    "delta_q_immediate": float(row["delta_q_immediate"]),
                    "incremental_progress_fraction": float(
                        row["incremental_progress_fraction"]
                    ),
                    "raw_barrier_if_chosen": float(row["raw_barrier_if_chosen"]),
                    "candidate_label_closure_extra_count": int(
                        row["candidate_label_closure_extra_count"]
                    ),
                }
            )
        lines.extend(_markdown_table(pd.DataFrame(first_rows), max_rows=12))

    lines.extend(["", "## Unit Summary", ""])
    if not unit_rows.empty:
        unit_summary = pd.DataFrame(
            [
                {
                    "units": int(len(unit_rows)),
                    "nodes": int(unit_rows["unit_node_count"].sum()),
                    "unit_nodes_min": int(unit_rows["unit_node_count"].min()),
                    "unit_nodes_median": float(unit_rows["unit_node_count"].median()),
                    "unit_nodes_max": int(unit_rows["unit_node_count"].max()),
                    "closure_extra_median": float(
                        unit_rows["candidate_label_closure_extra_count"].median()
                    ),
                    "boundary_edge_weight_median": float(
                        unit_rows["boundary_edge_weight"].median()
                    ),
                }
            ]
        )
        lines.extend(_markdown_table(unit_summary))

    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- This artifact contrasts block-level beam ordering against the existing node/group minimal pathway; it does not supersede it.",
            "- Frontier rows use cheap progress proxies; exact support distances are recorded on retained beam states.",
            "- If QF-positive first flips and progress-first flips disagree, a production operator needs an explicit choice between quality and basin progress.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(
    *,
    candidate_dirs: tuple[Path, ...],
    landscape_dir: Path,
    vanilla_dir: Path,
    minimal_pathway_dir: Path,
    output_dir: Path,
    candidate_index: int,
    vanilla_seed: int,
    vanilla_randomness: float,
    vanilla_n: str,
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    beam_width: int,
    max_steps: int,
    scoring_policies: tuple[str, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = pd.read_csv(landscape_dir / HYPOTHESES_FILENAME)
    target = _target_hypothesis(
        hypotheses,
        candidate_index=candidate_index,
        vanilla_seed=vanilla_seed,
        vanilla_randomness=vanilla_randomness,
        vanilla_n=vanilla_n,
    )
    case = str(target["case"])
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
        polish_iterations=polish_iterations,
    )
    vanilla = _run_leiden(
        graph,
        resolution=resolution,
        seed=vanilla_seed,
        n_iterations=int(_safe_int(vanilla_n, baseline_iterations) or baseline_iterations),
        randomness=vanilla_randomness,
    )
    sketch_nodes, sketch_context = compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=baseline.membership,
        node_weights=node_weights,
        candidate_rows=candidate_rows[candidate_rows["case"].astype(str) == case],
    )
    if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
        raise RuntimeError(f"sketch context mismatch for {case}")
    pair_id = f"c{candidate_index}-s{vanilla_seed}-r{vanilla_randomness:g}"
    context = {
        "case": case,
        "field": target.get("field"),
        "method": target.get("method"),
        "candidate_index": int(candidate_index),
        "vanilla_seed": int(vanilla_seed),
        "vanilla_randomness": float(vanilla_randomness),
        "vanilla_requested_n_iterations": str(vanilla_n),
        "pair_id": pair_id,
    }
    unit_rows, unit_summary = build_label_intersection_units(
        baseline_membership=baseline.membership,
        candidate_membership=candidate.recreated.membership,
        vanilla_membership=vanilla.membership,
        src=arrays.src,
        dst=arrays.dst,
        weight=arrays.weight,
        node_weights=node_weights,
        context=context,
    )
    frontier_rows, beam_rows = run_ordered_flip_beam(
        graph=graph,
        units=unit_rows,
        baseline_membership=baseline.membership,
        candidate_membership=candidate.recreated.membership,
        vanilla_membership=vanilla.membership,
        start_quality=vanilla.quality,
        candidate_quality=candidate.recreated.quality,
        vanilla_quality=vanilla.quality,
        sketch_nodes=sketch_nodes,
        resolution=resolution,
        beam_width=beam_width,
        max_steps=max_steps,
        scoring_policies=scoring_policies,
        context=context,
    )
    unit_rows.to_csv(output_dir / UNIT_ROWS_FILENAME, index=False)
    frontier_rows.to_csv(output_dir / FRONTIER_ROWS_FILENAME, index=False)
    beam_rows.to_csv(output_dir / BEAM_ROWS_FILENAME, index=False)

    minimal_rows = pd.DataFrame()
    minimal_path = minimal_pathway_dir / MINIMAL_PAIR_ROWS_FILENAME
    if minimal_path.exists():
        minimal = pd.read_csv(minimal_path)
        minimal["vanilla_requested_n_iterations"] = minimal[
            "vanilla_requested_n_iterations"
        ].astype(str)
        minimal_rows = minimal[
            (minimal["case"].astype(str) == case)
            & (pd.to_numeric(minimal["candidate_index"], errors="coerce") == candidate_index)
            & (pd.to_numeric(minimal["vanilla_seed"], errors="coerce") == vanilla_seed)
            & (
                np.isclose(
                    pd.to_numeric(minimal["vanilla_randomness"], errors="coerce"),
                    vanilla_randomness,
                )
            )
            & minimal["vanilla_requested_n_iterations"].eq(str(vanilla_n))
        ].copy()

    summary = {
        "schema": "leiden_basin_ordered_flip_profile.v0",
        "output_dir": str(output_dir),
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "minimal_pathway_dir": str(minimal_pathway_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "pair_id": pair_id,
        "case": case,
        "candidate_index": int(candidate_index),
        "vanilla_seed": int(vanilla_seed),
        "vanilla_randomness": float(vanilla_randomness),
        "vanilla_requested_n_iterations": str(vanilla_n),
        "direction": "vanilla_to_candidate_support",
        "unit_type": UNIT_TYPE_LABEL_INTERSECTION,
        "candidate_quality": float(candidate.recreated.quality),
        "vanilla_quality": float(vanilla.quality),
        "vanilla_minus_candidate_quality": float(
            vanilla.quality - candidate.recreated.quality
        ),
        "unit_rows": int(len(unit_rows)),
        "frontier_rows": int(len(frontier_rows)),
        "beam_rows": int(len(beam_rows)),
        "beam_width": int(beam_width),
        "max_steps": int(max_steps),
        "scoring_policies": list(scoring_policies),
        **unit_summary,
    }
    if not beam_rows.empty:
        best = _best_beam_rows(beam_rows)
        summary["best_beam_min_raw_barrier"] = float(best["raw_barrier_so_far"].min())
        summary["best_beam_min_support_distance_to_candidate"] = float(
            best["result_support_distance_to_candidate"].min()
        )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        unit_rows=unit_rows,
        frontier_rows=frontier_rows,
        beam_rows=beam_rows,
        minimal_rows=minimal_rows,
        summary=summary,
    )
    return summary


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
    parser.add_argument(
        "--minimal-pathway-dir",
        type=Path,
        default=DEFAULT_MINIMAL_PATHWAY_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-index", type=int, default=2)
    parser.add_argument("--vanilla-seed", type=int, default=11)
    parser.add_argument("--vanilla-randomness", type=float, default=0.0)
    parser.add_argument("--vanilla-n", default="10")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--scoring-policies",
        default=",".join(SCORING_POLICIES),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        minimal_pathway_dir=args.minimal_pathway_dir,
        output_dir=args.output_dir,
        candidate_index=args.candidate_index,
        vanilla_seed=args.vanilla_seed,
        vanilla_randomness=args.vanilla_randomness,
        vanilla_n=args.vanilla_n,
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        scoring_policies=_parse_csv_tuple(args.scoring_policies, SCORING_POLICIES),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
