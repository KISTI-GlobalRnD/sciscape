#!/usr/bin/env python3
"""Compare one branch target-growth candidate against vanilla seed controls."""

from __future__ import annotations

import argparse
import json
import math
import time
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

from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _parse_n_iterations_values,
    _read_candidate_rows,
    compatible_sketch_nodes,
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
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    endpoint_distance,
    membership_metric_row,
    support_distance,
    support_progress_from_vanilla,
    v_only_support_nodes,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    TARGET_SELECTION_FIXED_TAIL_BACKFILL,
    score_branch_path_rows,
    state_distance,
)
from search_leiden_basin_branch_target_growth import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_BRANCH_DIR,
    PATH_ROWS_FILENAME,
    STATE_ROWS_FILENAME,
)

DEFAULT_OUTPUT_DIR = DEFAULT_BRANCH_DIR.parent / (
    "basin_transition_branch_candidate_controls_field34_cc_c0_v0"
)
CONTROL_ROWS_FILENAME = "branch_candidate_control_rows.csv"
SUMMARY_ROWS_FILENAME = "branch_candidate_control_summary.csv"
SUMMARY_FILENAME = "branch_candidate_control_summary.json"
CONFIG_FILENAME = "branch_candidate_control_config.json"
REPORT_FILENAME = "branch_candidate_control_report.md"

def _parse_csv_tuple(value: str, *, cast: type = str) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())

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

def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

def select_branch_candidate_row(
    path_rows: pd.DataFrame,
    *,
    pair_id: str,
    selection_policy: str,
    support_gate: float,
) -> pd.Series:
    if path_rows.empty:
        raise ValueError("branch path rows are empty")
    rows = score_branch_path_rows(path_rows)
    if pair_id:
        rows = rows[rows["pair_id"].astype(str) == str(pair_id)].copy()
    if selection_policy:
        rows = rows[
            rows["path_selection_policy"].astype(str) == str(selection_policy)
        ].copy()
    if rows.empty:
        raise ValueError(
            f"No branch path rows match pair_id={pair_id!r}, "
            f"selection_policy={selection_policy!r}"
        )
    gated = rows[
        (rows["path_final_support_distance_to_vanilla"].astype(float) >= float(support_gate))
        & (rows["path_final_delta_q_vs_start"].astype(float) >= 0.0)
    ].copy()
    if gated.empty:
        gated = rows.copy()
    return gated.sort_values(
        [
            "path_branch_discovery_score",
            "path_final_support_distance_to_vanilla",
            "path_final_target_progress_from_vanilla",
            "path_final_delta_q_vs_start",
            "path_q_wall",
            "path_final_mutable_node_count",
        ],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]

def _metric_payload(
    *,
    membership: np.ndarray,
    quality: float,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    sketch_nodes: np.ndarray,
    vanilla_support_distance_to_candidate: float,
    vanilla_target_distance: float,
    start_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
) -> dict[str, Any]:
    metrics = membership_metric_row(
        membership=membership,
        quality=float(quality),
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
        vanilla_membership=vanilla_membership,
        sketch_nodes=sketch_nodes,
        start_quality=start_quality,
        candidate_quality=candidate_quality,
        vanilla_quality=vanilla_quality,
        prefix="result",
    )
    candidate_progress = support_progress_from_vanilla(
        support_distance_to_candidate=float(
            metrics["result_support_distance_to_candidate"]
        ),
        vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
    )
    target_distance = state_distance(
        support_distance_value=float(metrics["result_support_distance_to_candidate"]),
        endpoint_distance_value=float(metrics["result_endpoint_distance_to_candidate"]),
    )
    source_distance = state_distance(
        support_distance_value=float(metrics["result_support_distance_to_vanilla"]),
        endpoint_distance_value=float(metrics["result_endpoint_distance_to_vanilla"]),
    )
    return {
        "quality": float(metrics["result_quality"]),
        "delta_q_vs_vanilla": float(metrics["result_delta_q_vs_vanilla"]),
        "delta_q_vs_candidate": float(metrics["result_delta_q_vs_candidate"]),
        "support_size": int(metrics["result_support_size"]),
        "support_distance_to_vanilla": float(
            metrics["result_support_distance_to_vanilla"]
        ),
        "support_distance_to_candidate": float(
            metrics["result_support_distance_to_candidate"]
        ),
        "support_intersection_with_candidate": int(
            metrics["result_support_intersection_with_candidate"]
        ),
        "support_union_with_candidate": int(
            metrics["result_support_union_with_candidate"]
        ),
        "endpoint_distance_to_vanilla": float(
            metrics["result_endpoint_distance_to_vanilla"]
        ),
        "endpoint_distance_to_candidate": float(
            metrics["result_endpoint_distance_to_candidate"]
        ),
        "candidate_progress_from_vanilla": float(candidate_progress),
        "target_distance": float(target_distance),
        "source_distance": float(source_distance),
        "target_progress_from_vanilla": float(vanilla_target_distance - target_distance),
    }

def _state_branch_row(
    *,
    branch_path: pd.Series,
    branch_state: pd.Series,
    candidate_quality: float,
    vanilla_quality: float,
) -> dict[str, Any]:
    quality = _finite_float(branch_state.get("state_quality"))
    return {
        "row_type": "branch_candidate",
        "variant": "branch_target_growth",
        "run_id": str(branch_path.get("path_final_state_id", "")),
        "case": branch_path.get("case", ""),
        "pair_id": branch_path.get("pair_id", ""),
        "seed": "",
        "randomness": "",
        "requested_n_iterations": "",
        "iteration_mode": "",
        "elapsed_sec": _finite_float(branch_state.get("path_elapsed_sec"), 0.0),
        "quality": quality,
        "delta_q_vs_vanilla": quality - float(vanilla_quality),
        "delta_q_vs_candidate": quality - float(candidate_quality),
        "support_size": int(_finite_float(branch_state.get("state_support_size"), 0.0)),
        "support_distance_to_vanilla": _finite_float(
            branch_state.get("state_support_distance_to_vanilla")
        ),
        "support_distance_to_candidate": _finite_float(
            branch_state.get("state_support_distance_to_candidate")
        ),
        "endpoint_distance_to_vanilla": _finite_float(
            branch_state.get("state_endpoint_distance_to_vanilla")
        ),
        "endpoint_distance_to_candidate": _finite_float(
            branch_state.get("state_endpoint_distance_to_candidate")
        ),
        "candidate_progress_from_vanilla": _finite_float(
            branch_state.get("state_candidate_progress_from_vanilla")
        ),
        "target_distance": _finite_float(branch_state.get("state_target_distance")),
        "source_distance": _finite_float(branch_state.get("state_source_distance")),
        "target_progress_from_vanilla": _finite_float(
            branch_state.get("state_target_progress_from_vanilla")
        ),
        "q_wall": _finite_float(branch_path.get("path_q_wall"), 0.0),
        "mutable_node_count": int(
            _finite_float(branch_path.get("path_final_mutable_node_count"), 0.0)
        ),
        "branch_discovery_score": _finite_float(
            branch_path.get("path_branch_discovery_score")
        ),
        "path_selection_policy": branch_path.get("path_selection_policy", ""),
        "path_prefix_rank": int(_finite_float(branch_path.get("path_prefix_rank"), 0.0)),
        "path_applied_actions": branch_path.get("path_applied_actions", ""),
    }

def _reference_or_control_row(
    *,
    row_type: str,
    variant: str,
    run_id: str,
    membership: np.ndarray,
    quality: float,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    sketch_nodes: np.ndarray,
    vanilla_support_distance_to_candidate: float,
    vanilla_target_distance: float,
    candidate_quality: float,
    vanilla_quality: float,
    elapsed_sec: float = 0.0,
    seed: int | str = "",
    randomness: float | str = "",
    requested_n_iterations: str = "",
    iteration_mode: str = "",
) -> dict[str, Any]:
    payload = _metric_payload(
        membership=membership,
        quality=quality,
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
        vanilla_membership=vanilla_membership,
        sketch_nodes=sketch_nodes,
        vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
        vanilla_target_distance=vanilla_target_distance,
        start_quality=vanilla_quality,
        candidate_quality=candidate_quality,
        vanilla_quality=vanilla_quality,
    )
    return {
        "row_type": row_type,
        "variant": variant,
        "run_id": run_id,
        "seed": seed,
        "randomness": randomness,
        "requested_n_iterations": requested_n_iterations,
        "iteration_mode": iteration_mode,
        "elapsed_sec": float(elapsed_sec),
        **payload,
        "q_wall": 0.0,
        "mutable_node_count": int(np.asarray(membership).size),
        "branch_discovery_score": math.nan,
        "path_selection_policy": "",
        "path_prefix_rank": "",
        "path_applied_actions": "",
    }

def control_summary_rows(
    rows: pd.DataFrame,
    *,
    material_delta_q: float = 1.0,
    support_margin: float = 0.01,
    progress_margin: float = 0.005,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    branch = rows[rows["row_type"] == "branch_candidate"].copy()
    controls = rows[rows["row_type"] == "control"].copy()
    if branch.empty or controls.empty:
        return pd.DataFrame()
    branch_row = branch.iloc[0]
    best_quality = controls.sort_values(
        ["quality", "support_distance_to_vanilla", "target_progress_from_vanilla"],
        ascending=[False, False, False],
    ).iloc[0]
    best_support = controls.sort_values(
        ["support_distance_to_vanilla", "target_progress_from_vanilla", "quality"],
        ascending=[False, False, False],
    ).iloc[0]
    best_progress = controls.sort_values(
        ["target_progress_from_vanilla", "support_distance_to_vanilla", "quality"],
        ascending=[False, False, False],
    ).iloc[0]
    directed_controls = controls[
        controls["target_progress_from_vanilla"].astype(float) > 0.0
    ].copy()
    best_directed = None
    if not directed_controls.empty:
        best_directed = directed_controls.sort_values(
            [
                "target_progress_from_vanilla",
                "support_distance_to_vanilla",
                "quality",
            ],
            ascending=[False, False, False],
        ).iloc[0]
    dominance_mask = (
        (controls["quality"].astype(float) >= float(branch_row["quality"]))
        & (
            controls["support_distance_to_vanilla"].astype(float)
            >= float(branch_row["support_distance_to_vanilla"])
        )
        & (
            controls["target_progress_from_vanilla"].astype(float)
            >= float(branch_row["target_progress_from_vanilla"])
        )
    )
    branch_quality_margin = float(branch_row["quality"]) - float(best_quality["quality"])
    branch_support_margin = float(branch_row["support_distance_to_vanilla"]) - float(
        best_support["support_distance_to_vanilla"]
    )
    branch_progress_margin = float(branch_row["target_progress_from_vanilla"]) - float(
        best_progress["target_progress_from_vanilla"]
    )
    branch_directed = float(branch_row["target_progress_from_vanilla"]) > 0.0
    if dominance_mask.any():
        verdict = "seed_control_dominates_branch"
    elif branch_quality_margin >= float(material_delta_q) and branch_support_margin >= 0.0:
        verdict = "branch_material_quality_win"
    elif (
        branch_directed
        and directed_controls.empty
        and branch_progress_margin >= float(progress_margin)
        and branch_quality_margin >= -2.0 * float(material_delta_q)
    ):
        verdict = "branch_unique_candidate_directed_quality_lag"
    elif (
        branch_support_margin >= float(support_margin)
        and branch_progress_margin >= float(progress_margin)
        and branch_quality_margin >= -float(material_delta_q)
    ):
        verdict = "branch_support_progress_tradeoff"
    else:
        verdict = "branch_not_control_clear"
    return pd.DataFrame(
        [
            {
                "branch_run_id": branch_row["run_id"],
                "control_rows": int(len(controls)),
                "best_quality_control_run_id": best_quality["run_id"],
                "best_quality_control_quality": float(best_quality["quality"]),
                "best_quality_control_support": float(
                    best_quality["support_distance_to_vanilla"]
                ),
                "best_quality_control_target_progress": float(
                    best_quality["target_progress_from_vanilla"]
                ),
                "best_support_control_run_id": best_support["run_id"],
                "best_support_control_quality": float(best_support["quality"]),
                "best_support_control_support": float(
                    best_support["support_distance_to_vanilla"]
                ),
                "best_support_control_target_progress": float(
                    best_support["target_progress_from_vanilla"]
                ),
                "best_progress_control_run_id": best_progress["run_id"],
                "best_progress_control_quality": float(best_progress["quality"]),
                "best_progress_control_support": float(
                    best_progress["support_distance_to_vanilla"]
                ),
                "best_progress_control_target_progress": float(
                    best_progress["target_progress_from_vanilla"]
                ),
                "branch_quality": float(branch_row["quality"]),
                "branch_support": float(branch_row["support_distance_to_vanilla"]),
                "branch_target_progress": float(
                    branch_row["target_progress_from_vanilla"]
                ),
                "branch_q_wall": float(branch_row["q_wall"]),
                "branch_mutable_node_count": int(branch_row["mutable_node_count"]),
                "branch_quality_minus_best_control": branch_quality_margin,
                "branch_support_minus_best_control": branch_support_margin,
                "branch_target_progress_minus_best_control": branch_progress_margin,
                "candidate_directed_control_rows": int(len(directed_controls)),
                "best_candidate_directed_control_run_id": (
                    "" if best_directed is None else best_directed["run_id"]
                ),
                "best_candidate_directed_control_quality": (
                    math.nan if best_directed is None else float(best_directed["quality"])
                ),
                "best_candidate_directed_control_support": (
                    math.nan
                    if best_directed is None
                    else float(best_directed["support_distance_to_vanilla"])
                ),
                "best_candidate_directed_control_target_progress": (
                    math.nan
                    if best_directed is None
                    else float(best_directed["target_progress_from_vanilla"])
                ),
                "control_dominates_branch_rows": int(dominance_mask.sum()),
                "material_delta_q": float(material_delta_q),
                "support_margin": float(support_margin),
                "progress_margin": float(progress_margin),
                "verdict": verdict,
            }
        ]
    )

def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    lines = [
        "# Branch Candidate Seed-Control Comparison",
        "",
        "This artifact compares one branch target-growth candidate against same-case vanilla Leiden seed/iteration controls on the same support/progress/QF axes.",
        "",
        "## Config",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key in [
        "branch_dir",
        "pair_id",
        "selection_policy",
        "seeds",
        "randomness_values",
        "n_iterations_values",
        "support_gate",
        "material_delta_q",
        "support_margin",
        "progress_margin",
    ]:
        lines.append(f"| {key} | {config.get(key, '')} |")
    lines.extend(["", "## Summary", ""])
    lines.extend(_markdown_table(summary_rows, max_rows=10))
    lines.extend(["", "## Top Rows", ""])
    display_cols = [
        "row_type",
        "variant",
        "run_id",
        "seed",
        "randomness",
        "requested_n_iterations",
        "quality",
        "delta_q_vs_vanilla",
        "support_distance_to_vanilla",
        "target_progress_from_vanilla",
        "support_distance_to_candidate",
        "q_wall",
        "mutable_node_count",
        "elapsed_sec",
    ]
    top = rows.sort_values(
        [
            "row_type",
            "quality",
            "support_distance_to_vanilla",
            "target_progress_from_vanilla",
        ],
        ascending=[True, False, False, False],
    )
    lines.extend(_markdown_table(top[[c for c in display_cols if c in top.columns]], max_rows=80))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- A branch row is not a Dongdaemun-refinement claim unless it beats seed controls on material, cost-adjusted value.",
            "- Support/progress movement is reported separately from QF; a support-only win is a mechanism signal, not a quality win.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_evaluation(
    *,
    branch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_id: str,
    selection_policy: str,
    seeds: tuple[int, ...],
    randomness_values: tuple[float, ...],
    n_iterations_values: tuple[Any, ...],
    baseline_iterations: int,
    candidate_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    support_gate: float,
    material_delta_q: float,
    support_margin: float,
    progress_margin: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_rows = pd.read_csv(branch_dir / PATH_ROWS_FILENAME)
    state_rows = pd.read_csv(branch_dir / STATE_ROWS_FILENAME)
    branch_path = select_branch_candidate_row(
        path_rows,
        pair_id=pair_id,
        selection_policy=selection_policy,
        support_gate=support_gate,
    )
    branch_state_match = state_rows[
        state_rows["state_id"].astype(str) == str(branch_path["path_final_state_id"])
    ]
    if branch_state_match.empty:
        raise ValueError(f"Missing state row for branch path {branch_path['path_final_state_id']}")
    branch_state = branch_state_match.iloc[0]
    case = str(branch_path["case"])
    candidate_index = int(branch_path["candidate_index"])
    vanilla_seed = int(branch_path["vanilla_seed"])
    vanilla_randomness = float(branch_state["vanilla_randomness"])
    vanilla_n = str(branch_state["vanilla_requested_n_iterations"])
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
    graph, node_weights, arrays = _load_graph(Path(str(vanilla_row["graph_dir"])))
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
    sketch_nodes, sketch_context = compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=baseline.membership,
        node_weights=node_weights,
        candidate_rows=candidate_rows[candidate_rows["case"].astype(str) == case],
    )
    if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
        raise RuntimeError(f"sketch context mismatch for {case}")
    candidate_support, vanilla_support, _target_nodes = v_only_support_nodes(
        baseline.membership,
        candidate.recreated.membership,
        vanilla.membership,
    )
    vanilla_support_distance_to_candidate = support_distance(
        vanilla_support,
        candidate_support,
    )[0]
    vanilla_target_distance = state_distance(
        support_distance_value=float(vanilla_support_distance_to_candidate),
        endpoint_distance_value=endpoint_distance(
            vanilla.membership,
            candidate.recreated.membership,
            sketch_nodes,
        ),
    )
    rows: list[dict[str, Any]] = [
        _state_branch_row(
            branch_path=branch_path,
            branch_state=branch_state,
            candidate_quality=float(candidate.recreated.quality),
            vanilla_quality=float(vanilla.quality),
        ),
        _reference_or_control_row(
            row_type="reference",
            variant="reference_candidate",
            run_id=f"{case}|candidate={candidate_index}",
            membership=candidate.recreated.membership,
            quality=float(candidate.recreated.quality),
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=sketch_nodes,
            vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
            vanilla_target_distance=vanilla_target_distance,
            candidate_quality=float(candidate.recreated.quality),
            vanilla_quality=float(vanilla.quality),
        ),
        _reference_or_control_row(
            row_type="reference",
            variant="reference_vanilla",
            run_id=f"{case}|reference_vanilla",
            membership=vanilla.membership,
            quality=float(vanilla.quality),
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=sketch_nodes,
            vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
            vanilla_target_distance=vanilla_target_distance,
            candidate_quality=float(candidate.recreated.quality),
            vanilla_quality=float(vanilla.quality),
            seed=vanilla_seed,
            randomness=vanilla_randomness,
            requested_n_iterations=vanilla_n,
            iteration_mode="reference",
        ),
    ]
    for seed in seeds:
        for control_randomness in randomness_values:
            for budget in n_iterations_values:
                start = time.perf_counter()
                result = graph.run_leiden(
                    resolution=float(resolution),
                    seed=int(seed),
                    n_iterations=int(budget.n_iterations),
                    randomness=float(control_randomness),
                )
                elapsed = time.perf_counter() - start
                rows.append(
                    _reference_or_control_row(
                        row_type="control",
                        variant="standard_leiden",
                        run_id=(
                            f"{case}|control|seed={int(seed)}|"
                            f"randomness={float(control_randomness):g}|"
                            f"n={budget.requested}"
                        ),
                        membership=np.asarray(result.membership, dtype=np.uint64),
                        quality=float(result.quality),
                        baseline_membership=baseline.membership,
                        candidate_membership=candidate.recreated.membership,
                        vanilla_membership=vanilla.membership,
                        sketch_nodes=sketch_nodes,
                        vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
                        vanilla_target_distance=vanilla_target_distance,
                        candidate_quality=float(candidate.recreated.quality),
                        vanilla_quality=float(vanilla.quality),
                        elapsed_sec=float(elapsed),
                        seed=int(seed),
                        randomness=float(control_randomness),
                        requested_n_iterations=str(budget.requested),
                        iteration_mode=str(budget.mode),
                    )
                )
    control_rows = pd.DataFrame(rows)
    summary_rows = control_summary_rows(
        control_rows,
        material_delta_q=material_delta_q,
        support_margin=support_margin,
        progress_margin=progress_margin,
    )
    control_rows.to_csv(output_dir / CONTROL_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    config = {
        "branch_dir": str(branch_dir),
        "output_dir": str(output_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_id": str(pair_id),
        "selection_policy": str(selection_policy),
        "selected_branch_state_id": str(branch_path["path_final_state_id"]),
        "seeds": [int(seed) for seed in seeds],
        "randomness_values": [float(value) for value in randomness_values],
        "n_iterations_values": [str(value.requested) for value in n_iterations_values],
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "support_gate": float(support_gate),
        "material_delta_q": float(material_delta_q),
        "support_margin": float(support_margin),
        "progress_margin": float(progress_margin),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_branch_candidate_controls.v0",
        "output_dir": str(output_dir),
        "control_rows": int(len(control_rows)),
        "summary_rows": int(len(summary_rows)),
        "verdict": "" if summary_rows.empty else str(summary_rows.iloc[0]["verdict"]),
        "selected_branch_state_id": str(branch_path["path_final_state_id"]),
        "paths": {
            "rows": str(output_dir / CONTROL_ROWS_FILENAME),
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=control_rows,
        summary_rows=summary_rows,
        config=config,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=DEFAULT_BRANCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-id", default="c0-s11-r0.001")
    parser.add_argument("--selection-policy", default=TARGET_SELECTION_FIXED_TAIL_BACKFILL)
    parser.add_argument("--seeds", default="11,42,73,101,137")
    parser.add_argument("--randomness-values", default="0.01")
    parser.add_argument("--n-iterations-values", default="1,10,convergence")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--support-margin", type=float, default=0.01)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_evaluation(
        branch_dir=args.branch_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_id=args.pair_id,
        selection_policy=args.selection_policy,
        seeds=_parse_csv_tuple(args.seeds, cast=int),
        randomness_values=_parse_csv_tuple(args.randomness_values, cast=float),
        n_iterations_values=_parse_n_iterations_values(args.n_iterations_values),
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        support_gate=args.support_gate,
        material_delta_q=args.material_delta_q,
        support_margin=args.support_margin,
        progress_margin=args.progress_margin,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
