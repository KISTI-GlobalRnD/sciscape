#!/usr/bin/env python3
"""Measure closure context needed for support-level basin transitions.

This diagnostic follows the minimum-pathway framing: it does not ask whether
Leiden naturally performs a mutation. It asks how much label context surrounds
the direct support edit set `S_V - S_C`, so a later transition operator can
separate a true small support edit from a larger closure problem.
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

from analyze_leiden_basin_transition_boundaries import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_BOUNDARY_DIR,
    NODE_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
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
    _select_hypotheses,
    changed_support_nodes,
    endpoint_distance,
    support_distance,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_closure_context_field34_cc"
)
PAIR_ROWS_FILENAME = "basin_transition_closure_context_pairs.csv"
LABEL_ROWS_FILENAME = "basin_transition_closure_context_labels.csv"
SUMMARY_FILENAME = "basin_transition_closure_context_summary.json"
REPORT_FILENAME = "basin_transition_closure_context_report.md"

CLOSURE_MODES = ("baseline_label", "candidate_label", "vanilla_label_source")


def _as_set(nodes: np.ndarray | list[int] | set[int]) -> set[int]:
    return {int(node) for node in np.asarray(list(nodes), dtype=np.int64)}


def _label_closure_nodes(membership: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.asarray([], dtype=np.uint32)
    mask = np.isin(np.asarray(membership, dtype=np.uint64), labels)
    return np.flatnonzero(mask).astype(np.uint32, copy=False)


def _context_ratio(extra_count: int, direct_count: int) -> float:
    if direct_count <= 0:
        return math.nan
    return float(extra_count) / float(direct_count)


def support_set_summary(
    *,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
) -> dict[str, Any]:
    candidate_support = changed_support_nodes(baseline_membership, candidate_membership)
    vanilla_support = changed_support_nodes(baseline_membership, vanilla_membership)
    candidate_set = _as_set(candidate_support)
    vanilla_set = _as_set(vanilla_support)
    direct = vanilla_set - candidate_set
    missing = candidate_set - vanilla_set
    shared = candidate_set & vanilla_set
    dist, intersection, union = support_distance(candidate_support, vanilla_support)
    return {
        "candidate_support": candidate_support,
        "vanilla_support": vanilla_support,
        "direct_nodes": np.asarray(sorted(direct), dtype=np.uint32),
        "missing_nodes": np.asarray(sorted(missing), dtype=np.uint32),
        "shared_nodes": np.asarray(sorted(shared), dtype=np.uint32),
        "candidate_support_size": int(len(candidate_set)),
        "vanilla_support_size": int(len(vanilla_set)),
        "support_intersection_size": int(intersection),
        "support_union_size": int(union),
        "support_distance": float(dist),
        "direct_support_edit_lower_bound": int(len(direct)),
        "missing_candidate_support_count": int(len(missing)),
        "support_symmetric_edit_lower_bound": int(len(direct) + len(missing)),
        "candidate_containment_ratio": (
            float(len(shared)) / float(len(candidate_set)) if candidate_set else math.nan
        ),
        "vanilla_extra_fraction": (
            float(len(direct)) / float(len(vanilla_set)) if vanilla_set else math.nan
        ),
    }


def closure_rows_for_mode(
    *,
    mode: str,
    membership: np.ndarray,
    direct_nodes: np.ndarray,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mode not in CLOSURE_MODES:
        raise ValueError(f"Unsupported closure mode: {mode}")
    direct_set = _as_set(direct_nodes)
    candidate_set = _as_set(candidate_support)
    vanilla_set = _as_set(vanilla_support)
    support_union = candidate_set | vanilla_set
    if not direct_set:
        return pd.DataFrame(), {
            "closure_mode": mode,
            "closure_label_count": 0,
            "closure_node_count": 0,
            "closure_context_extra_count": 0,
            "closure_outside_support_count": 0,
            "closure_context_ratio": math.nan,
        }

    labels = np.unique(np.asarray(membership, dtype=np.uint64)[direct_nodes])
    rows: list[dict[str, Any]] = []
    closure_union: set[int] = set()
    outside_support_union: set[int] = set()
    for label in labels:
        label_nodes = _label_closure_nodes(
            membership,
            np.asarray([int(label)], dtype=np.uint64),
        )
        label_set = _as_set(label_nodes)
        direct_for_label = label_set & direct_set
        candidate_for_label = label_set & candidate_set
        vanilla_for_label = label_set & vanilla_set
        extra_for_label = label_set - direct_set
        outside_support_for_label = label_set - support_union
        closure_union |= label_set
        outside_support_union |= outside_support_for_label
        rows.append(
            {
                **context,
                "closure_mode": mode,
                "closure_label": int(label),
                "direct_node_count": int(len(direct_for_label)),
                "closure_node_count": int(len(label_set)),
                "closure_context_extra_count": int(len(extra_for_label)),
                "closure_outside_support_count": int(len(outside_support_for_label)),
                "closure_candidate_support_count": int(len(candidate_for_label)),
                "closure_vanilla_support_count": int(len(vanilla_for_label)),
                "closure_context_ratio": _context_ratio(
                    len(extra_for_label),
                    len(direct_for_label),
                ),
            }
        )
    summary = {
        "closure_mode": mode,
        "closure_label_count": int(len(labels)),
        "closure_node_count": int(len(closure_union)),
        "closure_context_extra_count": int(len(closure_union - direct_set)),
        "closure_outside_support_count": int(len(outside_support_union)),
        "closure_context_ratio": _context_ratio(
            len(closure_union - direct_set),
            len(direct_set),
        ),
    }
    return pd.DataFrame(rows), summary


def boundary_role_counts(
    *,
    node_rows: pd.DataFrame,
    pair_key: dict[str, Any],
) -> dict[str, int]:
    if node_rows.empty:
        return {}
    mask = np.ones(len(node_rows), dtype=np.bool_)
    for column, value in pair_key.items():
        if column == "vanilla_requested_n_iterations":
            mask &= node_rows[column].astype(str).eq(str(value)).to_numpy()
        else:
            mask &= node_rows[column].eq(value).to_numpy()
    frame = node_rows[mask & node_rows["support_class"].eq("vanilla_extra").to_numpy()]
    if frame.empty:
        return {}
    counts = frame["boundary_role"].value_counts().to_dict()
    return {f"direct_{str(role)}_nodes": int(count) for role, count in counts.items()}


def _hypothesis_rows(
    landscape_dir: Path,
    *,
    max_pairs: int,
    candidate_indices: set[int],
) -> pd.DataFrame:
    hypotheses = pd.read_csv(landscape_dir / HYPOTHESES_FILENAME)
    return _select_hypotheses(
        hypotheses,
        max_pairs=max_pairs,
        candidate_indices=candidate_indices,
    )


def _parse_candidate_indices(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def run_analysis(
    *,
    candidate_dirs: tuple[Path, ...],
    boundary_dir: Path,
    landscape_dir: Path,
    vanilla_dir: Path,
    output_dir: Path,
    max_pairs: int,
    candidate_indices: set[int],
    baseline_iterations: int,
    polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    node_rows = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    hypotheses = _hypothesis_rows(
        landscape_dir,
        max_pairs=max_pairs,
        candidate_indices=candidate_indices,
    )
    if hypotheses.empty:
        raise ValueError("No transition hypotheses selected")
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, Any] = {}
    candidate_cache: dict[tuple[str, int], Any] = {}
    vanilla_cache: dict[tuple[str, int, float, str], Any] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    label_frames: list[pd.DataFrame] = []

    for _, hypothesis in hypotheses.iterrows():
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
                n_iterations=int(
                    _safe_int(vanilla_n, baseline_iterations)
                    or baseline_iterations
                ),
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

        context = {
            "case": case,
            "field": hypothesis.get("field"),
            "method": hypothesis.get("method"),
            "candidate_index": int(candidate_index),
            "vanilla_seed": int(seed),
            "vanilla_randomness": float(vanilla_randomness),
            "vanilla_requested_n_iterations": vanilla_n,
        }
        pair_key = dict(context)
        pair_key.pop("field", None)
        pair_key.pop("method", None)
        support = support_set_summary(
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
        )
        mode_summaries: dict[str, Any] = {}
        modes = {
            "baseline_label": baseline.membership,
            "candidate_label": candidate.recreated.membership,
            "vanilla_label_source": vanilla.membership,
        }
        for mode, membership in modes.items():
            labels, mode_summary = closure_rows_for_mode(
                mode=mode,
                membership=membership,
                direct_nodes=support["direct_nodes"],
                candidate_support=support["candidate_support"],
                vanilla_support=support["vanilla_support"],
                context=context,
            )
            if not labels.empty:
                label_frames.append(labels)
            for key, value in mode_summary.items():
                if key == "closure_mode":
                    continue
                mode_summaries[f"{mode}_{key}"] = value
        pair_rows.append(
            {
                **context,
                "baseline_quality": float(baseline.quality),
                "candidate_quality": float(candidate.recreated.quality),
                "vanilla_quality": float(vanilla.quality),
                "candidate_delta_vs_baseline": float(
                    candidate.recreated.quality - baseline.quality
                ),
                "vanilla_delta_vs_baseline": float(
                    vanilla.quality - baseline.quality
                ),
                "vanilla_minus_candidate_delta": float(
                    vanilla.quality - candidate.recreated.quality
                ),
                "endpoint_distance": endpoint_distance(
                    candidate.recreated.membership,
                    vanilla.membership,
                    sketch_nodes,
                ),
                **{
                    key: value
                    for key, value in support.items()
                    if not isinstance(value, np.ndarray)
                },
                **boundary_role_counts(node_rows=node_rows, pair_key=pair_key),
                **mode_summaries,
            }
        )

    pair_frame = pd.DataFrame(pair_rows)
    label_frame = (
        pd.concat(label_frames, ignore_index=True)
        if label_frames
        else pd.DataFrame()
    )
    pair_frame.to_csv(output_dir / PAIR_ROWS_FILENAME, index=False)
    label_frame.to_csv(output_dir / LABEL_ROWS_FILENAME, index=False)
    summary = {
        "schema": "leiden_basin_transition_closure_context.v1",
        "boundary_dir": str(boundary_dir),
        "landscape_dir": str(landscape_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "pair_rows": int(len(pair_frame)),
        "label_rows": int(len(label_frame)),
        "baseline_iterations": int(baseline_iterations),
        "polish_iterations": int(polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "output_dir": str(output_dir),
    }
    if not pair_frame.empty:
        summary["median_direct_support_edit_lower_bound"] = float(
            pair_frame["direct_support_edit_lower_bound"].median()
        )
        summary["median_support_symmetric_edit_lower_bound"] = float(
            pair_frame["support_symmetric_edit_lower_bound"].median()
        )
        summary["median_candidate_label_context_ratio"] = float(
            pair_frame["candidate_label_closure_context_ratio"].median()
        )
        summary["median_baseline_label_context_ratio"] = float(
            pair_frame["baseline_label_closure_context_ratio"].median()
        )
        summary["median_vanilla_label_source_context_ratio"] = float(
            pair_frame["vanilla_label_source_closure_context_ratio"].median()
        )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, pair_frame, label_frame)
    return summary


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


def write_report(path: Path, pair_rows: pd.DataFrame, label_rows: pd.DataFrame) -> None:
    lines = [
        "# Basin Transition Closure Context",
        "",
        "This diagnostic computes label-context closure around the direct support edit set `S_V - S_C`.",
        "",
        "## Pair Summary",
        "",
    ]
    pair_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "direct_support_edit_lower_bound",
        "missing_candidate_support_count",
        "support_symmetric_edit_lower_bound",
        "candidate_containment_ratio",
        "support_distance",
        "endpoint_distance",
        "candidate_label_closure_node_count",
        "candidate_label_closure_context_extra_count",
        "candidate_label_closure_context_ratio",
        "baseline_label_closure_node_count",
        "baseline_label_closure_context_extra_count",
        "baseline_label_closure_context_ratio",
        "vanilla_label_source_closure_node_count",
        "vanilla_label_source_closure_context_extra_count",
        "vanilla_label_source_closure_context_ratio",
        "direct_collateral_like_nodes",
        "direct_ambiguous_nodes",
        "direct_bridge_like_nodes",
    ]
    display = (
        pair_rows.sort_values(
            [
                "support_symmetric_edit_lower_bound",
                "candidate_label_closure_context_ratio",
            ],
            ascending=[True, True],
        )
        if not pair_rows.empty
        else pair_rows
    )
    lines.extend(_markdown_table(display[[c for c in pair_cols if c in display.columns]]))
    lines.extend(["", "## Largest Candidate-Label Closure Groups", ""])
    if not label_rows.empty:
        candidate_labels = label_rows[
            label_rows["closure_mode"].eq("candidate_label")
        ].sort_values(
            [
                "closure_context_extra_count",
                "direct_node_count",
                "closure_node_count",
            ],
            ascending=[False, False, False],
        )
        label_cols = [
            "candidate_index",
            "vanilla_seed",
            "vanilla_randomness",
            "closure_mode",
            "closure_label",
            "direct_node_count",
            "closure_node_count",
            "closure_context_extra_count",
            "closure_outside_support_count",
            "closure_context_ratio",
        ]
        lines.extend(
            _markdown_table(candidate_labels[[c for c in label_cols if c in candidate_labels.columns]])
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- `direct_support_edit_lower_bound` is a support-set lower bound, not an executable Leiden operator.",
            "- Label closure rows estimate how much surrounding partition context a support edit touches under baseline, candidate, or vanilla labels.",
            "- Large closure ratios mean a nominally small support edit is likely a larger split/merge problem.",
            "- This remains a Dongdaemun diagnostic artifact, not a production policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        action="append",
        type=Path,
        dest="candidate_dirs",
        help="Candidate artifact dir. Can be repeated.",
    )
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--candidate-indices", default="0,2")
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    candidate_dirs = tuple(args.candidate_dirs or DEFAULT_CANDIDATE_DIRS)
    summary = run_analysis(
        candidate_dirs=candidate_dirs,
        boundary_dir=args.boundary_dir,
        landscape_dir=args.landscape_dir,
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        max_pairs=args.max_pairs,
        candidate_indices=_parse_candidate_indices(args.candidate_indices),
        baseline_iterations=args.baseline_iterations,
        polish_iterations=args.polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(summary)


if __name__ == "__main__":
    main()
