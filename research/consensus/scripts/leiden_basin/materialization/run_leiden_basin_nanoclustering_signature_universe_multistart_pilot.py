#!/usr/bin/env python3
"""Run a NanoClustering signature-universe multistart pilot.

The prior common-mask probes used a single top1 endpoint target. The frozen
local-panel success unit, however, is endpoint-family signature distance. This
runner therefore tests terminal multiplicity under a fixed-outside universe
defined as:

    dominant-host source handles union all top1 endpoint handles in a signature.

It is a terminal-multiplicity diagnostic only. It does not promote wall/pathway
claims, inspect basin quality/cost, claim real-data method success, or claim
algorithm novelty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_anchor_release_pilot import _run_leiden_or_hold
from run_leiden_basin_nanoclustering_common_mask_multistart_pilot import (
    _pair_singleton_initial,
    _random_pair_initial,
    _two_block_initial,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_READINESS_DIR,
    ENDPOINT_TARGET_ROWS_CSV,
    GRAPH_INPUT_ROWS_CSV,
    _compact_membership,
    _json_safe,
    _load_graph,
    _load_label_array,
    _mask_for_row,
    _mask_hash,
    _parse_csv_list,
    _read_csv,
    _score_target,
    _source_rows,
    _union_masks,
    _write_csv,
)


DEFAULT_UNIVERSE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_universe_redesign_20260601"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_signature_universe_multistart_pilot_smoke_20260601"
)

UNIVERSE_SIGNATURE_ROWS_CSV = "nanoclustering_basin_universe_signature_rows.csv"
SIGNATURE_ATTEMPT_ROWS_CSV = "nanoclustering_signature_universe_multistart_attempt_rows.csv"
SIGNATURE_SUMMARY_ROWS_CSV = "nanoclustering_signature_universe_multistart_signature_rows.csv"
SIGNATURE_CONFIG_JSON = "nanoclustering_signature_universe_multistart_config.json"
SIGNATURE_SUMMARY_JSON = "nanoclustering_signature_universe_multistart_summary.json"
SIGNATURE_REPORT_MD = "nanoclustering_signature_universe_multistart_report.md"

SELECTION_POLICIES = {
    "default",
    "largest_universe_node_count",
    "largest_target_union_node_count",
    "largest_baseline_expansion",
}

CLAIM_BOUNDARY = (
    "NanoClustering signature-universe multistart terminal-multiplicity pilot "
    "only; runs varied initializations under a fixed-outside endpoint-family "
    "signature universe. It does not promote wall/pathway claims, inspect basin "
    "quality/cost, claim real-data method success, or claim algorithm novelty."
)
RUN_STATUS = "executed_signature_universe_multistart_pilot"


def _parse_handles(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _select_signature_rows(
    universes: pd.DataFrame,
    *,
    case_ranks: tuple[int, ...],
    role_sides: tuple[str, ...],
    analysis_tiers: tuple[str, ...],
    strict_core_only: bool,
    selection_policy: str,
    max_signatures: int,
) -> pd.DataFrame:
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError(
            f"unsupported selection policy: {selection_policy}; "
            f"expected one of {sorted(SELECTION_POLICIES)}"
        )
    rows = universes.copy()
    if case_ranks:
        rows = rows[rows["panel_case_rank"].astype(int).isin(case_ranks)]
    if role_sides:
        rows = rows[rows["role_side"].astype(str).isin(role_sides)]
    if analysis_tiers:
        rows = rows[rows["analysis_tier"].astype(str).isin(analysis_tiers)]
    if strict_core_only:
        rows = rows[rows["strict_core_v0"].astype(bool)]
    if selection_policy == "largest_universe_node_count":
        sort_cols = ["universe_node_count", "target_union_node_count", "panel_case_rank"]
        ascending = [False, False, True]
    elif selection_policy == "largest_target_union_node_count":
        sort_cols = ["target_union_node_count", "universe_node_count", "panel_case_rank"]
        ascending = [False, False, True]
    elif selection_policy == "largest_baseline_expansion":
        sort_cols = [
            "node_expansion_vs_baseline_pair_median",
            "universe_node_count",
            "panel_case_rank",
        ]
        ascending = [False, False, True]
    else:
        sort_cols = ["panel_case_rank", "role_side", "endpoint_signature_id"]
        ascending = [True, True, True]
    rows = rows.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    if max_signatures > 0:
        rows = rows.head(max_signatures)
    return rows.reset_index(drop=True)


def _mask_from_handles(
    *,
    endpoint_targets: pd.DataFrame,
    handles: list[str],
    label_cache: dict[tuple[str, str], np.ndarray],
    n_nodes: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    mask = np.zeros(n_nodes, dtype=np.bool_)
    present: list[str] = []
    missing: list[str] = []
    for handle in handles:
        rows = endpoint_targets[
            endpoint_targets["endpoint_handle_id"].astype(str).eq(str(handle))
            & endpoint_targets["membership_path_exists"].astype(bool)
            & endpoint_targets["cluster_label_present"].astype(bool)
        ]
        if rows.empty:
            missing.append(str(handle))
            continue
        mask |= _mask_for_row(rows.iloc[0], label_cache)
        present.append(str(handle))
    return mask, present, missing


def _target_seeded_initial(
    *,
    initial_labels: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_labels, dtype=np.uint64).copy()
    initial[target_mask] = np.uint64(int(initial.max()) + 1)
    return initial


def _signature_start_specs(
    *,
    initial_labels: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    universe_mask: np.ndarray,
    method_seed: int,
    random_start_count: int,
    random_block_count: int,
) -> list[dict[str, Any]]:
    specs = [
        {
            "start_policy": "source_state",
            "start_index": 0,
            "leiden_seed": int(method_seed),
            "membership": np.asarray(initial_labels, dtype=np.uint64).copy(),
        },
        {
            "start_policy": "target_union_seeded",
            "start_index": 1,
            "leiden_seed": int(method_seed),
            "membership": _target_seeded_initial(
                initial_labels=initial_labels,
                target_mask=target_mask,
            ),
        },
        {
            "start_policy": "universe_singleton",
            "start_index": 2,
            "leiden_seed": int(method_seed),
            "membership": _pair_singleton_initial(
                initial_labels=initial_labels,
                pair_mask=universe_mask,
            ),
        },
        {
            "start_policy": "source_target_two_block",
            "start_index": 3,
            "leiden_seed": int(method_seed),
            "membership": _two_block_initial(
                initial_labels=initial_labels,
                source_mask=source_mask,
                target_mask=target_mask,
            ),
        },
    ]
    for idx in range(int(random_start_count)):
        start_seed = int(method_seed) * 1000 + idx + 1
        specs.append(
            {
                "start_policy": f"random_universe_blocks_{idx:03d}",
                "start_index": len(specs),
                "leiden_seed": start_seed,
                "membership": _random_pair_initial(
                    initial_labels=initial_labels,
                    pair_mask=universe_mask,
                    seed=start_seed,
                    block_count=int(random_block_count),
                ),
            }
        )
    return specs


def _signature_summary_rows(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if attempts.empty:
        return pd.DataFrame(rows)
    for universe_id, group in attempts.groupby("signature_universe_id", sort=False):
        terminal_hashes = sorted(set(group["terminal_universe_hash"].astype(str)))
        rows.append(
            {
                "signature_universe_id": universe_id,
                "panel_case_id": group["panel_case_id"].iloc[0],
                "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
                "analysis_tier": group["analysis_tier"].iloc[0],
                "strict_core_v0": bool(group["strict_core_v0"].iloc[0]),
                "role_id": group["role_id"].iloc[0],
                "role_side": group["role_side"].iloc[0],
                "primitive_id": group["primitive_id"].iloc[0],
                "branch": group["branch"].iloc[0],
                "endpoint_signature_id": group["endpoint_signature_id"].iloc[0],
                "method_seed": int(group["method_seed"].iloc[0]),
                "source_mask_hash": group["source_mask_hash"].iloc[0],
                "target_union_mask_hash": group["target_union_mask_hash"].iloc[0],
                "universe_mask_hash": group["universe_mask_hash"].iloc[0],
                "source_node_count": int(group["source_node_count"].iloc[0]),
                "target_union_node_count": int(group["target_union_node_count"].iloc[0]),
                "universe_node_count": int(group["universe_node_count"].iloc[0]),
                "fixed_outside_node_count": int(group["fixed_outside_node_count"].iloc[0]),
                "target_handle_count": int(group["target_handle_count"].iloc[0]),
                "start_attempt_count": int(len(group)),
                "executed_attempt_count": int(
                    group["execution_status"].astype(str).str.startswith("executed_").sum()
                ),
                "unique_terminal_universe_hash_count": len(terminal_hashes),
                "terminal_multiplicity_detected": len(terminal_hashes) > 1,
                "terminal_universe_hashes": ";".join(terminal_hashes),
                "target_best_cluster_doc_share_min": float(
                    group["target_best_cluster_doc_share"].min()
                ),
                "target_best_cluster_doc_share_median": float(
                    group["target_best_cluster_doc_share"].median()
                ),
                "target_best_cluster_doc_share_max": float(
                    group["target_best_cluster_doc_share"].max()
                ),
                "source_best_cluster_doc_share_min": float(
                    group["source_best_cluster_doc_share"].min()
                ),
                "source_best_cluster_doc_share_median": float(
                    group["source_best_cluster_doc_share"].median()
                ),
                "source_best_cluster_doc_share_max": float(
                    group["source_best_cluster_doc_share"].max()
                ),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "seconds_sum": float(group["seconds"].sum()),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    universe_dir = Path(args.universe_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint_targets = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    universe_rows = _read_csv(universe_dir / UNIVERSE_SIGNATURE_ROWS_CSV)
    selected = _select_signature_rows(
        universe_rows,
        case_ranks=_parse_csv_list(args.case_ranks, int),
        role_sides=_parse_csv_list(args.role_sides, str),
        analysis_tiers=_parse_csv_list(args.analysis_tiers, str),
        strict_core_only=bool(args.strict_core_only),
        selection_policy=str(args.selection_policy),
        max_signatures=int(args.max_signatures),
    )

    graph_by_branch = {
        str(row["branch"]): row
        for _, row in graph_rows.iterrows()
        if str(row.get("runtime_graph_status", "")).startswith("ready_")
    }
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, float]] = {}
    attempt_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in selected.itertuples(index=False):
        branch = str(row.branch)
        if branch not in graph_cache:
            graph, weights, load_seconds = _load_graph(
                graph_by_branch[branch],
                manifest_cache,
            )
            graph_cache[branch] = graph, weights, load_seconds
        graph, weights, graph_load_seconds = graph_cache[branch]
        n_nodes = int(graph.n_nodes)

        source = _source_rows(endpoint_targets, str(row.endpoint_signature_id))
        if source.empty:
            raise ValueError(f"missing source handles: {row.endpoint_signature_id}")
        source_full_path = Path(str(source.iloc[0]["membership_path"]))
        label_col = str(source.iloc[0]["label_cols"]).split(";")[0] or "candidate_micro_id"
        initial_labels = _compact_membership(_load_label_array(source_full_path, label_col))
        source_mask = _union_masks(source, label_cache, n_nodes)
        target_handles = _parse_handles(row.target_handles)
        target_mask, present_targets, missing_targets = _mask_from_handles(
            endpoint_targets=endpoint_targets,
            handles=target_handles,
            label_cache=label_cache,
            n_nodes=n_nodes,
        )
        if missing_targets:
            raise ValueError(
                f"missing target handles for {row.endpoint_signature_id}: {missing_targets}"
            )
        universe_mask = np.logical_or(source_mask, target_mask)
        fixed_nodes = ~universe_mask
        method_seed = int(args.method_seed)
        specs = _signature_start_specs(
            initial_labels=initial_labels,
            source_mask=source_mask,
            target_mask=target_mask,
            universe_mask=universe_mask,
            method_seed=method_seed,
            random_start_count=int(args.random_start_count),
            random_block_count=int(args.random_block_count),
        )
        for spec in specs:
            terminal, meta = _run_leiden_or_hold(
                graph=graph,
                membership=spec["membership"],
                fixed_nodes=fixed_nodes,
                resolution=float(args.gamma),
                seed=int(spec["leiden_seed"]),
                n_iterations=int(args.n_iterations),
                blocked_status="blocked_empty_signature_universe_free_mask",
                executed_status="executed_signature_universe_multistart",
            )
            score = _score_target(
                terminal=terminal,
                initial=spec["membership"],
                pair_mask=universe_mask,
                target_mask=target_mask,
                source_mask=source_mask,
                weights=weights,
            )
            attempt_rows.append(
                {
                    "signature_universe_id": row.universe_id,
                    "start_id": f"{row.universe_id}__{spec['start_policy']}",
                    "panel_case_id": row.panel_case_id,
                    "panel_case_rank": int(row.panel_case_rank),
                    "analysis_tier": row.analysis_tier,
                    "strict_core_v0": bool(row.strict_core_v0),
                    "role_id": row.role_id,
                    "role_side": row.role_side,
                    "primitive_id": row.primitive_id,
                    "branch": branch,
                    "endpoint_signature_id": row.endpoint_signature_id,
                    "method_seed": method_seed,
                    "start_policy": spec["start_policy"],
                    "start_index": int(spec["start_index"]),
                    "leiden_seed": int(spec["leiden_seed"]),
                    "random_block_count": int(args.random_block_count),
                    "n_nodes": n_nodes,
                    "n_edges": int(graph.n_edges),
                    "source_mask_hash": _mask_hash(source_mask),
                    "target_union_mask_hash": _mask_hash(target_mask),
                    "universe_mask_hash": _mask_hash(universe_mask),
                    "source_node_count": int(source_mask.sum()),
                    "target_union_node_count": int(target_mask.sum()),
                    "universe_node_count": int(universe_mask.sum()),
                    "fixed_outside_node_count": int(fixed_nodes.sum()),
                    "target_handle_count": len(target_handles),
                    "target_handles": ";".join(target_handles),
                    "graph_load_seconds_cached_branch": float(graph_load_seconds),
                    **meta,
                    "terminal_universe_hash": score["pair_terminal_hash"],
                    **score,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
        attempts = pd.DataFrame(attempt_rows)
        _write_csv(attempts, output_dir / SIGNATURE_ATTEMPT_ROWS_CSV)
        _write_csv(
            _signature_summary_rows(attempts),
            output_dir / SIGNATURE_SUMMARY_ROWS_CSV,
        )

    attempts = pd.DataFrame(attempt_rows)
    signatures = _signature_summary_rows(attempts)
    _write_csv(attempts, output_dir / SIGNATURE_ATTEMPT_ROWS_CSV)
    _write_csv(signatures, output_dir / SIGNATURE_SUMMARY_ROWS_CSV)
    summary = _build_summary(
        selected=selected,
        attempts=attempts,
        signatures=signatures,
        readiness_dir=readiness_dir,
        universe_dir=universe_dir,
        output_dir=output_dir,
        elapsed_seconds=time.perf_counter() - started,
    )
    (output_dir / SIGNATURE_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_signature_universe_multistart_pilot.v1",
        "readiness_dir": str(readiness_dir),
        "universe_dir": str(universe_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "role_sides": list(_parse_csv_list(args.role_sides, str)),
        "analysis_tiers": list(_parse_csv_list(args.analysis_tiers, str)),
        "strict_core_only": bool(args.strict_core_only),
        "selection_policy": str(args.selection_policy),
        "max_signatures": int(args.max_signatures),
        "method_seed": int(args.method_seed),
        "random_start_count": int(args.random_start_count),
        "random_block_count": int(args.random_block_count),
        "gamma": float(args.gamma),
        "n_iterations": int(args.n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SIGNATURE_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, signatures=signatures)
    return summary


def _build_summary(
    *,
    selected: pd.DataFrame,
    attempts: pd.DataFrame,
    signatures: pd.DataFrame,
    readiness_dir: Path,
    universe_dir: Path,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if signatures.empty:
        multiplicity_count = 0
        max_terminal_count = 0
        total_route_seconds = 0.0
    else:
        multiplicity_count = int(signatures["terminal_multiplicity_detected"].sum())
        max_terminal_count = int(signatures["unique_terminal_universe_hash_count"].max())
        total_route_seconds = float(signatures["seconds_sum"].sum())
    return {
        "schema": "nanoclustering_signature_universe_multistart_pilot_summary.v1",
        "status": RUN_STATUS if not signatures.empty else "no_signature_universes",
        "readiness_dir": str(readiness_dir),
        "universe_dir": str(universe_dir),
        "output_dir": str(output_dir),
        "selected_signature_count": int(len(selected)),
        "signature_universe_count": int(len(signatures)),
        "start_attempt_count": int(len(attempts)),
        "terminal_multiplicity_signature_count": multiplicity_count,
        "terminal_multiplicity_signature_share": (
            float(multiplicity_count / len(signatures)) if len(signatures) else None
        ),
        "max_unique_terminal_universe_hash_count": max_terminal_count,
        "unique_panel_case_count": (
            int(signatures["panel_case_id"].nunique()) if not signatures.empty else 0
        ),
        "branch_count": int(signatures["branch"].nunique()) if not signatures.empty else 0,
        "role_side_count": int(signatures["role_side"].nunique()) if not signatures.empty else 0,
        "universe_node_count_median": (
            float(signatures["universe_node_count"].median()) if not signatures.empty else None
        ),
        "universe_node_count_max": (
            int(signatures["universe_node_count"].max()) if not signatures.empty else 0
        ),
        "target_union_node_count_median": (
            float(signatures["target_union_node_count"].median())
            if not signatures.empty
            else None
        ),
        "target_best_cluster_doc_share_median": (
            float(signatures["target_best_cluster_doc_share_median"].median())
            if not signatures.empty
            else None
        ),
        "total_route_seconds": total_route_seconds,
        "elapsed_seconds": float(elapsed_seconds),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    signatures: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Signature-Universe Multistart Pilot",
        "",
        f"- status: `{summary['status']}`",
        f"- signature_universe_count: {summary['signature_universe_count']}",
        f"- start_attempt_count: {summary['start_attempt_count']}",
        f"- terminal_multiplicity_signature_count: {summary['terminal_multiplicity_signature_count']}",
        f"- terminal_multiplicity_signature_share: {summary['terminal_multiplicity_signature_share']}",
        f"- max_unique_terminal_universe_hash_count: {summary['max_unique_terminal_universe_hash_count']}",
        f"- universe_node_count_median: {summary['universe_node_count_median']}",
        f"- universe_node_count_max: {summary['universe_node_count_max']}",
        f"- total_route_seconds: {summary['total_route_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Signatures",
    ]
    if signatures.empty:
        lines.append("- no signature universes")
    else:
        for row in signatures.sort_values(
            ["terminal_multiplicity_detected", "universe_node_count"],
            ascending=[False, False],
        ).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['signature_universe_id']}: "
                f"unique_terminals={data['unique_terminal_universe_hash_count']}, "
                f"multiplicity={data['terminal_multiplicity_detected']}, "
                f"universe_nodes={data['universe_node_count']}, "
                f"target_handles={data['target_handle_count']}, "
                f"target_share_median={data['target_best_cluster_doc_share_median']}, "
                f"quality_range=[{data['quality_min']}, {data['quality_max']}]"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "Distinct terminals under the same endpoint-family signature "
                "universe are terminal-multiplicity evidence only. They are not "
                "yet wall, pathway, quality, cost, method-success, or algorithm "
                "evidence."
            ),
            "",
        ]
    )
    (output_dir / SIGNATURE_REPORT_MD).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ranks", default="")
    parser.add_argument("--role-sides", default="candidate")
    parser.add_argument("--analysis-tiers", default="strict_core_v0_primary")
    parser.add_argument("--strict-core-only", action="store_true")
    parser.add_argument(
        "--selection-policy",
        choices=sorted(SELECTION_POLICIES),
        default="largest_universe_node_count",
    )
    parser.add_argument("--max-signatures", type=int, default=6)
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--random-start-count", type=int, default=4)
    parser.add_argument("--random-block-count", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
