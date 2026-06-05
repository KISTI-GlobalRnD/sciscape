#!/usr/bin/env python3
"""Replay variable-pair synthetic endpoints and design the next route gate.

This diagnostic consumes the frozen 6-family synthetic CPM runner output. It
uses endpoint signatures as initial memberships for ordinary Leiden+CPM polish,
then materializes only route-candidate pairs whose source and target endpoints
are replay-stable and differ in L/R co-assignment.

It is not route execution, wall promotion, method comparison, quality/cost
evaluation, full NanoClustering replay, or an algorithm claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    DEFAULT_DESIGN_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_SYNTHETIC_DIR,
    DESIGN_FAMILY_ROWS_CSV,
    SEED_RUNS_CSV,
    _build_graph,
    _canonical_groups,
    _edges_for_variant,
    _json_safe,
    _mechanism_read,
    _signature_id,
    _synthetic_cases,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_variable_pair_synthetic_endpoint_replay_v1_20260603"
)

ENDPOINT_MANIFEST_CSV = "variable_pair_synthetic_endpoint_replay_manifest.csv"
REPLAY_RUNS_CSV = "variable_pair_synthetic_endpoint_replay_runs.csv"
REPLAY_SUMMARY_CSV = "variable_pair_synthetic_endpoint_replay_summary.csv"
ROUTE_CANDIDATES_CSV = "variable_pair_synthetic_route_gate_candidates.csv"
SUMMARY_JSON = "variable_pair_synthetic_endpoint_replay_summary.json"
CONFIG_JSON = "variable_pair_synthetic_endpoint_replay_config.json"
REPORT_MD = "variable_pair_synthetic_endpoint_replay_report.md"

CLAIM_BOUNDARY = (
    "Variable-pair synthetic endpoint-replay diagnostic only; frozen endpoint "
    "signatures are replayed as initial memberships under ordinary Leiden+CPM. "
    "No route/pathway execution, no wall promotion, no full NanoClustering "
    "replay, no quality/cost evaluation, no method comparison, and no "
    "algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_endpoint_replay_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "diagnostic_replay_and_route_design_not_method_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _initial_membership_from_groups(
    *,
    nodes: tuple[str, ...],
    endpoint_signature: str,
) -> list[int]:
    groups = json.loads(str(endpoint_signature))
    labels: dict[str, int] = {}
    for label, group in enumerate(groups):
        for node in group:
            labels[str(node)] = int(label)
    unknown = sorted(set(labels) - set(nodes))
    if unknown:
        raise ValueError(f"endpoint signature contains unknown nodes: {unknown}")
    next_label = len(groups)
    for node in nodes:
        if node not in labels:
            labels[node] = next_label
            next_label += 1
    return [labels[node] for node in nodes]


def _endpoint_manifest(seed_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "design_family",
        "synthetic_demo_role",
        "expected_signature",
        "graph_variant",
        "endpoint_signature_id",
        "endpoint_signature",
        "pair_coassigned",
        "mechanism_read",
    ]
    for keys, group in seed_runs.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **key,
                "endpoint_run_count": int(len(group)),
                "endpoint_run_share_within_variant": float(
                    len(group)
                    / len(
                        seed_runs[
                            (seed_runs["design_family"] == key["design_family"])
                            & (seed_runs["graph_variant"] == key["graph_variant"])
                        ]
                    )
                ),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "quality_min": float(group["quality"].min()),
                "quality_median": float(group["quality"].median()),
                "quality_max": float(group["quality"].max()),
                "cluster_count_min": int(group["cluster_count"].min()),
                "cluster_count_median": float(group["cluster_count"].median()),
                "cluster_count_max": int(group["cluster_count"].max()),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(
        [
            "design_family",
            "graph_variant",
            "pair_coassigned",
            "endpoint_run_count",
            "endpoint_signature_id",
        ],
        ascending=[True, True, False, False, True],
    )
    manifest["endpoint_rank_within_variant"] = (
        manifest.groupby(["design_family", "graph_variant"]).cumcount() + 1
    )
    manifest["endpoint_replay_id"] = [
        f"vp_endpoint_{index:04d}" for index in range(len(manifest))
    ]
    return _claim_columns(manifest)


def _endpoint_lookup(manifest: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in manifest.to_dict("records"):
        lookup[
            (
                str(row["design_family"]),
                str(row["graph_variant"]),
                str(row["endpoint_signature_id"]),
            )
        ] = row
    return lookup


def _run_replay(
    *,
    manifest: pd.DataFrame,
    design_dir: Path,
    replay_seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    families = pd.read_csv(design_dir / DESIGN_FAMILY_ROWS_CSV)
    cases = {case.design_family: case for case in _synthetic_cases(families)}
    lookup = _endpoint_lookup(manifest)
    rows: list[dict[str, Any]] = []

    for endpoint in manifest.itertuples(index=False):
        case = cases[str(endpoint.design_family)]
        graph_variant = str(endpoint.graph_variant)
        graph = _build_graph(case.nodes, _edges_for_variant(case, graph_variant))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        initial = _initial_membership_from_groups(
            nodes=case.nodes,
            endpoint_signature=str(endpoint.endpoint_signature),
        )
        for seed in range(int(replay_seeds)):
            result = runner.run(
                case.gamma,
                seed=int(seed),
                initial_membership=initial,
                node_sizes=case.node_sizes,
            )
            membership = list(map(int, result.membership))
            groups = _canonical_groups(case.nodes, membership)
            result_signature_id = _signature_id(groups)
            matched = lookup.get(
                (
                    str(endpoint.design_family),
                    graph_variant,
                    result_signature_id,
                )
            )
            if result_signature_id == str(endpoint.endpoint_signature_id):
                replay_outcome = "replayed_same_endpoint"
            elif matched is not None:
                replay_outcome = "collapsed_to_other_known_endpoint"
            else:
                replay_outcome = "created_new_endpoint"
            read = _mechanism_read(case, membership)
            rows.append(
                {
                    "endpoint_replay_id": str(endpoint.endpoint_replay_id),
                    "design_family": str(endpoint.design_family),
                    "synthetic_demo_role": str(endpoint.synthetic_demo_role),
                    "expected_signature": str(endpoint.expected_signature),
                    "graph_variant": graph_variant,
                    "source_endpoint_signature_id": str(endpoint.endpoint_signature_id),
                    "source_pair_coassigned": bool(endpoint.pair_coassigned),
                    "source_mechanism_read": str(endpoint.mechanism_read),
                    "source_endpoint_run_count": int(endpoint.endpoint_run_count),
                    "source_endpoint_run_share_within_variant": float(
                        endpoint.endpoint_run_share_within_variant
                    ),
                    "source_quality_median": float(endpoint.quality_median),
                    "source_cluster_count_median": float(endpoint.cluster_count_median),
                    "replay_seed": int(seed),
                    "result_endpoint_signature_id": result_signature_id,
                    "result_endpoint_replay_id": (
                        None if matched is None else str(matched["endpoint_replay_id"])
                    ),
                    "result_pair_coassigned": bool(read["pair_coassigned"]),
                    "result_mechanism_read": str(read["mechanism_read"]),
                    "replay_outcome": replay_outcome,
                    "result_quality": float(result.quality),
                    "quality_delta_vs_source_median": float(
                        result.quality - float(endpoint.quality_median)
                    ),
                    "result_cluster_count": int(result.cluster_count),
                    "result_endpoint_signature": json.dumps(groups, sort_keys=True),
                }
            )
    return _claim_columns(pd.DataFrame(rows))


def _summarize_replay(replay_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "endpoint_replay_id",
        "design_family",
        "synthetic_demo_role",
        "expected_signature",
        "graph_variant",
        "source_endpoint_signature_id",
        "source_pair_coassigned",
        "source_mechanism_read",
        "source_endpoint_run_count",
        "source_endpoint_run_share_within_variant",
        "source_quality_median",
        "source_cluster_count_median",
    ]
    for keys, group in replay_runs.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        same = group["replay_outcome"].eq("replayed_same_endpoint")
        known = group["replay_outcome"].eq("collapsed_to_other_known_endpoint")
        new = group["replay_outcome"].eq("created_new_endpoint")
        same_rate = float(same.mean())
        if same_rate == 1.0:
            stability_class = "stable_all_replays"
        elif same_rate >= 0.8:
            stability_class = "mostly_stable"
        elif bool(new.any()):
            stability_class = "unstable_creates_new_endpoint"
        else:
            stability_class = "unstable_collapses_to_known_endpoint"
        rows.append(
            {
                **key,
                "replay_count": int(len(group)),
                "same_endpoint_replay_count": int(same.sum()),
                "same_endpoint_replay_rate": same_rate,
                "known_other_endpoint_count": int(known.sum()),
                "new_endpoint_count": int(new.sum()),
                "distinct_result_endpoint_count": int(group["result_endpoint_signature_id"].nunique()),
                "stability_class": stability_class,
                "is_route_gate_stable_endpoint": bool(same_rate >= 0.8),
                "result_pair_coassigned_share": float(group["result_pair_coassigned"].mean()),
                "quality_delta_min": float(group["quality_delta_vs_source_median"].min()),
                "quality_delta_median": float(group["quality_delta_vs_source_median"].median()),
                "quality_delta_max": float(group["quality_delta_vs_source_median"].max()),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _route_candidates(replay_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stable = replay_summary[
        (replay_summary["graph_variant"] == "original")
        & (replay_summary["is_route_gate_stable_endpoint"].astype(bool))
    ].copy()
    for family, group in stable.groupby("design_family", sort=True):
        coassigned = group[group["source_pair_coassigned"].astype(bool)]
        separated = group[~group["source_pair_coassigned"].astype(bool)]
        if coassigned.empty or separated.empty:
            continue
        sources = pd.concat([coassigned, separated], ignore_index=True)
        for source in sources.to_dict("records"):
            targets = separated if bool(source["source_pair_coassigned"]) else coassigned
            for target in targets.to_dict("records"):
                rows.append(
                    {
                        "route_candidate_id": f"vp_route_{len(rows):04d}",
                        "design_family": str(family),
                        "graph_variant": "original",
                        "route_candidate_status": "g4_endpoint_relation_candidate",
                        "route_relation": (
                            "coassigned_to_separated"
                            if bool(source["source_pair_coassigned"])
                            else "separated_to_coassigned"
                        ),
                        "source_endpoint_replay_id": str(source["endpoint_replay_id"]),
                        "target_endpoint_replay_id": str(target["endpoint_replay_id"]),
                        "source_endpoint_signature_id": str(
                            source["source_endpoint_signature_id"]
                        ),
                        "target_endpoint_signature_id": str(
                            target["source_endpoint_signature_id"]
                        ),
                        "source_pair_coassigned": bool(source["source_pair_coassigned"]),
                        "target_pair_coassigned": bool(target["source_pair_coassigned"]),
                        "source_mechanism_read": str(source["source_mechanism_read"]),
                        "target_mechanism_read": str(target["source_mechanism_read"]),
                        "source_endpoint_run_share_within_variant": float(
                            source["source_endpoint_run_share_within_variant"]
                        ),
                        "target_endpoint_run_share_within_variant": float(
                            target["source_endpoint_run_share_within_variant"]
                        ),
                        "source_same_endpoint_replay_rate": float(
                            source["same_endpoint_replay_rate"]
                        ),
                        "target_same_endpoint_replay_rate": float(
                            target["same_endpoint_replay_rate"]
                        ),
                        "recommended_next_probe": "compact_initial_membership_route_trace",
                        "route_probe_boundary": (
                            "Candidate only: replay-stable endpoints differ in "
                            "L/R co-assignment. A future route probe must trace "
                            "intermediate initial memberships before any wall or "
                            "method claim."
                        ),
                    }
                )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    synthetic_dir: Path,
    output_dir: Path,
    manifest: pd.DataFrame,
    replay_summary: pd.DataFrame,
    route_candidates: pd.DataFrame,
) -> dict[str, Any]:
    original_summary = replay_summary[replay_summary["graph_variant"] == "original"]
    return {
        "schema": "variable_pair_synthetic_endpoint_replay_summary.v1",
        "status": "executed_endpoint_replay_and_route_design",
        "synthetic_dir": str(synthetic_dir),
        "output_dir": str(output_dir),
        "endpoint_count": int(len(manifest)),
        "original_endpoint_count": int(len(original_summary)),
        "stable_original_endpoint_count": int(
            original_summary["is_route_gate_stable_endpoint"].astype(bool).sum()
        ),
        "route_candidate_count": int(len(route_candidates)),
        "route_candidate_family_count": int(
            route_candidates["design_family"].nunique()
            if not route_candidates.empty
            else 0
        ),
        "route_candidate_status_counts": (
            route_candidates["route_candidate_status"].value_counts().to_dict()
            if not route_candidates.empty
            else {}
        ),
        "stability_class_counts": replay_summary["stability_class"].value_counts().to_dict(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    route_candidates: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic Endpoint Replay",
        "",
        f"- status: `{summary['status']}`",
        f"- endpoint_count: {summary['endpoint_count']}",
        f"- original_endpoint_count: {summary['original_endpoint_count']}",
        f"- stable_original_endpoint_count: {summary['stable_original_endpoint_count']}",
        f"- route_candidate_count: {summary['route_candidate_count']}",
        f"- route_candidate_family_count: {summary['route_candidate_family_count']}",
        f"- stability_class_counts: {summary['stability_class_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Route Candidate Families",
    ]
    if route_candidates.empty:
        lines.append("- none")
    else:
        for family, group in route_candidates.groupby("design_family", sort=True):
            relation_counts = group["route_relation"].value_counts().to_dict()
            lines.append(
                f"- {family}: candidates={len(group)}, relations={relation_counts}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This artifact verifies endpoint replay stability and lists "
                "candidate endpoint relations for a future G4 route trace. It "
                "does not execute a route, promote a wall, compare methods, or "
                "evaluate downstream basin value."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    design_dir = Path(args.design_dir)
    synthetic_dir = Path(args.synthetic_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_runs = pd.read_csv(synthetic_dir / SEED_RUNS_CSV)
    manifest = _endpoint_manifest(seed_runs)
    if str(args.graph_variant) != "all":
        manifest = manifest[manifest["graph_variant"].eq(str(args.graph_variant))].copy()
        manifest = manifest.reset_index(drop=True)
        manifest["endpoint_replay_id"] = [
            f"vp_endpoint_{index:04d}" for index in range(len(manifest))
        ]
    replay_runs = _run_replay(
        manifest=manifest,
        design_dir=design_dir,
        replay_seeds=int(args.replay_seeds),
        n_iterations=int(args.n_iterations),
    )
    replay_summary = _summarize_replay(replay_runs)
    route_candidates = _route_candidates(replay_summary)
    _write_csv(manifest, output_dir / ENDPOINT_MANIFEST_CSV)
    _write_csv(replay_runs, output_dir / REPLAY_RUNS_CSV)
    _write_csv(replay_summary, output_dir / REPLAY_SUMMARY_CSV)
    _write_csv(route_candidates, output_dir / ROUTE_CANDIDATES_CSV)
    summary = _summary(
        synthetic_dir=synthetic_dir,
        output_dir=output_dir,
        manifest=manifest,
        replay_summary=replay_summary,
        route_candidates=route_candidates,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_endpoint_replay_config.v1",
        "design_dir": str(design_dir),
        "synthetic_dir": str(synthetic_dir),
        "output_dir": str(output_dir),
        "graph_variant": str(args.graph_variant),
        "replay_seeds": int(args.replay_seeds),
        "n_iterations": int(args.n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        route_candidates=route_candidates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graph-variant", default="original")
    parser.add_argument("--replay-seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
