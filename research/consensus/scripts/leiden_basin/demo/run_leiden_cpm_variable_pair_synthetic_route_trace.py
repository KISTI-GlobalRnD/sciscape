#!/usr/bin/env python3
"""Run compact route traces over variable-pair synthetic endpoint candidates.

This runner consumes replay-stable endpoint-relation candidates from the
variable-pair synthetic endpoint-replay gate. It tests compact initial
membership perturbations and classifies ordinary Leiden+CPM polish outcomes as
target crossing, source bounce, known-endpoint collapse, mixed, or unknown.

It does not promote a basin wall, compare methods, evaluate downstream
quality/cost value, replay NanoClustering, or claim an algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from analyze_leiden_cpm_variable_pair_synthetic_endpoint_replay import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REPLAY_DIR,
    ENDPOINT_MANIFEST_CSV,
    ROUTE_CANDIDATES_CSV,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    DEFAULT_DESIGN_DIR,
    DESIGN_FAMILY_ROWS_CSV,
    _build_graph,
    _canonical_groups,
    _edges_for_variant,
    _json_safe,
    _mechanism_read,
    _renumber,
    _signature_id,
    _synthetic_cases,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_variable_pair_synthetic_route_trace_v1_20260603"
)

TRACE_RUNS_CSV = "variable_pair_synthetic_route_trace_runs.csv"
TRACE_POLICY_SUMMARY_CSV = "variable_pair_synthetic_route_trace_policy_summary.csv"
TRACE_CANDIDATE_SUMMARY_CSV = "variable_pair_synthetic_route_trace_candidate_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_route_trace_summary.json"
CONFIG_JSON = "variable_pair_synthetic_route_trace_config.json"
REPORT_MD = "variable_pair_synthetic_route_trace_report.md"

TRACE_POLICIES = (
    "source_replay",
    "target_replay",
    "pair_relation_only",
    "bridge_side_only",
    "pair_plus_bridge_side",
)
CLAIM_BOUNDARY = (
    "Variable-pair synthetic compact route-trace diagnostic only; replay-stable "
    "synthetic endpoints are used to initialize ordinary Leiden+CPM polish. "
    "No wall promotion, no method comparison, no full NanoClustering replay, "
    "no quality/cost evaluation, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "executed_compact_initial_membership_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_trace_classification_only"
METHOD_STATUS = "route_diagnostic_not_method_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _groups_to_labels(nodes: tuple[str, ...], endpoint_signature: str) -> dict[str, int]:
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
    return labels


def _labels_to_membership(nodes: tuple[str, ...], labels: dict[str, int]) -> list[int]:
    return _renumber([int(labels[node]) for node in nodes])


def _apply_pair_relation(
    *,
    nodes: tuple[str, ...],
    labels: dict[str, int],
    target_labels: dict[str, int],
) -> None:
    if target_labels["L"] == target_labels["R"]:
        labels["R"] = labels["L"]
        return
    if labels["L"] == labels["R"]:
        labels["R"] = max(labels.values()) + 1


def _apply_bridge_side(
    *,
    case,
    labels: dict[str, int],
    target_labels: dict[str, int],
) -> None:
    target_left = target_labels["L"]
    target_right = target_labels["R"]
    for bridge in case.bridge_nodes:
        if target_labels[bridge] == target_left:
            labels[bridge] = labels["L"]
        elif target_labels[bridge] == target_right:
            labels[bridge] = labels["R"]


def _initial_membership_for_policy(
    *,
    case,
    source_signature: str,
    target_signature: str,
    policy: str,
) -> list[int]:
    source_labels = _groups_to_labels(case.nodes, source_signature)
    target_labels = _groups_to_labels(case.nodes, target_signature)
    if policy == "source_replay":
        return _labels_to_membership(case.nodes, source_labels)
    if policy == "target_replay":
        return _labels_to_membership(case.nodes, target_labels)
    labels = dict(source_labels)
    if policy == "pair_relation_only":
        _apply_pair_relation(nodes=case.nodes, labels=labels, target_labels=target_labels)
    elif policy == "bridge_side_only":
        _apply_bridge_side(case=case, labels=labels, target_labels=target_labels)
    elif policy == "pair_plus_bridge_side":
        _apply_pair_relation(nodes=case.nodes, labels=labels, target_labels=target_labels)
        _apply_bridge_side(case=case, labels=labels, target_labels=target_labels)
    else:
        raise ValueError(f"unknown trace policy: {policy}")
    return _labels_to_membership(case.nodes, labels)


def _endpoint_maps(endpoint_manifest: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    by_id = {str(row["endpoint_replay_id"]): row for row in endpoint_manifest.to_dict("records")}
    by_family_sig = {
        (str(row["design_family"]), str(row["endpoint_signature_id"])): row
        for row in endpoint_manifest.to_dict("records")
    }
    return by_id, by_family_sig


def _outcome(
    *,
    result_signature_id: str,
    source_signature_id: str,
    target_signature_id: str,
    matched_endpoint: dict[str, Any] | None,
) -> str:
    if result_signature_id == target_signature_id:
        return "crosses_to_target"
    if result_signature_id == source_signature_id:
        return "bounces_to_source"
    if matched_endpoint is not None:
        return "collapses_to_other_known_endpoint"
    return "unknown_new_endpoint"


def _run_traces(
    *,
    design_dir: Path,
    endpoint_manifest: pd.DataFrame,
    route_candidates: pd.DataFrame,
    trace_seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    families = pd.read_csv(design_dir / DESIGN_FAMILY_ROWS_CSV)
    cases = {case.design_family: case for case in _synthetic_cases(families)}
    by_id, by_family_sig = _endpoint_maps(endpoint_manifest)
    rows: list[dict[str, Any]] = []

    for candidate in route_candidates.itertuples(index=False):
        family = str(candidate.design_family)
        case = cases[family]
        graph = _build_graph(case.nodes, _edges_for_variant(case, "original"))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        source = by_id[str(candidate.source_endpoint_replay_id)]
        target = by_id[str(candidate.target_endpoint_replay_id)]
        source_signature = str(source["endpoint_signature"])
        target_signature = str(target["endpoint_signature"])
        source_signature_id = str(source["endpoint_signature_id"])
        target_signature_id = str(target["endpoint_signature_id"])
        for policy in TRACE_POLICIES:
            initial = _initial_membership_for_policy(
                case=case,
                source_signature=source_signature,
                target_signature=target_signature,
                policy=policy,
            )
            for seed in range(int(trace_seeds)):
                result = runner.run(
                    case.gamma,
                    seed=int(seed),
                    initial_membership=initial,
                    node_sizes=case.node_sizes,
                )
                membership = list(map(int, result.membership))
                groups = _canonical_groups(case.nodes, membership)
                result_signature_id = _signature_id(groups)
                matched = by_family_sig.get((family, result_signature_id))
                read = _mechanism_read(case, membership)
                rows.append(
                    {
                        "route_candidate_id": str(candidate.route_candidate_id),
                        "design_family": family,
                        "route_relation": str(candidate.route_relation),
                        "trace_policy": policy,
                        "trace_seed": int(seed),
                        "source_endpoint_replay_id": str(candidate.source_endpoint_replay_id),
                        "target_endpoint_replay_id": str(candidate.target_endpoint_replay_id),
                        "source_endpoint_signature_id": source_signature_id,
                        "target_endpoint_signature_id": target_signature_id,
                        "result_endpoint_signature_id": result_signature_id,
                        "result_endpoint_replay_id": (
                            None if matched is None else str(matched["endpoint_replay_id"])
                        ),
                        "trace_outcome": _outcome(
                            result_signature_id=result_signature_id,
                            source_signature_id=source_signature_id,
                            target_signature_id=target_signature_id,
                            matched_endpoint=matched,
                        ),
                        "result_pair_coassigned": bool(read["pair_coassigned"]),
                        "result_mechanism_read": str(read["mechanism_read"]),
                        "result_quality": float(result.quality),
                        "result_cluster_count": int(result.cluster_count),
                        "result_endpoint_signature": json.dumps(groups, sort_keys=True),
                    }
                )
    return _claim_columns(pd.DataFrame(rows))


def _policy_summary(trace_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_candidate_id",
        "design_family",
        "route_relation",
        "trace_policy",
        "source_endpoint_replay_id",
        "target_endpoint_replay_id",
    ]
    for keys, group in trace_runs.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        outcomes = group["trace_outcome"].value_counts().to_dict()
        run_count = int(len(group))
        target_rate = float(group["trace_outcome"].eq("crosses_to_target").mean())
        source_rate = float(group["trace_outcome"].eq("bounces_to_source").mean())
        other_known_rate = float(
            group["trace_outcome"].eq("collapses_to_other_known_endpoint").mean()
        )
        unknown_rate = float(group["trace_outcome"].eq("unknown_new_endpoint").mean())
        if target_rate >= 0.8:
            route_trace_class = "crosses_to_target"
        elif source_rate >= 0.8:
            route_trace_class = "bounces_to_source"
        elif other_known_rate >= 0.8:
            route_trace_class = "collapses_to_other_known_endpoint"
        elif unknown_rate >= 0.8:
            route_trace_class = "unknown_new_endpoint"
        else:
            route_trace_class = "mixed_trace_outcomes"
        rows.append(
            {
                **key,
                "run_count": run_count,
                "target_cross_count": int(
                    group["trace_outcome"].eq("crosses_to_target").sum()
                ),
                "source_bounce_count": int(
                    group["trace_outcome"].eq("bounces_to_source").sum()
                ),
                "other_known_collapse_count": int(
                    group["trace_outcome"].eq("collapses_to_other_known_endpoint").sum()
                ),
                "unknown_new_endpoint_count": int(
                    group["trace_outcome"].eq("unknown_new_endpoint").sum()
                ),
                "target_cross_rate": target_rate,
                "source_bounce_rate": source_rate,
                "other_known_collapse_rate": other_known_rate,
                "unknown_new_endpoint_rate": unknown_rate,
                "distinct_result_endpoint_count": int(
                    group["result_endpoint_signature_id"].nunique()
                ),
                "route_trace_class": route_trace_class,
                "result_quality_min": float(group["result_quality"].min()),
                "result_quality_median": float(group["result_quality"].median()),
                "result_quality_max": float(group["result_quality"].max()),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _candidate_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_candidate_id",
        "design_family",
        "route_relation",
        "source_endpoint_replay_id",
        "target_endpoint_replay_id",
    ]
    active = policy_summary[~policy_summary["trace_policy"].isin(["source_replay", "target_replay"])]
    for keys, group in active.groupby(group_cols, sort=True):
        key = dict(zip(group_cols, keys, strict=True))
        rows.append(
            {
                **key,
                "policy_count": int(len(group)),
                "crossing_policy_count": int(
                    group["route_trace_class"].eq("crosses_to_target").sum()
                ),
                "bounce_policy_count": int(
                    group["route_trace_class"].eq("bounces_to_source").sum()
                ),
                "collapse_policy_count": int(
                    group["route_trace_class"].eq("collapses_to_other_known_endpoint").sum()
                ),
                "unknown_policy_count": int(
                    group["route_trace_class"].eq("unknown_new_endpoint").sum()
                ),
                "mixed_policy_count": int(
                    group["route_trace_class"].eq("mixed_trace_outcomes").sum()
                ),
                "candidate_trace_status": (
                    "has_compact_crossing_policy"
                    if group["route_trace_class"].eq("crosses_to_target").any()
                    else "no_compact_crossing_policy"
                ),
                "best_target_cross_rate": float(group["target_cross_rate"].max()),
                "best_source_bounce_rate": float(group["source_bounce_rate"].max()),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    replay_dir: Path,
    output_dir: Path,
    trace_runs: pd.DataFrame,
    policy_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> dict[str, Any]:
    active_policy_summary = policy_summary[
        ~policy_summary["trace_policy"].isin(["source_replay", "target_replay"])
    ]
    return {
        "schema": "variable_pair_synthetic_route_trace_summary.v1",
        "status": "executed_compact_route_trace",
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "trace_run_count": int(len(trace_runs)),
        "route_candidate_count": int(candidate_summary["route_candidate_id"].nunique()),
        "active_policy_count": int(len(active_policy_summary)),
        "candidate_trace_status_counts": candidate_summary[
            "candidate_trace_status"
        ].value_counts().to_dict(),
        "route_trace_class_counts": active_policy_summary[
            "route_trace_class"
        ].value_counts().to_dict(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    candidate_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic Route Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- trace_run_count: {summary['trace_run_count']}",
        f"- route_candidate_count: {summary['route_candidate_count']}",
        f"- candidate_trace_status_counts: {summary['candidate_trace_status_counts']}",
        f"- active_route_trace_class_counts: {summary['route_trace_class_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Candidate Status",
    ]
    for row in candidate_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.route_candidate_id} {row.design_family} {row.route_relation}: "
            f"{row.candidate_trace_status}, "
            f"best_cross={row.best_target_cross_rate:.3f}, "
            f"best_bounce={row.best_source_bounce_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Policy Classes",
        ]
    )
    for row in policy_summary.itertuples(index=False):
        if row.trace_policy in {"source_replay", "target_replay"}:
            continue
        lines.append(
            "- "
            f"{row.route_candidate_id} {row.trace_policy}: "
            f"{row.route_trace_class}, "
            f"target={row.target_cross_rate:.3f}, "
            f"source={row.source_bounce_rate:.3f}, "
            f"other={row.other_known_collapse_rate:.3f}, "
            f"unknown={row.unknown_new_endpoint_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This route trace tests compact initial-membership perturbations "
                "only. It classifies trace outcomes but does not promote a wall, "
                "method, quality/cost, NanoClustering, or algorithm claim."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    design_dir = Path(args.design_dir)
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_manifest = pd.read_csv(replay_dir / ENDPOINT_MANIFEST_CSV)
    route_candidates = pd.read_csv(replay_dir / ROUTE_CANDIDATES_CSV)
    trace_runs = _run_traces(
        design_dir=design_dir,
        endpoint_manifest=endpoint_manifest,
        route_candidates=route_candidates,
        trace_seeds=int(args.trace_seeds),
        n_iterations=int(args.n_iterations),
    )
    policy_summary = _policy_summary(trace_runs)
    candidate_summary = _candidate_summary(policy_summary)
    _write_csv(trace_runs, output_dir / TRACE_RUNS_CSV)
    _write_csv(policy_summary, output_dir / TRACE_POLICY_SUMMARY_CSV)
    _write_csv(candidate_summary, output_dir / TRACE_CANDIDATE_SUMMARY_CSV)
    summary = _summary(
        replay_dir=replay_dir,
        output_dir=output_dir,
        trace_runs=trace_runs,
        policy_summary=policy_summary,
        candidate_summary=candidate_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_route_trace_config.v1",
        "design_dir": str(design_dir),
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "trace_policies": list(TRACE_POLICIES),
        "trace_seeds": int(args.trace_seeds),
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
        candidate_summary=candidate_summary,
        policy_summary=policy_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trace-seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
