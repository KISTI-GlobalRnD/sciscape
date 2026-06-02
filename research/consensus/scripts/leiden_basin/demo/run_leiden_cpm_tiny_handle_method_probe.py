#!/usr/bin/env python3
"""Probe handle-conditioned Leiden + CPM initializations on tiny demo graphs.

This compares simple mechanism-aware initial memberships against the frozen
plain Leiden + CPM random-restart discovery curve. It is a controlled method
probe only: no NanoClustering claim, no wall/pathway promotion, no basin-quality
claim, no cost claim, and no algorithm-level claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from run_leiden_cpm_tiny_demo_seed_sweep import (
    _canonical_groups,
    _classify_mechanism,
    _graph_cases,
    _json_safe,
    _signature_id,
    _write_csv,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_BASELINE_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_demo_seed_sweep_20260531"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_handle_method_v1_20260531"

FROZEN_ENDPOINT_MANIFEST_CSV = "leiden_cpm_tiny_demo_frozen_endpoint_manifest.csv"
BASELINE_DISCOVERY_CURVE_CSV = "leiden_cpm_tiny_demo_discovery_curve.csv"
HANDLE_CANDIDATE_REGISTRY_CSV = "tiny_cpm_handle_candidate_registry.csv"
METHOD_SEED_RUNS_CSV = "tiny_cpm_method_seed_runs.csv"
METHOD_ENDPOINT_HITS_CSV = "tiny_cpm_method_endpoint_hits.csv"
BASELINE_VS_METHOD_DISCOVERY_CSV = "tiny_cpm_baseline_vs_method_discovery.csv"
NAVIGATION_ATTEMPTS_CSV = "tiny_cpm_navigation_attempts.csv"
GATE_MATRIX_CSV = "tiny_cpm_method_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_method_summary.json"
CONFIG_JSON = "tiny_cpm_method_config.json"
REPORT_MD = "tiny_cpm_method_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM handle-method probe only; controlled initial-membership probe "
    "against frozen random-restart baseline, no NanoClustering claim, no "
    "wall/pathway promotion, no basin-quality claim, no cost claim, and no "
    "algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_handle_initialization_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "candidate_method_probe_not_algorithm_claim"
DEFAULT_BUDGETS = (1, 2, 3, 5, 10, 20)


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class HandleCandidate:
    family: str
    handle_candidate_id: str
    handle_type: str
    target_mechanism_read: str
    groups: tuple[tuple[str, ...], ...]
    handle_node_count: int
    handle_description: str


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _group(*nodes: str) -> tuple[str, ...]:
    return tuple(nodes)


def _names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(count))


def _handle_candidates_v1() -> list[HandleCandidate]:
    a5 = _names("a", 5)
    b5 = _names("b", 5)
    a6 = _names("a", 6)
    b6 = _names("b", 6)
    s3 = _names("s", 3)
    m4 = _names("m", 4)
    x4 = _names("x", 4)
    h = {
        host: tuple(f"h{host}_{offset}" for offset in range(4))
        for host in range(4)
    }
    return [
        HandleCandidate(
            "near_tie_bridge_cliques",
            "near_tie_bridge_to_a",
            "bridge_handle_initialization",
            "bridge_to_a",
            (_group(*a5, "x"), _group(*b5)),
            1,
            "Attach bridge node x to host A before Leiden polish.",
        ),
        HandleCandidate(
            "near_tie_bridge_cliques",
            "near_tie_bridge_to_b",
            "bridge_handle_initialization",
            "bridge_to_b",
            (_group(*a5), _group(*b5, "x")),
            1,
            "Attach bridge node x to host B before Leiden polish.",
        ),
        HandleCandidate(
            "near_tie_bridge_cliques",
            "near_tie_bridge_separate",
            "bridge_handle_initialization",
            "bridge_separate",
            (_group(*a5), _group(*b5), _group("x")),
            1,
            "Keep bridge node x separate before Leiden polish.",
        ),
        HandleCandidate(
            "absorption_triad",
            "absorption_small_to_a",
            "small_module_handle_initialization",
            "small_module_absorbed_by_a",
            (_group(*a6, *s3), _group(*b6)),
            3,
            "Attach small module S to host A before Leiden polish.",
        ),
        HandleCandidate(
            "absorption_triad",
            "absorption_small_separate",
            "small_module_handle_initialization",
            "small_module_separate",
            (_group(*a6), _group(*b6), _group(*s3)),
            3,
            "Keep small module S separate before Leiden polish.",
        ),
        HandleCandidate(
            "absorption_triad",
            "absorption_small_to_b",
            "small_module_handle_initialization",
            "small_module_absorbed_by_b",
            (_group(*a6), _group(*b6, *s3)),
            3,
            "Attach small module S to host B before Leiden polish.",
        ),
        HandleCandidate(
            "balanced_split_module",
            "balanced_middle_split",
            "middle_module_split_initialization",
            "balanced_middle_split",
            (_group(*a5, "m0", "m1"), _group(*b5, "m2", "m3")),
            4,
            "Split middle module M across host A and host B.",
        ),
        HandleCandidate(
            "balanced_split_module",
            "balanced_middle_separate",
            "middle_module_split_initialization",
            "middle_module_separate",
            (_group(*a5), _group(*b5), _group(*m4)),
            4,
            "Keep middle module M separate before Leiden polish.",
        ),
        HandleCandidate(
            "balanced_split_module",
            "balanced_middle_to_a",
            "middle_module_split_initialization",
            "middle_module_absorbed_or_merged",
            (_group(*a5, *m4), _group(*b5)),
            4,
            "Attach the whole middle module M to host A.",
        ),
        HandleCandidate(
            "balanced_split_module",
            "balanced_middle_to_b",
            "middle_module_split_initialization",
            "middle_module_absorbed_or_merged",
            (_group(*a5), _group(*b5, *m4)),
            4,
            "Attach the whole middle module M to host B.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_aligned_hosts",
            "weak_fragment_handle_initialization",
            "diffuse_host_fragmentation",
            (
                _group(*h[0], "x0"),
                _group(*h[1], "x1"),
                _group(*h[2], "x2"),
                _group(*h[3], "x3"),
            ),
            4,
            "Attach each weak node x_i to its aligned host h_i.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_shifted_hosts_1",
            "weak_fragment_handle_initialization",
            "diffuse_host_fragmentation",
            (
                _group(*h[0], "x3"),
                _group(*h[1], "x0"),
                _group(*h[2], "x1"),
                _group(*h[3], "x2"),
            ),
            4,
            "Attach weak nodes to the next host cycle before polish.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_shifted_hosts_2",
            "weak_fragment_handle_initialization",
            "diffuse_host_fragmentation",
            (
                _group(*h[0], "x2"),
                _group(*h[1], "x3"),
                _group(*h[2], "x0"),
                _group(*h[3], "x1"),
            ),
            4,
            "Attach weak nodes to a second shifted host cycle.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_weak_separate",
            "weak_fragment_handle_initialization",
            "weak_module_separate",
            (_group(*h[0]), _group(*h[1]), _group(*h[2]), _group(*h[3]), _group(*x4)),
            4,
            "Keep weak module X separate from all hosts before polish.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_all_to_h0",
            "weak_fragment_handle_initialization",
            "weak_module_absorbed_or_merged",
            (_group(*h[0], *x4), _group(*h[1]), _group(*h[2]), _group(*h[3])),
            4,
            "Collapse weak module X into host h0 before polish.",
        ),
    ]


def _coverage_v2_candidates() -> list[HandleCandidate]:
    a5 = _names("a", 5)
    b5 = _names("b", 5)
    a6 = _names("a", 6)
    b6 = _names("b", 6)
    s3 = _names("s", 3)
    m4 = _names("m", 4)
    h = {
        host: tuple(f"h{host}_{offset}" for offset in range(4))
        for host in range(4)
    }
    return [
        HandleCandidate(
            "absorption_triad",
            "absorption_small_to_b_boundary_core",
            "small_module_boundary_core_initialization",
            "small_module_absorbed_by_b",
            (_group(*a6), _group("b0", "b1", *s3), _group("b2", "b3", "b4", "b5")),
            5,
            "Attach S to the B-side boundary core while leaving the B tail separate.",
        ),
        HandleCandidate(
            "balanced_split_module",
            "balanced_middle_to_a_boundary_core",
            "middle_module_boundary_core_initialization",
            "middle_module_absorbed_or_merged",
            (_group("a0", "a1", *m4), _group("a2", "a3", "a4"), _group(*b5)),
            6,
            "Attach M to the A boundary core while keeping the A tail separate.",
        ),
        HandleCandidate(
            "balanced_split_module",
            "balanced_middle_to_b_boundary_core",
            "middle_module_boundary_core_initialization",
            "middle_module_absorbed_or_merged",
            (_group(*a5), _group("b0", "b1", *m4), _group("b2", "b3", "b4")),
            6,
            "Attach M to the B boundary core while keeping the B tail separate.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_h2_pair_tail_split",
            "weak_pair_tail_split_initialization",
            "diffuse_host_fragmentation",
            (
                _group(*h[0], "x0"),
                _group(*h[1]),
                _group("h2_0", "h2_1", "h2_2", "x1", "x2"),
                _group("h2_3"),
                _group(*h[3], "x3"),
            ),
            6,
            "Attach x1/x2 to the h2 boundary core and split the h2 tail.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_h0_h2_pairs_tail_split",
            "weak_pair_tail_split_initialization",
            "diffuse_host_fragmentation",
            (
                _group("h0_0", "h0_1", "h0_2", "x0", "x3"),
                _group("h0_3"),
                _group(*h[1]),
                _group("h2_0", "h2_1", "h2_2", "x1", "x2"),
                _group("h2_3"),
                _group(*h[3]),
            ),
            8,
            "Attach paired weak nodes to h0/h2 boundary cores and split both tails.",
        ),
        HandleCandidate(
            "diffuse_fragment_star",
            "diffuse_h1_h3_pairs_tail_split",
            "weak_pair_tail_split_initialization",
            "diffuse_host_fragmentation",
            (
                _group(*h[0]),
                _group("h1_0", "h1_1", "h1_2", "x0", "x1"),
                _group("h1_3"),
                _group(*h[2]),
                _group("h3_0", "h3_1", "h3_2", "x2", "x3"),
                _group("h3_3"),
            ),
            8,
            "Attach paired weak nodes to h1/h3 boundary cores and split both tails.",
        ),
    ]


def _handle_candidates(candidate_set: str = "v1") -> list[HandleCandidate]:
    if candidate_set == "v1":
        return _handle_candidates_v1()
    if candidate_set == "coverage_v2":
        return [*_handle_candidates_v1(), *_coverage_v2_candidates()]
    raise ValueError(f"unknown candidate set: {candidate_set}")


def _initial_membership(node_names: list[str], groups: tuple[tuple[str, ...], ...]) -> list[int]:
    assigned: dict[str, int] = {}
    for label, group in enumerate(groups):
        for node in group:
            assigned[node] = label
    next_label = len(groups)
    for node in node_names:
        if node not in assigned:
            assigned[node] = next_label
            next_label += 1
    return [assigned[node] for node in node_names]


def _candidate_registry(candidates: list[HandleCandidate]) -> pd.DataFrame:
    rows = [
        {
            "family": candidate.family,
            "handle_candidate_id": candidate.handle_candidate_id,
            "handle_type": candidate.handle_type,
            "target_mechanism_read": candidate.target_mechanism_read,
            "handle_node_count": candidate.handle_node_count,
            "initial_group_count": len(candidate.groups),
            "initial_groups": json.dumps([list(group) for group in candidate.groups]),
            "handle_description": candidate.handle_description,
        }
        for candidate in candidates
    ]
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "handle_candidate_id"]))


def _outcome(
    *,
    target_mechanism_read: str,
    result_mechanism_read: str,
    frozen_endpoint_id: str | None,
) -> str:
    if frozen_endpoint_id is None:
        return "new_endpoint"
    if result_mechanism_read == target_mechanism_read:
        return "target_hit"
    if result_mechanism_read == "unclassified":
        return "unknown"
    return "collapse_to_other_frozen_endpoint"


def _run_method(
    *,
    max_budget: int,
    n_iterations: int,
    manifest: pd.DataFrame,
    candidate_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = {case.family: case for case in _graph_cases()}
    candidates_by_family: dict[str, list[HandleCandidate]] = {}
    for candidate in _handle_candidates(candidate_set):
        candidates_by_family.setdefault(candidate.family, []).append(candidate)
    manifest_by_family = {
        family: group.set_index("endpoint_signature_id")
        for family, group in manifest.groupby("family")
    }
    rows: list[dict[str, Any]] = []
    for family, candidates in sorted(candidates_by_family.items()):
        case = cases[family]
        graph = case.builder()
        node_names = list(map(str, graph.vs["name"]))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        for attempt_index in range(max_budget):
            candidate = candidates[attempt_index % len(candidates)]
            round_index = attempt_index // len(candidates)
            initial = _initial_membership(node_names, candidate.groups)
            result = runner.run(
                case.gamma,
                seed=round_index,
                initial_membership=initial,
            )
            membership = list(map(int, result.membership))
            groups = _canonical_groups(graph, membership)
            signature_id = _signature_id(groups)
            result_mechanism = _classify_mechanism(family, graph, membership)
            frozen_endpoint_id: str | None = None
            baseline_role: str | None = None
            if signature_id in manifest_by_family[family].index:
                matched = manifest_by_family[family].loc[signature_id]
                frozen_endpoint_id = str(matched["frozen_endpoint_id"])
                baseline_role = str(matched["baseline_role"])
            rows.append(
                {
                    "family": family,
                    "attempt_index": attempt_index + 1,
                    "handle_candidate_id": candidate.handle_candidate_id,
                    "handle_type": candidate.handle_type,
                    "target_mechanism_read": candidate.target_mechanism_read,
                    "result_mechanism_read": result_mechanism,
                    "navigation_outcome": _outcome(
                        target_mechanism_read=candidate.target_mechanism_read,
                        result_mechanism_read=result_mechanism,
                        frozen_endpoint_id=frozen_endpoint_id,
                    ),
                    "handle_node_count": candidate.handle_node_count,
                    "initial_group_count": len(candidate.groups),
                    "method_seed": round_index,
                    "gamma": case.gamma,
                    "cluster_count": result.cluster_count,
                    "quality": float(result.quality),
                    "endpoint_signature_id": signature_id,
                    "endpoint_signature": json.dumps(groups, sort_keys=True),
                    "frozen_endpoint_id": frozen_endpoint_id,
                    "baseline_role": baseline_role,
                    "is_frozen_endpoint_hit": frozen_endpoint_id is not None,
                    "is_target_hit": result_mechanism == candidate.target_mechanism_read
                    and frozen_endpoint_id is not None,
                }
            )
    runs = _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "attempt_index"]))
    hits = (
        runs.groupby(
            [
                "family",
                "endpoint_signature_id",
                "frozen_endpoint_id",
                "baseline_role",
                "result_mechanism_read",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            hit_count=("attempt_index", "size"),
            first_attempt=("attempt_index", "min"),
            best_quality=("quality", "max"),
            target_hit_count=("is_target_hit", "sum"),
        )
        .sort_values(["family", "first_attempt", "endpoint_signature_id"])
    )
    return runs, _with_claim_columns(hits)


def _method_discovery(
    *,
    method_runs: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    budgets: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in method_runs.groupby("family", sort=True):
        manifest_group = manifest[manifest["family"].eq(family)]
        recurrent_ids = set(
            manifest_group.loc[
                manifest_group["is_recurrent_endpoint"].astype(bool), "endpoint_signature_id"
            ].astype(str)
        )
        top_quality = float(manifest_group["quality_max"].max())
        top_quality_ids = set(
            manifest_group.loc[
                manifest_group["quality_max"].ge(top_quality - 1e-9),
                "endpoint_signature_id",
            ].astype(str)
        )
        total_ids = set(manifest_group["endpoint_signature_id"].astype(str))
        for budget in budgets:
            prefix = group[group["attempt_index"].le(budget)]
            found = set(prefix["endpoint_signature_id"].astype(str))
            frozen_found = found & total_ids
            recurrent_recall = len(frozen_found & recurrent_ids) / len(recurrent_ids)
            baseline_row = baseline_curve[
                baseline_curve["family"].eq(family) & baseline_curve["budget"].eq(budget)
            ]
            baseline_recall = (
                float(baseline_row["recurrent_endpoint_recall_mean"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            baseline_distinct = (
                float(baseline_row["distinct_endpoint_count_mean"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            baseline_all_recurrent = (
                float(baseline_row["all_recurrent_endpoint_hit_rate"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            rows.append(
                {
                    "family": str(family),
                    "budget": int(budget),
                    "method_distinct_endpoint_count": int(len(frozen_found)),
                    "method_new_endpoint_count": int(len(found - total_ids)),
                    "method_recurrent_endpoint_recall": float(recurrent_recall),
                    "method_all_recurrent_endpoint_hit": bool(recurrent_ids.issubset(frozen_found)),
                    "method_top_quality_endpoint_hit": bool(frozen_found & top_quality_ids),
                    "method_target_hit_count": int(prefix["is_target_hit"].astype(bool).sum()),
                    "method_attempt_count": int(len(prefix)),
                    "baseline_distinct_endpoint_count_mean": baseline_distinct,
                    "baseline_recurrent_endpoint_recall_mean": baseline_recall,
                    "baseline_all_recurrent_endpoint_hit_rate": baseline_all_recurrent,
                    "delta_distinct_endpoint_count": float(
                        len(frozen_found) - baseline_distinct
                    )
                    if math.isfinite(baseline_distinct)
                    else math.nan,
                    "delta_recurrent_endpoint_recall": float(
                        recurrent_recall - baseline_recall
                    )
                    if math.isfinite(baseline_recall)
                    else math.nan,
                    "delta_all_recurrent_endpoint_hit": float(
                        (1.0 if recurrent_ids.issubset(frozen_found) else 0.0)
                        - baseline_all_recurrent
                    )
                    if math.isfinite(baseline_all_recurrent)
                    else math.nan,
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "budget"]))


def _gate_matrix(discovery: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    budget20 = discovery[discovery["budget"].eq(20)]
    max_delta_recall = float(discovery["delta_recurrent_endpoint_recall"].max())
    diffuse20 = budget20[budget20["family"].eq("diffuse_fragment_star")]
    diffuse_delta = (
        float(diffuse20["delta_recurrent_endpoint_recall"].iloc[0])
        if not diffuse20.empty
        else 0.0
    )
    target_hit_rate = float(runs["is_target_hit"].astype(bool).mean()) if not runs.empty else 0.0
    rows = [
        {
            "gate_id": "H1_handle_method_executed",
            "gate_question": "Were handle-conditioned probes executed for every tiny graph family?",
            "evidence": (
                f"families={runs['family'].nunique()}, "
                f"attempts={len(runs)}"
            ),
            "status": "pass" if runs["family"].nunique() == 4 else "blocked_incomplete_family_grid",
            "decision": "use_as_candidate_method_probe_surface",
            "next_action": "inspect per-family discovery deltas",
        },
        {
            "gate_id": "H2_any_discovery_improvement",
            "gate_question": "Does the handle method beat the frozen restart curve anywhere?",
            "evidence": f"max_delta_recurrent_recall={max_delta_recall:.6f}",
            "status": "pass" if max_delta_recall > 0 else "blocked_no_discovery_gain",
            "decision": "candidate_signal_exists_if_pass",
            "next_action": "localize gains by mechanism family and budget",
        },
        {
            "gate_id": "H3_hard_family_diffuse_budget20",
            "gate_question": "Does the method improve the hard diffuse family at budget 20?",
            "evidence": f"diffuse_budget20_delta_recurrent_recall={diffuse_delta:.6f}",
            "status": "pass" if diffuse_delta > 0 else "caveat_required",
            "decision": "hard_family_navigation_signal_if_pass",
            "next_action": "inspect diffuse target misses and new endpoints",
        },
        {
            "gate_id": "H4_navigation_target_hit_rate",
            "gate_question": "Do mechanism handles tend to hit their target endpoint class?",
            "evidence": f"target_hit_rate={target_hit_rate:.6f}",
            "status": "pass" if target_hit_rate >= 0.5 else "caveat_required",
            "decision": "handle_navigation_is_directional_if_pass",
            "next_action": "separate deterministic handles from unstable handles",
        },
        {
            "gate_id": "H5_algorithm_claim_gate",
            "gate_question": "Can this be claimed as an algorithm improvement?",
            "evidence": "tiny controlled handle probe only",
            "status": "closed_excluded_by_design",
            "decision": "keep_algorithm_and_quality_claims_closed",
            "next_action": "promote only after robustness and external empirical stress test",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix = [f"\n_Showing {max_rows} of {len(frame)} rows._"] if len(frame) > max_rows else []
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    gate_matrix: pd.DataFrame,
    discovery: pd.DataFrame,
    runs: pd.DataFrame,
) -> None:
    nav = (
        runs.groupby(["family", "handle_candidate_id", "target_mechanism_read"], as_index=False)
        .agg(
            attempt_count=("attempt_index", "size"),
            target_hit_count=("is_target_hit", "sum"),
            first_attempt=("attempt_index", "min"),
        )
        .sort_values(["family", "first_attempt"])
    )
    nav["target_hit_rate"] = nav["target_hit_count"] / nav["attempt_count"]
    text = [
        f"# Tiny CPM Handle Method Probe ({summary['candidate_set']})",
        "",
        f"- handle_candidate_count: `{summary['handle_candidate_count']}`",
        f"- method_attempt_count: `{summary['method_attempt_count']}`",
        f"- max_delta_recurrent_endpoint_recall: `{summary['max_delta_recurrent_endpoint_recall']}`",
        f"- diffuse_budget20_delta_recurrent_recall: `{summary['diffuse_budget20_delta_recurrent_recall']}`",
        f"- target_hit_rate: `{summary['target_hit_rate']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=10,
        ),
        "",
        "## Baseline Vs Method Discovery",
        "",
        _markdown_table(
            discovery[discovery["budget"].isin([3, 5, 10, 20])],
            [
                "family",
                "budget",
                "method_recurrent_endpoint_recall",
                "baseline_recurrent_endpoint_recall_mean",
                "delta_recurrent_endpoint_recall",
                "method_target_hit_count",
            ],
            max_rows=32,
        ),
        "",
        "## Navigation Attempts",
        "",
        _markdown_table(
            nav,
            [
                "family",
                "handle_candidate_id",
                "target_mechanism_read",
                "attempt_count",
                "target_hit_count",
                "target_hit_rate",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- This is the first controlled candidate-method comparison against the frozen restart baseline.",
        "- Positive deltas are method signals only on tiny demo graphs, not algorithm claims.",
        "- Diffuse-fragment performance is the key hard-case read because restart discovery is weakest there.",
        "- The next step is robustness across handle ordering and larger seed budgets before NanoClustering stress tests.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_probe(
    *,
    baseline_dir: Path,
    output_dir: Path,
    max_budget: int,
    n_iterations: int,
    candidate_set: str,
) -> dict[str, Any]:
    manifest = pd.read_csv(baseline_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    baseline_curve = pd.read_csv(baseline_dir / BASELINE_DISCOVERY_CURVE_CSV)
    candidates = _handle_candidates(candidate_set)
    candidate_registry = _candidate_registry(candidates)
    method_runs, endpoint_hits = _run_method(
        max_budget=max_budget,
        n_iterations=n_iterations,
        manifest=manifest,
        candidate_set=candidate_set,
    )
    budgets = [budget for budget in DEFAULT_BUDGETS if budget <= max_budget]
    discovery = _method_discovery(
        method_runs=method_runs,
        manifest=manifest,
        baseline_curve=baseline_curve,
        budgets=budgets,
    )
    navigation_attempts = method_runs[
        [
            "family",
            "attempt_index",
            "handle_candidate_id",
            "target_mechanism_read",
            "result_mechanism_read",
            "navigation_outcome",
            "is_target_hit",
            "frozen_endpoint_id",
            "baseline_role",
            "quality",
            "cluster_count",
        ]
    ].copy()
    navigation_attempts = _with_claim_columns(navigation_attempts)
    gate_matrix = _gate_matrix(discovery, method_runs)

    budget20 = discovery[discovery["budget"].eq(20)]
    diffuse20 = budget20[budget20["family"].eq("diffuse_fragment_star")]
    summary = {
        "handle_candidate_count": int(len(candidate_registry)),
        "method_attempt_count": int(len(method_runs)),
        "method_family_count": int(method_runs["family"].nunique()),
        "candidate_set": candidate_set,
        "max_budget": int(max_budget),
        "max_delta_recurrent_endpoint_recall": float(
            discovery["delta_recurrent_endpoint_recall"].max()
        ),
        "diffuse_budget20_delta_recurrent_recall": float(
            diffuse20["delta_recurrent_endpoint_recall"].iloc[0]
        )
        if not diffuse20.empty
        else None,
        "target_hit_rate": float(method_runs["is_target_hit"].astype(bool).mean()),
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {"baseline_dir": _rel(baseline_dir)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(candidate_registry, output_dir / HANDLE_CANDIDATE_REGISTRY_CSV)
    _write_csv(method_runs, output_dir / METHOD_SEED_RUNS_CSV)
    _write_csv(endpoint_hits, output_dir / METHOD_ENDPOINT_HITS_CSV)
    _write_csv(discovery, output_dir / BASELINE_VS_METHOD_DISCOVERY_CSV)
    _write_csv(navigation_attempts, output_dir / NAVIGATION_ATTEMPTS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "baseline_dir": _rel(baseline_dir),
        "output_dir": _rel(output_dir),
        "candidate_set": candidate_set,
        "max_budget": int(max_budget),
        "n_iterations": int(n_iterations),
        "budgets": budgets,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        gate_matrix=gate_matrix,
        discovery=discovery,
        runs=method_runs,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument(
        "--candidate-set",
        choices=["v1", "coverage_v2"],
        default="v1",
        help="Handle candidate registry to execute. coverage_v2 appends replay-diagnosed coverage handles.",
    )
    args = parser.parse_args()
    summary = run_probe(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        max_budget=args.max_budget,
        n_iterations=args.n_iterations,
        candidate_set=args.candidate_set,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
