#!/usr/bin/env python3
"""Stress coverage-v2 tiny CPM handles under alternative order policies.

This is Stress 1 from the coverage-v2 robustness design. It tests whether the
small-demo coverage-v2 signal depends on a favorable handle order. It is not a
wall/pathway trace, a quality/cost claim, a NanoClustering claim, or an
algorithm-level claim.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
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
from run_leiden_cpm_tiny_handle_method_probe import (
    HandleCandidate,
    _coverage_v2_candidates,
    _handle_candidates,
    _handle_candidates_v1,
    _initial_membership,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_BASELINE_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_demo_seed_sweep_20260531"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_coverage_order_robustness_v1_20260531"

SEED_RUNS_CSV = "leiden_cpm_tiny_demo_seed_runs.csv"
FROZEN_ENDPOINT_MANIFEST_CSV = "leiden_cpm_tiny_demo_frozen_endpoint_manifest.csv"
ORDER_ATTEMPTS_CSV = "tiny_cpm_coverage_order_attempts.csv"
ORDER_DISCOVERY_CSV = "tiny_cpm_coverage_order_discovery.csv"
POLICY_SUMMARY_CSV = "tiny_cpm_coverage_order_policy_summary.csv"
BASELINE_DISTRIBUTION_CSV = "tiny_cpm_restart_baseline_distribution.csv"
BASELINE_SUMMARY_CSV = "tiny_cpm_restart_baseline_summary.csv"
GATE_MATRIX_CSV = "tiny_cpm_coverage_order_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_coverage_order_summary.json"
CONFIG_JSON = "tiny_cpm_coverage_order_config.json"
REPORT_MD = "tiny_cpm_coverage_order_report.md"

DEFAULT_BUDGETS = (1, 2, 3, 5, 10, 20)
DETERMINISTIC_POLICIES = (
    "canonical_v2",
    "v1_first_then_coverage",
    "coverage_first_then_v1",
    "handle_type_round_robin",
    "adversarial_delayed_coverage",
)
RANDOM_POLICY = "random_within_family"

CLAIM_BOUNDARY = (
    "Tiny CPM coverage-v2 order-robustness stress only; handle orders are "
    "varied against the frozen restart baseline, no route execution, no "
    "wall/pathway promotion, no basin-quality claim, no cost claim, no "
    "NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_order_stress_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "candidate_method_order_stress_not_algorithm_claim"


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _candidate_source(candidate: HandleCandidate, coverage_ids: set[str]) -> str:
    return "coverage_v2_addon" if candidate.handle_candidate_id in coverage_ids else "v1"


def _candidates_by_family() -> dict[str, list[HandleCandidate]]:
    rows: dict[str, list[HandleCandidate]] = defaultdict(list)
    for candidate in _handle_candidates("coverage_v2"):
        rows[candidate.family].append(candidate)
    return dict(rows)


def _coverage_ids() -> set[str]:
    return {candidate.handle_candidate_id for candidate in _coverage_v2_candidates()}


def _v1_ids() -> set[str]:
    return {candidate.handle_candidate_id for candidate in _handle_candidates_v1()}


def _round_robin_by_handle_type(candidates: list[HandleCandidate]) -> list[HandleCandidate]:
    by_type: dict[str, list[HandleCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.handle_type].append(candidate)
    ordered: list[HandleCandidate] = []
    type_order = sorted(by_type)
    cursor = 0
    while len(ordered) < len(candidates):
        for handle_type in type_order:
            values = by_type[handle_type]
            if cursor < len(values):
                ordered.append(values[cursor])
        cursor += 1
    return ordered


def _policy_base_order(
    *,
    policy: str,
    family: str,
    candidates: list[HandleCandidate],
    order_sample_id: int,
    rng_seed: int,
    coverage_ids: set[str],
) -> list[HandleCandidate]:
    if policy in {"canonical_v2", "v1_first_then_coverage"}:
        return list(candidates)
    if policy == "coverage_first_then_v1":
        coverage = [candidate for candidate in candidates if candidate.handle_candidate_id in coverage_ids]
        v1 = [candidate for candidate in candidates if candidate.handle_candidate_id not in coverage_ids]
        return [*coverage, *v1] if coverage else v1
    if policy == "handle_type_round_robin":
        return _round_robin_by_handle_type(candidates)
    if policy == RANDOM_POLICY:
        rng = random.Random(f"{rng_seed}:{family}:{order_sample_id}:{policy}")
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled
    if policy == "adversarial_delayed_coverage":
        return list(candidates)
    raise ValueError(f"unknown order policy: {policy}")


def _attempt_schedule(
    *,
    policy: str,
    family: str,
    candidates: list[HandleCandidate],
    max_budget: int,
    order_sample_id: int,
    rng_seed: int,
    coverage_ids: set[str],
) -> list[HandleCandidate]:
    base_order = _policy_base_order(
        policy=policy,
        family=family,
        candidates=candidates,
        order_sample_id=order_sample_id,
        rng_seed=rng_seed,
        coverage_ids=coverage_ids,
    )
    if not base_order:
        raise ValueError(f"no candidates for {family}")

    if policy != "adversarial_delayed_coverage":
        return [base_order[index % len(base_order)] for index in range(max_budget)]

    coverage = [candidate for candidate in candidates if candidate.handle_candidate_id in coverage_ids]
    v1 = [candidate for candidate in candidates if candidate.handle_candidate_id not in coverage_ids]
    if not coverage or not v1:
        return [base_order[index % len(base_order)] for index in range(max_budget)]

    delay_count = max(0, max_budget - len(coverage))
    schedule: list[HandleCandidate] = [
        v1[index % len(v1)]
        for index in range(delay_count)
    ]
    schedule.extend(coverage[: max_budget - len(schedule)])
    while len(schedule) < max_budget:
        schedule.append(v1[(len(schedule) - len(coverage)) % len(v1)])
    return schedule[:max_budget]


def _manifest_sets(manifest: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    sets: dict[str, dict[str, set[str]]] = {}
    for family, group in manifest.groupby("family", sort=True):
        recurrent = set(
            group.loc[group["is_recurrent_endpoint"].astype(bool), "endpoint_signature_id"]
            .astype(str)
            .tolist()
        )
        total = set(group["endpoint_signature_id"].astype(str).tolist())
        top_quality = float(group["quality_max"].max())
        top_quality_ids = set(
            group.loc[group["quality_max"].ge(top_quality - 1e-9), "endpoint_signature_id"]
            .astype(str)
            .tolist()
        )
        sets[str(family)] = {
            "recurrent": recurrent,
            "total": total,
            "top_quality": top_quality_ids,
        }
    return sets


def _manifest_by_signature(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(family): group.set_index("endpoint_signature_id", drop=False)
        for family, group in manifest.groupby("family", sort=True)
    }


def _run_cache(
    *,
    max_budget: int,
    n_iterations: int,
    manifest: pd.DataFrame,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    cases = {case.family: case for case in _graph_cases()}
    by_family = _candidates_by_family()
    coverage_ids = _coverage_ids()
    manifest_by_family = _manifest_by_signature(manifest)
    cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    for family, candidates in sorted(by_family.items()):
        case = cases[family]
        graph = case.builder()
        node_names = list(map(str, graph.vs["name"]))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        for candidate in candidates:
            initial = _initial_membership(node_names, candidate.groups)
            for method_seed in range(max_budget):
                result = runner.run(
                    case.gamma,
                    seed=method_seed,
                    initial_membership=initial,
                )
                membership = list(map(int, result.membership))
                groups = _canonical_groups(graph, membership)
                signature_id = _signature_id(groups)
                frozen_endpoint_id: str | None = None
                baseline_role: str | None = None
                if signature_id in manifest_by_family[family].index:
                    matched = manifest_by_family[family].loc[signature_id]
                    frozen_endpoint_id = str(matched["frozen_endpoint_id"])
                    baseline_role = str(matched["baseline_role"])
                cache[(family, candidate.handle_candidate_id, method_seed)] = {
                    "family": family,
                    "handle_candidate_id": candidate.handle_candidate_id,
                    "candidate_source": _candidate_source(candidate, coverage_ids),
                    "handle_type": candidate.handle_type,
                    "target_mechanism_read": candidate.target_mechanism_read,
                    "handle_node_count": candidate.handle_node_count,
                    "initial_group_count": len(candidate.groups),
                    "method_seed": int(method_seed),
                    "gamma": float(case.gamma),
                    "cluster_count": int(result.cluster_count),
                    "quality": float(result.quality),
                    "endpoint_signature_id": signature_id,
                    "endpoint_signature": json.dumps(groups, sort_keys=True),
                    "frozen_endpoint_id": frozen_endpoint_id,
                    "baseline_role": baseline_role,
                    "is_frozen_endpoint_hit": frozen_endpoint_id is not None,
                    "result_mechanism_read": _classify_mechanism(family, graph, membership),
                }
    return cache


def _run_order_attempts(
    *,
    max_budget: int,
    random_order_samples: int,
    rng_seed: int,
    n_iterations: int,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    by_family = _candidates_by_family()
    coverage_ids = _coverage_ids()
    cache = _run_cache(
        max_budget=max_budget,
        n_iterations=n_iterations,
        manifest=manifest,
    )
    rows: list[dict[str, Any]] = []
    policies = [*DETERMINISTIC_POLICIES, RANDOM_POLICY]
    for family, candidates in sorted(by_family.items()):
        for policy in policies:
            sample_count = random_order_samples if policy == RANDOM_POLICY else 1
            for order_sample_id in range(sample_count):
                schedule = _attempt_schedule(
                    policy=policy,
                    family=family,
                    candidates=candidates,
                    max_budget=max_budget,
                    order_sample_id=order_sample_id,
                    rng_seed=rng_seed,
                    coverage_ids=coverage_ids,
                )
                candidate_seen: dict[str, int] = defaultdict(int)
                for attempt_index, candidate in enumerate(schedule, start=1):
                    method_seed = candidate_seen[candidate.handle_candidate_id]
                    candidate_seen[candidate.handle_candidate_id] += 1
                    cached = cache[(family, candidate.handle_candidate_id, method_seed)]
                    is_target_hit = (
                        cached["is_frozen_endpoint_hit"]
                        and cached["result_mechanism_read"] == candidate.target_mechanism_read
                    )
                    rows.append(
                        {
                            "family": family,
                            "order_policy": policy,
                            "order_sample_id": int(order_sample_id),
                            "attempt_index": int(attempt_index),
                            **cached,
                            "is_target_hit": bool(is_target_hit),
                        }
                    )
    return _with_claim_columns(
        pd.DataFrame(rows).sort_values(
            ["family", "order_policy", "order_sample_id", "attempt_index"]
        )
    )


def _restart_baseline_distribution(
    *,
    seed_runs: pd.DataFrame,
    manifest: pd.DataFrame,
    budgets: list[int],
    permutations: int,
    rng_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = _manifest_sets(manifest)
    rows: list[dict[str, Any]] = []
    for family, group in seed_runs.groupby("family", sort=True):
        family = str(family)
        seed_to_endpoint = {
            int(row.seed): str(row.endpoint_signature_id)
            for row in group[["seed", "endpoint_signature_id"]].itertuples(index=False)
        }
        seed_values = sorted(seed_to_endpoint)
        family_sets = sets[family]
        recurrent_ids = family_sets["recurrent"]
        total_ids = family_sets["total"]
        top_quality_ids = family_sets["top_quality"]
        for budget in budgets:
            if budget > len(seed_values):
                continue
            for permutation in range(permutations):
                rng = random.Random(f"{rng_seed}:restart:{family}:{budget}:{permutation}")
                order = seed_values[:]
                rng.shuffle(order)
                found = {seed_to_endpoint[seed] for seed in order[:budget]}
                frozen_found = found & total_ids
                rows.append(
                    {
                        "family": family,
                        "budget": int(budget),
                        "permutation": int(permutation),
                        "distinct_endpoint_count": int(len(frozen_found)),
                        "recurrent_endpoint_recall": float(
                            len(frozen_found & recurrent_ids) / len(recurrent_ids)
                        ),
                        "all_recurrent_endpoint_hit": bool(recurrent_ids.issubset(frozen_found)),
                        "top_quality_endpoint_hit": bool(frozen_found & top_quality_ids),
                        "all_endpoint_hit": bool(total_ids.issubset(frozen_found)),
                    }
                )
    distribution = pd.DataFrame(rows).sort_values(["family", "budget", "permutation"])
    summary = (
        distribution.groupby(["family", "budget"], as_index=False)
        .agg(
            baseline_distinct_endpoint_count_mean=("distinct_endpoint_count", "mean"),
            baseline_recurrent_recall_mean=("recurrent_endpoint_recall", "mean"),
            baseline_recurrent_recall_p05=(
                "recurrent_endpoint_recall",
                lambda values: float(values.quantile(0.05)),
            ),
            baseline_recurrent_recall_median=("recurrent_endpoint_recall", "median"),
            baseline_recurrent_recall_p95=(
                "recurrent_endpoint_recall",
                lambda values: float(values.quantile(0.95)),
            ),
            baseline_all_recurrent_hit_rate=("all_recurrent_endpoint_hit", "mean"),
            baseline_top_quality_hit_rate=("top_quality_endpoint_hit", "mean"),
            baseline_all_endpoint_hit_rate=("all_endpoint_hit", "mean"),
        )
        .sort_values(["family", "budget"])
    )
    return _with_claim_columns(distribution), _with_claim_columns(summary)


def _order_discovery(
    *,
    attempts: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    budgets: list[int],
) -> pd.DataFrame:
    sets = _manifest_sets(manifest)
    rows: list[dict[str, Any]] = []
    group_cols = ["family", "order_policy", "order_sample_id"]
    for (family, policy, sample_id), group in attempts.groupby(group_cols, sort=True):
        family_sets = sets[str(family)]
        recurrent_ids = family_sets["recurrent"]
        total_ids = family_sets["total"]
        top_quality_ids = family_sets["top_quality"]
        for budget in budgets:
            prefix = group[group["attempt_index"].le(budget)]
            if prefix.empty:
                continue
            found = set(prefix["endpoint_signature_id"].astype(str))
            frozen_found = found & total_ids
            recall = float(len(frozen_found & recurrent_ids) / len(recurrent_ids))
            baseline_row = baseline_summary[
                baseline_summary["family"].eq(family) & baseline_summary["budget"].eq(budget)
            ]
            baseline_mean = (
                float(baseline_row["baseline_recurrent_recall_mean"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            baseline_median = (
                float(baseline_row["baseline_recurrent_recall_median"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            baseline_p95 = (
                float(baseline_row["baseline_recurrent_recall_p95"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            rows.append(
                {
                    "family": str(family),
                    "order_policy": str(policy),
                    "order_sample_id": int(sample_id),
                    "budget": int(budget),
                    "method_distinct_endpoint_count": int(len(frozen_found)),
                    "method_new_endpoint_count": int(len(found - total_ids)),
                    "method_recurrent_endpoint_recall": recall,
                    "method_all_recurrent_endpoint_hit": bool(recurrent_ids.issubset(frozen_found)),
                    "method_top_quality_endpoint_hit": bool(frozen_found & top_quality_ids),
                    "method_target_hit_count": int(prefix["is_target_hit"].astype(bool).sum()),
                    "coverage_addon_attempt_count": int(
                        prefix["candidate_source"].eq("coverage_v2_addon").sum()
                    ),
                    "v1_attempt_count": int(prefix["candidate_source"].eq("v1").sum()),
                    "baseline_recurrent_recall_mean": baseline_mean,
                    "baseline_recurrent_recall_median": baseline_median,
                    "baseline_recurrent_recall_p95": baseline_p95,
                    "delta_vs_baseline_mean": float(recall - baseline_mean)
                    if math.isfinite(baseline_mean)
                    else math.nan,
                    "delta_vs_baseline_median": float(recall - baseline_median)
                    if math.isfinite(baseline_median)
                    else math.nan,
                    "beats_baseline_mean": bool(
                        math.isfinite(baseline_mean) and recall >= baseline_mean
                    ),
                    "beats_baseline_p95": bool(
                        math.isfinite(baseline_p95) and recall >= baseline_p95
                    ),
                }
            )
    return _with_claim_columns(
        pd.DataFrame(rows).sort_values(["family", "order_policy", "order_sample_id", "budget"])
    )


def _policy_summary(discovery: pd.DataFrame) -> pd.DataFrame:
    rows = (
        discovery.groupby(["family", "order_policy", "budget"], as_index=False)
        .agg(
            order_sample_count=("order_sample_id", "nunique"),
            method_recurrent_recall_mean=("method_recurrent_endpoint_recall", "mean"),
            method_recurrent_recall_p05=(
                "method_recurrent_endpoint_recall",
                lambda values: float(values.quantile(0.05)),
            ),
            method_recurrent_recall_median=("method_recurrent_endpoint_recall", "median"),
            method_recurrent_recall_p95=(
                "method_recurrent_endpoint_recall",
                lambda values: float(values.quantile(0.95)),
            ),
            method_all_recurrent_hit_rate=("method_all_recurrent_endpoint_hit", "mean"),
            mean_delta_vs_baseline_mean=("delta_vs_baseline_mean", "mean"),
            median_delta_vs_baseline_mean=("delta_vs_baseline_mean", "median"),
            beat_baseline_mean_rate=("beats_baseline_mean", "mean"),
            beat_baseline_p95_rate=("beats_baseline_p95", "mean"),
            coverage_addon_attempt_count_median=("coverage_addon_attempt_count", "median"),
            v1_attempt_count_median=("v1_attempt_count", "median"),
            baseline_recurrent_recall_mean=("baseline_recurrent_recall_mean", "first"),
            baseline_recurrent_recall_median=("baseline_recurrent_recall_median", "first"),
        )
        .sort_values(["family", "order_policy", "budget"])
    )
    return _with_claim_columns(rows)


def _gate_matrix(policy_summary: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    budget20 = policy_summary[policy_summary["budget"].eq(20)]
    non_adversarial = budget20[
        ~budget20["order_policy"].eq("adversarial_delayed_coverage")
    ]
    under_baseline = non_adversarial[
        non_adversarial["method_recurrent_recall_median"].lt(
            non_adversarial["baseline_recurrent_recall_mean"]
        )
    ]
    diffuse20 = budget20[budget20["family"].eq("diffuse_fragment_star")]
    diffuse_positive = diffuse20[diffuse20["median_delta_vs_baseline_mean"].gt(0)]
    diffuse_policy_count = int(diffuse20["order_policy"].nunique())
    random20 = diffuse20[diffuse20["order_policy"].eq(RANDOM_POLICY)]
    random_beat_rate = (
        float(random20["beat_baseline_mean_rate"].iloc[0])
        if not random20.empty
        else 0.0
    )
    adversarial10 = policy_summary[
        policy_summary["order_policy"].eq("adversarial_delayed_coverage")
        & policy_summary["budget"].eq(10)
    ]
    adversarial20 = policy_summary[
        policy_summary["order_policy"].eq("adversarial_delayed_coverage")
        & policy_summary["budget"].eq(20)
    ]
    rows = [
        {
            "gate_id": "O1_order_stress_executed",
            "gate_question": "Were declared coverage-v2 order policies executed?",
            "evidence": (
                f"families={attempts['family'].nunique()}, "
                f"policies={attempts['order_policy'].nunique()}, "
                f"attempt_rows={len(attempts)}"
            ),
            "status": "pass" if attempts["order_policy"].nunique() >= 6 else "blocked_incomplete_order_grid",
            "decision": "use_as_order_robustness_stress_surface",
            "next_action": "inspect budget-20 deltas and early-budget ordering cost",
        },
        {
            "gate_id": "O2_budget20_non_adversarial_vs_restart",
            "gate_question": "Does budget-20 median performance stay above restart mean outside the adversarial-delay policy?",
            "evidence": f"non_adversarial_family_policy_failures={len(under_baseline)}",
            "status": "pass" if under_baseline.empty else "caveat_required",
            "decision": "coverage_v2_not_canonical_order_only_if_pass",
            "next_action": "localize any failing family-policy rows",
        },
        {
            "gate_id": "O3_diffuse_budget20_policy_robustness",
            "gate_question": "Does the hard diffuse family remain positive under most order policies at budget 20?",
            "evidence": (
                f"diffuse_positive_policies={len(diffuse_positive)}/"
                f"{diffuse_policy_count}, random_beat_rate={random_beat_rate:.6f}"
            ),
            "status": "pass"
            if diffuse_policy_count and len(diffuse_positive) / diffuse_policy_count >= 0.5
            else "caveat_required",
            "decision": "hard_family_signal_is_order_robust_if_pass",
            "next_action": "stress blind rule generation before external empirical use",
        },
        {
            "gate_id": "O4_adversarial_ordering_caveat_recorded",
            "gate_question": "Is delayed coverage recorded as an ordering/cost caveat?",
            "evidence": (
                f"adversarial_budget10_rows={len(adversarial10)}, "
                f"adversarial_budget20_rows={len(adversarial20)}"
            ),
            "status": "pass" if not adversarial10.empty and not adversarial20.empty else "blocked_no_adversarial_trace",
            "decision": "do_not_hide_ordering_cost",
            "next_action": "compare early-budget degradation before any method wording",
        },
        {
            "gate_id": "O5_algorithm_claim_gate",
            "gate_question": "Can ordering robustness claim an algorithmic contribution?",
            "evidence": "tiny controlled order stress only",
            "status": "closed_excluded_by_design",
            "decision": "keep_algorithm_wall_quality_cost_claims_closed",
            "next_action": "run endpoint-derived dependency stress next",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values: list[str] = []
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
    policy_summary: pd.DataFrame,
) -> None:
    budget20 = policy_summary[policy_summary["budget"].eq(20)].copy()
    early = policy_summary[
        policy_summary["budget"].isin([3, 5, 10])
        & policy_summary["family"].eq("diffuse_fragment_star")
    ].copy()
    text = [
        "# Tiny CPM Coverage-v2 Order Robustness Stress v1",
        "",
        f"- order_policy_count: `{summary['order_policy_count']}`",
        f"- random_order_samples: `{summary['random_order_samples']}`",
        f"- attempt_rows: `{summary['attempt_rows']}`",
        f"- budget20_non_adversarial_failures: `{summary['budget20_non_adversarial_failures']}`",
        f"- diffuse_budget20_positive_policy_share: `{summary['diffuse_budget20_positive_policy_share']}`",
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
        "## Budget-20 Policy Summary",
        "",
        _markdown_table(
            budget20,
            [
                "family",
                "order_policy",
                "method_recurrent_recall_median",
                "baseline_recurrent_recall_mean",
                "median_delta_vs_baseline_mean",
                "beat_baseline_mean_rate",
                "coverage_addon_attempt_count_median",
            ],
            max_rows=40,
        ),
        "",
        "## Diffuse Early-Budget Read",
        "",
        _markdown_table(
            early,
            [
                "budget",
                "order_policy",
                "method_recurrent_recall_median",
                "baseline_recurrent_recall_mean",
                "median_delta_vs_baseline_mean",
                "coverage_addon_attempt_count_median",
            ],
            max_rows=40,
        ),
        "",
        "## Read",
        "",
        "- This stress tests order dependence only; it does not validate blind handle generation.",
        "- Budget-20 robustness is necessary but not sufficient because all coverage handles may appear by that budget.",
        "- Early-budget degradation under adversarial delay is an ordering/cost caveat, not a failure of endpoint stability.",
        "- The next mandatory stress is endpoint-derived dependency: generate handles from graph rules without endpoint templates.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_stress(
    *,
    baseline_dir: Path,
    output_dir: Path,
    budgets: list[int],
    max_budget: int,
    baseline_permutations: int,
    random_order_samples: int,
    rng_seed: int,
    n_iterations: int,
) -> dict[str, Any]:
    manifest = pd.read_csv(baseline_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    seed_runs = pd.read_csv(baseline_dir / SEED_RUNS_CSV)
    baseline_distribution, baseline_summary = _restart_baseline_distribution(
        seed_runs=seed_runs,
        manifest=manifest,
        budgets=budgets,
        permutations=baseline_permutations,
        rng_seed=rng_seed,
    )
    attempts = _run_order_attempts(
        max_budget=max_budget,
        random_order_samples=random_order_samples,
        rng_seed=rng_seed,
        n_iterations=n_iterations,
        manifest=manifest,
    )
    discovery = _order_discovery(
        attempts=attempts,
        manifest=manifest,
        baseline_summary=baseline_summary,
        budgets=budgets,
    )
    policy_summary = _policy_summary(discovery)
    gate_matrix = _gate_matrix(policy_summary, attempts)

    budget20 = policy_summary[policy_summary["budget"].eq(20)]
    non_adversarial_failures = budget20[
        ~budget20["order_policy"].eq("adversarial_delayed_coverage")
        & budget20["method_recurrent_recall_median"].lt(
            budget20["baseline_recurrent_recall_mean"]
        )
    ]
    diffuse20 = budget20[budget20["family"].eq("diffuse_fragment_star")]
    diffuse_positive_share = (
        float(diffuse20["median_delta_vs_baseline_mean"].gt(0).mean())
        if not diffuse20.empty
        else None
    )
    summary = {
        "order_policy_count": int(attempts["order_policy"].nunique()),
        "random_order_samples": int(random_order_samples),
        "baseline_permutations": int(baseline_permutations),
        "max_budget": int(max_budget),
        "budgets": [int(budget) for budget in budgets],
        "attempt_rows": int(len(attempts)),
        "discovery_rows": int(len(discovery)),
        "budget20_non_adversarial_failures": int(len(non_adversarial_failures)),
        "diffuse_budget20_positive_policy_share": diffuse_positive_share,
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {"baseline_dir": _rel(baseline_dir)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(baseline_distribution, output_dir / BASELINE_DISTRIBUTION_CSV)
    _write_csv(baseline_summary, output_dir / BASELINE_SUMMARY_CSV)
    _write_csv(attempts, output_dir / ORDER_ATTEMPTS_CSV)
    _write_csv(discovery, output_dir / ORDER_DISCOVERY_CSV)
    _write_csv(policy_summary, output_dir / POLICY_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "baseline_dir": _rel(baseline_dir),
        "output_dir": _rel(output_dir),
        "budgets": budgets,
        "max_budget": int(max_budget),
        "baseline_permutations": int(baseline_permutations),
        "random_order_samples": int(random_order_samples),
        "rng_seed": int(rng_seed),
        "n_iterations": int(n_iterations),
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
        policy_summary=policy_summary,
    )
    return summary


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return sorted(set(budgets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budgets", type=_parse_budgets, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--baseline-permutations", type=int, default=1000)
    parser.add_argument("--random-order-samples", type=int, default=200)
    parser.add_argument("--rng-seed", type=int, default=314159)
    parser.add_argument("--n-iterations", type=int, default=-1)
    args = parser.parse_args()
    budgets = [budget for budget in args.budgets if budget <= args.max_budget]
    summary = run_stress(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        budgets=budgets,
        max_budget=args.max_budget,
        baseline_permutations=args.baseline_permutations,
        random_order_samples=args.random_order_samples,
        rng_seed=args.rng_seed,
        n_iterations=args.n_iterations,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
