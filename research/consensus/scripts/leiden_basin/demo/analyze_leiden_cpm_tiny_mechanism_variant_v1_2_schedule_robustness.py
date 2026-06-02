#!/usr/bin/env python3
"""Stress v1.2 joint weak-pair evidence under alternative schedule orders.

This diagnostic asks whether the Stress 4 v1.2 structural hit result depends on
the canonical candidate order. It reads the phase-locked P0-P4 registry and
the already-frozen P5-P8 endpoint universe, then replays candidate schedules
under deterministic and random order policies. It does not construct candidates
from endpoint misses and does not execute route/pathway traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from analyze_leiden_cpm_tiny_mechanism_variant_p8_failure_typing import (
    _structural_target_hit,
)
from run_leiden_cpm_tiny_demo_seed_sweep import (
    _canonical_groups,
    _json_safe,
    _signature_id,
    _write_csv,
)
from run_leiden_cpm_tiny_mechanism_variant_p5_p8 import (
    BASELINE_DISCOVERY_CSV,
    BLIND_CANDIDATE_REGISTRY_CSV,
    FROZEN_ENDPOINT_MANIFEST_CSV,
    GRAPH_EDGES_CSV,
    GRAPH_MANIFEST_CSV,
    GRAPH_ROLES_CSV,
    PHASE_LOCK_JSON,
    VariantInput,
    _initial_membership,
    _variant_inputs,
    _variant_mechanism_read,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_INPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_2_20260601"
DEFAULT_P5_P8_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_2_20260601"
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_schedule_robustness_v1_2_20260601"
)

SCHEDULE_ATTEMPTS_CSV = "tiny_cpm_variant_v1_2_schedule_attempts.csv"
ENDPOINT_DISCOVERY_CSV = "tiny_cpm_variant_v1_2_schedule_endpoint_discovery.csv"
POLICY_SUMMARY_CSV = "tiny_cpm_variant_v1_2_schedule_policy_summary.csv"
VARIANT_POLICY_SUMMARY_CSV = "tiny_cpm_variant_v1_2_schedule_variant_policy_summary.csv"
GATE_MATRIX_CSV = "tiny_cpm_variant_v1_2_schedule_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_variant_v1_2_schedule_robustness_summary.json"
CONFIG_JSON = "tiny_cpm_variant_v1_2_schedule_robustness_config.json"
REPORT_MD = "tiny_cpm_variant_v1_2_schedule_robustness_report.md"

DETERMINISTIC_POLICIES = (
    "canonical_sorted",
    "positive_first",
    "joint_first",
    "joint_delayed",
    "joint_suppressed_negative_control",
)
RANDOM_POLICY = "random_permutation"

CLAIM_BOUNDARY = (
    "Tiny CPM mechanism-variant v1.2 schedule-order robustness diagnostic only; "
    "reads phase-locked P0-P4 candidates and frozen P5-P8 endpoints, varies "
    "candidate schedule order, no new endpoint-informed candidate construction, "
    "no route/pathway execution, no wall promotion, no quality/cost claim, no "
    "NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_schedule_order_diagnostic_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "candidate_schedule_robustness_not_algorithm_claim"


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_phase_lock(input_dir: Path) -> tuple[dict[str, Any], bool, list[str]]:
    phase_lock = _load_json(input_dir / PHASE_LOCK_JSON)
    mismatches: list[str] = []
    for filename, expected_hash in sorted(phase_lock.get("artifact_hashes", {}).items()):
        path = input_dir / filename
        if not path.exists():
            mismatches.append(f"{filename}:missing")
            continue
        if _sha256_file(path) != expected_hash:
            mismatches.append(f"{filename}:hash_mismatch")
    return phase_lock, not mismatches, mismatches


def _load_inputs(input_dir: Path, p5_p8_dir: Path) -> dict[str, Any]:
    required_input = [
        GRAPH_MANIFEST_CSV,
        GRAPH_EDGES_CSV,
        GRAPH_ROLES_CSV,
        BLIND_CANDIDATE_REGISTRY_CSV,
        PHASE_LOCK_JSON,
    ]
    required_p5 = [FROZEN_ENDPOINT_MANIFEST_CSV, BASELINE_DISCOVERY_CSV]
    missing = [name for name in required_input if not (input_dir / name).exists()]
    missing.extend(name for name in required_p5 if not (p5_p8_dir / name).exists())
    if missing:
        raise FileNotFoundError(f"missing schedule robustness inputs: {missing}")
    phase_lock, phase_lock_verified, phase_lock_mismatches = _verify_phase_lock(input_dir)
    return {
        "phase_lock": phase_lock,
        "phase_lock_verified": phase_lock_verified,
        "phase_lock_mismatches": phase_lock_mismatches,
        "manifest": pd.read_csv(input_dir / GRAPH_MANIFEST_CSV),
        "edges": pd.read_csv(input_dir / GRAPH_EDGES_CSV),
        "roles": pd.read_csv(input_dir / GRAPH_ROLES_CSV),
        "registry": pd.read_csv(input_dir / BLIND_CANDIDATE_REGISTRY_CSV),
        "frozen_manifest": pd.read_csv(p5_p8_dir / FROZEN_ENDPOINT_MANIFEST_CSV),
        "baseline": pd.read_csv(p5_p8_dir / BASELINE_DISCOVERY_CSV),
    }


def _is_joint_candidate(row: pd.Series | Any) -> bool:
    handle_type = str(getattr(row, "handle_type", row["handle_type"] if isinstance(row, pd.Series) else ""))
    target_class = str(
        getattr(row, "target_mechanism_class", row["target_mechanism_class"] if isinstance(row, pd.Series) else "")
    )
    candidate_id = str(
        getattr(row, "handle_candidate_id", row["handle_candidate_id"] if isinstance(row, pd.Series) else "")
    )
    return "joint_weak_pair" in f"{handle_type} {target_class} {candidate_id}"


def _candidate_order(
    group: pd.DataFrame,
    *,
    policy: str,
    sample_id: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows = group.sort_values("handle_candidate_id").copy()
    rows["_is_positive"] = rows["target_claim_allowed"].astype(bool)
    rows["_is_joint"] = rows.apply(_is_joint_candidate, axis=1)
    if policy == "canonical_sorted":
        ordered = rows
    elif policy == "positive_first":
        ordered = rows.sort_values(["_is_positive", "handle_candidate_id"], ascending=[False, True])
    elif policy == "joint_first":
        ordered = rows.sort_values(["_is_joint", "_is_positive", "handle_candidate_id"], ascending=[False, False, True])
    elif policy == "joint_delayed":
        ordered = rows.sort_values(["_is_joint", "_is_positive", "handle_candidate_id"], ascending=[True, False, True])
    elif policy == "joint_suppressed_negative_control":
        ordered = rows[~rows["_is_joint"]].copy()
    elif policy == RANDOM_POLICY:
        rng = random.Random(f"{rng_seed}:{group['variant_id'].iloc[0]}:{policy}:{sample_id}")
        records = rows.to_dict("records")
        rng.shuffle(records)
        ordered = pd.DataFrame(records)
    else:
        raise ValueError(f"unknown schedule policy: {policy}")
    ordered = ordered.drop(columns=[column for column in ["_is_positive", "_is_joint"] if column in ordered.columns])
    if ordered.empty:
        raise ValueError(f"schedule policy {policy} produced no candidates for {group['variant_id'].iloc[0]}")
    return ordered.reset_index(drop=True)


def _schedule_rows(
    registry: pd.DataFrame,
    *,
    max_budget: int,
    random_samples: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedule_specs = [(policy, 0) for policy in DETERMINISTIC_POLICIES]
    schedule_specs.extend((RANDOM_POLICY, sample_id) for sample_id in range(random_samples))
    for variant_id, group in registry.groupby("variant_id", sort=True):
        for policy, sample_id in schedule_specs:
            ordered = _candidate_order(group, policy=policy, sample_id=sample_id, rng_seed=rng_seed)
            candidate_seen: dict[str, int] = defaultdict(int)
            for attempt_index in range(max_budget):
                candidate = ordered.iloc[attempt_index % len(ordered)]
                candidate_id = str(candidate["handle_candidate_id"])
                method_seed = candidate_seen[candidate_id]
                candidate_seen[candidate_id] += 1
                rows.append(
                    {
                        "variant_id": str(variant_id),
                        "schedule_policy": policy,
                        "schedule_sample_id": int(sample_id),
                        "attempt_index": int(attempt_index + 1),
                        "candidate_cycle_index": int(attempt_index % len(ordered)),
                        "schedule_candidate_count": int(len(ordered)),
                        "method_seed": int(method_seed),
                        "handle_candidate_id": candidate_id,
                        "candidate_signature_id": str(candidate["candidate_signature_id"]),
                        "candidate_class_signature_id": str(candidate["candidate_class_signature_id"]),
                        "handle_type": str(candidate["handle_type"]),
                        "target_mechanism_class": str(candidate["target_mechanism_class"]),
                        "target_claim_allowed": bool(candidate["target_claim_allowed"]),
                        "is_joint_candidate": bool(_is_joint_candidate(candidate)),
                    }
                )
    return rows


def _manifest_by_signature(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(variant): group.set_index("endpoint_signature_id", drop=False)
        for variant, group in manifest.groupby("variant_id", sort=True)
    }


def _eligible_endpoint_candidates(
    *,
    roles: pd.DataFrame,
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    positive = registry[registry["target_claim_allowed"].astype(bool)].copy()
    preserved = manifest[
        manifest["is_recurrent_endpoint"].astype(bool)
        & ~manifest["mechanism_state"].astype(str).str.contains("control")
    ].copy()
    rows: list[dict[str, Any]] = []
    for endpoint in preserved.itertuples(index=False):
        variant_id = str(endpoint.variant_id)
        variant_candidates = positive[positive["variant_id"].astype(str).eq(variant_id)]
        for candidate in variant_candidates.itertuples(index=False):
            if not _structural_target_hit(
                roles=roles,
                variant_id=variant_id,
                candidate_id=str(candidate.handle_candidate_id),
                target_mechanism_class=str(candidate.target_mechanism_class),
                endpoint_signature=str(endpoint.endpoint_signature),
            ):
                continue
            rows.append(
                {
                    "variant_id": variant_id,
                    "frozen_endpoint_id": str(endpoint.frozen_endpoint_id),
                    "endpoint_signature_id": str(endpoint.endpoint_signature_id),
                    "handle_candidate_id": str(candidate.handle_candidate_id),
                    "target_mechanism_class": str(candidate.target_mechanism_class),
                    "is_joint_candidate": bool(_is_joint_candidate(candidate)),
                }
            )
    return pd.DataFrame(rows)


def _run_attempts(
    *,
    variants: list[VariantInput],
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
    roles: pd.DataFrame,
    max_budget: int,
    random_samples: int,
    rng_seed: int,
    n_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_by_id = {variant.variant_id: variant for variant in variants}
    registry_by_id = registry.set_index("handle_candidate_id", drop=False)
    manifest_by_variant = _manifest_by_signature(manifest)
    eligible = _eligible_endpoint_candidates(roles=roles, registry=registry, manifest=manifest)
    eligible_pairs = {
        (str(row.frozen_endpoint_id), str(row.handle_candidate_id))
        for row in eligible.itertuples(index=False)
    }
    eligible_joint_endpoints = set(
        eligible.loc[eligible["is_joint_candidate"].astype(bool), "frozen_endpoint_id"].astype(str).tolist()
    )
    schedule = _schedule_rows(
        registry,
        max_budget=max_budget,
        random_samples=random_samples,
        rng_seed=rng_seed,
    )
    runners: dict[str, LeidenRunner] = {}
    node_names: dict[str, list[str]] = {}
    cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for item in schedule:
        variant_id = str(item["variant_id"])
        variant = variant_by_id[variant_id]
        if variant_id not in runners:
            runners[variant_id] = LeidenRunner(variant.graph, objective="cpm", default_iterations=n_iterations)
            node_names[variant_id] = list(map(str, variant.graph.vs["name"]))
        candidate_id = str(item["handle_candidate_id"])
        method_seed = int(item["method_seed"])
        cache_key = (variant_id, candidate_id, method_seed)
        if cache_key not in cache:
            candidate = registry_by_id.loc[candidate_id]
            groups = tuple(tuple(value) for value in json.loads(str(candidate["initial_groups"])))
            initial = _initial_membership(node_names[variant_id], groups)
            result = runners[variant_id].run(variant.gamma, seed=method_seed, initial_membership=initial)
            membership = list(map(int, result.membership))
            endpoint_groups = _canonical_groups(variant.graph, membership)
            signature_id = _signature_id(endpoint_groups)
            frozen_endpoint_id: str | None = None
            baseline_role: str | None = None
            endpoint_rank: int | None = None
            endpoint_signature: str | None = None
            endpoint_mechanism_read: str | None = None
            if signature_id in manifest_by_variant[variant_id].index:
                matched = manifest_by_variant[variant_id].loc[signature_id]
                frozen_endpoint_id = str(matched["frozen_endpoint_id"])
                baseline_role = str(matched["baseline_role"])
                endpoint_rank = int(matched["endpoint_rank_in_variant"])
                endpoint_signature = str(matched["endpoint_signature"])
                endpoint_mechanism_read = str(matched["mechanism_read"])
            cache[cache_key] = {
                "cluster_count": int(result.cluster_count),
                "quality": float(result.quality),
                "endpoint_signature_id": signature_id,
                "endpoint_signature": json.dumps(endpoint_groups, sort_keys=True),
                "frozen_endpoint_id": frozen_endpoint_id,
                "baseline_role": baseline_role,
                "endpoint_rank_in_variant": endpoint_rank,
                "endpoint_mechanism_read": endpoint_mechanism_read,
                "result_mechanism_read": _variant_mechanism_read(variant, membership),
                "matched_endpoint_signature": endpoint_signature,
            }
        result_row = cache[cache_key]
        structural_hit = False
        if result_row["frozen_endpoint_id"] is not None:
            structural_hit = (str(result_row["frozen_endpoint_id"]), candidate_id) in eligible_pairs
        rows.append(
            {
                **item,
                "mechanism_family": variant.mechanism_family,
                "mechanism_state": variant.mechanism_state,
                "responsible_rule": variant.responsible_rule,
                "gamma": float(variant.gamma),
                **result_row,
                "is_frozen_endpoint_hit": result_row["frozen_endpoint_id"] is not None,
                "is_recurrent_endpoint_hit": result_row["baseline_role"] == "recurrent_baseline_endpoint",
                "is_structural_target_hit": bool(structural_hit),
                "is_joint_eligible_endpoint_hit": bool(
                    structural_hit and str(result_row["frozen_endpoint_id"]) in eligible_joint_endpoints
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["schedule_policy", "schedule_sample_id", "variant_id", "attempt_index"])), eligible


def _endpoint_discovery(
    *,
    attempts: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    preserved = manifest[
        manifest["is_recurrent_endpoint"].astype(bool)
        & ~manifest["mechanism_state"].astype(str).str.contains("control")
    ].copy()
    eligible_by_endpoint = {
        str(endpoint_id): group
        for endpoint_id, group in eligible.groupby("frozen_endpoint_id", sort=True)
    }
    baseline_lookup = baseline.set_index(["variant_id", "frozen_endpoint_id"])
    schedule_keys = attempts[["schedule_policy", "schedule_sample_id"]].drop_duplicates().sort_values(
        ["schedule_policy", "schedule_sample_id"]
    )
    rows: list[dict[str, Any]] = []
    for schedule in schedule_keys.itertuples(index=False):
        policy = str(schedule.schedule_policy)
        sample_id = int(schedule.schedule_sample_id)
        schedule_attempts = attempts[
            attempts["schedule_policy"].astype(str).eq(policy)
            & attempts["schedule_sample_id"].astype(int).eq(sample_id)
        ]
        for endpoint in preserved.itertuples(index=False):
            endpoint_id = str(endpoint.frozen_endpoint_id)
            variant_id = str(endpoint.variant_id)
            endpoint_attempts = schedule_attempts[
                schedule_attempts["frozen_endpoint_id"].astype(str).eq(endpoint_id)
                & schedule_attempts["is_structural_target_hit"].astype(bool)
            ].sort_values("attempt_index")
            eligible_candidates = eligible_by_endpoint.get(endpoint_id, pd.DataFrame())
            baseline_row = baseline_lookup.loc[(variant_id, endpoint_id)]
            first_hit = int(endpoint_attempts["attempt_index"].iloc[0]) if len(endpoint_attempts) else None
            first_candidate = str(endpoint_attempts["handle_candidate_id"].iloc[0]) if len(endpoint_attempts) else ""
            rows.append(
                {
                    "schedule_policy": policy,
                    "schedule_sample_id": sample_id,
                    "variant_id": variant_id,
                    "mechanism_family": str(endpoint.mechanism_family),
                    "mechanism_state": str(endpoint.mechanism_state),
                    "frozen_endpoint_id": endpoint_id,
                    "endpoint_signature_id": str(endpoint.endpoint_signature_id),
                    "mechanism_read": str(endpoint.mechanism_read),
                    "seed_count": int(endpoint.seed_count),
                    "baseline_first_hit_p75": float(baseline_row["baseline_first_hit_p75"]),
                    "structural_target_eligible": bool(len(eligible_candidates)),
                    "joint_structural_target_eligible": bool(
                        len(eligible_candidates)
                        and eligible_candidates["is_joint_candidate"].astype(bool).any()
                    ),
                    "eligible_candidate_count": int(len(eligible_candidates)),
                    "eligible_joint_candidate_count": int(
                        eligible_candidates["is_joint_candidate"].astype(bool).sum()
                    )
                    if len(eligible_candidates)
                    else 0,
                    "structural_target_hit": bool(len(endpoint_attempts)),
                    "first_structural_target_attempt": first_hit,
                    "first_structural_target_candidate": first_candidate,
                    "structural_target_beats_restart_p75": bool(
                        first_hit is not None and first_hit <= float(baseline_row["baseline_first_hit_p75"])
                    ),
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["schedule_policy", "schedule_sample_id", "variant_id", "frozen_endpoint_id"]))


def _policy_summary(discovery: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in discovery.groupby(["schedule_policy", "schedule_sample_id"], sort=True):
        policy, sample_id = keys
        eligible = group[group["structural_target_eligible"].astype(bool)]
        joint_eligible = group[group["joint_structural_target_eligible"].astype(bool)]
        schedule_attempts = attempts[
            attempts["schedule_policy"].astype(str).eq(str(policy))
            & attempts["schedule_sample_id"].astype(int).eq(int(sample_id))
        ]
        control_positive_attempts = schedule_attempts[
            schedule_attempts["mechanism_state"].astype(str).str.contains("control")
            & schedule_attempts["target_claim_allowed"].astype(bool)
        ]
        first_joint_attempts = schedule_attempts[
            schedule_attempts["is_joint_candidate"].astype(bool)
            & schedule_attempts["variant_id"].astype(str).eq("df_two_pair")
        ]["attempt_index"]
        rows.append(
            {
                "schedule_policy": str(policy),
                "schedule_sample_id": int(sample_id),
                "eligible_endpoint_count": int(len(eligible)),
                "structural_hit_count": int(eligible["structural_target_hit"].astype(bool).sum()),
                "structural_recall": float(eligible["structural_target_hit"].astype(bool).mean()) if len(eligible) else 0.0,
                "structural_beats_restart_p75_count": int(
                    eligible["structural_target_beats_restart_p75"].astype(bool).sum()
                ),
                "joint_eligible_endpoint_count": int(len(joint_eligible)),
                "joint_structural_hit_count": int(joint_eligible["structural_target_hit"].astype(bool).sum()),
                "joint_structural_recall": float(joint_eligible["structural_target_hit"].astype(bool).mean())
                if len(joint_eligible)
                else 0.0,
                "joint_beats_restart_p75_count": int(
                    joint_eligible["structural_target_beats_restart_p75"].astype(bool).sum()
                ),
                "not_targeted_endpoint_count": int((~group["structural_target_eligible"].astype(bool)).sum()),
                "control_positive_attempt_count": int(len(control_positive_attempts)),
                "df_two_pair_first_joint_attempt": int(first_joint_attempts.min()) if len(first_joint_attempts) else None,
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["schedule_policy", "schedule_sample_id"]))


def _variant_policy_summary(discovery: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in discovery.groupby(["schedule_policy", "schedule_sample_id", "variant_id"], sort=True):
        policy, sample_id, variant_id = keys
        eligible = group[group["structural_target_eligible"].astype(bool)]
        joint_eligible = group[group["joint_structural_target_eligible"].astype(bool)]
        rows.append(
            {
                "schedule_policy": str(policy),
                "schedule_sample_id": int(sample_id),
                "variant_id": str(variant_id),
                "mechanism_family": str(group["mechanism_family"].iloc[0]),
                "mechanism_state": str(group["mechanism_state"].iloc[0]),
                "eligible_endpoint_count": int(len(eligible)),
                "structural_hit_count": int(eligible["structural_target_hit"].astype(bool).sum()),
                "structural_recall": float(eligible["structural_target_hit"].astype(bool).mean()) if len(eligible) else 0.0,
                "joint_eligible_endpoint_count": int(len(joint_eligible)),
                "joint_structural_hit_count": int(joint_eligible["structural_target_hit"].astype(bool).sum()),
                "joint_structural_recall": float(joint_eligible["structural_target_hit"].astype(bool).mean())
                if len(joint_eligible)
                else 0.0,
                "not_targeted_endpoint_count": int((~group["structural_target_eligible"].astype(bool)).sum()),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["schedule_policy", "schedule_sample_id", "variant_id"]))


def _gate_matrix(
    *,
    phase_lock_verified: bool,
    phase_lock_mismatches: list[str],
    policy_summary: pd.DataFrame,
    expected_eligible_endpoints: int,
    random_samples: int,
) -> pd.DataFrame:
    deterministic = policy_summary[
        policy_summary["schedule_policy"].astype(str).isin(
            [policy for policy in DETERMINISTIC_POLICIES if policy != "joint_suppressed_negative_control"]
        )
    ]
    random_rows = policy_summary[policy_summary["schedule_policy"].astype(str).eq(RANDOM_POLICY)]
    negative = policy_summary[
        policy_summary["schedule_policy"].astype(str).eq("joint_suppressed_negative_control")
    ]
    random_min_recall = float(random_rows["structural_recall"].min()) if len(random_rows) else 0.0
    random_min_joint_recall = float(random_rows["joint_structural_recall"].min()) if len(random_rows) else 0.0
    deterministic_min_recall = float(deterministic["structural_recall"].min()) if len(deterministic) else 0.0
    deterministic_min_joint_recall = float(deterministic["joint_structural_recall"].min()) if len(deterministic) else 0.0
    negative_joint_recall = float(negative["joint_structural_recall"].iloc[0]) if len(negative) else math.nan
    control_positive_attempts = int(policy_summary["control_positive_attempt_count"].astype(int).sum())
    rows = [
        {
            "gate_id": "S1_phase_lock_integrity",
            "gate_question": "Do P0-P4 inputs still match the v1.2 phase lock?",
            "status": "pass" if phase_lock_verified else "blocked_hash_mismatch",
            "evidence": "all locked hashes match" if phase_lock_verified else json.dumps(phase_lock_mismatches),
            "decision": "schedule robustness is interpretable only if pass",
        },
        {
            "gate_id": "S2_structural_denominator",
            "gate_question": "Is the structural target-eligible denominator the expected v1.2 surface?",
            "status": "pass" if expected_eligible_endpoints == 20 else "blocked_unexpected_denominator",
            "evidence": f"structural_target_eligible_endpoints={expected_eligible_endpoints}",
            "decision": "do not compare with v1.2 P8.1 if denominator drifts",
        },
        {
            "gate_id": "S3_deterministic_nonadversarial_recall",
            "gate_question": "Do deterministic nonadversarial schedules preserve all structural target hits?",
            "status": "pass" if deterministic_min_recall >= 1.0 and deterministic_min_joint_recall >= 1.0 else "caveat_order_sensitive_deterministic",
            "evidence": (
                f"deterministic_min_structural_recall={deterministic_min_recall}, "
                f"deterministic_min_joint_recall={deterministic_min_joint_recall}"
            ),
            "decision": "pre-endpoint joint rule is schedule-stable across named deterministic orders if pass",
        },
        {
            "gate_id": "S4_random_permutation_recall",
            "gate_question": "Do random within-variant candidate permutations preserve structural recall?",
            "status": "pass" if random_min_recall >= 1.0 and random_min_joint_recall >= 1.0 else "caveat_random_order_sensitive",
            "evidence": (
                f"random_samples={random_samples}, random_min_structural_recall={random_min_recall}, "
                f"random_min_joint_recall={random_min_joint_recall}"
            ),
            "decision": "order robustness can be claimed only for sampled random permutations if pass",
        },
        {
            "gate_id": "S5_joint_suppression_negative_control",
            "gate_question": "Does suppressing joint candidates damage joint-endpoint recall?",
            "status": "pass" if negative_joint_recall < 1.0 else "caveat_joint_candidates_not_necessary_under_negative_control",
            "evidence": f"joint_suppressed_joint_recall={negative_joint_recall}",
            "decision": "joint candidates have specific value if pass",
        },
        {
            "gate_id": "S6_control_positive_lock",
            "gate_question": "Do controls remain unable to make positive target claims under all schedules?",
            "status": "pass" if control_positive_attempts == 0 else "blocked_control_positive_attempts",
            "evidence": f"control_positive_attempt_count={control_positive_attempts}",
            "decision": "do not interpret robustness if controls can emit positive attempts",
        },
    ]
    return _with_claim_columns(pd.DataFrame(rows))


def _readiness(gates: pd.DataFrame) -> str:
    statuses = set(gates["status"].astype(str))
    if any(status.startswith("blocked") for status in statuses):
        return "blocked_fix_inputs_or_control_lock"
    if any(status.startswith("caveat") for status in statuses):
        return "caveated_schedule_order_sensitive"
    return "ready_for_v1_2_schedule_robustness_read"


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        table.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        table.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(table)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    gates: pd.DataFrame,
    policy_summary: pd.DataFrame,
    variant_policy_summary: pd.DataFrame,
) -> None:
    random_rows = policy_summary[policy_summary["schedule_policy"].astype(str).eq(RANDOM_POLICY)]
    lines = [
        "# Tiny CPM Mechanism Variant v1.2 Schedule Robustness",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- p5_p8_dir: `{summary['p5_p8_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- readiness: `{summary['schedule_robustness_readiness']}`",
        f"- schedule_count: `{summary['schedule_count']}`",
        f"- structural_target_eligible_endpoint_count: `{summary['structural_target_eligible_endpoint_count']}`",
        f"- nonadversarial_min_structural_recall: `{summary['nonadversarial_min_structural_recall']}`",
        f"- random_min_structural_recall: `{summary['random_min_structural_recall']}`",
        f"- random_min_joint_recall: `{summary['random_min_joint_recall']}`",
        f"- joint_suppressed_joint_recall: `{summary['joint_suppressed_joint_recall']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates, ["gate_id", "status", "evidence", "decision"], max_rows=12),
        "",
        "## Policy Summary",
        "",
        _markdown_table(
            policy_summary,
            [
                "schedule_policy",
                "schedule_sample_id",
                "structural_recall",
                "joint_structural_recall",
                "structural_beats_restart_p75_count",
                "joint_beats_restart_p75_count",
                "df_two_pair_first_joint_attempt",
            ],
            max_rows=20,
        ),
        "",
        "## Random Policy Distribution",
        "",
        _markdown_table(
            random_rows.describe(include="all").reset_index().rename(columns={"index": "stat"}),
            ["stat", "structural_recall", "joint_structural_recall", "structural_beats_restart_p75_count"],
            max_rows=12,
        )
        if len(random_rows)
        else "_No random rows._",
        "",
        "## Variant Policy Summary",
        "",
        _markdown_table(
            variant_policy_summary[
                variant_policy_summary["variant_id"].astype(str).isin(["df_two_pair", "df_one_pair"])
            ],
            [
                "schedule_policy",
                "schedule_sample_id",
                "variant_id",
                "eligible_endpoint_count",
                "structural_hit_count",
                "joint_eligible_endpoint_count",
                "joint_structural_hit_count",
                "not_targeted_endpoint_count",
            ],
            max_rows=32,
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This tests order sensitivity of the pre-endpoint v1.2 candidate registry.",
        "- `joint_suppressed_negative_control` is an intentional mechanism negative control, not a competing schedule.",
        "- Passing this diagnostic does not open route/pathway, wall, quality/cost, NanoClustering, or algorithm claims.",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_schedule_robustness(
    *,
    input_dir: Path,
    p5_p8_dir: Path,
    output_dir: Path,
    max_budget: int,
    random_samples: int,
    rng_seed: int,
    n_iterations: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    if summary_path.exists() and not force:
        raise FileExistsError(f"{_rel(summary_path)} already exists. Use --force to regenerate.")

    inputs = _load_inputs(input_dir, p5_p8_dir)
    variants = _variant_inputs(inputs["manifest"], inputs["edges"], inputs["roles"])
    attempts, eligible = _run_attempts(
        variants=variants,
        registry=inputs["registry"],
        manifest=inputs["frozen_manifest"],
        roles=inputs["roles"],
        max_budget=max_budget,
        random_samples=random_samples,
        rng_seed=rng_seed,
        n_iterations=n_iterations,
    )
    discovery = _endpoint_discovery(
        attempts=attempts,
        manifest=inputs["frozen_manifest"],
        baseline=inputs["baseline"],
        eligible=eligible,
    )
    policy_summary = _policy_summary(discovery, attempts)
    variant_policy_summary = _variant_policy_summary(discovery)
    expected_eligible_endpoints = int(eligible["frozen_endpoint_id"].nunique()) if len(eligible) else 0
    gates = _gate_matrix(
        phase_lock_verified=inputs["phase_lock_verified"],
        phase_lock_mismatches=inputs["phase_lock_mismatches"],
        policy_summary=policy_summary,
        expected_eligible_endpoints=expected_eligible_endpoints,
        random_samples=random_samples,
    )
    readiness = _readiness(gates)

    nonadversarial = policy_summary[
        ~policy_summary["schedule_policy"].astype(str).isin(["joint_suppressed_negative_control"])
    ]
    random_rows = policy_summary[policy_summary["schedule_policy"].astype(str).eq(RANDOM_POLICY)]
    joint_suppressed = policy_summary[
        policy_summary["schedule_policy"].astype(str).eq("joint_suppressed_negative_control")
    ]
    config = {
        "input_dir": _rel(input_dir),
        "p5_p8_dir": _rel(p5_p8_dir),
        "output_dir": _rel(output_dir),
        "phase_lock_hash": inputs["phase_lock"].get("phase_lock_hash"),
        "phase_lock_verified": inputs["phase_lock_verified"],
        "phase_lock_mismatches": inputs["phase_lock_mismatches"],
        "max_budget": max_budget,
        "random_samples": random_samples,
        "rng_seed": rng_seed,
        "n_iterations": n_iterations,
        "deterministic_policies": list(DETERMINISTIC_POLICIES),
        "random_policy": RANDOM_POLICY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    output_dir.joinpath(CONFIG_JSON).write_text(json.dumps(_json_safe(config), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(attempts, output_dir / SCHEDULE_ATTEMPTS_CSV)
    _write_csv(discovery, output_dir / ENDPOINT_DISCOVERY_CSV)
    _write_csv(policy_summary, output_dir / POLICY_SUMMARY_CSV)
    _write_csv(variant_policy_summary, output_dir / VARIANT_POLICY_SUMMARY_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)

    summary = {
        "input_dir": _rel(input_dir),
        "p5_p8_dir": _rel(p5_p8_dir),
        "output_dir": _rel(output_dir),
        "phase_lock_hash": inputs["phase_lock"].get("phase_lock_hash"),
        "phase_lock_verified": inputs["phase_lock_verified"],
        "schedule_count": int(policy_summary[["schedule_policy", "schedule_sample_id"]].drop_duplicates().shape[0]),
        "attempt_row_count": int(len(attempts)),
        "endpoint_discovery_row_count": int(len(discovery)),
        "structural_target_eligible_endpoint_count": expected_eligible_endpoints,
        "nonadversarial_min_structural_recall": float(nonadversarial["structural_recall"].min())
        if len(nonadversarial)
        else 0.0,
        "nonadversarial_min_joint_recall": float(nonadversarial["joint_structural_recall"].min())
        if len(nonadversarial)
        else 0.0,
        "random_min_structural_recall": float(random_rows["structural_recall"].min()) if len(random_rows) else 0.0,
        "random_min_joint_recall": float(random_rows["joint_structural_recall"].min()) if len(random_rows) else 0.0,
        "joint_suppressed_joint_recall": float(joint_suppressed["joint_structural_recall"].iloc[0])
        if len(joint_suppressed)
        else math.nan,
        "control_positive_attempt_count": int(policy_summary["control_positive_attempt_count"].astype(int).sum()),
        "schedule_robustness_readiness": readiness,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            SCHEDULE_ATTEMPTS_CSV,
            ENDPOINT_DISCOVERY_CSV,
            POLICY_SUMMARY_CSV,
            VARIANT_POLICY_SUMMARY_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        output_dir=output_dir,
        summary=summary,
        gates=gates,
        policy_summary=policy_summary,
        variant_policy_summary=variant_policy_summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress Stress-4 v1.2 joint weak-pair evidence under alternative schedule orders."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--p5-p8-dir", type=Path, default=DEFAULT_P5_P8_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--random-samples", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=31051984)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_schedule_robustness(
        input_dir=args.input_dir,
        p5_p8_dir=args.p5_p8_dir,
        output_dir=args.output_dir,
        max_budget=args.max_budget,
        random_samples=args.random_samples,
        rng_seed=args.rng_seed,
        n_iterations=args.n_iterations,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
