#!/usr/bin/env python3
"""Ablate blind graph-rule handle types on tiny CPM demos.

This is Stress 3 for the Track C tiny-demo method surface. It reuses the
materialized blind-rule candidate registry and tests whether the successful
blind-rule signal localizes to the intended mechanism families: boundary-core
handles for absorption/balanced hard endpoints and weak-pair tail-split handles
for diffuse hard endpoints.

The script reports both compacted and slot-preserving schedules. In the
slot-preserving schedule, removed handle types leave explicit no-op slots so
schedule compression cannot masquerade as mechanism attribution.

This remains a tiny-demo mechanism-attribution diagnostic. It is not a
route/pathway trace, a wall promotion, a quality/cost claim, a NanoClustering
generality claim, or an algorithm-level claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
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
from run_leiden_cpm_tiny_handle_method_probe import _initial_membership


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_BASELINE_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_demo_seed_sweep_20260531"
DEFAULT_BLIND_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_blind_rule_handle_probe_v1_20260531"
DEFAULT_REPLAY_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_endpoint_replay_v1_20260531"
DEFAULT_ORDER_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_coverage_order_robustness_v1_20260531"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_blind_rule_ablation_v1_20260531"

FROZEN_ENDPOINT_MANIFEST_CSV = "leiden_cpm_tiny_demo_frozen_endpoint_manifest.csv"
BASELINE_DISCOVERY_CURVE_CSV = "leiden_cpm_tiny_demo_discovery_curve.csv"
BLIND_CANDIDATE_REGISTRY_CSV = "tiny_cpm_blind_rule_candidate_registry.csv"
BLIND_ATTEMPTS_CSV = "tiny_cpm_blind_rule_attempts.csv"
MISSING_DIAGNOSIS_CSV = "tiny_cpm_missing_endpoint_replay_diagnosis.csv"
ORDER_ATTEMPTS_CSV = "tiny_cpm_coverage_order_attempts.csv"

ABLATION_REGISTRY_CSV = "tiny_cpm_blind_rule_ablation_registry.csv"
ABLATION_SCHEDULE_CSV = "tiny_cpm_blind_rule_ablation_schedule.csv"
ABLATION_ATTEMPTS_CSV = "tiny_cpm_blind_rule_ablation_attempts.csv"
ABLATION_DISCOVERY_CSV = "tiny_cpm_blind_rule_ablation_discovery.csv"
ABLATION_FIRST_HITS_CSV = "tiny_cpm_blind_rule_ablation_first_hits.csv"
ABLATION_ATTRIBUTION_CSV = "tiny_cpm_blind_rule_ablation_attribution_matrix.csv"
ABLATION_SEED_STABILITY_CSV = "tiny_cpm_blind_rule_ablation_candidate_seed_stability.csv"
GATE_MATRIX_CSV = "tiny_cpm_blind_rule_ablation_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_blind_rule_ablation_summary.json"
CONFIG_JSON = "tiny_cpm_blind_rule_ablation_config.json"
REPORT_MD = "tiny_cpm_blind_rule_ablation_report.md"

DEFAULT_BUDGETS = (1, 2, 3, 5, 10, 20)
SCHEDULE_MODES = ("compacted", "slot_preserving")
CLAIM_BOUNDARY = (
    "Tiny CPM blind-rule handle-type ablation only; subsets are built from "
    "the materialized blind-rule candidate registry, no route execution, no "
    "wall/pathway promotion, no basin-quality claim, no cost claim, no "
    "NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_blind_rule_ablation_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "candidate_ablation_not_algorithm_claim"

CONTROL_TYPES = {
    "blind_small_module_control_initialization",
    "blind_middle_control_initialization",
    "blind_weak_module_control_initialization",
}
BOUNDARY_CORE_TYPES = {
    "blind_small_module_boundary_core_initialization",
    "blind_middle_boundary_core_initialization",
}
WEAK_PAIR_TYPES = {"blind_weak_pair_tail_split_initialization"}
WEAK_TOP_TYPES = {"blind_weak_top_host_initialization"}
BRIDGE_TYPES = {"blind_bridge_contact_initialization"}


@dataclass(frozen=True)
class CandidateRow:
    family: str
    handle_candidate_id: str
    handle_type: str
    target_mechanism_read: str
    groups: tuple[tuple[str, ...], ...]
    handle_node_count: int
    initial_group_count: int
    ambiguous_node_count: int
    host_candidate: str


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


def _load_candidates(registry: pd.DataFrame) -> dict[str, CandidateRow]:
    candidates: dict[str, CandidateRow] = {}
    for row in registry.itertuples(index=False):
        groups = tuple(tuple(group) for group in json.loads(str(row.initial_groups)))
        ambiguous_nodes = json.loads(str(row.ambiguous_module_nodes))
        candidates[str(row.handle_candidate_id)] = CandidateRow(
            family=str(row.family),
            handle_candidate_id=str(row.handle_candidate_id),
            handle_type=str(row.handle_type),
            target_mechanism_read=str(row.target_mechanism_read),
            groups=groups,
            handle_node_count=int(row.handle_node_count),
            initial_group_count=int(row.initial_group_count),
            ambiguous_node_count=len(ambiguous_nodes),
            host_candidate=str(row.host_candidate),
        )
    return candidates


def _canonical_order(
    *,
    registry: pd.DataFrame,
    blind_dir: Path,
) -> dict[str, list[str]]:
    attempts_path = blind_dir / BLIND_ATTEMPTS_CSV
    if attempts_path.exists():
        attempts = pd.read_csv(attempts_path)
        first = (
            attempts.groupby(["family", "handle_candidate_id"], as_index=False)
            .agg(first_attempt=("attempt_index", "min"))
            .sort_values(["family", "first_attempt", "handle_candidate_id"])
        )
        return {
            str(family): group["handle_candidate_id"].astype(str).tolist()
            for family, group in first.groupby("family", sort=True)
        }
    return {
        str(family): group["handle_candidate_id"].astype(str).tolist()
        for family, group in registry.sort_values(["family", "handle_candidate_id"]).groupby("family", sort=True)
    }


def _type_sets(registry: pd.DataFrame) -> dict[str, set[str]]:
    by_type: dict[str, set[str]] = {}
    for handle_type, group in registry.groupby("handle_type", sort=True):
        by_type[str(handle_type)] = set(group["handle_candidate_id"].astype(str).tolist())
    return by_type


def _subset_specs(registry: pd.DataFrame) -> list[dict[str, Any]]:
    all_ids = set(registry["handle_candidate_id"].astype(str).tolist())
    by_type = _type_sets(registry)
    controls = set().union(*(by_type.get(handle_type, set()) for handle_type in CONTROL_TYPES))
    boundary_core = set().union(*(by_type.get(handle_type, set()) for handle_type in BOUNDARY_CORE_TYPES))
    weak_pair = set().union(*(by_type.get(handle_type, set()) for handle_type in WEAK_PAIR_TYPES))
    weak_top = set().union(*(by_type.get(handle_type, set()) for handle_type in WEAK_TOP_TYPES))
    bridge = set().union(*(by_type.get(handle_type, set()) for handle_type in BRIDGE_TYPES))
    specs: list[dict[str, Any]] = [
        {
            "ablation_subset_id": "all_blind_rules",
            "included_ids": all_ids,
            "target_scope": "global_context",
            "reason": "Full materialized blind-rule registry.",
        },
        {
            "ablation_subset_id": "all_blind_rules_no_controls",
            "included_ids": all_ids - controls,
            "target_scope": "global_context_without_controls",
            "reason": "Remove graph-rule control handles to test whether controls carry gains.",
        },
        {
            "ablation_subset_id": "boundary_core_only",
            "included_ids": boundary_core,
            "target_scope": "absorption_balanced_hard",
            "reason": "Boundary-core mechanism attribution subset.",
        },
        {
            "ablation_subset_id": "weak_pair_tail_split_only",
            "included_ids": weak_pair,
            "target_scope": "diffuse_hard",
            "reason": "Weak-pair tail-split mechanism attribution subset.",
        },
        {
            "ablation_subset_id": "weak_top_host_only",
            "included_ids": weak_top,
            "target_scope": "control_for_diffuse_top_host",
            "reason": "Check whether weak top-host alignment alone explains diffuse hard endpoints.",
        },
        {
            "ablation_subset_id": "bridge_contact_only",
            "included_ids": bridge,
            "target_scope": "near_tie",
            "reason": "Bridge-contact mechanism and negative control for non-near-tie hard endpoints.",
        },
        {
            "ablation_subset_id": "controls_only",
            "included_ids": controls,
            "target_scope": "negative_control",
            "reason": "Control handles should not explain hard endpoint gains.",
        },
        {
            "ablation_subset_id": "boundary_core_plus_weak_pair",
            "included_ids": boundary_core | weak_pair,
            "target_scope": "absorption_balanced_and_diffuse_hard",
            "reason": "Interaction subset for the two proposed responsible handle families.",
        },
    ]
    for handle_type, ids in sorted(by_type.items()):
        specs.append(
            {
                "ablation_subset_id": f"drop_{handle_type}",
                "included_ids": all_ids - ids,
                "target_scope": f"dropout_{handle_type}",
                "reason": f"Dropout subset removing {handle_type}.",
            }
        )
    return specs


def _ablation_registry(
    *,
    specs: list[dict[str, Any]],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    all_ids = set(registry["handle_candidate_id"].astype(str).tolist())
    handle_type_by_id = dict(
        zip(
            registry["handle_candidate_id"].astype(str),
            registry["handle_type"].astype(str),
        )
    )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        included = set(spec["included_ids"])
        excluded = all_ids - included
        rows.append(
            {
                "ablation_subset_id": str(spec["ablation_subset_id"]),
                "target_scope": str(spec["target_scope"]),
                "included_handle_count": int(len(included)),
                "excluded_handle_count": int(len(excluded)),
                "included_handle_ids": json.dumps(sorted(included)),
                "excluded_handle_ids": json.dumps(sorted(excluded)),
                "included_handle_types": json.dumps(
                    sorted({handle_type_by_id[handle_id] for handle_id in included})
                ),
                "excluded_handle_types": json.dumps(
                    sorted({handle_type_by_id[handle_id] for handle_id in excluded})
                ),
                "subset_reason": str(spec["reason"]),
                "construction_input_policy": "materialized_blind_rule_candidate_registry_only",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values("ablation_subset_id"))


def _schedule_rows(
    *,
    specs: list[dict[str, Any]],
    canonical_order: dict[str, list[str]],
    candidates: dict[str, CandidateRow],
    max_budget: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        subset_id = str(spec["ablation_subset_id"])
        included = set(spec["included_ids"])
        for family, full_order in sorted(canonical_order.items()):
            subset_order = [handle_id for handle_id in full_order if handle_id in included]
            for schedule_mode in SCHEDULE_MODES:
                for attempt_index in range(1, max_budget + 1):
                    if schedule_mode == "compacted":
                        if subset_order:
                            handle_id = subset_order[(attempt_index - 1) % len(subset_order)]
                            is_noop = False
                            noop_reason = ""
                        else:
                            handle_id = None
                            is_noop = True
                            noop_reason = "empty_subset"
                    else:
                        handle_id = full_order[(attempt_index - 1) % len(full_order)]
                        if handle_id not in included:
                            handle_id = None
                            is_noop = True
                            noop_reason = "slot_preserved_for_removed_handle_type"
                        else:
                            is_noop = False
                            noop_reason = ""
                    candidate = candidates[handle_id] if handle_id is not None else None
                    rows.append(
                        {
                            "ablation_subset_id": subset_id,
                            "target_scope": str(spec["target_scope"]),
                            "schedule_mode": schedule_mode,
                            "family": family,
                            "attempt_index": int(attempt_index),
                            "handle_candidate_id": handle_id,
                            "handle_type": candidate.handle_type if candidate else None,
                            "is_noop": bool(is_noop),
                            "noop_reason": noop_reason,
                        }
                    )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["ablation_subset_id", "schedule_mode", "family", "attempt_index"]))


def _manifest_by_signature(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(family): group.set_index("endpoint_signature_id", drop=False)
        for family, group in manifest.groupby("family", sort=True)
    }


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


def _run_cache(
    *,
    candidates: dict[str, CandidateRow],
    manifest: pd.DataFrame,
    max_seed: int,
    n_iterations: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    cases = {case.family: case for case in _graph_cases()}
    manifest_by_family = _manifest_by_signature(manifest)
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for handle_id, candidate in sorted(candidates.items()):
        case = cases[candidate.family]
        graph = case.builder()
        node_names = list(map(str, graph.vs["name"]))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        initial = _initial_membership(node_names, candidate.groups)
        for method_seed in range(max_seed):
            result = runner.run(case.gamma, seed=method_seed, initial_membership=initial)
            membership = list(map(int, result.membership))
            groups = _canonical_groups(graph, membership)
            signature_id = _signature_id(groups)
            frozen_endpoint_id: str | None = None
            baseline_role: str | None = None
            if signature_id in manifest_by_family[candidate.family].index:
                matched = manifest_by_family[candidate.family].loc[signature_id]
                frozen_endpoint_id = str(matched["frozen_endpoint_id"])
                baseline_role = str(matched["baseline_role"])
            result_mechanism = _classify_mechanism(candidate.family, graph, membership)
            cache[(handle_id, method_seed)] = {
                "method_seed": int(method_seed),
                "gamma": float(case.gamma),
                "cluster_count": int(result.cluster_count),
                "quality": float(result.quality),
                "endpoint_signature_id": signature_id,
                "endpoint_signature": json.dumps(groups, sort_keys=True),
                "frozen_endpoint_id": frozen_endpoint_id,
                "baseline_role": baseline_role,
                "is_frozen_endpoint_hit": frozen_endpoint_id is not None,
                "result_mechanism_read": result_mechanism,
                "is_target_hit": bool(
                    frozen_endpoint_id is not None
                    and result_mechanism == candidate.target_mechanism_read
                ),
            }
    return cache


def _attempt_rows(
    *,
    schedule: pd.DataFrame,
    candidates: dict[str, CandidateRow],
    cache: dict[tuple[str, int], dict[str, Any]],
) -> pd.DataFrame:
    seen: dict[tuple[str, str, str, str], int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for row in schedule.itertuples(index=False):
        handle_id = getattr(row, "handle_candidate_id")
        base = {
            "ablation_subset_id": str(row.ablation_subset_id),
            "target_scope": str(row.target_scope),
            "schedule_mode": str(row.schedule_mode),
            "family": str(row.family),
            "attempt_index": int(row.attempt_index),
            "handle_candidate_id": handle_id if isinstance(handle_id, str) else None,
            "handle_type": getattr(row, "handle_type") if isinstance(getattr(row, "handle_type"), str) else None,
            "is_noop": bool(row.is_noop),
            "noop_reason": str(row.noop_reason) if isinstance(row.noop_reason, str) else "",
        }
        if bool(row.is_noop):
            rows.append(
                {
                    **base,
                    "method_seed": None,
                    "gamma": None,
                    "cluster_count": None,
                    "quality": None,
                    "endpoint_signature_id": None,
                    "endpoint_signature": None,
                    "frozen_endpoint_id": None,
                    "baseline_role": None,
                    "is_frozen_endpoint_hit": False,
                    "result_mechanism_read": None,
                    "is_target_hit": False,
                    "handle_node_count": 0,
                    "initial_group_count": 0,
                    "ambiguous_node_count": 0,
                }
            )
            continue
        candidate = candidates[str(handle_id)]
        seen_key = (
            str(row.ablation_subset_id),
            str(row.schedule_mode),
            str(row.family),
            str(handle_id),
        )
        method_seed = seen[seen_key]
        seen[seen_key] += 1
        cached = cache[(str(handle_id), method_seed)]
        rows.append(
            {
                **base,
                **cached,
                "handle_node_count": candidate.handle_node_count,
                "initial_group_count": candidate.initial_group_count,
                "ambiguous_node_count": candidate.ambiguous_node_count,
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["ablation_subset_id", "schedule_mode", "family", "attempt_index"]))


def _discovery(
    *,
    attempts: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    budgets: list[int],
) -> pd.DataFrame:
    sets = _manifest_sets(manifest)
    rows: list[dict[str, Any]] = []
    group_cols = ["ablation_subset_id", "target_scope", "schedule_mode", "family"]
    for keys, group in attempts.groupby(group_cols, sort=True):
        subset_id, target_scope, schedule_mode, family = keys
        family_sets = sets[str(family)]
        recurrent_ids = family_sets["recurrent"]
        total_ids = family_sets["total"]
        top_quality_ids = family_sets["top_quality"]
        for budget in budgets:
            prefix = group[group["attempt_index"].le(budget)]
            found = set(prefix["endpoint_signature_id"].dropna().astype(str))
            frozen_found = found & total_ids
            recall = float(len(frozen_found & recurrent_ids) / len(recurrent_ids))
            baseline_row = baseline_curve[
                baseline_curve["family"].eq(family) & baseline_curve["budget"].eq(budget)
            ]
            baseline_recall = (
                float(baseline_row["recurrent_endpoint_recall_mean"].iloc[0])
                if not baseline_row.empty
                else math.nan
            )
            rows.append(
                {
                    "ablation_subset_id": str(subset_id),
                    "target_scope": str(target_scope),
                    "schedule_mode": str(schedule_mode),
                    "family": str(family),
                    "budget": int(budget),
                    "method_distinct_endpoint_count": int(len(frozen_found)),
                    "method_new_endpoint_count": int(len(found - total_ids)),
                    "method_recurrent_endpoint_recall": recall,
                    "method_all_recurrent_endpoint_hit": bool(recurrent_ids.issubset(frozen_found)),
                    "method_top_quality_endpoint_hit": bool(frozen_found & top_quality_ids),
                    "method_target_hit_count": int(prefix["is_target_hit"].astype(bool).sum()),
                    "actual_attempt_count": int(prefix["is_noop"].eq(False).sum()),
                    "noop_count": int(prefix["is_noop"].eq(True).sum()),
                    "baseline_recurrent_endpoint_recall_mean": baseline_recall,
                    "delta_recurrent_endpoint_recall": float(recall - baseline_recall)
                    if math.isfinite(baseline_recall)
                    else math.nan,
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["ablation_subset_id", "schedule_mode", "family", "budget"]))


def _hard_endpoints(
    *,
    replay_dir: Path,
) -> pd.DataFrame:
    path = replay_dir / "tiny_cpm_missing_endpoint_replay_diagnosis.csv"
    hard = pd.read_csv(path)
    hard = hard[hard["diagnosis"].astype(str).str.contains("stable_endpoint", na=False)].copy()
    def target_class(row: pd.Series) -> str:
        family = str(row["family"])
        if family == "absorption_triad":
            return "boundary_core_absorption"
        if family == "balanced_split_module":
            return "boundary_core_balanced"
        if family == "diffuse_fragment_star":
            return "weak_pair_diffuse"
        return "other"
    hard["target_endpoint_class"] = hard.apply(target_class, axis=1)
    return hard


def _adversarial_first_hits(order_dir: Path) -> pd.DataFrame:
    path = order_dir / ORDER_ATTEMPTS_CSV
    attempts = pd.read_csv(path)
    rows = attempts[
        attempts["order_policy"].eq("adversarial_delayed_coverage")
        & attempts["frozen_endpoint_id"].notna()
        & attempts["baseline_role"].eq("recurrent_baseline_endpoint")
    ]
    return (
        rows.groupby(["family", "frozen_endpoint_id"], as_index=False)
        .agg(adversarial_first_attempt=("attempt_index", "min"))
        .sort_values(["family", "frozen_endpoint_id"])
    )


def _first_hits(
    *,
    attempts: pd.DataFrame,
    manifest: pd.DataFrame,
    hard: pd.DataFrame,
    adversarial: pd.DataFrame,
) -> pd.DataFrame:
    recurrent = manifest[manifest["is_recurrent_endpoint"].astype(bool)].copy()
    first = (
        attempts[
            attempts["frozen_endpoint_id"].notna()
            & attempts["baseline_role"].eq("recurrent_baseline_endpoint")
        ]
        .groupby(
            [
                "ablation_subset_id",
                "target_scope",
                "schedule_mode",
                "family",
                "frozen_endpoint_id",
            ],
            as_index=False,
        )
        .agg(
            first_attempt=("attempt_index", "min"),
            hit_candidate_ids=("handle_candidate_id", lambda values: ";".join(sorted(set(map(str, values))))),
            hit_handle_types=("handle_type", lambda values: ";".join(sorted(set(map(str, values))))),
        )
    )
    grid_keys = attempts[["ablation_subset_id", "target_scope", "schedule_mode"]].drop_duplicates()
    endpoint_grid = recurrent[["family", "frozen_endpoint_id", "mechanism_read", "seed_count"]].drop_duplicates()
    grid = grid_keys.merge(endpoint_grid, how="cross")
    rows = grid.merge(
        first,
        on=["ablation_subset_id", "target_scope", "schedule_mode", "family", "frozen_endpoint_id"],
        how="left",
    )
    hard_flags = hard[["family", "frozen_endpoint_id", "target_endpoint_class"]].copy()
    rows = rows.merge(hard_flags, on=["family", "frozen_endpoint_id"], how="left")
    rows["is_hard_endpoint"] = rows["target_endpoint_class"].notna()
    rows = rows.merge(adversarial, on=["family", "frozen_endpoint_id"], how="left")
    rows["beats_adversarial_first_hit"] = (
        rows["first_attempt"].notna()
        & rows["adversarial_first_attempt"].notna()
        & rows["first_attempt"].lt(rows["adversarial_first_attempt"])
    )
    return _with_claim_columns(rows.sort_values(["ablation_subset_id", "schedule_mode", "family", "frozen_endpoint_id"]))


def _target_scope_score(first_hits: pd.DataFrame) -> pd.DataFrame:
    target_map = {
        "boundary_core_only": {"boundary_core_absorption", "boundary_core_balanced"},
        "weak_pair_tail_split_only": {"weak_pair_diffuse"},
        "bridge_contact_only": set(),
        "controls_only": {"boundary_core_absorption", "boundary_core_balanced", "weak_pair_diffuse"},
        "weak_top_host_only": {"weak_pair_diffuse"},
        "boundary_core_plus_weak_pair": {
            "boundary_core_absorption",
            "boundary_core_balanced",
            "weak_pair_diffuse",
        },
        "all_blind_rules": {
            "boundary_core_absorption",
            "boundary_core_balanced",
            "weak_pair_diffuse",
        },
        "all_blind_rules_no_controls": {
            "boundary_core_absorption",
            "boundary_core_balanced",
            "weak_pair_diffuse",
        },
        "drop_blind_small_module_boundary_core_initialization": {"boundary_core_absorption"},
        "drop_blind_middle_boundary_core_initialization": {"boundary_core_balanced"},
        "drop_blind_weak_pair_tail_split_initialization": {"weak_pair_diffuse"},
    }
    rows: list[dict[str, Any]] = []
    for (subset_id, schedule_mode), group in first_hits.groupby(["ablation_subset_id", "schedule_mode"], sort=True):
        target_classes = target_map.get(str(subset_id), set())
        target_rows = group[group["target_endpoint_class"].isin(target_classes)].copy()
        if target_rows.empty:
            continue
        rows.append(
            {
                "ablation_subset_id": str(subset_id),
                "schedule_mode": str(schedule_mode),
                "target_classes": ";".join(sorted(target_classes)),
                "target_endpoint_count": int(len(target_rows)),
                "target_endpoint_hit_count": int(target_rows["first_attempt"].notna().sum()),
                "target_endpoint_hit_by_budget10_count": int(target_rows["first_attempt"].le(10).sum()),
                "target_endpoint_beats_adversarial_count": int(
                    target_rows["beats_adversarial_first_hit"].astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _attribution_matrix(
    *,
    first_hits: pd.DataFrame,
) -> pd.DataFrame:
    target_scores = _target_scope_score(first_hits)
    rows: list[dict[str, Any]] = []
    for row in target_scores.itertuples(index=False):
        rows.append(
            {
                "attribution_id": f"{row.ablation_subset_id}:{row.schedule_mode}",
                "ablation_subset_id": row.ablation_subset_id,
                "schedule_mode": row.schedule_mode,
                "target_classes": row.target_classes,
                "target_endpoint_count": int(row.target_endpoint_count),
                "target_endpoint_hit_count": int(row.target_endpoint_hit_count),
                "target_endpoint_hit_by_budget10_count": int(row.target_endpoint_hit_by_budget10_count),
                "target_endpoint_beats_adversarial_count": int(row.target_endpoint_beats_adversarial_count),
                "target_hit_rate": float(row.target_endpoint_hit_count / row.target_endpoint_count),
                "budget10_target_hit_rate": float(
                    row.target_endpoint_hit_by_budget10_count / row.target_endpoint_count
                ),
                "beats_adversarial_rate": float(
                    row.target_endpoint_beats_adversarial_count / row.target_endpoint_count
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["ablation_subset_id", "schedule_mode"]))


def _candidate_seed_stability(
    *,
    cache: dict[tuple[str, int], dict[str, Any]],
    candidates: dict[str, CandidateRow],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (handle_id, _seed), result in cache.items():
        by_candidate[handle_id].append(result)
    for handle_id, values in sorted(by_candidate.items()):
        candidate = candidates[handle_id]
        endpoint_counts = Counter(str(value["frozen_endpoint_id"]) for value in values)
        mechanism_counts = Counter(str(value["result_mechanism_read"]) for value in values)
        top_endpoint, top_endpoint_count = endpoint_counts.most_common(1)[0]
        rows.append(
            {
                "family": candidate.family,
                "handle_candidate_id": handle_id,
                "handle_type": candidate.handle_type,
                "target_mechanism_read": candidate.target_mechanism_read,
                "seed_count": int(len(values)),
                "unique_result_endpoint_count": int(len(endpoint_counts)),
                "dominant_result_endpoint_id": top_endpoint,
                "dominant_result_endpoint_rate": float(top_endpoint_count / len(values)),
                "target_hit_rate": float(
                    sum(bool(value["is_target_hit"]) for value in values) / len(values)
                ),
                "result_mechanism_reads": ";".join(sorted(mechanism_counts)),
                "handle_node_count": candidate.handle_node_count,
                "initial_group_count": candidate.initial_group_count,
                "ambiguous_node_count": candidate.ambiguous_node_count,
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "handle_type", "handle_candidate_id"]))


def _gate_matrix(
    *,
    ablation_registry: pd.DataFrame,
    schedule: pd.DataFrame,
    attribution: pd.DataFrame,
    first_hits: pd.DataFrame,
    seed_stability: pd.DataFrame,
) -> pd.DataFrame:
    subset_ids = set(ablation_registry["ablation_subset_id"].astype(str))
    required_subsets = {
        "all_blind_rules",
        "all_blind_rules_no_controls",
        "boundary_core_only",
        "weak_pair_tail_split_only",
        "weak_top_host_only",
        "bridge_contact_only",
        "controls_only",
        "boundary_core_plus_weak_pair",
    }
    schedule_modes = set(schedule["schedule_mode"].astype(str))
    def target_score(subset_id: str, mode: str) -> pd.Series | None:
        rows = attribution[
            attribution["ablation_subset_id"].eq(subset_id)
            & attribution["schedule_mode"].eq(mode)
        ]
        if rows.empty:
            return None
        return rows.iloc[0]
    boundary_pass = all(
        (target_score("boundary_core_only", mode) is not None)
        and int(target_score("boundary_core_only", mode)["target_endpoint_beats_adversarial_count"]) == 3
        for mode in SCHEDULE_MODES
    )
    weak_pair_pass = all(
        (target_score("weak_pair_tail_split_only", mode) is not None)
        and int(target_score("weak_pair_tail_split_only", mode)["target_endpoint_beats_adversarial_count"]) == 3
        for mode in SCHEDULE_MODES
    )
    controls = attribution[
        attribution["ablation_subset_id"].isin(["controls_only", "bridge_contact_only", "weak_top_host_only"])
    ]
    control_explains = bool(
        not controls.empty and controls["target_endpoint_beats_adversarial_count"].max() > 0
    )
    dropout_expectations = {
        "drop_blind_small_module_boundary_core_initialization": "boundary_core_absorption",
        "drop_blind_middle_boundary_core_initialization": "boundary_core_balanced",
        "drop_blind_weak_pair_tail_split_initialization": "weak_pair_diffuse",
    }
    dropout_damage_details: list[str] = []
    dropout_damage_pass = True
    for subset_id, target_class in dropout_expectations.items():
        for mode in SCHEDULE_MODES:
            score = target_score(subset_id, mode)
            if score is None:
                dropout_damage_details.append(f"{subset_id}:{mode}:missing")
                dropout_damage_pass = False
                continue
            hit_count = int(score["target_endpoint_hit_count"])
            target_count = int(score["target_endpoint_count"])
            class_matches = str(score["target_classes"]) == target_class
            damaged = class_matches and target_count > 0 and hit_count < target_count
            dropout_damage_details.append(f"{subset_id}:{mode}:{hit_count}/{target_count}")
            dropout_damage_pass = dropout_damage_pass and damaged
    exact_rows = first_hits[first_hits["is_hard_endpoint"].astype(bool)]
    exact_complete = exact_rows["hit_candidate_ids"].notna().any()
    responsible = seed_stability[
        seed_stability["handle_type"].isin(
            [
                "blind_small_module_boundary_core_initialization",
                "blind_middle_boundary_core_initialization",
                "blind_weak_pair_tail_split_initialization",
            ]
        )
    ]
    min_responsible_dominant_rate = (
        float(responsible["dominant_result_endpoint_rate"].min())
        if not responsible.empty
        else 0.0
    )
    rows = [
        {
            "gate_id": "A1_ablation_registry_integrity",
            "gate_question": "Were ablation subsets built from the materialized blind-rule registry?",
            "evidence": f"subset_count={len(subset_ids)}, required_missing={sorted(required_subsets - subset_ids)}",
            "status": "pass" if required_subsets.issubset(subset_ids) else "blocked_missing_required_subset",
            "decision": "use_as_ablation_surface_if_pass",
            "next_action": "inspect schedule normalization",
        },
        {
            "gate_id": "A2_baseline_controls",
            "gate_question": "Are required controls and dropouts present?",
            "evidence": f"dropout_subset_count={sum(subset.startswith('drop_') for subset in subset_ids)}",
            "status": "pass" if any(subset.startswith("drop_") for subset in subset_ids) else "blocked_missing_dropouts",
            "decision": "controls_available_if_pass",
            "next_action": "evaluate target-scoped attribution",
        },
        {
            "gate_id": "A3_mechanism_positive_attribution",
            "gate_question": "Do responsible subsets hit their hard endpoints before adversarial delay?",
            "evidence": f"boundary_core_pass={boundary_pass}, weak_pair_pass={weak_pair_pass}",
            "status": "pass" if boundary_pass and weak_pair_pass else "caveat_required",
            "decision": "positive_mechanism_localization_if_pass",
            "next_action": "check dropout damage",
        },
        {
            "gate_id": "A4_mechanism_negative_attribution",
            "gate_question": "Do handle-type removals localize coverage damage?",
            "evidence": "dropout_damage=" + ";".join(dropout_damage_details),
            "status": "pass" if dropout_damage_pass else "caveat_required",
            "decision": "negative_attribution_surface_available_if_pass",
            "next_action": "inspect endpoint-level dropout rows",
        },
        {
            "gate_id": "A5_control_non_sufficiency",
            "gate_question": "Do controls fail to explain hard endpoint gains?",
            "evidence": f"control_explains_hard_endpoint={control_explains}",
            "status": "pass" if not control_explains else "caveat_required",
            "decision": "responsible_handles_not_generic_controls_if_pass",
            "next_action": "inspect compacted versus slot-preserving schedules",
        },
        {
            "gate_id": "A6_budget_profile",
            "gate_question": "Are budget profiles available for both schedule modes?",
            "evidence": f"schedule_modes={sorted(schedule_modes)}",
            "status": "pass" if set(SCHEDULE_MODES).issubset(schedule_modes) else "blocked_missing_schedule_mode",
            "decision": "schedule_compression_confound_controlled_if_pass",
            "next_action": "record interaction accounting",
        },
        {
            "gate_id": "A7_interaction_accounting",
            "gate_question": "Is boundary-core plus weak-pair interaction explicitly measured?",
            "evidence": f"boundary_core_plus_weak_pair_present={'boundary_core_plus_weak_pair' in subset_ids}",
            "status": "pass" if "boundary_core_plus_weak_pair" in subset_ids else "blocked_missing_interaction_subset",
            "decision": "interaction_accounting_available_if_pass",
            "next_action": "inspect exact endpoint attribution",
        },
        {
            "gate_id": "A8_claim_gate",
            "gate_question": "Can this ablation open algorithm or wall/pathway claims?",
            "evidence": "tiny mechanism-attribution ablation only",
            "status": "closed_excluded_by_design",
            "decision": "keep_algorithm_wall_quality_cost_claims_closed",
            "next_action": "use only as precondition for mechanism variant panel",
        },
        {
            "gate_id": "A9_schedule_normalization",
            "gate_question": "Are compacted and slot-preserving schedules both reported?",
            "evidence": f"schedule_modes={sorted(schedule_modes)}",
            "status": "pass" if set(SCHEDULE_MODES).issubset(schedule_modes) else "blocked_missing_slot_preserving_schedule",
            "decision": "subset_size_confound_visible_if_pass",
            "next_action": "inspect first-hit differences by schedule",
        },
        {
            "gate_id": "A10_target_scope_scoring",
            "gate_question": "Are target-scoped attribution rows present?",
            "evidence": f"attribution_rows={len(attribution)}",
            "status": "pass" if not attribution.empty else "blocked_missing_target_scope_rows",
            "decision": "avoid_global_recall_only_if_pass",
            "next_action": "inspect exact endpoint attribution",
        },
        {
            "gate_id": "A11_exact_endpoint_attribution",
            "gate_question": "Can hard endpoint hits be traced to candidate ids?",
            "evidence": f"hard_endpoint_rows={len(exact_rows)}, exact_complete={exact_complete}",
            "status": "pass" if exact_complete else "blocked_no_candidate_id_trace",
            "decision": "endpoint_level_attribution_available_if_pass",
            "next_action": "inspect seed stability",
        },
        {
            "gate_id": "A12_seed_robustness",
            "gate_question": "Are responsible handle outcomes stable across method seeds?",
            "evidence": f"min_responsible_dominant_endpoint_rate={min_responsible_dominant_rate:.6f}",
            "status": "pass" if min_responsible_dominant_rate >= 0.8 else "caveat_required",
            "decision": "responsible_handles_seed_stable_if_pass",
            "next_action": "inspect intervention-size caveats",
        },
        {
            "gate_id": "A13_intervention_size_caveat",
            "gate_question": "Are intervention-size diagnostics recorded?",
            "evidence": f"seed_stability_rows={len(seed_stability)}",
            "status": "pass"
            if {"handle_node_count", "initial_group_count", "ambiguous_node_count"}.issubset(seed_stability.columns)
            else "blocked_missing_size_metrics",
            "decision": "do_not_claim_cost_without_size_context",
            "next_action": "keep cost claims closed",
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
    attribution: pd.DataFrame,
    first_hits: pd.DataFrame,
    discovery: pd.DataFrame,
) -> None:
    hard_hits = first_hits[first_hits["is_hard_endpoint"].astype(bool)].copy()
    key_discovery = discovery[
        discovery["budget"].isin([5, 10, 20])
        & discovery["ablation_subset_id"].isin(
            [
                "all_blind_rules",
                "boundary_core_only",
                "weak_pair_tail_split_only",
                "boundary_core_plus_weak_pair",
                "controls_only",
                "weak_top_host_only",
            ]
        )
    ]
    text = [
        "# Tiny CPM Blind-Rule Handle Ablation v1",
        "",
        f"- subset_count: `{summary['subset_count']}`",
        f"- schedule_rows: `{summary['schedule_rows']}`",
        f"- attempt_rows: `{summary['attempt_rows']}`",
        f"- attribution_rows: `{summary['attribution_rows']}`",
        f"- gate_status_counts: `{summary['gate_status_counts']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=20,
        ),
        "",
        "## Target-Scoped Attribution",
        "",
        _markdown_table(
            attribution,
            [
                "ablation_subset_id",
                "schedule_mode",
                "target_classes",
                "target_endpoint_hit_count",
                "target_endpoint_count",
                "target_endpoint_beats_adversarial_count",
                "budget10_target_hit_rate",
            ],
            max_rows=30,
        ),
        "",
        "## Hard Endpoint First Hits",
        "",
        _markdown_table(
            hard_hits,
            [
                "ablation_subset_id",
                "schedule_mode",
                "family",
                "frozen_endpoint_id",
                "first_attempt",
                "adversarial_first_attempt",
                "hit_candidate_ids",
            ],
            max_rows=40,
        ),
        "",
        "## Key Discovery Rows",
        "",
        _markdown_table(
            key_discovery,
            [
                "ablation_subset_id",
                "schedule_mode",
                "family",
                "budget",
                "method_recurrent_endpoint_recall",
                "delta_recurrent_endpoint_recall",
                "actual_attempt_count",
                "noop_count",
            ],
            max_rows=40,
        ),
        "",
        "## Read",
        "",
        "- This ablation localizes mechanism contribution only on tiny demo graphs.",
        "- Compacted and slot-preserving schedules should be read together; compacted gains can include schedule compression.",
        "- Algorithm, wall/pathway, quality/cost, and NanoClustering claims remain closed.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_ablation(
    *,
    baseline_dir: Path,
    blind_dir: Path,
    replay_dir: Path,
    order_dir: Path,
    output_dir: Path,
    budgets: list[int],
    max_budget: int,
    n_iterations: int,
) -> dict[str, Any]:
    registry = pd.read_csv(blind_dir / BLIND_CANDIDATE_REGISTRY_CSV)
    candidates = _load_candidates(registry)
    manifest = pd.read_csv(baseline_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    baseline_curve = pd.read_csv(baseline_dir / BASELINE_DISCOVERY_CURVE_CSV)
    specs = _subset_specs(registry)
    ablation_registry = _ablation_registry(specs=specs, registry=registry)
    canonical_order = _canonical_order(registry=registry, blind_dir=blind_dir)
    schedule = _schedule_rows(
        specs=specs,
        canonical_order=canonical_order,
        candidates=candidates,
        max_budget=max_budget,
    )
    cache = _run_cache(
        candidates=candidates,
        manifest=manifest,
        max_seed=max_budget,
        n_iterations=n_iterations,
    )
    attempts = _attempt_rows(schedule=schedule, candidates=candidates, cache=cache)
    discovery = _discovery(
        attempts=attempts,
        manifest=manifest,
        baseline_curve=baseline_curve,
        budgets=budgets,
    )
    hard = _hard_endpoints(replay_dir=replay_dir)
    adversarial = _adversarial_first_hits(order_dir=order_dir)
    first_hits = _first_hits(
        attempts=attempts,
        manifest=manifest,
        hard=hard,
        adversarial=adversarial,
    )
    attribution = _attribution_matrix(first_hits=first_hits)
    seed_stability = _candidate_seed_stability(cache=cache, candidates=candidates)
    gate_matrix = _gate_matrix(
        ablation_registry=ablation_registry,
        schedule=schedule,
        attribution=attribution,
        first_hits=first_hits,
        seed_stability=seed_stability,
    )
    summary = {
        "subset_count": int(ablation_registry["ablation_subset_id"].nunique()),
        "schedule_rows": int(len(schedule)),
        "attempt_rows": int(len(attempts)),
        "discovery_rows": int(len(discovery)),
        "attribution_rows": int(len(attribution)),
        "seed_stability_rows": int(len(seed_stability)),
        "budgets": [int(budget) for budget in budgets],
        "max_budget": int(max_budget),
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "baseline_dir": _rel(baseline_dir),
            "blind_dir": _rel(blind_dir),
            "replay_dir": _rel(replay_dir),
            "order_dir": _rel(order_dir),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(ablation_registry, output_dir / ABLATION_REGISTRY_CSV)
    _write_csv(schedule, output_dir / ABLATION_SCHEDULE_CSV)
    _write_csv(attempts, output_dir / ABLATION_ATTEMPTS_CSV)
    _write_csv(discovery, output_dir / ABLATION_DISCOVERY_CSV)
    _write_csv(first_hits, output_dir / ABLATION_FIRST_HITS_CSV)
    _write_csv(attribution, output_dir / ABLATION_ATTRIBUTION_CSV)
    _write_csv(seed_stability, output_dir / ABLATION_SEED_STABILITY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "baseline_dir": _rel(baseline_dir),
        "blind_dir": _rel(blind_dir),
        "replay_dir": _rel(replay_dir),
        "order_dir": _rel(order_dir),
        "output_dir": _rel(output_dir),
        "budgets": [int(budget) for budget in budgets],
        "max_budget": int(max_budget),
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
        attribution=attribution,
        first_hits=first_hits,
        discovery=discovery,
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
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--order-dir", type=Path, default=DEFAULT_ORDER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budgets", type=_parse_budgets, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--max-budget", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=-1)
    args = parser.parse_args()
    budgets = [budget for budget in args.budgets if budget <= args.max_budget]
    summary = run_ablation(
        baseline_dir=args.baseline_dir,
        blind_dir=args.blind_dir,
        replay_dir=args.replay_dir,
        order_dir=args.order_dir,
        output_dir=args.output_dir,
        budgets=budgets,
        max_budget=args.max_budget,
        n_iterations=args.n_iterations,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
