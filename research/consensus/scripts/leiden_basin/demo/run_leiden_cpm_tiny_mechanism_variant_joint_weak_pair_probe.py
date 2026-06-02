#!/usr/bin/env python3
"""Probe joint weak-pair initializations for Stress 4 P8.1 eligible misses.

This is a downstream diagnostic. It does not mutate the phase-locked P0-P4
candidate registry. It reads P8.1 failure typing, selects only eligible misses
whose frozen positive registry contains multiple compatible weak-pair
candidate handles, and tests whether applying those handles jointly recovers
the missed recurrent endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from run_leiden_cpm_tiny_demo_seed_sweep import (
    _canonical_groups,
    _json_safe,
    _signature_id,
    _write_csv,
)
from run_leiden_cpm_tiny_mechanism_variant_p5_p8 import (
    _initial_membership,
    _variant_inputs,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_INPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_20260601"
DEFAULT_P5_P8_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_1_20260601"
DEFAULT_FAILURE_TYPING_DIR = (
    BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p8_1_failure_typing_v1_1_20260601"
)
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p8_2_joint_weak_pair_probe_v1_1_20260601"

GRAPH_MANIFEST_CSV = "tiny_cpm_variant_graph_manifest.csv"
GRAPH_EDGES_CSV = "tiny_cpm_variant_graph_edges.csv"
GRAPH_ROLES_CSV = "tiny_cpm_variant_graph_roles.csv"
P8_1_ENDPOINT_TYPING_CSV = "tiny_cpm_variant_p8_1_structural_target_endpoint_typing.csv"
FROZEN_ENDPOINT_MANIFEST_CSV = "tiny_cpm_variant_p6_frozen_endpoint_manifest.csv"
BASELINE_DISCOVERY_CSV = "tiny_cpm_variant_p6_baseline_endpoint_discovery.csv"

JOINT_REGISTRY_CSV = "tiny_cpm_variant_p8_2_joint_weak_pair_registry.csv"
JOINT_ATTEMPTS_CSV = "tiny_cpm_variant_p8_2_joint_weak_pair_attempts.csv"
JOINT_RECOVERY_CSV = "tiny_cpm_variant_p8_2_joint_weak_pair_recovery.csv"
SUMMARY_JSON = "tiny_cpm_variant_p8_2_joint_weak_pair_summary.json"
REPORT_MD = "tiny_cpm_variant_p8_2_joint_weak_pair_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM mechanism-variant P8.2 joint weak-pair probe only; reads P0-P4 "
    "roles/graph artifacts plus P5-P8/P8.1 diagnostics, constructs downstream "
    "joint initializations without mutating the phase-locked candidate registry, "
    "no route/pathway execution, no wall promotion, no quality/cost claim, no "
    "NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_p8_2_joint_probe_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "downstream_joint_initialization_probe_not_registry_claim"


@dataclass(frozen=True)
class WeakPairSpec:
    source_candidate_id: str
    weak_pair_slot: str
    host_slot: str


@dataclass(frozen=True)
class JointCandidate:
    variant_id: str
    frozen_endpoint_id: str
    endpoint_signature_id: str
    joint_candidate_id: str
    source_candidate_ids: tuple[str, ...]
    specs: tuple[WeakPairSpec, ...]
    initial_groups: tuple[tuple[str, ...], ...]
    target_endpoint_structure: str
    baseline_first_hit_p75: float


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


def _role_nodes(roles: pd.DataFrame, variant_id: str, role_type: str, role_slot: str | None = None) -> tuple[str, ...]:
    rows = roles[roles["variant_id"].astype(str).eq(variant_id) & roles["role_type"].astype(str).eq(role_type)]
    if role_slot is not None:
        rows = rows[rows["role_slot"].astype(str).eq(role_slot)]
    nodes: list[str] = []
    for value in rows["node_ids"].astype(str):
        nodes.extend(str(node) for node in json.loads(value))
    return tuple(sorted(dict.fromkeys(nodes)))


def _parse_weak_pair_candidate(candidate_id: str) -> WeakPairSpec | None:
    match = re.search(r"__(weak_pair_\d+)_to_(host_\d+)_tail_split$", str(candidate_id))
    if not match:
        return None
    return WeakPairSpec(
        source_candidate_id=str(candidate_id),
        weak_pair_slot=match.group(1),
        host_slot=match.group(2),
    )


def _joint_initial_groups(
    *,
    roles: pd.DataFrame,
    variant_id: str,
    specs: tuple[WeakPairSpec, ...],
) -> tuple[tuple[str, ...], ...]:
    selected_hosts = {spec.host_slot for spec in specs}
    selected_pair_nodes = {
        node
        for spec in specs
        for node in _role_nodes(roles, variant_id, "weak_pair", spec.weak_pair_slot)
    }
    groups: list[tuple[str, ...]] = []
    used_nodes: set[str] = set()
    for spec in sorted(specs, key=lambda item: (item.weak_pair_slot, item.host_slot)):
        core = _role_nodes(roles, variant_id, "boundary_core", spec.host_slot)
        pair = _role_nodes(roles, variant_id, "weak_pair", spec.weak_pair_slot)
        group = tuple(sorted((*core, *pair)))
        groups.append(group)
        used_nodes.update(group)
        tail = tuple(node for node in _role_nodes(roles, variant_id, "host_tail", spec.host_slot) if node not in used_nodes)
        if tail:
            groups.append(tail)
            used_nodes.update(tail)

    residual_weak = tuple(
        node for node in _role_nodes(roles, variant_id, "weak_module", "weak_module") if node not in selected_pair_nodes
    )
    if residual_weak:
        groups.append(residual_weak)
        used_nodes.update(residual_weak)

    host_rows = roles[
        roles["variant_id"].astype(str).eq(variant_id) & roles["role_type"].astype(str).eq("host")
    ]
    for host_slot in sorted(host_rows["role_slot"].astype(str).unique().tolist()):
        if host_slot in selected_hosts:
            continue
        host_nodes = tuple(node for node in _role_nodes(roles, variant_id, "host", host_slot) if node not in used_nodes)
        if host_nodes:
            groups.append(host_nodes)
            used_nodes.update(host_nodes)
    return tuple(groups)


def _load_joint_candidates(
    *,
    roles: pd.DataFrame,
    failure_typing: pd.DataFrame,
    baseline: pd.DataFrame,
) -> list[JointCandidate]:
    baseline_lookup = baseline.set_index(["variant_id", "frozen_endpoint_id"])
    miss_rows = failure_typing[
        failure_typing["failure_type"].astype(str).eq("eligible_missed_by_current_schedule_or_candidate_coupling")
    ].copy()
    candidates: list[JointCandidate] = []
    for row in miss_rows.sort_values(["variant_id", "frozen_endpoint_id"]).itertuples(index=False):
        source_ids = tuple(item for item in str(row.eligible_candidates).split(";") if item)
        specs = tuple(
            spec
            for spec in (_parse_weak_pair_candidate(candidate_id) for candidate_id in source_ids)
            if spec is not None
        )
        if len(specs) < 2:
            continue
        variant_id = str(row.variant_id)
        frozen_endpoint_id = str(row.frozen_endpoint_id)
        baseline_row = baseline_lookup.loc[(variant_id, frozen_endpoint_id)]
        slug = "__".join(f"{spec.weak_pair_slot}_to_{spec.host_slot}" for spec in specs)
        candidates.append(
            JointCandidate(
                variant_id=variant_id,
                frozen_endpoint_id=frozen_endpoint_id,
                endpoint_signature_id=str(row.endpoint_signature_id),
                joint_candidate_id=f"{variant_id}__joint__{slug}",
                source_candidate_ids=source_ids,
                specs=specs,
                initial_groups=_joint_initial_groups(roles=roles, variant_id=variant_id, specs=specs),
                target_endpoint_structure=str(row.endpoint_structure),
                baseline_first_hit_p75=float(baseline_row["baseline_first_hit_p75"]),
            )
        )
    return candidates


def _registry_frame(candidates: list[JointCandidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "variant_id": candidate.variant_id,
                "frozen_endpoint_id": candidate.frozen_endpoint_id,
                "endpoint_signature_id": candidate.endpoint_signature_id,
                "joint_candidate_id": candidate.joint_candidate_id,
                "source_candidate_ids": json.dumps(list(candidate.source_candidate_ids), sort_keys=True),
                "joint_component_count": len(candidate.specs),
                "initial_groups": json.dumps([list(group) for group in candidate.initial_groups], sort_keys=True),
                "target_endpoint_structure": candidate.target_endpoint_structure,
                "baseline_first_hit_p75": candidate.baseline_first_hit_p75,
                "construction_policy": "downstream_joint_of_phase_locked_single_weak_pair_candidates",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "frozen_endpoint_id"]))


def _run_joint_attempts(
    *,
    candidates: list[JointCandidate],
    variants_by_id: dict[str, Any],
    seeds_per_joint: int,
    n_iterations: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        variant = variants_by_id[candidate.variant_id]
        node_names = list(map(str, variant.graph.vs["name"]))
        initial = _initial_membership(node_names, candidate.initial_groups)
        runner = LeidenRunner(variant.graph, objective="cpm", default_iterations=n_iterations)
        for method_seed in range(seeds_per_joint):
            result = runner.run(variant.gamma, seed=method_seed, initial_membership=initial)
            groups = _canonical_groups(variant.graph, list(map(int, result.membership)))
            signature_id = _signature_id(groups)
            exact_hit = signature_id == candidate.endpoint_signature_id
            rows.append(
                {
                    "variant_id": candidate.variant_id,
                    "frozen_endpoint_id": candidate.frozen_endpoint_id,
                    "joint_candidate_id": candidate.joint_candidate_id,
                    "attempt_index_within_joint": int(method_seed + 1),
                    "method_seed": int(method_seed),
                    "gamma": float(variant.gamma),
                    "cluster_count": int(result.cluster_count),
                    "quality": float(result.quality),
                    "endpoint_signature_id": signature_id,
                    "endpoint_signature": json.dumps(groups, sort_keys=True),
                    "exact_target_endpoint_hit": bool(exact_hit),
                    "target_endpoint_signature_id": candidate.endpoint_signature_id,
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "frozen_endpoint_id", "method_seed"]))


def _recovery_frame(candidates: list[JointCandidate], attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        candidate_attempts = attempts[
            attempts["joint_candidate_id"].astype(str).eq(candidate.joint_candidate_id)
        ].copy()
        hits = candidate_attempts[candidate_attempts["exact_target_endpoint_hit"].astype(bool)]
        first_hit = int(hits["attempt_index_within_joint"].min()) if len(hits) else None
        rows.append(
            {
                "variant_id": candidate.variant_id,
                "frozen_endpoint_id": candidate.frozen_endpoint_id,
                "joint_candidate_id": candidate.joint_candidate_id,
                "source_candidate_ids": json.dumps(list(candidate.source_candidate_ids), sort_keys=True),
                "joint_component_count": len(candidate.specs),
                "attempt_count": int(len(candidate_attempts)),
                "exact_target_endpoint_hit": bool(len(hits)),
                "first_hit_attempt_within_joint": first_hit,
                "hit_seed_count": int(len(hits)),
                "dominant_result_endpoint_id": str(
                    candidate_attempts["endpoint_signature_id"].value_counts().index[0]
                )
                if len(candidate_attempts)
                else "",
                "unique_result_endpoint_count": int(candidate_attempts["endpoint_signature_id"].nunique()),
                "target_endpoint_signature_id": candidate.endpoint_signature_id,
                "baseline_first_hit_p75": candidate.baseline_first_hit_p75,
                "single_joint_initialization_beats_restart_p75": bool(
                    first_hit is not None and 1 <= candidate.baseline_first_hit_p75
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "frozen_endpoint_id"]))


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.loc[:, columns].head(max_rows).iterrows():
        values = []
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
    recovery: pd.DataFrame,
) -> None:
    lines = [
        "# Tiny CPM Mechanism Variant P8.2 Joint Weak-Pair Probe",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- p5_p8_dir: `{summary['p5_p8_dir']}`",
        f"- failure_typing_dir: `{summary['failure_typing_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- joint_candidate_count: `{summary['joint_candidate_count']}`",
        f"- recovered_endpoint_count: `{summary['recovered_endpoint_count']}`",
        f"- all_joint_misses_recovered: `{summary['all_joint_misses_recovered']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Recovery",
        "",
        _markdown_table(
            recovery,
            [
                "variant_id",
                "frozen_endpoint_id",
                "joint_component_count",
                "attempt_count",
                "exact_target_endpoint_hit",
                "first_hit_attempt_within_joint",
                "hit_seed_count",
                "baseline_first_hit_p75",
            ],
            max_rows=12,
        ),
        "",
        "## Interpretation",
        "",
        "- A hit here means the missed endpoint is reachable by jointly applying already phase-locked single weak-pair handles.",
        "- This is not a new P0-P4 registry claim because the joint candidates are downstream diagnostics.",
        "- If all joint misses recover, the next design question is whether to promote a pre-endpoint joint-candidate generator into a new panel version.",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    input_dir: Path,
    p5_p8_dir: Path,
    failure_typing_dir: Path,
    output_dir: Path,
    seeds_per_joint: int,
    n_iterations: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    if summary_path.exists() and not force:
        raise FileExistsError(f"{_rel(summary_path)} already exists. Use --force to regenerate.")

    manifest = pd.read_csv(input_dir / GRAPH_MANIFEST_CSV)
    edges = pd.read_csv(input_dir / GRAPH_EDGES_CSV)
    roles = pd.read_csv(input_dir / GRAPH_ROLES_CSV)
    variants = _variant_inputs(manifest, edges, roles)
    variants_by_id = {variant.variant_id: variant for variant in variants}
    failure_typing = pd.read_csv(failure_typing_dir / P8_1_ENDPOINT_TYPING_CSV)
    baseline = pd.read_csv(p5_p8_dir / BASELINE_DISCOVERY_CSV)
    candidates = _load_joint_candidates(roles=roles, failure_typing=failure_typing, baseline=baseline)
    registry = _registry_frame(candidates)
    attempts = _run_joint_attempts(
        candidates=candidates,
        variants_by_id=variants_by_id,
        seeds_per_joint=seeds_per_joint,
        n_iterations=n_iterations,
    )
    recovery = _recovery_frame(candidates, attempts)
    recovered_count = int(recovery["exact_target_endpoint_hit"].astype(bool).sum()) if len(recovery) else 0
    summary = {
        "input_dir": _rel(input_dir),
        "p5_p8_dir": _rel(p5_p8_dir),
        "failure_typing_dir": _rel(failure_typing_dir),
        "output_dir": _rel(output_dir),
        "joint_candidate_count": int(len(candidates)),
        "joint_attempt_count": int(len(attempts)),
        "seeds_per_joint": int(seeds_per_joint),
        "recovered_endpoint_count": recovered_count,
        "all_joint_misses_recovered": bool(len(candidates) and recovered_count == len(candidates)),
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            JOINT_REGISTRY_CSV,
            JOINT_ATTEMPTS_CSV,
            JOINT_RECOVERY_CSV,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }
    _write_csv(registry, output_dir / JOINT_REGISTRY_CSV)
    _write_csv(attempts, output_dir / JOINT_ATTEMPTS_CSV)
    _write_csv(recovery, output_dir / JOINT_RECOVERY_CSV)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir=output_dir, summary=summary, recovery=recovery)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe joint weak-pair initializations for Stress 4 P8.1 eligible misses."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--p5-p8-dir", type=Path, default=DEFAULT_P5_P8_DIR)
    parser.add_argument("--failure-typing-dir", type=Path, default=DEFAULT_FAILURE_TYPING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-per-joint", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_probe(
        input_dir=args.input_dir,
        p5_p8_dir=args.p5_p8_dir,
        failure_typing_dir=args.failure_typing_dir,
        output_dir=args.output_dir,
        seeds_per_joint=args.seeds_per_joint,
        n_iterations=args.n_iterations,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
