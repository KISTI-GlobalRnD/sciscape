#!/usr/bin/env python3
"""Failure-type Stress 4 P8 misses using endpoint structure.

The P5-P8 runner intentionally reports conservative target-class hits. This
post-hoc diagnostic separates three cases:

1. recurrent endpoints hit by a structurally compatible positive candidate;
2. recurrent endpoints structurally targetable by the frozen positive registry
   but missed by the current candidate execution schedule;
3. recurrent endpoints that the frozen positive registry does not currently
   target, such as explicit separate endpoints or single-weak-node splits.

It reads only phase-locked P0-P4 registry/role artifacts and P5-P8 endpoint
diagnostic outputs. It does not rerun Leiden.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_tiny_demo_seed_sweep import _json_safe, _write_csv


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_INPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_v1_1_20260601"
DEFAULT_P5_P8_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p5_p8_v1_1_20260601"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_mechanism_variant_panel_p8_1_failure_typing_v1_1_20260601"

GRAPH_ROLES_CSV = "tiny_cpm_variant_graph_roles.csv"
BLIND_CANDIDATE_REGISTRY_CSV = "tiny_cpm_variant_blind_candidate_registry.csv"
FROZEN_ENDPOINT_MANIFEST_CSV = "tiny_cpm_variant_p6_frozen_endpoint_manifest.csv"
BASELINE_DISCOVERY_CSV = "tiny_cpm_variant_p6_baseline_endpoint_discovery.csv"
CANDIDATE_ATTEMPTS_CSV = "tiny_cpm_variant_p7_candidate_attempts.csv"

ENDPOINT_TYPING_CSV = "tiny_cpm_variant_p8_1_structural_target_endpoint_typing.csv"
VARIANT_SUMMARY_CSV = "tiny_cpm_variant_p8_1_structural_target_variant_summary.csv"
SUMMARY_JSON = "tiny_cpm_variant_p8_1_failure_typing_summary.json"
REPORT_MD = "tiny_cpm_variant_p8_1_failure_typing_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM mechanism-variant P8.1 structural failure typing only; reads "
    "P0-P4 roles/candidate registry and P5-P8 endpoint outputs, no Leiden rerun, "
    "no route/pathway execution, no wall promotion, no quality/cost claim, no "
    "NanoClustering generality claim, and no algorithm-level claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_p8_1_failure_typing_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "posthoc_structural_failure_typing_not_method_claim"


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


def _role_nodes(roles: pd.DataFrame, variant_id: str, role_type: str, role_slot: str | None = None) -> set[str]:
    rows = roles[roles["variant_id"].astype(str).eq(variant_id) & roles["role_type"].astype(str).eq(role_type)]
    if role_slot is not None:
        rows = rows[rows["role_slot"].astype(str).eq(role_slot)]
    nodes: list[str] = []
    for value in rows["node_ids"].astype(str):
        nodes.extend(str(node) for node in json.loads(value))
    return set(nodes)


def _labels(endpoint_signature: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for label, group in enumerate(json.loads(str(endpoint_signature))):
        for node in group:
            labels[str(node)] = int(label)
    return labels


def _cluster_for(labels: dict[str, int], nodes: set[str]) -> int | None:
    if not nodes:
        return None
    present = [labels[node] for node in nodes if node in labels]
    if len(present) != len(nodes):
        return None
    if len(set(present)) != 1:
        return None
    return int(present[0])


def _contains_any(labels: dict[str, int], cluster: int | None, nodes: set[str]) -> bool:
    return cluster is not None and any(labels.get(node) == cluster for node in nodes)


def _excludes_all(labels: dict[str, int], cluster: int | None, nodes: set[str]) -> bool:
    return cluster is not None and all(labels.get(node) != cluster for node in nodes)


def _structural_target_hit(
    *,
    roles: pd.DataFrame,
    variant_id: str,
    candidate_id: str,
    target_mechanism_class: str,
    endpoint_signature: str,
) -> bool:
    labels = _labels(endpoint_signature)
    target = str(target_mechanism_class)

    if target.startswith("bridge_to_"):
        host = target.replace("bridge_to_", "")
        cluster = _cluster_for(labels, _role_nodes(roles, variant_id, "bridge", "bridge"))
        return _contains_any(labels, cluster, _role_nodes(roles, variant_id, "host", host))

    match = re.match(r"small_module_to_(host_[ab])_boundary_core", target)
    if match:
        host = match.group(1)
        cluster = _cluster_for(labels, _role_nodes(roles, variant_id, "small_module", "small_module"))
        core_nodes = _role_nodes(roles, variant_id, "boundary_core", host) or _role_nodes(roles, variant_id, "host", host)
        return _contains_any(labels, cluster, core_nodes)

    if target == "middle_contact_split":
        pull_a = _role_nodes(roles, variant_id, "middle_submodule", "middle_pull_a")
        pull_b = _role_nodes(roles, variant_id, "middle_submodule", "middle_pull_b")
        cluster_a = _cluster_for(labels, pull_a)
        cluster_b = _cluster_for(labels, pull_b)
        return (
            cluster_a is not None
            and cluster_b is not None
            and cluster_a != cluster_b
            and _contains_any(labels, cluster_a, _role_nodes(roles, variant_id, "host", "host_a"))
            and _contains_any(labels, cluster_b, _role_nodes(roles, variant_id, "host", "host_b"))
        )

    match = re.match(r"middle_module_to_(host_[ab])_boundary_core", target)
    if match:
        host = match.group(1)
        cluster = _cluster_for(labels, _role_nodes(roles, variant_id, "middle_module", "middle_module"))
        core_nodes = _role_nodes(roles, variant_id, "boundary_core", host) or _role_nodes(roles, variant_id, "host", host)
        return _contains_any(labels, cluster, core_nodes)

    if target == "weak_pair_tail_split":
        match = re.search(r"__weak_pair_(\d+)_to_(host_\d+)_tail_split", str(candidate_id))
        if not match:
            return False
        pair_nodes = _role_nodes(roles, variant_id, "weak_pair", f"weak_pair_{match.group(1)}")
        host = match.group(2)
        cluster = _cluster_for(labels, pair_nodes)
        core_nodes = _role_nodes(roles, variant_id, "boundary_core", host) or _role_nodes(roles, variant_id, "host", host)
        tail_nodes = _role_nodes(roles, variant_id, "host_tail", host)
        return _contains_any(labels, cluster, core_nodes) and _excludes_all(labels, cluster, tail_nodes)

    if target == "joint_weak_pair_tail_split":
        matches = re.findall(r"weak_pair_(\d+)_to_(host_\d+)", str(candidate_id))
        if len(matches) < 2:
            return False
        component_clusters: list[int] = []
        for pair_index, host in matches:
            pair_nodes = _role_nodes(roles, variant_id, "weak_pair", f"weak_pair_{pair_index}")
            cluster = _cluster_for(labels, pair_nodes)
            core_nodes = _role_nodes(roles, variant_id, "boundary_core", host) or _role_nodes(
                roles, variant_id, "host", host
            )
            tail_nodes = _role_nodes(roles, variant_id, "host_tail", host)
            if not (_contains_any(labels, cluster, core_nodes) and _excludes_all(labels, cluster, tail_nodes)):
                return False
            component_clusters.append(int(cluster))
        return len(set(component_clusters)) == len(component_clusters)

    return False


def _host_slot_of_node(roles: pd.DataFrame, variant_id: str, node: str) -> str | None:
    host_rows = roles[roles["variant_id"].astype(str).eq(variant_id) & roles["role_type"].astype(str).eq("host")]
    for slot in sorted(host_rows["role_slot"].astype(str).unique().tolist()):
        if node in _role_nodes(roles, variant_id, "host", slot):
            return slot
    return None


def _endpoint_structure(roles: pd.DataFrame, variant_id: str, endpoint_signature: str) -> str:
    weak_nodes = _role_nodes(roles, variant_id, "weak_module", "weak_module")
    if not weak_nodes:
        return ""
    pair_rows = roles[
        roles["variant_id"].astype(str).eq(variant_id)
        & roles["role_type"].astype(str).isin(["weak_pair", "weak_pair_decoy"])
    ]
    weak_pairs = [
        (str(row.role_slot), set(json.loads(str(row.node_ids))))
        for row in pair_rows.itertuples(index=False)
    ]
    pieces: list[str] = []
    for index, group in enumerate(json.loads(str(endpoint_signature))):
        group_set = set(map(str, group))
        xs = sorted(group_set & weak_nodes)
        if not xs:
            continue
        host_parts: list[str] = []
        host_slots = sorted(
            {
                slot
                for node in group_set
                for slot in [_host_slot_of_node(roles, variant_id, node)]
                if slot is not None
            }
        )
        for slot in host_slots:
            core = _role_nodes(roles, variant_id, "boundary_core", slot) | _role_nodes(
                roles, variant_id, "boundary_core_decoy", slot
            )
            tail = _role_nodes(roles, variant_id, "host_tail", slot)
            host_nodes = group_set & _role_nodes(roles, variant_id, "host", slot)
            host_parts.append(f"{slot}:core{len(host_nodes & core)}/tail{len(host_nodes & tail)}")
        pair_parts: list[str] = []
        for slot, pair_nodes in weak_pairs:
            hits = sorted(group_set & pair_nodes)
            if hits:
                pair_parts.append(f"{slot}({'+'.join(node.split('_')[-1] for node in hits)})")
        pieces.append(
            f"G{index}:xs={'+'.join(node.split('_')[-1] for node in xs)} "
            f"hosts={';'.join(host_parts) or 'none'} pairs={';'.join(pair_parts) or 'none'}"
        )
    return " | ".join(pieces)


def _endpoint_typing(
    *,
    roles: pd.DataFrame,
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
    attempts: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    baseline_lookup = baseline.set_index(["variant_id", "frozen_endpoint_id"])
    positive_registry = registry[registry["target_claim_allowed"].astype(bool)].copy()
    preserved = manifest[
        manifest["is_recurrent_endpoint"].astype(bool)
        & ~manifest["mechanism_state"].astype(str).str.contains("control")
    ].copy()
    rows: list[dict[str, Any]] = []
    for endpoint in preserved.itertuples(index=False):
        variant_id = str(endpoint.variant_id)
        endpoint_id = str(endpoint.frozen_endpoint_id)
        endpoint_signature = str(endpoint.endpoint_signature)
        variant_candidates = positive_registry[positive_registry["variant_id"].astype(str).eq(variant_id)]
        eligible_candidates = [
            str(candidate.handle_candidate_id)
            for candidate in variant_candidates.itertuples(index=False)
            if _structural_target_hit(
                roles=roles,
                variant_id=variant_id,
                candidate_id=str(candidate.handle_candidate_id),
                target_mechanism_class=str(candidate.target_mechanism_class),
                endpoint_signature=endpoint_signature,
            )
        ]
        endpoint_attempts = attempts[
            attempts["variant_id"].astype(str).eq(variant_id)
            & attempts["frozen_endpoint_id"].astype(str).eq(endpoint_id)
            & attempts["target_claim_allowed"].astype(bool)
        ]
        hit_rows: list[tuple[int, str]] = []
        for attempt in endpoint_attempts.itertuples(index=False):
            candidate_row = registry[registry["handle_candidate_id"].astype(str).eq(str(attempt.handle_candidate_id))].iloc[0]
            if _structural_target_hit(
                roles=roles,
                variant_id=variant_id,
                candidate_id=str(candidate_row.handle_candidate_id),
                target_mechanism_class=str(candidate_row.target_mechanism_class),
                endpoint_signature=endpoint_signature,
            ):
                hit_rows.append((int(attempt.attempt_index), str(attempt.handle_candidate_id)))
        baseline_row = baseline_lookup.loc[(variant_id, endpoint_id)]
        first_hit = min((item[0] for item in hit_rows), default=None)
        if hit_rows:
            failure_type = "hit"
        elif eligible_candidates:
            failure_type = "eligible_missed_by_current_schedule_or_candidate_coupling"
        else:
            failure_type = "not_targeted_by_current_positive_registry"
        rows.append(
            {
                "variant_id": variant_id,
                "mechanism_family": str(endpoint.mechanism_family),
                "mechanism_state": str(endpoint.mechanism_state),
                "frozen_endpoint_id": endpoint_id,
                "endpoint_signature_id": str(endpoint.endpoint_signature_id),
                "mechanism_read": str(endpoint.mechanism_read),
                "seed_count": int(endpoint.seed_count),
                "seed_share": float(endpoint.seed_share),
                "quality_median": float(endpoint.quality_median),
                "baseline_first_hit_p75": float(baseline_row["baseline_first_hit_p75"]),
                "structural_target_eligible": bool(eligible_candidates),
                "eligible_candidate_count": int(len(eligible_candidates)),
                "eligible_candidates": ";".join(eligible_candidates),
                "structural_target_hit": bool(hit_rows),
                "first_structural_target_attempt": first_hit,
                "first_structural_target_candidate": sorted(hit_rows)[0][1] if hit_rows else "",
                "structural_target_beats_restart_p75": bool(
                    first_hit is not None and first_hit <= float(baseline_row["baseline_first_hit_p75"])
                ),
                "failure_type": failure_type,
                "endpoint_structure": _endpoint_structure(roles, variant_id, endpoint_signature),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["variant_id", "frozen_endpoint_id"]))


def _variant_summary(endpoint_typing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_id, group in endpoint_typing.groupby("variant_id", sort=True):
        rows.append(
            {
                "variant_id": str(variant_id),
                "mechanism_family": str(group["mechanism_family"].iloc[0]),
                "mechanism_state": str(group["mechanism_state"].iloc[0]),
                "recurrent_endpoint_count": int(len(group)),
                "structural_target_eligible_count": int(group["structural_target_eligible"].astype(bool).sum()),
                "structural_target_hit_count": int(group["structural_target_hit"].astype(bool).sum()),
                "structural_target_beats_restart_p75_count": int(
                    group["structural_target_beats_restart_p75"].astype(bool).sum()
                ),
                "eligible_missed_count": int(
                    group["failure_type"].astype(str).eq("eligible_missed_by_current_schedule_or_candidate_coupling").sum()
                ),
                "not_targeted_count": int(
                    group["failure_type"].astype(str).eq("not_targeted_by_current_positive_registry").sum()
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows).sort_values("variant_id"))


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
    variant_summary: pd.DataFrame,
    endpoint_typing: pd.DataFrame,
) -> None:
    misses = endpoint_typing[endpoint_typing["failure_type"].astype(str).ne("hit")]
    lines = [
        "# Tiny CPM Mechanism Variant P8.1 Failure Typing",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- p5_p8_dir: `{summary['p5_p8_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- preserved_recurrent_endpoint_count: `{summary['preserved_recurrent_endpoint_count']}`",
        f"- structural_target_eligible_count: `{summary['structural_target_eligible_count']}`",
        f"- structural_target_hit_count: `{summary['structural_target_hit_count']}`",
        f"- eligible_missed_count: `{summary['eligible_missed_count']}`",
        f"- not_targeted_count: `{summary['not_targeted_count']}`",
        f"- structural_target_hit_rate_among_eligible: `{summary['structural_target_hit_rate_among_eligible']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Variant Summary",
        "",
        _markdown_table(
            variant_summary,
            [
                "variant_id",
                "recurrent_endpoint_count",
                "structural_target_eligible_count",
                "structural_target_hit_count",
                "eligible_missed_count",
                "not_targeted_count",
            ],
            max_rows=16,
        ),
        "",
        "## Non-Hit Endpoint Typing",
        "",
        _markdown_table(
            misses,
            [
                "variant_id",
                "frozen_endpoint_id",
                "mechanism_read",
                "seed_count",
                "structural_target_eligible",
                "failure_type",
                "eligible_candidates",
            ],
            max_rows=32,
        ),
        "",
        "## Interpretation",
        "",
        "- Non-targeted endpoints are not failures of the current positive registry; they expose missing endpoint classes.",
        "- Eligible misses are stronger failures: the current registry has a compatible target predicate, but the schedule/candidate coupling did not reach the endpoint.",
        "- In `v1.1`, the main eligible misses were joint weak-pair endpoints in `df_two_pair`; later panels should use this as a regression check.",
    ]
    output_dir.joinpath(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    input_dir: Path,
    p5_p8_dir: Path,
    output_dir: Path,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_JSON
    if summary_path.exists() and not force:
        raise FileExistsError(f"{_rel(summary_path)} already exists. Use --force to regenerate.")

    roles = pd.read_csv(input_dir / GRAPH_ROLES_CSV)
    registry = pd.read_csv(input_dir / BLIND_CANDIDATE_REGISTRY_CSV)
    manifest = pd.read_csv(p5_p8_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    baseline = pd.read_csv(p5_p8_dir / BASELINE_DISCOVERY_CSV)
    attempts = pd.read_csv(p5_p8_dir / CANDIDATE_ATTEMPTS_CSV)
    endpoint_typing = _endpoint_typing(
        roles=roles,
        registry=registry,
        manifest=manifest,
        attempts=attempts,
        baseline=baseline,
    )
    variant_summary = _variant_summary(endpoint_typing)

    eligible = endpoint_typing[endpoint_typing["structural_target_eligible"].astype(bool)]
    hit_count = int(endpoint_typing["structural_target_hit"].astype(bool).sum())
    eligible_count = int(len(eligible))
    summary = {
        "input_dir": _rel(input_dir),
        "p5_p8_dir": _rel(p5_p8_dir),
        "output_dir": _rel(output_dir),
        "preserved_recurrent_endpoint_count": int(len(endpoint_typing)),
        "structural_target_eligible_count": eligible_count,
        "structural_target_hit_count": hit_count,
        "structural_target_beats_restart_p75_count": int(
            endpoint_typing["structural_target_beats_restart_p75"].astype(bool).sum()
        ),
        "eligible_missed_count": int(
            endpoint_typing["failure_type"].astype(str).eq("eligible_missed_by_current_schedule_or_candidate_coupling").sum()
        ),
        "not_targeted_count": int(
            endpoint_typing["failure_type"].astype(str).eq("not_targeted_by_current_positive_registry").sum()
        ),
        "structural_target_hit_rate_among_eligible": float(hit_count / eligible_count) if eligible_count else 0.0,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            ENDPOINT_TYPING_CSV,
            VARIANT_SUMMARY_CSV,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }
    _write_csv(endpoint_typing, output_dir / ENDPOINT_TYPING_CSV)
    _write_csv(variant_summary, output_dir / VARIANT_SUMMARY_CSV)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        output_dir=output_dir,
        summary=summary,
        variant_summary=variant_summary,
        endpoint_typing=endpoint_typing,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc structural failure typing for Stress 4 P5-P8 endpoint misses."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--p5-p8-dir", type=Path, default=DEFAULT_P5_P8_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_analysis(
        input_dir=args.input_dir,
        p5_p8_dir=args.p5_p8_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
