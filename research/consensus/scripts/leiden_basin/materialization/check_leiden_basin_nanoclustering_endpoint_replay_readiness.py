#!/usr/bin/env python3
"""Check NanoClustering endpoint-replay readiness for the frozen local panel.

This is a contract/readiness runner, not endpoint replay. It consumes the
frozen joint weak-pair local panel, resolves endpoint-family target handles to
existing NanoClustering membership artifacts, freezes the strict-core attempt
plan, and checks whether the raw graph inputs needed for Leiden replay are
locally available. It preserves the original absolute graph paths from the
NanoClustering sidecar run requests, but can use a verified local mirror when
the original `/data` path is not mounted. It does not run clustering, replay
endpoints, execute routes/pathways, promote walls, inspect quality/cost, or
claim real-data method success.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
DEFAULT_NANO_ROOT = REPO_ROOT.parent / "1.1.4.KISTI_NanoClustering"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_PANEL_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_joint_weak_pair_local_panel_design_20260601"
)
DEFAULT_EXTERNAL_LANDSCAPE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_endpoint_replay_readiness_20260601"
)

PANEL_CASE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_case_rows.csv"
PANEL_ROLE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_role_rows.csv"
PANEL_EVENT_ROLE_ROWS_CSV = "nanoclustering_joint_weak_pair_local_panel_event_role_rows.csv"
PANEL_ENDPOINT_SIGNATURE_ROWS_CSV = (
    "nanoclustering_joint_weak_pair_local_panel_endpoint_signature_rows.csv"
)
PANEL_ENDPOINT_REPLAY_CONTRACT_CSV = (
    "nanoclustering_joint_weak_pair_local_panel_endpoint_replay_contract.csv"
)
EXTERNAL_ENDPOINT_REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"

INPUT_MANIFEST_CSV = "nanoclustering_endpoint_replay_readiness_input_manifest.csv"
GRAPH_INPUT_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_graph_input_rows.csv"
ENDPOINT_TARGET_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_endpoint_target_rows.csv"
ATTEMPT_PLAN_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_attempt_plan_rows.csv"
MISSING_INPUT_ROWS_CSV = "nanoclustering_endpoint_replay_readiness_missing_input_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_endpoint_replay_readiness_gate_matrix.csv"
RUNNER_CONFIG_TEMPLATE_JSON = (
    "nanoclustering_endpoint_replay_readiness_runner_config_template.json"
)
SUMMARY_JSON = "nanoclustering_endpoint_replay_readiness_summary.json"
CONFIG_JSON = "nanoclustering_endpoint_replay_readiness_config.json"
REPORT_MD = "nanoclustering_endpoint_replay_readiness_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering endpoint-replay readiness only; resolves frozen local-panel "
    "roles, endpoint-family target memberships, and raw graph runtime inputs. "
    "It does not run clustering, execute endpoint replay, execute routes/pathways, "
    "promote walls, inspect quality/cost, or claim real-data method success."
)
REPLAY_EXECUTION_STATUS = "not_executed_readiness_only"
ROUTE_EXECUTION_STATUS = "not_executed_no_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
QUALITY_COST_STATUS = "excluded_readiness_only"
DEFAULT_METHOD_SEEDS = tuple(range(10))
DEFAULT_ANALYSIS_TIER = "strict_core_v0_primary"
HANDLE_RE = re.compile(
    r"^(?P<run_id>sidecar_(?P<branch>java|rust)_g0005_min250_gamma0p7_seed(?P<seed>\d{3}))"
    r":(?P<handle_kind>run|ref)(?P<cluster_id>\d+)$"
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _split_handles(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [item for item in str(value).split(";") if item]


def _parse_seed_list(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return DEFAULT_METHOD_SEEDS
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["replay_execution_status"] = REPLAY_EXECUTION_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _required_artifact_rows(
    *,
    panel_dir: Path,
    external_landscape_dir: Path,
    nano_root: Path,
) -> pd.DataFrame:
    specs = [
        (
            "panel_case_rows",
            "frozen_panel_design",
            panel_dir / PANEL_CASE_ROWS_CSV,
            True,
            "case denominator and analysis tiers",
        ),
        (
            "panel_role_rows",
            "frozen_panel_design",
            panel_dir / PANEL_ROLE_ROWS_CSV,
            True,
            "candidate/control role IDs",
        ),
        (
            "panel_event_role_rows",
            "frozen_panel_design",
            panel_dir / PANEL_EVENT_ROLE_ROWS_CSV,
            True,
            "pre-endpoint event roles",
        ),
        (
            "panel_endpoint_signature_rows",
            "frozen_panel_design",
            panel_dir / PANEL_ENDPOINT_SIGNATURE_ROWS_CSV,
            True,
            "endpoint-family target signatures",
        ),
        (
            "panel_endpoint_replay_contract",
            "frozen_panel_design",
            panel_dir / PANEL_ENDPOINT_REPLAY_CONTRACT_CSV,
            True,
            "future replay contract",
        ),
        (
            "external_endpoint_registry",
            "endpoint_target_resolution",
            external_landscape_dir / EXTERNAL_ENDPOINT_REGISTRY_CSV,
            True,
            "NanoClustering seed endpoint membership registry",
        ),
        (
            "nano_root",
            "endpoint_target_resolution",
            nano_root,
            True,
            "NanoClustering repository root",
        ),
    ]
    rows = []
    for input_id, role, path, required, notes in specs:
        exists = path.exists()
        rows.append(
            {
                "input_id": input_id,
                "input_role": role,
                "path": str(path),
                "path_read": _rel(path),
                "required_for_replay_execution": required,
                "exists": exists,
                "status": "present" if exists else "missing_required_input",
                "notes": notes,
            }
        )
    return _with_claim_columns(pd.DataFrame(rows))


def _load_run_request(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)


def _graph_input_rows(
    *,
    nano_root: Path,
    endpoint_registry: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for branch in ["java", "rust"]:
        run_request_path = (
            nano_root
            / "outputs/specter2/_current/sidecar_g0005_min250_micro_seed_sweep_gamma0p7_20260512"
            / branch
            / "run_request.json"
        )
        run_request = _load_run_request(run_request_path)
        node_manifest = Path(str(run_request.get("inputs", {}).get("node_manifest", "")))
        int_edges = Path(str(run_request.get("inputs", {}).get("int_edges", "")))
        local_mirror_dir = (
            nano_root
            / "outputs/specter2/_current/sidecar_g0005_min250_candidate_nano_doc_graph_20260512"
            / branch
        )
        local_mirror_node_manifest = local_mirror_dir / "node_manifest.parquet"
        local_mirror_int_edges = local_mirror_dir / "int_edges.parquet"
        branch_registry = endpoint_registry[
            endpoint_registry["run_id"].astype(str).str.startswith(f"sidecar_{branch}_")
        ]
        expected_node_count = (
            int(branch_registry["row_count"].dropna().astype(int).mode().iloc[0])
            if not branch_registry.empty
            else 0
        )
        raw_node_count = _parquet_row_count(node_manifest)
        local_mirror_node_count = _parquet_row_count(local_mirror_node_manifest)
        local_mirror_edge_count = _parquet_row_count(local_mirror_int_edges)
        local_mirror_valid = (
            local_mirror_node_manifest.exists()
            and local_mirror_int_edges.exists()
            and local_mirror_node_count == expected_node_count
        )
        if node_manifest.exists() and int_edges.exists():
            runtime_node_manifest = node_manifest
            runtime_int_edges = int_edges
            runtime_graph_source = "original_run_request_absolute_path"
            runtime_graph_status = "ready_original_raw_graph_inputs_present"
        elif local_mirror_valid:
            runtime_node_manifest = local_mirror_node_manifest
            runtime_int_edges = local_mirror_int_edges
            runtime_graph_source = "verified_local_mirror_of_run_request_outputs"
            runtime_graph_status = "ready_local_mirror_graph_inputs_present"
        else:
            runtime_node_manifest = node_manifest
            runtime_int_edges = int_edges
            runtime_graph_source = "missing_runtime_graph_inputs"
            runtime_graph_status = "blocked_missing_raw_graph_inputs"
        rows.append(
            {
                "branch": branch,
                "run_request_path": str(run_request_path),
                "run_request_exists": run_request_path.exists(),
                "expected_node_count_from_registry": expected_node_count,
                "original_node_manifest_path": str(node_manifest),
                "original_node_manifest_exists": node_manifest.exists(),
                "original_node_manifest_node_count": raw_node_count,
                "original_int_edges_path": str(int_edges),
                "original_int_edges_exists": int_edges.exists(),
                "local_mirror_node_manifest_path": str(local_mirror_node_manifest),
                "local_mirror_node_manifest_exists": local_mirror_node_manifest.exists(),
                "local_mirror_node_manifest_node_count": local_mirror_node_count,
                "local_mirror_int_edges_path": str(local_mirror_int_edges),
                "local_mirror_int_edges_exists": local_mirror_int_edges.exists(),
                "local_mirror_int_edges_edge_count": local_mirror_edge_count,
                "local_mirror_status": (
                    "valid_row_count_matched_local_mirror"
                    if local_mirror_valid
                    else "missing_or_row_count_mismatch_local_mirror"
                ),
                "runtime_node_manifest_path": str(runtime_node_manifest),
                "runtime_node_manifest_exists": runtime_node_manifest.exists(),
                "runtime_int_edges_path": str(runtime_int_edges),
                "runtime_int_edges_exists": runtime_int_edges.exists(),
                "runtime_graph_source": runtime_graph_source,
                "runtime_graph_status": runtime_graph_status,
                "notes": (
                    "Replay must use the branch-specific sidecar candidate graph. "
                    "A local mirror is acceptable only when it is the mirrored "
                    "sidecar graph output and its node count matches the endpoint "
                    "registry; unrelated active hierarchy graphs are not safe "
                    "substitutes."
                ),
            }
        )
    return _with_claim_columns(pd.DataFrame(rows))


def _registry_by_run_id(endpoint_registry: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["run_id"]): row for _, row in endpoint_registry.iterrows()}


def _membership_cluster_stats(
    *,
    row: pd.Series,
    cluster_id: int,
    cache: dict[str, pd.DataFrame],
) -> tuple[bool, int, int, str]:
    path = Path(str(row.get("absolute_path", "")))
    if not path.exists():
        return False, 0, 0, "missing_membership_path"
    label_cols = [item for item in str(row.get("label_cols", "")).split(";") if item]
    if not label_cols:
        return False, 0, 0, "missing_label_col"
    label_col = label_cols[0]
    unit_col = str(row.get("unit_col", ""))
    weight_col = str(row.get("weight_col", ""))
    run_id = str(row.get("run_id", ""))
    if run_id not in cache:
        try:
            cache[run_id] = pd.read_parquet(
                path,
                columns=list(dict.fromkeys([unit_col, weight_col, label_col])),
            )
        except Exception as exc:
            return False, 0, 0, f"membership_read_failed:{type(exc).__name__}"
    frame = cache[run_id]
    if label_col not in frame:
        return False, 0, 0, "missing_label_col_in_membership"
    selected = frame[frame[label_col].astype("int64").eq(int(cluster_id))]
    if selected.empty:
        return False, 0, 0, "cluster_label_missing"
    return (
        True,
        int(selected[unit_col].nunique()) if unit_col in selected else int(len(selected)),
        int(selected[weight_col].sum()) if weight_col in selected else 0,
        "cluster_label_present",
    )


def _endpoint_target_rows(
    *,
    endpoint_signatures: pd.DataFrame,
    endpoint_registry: pd.DataFrame,
) -> pd.DataFrame:
    registry = _registry_by_run_id(endpoint_registry)
    membership_cache: dict[str, pd.DataFrame] = {}
    rows = []
    signature_rows = endpoint_signatures.sort_values(
        ["analysis_tier", "panel_case_rank", "role_side", "endpoint_signature_id"],
        kind="mergesort",
    )
    for signature in signature_rows.itertuples(index=False):
        sig = signature._asdict()
        for handle_field, handle_role in [
            ("dominant_host_handle_ids", "dominant_host_context_member"),
            ("top1_endpoint_handle_ids", "top1_endpoint_target_member"),
        ]:
            for handle in _split_handles(sig.get(handle_field, "")):
                parsed = HANDLE_RE.match(handle)
                parsed_values: dict[str, Any] = {
                    "parsed_handle": bool(parsed),
                    "run_id": "",
                    "branch": "",
                    "seed": "",
                    "handle_kind": "",
                    "cluster_id": "",
                }
                registry_row: pd.Series | None = None
                present = False
                unit_count = 0
                weight_sum = 0
                status = "handle_parse_failed"
                membership_path = ""
                label_cols = ""
                if parsed is not None:
                    parsed_values.update(
                        {
                            "run_id": parsed.group("run_id"),
                            "branch": parsed.group("branch"),
                            "seed": int(parsed.group("seed")),
                            "handle_kind": parsed.group("handle_kind"),
                            "cluster_id": int(parsed.group("cluster_id")),
                        }
                    )
                    registry_row = registry.get(str(parsed_values["run_id"]))
                    if registry_row is None:
                        status = "missing_endpoint_registry_row"
                    else:
                        membership_path = str(registry_row.get("absolute_path", ""))
                        label_cols = str(registry_row.get("label_cols", ""))
                        present, unit_count, weight_sum, status = _membership_cluster_stats(
                            row=registry_row,
                            cluster_id=int(parsed_values["cluster_id"]),
                            cache=membership_cache,
                        )
                rows.append(
                    {
                        "endpoint_signature_id": sig["endpoint_signature_id"],
                        "panel_case_id": sig["panel_case_id"],
                        "panel_case_rank": int(sig["panel_case_rank"]),
                        "panel_scope": sig["panel_scope"],
                        "analysis_tier": sig["analysis_tier"],
                        "strict_core_v0": _as_bool(sig["strict_core_v0"]),
                        "role_side": sig["role_side"],
                        "primitive_id": sig["primitive_id"],
                        "target_handle_role": handle_role,
                        "endpoint_handle_id": handle,
                        **parsed_values,
                        "membership_path": membership_path,
                        "membership_path_exists": Path(membership_path).exists()
                        if membership_path
                        else False,
                        "label_cols": label_cols,
                        "cluster_label_present": present,
                        "cluster_unit_count": unit_count,
                        "cluster_weight_sum": weight_sum,
                        "target_resolution_status": status,
                    }
                )
    return _with_claim_columns(pd.DataFrame(rows))


def _attempt_plan_rows(
    *,
    case_rows: pd.DataFrame,
    role_rows: pd.DataFrame,
    endpoint_signatures: pd.DataFrame,
    graph_rows: pd.DataFrame,
    method_seeds: tuple[int, ...],
    analysis_tier: str,
) -> pd.DataFrame:
    selected_cases = case_rows[case_rows["analysis_tier"].astype(str).eq(analysis_tier)]
    selected_case_ids = set(selected_cases["panel_case_id"].astype(str))
    selected_roles = role_rows[role_rows["panel_case_id"].astype(str).isin(selected_case_ids)]
    sig_lookup = {
        (str(row["panel_case_id"]), str(row["role_side"])): str(
            row["endpoint_signature_id"]
        )
        for _, row in endpoint_signatures.iterrows()
    }
    graph_status_by_branch = {
        str(row["branch"]): str(row["runtime_graph_status"])
        for _, row in graph_rows.iterrows()
    }
    rows = []
    for role in selected_roles.sort_values(
        ["panel_case_rank", "role_side"], kind="mergesort"
    ).itertuples(index=False):
        role_dict = role._asdict()
        branch = str(role_dict["branch"])
        for method_seed in method_seeds:
            attempt_id = (
                f"{role_dict['role_id']}__method_seed{int(method_seed):03d}"
            )
            runtime_graph_status = graph_status_by_branch.get(
                branch, "missing_branch_graph_row"
            )
            rows.append(
                {
                    "attempt_id": attempt_id,
                    "panel_case_id": role_dict["panel_case_id"],
                    "panel_case_rank": int(role_dict["panel_case_rank"]),
                    "analysis_tier": role_dict["analysis_tier"],
                    "strict_core_v0": _as_bool(role_dict["strict_core_v0"]),
                    "role_id": role_dict["role_id"],
                    "role_side": role_dict["role_side"],
                    "primitive_id": role_dict["primitive_id"],
                    "branch": branch,
                    "method_seed": int(method_seed),
                    "target_endpoint_signature_id": sig_lookup.get(
                        (str(role_dict["panel_case_id"]), str(role_dict["role_side"])),
                        "",
                    ),
                    "initial_membership_policy": (
                        "future_runner_must_construct_from_frozen_endpoint_family_signature"
                    ),
                    "endpoint_success_unit": role_dict["endpoint_success_unit"],
                    "runtime_graph_status": runtime_graph_status,
                    "attempt_execution_status": (
                        "ready_to_execute"
                        if runtime_graph_status.startswith("ready_")
                        else "blocked_missing_raw_graph_inputs"
                    ),
                }
            )
    return _with_claim_columns(pd.DataFrame(rows))


def _missing_input_rows(
    *,
    input_manifest: pd.DataFrame,
    graph_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "missing_input_type",
        "input_id",
        "path",
        "status",
        "blocking_scope",
    ]
    rows = []
    for row in input_manifest.itertuples(index=False):
        data = row._asdict()
        if not _as_bool(data["exists"]):
            rows.append(
                {
                    "missing_input_type": "required_artifact",
                    "input_id": data["input_id"],
                    "path": data["path"],
                    "status": data["status"],
                    "blocking_scope": "readiness_construction",
                }
            )
    for row in graph_rows.itertuples(index=False):
        data = row._asdict()
        if not str(data["runtime_graph_status"]).startswith("ready_") and not _as_bool(
            data["runtime_node_manifest_exists"]
        ):
            rows.append(
                {
                    "missing_input_type": "runtime_graph_node_manifest",
                    "input_id": f"{data['branch']}_node_manifest",
                    "path": data["runtime_node_manifest_path"],
                    "status": "missing_required_runtime_graph_input",
                    "blocking_scope": "endpoint_replay_execution",
                }
            )
        if not str(data["runtime_graph_status"]).startswith("ready_") and not _as_bool(
            data["runtime_int_edges_exists"]
        ):
            rows.append(
                {
                    "missing_input_type": "runtime_graph_int_edges",
                    "input_id": f"{data['branch']}_int_edges",
                    "path": data["runtime_int_edges_path"],
                    "status": "missing_required_runtime_graph_input",
                    "blocking_scope": "endpoint_replay_execution",
                }
            )
    unresolved_targets = target_rows[
        ~target_rows["cluster_label_present"].astype(bool)
        | ~target_rows["membership_path_exists"].astype(bool)
        | ~target_rows["parsed_handle"].astype(bool)
    ]
    for row in unresolved_targets.itertuples(index=False):
        data = row._asdict()
        rows.append(
            {
                "missing_input_type": "endpoint_target_membership",
                "input_id": data["endpoint_handle_id"],
                "path": data["membership_path"],
                "status": data["target_resolution_status"],
                "blocking_scope": "endpoint_signature_readout",
            }
        )
    return _with_claim_columns(pd.DataFrame(rows, columns=columns))


def _gate_matrix(
    *,
    input_manifest: pd.DataFrame,
    graph_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    attempt_plan: pd.DataFrame,
    case_rows: pd.DataFrame,
    endpoint_signatures: pd.DataFrame,
    analysis_tier: str,
) -> pd.DataFrame:
    strict_cases = case_rows[case_rows["analysis_tier"].astype(str).eq(analysis_tier)]
    target_ok = (
        not target_rows.empty
        and bool(target_rows["parsed_handle"].astype(bool).all())
        and bool(target_rows["membership_path_exists"].astype(bool).all())
        and bool(target_rows["cluster_label_present"].astype(bool).all())
    )
    graph_ok = bool(
        graph_rows["runtime_graph_status"].astype(str).str.startswith("ready_").all()
    )
    single_handle_ok = (
        set(case_rows["endpoint_success_unit"].astype(str).unique())
        == {"endpoint_family_signature_distance"}
        and not case_rows["single_endpoint_hit_allowed"].map(_as_bool).any()
        and set(endpoint_signatures["endpoint_success_unit"].astype(str).unique())
        == {"endpoint_family_signature_distance"}
    )
    rows = [
        {
            "gate_id": "G1_required_design_artifacts_present",
            "gate_question": "Are frozen panel design artifacts present?",
            "status": (
                "pass"
                if input_manifest["exists"].astype(bool).all()
                else "fail_missing_required_artifact"
            ),
            "evidence": f"{int(input_manifest['exists'].astype(bool).sum())}/{len(input_manifest)} present",
        },
        {
            "gate_id": "G2_strict_core_denominator_frozen",
            "gate_question": "Is the strict-core replay denominator frozen?",
            "status": "pass" if len(strict_cases) == 10 else "caveat_unexpected_denominator",
            "evidence": f"{len(strict_cases)} {analysis_tier} cases",
        },
        {
            "gate_id": "G3_endpoint_family_readout_only",
            "gate_question": "Is success defined by endpoint-family signature distance, not a single handle?",
            "status": "pass" if single_handle_ok else "fail_single_handle_readout_detected",
            "evidence": "single_endpoint_hit_allowed=false and endpoint_success_unit=endpoint_family_signature_distance",
        },
        {
            "gate_id": "G4_endpoint_target_memberships_materialized",
            "gate_question": "Can all target endpoint handles be resolved to local membership artifacts?",
            "status": "pass" if target_ok else "fail_missing_endpoint_target_membership",
            "evidence": f"{int(target_rows['cluster_label_present'].astype(bool).sum())}/{len(target_rows)} handle targets resolved",
        },
        {
            "gate_id": "G5_raw_graph_runtime_inputs_present",
            "gate_question": "Are branch-specific raw graph inputs available for Leiden replay?",
            "status": "pass" if graph_ok else "blocked_missing_raw_graph_inputs",
            "evidence": "; ".join(
                f"{row.branch}:{row.runtime_graph_status}"
                for row in graph_rows.itertuples(index=False)
            ),
        },
        {
            "gate_id": "G6_attempt_plan_frozen",
            "gate_question": "Is a strict-core symmetric candidate/control attempt plan frozen?",
            "status": "pass" if not attempt_plan.empty else "fail_empty_attempt_plan",
            "evidence": f"{len(attempt_plan)} planned attempts",
        },
        {
            "gate_id": "G7_claim_boundary_closed",
            "gate_question": "Are replay, route/wall, quality/cost, and method-success claims closed?",
            "status": "closed_by_design",
            "evidence": CLAIM_BOUNDARY,
        },
    ]
    return _with_claim_columns(pd.DataFrame(rows))


def _runner_config_template(
    *,
    nano_root: Path,
    panel_dir: Path,
    external_landscape_dir: Path,
    output_dir: Path,
    method_seeds: tuple[int, ...],
    analysis_tier: str,
    graph_rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_endpoint_replay_runner_config_template.v1",
        "status": "template_only_not_executed",
        "analysis_tier": analysis_tier,
        "method_seeds": list(method_seeds),
        "panel_dir": _rel(panel_dir),
        "external_landscape_dir": _rel(external_landscape_dir),
        "readiness_output_dir": _rel(output_dir),
        "nano_root": str(nano_root),
        "required_runtime_inputs": [
            {
                "branch": str(row["branch"]),
                "original_node_manifest": str(row["original_node_manifest_path"]),
                "original_int_edges": str(row["original_int_edges_path"]),
                "runtime_node_manifest": str(row["runtime_node_manifest_path"]),
                "runtime_int_edges": str(row["runtime_int_edges_path"]),
                "runtime_graph_source": str(row["runtime_graph_source"]),
                "runtime_graph_status": str(row["runtime_graph_status"]),
            }
            for _, row in graph_rows.iterrows()
        ],
        "success_readout": {
            "unit": "endpoint_family_signature_distance",
            "single_endpoint_hit_allowed": False,
            "required_future_outputs": [
                "replay_config.json",
                "endpoint_replay_attempt_rows.csv",
                "endpoint_signature_rows.csv",
                "endpoint_replay_case_summary.csv",
                "endpoint_replay_control_sensitivity_summary.csv",
            ],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _report(
    *,
    summary: dict[str, Any],
    output_dir: Path,
    graph_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    missing_graph_lines = []
    for row in graph_rows.itertuples(index=False):
        missing_graph_lines.append(
            f"- {row.branch}: runtime `{row.runtime_graph_status}` via "
            f"`{row.runtime_node_manifest_path}` and `{row.runtime_int_edges_path}`; "
            f"original `/data` paths exist={row.original_node_manifest_exists and row.original_int_edges_exists}"
        )
    gate_lines = [
        f"| {row.gate_id} | {row.status} | {row.evidence} |"
        for row in gate_matrix.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# NanoClustering Endpoint Replay Readiness",
            "",
            "## Scope",
            "",
            CLAIM_BOUNDARY,
            "",
            "## Readiness Summary",
            "",
            f"- Readiness: `{summary['readiness']}`",
            f"- Analysis tier: `{summary['analysis_tier']}`",
            f"- Strict-core cases planned: {summary['planned_case_count']}",
            f"- Planned symmetric attempts: {summary['planned_attempt_count']}",
            f"- Endpoint target handles checked: {summary['endpoint_target_handle_count']}",
            f"- Endpoint target handles resolved: {summary['endpoint_target_resolved_count']}",
            f"- Missing/blocking input rows: {summary['missing_input_count']}",
            "",
            "## Raw Graph Runtime Inputs",
            "",
            *missing_graph_lines,
            "",
            "The adjacent NanoClustering membership endpoints are present. If the original `/data` graph paths from `run_request.json` are not mounted, the runner accepts only the mirrored sidecar candidate graph package whose node count matches the endpoint registry; unrelated active hierarchy graphs are not substituted.",
            "",
            "## Gates",
            "",
            "| gate | status | evidence |",
            "| --- | --- | --- |",
            *gate_lines,
            "",
            "## Artifacts",
            "",
            f"- `{_rel(output_dir / INPUT_MANIFEST_CSV)}`",
            f"- `{_rel(output_dir / GRAPH_INPUT_ROWS_CSV)}`",
            f"- `{_rel(output_dir / ENDPOINT_TARGET_ROWS_CSV)}`",
            f"- `{_rel(output_dir / ATTEMPT_PLAN_ROWS_CSV)}`",
            f"- `{_rel(output_dir / MISSING_INPUT_ROWS_CSV)}`",
            f"- `{_rel(output_dir / GATE_MATRIX_CSV)}`",
            f"- `{_rel(output_dir / RUNNER_CONFIG_TEMPLATE_JSON)}`",
            "",
            "## Next Valid Step",
            "",
            "Execute endpoint replay only against the runtime graph paths frozen in the config template, and keep the endpoint-family signature readout as the success unit.",
            "",
        ]
    )


def build_outputs(
    *,
    panel_dir: Path,
    external_landscape_dir: Path,
    nano_root: Path,
    output_dir: Path,
    method_seeds: tuple[int, ...],
    analysis_tier: str,
) -> dict[str, Any]:
    case_rows = _read_csv(panel_dir / PANEL_CASE_ROWS_CSV)
    role_rows = _read_csv(panel_dir / PANEL_ROLE_ROWS_CSV)
    event_role_rows = _read_csv(panel_dir / PANEL_EVENT_ROLE_ROWS_CSV)
    endpoint_signatures = _read_csv(panel_dir / PANEL_ENDPOINT_SIGNATURE_ROWS_CSV)
    endpoint_contract = _read_csv(panel_dir / PANEL_ENDPOINT_REPLAY_CONTRACT_CSV)
    endpoint_registry = _read_csv(external_landscape_dir / EXTERNAL_ENDPOINT_REGISTRY_CSV)

    input_manifest = _required_artifact_rows(
        panel_dir=panel_dir,
        external_landscape_dir=external_landscape_dir,
        nano_root=nano_root,
    )
    graph_rows = _graph_input_rows(nano_root=nano_root, endpoint_registry=endpoint_registry)
    target_rows = _endpoint_target_rows(
        endpoint_signatures=endpoint_signatures,
        endpoint_registry=endpoint_registry,
    )
    attempt_plan = _attempt_plan_rows(
        case_rows=case_rows,
        role_rows=role_rows,
        endpoint_signatures=endpoint_signatures,
        graph_rows=graph_rows,
        method_seeds=method_seeds,
        analysis_tier=analysis_tier,
    )
    missing_inputs = _missing_input_rows(
        input_manifest=input_manifest,
        graph_rows=graph_rows,
        target_rows=target_rows,
    )
    gate_matrix = _gate_matrix(
        input_manifest=input_manifest,
        graph_rows=graph_rows,
        target_rows=target_rows,
        attempt_plan=attempt_plan,
        case_rows=case_rows,
        endpoint_signatures=endpoint_signatures,
        analysis_tier=analysis_tier,
    )

    graph_ok = bool(
        graph_rows["runtime_graph_status"].astype(str).str.startswith("ready_").all()
    )
    target_ok = (
        not target_rows.empty
        and bool(target_rows["cluster_label_present"].astype(bool).all())
        and bool(target_rows["membership_path_exists"].astype(bool).all())
        and bool(target_rows["parsed_handle"].astype(bool).all())
    )
    readiness = (
        "ready_for_endpoint_replay_execution"
        if graph_ok and target_ok
        else (
            "blocked_missing_endpoint_target_membership"
            if not target_ok
            else "blocked_missing_runtime_graph_inputs"
        )
    )
    planned_case_count = int(
        case_rows["analysis_tier"].astype(str).eq(analysis_tier).sum()
    )
    summary = {
        "schema": "nanoclustering_endpoint_replay_readiness.v1",
        "readiness": readiness,
        "analysis_tier": analysis_tier,
        "method_seed_count": len(method_seeds),
        "planned_case_count": planned_case_count,
        "planned_role_count": int(
            role_rows["analysis_tier"].astype(str).eq(analysis_tier).sum()
        ),
        "planned_attempt_count": int(len(attempt_plan)),
        "panel_case_count": int(len(case_rows)),
        "endpoint_signature_row_count": int(len(endpoint_signatures)),
        "event_role_row_count": int(len(event_role_rows)),
        "endpoint_contract_row_count": int(len(endpoint_contract)),
        "endpoint_target_handle_count": int(len(target_rows)),
        "endpoint_target_resolved_count": int(
            target_rows["cluster_label_present"].astype(bool).sum()
        ),
        "unique_endpoint_membership_run_count": int(target_rows["run_id"].nunique()),
        "runtime_graph_branch_count": int(len(graph_rows)),
        "runtime_graph_ready_branch_count": int(
            graph_rows["runtime_graph_status"].astype(str).str.startswith("ready_").sum()
        ),
        "missing_input_count": int(len(missing_inputs)),
        "gate_status_counts": {
            str(k): int(v) for k, v in gate_matrix["status"].value_counts().to_dict().items()
        },
        "panel_dir": _rel(panel_dir),
        "external_landscape_dir": _rel(external_landscape_dir),
        "nano_root": str(nano_root),
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(input_manifest, output_dir / INPUT_MANIFEST_CSV)
    _write_csv(graph_rows, output_dir / GRAPH_INPUT_ROWS_CSV)
    _write_csv(target_rows, output_dir / ENDPOINT_TARGET_ROWS_CSV)
    _write_csv(attempt_plan, output_dir / ATTEMPT_PLAN_ROWS_CSV)
    _write_csv(missing_inputs, output_dir / MISSING_INPUT_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / RUNNER_CONFIG_TEMPLATE_JSON).write_text(
        json.dumps(
            _json_safe(
                _runner_config_template(
                    nano_root=nano_root,
                    panel_dir=panel_dir,
                    external_landscape_dir=external_landscape_dir,
                    output_dir=output_dir,
                    method_seeds=method_seeds,
                    analysis_tier=analysis_tier,
                    graph_rows=graph_rows,
                )
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "script": _rel(Path(__file__)),
        "panel_dir": _rel(panel_dir),
        "external_landscape_dir": _rel(external_landscape_dir),
        "nano_root": str(nano_root),
        "output_dir": _rel(output_dir),
        "method_seeds": list(method_seeds),
        "analysis_tier": analysis_tier,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            output_dir=output_dir,
            graph_rows=graph_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument(
        "--external-landscape-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_LANDSCAPE_DIR,
    )
    parser.add_argument("--nano-root", type=Path, default=DEFAULT_NANO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method-seeds", default=",".join(map(str, DEFAULT_METHOD_SEEDS)))
    parser.add_argument("--analysis-tier", default=DEFAULT_ANALYSIS_TIER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_outputs(
        panel_dir=args.panel_dir,
        external_landscape_dir=args.external_landscape_dir,
        nano_root=args.nano_root,
        output_dir=args.output_dir,
        method_seeds=_parse_seed_list(args.method_seeds),
        analysis_tier=str(args.analysis_tier),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
