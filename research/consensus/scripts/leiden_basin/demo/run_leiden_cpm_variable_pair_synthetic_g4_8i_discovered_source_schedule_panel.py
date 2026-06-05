#!/usr/bin/env python3
"""Run a fresh discovered-source schedule panel after G4.8H.

G4.8H freezes a target-free source-discovery rule over materialized G4.8G
endpoint pools. This G4.8I runner moves one step forward: it builds a fresh
predeclared edge-mid direct/host panel, runs ordinary Leiden+CPM plus the
frozen G4.3 bridge-release handle, discovers sources from target-free
endpoint/source-local features, and accounts for the G4.6 schedule using only
those discovered sources.

This is still a synthetic schedule diagnostic. The source rule is fixed; the
panel is predeclared; oracle source-signature reads are not used to decide
whether a handle fires. It does not promote walls/pathways, quality/cost value,
NanoClustering replay, or an algorithm-level method claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression import (
    DIRECT_PAIR_SUPPORT_MIN,
    SOURCE_NEUTRAL_DELTA_ABS_MAX,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_8f_centerline_signature_audit import (
    _endpoint_features,
    _median_or_none,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_8h_source_discovery_smoke import (
    DECISION_INPUT_COLUMNS,
    EVALUATION_ONLY_COLUMNS,
    SOURCE_DISCOVERY_RULE_ID,
    _bool,
    _float,
    _set_text,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    BASELINE_RUNS_CSV,
    CONFIG_JSON as G4_3_CONFIG_JSON,
    ENDPOINT_SUMMARY_CSV,
    GRAPH_EDGES_CSV,
    GRAPH_MANIFEST_CSV,
    HANDLE_POLICY_SUMMARY_CSV,
    HANDLE_RUNS_CSV,
    PANEL_CASES_CSV,
    REPORT_MD as G4_3_REPORT_MD,
    SUMMARY_JSON as G4_3_SUMMARY_JSON,
    VARIANT_GATE_ROWS_CSV,
    CLAIM_BOUNDARY as G4_3_CLAIM_BOUNDARY,
    HANDLE_POLICIES,
    PanelCase,
    _endpoint_summary,
    _graph_manifest_and_edges,
    _handle_policy_summary,
    _panel_case_to_synthetic,
    _run_baseline,
    _run_handles,
    _summary as _g4_3_summary,
    _variant_gate_rows,
    _write_report as _write_g4_3_report,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8i_discovered_source_schedule_panel_v1_20260604"
)

G4_3_DIRNAME = "g4_3_handle_probe"
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8i_panel_design.csv"
ENDPOINT_DISCOVERY_ROWS_CSV = (
    "variable_pair_synthetic_g4_8i_endpoint_discovery_rows.csv"
)
SCHEDULE_RUN_ROWS_CSV = "variable_pair_synthetic_g4_8i_schedule_run_rows.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8i_case_summary.csv"
CONTEXT_SUMMARY_CSV = "variable_pair_synthetic_g4_8i_context_summary.csv"
ROLE_SUMMARY_CSV = "variable_pair_synthetic_g4_8i_role_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8i_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8i_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8i_report.md"

HANDLE_POLICY = "bridge_context_release_without_pair_merge"
SCHEDULE_RULE_ID = "restart_then_g4_8h_discovered_source_then_one_g4_3_handle_v1"
HANDLE_UNIT_COST = 1.0
RESTART_UNIT_COST = 1.0
EPS = 1.0e-12

HOST_CONTEXTS = (
    ("direct_mid_host_low", 1.08, 1.23),
    ("direct_mid_host_high", 1.08, 1.27),
    ("direct_low_host_mid", 1.06, 1.25),
    ("direct_high_host_mid", 1.10, 1.25),
)
PAIR_BRIDGE_VALUES = tuple(round(1.320 + step * 0.005, 3) for step in range(13))
DIAGONAL_START_PAIR = 1.32
DIAGONAL_START_BRIDGE = 1.44
DIAGONAL_SLOPE = 1.0 / 3.0

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8I discovered-source schedule panel only; a "
    "fresh predeclared direct/host edge-mid panel runs ordinary Leiden+CPM and "
    "the frozen G4.3 bridge-release handle, then drives schedule accounting "
    "with the frozen G4.8H target-free source-discovery rule. No selector "
    "retuning, no oracle source-signature reads for schedule decisions, no "
    "wall or pathway promotion, no quality/cost value, no NanoClustering "
    "replay, and no algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8i_discovered_source_schedule_panel"
WALL_PROMOTION_STATUS = "not_promoted_discovered_schedule_panel_only"
METHOD_STATUS = "discovered_source_schedule_panel_not_method_claim"


@dataclass(frozen=True)
class SchedulePanelCase:
    case_id: str
    context_id: str
    context_index: int
    direct_weight: float
    host_clique_weight: float
    pair_bridge_weight: float
    bridge_host_weight: float
    centerline_index: int
    expected_role_symbol: str

    def to_panel_case(self) -> PanelCase:
        return PanelCase(
            case_id=self.case_id,
            panel_role="positive_holdout",
            expected_gate=f"discovered_schedule_expected_{self.expected_role_symbol}",
            direct_weight=self.direct_weight,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=self.host_clique_weight,
            note=(
                "Fresh edge-mid context panel for target-free discovered-source "
                "schedule accounting."
            ),
        )


def _diagonal_bridge_host(pair_bridge: float) -> float:
    return round(
        DIAGONAL_START_BRIDGE
        + (float(pair_bridge) - DIAGONAL_START_PAIR) * DIAGONAL_SLOPE,
        3,
    )


def _expected_role(centerline_index: int) -> str:
    return ("R", "T", "N")[centerline_index % 3]


def _case_id(
    *,
    context_id: str,
    pair_bridge: float,
    bridge_host: float,
    expected_role: str,
) -> str:
    return (
        f"g4_8i_{context_id}_pair{int(round(pair_bridge * 1000)):04d}"
        f"_bridge{int(round(bridge_host * 1000)):04d}_exp{expected_role}"
    )


def _panel_cases() -> tuple[SchedulePanelCase, ...]:
    cases: list[SchedulePanelCase] = []
    for context_index, (context_id, direct_weight, host_clique_weight) in enumerate(
        HOST_CONTEXTS
    ):
        for centerline_index, pair_bridge in enumerate(PAIR_BRIDGE_VALUES):
            bridge_host = _diagonal_bridge_host(pair_bridge)
            expected_role = _expected_role(centerline_index)
            cases.append(
                SchedulePanelCase(
                    case_id=_case_id(
                        context_id=context_id,
                        pair_bridge=pair_bridge,
                        bridge_host=bridge_host,
                        expected_role=expected_role,
                    ),
                    context_id=context_id,
                    context_index=context_index,
                    direct_weight=float(direct_weight),
                    host_clique_weight=float(host_clique_weight),
                    pair_bridge_weight=float(pair_bridge),
                    bridge_host_weight=float(bridge_host),
                    centerline_index=centerline_index,
                    expected_role_symbol=expected_role,
                )
            )
    return tuple(cases)


PANEL_CASES = _panel_cases()


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.drop(
        columns=[
            "route_execution_status",
            "wall_promotion_status",
            "method_status",
            "claim_boundary",
        ],
        errors="ignore",
    ).copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_design_rows() -> pd.DataFrame:
    return _claim_columns(
        pd.DataFrame(
            [
                {
                    "case_id": case.case_id,
                    "context_id": case.context_id,
                    "context_index": case.context_index,
                    "centerline_index": case.centerline_index,
                    "expected_role_symbol": case.expected_role_symbol,
                    "pair_bridge_weight": case.pair_bridge_weight,
                    "bridge_host_weight": case.bridge_host_weight,
                    "direct_weight": case.direct_weight,
                    "host_clique_weight": case.host_clique_weight,
                    "panel_role": "positive_holdout",
                    "expected_gate": (
                        f"discovered_schedule_expected_{case.expected_role_symbol}"
                    ),
                }
                for case in PANEL_CASES
            ]
        )
    )


def _run_g4_3_stage(
    *,
    output_dir: Path,
    baseline_seeds: int,
    handle_seeds: int,
    n_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_cases = tuple(case.to_panel_case() for case in PANEL_CASES)
    cases = [_panel_case_to_synthetic(case) for case in panel_cases]
    panel_rows, graph_manifest, graph_edges = _graph_manifest_and_edges(
        cases,
        panel_cases,
    )
    baseline_runs = _run_baseline(
        cases=cases,
        panel_cases=panel_cases,
        seeds=baseline_seeds,
        n_iterations=n_iterations,
    )
    endpoint_summary = _endpoint_summary(baseline_runs)
    handle_runs = _run_handles(
        cases=cases,
        panel_cases=panel_cases,
        endpoint_summary=endpoint_summary,
        seeds=handle_seeds,
        n_iterations=n_iterations,
    )
    policy_summary = _handle_policy_summary(handle_runs)
    gate_rows = _variant_gate_rows(
        endpoint_summary=endpoint_summary,
        policy_summary=policy_summary,
        panel_cases=panel_cases,
    )
    _write_csv(panel_rows, output_dir / PANEL_CASES_CSV)
    _write_csv(graph_manifest, output_dir / GRAPH_MANIFEST_CSV)
    _write_csv(graph_edges, output_dir / GRAPH_EDGES_CSV)
    _write_csv(baseline_runs, output_dir / BASELINE_RUNS_CSV)
    _write_csv(endpoint_summary, output_dir / ENDPOINT_SUMMARY_CSV)
    _write_csv(handle_runs, output_dir / HANDLE_RUNS_CSV)
    _write_csv(policy_summary, output_dir / HANDLE_POLICY_SUMMARY_CSV)
    _write_csv(gate_rows, output_dir / VARIANT_GATE_ROWS_CSV)
    summary = _g4_3_summary(
        output_dir=output_dir,
        baseline_runs=baseline_runs,
        endpoint_summary=endpoint_summary,
        handle_runs=handle_runs,
        policy_summary=policy_summary,
        gate_rows=gate_rows,
    )
    (output_dir / G4_3_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8i_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in PANEL_CASES],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_8i_claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / G4_3_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_g4_3_report(
        output_dir=output_dir,
        summary=summary,
        gate_rows=gate_rows,
        policy_summary=policy_summary,
    )
    return summary


def _endpoint_discovery_rows(
    *,
    panel_design: pd.DataFrame,
    g4_3_dir: Path,
) -> pd.DataFrame:
    endpoints = pd.read_csv(g4_3_dir / ENDPOINT_SUMMARY_CSV)
    handle_policy = pd.read_csv(g4_3_dir / HANDLE_POLICY_SUMMARY_CSV)
    features = pd.DataFrame(
        [
            _endpoint_features(signature)
            for signature in endpoints["endpoint_signature"].astype(str)
        ]
    )
    endpoints = pd.concat([endpoints.reset_index(drop=True), features], axis=1)
    panel_cols = [
        "case_id",
        "context_id",
        "context_index",
        "centerline_index",
        "expected_role_symbol",
        "direct_weight",
        "pair_bridge_weight",
        "bridge_host_weight",
        "host_clique_weight",
    ]
    rows = endpoints.merge(panel_design[panel_cols], on="case_id", how="left")
    bridge_policy = handle_policy[handle_policy["handle_policy"].eq(HANDLE_POLICY)]
    handle_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "handle_eligible",
        "released_bridge_nodes",
        "released_bridge_count",
        "changed_nodes_vs_source",
        "source_pair_coassigned",
        "initial_pair_coassigned",
        "initial_keeps_pair_relation",
        "initial_quality",
        "initial_cluster_count",
        "initial_quality_delta_vs_source",
        "initial_coassoc_distance_vs_source",
        "known_coassigned_endpoint_rate",
        "pair_coassigned_rate",
        "source_bounce_rate",
        "distinct_result_endpoint_count",
        "handle_policy_class",
        "result_quality_delta_vs_source_median",
    ]
    rows = rows.merge(
        bridge_policy[handle_cols],
        left_on=["case_id", "endpoint_signature_id"],
        right_on=["case_id", "source_endpoint_signature_id"],
        how="left",
    )
    rows["endpoint_bridge_candidate"] = [
        (not _bool(row["pair_coassigned"]))
        and int(row["pair_attached_bridge_count"]) > 0
        for row in rows.to_dict("records")
    ]
    rows["release_source_candidate"] = [
        _release_source_candidate(row) for row in rows.to_dict("records")
    ]
    rows["source_neutral_release"] = (
        rows["initial_quality_delta_vs_source"].astype(float).abs()
        <= SOURCE_NEUTRAL_DELTA_ABS_MAX
    ).fillna(False)
    rows["direct_pair_support_floor_passed"] = (
        rows["direct_weight"].astype(float) >= DIRECT_PAIR_SUPPORT_MIN
    )
    rows["ready_source_candidate"] = [
        _ready_source_candidate(row) for row in rows.to_dict("records")
    ]
    rows["source_discovery_decision_status"] = [
        _decision_status(row) for row in rows.to_dict("records")
    ]
    rows["source_discovery_rule_id"] = SOURCE_DISCOVERY_RULE_ID
    rows["decision_input_columns"] = ",".join(DECISION_INPUT_COLUMNS)
    rows["evaluation_only_columns"] = ",".join(EVALUATION_ONLY_COLUMNS)
    return _claim_columns(
        rows.sort_values(
            [
                "context_id",
                "centerline_index",
                "pair_coassigned",
                "bridge_signature_family",
                "endpoint_rank_within_case",
                "endpoint_signature_id",
            ],
            ascending=[True, True, True, True, True, True],
            kind="stable",
        )
    )


def _release_source_candidate(row: dict[str, Any]) -> bool:
    return bool(
        _bool(row["endpoint_bridge_candidate"])
        and _bool(row["handle_eligible"])
        and int(_float(row["released_bridge_count"])) > 0
        and not _bool(row["initial_pair_coassigned"])
        and _bool(row["initial_keeps_pair_relation"])
    )


def _ready_source_candidate(row: dict[str, Any]) -> bool:
    return bool(
        _bool(row["release_source_candidate"])
        and abs(_float(row["initial_quality_delta_vs_source"], default=1.0))
        <= SOURCE_NEUTRAL_DELTA_ABS_MAX
        and _float(row["direct_weight"]) >= DIRECT_PAIR_SUPPORT_MIN
    )


def _decision_status(row: dict[str, Any]) -> str:
    if _bool(row["ready_source_candidate"]):
        return "ready_source_discovered"
    if _bool(row["release_source_candidate"]):
        if abs(_float(row["initial_quality_delta_vs_source"], default=1.0)) > SOURCE_NEUTRAL_DELTA_ABS_MAX:
            return "release_source_suppressed_nonneutral"
        if _float(row["direct_weight"]) < DIRECT_PAIR_SUPPORT_MIN:
            return "release_source_suppressed_low_direct_support"
        return "release_source_suppressed_other"
    if _bool(row["endpoint_bridge_candidate"]):
        return "endpoint_bridge_candidate_no_valid_release"
    if _bool(row["pair_coassigned"]):
        return "coassigned_endpoint_not_source"
    return "separated_endpoint_without_pair_bridge"


def _schedule_run_rows(
    *,
    g4_3_dir: Path,
    endpoint_discovery: pd.DataFrame,
) -> pd.DataFrame:
    baseline_runs = pd.read_csv(g4_3_dir / BASELINE_RUNS_CSV)
    decision_cols = [
        "case_id",
        "endpoint_signature_id",
        "ready_source_candidate",
        "release_source_candidate",
        "known_coassigned_endpoint_rate",
        "pair_coassigned_rate",
        "source_discovery_decision_status",
    ]
    decisions = endpoint_discovery[decision_cols].copy()
    rows = baseline_runs.merge(
        decisions,
        on=["case_id", "endpoint_signature_id"],
        how="left",
    )
    schedule_rows: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        pair_coassigned = _bool(row["pair_coassigned"])
        ready = _bool(row["ready_source_candidate"])
        release = _bool(row["release_source_candidate"])
        handle_rate = _float(row["known_coassigned_endpoint_rate"])
        pair_rate = _float(row["pair_coassigned_rate"])
        if pair_coassigned:
            schedule_hit_probability = 1.0
            schedule_pair_coassigned_probability = 1.0
            handle_applied = False
            status = "coassigned_endpoint_without_handle"
        elif ready:
            schedule_hit_probability = handle_rate
            schedule_pair_coassigned_probability = pair_rate
            handle_applied = True
            status = "discovered_ready_source_handle_applied"
        elif release:
            schedule_hit_probability = 0.0
            schedule_pair_coassigned_probability = 0.0
            handle_applied = False
            status = "release_source_not_ready_noop"
        else:
            schedule_hit_probability = 0.0
            schedule_pair_coassigned_probability = 0.0
            handle_applied = False
            status = "no_discovered_source_noop"
        schedule_rows.append(
            {
                "case_id": str(row["case_id"]),
                "panel_role": str(row["panel_role"]),
                "expected_gate": str(row["expected_gate"]),
                "start_condition": str(row["start_condition"]),
                "seed": int(row["seed"]),
                "endpoint_signature_id": str(row["endpoint_signature_id"]),
                "baseline_pair_coassigned": bool(pair_coassigned),
                "release_source_candidate": bool(release),
                "ready_source_candidate": bool(ready),
                "handle_applied": bool(handle_applied),
                "handle_known_coassigned_hit_rate": float(handle_rate),
                "handle_pair_coassigned_hit_rate": float(pair_rate),
                "schedule_hit_probability": float(schedule_hit_probability),
                "schedule_pair_coassigned_probability": float(
                    schedule_pair_coassigned_probability
                ),
                "schedule_run_status": status,
                "source_discovery_decision_status": str(
                    row["source_discovery_decision_status"]
                ),
                "restart_unit_cost": RESTART_UNIT_COST,
                "expected_handle_unit_cost": (
                    HANDLE_UNIT_COST if handle_applied else 0.0
                ),
                "expected_restart_plus_handle_unit_cost": (
                    RESTART_UNIT_COST + (HANDLE_UNIT_COST if handle_applied else 0.0)
                ),
                "schedule_rule_id": SCHEDULE_RULE_ID,
            }
        )
    return _claim_columns(pd.DataFrame(schedule_rows))


def _case_summary(
    *,
    panel_design: pd.DataFrame,
    endpoint_discovery: pd.DataFrame,
    schedule_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in endpoint_discovery.groupby("case_id", sort=True):
        schedule = schedule_rows[schedule_rows["case_id"].astype(str).eq(str(case_id))]
        coassigned = group[group["pair_coassigned"].astype(bool)]
        release = group[group["release_source_candidate"].astype(bool)]
        ready = group[group["ready_source_candidate"].astype(bool)]
        release_set = set(release["endpoint_signature_id"].astype(str))
        ready_set = set(ready["endpoint_signature_id"].astype(str))
        baseline_pair_share = float(
            schedule["baseline_pair_coassigned"].astype(bool).mean()
        )
        schedule_hit = float(schedule["schedule_hit_probability"].mean())
        schedule_pair_hit = float(
            schedule["schedule_pair_coassigned_probability"].mean()
        )
        handle_apply_rate = float(schedule["handle_applied"].astype(bool).mean())
        expected_unit = float(
            schedule["expected_restart_plus_handle_unit_cost"].mean()
        )
        baseline_expected_runs = (
            float(1.0 / baseline_pair_share) if baseline_pair_share > 0.0 else None
        )
        schedule_expected_units = (
            float(expected_unit / schedule_hit) if schedule_hit > 0.0 else None
        )
        ratio = (
            float(baseline_expected_runs / schedule_expected_units)
            if baseline_expected_runs is not None and schedule_expected_units
            else None
        )
        design = panel_design[panel_design["case_id"].astype(str).eq(str(case_id))].iloc[0]
        observed_role = _observed_role(
            baseline_pair_share=baseline_pair_share,
            release_count=len(release_set),
            ready_count=len(ready_set),
        )
        rows.append(
            {
                "case_id": str(case_id),
                "context_id": str(design["context_id"]),
                "context_index": int(design["context_index"]),
                "centerline_index": int(design["centerline_index"]),
                "expected_role_symbol": str(design["expected_role_symbol"]),
                "role_symbol": observed_role,
                "pair_bridge_weight": float(design["pair_bridge_weight"]),
                "bridge_host_weight": float(design["bridge_host_weight"]),
                "direct_weight": float(design["direct_weight"]),
                "host_clique_weight": float(design["host_clique_weight"]),
                "endpoint_signature_count": int(len(group)),
                "coassigned_endpoint_count": int(len(coassigned)),
                "release_source_candidate_count": int(len(release_set)),
                "ready_source_candidate_count": int(len(ready_set)),
                "release_source_candidate_ids": _set_text(release_set),
                "ready_source_candidate_ids": _set_text(ready_set),
                "baseline_pair_coassigned_run_share": baseline_pair_share,
                "schedule_known_coassigned_hit_rate": schedule_hit,
                "schedule_pair_coassigned_hit_rate": schedule_pair_hit,
                "discovered_source_availability_rate": handle_apply_rate,
                "target_free_noop_run_share": float(
                    schedule["schedule_run_status"]
                    .isin(["release_source_not_ready_noop", "no_discovered_source_noop"])
                    .mean()
                ),
                "expected_restart_plus_handle_unit_per_restart": expected_unit,
                "baseline_expected_runs_to_pair_coassigned": baseline_expected_runs,
                "schedule_expected_units_to_known_coassigned": schedule_expected_units,
                "baseline_over_discovered_schedule_unit_ratio": ratio,
                "ready_source_initial_delta_median": _median_or_none(
                    ready["initial_quality_delta_vs_source"]
                ),
                "ready_source_handle_hit_rate_median": _median_or_none(
                    ready["known_coassigned_endpoint_rate"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["observed_role_matches_expected"] = frame["role_symbol"].astype(str).eq(
        frame["expected_role_symbol"].astype(str)
    )
    frame["schedule_expectation_status"] = [
        _schedule_expectation_status(row) for row in frame.to_dict("records")
    ]
    frame["schedule_expectation_passed"] = frame["schedule_expectation_status"].eq(
        "schedule_expectation_passed"
    )
    return _claim_columns(frame.sort_values(["context_id", "centerline_index"], kind="stable"))


def _observed_role(
    *,
    baseline_pair_share: float,
    release_count: int,
    ready_count: int,
) -> str:
    if baseline_pair_share >= 1.0 - EPS and release_count == 0:
        return "T"
    if ready_count >= 8:
        return "R"
    if release_count > 0 and ready_count == 0:
        return "N"
    return "U"


def _schedule_expectation_status(row: dict[str, Any]) -> str:
    if str(row["role_symbol"]) != str(row["expected_role_symbol"]):
        return "role_mismatch"
    role = str(row["role_symbol"])
    release_count = int(row["release_source_candidate_count"])
    ready_count = int(row["ready_source_candidate_count"])
    baseline = float(row["baseline_pair_coassigned_run_share"])
    schedule = float(row["schedule_known_coassigned_hit_rate"])
    ratio = row["baseline_over_discovered_schedule_unit_ratio"]
    if role == "R":
        if (
            release_count == 8
            and ready_count == 8
            and schedule >= 1.0 - EPS
            and ratio is not None
            and float(ratio) > 1.0
        ):
            return "schedule_expectation_passed"
        return "ready_schedule_accounting_failed"
    if role == "N":
        if release_count == 4 and ready_count == 0 and abs(schedule - baseline) <= EPS:
            return "schedule_expectation_passed"
        return "nonrobust_noop_or_leak_failed"
    if role == "T":
        if release_count == 0 and ready_count == 0 and schedule >= 1.0 - EPS:
            return "schedule_expectation_passed"
        return "target_saturation_noop_failed"
    return "unknown_role"


def _group_summary(case_rows: pd.DataFrame, group_col: str, key_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in case_rows.groupby(group_col, sort=True):
        rows.append(
            {
                key_col: str(key),
                "case_count": int(len(group)),
                "schedule_expectation_pass_count": int(
                    group["schedule_expectation_passed"].astype(bool).sum()
                ),
                "observed_role_match_count": int(
                    group["observed_role_matches_expected"].astype(bool).sum()
                ),
                "release_source_candidate_count_sum": int(
                    group["release_source_candidate_count"].sum()
                ),
                "ready_source_candidate_count_sum": int(
                    group["ready_source_candidate_count"].sum()
                ),
                "schedule_known_coassigned_hit_rate_median": float(
                    group["schedule_known_coassigned_hit_rate"].median()
                ),
                "discovered_source_availability_rate_median": float(
                    group["discovered_source_availability_rate"].median()
                ),
                "expected_restart_plus_handle_unit_per_restart_median": float(
                    group[
                        "expected_restart_plus_handle_unit_per_restart"
                    ].median()
                ),
                "role_sequence": "".join(
                    group.sort_values("centerline_index")["role_symbol"].astype(str)
                ),
                "expected_role_sequence": "".join(
                    group.sort_values("centerline_index")[
                        "expected_role_symbol"
                    ].astype(str)
                ),
                "status_counts": json.dumps(
                    group["schedule_expectation_status"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    endpoint_discovery: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> dict[str, Any]:
    case_count = int(len(case_rows))
    pass_count = int(case_rows["schedule_expectation_passed"].astype(bool).sum())
    role_match_count = int(
        case_rows["observed_role_matches_expected"].astype(bool).sum()
    )
    if pass_count == case_count and role_match_count == case_count:
        schedule_status = "discovered_source_schedule_panel_passed"
    elif role_match_count == case_count:
        schedule_status = "roles_passed_schedule_failed"
    else:
        schedule_status = "discovered_source_schedule_panel_failed"
    return {
        "schema": "variable_pair_synthetic_g4_8i_discovered_source_schedule_panel_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "schedule_status": schedule_status,
        "output_dir": str(output_dir),
        "case_count": case_count,
        "endpoint_discovery_row_count": int(len(endpoint_discovery)),
        "schedule_run_row_count": int(len(schedule_rows)),
        "observed_role_match_count": role_match_count,
        "schedule_expectation_pass_count": pass_count,
        "schedule_status_counts": case_rows[
            "schedule_expectation_status"
        ].value_counts().to_dict(),
        "role_counts": case_rows["role_symbol"].value_counts().to_dict(),
        "release_source_candidate_count_by_role": case_rows.groupby("role_symbol")[
            "release_source_candidate_count"
        ].sum().astype(int).to_dict(),
        "ready_source_candidate_count_by_role": case_rows.groupby("role_symbol")[
            "ready_source_candidate_count"
        ].sum().astype(int).to_dict(),
        "schedule_hit_rate_median_by_role": case_rows.groupby("role_symbol")[
            "schedule_known_coassigned_hit_rate"
        ].median().to_dict(),
        "discovered_source_availability_median_by_role": case_rows.groupby(
            "role_symbol"
        )["discovered_source_availability_rate"].median().to_dict(),
        "restart_plus_handle_unit_median_by_role": case_rows.groupby("role_symbol")[
            "expected_restart_plus_handle_unit_per_restart"
        ].median().to_dict(),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "context_summary_row_count": int(len(context_summary)),
        "role_summary_row_count": int(len(role_summary)),
        "source_discovery_rule_id": SOURCE_DISCOVERY_RULE_ID,
        "schedule_rule_id": SCHEDULE_RULE_ID,
        "recommended_next_gate": _recommended_next_gate(schedule_status),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _recommended_next_gate(schedule_status: str) -> str:
    if schedule_status == "discovered_source_schedule_panel_passed":
        return (
            "Freeze discovered-source schedule accounting and design the next "
            "fresh panel around first failure modes: context expansion, "
            "off-center offsets, or a tiny real-data analog screen."
        )
    if schedule_status == "roles_passed_schedule_failed":
        return (
            "Inspect discovered-source schedule accounting failures before "
            "expanding contexts or offsets."
        )
    return (
        "Do not expand schedule claims; revisit construction-read or "
        "source-discovery assumptions on the failed fresh panel."
    )


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8I Discovered-Source Schedule Panel",
        "",
        f"- status: `{summary['status']}`",
        f"- schedule_status: {summary['schedule_status']}",
        f"- case_count: {summary['case_count']}",
        f"- observed_role_match_count: {summary['observed_role_match_count']}",
        f"- schedule_expectation_pass_count: {summary['schedule_expectation_pass_count']}",
        f"- schedule_status_counts: {summary['schedule_status_counts']}",
        f"- source_discovery_rule_id: `{SOURCE_DISCOVERY_RULE_ID}`",
        f"- schedule_rule_id: `{SCHEDULE_RULE_ID}`",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Context Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            context_summary[
                [
                    "context_id",
                    "case_count",
                    "schedule_expectation_pass_count",
                    "observed_role_match_count",
                    "release_source_candidate_count_sum",
                    "ready_source_candidate_count_sum",
                    "schedule_known_coassigned_hit_rate_median",
                    "discovered_source_availability_rate_median",
                    "expected_restart_plus_handle_unit_per_restart_median",
                    "role_sequence",
                    "expected_role_sequence",
                    "status_counts",
                ]
            ]
        )
    )
    lines.extend(["", "## Role Summary", ""])
    lines.extend(
        _markdown_table(
            role_summary[
                [
                    "role_symbol",
                    "case_count",
                    "schedule_expectation_pass_count",
                    "release_source_candidate_count_sum",
                    "ready_source_candidate_count_sum",
                    "schedule_known_coassigned_hit_rate_median",
                    "discovered_source_availability_rate_median",
                    "expected_restart_plus_handle_unit_per_restart_median",
                    "status_counts",
                ]
            ]
        )
    )
    lines.extend(["", "## Case Summary", ""])
    lines.extend(
        _markdown_table(
            case_rows[
                [
                    "context_id",
                    "centerline_index",
                    "expected_role_symbol",
                    "role_symbol",
                    "release_source_candidate_count",
                    "ready_source_candidate_count",
                    "baseline_pair_coassigned_run_share",
                    "schedule_known_coassigned_hit_rate",
                    "discovered_source_availability_rate",
                    "expected_restart_plus_handle_unit_per_restart",
                    "baseline_over_discovered_schedule_unit_ratio",
                    "schedule_expectation_status",
                ]
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8I is a fresh synthetic schedule-accounting panel. It uses "
                "the frozen G4.8H target-free source-discovery rule to decide "
                "which endpoints receive the frozen G4.3 handle. It does not "
                "use oracle source-signature reads for schedule decisions, and "
                "does not establish wall/pathway, quality/cost, NanoClustering, "
                "or algorithm-level method claims."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    cols = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in cols) + " |")
    return lines


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    g4_3_dir = output_dir / G4_3_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_design = _panel_design_rows()
    _write_csv(panel_design, output_dir / PANEL_DESIGN_CSV)
    g4_3_summary = _run_g4_3_stage(
        output_dir=g4_3_dir,
        baseline_seeds=int(args.baseline_seeds),
        handle_seeds=int(args.handle_seeds),
        n_iterations=int(args.n_iterations),
    )
    endpoint_discovery = _endpoint_discovery_rows(
        panel_design=panel_design,
        g4_3_dir=g4_3_dir,
    )
    schedule_rows = _schedule_run_rows(
        g4_3_dir=g4_3_dir,
        endpoint_discovery=endpoint_discovery,
    )
    case_rows = _case_summary(
        panel_design=panel_design,
        endpoint_discovery=endpoint_discovery,
        schedule_rows=schedule_rows,
    )
    context_summary = _group_summary(case_rows, "context_id", "context_id")
    role_summary = _group_summary(case_rows, "role_symbol", "role_symbol")
    _write_csv(endpoint_discovery, output_dir / ENDPOINT_DISCOVERY_ROWS_CSV)
    _write_csv(schedule_rows, output_dir / SCHEDULE_RUN_ROWS_CSV)
    _write_csv(case_rows, output_dir / CASE_SUMMARY_CSV)
    _write_csv(context_summary, output_dir / CONTEXT_SUMMARY_CSV)
    _write_csv(role_summary, output_dir / ROLE_SUMMARY_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        endpoint_discovery=endpoint_discovery,
        schedule_rows=schedule_rows,
        case_rows=case_rows,
        context_summary=context_summary,
        role_summary=role_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8i_discovered_source_schedule_panel_config.v1",
        "output_dir": str(output_dir),
        "g4_3_dir": str(g4_3_dir),
        "host_contexts": [
            {
                "context_id": context_id,
                "direct_weight": direct_weight,
                "host_clique_weight": host_clique_weight,
            }
            for context_id, direct_weight, host_clique_weight in HOST_CONTEXTS
        ],
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "diagonal_rule": (
            "bridge_host = 1.44 + (pair_bridge - 1.32) / 3, rounded to 0.001"
        ),
        "expected_role_pattern": "R,T,N repeated from pair_bridge=1.320",
        "source_discovery_rule_id": SOURCE_DISCOVERY_RULE_ID,
        "schedule_rule_id": SCHEDULE_RULE_ID,
        "decision_input_columns": list(DECISION_INPUT_COLUMNS),
        "evaluation_only_columns": list(EVALUATION_ONLY_COLUMNS),
        "baseline_seeds": int(args.baseline_seeds),
        "handle_seeds": int(args.handle_seeds),
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
        case_rows=case_rows,
        context_summary=context_summary,
        role_summary=role_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-seeds", type=int, default=16)
    parser.add_argument("--handle-seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
