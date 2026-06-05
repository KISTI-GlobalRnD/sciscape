#!/usr/bin/env python3
"""Read out held-out local behavior for the frozen G4.8 analog panel.

This consumes the frozen NanoClustering G4.8 local analog validation panel and
the existing local-ablation seed runs. It does not run Leiden. The readout
splits existing local seeds into discovery and held-out halves, materializes
source-signature proxies from endpoint signatures, and checks whether frozen
strata remain visible under the held-out local split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_PANEL_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_analog_validation_panel_gamma1e5_20260604"
)
DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_validation_readout_gamma1e5_20260604"
)

PANEL_ROWS_CSV = "nanoclustering_g4_8_local_analog_validation_panel_rows.csv"
LOCAL_SEED_RUNS_CSV = "nanoclustering_symmetric_object_variable_pair_local_ablation_seed_runs.csv"

PAIR_ROWS_CSV = "nanoclustering_g4_8_local_validation_readout_pair_rows.csv"
SEED_SPLIT_ROWS_CSV = "nanoclustering_g4_8_local_validation_readout_seed_split_rows.csv"
START_CONDITION_ROWS_CSV = "nanoclustering_g4_8_local_validation_readout_start_condition_rows.csv"
STRATUM_SUMMARY_CSV = "nanoclustering_g4_8_local_validation_readout_stratum_summary.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_local_validation_readout_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_local_validation_readout_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_local_validation_readout_summary.json"
REPORT_MD = "nanoclustering_g4_8_local_validation_readout_report.md"

GRAPH_VARIANTS = (
    "original",
    "drop_direct_edge",
    "drop_bridge_edges",
    "drop_direct_and_bridge_edges",
)
START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)
DISCOVERY_SPLIT = "discovery_seed_0_3"
HELDOUT_SPLIT = "heldout_seed_4_7"

RUN_STATUS = "read_only_nanoclustering_g4_8_local_validation_readout"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 local validation readout only; reads the frozen "
    "23-pair analog panel and existing local-ablation seed runs to materialize "
    "held-out local behavior and source-signature proxies. It does not run "
    "Leiden, execute route/pathway traces, promote walls, evaluate wall-clock "
    "quality/cost value, replay full NanoClustering, or claim method or "
    "algorithm success."
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _prefix_stats(prefix: str, values: pd.Series | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_mean": float(array.mean()),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in frame[cols].itertuples(index=False):
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _classify_condition(
    *,
    original_share: float,
    drop_direct_share: float,
    drop_bridge_share: float,
    drop_both_share: float,
    thresholds: dict[str, float],
) -> tuple[str, str]:
    suppressed_max = float(thresholds["suppressed_share_max"])
    released_min = float(thresholds["released_share_min"])
    partial_min = float(thresholds["partial_source_min"])
    partial_max = float(thresholds["partial_source_max"])
    target_min = float(thresholds["target_saturated_min"])

    direct_suppressed = float(drop_direct_share) <= suppressed_max
    both_suppressed = float(drop_both_share) <= suppressed_max
    bridge_released = float(drop_bridge_share) >= released_min

    if (
        partial_min <= float(original_share) <= partial_max
        and direct_suppressed
        and bridge_released
        and both_suppressed
    ):
        return "R_candidate", "strict_partial_release_ready_analog"
    if (
        0.0 < float(original_share) < partial_min
        and direct_suppressed
        and bridge_released
        and both_suppressed
    ):
        return "R_weak", "rare_start_release_ready_analog"
    if float(original_share) >= target_min and direct_suppressed and bridge_released and both_suppressed:
        return "T_like", "target_saturated_direct_contact_no_handle_analog"
    if float(original_share) >= target_min and direct_suppressed and float(drop_bridge_share) <= suppressed_max and both_suppressed:
        return "T_or_failure", "coupled_direct_bridge_context_failure_control"
    if float(original_share) <= suppressed_max and direct_suppressed and bridge_released and both_suppressed:
        return "N_like", "latent_release_without_original_source_control"
    if float(original_share) <= suppressed_max and direct_suppressed and float(drop_bridge_share) <= suppressed_max and both_suppressed:
        return "N_like", "no_local_source_or_release_control"
    return "mixed", "mixed_or_unclassified_source_condition"


def _expected_readout_pass(
    *,
    validation_stratum: str,
    macro_role: str,
    source_condition: str,
) -> tuple[bool, str]:
    stratum = str(validation_stratum)
    macro = str(macro_role)
    condition = str(source_condition)
    if stratum == "strict_ready":
        passed = macro == "R_candidate"
        return passed, "strict-ready should remain R_candidate"
    if stratum == "rare_ready":
        passed = macro in {"R_candidate", "R_weak"}
        return passed, "rare-ready may remain R_weak or strengthen to R_candidate"
    if stratum == "target_saturated_no_handle":
        passed = macro == "T_like"
        return passed, "target-saturated no-handle should remain T_like"
    if stratum == "latent_release_no_source_control":
        passed = condition == "latent_release_without_original_source_control"
        return passed, "latent-release control should remain no-original-source release"
    if stratum == "no_release_control":
        passed = condition == "no_local_source_or_release_control"
        return passed, "no-release control should remain no source and no release"
    if stratum == "coupled_direct_bridge_failure_control":
        passed = macro == "T_or_failure"
        return passed, "coupled direct-bridge failure should remain T_or_failure"
    return False, "unclassified validation stratum requires review"


def _seed_split(seed: int) -> str:
    if int(seed) <= 3:
        return DISCOVERY_SPLIT
    return HELDOUT_SPLIT


def _signature_stats(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {
            "original_run_count": 0,
            "original_distinct_endpoint_count": 0,
            "original_top_endpoint_share": 0.0,
            "original_coassigned_signature_count": 0,
            "original_source_endpoint_signature_proxy_count": 0,
            "original_bridge_split_source_signature_proxy_count": 0,
            "original_single_side_source_signature_proxy_count": 0,
            "original_no_bridge_separated_signature_count": 0,
            "original_mechanism_read_counts": "{}",
        }
    endpoint_counts = group["endpoint_signature_id"].value_counts()
    coassigned = group["pair_coassigned"].fillna(False).astype(bool)
    bridge_split = group["mechanism_read"].astype(str).eq("pair_separated_bridge_split")
    single_side = group["mechanism_read"].astype(str).eq("pair_separated_single_side_bridge")
    no_bridge = group["mechanism_read"].astype(str).eq("pair_separated_no_selected_bridge")
    source_proxy = bridge_split | single_side
    return {
        "original_run_count": int(len(group)),
        "original_distinct_endpoint_count": int(group["endpoint_signature_id"].nunique()),
        "original_top_endpoint_share": float(endpoint_counts.max() / len(group)),
        "original_coassigned_signature_count": int(group.loc[coassigned, "endpoint_signature_id"].nunique()),
        "original_source_endpoint_signature_proxy_count": int(
            group.loc[source_proxy, "endpoint_signature_id"].nunique()
        ),
        "original_bridge_split_source_signature_proxy_count": int(
            group.loc[bridge_split, "endpoint_signature_id"].nunique()
        ),
        "original_single_side_source_signature_proxy_count": int(
            group.loc[single_side, "endpoint_signature_id"].nunique()
        ),
        "original_no_bridge_separated_signature_count": int(
            group.loc[no_bridge, "endpoint_signature_id"].nunique()
        ),
        "original_mechanism_read_counts": json.dumps(
            _count_dict(group["mechanism_read"]),
            sort_keys=True,
        ),
    }


def _identity_columns() -> list[str]:
    return [
        "local_pair_id",
        "object_role_universe_id",
        "branch",
        "left_node_id",
        "right_node_id",
        "pair_scope",
        "counterfactual_class",
        "design_family",
        "validation_stratum",
        "validation_family",
        "analog_macro_role",
        "analog_source_condition",
    ]


def _panel_lookup(panel: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cols = [col for col in _identity_columns() if col in panel.columns]
    return {
        str(row.local_pair_id): {col: getattr(row, col) for col in cols}
        for row in panel[cols].itertuples(index=False)
    }


def _build_readout_rows(
    *,
    panel: pd.DataFrame,
    seed_runs: pd.DataFrame,
    group_cols: list[str],
    group_label_col: str,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    lookup = _panel_lookup(panel)
    rows: list[dict[str, Any]] = []
    for keys, group in seed_runs.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_data = dict(zip(group_cols, keys, strict=True))
        local_pair_id = str(key_data["local_pair_id"])
        data = dict(lookup[local_pair_id])
        data.update(key_data)

        variant_shares: dict[str, float] = {}
        variant_run_counts: dict[str, int] = {}
        for variant in GRAPH_VARIANTS:
            variant_group = group[group["graph_variant"].astype(str).eq(variant)]
            variant_run_counts[variant] = int(len(variant_group))
            variant_shares[variant] = (
                float(variant_group["pair_coassigned"].fillna(False).astype(bool).mean())
                if not variant_group.empty
                else 0.0
            )
        macro, condition = _classify_condition(
            original_share=variant_shares["original"],
            drop_direct_share=variant_shares["drop_direct_edge"],
            drop_bridge_share=variant_shares["drop_bridge_edges"],
            drop_both_share=variant_shares["drop_direct_and_bridge_edges"],
            thresholds=thresholds,
        )
        passed, expectation = _expected_readout_pass(
            validation_stratum=str(data["validation_stratum"]),
            macro_role=macro,
            source_condition=condition,
        )
        original_group = group[group["graph_variant"].astype(str).eq("original")]
        data.update(_signature_stats(original_group))
        data.update(
            {
                f"{group_label_col}_macro_role": macro,
                f"{group_label_col}_source_condition": condition,
                f"{group_label_col}_expected_validation_pass": bool(passed),
                f"{group_label_col}_expectation": expectation,
                "original_pair_coassigned_share": variant_shares["original"],
                "drop_direct_pair_coassigned_share": variant_shares["drop_direct_edge"],
                "drop_bridge_pair_coassigned_share": variant_shares["drop_bridge_edges"],
                "drop_direct_and_bridge_pair_coassigned_share": variant_shares[
                    "drop_direct_and_bridge_edges"
                ],
                "bridge_release_lift_proxy": (
                    variant_shares["drop_bridge_edges"] - variant_shares["original"]
                ),
                "direct_dependency_proxy": (
                    variant_shares["original"] - variant_shares["drop_direct_edge"]
                ),
                "graph_variant_run_counts": json.dumps(variant_run_counts, sort_keys=True),
                "readout_source_signature_proxy_available": True,
                "exact_g4_8f_signature_available": False,
                "signature_limit_note": (
                    "endpoint-signature proxy from local ablation only; no exact "
                    "G4.8F NanoClustering source-signature set is materialized"
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
        rows.append(data)
    return pd.DataFrame(rows)


def _pair_rows(seed_split_rows: pd.DataFrame) -> pd.DataFrame:
    discovery = seed_split_rows[seed_split_rows["seed_split"].eq(DISCOVERY_SPLIT)].copy()
    heldout = seed_split_rows[seed_split_rows["seed_split"].eq(HELDOUT_SPLIT)].copy()
    base_cols = [col for col in _identity_columns() if col in seed_split_rows.columns]
    rows = seed_split_rows[base_cols].drop_duplicates("local_pair_id").copy()

    copy_cols = [
        "local_pair_id",
        "seed_split_macro_role",
        "seed_split_source_condition",
        "seed_split_expected_validation_pass",
        "original_pair_coassigned_share",
        "drop_direct_pair_coassigned_share",
        "drop_bridge_pair_coassigned_share",
        "drop_direct_and_bridge_pair_coassigned_share",
        "bridge_release_lift_proxy",
        "direct_dependency_proxy",
        "original_distinct_endpoint_count",
        "original_coassigned_signature_count",
        "original_source_endpoint_signature_proxy_count",
        "original_bridge_split_source_signature_proxy_count",
        "original_single_side_source_signature_proxy_count",
        "original_no_bridge_separated_signature_count",
        "original_mechanism_read_counts",
    ]

    def renamed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = [col for col in copy_cols if col in frame.columns]
        out = frame[cols].copy()
        return out.rename(
            columns={
                col: f"{prefix}_{col}"
                for col in cols
                if col != "local_pair_id"
            }
        )

    rows = rows.merge(renamed(discovery, "discovery"), on="local_pair_id", how="left")
    rows = rows.merge(renamed(heldout, "heldout"), on="local_pair_id", how="left")
    rows["heldout_validation_status"] = np.where(
        rows["heldout_seed_split_expected_validation_pass"].fillna(False).astype(bool),
        "heldout_expected_stratum_preserved",
        "heldout_stratum_fragile_or_shifted",
    )
    rows["discovery_vs_heldout_macro_match"] = (
        rows["discovery_seed_split_macro_role"].astype(str)
        == rows["heldout_seed_split_macro_role"].astype(str)
    )
    rows["discovery_vs_heldout_source_condition_match"] = (
        rows["discovery_seed_split_source_condition"].astype(str)
        == rows["heldout_seed_split_source_condition"].astype(str)
    )
    rows["readout_source_signature_proxy_available"] = True
    rows["exact_g4_8f_signature_available"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["validation_stratum", "local_pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _stratum_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, group in pair_rows.groupby("validation_stratum", sort=True):
        data: dict[str, Any] = {
            "validation_stratum": str(stratum),
            "pair_count": int(len(group)),
            "object_count": int(group["object_role_universe_id"].nunique()),
            "branch_count": int(group["branch"].nunique()),
            "heldout_expected_pass_count": int(
                group["heldout_seed_split_expected_validation_pass"].fillna(False).astype(bool).sum()
            ),
            "heldout_fragile_pair_count": int(
                (~group["heldout_seed_split_expected_validation_pass"].fillna(False).astype(bool)).sum()
            ),
            "discovery_heldout_macro_match_count": int(
                group["discovery_vs_heldout_macro_match"].fillna(False).astype(bool).sum()
            ),
            "heldout_macro_role_counts": json.dumps(
                _count_dict(group["heldout_seed_split_macro_role"]),
                sort_keys=True,
            ),
            "heldout_source_condition_counts": json.dumps(
                _count_dict(group["heldout_seed_split_source_condition"]),
                sort_keys=True,
            ),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for col in [
            "heldout_original_pair_coassigned_share",
            "heldout_drop_bridge_pair_coassigned_share",
            "heldout_bridge_release_lift_proxy",
            "heldout_original_source_endpoint_signature_proxy_count",
            "heldout_original_bridge_split_source_signature_proxy_count",
            "heldout_original_single_side_source_signature_proxy_count",
            "heldout_original_coassigned_signature_count",
        ]:
            if col in group.columns:
                data.update(_prefix_stats(col, group[col]))
        rows.append(data)
    return pd.DataFrame(rows)


def _gate_row(gate_id: str, question: str, passed: bool, observed: Any, minimum: Any) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "gate_status": "pass" if bool(passed) else "fail",
        "observed": observed,
        "minimum_or_rule": minimum,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_gate_matrix(
    *,
    panel: pd.DataFrame,
    seed_runs: pd.DataFrame,
    pair_rows: pd.DataFrame,
    seed_split_rows: pd.DataFrame,
    start_condition_rows: pd.DataFrame,
) -> pd.DataFrame:
    split_counts = seed_runs.groupby(["local_pair_id", "seed_split", "graph_variant"]).size()
    start_counts = seed_runs.groupby(["local_pair_id", "start_condition", "graph_variant"]).size()
    heldout_pass = pair_rows["heldout_seed_split_expected_validation_pass"].fillna(False).astype(bool)
    heldout_strata_with_pass = set(
        pair_rows.loc[heldout_pass, "validation_stratum"].astype(str).unique()
    )
    ready_pass = bool(
        pair_rows.loc[
            heldout_pass & pair_rows["validation_stratum"].isin(["strict_ready", "rare_ready"])
        ].shape[0]
        > 0
    )
    control_strata = {
        "target_saturated_no_handle",
        "latent_release_no_source_control",
        "no_release_control",
        "coupled_direct_bridge_failure_control",
    }
    control_pass = control_strata.issubset(heldout_strata_with_pass)
    rows = [
        _gate_row(
            "G1_panel_rows_preserved",
            "Does the readout preserve all frozen panel rows?",
            int(pair_rows["local_pair_id"].nunique()) == int(panel["local_pair_id"].nunique()),
            f"pair_rows={pair_rows['local_pair_id'].nunique()} panel_rows={panel['local_pair_id'].nunique()}",
            "all 23 frozen panel rows",
        ),
        _gate_row(
            "G2_seed_split_coverage",
            "Are discovery and held-out seed splits available for every pair and graph variant?",
            not split_counts.empty and int(split_counts.min()) >= len(START_CONDITIONS) * 4,
            f"min_split_variant_run_count={int(split_counts.min()) if not split_counts.empty else 0}",
            "at least 20 runs per pair/split/variant",
        ),
        _gate_row(
            "G3_start_condition_coverage",
            "Are start-condition readouts available for every pair and graph variant?",
            not start_counts.empty and int(start_counts.min()) >= 8,
            f"min_start_variant_run_count={int(start_counts.min()) if not start_counts.empty else 0}",
            "at least 8 seeds per pair/start/variant",
        ),
        _gate_row(
            "G4_source_signature_proxy_materialized",
            "Are endpoint-derived source-signature proxies materialized?",
            bool(seed_split_rows["readout_source_signature_proxy_available"].fillna(False).all())
            and bool(start_condition_rows["readout_source_signature_proxy_available"].fillna(False).all()),
            "proxy columns present for seed-split and start-condition rows",
            "proxy readout only; exact G4.8F signatures remain unavailable",
        ),
        _gate_row(
            "G5_heldout_stratum_stability",
            "Do all frozen strata preserve their expected role under held-out seeds?",
            bool(heldout_pass.all()),
            f"heldout_expected_pass={int(heldout_pass.sum())}/{len(heldout_pass)}",
            "all 23 held-out rows preserve expected stratum",
        ),
        _gate_row(
            "G6_heldout_ready_signal_present",
            "Does held-out readout retain at least one ready signal?",
            ready_pass,
            f"ready_heldout_pass_count={int(pair_rows.loc[heldout_pass & pair_rows['validation_stratum'].isin(['strict_ready', 'rare_ready'])].shape[0])}",
            "strict or rare ready signal survives held-out split",
        ),
        _gate_row(
            "G7_heldout_control_surface_present",
            "Do held-out controls remain represented across control strata?",
            control_pass,
            json.dumps(sorted(heldout_strata_with_pass & control_strata)),
            "target-saturated, nonready, and coupled-failure controls represented",
        ),
        _gate_row(
            "G8_exact_signature_gap_closed",
            "Is the exact G4.8F source-signature gap kept closed?",
            not bool(pair_rows["exact_g4_8f_signature_available"].fillna(False).any()),
            "exact_g4_8f_signature_available=false",
            "do not claim exact source signatures",
        ),
        _gate_row(
            "G9_no_new_leiden_execution",
            "Is this readout over existing seed runs rather than new Leiden execution?",
            True,
            RUN_STATUS,
            "read-only over existing local-ablation seed runs",
        ),
        _gate_row(
            "G10_no_method_or_wall_claim",
            "Are wall/pathway, quality/cost, replay, and method claims closed?",
            True,
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
        ),
    ]
    return pd.DataFrame(rows)


def _readout_status(gate_matrix: pd.DataFrame) -> str:
    if gate_matrix.empty:
        return "local_validation_readout_gate_failed"
    material_gates = gate_matrix[
        ~gate_matrix["gate_id"].astype(str).eq("G5_heldout_stratum_stability")
    ]
    if not bool(material_gates["gate_status"].astype(str).eq("pass").all()):
        return "local_validation_readout_gate_failed"
    stability = gate_matrix[gate_matrix["gate_id"].eq("G5_heldout_stratum_stability")]
    if not stability.empty and stability["gate_status"].iloc[0] == "pass":
        return "local_validation_readout_heldout_stable"
    return "local_validation_readout_materialized_with_heldout_fragility"


def _build_summary(
    *,
    pair_rows: pd.DataFrame,
    seed_split_rows: pd.DataFrame,
    start_condition_rows: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    panel_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    heldout_pass = pair_rows["heldout_seed_split_expected_validation_pass"].fillna(False).astype(bool)
    return {
        "schema": "nanoclustering_g4_8_local_validation_readout_summary.v1",
        "status": _readout_status(gate_matrix),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "panel_dir": str(panel_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "thresholds": thresholds,
        "pair_count": int(len(pair_rows)),
        "seed_split_readout_rows": int(len(seed_split_rows)),
        "start_condition_readout_rows": int(len(start_condition_rows)),
        "stratum_summary_rows": int(len(stratum_summary)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": [
            str(row.gate_id)
            for row in gate_matrix.itertuples(index=False)
            if str(row.gate_status) != "pass"
        ],
        "heldout_expected_pass_count": int(heldout_pass.sum()),
        "heldout_fragile_pair_count": int((~heldout_pass).sum()),
        "heldout_macro_role_counts": _count_dict(pair_rows["heldout_seed_split_macro_role"]),
        "heldout_source_condition_counts": _count_dict(
            pair_rows["heldout_seed_split_source_condition"]
        ),
        "validation_stratum_counts": _count_dict(pair_rows["validation_stratum"]),
        "exact_g4_8f_signature_available": False,
        "recommended_next_gate": (
            "Inspect held-out fragile rows and decide whether the local validation "
            "surface needs seed/start-stratified contracts before any route/pathway, "
            "quality/cost, full NanoClustering replay, or method claim."
            if int((~heldout_pass).sum()) > 0
            else "Proceed to a predeclared local validation execution contract over this panel."
        ),
        "written_artifacts": [
            PAIR_ROWS_CSV,
            SEED_SPLIT_ROWS_CSV,
            START_CONDITION_ROWS_CSV,
            STRATUM_SUMMARY_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    stratum_summary: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> None:
    fragile = pair_rows[
        ~pair_rows["heldout_seed_split_expected_validation_pass"].fillna(False).astype(bool)
    ].copy()
    lines = [
        "# NanoClustering G4.8 Local Validation Readout",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- heldout_expected_pass_count: {summary['heldout_expected_pass_count']}",
        f"- heldout_fragile_pair_count: {summary['heldout_fragile_pair_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- exact_g4_8f_signature_available: {summary['exact_g4_8f_signature_available']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Stratum Summary",
        "",
    ]
    stratum_cols = [
        "validation_stratum",
        "pair_count",
        "heldout_expected_pass_count",
        "heldout_fragile_pair_count",
        "heldout_macro_role_counts",
        "heldout_original_pair_coassigned_share_median",
        "heldout_drop_bridge_pair_coassigned_share_median",
        "heldout_original_source_endpoint_signature_proxy_count_median",
        "heldout_original_coassigned_signature_count_median",
    ]
    lines.append(_markdown_table(stratum_summary, stratum_cols))
    lines.extend(["", "## Held-Out Fragile Rows", ""])
    if fragile.empty:
        lines.append("No held-out fragile rows.")
    else:
        fragile_cols = [
            "local_pair_id",
            "validation_stratum",
            "analog_macro_role",
            "heldout_seed_split_macro_role",
            "heldout_seed_split_source_condition",
            "heldout_original_pair_coassigned_share",
            "heldout_drop_bridge_pair_coassigned_share",
            "heldout_original_source_endpoint_signature_proxy_count",
            "heldout_original_coassigned_signature_count",
        ]
        lines.append(_markdown_table(fragile, fragile_cols))
    lines.extend(["", "## Gate Matrix", ""])
    lines.append(
        _markdown_table(
            gate_matrix,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        )
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This readout reuses existing local-ablation seed runs. It materializes "
            "held-out local behavior and endpoint-derived source-signature proxies, "
            "but it does not materialize exact G4.8F source signatures and does not "
            "open route/pathway, wall, quality/cost, full NanoClustering replay, or "
            "method claims.",
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_readout(args: argparse.Namespace) -> dict[str, Any]:
    panel_dir = Path(args.panel_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "partial_source_min": float(args.partial_source_min),
        "partial_source_max": float(args.partial_source_max),
        "target_saturated_min": float(args.target_saturated_min),
        "released_share_min": float(args.released_share_min),
        "suppressed_share_max": float(args.suppressed_share_max),
    }
    panel = _read_csv(panel_dir / PANEL_ROWS_CSV)
    seed_runs = _read_csv(local_ablation_dir / LOCAL_SEED_RUNS_CSV)
    panel_ids = set(panel["local_pair_id"].astype(str))
    seed_runs = seed_runs[seed_runs["local_pair_id"].astype(str).isin(panel_ids)].copy()
    seed_runs["seed_split"] = seed_runs["seed"].astype(int).map(_seed_split)

    seed_split_rows = _build_readout_rows(
        panel=panel,
        seed_runs=seed_runs,
        group_cols=["local_pair_id", "seed_split"],
        group_label_col="seed_split",
        thresholds=thresholds,
    )
    start_condition_rows = _build_readout_rows(
        panel=panel,
        seed_runs=seed_runs,
        group_cols=["local_pair_id", "start_condition"],
        group_label_col="start_condition",
        thresholds=thresholds,
    )
    pair_rows = _pair_rows(seed_split_rows)
    stratum_summary = _stratum_summary(pair_rows)
    gate_matrix = _build_gate_matrix(
        panel=panel,
        seed_runs=seed_runs,
        pair_rows=pair_rows,
        seed_split_rows=seed_split_rows,
        start_condition_rows=start_condition_rows,
    )
    summary = _build_summary(
        pair_rows=pair_rows,
        seed_split_rows=seed_split_rows,
        start_condition_rows=start_condition_rows,
        stratum_summary=stratum_summary,
        gate_matrix=gate_matrix,
        panel_dir=panel_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        thresholds=thresholds,
    )
    config = {
        "schema": "nanoclustering_g4_8_local_validation_readout_config.v1",
        "panel_dir": str(panel_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "seed_split_rule": {
            DISCOVERY_SPLIT: "seed <= 3",
            HELDOUT_SPLIT: "seed >= 4",
        },
        "graph_variants": GRAPH_VARIANTS,
        "start_conditions": START_CONDITIONS,
        "thresholds": thresholds,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(seed_split_rows, output_dir / SEED_SPLIT_ROWS_CSV)
    _write_csv(start_condition_rows, output_dir / START_CONDITION_ROWS_CSV)
    _write_csv(stratum_summary, output_dir / STRATUM_SUMMARY_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        stratum_summary=stratum_summary,
        gate_matrix=gate_matrix,
        pair_rows=pair_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--partial-source-min", type=float, default=0.20)
    parser.add_argument("--partial-source-max", type=float, default=0.80)
    parser.add_argument("--target-saturated-min", type=float, default=0.80)
    parser.add_argument("--released-share-min", type=float, default=0.95)
    parser.add_argument("--suppressed-share-max", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    summary = run_readout(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
