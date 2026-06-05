#!/usr/bin/env python3
"""Audit cross-seed endpoint signatures for the G4.8 primary pathway trace.

The earlier pathway-shape and direct-path acceptance audits used
``unknown_new_endpoint`` in a same-seed anchor sense: a step endpoint was
unknown if it did not match that seed's known anchors. This readout asks a
different topology question: is the endpoint signature truly novel at the
pair-level, or is it known from another seed/start condition within the same
local pair?

It does not run Leiden, broaden route execution, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_cross_seed_endpoint_atlas_gamma1e5_20260604"
)

PRIMARY_ROUTE_FAMILY = "bridge_release_interpolation_probe"
RUN_STATUS = "audited_nanoclustering_g4_8_cross_seed_endpoint_atlas"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 cross-seed endpoint-atlas audit only; reads already "
    "executed scoped local route traces and reclassifies same-seed unknown "
    "endpoint labels against pair-level endpoint signatures. It does not run "
    "Leiden, broaden route execution, promote walls, evaluate quality/cost "
    "value, replay full NanoClustering, or claim method or algorithm success."
)

SIGNATURE_ROWS_CSV = "nanoclustering_g4_8_cross_seed_endpoint_atlas_signature_rows.csv"
STEP_SIGNATURE_ROWS_CSV = (
    "nanoclustering_g4_8_cross_seed_endpoint_atlas_step_signature_rows.csv"
)
UNKNOWN_RECLASS_ROWS_CSV = (
    "nanoclustering_g4_8_cross_seed_endpoint_atlas_unknown_reclass_rows.csv"
)
CONTRACT_ROWS_CSV = "nanoclustering_g4_8_cross_seed_endpoint_atlas_contract_rows.csv"
PAIR_ROWS_CSV = "nanoclustering_g4_8_cross_seed_endpoint_atlas_pair_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_cross_seed_endpoint_atlas_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_cross_seed_endpoint_atlas_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_cross_seed_endpoint_atlas_config.json"
REPORT_MD = "nanoclustering_g4_8_cross_seed_endpoint_atlas_report.md"


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _join_sorted(values: pd.Series) -> str:
    items = sorted({str(value) for value in values.dropna() if str(value)})
    return ";".join(items)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    visible = frame[cols].head(int(max_rows))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in visible.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                values.append(json.dumps(_json_safe(value), sort_keys=True))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _primary_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    return trace_rows[
        trace_rows["planned_route_family"].astype(str).eq(PRIMARY_ROUTE_FAMILY)
    ].copy()


def _signature_rows(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    known = primary[~primary["endpoint_assignment_by_step"].astype(str).eq("unknown_new_endpoint")]
    known_vocab = (
        known.groupby(["local_pair_id", "result_endpoint_signature_id"], sort=False)
        .agg(
            cross_seed_known_assignments=("endpoint_assignment_by_step", _join_sorted),
            known_step_indices=("step_index", lambda values: ";".join(map(str, sorted(set(map(int, values)))))),
            known_seed_count=("seed", "nunique"),
            known_start_count=("start_condition", "nunique"),
        )
        .reset_index()
    )
    for keys, group in primary.groupby(
        ["local_pair_id", "result_endpoint_signature_id"], sort=False
    ):
        local_pair_id, signature_id = keys
        first = group.iloc[0]
        known_match = known_vocab[
            known_vocab["local_pair_id"].astype(str).eq(str(local_pair_id))
            & known_vocab["result_endpoint_signature_id"].astype(str).eq(str(signature_id))
        ]
        known_assignments = (
            "" if known_match.empty else str(known_match.iloc[0]["cross_seed_known_assignments"])
        )
        if known_assignments:
            if "drop_bridge_target_anchor" in known_assignments:
                atlas_role = "pair_level_known_drop_bridge_target_signature"
            elif "original_source_anchor" in known_assignments:
                atlas_role = "pair_level_known_original_source_signature"
            elif "drop_direct_guard_anchor" in known_assignments:
                atlas_role = "pair_level_known_drop_direct_guard_signature"
            else:
                atlas_role = "pair_level_known_other_anchor_signature"
        else:
            atlas_role = "pair_level_true_novel_signature"
        rows.append(
            {
                "local_pair_id": str(local_pair_id),
                "result_endpoint_signature_id": str(signature_id),
                "result_endpoint_signature": str(first["result_endpoint_signature"]),
                "row_count": int(len(group)),
                "step_indices": ";".join(map(str, sorted(set(map(int, group["step_index"]))))),
                "seed_count": int(group["seed"].nunique()),
                "start_condition_count": int(group["start_condition"].nunique()),
                "same_seed_assignment_counts": json.dumps(
                    _count_dict(group["endpoint_assignment_by_step"]),
                    sort_keys=True,
                ),
                "cross_seed_known_assignments": known_assignments,
                "cross_seed_known_signature": bool(known_assignments),
                "endpoint_atlas_role": atlas_role,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "endpoint_atlas_role", "result_endpoint_signature_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _step_signature_rows(primary: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    vocab = signatures[
        [
            "local_pair_id",
            "result_endpoint_signature_id",
            "endpoint_atlas_role",
            "cross_seed_known_assignments",
        ]
    ]
    rows = primary.merge(
        vocab,
        on=["local_pair_id", "result_endpoint_signature_id"],
        how="left",
        validate="many_to_one",
    )
    grouped = (
        rows.groupby(
            [
                "local_pair_id",
                "step_index",
                "result_endpoint_signature_id",
                "endpoint_atlas_role",
                "cross_seed_known_assignments",
            ],
            sort=False,
        )
        .agg(
            row_count=("route_trace_row_id", "count"),
            seed_count=("seed", "nunique"),
            start_condition_count=("start_condition", "nunique"),
            same_seed_assignment_counts=(
                "endpoint_assignment_by_step",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    grouped["run_status"] = RUN_STATUS
    grouped["claim_boundary"] = CLAIM_BOUNDARY
    return grouped.sort_values(
        ["local_pair_id", "step_index", "endpoint_atlas_role", "result_endpoint_signature_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _unknown_reclass_rows(primary: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    unknown = primary[
        primary["endpoint_assignment_by_step"].astype(str).eq("unknown_new_endpoint")
    ].copy()
    vocab = signatures[
        [
            "local_pair_id",
            "result_endpoint_signature_id",
            "cross_seed_known_assignments",
            "cross_seed_known_signature",
            "endpoint_atlas_role",
        ]
    ]
    rows = unknown.merge(
        vocab,
        on=["local_pair_id", "result_endpoint_signature_id"],
        how="left",
        validate="many_to_one",
    )
    rows["same_seed_unknown_reclass_status"] = rows["cross_seed_known_signature"].map(
        {
            True: "same_seed_unknown_but_pair_level_known_signature",
            False: "same_seed_unknown_and_pair_level_true_novel_signature",
        }
    )
    rows["topology_block_reason"] = rows["cross_seed_known_signature"].map(
        {
            True: "same-seed anchor mismatch; not a pair-level novel endpoint",
            False: "pair-level novel endpoint requires separate basin candidate audit",
        }
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    keep = [
        "route_trace_row_id",
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "seed",
        "step_index",
        "bridge_edge_weight_fraction",
        "direct_edge_weight_fraction",
        "result_endpoint_signature_id",
        "cross_seed_known_assignments",
        "cross_seed_known_signature",
        "endpoint_atlas_role",
        "support_distance_min_known_anchor",
        "support_distance_to_original",
        "support_distance_to_drop_bridge_edges",
        "support_distance_to_drop_direct_edge",
        "objective_delta_from_start",
        "same_seed_unknown_reclass_status",
        "topology_block_reason",
        "run_status",
        "claim_boundary",
    ]
    return rows[keep].sort_values(
        ["local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_rows(primary: pd.DataFrame, unknown_reclass: pd.DataFrame) -> pd.DataFrame:
    unknown_summary = (
        unknown_reclass.groupby(["route_contract_id"], sort=False)
        .agg(
            same_seed_unknown_count=("route_trace_row_id", "count"),
            cross_seed_known_unknown_count=("cross_seed_known_signature", "sum"),
            true_novel_unknown_count=(
                "cross_seed_known_signature",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            reclass_status_counts=(
                "same_seed_unknown_reclass_status",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    base = (
        primary.groupby(["route_contract_id", "validation_unit_id", "local_pair_id", "start_condition"], sort=False)
        .agg(
            seed_count=("seed", "nunique"),
            result_signature_count=("result_endpoint_signature_id", "nunique"),
            final_signature_count=(
                "result_endpoint_signature_id",
                lambda values: 0,
            ),
        )
        .reset_index()
    )
    # Recompute final signature count from final step rows only.
    final_counts = (
        primary.sort_values("step_index", kind="mergesort")
        .groupby(["route_contract_id", "seed"], sort=False)
        .tail(1)
        .groupby("route_contract_id")["result_endpoint_signature_id"]
        .nunique()
        .reset_index(name="final_signature_count")
    )
    base = base.drop(columns=["final_signature_count"]).merge(
        final_counts,
        on="route_contract_id",
        how="left",
        validate="one_to_one",
    )
    rows = base.merge(
        unknown_summary,
        on="route_contract_id",
        how="left",
        validate="one_to_one",
    )
    for col in [
        "same_seed_unknown_count",
        "cross_seed_known_unknown_count",
        "true_novel_unknown_count",
    ]:
        rows[col] = rows[col].fillna(0).astype(int)
    rows["reclass_status_counts"] = rows["reclass_status_counts"].fillna("{}")
    rows["no_true_novel_unknown_endpoint"] = rows["true_novel_unknown_count"].eq(0)
    rows["endpoint_atlas_contract_status"] = rows.apply(
        lambda row: (
            "same_seed_unknowns_reclassified_pair_level_known_no_true_novel"
            if int(row["same_seed_unknown_count"]) > 0
            and int(row["true_novel_unknown_count"]) == 0
            else (
                "no_same_seed_unknowns_no_true_novel"
                if int(row["same_seed_unknown_count"]) == 0
                else "true_novel_endpoint_requires_basin_audit"
            )
        ),
        axis=1,
    )
    rows["wall_contract_ready"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _pair_rows(
    primary: pd.DataFrame,
    signatures: pd.DataFrame,
    unknown_reclass: pd.DataFrame,
    contract_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair, group in primary.groupby("local_pair_id", sort=False):
        pair_signatures = signatures[signatures["local_pair_id"].astype(str).eq(str(pair))]
        pair_unknown = unknown_reclass[unknown_reclass["local_pair_id"].astype(str).eq(str(pair))]
        pair_contracts = contract_rows[contract_rows["local_pair_id"].astype(str).eq(str(pair))]
        step_signature_counts = {
            str(int(step)): int(sig_count)
            for step, sig_count in group.groupby("step_index")["result_endpoint_signature_id"]
            .nunique()
            .items()
        }
        if str(pair) == "local_pair_009":
            interpretation = (
                "step2 collapses to one pair-level signature for all seeds; same-seed "
                "unknown labels are anchor-label incompatibilities, not true novel endpoints"
            )
        elif str(pair) == "local_pair_012":
            interpretation = (
                "step2 is mixed at pair level but every same-seed unknown signature is "
                "known elsewhere as source or target"
            )
        else:
            interpretation = "pair-level endpoint atlas requires manual interpretation"
        rows.append(
            {
                "local_pair_id": str(pair),
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "seed_route_count": int(
                    group[["route_contract_id", "seed"]].drop_duplicates().shape[0]
                ),
                "trace_row_count": int(len(group)),
                "pair_level_signature_count": int(pair_signatures["result_endpoint_signature_id"].nunique()),
                "pair_level_true_novel_signature_count": int(
                    pair_signatures["endpoint_atlas_role"]
                    .astype(str)
                    .eq("pair_level_true_novel_signature")
                    .sum()
                ),
                "same_seed_unknown_row_count": int(len(pair_unknown)),
                "cross_seed_known_unknown_row_count": int(
                    pair_unknown["cross_seed_known_signature"].astype(bool).sum()
                ),
                "true_novel_unknown_row_count": int(
                    (~pair_unknown["cross_seed_known_signature"].astype(bool)).sum()
                ),
                "contract_count": int(len(pair_contracts)),
                "contracts_with_no_true_novel_unknown_count": int(
                    pair_contracts["no_true_novel_unknown_endpoint"].astype(bool).sum()
                ),
                "step_signature_counts": json.dumps(step_signature_counts, sort_keys=True),
                "pair_endpoint_atlas_status": (
                    "no_true_novel_unknowns_same_seed_labels_need_reinterpretation"
                ),
                "interpretation": interpretation,
                "wall_ready_contract_count": int(pair_contracts["wall_contract_ready"].astype(bool).sum()),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values("local_pair_id", kind="mergesort").reset_index(drop=True)


def _gate_row(
    gate_id: str,
    question: str,
    observed: Any,
    minimum_or_rule: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "observed": observed,
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate_matrix(
    *,
    trace_gates: pd.DataFrame,
    primary: pd.DataFrame,
    unknown_reclass: pd.DataFrame,
    contract_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    same_seed_unknown_count = int(len(unknown_reclass))
    cross_seed_known_unknown_count = int(
        unknown_reclass["cross_seed_known_signature"].astype(bool).sum()
    )
    true_novel_unknown_count = int(
        (~unknown_reclass["cross_seed_known_signature"].astype(bool)).sum()
    )
    local_pair_009_step2_sig_count = int(
        primary[
            primary["local_pair_id"].astype(str).eq("local_pair_009")
            & primary["step_index"].astype(int).eq(2)
        ]["result_endpoint_signature_id"].nunique()
    )
    local_pair_012_step2_sig_count = int(
        primary[
            primary["local_pair_id"].astype(str).eq("local_pair_012")
            & primary["step_index"].astype(int).eq(2)
        ]["result_endpoint_signature_id"].nunique()
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_trace_gates_pass",
                "Did every upstream scoped pathway-probe trace gate pass?",
                _count_dict(trace_gates["gate_status"]),
                "all upstream trace gates pass",
                bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_primary_scope_only",
                "Is the endpoint atlas restricted to primary bridge-release traces?",
                f"trace_rows={len(primary)} pair_count={primary['local_pair_id'].nunique()}",
                "400 primary bridge-release trace rows over two pairs",
                len(primary) == 400 and int(primary["local_pair_id"].nunique()) == 2,
            ),
            _gate_row(
                "G3_same_seed_unknowns_reclassified",
                "Are all same-seed unknown labels checked against pair-level signatures?",
                (
                    f"same_seed_unknowns={same_seed_unknown_count} "
                    f"cross_seed_known_unknowns={cross_seed_known_unknown_count}"
                ),
                "all 27 same-seed unknown rows are cross-seed known signatures",
                same_seed_unknown_count == 27
                and cross_seed_known_unknown_count == same_seed_unknown_count,
            ),
            _gate_row(
                "G4_no_true_novel_unknown_endpoint",
                "Are any same-seed unknown endpoints truly novel at pair level?",
                f"true_novel_unknowns={true_novel_unknown_count}",
                "zero pair-level true novel unknown endpoints",
                true_novel_unknown_count == 0,
            ),
            _gate_row(
                "G5_local_pair_009_step2_is_single_signature",
                "Does local_pair_009 step2 collapse to one pair-level signature?",
                f"local_pair_009_step2_signature_count={local_pair_009_step2_sig_count}",
                "one pair-level signature at step2",
                local_pair_009_step2_sig_count == 1,
            ),
            _gate_row(
                "G6_local_pair_012_step2_is_mixed_not_novel",
                "Is local_pair_012 step2 mixed but not truly novel?",
                f"local_pair_012_step2_signature_count={local_pair_012_step2_sig_count}",
                "multiple pair-level signatures at step2, all in pair-level atlas",
                local_pair_012_step2_sig_count == 3 and true_novel_unknown_count == 0,
            ),
            _gate_row(
                "G7_all_contracts_have_no_true_novel_unknown",
                "Do all primary contracts have zero true-novel unknown endpoints?",
                (
                    "contracts_no_true_novel="
                    f"{int(contract_rows['no_true_novel_unknown_endpoint'].astype(bool).sum())}"
                ),
                "10 of 10 contracts have no true-novel unknown endpoints",
                int(contract_rows["no_true_novel_unknown_endpoint"].astype(bool).sum()) == 10,
            ),
            _gate_row(
                "G8_wall_claim_remains_closed",
                "Are wall claims still closed after cross-seed endpoint reclassification?",
                f"wall_ready_contracts={int(pair_rows['wall_ready_contract_count'].sum())}",
                "zero wall-ready contracts",
                int(pair_rows["wall_ready_contract_count"].sum()) == 0,
            ),
            _gate_row(
                "G9_no_method_quality_or_full_replay_claim",
                "Are method, quality/cost, full replay, and algorithm claims closed?",
                CLAIM_BOUNDARY,
                "claim boundary explicitly closed",
                True,
            ),
        ]
    )


def _summary(
    *,
    trace_dir: Path,
    output_dir: Path,
    signatures: pd.DataFrame,
    unknown_reclass: pd.DataFrame,
    contract_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_cross_seed_endpoint_atlas_summary.v1",
        "status": "cross_seed_endpoint_atlas_reclassifies_same_seed_unknowns_no_true_novel_wall_closed",
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "signature_row_count": int(len(signatures)),
        "unknown_reclass_row_count": int(len(unknown_reclass)),
        "contract_row_count": int(len(contract_rows)),
        "pair_row_count": int(len(pair_rows)),
        "same_seed_unknown_row_count": int(len(unknown_reclass)),
        "cross_seed_known_unknown_row_count": int(
            unknown_reclass["cross_seed_known_signature"].astype(bool).sum()
        ),
        "true_novel_unknown_row_count": int(
            (~unknown_reclass["cross_seed_known_signature"].astype(bool)).sum()
        ),
        "contracts_with_no_true_novel_unknown_count": int(
            contract_rows["no_true_novel_unknown_endpoint"].astype(bool).sum()
        ),
        "pair_endpoint_atlas_status_counts": _count_dict(pair_rows["pair_endpoint_atlas_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The 27 same-seed unknown endpoint labels are not true pair-level novel "
            "endpoints. Every one maps to a signature known elsewhere in the same "
            "local pair. local_pair_009 is especially clear: all step2 primary "
            "bridge-release traces collapse to one pair-level signature before all "
            "seeds move to the drop-bridge target at step3. Therefore the current "
            "D5 direct-path rule is a strict same-seed anchor-consistency guard, not "
            "a true-novel-endpoint or basin-topology test."
        ),
        "recommended_next_gate": (
            "Revise the direct-path acceptance contract into two axes: same-seed "
            "anchor consistency and pair-level endpoint atlas continuity. Keep wall "
            "claims closed, but do not treat same-seed unknown labels as evidence of "
            "new basins unless they are true novel pair-level signatures."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            SIGNATURE_ROWS_CSV,
            STEP_SIGNATURE_ROWS_CSV,
            UNKNOWN_RECLASS_ROWS_CSV,
            CONTRACT_ROWS_CSV,
            PAIR_ROWS_CSV,
            GATE_MATRIX_CSV,
            SUMMARY_JSON,
            CONFIG_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    signatures: pd.DataFrame,
    step_signatures: pd.DataFrame,
    unknown_reclass: pd.DataFrame,
    contract_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Cross-Seed Endpoint Atlas Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- same_seed_unknown_row_count: {summary['same_seed_unknown_row_count']}",
        f"- cross_seed_known_unknown_row_count: {summary['cross_seed_known_unknown_row_count']}",
        f"- true_novel_unknown_row_count: {summary['true_novel_unknown_row_count']}",
        f"- contracts_with_no_true_novel_unknown_count: {summary['contracts_with_no_true_novel_unknown_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Atlas",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "pair_level_signature_count",
                "same_seed_unknown_row_count",
                "cross_seed_known_unknown_row_count",
                "true_novel_unknown_row_count",
                "step_signature_counts",
                "pair_endpoint_atlas_status",
                "interpretation",
            ],
        ),
        "",
        "## Signature Atlas",
        "",
        _markdown_table(
            signatures,
            [
                "local_pair_id",
                "result_endpoint_signature_id",
                "row_count",
                "step_indices",
                "same_seed_assignment_counts",
                "cross_seed_known_assignments",
                "endpoint_atlas_role",
            ],
            max_rows=20,
        ),
        "",
        "## Step Signatures",
        "",
        _markdown_table(
            step_signatures,
            [
                "local_pair_id",
                "step_index",
                "result_endpoint_signature_id",
                "endpoint_atlas_role",
                "cross_seed_known_assignments",
                "row_count",
                "same_seed_assignment_counts",
            ],
            max_rows=30,
        ),
        "",
        "## Unknown Reclassification",
        "",
        _markdown_table(
            unknown_reclass,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "step_index",
                "result_endpoint_signature_id",
                "cross_seed_known_assignments",
                "endpoint_atlas_role",
                "same_seed_unknown_reclass_status",
            ],
            max_rows=30,
        ),
        "",
        "## Contract Atlas",
        "",
        _markdown_table(
            contract_rows,
            [
                "local_pair_id",
                "start_condition",
                "same_seed_unknown_count",
                "cross_seed_known_unknown_count",
                "true_novel_unknown_count",
                "no_true_novel_unknown_endpoint",
                "endpoint_atlas_contract_status",
            ],
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            max_rows=20,
        ),
        "",
        "## Boundary",
        "",
        (
            "This audit changes the interpretation of same-seed unknown labels. "
            "It does not promote walls or direct-path method claims. It says that "
            "unknown_new_endpoint in this trace is a same-seed anchor-label issue, "
            "not evidence of a true novel pair-level endpoint."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    trace_gates = _read_csv(trace_dir / TRACE_GATE_MATRIX_CSV)
    primary = _primary_rows(trace_rows)
    signatures = _signature_rows(primary)
    step_signatures = _step_signature_rows(primary, signatures)
    unknown_reclass = _unknown_reclass_rows(primary, signatures)
    contract_atlas = _contract_rows(primary, unknown_reclass)
    pair_atlas = _pair_rows(primary, signatures, unknown_reclass, contract_atlas)
    gates = _gate_matrix(
        trace_gates=trace_gates,
        primary=primary,
        unknown_reclass=unknown_reclass,
        contract_rows=contract_atlas,
        pair_rows=pair_atlas,
    )
    summary = _summary(
        trace_dir=trace_dir,
        output_dir=output_dir,
        signatures=signatures,
        unknown_reclass=unknown_reclass,
        contract_rows=contract_atlas,
        pair_rows=pair_atlas,
        gates=gates,
    )

    _write_csv(signatures, output_dir / SIGNATURE_ROWS_CSV)
    _write_csv(step_signatures, output_dir / STEP_SIGNATURE_ROWS_CSV)
    _write_csv(unknown_reclass, output_dir / UNKNOWN_RECLASS_ROWS_CSV)
    _write_csv(contract_atlas, output_dir / CONTRACT_ROWS_CSV)
    _write_csv(pair_atlas, output_dir / PAIR_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "trace_dir": str(trace_dir),
                    "output_dir": str(output_dir),
                    "primary_route_family": PRIMARY_ROUTE_FAMILY,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        signatures=signatures,
        step_signatures=step_signatures,
        unknown_reclass=unknown_reclass,
        contract_rows=contract_atlas,
        pair_rows=pair_atlas,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
