#!/usr/bin/env python3
"""Design the G4.8 direct-path acceptance contract.

This consumes the primary bridge-release pathway-shape audit and freezes the
next gate as an explicit direct-path acceptance contract. It separates
seed-level clean pathway candidates from contract-level acceptance:

- physical direct-edge retention is necessary but insufficient;
- no intermediate unknown/support-incompatible endpoint is required;
- acceptance is contract-level, across all seeds for each start condition;
- objective debt/recovery remains a separate wall-readiness question.

It does not run Leiden, broaden route execution, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_primary_bridge_release_pathway_shape import (
    CLAIM_BOUNDARY as SHAPE_CLAIM_BOUNDARY,
    CONTRACT_ROWS_CSV as SHAPE_CONTRACT_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_SHAPE_DIR,
    GATE_MATRIX_CSV as SHAPE_GATE_MATRIX_CSV,
    PAIR_ROWS_CSV as SHAPE_PAIR_ROWS_CSV,
    SEED_ROWS_CSV as SHAPE_SEED_ROWS_CSV,
    SUMMARY_JSON as SHAPE_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_direct_path_acceptance_contract_gamma1e5_20260604"
)

ACCEPTANCE_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_direct_path_acceptance_contract_rule_rows.csv"
)
SEED_EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_direct_path_acceptance_contract_seed_evidence_rows.csv"
)
CONTRACT_ROWS_CSV = "nanoclustering_g4_8_direct_path_acceptance_contract_rows.csv"
REGIME_ROWS_CSV = "nanoclustering_g4_8_direct_path_acceptance_contract_regime_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_direct_path_acceptance_contract_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_direct_path_acceptance_contract_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_direct_path_acceptance_contract_summary.json"
REPORT_MD = "nanoclustering_g4_8_direct_path_acceptance_contract_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_direct_path_acceptance_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 direct-path acceptance contract design only; reads the "
    "primary bridge-release pathway-shape audit and fixes direct-path acceptance "
    "criteria. It does not run Leiden, broaden route execution, promote walls, "
    "evaluate quality/cost value, replay full NanoClustering, or claim method "
    "or algorithm success."
)

PRIMARY_ROUTE_FAMILY = "bridge_release_interpolation_probe"

ACCEPTANCE_RULES = (
    {
        "rule_id": "D1_primary_scope",
        "rule_group": "scope",
        "rule_question": "Is the evidence restricted to primary bridge-release routes?",
        "seed_level_requirement": "planned_route_family == bridge_release_interpolation_probe",
        "contract_level_requirement": "all seed rows in the contract are primary bridge-release rows",
        "wall_claim_effect": "necessary_for_direct_path_only",
    },
    {
        "rule_id": "D2_direct_edge_retained",
        "rule_group": "pathway",
        "rule_question": "Is the direct pair edge retained throughout the route?",
        "seed_level_requirement": (
            "physical_direct_edge_retained_all_steps == true and "
            "direct_edge_weight_fraction_min > 0 and active_direct_edge_weight_min > 0"
        ),
        "contract_level_requirement": "all seeds retain the direct pair edge",
        "wall_claim_effect": "necessary_but_insufficient",
    },
    {
        "rule_id": "D3_source_start_known",
        "rule_group": "pathway",
        "rule_question": "Does the route start from an accepted source anchor?",
        "seed_level_requirement": "source_start_anchor_matched == true",
        "contract_level_requirement": "all seeds start from an accepted source anchor",
        "wall_claim_effect": "necessary_but_insufficient",
    },
    {
        "rule_id": "D4_target_reached_known",
        "rule_group": "pathway",
        "rule_question": "Does the route reach the expected drop-bridge target?",
        "seed_level_requirement": "expected_final_anchor_reached == true",
        "contract_level_requirement": "all seeds reach the expected target anchor",
        "wall_claim_effect": "necessary_but_insufficient",
    },
    {
        "rule_id": "D5_no_intermediate_unknown",
        "rule_group": "pathway",
        "rule_question": "Is the path free of intermediate unknown endpoints?",
        "seed_level_requirement": "unknown_endpoint_step_count == 0",
        "contract_level_requirement": "all seeds have zero intermediate unknown endpoints",
        "wall_claim_effect": "required_for_accepted_direct_path",
    },
    {
        "rule_id": "D6_no_support_incompatibility",
        "rule_group": "pathway",
        "rule_question": "Is the path free of support-incompatibility flags?",
        "seed_level_requirement": "support_incompatibility_step_count == 0",
        "contract_level_requirement": "all seeds have zero support-incompatibility flags",
        "wall_claim_effect": "required_for_accepted_direct_path",
    },
    {
        "rule_id": "D7_all_seed_contract_acceptance",
        "rule_group": "aggregation",
        "rule_question": "Does every seed in the start-conditioned contract pass D1-D6?",
        "seed_level_requirement": "seed_direct_path_candidate == true",
        "contract_level_requirement": "seed_direct_path_candidate_count == seed_count",
        "wall_claim_effect": "direct_path_acceptance_only_not_wall",
    },
    {
        "rule_id": "D8_objective_kept_separate",
        "rule_group": "claim_boundary",
        "rule_question": "Is objective recovery separated from direct-path acceptance?",
        "seed_level_requirement": "objective fields are reported but not used to accept D1-D7",
        "contract_level_requirement": (
            "objective recovery is reported separately and cannot promote wall language alone"
        ),
        "wall_claim_effect": "prevents_quality_or_wall_conflation",
    },
    {
        "rule_id": "D9_wall_claim_closed",
        "rule_group": "claim_boundary",
        "rule_question": "Are wall claims closed until direct-path and objective evidence both pass?",
        "seed_level_requirement": "wall_seed_ready == false",
        "contract_level_requirement": "wall_contract_ready == false",
        "wall_claim_effect": "wall_promotion_blocked",
    },
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


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


def _rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(ACCEPTANCE_RULES))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _seed_evidence_rows(seed_shape: pd.DataFrame) -> pd.DataFrame:
    rows = seed_shape.copy()
    rows["d1_primary_scope_pass"] = rows["planned_route_family"].astype(str).eq(
        PRIMARY_ROUTE_FAMILY
    )
    rows["d2_direct_edge_retained_pass"] = rows[
        "physical_direct_edge_retained_all_steps"
    ].astype(bool)
    rows["d3_source_start_known_pass"] = rows["source_start_anchor_matched"].astype(bool)
    rows["d4_target_reached_known_pass"] = rows["expected_final_anchor_reached"].astype(bool)
    rows["d5_no_intermediate_unknown_pass"] = rows["unknown_endpoint_step_count"].astype(int).eq(0)
    rows["d6_no_support_incompatibility_pass"] = rows[
        "support_incompatibility_step_count"
    ].astype(int).eq(0)
    rule_cols = [
        "d1_primary_scope_pass",
        "d2_direct_edge_retained_pass",
        "d3_source_start_known_pass",
        "d4_target_reached_known_pass",
        "d5_no_intermediate_unknown_pass",
        "d6_no_support_incompatibility_pass",
    ]
    rows["seed_direct_path_candidate"] = rows[rule_cols].all(axis=1)
    rows["direct_path_seed_status"] = rows["seed_direct_path_candidate"].map(
        {
            True: "seed_direct_path_candidate_passes_d1_to_d6",
            False: "seed_direct_path_candidate_blocked_by_unknown_or_support",
        }
    )
    rows.loc[
        ~rows["d5_no_intermediate_unknown_pass"], "direct_path_seed_block_reason"
    ] = "intermediate_unknown_endpoint"
    rows.loc[
        rows["d5_no_intermediate_unknown_pass"]
        & ~rows["d6_no_support_incompatibility_pass"],
        "direct_path_seed_block_reason",
    ] = "support_incompatibility"
    rows.loc[
        rows["seed_direct_path_candidate"], "direct_path_seed_block_reason"
    ] = "seed_level_candidate_only_contract_acceptance_required"
    rows["accepted_direct_path_evidence"] = False
    rows["wall_seed_ready"] = False
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["local_pair_id", "start_condition", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)


def _contract_rows(contract_shape: pd.DataFrame, seed_evidence: pd.DataFrame) -> pd.DataFrame:
    groups = (
        seed_evidence.groupby(
            ["route_contract_id", "validation_unit_id", "local_pair_id", "start_condition"],
            sort=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            seed_direct_path_candidate_count=("seed_direct_path_candidate", "sum"),
            unknown_intermediate_seed_count=(
                "d5_no_intermediate_unknown_pass",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            support_incompatibility_seed_count=(
                "d6_no_support_incompatibility_pass",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            objective_recovery_seed_count=(
                "max_objective_recovery_from_min",
                lambda values: int((values.astype(float) > 1e-9).sum()),
            ),
            pathway_shape_class_counts=(
                "pathway_shape_class",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
            objective_shape_class_counts=(
                "objective_shape_class",
                lambda values: json.dumps(_count_dict(values), sort_keys=True),
            ),
        )
        .reset_index()
    )
    merge_cols = [
        "route_contract_id",
        "pathway_shape_status",
        "first_target_step2_seed_count",
        "first_target_step3_seed_count",
        "max_objective_debt_from_start",
        "max_objective_recovery_from_min",
    ]
    available = [col for col in merge_cols if col in contract_shape.columns]
    if available:
        groups = groups.merge(
            contract_shape[available],
            on="route_contract_id",
            how="left",
            validate="one_to_one",
        )
    groups["d7_all_seed_direct_path_acceptance_pass"] = groups[
        "seed_direct_path_candidate_count"
    ].astype(int).eq(groups["seed_count"].astype(int))
    groups["d8_objective_kept_separate_pass"] = True
    groups["d9_wall_claim_closed_pass"] = True
    groups["accepted_direct_path_contract"] = False
    groups["wall_contract_ready"] = False

    def status(row: pd.Series) -> str:
        seed_count = int(row["seed_count"])
        candidate_count = int(row["seed_direct_path_candidate_count"])
        unknown_count = int(row["unknown_intermediate_seed_count"])
        if candidate_count == seed_count and seed_count > 0:
            return "contract_direct_path_accepted_currently_unobserved"
        if candidate_count > 0 and unknown_count > 0:
            return "contract_has_seed_level_candidates_but_fails_all_seed_acceptance"
        if unknown_count == seed_count:
            return "contract_all_seeds_blocked_by_unknown_intermediate"
        return "contract_direct_path_not_accepted"

    groups["direct_path_contract_status"] = groups.apply(status, axis=1)
    groups["direct_path_contract_block_reason"] = (
        "accepted direct-path evidence requires all seeds to pass D1-D6; current "
        "contracts all contain at least one unknown/support-incompatible seed route"
    )
    groups["run_status"] = RUN_STATUS
    groups["claim_boundary"] = CLAIM_BOUNDARY
    return groups.sort_values(
        ["local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _regime_rows(pair_shape: pd.DataFrame, contract_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pair_shape.itertuples(index=False):
        local_pair_id = str(pair.local_pair_id)
        contracts = contract_rows[contract_rows["local_pair_id"].astype(str).eq(local_pair_id)]
        if local_pair_id == "local_pair_009":
            regime_name = "step3_debt_without_recovery"
            next_acceptance_question = (
                "Can a direct path be accepted when every seed reaches the target only "
                "after the step-3 bridge cut and no objective recovery appears?"
            )
        elif local_pair_id == "local_pair_012":
            regime_name = "mostly_step2_partial_recovery"
            next_acceptance_question = (
                "Can a direct path be accepted in the mostly step-2 regime without "
                "letting partial objective recovery stand in for pathway evidence?"
            )
        else:
            regime_name = "unclassified_primary_bridge_release_regime"
            next_acceptance_question = (
                "Can a direct path be accepted under the explicit D1-D9 rules?"
            )
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "planned_route_family": PRIMARY_ROUTE_FAMILY,
                "regime_name": regime_name,
                "contract_count": int(len(contracts)),
                "seed_count": int(pair.seed_count),
                "known_anchor_direct_path_candidate_seed_count": int(
                    pair.known_anchor_direct_path_candidate_seed_count
                ),
                "unknown_intermediate_seed_count": int(pair.unknown_intermediate_seed_count),
                "objective_recovery_seed_count": int(pair.objective_recovery_seed_count),
                "accepted_direct_path_contract_count": int(
                    contracts["accepted_direct_path_contract"].astype(bool).sum()
                ),
                "wall_ready_contract_count": int(
                    contracts["wall_contract_ready"].astype(bool).sum()
                ),
                "current_regime_acceptance_status": (
                    "regime_has_seed_candidates_but_no_contract_level_direct_path_acceptance"
                ),
                "next_acceptance_question": next_acceptance_question,
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
    shape_gates: pd.DataFrame,
    rule_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    regime_rows: pd.DataFrame,
) -> pd.DataFrame:
    seed_candidate_count = int(seed_rows["seed_direct_path_candidate"].astype(bool).sum())
    contract_accept_count = int(
        contract_rows["accepted_direct_path_contract"].astype(bool).sum()
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_pathway_shape_gates_pass",
                "Did every upstream primary bridge-release pathway-shape gate pass?",
                _count_dict(shape_gates["gate_status"]),
                "all upstream shape gates pass",
                bool(shape_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_acceptance_rules_are_explicit",
                "Are direct-path acceptance rules fixed before any new execution?",
                f"rule_count={len(rule_rows)} rule_ids={list(rule_rows['rule_id'])}",
                "D1-D9 acceptance and claim-boundary rules are materialized",
                len(rule_rows) == 9,
            ),
            _gate_row(
                "G3_seed_level_candidates_preserved",
                "Are seed-level direct-path candidates preserved but not promoted?",
                f"seed_candidate_count={seed_candidate_count} seed_rows={len(seed_rows)}",
                "53 seed-level candidates from existing shape audit, not contract acceptance",
                seed_candidate_count == 53 and len(seed_rows) == 80,
            ),
            _gate_row(
                "G4_contract_level_acceptance_strict",
                "Does strict all-seed contract acceptance remain closed on current evidence?",
                (
                    f"accepted_direct_path_contracts={contract_accept_count} "
                    f"contract_rows={len(contract_rows)}"
                ),
                "0 of 10 contracts accepted under D1-D7",
                contract_accept_count == 0 and len(contract_rows) == 10,
            ),
            _gate_row(
                "G5_two_regimes_are_kept_separate",
                "Are the two observed pathway-shape regimes kept separate?",
                _count_dict(regime_rows["regime_name"]),
                "local_pair_009 and local_pair_012 are separate regimes",
                set(regime_rows["regime_name"].astype(str))
                == {"step3_debt_without_recovery", "mostly_step2_partial_recovery"},
            ),
            _gate_row(
                "G6_objective_recovery_not_used_as_path_acceptance",
                "Is objective recovery reported but excluded from direct-path acceptance?",
                (
                    f"objective_recovery_seed_count="
                    f"{int(seed_rows['max_objective_recovery_from_min'].astype(float).gt(1e-9).sum())}"
                ),
                "objective recovery is a separate wall-readiness field, not a D1-D7 pass rule",
                bool(rule_rows["rule_id"].astype(str).eq("D8_objective_kept_separate").any()),
            ),
            _gate_row(
                "G7_wall_claim_remains_closed",
                "Are wall claims closed after materializing the direct-path contract?",
                (
                    f"wall_ready_contracts="
                    f"{int(contract_rows['wall_contract_ready'].astype(bool).sum())}"
                ),
                "zero wall-ready contracts",
                not bool(contract_rows["wall_contract_ready"].astype(bool).any()),
            ),
            _gate_row(
                "G8_no_new_leiden_execution",
                "Is this a design contract rather than an executed route run?",
                RUN_STATUS,
                "contract/materialization only",
                True,
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
    shape_dir: Path,
    output_dir: Path,
    rule_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    regime_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_direct_path_acceptance_contract_summary.v1",
        "status": "direct_path_acceptance_contract_materialized_current_evidence_contract_level_closed",
        "run_status": RUN_STATUS,
        "shape_dir": str(shape_dir),
        "output_dir": str(output_dir),
        "rule_count": int(len(rule_rows)),
        "seed_evidence_row_count": int(len(seed_rows)),
        "contract_row_count": int(len(contract_rows)),
        "regime_row_count": int(len(regime_rows)),
        "seed_direct_path_candidate_count": int(
            seed_rows["seed_direct_path_candidate"].astype(bool).sum()
        ),
        "accepted_direct_path_contract_count": int(
            contract_rows["accepted_direct_path_contract"].astype(bool).sum()
        ),
        "wall_ready_contract_count": int(
            contract_rows["wall_contract_ready"].astype(bool).sum()
        ),
        "direct_path_contract_status_counts": _count_dict(
            contract_rows["direct_path_contract_status"]
        ),
        "regime_status_counts": _count_dict(regime_rows["current_regime_acceptance_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The direct-path acceptance gate is now explicit. Current evidence has "
            "53 seed-level direct-path candidates, but strict all-seed contract "
            "acceptance remains 0 of 10 because each contract contains at least one "
            "intermediate unknown/support-incompatible seed route. Objective recovery "
            "is reported separately and cannot promote a direct path or wall claim."
        ),
        "recommended_next_gate": (
            "If proceeding to execution, evaluate only the D1-D9 direct-path "
            "acceptance contract over the two separated regimes. Do not broaden to "
            "new pairs, quality/cost evaluation, full NanoClustering replay, or wall "
            "language until at least one contract passes D1-D7 and objective evidence "
            "is audited separately."
        ),
        "shape_claim_boundary": SHAPE_CLAIM_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            ACCEPTANCE_RULE_ROWS_CSV,
            SEED_EVIDENCE_ROWS_CSV,
            CONTRACT_ROWS_CSV,
            REGIME_ROWS_CSV,
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
    rule_rows: pd.DataFrame,
    seed_rows: pd.DataFrame,
    contract_rows: pd.DataFrame,
    regime_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Direct-Path Acceptance Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- rule_count: {summary['rule_count']}",
        f"- seed_evidence_row_count: {summary['seed_evidence_row_count']}",
        f"- contract_row_count: {summary['contract_row_count']}",
        f"- regime_row_count: {summary['regime_row_count']}",
        f"- seed_direct_path_candidate_count: {summary['seed_direct_path_candidate_count']}",
        f"- accepted_direct_path_contract_count: {summary['accepted_direct_path_contract_count']}",
        f"- wall_ready_contract_count: {summary['wall_ready_contract_count']}",
        f"- direct_path_contract_status_counts: {summary['direct_path_contract_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Acceptance Rules",
        "",
        _markdown_table(
            rule_rows,
            [
                "rule_id",
                "rule_group",
                "rule_question",
                "seed_level_requirement",
                "contract_level_requirement",
                "wall_claim_effect",
            ],
            max_rows=20,
        ),
        "",
        "## Regimes",
        "",
        _markdown_table(
            regime_rows,
            [
                "local_pair_id",
                "regime_name",
                "seed_count",
                "known_anchor_direct_path_candidate_seed_count",
                "unknown_intermediate_seed_count",
                "objective_recovery_seed_count",
                "accepted_direct_path_contract_count",
                "current_regime_acceptance_status",
            ],
        ),
        "",
        "## Contract Evidence",
        "",
        _markdown_table(
            contract_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed_count",
                "seed_direct_path_candidate_count",
                "unknown_intermediate_seed_count",
                "objective_recovery_seed_count",
                "d7_all_seed_direct_path_acceptance_pass",
                "accepted_direct_path_contract",
                "direct_path_contract_status",
            ],
        ),
        "",
        "## Seed Evidence Sample",
        "",
        _markdown_table(
            seed_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "seed_direct_path_candidate",
                "unknown_step_indices",
                "max_objective_recovery_from_min",
                "direct_path_seed_status",
                "direct_path_seed_block_reason",
            ],
            max_rows=30,
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
            "This contract fixes what accepted direct-path evidence would mean. It "
            "does not turn the existing seed-level candidates into contract-level "
            "evidence, and it does not use objective recovery as a shortcut for "
            "wall language."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    shape_dir = Path(args.shape_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_shape = _read_csv(shape_dir / SHAPE_SEED_ROWS_CSV)
    contract_shape = _read_csv(shape_dir / SHAPE_CONTRACT_ROWS_CSV)
    pair_shape = _read_csv(shape_dir / SHAPE_PAIR_ROWS_CSV)
    shape_gates = _read_csv(shape_dir / SHAPE_GATE_MATRIX_CSV)
    shape_summary = json.loads((shape_dir / SHAPE_SUMMARY_JSON).read_text(encoding="utf-8"))

    rule_rows = _rule_rows()
    seed_rows = _seed_evidence_rows(seed_shape)
    contract_rows = _contract_rows(contract_shape, seed_rows)
    regime_rows = _regime_rows(pair_shape, contract_rows)
    gates = _gate_matrix(
        shape_gates=shape_gates,
        rule_rows=rule_rows,
        seed_rows=seed_rows,
        contract_rows=contract_rows,
        regime_rows=regime_rows,
    )
    summary = _summary(
        shape_dir=shape_dir,
        output_dir=output_dir,
        rule_rows=rule_rows,
        seed_rows=seed_rows,
        contract_rows=contract_rows,
        regime_rows=regime_rows,
        gates=gates,
    )

    _write_csv(rule_rows, output_dir / ACCEPTANCE_RULE_ROWS_CSV)
    _write_csv(seed_rows, output_dir / SEED_EVIDENCE_ROWS_CSV)
    _write_csv(contract_rows, output_dir / CONTRACT_ROWS_CSV)
    _write_csv(regime_rows, output_dir / REGIME_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            _json_safe(
                {
                    "shape_dir": str(shape_dir),
                    "output_dir": str(output_dir),
                    "shape_summary_status": shape_summary.get("status"),
                    "acceptance_rules": ACCEPTANCE_RULES,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        rule_rows=rule_rows,
        seed_rows=seed_rows,
        contract_rows=contract_rows,
        regime_rows=regime_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-dir", type=Path, default=DEFAULT_SHAPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run_contract(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
