#!/usr/bin/env python3
"""Derive synthetic-demo design families from local variable-pair ablations.

This script reads the frozen variable-pair counterfactual panel and the first
local Leiden+CPM ablation gate. It does not run Leiden. It classifies each
panel pair into a synthetic-demo design role and materializes the design axes
needed for the next controlled demo.

The intent is to prevent the next step from becoming another broad sweep. The
design target is direct-contact phase sensitivity under bridge-context
competition, with explicit controls for coupled direct+bridge cases and
non-local context misses.
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
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_counterfactual_panel_gamma1e5_20260603"
)
DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_gamma1e5_20260603"
)

PANEL_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_counterfactual_panel_rows.csv"
)
LOCAL_GATE_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_pair_gate_rows.csv"
)
LOCAL_GRAPH_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_graph_rows.csv"
)
LOCAL_VARIANT_SUMMARY_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_variant_summary.csv"
)

PAIR_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_pair_rows.csv"
)
FAMILY_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_family_rows.csv"
)
AXIS_ROWS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_axis_rows.csv"
)
SUMMARY_JSON = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_config.json"
)
REPORT_MD = (
    "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_report.md"
)

RUN_STATUS = "designed_symmetric_object_variable_pair_synthetic_demo"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object variable-pair synthetic demo design only; "
    "derives controlled demo families from frozen panel and local ablation "
    "artifacts. It does not run Leiden, execute full-graph replay, promote "
    "walls/pathways, inspect quality/cost as success claims, or claim "
    "method/algorithm success."
)


def _prefix_stats(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_median": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
    }


def _design_family(row: pd.Series) -> tuple[str, str]:
    gate_class = str(row["gate_class"])
    original = float(row["original_pair_coassigned_share"])
    drop_bridge = float(row["drop_bridge_pair_coassigned_share"])
    counterfactual = str(row["counterfactual_class"])
    if gate_class == "direct_and_bridge_sensitive_local_switch":
        return (
            "coupled_negative_direct_bridge_contact",
            "negative direct CPM pair still coassigns locally only when direct contact and bridge context are both present",
        )
    if gate_class == "direct_edge_sensitive_local_switch":
        if original >= 0.70:
            return (
                "stable_direct_contact_competition",
                "direct pair contact is a stable local necessary condition under bridge-context competition",
            )
        return (
            "partial_direct_contact_competition",
            "direct pair contact is necessary but local endpoint choice remains partial under bridge-context competition",
        )
    if gate_class == "local_seed_or_start_sensitive_switch":
        return (
            "rare_start_sensitive_direct_contact",
            "direct contact can reproduce coassignment only in rare seed/start endpoints near a phase boundary",
        )
    if gate_class == "not_reproduced_no_original_local_coassignment":
        if drop_bridge >= 0.90:
            return (
                "overcompeting_bridge_context_control",
                "selected bridge context suppresses local pair coassignment; removing pair-to-bridge edges collapses the pair",
            )
        if counterfactual == "bridge_mediated_negative_direct_pair":
            return (
                "nonlocal_negative_direct_context_control",
                "negative-direct bridge candidate does not reproduce in the recoverable local induced graph",
            )
        return (
            "nonlocal_or_missing_context_control",
            "recoverable local induced graph does not reproduce the observed terminal coassignment",
        )
    return ("unclassified_design_family", "unclassified local ablation pattern")


def _axis_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    axis_defs = [
        (
            "direct_dependency",
            "original_pair_coassigned_share - drop_direct_pair_coassigned_share",
            "How much direct edge removal suppresses pair coassignment.",
            "higher means direct contact is locally necessary",
            "direct_dependency_score",
        ),
        (
            "bridge_context_competition",
            "drop_bridge_pair_coassigned_share - original_pair_coassigned_share",
            "How much removing pair-to-bridge edges increases pair coassignment.",
            "higher means selected bridge context competes with pair coassignment",
            "bridge_competition_score",
        ),
        (
            "direct_and_bridge_coupling",
            "min(original-drop_direct, original-drop_bridge)",
            "Whether direct and bridge edges are jointly needed.",
            "higher means the local coassignment depends on both direct and bridge edges",
            "direct_bridge_coupling_score",
        ),
        (
            "local_reproducibility",
            "original_pair_coassigned_share",
            "Whether the small induced graph reproduces the observed variable-pair coassignment.",
            "higher means the local graph can reproduce the pair coassignment",
            "local_reproducibility_score",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for axis_id, formula, description, interpretation, column in axis_defs:
        family_stats = []
        for family, group in pair_rows.groupby("design_family", sort=True):
            family_stats.append(
                {
                    "design_family": family,
                    "count": int(len(group)),
                    "median": float(group[column].median()),
                    "min": float(group[column].min()),
                    "max": float(group[column].max()),
                }
            )
        rows.append(
            {
                "axis_id": axis_id,
                "formula": formula,
                "description": description,
                "interpretation": interpretation,
                "source_column": column,
                "family_stats": json.dumps(family_stats, sort_keys=True),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_rows(
    *,
    panel: pd.DataFrame,
    gate: pd.DataFrame,
    graph_rows: pd.DataFrame,
    variant_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = gate.merge(
        panel,
        on=["object_role_universe_id", "left_node_id", "right_node_id", "counterfactual_class"],
        suffixes=("", "_panel"),
    ).merge(
        graph_rows[
            [
                "local_pair_id",
                "selected_bridge_count",
                "excluded_outside_bridge_count",
                "local_node_count",
                "local_doc_missing_node_count",
            ]
        ],
        on="local_pair_id",
        how="left",
    )
    original = variant_summary[variant_summary["graph_variant"].astype(str).eq("original")]
    rows = rows.merge(
        original[
            [
                "local_pair_id",
                "distinct_endpoint_count",
                "recurrent_endpoint_count",
                "top_endpoint_share",
                "pair_bridge_same_cluster_median",
                "mechanism_read_counts",
            ]
        ].rename(
            columns={
                "distinct_endpoint_count": "original_distinct_endpoint_count_variant",
                "recurrent_endpoint_count": "original_recurrent_endpoint_count_variant",
                "top_endpoint_share": "original_top_endpoint_share",
                "pair_bridge_same_cluster_median": "original_pair_bridge_same_cluster_median",
                "mechanism_read_counts": "original_mechanism_read_counts",
            }
        ),
        on="local_pair_id",
        how="left",
    )
    family_payloads = rows.apply(_design_family, axis=1)
    rows["design_family"] = [item[0] for item in family_payloads]
    rows["design_rationale"] = [item[1] for item in family_payloads]
    rows["direct_dependency_score"] = (
        rows["original_pair_coassigned_share"].astype(float)
        - rows["drop_direct_pair_coassigned_share"].astype(float)
    )
    rows["bridge_competition_score"] = (
        rows["drop_bridge_pair_coassigned_share"].astype(float)
        - rows["original_pair_coassigned_share"].astype(float)
    )
    rows["direct_bridge_coupling_score"] = np.minimum(
        rows["original_pair_coassigned_share"].astype(float)
        - rows["drop_direct_pair_coassigned_share"].astype(float),
        rows["original_pair_coassigned_share"].astype(float)
        - rows["drop_bridge_pair_coassigned_share"].astype(float),
    )
    rows["local_reproducibility_score"] = rows["original_pair_coassigned_share"].astype(float)
    rows["synthetic_demo_role"] = rows["design_family"].map(
        {
            "stable_direct_contact_competition": "positive_family_primary",
            "partial_direct_contact_competition": "positive_family_boundary",
            "coupled_negative_direct_bridge_contact": "positive_family_coupled_control",
            "rare_start_sensitive_direct_contact": "near_boundary_stress_case",
            "overcompeting_bridge_context_control": "negative_context_control",
            "nonlocal_negative_direct_context_control": "negative_locality_control",
            "nonlocal_or_missing_context_control": "negative_locality_control",
        }
    ).fillna("unclassified_role")
    rows["next_demo_requirements"] = rows["design_family"].map(
        {
            "stable_direct_contact_competition": (
                "include direct pair edge, symmetric bridge competitors, and host-context edges; "
                "direct removal must separate the pair while bridge removal collapses it"
            ),
            "partial_direct_contact_competition": (
                "same as stable family but tune bridge/context strength until original endpoints are mixed"
            ),
            "coupled_negative_direct_bridge_contact": (
                "use negative direct CPM margin but enough bridge-supported contact that original coassigns; "
                "both direct and bridge removal should break coassignment"
            ),
            "rare_start_sensitive_direct_contact": (
                "tune near phase boundary so coassignment is rare and start/seed-sensitive"
            ),
            "overcompeting_bridge_context_control": (
                "preserve as a negative control where bridge context suppresses coassignment"
            ),
            "nonlocal_negative_direct_context_control": (
                "do not use as primary synthetic positive; requires missing nonlocal context audit"
            ),
            "nonlocal_or_missing_context_control": (
                "do not use as primary synthetic positive; requires larger context or outside bridge inclusion"
            ),
        }
    ).fillna("unclassified requirement")
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    preferred = [
        "local_pair_id",
        "object_role_universe_id",
        "branch",
        "left_node_id",
        "right_node_id",
        "pair_scope",
        "counterfactual_class",
        "gate_class",
        "gate_status",
        "design_family",
        "synthetic_demo_role",
        "design_rationale",
        "next_demo_requirements",
        "original_pair_coassigned_share",
        "drop_direct_pair_coassigned_share",
        "drop_bridge_pair_coassigned_share",
        "drop_direct_and_bridge_pair_coassigned_share",
        "direct_dependency_score",
        "bridge_competition_score",
        "direct_bridge_coupling_score",
        "local_reproducibility_score",
        "direct_cpm_delta_q",
        "direct_edge_weight",
        "direct_critical_gamma",
        "common_neighbor_min_weight_sum",
        "bridge_to_direct_weight_ratio",
        "selected_bridge_count",
        "excluded_outside_bridge_count",
        "local_node_count",
        "local_doc_missing_node_count",
        "original_top_endpoint_share",
        "original_pair_bridge_same_cluster_median",
        "original_mechanism_read_counts",
        "selection_reason",
        "run_status",
        "claim_boundary",
    ]
    return rows[preferred].sort_values(
        ["synthetic_demo_role", "design_family", "local_pair_id"],
        kind="mergesort",
    )


def _family_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in pair_rows.groupby("design_family", sort=True):
        row = {
            "design_family": family,
            "synthetic_demo_role": str(group["synthetic_demo_role"].iloc[0]),
            "pair_count": int(len(group)),
            "pair_ids": ";".join(group["local_pair_id"].astype(str).tolist()),
            "pair_scope_counts": json.dumps(group["pair_scope"].value_counts().to_dict(), sort_keys=True),
            "counterfactual_class_counts": json.dumps(
                group["counterfactual_class"].value_counts().to_dict(),
                sort_keys=True,
            ),
            "gate_class_counts": json.dumps(group["gate_class"].value_counts().to_dict(), sort_keys=True),
            "representative_pair_id": str(
                group.sort_values(
                    ["local_reproducibility_score", "direct_dependency_score"],
                    ascending=[False, False],
                    kind="mergesort",
                )["local_pair_id"].iloc[0]
            ),
            "next_demo_requirements": str(group["next_demo_requirements"].iloc[0]),
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        for prefix, column in [
            ("direct_dependency", "direct_dependency_score"),
            ("bridge_competition", "bridge_competition_score"),
            ("direct_bridge_coupling", "direct_bridge_coupling_score"),
            ("local_reproducibility", "local_reproducibility_score"),
            ("direct_edge_weight", "direct_edge_weight"),
            ("direct_critical_gamma", "direct_critical_gamma"),
            ("bridge_to_direct_ratio", "bridge_to_direct_weight_ratio"),
        ]:
            row.update(_prefix_stats(prefix, group[column].to_numpy(dtype=np.float64)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["synthetic_demo_role", "design_family"])


def _build_summary(
    *,
    pair_rows: pd.DataFrame,
    family_rows: pd.DataFrame,
    output_dir: Path,
    panel_dir: Path,
    local_ablation_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "panel_dir": str(panel_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "pair_count": int(len(pair_rows)),
        "design_family_count": int(pair_rows["design_family"].nunique()),
        "design_family_counts": pair_rows["design_family"].value_counts().to_dict(),
        "synthetic_demo_role_counts": pair_rows["synthetic_demo_role"].value_counts().to_dict(),
        "primary_positive_pair_count": int(
            pair_rows["synthetic_demo_role"].astype(str).isin(
                ["positive_family_primary", "positive_family_boundary"]
            ).sum()
        ),
        "near_boundary_pair_count": int(
            pair_rows["synthetic_demo_role"].astype(str).eq("near_boundary_stress_case").sum()
        ),
        "negative_control_pair_count": int(
            pair_rows["synthetic_demo_role"].astype(str).str.startswith("negative_").sum()
        ),
        "family_rows": family_rows.to_dict("records"),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    family_rows: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Variable-Pair Synthetic Demo Design",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- design_family_counts: {summary['design_family_counts']}",
        f"- synthetic_demo_role_counts: {summary['synthetic_demo_role_counts']}",
        f"- primary_positive_pair_count: {summary['primary_positive_pair_count']}",
        f"- near_boundary_pair_count: {summary['near_boundary_pair_count']}",
        f"- negative_control_pair_count: {summary['negative_control_pair_count']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Family Design",
    ]
    for row in family_rows.itertuples(index=False):
        lines.append(
            "- "
            f"{row.design_family}: role={row.synthetic_demo_role}, "
            f"pairs={row.pair_count}, representative={row.representative_pair_id}, "
            f"direct_dependency_median={row.direct_dependency_median}, "
            f"bridge_competition_median={row.bridge_competition_median}, "
            f"local_reproducibility_median={row.local_reproducibility_median}, "
            f"requirements={row.next_demo_requirements}"
        )
    lines.extend(
        [
            "",
            "## Design Read",
            "",
            (
                "The next synthetic demo should not be a generic bridge-merge "
                "example. The local ablation evidence points to direct-contact "
                "phase sensitivity under bridge-context competition. Positive "
                "families should therefore include a direct pair edge, bridge "
                "competitors, and host-context edges; controls should include "
                "overcompeting bridge context and missing/nonlocal-context cases."
            ),
            "",
            "## Boundary",
            "",
            (
                "This is a design artifact only. It does not run Leiden, replay "
                "the full graph, execute routes/pathways, promote walls, compare "
                "quality/cost, or claim a method."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    panel_dir = Path(args.panel_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _read_csv(panel_dir / PANEL_ROWS_CSV)
    gate = _read_csv(local_ablation_dir / LOCAL_GATE_ROWS_CSV)
    graph_rows = _read_csv(local_ablation_dir / LOCAL_GRAPH_ROWS_CSV)
    variant_summary = _read_csv(local_ablation_dir / LOCAL_VARIANT_SUMMARY_CSV)

    pair_rows = _pair_rows(
        panel=panel,
        gate=gate,
        graph_rows=graph_rows,
        variant_summary=variant_summary,
    )
    family_rows = _family_rows(pair_rows)
    axis_rows = _axis_rows(pair_rows)

    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(family_rows, output_dir / FAMILY_ROWS_CSV)
    _write_csv(axis_rows, output_dir / AXIS_ROWS_CSV)
    summary = _build_summary(
        pair_rows=pair_rows,
        family_rows=family_rows,
        output_dir=output_dir,
        panel_dir=panel_dir,
        local_ablation_dir=local_ablation_dir,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_variable_pair_synthetic_demo_design.v1",
        "panel_dir": str(panel_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, family_rows=family_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
