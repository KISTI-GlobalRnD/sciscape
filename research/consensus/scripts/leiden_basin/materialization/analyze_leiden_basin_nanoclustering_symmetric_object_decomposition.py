#!/usr/bin/env python3
"""Decompose symmetric NanoClustering endpoint objects into next-use classes.

This reads the all-seed symmetric endpoint-object audit and separates stable
objects, anchor-local fragments, multi-cluster objects, and seed0 mapping
failure modes. It is a definition-diagnostic pass only. It does not rerun
clustering, execute routes, promote wall/pathway claims, inspect quality/cost,
or claim a method improvement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_SYMMETRIC_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_symmetric_endpoint_objects_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_decomposition_v1_20260531"
)

OBJECT_COMPONENTS_CSV = "nanoclustering_symmetric_endpoint_object_components.csv"
OVERLAP_EDGES_CSV = "nanoclustering_symmetric_endpoint_overlap_edges.csv"
SEED0_MAPPING_CSV = "nanoclustering_seed0_v2_2_mapping_to_symmetric_objects.csv"

OBJECT_DECOMPOSITION_CSV = "nanoclustering_symmetric_object_decomposition.csv"
SEED0_FAILURE_MODES_CSV = "nanoclustering_seed0_tier_mapping_failure_modes.csv"
STABLE_OBJECT_REGISTRY_CSV = "nanoclustering_stable_object_registry.csv"
ANCHOR_LOCAL_FRAGMENT_REGISTRY_CSV = "nanoclustering_anchor_local_fragment_registry.csv"
MULTI_CLUSTER_OBJECT_REGISTRY_CSV = "nanoclustering_multi_cluster_object_registry.csv"
MECHANISM_PROBE_CANDIDATES_CSV = "nanoclustering_object_to_mechanism_probe_candidates.csv"
GATE_MATRIX_CSV = "nanoclustering_symmetric_object_decomposition_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_symmetric_object_decomposition_summary.json"
CONFIG_JSON = "nanoclustering_symmetric_object_decomposition_config.json"
REPORT_MD = "nanoclustering_symmetric_object_decomposition_report.md"

CLAIM_BOUNDARY = (
    "Symmetric object decomposition only; membership-derived object/failure-mode "
    "diagnostics, no route execution, no wall/pathway promotion, no "
    "basin-quality claim, no cost claim, no directed-search claim, and no "
    "algorithm claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_symmetric_object_decomposition"

GOOD_OBJECT_SEED_COVERAGE_MIN = 5
STRONG_OBJECT_SEED_COVERAGE_MIN = 8


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _object_decomposition_class(row: pd.Series) -> str:
    seed_coverage = int(row["seed_coverage_count"])
    max_per_seed = int(row["max_endpoint_nodes_per_seed"])
    if seed_coverage >= GOOD_OBJECT_SEED_COVERAGE_MIN and max_per_seed <= 1:
        if seed_coverage >= STRONG_OBJECT_SEED_COVERAGE_MIN:
            return "stable_one_per_seed_object_strong"
        return "stable_one_per_seed_object_good"
    if seed_coverage >= GOOD_OBJECT_SEED_COVERAGE_MIN and max_per_seed > 1:
        if seed_coverage >= STRONG_OBJECT_SEED_COVERAGE_MIN:
            return "stable_multi_cluster_object_strong"
        return "stable_multi_cluster_object_good"
    if seed_coverage >= 2:
        return "partial_object"
    return "anchor_local_fragment"


def _mechanism_probe_label(row: pd.Series) -> str:
    if int(row["seed0_source_family_count"]) > 1:
        return "merged_seed0_family_probe"
    if int(row["max_endpoint_nodes_per_seed"]) > 1:
        if int(row["multi_endpoint_seed_count"]) >= 3:
            return "multi_seed_split_merge_probe"
        return "localized_multi_cluster_probe"
    if int(row["seed_coverage_count"]) < GOOD_OBJECT_SEED_COVERAGE_MIN:
        return "anchor_local_fragment_review"
    if int(row["seed0_source_family_count"]) > 0:
        return "mapped_stable_object_control_or_probe"
    return "unmapped_stable_object_control"


def _probe_priority(row: pd.Series) -> str:
    if str(row["mechanism_probe_label"]) == "multi_seed_split_merge_probe":
        return "P1_multi_seed_mechanism_probe"
    if int(row["seed0_source_family_count"]) > 1:
        return "P1_merged_seed0_family_probe"
    if int(row["seed0_t1_family_count"]) > 0 and str(row["decomposition_class"]).startswith(
        "stable"
    ):
        return "P2_t1_stable_object_probe"
    if str(row["decomposition_class"]) == "anchor_local_fragment":
        return "P4_anchor_local_review"
    return "P3_control_or_background"


def _object_summary(components: pd.DataFrame, seed0_mapping: pd.DataFrame) -> pd.DataFrame:
    object_rows: list[dict[str, Any]] = []
    for (branch, object_id), group in components.groupby(
        ["branch", "symmetric_object_id"],
        sort=True,
    ):
        seed_counts = group.groupby("seed").size()
        object_rows.append(
            {
                "branch": str(branch),
                "symmetric_object_id": str(object_id),
                "endpoint_node_count": int(len(group)),
                "seed_coverage_count": int(group["seed"].nunique()),
                "seed_coverage_share": float(group["seed"].nunique() / 10.0),
                "seed_list": ";".join(str(int(seed)) for seed in sorted(group["seed"].unique())),
                "max_endpoint_nodes_per_seed": int(seed_counts.max()),
                "multi_endpoint_seed_count": int((seed_counts > 1).sum()),
                "endpoint_weight_sum_total": int(group["endpoint_weight_sum"].sum()),
                "endpoint_weight_sum_median": float(group["endpoint_weight_sum"].median()),
                "endpoint_unit_count_median": float(group["endpoint_unit_count"].median()),
                "contains_seed0_endpoint": bool(group["seed"].eq(0).any()),
            }
        )
    objects = pd.DataFrame(object_rows)

    mapping = seed0_mapping.copy()
    mapping["is_t1"] = mapping["claim_tiers"].astype(str).str.contains(
        "T1_stable_high_support_nucleus",
        regex=False,
    )
    seed0_counts = (
        mapping.groupby(["branch", "symmetric_object_id"], dropna=False)
        .agg(
            seed0_source_family_count=("source_family_id", "nunique"),
            seed0_primitive_count=("primitive_count", "sum"),
            seed0_event_count=("event_count", "sum"),
            seed0_t1_family_count=("is_t1", "sum"),
            mapped_claim_tiers=("claim_tiers", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    objects = objects.merge(
        seed0_counts,
        on=["branch", "symmetric_object_id"],
        how="left",
        validate="one_to_one",
    )
    fill_ints = [
        "seed0_source_family_count",
        "seed0_primitive_count",
        "seed0_event_count",
        "seed0_t1_family_count",
    ]
    for column in fill_ints:
        objects[column] = objects[column].fillna(0).astype(int)
    objects["mapped_claim_tiers"] = objects["mapped_claim_tiers"].fillna("")
    objects["decomposition_class"] = objects.apply(_object_decomposition_class, axis=1)
    objects["mechanism_probe_label"] = objects.apply(_mechanism_probe_label, axis=1)
    objects["probe_priority"] = objects.apply(_probe_priority, axis=1)
    return _with_claim_columns(
        objects.sort_values(
            [
                "branch",
                "probe_priority",
                "seed_coverage_count",
                "endpoint_node_count",
                "symmetric_object_id",
            ],
            ascending=[True, True, False, False, True],
        )
    )


def _edge_relation_by_object(sym_dir: Path, objects: pd.DataFrame) -> pd.DataFrame:
    mapping = objects[["branch", "symmetric_object_id"]].drop_duplicates()
    components = _read_csv(
        sym_dir / OBJECT_COMPONENTS_CSV,
        usecols=["endpoint_node_id", "symmetric_object_id"],
    )
    node_to_object = dict(
        zip(
            components["endpoint_node_id"].astype(str),
            components["symmetric_object_id"].astype(str),
        )
    )
    edges = _read_csv(
        sym_dir / OVERLAP_EDGES_CSV,
        usecols=[
            "branch",
            "left_endpoint_node_id",
            "right_endpoint_node_id",
            "relation_class",
            "component_link",
        ],
    )
    edges = edges[_bool_series(edges["component_link"])].copy()
    edges["left_object_id"] = edges["left_endpoint_node_id"].map(node_to_object)
    edges["right_object_id"] = edges["right_endpoint_node_id"].map(node_to_object)
    internal = edges[edges["left_object_id"].eq(edges["right_object_id"])].copy()
    relation = (
        internal.groupby(["branch", "left_object_id", "relation_class"], as_index=False)
        .size()
        .rename(columns={"left_object_id": "symmetric_object_id", "size": "edge_count"})
    )
    if relation.empty:
        mapping["dominant_internal_relation_class"] = ""
        mapping["dominant_internal_relation_edge_count"] = 0
        return mapping
    relation = relation.sort_values(
        ["branch", "symmetric_object_id", "edge_count", "relation_class"],
        ascending=[True, True, False, True],
    )
    dominant = relation.drop_duplicates(["branch", "symmetric_object_id"], keep="first")
    dominant = dominant.rename(
        columns={
            "relation_class": "dominant_internal_relation_class",
            "edge_count": "dominant_internal_relation_edge_count",
        }
    )
    return mapping.merge(
        dominant,
        on=["branch", "symmetric_object_id"],
        how="left",
        validate="one_to_one",
    ).fillna(
        {
            "dominant_internal_relation_class": "",
            "dominant_internal_relation_edge_count": 0,
        }
    )


def _mapping_failure_mode(row: pd.Series) -> str:
    if not bool(row["mapped_to_symmetric_object"]):
        return "seed0_mapping_missing"
    if bool(row["anchor_independent_candidate"]):
        if int(row["object_seed0_source_family_count"]) > 1:
            return "anchor_independent_but_seed0_merged"
        return "anchor_independent_candidate"
    if int(row["object_seed0_source_family_count"]) > 1:
        return "seed0_mapping_merged"
    if int(row["seed_coverage_count"]) <= 1:
        return "seed0_mapping_anchor_local_fragment"
    if int(row["seed_coverage_count"]) < GOOD_OBJECT_SEED_COVERAGE_MIN:
        return "seed0_mapping_partial_object"
    if int(row["max_endpoint_nodes_per_seed"]) > 1:
        return "seed0_mapping_multi_cluster_object"
    if bool(row["good_seed_coverage_object"]) and not bool(row["strong_seed_coverage_object"]):
        return "seed0_mapping_good_not_strong"
    return "seed0_mapping_other_caveat"


def _seed0_failure_modes(
    seed0_mapping: pd.DataFrame,
    object_decomposition: pd.DataFrame,
) -> pd.DataFrame:
    object_counts = object_decomposition[
        [
            "branch",
            "symmetric_object_id",
            "seed0_source_family_count",
            "decomposition_class",
            "mechanism_probe_label",
            "probe_priority",
            "dominant_internal_relation_class",
        ]
    ].rename(columns={"seed0_source_family_count": "object_seed0_source_family_count"})
    rows = seed0_mapping.merge(
        object_counts,
        on=["branch", "symmetric_object_id"],
        how="left",
        validate="many_to_one",
    )
    for column in [
        "mapped_to_symmetric_object",
        "good_seed_coverage_object",
        "strong_seed_coverage_object",
        "single_cluster_per_seed_object",
        "anchor_independent_candidate",
    ]:
        rows[column] = _bool_series(rows[column])
    rows["object_seed0_source_family_count"] = (
        rows["object_seed0_source_family_count"].fillna(0).astype(int)
    )
    rows["mapping_failure_mode"] = rows.apply(_mapping_failure_mode, axis=1)
    rows["mapping_failure_family"] = rows["mapping_failure_mode"].str.replace(
        r"^(anchor_independent|seed0_mapping)_?",
        "",
        regex=True,
    )
    return _with_claim_columns(
        rows.sort_values(["branch", "best_claim_tier_rank", "mapping_failure_mode", "source_family_id"])
    )


def _mechanism_probe_candidates(object_decomposition: pd.DataFrame) -> pd.DataFrame:
    candidates = object_decomposition[
        object_decomposition["probe_priority"].isin(
            [
                "P1_multi_seed_mechanism_probe",
                "P1_merged_seed0_family_probe",
                "P2_t1_stable_object_probe",
            ]
        )
    ].copy()
    candidates["candidate_reason"] = candidates["probe_priority"]
    candidates["candidate_status"] = "candidate_only_no_route_or_quality_claim"
    return _with_claim_columns(
        candidates.sort_values(
            ["probe_priority", "seed0_t1_family_count", "seed_coverage_count", "endpoint_node_count"],
            ascending=[True, False, False, False],
        )
    )


def _gate_matrix(
    *,
    object_decomposition: pd.DataFrame,
    failure_modes: pd.DataFrame,
    stable_registry: pd.DataFrame,
    mechanism_candidates: pd.DataFrame,
) -> pd.DataFrame:
    t1 = failure_modes[
        failure_modes["claim_tiers"].astype(str).str.contains(
            "T1_stable_high_support_nucleus",
            regex=False,
        )
    ]
    t1_anchor_share = (
        float(t1["mapping_failure_mode"].eq("anchor_independent_candidate").mean())
        if not t1.empty
        else 0.0
    )
    t1_explained_share = (
        float(t1["mapping_failure_mode"].ne("seed0_mapping_other_caveat").mean())
        if not t1.empty
        else 0.0
    )
    rows = [
        {
            "gate_id": "D1_decomposition_executed",
            "gate_question": "Were symmetric endpoint objects decomposed into next-use classes?",
            "evidence": (
                f"objects={len(object_decomposition)}, "
                f"failure_rows={len(failure_modes)}"
            ),
            "status": "pass" if len(object_decomposition) > 0 else "blocked_no_objects",
            "decision": "use_decomposition_as_object_definition_diagnostic",
            "next_action": "separate stable controls from mechanism probes",
        },
        {
            "gate_id": "D2_stable_registry_exists",
            "gate_question": "Is there a stable one-per-seed object registry?",
            "evidence": f"stable_objects={len(stable_registry)}",
            "status": "pass" if len(stable_registry) > 0 else "blocked_no_stable_registry",
            "decision": "stable_objects_can_be_used_as_controls_or_taxonomy_nucleus",
            "next_action": "inspect mapped T1 and unmapped stable objects separately",
        },
        {
            "gate_id": "D3_t1_failure_modes_explained",
            "gate_question": "Are seed0 T1 non-promotions decomposed into named failure modes?",
            "evidence": (
                f"T1_anchor_independent_share={t1_anchor_share:.6f}, "
                f"T1_named_failure_share={t1_explained_share:.6f}"
            ),
            "status": "pass" if t1_explained_share >= 0.95 else "caveat_required",
            "decision": "do_not_promote_taxonomy_but_keep_structural_failure_surface",
            "next_action": "audit top T1 partial and multi-cluster failures",
        },
        {
            "gate_id": "D4_mechanism_probe_surface",
            "gate_question": "Are there object-level candidates for mechanism probes?",
            "evidence": f"mechanism_probe_candidates={len(mechanism_candidates)}",
            "status": "pass" if len(mechanism_candidates) > 0 else "blocked_no_probe_candidates",
            "decision": "use_candidates_for_future_mechanism_extraction_only",
            "next_action": "add topology measures before route or method work",
        },
        {
            "gate_id": "D5_route_quality_method_gate",
            "gate_question": "Can this decomposition open wall/pathway, quality/cost, or method claims?",
            "evidence": "object and mapping diagnostics only",
            "status": "closed_excluded_by_design",
            "decision": "keep_wall_quality_method_claims_closed",
            "next_action": "use only as definition and mechanism-mining input",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values = []
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
    object_decomposition: pd.DataFrame,
    failure_modes: pd.DataFrame,
    mechanism_candidates: pd.DataFrame,
) -> None:
    class_counts = (
        object_decomposition["decomposition_class"].value_counts().reset_index()
    )
    class_counts.columns = ["decomposition_class", "object_count"]
    failure_counts = failure_modes["mapping_failure_mode"].value_counts().reset_index()
    failure_counts.columns = ["mapping_failure_mode", "seed0_family_count"]
    text = [
        "# NanoClustering Symmetric Object Decomposition v1",
        "",
        f"- symmetric_object_count: `{summary['symmetric_object_count']}`",
        f"- stable_one_per_seed_object_count: `{summary['stable_one_per_seed_object_count']}`",
        f"- multi_cluster_object_count: `{summary['multi_cluster_object_count']}`",
        f"- anchor_local_fragment_count: `{summary['anchor_local_fragment_count']}`",
        f"- mechanism_probe_candidate_count: `{summary['mechanism_probe_candidate_count']}`",
        f"- T1_anchor_independent_share: `{summary['t1_anchor_independent_share']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=10,
        ),
        "",
        "## Object Classes",
        "",
        _markdown_table(class_counts, ["decomposition_class", "object_count"], max_rows=20),
        "",
        "## Seed0 Mapping Failure Modes",
        "",
        _markdown_table(
            failure_counts,
            ["mapping_failure_mode", "seed0_family_count"],
            max_rows=20,
        ),
        "",
        "## Top Mechanism Probe Candidates",
        "",
        _markdown_table(
            mechanism_candidates,
            [
                "branch",
                "symmetric_object_id",
                "probe_priority",
                "mechanism_probe_label",
                "seed_coverage_count",
                "endpoint_node_count",
                "seed0_source_family_count",
                "seed0_t1_family_count",
                "dominant_internal_relation_class",
            ],
            max_rows=25,
        ),
        "",
        "## Read",
        "",
        "- This pass decomposes why the symmetric-object audit did not justify seed-invariant taxonomy promotion.",
        "- Stable one-per-seed objects are controls or a conservative taxonomy nucleus, not wall/pathway claims.",
        "- Multi-cluster and merged seed0-family objects are mechanism-mining candidates.",
        "- Anchor-local fragments explain part of the seed0-specific primitive surface and should not be promoted.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(*, symmetric_dir: Path, output_dir: Path) -> dict[str, Any]:
    components = _read_csv(symmetric_dir / OBJECT_COMPONENTS_CSV)
    seed0_mapping = _read_csv(symmetric_dir / SEED0_MAPPING_CSV)
    object_decomposition = _object_summary(components, seed0_mapping)
    relation = _edge_relation_by_object(symmetric_dir, object_decomposition)
    object_decomposition = object_decomposition.merge(
        relation,
        on=["branch", "symmetric_object_id"],
        how="left",
        validate="one_to_one",
    )
    object_decomposition["dominant_internal_relation_class"] = object_decomposition[
        "dominant_internal_relation_class"
    ].fillna("")
    object_decomposition["dominant_internal_relation_edge_count"] = object_decomposition[
        "dominant_internal_relation_edge_count"
    ].fillna(0).astype(int)

    failure_modes = _seed0_failure_modes(seed0_mapping, object_decomposition)
    stable_registry = object_decomposition[
        object_decomposition["decomposition_class"].isin(
            ["stable_one_per_seed_object_strong", "stable_one_per_seed_object_good"]
        )
    ].copy()
    anchor_local_registry = object_decomposition[
        object_decomposition["decomposition_class"].eq("anchor_local_fragment")
    ].copy()
    multi_cluster_registry = object_decomposition[
        object_decomposition["decomposition_class"].str.contains("multi_cluster", regex=False)
    ].copy()
    mechanism_candidates = _mechanism_probe_candidates(object_decomposition)
    gate_matrix = _gate_matrix(
        object_decomposition=object_decomposition,
        failure_modes=failure_modes,
        stable_registry=stable_registry,
        mechanism_candidates=mechanism_candidates,
    )

    t1 = failure_modes[
        failure_modes["claim_tiers"].astype(str).str.contains(
            "T1_stable_high_support_nucleus",
            regex=False,
        )
    ]
    summary = {
        "symmetric_object_count": int(len(object_decomposition)),
        "stable_one_per_seed_object_count": int(len(stable_registry)),
        "multi_cluster_object_count": int(len(multi_cluster_registry)),
        "anchor_local_fragment_count": int(len(anchor_local_registry)),
        "mechanism_probe_candidate_count": int(len(mechanism_candidates)),
        "seed0_mapping_failure_mode_counts": {
            str(key): int(value)
            for key, value in failure_modes["mapping_failure_mode"]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "t1_anchor_independent_share": float(
            t1["mapping_failure_mode"].eq("anchor_independent_candidate").mean()
        )
        if not t1.empty
        else None,
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {"symmetric_dir": _rel(symmetric_dir)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(object_decomposition, output_dir / OBJECT_DECOMPOSITION_CSV)
    _write_csv(failure_modes, output_dir / SEED0_FAILURE_MODES_CSV)
    _write_csv(stable_registry, output_dir / STABLE_OBJECT_REGISTRY_CSV)
    _write_csv(anchor_local_registry, output_dir / ANCHOR_LOCAL_FRAGMENT_REGISTRY_CSV)
    _write_csv(multi_cluster_registry, output_dir / MULTI_CLUSTER_OBJECT_REGISTRY_CSV)
    _write_csv(mechanism_candidates, output_dir / MECHANISM_PROBE_CANDIDATES_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "symmetric_dir": _rel(symmetric_dir),
        "output_dir": _rel(output_dir),
        "good_object_seed_coverage_min": GOOD_OBJECT_SEED_COVERAGE_MIN,
        "strong_object_seed_coverage_min": STRONG_OBJECT_SEED_COVERAGE_MIN,
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
        object_decomposition=object_decomposition,
        failure_modes=failure_modes,
        mechanism_candidates=mechanism_candidates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetric-dir", type=Path, default=DEFAULT_SYMMETRIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = materialize(symmetric_dir=args.symmetric_dir, output_dir=args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
