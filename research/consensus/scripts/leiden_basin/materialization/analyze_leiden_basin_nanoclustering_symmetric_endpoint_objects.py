#!/usr/bin/env python3
"""Build symmetric all-seed endpoint objects for NanoClustering basin candidates.

The v2.2 primitive surface is seed0-anchored. This audit treats every
``(branch, seed, cluster_id)`` endpoint cluster as a node, builds cross-seed
overlap edges, and forms symmetric endpoint objects from high-confidence
overlap links. It is an anchor-dependence audit only: no clustering is run, no
route is executed, and no wall/pathway, quality/cost, or method claim is
opened.
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
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_LANDSCAPE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
)
DEFAULT_CLAIM_TIER_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_v2_2_claim_tier_ladder_20260531"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_endpoint_objects_20260531"
)

ENDPOINT_REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
CLAIM_TIER_PRIMITIVE_ROWS_CSV = "nanoclustering_v2_2_claim_tier_primitive_rows.csv"

ENDPOINT_NODE_REGISTRY_CSV = "nanoclustering_symmetric_endpoint_node_registry.csv"
ENDPOINT_OVERLAP_EDGES_CSV = "nanoclustering_symmetric_endpoint_overlap_edges.csv"
OBJECT_COMPONENTS_CSV = "nanoclustering_symmetric_endpoint_object_components.csv"
OBJECT_SEED_COVERAGE_CSV = "nanoclustering_symmetric_endpoint_object_seed_coverage.csv"
OBJECT_RELATION_ARCHETYPE_CSV = (
    "nanoclustering_symmetric_endpoint_object_relation_archetypes.csv"
)
SEED0_MAPPING_CSV = "nanoclustering_seed0_v2_2_mapping_to_symmetric_objects.csv"
GATE_MATRIX_CSV = "nanoclustering_symmetric_endpoint_object_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_symmetric_endpoint_object_summary.json"
CONFIG_JSON = "nanoclustering_symmetric_endpoint_object_config.json"
REPORT_MD = "nanoclustering_symmetric_endpoint_object_report.md"

CLAIM_BOUNDARY = (
    "Symmetric endpoint object audit only; membership-derived all-seed overlap "
    "cartography, no route execution, no wall/pathway promotion, no "
    "basin-quality claim, no cost claim, no directed-search claim, and no "
    "algorithm claim."
)
ROUTE_EXECUTION_STATUS = "not_executed_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_route_trace"
QUALITY_COST_STATUS = "excluded_symmetric_endpoint_object_audit"

TIER_ORDER = [
    "T1_stable_high_support_nucleus",
    "T2_thin_clean_extension",
    "T3_thin_concentration_caveat",
    "T4_concentration_caveat_no_residual",
    "T5_standard_residual_debt_caveat",
    "T6_high_residual_debt_priority",
]

OUTPUT_SHARE_MIN = 0.10
OUTPUT_JACCARD_MIN = 0.10
RECIPROCAL_LINK_MIN_SHARE = 0.50
LINK_JACCARD_MIN = 0.50
ONE_SIDED_LINK_MAJOR_SHARE = 0.90
ONE_SIDED_LINK_MINOR_SHARE = 0.25
GOOD_OBJECT_SEED_COVERAGE_MIN = 5
STRONG_OBJECT_SEED_COVERAGE_MIN = 8


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


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


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _endpoint_node_id(branch: str, seed: int, cluster_id: int) -> str:
    return f"{branch}_seed{seed}_cluster{cluster_id}"


def _symmetric_object_id(branch: str, index: int) -> str:
    return f"{branch}_symobj{index:05d}"


def _pure_seed_registry(landscape_dir: Path) -> pd.DataFrame:
    registry = _read_csv(landscape_dir / ENDPOINT_REGISTRY_CSV)
    pure = registry[registry["pure_seed_ensemble"].astype(str).eq("True")].copy()
    pure["seed"] = pure["seed"].astype(int)
    if pure.empty:
        raise ValueError("no pure seed endpoints in registry")
    return pure.sort_values(["branch", "seed"]).reset_index(drop=True)


def _load_seed_memberships(registry: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for row in registry.itertuples(index=False):
        label_cols = str(row.label_cols).split(";")
        if "candidate_micro_id" not in label_cols:
            raise ValueError(f"expected candidate_micro_id label: {row.run_id}")
        table = pq.read_table(
            Path(row.absolute_path),
            columns=[row.unit_col, row.weight_col, "candidate_micro_id"],
        )
        frame = table.to_pandas().rename(
            columns={
                row.unit_col: "unit_id",
                row.weight_col: "unit_weight",
                "candidate_micro_id": "cluster_id",
            }
        )
        frame["unit_id"] = frame["unit_id"].astype("int64")
        frame["unit_weight"] = frame["unit_weight"].astype("int64")
        frame["cluster_id"] = frame["cluster_id"].astype("int64")
        frames[(str(row.branch), int(row.seed))] = frame.sort_values(
            ["unit_id", "unit_weight"]
        ).reset_index(drop=True)
    return frames


def _endpoint_node_registry(
    memberships: dict[tuple[str, int], pd.DataFrame]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (branch, seed), frame in sorted(memberships.items()):
        totals = frame.groupby("cluster_id", as_index=False).agg(
            endpoint_unit_count=("unit_id", "size"),
            endpoint_weight_sum=("unit_weight", "sum"),
        )
        totals.insert(0, "seed", int(seed))
        totals.insert(0, "branch", str(branch))
        totals["endpoint_node_id"] = [
            _endpoint_node_id(str(branch), int(seed), int(cluster_id))
            for cluster_id in totals["cluster_id"]
        ]
        rows.append(totals)
    registry = pd.concat(rows, ignore_index=True, sort=False)
    preferred = [
        "branch",
        "seed",
        "cluster_id",
        "endpoint_node_id",
        "endpoint_unit_count",
        "endpoint_weight_sum",
    ]
    return _with_claim_columns(
        registry[preferred].sort_values(["branch", "seed", "cluster_id"])
    )


def _rank_overlap_edges(
    overlap: pd.DataFrame,
    *,
    left_cluster_col: str,
    right_cluster_col: str,
) -> pd.DataFrame:
    left_ranked = overlap.sort_values(
        [left_cluster_col, "overlap_weight_sum", "overlap_unit_count", right_cluster_col],
        ascending=[True, False, False, True],
    ).copy()
    left_ranked["left_overlap_rank"] = left_ranked.groupby(left_cluster_col).cumcount() + 1
    right_ranked = left_ranked.sort_values(
        [right_cluster_col, "overlap_weight_sum", "overlap_unit_count", left_cluster_col],
        ascending=[True, False, False, True],
    ).copy()
    right_ranked["right_overlap_rank"] = (
        right_ranked.groupby(right_cluster_col).cumcount() + 1
    )
    return right_ranked


def _relation_class(row: pd.Series) -> str:
    left_rank = int(row["left_overlap_rank"])
    right_rank = int(row["right_overlap_rank"])
    left_share = float(row["left_share_weight"])
    right_share = float(row["right_share_weight"])
    min_share = min(left_share, right_share)
    if left_rank == 1 and right_rank == 1 and min_share >= RECIPROCAL_LINK_MIN_SHARE:
        return "reciprocal_high_overlap"
    if left_rank == 1 and right_rank == 1:
        return "reciprocal_weak_overlap"
    if left_rank > 1 and right_rank == 1 and right_share >= RECIPROCAL_LINK_MIN_SHARE:
        return "split_segment_overlap"
    if left_rank == 1 and right_rank > 1 and left_share >= RECIPROCAL_LINK_MIN_SHARE:
        return "absorption_host_overlap"
    if left_rank > 1 and right_rank > 1:
        return "diffuse_multiway_overlap"
    return "asymmetric_low_share_overlap"


def _component_link(row: pd.Series) -> bool:
    left_share = float(row["left_share_weight"])
    right_share = float(row["right_share_weight"])
    min_share = min(left_share, right_share)
    max_share = max(left_share, right_share)
    reciprocal = int(row["left_overlap_rank"]) == 1 and int(row["right_overlap_rank"]) == 1
    if reciprocal and min_share >= RECIPROCAL_LINK_MIN_SHARE:
        return True
    if float(row["jaccard_weight"]) >= LINK_JACCARD_MIN:
        return True
    return (
        max_share >= ONE_SIDED_LINK_MAJOR_SHARE
        and min_share >= ONE_SIDED_LINK_MINOR_SHARE
    )


def _overlap_edges_for_pair(
    *,
    branch: str,
    left_seed: int,
    right_seed: int,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    left_frame = left[["unit_id", "unit_weight", "cluster_id"]].rename(
        columns={"cluster_id": "left_cluster_id"}
    )
    right_frame = right[["unit_id", "unit_weight", "cluster_id"]].rename(
        columns={"cluster_id": "right_cluster_id"}
    )
    left_totals = left_frame.groupby("left_cluster_id", as_index=False).agg(
        left_unit_count=("unit_id", "size"),
        left_weight_sum=("unit_weight", "sum"),
    )
    right_totals = right_frame.groupby("right_cluster_id", as_index=False).agg(
        right_unit_count=("unit_id", "size"),
        right_weight_sum=("unit_weight", "sum"),
    )
    aligned = left_frame.merge(right_frame, on=["unit_id", "unit_weight"], how="inner")
    overlap = aligned.groupby(["left_cluster_id", "right_cluster_id"], as_index=False).agg(
        overlap_unit_count=("unit_id", "size"),
        overlap_weight_sum=("unit_weight", "sum"),
    )
    overlap = overlap.merge(left_totals, on="left_cluster_id", how="left")
    overlap = overlap.merge(right_totals, on="right_cluster_id", how="left")
    overlap["left_share_weight"] = overlap["overlap_weight_sum"] / overlap["left_weight_sum"]
    overlap["right_share_weight"] = overlap["overlap_weight_sum"] / overlap["right_weight_sum"]
    overlap["left_share_units"] = overlap["overlap_unit_count"] / overlap["left_unit_count"]
    overlap["right_share_units"] = overlap["overlap_unit_count"] / overlap["right_unit_count"]
    overlap["jaccard_weight"] = overlap["overlap_weight_sum"] / (
        overlap["left_weight_sum"] + overlap["right_weight_sum"] - overlap["overlap_weight_sum"]
    )
    overlap = _rank_overlap_edges(
        overlap,
        left_cluster_col="left_cluster_id",
        right_cluster_col="right_cluster_id",
    )
    overlap["relation_class"] = overlap.apply(_relation_class, axis=1)
    overlap["component_link"] = overlap.apply(_component_link, axis=1)
    keep = (
        overlap["component_link"]
        | overlap["left_share_weight"].ge(OUTPUT_SHARE_MIN)
        | overlap["right_share_weight"].ge(OUTPUT_SHARE_MIN)
        | overlap["jaccard_weight"].ge(OUTPUT_JACCARD_MIN)
        | overlap["left_overlap_rank"].eq(1)
        | overlap["right_overlap_rank"].eq(1)
    )
    overlap = overlap[keep].copy()
    overlap.insert(0, "branch", branch)
    overlap.insert(1, "left_seed", int(left_seed))
    overlap.insert(2, "right_seed", int(right_seed))
    overlap["left_endpoint_node_id"] = [
        _endpoint_node_id(branch, int(left_seed), int(cluster_id))
        for cluster_id in overlap["left_cluster_id"]
    ]
    overlap["right_endpoint_node_id"] = [
        _endpoint_node_id(branch, int(right_seed), int(cluster_id))
        for cluster_id in overlap["right_cluster_id"]
    ]
    preferred = [
        "branch",
        "left_seed",
        "right_seed",
        "left_cluster_id",
        "right_cluster_id",
        "left_endpoint_node_id",
        "right_endpoint_node_id",
        "overlap_unit_count",
        "overlap_weight_sum",
        "left_unit_count",
        "right_unit_count",
        "left_weight_sum",
        "right_weight_sum",
        "left_share_weight",
        "right_share_weight",
        "left_share_units",
        "right_share_units",
        "jaccard_weight",
        "left_overlap_rank",
        "right_overlap_rank",
        "relation_class",
        "component_link",
    ]
    return overlap[preferred].sort_values(
        ["branch", "left_seed", "right_seed", "left_cluster_id", "right_cluster_id"]
    )


def _endpoint_overlap_edges(
    registry: pd.DataFrame,
    memberships: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for branch, group in registry.groupby("branch", sort=True):
        branch = str(branch)
        seeds = sorted(int(seed) for seed in group["seed"].unique())
        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                rows.append(
                    _overlap_edges_for_pair(
                        branch=branch,
                        left_seed=left_seed,
                        right_seed=right_seed,
                        left=memberships[(branch, left_seed)],
                        right=memberships[(branch, right_seed)],
                    )
                )
    if not rows:
        return pd.DataFrame()
    return _with_claim_columns(pd.concat(rows, ignore_index=True, sort=False))


def _object_class(seed_coverage_count: int, max_endpoint_nodes_per_seed: int) -> str:
    if seed_coverage_count >= STRONG_OBJECT_SEED_COVERAGE_MIN:
        if max_endpoint_nodes_per_seed <= 1:
            return "strong_seed_spanning_one_per_seed_object"
        return "strong_seed_spanning_multi_cluster_object"
    if seed_coverage_count >= GOOD_OBJECT_SEED_COVERAGE_MIN:
        if max_endpoint_nodes_per_seed <= 1:
            return "good_seed_spanning_one_per_seed_object"
        return "good_seed_spanning_multi_cluster_object"
    if seed_coverage_count >= 2:
        return "partial_seed_object"
    return "singleton_anchor_local_object"


def _component_frames(
    endpoint_nodes: pd.DataFrame,
    overlap_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_ids = endpoint_nodes["endpoint_node_id"].astype(str).tolist()
    uf = UnionFind(node_ids)
    for row in overlap_edges[overlap_edges["component_link"].astype(bool)].itertuples(
        index=False
    ):
        uf.union(str(row.left_endpoint_node_id), str(row.right_endpoint_node_id))

    root_to_object: dict[tuple[str, str], str] = {}
    object_indices: dict[str, int] = {}
    assignments: list[dict[str, Any]] = []
    endpoint_lookup = endpoint_nodes.set_index("endpoint_node_id")
    for node_id in node_ids:
        branch = str(endpoint_lookup.loc[node_id, "branch"])
        root = uf.find(node_id)
        key = (branch, root)
        if key not in root_to_object:
            object_indices[branch] = object_indices.get(branch, 0) + 1
            root_to_object[key] = _symmetric_object_id(branch, object_indices[branch])
        assignments.append(
            {
                "endpoint_node_id": node_id,
                "component_root": root,
                "symmetric_object_id": root_to_object[key],
            }
        )
    assigned = endpoint_nodes.merge(pd.DataFrame(assignments), on="endpoint_node_id")

    component_rows: list[dict[str, Any]] = []
    for (branch, object_id), group in assigned.groupby(
        ["branch", "symmetric_object_id"],
        sort=True,
    ):
        seed_counts = group.groupby("seed").size()
        seed_list = ";".join(str(int(seed)) for seed in sorted(group["seed"].unique()))
        component_rows.append(
            {
                "branch": str(branch),
                "symmetric_object_id": str(object_id),
                "endpoint_node_count": int(len(group)),
                "seed_coverage_count": int(group["seed"].nunique()),
                "seed_coverage_share": float(group["seed"].nunique() / 10.0),
                "seed_list": seed_list,
                "max_endpoint_nodes_per_seed": int(seed_counts.max()),
                "multi_endpoint_seed_count": int((seed_counts > 1).sum()),
                "endpoint_weight_sum_total": int(group["endpoint_weight_sum"].sum()),
                "endpoint_weight_sum_median": float(group["endpoint_weight_sum"].median()),
                "endpoint_weight_sum_max": int(group["endpoint_weight_sum"].max()),
                "endpoint_unit_count_median": float(group["endpoint_unit_count"].median()),
                "endpoint_unit_count_max": int(group["endpoint_unit_count"].max()),
                "object_class": _object_class(
                    int(group["seed"].nunique()),
                    int(seed_counts.max()),
                ),
            }
        )
    components = pd.DataFrame(component_rows).sort_values(
        ["branch", "seed_coverage_count", "endpoint_node_count", "symmetric_object_id"],
        ascending=[True, False, False, True],
    )
    return _with_claim_columns(assigned), _with_claim_columns(components)


def _seed_coverage_summary(components: pd.DataFrame) -> pd.DataFrame:
    rows = (
        components.groupby(["branch", "seed_coverage_count", "object_class"], as_index=False)
        .agg(
            symmetric_object_count=("symmetric_object_id", "nunique"),
            endpoint_node_count=("endpoint_node_count", "sum"),
            median_endpoint_nodes_per_object=("endpoint_node_count", "median"),
        )
        .sort_values(["branch", "seed_coverage_count", "object_class"])
    )
    return _with_claim_columns(rows)


def _relation_archetype_summary(overlap_edges: pd.DataFrame) -> pd.DataFrame:
    rows = (
        overlap_edges.groupby(["branch", "relation_class", "component_link"], as_index=False)
        .agg(
            edge_count=("relation_class", "size"),
            overlap_weight_sum=("overlap_weight_sum", "sum"),
            median_left_share_weight=("left_share_weight", "median"),
            median_right_share_weight=("right_share_weight", "median"),
            median_jaccard_weight=("jaccard_weight", "median"),
        )
        .sort_values(["branch", "component_link", "edge_count"], ascending=[True, False, False])
    )
    return _with_claim_columns(rows)


def _seed0_family_tiers(claim_tier_rows: pd.DataFrame) -> pd.DataFrame:
    rows = claim_tier_rows.copy()
    rows["source_family_id"] = rows["source_family_id"].astype(str)
    parsed = rows["source_family_id"].str.extract(
        r"^(?P<parsed_branch>[^_]+)_seed0_ref(?P<seed0_ref_cluster_id>\d+)$"
    )
    rows = pd.concat([rows, parsed], axis=1)
    rows = rows.dropna(subset=["parsed_branch", "seed0_ref_cluster_id"]).copy()
    if "branch" not in rows.columns:
        rows["branch"] = rows["parsed_branch"]
    elif not rows["branch"].astype(str).eq(rows["parsed_branch"].astype(str)).all():
        raise ValueError("source_family_id branch disagrees with claim-tier branch column")
    rows = rows.drop(columns=["parsed_branch"])
    rows["seed0_ref_cluster_id"] = rows["seed0_ref_cluster_id"].astype(int)
    rows["claim_tier_rank"] = rows["claim_tier"].map(
        {tier: index + 1 for index, tier in enumerate(TIER_ORDER)}
    )
    families = (
        rows.groupby(["branch", "source_family_id", "seed0_ref_cluster_id"], as_index=False)
        .agg(
            best_claim_tier_rank=("claim_tier_rank", "min"),
            claim_tiers=("claim_tier", lambda s: ";".join(sorted(set(map(str, s))))),
            primitive_count=("primitive_id", "nunique"),
            event_count=("event_count", "sum"),
        )
    )
    families["best_claim_tier"] = families["best_claim_tier_rank"].map(
        {index + 1: tier for index, tier in enumerate(TIER_ORDER)}
    )
    return families


def _seed0_mapping_to_objects(
    *,
    claim_tier_rows: pd.DataFrame,
    endpoint_assignments: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    families = _seed0_family_tiers(claim_tier_rows)
    seed0_nodes = endpoint_assignments[endpoint_assignments["seed"].eq(0)][
        [
            "branch",
            "cluster_id",
            "endpoint_node_id",
            "endpoint_unit_count",
            "endpoint_weight_sum",
            "symmetric_object_id",
        ]
    ].rename(columns={"cluster_id": "seed0_ref_cluster_id"})
    component_cols = [
        "branch",
        "symmetric_object_id",
        "endpoint_node_count",
        "seed_coverage_count",
        "seed_coverage_share",
        "max_endpoint_nodes_per_seed",
        "multi_endpoint_seed_count",
        "object_class",
    ]
    mapped = families.merge(
        seed0_nodes,
        on=["branch", "seed0_ref_cluster_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        components[component_cols],
        on=["branch", "symmetric_object_id"],
        how="left",
        validate="many_to_one",
    )
    mapped["mapped_to_symmetric_object"] = mapped["symmetric_object_id"].notna()
    mapped["good_seed_coverage_object"] = mapped["seed_coverage_count"].ge(
        GOOD_OBJECT_SEED_COVERAGE_MIN
    )
    mapped["strong_seed_coverage_object"] = mapped["seed_coverage_count"].ge(
        STRONG_OBJECT_SEED_COVERAGE_MIN
    )
    mapped["single_cluster_per_seed_object"] = mapped["max_endpoint_nodes_per_seed"].le(1)
    mapped["anchor_independent_candidate"] = (
        mapped["good_seed_coverage_object"].fillna(False)
        & mapped["single_cluster_per_seed_object"].fillna(False)
    )
    return _with_claim_columns(
        mapped.sort_values(["branch", "best_claim_tier_rank", "source_family_id"])
    )


def _tier_summary(seed0_mapping: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tier in TIER_ORDER:
        group = seed0_mapping[
            seed0_mapping["claim_tiers"].astype(str).str.contains(tier, regex=False)
        ]
        if group.empty:
            continue
        rows.append(
            {
                "claim_tier": tier,
                "seed0_source_family_count": int(group["source_family_id"].nunique()),
                "mapped_source_family_count": int(
                    group.loc[group["mapped_to_symmetric_object"], "source_family_id"].nunique()
                ),
                "good_seed_coverage_count": int(
                    group.loc[group["good_seed_coverage_object"], "source_family_id"].nunique()
                ),
                "strong_seed_coverage_count": int(
                    group.loc[group["strong_seed_coverage_object"], "source_family_id"].nunique()
                ),
                "anchor_independent_candidate_count": int(
                    group.loc[
                        group["anchor_independent_candidate"], "source_family_id"
                    ].nunique()
                ),
                "median_seed_coverage_count": float(group["seed_coverage_count"].median()),
                "median_endpoint_nodes_per_object": float(
                    group["endpoint_node_count"].median()
                ),
            }
        )
    tier_rows = pd.DataFrame(rows)
    if tier_rows.empty:
        return tier_rows
    tier_rows["good_seed_coverage_share"] = (
        tier_rows["good_seed_coverage_count"] / tier_rows["seed0_source_family_count"]
    )
    tier_rows["anchor_independent_candidate_share"] = (
        tier_rows["anchor_independent_candidate_count"]
        / tier_rows["seed0_source_family_count"]
    )
    return _with_claim_columns(tier_rows)


def _gate_matrix(
    *,
    endpoint_nodes: pd.DataFrame,
    components: pd.DataFrame,
    seed0_mapping: pd.DataFrame,
    tier_summary: pd.DataFrame,
) -> pd.DataFrame:
    t1 = tier_summary[tier_summary["claim_tier"].eq("T1_stable_high_support_nucleus")]
    t1_anchor_share = (
        float(t1["anchor_independent_candidate_share"].iloc[0]) if not t1.empty else 0.0
    )
    good_objects = components[
        components["seed_coverage_count"].ge(GOOD_OBJECT_SEED_COVERAGE_MIN)
    ]
    strong_objects = components[
        components["seed_coverage_count"].ge(STRONG_OBJECT_SEED_COVERAGE_MIN)
    ]
    singleton_share = float(
        components["object_class"].eq("singleton_anchor_local_object").mean()
    )
    rows = [
        {
            "gate_id": "O1_symmetric_object_audit_executed",
            "gate_question": "Were all pure Java/Rust endpoint clusters included?",
            "evidence": (
                f"endpoint_nodes={len(endpoint_nodes)}, "
                f"branches={endpoint_nodes['branch'].nunique()}, "
                f"seeds={endpoint_nodes['seed'].nunique()}"
            ),
            "status": "pass"
            if endpoint_nodes["branch"].nunique() == 2
            and endpoint_nodes["seed"].nunique() == 10
            else "blocked_incomplete_endpoint_grid",
            "decision": "use_symmetric_objects_as_anchor_dependence_audit",
            "next_action": "inspect object seed coverage and seed0 tier mapping",
        },
        {
            "gate_id": "O2_multi_seed_object_surface",
            "gate_question": "Does the overlap graph produce multi-seed endpoint objects?",
            "evidence": (
                f"good_seed_objects={len(good_objects)}, "
                f"strong_seed_objects={len(strong_objects)}, "
                f"singleton_object_share={singleton_share:.6f}"
            ),
            "status": "pass" if len(good_objects) > 0 else "blocked_no_multi_seed_objects",
            "decision": "endpoint_objects_exist_but_require_tier_mapping",
            "next_action": "evaluate seed0 T1/T2 recovery into symmetric objects",
        },
        {
            "gate_id": "O3_seed0_t1_anchor_independence",
            "gate_question": "Do seed0 T1 source families map to anchor-independent objects?",
            "evidence": f"T1_anchor_independent_candidate_share={t1_anchor_share:.6f}",
            "status": "pass" if t1_anchor_share >= 0.5 else "caveat_required",
            "decision": "do_not_claim_seed_invariant_taxonomy_if_caveat",
            "next_action": "separate stable symmetric objects from anchor-local fragments",
        },
        {
            "gate_id": "O4_route_quality_method_gate",
            "gate_question": "Can symmetric objects open wall/pathway, quality/cost, or method claims?",
            "evidence": "membership overlap components only",
            "status": "closed_excluded_by_design",
            "decision": "keep_wall_quality_method_claims_closed",
            "next_action": "use only as object-definition evidence",
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
    seed_coverage: pd.DataFrame,
    relation_archetypes: pd.DataFrame,
    tier_summary: pd.DataFrame,
) -> None:
    text = [
        "# NanoClustering Symmetric Endpoint Object Audit",
        "",
        f"- endpoint_node_count: `{summary['endpoint_node_count']}`",
        f"- retained_overlap_edge_count: `{summary['retained_overlap_edge_count']}`",
        f"- component_link_edge_count: `{summary['component_link_edge_count']}`",
        f"- symmetric_object_count: `{summary['symmetric_object_count']}`",
        f"- good_seed_coverage_object_count: `{summary['good_seed_coverage_object_count']}`",
        f"- strong_seed_coverage_object_count: `{summary['strong_seed_coverage_object_count']}`",
        f"- T1_anchor_independent_candidate_share: `{summary['t1_anchor_independent_candidate_share']}`",
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
        "## Seed Coverage",
        "",
        _markdown_table(
            seed_coverage,
            [
                "branch",
                "seed_coverage_count",
                "object_class",
                "symmetric_object_count",
                "endpoint_node_count",
            ],
            max_rows=25,
        ),
        "",
        "## Relation Archetypes",
        "",
        _markdown_table(
            relation_archetypes,
            [
                "branch",
                "relation_class",
                "component_link",
                "edge_count",
                "median_left_share_weight",
                "median_right_share_weight",
                "median_jaccard_weight",
            ],
            max_rows=25,
        ),
        "",
        "## Seed0 Claim Tier Mapping",
        "",
        _markdown_table(
            tier_summary,
            [
                "claim_tier",
                "seed0_source_family_count",
                "good_seed_coverage_count",
                "strong_seed_coverage_count",
                "anchor_independent_candidate_count",
                "anchor_independent_candidate_share",
                "median_seed_coverage_count",
            ],
            max_rows=10,
        ),
        "",
        "## Read",
        "",
        "- This audit replaces the seed0 coordinate with all-seed endpoint-cluster overlap components.",
        "- A good symmetric object means a cluster identity has at least five-seed support under the current component-link rule.",
        "- O3 is intentionally strict: it asks whether seed0 T1 families map to good objects with at most one endpoint node per seed.",
        "- Even if O3 passes, this remains object-definition evidence only; wall/pathway, quality/cost, and method claims stay closed.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(
    *,
    landscape_dir: Path,
    claim_tier_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _pure_seed_registry(landscape_dir)
    claim_tier_rows = _read_csv(claim_tier_dir / CLAIM_TIER_PRIMITIVE_ROWS_CSV)
    memberships = _load_seed_memberships(registry)

    endpoint_nodes = _endpoint_node_registry(memberships)
    overlap_edges = _endpoint_overlap_edges(registry, memberships)
    endpoint_assignments, components = _component_frames(endpoint_nodes, overlap_edges)
    seed_coverage = _seed_coverage_summary(components)
    relation_archetypes = _relation_archetype_summary(overlap_edges)
    seed0_mapping = _seed0_mapping_to_objects(
        claim_tier_rows=claim_tier_rows,
        endpoint_assignments=endpoint_assignments,
        components=components,
    )
    tier_summary = _tier_summary(seed0_mapping)
    gate_matrix = _gate_matrix(
        endpoint_nodes=endpoint_nodes,
        components=components,
        seed0_mapping=seed0_mapping,
        tier_summary=tier_summary,
    )

    good_objects = components[
        components["seed_coverage_count"].ge(GOOD_OBJECT_SEED_COVERAGE_MIN)
    ]
    strong_objects = components[
        components["seed_coverage_count"].ge(STRONG_OBJECT_SEED_COVERAGE_MIN)
    ]
    t1 = tier_summary[tier_summary["claim_tier"].eq("T1_stable_high_support_nucleus")]
    t1_anchor_share = (
        float(t1["anchor_independent_candidate_share"].iloc[0]) if not t1.empty else None
    )
    summary = {
        "endpoint_node_count": int(len(endpoint_nodes)),
        "retained_overlap_edge_count": int(len(overlap_edges)),
        "component_link_edge_count": int(overlap_edges["component_link"].astype(bool).sum()),
        "symmetric_object_count": int(components["symmetric_object_id"].nunique()),
        "good_seed_coverage_object_count": int(len(good_objects)),
        "strong_seed_coverage_object_count": int(len(strong_objects)),
        "singleton_anchor_local_object_count": int(
            components["object_class"].eq("singleton_anchor_local_object").sum()
        ),
        "seed0_v2_2_source_family_count": int(seed0_mapping["source_family_id"].nunique()),
        "t1_anchor_independent_candidate_share": t1_anchor_share,
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "landscape_dir": _rel(landscape_dir),
            "claim_tier_dir": _rel(claim_tier_dir),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(endpoint_nodes, output_dir / ENDPOINT_NODE_REGISTRY_CSV)
    _write_csv(overlap_edges, output_dir / ENDPOINT_OVERLAP_EDGES_CSV)
    _write_csv(endpoint_assignments, output_dir / OBJECT_COMPONENTS_CSV)
    _write_csv(seed_coverage, output_dir / OBJECT_SEED_COVERAGE_CSV)
    _write_csv(relation_archetypes, output_dir / OBJECT_RELATION_ARCHETYPE_CSV)
    _write_csv(seed0_mapping, output_dir / SEED0_MAPPING_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "landscape_dir": _rel(landscape_dir),
        "claim_tier_dir": _rel(claim_tier_dir),
        "output_dir": _rel(output_dir),
        "thresholds": {
            "output_share_min": OUTPUT_SHARE_MIN,
            "output_jaccard_min": OUTPUT_JACCARD_MIN,
            "reciprocal_link_min_share": RECIPROCAL_LINK_MIN_SHARE,
            "link_jaccard_min": LINK_JACCARD_MIN,
            "one_sided_link_major_share": ONE_SIDED_LINK_MAJOR_SHARE,
            "one_sided_link_minor_share": ONE_SIDED_LINK_MINOR_SHARE,
            "good_object_seed_coverage_min": GOOD_OBJECT_SEED_COVERAGE_MIN,
            "strong_object_seed_coverage_min": STRONG_OBJECT_SEED_COVERAGE_MIN,
        },
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
        seed_coverage=seed_coverage,
        relation_archetypes=relation_archetypes,
        tier_summary=tier_summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--claim-tier-dir", type=Path, default=DEFAULT_CLAIM_TIER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = materialize(
        landscape_dir=args.landscape_dir,
        claim_tier_dir=args.claim_tier_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
