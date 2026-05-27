"""Run branch-adaptive quality-first split diagnostics.

This runner consumes prepared hierarchy postprocess validation artifacts:
source-seed memberships plus graph sidecars.  Local split candidates are
generated with induced parent subgraph probes; acceptance diagnostics are
computed with original-graph CPM split accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciscape_leiden


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from evaluate_hierarchy_postprocess import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    _cluster_weights,
    _gini,
    _load_membership,
    _load_node_weights,
    _markdown_table,
    _normalized_entropy,
    _repo_path,
    _write_table,
)
from run_hierarchy_postprocess_next_level import _rel, _safe_slug  # noqa: E402
from sciscape.clustering.branch_adaptive import (  # noqa: E402
    DEFAULT_LOCAL_SEEDS,
    DEFAULT_SOURCE_SEEDS,
    DEFAULT_STAGE1_ALPHAS,
    DEFAULT_STAGE2_ALPHAS,
    DEFAULT_TAU_SPLIT_RATIOS,
    best_match_child_jaccard,
    child_size_diagnostics,
    mean_pairwise_ami,
    rank_branch_split_candidates,
    source_max_ratio_delta_if_applied,
    split_accounting_from_delta_cut,
)
from sciscape.clustering.hierarchy_postprocess import (  # noqa: E402
    current_oversize_candidate_clusters,
)
from sciscape.clustering.leiden_rust import RustLeidenGraph, build_leiden_graph  # noqa: E402


DEFAULT_BRANCH_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "branch_adaptive_quality_first_pilot"
DEFAULT_FIELDS = (30, 26)
APPLICATION_KERNEL = "split_merge_repair_candidates"
APPLY_CACHE_SCHEMA_VERSION = 2
CHILD_WEIGHT_ENTROPY_SOURCE = "proxy_largest_second_equal_remainder_from_probe_summary"

REQUIRED_CANDIDATE_COLUMNS = [
    "field",
    "sample",
    "source_seed",
    "parent_cluster",
    "alpha",
    "local_seed",
    "k_children",
    "parent_doc_weight",
    "target_max_doc_weight",
    "delta_q_split_original",
    "w_between",
    "e_between",
    "gamma_star_split",
    "raw_split_gap",
    "normalized_split_gain",
    "child_weight_entropy",
    "child_weight_entropy_is_exact",
    "child_weight_entropy_source",
    "n_children_below_min",
    "largest_child_fraction",
    "source_max_ratio_delta_if_applied",
    "status",
]


@dataclass(frozen=True)
class SourceRunConfig:
    field: int
    sample: str
    source_seed: int
    prepare_summary_path: Path
    graph_dir: Path
    membership_path: Path
    node_weights_path: Path
    resolution: float
    target_min_doc_weight: float
    target_max_doc_weight: float
    n_nodes: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cache_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": _rel(path), "exists": False}
    return {
        "path": _rel(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _hash_json(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _field_from_sample(sample: str) -> int | None:
    match = re.search(r"field_?(\d+)", str(sample))
    return int(match.group(1)) if match else None


def _source_seed_roots(validation_dir: Path) -> list[Path]:
    return [
        validation_dir / "field_expansion_runs" / "source_seed_sweep_runs",
        validation_dir / "source_seed_sweep_runs",
    ]


def _discover_source_run(
    *,
    field: int,
    source_seed: int,
    validation_dir: Path,
) -> SourceRunConfig:
    candidates: list[Path] = []
    for root in _source_seed_roots(validation_dir):
        if not root.exists():
            continue
        for summary_path in sorted(root.glob(f"field*{field}*/seed_{source_seed}/prepare_summary.json")):
            if _field_from_sample(summary_path.parts[-3]) == int(field):
                candidates.append(summary_path)
    if not candidates:
        raise FileNotFoundError(
            f"No source-seed prepare_summary.json for field={field}, seed={source_seed}"
        )

    summary_path = candidates[0]
    summary = _read_json(summary_path)
    paths = summary.get("paths", {})
    graph_dir = _repo_path(paths.get("graph_dir")) or summary_path.parent / "graph"
    membership_path = _repo_path(paths.get("membership")) or summary_path.parent / "membership.parquet"
    if graph_dir is None or membership_path is None:
        raise FileNotFoundError(f"Invalid source run paths in {summary_path}")
    node_weights_path = graph_dir / "node_weights.f64.bin"
    if not node_weights_path.exists():
        raise FileNotFoundError(f"Missing node weights sidecar: {node_weights_path}")
    n_nodes = int(node_weights_path.stat().st_size // np.dtype(np.float64).itemsize)
    sample = str(summary.get("sample") or summary_path.parts[-3])
    return SourceRunConfig(
        field=int(field),
        sample=sample,
        source_seed=int(source_seed),
        prepare_summary_path=summary_path,
        graph_dir=graph_dir,
        membership_path=membership_path,
        node_weights_path=node_weights_path,
        resolution=float(summary.get("resolution") or 0.01),
        target_min_doc_weight=float(
            summary.get("target_min_doc_weight") or summary.get("min_size") or 50.0
        ),
        target_max_doc_weight=float(summary["target_max_doc_weight"]),
        n_nodes=n_nodes,
    )


def _load_graph(config: SourceRunConfig) -> RustLeidenGraph:
    native = sciscape_leiden.load_graph_raw_files(
        int(config.n_nodes),
        str(config.graph_dir / "src.u32.bin"),
        str(config.graph_dir / "dst.u32.bin"),
        str(config.graph_dir / "weight.f64.bin"),
        str(config.node_weights_path),
    )
    return RustLeidenGraph(
        graph=native,
        n_nodes=int(native.n_nodes),
        n_edges=int(native.n_edges),
        node_weights=None,
    )


def _load_graph_arrays(graph_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.memmap(graph_dir / "src.u32.bin", dtype=np.uint32, mode="r")
    dst = np.memmap(graph_dir / "dst.u32.bin", dtype=np.uint32, mode="r")
    weight = np.memmap(graph_dir / "weight.f64.bin", dtype=np.float64, mode="r")
    return src, dst, weight


def _candidate_clusters(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_doc_weight: float,
    max_parents: int,
) -> np.ndarray:
    clusters = current_oversize_candidate_clusters(
        membership,
        node_weights,
        max_weight=float(target_max_doc_weight),
        max_candidates=max(0, int(max_parents)),
    )
    return np.asarray(clusters, dtype=np.uint64)


def _approx_child_weights(
    *,
    parent_doc_weight: float,
    k_children: int,
    largest_child_weight: float,
    second_child_weight: float,
) -> np.ndarray:
    k = max(0, int(k_children))
    if k == 0:
        return np.asarray([], dtype=np.float64)
    weights: list[float] = []
    largest = max(0.0, float(largest_child_weight))
    second = max(0.0, float(second_child_weight))
    if largest > 0.0:
        weights.append(largest)
    if k > 1 and second > 0.0:
        weights.append(second)
    remaining_slots = max(0, k - len(weights))
    remaining_weight = max(0.0, float(parent_doc_weight) - float(sum(weights)))
    if remaining_slots:
        weights.extend([remaining_weight / remaining_slots] * remaining_slots)
    return np.asarray(weights[:k], dtype=np.float64)


def _probe_rows(
    *,
    config: SourceRunConfig,
    local_seed: int,
    probes: Any,
    start_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(probes.cluster.shape[0])
    for idx in range(n):
        delta_q = float(probes.split_delta_q_base[idx])
        e_between = float(probes.cut_weight[idx])
        accounting = split_accounting_from_delta_cut(
            delta_q_split_original=delta_q,
            e_between=e_between,
            gamma=float(config.resolution),
        )
        k_children = int(probes.n_parts[idx])
        parent_doc_weight = float(probes.doc_weight[idx])
        child_weights = _approx_child_weights(
            parent_doc_weight=parent_doc_weight,
            k_children=k_children,
            largest_child_weight=float(probes.largest_part_weight[idx]),
            second_child_weight=float(probes.second_part_weight[idx]),
        )
        child_diag = child_size_diagnostics(
            child_weights,
            min_doc_weight=float(config.target_min_doc_weight),
        )
        child_diag["n_children_below_min"] = max(
            0,
            k_children - int(probes.core_part_count[idx]),
        )
        largest_child_fraction = float(probes.largest_part_fraction[idx])
        status = "ok"
        if k_children < 2:
            status = "no_split"
        elif not math.isfinite(accounting.w_between) or accounting.w_between <= 0.0:
            status = "zero_w_between"
        rows.append(
            {
                "candidate_index": start_index + len(rows),
                "field": int(config.field),
                "sample": config.sample,
                "source_seed": int(config.source_seed),
                "parent_cluster": int(probes.cluster[idx]),
                "alpha": float(probes.gamma_multiplier[idx]),
                "local_seed": int(local_seed),
                "resolution": float(config.resolution),
                "probe_resolution": float(probes.probe_resolution[idx]),
                "k_children": k_children,
                "parent_doc_weight": parent_doc_weight,
                "target_min_doc_weight": float(config.target_min_doc_weight),
                "target_max_doc_weight": float(config.target_max_doc_weight),
                "delta_q_split_original": accounting.delta_q_split_original,
                "w_between": accounting.w_between,
                "e_between": accounting.e_between,
                "gamma_star_split": accounting.gamma_star_split,
                "raw_split_gap": accounting.raw_split_gap,
                "normalized_split_gain": accounting.normalized_split_gain,
                "child_weight_entropy": float(child_diag["child_weight_entropy"]),
                "child_weight_entropy_is_exact": False,
                "child_weight_entropy_source": CHILD_WEIGHT_ENTROPY_SOURCE,
                "n_children_below_min": int(child_diag["n_children_below_min"]),
                "largest_child_fraction": largest_child_fraction,
                "source_max_ratio_delta_if_applied": source_max_ratio_delta_if_applied(
                    parent_doc_weight=parent_doc_weight,
                    largest_child_fraction=largest_child_fraction,
                    target_max_doc_weight=float(config.target_max_doc_weight),
                ),
                "block_count": int(probes.block_count[idx]),
                "induced_directed_edges": int(probes.induced_directed_edges[idx]),
                "non_singleton_children": int(probes.non_singleton_parts[idx]),
                "singleton_children": int(probes.singleton_parts[idx]),
                "singleton_weight": float(probes.singleton_weight[idx]),
                "core_child_count": int(probes.core_part_count[idx]),
                "core_child_weight": float(probes.core_part_weight[idx]),
                "largest_child_weight": float(probes.largest_part_weight[idx]),
                "second_child_weight": float(probes.second_part_weight[idx]),
                "split_delta_q_probe": float(probes.split_delta_q_probe[idx]),
                "hysteresis_only": bool(probes.hysteresis_only[idx]),
                "status": status,
            }
        )
    return rows


def _run_candidate_diagnostics(
    *,
    configs: list[SourceRunConfig],
    alphas: tuple[float, ...],
    local_seeds: tuple[int, ...],
    max_parents: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    candidate_index = 0
    for config in configs:
        t0 = time.perf_counter()
        membership = _load_membership(config.membership_path).astype(np.uint64, copy=False)
        node_weights = _load_node_weights(config.node_weights_path, int(membership.shape[0]))
        parents = _candidate_clusters(
            membership,
            node_weights,
            target_max_doc_weight=float(config.target_max_doc_weight),
            max_parents=max_parents,
        )
        graph = _load_graph(config)
        alpha_array = np.asarray(alphas, dtype=np.float64)
        before_rows = len(rows)
        for local_seed in local_seeds:
            probes = graph.multi_core_split_probes(
                membership=membership,
                candidate_clusters=parents,
                resolution=float(config.resolution),
                gamma_multipliers=alpha_array,
                min_core_weight=float(config.target_min_doc_weight),
                randomness=0.01,
                seed=int(local_seed),
            )
            new_rows = _probe_rows(
                config=config,
                local_seed=int(local_seed),
                probes=probes,
                start_index=candidate_index,
            )
            rows.extend(new_rows)
            candidate_index += len(new_rows)
        run_summaries.append(
            {
                "field": int(config.field),
                "sample": config.sample,
                "source_seed": int(config.source_seed),
                "n_nodes": int(config.n_nodes),
                "n_parent_candidates": int(parents.size),
                "n_probe_rows": int(len(rows) - before_rows),
                "elapsed_sec": float(time.perf_counter() - t0),
                "membership_path": _rel(config.membership_path),
                "graph_dir": _rel(config.graph_dir),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=REQUIRED_CANDIDATE_COLUMNS)
    return df, run_summaries


def _parent_summary(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["field", "sample", "source_seed", "parent_cluster"]
    for key, group in candidate_df.groupby(keys, sort=True):
        ok = group[group["status"] == "ok"].copy()
        best = ok.sort_values(
            ["normalized_split_gain", "delta_q_split_original"],
            ascending=[False, False],
        ).head(1)
        best_row = best.iloc[0].to_dict() if not best.empty else {}
        rows.append(
            {
                "field": int(key[0]),
                "sample": key[1],
                "source_seed": int(key[2]),
                "parent_cluster": int(key[3]),
                "parent_doc_weight": float(group["parent_doc_weight"].iloc[0]),
                "target_max_doc_weight": float(group["target_max_doc_weight"].iloc[0]),
                "n_candidates": int(len(group)),
                "n_ok": int(len(ok)),
                "n_quality_positive": int((ok["delta_q_split_original"] >= 0.0).sum())
                if not ok.empty
                else 0,
                "best_alpha": best_row.get("alpha"),
                "best_local_seed": best_row.get("local_seed"),
                "best_k_children": best_row.get("k_children"),
                "best_delta_q_split_original": best_row.get("delta_q_split_original"),
                "best_normalized_split_gain": best_row.get("normalized_split_gain"),
                "best_gamma_star_split": best_row.get("gamma_star_split"),
                "best_largest_child_fraction": best_row.get("largest_child_fraction"),
                "best_source_max_ratio_delta_if_applied": best_row.get(
                    "source_max_ratio_delta_if_applied"
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_for_tau(candidate_df: pd.DataFrame, tau_ratio: float) -> pd.DataFrame:
    ranked: list[dict[str, Any]] = []
    if candidate_df.empty:
        return pd.DataFrame()
    for _resolution, group in candidate_df.groupby("resolution", sort=False):
        ranked.extend(
            rank_branch_split_candidates(
                group.to_dict("records"),
                gamma=float(_resolution),
                tau_split_ratio=float(tau_ratio),
                epsilon_q=0.0,
            )
        )
    return pd.DataFrame(ranked)


def _tau_sensitivity(candidate_df: pd.DataFrame, tau_ratios: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_ranked: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for tau_ratio in tau_ratios:
        ranked = _rank_for_tau(candidate_df, tau_ratio)
        if ranked.empty:
            rows.append(
                {
                    "tau_split_ratio": float(tau_ratio),
                    "n_candidates": 0,
                    "n_policy_accepted": 0,
                    "n_selected": 0,
                    "selected_delta_q_sum": 0.0,
                    "selected_source_max_ratio_delta_sum": 0.0,
                }
            )
            continue
        all_ranked.append(ranked)
        selected = ranked[ranked["selected_for_apply"]].copy()
        rows.append(
            {
                "tau_split_ratio": float(tau_ratio),
                "tau_split_abs_mean": float(ranked["tau_split_abs"].mean()),
                "n_candidates": int(len(ranked)),
                "n_policy_accepted": int(ranked["accepted_by_policy"].sum()),
                "n_selected": int(selected.shape[0]),
                "n_selected_fields": int(selected["field"].nunique()) if not selected.empty else 0,
                "n_selected_parents": int(
                    selected[["sample", "source_seed", "parent_cluster"]].drop_duplicates().shape[0]
                )
                if not selected.empty
                else 0,
                "selected_delta_q_sum": float(selected["delta_q_split_original"].sum())
                if not selected.empty
                else 0.0,
                "selected_source_max_ratio_delta_sum": float(
                    selected["source_max_ratio_delta_if_applied"].sum()
                )
                if not selected.empty
                else 0.0,
                "selected_normalized_gain_mean": float(selected["normalized_split_gain"].mean())
                if not selected.empty
                else 0.0,
                "selected_largest_child_fraction_mean": float(
                    selected["largest_child_fraction"].mean()
                )
                if not selected.empty
                else 0.0,
                "selected_children_below_min_sum": int(selected["n_children_below_min"].sum())
                if not selected.empty
                else 0,
            }
        )
    selection_df = pd.concat(all_ranked, ignore_index=True) if all_ranked else pd.DataFrame()
    return pd.DataFrame(rows), selection_df


def _proxy_stability(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["field", "sample", "source_seed", "parent_cluster"]
    for key, group in candidate_df.groupby(keys, sort=True):
        rows.append(
            {
                "field": int(key[0]),
                "sample": key[1],
                "source_seed": int(key[2]),
                "parent_cluster": int(key[3]),
                "n_candidate_variants": int(len(group)),
                "mean_pairwise_ami": math.nan,
                "gamma_axis_ami": math.nan,
                "optional_best_match_child_jaccard": math.nan,
                "k_children_mean": float(group["k_children"].mean()),
                "k_children_std": float(group["k_children"].std(ddof=0)),
                "positive_quality_rate": float(
                    (group["delta_q_split_original"] >= 0.0).mean()
                ),
                "stability_status": "proxy_only_assignments_not_computed",
            }
        )
    return pd.DataFrame(rows)


def _induced_edges_for_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    nodes: np.ndarray,
    local_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_index[nodes] = np.arange(nodes.shape[0], dtype=np.int64)
    src_local_all = local_index[src]
    src_mask = src_local_all >= 0
    src_positions = np.flatnonzero(src_mask)
    dst_local = local_index[dst[src_positions]]
    keep = dst_local >= 0
    local_src = src_local_all[src_positions][keep].astype(np.uint32, copy=False)
    local_dst = dst_local[keep].astype(np.uint32, copy=False)
    local_weight = np.asarray(weight[src_positions][keep], dtype=np.float64)
    local_index[nodes] = -1
    return local_src, local_dst, local_weight


def _exact_stability(
    *,
    configs: list[SourceRunConfig],
    candidate_df: pd.DataFrame,
    alphas: tuple[float, ...],
    local_seeds: tuple[int, ...],
    max_stability_parents: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidate_df.empty:
        return pd.DataFrame()
    for config in configs:
        membership = _load_membership(config.membership_path).astype(np.uint64, copy=False)
        node_weights = _load_node_weights(config.node_weights_path, int(membership.shape[0]))
        src, dst, edge_weight = _load_graph_arrays(config.graph_dir)
        local_index = np.full(config.n_nodes, -1, dtype=np.int64)
        parent_ids = (
            candidate_df[
                (candidate_df["sample"] == config.sample)
                & (candidate_df["source_seed"] == int(config.source_seed))
            ]["parent_cluster"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
        if max_stability_parents > 0:
            parent_ids = parent_ids[: int(max_stability_parents)]
        for parent_cluster in parent_ids:
            nodes = np.flatnonzero(membership == int(parent_cluster)).astype(np.int64)
            if nodes.size < 2:
                continue
            local_src, local_dst, local_weight = _induced_edges_for_nodes(
                src=src,
                dst=dst,
                weight=edge_weight,
                nodes=nodes,
                local_index=local_index,
            )
            if local_src.size == 0:
                rows.append(
                    {
                        "field": int(config.field),
                        "sample": config.sample,
                        "source_seed": int(config.source_seed),
                        "parent_cluster": int(parent_cluster),
                        "n_candidate_variants": 0,
                        "mean_pairwise_ami": math.nan,
                        "gamma_axis_ami": math.nan,
                        "optional_best_match_child_jaccard": math.nan,
                        "k_children_mean": 0.0,
                        "k_children_std": 0.0,
                        "positive_quality_rate": math.nan,
                        "stability_status": "no_induced_edges",
                    }
                )
                continue
            local_graph = build_leiden_graph(
                n_nodes=int(nodes.size),
                edges_src=local_src,
                edges_dst=local_dst,
                edges_weight=local_weight,
                node_weights=node_weights[nodes],
            )
            partitions: list[np.ndarray] = []
            labels: list[tuple[float, int]] = []
            for alpha in alphas:
                for local_seed in local_seeds:
                    result = local_graph.run_leiden(
                        resolution=float(config.resolution) * float(alpha),
                        seed=int(local_seed),
                        n_iterations=5,
                        randomness=0.01,
                    )
                    partitions.append(np.asarray(result.membership, dtype=np.int64))
                    labels.append((float(alpha), int(local_seed)))
            pair_jaccards: list[float] = []
            gamma_axis_amis: list[float] = []
            for i in range(len(partitions)):
                for j in range(i + 1, len(partitions)):
                    pair_jaccards.append(best_match_child_jaccard(partitions[i], partitions[j]))
                    if labels[i][0] != labels[j][0]:
                        gamma_axis_amis.append(mean_pairwise_ami([partitions[i], partitions[j]]))
            k_values = np.asarray([np.unique(partition).size for partition in partitions], dtype=np.float64)
            group = candidate_df[
                (candidate_df["sample"] == config.sample)
                & (candidate_df["source_seed"] == int(config.source_seed))
                & (candidate_df["parent_cluster"] == int(parent_cluster))
            ]
            rows.append(
                {
                    "field": int(config.field),
                    "sample": config.sample,
                    "source_seed": int(config.source_seed),
                    "parent_cluster": int(parent_cluster),
                    "n_candidate_variants": int(len(partitions)),
                    "mean_pairwise_ami": mean_pairwise_ami(partitions),
                    "gamma_axis_ami": float(np.mean(gamma_axis_amis)) if gamma_axis_amis else math.nan,
                    "optional_best_match_child_jaccard": float(np.mean(pair_jaccards))
                    if pair_jaccards
                    else math.nan,
                    "k_children_mean": float(k_values.mean()) if k_values.size else 0.0,
                    "k_children_std": float(k_values.std()) if k_values.size else 0.0,
                    "positive_quality_rate": float(
                        (group["delta_q_split_original"] >= 0.0).mean()
                    )
                    if not group.empty
                    else math.nan,
                    "stability_status": "exact_local_leiden",
                }
            )
    return pd.DataFrame(rows)


def _membership_metrics(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_doc_weight: float,
) -> dict[str, Any]:
    weights = _cluster_weights(np.asarray(membership, dtype=np.int64), node_weights)
    total = float(weights.sum()) if weights.size else 0.0
    sorted_desc = np.sort(weights)[::-1]
    target = float(target_max_doc_weight)
    return {
        "n_clusters": int(weights.size),
        "max_doc_weight": float(sorted_desc[0]) if sorted_desc.size else 0.0,
        "max_doc_weight_ratio": float(sorted_desc[0] / target)
        if sorted_desc.size and target > 0.0
        else 0.0,
        "n_above_max_doc_weight": int((weights > target).sum()) if target > 0.0 else 0,
        "gini_doc_weight": _gini(weights),
        "entropy_doc_weight": _normalized_entropy(weights),
        "top1_doc_weight_share": float(sorted_desc[:1].sum() / total) if total else 0.0,
        "top5_doc_weight_share": float(sorted_desc[:5].sum() / total) if total else 0.0,
        "target_max_satisfied": bool(target <= 0.0 or not np.any(weights > target)),
    }


def _write_membership(path: Path, membership: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "node_idx": np.arange(membership.shape[0], dtype=np.uint64),
            "cluster": np.asarray(membership, dtype=np.uint64),
        }
    )
    pq.write_table(table, path, compression="zstd")


def _selected_cache_records(selected: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "candidate_index",
        "field",
        "sample",
        "source_seed",
        "parent_cluster",
        "alpha",
        "local_seed",
        "tau_split_ratio",
        "selection_rank",
        "normalized_split_gain",
        "delta_q_split_original",
        "w_between",
        "e_between",
        "status",
    ]
    present = [column for column in columns if column in selected.columns]
    records: list[dict[str, Any]] = []
    for row in selected[present].to_dict("records"):
        records.append({key: _cache_scalar(value) for key, value in row.items()})
    return records


def _apply_cache_metadata(
    *,
    config: SourceRunConfig,
    selected: pd.DataFrame,
    tau_ratio: float,
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    graph_inputs = [
        _file_fingerprint(config.graph_dir / "src.u32.bin"),
        _file_fingerprint(config.graph_dir / "dst.u32.bin"),
        _file_fingerprint(config.graph_dir / "weight.f64.bin"),
    ]
    payload = {
        "schema_version": APPLY_CACHE_SCHEMA_VERSION,
        "application_kernel": APPLICATION_KERNEL,
        "field": int(config.field),
        "sample": config.sample,
        "source_seed": int(config.source_seed),
        "tau_split_ratio": float(tau_ratio),
        "alphas": [float(alpha) for alpha in alphas],
        "resolution": float(config.resolution),
        "target_min_doc_weight": float(config.target_min_doc_weight),
        "target_max_doc_weight": float(config.target_max_doc_weight),
        "n_nodes": int(config.n_nodes),
        "membership_input": _file_fingerprint(config.membership_path),
        "node_weights_input": _file_fingerprint(config.node_weights_path),
        "graph_inputs": graph_inputs,
        "selected_candidates": _selected_cache_records(selected),
    }
    return {
        "schema_version": APPLY_CACHE_SCHEMA_VERSION,
        "cache_key": _hash_json(payload),
        "payload": payload,
    }


def _cache_matches(summary: dict[str, Any], expected: dict[str, Any]) -> bool:
    observed = summary.get("cache_metadata")
    if not isinstance(observed, dict):
        return False
    return (
        observed.get("schema_version") == expected.get("schema_version")
        and observed.get("cache_key") == expected.get("cache_key")
    )


def _apply_selected_for_tau(
    *,
    configs: list[SourceRunConfig],
    selection_df: pd.DataFrame,
    tau_ratio: float,
    alphas: tuple[float, ...],
    output_dir: Path,
    force: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_all = selection_df[
        (selection_df["tau_split_ratio"] == float(tau_ratio))
        & (selection_df["selected_for_apply"])
    ].copy()
    for config in configs:
        run_dir = (
            output_dir
            / "applied_memberships"
            / f"tau_{str(tau_ratio).replace('.', 'p')}"
            / _safe_slug(config.sample)
            / f"source_seed_{config.source_seed}"
        )
        summary_path = run_dir / "summary.json"
        selected = selected_all[
            (selected_all["sample"] == config.sample)
            & (selected_all["source_seed"] == int(config.source_seed))
        ].sort_values("selection_rank")
        cache_metadata = _apply_cache_metadata(
            config=config,
            selected=selected,
            tau_ratio=float(tau_ratio),
            alphas=alphas,
        )
        if summary_path.exists() and not force:
            summary = _read_json(summary_path)
            if _cache_matches(summary, cache_metadata):
                rows.append(summary["effect_row"])
                continue
        membership = _load_membership(config.membership_path).astype(np.uint64, copy=False)
        node_weights = _load_node_weights(config.node_weights_path, int(membership.shape[0]))
        before_metrics = _membership_metrics(
            membership,
            node_weights,
            target_max_doc_weight=float(config.target_max_doc_weight),
        )
        if selected.empty:
            effect_row = {
                "field": int(config.field),
                "sample": config.sample,
                "source_seed": int(config.source_seed),
                "policy": "branch_adaptive_quality_first",
                "application_kernel": APPLICATION_KERNEL,
                "tau_split_ratio": float(tau_ratio),
                "status": "unchanged_no_selected_candidates",
                "accepted_for_contraction": True,
                "n_selected": 0,
                "n_committed": 0,
                "n_skipped": 0,
                "delta_q": 0.0,
                "membership_path": _rel(config.membership_path),
                "source_membership_path": _rel(config.membership_path),
                "initial_max_doc_weight": before_metrics["max_doc_weight"],
                "max_doc_weight": before_metrics["max_doc_weight"],
                "max_doc_weight_ratio": before_metrics["max_doc_weight_ratio"],
                "delta_max_doc_weight": 0.0,
                "n_above_max_doc_weight": before_metrics["n_above_max_doc_weight"],
                "delta_oversize_count": 0,
                "gini_doc_weight": before_metrics["gini_doc_weight"],
                "delta_gini_doc_weight": 0.0,
                "target_max_satisfied": before_metrics["target_max_satisfied"],
                "cache_key": cache_metadata["cache_key"],
            }
            summary = {
                "effect_row": effect_row,
                "committed_candidates": [],
                "skipped_candidates": [],
                "quality_before": None,
                "quality_after": None,
                "exact_delta_q": 0.0,
                "application_kernel": APPLICATION_KERNEL,
                "cache_metadata": cache_metadata,
                "paths": {
                    "membership": None,
                    "summary": _rel(summary_path),
                },
            }
            _write_json(summary_path, summary)
            rows.append(effect_row)
            continue
        graph = _load_graph(config)
        initial_quality = float(
            graph.cpm_quality(membership=membership, resolution=float(config.resolution))
        )
        current = membership.copy()
        committed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        current_quality = initial_quality
        alpha_array = np.asarray(alphas, dtype=np.float64)
        for row in selected.to_dict("records"):
            parent = int(row["parent_cluster"])
            alpha = float(row["alpha"])
            local_seed = int(row["local_seed"])
            result = graph.apply_split_merge_repair_candidates(
                membership=current,
                candidate_clusters=np.asarray([parent], dtype=np.uint64),
                selected_clusters=np.asarray([parent], dtype=np.uint64),
                selected_gamma_multipliers=np.asarray([alpha], dtype=np.float64),
                resolution=float(config.resolution),
                gamma_multipliers=alpha_array,
                min_core_weight=float(config.target_min_doc_weight),
                randomness=0.01,
                repair_epsilon=0.0,
                seed=local_seed,
            )
            if result.n_applied == 0:
                skipped.append({**row, "apply_skip_reason": "candidate_not_replayed"})
                continue
            proposed = np.asarray(result.membership, dtype=np.uint64)
            proposed_quality = float(
                graph.cpm_quality(membership=proposed, resolution=float(config.resolution))
            )
            previous_quality = current_quality
            delta_vs_initial = proposed_quality - initial_quality
            if delta_vs_initial < -1e-9:
                skipped.append({**row, "apply_skip_reason": "quality_regression"})
                continue
            current = proposed
            current_quality = proposed_quality
            committed.append(
                {
                    **row,
                    "apply_delta_q_step": float(proposed_quality - previous_quality),
                    "apply_delta_q_vs_initial": float(delta_vs_initial),
                }
            )
        final_metrics = _membership_metrics(
            current,
            node_weights,
            target_max_doc_weight=float(config.target_max_doc_weight),
        )
        exact_delta_q = float(current_quality - initial_quality)
        membership_path: Path | None = None
        status = (
            "committed"
            if committed
            else (
                "unchanged_no_committed_candidates"
                if int(selected.shape[0])
                else "unchanged_no_selected_candidates"
            )
        )
        if committed:
            membership_path = run_dir / "branch_adaptive_quality_first_membership.parquet"
            _write_membership(membership_path, current)
        effect_row = {
            "field": int(config.field),
            "sample": config.sample,
            "source_seed": int(config.source_seed),
            "policy": "branch_adaptive_quality_first",
            "application_kernel": APPLICATION_KERNEL,
            "tau_split_ratio": float(tau_ratio),
            "status": status,
            "accepted_for_contraction": bool(exact_delta_q >= -1e-9),
            "n_selected": int(selected.shape[0]),
            "n_committed": int(len(committed)),
            "n_skipped": int(len(skipped)),
            "delta_q": exact_delta_q,
            "membership_path": _rel(membership_path) if membership_path else _rel(config.membership_path),
            "source_membership_path": _rel(config.membership_path),
            "initial_max_doc_weight": before_metrics["max_doc_weight"],
            "max_doc_weight": final_metrics["max_doc_weight"],
            "max_doc_weight_ratio": final_metrics["max_doc_weight_ratio"],
            "delta_max_doc_weight": final_metrics["max_doc_weight"]
            - before_metrics["max_doc_weight"],
            "n_above_max_doc_weight": final_metrics["n_above_max_doc_weight"],
            "delta_oversize_count": final_metrics["n_above_max_doc_weight"]
            - before_metrics["n_above_max_doc_weight"],
            "gini_doc_weight": final_metrics["gini_doc_weight"],
            "delta_gini_doc_weight": final_metrics["gini_doc_weight"]
            - before_metrics["gini_doc_weight"],
            "target_max_satisfied": final_metrics["target_max_satisfied"],
            "cache_key": cache_metadata["cache_key"],
        }
        summary = {
            "effect_row": effect_row,
            "committed_candidates": committed,
            "skipped_candidates": skipped,
            "quality_before": initial_quality,
            "quality_after": current_quality,
            "exact_delta_q": exact_delta_q,
            "application_kernel": APPLICATION_KERNEL,
            "cache_metadata": cache_metadata,
            "paths": {
                "membership": _rel(membership_path) if membership_path else None,
                "summary": _rel(summary_path),
            },
        }
        _write_json(summary_path, summary)
        rows.append(effect_row)
    return pd.DataFrame(rows)


def _load_current_policy_effects(validation_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    paths = [
        validation_dir / "source_seed_sweep_effects.csv",
        validation_dir / "field_expansion_source_seed_effects.csv",
    ]
    for source_rank, path in enumerate(paths):
        if path.exists():
            frame = pd.read_csv(path)
            frame["_source_file"] = path.name
            frame["_source_priority"] = source_rank
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "seed" in df.columns and "source_seed" not in df.columns:
        df["source_seed"] = df["seed"]
    dedupe_cols = [
        column
        for column in ["sample", "source_seed", "policy", "membership_role"]
        if column in df.columns
    ]
    if dedupe_cols:
        df = (
            df.sort_values("_source_priority")
            .drop_duplicates(subset=dedupe_cols, keep="last")
            .reset_index(drop=True)
        )
    df = df.drop(columns=[c for c in ["_source_file", "_source_priority"] if c in df.columns])
    return df


def _compare_vs_current(branch_effects: pd.DataFrame, validation_dir: Path) -> pd.DataFrame:
    current = _load_current_policy_effects(validation_dir)
    if branch_effects.empty or current.empty:
        return pd.DataFrame()
    qf = current[current["policy"] == "two_stage_quality_first"].copy()
    if "membership_role" in qf.columns:
        qf = qf[qf["membership_role"] == "effective"].copy()
    if qf.empty:
        return pd.DataFrame()
    keep = [
        "sample",
        "source_seed",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
        "gini_doc_weight",
        "delta_q",
    ]
    missing = [column for column in keep if column not in qf.columns]
    if missing:
        return pd.DataFrame()
    merged = branch_effects.merge(
        qf[keep],
        on=["sample", "source_seed"],
        suffixes=("_branch_adaptive", "_two_stage_quality_first"),
    )
    if merged.empty:
        return merged
    merged["delta_max_ratio_vs_quality_first"] = (
        merged["max_doc_weight_ratio_branch_adaptive"]
        - merged["max_doc_weight_ratio_two_stage_quality_first"]
    )
    merged["delta_oversize_count_vs_quality_first"] = (
        merged["n_above_max_doc_weight_branch_adaptive"]
        - merged["n_above_max_doc_weight_two_stage_quality_first"]
    )
    merged["delta_gini_vs_quality_first"] = (
        merged["gini_doc_weight_branch_adaptive"]
        - merged["gini_doc_weight_two_stage_quality_first"]
    )
    merged["delta_q_vs_quality_first"] = (
        merged["delta_q_branch_adaptive"] - merged["delta_q_two_stage_quality_first"]
    )
    return merged


def _plot_tau_sensitivity(tau_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if tau_df.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(tau_df.shape[0])
    labels = [f"{value:g}" for value in tau_df["tau_split_ratio"]]
    ax1.plot(x, tau_df["n_selected"], marker="o", color="#2f6f9f", label="selected")
    ax1.plot(
        x,
        tau_df["n_policy_accepted"],
        marker="s",
        color="#7c4d27",
        label="policy accepted",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("tau_split / gamma")
    ax1.set_ylabel("candidate count")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="best")
    fig.tight_layout()
    out = output_dir / "figure12_branch_adaptive_tau_sensitivity.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _write_report(
    *,
    candidate_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: Path,
    figure_path: Path | None,
) -> Path:
    lines = [
        "# Branch-Adaptive Quality-First Diagnostics",
        "",
        "This diagnostics-first pilot generates local split candidates on induced parent subgraphs and evaluates CPM split accounting at the original graph resolution.",
        "",
        "## Scope",
        "",
        f"- Candidate rows: {len(candidate_df)}",
        f"- Parent rows: {len(parent_df)}",
        f"- Fields: {', '.join(str(x) for x in sorted(candidate_df['field'].unique())) if not candidate_df.empty else ''}",
        f"- Source seeds: {', '.join(str(x) for x in sorted(candidate_df['source_seed'].unique())) if not candidate_df.empty else ''}",
        f"- Application kernel: `{APPLICATION_KERNEL}` with `repair_epsilon = 0.0` when `--apply` is used",
        f"- Child weight entropy: `{CHILD_WEIGHT_ENTROPY_SOURCE}` (`child_weight_entropy_is_exact = false`)",
        "",
    ]
    if not tau_df.empty:
        lines.extend(
            [
                "## Tau Sensitivity",
                "",
                _markdown_table(tau_df),
                "",
            ]
        )
    if not parent_df.empty:
        field_summary = (
            parent_df.groupby("field")
            .agg(
                n_parents=("parent_cluster", "count"),
                mean_best_gain=("best_normalized_split_gain", "mean"),
                positive_parent_rate=("n_quality_positive", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        lines.extend(
            [
                "## Field Breakdown",
                "",
                _markdown_table(field_summary),
                "",
            ]
        )
    if not stability_df.empty:
        status_counts = stability_df["stability_status"].value_counts().reset_index()
        status_counts.columns = ["stability_status", "rows"]
        lines.extend(
            [
                "## Stability Diagnostics",
                "",
                _markdown_table(status_counts),
                "",
            ]
        )
    if figure_path is not None:
        lines.extend(["## Figures", "", f"- `{_rel(figure_path)}`", ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `branch_adaptive_split_candidates.csv` / `.parquet`",
            "- `branch_adaptive_parent_summary.csv` / `.parquet`",
            "- `branch_adaptive_tau_sensitivity.csv` / `.parquet`",
            "- `branch_adaptive_candidate_stability.csv` / `.parquet`",
            "- `branch_adaptive_compute_summary.json`",
            "",
            "Semantic coherence is intentionally excluded from acceptance in this pilot. It remains a post-hoc explanation layer.",
            "",
            "Applied membership artifacts replay selected branch candidates through the split-merge-repair kernel. Candidate diagnostics remain induced-generation/original-graph CPM accounting, while the application path records the kernel explicitly for auditability.",
            "",
        ]
    )
    path = output_dir / "branch_adaptive_diagnostics_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_manuscript_skeletons(output_dir: Path) -> None:
    docs_dir = REPO_ROOT / "docs"
    outline = docs_dir / "scientometrics_manuscript_outline.md"
    notes = docs_dir / "branch_adaptive_case_study_notes.md"
    if not outline.exists():
        outline.write_text(
            """# Scientometrics Manuscript Outline

## Working Claim

Global size thresholds are insufficient for heterogeneous science maps. CPM critical-gamma evidence makes local split and merge decisions auditable while preserving exact CPM quality as the source of truth.

## Method Spine

1. Baseline CPM Leiden hierarchy construction.
2. Small-cluster repair as operational map-unit cleanup.
3. Quality-first oversize split repair as the robust default.
4. Branch-adaptive diagnostics using normalized split gain and tau sensitivity.
5. Contraction-aware next-level validation.

## Main Figures And Tables

- Six-field quality-first evidence.
- Hard-cap fallback diagnostic.
- Semantic sanity check as post-hoc explanation.
- Branch-adaptive tau sensitivity.
- Field30 and field26 case study.

## Claim Boundary

MCMC adaptive cut is related work only. Semantic coherence does not enter primary acceptance. Stability is a ranking/reporting diagnostic, not a hard threshold in the pilot.
""",
            encoding="utf-8",
        )
    if not notes.exists():
        notes.write_text(
            f"""# Branch-Adaptive Case Study Notes

Generated by `{_rel(output_dir)}`.

## Field30 Positive Case

- Inspect tau rows where branch-adaptive selects parents with non-negative exact CPM split delta.
- Compare source max/target reduction against current `two_stage_quality_first`.
- Carry accepted memberships to next-level propagation before freezing figures.

## Field26 Mixed Case

- Use normalized split gain and stability diagnostics to explain accepted versus rejected parents.
- Treat no-split or low-gain parents as evidence that the broad source topic should not be forced under `quality_first`.

## Acceptance Defaults

- `epsilon_Q = 0`
- `tau_split / gamma in {{0.0, 0.001, 0.005, 0.01, 0.05}}`
- applied memberships use `{APPLICATION_KERNEL}` with `repair_epsilon = 0.0`
- `child_weight_entropy` is a proxy unless `child_weight_entropy_is_exact` is true
- semantic coherence post-hoc only
- stability diagnostic only
""",
            encoding="utf-8",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BRANCH_OUTPUT_DIR)
    parser.add_argument("--fields", default=",".join(str(x) for x in DEFAULT_FIELDS))
    parser.add_argument("--source-seeds", default=",".join(str(x) for x in DEFAULT_SOURCE_SEEDS))
    parser.add_argument("--local-seeds", default=",".join(str(x) for x in DEFAULT_LOCAL_SEEDS))
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_STAGE1_ALPHAS))
    parser.add_argument(
        "--stage2-alphas",
        default=",".join(str(x) for x in DEFAULT_STAGE2_ALPHAS),
        help="Recorded in compute summary for conservative expansion; not used unless passed via --alphas.",
    )
    parser.add_argument(
        "--tau-split-ratios",
        default=",".join(str(x) for x in DEFAULT_TAU_SPLIT_RATIOS),
    )
    parser.add_argument("--max-parents", type=int, default=1000)
    parser.add_argument(
        "--compute-stability",
        action="store_true",
        help="Run extra local Leiden passes to compute exact AMI/Jaccard stability.",
    )
    parser.add_argument("--max-stability-parents", type=int, default=0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply selected candidates per tau into branch-adaptive memberships.",
    )
    parser.add_argument("--apply-tau-ratios", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    validation_dir = _repo_path(args.validation_dir) or args.validation_dir
    output_dir = _repo_path(args.output_dir) or args.output_dir
    assert validation_dir is not None and output_dir is not None

    fields = _parse_int_list(args.fields)
    source_seeds = _parse_int_list(args.source_seeds)
    local_seeds = _parse_int_list(args.local_seeds)
    alphas = _parse_float_list(args.alphas)
    tau_ratios = _parse_float_list(args.tau_split_ratios)
    stage2_alphas = _parse_float_list(args.stage2_alphas)
    configs = [
        _discover_source_run(field=field, source_seed=seed, validation_dir=validation_dir)
        for field in fields
        for seed in source_seeds
    ]

    if args.dry_run:
        payload = {
            "status": "dry_run_ok",
            "n_configs": len(configs),
            "configs": [
                {
                    "field": config.field,
                    "sample": config.sample,
                    "source_seed": config.source_seed,
                    "graph_dir": _rel(config.graph_dir),
                    "membership_path": _rel(config.membership_path),
                    "resolution": config.resolution,
                    "target_min_doc_weight": config.target_min_doc_weight,
                    "target_max_doc_weight": config.target_max_doc_weight,
                    "n_nodes": config.n_nodes,
                }
                for config in configs
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    candidate_df, run_summaries = _run_candidate_diagnostics(
        configs=configs,
        alphas=alphas,
        local_seeds=local_seeds,
        max_parents=int(args.max_parents),
    )
    for column in REQUIRED_CANDIDATE_COLUMNS:
        if column not in candidate_df.columns:
            candidate_df[column] = pd.Series(dtype="float64")
    candidate_df = candidate_df[[*REQUIRED_CANDIDATE_COLUMNS, *[c for c in candidate_df.columns if c not in REQUIRED_CANDIDATE_COLUMNS]]]
    parent_df = _parent_summary(candidate_df)
    tau_df, selection_df = _tau_sensitivity(candidate_df, tau_ratios)
    if args.compute_stability:
        stability_df = _exact_stability(
            configs=configs,
            candidate_df=candidate_df,
            alphas=alphas,
            local_seeds=local_seeds,
            max_stability_parents=int(args.max_stability_parents),
        )
    else:
        stability_df = _proxy_stability(candidate_df)

    _write_table(candidate_df, output_dir / "branch_adaptive_split_candidates")
    _write_table(parent_df, output_dir / "branch_adaptive_parent_summary")
    _write_table(tau_df, output_dir / "branch_adaptive_tau_sensitivity")
    _write_table(stability_df, output_dir / "branch_adaptive_candidate_stability")
    if not selection_df.empty:
        _write_table(selection_df, output_dir / "branch_adaptive_tau_candidate_selection")

    branch_effect_frames: list[pd.DataFrame] = []
    if args.apply:
        apply_tau_ratios = (
            _parse_float_list(args.apply_tau_ratios)
            if str(args.apply_tau_ratios).strip()
            else tau_ratios
        )
        for tau_ratio in apply_tau_ratios:
            branch_effect_frames.append(
                _apply_selected_for_tau(
                    configs=configs,
                    selection_df=selection_df,
                    tau_ratio=float(tau_ratio),
                    alphas=alphas,
                    output_dir=output_dir,
                    force=bool(args.force),
                )
            )
        branch_effects = (
            pd.concat(branch_effect_frames, ignore_index=True, sort=False)
            if branch_effect_frames
            else pd.DataFrame()
        )
        _write_table(branch_effects, output_dir / "branch_adaptive_policy_effects")
        comparison = _compare_vs_current(branch_effects, validation_dir)
        _write_table(comparison, output_dir / "branch_adaptive_quality_first_vs_current")

    figure_path = _plot_tau_sensitivity(tau_df, output_dir)
    report_path = _write_report(
        candidate_df=candidate_df,
        parent_df=parent_df,
        tau_df=tau_df,
        stability_df=stability_df,
        output_dir=output_dir,
        figure_path=figure_path,
    )
    _write_manuscript_skeletons(output_dir)
    compute_summary = {
        "status": "completed",
        "fields": list(fields),
        "source_seeds": list(source_seeds),
        "local_seeds": list(local_seeds),
        "alphas": list(alphas),
        "stage2_alphas": list(stage2_alphas),
        "tau_split_ratios": list(tau_ratios),
        "epsilon_q": 0.0,
        "candidate_generation": "induced_local_graph",
        "acceptance_evaluation": "original_graph_exact_cpm",
        "application_kernel": APPLICATION_KERNEL,
        "apply_cache_schema_version": APPLY_CACHE_SCHEMA_VERSION,
        "child_weight_entropy_source": CHILD_WEIGHT_ENTROPY_SOURCE,
        "child_weight_entropy_is_exact": False,
        "stability": "exact" if args.compute_stability else "proxy_only",
        "semantic_coherence": "post_hoc_only",
        "mcmc": "related_work_only",
        "n_candidate_rows": int(len(candidate_df)),
        "n_parent_rows": int(len(parent_df)),
        "n_tau_rows": int(len(tau_df)),
        "n_stability_rows": int(len(stability_df)),
        "run_summaries": run_summaries,
        "elapsed_sec": float(time.perf_counter() - t0),
        "paths": {
            "split_candidates": _rel(output_dir / "branch_adaptive_split_candidates.csv"),
            "parent_summary": _rel(output_dir / "branch_adaptive_parent_summary.csv"),
            "tau_sensitivity": _rel(output_dir / "branch_adaptive_tau_sensitivity.csv"),
            "candidate_stability": _rel(output_dir / "branch_adaptive_candidate_stability.csv"),
            "report": _rel(report_path),
            "tau_figure": _rel(figure_path) if figure_path else None,
        },
    }
    _write_json(output_dir / "branch_adaptive_compute_summary.json", compute_summary)
    print(json.dumps(compute_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
