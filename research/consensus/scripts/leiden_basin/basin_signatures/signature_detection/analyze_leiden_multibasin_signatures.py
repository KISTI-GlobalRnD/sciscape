#!/usr/bin/env python3
"""Summarize p5 basin signatures from Leiden multifidelity candidate rows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import numpy as np
import pandas as pd

GROUP_COLUMNS = [
    "candidate_eval_mode",
    "case",
    "seed",
    "candidate_budget",
    "max_group_candidates",
]
TOP_K_VALUES = (1, 2, 3, 5, 10, 20)
SKETCH_BASELINE_COLUMN = "p5_basin_sketch_baseline_membership"
SKETCH_MEMBERSHIP_COLUMN = "p5_basin_sketch_membership"
SKETCH_HASH_COLUMN = "p5_basin_sketch_node_hash"
CHANGED_SUPPORT_COLUMN = "p5_basin_changed_support_nodes"
ALIGNMENT_ERROR_NODE_COUNT_COLUMN = "p5_alignment_error_nodes_vs_baseline"
ALIGNMENT_ERROR_FRACTION_COLUMN = "p5_alignment_error_fraction_vs_baseline"
ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN = "p5_aligned_changed_support_node_count"
ALIGNED_CHANGED_SUPPORT_NODES_COLUMN = "p5_aligned_changed_support_nodes"

def _basin_scale_tier(changed_fraction: float) -> str:
    if not math.isfinite(changed_fraction) or changed_fraction <= 0.0:
        return "none"
    if changed_fraction < 0.001:
        return "micro"
    if changed_fraction < 0.01:
        return "meso"
    return "macro"

def _read_csvs(input_dir: Path, filename: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(f"**/{filename}"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame["source_path"] = str(path.relative_to(input_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default

def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")

def _copy_first_available_column(
    frame: pd.DataFrame,
    *,
    target: str,
    sources: tuple[str, ...],
) -> None:
    if target in frame.columns:
        return
    for source in sources:
        if source in frame.columns:
            frame[target] = frame[source]
            return

def _ensure_alignment_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    _copy_first_available_column(
        out,
        target=ALIGNMENT_ERROR_NODE_COUNT_COLUMN,
        sources=("p5_changed_nodes_vs_baseline",),
    )
    _copy_first_available_column(
        out,
        target=ALIGNMENT_ERROR_FRACTION_COLUMN,
        sources=("p5_changed_fraction_vs_baseline",),
    )
    _copy_first_available_column(
        out,
        target=ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN,
        sources=("p5_basin_changed_support_node_count", ALIGNMENT_ERROR_NODE_COUNT_COLUMN),
    )
    _copy_first_available_column(
        out,
        target=ALIGNED_CHANGED_SUPPORT_NODES_COLUMN,
        sources=(CHANGED_SUPPORT_COLUMN,),
    )
    _copy_first_available_column(
        out,
        target="p5_changed_nodes_vs_baseline",
        sources=(ALIGNMENT_ERROR_NODE_COUNT_COLUMN,),
    )
    _copy_first_available_column(
        out,
        target="p5_changed_fraction_vs_baseline",
        sources=(ALIGNMENT_ERROR_FRACTION_COLUMN,),
    )
    _copy_first_available_column(
        out,
        target="p5_basin_changed_support_node_count",
        sources=(ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN,),
    )
    _copy_first_available_column(
        out,
        target=CHANGED_SUPPORT_COLUMN,
        sources=(ALIGNED_CHANGED_SUPPORT_NODES_COLUMN,),
    )
    return out

def _parse_sketch(value: Any) -> np.ndarray:
    if value is None or pd.isna(value):
        return np.asarray([], dtype=np.int64)
    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=np.int64)
    try:
        return np.asarray([int(part) for part in text.split(";") if part], dtype=np.int64)
    except ValueError:
        return np.asarray([], dtype=np.int64)

def _jaccard_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = len(left_set | right_set)
    if union == 0:
        return 0.0, 0, 0
    intersection = len(left_set & right_set)
    return 1.0 - intersection / union, intersection, union

def _coassignment_bits(labels: np.ndarray) -> np.ndarray:
    if labels.size < 2:
        return np.asarray([], dtype=bool)
    same = labels[:, None] == labels[None, :]
    upper = np.triu_indices(labels.size, k=1)
    return same[upper]

class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_COLUMNS if column in frame.columns]

def _signature_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or "p5_basin_signature" not in candidates.columns:
        return pd.DataFrame()
    frame = candidates.copy()
    frame["p5_basin_signature"] = frame["p5_basin_signature"].fillna("").astype(str)
    frame = frame[frame["p5_basin_signature"].str.len() > 0].copy()
    if frame.empty:
        return frame
    frame = _ensure_alignment_aliases(frame)
    for column in (
        "candidate_index",
        "p1_delta_q",
        "p5_delta_q",
        "p5_relative_delta_q_ppm",
        "p5_changed_fraction_vs_baseline",
        "p5_changed_nodes_vs_baseline",
        ALIGNMENT_ERROR_FRACTION_COLUMN,
        ALIGNMENT_ERROR_NODE_COUNT_COLUMN,
        ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN,
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame

def _mark_material_gain(
    frame: pd.DataFrame, *, material_delta_q: float, material_relative_ppm: float
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    delta = pd.to_numeric(out.get("p5_delta_q"), errors="coerce")
    rel = pd.to_numeric(out.get("p5_relative_delta_q_ppm"), errors="coerce")
    clears_abs = delta >= material_delta_q
    clears_rel = rel >= material_relative_ppm
    if rel.notna().any():
        out["material_gain"] = clears_abs & clears_rel
    else:
        out["material_gain"] = clears_abs
    out["low_roi_positive_gain"] = (delta > 0) & ~out["material_gain"]
    return out

def _entropy(signatures: pd.Series) -> float:
    if signatures.empty:
        return math.nan
    counts = signatures.value_counts(dropna=True)
    total = float(counts.sum())
    if total <= 0.0:
        return math.nan
    entropy = 0.0
    for count in counts:
        probability = float(count) / total
        entropy -= probability * math.log(probability)
    return entropy

def _ranked_indices(group: pd.DataFrame, metric: str) -> list[Any]:
    if metric not in group.columns:
        metric = "p5_delta_q"
    order = group.assign(
        _rank_metric=pd.to_numeric(group.get(metric), errors="coerce"),
        _candidate_index=pd.to_numeric(group.get("candidate_index"), errors="coerce").fillna(0),
    ).sort_values(
        ["_rank_metric", "_candidate_index"],
        ascending=[False, True],
        na_position="last",
    )
    return list(order.index)

def build_basin_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        for signature, basin in group.groupby("p5_basin_signature", dropna=False):
            p5 = pd.to_numeric(basin.get("p5_delta_q"), errors="coerce")
            if p5.notna().sum() == 0:
                continue
            best_idx = p5.idxmax()
            best = basin.loc[best_idx]
            mean_alignment_error_fraction = _finite_float(
                _numeric_series(basin, ALIGNMENT_ERROR_FRACTION_COLUMN).mean()
            )
            max_alignment_error_nodes = _finite_float(
                _numeric_series(basin, ALIGNMENT_ERROR_NODE_COUNT_COLUMN).max()
            )
            mean_aligned_changed_support = _finite_float(
                _numeric_series(basin, ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN).mean()
            )
            max_aligned_changed_support = _finite_float(
                _numeric_series(basin, ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN).max()
            )
            rows.append(
                {
                    **base,
                    "dongdaemun_family": "diagnostic",
                    "dongdaemun_claim_level": "diagnostic",
                    "effective_output": False,
                    "p5_basin_signature": signature,
                    "basin_candidate_count": int(len(basin)),
                    "basin_best_delta_q": _finite_float(best.get("p5_delta_q")),
                    "basin_best_relative_delta_q_ppm": _finite_float(
                        best.get("p5_relative_delta_q_ppm")
                    ),
                    "basin_best_candidate_index": int(best.get("candidate_index", -1)),
                    "material_basin": bool(basin.get("material_gain", False).map(bool).any()),
                    "mean_alignment_error_fraction_vs_baseline": (
                        mean_alignment_error_fraction
                    ),
                    "max_alignment_error_nodes_vs_baseline": max_alignment_error_nodes,
                    "mean_aligned_changed_support_node_count": (
                        mean_aligned_changed_support
                    ),
                    "max_aligned_changed_support_node_count": (
                        max_aligned_changed_support
                    ),
                    "mean_changed_fraction_vs_baseline": mean_alignment_error_fraction,
                    "max_changed_nodes_vs_baseline": max_alignment_error_nodes,
                }
            )
            rows[-1]["basin_scale_tier"] = _basin_scale_tier(
                _finite_float(rows[-1]["mean_alignment_error_fraction_vs_baseline"])
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = group_cols + ["basin_best_delta_q", "basin_best_candidate_index"]
    return out.sort_values(sort_cols, ascending=[True] * len(group_cols) + [False, True])

def build_basin_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        p5 = pd.to_numeric(group.get("p5_delta_q"), errors="coerce")
        labeled = group[p5.notna()]
        if labeled.empty:
            continue
        full_best_idx = p5.idxmax()
        full_best = group.loc[full_best_idx]
        basin_counts = labeled["p5_basin_signature"].value_counts()
        material = labeled[labeled.get("material_gain", False).map(bool)]
        material_signatures = set(material["p5_basin_signature"])
        basin_scale: dict[str, str] = {}
        for signature, basin in labeled.groupby("p5_basin_signature", dropna=False):
            changed = _numeric_series(basin, ALIGNMENT_ERROR_FRACTION_COLUMN)
            basin_scale[str(signature)] = _basin_scale_tier(_finite_float(changed.mean()))
        macro_signatures = {
            signature for signature, tier in basin_scale.items() if tier == "macro"
        }
        meso_or_macro_signatures = {
            signature
            for signature, tier in basin_scale.items()
            if tier in {"meso", "macro"}
        }
        top_dominance = (
            float(basin_counts.iloc[0]) / float(basin_counts.sum())
            if not basin_counts.empty
            else math.nan
        )
        max_alignment_error_fraction = _finite_float(
            _numeric_series(labeled, ALIGNMENT_ERROR_FRACTION_COLUMN).max()
        )
        mean_alignment_error_fraction = _finite_float(
            _numeric_series(labeled, ALIGNMENT_ERROR_FRACTION_COLUMN).mean()
        )
        max_alignment_error_nodes = _finite_float(
            _numeric_series(labeled, ALIGNMENT_ERROR_NODE_COUNT_COLUMN).max()
        )
        mean_aligned_changed_support = _finite_float(
            _numeric_series(labeled, ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN).mean()
        )
        max_aligned_changed_support = _finite_float(
            _numeric_series(labeled, ALIGNED_CHANGED_SUPPORT_NODE_COUNT_COLUMN).max()
        )
        rows.append(
            {
                **base,
                "dongdaemun_family": "diagnostic",
                "dongdaemun_claim_level": "diagnostic",
                "effective_output": False,
                "candidate_count": int(len(group)),
                "p5_labeled_count": int(len(labeled)),
                "distinct_basin_count": int(labeled["p5_basin_signature"].nunique()),
                "distinct_material_basin_count": int(len(material_signatures)),
                "distinct_meso_or_macro_basin_count": int(len(meso_or_macro_signatures)),
                "distinct_meso_or_macro_material_basin_count": int(
                    len(meso_or_macro_signatures & material_signatures)
                ),
                "distinct_macro_basin_count": int(len(macro_signatures)),
                "distinct_macro_material_basin_count": int(
                    len(macro_signatures & material_signatures)
                ),
                "basin_entropy": _entropy(labeled["p5_basin_signature"]),
                "top_basin_dominance": top_dominance,
                "full_p5_best_delta_q": _finite_float(full_best.get("p5_delta_q")),
                "full_p5_best_relative_delta_q_ppm": _finite_float(
                    full_best.get("p5_relative_delta_q_ppm")
                ),
                "full_p5_best_candidate_index": int(
                    full_best.get("candidate_index", -1)
                ),
                "best_basin_signature": str(full_best.get("p5_basin_signature", "")),
                "best_basin_is_material": bool(full_best.get("material_gain", False)),
                "low_roi_positive_count": int(
                    labeled.get("low_roi_positive_gain", False).map(bool).sum()
                ),
                "max_alignment_error_fraction_vs_baseline": (
                    max_alignment_error_fraction
                ),
                "mean_alignment_error_fraction_vs_baseline": (
                    mean_alignment_error_fraction
                ),
                "max_alignment_error_nodes_vs_baseline": max_alignment_error_nodes,
                "mean_aligned_changed_support_node_count": (
                    mean_aligned_changed_support
                ),
                "max_aligned_changed_support_node_count": max_aligned_changed_support,
                "max_changed_fraction_vs_baseline": max_alignment_error_fraction,
                "mean_changed_fraction_vs_baseline": mean_alignment_error_fraction,
            }
        )
    return pd.DataFrame(rows)

def build_coverage_rows(candidates: pd.DataFrame, *, rank_metric: str = "p1_delta_q") -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        p5 = pd.to_numeric(group.get("p5_delta_q"), errors="coerce")
        labeled = group[p5.notna()]
        if labeled.empty:
            continue
        full_best_idx = p5.idxmax()
        full_best = group.loc[full_best_idx]
        full_best_delta = _finite_float(full_best.get("p5_delta_q"))
        best_signature = str(full_best.get("p5_basin_signature", ""))
        all_signatures = set(labeled["p5_basin_signature"])
        material_signatures = set(
            labeled[labeled.get("material_gain", False).map(bool)]["p5_basin_signature"]
        )
        ranked = _ranked_indices(labeled, rank_metric)
        for top_k in TOP_K_VALUES:
            selected = labeled.loc[ranked[: min(top_k, len(ranked))]]
            selected_p5 = pd.to_numeric(selected.get("p5_delta_q"), errors="coerce")
            best_selected_delta = _finite_float(selected_p5.max())
            selected_signatures = set(selected["p5_basin_signature"])
            selected_material_signatures = selected_signatures & material_signatures
            rows.append(
                {
                    **base,
                    "dongdaemun_family": "diagnostic",
                    "dongdaemun_claim_level": "diagnostic",
                    "effective_output": False,
                    "rank_metric": rank_metric,
                    "top_k": top_k,
                    "p5_evaluated": int(len(selected)),
                    "best_basin_hit_at_k": bool(best_signature in selected_signatures),
                    "best_quality_regret_at_k": (
                        full_best_delta - best_selected_delta
                        if math.isfinite(full_best_delta)
                        and math.isfinite(best_selected_delta)
                        else math.nan
                    ),
                    "distinct_basin_coverage_at_k": (
                        len(selected_signatures) / len(all_signatures)
                        if all_signatures
                        else math.nan
                    ),
                    "distinct_material_basin_coverage_at_k": (
                        len(selected_material_signatures) / len(material_signatures)
                        if material_signatures
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)

def build_pairwise_basin_matrix(
    candidates: pd.DataFrame,
    *,
    coarse_endpoint_tau: float = 0.02,
    coarse_support_tau: float = 0.5,
    iso_q_delta: float = 10.0,
    iso_q_relative_ppm: float = 10.0,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    required = {SKETCH_BASELINE_COLUMN, SKETCH_MEMBERSHIP_COLUMN, SKETCH_HASH_COLUMN}
    if not required <= set(candidates.columns):
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        labeled = group[
            pd.to_numeric(group.get("p5_delta_q"), errors="coerce").notna()
        ].copy()
        labeled = labeled[
            labeled[SKETCH_MEMBERSHIP_COLUMN].fillna("").astype(str).str.len() > 0
        ].copy()
        if len(labeled) < 2:
            continue
        sketch_hashes = set(labeled[SKETCH_HASH_COLUMN].fillna("").astype(str))
        if len(sketch_hashes) != 1 or "" in sketch_hashes:
            continue
        baseline = _parse_sketch(labeled.iloc[0].get(SKETCH_BASELINE_COLUMN))
        if baseline.size < 2:
            continue
        baseline_bits = _coassignment_bits(baseline)
        sketches: list[dict[str, Any]] = []
        for row_idx, row in labeled.iterrows():
            membership = _parse_sketch(row.get(SKETCH_MEMBERSHIP_COLUMN))
            if membership.size != baseline.size:
                continue
            endpoint_bits = _coassignment_bits(membership)
            if endpoint_bits.size != baseline_bits.size:
                continue
            sketches.append(
                {
                    "row_idx": row_idx,
                    "candidate_index": int(row.get("candidate_index", -1)),
                    "signature": str(row.get("p5_basin_signature", "")),
                    "p5_delta_q": _finite_float(row.get("p5_delta_q")),
                    "p5_relative_delta_q_ppm": _finite_float(
                        row.get("p5_relative_delta_q_ppm")
                    ),
                    "changed_fraction": _finite_float(
                        row.get(ALIGNMENT_ERROR_FRACTION_COLUMN)
                    ),
                    "changed_support_nodes": (
                        _parse_sketch(row.get(ALIGNED_CHANGED_SUPPORT_NODES_COLUMN))
                        if ALIGNED_CHANGED_SUPPORT_NODES_COLUMN in labeled.columns
                        else np.asarray([], dtype=np.int64)
                    ),
                    "endpoint_bits": endpoint_bits,
                    "changed_pair_bits": endpoint_bits != baseline_bits,
                }
            )
        for left_pos, left in enumerate(sketches):
            for right in sketches[left_pos + 1 :]:
                endpoint_distance = float(
                    np.mean(left["endpoint_bits"] != right["endpoint_bits"])
                )
                left_changed = left["changed_pair_bits"]
                right_changed = right["changed_pair_bits"]
                intersection = int(np.logical_and(left_changed, right_changed).sum())
                union = int(np.logical_or(left_changed, right_changed).sum())
                support_jaccard_distance = 0.0 if union == 0 else 1.0 - intersection / union
                (
                    changed_node_support_jaccard_distance,
                    changed_node_support_intersection,
                    changed_node_support_union,
                ) = _jaccard_distance(
                    left["changed_support_nodes"], right["changed_support_nodes"]
                )
                coarse_support_distance = (
                    changed_node_support_jaccard_distance
                    if changed_node_support_union > 0
                    else support_jaccard_distance
                )
                q_delta_abs = abs(left["p5_delta_q"] - right["p5_delta_q"])
                q_relative_ppm_abs = abs(
                    left["p5_relative_delta_q_ppm"] - right["p5_relative_delta_q_ppm"]
                )
                same_coarse = (
                    endpoint_distance <= coarse_endpoint_tau
                    and coarse_support_distance <= coarse_support_tau
                )
                iso_q_pair = (
                    q_delta_abs <= iso_q_delta
                    and q_relative_ppm_abs <= iso_q_relative_ppm
                )
                rows.append(
                    {
                        **base,
                        "dongdaemun_family": "diagnostic",
                        "dongdaemun_claim_level": "diagnostic",
                        "effective_output": False,
                        "left_candidate_index": left["candidate_index"],
                        "right_candidate_index": right["candidate_index"],
                        "left_basin_signature": left["signature"],
                        "right_basin_signature": right["signature"],
                        "sketch_node_hash": next(iter(sketch_hashes)),
                        "sketch_sample_size": int(baseline.size),
                        "sample_pair_count": int(baseline_bits.size),
                        "sample_coassignment_distance": endpoint_distance,
                        "changed_pair_support_jaccard_distance": support_jaccard_distance,
                        "changed_pair_support_intersection": intersection,
                        "changed_pair_support_union": union,
                        "changed_node_support_jaccard_distance": (
                            changed_node_support_jaccard_distance
                        ),
                        "changed_node_support_intersection": (
                            changed_node_support_intersection
                        ),
                        "changed_node_support_union": changed_node_support_union,
                        "coarse_support_distance": coarse_support_distance,
                        "coarse_support_distance_source": (
                            "changed_node_support"
                            if changed_node_support_union > 0
                            else "changed_pair_support"
                        ),
                        "left_p5_delta_q": left["p5_delta_q"],
                        "right_p5_delta_q": right["p5_delta_q"],
                        "q_delta_abs": q_delta_abs,
                        "q_relative_ppm_abs": q_relative_ppm_abs,
                        "left_changed_fraction_vs_baseline": left["changed_fraction"],
                        "right_changed_fraction_vs_baseline": right["changed_fraction"],
                        "same_coarse_basin": same_coarse,
                        "iso_q_pair": iso_q_pair,
                        "partition_distinct_iso_q_pair": (not same_coarse) and iso_q_pair,
                        "coarse_endpoint_tau": coarse_endpoint_tau,
                        "coarse_support_tau": coarse_support_tau,
                        "iso_q_delta": iso_q_delta,
                        "iso_q_relative_ppm": iso_q_relative_ppm,
                    }
                )
    return pd.DataFrame(rows)

def build_coarse_basin_rows(
    candidates: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty or pairwise.empty:
        return pd.DataFrame()
    group_cols = _group_columns(candidates)
    if not group_cols:
        candidates = candidates.copy()
        candidates["_all"] = "all"
        group_cols = ["_all"]
    rows: list[dict[str, Any]] = []
    for group_key, group in candidates.groupby(group_cols, dropna=False):
        group_key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base = dict(zip(group_cols, group_key_values, strict=False))
        labeled = group[
            pd.to_numeric(group.get("p5_delta_q"), errors="coerce").notna()
        ].copy()
        if labeled.empty:
            continue
        labeled["_candidate_index_int"] = pd.to_numeric(
            labeled.get("candidate_index"), errors="coerce"
        ).astype("Int64")
        candidate_indices = [int(value) for value in labeled["_candidate_index_int"].dropna()]
        if not candidate_indices:
            continue
        index_to_pos = {candidate_index: pos for pos, candidate_index in enumerate(candidate_indices)}
        union_find = _UnionFind(len(candidate_indices))
        mask = pd.Series([True] * len(pairwise), index=pairwise.index)
        for column, value in base.items():
            if column in pairwise.columns:
                mask &= pairwise[column] == value
        for _, pair in pairwise[mask].iterrows():
            if not bool(pair.get("same_coarse_basin", False)):
                continue
            left = int(pair.get("left_candidate_index", -1))
            right = int(pair.get("right_candidate_index", -1))
            if left in index_to_pos and right in index_to_pos:
                union_find.union(index_to_pos[left], index_to_pos[right])
        root_to_coarse: dict[int, int] = {}
        coarse_by_candidate: dict[int, int] = {}
        for candidate_index in candidate_indices:
            root = union_find.find(index_to_pos[candidate_index])
            coarse_id = root_to_coarse.setdefault(root, len(root_to_coarse))
            coarse_by_candidate[candidate_index] = coarse_id
        labeled["_coarse_basin_id"] = labeled["_candidate_index_int"].map(coarse_by_candidate)
        for coarse_id, basin in labeled.groupby("_coarse_basin_id", dropna=False):
            p5 = pd.to_numeric(basin.get("p5_delta_q"), errors="coerce")
            best_idx = p5.idxmax()
            best = basin.loc[best_idx]
            mean_alignment_error_fraction = _finite_float(
                _numeric_series(basin, ALIGNMENT_ERROR_FRACTION_COLUMN).mean()
            )
            rows.append(
                {
                    **base,
                    "dongdaemun_family": "diagnostic",
                    "dongdaemun_claim_level": "diagnostic",
                    "effective_output": False,
                    "coarse_basin_id": int(coarse_id),
                    "member_candidate_indices": ";".join(
                        str(int(value))
                        for value in sorted(
                            pd.to_numeric(
                                basin.get("candidate_index"), errors="coerce"
                            )
                            .dropna()
                            .tolist()
                        )
                    ),
                    "candidate_count": int(len(basin)),
                    "exact_basin_count": int(basin["p5_basin_signature"].nunique()),
                    "material_candidate_count": int(
                        basin.get("material_gain", False).map(bool).sum()
                    ),
                    "best_candidate_index": int(best.get("candidate_index", -1)),
                    "best_p5_delta_q": _finite_float(best.get("p5_delta_q")),
                    "best_relative_delta_q_ppm": _finite_float(
                        best.get("p5_relative_delta_q_ppm")
                    ),
                    "mean_p5_delta_q": _finite_float(p5.mean()),
                    "min_p5_delta_q": _finite_float(p5.min()),
                    "max_p5_delta_q": _finite_float(p5.max()),
                    "q_range": _finite_float(p5.max() - p5.min()),
                    "mean_alignment_error_fraction_vs_baseline": (
                        mean_alignment_error_fraction
                    ),
                    "mean_changed_fraction_vs_baseline": mean_alignment_error_fraction,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols = group_cols + ["best_p5_delta_q", "coarse_basin_id"]
    return out.sort_values(sort_cols, ascending=[True] * len(group_cols) + [False, True])

def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)

def write_report(
    output_dir: Path,
    basin_summary: pd.DataFrame,
    coverage_rows: pd.DataFrame,
    pairwise_rows: pd.DataFrame | None = None,
    coarse_rows: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Leiden Multi-Basin Signature Review",
        "",
        "This is a Dongdaemun diagnostic artifact. It does not by itself establish an accepted Dongdaemun output.",
        "",
        "## Basin Summary",
        "",
    ]
    if basin_summary.empty:
        lines.append("- No p5 basin signature rows were available.")
    else:
        display_cols = [
            column
            for column in [
                "candidate_eval_mode",
                "case",
                "seed",
                "candidate_budget",
                "distinct_basin_count",
                "distinct_material_basin_count",
                "distinct_meso_or_macro_material_basin_count",
                "distinct_macro_material_basin_count",
                "full_p5_best_delta_q",
                "full_p5_best_relative_delta_q_ppm",
                "best_basin_is_material",
                "low_roi_positive_count",
                "mean_alignment_error_fraction_vs_baseline",
                "max_alignment_error_nodes_vs_baseline",
                "mean_aligned_changed_support_node_count",
                ]
            if column in basin_summary.columns
        ]
        lines.extend(_markdown_table(basin_summary[display_cols]).splitlines())
    lines.extend(["", "## Coverage", ""])
    if coverage_rows.empty:
        lines.append("- No top-k coverage rows were available.")
    else:
        top_rows = coverage_rows[coverage_rows["top_k"].isin([1, 2, 3, 5])].copy()
        display_cols = [
            column
            for column in [
                "candidate_eval_mode",
                "case",
                "seed",
                "candidate_budget",
                "top_k",
                "best_basin_hit_at_k",
                "best_quality_regret_at_k",
                "distinct_basin_coverage_at_k",
                "distinct_material_basin_coverage_at_k",
            ]
            if column in top_rows.columns
        ]
        lines.extend(_markdown_table(top_rows[display_cols]).splitlines())
    if coarse_rows is not None:
        lines.extend(["", "## Coarse Basin Diagnostic", ""])
        if coarse_rows.empty:
            lines.append("- No basin sketch rows were available for coarse clustering.")
        else:
            display_cols = [
                column
                for column in [
                    "candidate_eval_mode",
                    "case",
                    "seed",
                    "candidate_budget",
                    "coarse_basin_id",
                    "member_candidate_indices",
                    "candidate_count",
                    "exact_basin_count",
                    "material_candidate_count",
                    "best_candidate_index",
                    "best_p5_delta_q",
                    "q_range",
                    "mean_alignment_error_fraction_vs_baseline",
                ]
                if column in coarse_rows.columns
            ]
            lines.extend(_markdown_table(coarse_rows[display_cols]).splitlines())
    if pairwise_rows is not None:
        lines.extend(["", "## Pairwise Basin Matrix", ""])
        if pairwise_rows.empty:
            lines.append("- No pairwise basin matrix rows were available.")
        else:
            iso_count = int(pairwise_rows.get("iso_q_pair", False).map(bool).sum())
            distinct_iso_count = int(
                pairwise_rows.get("partition_distinct_iso_q_pair", False).map(bool).sum()
            )
            lines.append(f"- pairwise rows: {len(pairwise_rows)}")
            lines.append(f"- iso-Q pairs: {iso_count}")
            lines.append(f"- partition-distinct iso-Q pairs: {distinct_iso_count}")
    (output_dir / "leiden_multibasin_signature_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

def analyze_signatures(
    candidates: pd.DataFrame,
    *,
    material_delta_q: float = 1.0,
    material_relative_ppm: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signature_rows = _signature_frame(candidates)
    signature_rows = _mark_material_gain(
        signature_rows,
        material_delta_q=material_delta_q,
        material_relative_ppm=material_relative_ppm,
    )
    basin_rows = build_basin_rows(signature_rows)
    basin_summary = build_basin_summary(signature_rows)
    coverage_rows = build_coverage_rows(signature_rows)
    return basin_rows, basin_summary, coverage_rows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--material-relative-ppm", type=float, default=10.0)
    parser.add_argument("--coarse-endpoint-tau", type=float, default=0.02)
    parser.add_argument("--coarse-support-tau", type=float, default=0.5)
    parser.add_argument("--iso-q-delta", type=float, default=10.0)
    parser.add_argument("--iso-q-relative-ppm", type=float, default=10.0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _read_csvs(input_dir, "candidate_level_rows.csv")
    signature_rows = _signature_frame(candidates)
    signature_rows = _mark_material_gain(
        signature_rows,
        material_delta_q=args.material_delta_q,
        material_relative_ppm=args.material_relative_ppm,
    )
    basin_rows = build_basin_rows(signature_rows)
    basin_summary = build_basin_summary(signature_rows)
    coverage_rows = build_coverage_rows(signature_rows)
    pairwise_rows = build_pairwise_basin_matrix(
        signature_rows,
        coarse_endpoint_tau=args.coarse_endpoint_tau,
        coarse_support_tau=args.coarse_support_tau,
        iso_q_delta=args.iso_q_delta,
        iso_q_relative_ppm=args.iso_q_relative_ppm,
    )
    coarse_rows = build_coarse_basin_rows(signature_rows, pairwise_rows)
    basin_rows.to_csv(output_dir / "leiden_multibasin_basin_rows.csv", index=False)
    basin_summary.to_csv(output_dir / "leiden_multibasin_basin_summary.csv", index=False)
    coverage_rows.to_csv(output_dir / "leiden_multibasin_coverage_curves.csv", index=False)
    pairwise_rows.to_csv(output_dir / "leiden_multibasin_pairwise_basin_matrix.csv", index=False)
    coarse_rows.to_csv(output_dir / "leiden_multibasin_coarse_basin_rows.csv", index=False)
    write_report(output_dir, basin_summary, coverage_rows, pairwise_rows, coarse_rows)
    print(
        {
            "candidate_rows": int(len(candidates)),
            "basin_rows": int(len(basin_rows)),
            "summary_rows": int(len(basin_summary)),
            "pairwise_rows": int(len(pairwise_rows)),
            "coarse_rows": int(len(coarse_rows)),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
