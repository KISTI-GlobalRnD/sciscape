#!/usr/bin/env python3
"""Run a small cross-field Leiden hysteresis work-acceleration monitor.

The monitor is deliberately narrower than the original smoke run. It evaluates
whether an external-grain perturbation reaches the same qf target with less
refinement work on a few additional graph fields. The primary axis is
``k_work``: cumulative ``n_clusters`` observed at ``after_refinement`` phase
checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_leiden_hysteresis_work_acceleration import (  # noqa: E402
    _classify_quality_guard,
    _classify_role,
    _classify_work_saving,
    _score_target_policy,
    _target_policy_specs,
)
from run_leiden_hysteresis_shatter_smoke import (  # noqa: E402
    _candidate_clusters,
    _case_name,
    _load_graph_arrays,
)
from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402


DEFAULT_GRAPH_DIRS = (
    REPO_ROOT / "research/consensus/results/adaptive_refinement/"
    "dongdaemun_safe_fast_layer_comparison/bc_cosine_20260507/graphs/"
    "field30_gcc_emb_full_knn30/seed_11/bc_cosine",
    REPO_ROOT / "research/consensus/results/adaptive_refinement/"
    "dongdaemun_safe_fast_layer_comparison/cc_cosine_20260507/graphs/"
    "field30_gcc_emb_full_knn30/seed_11/cc_cosine",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_work_acceleration_monitor_20260513"
)

QUALITY_TRACE_PATH_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_PATH"
QUALITY_TRACE_RUN_ID_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_RUN_ID"
QUALITY_TRACE_EPOCH_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_EPOCH"
QUALITY_TRACE_TARGET_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_TARGET_MAX_WEIGHT"
TRAJECTORY_TRACE_PATH_ENV = "SCISCAPE_DDM_TRAJECTORY_TRACE_PATH"
TRAJECTORY_TRACE_RUN_ID_ENV = "SCISCAPE_DDM_TRAJECTORY_TRACE_RUN_ID"
TRAJECTORY_TRACE_EPOCH_ENV = "SCISCAPE_DDM_TRAJECTORY_TRACE_EPOCH"

APPROX_POLISH_LABEL_MODES = {
    "localized_label",
    "quotient_label",
    "upper_bound_label",
}
APPROX_POLISH_DEFAULT_POLICY = {
    "localized_label": "localized_top2_then_p5",
    "quotient_label": "quotient_top5_then_p5",
    "upper_bound_label": "ub_shadow_skip_margin_0",
}


def _parse_int_list(value: str) -> list[int]:
    out = [int(part) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one integer")
    return out


def _parse_graph_dirs(value: str | None) -> list[Path]:
    if value is None:
        return [path.resolve() for path in DEFAULT_GRAPH_DIRS]
    return [
        Path(part).expanduser().resolve() for part in value.split(",") if part.strip()
    ]


def _build_monitor_graph(
    graph_dir: Path,
    *,
    probe_only: bool,
) -> tuple[Any, np.ndarray, Any | None]:
    if probe_only:
        node_weights_path = graph_dir / "node_weights.f64.bin"
        node_weights = np.memmap(node_weights_path, dtype=np.float64, mode="r")
        graph = build_leiden_graph(
            edge_path=graph_dir / "int_edges.parquet",
            n_nodes=int(node_weights.shape[0]),
            node_weights_path=node_weights_path,
        )
        return graph, node_weights, None

    arrays = _load_graph_arrays(graph_dir)
    graph = build_leiden_graph(
        edges_src=arrays.src,
        edges_dst=arrays.dst,
        edges_weight=arrays.weight,
        n_nodes=int(arrays.node_weights.shape[0]),
        node_weights=arrays.node_weights,
    )
    return graph, arrays.node_weights, arrays


def _release_memmap_array(array: Any) -> None:
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@contextmanager
def _trace_file_context(
    quality_path: Path, trajectory_path: Path, *, resume: bool
) -> Iterator[None]:
    previous = {
        QUALITY_TRACE_PATH_ENV: os.environ.get(QUALITY_TRACE_PATH_ENV),
        QUALITY_TRACE_EPOCH_ENV: os.environ.get(QUALITY_TRACE_EPOCH_ENV),
        TRAJECTORY_TRACE_PATH_ENV: os.environ.get(TRAJECTORY_TRACE_PATH_ENV),
        TRAJECTORY_TRACE_EPOCH_ENV: os.environ.get(TRAJECTORY_TRACE_EPOCH_ENV),
    }
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        for path in (quality_path, trajectory_path):
            if path.exists():
                path.unlink()
    os.environ[QUALITY_TRACE_PATH_ENV] = str(quality_path)
    os.environ[QUALITY_TRACE_EPOCH_ENV] = uuid.uuid4().hex
    os.environ[TRAJECTORY_TRACE_PATH_ENV] = str(trajectory_path)
    os.environ[TRAJECTORY_TRACE_EPOCH_ENV] = uuid.uuid4().hex
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _trace_run_context(run_id: str, *, target_max_weight: float) -> Iterator[None]:
    previous = {
        QUALITY_TRACE_RUN_ID_ENV: os.environ.get(QUALITY_TRACE_RUN_ID_ENV),
        QUALITY_TRACE_TARGET_ENV: os.environ.get(QUALITY_TRACE_TARGET_ENV),
        TRAJECTORY_TRACE_RUN_ID_ENV: os.environ.get(TRAJECTORY_TRACE_RUN_ID_ENV),
    }
    os.environ[QUALITY_TRACE_RUN_ID_ENV] = run_id
    os.environ[QUALITY_TRACE_TARGET_ENV] = str(float(target_max_weight))
    os.environ[TRAJECTORY_TRACE_RUN_ID_ENV] = run_id
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _trace_disabled_context() -> Iterator[None]:
    trace_keys = (
        QUALITY_TRACE_PATH_ENV,
        QUALITY_TRACE_RUN_ID_ENV,
        QUALITY_TRACE_TARGET_ENV,
        TRAJECTORY_TRACE_PATH_ENV,
        TRAJECTORY_TRACE_RUN_ID_ENV,
    )
    previous = {key: os.environ.get(key) for key in trace_keys}
    for key in trace_keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _append_run_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        existing_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    new_fields = [key for key in sorted(row.keys()) if key not in fieldnames]
    if not new_fields:
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writerow(row)
        return

    fieldnames.extend(new_fields)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
        writer.writerow(row)


def _best_candidate_row(candidate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidate_rows:
        return None
    return max(
        candidate_rows, key=lambda row: float(row.get("post_polish_delta_q", 0.0))
    )


def _policy_rows_by_name(
    policy_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(row.get("policy", "")): row for row in policy_rows}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value)
    except ValueError:
        return False


def _ranked_candidate_row_indices(
    candidate_rows: list[dict[str, Any]],
    *,
    metric: str,
) -> list[int]:
    indexed = list(enumerate(candidate_rows))

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        row_idx, row = item
        value = _finite_float(row.get(metric), math.nan)
        candidate_index = int(row.get("candidate_index", row_idx))
        if math.isfinite(value):
            return (0, -value, candidate_index)
        return (1, 0.0, candidate_index)

    return [row_idx for row_idx, _ in sorted(indexed, key=key)]


def _candidate_baseline_quality(candidate_rows: list[dict[str, Any]]) -> float:
    for row in candidate_rows:
        p5_quality = _finite_float(row.get("p5_quality"), math.nan)
        p5_delta = _finite_float(row.get("p5_delta_q"), math.nan)
        if math.isfinite(p5_quality) and math.isfinite(p5_delta):
            return p5_quality - p5_delta
        p1_quality = _finite_float(row.get("p1_quality"), math.nan)
        p1_delta = _finite_float(row.get("p1_delta_q"), math.nan)
        if math.isfinite(p1_quality) and math.isfinite(p1_delta):
            return p1_quality - p1_delta
    return math.nan


def _build_multifidelity_policy_row_from_candidates(
    *,
    policy: str,
    candidate_rows: list[dict[str, Any]],
    selected_row_indices: list[int],
    full_p5_winner_row_idx: int | None,
    quality_eps: float,
) -> dict[str, Any]:
    selected_rows = [
        candidate_rows[row_idx]
        for row_idx in selected_row_indices
        if 0 <= row_idx < len(candidate_rows)
    ]
    p5_rows = [
        row
        for row in selected_rows
        if math.isfinite(_finite_float(row.get("p5_delta_q"), math.nan))
    ]
    available = bool(selected_rows) and len(p5_rows) == len(selected_rows)
    p1_elapsed_ms = sum(
        _finite_float(row.get("p1_elapsed_ms"), 0.0)
        for row in candidate_rows
        if math.isfinite(_finite_float(row.get("p1_elapsed_ms"), math.nan))
    )
    p5_elapsed_ms = sum(
        _finite_float(row.get("p5_elapsed_ms"), 0.0)
        for row in p5_rows
        if math.isfinite(_finite_float(row.get("p5_elapsed_ms"), math.nan))
    )
    selected: dict[str, Any] | None = None
    if available:
        selected = max(
            p5_rows,
            key=lambda row: (
                _finite_float(row.get("p5_delta_q"), -math.inf),
                -int(row.get("candidate_index", 0)),
            ),
        )
    baseline_quality = _candidate_baseline_quality(candidate_rows)
    selected_candidate_index = -1
    final_delta_q = math.nan
    quality = math.nan
    accepted = False
    matches_full_p5 = False
    if selected is not None:
        selected_candidate_index = int(selected.get("candidate_index", -1))
        final_delta_q = _finite_float(selected.get("p5_delta_q"), math.nan)
        quality = _finite_float(selected.get("p5_quality"), math.nan)
        if math.isfinite(quality) and math.isfinite(baseline_quality):
            accepted = quality >= baseline_quality + quality_eps
        elif math.isfinite(final_delta_q):
            accepted = final_delta_q >= quality_eps
        if full_p5_winner_row_idx is not None and 0 <= full_p5_winner_row_idx < len(
            candidate_rows
        ):
            matches_full_p5 = selected_candidate_index == int(
                candidate_rows[full_p5_winner_row_idx].get("candidate_index", -1)
            )
    return {
        "policy": policy,
        "selected_candidate_index": selected_candidate_index,
        "candidate_count": len(candidate_rows),
        "p1_evaluated": len(candidate_rows),
        "p5_evaluated": len(p5_rows),
        "p1_elapsed_ms": p1_elapsed_ms,
        "p5_elapsed_ms": p5_elapsed_ms,
        "total_elapsed_ms": p1_elapsed_ms + p5_elapsed_ms,
        "final_delta_q": final_delta_q,
        "quality": quality,
        "accepted": accepted,
        "available": available,
        "matches_full_p5": matches_full_p5,
    }


def _ensure_multifidelity_policy_rows(
    policy_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    quality_eps: float = 0.0,
) -> list[dict[str, Any]]:
    """Add diagnostic p1-topN policy rows when older Rust builds omit them."""
    if not candidate_rows:
        return list(policy_rows)
    out = [dict(row) for row in policy_rows]
    present = {str(row.get("policy", "")) for row in out}
    p1_order = _ranked_candidate_row_indices(candidate_rows, metric="p1_delta_q")
    p5_order = [
        idx
        for idx in _ranked_candidate_row_indices(candidate_rows, metric="p5_delta_q")
        if math.isfinite(_finite_float(candidate_rows[idx].get("p5_delta_q"), math.nan))
    ]
    full_p5_winner = p5_order[0] if p5_order else None
    for top_n in (1, 2, 3):
        policy = f"p1_top{top_n}_then_p5"
        if policy in present:
            continue
        out.append(
            _build_multifidelity_policy_row_from_candidates(
                policy=policy,
                candidate_rows=candidate_rows,
                selected_row_indices=p1_order[:top_n],
                full_p5_winner_row_idx=full_p5_winner,
                quality_eps=quality_eps,
            )
        )
        present.add(policy)
    return out


def _best_full_p5_row_idx(candidate_rows: list[dict[str, Any]]) -> int | None:
    p5_order = [
        idx
        for idx in _ranked_candidate_row_indices(candidate_rows, metric="p5_delta_q")
        if math.isfinite(_finite_float(candidate_rows[idx].get("p5_delta_q"), math.nan))
    ]
    return p5_order[0] if p5_order else None


def _build_approx_topk_policy_row_from_candidates(
    *,
    policy: str,
    candidate_rows: list[dict[str, Any]],
    rank_metric: str,
    elapsed_metric: str,
    top_n: int,
    full_p5_winner_row_idx: int | None,
    quality_eps: float,
) -> dict[str, Any]:
    ranked = _ranked_candidate_row_indices(candidate_rows, metric=rank_metric)
    selected_indices = [
        idx
        for idx in ranked
        if math.isfinite(_finite_float(candidate_rows[idx].get(rank_metric), math.nan))
    ][:top_n]
    selected_rows = [candidate_rows[idx] for idx in selected_indices]
    p5_rows = [
        row
        for row in selected_rows
        if math.isfinite(_finite_float(row.get("p5_delta_q"), math.nan))
    ]
    available = bool(selected_rows) and len(p5_rows) == len(selected_rows)
    screen_elapsed_ms = sum(
        _finite_float(row.get(elapsed_metric), 0.0)
        for row in candidate_rows
        if math.isfinite(_finite_float(row.get(elapsed_metric), math.nan))
    )
    p5_elapsed_ms = sum(
        _finite_float(row.get("p5_elapsed_ms"), 0.0)
        for row in p5_rows
        if math.isfinite(_finite_float(row.get("p5_elapsed_ms"), math.nan))
    )
    selected: dict[str, Any] | None = None
    if available:
        selected = max(
            p5_rows,
            key=lambda row: (
                _finite_float(row.get("p5_delta_q"), -math.inf),
                -int(row.get("candidate_index", 0)),
            ),
        )
    baseline_quality = _candidate_baseline_quality(candidate_rows)
    selected_candidate_index = -1
    final_delta_q = math.nan
    quality = math.nan
    accepted = False
    matches_full_p5 = False
    if selected is not None:
        selected_candidate_index = int(selected.get("candidate_index", -1))
        final_delta_q = _finite_float(selected.get("p5_delta_q"), math.nan)
        quality = _finite_float(selected.get("p5_quality"), math.nan)
        if math.isfinite(quality) and math.isfinite(baseline_quality):
            accepted = quality >= baseline_quality + quality_eps
        elif math.isfinite(final_delta_q):
            accepted = final_delta_q >= quality_eps
        if full_p5_winner_row_idx is not None and 0 <= full_p5_winner_row_idx < len(
            candidate_rows
        ):
            matches_full_p5 = selected_candidate_index == int(
                candidate_rows[full_p5_winner_row_idx].get("candidate_index", -1)
            )
    return {
        "policy": policy,
        "selected_candidate_index": selected_candidate_index,
        "candidate_count": len(candidate_rows),
        "p1_evaluated": len(candidate_rows),
        "p5_evaluated": len(p5_rows),
        "p1_elapsed_ms": screen_elapsed_ms,
        "p5_elapsed_ms": p5_elapsed_ms,
        "total_elapsed_ms": screen_elapsed_ms + p5_elapsed_ms,
        "final_delta_q": final_delta_q,
        "quality": quality,
        "accepted": accepted,
        "available": available,
        "matches_full_p5": matches_full_p5,
        "rank_metric": rank_metric,
        "screen_elapsed_metric": elapsed_metric,
    }


def _build_upper_bound_shadow_policy_row(
    *,
    policy: str,
    candidate_rows: list[dict[str, Any]],
    margin: float,
    full_p5_winner_row_idx: int | None,
    quality_eps: float,
) -> dict[str, Any]:
    baseline_quality = _candidate_baseline_quality(candidate_rows)
    screen_elapsed_ms = sum(
        _finite_float(row.get("ub_elapsed_ms"), 0.0)
        for row in candidate_rows
        if math.isfinite(_finite_float(row.get("ub_elapsed_ms"), math.nan))
    )
    evaluated_rows: list[dict[str, Any]] = []
    skipped = 0
    best_seen_delta = -math.inf
    for row in sorted(candidate_rows, key=lambda item: int(item.get("candidate_index", 0))):
        ub_delta = _finite_float(row.get("ub_delta_q"), math.nan)
        p5_delta = _finite_float(row.get("p5_delta_q"), math.nan)
        if evaluated_rows and math.isfinite(ub_delta) and ub_delta + margin < best_seen_delta:
            skipped += 1
            continue
        if math.isfinite(p5_delta):
            evaluated_rows.append(row)
            if p5_delta > best_seen_delta:
                best_seen_delta = p5_delta
    available = bool(evaluated_rows)
    selected = None
    if available:
        selected = max(
            evaluated_rows,
            key=lambda row: (
                _finite_float(row.get("p5_delta_q"), -math.inf),
                -int(row.get("candidate_index", 0)),
            ),
        )
    p5_elapsed_ms = sum(
        _finite_float(row.get("p5_elapsed_ms"), 0.0)
        for row in evaluated_rows
        if math.isfinite(_finite_float(row.get("p5_elapsed_ms"), math.nan))
    )
    selected_candidate_index = -1
    final_delta_q = math.nan
    quality = math.nan
    accepted = False
    matches_full_p5 = False
    if selected is not None:
        selected_candidate_index = int(selected.get("candidate_index", -1))
        final_delta_q = _finite_float(selected.get("p5_delta_q"), math.nan)
        quality = _finite_float(selected.get("p5_quality"), math.nan)
        if math.isfinite(quality) and math.isfinite(baseline_quality):
            accepted = quality >= baseline_quality + quality_eps
        elif math.isfinite(final_delta_q):
            accepted = final_delta_q >= quality_eps
        if full_p5_winner_row_idx is not None and 0 <= full_p5_winner_row_idx < len(
            candidate_rows
        ):
            matches_full_p5 = selected_candidate_index == int(
                candidate_rows[full_p5_winner_row_idx].get("candidate_index", -1)
            )
    return {
        "policy": policy,
        "selected_candidate_index": selected_candidate_index,
        "candidate_count": len(candidate_rows),
        "p1_evaluated": len(candidate_rows),
        "p5_evaluated": len(evaluated_rows),
        "p1_elapsed_ms": screen_elapsed_ms,
        "p5_elapsed_ms": p5_elapsed_ms,
        "total_elapsed_ms": screen_elapsed_ms + p5_elapsed_ms,
        "final_delta_q": final_delta_q,
        "quality": quality,
        "accepted": accepted,
        "available": available,
        "matches_full_p5": matches_full_p5,
        "ub_margin": margin,
        "ub_skipped": skipped,
    }


def _ensure_approx_polish_policy_rows(
    policy_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    quality_eps: float = 0.0,
) -> list[dict[str, Any]]:
    if not candidate_rows:
        return list(policy_rows)
    out = [dict(row) for row in policy_rows]
    present = {str(row.get("policy", "")) for row in out}
    full_p5_winner = _best_full_p5_row_idx(candidate_rows)
    approx_specs = (
        ("localized", "localized_delta_q", "localized_elapsed_ms", (1, 2, 3)),
        ("quotient", "quotient_delta_q", "quotient_elapsed_ms", (1, 3, 5)),
    )
    for prefix, metric, elapsed_metric, top_ns in approx_specs:
        if not any(
            math.isfinite(_finite_float(row.get(metric), math.nan))
            for row in candidate_rows
        ):
            continue
        for top_n in top_ns:
            policy = f"{prefix}_top{top_n}_then_p5"
            if policy in present:
                continue
            out.append(
                _build_approx_topk_policy_row_from_candidates(
                    policy=policy,
                    candidate_rows=candidate_rows,
                    rank_metric=metric,
                    elapsed_metric=elapsed_metric,
                    top_n=top_n,
                    full_p5_winner_row_idx=full_p5_winner,
                    quality_eps=quality_eps,
                )
            )
            present.add(policy)
    if any(
        math.isfinite(_finite_float(row.get("ub_delta_q"), math.nan))
        for row in candidate_rows
    ):
        baseline_quality = _candidate_baseline_quality(candidate_rows)
        ppm_margin = abs(baseline_quality) * 1e-6 if math.isfinite(baseline_quality) else 0.0
        for policy, margin in (
            ("ub_shadow_skip_margin_0", 0.0),
            ("ub_shadow_skip_margin_1ppm", ppm_margin),
        ):
            if policy in present:
                continue
            out.append(
                _build_upper_bound_shadow_policy_row(
                    policy=policy,
                    candidate_rows=candidate_rows,
                    margin=margin,
                    full_p5_winner_row_idx=full_p5_winner,
                    quality_eps=quality_eps,
                )
            )
            present.add(policy)
    return out


def _candidate_row_by_index(
    candidate_rows: list[dict[str, Any]], index: Any
) -> dict[str, Any] | None:
    try:
        selected_index = int(index)
    except (TypeError, ValueError):
        return None
    for row in candidate_rows:
        try:
            if int(row.get("candidate_index", -1)) == selected_index:
                return row
        except (TypeError, ValueError):
            continue
    return None


def _append_rows(path: Path, base: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        _append_run_row(path, {**base, **dict(row)})


def _portfolio_candidate_rows(
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidate_rows:
        return []
    best = _best_candidate_row(candidate_rows)
    best_index = None if best is None else int(best.get("candidate_index", -1))
    out: list[dict[str, Any]] = []
    for row in candidate_rows:
        row_out = dict(row)
        row_out["p5_quality"] = row_out.get("post_polish_quality", math.nan)
        row_out["p5_delta_q"] = row_out.get("post_polish_delta_q", math.nan)
        row_out["p5_elapsed_ms"] = row_out.get("elapsed_ms", math.nan)
        row_out["selected_by_full_p5"] = (
            int(row_out.get("candidate_index", -1)) == best_index
        )
        row_out["selected_by_p1_top1"] = False
        row_out["selected_by_p1_top2"] = False
        return_pre = row_out.get("pre_polish_delta_q", math.nan)
        row_out["pre_delta_q"] = return_pre
        out.append(row_out)
    return out


def _portfolio_policy_row(
    *,
    policy: str,
    probe: Any,
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_count = len(getattr(probe, "candidate_rows", []))
    selected_index = -1 if best is None else int(best.get("candidate_index", -1))
    final_delta_q = (
        math.nan if best is None else _finite_float(best.get("post_polish_delta_q"))
    )
    quality = (
        math.nan if best is None else _finite_float(best.get("post_polish_quality"))
    )
    total_elapsed_ms = _finite_float(getattr(probe, "elapsed_ms", math.nan))
    wall_elapsed_ms = _finite_float(
        getattr(probe, "candidate_eval_wall_elapsed_ms", total_elapsed_ms)
    )
    cpu_sum_elapsed_ms = _finite_float(
        getattr(
            probe,
            "candidate_eval_cpu_sum_elapsed_ms",
            sum(
                _finite_float(row.get("elapsed_ms"), 0.0)
                for row in getattr(probe, "candidate_rows", [])
            ),
        )
    )
    return {
        "policy": policy,
        "selected_candidate_index": selected_index,
        "candidate_count": candidate_count,
        "p1_evaluated": 0,
        "p5_evaluated": candidate_count,
        "p1_elapsed_ms": 0.0,
        "p5_elapsed_ms": cpu_sum_elapsed_ms,
        "total_elapsed_ms": total_elapsed_ms,
        "candidate_eval_wall_elapsed_ms": wall_elapsed_ms,
        "candidate_eval_cpu_sum_elapsed_ms": cpu_sum_elapsed_ms,
        "candidate_eval_parallel": bool(
            getattr(probe, "candidate_eval_parallel", False)
        ),
        "candidate_eval_parallel_speedup": _finite_float(
            getattr(probe, "candidate_eval_parallel_speedup", math.nan)
        ),
        "candidate_eval_parallel_workers": int(
            getattr(probe, "candidate_eval_parallel_workers", 0)
        ),
        "final_delta_q": final_delta_q,
        "quality": quality,
        "accepted": bool(getattr(probe, "accepted", False)),
        "available": bool(candidate_count > 0),
        "matches_full_p5": True,
    }


def _probe_run_row(
    *,
    case: str,
    seed: int,
    candidate_budget: int,
    baseline_quality: float,
    baseline_elapsed: float,
    candidate_selection_elapsed: float,
    probe_elapsed: float,
    probe: Any,
    candidate_eval_mode: str,
    selected_policy: str,
    selected_policy_row: dict[str, Any],
    candidate_clusters: list[int],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": case,
        "seed": seed,
        "candidate_budget": candidate_budget,
        "max_group_candidates": candidate_budget,
        "branch": "probe",
        "run_id": f"{case}|seed={seed}|budget={candidate_budget}|probe",
        "accepted": bool(getattr(probe, "accepted", False)),
        "baseline_quality": float(baseline_quality),
        "quality": _finite_float(
            selected_policy_row.get("quality"),
            float(baseline_quality),
        ),
        "elapsed_sec": probe_elapsed,
        "baseline_elapsed_sec": baseline_elapsed,
        "candidate_cluster_selection_elapsed_sec": candidate_selection_elapsed,
        "candidate_probe_elapsed_sec": probe_elapsed,
        "candidate_probe_eval_wall_elapsed_sec": _finite_float(
            getattr(probe, "candidate_eval_wall_elapsed_ms", math.nan)
        )
        / 1000.0,
        "candidate_probe_cpu_sum_elapsed_sec": _finite_float(
            getattr(probe, "candidate_eval_cpu_sum_elapsed_ms", math.nan)
        )
        / 1000.0,
        "candidate_probe_parallel": bool(
            getattr(probe, "candidate_eval_parallel", False)
        ),
        "candidate_probe_parallel_speedup": _finite_float(
            getattr(probe, "candidate_eval_parallel_speedup", math.nan)
        ),
        "candidate_probe_parallel_workers": int(
            getattr(probe, "candidate_eval_parallel_workers", 0)
        ),
        "candidate_eval_mode": candidate_eval_mode,
        "selected_policy": selected_policy,
        "selected_policy_available": bool(
            selected_policy_row.get("available", best is not None)
        ),
        "process_hwm_mb": _process_hwm_mb(),
        "ranking_strategy": "external_grain_priority_v1",
        "candidate_clusters": json.dumps(candidate_clusters),
    }
    if best is not None:
        row.update(
            {
                "candidate_index": int(best["candidate_index"]),
                "source_cluster": int(best["source_cluster"]),
                "target_cluster": int(best["target_cluster"]),
                "group_kind": best["group_kind"],
                "group_count": int(best["group_count"]),
                "group_weight": float(best["group_weight"]),
                "pre_polish_delta_q": float(
                    best.get("pre_polish_delta_q", best.get("pre_delta_q", math.nan))
                ),
                "expected_post_polish_delta_q": float(
                    best.get("post_polish_delta_q", best.get("p5_delta_q", math.nan))
                ),
                "p1_delta_q": best.get("p1_delta_q", math.nan),
                "p5_delta_q": best.get("p5_delta_q", math.nan),
                "accepted_by_probe_quality": bool(
                    best.get("accepted_by_quality", getattr(probe, "accepted", False))
                ),
                "probe_accepted": bool(getattr(probe, "accepted", False)),
            }
        )
    return row


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _process_hwm_mb() -> float:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return math.nan
    # Linux reports KiB; macOS reports bytes. The monitor runs on Linux in the
    # large-memory workflow, but keep the conversion portable for local tests.
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _compact_membership(membership: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(np.asarray(membership, dtype=np.uint64), return_inverse=True)
    return np.ascontiguousarray(inverse, dtype=np.uint64)


def _reconstruct_external_group(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    membership: np.ndarray,
    node_weights: np.ndarray,
    source_cluster: int,
    target_cluster: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct nodes assigned to a target by strongest external cluster."""
    membership = np.asarray(membership, dtype=np.int64)
    nodes = np.flatnonzero(membership == int(source_cluster))
    if nodes.size == 0:
        return nodes.astype(np.uint32), {
            "reconstructed_group_count": 0,
            "reconstructed_group_weight": 0.0,
            "reconstruction_status": "empty_source",
        }
    source_mask = membership == int(source_cluster)
    src_incident = source_mask[np.asarray(src, dtype=np.int64)]
    dst_incident = source_mask[np.asarray(dst, dtype=np.int64)]
    incident_node = np.concatenate(
        [
            np.asarray(src, dtype=np.uint32)[src_incident],
            np.asarray(dst, dtype=np.uint32)[dst_incident],
        ]
    )
    incident_nbr = np.concatenate(
        [
            np.asarray(dst, dtype=np.uint32)[src_incident],
            np.asarray(src, dtype=np.uint32)[dst_incident],
        ]
    )
    incident_weight = np.concatenate(
        [
            np.asarray(weight, dtype=np.float64)[src_incident],
            np.asarray(weight, dtype=np.float64)[dst_incident],
        ]
    )
    by_node: dict[int, dict[int, float]] = {int(node): {} for node in nodes}
    for node, nbr, edge_weight in zip(
        incident_node, incident_nbr, incident_weight, strict=False
    ):
        nbr_cluster = int(membership[int(nbr)])
        if nbr_cluster == int(source_cluster):
            continue
        node_targets = by_node[int(node)]
        node_targets[nbr_cluster] = node_targets.get(nbr_cluster, 0.0) + float(
            edge_weight
        )
    selected: list[int] = []
    for node in nodes:
        best_target = -1
        best_weight = 0.0
        for nbr_cluster, total_weight in by_node[int(node)].items():
            if total_weight > best_weight:
                best_weight = total_weight
                best_target = nbr_cluster
        if best_target == int(target_cluster):
            selected.append(int(node))

    selected_nodes = np.asarray(selected, dtype=np.uint32)
    selected_weight = (
        float(np.asarray(node_weights)[selected_nodes].sum())
        if selected_nodes.size
        else 0.0
    )
    return selected_nodes, {
        "reconstructed_group_count": int(selected_nodes.size),
        "reconstructed_group_weight": selected_weight,
        "reconstruction_status": "ok",
    }


def _extract_phase_checkpoints(trajectory_path: Path, out_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in _read_jsonl(trajectory_path):
        if event.get("event") != "phase_checkpoint":
            continue
        rows.append(event)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    return frame


LOCAL_MERGE_SUMMARY_COLUMNS = [
    "run_id",
    "iteration",
    "depth",
    "parent_id",
    "parent_visit_index",
    "source",
    "parent_size",
    "parent_weight",
    "decision_count",
    "low_margin_decision_count",
    "changed_decision_count",
    "min_margin",
    "p10_margin",
    "p50_margin",
    "selected_child_count",
    "largest_child_fraction",
]


LOCAL_MERGE_PARENT_SUMMARY_COLUMNS = [
    "case",
    "seed",
    "candidate_budget",
    "run_id",
    "branch",
    "first_divergence_iteration",
    "n_parent_rows",
    "total_decision_count",
    "total_low_margin_decision_count",
    "total_changed_decision_count",
    "min_margin_min",
    "p10_margin_min",
    "largest_child_fraction_max",
    "top_low_margin_parent_ids",
    "top_changed_parent_ids",
    "top_decision_parent_ids",
]


FIRST_DIVERGENCE_COLUMNS = [
    "case",
    "seed",
    "candidate_budget",
    "group_size_class",
    "run_id_extra",
    "run_id_perturb",
    "first_divergence_phase",
    "first_divergence_iteration",
    "first_divergence_depth",
    "extra_membership_hash",
    "perturb_membership_hash",
    "extra_n_clusters",
    "perturb_n_clusters",
    "extra_quality",
    "perturb_quality",
    "extra_moved_nodes_iter",
    "perturb_moved_nodes_iter",
    "local_merge_parent_ids_sample",
    "final_quality_delta",
]


def _extract_local_merge_summaries(
    trajectory_path: Path,
    out_path: Path,
    run_iteration_filter: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in _read_jsonl(trajectory_path):
        if event.get("event") != "local_merge_margin_summary":
            continue
        if run_iteration_filter is not None:
            try:
                key = (str(event.get("run_id") or ""), int(event.get("iteration") or 0))
            except (TypeError, ValueError):
                continue
            if key not in run_iteration_filter:
                continue
        rows.append(
            {column: event.get(column, "") for column in LOCAL_MERGE_SUMMARY_COLUMNS}
        )
    frame = pd.DataFrame(rows, columns=LOCAL_MERGE_SUMMARY_COLUMNS)
    frame.to_csv(out_path, index=False)
    return frame


def _float_event_value(event: dict[str, Any], key: str) -> float:
    try:
        return float(event.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _top_parent_ids(
    parent_totals: dict[str, dict[str, float]], metric: str, limit: int = 8
) -> str:
    ranked = sorted(
        parent_totals.items(),
        key=lambda item: (-float(item[1].get(metric, 0.0)), item[0]),
    )
    return ",".join(
        parent_id
        for parent_id, values in ranked[:limit]
        if values.get(metric, 0.0) > 0.0
    )


def _local_merge_contexts(
    first_divergence: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, Any]]:
    contexts: dict[tuple[str, int], dict[str, Any]] = {}
    if first_divergence.empty:
        return contexts
    for _, row in first_divergence.iterrows():
        iteration = row.get("first_divergence_iteration")
        if pd.isna(iteration) or iteration == "":
            continue
        iteration_value = int(iteration)
        for branch, run_id_column in (
            ("extra", "run_id_extra"),
            ("perturb", "run_id_perturb"),
        ):
            run_id = str(row.get(run_id_column, ""))
            if not run_id:
                continue
            contexts[(run_id, iteration_value)] = {
                "case": row.get("case", ""),
                "seed": row.get("seed", ""),
                "candidate_budget": row.get("candidate_budget", ""),
                "run_id": run_id,
                "branch": branch,
                "first_divergence_iteration": iteration_value,
            }
    return contexts


def _extract_compact_local_merge_parent_summary(
    trajectory_path: Path,
    first_divergence: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    contexts = _local_merge_contexts(first_divergence)
    aggregates: dict[tuple[str, int], dict[str, Any]] = {}
    for key, context in contexts.items():
        aggregates[key] = {
            **context,
            "n_parent_rows": 0,
            "total_decision_count": 0.0,
            "total_low_margin_decision_count": 0.0,
            "total_changed_decision_count": 0.0,
            "min_margin_min": math.nan,
            "p10_margin_min": math.nan,
            "largest_child_fraction_max": math.nan,
            "parent_totals": {},
        }
    if aggregates:
        for event in _read_jsonl(trajectory_path):
            if event.get("event") != "local_merge_margin_summary":
                continue
            try:
                key = (str(event.get("run_id") or ""), int(event.get("iteration") or 0))
            except (TypeError, ValueError):
                continue
            row = aggregates.get(key)
            if row is None:
                continue
            decision = _float_event_value(event, "decision_count")
            low = _float_event_value(event, "low_margin_decision_count")
            changed = _float_event_value(event, "changed_decision_count")
            min_margin = _float_event_value(event, "min_margin")
            p10_margin = _float_event_value(event, "p10_margin")
            largest_child = _float_event_value(event, "largest_child_fraction")
            row["n_parent_rows"] += 1
            row["total_decision_count"] += decision
            row["total_low_margin_decision_count"] += low
            row["total_changed_decision_count"] += changed
            row["min_margin_min"] = (
                min_margin
                if not math.isfinite(row["min_margin_min"])
                else min(row["min_margin_min"], min_margin)
            )
            row["p10_margin_min"] = (
                p10_margin
                if not math.isfinite(row["p10_margin_min"])
                else min(row["p10_margin_min"], p10_margin)
            )
            row["largest_child_fraction_max"] = (
                largest_child
                if not math.isfinite(row["largest_child_fraction_max"])
                else max(row["largest_child_fraction_max"], largest_child)
            )
            parent_id = str(event.get("parent_id", ""))
            parent_totals = row["parent_totals"].setdefault(
                parent_id,
                {
                    "decision_count": 0.0,
                    "low_margin_decision_count": 0.0,
                    "changed_decision_count": 0.0,
                },
            )
            parent_totals["decision_count"] += decision
            parent_totals["low_margin_decision_count"] += low
            parent_totals["changed_decision_count"] += changed

    rows: list[dict[str, Any]] = []
    for key in sorted(aggregates):
        row = aggregates[key]
        parent_totals = row.pop("parent_totals")
        row["top_low_margin_parent_ids"] = _top_parent_ids(
            parent_totals, "low_margin_decision_count"
        )
        row["top_changed_parent_ids"] = _top_parent_ids(
            parent_totals, "changed_decision_count"
        )
        row["top_decision_parent_ids"] = _top_parent_ids(
            parent_totals, "decision_count"
        )
        rows.append(
            {
                column: row.get(column, "")
                for column in LOCAL_MERGE_PARENT_SUMMARY_COLUMNS
            }
        )
    frame = pd.DataFrame(rows, columns=LOCAL_MERGE_PARENT_SUMMARY_COLUMNS)
    frame.to_csv(out_path, index=False)
    return frame


def _group_size_class(group_count: Any) -> str:
    try:
        count = int(group_count)
    except (TypeError, ValueError):
        return "unknown"
    if count == 1:
        return "single_node"
    if 2 <= count <= 12:
        return "small_group_2_12"
    return "larger_group"


def _candidate_budget_from_row(row: pd.Series) -> int:
    if "candidate_budget" in row and not pd.isna(row["candidate_budget"]):
        return int(row["candidate_budget"])
    if "max_group_candidates" in row and not pd.isna(row["max_group_candidates"]):
        return int(row["max_group_candidates"])
    return 0


def _point_at_iteration(
    points: pd.DataFrame, run_id: str, iteration: Any
) -> pd.Series | None:
    if points.empty:
        return None
    try:
        iteration_value = int(iteration)
    except (TypeError, ValueError):
        return None
    rows = points[(points["run_id"] == run_id) & (points["t_i"] == iteration_value)]
    if rows.empty:
        return None
    return rows.sort_values(["t_k_work", "t_k_phase"]).iloc[-1]


def _local_merge_parent_sample(
    local_merge_frame: pd.DataFrame, run_ids: tuple[str, str], iteration: Any
) -> str:
    if local_merge_frame.empty:
        return ""
    try:
        iteration_value = int(iteration)
    except (TypeError, ValueError):
        return ""
    subset = local_merge_frame[
        local_merge_frame["run_id"].isin(run_ids)
        & (local_merge_frame["iteration"].astype(str) == str(iteration_value))
    ]
    if subset.empty or "parent_id" not in subset:
        return ""
    parent_ids = [
        str(value) for value in subset["parent_id"].dropna().astype(str).unique()[:8]
    ]
    return ",".join(parent_ids)


def _build_first_divergence_rows(
    *,
    phase_frame: pd.DataFrame,
    points: pd.DataFrame,
    run_rows: pd.DataFrame,
    local_merge_frame: pd.DataFrame,
) -> pd.DataFrame:
    if phase_frame.empty or run_rows.empty:
        return pd.DataFrame(columns=FIRST_DIVERGENCE_COLUMNS)
    if "candidate_budget" not in run_rows:
        run_rows = run_rows.copy()
        run_rows["candidate_budget"] = 0
    phase_by_run = {
        str(run_id): group.sort_values(["iteration", "depth", "phase"]).reset_index(
            drop=True
        )
        for run_id, group in phase_frame.groupby("run_id", dropna=False)
    }
    rows: list[dict[str, Any]] = []
    for (case, seed, candidate_budget), branch_rows in run_rows.groupby(
        ["case", "seed", "candidate_budget"], dropna=False
    ):
        extra_rows = branch_rows[branch_rows["branch"] == "extra"]
        perturb_rows = branch_rows[branch_rows["branch"] == "perturb"]
        if extra_rows.empty or perturb_rows.empty:
            continue
        extra_meta = extra_rows.iloc[0]
        perturb_meta = perturb_rows.iloc[0]
        extra_run_id = str(extra_meta.get("run_id", ""))
        perturb_run_id = str(perturb_meta.get("run_id", ""))
        extra_phase = phase_by_run.get(extra_run_id, pd.DataFrame())
        perturb_phase = phase_by_run.get(perturb_run_id, pd.DataFrame())
        final_extra = extra_phase.iloc[-1] if not extra_phase.empty else {}
        final_perturb = perturb_phase.iloc[-1] if not perturb_phase.empty else {}
        row: dict[str, Any] = {
            "case": case,
            "seed": int(seed),
            "candidate_budget": int(candidate_budget),
            "group_size_class": _group_size_class(perturb_meta.get("group_count", "")),
            "run_id_extra": extra_run_id,
            "run_id_perturb": perturb_run_id,
            "first_divergence_phase": "",
            "first_divergence_iteration": "",
            "first_divergence_depth": "",
            "extra_membership_hash": "",
            "perturb_membership_hash": "",
            "extra_n_clusters": "",
            "perturb_n_clusters": "",
            "extra_quality": final_extra.get("quality", "")
            if isinstance(final_extra, pd.Series)
            else "",
            "perturb_quality": final_perturb.get("quality", "")
            if isinstance(final_perturb, pd.Series)
            else "",
            "extra_moved_nodes_iter": "",
            "perturb_moved_nodes_iter": "",
            "local_merge_parent_ids_sample": "",
            "final_quality_delta": float(perturb_meta.get("quality", 0.0))
            - float(extra_meta.get("quality", 0.0)),
        }
        for _, (extra_event, perturb_event) in enumerate(
            zip(extra_phase.to_dict("records"), perturb_phase.to_dict("records"))
        ):
            if extra_event.get("membership_hash") == perturb_event.get(
                "membership_hash"
            ):
                continue
            iteration = extra_event.get("iteration", "")
            extra_point = _point_at_iteration(points, extra_run_id, iteration)
            perturb_point = _point_at_iteration(points, perturb_run_id, iteration)
            row.update(
                {
                    "first_divergence_phase": extra_event.get("phase", ""),
                    "first_divergence_iteration": iteration,
                    "first_divergence_depth": extra_event.get("depth", ""),
                    "extra_membership_hash": extra_event.get("membership_hash", ""),
                    "perturb_membership_hash": perturb_event.get("membership_hash", ""),
                    "extra_n_clusters": extra_event.get("n_clusters", ""),
                    "perturb_n_clusters": perturb_event.get("n_clusters", ""),
                    "extra_quality": extra_event.get("quality", ""),
                    "perturb_quality": perturb_event.get("quality", ""),
                    "extra_moved_nodes_iter": ""
                    if extra_point is None
                    else int(extra_point.get("moved_nodes", 0)),
                    "perturb_moved_nodes_iter": ""
                    if perturb_point is None
                    else int(perturb_point.get("moved_nodes", 0)),
                    "local_merge_parent_ids_sample": _local_merge_parent_sample(
                        local_merge_frame,
                        (extra_run_id, perturb_run_id),
                        iteration,
                    ),
                }
            )
            break
        rows.append(row)
    return pd.DataFrame(rows, columns=FIRST_DIVERGENCE_COLUMNS)


def _quality_points(
    quality_path: Path,
    phase_frame: pd.DataFrame,
    baseline_quality_by_run_id: dict[str, float] | None = None,
) -> pd.DataFrame:
    quality_rows = [
        event
        for event in _read_jsonl(quality_path)
        if event.get("event") == "quality_checkpoint"
        and event.get("phase") in {"start", "after_iteration", "final"}
    ]
    quality = pd.DataFrame(quality_rows)
    if quality.empty:
        return pd.DataFrame()

    if phase_frame.empty:
        phase_summary = pd.DataFrame(
            columns=[
                "run_id",
                "iteration",
                "k_phase_iter",
                "k_work_iter_refined_clusters",
            ]
        )
    else:
        phase_summary = (
            phase_frame[phase_frame["phase"] == "after_refinement"]
            .groupby(["run_id", "iteration"], dropna=False)
            .agg(
                k_phase_iter=("phase", "size"),
                k_work_iter_refined_clusters=("n_clusters", "sum"),
                max_refined_clusters_this_i=("n_clusters", "max"),
            )
            .reset_index()
        )

    rows: list[dict[str, Any]] = []
    baseline_quality_by_run_id = baseline_quality_by_run_id or {}
    for run_id, group in quality.groupby("run_id", dropna=False):
        run = group.sort_values(["checkpoint_index", "iteration"]).copy()
        trace_start_quality = float(run.iloc[0]["quality"])
        baseline_quality = float(
            baseline_quality_by_run_id.get(str(run_id), trace_start_quality)
        )
        k_phase_cum = 0
        k_work_cum = 0
        phase_by_iteration = {
            int(row["iteration"]): row
            for _, row in phase_summary[phase_summary["run_id"] == run_id].iterrows()
        }
        seen_iterations: set[int] = set()
        for _, event in run.iterrows():
            phase = str(event["phase"])
            iteration = int(event["iteration"])
            if phase == "final":
                continue
            if phase == "start":
                k_phase_iter = 0
                k_work_iter = 0
                max_refined = 0
            else:
                phase_row = phase_by_iteration.get(iteration)
                k_phase_iter = (
                    int(phase_row["k_phase_iter"]) if phase_row is not None else 0
                )
                k_work_iter = (
                    int(phase_row["k_work_iter_refined_clusters"])
                    if phase_row is not None
                    else 0
                )
                max_refined = (
                    int(phase_row["max_refined_clusters_this_i"])
                    if phase_row is not None
                    else 0
                )
                if iteration not in seen_iterations:
                    k_phase_cum += k_phase_iter
                    k_work_cum += k_work_iter
                    seen_iterations.add(iteration)
            quality_value = float(event["quality"])
            qf_delta = quality_value - baseline_quality
            rows.append(
                {
                    "run_id": run_id,
                    "branch": str(run_id).split("|")[-1].split(":")[0],
                    "t_i": iteration,
                    "t_k_phase": k_phase_cum,
                    "t_k_work": k_work_cum,
                    "t_label": f"({iteration},{k_phase_cum},{k_work_cum})",
                    "k_phase_iter": k_phase_iter,
                    "k_work_iter_refined_clusters": k_work_iter,
                    "max_refined_clusters_this_i": max_refined,
                    "quality": quality_value,
                    "trace_start_quality": trace_start_quality,
                    "baseline_quality": baseline_quality,
                    "qf_delta": qf_delta,
                    "qf_delta_ppm": qf_delta / baseline_quality * 1_000_000.0
                    if baseline_quality
                    else 0.0,
                    "moved_nodes": int(event.get("moved_nodes", 0)),
                    "n_clusters": int(event.get("n_clusters", 0)),
                }
            )
    return pd.DataFrame(rows)


def _interp_ppm_at_k_work(points: pd.DataFrame, k_work: float) -> float:
    branch = points.sort_values("t_k_work").reset_index(drop=True)
    if branch.empty or not math.isfinite(k_work):
        return math.nan
    if k_work <= float(branch.iloc[0]["t_k_work"]):
        return float(branch.iloc[0]["qf_delta_ppm"])
    for idx in range(1, len(branch)):
        prev = branch.iloc[idx - 1]
        curr = branch.iloc[idx]
        prev_k = float(prev["t_k_work"])
        curr_k = float(curr["t_k_work"])
        if k_work <= curr_k:
            span = curr_k - prev_k
            frac = 1.0 if span == 0.0 else (k_work - prev_k) / span
            return float(prev["qf_delta_ppm"]) + frac * (
                float(curr["qf_delta_ppm"]) - float(prev["qf_delta_ppm"])
            )
    return float(branch.iloc[-1]["qf_delta_ppm"])


def _scorecard(points: pd.DataFrame, run_rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    if run_rows.empty or "branch" not in run_rows:
        run_rows = pd.DataFrame(columns=["case", "seed", "candidate_budget", "branch"])
    run_meta = {
        (row["case"], int(row["seed"]), _candidate_budget_from_row(row)): row
        for _, row in run_rows[run_rows["branch"] == "perturb"].iterrows()
    }
    extra_meta = {
        (row["case"], int(row["seed"]), _candidate_budget_from_row(row)): row
        for _, row in run_rows[run_rows["branch"] == "extra"].iterrows()
    }
    group_cols = ["case", "seed", "candidate_budget"]
    if "candidate_budget" not in points:
        points = points.copy()
        points["candidate_budget"] = 0
    for (case, seed, candidate_budget), group in points.groupby(group_cols):
        extra = group[group["branch"] == "extra"]
        perturb = group[group["branch"] == "perturb"]
        if extra.empty or perturb.empty:
            continue
        extra_final = extra.sort_values("t_i").iloc[-1]
        perturb_final = perturb.sort_values("t_i").iloc[-1]
        extra_final_ppm = float(extra_final["qf_delta_ppm"])
        perturb_final_ppm = float(perturb_final["qf_delta_ppm"])
        extra_at_perturb = _interp_ppm_at_k_work(
            extra, float(perturb_final["t_k_work"])
        )
        same_work_adv = perturb_final_ppm - extra_at_perturb
        quality_class = _classify_quality_guard(same_work_adv, math.nan)
        key = (case, int(seed), int(candidate_budget))
        meta = run_meta.get(key, {})
        extra_run = extra_meta.get(key, {})
        operational_extra_elapsed = float(extra_run.get("elapsed_sec", math.nan))
        candidate_selection_elapsed = float(
            meta.get("candidate_cluster_selection_elapsed_sec", math.nan)
        )
        candidate_probe_elapsed = float(
            meta.get("candidate_probe_elapsed_sec", math.nan)
        )
        candidate_probe_eval_wall_elapsed = float(
            meta.get("candidate_probe_eval_wall_elapsed_sec", math.nan)
        )
        candidate_probe_cpu_sum_elapsed = float(
            meta.get("candidate_probe_cpu_sum_elapsed_sec", math.nan)
        )
        candidate_probe_parallel_speedup = float(
            meta.get("candidate_probe_parallel_speedup", math.nan)
        )
        operational_perturb_elapsed = (
            candidate_selection_elapsed + candidate_probe_elapsed
        )
        net_elapsed_delta = operational_extra_elapsed - operational_perturb_elapsed
        net_elapsed_saving_pct = (
            net_elapsed_delta / operational_extra_elapsed * 100.0
            if operational_extra_elapsed > 0.0
            else math.nan
        )
        group_count = meta.get("group_count", "")
        base = {
            "case": case,
            "seed": int(seed),
            "candidate_budget": int(candidate_budget),
            "ranking_strategy": meta.get(
                "ranking_strategy", "external_grain_priority_v1"
            ),
            "extra_final_ppm": extra_final_ppm,
            "perturb_final_ppm": perturb_final_ppm,
            "same_k_work_advantage_ppm": same_work_adv,
            "final_perturb_minus_extra_ppm": perturb_final_ppm - extra_final_ppm,
            "extra_final_t": extra_final["t_label"],
            "perturb_final_t": perturb_final["t_label"],
            "source_cluster": meta.get("source_cluster", ""),
            "target_cluster": meta.get("target_cluster", ""),
            "group_kind": meta.get("group_kind", ""),
            "group_count": group_count,
            "group_size_class": _group_size_class(group_count),
            "group_weight": meta.get("group_weight", ""),
            "reconstructed_group_count": meta.get("reconstructed_group_count", ""),
            "reconstructed_group_weight": meta.get("reconstructed_group_weight", ""),
            "reconstruction_status": meta.get("reconstruction_status", ""),
            "candidate_cluster_selection_elapsed_sec": candidate_selection_elapsed,
            "candidate_probe_elapsed_sec": candidate_probe_elapsed,
            "candidate_probe_eval_wall_elapsed_sec": candidate_probe_eval_wall_elapsed,
            "candidate_probe_cpu_sum_elapsed_sec": candidate_probe_cpu_sum_elapsed,
            "candidate_probe_parallel": meta.get("candidate_probe_parallel", ""),
            "candidate_probe_parallel_speedup": candidate_probe_parallel_speedup,
            "candidate_probe_parallel_workers": meta.get(
                "candidate_probe_parallel_workers", ""
            ),
            "process_hwm_mb": float(meta.get("process_hwm_mb", math.nan)),
            "extra_polish_elapsed_sec": operational_extra_elapsed,
            "perturb_trace_polish_elapsed_sec": float(
                meta.get("elapsed_sec", math.nan)
            ),
            "operational_extra_elapsed_sec": operational_extra_elapsed,
            "operational_perturb_elapsed_sec": operational_perturb_elapsed,
            "net_elapsed_delta_sec": net_elapsed_delta,
            "net_elapsed_saving_pct": net_elapsed_saving_pct,
            "analysis_trace_overhead_sec": float(meta.get("elapsed_sec", math.nan)),
            "quality_guard_class": quality_class,
        }
        for spec in _target_policy_specs(extra_final_ppm, perturb_final_ppm):
            scored = _score_target_policy(
                extra_points=extra,
                perturb_points=perturb,
                target_policy=str(spec["target_policy"]),
                target_ppm=float(spec["target_ppm"]),
                target_note=str(spec["target_note"]),
            )
            work_class = _classify_work_saving(float(scored["k_work_saving_pct"]))
            role = _classify_role(work_class, quality_class)
            out.append(
                {
                    **base,
                    **scored,
                    "work_speed_class": work_class,
                    "acceleration_role": role,
                }
            )
    return pd.DataFrame(out).sort_values(
        ["case", "seed", "candidate_budget", "target_policy"]
    )


def _plot_curves(points: pd.DataFrame, scorecard: pd.DataFrame, out_path: Path) -> None:
    if scorecard.empty:
        return
    plot_rows = scorecard[scorecard["target_policy"] == "matched_min"].copy()
    if plot_rows.empty:
        plot_rows = scorecard.drop_duplicates(
            ["case", "seed", "candidate_budget"]
        ).copy()
    n = len(plot_rows)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.6 * cols, 3.7 * rows), squeeze=False
    )
    axes_flat = axes.ravel()
    for ax, (_, score) in zip(axes_flat, plot_rows.iterrows(), strict=False):
        subset = points[
            (points["case"] == score["case"])
            & (points["seed"] == score["seed"])
            & (points["candidate_budget"] == score["candidate_budget"])
        ]
        for branch, color in (("extra", "#4C78A8"), ("perturb", "#F58518")):
            branch_points = subset[subset["branch"] == branch].sort_values("t_k_work")
            ax.plot(
                branch_points["t_k_work"],
                branch_points["qf_delta_ppm"],
                marker="o",
                linewidth=1.8,
                color=color,
                label=branch,
            )
            for _, point in branch_points.iterrows():
                ax.annotate(
                    str(int(point["t_i"])),
                    (point["t_k_work"], point["qf_delta_ppm"]),
                    fontsize=7,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        ax.axhline(
            float(score["common_target_ppm"]),
            color="#666666",
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_title(
            f"{score['case']} seed {int(score['seed'])} b{int(score['candidate_budget'])}",
            fontsize=9,
        )
        ax.set_xlabel("k_work")
        ax.set_ylabel("qf delta ppm")
        ax.grid(alpha=0.25)
    for ax in axes_flat[n:]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_report(out_dir: Path, scorecard: pd.DataFrame) -> None:
    if scorecard.empty or "target_policy" not in scorecard:
        report_rows = pd.DataFrame()
    else:
        report_rows = scorecard[
            scorecard["target_policy"].isin(["extra_p5_final", "baseline_plus_25ppm"])
        ].copy()
        if report_rows.empty:
            report_rows = scorecard.copy()
    lines = [
        "# Leiden Hysteresis Work Acceleration Monitor",
        "",
        "Small cross-field monitor using qf ppm versus cumulative refinement `k_work`, with candidate-evaluation elapsed cost separated from analysis trace replay.",
        "",
        "## Scorecard",
        "",
        "| case | seed | budget | target | target ppm | tau status | k_work saving % | net elapsed saving % | same-k qf adv ppm | group | role |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in report_rows.iterrows():
        tau_status = (
            f"{row.get('extra_tau_status', '')}/{row.get('perturb_tau_status', '')}"
        )
        lines.append(
            "| {case} | {seed} | {budget} | {policy} | {target:.1f} | {tau} | {saving:.1f} | {elapsed:.1f} | {same:.1f} | {group} | {role} |".format(
                case=row["case"],
                seed=int(row["seed"]),
                budget=int(row["candidate_budget"]),
                policy=row["target_policy"],
                target=float(row["target_ppm"]),
                tau=tau_status,
                saving=float(row["k_work_saving_pct"])
                if pd.notna(row["k_work_saving_pct"])
                else math.nan,
                elapsed=float(row["net_elapsed_saving_pct"])
                if pd.notna(row["net_elapsed_saving_pct"])
                else math.nan,
                same=float(row["same_k_work_advantage_ppm"]),
                group=row.get("group_count", ""),
                role=row["acceleration_role"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `k_work saving %` is target-policy specific; `matched_min` is diagnostic and branch-biased.",
            "- `net elapsed saving %` uses candidate selection + probe cost for perturb and excludes analysis-only perturb trace replay.",
            "- `did_not_reach_target` rows should not be read as successful acceleration even if another target policy is positive.",
            "- This monitor does not run long-polish guards by default; promoted rows still need p20 checks.",
            "- Raw trajectory traces are deleted by default after extracting phase checkpoints.",
        ]
    )
    (out_dir / "work_acceleration_monitor_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _postprocess_outputs(
    *,
    out_dir: Path,
    quality_trace_path: Path,
    trajectory_trace_path: Path,
    phase_checkpoint_path: Path,
    local_merge_summary_path: Path,
    local_merge_parent_summary_path: Path,
    first_divergence_path: Path,
    run_rows_path: Path,
    qf_points_path: Path,
    scorecard_path: Path,
    curve_path: Path,
    local_merge_summary_mode: str,
    keep_raw_trajectory: bool,
) -> None:
    phase_frame = _extract_phase_checkpoints(
        trajectory_trace_path, phase_checkpoint_path
    )
    run_rows = pd.read_csv(run_rows_path) if run_rows_path.exists() else pd.DataFrame()
    baseline_by_run_id = {
        str(row["run_id"]): float(row["baseline_quality"])
        for _, row in run_rows.iterrows()
        if "run_id" in run_rows and pd.notna(row.get("run_id"))
    }
    points = _quality_points(quality_trace_path, phase_frame, baseline_by_run_id)
    if not points.empty:
        parsed = points["run_id"].str.extract(
            r"^(?P<case>.+)\|seed=(?P<seed>\d+)(?:\|budget=(?P<candidate_budget>\d+))?\|(?P<branch>[^|]+)$"
        )
        valid = parsed["seed"].notna()
        points = points[valid].copy()
        parsed = parsed[valid]
        points["case"] = parsed["case"]
        points["seed"] = parsed["seed"].astype(int)
        points["candidate_budget"] = parsed["candidate_budget"].fillna(0).astype(int)
        points["branch"] = parsed["branch"]
    points.to_csv(qf_points_path, index=False)
    scorecard = _scorecard(points, run_rows) if not points.empty else pd.DataFrame()
    scorecard.to_csv(scorecard_path, index=False)
    first_divergence_without_parents = _build_first_divergence_rows(
        phase_frame=phase_frame,
        points=points,
        run_rows=run_rows,
        local_merge_frame=pd.DataFrame(columns=LOCAL_MERGE_SUMMARY_COLUMNS),
    )
    local_merge_filter: set[tuple[str, int]] = set()
    for _, row in first_divergence_without_parents.iterrows():
        if (
            pd.isna(row.get("first_divergence_iteration"))
            or row.get("first_divergence_iteration") == ""
        ):
            continue
        iteration = int(row["first_divergence_iteration"])
        local_merge_filter.add((str(row["run_id_extra"]), iteration))
        local_merge_filter.add((str(row["run_id_perturb"]), iteration))
    if local_merge_summary_mode == "full":
        local_merge_frame = _extract_local_merge_summaries(
            trajectory_trace_path,
            local_merge_summary_path,
            None,
        )
    elif local_merge_summary_mode == "focused":
        local_merge_frame = _extract_local_merge_summaries(
            trajectory_trace_path,
            local_merge_summary_path,
            local_merge_filter,
        )
    else:
        local_merge_frame = pd.DataFrame(columns=LOCAL_MERGE_SUMMARY_COLUMNS)
        pd.DataFrame(columns=LOCAL_MERGE_SUMMARY_COLUMNS).to_csv(
            local_merge_summary_path,
            index=False,
        )
    _extract_compact_local_merge_parent_summary(
        trajectory_trace_path,
        first_divergence_without_parents,
        local_merge_parent_summary_path,
    )
    first_divergence = _build_first_divergence_rows(
        phase_frame=phase_frame,
        points=points,
        run_rows=run_rows,
        local_merge_frame=local_merge_frame,
    )
    first_divergence.to_csv(first_divergence_path, index=False)
    _plot_curves(points, scorecard, curve_path)
    _write_report(out_dir, scorecard)

    if not keep_raw_trajectory and trajectory_trace_path.exists():
        trajectory_trace_path.unlink()


def run_monitor(args: argparse.Namespace) -> None:
    if args.candidate_eval_mode in APPROX_POLISH_LABEL_MODES and not args.probe_only:
        raise ValueError(
            "approximate polish label modes are shadow diagnostics and require --probe-only"
        )
    parallel_workers = int(getattr(args, "parallel_candidate_workers", 0) or 0)
    if parallel_workers > 0:
        os.environ["RAYON_NUM_THREADS"] = str(parallel_workers)
        os.environ["SCISCAPE_PARALLEL_CANDIDATE_WORKER_LIMIT"] = str(parallel_workers)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_trace_path = out_dir / "quality_trace.jsonl"
    trajectory_trace_path = out_dir / "trajectory_trace_raw.jsonl"
    phase_checkpoint_path = out_dir / "trajectory_phase_checkpoints.csv"
    local_merge_summary_path = out_dir / "trajectory_local_merge_summaries.csv"
    local_merge_parent_summary_path = out_dir / "first_divergence_parent_summary.csv"
    first_divergence_path = out_dir / "first_divergence_rows.csv"
    run_rows_path = out_dir / "monitor_run_rows.csv"
    candidate_level_path = out_dir / "candidate_level_rows.csv"
    policy_comparison_path = out_dir / "policy_comparison_rows.csv"
    qf_points_path = out_dir / "qf_i_k_points.csv"
    scorecard_path = out_dir / "work_acceleration_monitor_scorecard.csv"
    curve_path = out_dir / "qf_ppm_vs_k_work_monitor_grid.png"

    if not args.resume and not args.postprocess_only:
        for path in (
            quality_trace_path,
            trajectory_trace_path,
            phase_checkpoint_path,
            local_merge_summary_path,
            local_merge_parent_summary_path,
            first_divergence_path,
            run_rows_path,
            candidate_level_path,
            policy_comparison_path,
            qf_points_path,
            scorecard_path,
            curve_path,
            out_dir / "work_acceleration_monitor_report.md",
        ):
            if path.exists():
                path.unlink()

    if args.postprocess_only:
        _postprocess_outputs(
            out_dir=out_dir,
            quality_trace_path=quality_trace_path,
            trajectory_trace_path=trajectory_trace_path,
            phase_checkpoint_path=phase_checkpoint_path,
            local_merge_summary_path=local_merge_summary_path,
            local_merge_parent_summary_path=local_merge_parent_summary_path,
            first_divergence_path=first_divergence_path,
            run_rows_path=run_rows_path,
            qf_points_path=qf_points_path,
            scorecard_path=scorecard_path,
            curve_path=curve_path,
            local_merge_summary_mode=args.local_merge_summary_mode,
            keep_raw_trajectory=args.keep_raw_trajectory,
        )
        summary = {
            "schema": "leiden_hysteresis_work_acceleration_monitor.v2",
            "local_merge_summary_mode": args.local_merge_summary_mode,
            "paths": {
                "run_rows": _display_path(run_rows_path),
                "candidate_level_rows": _display_path(candidate_level_path),
                "policy_comparison_rows": _display_path(policy_comparison_path),
                "qf_i_k_points": _display_path(qf_points_path),
                "scorecard": _display_path(scorecard_path),
                "report": _display_path(
                    out_dir / "work_acceleration_monitor_report.md"
                ),
                "plot": _display_path(curve_path),
                "phase_checkpoints": _display_path(phase_checkpoint_path),
                "local_merge_summaries": _display_path(local_merge_summary_path),
                "first_divergence_parent_summary": _display_path(
                    local_merge_parent_summary_path
                ),
                "first_divergence": _display_path(first_divergence_path),
                "quality_trace": _display_path(quality_trace_path),
            },
        }
        (out_dir / "work_acceleration_monitor_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary["paths"], indent=2, sort_keys=True))
        return

    graph_dirs = _parse_graph_dirs(args.graph_dirs)
    seeds = _parse_int_list(args.seeds)
    candidate_budgets = (
        _parse_int_list(args.candidate_budgets)
        if args.candidate_budgets
        else [int(args.max_group_candidates)]
    )

    with _trace_file_context(
        quality_trace_path, trajectory_trace_path, resume=args.resume
    ):
        for graph_dir in graph_dirs:
            graph, node_weights, arrays = _build_monitor_graph(
                graph_dir,
                probe_only=bool(args.probe_only),
            )
            case = _case_name(graph_dir)
            for seed in seeds:
                print(f"[monitor] {case} seed={seed}: baseline", flush=True)
                baseline_t0 = time.perf_counter()
                with _trace_disabled_context():
                    baseline = graph.run_leiden(
                        resolution=args.resolution,
                        seed=seed,
                        n_iterations=args.baseline_iterations,
                        randomness=args.randomness,
                        membership_dtype=np.uint32 if args.probe_only else np.uint64,
                    )
                baseline_elapsed = time.perf_counter() - baseline_t0
                baseline_membership = np.asarray(
                    baseline.membership,
                    dtype=np.uint32 if args.probe_only else np.uint64,
                )
                if args.probe_only and arrays is None and node_weights is None:
                    node_weights = np.memmap(
                        graph_dir / "node_weights.f64.bin",
                        dtype=np.float64,
                        mode="r",
                    )
                candidate_t0 = time.perf_counter()
                candidate_clusters = _candidate_clusters(
                    graph,
                    baseline_membership,
                    node_weights,
                    resolution=args.resolution,
                    top_weight_count=args.max_suspect_clusters,
                    external_priority_count=args.max_external_priority_clusters,
                    min_suspect_weight=args.min_suspect_weight,
                    min_doc_weight=args.min_doc_weight,
                    min_assigned_fraction=args.min_assigned_fraction,
                    min_best_group_fraction=args.min_best_group_fraction,
                )
                if args.probe_only and arrays is None:
                    _release_memmap_array(node_weights)
                    node_weights = None
                candidate_selection_elapsed = time.perf_counter() - candidate_t0
                print(
                    f"[monitor] {case} seed={seed}: candidates={candidate_clusters}",
                    flush=True,
                )
                for candidate_budget in candidate_budgets:
                    print(
                        f"[monitor] {case} seed={seed}: candidate_budget={candidate_budget}",
                        flush=True,
                    )
                    probe_t0 = time.perf_counter()
                    if args.candidate_eval_mode in {
                        "full_p5",
                        "parallel_full_p5_portfolio",
                    }:
                        parallel_probe = (
                            args.candidate_eval_mode == "parallel_full_p5_portfolio"
                        )
                        with _trace_disabled_context():
                            probe = graph.non_monotone_group_escape_probe(
                                baseline_membership,
                                np.asarray(candidate_clusters, dtype=np.uint64),
                                resolution=args.resolution,
                                max_candidates=candidate_budget,
                                polish_iterations=args.polish_iterations,
                                randomness=args.randomness,
                                seed=seed + args.perturb_seed_offset,
                                min_doc_weight=args.min_doc_weight,
                                min_assigned_fraction=args.min_assigned_fraction,
                                min_best_group_fraction=args.min_best_group_fraction,
                                quality_eps=args.quality_eps,
                                parallel_candidates=parallel_probe,
                                return_membership=not args.probe_only,
                            )
                        selected_policy = (
                            "parallel_full_p5_portfolio"
                            if parallel_probe
                            else "full_top3_p5"
                        )
                        best = _best_candidate_row(probe.candidate_rows)
                        selected_policy_row = _portfolio_policy_row(
                            policy=selected_policy,
                            probe=probe,
                            best=best,
                        )
                        operational_probe_elapsed = (
                            _finite_float(
                                selected_policy_row.get("total_elapsed_ms"), math.nan
                            )
                            / 1000.0
                        )
                    else:
                        label_full_p5 = (
                            args.candidate_eval_mode == "multifidelity_label"
                            or args.candidate_eval_mode in APPROX_POLISH_LABEL_MODES
                        )
                        approx_polish_labels = (
                            args.candidate_eval_mode in APPROX_POLISH_LABEL_MODES
                        )
                        with _trace_disabled_context():
                            probe = graph.non_monotone_group_escape_multifidelity_probe(
                                baseline_membership,
                                np.asarray(candidate_clusters, dtype=np.uint64),
                                resolution=args.resolution,
                                max_candidates=candidate_budget,
                                prescreen_iterations=args.prescreen_iterations,
                                final_iterations=args.final_iterations,
                                finalists=args.multifidelity_finalists,
                                label_full_p5=label_full_p5,
                                randomness=args.randomness,
                                seed=seed + args.perturb_seed_offset,
                                min_doc_weight=args.min_doc_weight,
                                min_assigned_fraction=args.min_assigned_fraction,
                                min_best_group_fraction=args.min_best_group_fraction,
                                quality_eps=args.quality_eps,
                                return_membership=not args.probe_only,
                                approx_polish_labels=approx_polish_labels,
                                basin_signatures=args.basin_signatures,
                            )
                        selected_policy = APPROX_POLISH_DEFAULT_POLICY.get(
                            args.candidate_eval_mode, str(probe.selected_policy)
                        )
                        policy_rows = _ensure_multifidelity_policy_rows(
                            probe.policy_rows,
                            probe.candidate_rows,
                            quality_eps=args.quality_eps,
                        )
                        if approx_polish_labels:
                            policy_rows = _ensure_approx_polish_policy_rows(
                                policy_rows,
                                probe.candidate_rows,
                                quality_eps=args.quality_eps,
                            )
                        policy_by_name = _policy_rows_by_name(policy_rows)
                        selected_policy_row = policy_by_name.get(selected_policy, {})
                        best = _candidate_row_by_index(
                            probe.candidate_rows,
                            selected_policy_row.get(
                                "selected_candidate_index",
                                probe.selected_candidate_index,
                            ),
                        )
                        operational_probe_elapsed = (
                            _finite_float(
                                selected_policy_row.get("total_elapsed_ms"), math.nan
                            )
                            / 1000.0
                        )
                    probe_elapsed = time.perf_counter() - probe_t0
                    if args.candidate_eval_mode in {
                        "multifidelity_label",
                        "multifidelity_operational",
                    } | APPROX_POLISH_LABEL_MODES:
                        row_base = {
                            "case": case,
                            "seed": seed,
                            "candidate_budget": candidate_budget,
                            "max_group_candidates": candidate_budget,
                            "candidate_eval_mode": args.candidate_eval_mode,
                            "selected_policy": selected_policy,
                            "label_full_p5": bool(
                                args.candidate_eval_mode == "multifidelity_label"
                                or args.candidate_eval_mode in APPROX_POLISH_LABEL_MODES
                            ),
                            "approx_polish_labels": bool(
                                args.candidate_eval_mode in APPROX_POLISH_LABEL_MODES
                            ),
                            "basin_signatures": bool(args.basin_signatures),
                        }
                        _append_rows(
                            candidate_level_path, row_base, probe.candidate_rows
                        )
                        _append_rows(
                            policy_comparison_path,
                            row_base,
                            policy_rows,
                        )
                    elif args.candidate_eval_mode in {
                        "full_p5",
                        "parallel_full_p5_portfolio",
                    } and (
                        args.probe_only
                        or args.candidate_eval_mode == "parallel_full_p5_portfolio"
                    ):
                        row_base = {
                            "case": case,
                            "seed": seed,
                            "candidate_budget": candidate_budget,
                            "max_group_candidates": candidate_budget,
                            "candidate_eval_mode": args.candidate_eval_mode,
                            "selected_policy": selected_policy,
                            "label_full_p5": True,
                            "candidate_eval_parallel": bool(
                                getattr(probe, "candidate_eval_parallel", False)
                            ),
                        }
                        _append_rows(
                            candidate_level_path,
                            row_base,
                            _portfolio_candidate_rows(probe.candidate_rows),
                        )
                        _append_rows(
                            policy_comparison_path,
                            row_base,
                            [selected_policy_row],
                        )
                    if args.probe_only or best is None:
                        _append_run_row(
                            run_rows_path,
                            _probe_run_row(
                                case=case,
                                seed=seed,
                                candidate_budget=candidate_budget,
                                baseline_quality=float(baseline.quality),
                                baseline_elapsed=baseline_elapsed,
                                candidate_selection_elapsed=candidate_selection_elapsed,
                                probe_elapsed=probe_elapsed,
                                probe=probe,
                                candidate_eval_mode=args.candidate_eval_mode,
                                selected_policy=selected_policy,
                                selected_policy_row=selected_policy_row,
                                candidate_clusters=candidate_clusters,
                                best=best,
                            ),
                        )
                        continue

                    if arrays is None:
                        raise RuntimeError(
                            "edge arrays are required outside probe-only mode"
                        )
                    group_nodes, reconstruction = _reconstruct_external_group(
                        src=arrays.src,
                        dst=arrays.dst,
                        weight=arrays.weight,
                        membership=baseline_membership,
                        node_weights=node_weights,
                        source_cluster=int(best["source_cluster"]),
                        target_cluster=int(best["target_cluster"]),
                    )
                    perturbed = baseline_membership.copy()
                    perturbed[group_nodes] = np.uint64(int(best["target_cluster"]))
                    perturbed = _compact_membership(perturbed)
                    selected_seed = (
                        seed + args.perturb_seed_offset + int(best["candidate_index"])
                    )
                    candidate_probe_cost = (
                        operational_probe_elapsed
                        if math.isfinite(operational_probe_elapsed)
                        else probe_elapsed
                    )
                    trace_polish_iterations = (
                        args.polish_iterations
                        if args.candidate_eval_mode
                        in {"full_p5", "parallel_full_p5_portfolio"}
                        else args.final_iterations
                    )

                    for branch, initial, run_seed in (
                        ("extra", baseline_membership, seed + args.perturb_seed_offset),
                        ("perturb", perturbed, selected_seed),
                    ):
                        run_id = (
                            f"{case}|seed={seed}|budget={candidate_budget}|{branch}"
                        )
                        print(f"[monitor] {run_id}: polish", flush=True)
                        t0 = time.perf_counter()
                        with _trace_run_context(
                            run_id, target_max_weight=args.target_max_weight
                        ):
                            result = graph.run_leiden(
                                resolution=args.resolution,
                                seed=run_seed,
                                n_iterations=trace_polish_iterations,
                                randomness=args.randomness,
                                initial_membership=initial,
                            )
                        elapsed = time.perf_counter() - t0
                        row = {
                            "case": case,
                            "seed": seed,
                            "candidate_budget": candidate_budget,
                            "max_group_candidates": candidate_budget,
                            "ranking_strategy": "external_grain_priority_v1",
                            "branch": branch,
                            "run_id": run_id,
                            "baseline_quality": float(baseline.quality),
                            "quality": float(result.quality),
                            "quality_delta_vs_baseline": float(
                                result.quality - baseline.quality
                            ),
                            "elapsed_sec": elapsed,
                            "baseline_elapsed_sec": baseline_elapsed,
                            "candidate_cluster_selection_elapsed_sec": candidate_selection_elapsed,
                            "candidate_probe_elapsed_sec": candidate_probe_cost,
                            "probe_elapsed_sec": probe_elapsed,
                            "candidate_probe_eval_wall_elapsed_sec": _finite_float(
                                getattr(
                                    probe, "candidate_eval_wall_elapsed_ms", math.nan
                                )
                            )
                            / 1000.0,
                            "candidate_probe_cpu_sum_elapsed_sec": _finite_float(
                                getattr(
                                    probe, "candidate_eval_cpu_sum_elapsed_ms", math.nan
                                )
                            )
                            / 1000.0,
                            "candidate_probe_parallel": bool(
                                getattr(probe, "candidate_eval_parallel", False)
                            ),
                            "candidate_probe_parallel_speedup": _finite_float(
                                getattr(
                                    probe, "candidate_eval_parallel_speedup", math.nan
                                )
                            ),
                            "candidate_probe_parallel_workers": int(
                                getattr(probe, "candidate_eval_parallel_workers", 0)
                            ),
                            "n_clusters": int(result.n_clusters),
                            "candidate_clusters": json.dumps(candidate_clusters),
                            "candidate_index": int(best["candidate_index"]),
                            "source_cluster": int(best["source_cluster"]),
                            "target_cluster": int(best["target_cluster"]),
                            "group_kind": best["group_kind"],
                            "group_count": int(best["group_count"]),
                            "group_weight": float(best["group_weight"]),
                            "pre_polish_delta_q": float(
                                best.get(
                                    "pre_polish_delta_q",
                                    best.get("pre_delta_q", math.nan),
                                )
                            ),
                            "expected_post_polish_delta_q": float(
                                best.get(
                                    "post_polish_delta_q",
                                    best.get("p5_delta_q", math.nan),
                                )
                            ),
                            "p1_delta_q": best.get("p1_delta_q", math.nan),
                            "p5_delta_q": best.get("p5_delta_q", math.nan),
                            "selected_policy": selected_policy,
                            "selected_policy_available": bool(
                                selected_policy_row.get("available", True)
                            ),
                            "candidate_eval_mode": args.candidate_eval_mode,
                            "accepted_by_probe_quality": bool(
                                best.get("accepted_by_quality", probe.accepted)
                            ),
                            "probe_accepted": bool(probe.accepted),
                        }
                        row.update(reconstruction)
                        _append_run_row(run_rows_path, row)

    _postprocess_outputs(
        out_dir=out_dir,
        quality_trace_path=quality_trace_path,
        trajectory_trace_path=trajectory_trace_path,
        phase_checkpoint_path=phase_checkpoint_path,
        local_merge_summary_path=local_merge_summary_path,
        local_merge_parent_summary_path=local_merge_parent_summary_path,
        first_divergence_path=first_divergence_path,
        run_rows_path=run_rows_path,
        qf_points_path=qf_points_path,
        scorecard_path=scorecard_path,
        curve_path=curve_path,
        local_merge_summary_mode=args.local_merge_summary_mode,
        keep_raw_trajectory=args.keep_raw_trajectory,
    )

    summary = {
        "schema": "leiden_hysteresis_work_acceleration_monitor.v2",
        "graph_dirs": [_display_path(path) for path in graph_dirs],
        "seeds": seeds,
        "candidate_budgets": candidate_budgets,
        "candidate_eval_mode": args.candidate_eval_mode,
        "probe_only": bool(args.probe_only),
        "prescreen_iterations": args.prescreen_iterations,
        "final_iterations": args.final_iterations,
        "multifidelity_finalists": args.multifidelity_finalists,
        "local_merge_summary_mode": args.local_merge_summary_mode,
        "paths": {
            "run_rows": _display_path(run_rows_path),
            "candidate_level_rows": _display_path(candidate_level_path),
            "policy_comparison_rows": _display_path(policy_comparison_path),
            "qf_i_k_points": _display_path(qf_points_path),
            "scorecard": _display_path(scorecard_path),
            "report": _display_path(out_dir / "work_acceleration_monitor_report.md"),
            "plot": _display_path(curve_path),
            "phase_checkpoints": _display_path(phase_checkpoint_path),
            "local_merge_summaries": _display_path(local_merge_summary_path),
            "first_divergence_parent_summary": _display_path(
                local_merge_parent_summary_path
            ),
            "first_divergence": _display_path(first_divergence_path),
            "quality_trace": _display_path(quality_trace_path),
        },
    }
    (out_dir / "work_acceleration_monitor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary["paths"], indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dirs", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=str, default="11,42")
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--target-max-weight", type=float, default=1000.0)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument(
        "--candidate-eval-mode",
        choices=(
            "full_p5",
            "multifidelity_label",
            "multifidelity_operational",
            "parallel_full_p5_portfolio",
            "localized_label",
            "quotient_label",
            "upper_bound_label",
        ),
        default="full_p5",
        help=(
            "full_p5 preserves the v2 full candidate polish path; "
            "multifidelity_label computes p1 prescreen plus full p5 labels; "
            "multifidelity_operational only promotes p1 finalists to p5; "
            "parallel_full_p5_portfolio evaluates full-p5 candidates in parallel; "
            "localized_label, quotient_label, and upper_bound_label add shadow "
            "approximate-polish diagnostics and require --probe-only."
        ),
    )
    parser.add_argument("--prescreen-iterations", type=int, default=1)
    parser.add_argument("--final-iterations", type=int, default=5)
    parser.add_argument("--multifidelity-finalists", type=int, default=1)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run baseline, candidate selection, and candidate probe only; skip extra/perturb trace polish.",
    )
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--max-suspect-clusters", type=int, default=2)
    parser.add_argument("--max-external-priority-clusters", type=int, default=3)
    parser.add_argument("--min-suspect-weight", type=float, default=500.0)
    parser.add_argument("--max-group-candidates", type=int, default=5)
    parser.add_argument(
        "--candidate-budgets",
        type=str,
        default=None,
        help="Comma-separated candidate budgets to evaluate; defaults to --max-group-candidates.",
    )
    parser.add_argument("--min-doc-weight", type=float, default=0.0)
    parser.add_argument("--min-assigned-fraction", type=float, default=0.0)
    parser.add_argument("--min-best-group-fraction", type=float, default=0.0)
    parser.add_argument("--quality-eps", type=float, default=0.0)
    parser.add_argument(
        "--basin-signatures",
        action="store_true",
        help=(
            "Attach compact p5 membership basin signatures to multifidelity "
            "candidate rows. This is diagnostic-only and requires p5 labels for "
            "complete basin coverage."
        ),
    )
    parser.add_argument(
        "--local-merge-summary-mode",
        choices=("compact", "focused", "full"),
        default="compact",
        help=(
            "compact writes only first_divergence_parent_summary.csv; "
            "focused also writes parent rows at first-divergence iterations; "
            "full writes every local-merge parent row."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--keep-raw-trajectory", action="store_true")
    parser.add_argument(
        "--parallel-candidate-workers",
        type=int,
        default=0,
        help="Set RAYON_NUM_THREADS before parallel candidate evaluation; 0 leaves Rayon default.",
    )
    return parser.parse_args()


def main() -> None:
    run_monitor(parse_args())


if __name__ == "__main__":
    main()
