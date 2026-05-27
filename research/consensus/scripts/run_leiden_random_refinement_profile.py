"""Profile standard Leiden seed/randomness refinement landscapes.

This runner uses prepared graph summaries from the Dongdaemun/adaptive
experiments, but only runs the standard Rust Leiden path.  It records per-run
quality checkpoints through the standard Leiden quality trace and summarizes
time-to-quality/pressure metrics for seed and randomness comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import evaluate_dongdaemun_refinement_slice4 as pilot  # noqa: E402
import summarize_dongdaemun_quality_trace as quality_summary  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "leiden_iteration_budget_profile_20260511"
)
DEFAULT_SUMMARIES = (
    Path(
        "research/consensus/results/adaptive_refinement/"
        "dongdaemun_safe_fast_layer_comparison/bc_cosine_20260507/summaries/"
        "field15_gcc_emb_full_knn30/seed_42/bc_cosine_prepare_summary.json"
    ),
    Path(
        "research/consensus/results/adaptive_refinement/"
        "dongdaemun_safe_fast_layer_comparison/cc_cosine_20260507/summaries/"
        "field15_gcc_emb_full_knn30/seed_42/cc_cosine_prepare_summary.json"
    ),
)
DEFAULT_SEEDS = (11, 42, 73, 101, 137)
DEFAULT_RANDOMNESS = (0.0, 0.001, 0.01)
DEFAULT_N_ITERATIONS_VALUES = ("1", "2", "3", "5", "10", "convergence")
VARIANT_STANDARD = "standard_leiden"
SCHEMA_VERSION = 1

ROWS_JSONL_FILENAME = "leiden_random_refinement_profile_rows.jsonl"
ROWS_CSV_FILENAME = "leiden_random_refinement_profile_rows.csv"
SUMMARY_FILENAME = "leiden_random_refinement_profile_summary.json"
REPORT_FILENAME = "leiden_random_refinement_profile_report.md"
QUALITY_TRACE_FILENAME = "quality_trace.jsonl"
QUALITY_TRACE_RUNS_FILENAME = "quality_trace_runs.jsonl"
ITERATION_BUDGET_BY_RUN_FILENAME = "iteration_budget_by_run.csv"
ITERATION_BUDGET_BY_GROUP_FILENAME = "iteration_budget_by_group.csv"
SHORTLIST_POLICY_SIMULATION_FILENAME = "shortlist_policy_simulation.csv"
QUALITY_BY_BUDGET_PLOT_FILENAME = "quality_vs_iteration_budget.png"
PRESSURE_BY_BUDGET_PLOT_FILENAME = "pressure_vs_iteration_budget.png"
ELAPSED_BY_BUDGET_PLOT_FILENAME = "elapsed_vs_iteration_budget.png"
RECOVERY_BY_ELAPSED_PLOT_FILENAME = "quality_recovery_vs_elapsed_budget.png"

QUALITY_TRACE_PATH_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_PATH"
QUALITY_TRACE_RUN_ID_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_RUN_ID"
QUALITY_TRACE_EPOCH_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_EPOCH"
QUALITY_TRACE_TARGET_ENV = "SCISCAPE_LEIDEN_QUALITY_TRACE_TARGET_MAX_WEIGHT"

ROW_FIELDS = [
    "sample",
    "source_sample",
    "edge_layer",
    "summary_path",
    "variant",
    "run_id",
    "row_key",
    "seed",
    "randomness",
    "requested_n_iterations",
    "iteration_mode",
    "n_iterations",
    "n_iterations_used",
    "resolution",
    "target_max_doc_weight",
    "supported",
    "unsupported_reason",
    "elapsed_sec",
    "elapsed_ms_final",
    "time_to_95pct_final_quality_gain_ms",
    "quality",
    "quality_recomputed",
    "quality_recompute_delta",
    "quality_recompute_abs_delta",
    "quality_recompute_ok",
    "quality_gain_per_sec",
    "best_quality_delta_per_sec",
    "final_pressure_reduction_per_sec",
    "n_clusters",
    "max_doc_weight",
    "max_doc_weight_ratio",
    "n_above_max_doc_weight",
    "top10_doc_weights",
    "trace_checkpoint_count",
    "trace_has_start",
    "trace_has_final",
    "trace_final_quality",
    "trace_final_iteration",
    "trace_quality_matches_result",
    "late_quality_gain_vs_iter1",
    "late_pressure_delta_vs_iter1",
    "quality_recovery_ratio_vs_best_10",
    "quality_recovery_ratio_vs_best_convergence",
]


@dataclass(frozen=True)
class ProfileInput:
    input_cfg: pilot.Slice4Input
    summary: dict[str, Any]
    summary_path: Path
    sample: str
    source_sample: str | None
    edge_layer: str | None


@dataclass(frozen=True)
class IterationBudget:
    requested: str
    n_iterations: int
    mode: str

    @property
    def sort_key(self) -> tuple[int, int]:
        return (1, 0) if self.mode == "convergence" else (0, int(self.n_iterations))


def _json_safe(value: Any) -> Any:
    return pilot._json_safe(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    base_fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ROW_FIELDS if base_fields is None else base_fields)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":")))
        fh.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_json_safe(row), sort_keys=True, separators=(",", ":")))
            fh.write("\n")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parse_csv_tuple(value: str, *, cast: type) -> tuple[Any, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(cast(item) for item in items)


def _parse_n_iterations_value(value: Any) -> IterationBudget:
    token = str(value).strip().lower()
    if token in {"convergence", "converge", "until_convergence", "0"}:
        return IterationBudget(
            requested="convergence",
            n_iterations=0,
            mode="convergence",
        )
    try:
        n_iterations = int(token)
    except ValueError as exc:
        raise ValueError(
            f"Invalid n_iterations value {value!r}; use a positive integer or convergence"
        ) from exc
    if n_iterations <= 0:
        raise ValueError(
            f"Invalid n_iterations value {value!r}; use a positive integer or convergence"
        )
    return IterationBudget(
        requested=str(n_iterations),
        n_iterations=n_iterations,
        mode="fixed",
    )


def _parse_n_iterations_values(value: str) -> tuple[IterationBudget, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("--n-iterations-values must contain at least one value")
    budgets = tuple(_parse_n_iterations_value(item) for item in items)
    seen: set[str] = set()
    for budget in budgets:
        if budget.requested in seen:
            raise ValueError(f"Duplicate n_iterations value: {budget.requested}")
        seen.add(budget.requested)
    return budgets


def _row_key(
    *,
    summary_path: Path,
    seed: int,
    randomness: float,
    requested_n_iterations: str,
) -> str:
    return json.dumps(
        [
            str(pilot._rel(summary_path)),
            int(seed),
            float(randomness),
            str(requested_n_iterations),
            VARIANT_STANDARD,
        ],
        separators=(",", ":"),
    )


def _run_id(row_key: str) -> str:
    return hashlib.sha1(row_key.encode("utf-8")).hexdigest()[:16]


def _resolve_profile_input(summary_path: Path) -> ProfileInput:
    summary_path = summary_path if summary_path.is_absolute() else REPO_ROOT / summary_path
    summary = _read_json(summary_path)
    input_cfg = pilot._resolve_input_from_summary(summary_path)
    return ProfileInput(
        input_cfg=input_cfg,
        summary=summary,
        summary_path=summary_path,
        sample=str(summary.get("sample") or input_cfg.sample),
        source_sample=summary.get("source_sample"),
        edge_layer=summary.get("edge_layer"),
    )


def _quality_trace_run_metadata(
    *,
    profile: ProfileInput,
    seed: int,
    randomness: float,
    budget: IterationBudget,
    n_iterations_used: int | None,
    run_id: str,
    row_key: str,
) -> dict[str, Any]:
    return {
        "schema": f"leiden_random_refinement_quality_trace_run.v{SCHEMA_VERSION}",
        "run_id": run_id,
        "row_key": row_key,
        "sample": profile.sample,
        "source_sample": profile.source_sample,
        "edge_layer": profile.edge_layer,
        "summary_path": pilot._rel(profile.summary_path),
        "variant": VARIANT_STANDARD,
        "seed": int(seed),
        "randomness": float(randomness),
        "requested_n_iterations": budget.requested,
        "iteration_mode": budget.mode,
        "n_iterations": int(budget.n_iterations),
        "n_iterations_used": n_iterations_used,
        "resolution": float(profile.input_cfg.resolution),
        "target_max_doc_weight": float(profile.input_cfg.target_max_doc_weight),
    }


@contextmanager
def _quality_trace_path_context(path: Path, *, resume: bool):
    previous_path = os.environ.get(QUALITY_TRACE_PATH_ENV)
    previous_epoch = os.environ.get(QUALITY_TRACE_EPOCH_ENV)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and path.exists():
        path.unlink()
    os.environ[QUALITY_TRACE_PATH_ENV] = str(path)
    os.environ[QUALITY_TRACE_EPOCH_ENV] = uuid.uuid4().hex
    try:
        yield
    finally:
        if previous_path is None:
            os.environ.pop(QUALITY_TRACE_PATH_ENV, None)
        else:
            os.environ[QUALITY_TRACE_PATH_ENV] = previous_path
        if previous_epoch is None:
            os.environ.pop(QUALITY_TRACE_EPOCH_ENV, None)
        else:
            os.environ[QUALITY_TRACE_EPOCH_ENV] = previous_epoch


@contextmanager
def _quality_trace_run_context(run_id: str, target_max_doc_weight: float):
    previous_run_id = os.environ.get(QUALITY_TRACE_RUN_ID_ENV)
    previous_target = os.environ.get(QUALITY_TRACE_TARGET_ENV)
    os.environ[QUALITY_TRACE_RUN_ID_ENV] = run_id
    os.environ[QUALITY_TRACE_TARGET_ENV] = str(float(target_max_doc_weight))
    try:
        yield
    finally:
        if previous_run_id is None:
            os.environ.pop(QUALITY_TRACE_RUN_ID_ENV, None)
        else:
            os.environ[QUALITY_TRACE_RUN_ID_ENV] = previous_run_id
        if previous_target is None:
            os.environ.pop(QUALITY_TRACE_TARGET_ENV, None)
        else:
            os.environ[QUALITY_TRACE_TARGET_ENV] = previous_target


def _trace_summary_for_run(trace_path: Path, run_id: str) -> dict[str, Any]:
    events = [
        event
        for event in _read_jsonl(trace_path)
        if event.get("event") == "quality_checkpoint" and str(event.get("run_id")) == run_id
    ]
    final = next(
        (event for event in reversed(events) if event.get("phase") == "final"),
        events[-1] if events else {},
    )
    return {
        "trace_checkpoint_count": len(events),
        "trace_has_start": any(event.get("phase") == "start" for event in events),
        "trace_has_final": any(event.get("phase") == "final" for event in events),
        "trace_final_quality": final.get("quality"),
        "trace_final_iteration": final.get("iteration"),
    }


def _recompute_quality(graph: Any, membership: np.ndarray, resolution: float, quality: float) -> dict[str, Any]:
    recomputed = float(graph.cpm_quality(membership, resolution=float(resolution)))
    delta = recomputed - float(quality)
    return {
        "quality_recomputed": recomputed,
        "quality_recompute_delta": delta,
        "quality_recompute_abs_delta": abs(delta),
        "quality_recompute_ok": abs(delta) <= 1.0e-6,
    }


def _run_one(
    *,
    profile: ProfileInput,
    graph: Any,
    node_weights: np.ndarray,
    seed: int,
    randomness: float,
    budget: IterationBudget,
    trace_path: Path,
    trace_runs_path: Path,
) -> dict[str, Any]:
    row_key = _row_key(
        summary_path=profile.summary_path,
        seed=seed,
        randomness=randomness,
        requested_n_iterations=budget.requested,
    )
    run_id = _run_id(row_key)
    start = time.perf_counter()
    with _quality_trace_run_context(run_id, profile.input_cfg.target_max_doc_weight):
        result = graph.run_leiden(
            resolution=float(profile.input_cfg.resolution),
            seed=int(seed),
            n_iterations=int(budget.n_iterations),
            randomness=float(randomness),
        )
    elapsed_sec = time.perf_counter() - start
    membership = np.asarray(result.membership, dtype=np.uint64)
    pressure = pilot._cluster_weight_summary(
        membership,
        node_weights,
        float(profile.input_cfg.target_max_doc_weight),
    )
    recompute = _recompute_quality(
        graph,
        membership,
        float(profile.input_cfg.resolution),
        float(result.quality),
    )
    trace_summary = _trace_summary_for_run(trace_path, run_id)
    trace_quality = _finite_float(trace_summary.get("trace_final_quality"))
    trace_matches = (
        trace_quality is not None and abs(trace_quality - float(result.quality)) <= 1.0e-6
    )
    n_iterations_used_value = _finite_float(trace_summary.get("trace_final_iteration"))
    n_iterations_used = None if n_iterations_used_value is None else int(n_iterations_used_value)
    row = {
        "sample": profile.sample,
        "source_sample": profile.source_sample,
        "edge_layer": profile.edge_layer,
        "summary_path": pilot._rel(profile.summary_path),
        "variant": VARIANT_STANDARD,
        "run_id": run_id,
        "row_key": row_key,
        "seed": int(seed),
        "randomness": float(randomness),
        "requested_n_iterations": budget.requested,
        "iteration_mode": budget.mode,
        "n_iterations": int(budget.n_iterations),
        "n_iterations_used": n_iterations_used,
        "resolution": float(profile.input_cfg.resolution),
        "target_max_doc_weight": float(profile.input_cfg.target_max_doc_weight),
        "supported": True,
        "unsupported_reason": "",
        "elapsed_sec": float(elapsed_sec),
        "quality": float(result.quality),
        "n_clusters": int(result.n_clusters),
        **pressure,
        **recompute,
        **trace_summary,
        "trace_quality_matches_result": bool(trace_matches),
    }
    _append_jsonl(
        trace_runs_path,
        _quality_trace_run_metadata(
            profile=profile,
            seed=seed,
            randomness=randomness,
            budget=budget,
            n_iterations_used=n_iterations_used,
            run_id=run_id,
            row_key=row_key,
        ),
    )
    return row


def _load_trace_by_run(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            run_id = row.get("run_id")
            if run_id:
                rows[str(run_id)] = row
    return rows


def _enrich_rows_with_trace_summary(rows: list[dict[str, Any]], by_run_csv: Path) -> None:
    by_run = _load_trace_by_run(by_run_csv)
    for row in rows:
        metrics = by_run.get(str(row.get("run_id")), {})
        elapsed_ms = _finite_float(metrics.get("elapsed_ms_final"))
        final_gain = _finite_float(metrics.get("final_quality_delta_vs_start"))
        final_iteration = _finite_float(metrics.get("final_iteration"))
        if final_iteration is not None:
            row["n_iterations_used"] = int(final_iteration)
        row["elapsed_ms_final"] = elapsed_ms
        row["time_to_95pct_final_quality_gain_ms"] = _finite_float(
            metrics.get("time_to_95pct_final_quality_gain_ms")
        )
        row["best_quality_delta_per_sec"] = _finite_float(
            metrics.get("best_quality_delta_per_sec")
        )
        row["final_pressure_reduction_per_sec"] = _finite_float(
            metrics.get("final_pressure_reduction_per_sec")
        )
        row["quality_gain_per_sec"] = (
            final_gain / (elapsed_ms / 1000.0)
            if final_gain is not None and elapsed_ms is not None and elapsed_ms > 0.0
            else None
        )


def _iteration_label(row: dict[str, Any]) -> str:
    requested = row.get("requested_n_iterations")
    if requested is not None and str(requested):
        return str(requested)
    n_iterations = int(row.get("n_iterations") or 0)
    return "convergence" if n_iterations == 0 else str(n_iterations)


def _iteration_sort_key(label: str) -> tuple[int, int]:
    return (1, 0) if label == "convergence" else (0, int(label))


def _ladder_key(row: dict[str, Any]) -> tuple[str, int, float]:
    return (
        str(row.get("sample")),
        int(row.get("seed")),
        float(row.get("randomness")),
    )


def _quality_pressure_key(row: dict[str, Any]) -> tuple[float, float, float]:
    quality = _finite_float(row.get("quality"))
    pressure = _finite_float(row.get("max_doc_weight_ratio"))
    elapsed = _finite_float(row.get("elapsed_sec"))
    return (
        -(quality if quality is not None else -math.inf),
        pressure if pressure is not None else math.inf,
        elapsed if elapsed is not None else math.inf,
    )


def _best_quality_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    supported = [row for row in rows if row.get("supported")]
    if not supported:
        return None
    return min(supported, key=_quality_pressure_key)


def _recovery_ratio(
    *,
    quality: float | None,
    baseline_quality: float | None,
    reference_quality: float | None,
) -> float | None:
    if quality is None or baseline_quality is None or reference_quality is None:
        return None
    denominator = reference_quality - baseline_quality
    if abs(denominator) <= 1.0e-12:
        return 1.0 if quality >= reference_quality - 1.0e-12 else None
    return (quality - baseline_quality) / denominator


def _enrich_rows_with_iteration_budget_metrics(rows: list[dict[str, Any]]) -> None:
    ladders: dict[tuple[str, int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not row.get("supported"):
            continue
        ladders.setdefault(_ladder_key(row), {})[_iteration_label(row)] = row

    for row in rows:
        label = _iteration_label(row)
        ladder = ladders.get(_ladder_key(row), {})
        iter1 = ladder.get("1")
        iter10 = ladder.get("10")
        convergence = ladder.get("convergence")
        quality = _finite_float(row.get("quality"))
        pressure = _finite_float(row.get("max_doc_weight_ratio"))
        iter1_quality = _finite_float(iter1.get("quality")) if iter1 else None
        iter1_pressure = (
            _finite_float(iter1.get("max_doc_weight_ratio")) if iter1 else None
        )
        best10_quality = _finite_float(iter10.get("quality")) if iter10 else None
        convergence_quality = (
            _finite_float(convergence.get("quality")) if convergence else None
        )
        row["late_quality_gain_vs_iter1"] = (
            quality - iter1_quality
            if quality is not None and iter1_quality is not None
            else None
        )
        row["late_pressure_delta_vs_iter1"] = (
            pressure - iter1_pressure
            if pressure is not None and iter1_pressure is not None
            else None
        )
        row["quality_recovery_ratio_vs_best_10"] = _recovery_ratio(
            quality=quality,
            baseline_quality=iter1_quality,
            reference_quality=best10_quality,
        )
        row["quality_recovery_ratio_vs_best_convergence"] = _recovery_ratio(
            quality=quality,
            baseline_quality=iter1_quality,
            reference_quality=convergence_quality,
        )
        if label == "1":
            row["late_quality_gain_vs_iter1"] = 0.0
            row["late_pressure_delta_vs_iter1"] = 0.0


def _best_rows_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample = str(row.get("sample"))
        if not row.get("supported"):
            continue
        current = best.get(sample)
        if current is None or float(row["quality"]) > float(current["quality"]):
            best[sample] = row
    return best


def _time_quality_frontier(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("supported"):
            by_sample.setdefault(str(row.get("sample")), []).append(row)

    frontiers: dict[str, list[dict[str, Any]]] = {}
    for sample, sample_rows in by_sample.items():
        best_quality = -math.inf
        frontier: list[dict[str, Any]] = []
        for row in sorted(sample_rows, key=lambda item: float(item.get("elapsed_sec") or math.inf)):
            quality = float(row.get("quality") or -math.inf)
            if quality > best_quality:
                best_quality = quality
                frontier.append(row)
        frontiers[sample] = frontier
    return frontiers


def _basin_mismatch_count(rows: list[dict[str, Any]]) -> int:
    by_key: dict[tuple[str, int], dict[float, dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("sample")), int(row.get("seed"))), {})[
            float(row.get("randomness"))
        ] = row
    count = 0
    for variants in by_key.values():
        baseline = variants.get(0.0)
        if baseline is None:
            continue
        base_quality = _finite_float(baseline.get("quality"))
        base_pressure = _finite_float(baseline.get("max_doc_weight_ratio"))
        if base_quality is None or base_pressure is None:
            continue
        for randomness, row in variants.items():
            if randomness == 0.0:
                continue
            quality = _finite_float(row.get("quality"))
            pressure = _finite_float(row.get("max_doc_weight_ratio"))
            if quality is not None and pressure is not None:
                count += int(quality > base_quality and pressure > base_pressure)
    return count


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _iteration_budget_by_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ladders: dict[tuple[str, int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("supported"):
            ladders.setdefault(_ladder_key(row), {})[_iteration_label(row)] = row

    output: list[dict[str, Any]] = []
    for (sample, seed, randomness), ladder in sorted(ladders.items()):
        labels = sorted(ladder, key=_iteration_sort_key)
        iter1 = ladder.get("1")
        iter10 = ladder.get("10")
        convergence = ladder.get("convergence")
        best_row = _best_quality_row(list(ladder.values()))
        row: dict[str, Any] = {
            "sample": sample,
            "seed": seed,
            "randomness": randomness,
            "iteration_budgets": labels,
            "best_requested_n_iterations": (
                None if best_row is None else _iteration_label(best_row)
            ),
            "best_quality": None if best_row is None else best_row.get("quality"),
            "best_max_doc_weight_ratio": (
                None if best_row is None else best_row.get("max_doc_weight_ratio")
            ),
            "iter1_quality": None if iter1 is None else iter1.get("quality"),
            "iter10_quality": None if iter10 is None else iter10.get("quality"),
            "convergence_quality": (
                None if convergence is None else convergence.get("quality")
            ),
            "iter1_max_doc_weight_ratio": (
                None if iter1 is None else iter1.get("max_doc_weight_ratio")
            ),
            "iter10_max_doc_weight_ratio": (
                None if iter10 is None else iter10.get("max_doc_weight_ratio")
            ),
            "convergence_max_doc_weight_ratio": (
                None if convergence is None else convergence.get("max_doc_weight_ratio")
            ),
            "iter1_elapsed_sec": None if iter1 is None else iter1.get("elapsed_sec"),
            "iter10_elapsed_sec": None if iter10 is None else iter10.get("elapsed_sec"),
            "convergence_elapsed_sec": (
                None if convergence is None else convergence.get("elapsed_sec")
            ),
        }
        for label in labels:
            budget_row = ladder[label]
            safe_label = "convergence" if label == "convergence" else f"iter{label}"
            row[f"{safe_label}_quality"] = budget_row.get("quality")
            row[f"{safe_label}_max_doc_weight_ratio"] = budget_row.get(
                "max_doc_weight_ratio"
            )
            row[f"{safe_label}_elapsed_sec"] = budget_row.get("elapsed_sec")
            row[f"{safe_label}_n_iterations_used"] = budget_row.get("n_iterations_used")
            row[f"{safe_label}_late_quality_gain_vs_iter1"] = budget_row.get(
                "late_quality_gain_vs_iter1"
            )
            row[f"{safe_label}_late_pressure_delta_vs_iter1"] = budget_row.get(
                "late_pressure_delta_vs_iter1"
            )
        output.append(row)
    return output


def _iteration_budget_by_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("supported"):
            groups.setdefault(
                (
                    str(row.get("sample")),
                    float(row.get("randomness")),
                    _iteration_label(row),
                ),
                [],
            ).append(row)

    output: list[dict[str, Any]] = []
    for (sample, randomness, label), group_rows in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], _iteration_sort_key(item[0][2]))
    ):
        qualities = [
            value
            for value in (_finite_float(row.get("quality")) for row in group_rows)
            if value is not None
        ]
        pressures = [
            value
            for value in (
                _finite_float(row.get("max_doc_weight_ratio")) for row in group_rows
            )
            if value is not None
        ]
        elapsed = [
            value
            for value in (_finite_float(row.get("elapsed_sec")) for row in group_rows)
            if value is not None
        ]
        used = [
            value
            for value in (
                _finite_float(row.get("n_iterations_used")) for row in group_rows
            )
            if value is not None
        ]
        best_row = _best_quality_row(group_rows)
        output.append(
            {
                "sample": sample,
                "randomness": randomness,
                "requested_n_iterations": label,
                "iteration_mode": (
                    "convergence" if label == "convergence" else "fixed"
                ),
                "n_runs": len(group_rows),
                "mean_quality": _mean(qualities),
                "median_quality": _median(qualities),
                "best_quality": None if best_row is None else best_row.get("quality"),
                "mean_max_doc_weight_ratio": _mean(pressures),
                "median_max_doc_weight_ratio": _median(pressures),
                "best_max_doc_weight_ratio": (
                    None if best_row is None else best_row.get("max_doc_weight_ratio")
                ),
                "mean_elapsed_sec": _mean(elapsed),
                "median_elapsed_sec": _median(elapsed),
                "mean_n_iterations_used": _mean(used),
                "median_n_iterations_used": _median(used),
            }
        )
    return output


def _sample_rows_by_iteration(rows: list[dict[str, Any]], sample: str) -> dict[str, list[dict[str, Any]]]:
    by_iteration: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("supported") and str(row.get("sample")) == sample:
            by_iteration.setdefault(_iteration_label(row), []).append(row)
    return by_iteration


def _row_by_candidate(rows: list[dict[str, Any]]) -> dict[tuple[int, float], dict[str, Any]]:
    return {
        (int(row.get("seed")), float(row.get("randomness"))): row
        for row in rows
        if row.get("supported")
    }


def _sum_elapsed(rows: list[dict[str, Any]]) -> float:
    return float(
        sum(value for value in (_finite_float(row.get("elapsed_sec")) for row in rows) if value)
    )


def _shortlist_policy_simulation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = sorted({str(row.get("sample")) for row in rows if row.get("supported")})
    output: list[dict[str, Any]] = []
    for sample in samples:
        by_iteration = _sample_rows_by_iteration(rows, sample)
        fixed10_rows = by_iteration.get("10", [])
        convergence_rows = by_iteration.get("convergence", [])
        fixed10_by_candidate = _row_by_candidate(fixed10_rows)
        convergence_by_candidate = _row_by_candidate(convergence_rows)
        best10 = _best_quality_row(fixed10_rows)
        best_convergence = _best_quality_row(convergence_rows)
        full10_elapsed = _sum_elapsed(fixed10_rows)
        full_convergence_elapsed = _sum_elapsed(convergence_rows)

        for stage1_label in ("1", "2"):
            stage1_rows = sorted(
                by_iteration.get(stage1_label, []),
                key=_quality_pressure_key,
            )
            if not stage1_rows:
                continue
            stage1_elapsed = _sum_elapsed(stage1_rows)
            for top_k in (1, 2, 3, 5):
                promoted = stage1_rows[: min(top_k, len(stage1_rows))]
                promoted_keys = [
                    (int(row.get("seed")), float(row.get("randomness")))
                    for row in promoted
                ]
                final10_candidates = [
                    fixed10_by_candidate[key]
                    for key in promoted_keys
                    if key in fixed10_by_candidate
                ]
                if not final10_candidates:
                    continue
                best_final10 = _best_quality_row(final10_candidates)
                if best_final10 is None:
                    continue
                for polish_convergence in (False, True):
                    selected = best_final10
                    polish_elapsed = 0.0
                    if polish_convergence:
                        selected_key = (
                            int(best_final10.get("seed")),
                            float(best_final10.get("randomness")),
                        )
                        polish_candidate = convergence_by_candidate.get(selected_key)
                        if polish_candidate is not None:
                            selected = polish_candidate
                            polish_elapsed = (
                                _finite_float(selected.get("elapsed_sec")) or 0.0
                            )
                    final_elapsed = _sum_elapsed(final10_candidates)
                    estimated_elapsed = stage1_elapsed + final_elapsed + polish_elapsed
                    selected_quality = _finite_float(selected.get("quality"))
                    selected_pressure = _finite_float(
                        selected.get("max_doc_weight_ratio")
                    )
                    best10_quality = (
                        _finite_float(best10.get("quality")) if best10 else None
                    )
                    best10_pressure = (
                        _finite_float(best10.get("max_doc_weight_ratio"))
                        if best10
                        else None
                    )
                    best_conv_quality = (
                        _finite_float(best_convergence.get("quality"))
                        if best_convergence
                        else None
                    )
                    best_conv_pressure = (
                        _finite_float(best_convergence.get("max_doc_weight_ratio"))
                        if best_convergence
                        else None
                    )
                    output.append(
                        {
                            "sample": sample,
                            "stage1_requested_n_iterations": stage1_label,
                            "promote_top_k": top_k,
                            "final_requested_n_iterations": "10",
                            "polish_top1_convergence": polish_convergence,
                            "n_stage1_candidates": len(stage1_rows),
                            "n_promoted": len(promoted),
                            "selected_seed": int(selected.get("seed")),
                            "selected_randomness": float(selected.get("randomness")),
                            "selected_requested_n_iterations": _iteration_label(selected),
                            "selected_quality": selected_quality,
                            "selected_max_doc_weight_ratio": selected_pressure,
                            "best10_quality": best10_quality,
                            "best10_max_doc_weight_ratio": best10_pressure,
                            "best_convergence_quality": best_conv_quality,
                            "best_convergence_max_doc_weight_ratio": best_conv_pressure,
                            "best10_quality_recovery_ratio": (
                                selected_quality / best10_quality
                                if selected_quality is not None
                                and best10_quality not in (None, 0.0)
                                else None
                            ),
                            "best_convergence_quality_recovery_ratio": (
                                selected_quality / best_conv_quality
                                if selected_quality is not None
                                and best_conv_quality not in (None, 0.0)
                                else None
                            ),
                            "quality_gap_to_best10": (
                                best10_quality - selected_quality
                                if best10_quality is not None
                                and selected_quality is not None
                                else None
                            ),
                            "quality_gap_to_best_convergence": (
                                best_conv_quality - selected_quality
                                if best_conv_quality is not None
                                and selected_quality is not None
                                else None
                            ),
                            "pressure_delta_vs_best10": (
                                selected_pressure - best10_pressure
                                if selected_pressure is not None
                                and best10_pressure is not None
                                else None
                            ),
                            "pressure_delta_vs_best_convergence": (
                                selected_pressure - best_conv_pressure
                                if selected_pressure is not None
                                and best_conv_pressure is not None
                                else None
                            ),
                            "pressure_worse_vs_best10": (
                                selected_pressure > best10_pressure
                                if selected_pressure is not None
                                and best10_pressure is not None
                                else None
                            ),
                            "pressure_worse_vs_best_convergence": (
                                selected_pressure > best_conv_pressure
                                if selected_pressure is not None
                                and best_conv_pressure is not None
                                else None
                            ),
                            "estimated_elapsed_sec": estimated_elapsed,
                            "full10_elapsed_sec": full10_elapsed,
                            "full_convergence_elapsed_sec": full_convergence_elapsed,
                            "estimated_elapsed_saving_vs_full10_ratio": (
                                1.0 - estimated_elapsed / full10_elapsed
                                if full10_elapsed > 0.0
                                else None
                            ),
                            "estimated_elapsed_saving_vs_full_convergence_ratio": (
                                1.0 - estimated_elapsed / full_convergence_elapsed
                                if full_convergence_elapsed > 0.0
                                else None
                            ),
                            "missed_best10": (
                                (best10_quality - selected_quality) > 1.0e-9
                                if best10_quality is not None
                                and selected_quality is not None
                                else None
                            ),
                        }
                    )
    return output


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010804000000b51c0c02"
            "0000000b4944415478da63fcff1f0003030200efbfa7db0000000049454e44ae426082"
        )
    )


def _write_budget_line_plot(
    path: Path,
    group_rows: list[dict[str, Any]],
    *,
    y_field: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in group_rows:
        by_sample.setdefault(str(row.get("sample")), []).append(row)

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for sample, sample_rows in sorted(by_sample.items()):
        by_label: dict[str, list[float]] = {}
        for row in sample_rows:
            value = _finite_float(row.get(y_field))
            if value is not None:
                by_label.setdefault(str(row.get("requested_n_iterations")), []).append(value)
        labels = sorted(by_label, key=_iteration_sort_key)
        if not labels:
            continue
        x_values = list(range(len(labels)))
        y_values = [_mean(by_label[label]) for label in labels]
        ax.plot(x_values, y_values, marker="o", linewidth=1.5, label=sample)
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels)
        plotted = True
    ax.set_xlabel("requested n_iterations")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_policy_scatter_plot(path: Path, policy_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    points = [
        (
            _finite_float(row.get("estimated_elapsed_sec")),
            _finite_float(row.get("best10_quality_recovery_ratio")),
            str(row.get("sample")),
        )
        for row in policy_rows
    ]
    points = [(x, y, sample) for x, y, sample in points if x is not None and y is not None]
    samples = sorted({sample for _, _, sample in points})
    for sample in samples:
        sample_points = [(x, y) for x, y, point_sample in points if point_sample == sample]
        ax.scatter(
            [x for x, _ in sample_points],
            [y for _, y in sample_points],
            s=30,
            alpha=0.75,
            label=sample,
        )
    ax.set_xlabel("estimated elapsed sec")
    ax.set_ylabel("best-10 quality recovery ratio")
    ax.grid(True, alpha=0.25)
    if samples:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_iteration_budget_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    by_run_rows = _iteration_budget_by_run_rows(rows)
    by_group_rows = _iteration_budget_by_group_rows(rows)
    policy_rows = _shortlist_policy_simulation_rows(rows)
    by_run_path = output_dir / ITERATION_BUDGET_BY_RUN_FILENAME
    by_group_path = output_dir / ITERATION_BUDGET_BY_GROUP_FILENAME
    policy_path = output_dir / SHORTLIST_POLICY_SIMULATION_FILENAME
    quality_plot_path = output_dir / QUALITY_BY_BUDGET_PLOT_FILENAME
    pressure_plot_path = output_dir / PRESSURE_BY_BUDGET_PLOT_FILENAME
    elapsed_plot_path = output_dir / ELAPSED_BY_BUDGET_PLOT_FILENAME
    recovery_plot_path = output_dir / RECOVERY_BY_ELAPSED_PLOT_FILENAME

    _write_csv(by_run_path, by_run_rows, base_fields=[])
    _write_csv(by_group_path, by_group_rows, base_fields=[])
    _write_csv(policy_path, policy_rows, base_fields=[])
    _write_budget_line_plot(
        quality_plot_path,
        by_group_rows,
        y_field="mean_quality",
        ylabel="mean quality",
    )
    _write_budget_line_plot(
        pressure_plot_path,
        by_group_rows,
        y_field="mean_max_doc_weight_ratio",
        ylabel="mean max doc weight ratio",
    )
    _write_budget_line_plot(
        elapsed_plot_path,
        by_group_rows,
        y_field="median_elapsed_sec",
        ylabel="median elapsed sec",
    )
    _write_policy_scatter_plot(recovery_plot_path, policy_rows)
    return {
        "iteration_budget_by_run": str(by_run_path),
        "iteration_budget_by_group": str(by_group_path),
        "shortlist_policy_simulation": str(policy_path),
        "quality_vs_iteration_budget": str(quality_plot_path),
        "pressure_vs_iteration_budget": str(pressure_plot_path),
        "elapsed_vs_iteration_budget": str(elapsed_plot_path),
        "quality_recovery_vs_elapsed_budget": str(recovery_plot_path),
    }


def _write_report(path: Path, rows: list[dict[str, Any]], trace_payload: dict[str, Any]) -> None:
    best = _best_rows_by_sample(rows)
    frontiers = _time_quality_frontier(rows)
    mismatch_count = _basin_mismatch_count(rows)
    policy_rows = _shortlist_policy_simulation_rows(rows)
    lines = [
        "# Leiden Iteration Budget Profile",
        "",
        f"- Runs: {len(rows)}",
        f"- Quality trace checkpoints: {trace_payload.get('n_checkpoints')}",
        f"- Basin mismatch rows: {mismatch_count}",
        "",
        "## Best Final Quality By Sample",
        "",
        "| sample | n_iterations | seed | randomness | quality | max_doc_weight_ratio | n_above_max_doc_weight | elapsed_sec | quality_gain_per_sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample, row in sorted(best.items()):
        lines.append(
            "| {sample} | {budget} | {seed} | {randomness:g} | {quality:.6f} | {pressure:.6f} | {above} | {elapsed:.3f} | {gain} |".format(
                sample=sample,
                budget=_iteration_label(row),
                seed=int(row.get("seed")),
                randomness=float(row.get("randomness")),
                quality=float(row.get("quality")),
                pressure=float(row.get("max_doc_weight_ratio")),
                above=int(row.get("n_above_max_doc_weight") or 0),
                elapsed=float(row.get("elapsed_sec") or 0.0),
                gain=""
                if row.get("quality_gain_per_sec") is None
                else f"{float(row['quality_gain_per_sec']):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Fixed-10 Vs Convergence",
            "",
            "| sample | best_fixed10_seed | best_fixed10_randomness | best_fixed10_quality | best_fixed10_pressure | best_convergence_seed | best_convergence_randomness | best_convergence_quality | best_convergence_pressure |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample in sorted({str(row.get("sample")) for row in rows if row.get("supported")}):
        sample_rows = _sample_rows_by_iteration(rows, sample)
        best10 = _best_quality_row(sample_rows.get("10", []))
        best_convergence = _best_quality_row(sample_rows.get("convergence", []))
        lines.append(
            "| {sample} | {s10} | {r10} | {q10} | {p10} | {sc} | {rc} | {qc} | {pc} |".format(
                sample=sample,
                s10="" if best10 is None else int(best10.get("seed")),
                r10="" if best10 is None else f"{float(best10.get('randomness')):g}",
                q10="" if best10 is None else f"{float(best10.get('quality')):.6f}",
                p10=""
                if best10 is None
                else f"{float(best10.get('max_doc_weight_ratio')):.6f}",
                sc="" if best_convergence is None else int(best_convergence.get("seed")),
                rc=""
                if best_convergence is None
                else f"{float(best_convergence.get('randomness')):g}",
                qc=""
                if best_convergence is None
                else f"{float(best_convergence.get('quality')):.6f}",
                pc=""
                if best_convergence is None
                else f"{float(best_convergence.get('max_doc_weight_ratio')):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Best Shortlist Policy",
            "",
            "| sample | stage1_iter | top_k | polish_convergence | recovery_vs_best10 | gap_to_best10 | elapsed_saving_vs_full10 | pressure_delta_vs_best10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_policy_sample: dict[str, list[dict[str, Any]]] = {}
    for row in policy_rows:
        by_policy_sample.setdefault(str(row.get("sample")), []).append(row)
    for sample, sample_policy_rows in sorted(by_policy_sample.items()):
        best_policy = min(
            sample_policy_rows,
            key=lambda row: (
                -(_finite_float(row.get("best10_quality_recovery_ratio")) or -math.inf),
                _finite_float(row.get("quality_gap_to_best10")) or math.inf,
                -(
                    _finite_float(row.get("estimated_elapsed_saving_vs_full10_ratio"))
                    or -math.inf
                ),
            ),
        )
        lines.append(
            "| {sample} | {stage1} | {top_k} | {polish} | {recovery} | {gap} | {saving} | {pressure_delta} |".format(
                sample=sample,
                stage1=best_policy.get("stage1_requested_n_iterations"),
                top_k=best_policy.get("promote_top_k"),
                polish=best_policy.get("polish_top1_convergence"),
                recovery=""
                if best_policy.get("best10_quality_recovery_ratio") is None
                else f"{float(best_policy['best10_quality_recovery_ratio']):.6f}",
                gap=""
                if best_policy.get("quality_gap_to_best10") is None
                else f"{float(best_policy['quality_gap_to_best10']):.6f}",
                saving=""
                if best_policy.get("estimated_elapsed_saving_vs_full10_ratio") is None
                else f"{float(best_policy['estimated_elapsed_saving_vs_full10_ratio']):.3f}",
                pressure_delta=""
                if best_policy.get("pressure_delta_vs_best10") is None
                else f"{float(best_policy['pressure_delta_vs_best10']):.6f}",
            )
        )
    lines.extend(["", "## Time-Quality Frontier", ""])
    for sample, frontier in sorted(frontiers.items()):
        lines.extend(
            [
                f"### {sample}",
                "",
                "| n_iterations | seed | randomness | quality | elapsed_sec | max_doc_weight_ratio | time_to_95pct_gain_ms |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in frontier[:10]:
            time_to_95 = row.get("time_to_95pct_final_quality_gain_ms")
            lines.append(
                "| {budget} | {seed} | {randomness:g} | {quality:.6f} | {elapsed:.3f} | {pressure:.6f} | {time_to_95} |".format(
                    budget=_iteration_label(row),
                    seed=int(row.get("seed")),
                    randomness=float(row.get("randomness")),
                    quality=float(row.get("quality")),
                    elapsed=float(row.get("elapsed_sec") or 0.0),
                    pressure=float(row.get("max_doc_weight_ratio")),
                    time_to_95=""
                    if time_to_95 is None
                    else f"{float(time_to_95):.3f}",
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_profile(
    *,
    summaries: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    randomness_values: tuple[float, ...],
    n_iterations: int | None = None,
    n_iterations_values: tuple[IterationBudget | str | int, ...] | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if n_iterations_values is None:
        n_iterations_values = (
            (_parse_n_iterations_value(n_iterations),)
            if n_iterations is not None
            else tuple(
                _parse_n_iterations_value(value)
                for value in DEFAULT_N_ITERATIONS_VALUES
            )
        )
    else:
        n_iterations_values = tuple(
            budget
            if isinstance(budget, IterationBudget)
            else _parse_n_iterations_value(budget)
            for budget in n_iterations_values
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_jsonl_path = output_dir / ROWS_JSONL_FILENAME
    rows_csv_path = output_dir / ROWS_CSV_FILENAME
    trace_path = output_dir / QUALITY_TRACE_FILENAME
    trace_runs_path = output_dir / QUALITY_TRACE_RUNS_FILENAME
    trace_summary_dir = output_dir / "quality_trace_summary"
    summary_path = output_dir / SUMMARY_FILENAME
    report_path = output_dir / REPORT_FILENAME

    if not resume:
        for path in (rows_jsonl_path, rows_csv_path, trace_runs_path):
            if path.exists():
                path.unlink()

    rows: list[dict[str, Any]] = []
    run_count = 0
    with _quality_trace_path_context(trace_path, resume=resume):
        for summary_path_input in summaries:
            profile = _resolve_profile_input(summary_path_input)
            n_nodes = pilot._infer_n_nodes(profile.input_cfg)
            node_weights = pilot._load_node_weights(profile.input_cfg.node_weights_path, n_nodes)
            graph = pilot._load_graph(profile.input_cfg, node_weights)
            for seed in seeds:
                for randomness in randomness_values:
                    for budget in n_iterations_values:
                        if limit is not None and run_count >= limit:
                            break
                        row = _run_one(
                            profile=profile,
                            graph=graph,
                            node_weights=node_weights,
                            seed=int(seed),
                            randomness=float(randomness),
                            budget=budget,
                            trace_path=trace_path,
                            trace_runs_path=trace_runs_path,
                        )
                        rows.append(row)
                        _append_jsonl(rows_jsonl_path, row)
                        run_count += 1
                    if limit is not None and run_count >= limit:
                        break
                if limit is not None and run_count >= limit:
                    break
            if limit is not None and run_count >= limit:
                break

    trace_payload = quality_summary.summarize_quality_trace(
        trace_path=trace_path,
        runs_path=trace_runs_path,
        output_dir=trace_summary_dir,
        group_fields=("sample", "variant", "seed", "randomness", "requested_n_iterations"),
    )
    _enrich_rows_with_trace_summary(rows, Path(trace_payload["paths"]["by_run"]))
    _enrich_rows_with_iteration_budget_metrics(rows)
    _write_jsonl(rows_jsonl_path, rows)
    _write_csv(rows_csv_path, rows)
    iteration_budget_paths = _write_iteration_budget_outputs(
        output_dir=output_dir,
        rows=rows,
    )

    expected_runs = (
        len(summaries)
        * len(seeds)
        * len(randomness_values)
        * len(n_iterations_values)
    )
    quality_mismatches = [
        row for row in rows if not row.get("quality_recompute_ok") or not row.get("trace_quality_matches_result")
    ]
    payload = {
        "schema": f"leiden_random_refinement_profile.v{SCHEMA_VERSION}",
        "output_dir": str(output_dir),
        "grid": {
            "variant": VARIANT_STANDARD,
            "seeds": list(seeds),
            "randomness_values": list(randomness_values),
            "n_iterations_values": [
                {
                    "requested_n_iterations": budget.requested,
                    "iteration_mode": budget.mode,
                    "n_iterations": int(budget.n_iterations),
                }
                for budget in n_iterations_values
            ],
        },
        "n_expected_runs": int(expected_runs),
        "n_rows": len(rows),
        "n_quality_or_trace_mismatches": len(quality_mismatches),
        "n_runs_with_start_and_final_trace": sum(
            1 for row in rows if row.get("trace_has_start") and row.get("trace_has_final")
        ),
        "best_by_sample": {
            sample: {
                "requested_n_iterations": _iteration_label(row),
                "iteration_mode": row.get("iteration_mode"),
                "n_iterations": int(row["n_iterations"]),
                "n_iterations_used": (
                    None
                    if row.get("n_iterations_used") is None
                    else int(row["n_iterations_used"])
                ),
                "seed": int(row["seed"]),
                "randomness": float(row["randomness"]),
                "quality": float(row["quality"]),
                "max_doc_weight_ratio": float(row["max_doc_weight_ratio"]),
                "n_above_max_doc_weight": int(row["n_above_max_doc_weight"]),
                "elapsed_sec": float(row["elapsed_sec"]),
            }
            for sample, row in _best_rows_by_sample(rows).items()
        },
        "basin_mismatch_count": _basin_mismatch_count(rows),
        "paths": {
            "rows_jsonl": str(rows_jsonl_path),
            "rows_csv": str(rows_csv_path),
            "quality_trace": str(trace_path),
            "quality_trace_runs": str(trace_runs_path),
            "quality_trace_summary": trace_payload["paths"]["summary"],
            "summary": str(summary_path),
            "report": str(report_path),
            **iteration_budget_paths,
        },
    }
    _write_json(summary_path, payload)
    _write_report(report_path, rows, trace_payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        dest="summaries",
        action="append",
        type=Path,
        help="Prepared summary JSON. Repeat for multiple graph layers.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated Leiden seeds.",
    )
    parser.add_argument(
        "--randomness-values",
        default=",".join(str(value) for value in DEFAULT_RANDOMNESS),
        help="Comma-separated refinement randomness values.",
    )
    parser.add_argument(
        "--n-iterations-values",
        help=(
            "Comma-separated iteration budgets. Use convergence to pass "
            "n_iterations=0 to the Rust backend. Default: "
            + ",".join(DEFAULT_N_ITERATIONS_VALUES)
        ),
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        help=(
            "Backward-compatible single iteration budget. If omitted, "
            "--n-iterations-values/default grid is used."
        ),
    )
    parser.add_argument("--limit", type=int, help="Optional maximum number of runs.")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summaries = tuple(args.summaries) if args.summaries else DEFAULT_SUMMARIES
    if args.n_iterations_values:
        n_iterations_values = _parse_n_iterations_values(args.n_iterations_values)
    elif args.n_iterations is not None:
        n_iterations_values = (_parse_n_iterations_value(args.n_iterations),)
    else:
        n_iterations_values = _parse_n_iterations_values(
            ",".join(DEFAULT_N_ITERATIONS_VALUES)
        )
    payload = run_profile(
        summaries=summaries,
        output_dir=args.output_dir,
        seeds=tuple(int(value) for value in _parse_csv_tuple(args.seeds, cast=int)),
        randomness_values=tuple(
            float(value) for value in _parse_csv_tuple(args.randomness_values, cast=float)
        ),
        n_iterations_values=n_iterations_values,
        limit=args.limit,
        resume=bool(args.resume),
    )
    print(f"Saved profile summary to {payload['paths']['summary']}")
    print(f"Saved report to {payload['paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
