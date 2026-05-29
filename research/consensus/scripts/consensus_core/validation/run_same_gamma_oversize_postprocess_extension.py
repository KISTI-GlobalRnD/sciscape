"""Run same-gamma oversize postprocess extension experiments.

This runner reuses prepared source-seed memberships and graph sidecars from the
hierarchy postprocess validation directory.  It keeps the original CPM gamma as
the acceptance objective while comparing iterative split-repair and conservative
boundary-trim variants against the current two-stage quality-first baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
import sciscape_leiden

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
from run_hierarchy_postprocess_seed_sweep import _parse_int_list  # noqa: E402
from scripts.run_adaptive_split_merge_repair_probe import (  # noqa: E402
    _run_iterative_apply,
)

DEFAULT_OUTPUT_DIR_EXTENSION = DEFAULT_OUTPUT_DIR / "same_gamma_oversize_extension"
DEFAULT_FIELDS = (12, 26, 30)
DEFAULT_SOURCE_SEEDS = (11, 42, 73)
DEFAULT_VARIANTS = (
    "iterative_quality_first",
    "iterative_quality_first_plus_trim",
    "target_reaching_trim_diagnostic",
)
DEFAULT_GAMMA_MULTIPLIERS = (1.02, 1.05, 1.10, 1.15, 1.20, 1.25)
CACHE_SCHEMA_VERSION = 1

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

@dataclass(frozen=True)

class VariantConfig:
    name: str
    acceptance_mode: str
    apply_iterations: int
    apply_trim: bool
    trim_min_delta_q: float
    trim_max_moves_per_cluster: int
    singleton_budget: float
    membership_role: str

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

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
    for root in _source_seed_roots(validation_dir):
        for summary_path in sorted(root.glob(f"field*{field}*/seed_{source_seed}/prepare_summary.json")):
            summary = _read_json(summary_path)
            sample = str(summary.get("sample") or summary_path.parents[1].name)
            if _field_from_sample(sample) != int(field):
                continue
            paths = summary.get("paths", {})
            graph_dir = _repo_path(paths.get("graph_dir")) or summary_path.parents[2] / sample / "graph"
            membership_path = _repo_path(paths.get("membership")) or summary_path.parent / "membership.parquet"
            if graph_dir is None or membership_path is None:
                continue
            node_weights_path = graph_dir / "node_weights.f64.bin"
            if not membership_path.exists():
                raise FileNotFoundError(f"Missing source membership: {membership_path}")
            if not node_weights_path.exists():
                raise FileNotFoundError(f"Missing node weights: {node_weights_path}")
            return SourceRunConfig(
                field=int(field),
                sample=sample,
                source_seed=int(source_seed),
                prepare_summary_path=summary_path,
                graph_dir=graph_dir,
                membership_path=membership_path,
                node_weights_path=node_weights_path,
                resolution=float(summary.get("resolution") or 0.01),
                target_min_doc_weight=float(summary.get("target_min_doc_weight") or 50.0),
                target_max_doc_weight=float(summary.get("target_max_doc_weight") or 0.0),
                n_nodes=int(summary.get("n_nodes") or 0),
            )
    raise FileNotFoundError(
        f"No prepared source-seed run for field={field}, source_seed={source_seed}"
    )

def _load_graph(config: SourceRunConfig):
    n_nodes = int(config.n_nodes)
    if n_nodes <= 0:
        n_nodes = int(_load_membership(config.membership_path).shape[0])
    return sciscape_leiden.load_graph_raw_files(
        n_nodes,
        str(config.graph_dir / "src.u32.bin"),
        str(config.graph_dir / "dst.u32.bin"),
        str(config.graph_dir / "weight.f64.bin"),
        str(config.node_weights_path),
    )

def _membership_metrics(
    membership: np.ndarray,
    node_weights: np.ndarray,
    *,
    target_max_doc_weight: float,
) -> dict[str, Any]:
    weights = _cluster_weights(np.asarray(membership, dtype=np.int64), node_weights)
    target = float(target_max_doc_weight)
    sorted_desc = np.sort(weights)[::-1]
    oversize_excess = (
        float(np.maximum(0.0, weights - target).sum()) if target > 0.0 else 0.0
    )
    total = float(weights.sum()) if weights.size else 0.0
    return {
        "n_clusters": int(weights.size),
        "max_doc_weight": float(sorted_desc[0]) if sorted_desc.size else 0.0,
        "target_max_doc_weight": target,
        "max_doc_weight_ratio": (
            float(sorted_desc[0] / target) if sorted_desc.size and target > 0.0 else 0.0
        ),
        "n_above_max_doc_weight": int((weights > target).sum()) if target > 0.0 else 0,
        "oversize_excess_mass": oversize_excess,
        "gini_doc_weight": _gini(weights),
        "entropy_doc_weight": _normalized_entropy(weights),
        "top1_doc_weight_share": float(sorted_desc[:1].sum() / total) if total else 0.0,
        "top5_doc_weight_share": float(sorted_desc[:5].sum() / total) if total else 0.0,
        "target_max_satisfied": bool(target <= 0.0 or not np.any(weights > target)),
    }

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

def _variant_config(
    name: str,
    *,
    apply_iterations: int,
    trim_max_moves_per_cluster: int,
    singleton_budget: float,
) -> VariantConfig:
    if name == "iterative_quality_first":
        return VariantConfig(
            name=name,
            acceptance_mode="quality_first",
            apply_iterations=int(apply_iterations),
            apply_trim=False,
            trim_min_delta_q=0.0,
            trim_max_moves_per_cluster=0,
            singleton_budget=float(singleton_budget),
            membership_role="effective",
        )
    if name == "iterative_quality_first_plus_trim":
        return VariantConfig(
            name=name,
            acceptance_mode="quality_first",
            apply_iterations=int(apply_iterations),
            apply_trim=True,
            trim_min_delta_q=0.0,
            trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
            singleton_budget=float(singleton_budget),
            membership_role="effective",
        )
    if name == "target_reaching_trim_diagnostic":
        return VariantConfig(
            name=name,
            acceptance_mode="hard_cap",
            apply_iterations=int(apply_iterations),
            apply_trim=True,
            trim_min_delta_q=-1.0,
            trim_max_moves_per_cluster=int(trim_max_moves_per_cluster),
            singleton_budget=float(singleton_budget),
            membership_role="diagnostic",
        )
    raise ValueError(f"Unsupported variant: {name}")

def _args_for_variant(
    *,
    config: SourceRunConfig,
    variant: VariantConfig,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        graph_dir=None,
        membership=None,
        candidates=None,
        output_dir=None,
        resolution=float(config.resolution),
        gamma_multipliers=",".join(str(x) for x in DEFAULT_GAMMA_MULTIPLIERS),
        min_core_weight=25.0,
        randomness=0.01,
        repair_epsilon=0.0,
        seed=int(seed),
        pair_seeded_probes=False,
        policy="",
        max_candidates=1000,
        target_min_doc_weight=float(config.target_min_doc_weight),
        target_max_doc_weight=float(config.target_max_doc_weight),
        oversize_acceptance_mode=variant.acceptance_mode,
        selection_mode="oversize_first",
        selection_singleton_budget=float(variant.singleton_budget),
        selection_max_selected=0,
        apply_split_repair_candidates=True,
        apply_iterations=int(variant.apply_iterations),
        applied_membership_output=None,
        apply_min_quality_delta=0.0,
        apply_oversize_boundary_trim=bool(variant.apply_trim),
        trim_min_delta_q=float(variant.trim_min_delta_q),
        trim_min_delta_q_source="variant_default",
        trim_max_moves_per_cluster=int(variant.trim_max_moves_per_cluster),
    )

def _cache_metadata(
    *,
    config: SourceRunConfig,
    variant: VariantConfig,
    gamma_multipliers: tuple[float, ...],
) -> dict[str, Any]:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "field": int(config.field),
        "sample": config.sample,
        "source_seed": int(config.source_seed),
        "resolution": float(config.resolution),
        "target_min_doc_weight": float(config.target_min_doc_weight),
        "target_max_doc_weight": float(config.target_max_doc_weight),
        "variant": variant.__dict__,
        "gamma_multipliers": [float(x) for x in gamma_multipliers],
        "membership_input": _file_fingerprint(config.membership_path),
        "node_weights_input": _file_fingerprint(config.node_weights_path),
        "graph_inputs": [
            _file_fingerprint(config.graph_dir / "src.u32.bin"),
            _file_fingerprint(config.graph_dir / "dst.u32.bin"),
            _file_fingerprint(config.graph_dir / "weight.f64.bin"),
        ],
    }
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": _hash_json(payload),
        "payload": payload,
    }

def _cache_matches(summary: dict[str, Any], expected: dict[str, Any]) -> bool:
    observed = summary.get("extension_cache_metadata")
    return (
        isinstance(observed, dict)
        and observed.get("schema_version") == expected.get("schema_version")
        and observed.get("cache_key") == expected.get("cache_key")
    )

def _summary_membership_path(summary: dict[str, Any]) -> Path | None:
    paths = summary.get("paths", {})
    raw = (
        paths.get("applied_membership")
        or paths.get("diagnostic_membership")
        or paths.get("trim_membership")
    )
    return _repo_path(raw)

def _effect_row(
    *,
    config: SourceRunConfig,
    variant_name: str,
    summary: dict[str, Any] | None,
    membership: np.ndarray,
    node_weights: np.ndarray,
    before_metrics: dict[str, Any],
    membership_path: Path,
    membership_role: str,
    cache_key: str | None,
) -> dict[str, Any]:
    final_metrics = _membership_metrics(
        membership,
        node_weights,
        target_max_doc_weight=float(config.target_max_doc_weight),
    )
    exact_delta_q = float((summary or {}).get("exact_delta_q_total", 0.0))
    status = str((summary or {}).get("status", "committed"))
    if summary is None:
        status = "small_only"
    accepted = bool(exact_delta_q >= -1e-9)
    if membership_role == "diagnostic":
        accepted = bool(accepted and final_metrics["target_max_satisfied"])
    return {
        "field": int(config.field),
        "sample": config.sample,
        "source_seed": int(config.source_seed),
        "policy": variant_name,
        "membership_role": membership_role,
        "status": status,
        "accepted_for_contraction": accepted,
        "fallback_used": bool(membership_role == "effective" and not accepted),
        "delta_q": exact_delta_q,
        "split_repair_exact_delta_q": float((summary or {}).get("split_repair_exact_delta_q", 0.0)),
        "trim_exact_delta_q": float((summary or {}).get("trim_exact_delta_q", 0.0)),
        "n_iterations_run": int((summary or {}).get("n_iterations_run", 0)),
        "n_committed_iterations": int((summary or {}).get("n_committed_iterations", 0)),
        "trim_committed": bool((summary or {}).get("trim_committed", False)),
        "changed_nodes_vs_initial": int((summary or {}).get("changed_nodes_vs_initial", 0)),
        "changed_nodes_step_sum": int((summary or {}).get("changed_nodes_step_sum", 0)),
        "stop_reason": (summary or {}).get("stop_reason"),
        "membership_path": _rel(membership_path),
        "source_membership_path": _rel(config.membership_path),
        "initial_max_doc_weight": before_metrics["max_doc_weight"],
        "max_doc_weight": final_metrics["max_doc_weight"],
        "target_max_doc_weight": final_metrics["target_max_doc_weight"],
        "max_doc_weight_ratio": final_metrics["max_doc_weight_ratio"],
        "delta_max_doc_weight": final_metrics["max_doc_weight"] - before_metrics["max_doc_weight"],
        "n_above_max_doc_weight": final_metrics["n_above_max_doc_weight"],
        "delta_oversize_count": (
            final_metrics["n_above_max_doc_weight"]
            - before_metrics["n_above_max_doc_weight"]
        ),
        "oversize_excess_mass": final_metrics["oversize_excess_mass"],
        "delta_oversize_excess_mass": (
            final_metrics["oversize_excess_mass"]
            - before_metrics["oversize_excess_mass"]
        ),
        "gini_doc_weight": final_metrics["gini_doc_weight"],
        "delta_gini_doc_weight": final_metrics["gini_doc_weight"] - before_metrics["gini_doc_weight"],
        "entropy_doc_weight": final_metrics["entropy_doc_weight"],
        "target_max_satisfied": final_metrics["target_max_satisfied"],
        "cache_key": cache_key,
    }

def _flatten_pass_rows(
    *,
    config: SourceRunConfig,
    variant: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("iterations", []):
        before = item.get("before_membership", {})
        after = item.get("after_membership", {})
        rows.append(
            {
                "field": int(config.field),
                "sample": config.sample,
                "source_seed": int(config.source_seed),
                "policy": variant,
                "iteration": int(item.get("iteration", 0)),
                "status": str(item.get("status", "")),
                "candidate_clusters": int(item.get("candidate_clusters", 0)),
                "n_selected": int(item.get("n_selected", 0)),
                "n_applied": int(item.get("n_applied", 0)),
                "exact_delta_q": float(item.get("exact_delta_q", 0.0)),
                "predicted_delta_q_sum": float(item.get("predicted_delta_q_sum", 0.0)),
                "changed_nodes": int(item.get("changed_nodes", 0)),
                "before_max_doc_weight": float(before.get("max_doc_weight", 0.0)),
                "after_max_doc_weight": float(after.get("max_doc_weight", 0.0)),
                "delta_max_doc_weight": float(after.get("max_doc_weight", 0.0))
                - float(before.get("max_doc_weight", 0.0)),
                "before_n_above_max_doc_weight": int(before.get("n_above_max_doc_weight", 0)),
                "after_n_above_max_doc_weight": int(after.get("n_above_max_doc_weight", 0)),
            }
        )
    return rows

def _candidate_rows_from_summary(
    *,
    config: SourceRunConfig,
    variant: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("iterations", []):
        paths = item.get("paths", {})
        selection_path = _repo_path(paths.get("selection_candidates"))
        if selection_path is None or not selection_path.exists():
            continue
        frame = pd.read_csv(selection_path)
        if frame.empty:
            continue
        frame["field"] = int(config.field)
        frame["sample"] = config.sample
        frame["source_seed"] = int(config.source_seed)
        frame["extension_policy"] = variant
        frame["iteration"] = int(item.get("iteration", 0))
        front = ["field", "sample", "source_seed", "extension_policy", "iteration"]
        frame = frame[[*front, *[column for column in frame.columns if column not in front]]]
        rows.extend(frame.to_dict("records"))
    return rows

def _run_variant(
    *,
    graph: Any,
    config: SourceRunConfig,
    source_membership: np.ndarray,
    node_weights: np.ndarray,
    variant: VariantConfig,
    output_dir: Path,
    gamma_multipliers: tuple[float, ...],
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = (
        output_dir
        / "source_seed_runs"
        / _safe_slug(config.sample)
        / f"seed_{config.source_seed}"
        / variant.name
    )
    summary_path = run_dir / "iterative_split_repair_apply_summary.json"
    metadata = _cache_metadata(
        config=config,
        variant=variant,
        gamma_multipliers=gamma_multipliers,
    )
    if summary_path.exists() and not force:
        summary = _read_json(summary_path)
        if not _cache_matches(summary, metadata):
            summary = {}
    else:
        summary = {}
    if not summary:
        args = _args_for_variant(config=config, variant=variant, seed=config.source_seed)
        summary = _run_iterative_apply(
            graph,
            source_membership,
            node_weights,
            np.asarray(gamma_multipliers, dtype=np.float64),
            run_dir,
            args,
            [],
        )
        summary["extension_cache_metadata"] = metadata
        summary["extension_variant"] = variant.__dict__
        _write_json(summary_path, summary)

    membership_path = _summary_membership_path(summary) or config.membership_path
    final_membership = (
        _load_membership(membership_path).astype(np.uint64, copy=False)
        if membership_path != config.membership_path
        else source_membership
    )
    before_metrics = _membership_metrics(
        source_membership,
        node_weights,
        target_max_doc_weight=float(config.target_max_doc_weight),
    )
    row = _effect_row(
        config=config,
        variant_name=variant.name,
        summary=summary,
        membership=final_membership,
        node_weights=node_weights,
        before_metrics=before_metrics,
        membership_path=membership_path,
        membership_role=variant.membership_role,
        cache_key=metadata["cache_key"],
    )
    return (
        row,
        _flatten_pass_rows(config=config, variant=variant.name, summary=summary),
        _candidate_rows_from_summary(config=config, variant=variant.name, summary=summary),
    )

def _current_quality_first(validation_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    paths = [
        validation_dir / "source_seed_sweep_effects.csv",
        validation_dir / "field_expansion_source_seed_effects.csv",
    ]
    for priority, path in enumerate(paths):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["_priority"] = int(priority)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "seed" in df.columns and "source_seed" not in df.columns:
        df["source_seed"] = df["seed"]
    df["field"] = df["sample"].map(_field_from_sample)
    qf = df[
        (df["policy"] == "two_stage_quality_first")
        & (df["membership_role"] == "effective")
    ].copy()
    qf = (
        qf.sort_values("_priority")
        .drop_duplicates(subset=["sample", "source_seed", "policy"], keep="last")
        .reset_index(drop=True)
    )
    return qf.drop(columns=["_priority"], errors="ignore")

def _compare_vs_current(effects: pd.DataFrame, validation_dir: Path) -> pd.DataFrame:
    current = _current_quality_first(validation_dir)
    if effects.empty or current.empty:
        return pd.DataFrame()
    extensions = effects[effects["policy"] != "small_only"].copy()
    extensions = extensions[extensions["membership_role"].isin(["effective", "diagnostic"])]
    columns = [
        "field",
        "sample",
        "source_seed",
        "policy",
        "membership_role",
        "status",
        "delta_q",
        "max_doc_weight",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
        "oversize_excess_mass",
        "gini_doc_weight",
        "target_max_satisfied",
    ]
    current_columns = [
        "sample",
        "source_seed",
        "delta_q",
        "max_doc_weight",
        "max_doc_weight_ratio",
        "n_above_max_doc_weight",
        "gini_doc_weight",
        "target_max_satisfied",
    ]
    merged = extensions[columns].merge(
        current[current_columns],
        on=["sample", "source_seed"],
        how="inner",
        suffixes=("_extension", "_current_quality_first"),
    )
    if merged.empty:
        return merged
    merged["delta_max_ratio_vs_current_quality_first"] = (
        merged["max_doc_weight_ratio_extension"]
        - merged["max_doc_weight_ratio_current_quality_first"]
    )
    merged["delta_oversize_count_vs_current_quality_first"] = (
        merged["n_above_max_doc_weight_extension"]
        - merged["n_above_max_doc_weight_current_quality_first"]
    )
    merged["delta_gini_vs_current_quality_first"] = (
        merged["gini_doc_weight_extension"]
        - merged["gini_doc_weight_current_quality_first"]
    )
    merged["delta_q_vs_current_quality_first"] = (
        merged["delta_q_extension"] - merged["delta_q_current_quality_first"]
    )
    return merged

def _policy_summary(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for policy, group in effects.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "n_rows": int(len(group)),
                "accepted_rate": float(group["accepted_for_contraction"].mean()),
                "mean_delta_q": float(group["delta_q"].mean()),
                "mean_max_doc_weight_ratio": float(group["max_doc_weight_ratio"].mean()),
                "mean_oversize_count": float(group["n_above_max_doc_weight"].mean()),
                "mean_oversize_excess_mass": float(group["oversize_excess_mass"].mean()),
                "mean_gini_doc_weight": float(group["gini_doc_weight"].mean()),
                "target_satisfied_rate": float(group["target_max_satisfied"].mean()),
            }
        )
    return pd.DataFrame(rows)

def _write_report(
    *,
    output_dir: Path,
    effects: pd.DataFrame,
    comparison: pd.DataFrame,
    policy_summary: pd.DataFrame,
    elapsed_sec: float,
) -> Path:
    lines = [
        "# Same-Gamma Oversize Postprocess Extension",
        "",
        "This run keeps the original CPM gamma fixed and compares iterative split-repair and boundary-trim variants against the current two-stage quality-first baseline.",
        "",
        "## Scope",
        "",
        f"- Effect rows: {len(effects)}",
        f"- Comparison rows: {len(comparison)}",
        f"- Elapsed seconds: {elapsed_sec:.3f}",
        "",
    ]
    if not policy_summary.empty:
        lines.extend(["## Policy Summary", "", _markdown_table(policy_summary), ""])
    if not comparison.empty:
        compact = (
            comparison.groupby("policy")
            .agg(
                n_rows=("policy", "count"),
                mean_delta_max_ratio=(
                    "delta_max_ratio_vs_current_quality_first",
                    "mean",
                ),
                mean_delta_oversize_count=(
                    "delta_oversize_count_vs_current_quality_first",
                    "mean",
                ),
                mean_delta_gini=("delta_gini_vs_current_quality_first", "mean"),
                mean_delta_q=("delta_q_vs_current_quality_first", "mean"),
            )
            .reset_index()
        )
        lines.extend(["## Compared With Current Quality-First", "", _markdown_table(compact), ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `iterative_quality_first_effects.csv` / `.parquet`",
            "- `iterative_quality_first_passes.csv` / `.parquet`",
            "- `iterative_quality_first_candidates.csv` / `.parquet`",
            "- `iterative_quality_first_vs_current.csv` / `.parquet`",
            "- `iterative_quality_first_policy_summary.csv` / `.parquet`",
            "- `iterative_quality_first_compute_summary.json`",
            "",
        ]
    )
    path = output_dir / "iterative_quality_first_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_EXTENSION)
    parser.add_argument("--fields", default=",".join(str(x) for x in DEFAULT_FIELDS))
    parser.add_argument("--source-seeds", default=",".join(str(x) for x in DEFAULT_SOURCE_SEEDS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--apply-iterations", type=int, default=4)
    parser.add_argument("--trim-max-moves-per-cluster", type=int, default=100)
    parser.add_argument("--selection-singleton-budget", type=float, default=100.0)
    parser.add_argument(
        "--gamma-multipliers",
        default=",".join(str(x) for x in DEFAULT_GAMMA_MULTIPLIERS),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    validation_dir = _repo_path(args.validation_dir) or args.validation_dir
    output_dir = _repo_path(args.output_dir) or args.output_dir
    assert validation_dir is not None and output_dir is not None

    fields = _parse_int_list(str(args.fields))
    source_seeds = _parse_int_list(str(args.source_seeds))
    variant_names = tuple(part.strip() for part in str(args.variants).split(",") if part.strip())
    variants = tuple(
        _variant_config(
            name,
            apply_iterations=int(args.apply_iterations),
            trim_max_moves_per_cluster=int(args.trim_max_moves_per_cluster),
            singleton_budget=float(args.selection_singleton_budget),
        )
        for name in variant_names
    )
    gamma_multipliers = _parse_float_list(str(args.gamma_multipliers))
    configs = [
        _discover_source_run(field=field, source_seed=seed, validation_dir=validation_dir)
        for field in fields
        for seed in source_seeds
    ]
    if args.dry_run:
        payload = {
            "status": "dry_run_ok",
            "n_configs": len(configs),
            "variants": [variant.__dict__ for variant in variants],
            "configs": [
                {
                    "field": config.field,
                    "sample": config.sample,
                    "source_seed": config.source_seed,
                    "graph_dir": _rel(config.graph_dir),
                    "membership_path": _rel(config.membership_path),
                    "target_max_doc_weight": config.target_max_doc_weight,
                }
                for config in configs
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    effect_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for index, config in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] same-gamma extension "
            f"{config.sample} seed={config.source_seed}",
            flush=True,
        )
        source_membership = _load_membership(config.membership_path).astype(np.uint64, copy=False)
        node_weights = _load_node_weights(config.node_weights_path, int(source_membership.shape[0]))
        before_metrics = _membership_metrics(
            source_membership,
            node_weights,
            target_max_doc_weight=float(config.target_max_doc_weight),
        )
        effect_rows.append(
            _effect_row(
                config=config,
                variant_name="small_only",
                summary=None,
                membership=source_membership,
                node_weights=node_weights,
                before_metrics=before_metrics,
                membership_path=config.membership_path,
                membership_role="effective",
                cache_key=None,
            )
        )
        graph = _load_graph(config)
        for variant in variants:
            row, passes, candidates = _run_variant(
                graph=graph,
                config=config,
                source_membership=source_membership,
                node_weights=node_weights,
                variant=variant,
                output_dir=output_dir,
                gamma_multipliers=gamma_multipliers,
                force=bool(args.force),
            )
            effect_rows.append(row)
            pass_rows.extend(passes)
            candidate_rows.extend(candidates)
        run_summaries.append(
            {
                "field": int(config.field),
                "sample": config.sample,
                "source_seed": int(config.source_seed),
                "source_max_doc_weight": before_metrics["max_doc_weight"],
                "source_n_above_max_doc_weight": before_metrics["n_above_max_doc_weight"],
                "graph_dir": _rel(config.graph_dir),
                "membership_path": _rel(config.membership_path),
            }
        )

    effects = pd.DataFrame(effect_rows)
    passes = pd.DataFrame(pass_rows)
    candidates = pd.DataFrame(candidate_rows)
    comparison = _compare_vs_current(effects, validation_dir)
    policy_summary = _policy_summary(effects)

    _write_table(effects, output_dir / "iterative_quality_first_effects")
    _write_table(passes, output_dir / "iterative_quality_first_passes")
    _write_table(candidates, output_dir / "iterative_quality_first_candidates")
    _write_table(comparison, output_dir / "iterative_quality_first_vs_current")
    _write_table(policy_summary, output_dir / "iterative_quality_first_policy_summary")

    elapsed_sec = float(time.perf_counter() - t0)
    report_path = _write_report(
        output_dir=output_dir,
        effects=effects,
        comparison=comparison,
        policy_summary=policy_summary,
        elapsed_sec=elapsed_sec,
    )
    compute_summary = {
        "status": "completed",
        "fields": list(fields),
        "source_seeds": list(source_seeds),
        "variants": [variant.__dict__ for variant in variants],
        "gamma_multipliers": [float(x) for x in gamma_multipliers],
        "same_gamma_objective": True,
        "epsilon_q": 0.0,
        "n_effect_rows": int(len(effects)),
        "n_pass_rows": int(len(passes)),
        "n_candidate_rows": int(len(candidates)),
        "n_comparison_rows": int(len(comparison)),
        "run_summaries": run_summaries,
        "elapsed_sec": elapsed_sec,
        "paths": {
            "effects": _rel(output_dir / "iterative_quality_first_effects.csv"),
            "passes": _rel(output_dir / "iterative_quality_first_passes.csv"),
            "candidates": _rel(output_dir / "iterative_quality_first_candidates.csv"),
            "comparison": _rel(output_dir / "iterative_quality_first_vs_current.csv"),
            "policy_summary": _rel(output_dir / "iterative_quality_first_policy_summary.csv"),
            "report": _rel(report_path),
        },
    }
    _write_json(output_dir / "iterative_quality_first_compute_summary.json", compute_summary)
    print(json.dumps(compute_summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
