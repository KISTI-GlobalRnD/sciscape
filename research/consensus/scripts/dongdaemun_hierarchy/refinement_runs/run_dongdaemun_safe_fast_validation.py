"""Validate the conservative Dongdaemun safe-fast preset on prepared summaries.

This runner exercises ``RustLeidenGraph.run_leiden_dongdaemun_safe_fast_refinement``
directly.  It records the pressure trigger, fallback path, quality delta versus
the internally computed standard Leiden result, and a recomputed CPM quality
sanity check for each prepared source-level graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
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

SCRIPT_DIR = Path(__file__).resolve().parent

import evaluate_dongdaemun_refinement_slice4 as pilot  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_safe_fast_validation"
)
QUALITY_RECOMPUTE_ABS_TOL = 1.0e-6
SCHEMA_VERSION = 1
SAFE_FAST_PRESET = {
    "max_extra_parents_per_iteration": 4,
    "max_extra_children_per_parent": 16,
    "parent_selection_policy": "weight",
    "gamma_multipliers": (1.02, 1.05),
    "candidate_quality_policy": "structural",
    "use_quotient_diagnostic": True,
    "allow_repair_escalation": False,
    "baseline_repair_policy": "adaptive",
}

CSV_FIELDS = [
    "sample",
    "seed",
    "summary_path",
    "supported",
    "unsupported_reason",
    "skipped_by_prepare",
    "elapsed_sec",
    "selected_variant",
    "triggered",
    "fallback_triggered",
    "fallback_reason",
    "quality",
    "standard_quality",
    "quality_delta_vs_standard",
    "quality_improved_vs_standard",
    "quality_recomputed",
    "quality_recompute_delta",
    "quality_recompute_abs_delta",
    "quality_recompute_ok",
    "n_clusters",
    "standard_max_doc_weight",
    "standard_max_doc_weight_ratio",
    "standard_n_above_max_doc_weight",
    "selected_max_doc_weight",
    "selected_max_doc_weight_ratio",
    "selected_n_above_max_doc_weight",
    "membership_equal_to_standard",
    "membership_diff_nodes_vs_standard",
    "max_extra_parents_per_iteration",
    "max_extra_children_per_parent",
    "severe_tier_triggered",
    "repair_escalated",
    "repair_escalation_accepted",
    "prepare_max_doc_weight",
    "prepare_max_doc_weight_ratio",
    "prepare_n_above_max_doc_weight",
    "prepare_triggered",
]

@dataclass(frozen=True)
class SafeFastValidationConfig:
    n_iterations: int = 10
    randomness: float = 0.01
    trigger_max_doc_weight_ratio: float | None = 1.03
    trigger_min_above_max_doc_weight: int | None = 2
    accept_max_doc_weight_ratio: float = 1.01
    accept_min_quality_delta: float | None = 0.0
    accept_min_quality_delta_ratio: float | None = None
    quality_recompute_abs_tol: float = QUALITY_RECOMPUTE_ABS_TOL
    skip_prepare_untriggered: bool = False

def _json_safe(value: Any) -> Any:
    return pilot._json_safe(value)

def _csv_value(value: Any) -> Any:
    safe = _json_safe(value)
    if safe is None:
        return ""
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, sort_keys=True, separators=(",", ":"))
    return safe

def _repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    parsed = Path(path)
    if parsed.is_absolute():
        return parsed
    return REPO_ROOT / parsed

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})

def _pressure_triggered(
    *,
    max_doc_weight_ratio: float | None,
    n_above_max_doc_weight: int | None,
    trigger_max_doc_weight_ratio: float | None,
    trigger_min_above_max_doc_weight: int | None,
) -> bool | None:
    checks: list[bool] = []
    if trigger_max_doc_weight_ratio is not None and max_doc_weight_ratio is not None:
        checks.append(float(max_doc_weight_ratio) > float(trigger_max_doc_weight_ratio))
    if (
        trigger_min_above_max_doc_weight is not None
        and n_above_max_doc_weight is not None
    ):
        checks.append(
            int(n_above_max_doc_weight) >= int(trigger_min_above_max_doc_weight)
        )
    return any(checks) if checks else None

def _prepare_pressure_summary(
    summary_path: Path,
    *,
    target_max_doc_weight: float,
    trigger_max_doc_weight_ratio: float | None,
    trigger_min_above_max_doc_weight: int | None,
) -> dict[str, Any]:
    summary = pilot._read_json(summary_path)
    max_doc_weight = pilot._nested_value(
        summary,
        "post_max_cluster_size",
        "postprocess.max_doc_weight",
        "postprocess_summary.max_doc_weight",
        "oversize_summary.after.max_doc_weight",
        "oversize_summary.before.max_doc_weight",
    )
    n_above = pilot._nested_value(
        summary,
        "post_n_clusters_gt_target_max",
        "postprocess.n_above_max_doc_weight",
        "postprocess_summary.n_above_max_doc_weight",
        "oversize_summary.after.n_above_max_doc_weight",
        "oversize_summary.before.n_above_max_doc_weight",
    )
    max_doc_weight_float = None if max_doc_weight is None else float(max_doc_weight)
    n_above_int = None if n_above is None else int(n_above)
    ratio = (
        None
        if max_doc_weight_float is None or float(target_max_doc_weight) <= 0.0
        else max_doc_weight_float / float(target_max_doc_weight)
    )
    return {
        "prepare_max_doc_weight": max_doc_weight_float,
        "prepare_max_doc_weight_ratio": ratio,
        "prepare_n_above_max_doc_weight": n_above_int,
        "prepare_triggered": _pressure_triggered(
            max_doc_weight_ratio=ratio,
            n_above_max_doc_weight=n_above_int,
            trigger_max_doc_weight_ratio=trigger_max_doc_weight_ratio,
            trigger_min_above_max_doc_weight=trigger_min_above_max_doc_weight,
        ),
    }

def _blank_row(input_cfg: pilot.Slice4Input, prepare: dict[str, Any]) -> dict[str, Any]:
    row = {field: None for field in CSV_FIELDS}
    row.update(
        {
            "sample": input_cfg.sample,
            "seed": int(input_cfg.seed),
            "summary_path": pilot._rel(input_cfg.summary_path),
            **prepare,
        }
    )
    return row

def _unsupported_row(
    *,
    input_cfg: pilot.Slice4Input,
    prepare: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    row = _blank_row(input_cfg, prepare)
    row.update(
        {
            "supported": False,
            "unsupported_reason": reason,
            "skipped_by_prepare": False,
            "elapsed_sec": 0.0,
        }
    )
    return row

def _prepare_skipped_row(
    *,
    input_cfg: pilot.Slice4Input,
    prepare: dict[str, Any],
) -> dict[str, Any]:
    row = _blank_row(input_cfg, prepare)
    row.update(
        {
            "supported": True,
            "unsupported_reason": "",
            "skipped_by_prepare": True,
            "elapsed_sec": 0.0,
            "selected_variant": "skipped_prepare_untriggered",
            "triggered": False,
            "fallback_triggered": True,
            "fallback_reason": "prepare_trigger_not_met",
        }
    )
    return row

def _flatten_result(
    *,
    input_cfg: pilot.Slice4Input,
    result: Any,
    graph: Any,
    elapsed_sec: float,
    config: SafeFastValidationConfig,
    prepare: dict[str, Any],
) -> dict[str, Any]:
    membership = np.asarray(result.membership, dtype=np.uint64)
    standard = result.standard
    diff = pilot._membership_diff_summary(
        np.asarray(standard.membership, dtype=np.uint64),
        membership,
    )
    quality = float(result.quality)
    standard_quality = float(standard.quality)
    quality_delta = quality - standard_quality
    try:
        quality_recomputed = float(
            graph.cpm_quality(membership, resolution=float(input_cfg.resolution))
        )
    except (AttributeError, TypeError):
        quality_recomputed = None
    quality_recompute_delta = (
        None if quality_recomputed is None else quality_recomputed - quality
    )
    quality_recompute_abs_delta = (
        None if quality_recompute_delta is None else abs(quality_recompute_delta)
    )
    quality_recompute_ok = (
        None
        if quality_recompute_abs_delta is None
        else quality_recompute_abs_delta <= float(config.quality_recompute_abs_tol)
    )
    row = _blank_row(input_cfg, prepare)
    row.update(
        {
            "supported": True,
            "unsupported_reason": "",
            "skipped_by_prepare": False,
            "elapsed_sec": float(elapsed_sec),
            "selected_variant": str(result.selected_variant),
            "triggered": bool(result.triggered),
            "fallback_triggered": bool(result.fallback_triggered),
            "fallback_reason": str(result.fallback_reason),
            "quality": quality,
            "standard_quality": standard_quality,
            "quality_delta_vs_standard": quality_delta,
            "quality_improved_vs_standard": quality_delta > 0.0,
            "quality_recomputed": quality_recomputed,
            "quality_recompute_delta": quality_recompute_delta,
            "quality_recompute_abs_delta": quality_recompute_abs_delta,
            "quality_recompute_ok": quality_recompute_ok,
            "n_clusters": int(result.n_clusters),
            "standard_max_doc_weight": float(result.standard_max_doc_weight),
            "standard_max_doc_weight_ratio": float(result.standard_max_doc_weight_ratio),
            "standard_n_above_max_doc_weight": int(
                result.standard_n_above_max_doc_weight
            ),
            "selected_max_doc_weight": float(result.selected_max_doc_weight),
            "selected_max_doc_weight_ratio": float(result.selected_max_doc_weight_ratio),
            "selected_n_above_max_doc_weight": int(
                result.selected_n_above_max_doc_weight
            ),
            "max_extra_parents_per_iteration": int(
                result.max_extra_parents_per_iteration
            ),
            "max_extra_children_per_parent": int(
                result.max_extra_children_per_parent
            ),
            "severe_tier_triggered": bool(result.severe_tier_triggered),
            "repair_escalated": bool(result.repair_escalated),
            "repair_escalation_accepted": bool(result.repair_escalation_accepted),
            **diff,
        }
    )
    return {field: row.get(field) for field in CSV_FIELDS}

def _run_one_summary(
    input_cfg: pilot.Slice4Input,
    *,
    config: SafeFastValidationConfig,
) -> dict[str, Any]:
    prepare = (
        _prepare_pressure_summary(
            input_cfg.summary_path,
            target_max_doc_weight=float(input_cfg.target_max_doc_weight),
            trigger_max_doc_weight_ratio=config.trigger_max_doc_weight_ratio,
            trigger_min_above_max_doc_weight=config.trigger_min_above_max_doc_weight,
        )
        if input_cfg.summary_path is not None
        else {
            "prepare_max_doc_weight": None,
            "prepare_max_doc_weight_ratio": None,
            "prepare_n_above_max_doc_weight": None,
            "prepare_triggered": None,
        }
    )
    if config.skip_prepare_untriggered and prepare.get("prepare_triggered") is False:
        return _prepare_skipped_row(input_cfg=input_cfg, prepare=prepare)
    n_nodes = pilot._infer_n_nodes(input_cfg)
    node_weights = pilot._load_node_weights(input_cfg.node_weights_path, n_nodes)
    graph = pilot._load_graph(input_cfg, node_weights)
    start = time.perf_counter()
    try:
        result = graph.run_leiden_dongdaemun_safe_fast_refinement(
            target_max_weight=float(input_cfg.target_max_doc_weight),
            resolution=float(input_cfg.resolution),
            seed=int(input_cfg.seed),
            n_iterations=int(config.n_iterations),
            randomness=float(config.randomness),
            trigger_max_doc_weight_ratio=config.trigger_max_doc_weight_ratio,
            trigger_min_above_max_doc_weight=config.trigger_min_above_max_doc_weight,
            accept_max_doc_weight_ratio=float(config.accept_max_doc_weight_ratio),
            accept_min_quality_delta=config.accept_min_quality_delta,
            accept_min_quality_delta_ratio=config.accept_min_quality_delta_ratio,
        )
    except (AttributeError, TypeError, ImportError) as exc:
        return _unsupported_row(
            input_cfg=input_cfg,
            prepare=prepare,
            reason=str(exc),
        )
    elapsed = time.perf_counter() - start
    return _flatten_result(
        input_cfg=input_cfg,
        result=result,
        graph=graph,
        elapsed_sec=elapsed,
        config=config,
        prepare=prepare,
    )

def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        key = "" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts

def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if bool(row.get("supported"))]
    direct = [
        row
        for row in supported
        if not bool(row.get("skipped_by_prepare"))
        and row.get("quality_delta_vs_standard") is not None
    ]
    deltas = [float(row["quality_delta_vs_standard"]) for row in direct]
    elapsed = [float(row.get("elapsed_sec") or 0.0) for row in direct]
    recompute_abs = [
        float(row["quality_recompute_abs_delta"])
        for row in direct
        if row.get("quality_recompute_abs_delta") is not None
    ]
    recompute_flags = [
        bool(row["quality_recompute_ok"])
        for row in direct
        if row.get("quality_recompute_ok") is not None
    ]
    return {
        "n_rows": int(len(rows)),
        "n_supported": int(len(supported)),
        "n_unsupported": int(len(rows) - len(supported)),
        "n_skipped_by_prepare": int(
            sum(1 for row in rows if bool(row.get("skipped_by_prepare")))
        ),
        "n_direct_rows": int(len(direct)),
        "n_triggered": int(sum(1 for row in direct if bool(row.get("triggered")))),
        "n_fallback": int(
            sum(1 for row in direct if bool(row.get("fallback_triggered")))
        ),
        "n_improved_vs_standard": int(sum(1 for delta in deltas if delta > 0.0)),
        "best_quality_delta_vs_standard": max(deltas) if deltas else None,
        "min_quality_delta_vs_standard": min(deltas) if deltas else None,
        "mean_elapsed_sec": (
            sum(elapsed) / float(len(elapsed)) if elapsed else None
        ),
        "max_elapsed_sec": max(elapsed) if elapsed else None,
        "max_quality_recompute_abs_delta": (
            max(recompute_abs) if recompute_abs else None
        ),
        "quality_recompute_all_ok": (
            all(recompute_flags) if recompute_flags else None
        ),
        "selected_variant_counts": _count_by(direct, "selected_variant"),
        "fallback_reason_counts": _count_by(direct, "fallback_reason"),
    }

def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| sample | seed | selected | triggered | fallback | delta q | std ratio | selected ratio | elapsed sec |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {seed} | {selected} | {triggered} | {fallback} | {delta:.6g} | {std_ratio:.6g} | {sel_ratio:.6g} | {elapsed:.3f} |".format(
                sample=row.get("sample", ""),
                seed=int(row.get("seed") or 0),
                selected=row.get("selected_variant") or "",
                triggered=bool(row.get("triggered")),
                fallback=row.get("fallback_reason") or "",
                delta=float(row.get("quality_delta_vs_standard") or 0.0),
                std_ratio=float(row.get("standard_max_doc_weight_ratio") or 0.0),
                sel_ratio=float(row.get("selected_max_doc_weight_ratio") or 0.0),
                elapsed=float(row.get("elapsed_sec") or 0.0),
            )
        )
    return lines

def _write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    config: SafeFastValidationConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dongdaemun Safe-Fast Validation",
        "",
        "This validation runs the conservative safe-fast preset directly on prepared source-level summaries.",
        "",
        "## Aggregate",
        "",
        f"- Direct rows: {aggregate.get('n_direct_rows')} / {aggregate.get('n_rows')}",
        f"- Triggered rows: {aggregate.get('n_triggered')}",
        f"- Fallback rows: {aggregate.get('n_fallback')}",
        f"- Improved rows: {aggregate.get('n_improved_vs_standard')}",
        f"- Best quality delta vs standard: {aggregate.get('best_quality_delta_vs_standard')}",
        f"- Max recompute abs delta: {aggregate.get('max_quality_recompute_abs_delta')}",
        "",
        "## Rows",
        "",
        *_markdown_table(rows),
        "",
        "## Config",
        "",
        f"- n_iterations: {config.n_iterations}",
        f"- randomness: {config.randomness:g}",
        f"- trigger_max_doc_weight_ratio: {config.trigger_max_doc_weight_ratio}",
        f"- trigger_min_above_max_doc_weight: {config.trigger_min_above_max_doc_weight}",
        f"- accept_max_doc_weight_ratio: {config.accept_max_doc_weight_ratio:g}",
        f"- accept_min_quality_delta: {config.accept_min_quality_delta}",
        f"- accept_min_quality_delta_ratio: {config.accept_min_quality_delta_ratio}",
        f"- safe_fast_preset: {SAFE_FAST_PRESET}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def _write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    config: SafeFastValidationConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dongdaemun_safe_fast_validation.csv"
    summary_path = output_dir / "dongdaemun_safe_fast_validation_summary.json"
    report_path = output_dir / "dongdaemun_safe_fast_validation_report.md"
    _write_csv(csv_path, rows)
    payload = {
        "schema": f"dongdaemun_safe_fast_validation.v{SCHEMA_VERSION}",
        "config": asdict(config),
        "safe_fast_preset": SAFE_FAST_PRESET,
        "aggregate": aggregate,
        "rows": rows,
        "paths": {
            "csv": csv_path,
            "summary": summary_path,
            "report": report_path,
        },
    }
    _write_json(summary_path, payload)
    _write_report(report_path, rows=rows, aggregate=aggregate, config=config)
    return {"csv": csv_path, "summary": summary_path, "report": report_path}

def run_validation(
    input_cfgs: list[pilot.Slice4Input],
    *,
    output_dir: Path,
    config: SafeFastValidationConfig | None = None,
) -> dict[str, Any]:
    config = config or SafeFastValidationConfig()
    rows = [_run_one_summary(input_cfg, config=config) for input_cfg in input_cfgs]
    aggregate = _aggregate_rows(rows)
    paths = _write_outputs(
        output_dir=output_dir,
        rows=rows,
        aggregate=aggregate,
        config=config,
    )
    return {
        "config": asdict(config),
        "aggregate": aggregate,
        "rows": rows,
        "paths": paths,
    }

def _summary_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in (args.summary or [])]
    if args.summary_list is not None:
        for line in Path(args.summary_list).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                paths.append(Path(stripped))
    if args.max_runs is not None:
        paths = paths[: int(args.max_runs)]
    return paths

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Prepared source-level summary JSON. May be passed multiple times.",
    )
    parser.add_argument(
        "--summary-list",
        type=Path,
        help="Text file with one summary JSON path per line.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--trigger-max-doc-weight-ratio", type=float, default=1.03)
    parser.add_argument("--trigger-min-above-max-doc-weight", type=int, default=2)
    parser.add_argument("--accept-max-doc-weight-ratio", type=float, default=1.01)
    parser.add_argument("--accept-min-quality-delta", type=float, default=0.0)
    parser.add_argument("--accept-min-quality-delta-ratio", type=float)
    parser.add_argument(
        "--quality-recompute-abs-tol",
        type=float,
        default=QUALITY_RECOMPUTE_ABS_TOL,
    )
    parser.add_argument(
        "--skip-prepare-untriggered",
        action="store_true",
        help="Skip direct safe-fast runs when prepare-summary pressure is below trigger.",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary_paths = _summary_paths_from_args(args)
    if not summary_paths:
        parser.error("at least one --summary or --summary-list entry is required")
    input_cfgs = [
        pilot._resolve_input_from_summary(_repo_path(path) or Path(path))
        for path in summary_paths
    ]
    config = SafeFastValidationConfig(
        n_iterations=int(args.n_iterations),
        randomness=float(args.randomness),
        trigger_max_doc_weight_ratio=args.trigger_max_doc_weight_ratio,
        trigger_min_above_max_doc_weight=args.trigger_min_above_max_doc_weight,
        accept_max_doc_weight_ratio=float(args.accept_max_doc_weight_ratio),
        accept_min_quality_delta=args.accept_min_quality_delta,
        accept_min_quality_delta_ratio=args.accept_min_quality_delta_ratio,
        quality_recompute_abs_tol=float(args.quality_recompute_abs_tol),
        skip_prepare_untriggered=bool(args.skip_prepare_untriggered),
    )
    result = run_validation(
        input_cfgs,
        output_dir=_repo_path(args.output_dir) or args.output_dir,
        config=config,
    )
    print(json.dumps(_json_safe(result["paths"]), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
