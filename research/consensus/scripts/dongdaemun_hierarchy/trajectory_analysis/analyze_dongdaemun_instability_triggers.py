"""Analyze Dongdaemun local candidate instability triggers.

This is the first-pass test for a local delayed-commitment direction.  It does
not change Rust behavior.  It asks how often parent-local candidate selection
looks unstable enough to justify a small lookahead beam.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
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


SCRIPT_DIR = Path(__file__).resolve().parent
from analyze_dongdaemun_local_candidate_beam import (  # noqa: E402
    DEFAULT_GROUP_FIELDS,
    DEFAULT_TRACE_PATH,
    _bool_value,
    _candidate_groups_from_events,
    _candidate_id,
    _child_ratio,
    _csv_value,
    _finite_float,
    _float_value,
    _int_value,
    _is_selectable,
    _largest_fraction,
    _load_run_metadata,
    _pressure,
    _pressure_sort_key,
    _quality,
    _quality_sort_key,
    _read_jsonl,
    _signature,
    _singleton_fraction,
)

DEFAULT_RUNS_PATH = DEFAULT_TRACE_PATH.with_name("candidate_trace_runs.jsonl")
DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_instability_triggers_20260511"
)

SCHEMA_VERSION = 1
PARENT_ROWS_FILENAME = "instability_parent_rows.csv"
SUMMARY_FILENAME = "instability_summary.csv"
LOOKAHEAD_CANDIDATES_FILENAME = "instability_lookahead_candidates.csv"
REPORT_FILENAME = "instability_report.md"
SUMMARY_JSON_FILENAME = "instability_summary.json"

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _candidate_columns(prefix: str, candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {
            f"{prefix}_candidate_id": None,
            f"{prefix}_source": None,
            f"{prefix}_candidate_delta_q": None,
            f"{prefix}_pressure_reduction": None,
            f"{prefix}_child_ratio": None,
            f"{prefix}_largest_child_fraction": None,
            f"{prefix}_singleton_weight_fraction": None,
            f"{prefix}_signature": None,
        }
    return {
        f"{prefix}_candidate_id": _candidate_id(candidate),
        f"{prefix}_source": candidate.get("source"),
        f"{prefix}_candidate_delta_q": _quality(candidate),
        f"{prefix}_pressure_reduction": _pressure(candidate),
        f"{prefix}_child_ratio": _child_ratio(candidate),
        f"{prefix}_largest_child_fraction": _largest_fraction(candidate),
        f"{prefix}_singleton_weight_fraction": _singleton_fraction(candidate),
        f"{prefix}_signature": _signature(candidate, 2),
    }

def _quality_margin(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    ordered = sorted(candidates, key=_quality_sort_key)
    return _quality(ordered[0]) - _quality(ordered[1])

def _relative_margin(margin: float | None, best_quality: float | None) -> float | None:
    if margin is None or best_quality is None:
        return None
    denom = max(1.0, abs(float(best_quality)))
    return float(margin) / denom

def _within_quality_band(
    candidates: list[dict[str, Any]],
    *,
    quality_band_abs: float,
    quality_band_rel: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    best_quality = max(_quality(candidate) for candidate in candidates)
    band = max(float(quality_band_abs), abs(best_quality) * float(quality_band_rel))
    return [
        candidate
        for candidate in candidates
        if _quality(candidate) >= best_quality - band
    ]

def _retained_beam_candidates(
    selectable: list[dict[str, Any]],
    *,
    beam_width: int,
    quality_band_abs: float,
    quality_band_rel: float,
    signature_precision: int,
) -> list[dict[str, Any]]:
    retained_by_id: dict[int, dict[str, Any]] = {}

    def add(candidate: dict[str, Any]) -> None:
        retained_by_id.setdefault(_candidate_id(candidate), candidate)

    for candidate in sorted(selectable, key=_quality_sort_key)[:2]:
        add(candidate)
    for candidate in sorted(selectable, key=_pressure_sort_key)[:1]:
        add(candidate)
    in_band = _within_quality_band(
        selectable,
        quality_band_abs=quality_band_abs,
        quality_band_rel=quality_band_rel,
    )
    by_signature: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in sorted(in_band, key=_quality_sort_key):
        by_signature.setdefault(_signature(candidate, signature_precision), candidate)
    for candidate in by_signature.values():
        add(candidate)

    retained = list(retained_by_id.values())
    if len(retained) > int(beam_width):
        retained = sorted(retained, key=_quality_sort_key)[: int(beam_width)]
    return retained

def _metadata_columns(
    run_id: str,
    run_metadata: dict[str, dict[str, Any]],
    group_fields: Iterable[str],
) -> dict[str, Any]:
    metadata = run_metadata.get(run_id, {})
    return {field: metadata.get(field) for field in group_fields}

def build_instability_rows(
    *,
    events: list[dict[str, Any]],
    run_metadata: dict[str, dict[str, Any]],
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    quality_margin_abs: float = 1.0,
    quality_margin_rel: float = 1.0e-4,
    quality_band_abs: float = 1.0,
    quality_band_rel: float = 1.0e-4,
    beam_width: int = 5,
    signature_precision: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _candidate_groups_from_events(events)
    for group in groups:
        selectable = [candidate for candidate in group.profiles if _is_selectable(candidate)]
        current = None
        if group.applied_candidate_id is not None:
            for candidate in group.profiles:
                if _candidate_id(candidate) == group.applied_candidate_id:
                    current = candidate
                    break
        if current is None:
            selected_by_policy = [
                candidate
                for candidate in group.profiles
                if str(candidate.get("decision") or "") == "selected_by_policy"
            ]
            current = selected_by_policy[-1] if selected_by_policy else None

        ordered_quality = sorted(selectable, key=_quality_sort_key)
        ordered_pressure = sorted(selectable, key=_pressure_sort_key)
        quality_top1 = ordered_quality[0] if ordered_quality else None
        quality_top2 = ordered_quality[1] if len(ordered_quality) > 1 else None
        pressure_top1 = ordered_pressure[0] if ordered_pressure else None
        margin = _quality_margin(selectable)
        best_quality = None if quality_top1 is None else _quality(quality_top1)
        rel_margin = _relative_margin(margin, best_quality)
        in_band = _within_quality_band(
            selectable,
            quality_band_abs=quality_band_abs,
            quality_band_rel=quality_band_rel,
        )
        signatures = {_signature(candidate, signature_precision) for candidate in in_band}

        low_margin = (
            margin is not None
            and (
                margin <= float(quality_margin_abs)
                or (
                    rel_margin is not None
                    and rel_margin <= float(quality_margin_rel)
                )
            )
        )
        q_pressure_disagree = (
            quality_top1 is not None
            and pressure_top1 is not None
            and _candidate_id(quality_top1) != _candidate_id(pressure_top1)
        )
        current_not_quality = (
            current is not None
            and quality_top1 is not None
            and _candidate_id(current) != _candidate_id(quality_top1)
        )
        signature_diverse = len(signatures) >= 2
        unstable_reasons: list[str] = []
        if low_margin:
            unstable_reasons.append("low_quality_margin")
        if q_pressure_disagree:
            unstable_reasons.append("quality_pressure_disagree")
        if current_not_quality:
            unstable_reasons.append("current_not_quality_top1")
        if signature_diverse:
            unstable_reasons.append("signature_diverse_in_band")
        unstable = bool(unstable_reasons)
        retained = (
            _retained_beam_candidates(
                selectable,
                beam_width=beam_width,
                quality_band_abs=quality_band_abs,
                quality_band_rel=quality_band_rel,
                signature_precision=signature_precision,
            )
            if unstable
            else []
        )

        row = {
            "run_id": group.run_id,
            "depth": group.depth,
            "parent_id": group.parent_id,
            "parent_visit_index": group.parent_visit_index,
            **_metadata_columns(group.run_id, run_metadata, group_fields),
            "n_profiles": len(group.profiles),
            "n_selectable_candidates": len(selectable),
            "n_in_quality_band": len(in_band),
            "n_quality_band_signatures": len(signatures),
            "quality_margin_abs": margin,
            "quality_margin_rel": rel_margin,
            "low_quality_margin": low_margin,
            "quality_pressure_disagree": q_pressure_disagree,
            "current_not_quality_top1": current_not_quality,
            "signature_diverse_in_band": signature_diverse,
            "unstable": unstable,
            "unstable_reasons": unstable_reasons,
            "beam_width": len(retained),
            "retained_candidate_ids": [_candidate_id(candidate) for candidate in retained],
            "estimated_extra_replays": max(0, len(retained) - 1),
            "parent_weight": _finite_float(group.profiles[0].get("parent_weight"))
            if group.profiles
            else None,
            "parent_size": _int_value(group.profiles[0].get("parent_size"))
            if group.profiles
            else None,
        }
        row.update(_candidate_columns("current", current))
        row.update(_candidate_columns("quality_top1", quality_top1))
        row.update(_candidate_columns("quality_top2", quality_top2))
        row.update(_candidate_columns("pressure_top1", pressure_top1))
        rows.append(row)
    return rows

def _group_key(row: dict[str, Any], group_fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in group_fields)

def summarize_instability(
    rows: list[dict[str, Any]],
    *,
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row, group_fields)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        n = len(group_rows)
        unstable = [row for row in group_rows if _bool_value(row.get("unstable"))]
        selectable = [
            row for row in group_rows if _int_value(row.get("n_selectable_candidates")) > 0
        ]
        summary = dict(zip(group_fields, key))
        summary.update(
            {
                "n_parent_visits": n,
                "n_selectable_parent_visits": len(selectable),
                "n_unstable_parent_visits": len(unstable),
                "unstable_parent_fraction": 0.0 if n == 0 else len(unstable) / n,
                "mean_selectable_candidates": _mean(
                    [_finite_float(row.get("n_selectable_candidates")) for row in group_rows]
                ),
                "mean_quality_margin_abs": _mean(
                    [_finite_float(row.get("quality_margin_abs")) for row in group_rows]
                ),
                "n_low_quality_margin": _count_true(group_rows, "low_quality_margin"),
                "n_quality_pressure_disagree": _count_true(
                    group_rows, "quality_pressure_disagree"
                ),
                "n_current_not_quality_top1": _count_true(
                    group_rows, "current_not_quality_top1"
                ),
                "n_signature_diverse_in_band": _count_true(
                    group_rows, "signature_diverse_in_band"
                ),
                "mean_unstable_beam_width": _mean(
                    [_finite_float(row.get("beam_width")) for row in unstable]
                ),
                "estimated_extra_replays": sum(
                    _int_value(row.get("estimated_extra_replays")) for row in unstable
                ),
            }
        )
        summary_rows.append(summary)
    return summary_rows

def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return None if not finite else sum(finite) / len(finite)

def _count_true(rows: list[dict[str, Any]], field: str) -> int:
    return sum(_bool_value(row.get(field)) for row in rows)

def _display_value(value: Any) -> Any:
    return "" if value is None else value

def lookahead_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _bool_value(row.get("unstable"))]
    return sorted(
        candidates,
        key=lambda row: (
            -_int_value(row.get("estimated_extra_replays")),
            _finite_float(row.get("quality_margin_abs")) or math.inf,
            -_float_value(row.get("parent_weight"), 0.0),
            str(row.get("run_id")),
            _int_value(row.get("parent_id")),
        ),
    )

def _build_report(
    *,
    parent_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    lookahead_rows: list[dict[str, Any]],
) -> str:
    n = len(parent_rows)
    n_unstable = _count_true(parent_rows, "unstable")
    lines = [
        "# Dongdaemun Instability Trigger Analysis",
        "",
        f"- Parent visits: {n}",
        f"- Unstable parent visits: {n_unstable} ({0.0 if n == 0 else n_unstable / n:.2%})",
        f"- Estimated extra replays: {sum(_int_value(row.get('estimated_extra_replays')) for row in lookahead_rows)}",
        "",
        "## Summary",
        "",
        "| sample | variant | config | seed | candidate policy | band | parents | unstable | fraction | mean beam | extra replays |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {sample} | {variant} | {config} | {seed} | {policy} | {band} | {parents} | {unstable} | {fraction:.2%} | {beam} | {extra} |".format(
                sample=_display_value(row.get("sample")),
                variant=_display_value(row.get("variant")),
                config=_display_value(row.get("config_id")),
                seed=_display_value(row.get("seed_perturbations")),
                policy=_display_value(row.get("candidate_quality_policy")),
                band=_display_value(row.get("adaptive_plateau_quality_band")),
                parents=_int_value(row.get("n_parent_visits")),
                unstable=_int_value(row.get("n_unstable_parent_visits")),
                fraction=_float_value(row.get("unstable_parent_fraction")),
                beam="" if row.get("mean_unstable_beam_width") is None else f"{float(row['mean_unstable_beam_width']):.2f}",
                extra=_int_value(row.get("estimated_extra_replays")),
            )
        )
    lines.extend(
        [
            "",
            "## Top Lookahead Candidates",
            "",
            "| run | parent | visit | reasons | margin | beam | retained |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in lookahead_rows[:20]:
        lines.append(
            "| {run} | {parent} | {visit} | {reasons} | {margin} | {beam} | {retained} |".format(
                run=row.get("run_id"),
                parent=_int_value(row.get("parent_id")),
                visit=_int_value(row.get("parent_visit_index")),
                reasons=";".join(row.get("unstable_reasons") or []),
                margin="" if row.get("quality_margin_abs") is None else f"{float(row['quality_margin_abs']):.4g}",
                beam=_int_value(row.get("beam_width")),
                retained=",".join(str(x) for x in row.get("retained_candidate_ids") or []),
            )
        )
    lines.append("")
    return "\n".join(lines)

def analyze_instability_triggers(
    *,
    trace_path: Path,
    runs_path: Path | None,
    output_dir: Path,
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    quality_margin_abs: float = 1.0,
    quality_margin_rel: float = 1.0e-4,
    quality_band_abs: float = 1.0,
    quality_band_rel: float = 1.0e-4,
    beam_width: int = 5,
    signature_precision: int = 2,
) -> dict[str, Any]:
    events = _read_jsonl(trace_path)
    run_metadata = _load_run_metadata(runs_path)
    parent_rows = build_instability_rows(
        events=events,
        run_metadata=run_metadata,
        group_fields=group_fields,
        quality_margin_abs=quality_margin_abs,
        quality_margin_rel=quality_margin_rel,
        quality_band_abs=quality_band_abs,
        quality_band_rel=quality_band_rel,
        beam_width=beam_width,
        signature_precision=signature_precision,
    )
    summary_rows = summarize_instability(parent_rows, group_fields=group_fields)
    lookahead_rows = lookahead_candidate_rows(parent_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    parent_path = output_dir / PARENT_ROWS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    lookahead_path = output_dir / LOOKAHEAD_CANDIDATES_FILENAME
    report_path = output_dir / REPORT_FILENAME
    summary_json_path = output_dir / SUMMARY_JSON_FILENAME
    _write_csv(parent_path, parent_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(lookahead_path, lookahead_rows)
    report_path.write_text(
        _build_report(
            parent_rows=parent_rows,
            summary_rows=summary_rows,
            lookahead_rows=lookahead_rows,
        ),
        encoding="utf-8",
    )
    payload = {
        "schema": "dongdaemun_instability_triggers.v1",
        "schema_version": SCHEMA_VERSION,
        "trace_path": str(trace_path),
        "runs_path": None if runs_path is None else str(runs_path),
        "n_parent_rows": len(parent_rows),
        "n_unstable_parent_rows": _count_true(parent_rows, "unstable"),
        "unstable_parent_fraction": 0.0
        if not parent_rows
        else _count_true(parent_rows, "unstable") / len(parent_rows),
        "estimated_extra_replays": sum(
            _int_value(row.get("estimated_extra_replays")) for row in lookahead_rows
        ),
        "unstable_reason_counts": {
            "low_quality_margin": _count_true(parent_rows, "low_quality_margin"),
            "quality_pressure_disagree": _count_true(
                parent_rows, "quality_pressure_disagree"
            ),
            "current_not_quality_top1": _count_true(
                parent_rows, "current_not_quality_top1"
            ),
            "signature_diverse_in_band": _count_true(
                parent_rows, "signature_diverse_in_band"
            ),
        },
        "paths": {
            "parent_rows": str(parent_path),
            "summary": str(summary_path),
            "lookahead_candidates": str(lookahead_path),
            "report": str(report_path),
            "summary_json": str(summary_json_path),
        },
    }
    _write_json(summary_json_path, payload)
    return payload

def _parse_group_fields(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(DEFAULT_GROUP_FIELDS)
    return tuple(item.strip() for item in value.split(",") if item.strip())

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-fields")
    parser.add_argument("--quality-margin-abs", type=float, default=1.0)
    parser.add_argument("--quality-margin-rel", type=float, default=1.0e-4)
    parser.add_argument("--quality-band-abs", type=float, default=1.0)
    parser.add_argument("--quality-band-rel", type=float, default=1.0e-4)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--signature-precision", type=int, default=2)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = analyze_instability_triggers(
        trace_path=args.trace,
        runs_path=args.runs,
        output_dir=args.output_dir,
        group_fields=_parse_group_fields(args.group_fields),
        quality_margin_abs=float(args.quality_margin_abs),
        quality_margin_rel=float(args.quality_margin_rel),
        quality_band_abs=float(args.quality_band_abs),
        quality_band_rel=float(args.quality_band_rel),
        beam_width=int(args.beam_width),
        signature_precision=int(args.signature_precision),
    )
    print(f"Saved instability analysis to {payload['paths']['summary_json']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
