"""Summarize opt-in Dongdaemun quality checkpoint trace JSONL files."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GROUP_FIELDS = (
    "sample",
    "variant",
    "config_id",
    "candidate_quality_policy",
    "adaptive_plateau_quality_band",
    "gamma_preset",
    "seed_perturbations",
)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
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


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _sort_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_order = {
        "start": 0,
        "after_iteration": 1,
        "pre_final_guard": 2,
        "final": 3,
    }
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("checkpoint_index") or 0),
            int(row.get("iteration") or 0),
            phase_order.get(str(row.get("phase") or ""), 99),
        ),
    )


def _metadata_label(metadata: dict[str, Any], group_fields: tuple[str, ...]) -> str:
    values = [metadata.get(field) for field in group_fields]
    label = " | ".join("" if value is None else str(value) for value in values)
    return label or str(metadata.get("run_id") or "unknown")


def _joined_checkpoint_rows(
    trace_rows: list[dict[str, Any]],
    run_metadata: dict[str, dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in trace_rows:
        if event.get("event") not in (None, "quality_checkpoint"):
            continue
        run_id = event.get("run_id")
        metadata = run_metadata.get(str(run_id), {})
        row: dict[str, Any] = {}
        for field in group_fields:
            row[field] = metadata.get(field)
        for field in (
            "row_key",
            "summary_path",
            "seed",
            "use_baseline_repair",
            "parent_selection_policy",
            "max_extra_parents_per_iteration",
            "max_extra_children_per_parent",
        ):
            if field in metadata:
                row[field] = metadata.get(field)
        row.update(event)
        rows.append(row)
    return rows


def _by_run_rows(
    checkpoints: list[dict[str, Any]],
    run_metadata: dict[str, dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        by_run[str(row.get("run_id"))].append(row)

    rows: list[dict[str, Any]] = []
    for run_id, run_rows in sorted(by_run.items()):
        ordered = _sort_checkpoints(run_rows)
        start = next((row for row in ordered if row.get("phase") == "start"), ordered[0])
        final = next((row for row in reversed(ordered) if row.get("phase") == "final"), ordered[-1])
        qualities = [
            value
            for value in (_finite_float(row.get("quality")) for row in ordered)
            if value is not None
        ]
        elapsed_ms_final = _finite_float(final.get("elapsed_ms_since_run_start"))
        final_gain = _finite_float(final.get("quality_delta_vs_start"))
        elapsed_sec_final = (
            elapsed_ms_final / 1000.0
            if elapsed_ms_final is not None and elapsed_ms_final > 0.0
            else None
        )
        time_to_95pct = _time_to_final_quality_gain_ms(ordered, final_gain, 0.95)
        best_quality_delta_per_sec = _best_quality_delta_per_sec(ordered)
        start_pressure = _finite_float(start.get("max_doc_weight_ratio"))
        final_pressure = _finite_float(final.get("max_doc_weight_ratio"))
        final_pressure_reduction_per_sec = (
            (start_pressure - final_pressure) / elapsed_sec_final
            if start_pressure is not None
            and final_pressure is not None
            and elapsed_sec_final is not None
            else None
        )
        metadata = run_metadata.get(run_id, {})
        row: dict[str, Any] = {"run_id": run_id}
        for field in group_fields:
            row[field] = metadata.get(field)
        row.update(
            {
                "row_key": metadata.get("row_key"),
                "n_checkpoints": len(ordered),
                "start_quality": start.get("quality"),
                "final_quality": final.get("quality"),
                "final_quality_delta_vs_start": final.get("quality_delta_vs_start"),
                "max_quality": max(qualities) if qualities else None,
                "min_quality": min(qualities) if qualities else None,
                "final_iteration": final.get("iteration"),
                "elapsed_ms_final": elapsed_ms_final,
                "time_to_95pct_final_quality_gain_ms": time_to_95pct,
                "best_quality_delta_per_sec": best_quality_delta_per_sec,
                "final_pressure_reduction_per_sec": final_pressure_reduction_per_sec,
                "final_n_clusters": final.get("n_clusters"),
                "start_max_doc_weight": start.get("max_doc_weight"),
                "final_max_doc_weight": final.get("max_doc_weight"),
                "final_max_doc_weight_ratio": final.get("max_doc_weight_ratio"),
                "final_n_above_max_doc_weight": final.get("n_above_max_doc_weight"),
                "selected_parent_count_total": final.get("selected_parent_count_total"),
                "applied_parent_count_total": final.get("applied_parent_count_total"),
            }
        )
        rows.append(row)
    return rows


def _time_to_final_quality_gain_ms(
    ordered: list[dict[str, Any]],
    final_gain: float | None,
    fraction: float,
) -> float | None:
    if final_gain is None or final_gain <= 0.0:
        return None
    target = final_gain * fraction
    for row in ordered:
        gain = _finite_float(row.get("quality_delta_vs_start"))
        elapsed = _finite_float(row.get("elapsed_ms_since_run_start"))
        if gain is not None and elapsed is not None and gain >= target:
            return elapsed
    return None


def _best_quality_delta_per_sec(ordered: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for row in ordered:
        gain = _finite_float(row.get("quality_delta_vs_start"))
        elapsed_ms = _finite_float(row.get("elapsed_ms_since_run_start"))
        if gain is None or elapsed_ms is None or elapsed_ms <= 0.0:
            continue
        rate = gain / (elapsed_ms / 1000.0)
        if best is None or rate > best:
            best = rate
    return best


def _quality_gain_per_sec_rows(by_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in by_run:
        final_gain = _finite_float(row.get("final_quality_delta_vs_start"))
        elapsed_ms = _finite_float(row.get("elapsed_ms_final"))
        final_rate = (
            final_gain / (elapsed_ms / 1000.0)
            if final_gain is not None and elapsed_ms is not None and elapsed_ms > 0.0
            else None
        )
        rows.append(
            {
                **row,
                "final_quality_gain_per_sec": final_rate,
                "best_quality_delta_per_sec": row.get("best_quality_delta_per_sec"),
            }
        )
    return rows


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )


def _write_line_plot(
    path: Path,
    rows_by_run: dict[str, list[dict[str, Any]]],
    run_metadata: dict[str, dict[str, Any]],
    group_fields: tuple[str, ...],
    *,
    x_field: str = "iteration",
    xlabel: str = "iteration",
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

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted_count = 0
    for index, (run_id, run_rows) in enumerate(sorted(rows_by_run.items())):
        ordered = _sort_checkpoints(run_rows)
        x_values = [_finite_float(row.get(x_field)) for row in ordered]
        y_values = [_finite_float(row.get(y_field)) for row in ordered]
        points = [
            (x, y)
            for x, y in zip(x_values, y_values)
            if x is not None and y is not None
        ]
        if not points:
            continue
        label = _metadata_label(run_metadata.get(run_id, {}), group_fields)
        ax.plot(
            [x for x, _ in points],
            [y for _, y in points],
            marker="o",
            linewidth=1.25,
            label=label if index < 12 else None,
        )
        plotted_count += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted_count and len(rows_by_run) <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_scatter_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _write_placeholder_png(path)
        return

    x_values = [_finite_float(row.get("max_doc_weight_ratio")) for row in rows]
    y_values = [_finite_float(row.get("quality_delta_vs_start")) for row in rows]
    points = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    if points:
        ax.scatter([x for x, _ in points], [y for _, y in points], s=18, alpha=0.75)
    ax.set_xlabel("max doc weight ratio")
    ax.set_ylabel("quality delta vs start")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_quality_trace(
    *,
    trace_path: Path,
    runs_path: Path | None,
    output_dir: Path,
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS,
) -> dict[str, Any]:
    run_metadata: dict[str, dict[str, Any]] = {}
    if runs_path is not None:
        run_metadata = {
            str(row.get("run_id")): row
            for row in _read_jsonl(runs_path)
            if row.get("run_id")
        }
    trace_rows = _read_jsonl(trace_path)
    checkpoints = _joined_checkpoint_rows(trace_rows, run_metadata, group_fields)
    by_run = _by_run_rows(checkpoints, run_metadata, group_fields)
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoints:
        rows_by_run[str(row.get("run_id"))].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_path = output_dir / "quality_trace_checkpoints.csv"
    by_run_path = output_dir / "quality_trace_by_run.csv"
    gain_per_sec_path = output_dir / "quality_gain_per_sec_by_run.csv"
    quality_plot_path = output_dir / "quality_vs_iteration.png"
    delta_plot_path = output_dir / "quality_delta_vs_iteration.png"
    delta_elapsed_plot_path = output_dir / "quality_delta_vs_elapsed_ms.png"
    pressure_elapsed_plot_path = output_dir / "max_doc_weight_ratio_vs_elapsed_ms.png"
    pressure_plot_path = output_dir / "quality_delta_vs_max_doc_weight_ratio.png"
    summary_path = output_dir / "quality_trace_summary.json"

    _write_csv(checkpoints_path, checkpoints)
    _write_csv(by_run_path, by_run)
    _write_csv(gain_per_sec_path, _quality_gain_per_sec_rows(by_run))
    _write_line_plot(
        quality_plot_path,
        rows_by_run,
        run_metadata,
        group_fields,
        x_field="iteration",
        xlabel="iteration",
        y_field="quality",
        ylabel="quality",
    )
    _write_line_plot(
        delta_plot_path,
        rows_by_run,
        run_metadata,
        group_fields,
        x_field="iteration",
        xlabel="iteration",
        y_field="quality_delta_vs_start",
        ylabel="quality delta vs start",
    )
    _write_line_plot(
        delta_elapsed_plot_path,
        rows_by_run,
        run_metadata,
        group_fields,
        x_field="elapsed_ms_since_run_start",
        xlabel="elapsed ms since run start",
        y_field="quality_delta_vs_start",
        ylabel="quality delta vs start",
    )
    _write_line_plot(
        pressure_elapsed_plot_path,
        rows_by_run,
        run_metadata,
        group_fields,
        x_field="elapsed_ms_since_run_start",
        xlabel="elapsed ms since run start",
        y_field="max_doc_weight_ratio",
        ylabel="max doc weight ratio",
    )
    _write_scatter_plot(pressure_plot_path, checkpoints)

    final_deltas = [
        value
        for value in (
            _finite_float(row.get("final_quality_delta_vs_start")) for row in by_run
        )
        if value is not None
    ]
    payload = {
        "schema": "dongdaemun_quality_trace_summary.v1",
        "trace_path": str(trace_path),
        "runs_path": None if runs_path is None else str(runs_path),
        "group_fields": list(group_fields),
        "n_checkpoints": len(checkpoints),
        "n_runs": len(by_run),
        "best_final_quality_delta_vs_start": max(final_deltas) if final_deltas else None,
        "worst_final_quality_delta_vs_start": min(final_deltas) if final_deltas else None,
        "paths": {
            "checkpoints": str(checkpoints_path),
            "by_run": str(by_run_path),
            "quality_gain_per_sec_by_run": str(gain_per_sec_path),
            "quality_vs_iteration": str(quality_plot_path),
            "quality_delta_vs_iteration": str(delta_plot_path),
            "quality_delta_vs_elapsed_ms": str(delta_elapsed_plot_path),
            "max_doc_weight_ratio_vs_elapsed_ms": str(pressure_elapsed_plot_path),
            "quality_delta_vs_max_doc_weight_ratio": str(pressure_plot_path),
            "summary": str(summary_path),
        },
    }
    _write_json(summary_path, payload)
    return payload


def _parse_group_fields(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_GROUP_FIELDS
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    if not fields:
        raise ValueError("--group-fields must contain at least one field")
    return fields


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--group-fields",
        help=(
            "Comma-separated run metadata fields for plot grouping. "
            f"Default: {', '.join(DEFAULT_GROUP_FIELDS)}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = summarize_quality_trace(
        trace_path=args.trace,
        runs_path=args.runs,
        output_dir=args.output_dir,
        group_fields=_parse_group_fields(args.group_fields),
    )
    print(f"Saved quality trace summary to {payload['paths']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
