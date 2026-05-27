"""Summarize opt-in Dongdaemun candidate trace JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GROUP_FIELDS = (
    "sample",
    "variant",
    "use_baseline_repair",
    "config_id",
    "gamma_preset",
    "seed_perturbations",
    "parent_selection_policy",
    "candidate_quality_policy",
    "adaptive_plateau_quality_band",
)
NUMERIC_FIELDS = (
    "candidate_delta_q",
    "largest_child_fraction_improvement",
    "largest_child_fraction",
    "standard_max_child_weight_ratio",
    "candidate_max_child_weight_ratio",
    "pressure_reduction",
    "singleton_weight_fraction",
    "quotient_score",
    "adaptive_diagnostic_score",
    "baseline_repair_delta_sum",
)


@dataclass
class NumericSummary:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: Any) -> None:
        if value is None:
            return
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_columns(self, prefix: str) -> dict[str, Any]:
        mean = None if self.count == 0 else self.total / self.count
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_mean": mean,
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
        }


@dataclass
class CandidateTraceAggregate:
    counts: Counter[str] = field(default_factory=Counter)
    numeric: dict[str, NumericSummary] = field(
        default_factory=lambda: defaultdict(NumericSummary)
    )
    selected_applied_numeric: dict[str, NumericSummary] = field(
        default_factory=lambda: defaultdict(NumericSummary)
    )

    def add_selected_applied_profile(self, profile: dict[str, Any] | None) -> None:
        if profile is None:
            self.counts["selected_applied_profile_missing"] += 1
            return
        self.counts["selected_applied_profiles"] += 1
        for key in ("source", "quadrant", "decision"):
            value = profile.get(key)
            if value:
                self.counts[f"selected_applied_{key}_{value}"] += 1
        for field_name in NUMERIC_FIELDS:
            self.selected_applied_numeric[field_name].add(profile.get(field_name))

    def add_event(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "unknown")
        self.counts[f"event_{event_name}"] += 1
        if event_name == "candidate_profile":
            self.counts["candidate_profiles"] += 1
            for key in ("source", "quadrant", "decision"):
                value = event.get(key)
                if value:
                    self.counts[f"{key}_{value}"] += 1
            if event.get("valid") is True:
                self.counts["valid_true"] += 1
            elif event.get("valid") is False:
                self.counts["valid_false"] += 1
            if event.get("quality_passes") is True:
                self.counts["quality_passes_true"] += 1
            elif event.get("quality_passes") is False:
                self.counts["quality_passes_false"] += 1
            for field_name in NUMERIC_FIELDS:
                self.numeric[field_name].add(event.get(field_name))
        elif event_name == "candidate_decision":
            decision = event.get("decision")
            if decision:
                self.counts[f"decision_{decision}"] += 1

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = dict(sorted(self.counts.items()))
        for field_name in NUMERIC_FIELDS:
            row.update(self.numeric[field_name].as_columns(field_name))
            row.update(
                self.selected_applied_numeric[field_name].as_columns(
                    f"selected_applied_{field_name}"
                )
            )
        return row


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


def _group_key(metadata: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(metadata.get(field) for field in fields)


def summarize_candidate_trace(
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
    overall = CandidateTraceAggregate()
    by_run: dict[str, CandidateTraceAggregate] = defaultdict(CandidateTraceAggregate)
    by_group: dict[tuple[Any, ...], CandidateTraceAggregate] = defaultdict(
        CandidateTraceAggregate
    )

    events = _read_jsonl(trace_path)
    profile_by_key = {
        (
            str(event.get("run_id")),
            int(event.get("depth") or 0),
            int(event.get("parent_id") or 0),
            int(event.get("candidate_id") or 0),
        ): event
        for event in events
        if event.get("event") == "candidate_profile"
    }

    for event in events:
        run_id = event.get("run_id")
        metadata = run_metadata.get(str(run_id), {})
        profile = None
        if (
            event.get("event") == "candidate_decision"
            and event.get("decision") == "selected_applied"
        ):
            profile = profile_by_key.get(
                (
                    str(run_id),
                    int(event.get("depth") or 0),
                    int(event.get("parent_id") or 0),
                    int(event.get("candidate_id") or 0),
                )
            )
        overall.add_event(event)
        by_run[str(run_id)].add_event(event)
        by_group[_group_key(metadata, group_fields)].add_event(event)
        if event.get("event") == "candidate_decision" and event.get(
            "decision"
        ) == "selected_applied":
            overall.add_selected_applied_profile(profile)
            by_run[str(run_id)].add_selected_applied_profile(profile)
            by_group[_group_key(metadata, group_fields)].add_selected_applied_profile(
                profile
            )

    by_run_rows: list[dict[str, Any]] = []
    for run_id, aggregate in sorted(by_run.items()):
        row = {"run_id": run_id}
        row.update(run_metadata.get(run_id, {}))
        row.update(aggregate.as_row())
        by_run_rows.append(row)

    by_group_rows: list[dict[str, Any]] = []
    for key, aggregate in sorted(
        by_group.items(),
        key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
    ):
        row = dict(zip(group_fields, key))
        row.update(aggregate.as_row())
        by_group_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    by_run_path = output_dir / "candidate_trace_by_run.csv"
    by_group_path = output_dir / "candidate_trace_by_group.csv"
    summary_path = output_dir / "candidate_trace_summary.json"
    _write_csv(by_run_path, by_run_rows)
    _write_csv(by_group_path, by_group_rows)
    payload = {
        "schema": "dongdaemun_candidate_trace_summary.v1",
        "trace_path": str(trace_path),
        "runs_path": None if runs_path is None else str(runs_path),
        "group_fields": list(group_fields),
        "overall": overall.as_row(),
        "n_runs": len(by_run_rows),
        "n_groups": len(by_group_rows),
        "paths": {
            "by_run": str(by_run_path),
            "by_group": str(by_group_path),
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
            "Comma-separated run metadata fields for candidate_trace_by_group.csv. "
            f"Default: {', '.join(DEFAULT_GROUP_FIELDS)}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = summarize_candidate_trace(
        trace_path=args.trace,
        runs_path=args.runs,
        output_dir=args.output_dir,
        group_fields=_parse_group_fields(args.group_fields),
    )
    print(f"Saved candidate trace summary to {payload['paths']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
