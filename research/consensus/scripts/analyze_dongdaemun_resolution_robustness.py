"""Summarize parent-local resolution perturbation robustness traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SUMMARY_COLUMNS = [
    "run_id",
    "direction",
    "gamma_multiplier",
    "n_candidates",
    "valid_count",
    "quality_passes_count",
    "selected_count",
    "qpos_spos_count",
    "qpos_sneg_count",
    "qneg_spos_count",
    "qneg_sneg_count",
    "mean_candidate_delta_q",
    "max_candidate_delta_q",
    "mean_largest_child_fraction",
    "mean_largest_child_fraction_improvement",
]


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_resolution_robustness_rows(
    events: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "candidate_profile":
            continue
        gamma = _as_float(event.get("gamma_multiplier"), default=1.0)
        if not math.isfinite(gamma) or gamma <= 0.0:
            continue
        if abs(gamma - 1.0) <= 1e-12:
            continue
        direction = "down" if gamma < 1.0 else "up"
        run_id = str(event.get("run_id") or "")
        if not run_id:
            continue
        groups[(run_id, direction, gamma)].append(event)

    rows: list[dict[str, object]] = []
    for (run_id, direction, gamma), group in sorted(groups.items()):
        deltas = [_as_float(row.get("candidate_delta_q")) for row in group]
        largest = [_as_float(row.get("largest_child_fraction")) for row in group]
        improvements = [
            _as_float(row.get("largest_child_fraction_improvement")) for row in group
        ]
        rows.append(
            {
                "run_id": run_id,
                "direction": direction,
                "gamma_multiplier": gamma,
                "n_candidates": len(group),
                "valid_count": sum(_truthy(row.get("valid")) for row in group),
                "quality_passes_count": sum(
                    _truthy(row.get("quality_passes")) for row in group
                ),
                "selected_count": sum(
                    row.get("decision") == "selected_by_policy" for row in group
                ),
                "qpos_spos_count": sum(row.get("quadrant") == "qpos_spos" for row in group),
                "qpos_sneg_count": sum(row.get("quadrant") == "qpos_sneg" for row in group),
                "qneg_spos_count": sum(row.get("quadrant") == "qneg_spos" for row in group),
                "qneg_sneg_count": sum(row.get("quadrant") == "qneg_sneg" for row in group),
                "mean_candidate_delta_q": _mean(deltas),
                "max_candidate_delta_q": max(deltas) if deltas else "",
                "mean_largest_child_fraction": _mean(largest),
                "mean_largest_child_fraction_improvement": _mean(improvements),
            }
        )
    return rows


def analyze_dongdaemun_resolution_robustness_for_test(
    *,
    candidate_events: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_resolution_robustness_rows(candidate_events)
    paths = {
        "summary_csv": output_dir / "resolution_perturbation_summary.csv",
        "report": output_dir / "resolution_perturbation_report.md",
    }
    _write_csv(paths["summary_csv"], rows)
    paths["report"].write_text(_build_report(rows), encoding="utf-8")
    return paths


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _build_report(rows: list[dict[str, object]]) -> str:
    down = [row for row in rows if row.get("direction") == "down"]
    up = [row for row in rows if row.get("direction") == "up"]
    selected_down = sum(int(row.get("selected_count") or 0) for row in down)
    selected_up = sum(int(row.get("selected_count") or 0) for row in up)
    return "\n".join(
        [
            "# Dongdaemun Resolution Perturbation Robustness",
            "",
            f"- Summary rows: {len(rows)}",
            f"- Down perturbation rows: {len(down)}",
            f"- Up perturbation rows: {len(up)}",
            f"- Down selected candidates: {selected_down}",
            f"- Up selected candidates: {selected_up}",
            "",
        ]
    )


def _mean(values: list[float]) -> float | str:
    finite = [value for value in values if math.isfinite(value)]
    return "" if not finite else sum(finite) / len(finite)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze_dongdaemun_resolution_robustness_for_test(
        candidate_events=load_jsonl(args.candidate_trace),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
