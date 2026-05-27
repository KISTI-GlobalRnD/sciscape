"""Analyze Dongdaemun trajectory divergence and near-tie probe traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FIRST_DIVERGENCE_COLUMNS = [
    "run_id_left",
    "run_id_right",
    "first_divergence_phase",
    "first_divergence_iteration",
    "first_divergence_depth",
    "left_membership_hash",
    "right_membership_hash",
    "left_quality",
    "right_quality",
    "final_quality_delta",
]

MARGIN_SUMMARY_COLUMNS = [
    "run_id",
    "depth",
    "parent_id",
    "parent_visit_index",
    "source",
    "decision_count",
    "low_margin_decision_count",
    "changed_decision_count",
    "min_margin",
    "p10_margin",
    "p50_margin",
    "selected_child_count",
    "largest_child_fraction",
]

POLICY_COMPARISON_COLUMNS = [
    "run_id",
    "mode",
    "source",
    "candidate_delta_q",
    "baseline_candidate_delta_q",
    "gain_vs_baseline",
    "valid",
    "quality_passes",
    "local_win",
    "commit_eligible",
    "committed",
    "commit_block_reason",
    "near_tie_low_margin_decision_count",
    "near_tie_changed_decision_count",
]

LOCAL_SHAKE_COLUMNS = [
    "run_id",
    "mode",
    "arm",
    "candidate_delta_q",
    "current_candidate_delta_q",
    "gain_vs_current",
    "distinct",
    "valid",
    "quality_passes",
    "commit_eligible",
    "commit_block_reason",
    "committed",
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


def phase_checkpoints(events: Iterable[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        if event.get("event") != "phase_checkpoint":
            continue
        run_id = str(event.get("run_id") or "")
        if run_id:
            grouped[run_id].append(event)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                int(row.get("iteration") or 0),
                int(row.get("depth") or 0),
                str(row.get("phase") or ""),
            )
        )
    return grouped


def first_divergence_row(
    left_run_id: str,
    right_run_id: str,
    checkpoints_by_run: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    left = checkpoints_by_run.get(left_run_id, [])
    right = checkpoints_by_run.get(right_run_id, [])
    final_left = left[-1] if left else {}
    final_right = right[-1] if right else {}
    row: dict[str, object] = {
        "run_id_left": left_run_id,
        "run_id_right": right_run_id,
        "first_divergence_phase": "",
        "first_divergence_iteration": "",
        "first_divergence_depth": "",
        "left_membership_hash": "",
        "right_membership_hash": "",
        "left_quality": final_left.get("quality", ""),
        "right_quality": final_right.get("quality", ""),
        "final_quality_delta": _as_float(final_right.get("quality"))
        - _as_float(final_left.get("quality")),
    }
    for left_event, right_event in zip(left, right):
        if left_event.get("membership_hash") == right_event.get("membership_hash"):
            continue
        row.update(
            {
                "first_divergence_phase": left_event.get("phase", ""),
                "first_divergence_iteration": left_event.get("iteration", ""),
                "first_divergence_depth": left_event.get("depth", ""),
                "left_membership_hash": left_event.get("membership_hash", ""),
                "right_membership_hash": right_event.get("membership_hash", ""),
                "left_quality": left_event.get("quality", ""),
                "right_quality": right_event.get("quality", ""),
            }
        )
        break
    return row


def build_first_divergence_rows(
    events: Iterable[dict[str, object]],
    pairs: Iterable[tuple[str, str]],
) -> list[dict[str, object]]:
    checkpoints_by_run = phase_checkpoints(events)
    return [
        first_divergence_row(left, right, checkpoints_by_run)
        for left, right in pairs
    ]


def build_margin_summary_rows(events: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for event in events:
        if event.get("event") != "local_merge_margin_summary":
            continue
        rows.append({column: event.get(column, "") for column in MARGIN_SUMMARY_COLUMNS})
    return rows


def build_policy_comparison_rows(events: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for event in events:
        if event.get("event") != "adaptive_probe_candidate":
            continue
        if event.get("source") != "near_tie_refinement_probe":
            continue
        rows.append({column: event.get(column, "") for column in POLICY_COMPARISON_COLUMNS})
    return rows


def build_local_shake_rows(events: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for event in events:
        name = event.get("event")
        if name == "adaptive_local_shake_candidate":
            key = (
                str(event.get("run_id") or ""),
                str(event.get("depth") or ""),
                str(event.get("parent_id") or ""),
                str(event.get("parent_visit_index") or ""),
                str(event.get("candidate_index") or ""),
            )
            rows_by_key[key] = {column: event.get(column, "") for column in LOCAL_SHAKE_COLUMNS}
        elif name == "adaptive_local_shake_decision" and _truthy(event.get("committed")):
            key = (
                str(event.get("run_id") or ""),
                str(event.get("depth") or ""),
                str(event.get("parent_id") or ""),
                str(event.get("parent_visit_index") or ""),
                str(event.get("selected_candidate_index") or ""),
            )
            row = rows_by_key.get(key)
            if row is not None:
                row["committed"] = True
    return list(rows_by_key.values())


def write_report(
    output_dir: Path,
    first_rows: list[dict[str, object]],
    margin_rows: list[dict[str, object]],
    policy_rows: list[dict[str, object]],
    local_shake_rows: list[dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "trajectory_divergence_report.md"
    divergent = sum(1 for row in first_rows if row.get("first_divergence_phase"))
    committed = sum(1 for row in policy_rows if _truthy(row.get("committed")))
    local_wins = sum(1 for row in policy_rows if _truthy(row.get("local_win")))
    local_shake_commits = sum(
        1 for row in local_shake_rows if _truthy(row.get("committed"))
    )
    local_shake_candidates = len(local_shake_rows)
    changed = sum(
        1
        for row in policy_rows
        if _as_float(row.get("near_tie_changed_decision_count")) > 0.0
    )
    block_reasons: dict[str, int] = defaultdict(int)
    for row in policy_rows:
        reason = str(row.get("commit_block_reason") or "")
        if reason:
            block_reasons[reason] += 1
    block_text = ", ".join(
        f"{reason}={count}" for reason, count in sorted(block_reasons.items())
    )
    report_path.write_text(
        "\n".join(
            [
                "# Dongdaemun Trajectory Divergence Report",
                "",
                f"- Pair comparisons: {len(first_rows)}",
                f"- Divergent pairs: {divergent}",
                f"- Margin summaries: {len(margin_rows)}",
                f"- Near-tie probe rows: {len(policy_rows)}",
                f"- Near-tie changed rows: {changed}",
                f"- Near-tie local wins: {local_wins}",
                f"- Near-tie commits: {committed}",
                f"- Near-tie commit blocks: {block_text}",
                f"- Local-shake candidates: {local_shake_candidates}",
                f"- Local-shake commits: {local_shake_commits}",
                "",
            ]
        )
    )
    return report_path


def analyze_dongdaemun_trajectory_divergence_for_test(
    *,
    trajectory_events: list[dict[str, object]],
    candidate_events: list[dict[str, object]] | None,
    pairs: list[tuple[str, str]],
    output_dir: Path,
) -> dict[str, Path]:
    all_events = trajectory_events + list(candidate_events or [])
    first_rows = build_first_divergence_rows(trajectory_events, pairs)
    margin_rows = build_margin_summary_rows(trajectory_events)
    policy_rows = build_policy_comparison_rows(all_events)
    local_shake_rows = build_local_shake_rows(all_events)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": write_report(
            output_dir, first_rows, margin_rows, policy_rows, local_shake_rows
        ),
        "first_divergence": output_dir / "first_divergence_by_pair.csv",
        "margin_summary": output_dir / "near_tie_margin_summary.csv",
        "policy_comparison": output_dir / "near_tie_probe_policy_comparison.csv",
        "local_shake": output_dir / "adaptive_local_shake_policy_comparison.csv",
    }
    _write_csv(paths["first_divergence"], FIRST_DIVERGENCE_COLUMNS, first_rows)
    _write_csv(paths["margin_summary"], MARGIN_SUMMARY_COLUMNS, margin_rows)
    _write_csv(paths["policy_comparison"], POLICY_COMPARISON_COLUMNS, policy_rows)
    _write_csv(paths["local_shake"], LOCAL_SHAKE_COLUMNS, local_shake_rows)
    return paths


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_pair(value: str) -> tuple[str, str]:
    left, sep, right = value.partition(":")
    if not sep or not left or not right:
        raise argparse.ArgumentTypeError("pairs must use left_run_id:right_run_id")
    return left, right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path)
    parser.add_argument("--pair", action="append", type=_parse_pair, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    trajectory_events = load_jsonl(args.trajectory_trace)
    candidate_events = load_jsonl(args.candidate_trace) if args.candidate_trace else []
    analyze_dongdaemun_trajectory_divergence_for_test(
        trajectory_events=trajectory_events,
        candidate_events=candidate_events,
        pairs=args.pair,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
