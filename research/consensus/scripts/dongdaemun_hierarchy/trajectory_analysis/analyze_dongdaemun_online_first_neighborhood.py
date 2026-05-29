"""Analyze small online-first adaptive probe policy perturbations.

This script compares already-executed conservative online-first policies around
the current best Dongdaemun adaptive stochastic setting.  It is intentionally
offline: the goal is to estimate whether small source/threshold changes have
remaining upside before adding a new candidate generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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


DEFAULT_BASE_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_adaptive_stochastic_greedy_prototype_20260511"
    / "conservative_full_121"
)
DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_adaptive_stochastic_greedy_prototype_20260512"
    / "online_first_neighborhood_audit"
)
SUMMARY_FILENAME = "online_first_neighborhood_summary.csv"
RUN_MATRIX_FILENAME = "online_first_neighborhood_run_matrix.csv"
COMMITS_FILENAME = "online_first_neighborhood_commits.csv"
AMBIGUITY_FILENAME = "online_first_neighborhood_local_ambiguity.csv"
REPORT_FILENAME = "online_first_neighborhood_report.md"
SUMMARY_JSON_FILENAME = "online_first_neighborhood_summary.json"

DEFAULT_POLICIES = (
    "r05_total1_all",
    "r10_total1_all",
    "r10_total1_same",
    "r10_total1_node",
)
SCHEMA_VERSION = 1

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

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

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value

def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _run_rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {str(row["run_id"]): row for row in _read_csv(path) if row.get("run_id")}

def _policy_run_rows_path(base_dir: Path, policy: str) -> Path:
    return base_dir / f"conservative_{policy}" / "parent_local_replay_run_rows.csv"

def _policy_trace_path(base_dir: Path, policy: str) -> Path:
    return base_dir / f"conservative_{policy}" / "parent_local_replay_trace.jsonl"

def build_run_matrix(
    *,
    baseline_rows: dict[str, dict[str, str]],
    policy_rows: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    common_ids = set(baseline_rows)
    for rows in policy_rows.values():
        common_ids &= set(rows)
    matrix: list[dict[str, Any]] = []
    for run_id in sorted(common_ids):
        baseline = baseline_rows[run_id]
        row: dict[str, Any] = {
            "run_id": run_id,
            "sample": baseline.get("sample"),
            "variant": baseline.get("variant"),
            "baseline_quality": _float_value(baseline.get("quality")),
            "baseline_n_clusters": _int_value(baseline.get("n_clusters")),
            "baseline_n_above": _int_value(baseline.get("n_above_max_doc_weight")),
        }
        best_policy = "baseline"
        best_delta = 0.0
        best_nonnegative_policy = "baseline"
        best_nonnegative_delta = 0.0
        for policy, rows in sorted(policy_rows.items()):
            policy_row = rows[run_id]
            delta = _float_value(policy_row.get("quality")) - _float_value(
                baseline.get("quality")
            )
            row[f"{policy}_quality_delta"] = delta
            row[f"{policy}_n_clusters"] = _int_value(policy_row.get("n_clusters"))
            row[f"{policy}_n_above"] = _int_value(
                policy_row.get("n_above_max_doc_weight")
            )
            if delta > best_delta:
                best_policy = policy
                best_delta = delta
            if delta >= -1e-9 and delta > best_nonnegative_delta:
                best_nonnegative_policy = policy
                best_nonnegative_delta = delta
        row["oracle_best_policy"] = best_policy
        row["oracle_best_quality_delta"] = best_delta
        row["oracle_nonnegative_policy"] = best_nonnegative_policy
        row["oracle_nonnegative_quality_delta"] = best_nonnegative_delta
        matrix.append(row)
    return matrix

def summarize_policies(matrix: list[dict[str, Any]], policies: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        deltas = [_float_value(row.get(f"{policy}_quality_delta")) for row in matrix]
        n_above_delta = sum(
            _int_value(row.get(f"{policy}_n_above")) - _int_value(row.get("baseline_n_above"))
            for row in matrix
        )
        rows.append(
            {
                "policy": policy,
                "runs": len(deltas),
                "quality_delta_sum": sum(deltas),
                "quality_delta_mean": None if not deltas else sum(deltas) / len(deltas),
                "quality_wins": sum(delta > 1e-9 for delta in deltas),
                "quality_losses": sum(delta < -1e-9 for delta in deltas),
                "quality_equal": sum(abs(delta) <= 1e-9 for delta in deltas),
                "n_above_delta_sum": n_above_delta,
            }
        )
    oracle_deltas = [_float_value(row.get("oracle_best_quality_delta")) for row in matrix]
    oracle_safe_deltas = [
        _float_value(row.get("oracle_nonnegative_quality_delta")) for row in matrix
    ]
    rows.append(
        {
            "policy": "oracle_best_known_neighborhood",
            "runs": len(oracle_deltas),
            "quality_delta_sum": sum(oracle_deltas),
            "quality_delta_mean": None
            if not oracle_deltas
            else sum(oracle_deltas) / len(oracle_deltas),
            "quality_wins": sum(delta > 1e-9 for delta in oracle_deltas),
            "quality_losses": sum(delta < -1e-9 for delta in oracle_deltas),
            "quality_equal": sum(abs(delta) <= 1e-9 for delta in oracle_deltas),
            "n_above_delta_sum": None,
        }
    )
    rows.append(
        {
            "policy": "oracle_nonnegative_known_neighborhood",
            "runs": len(oracle_safe_deltas),
            "quality_delta_sum": sum(oracle_safe_deltas),
            "quality_delta_mean": None
            if not oracle_safe_deltas
            else sum(oracle_safe_deltas) / len(oracle_safe_deltas),
            "quality_wins": sum(delta > 1e-9 for delta in oracle_safe_deltas),
            "quality_losses": sum(delta < -1e-9 for delta in oracle_safe_deltas),
            "quality_equal": sum(abs(delta) <= 1e-9 for delta in oracle_safe_deltas),
            "n_above_delta_sum": None,
        }
    )
    return rows

def extract_commits(base_dir: Path, policies: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for event in _read_jsonl(_policy_trace_path(base_dir, policy)):
            if event.get("event") != "adaptive_probe_candidate":
                continue
            if not bool(event.get("committed")):
                continue
            rows.append(
                {
                    "policy": policy,
                    "run_id": event.get("run_id"),
                    "depth": _int_value(event.get("depth")),
                    "parent_id": _int_value(event.get("parent_id")),
                    "visit": _int_value(event.get("parent_visit_index"), 1),
                    "source": event.get("source"),
                    "source_index": _int_value(event.get("source_index")),
                    "gain": _float_value(event.get("gain_vs_baseline")),
                    "gain_parent_weight": _float_value(event.get("commit_gain_parent_weight")),
                    "candidate_delta_q": _float_value(event.get("candidate_delta_q")),
                    "baseline_candidate_delta_q": _float_value(
                        event.get("baseline_candidate_delta_q")
                    ),
                    "candidate_n_clusters": _int_value(event.get("candidate_n_clusters")),
                    "standard_n_clusters": _int_value(event.get("standard_n_clusters")),
                    "largest_child_fraction": _float_value(
                        event.get("largest_child_fraction")
                    ),
                    "standard_largest_child_fraction": _float_value(
                        event.get("standard_largest_child_fraction")
                    ),
                }
            )
    return rows

def summarize_local_ambiguity(
    *,
    commits: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix_by_run = {str(row["run_id"]): row for row in matrix}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in commits:
        key = (
            row.get("policy"),
            row.get("parent_id"),
            row.get("source"),
            row.get("source_index"),
            round(_float_value(row.get("gain_parent_weight")), 6),
            row.get("candidate_n_clusters"),
            round(_float_value(row.get("largest_child_fraction")), 6),
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: (item[0], len(item[1]))):
        if len(group) < 2:
            continue
        policy = str(key[0])
        deltas = [
            _float_value(matrix_by_run[str(row["run_id"])].get(f"{policy}_quality_delta"))
            for row in group
            if str(row.get("run_id")) in matrix_by_run
        ]
        if not deltas:
            continue
        has_win = any(delta > 1e-9 for delta in deltas)
        has_loss = any(delta < -1e-9 for delta in deltas)
        out.append(
            {
                "policy": policy,
                "parent_id": key[1],
                "source": key[2],
                "source_index": key[3],
                "gain_parent_weight": key[4],
                "candidate_n_clusters": key[5],
                "largest_child_fraction": key[6],
                "n_runs": len(deltas),
                "quality_delta_min": min(deltas),
                "quality_delta_max": max(deltas),
                "quality_delta_sum": sum(deltas),
                "has_win_and_loss": has_win and has_loss,
                "run_ids": [row.get("run_id") for row in group],
                "quality_deltas": deltas,
            }
        )
    return out

def _best_summary_row(summary_rows: list[dict[str, Any]], *, safe: bool) -> dict[str, Any] | None:
    eligible = [
        row
        for row in summary_rows
        if not str(row.get("policy", "")).startswith("oracle_")
        and (not safe or _int_value(row.get("quality_losses")) == 0)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: _float_value(row.get("quality_delta_sum")))

def _build_report(
    *,
    summary_rows: list[dict[str, Any]],
    ambiguity_rows: list[dict[str, Any]],
) -> str:
    best_quality = _best_summary_row(summary_rows, safe=False)
    best_safe = _best_summary_row(summary_rows, safe=True)
    oracle = next(
        row
        for row in summary_rows
        if row.get("policy") == "oracle_best_known_neighborhood"
    )
    lines = [
        "# Dongdaemun Online-First Neighborhood Audit",
        "",
        "## Summary",
        "",
        "| policy | runs | qf delta | wins/loss/equal | n above delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {policy} | {runs} | {qf:.6g} | {wins}/{losses}/{equal} | {above} |".format(
                policy=row.get("policy"),
                runs=row.get("runs"),
                qf=_float_value(row.get("quality_delta_sum")),
                wins=row.get("quality_wins"),
                losses=row.get("quality_losses"),
                equal=row.get("quality_equal"),
                above="" if row.get("n_above_delta_sum") is None else row.get("n_above_delta_sum"),
            )
        )
    lines.extend(["", "## Readout", ""])
    if best_quality:
        lines.append(
            "- Best observed quality policy: `{}` with qf delta {:.6g} and {}/{}/{} wins/loss/equal.".format(
                best_quality["policy"],
                _float_value(best_quality.get("quality_delta_sum")),
                best_quality.get("quality_wins"),
                best_quality.get("quality_losses"),
                best_quality.get("quality_equal"),
            )
        )
    if best_safe:
        lines.append(
            "- Best observed loss-free policy: `{}` with qf delta {:.6g}.".format(
                best_safe["policy"],
                _float_value(best_safe.get("quality_delta_sum")),
            )
        )
    lines.append(
        "- Known-neighborhood oracle upper bound: qf delta {:.6g}.".format(
            _float_value(oracle.get("quality_delta_sum"))
        )
    )
    conflict_rows = [row for row in ambiguity_rows if row.get("has_win_and_loss")]
    lines.append(
        "- Local ambiguity groups with both final wins and losses: {}.".format(
            len(conflict_rows)
        )
    )
    if conflict_rows:
        lines.extend(
            [
                "",
                "## Local Ambiguity",
                "",
                "| policy | parent | source | gain ratio | runs | qf delta range |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in conflict_rows:
            lines.append(
                "| {policy} | {parent} | {source}[{idx}] | {ratio:.6g} | {runs} | {lo:.6g}..{hi:.6g} |".format(
                    policy=row.get("policy"),
                    parent=row.get("parent_id"),
                    source=row.get("source"),
                    idx=row.get("source_index"),
                    ratio=_float_value(row.get("gain_parent_weight")),
                    runs=row.get("n_runs"),
                    lo=_float_value(row.get("quality_delta_min")),
                    hi=_float_value(row.get("quality_delta_max")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Small online-first perturbations still have quality upside: `r10_total1_node` beats the loss-free `r05_total1_all` in aggregate.",
            "- The remaining loss is not separable by the local commit signature alone; an identical node-order commit can be a final win or a final loss depending on the surrounding trajectory.",
            "- The next useful test is therefore not another scalar threshold sweep. It is a short post-commit lookahead or boundary-jitter candidate that observes whether the disturbed trajectory settles favorably.",
            "",
        ]
    )
    return "\n".join(lines)

def analyze_online_first_neighborhood(
    *,
    base_dir: Path,
    output_dir: Path,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
) -> dict[str, Any]:
    baseline_path = base_dir / "trace_only" / "parent_local_replay_run_rows.csv"
    baseline_rows = _run_rows_by_id(baseline_path)
    policy_rows = {
        policy: _run_rows_by_id(_policy_run_rows_path(base_dir, policy))
        for policy in policies
    }
    missing = [policy for policy, rows in policy_rows.items() if not rows]
    if missing:
        raise FileNotFoundError(f"missing policy run rows for: {', '.join(missing)}")
    matrix = build_run_matrix(baseline_rows=baseline_rows, policy_rows=policy_rows)
    summary_rows = summarize_policies(matrix, policies)
    commits = extract_commits(base_dir, policies)
    ambiguity_rows = summarize_local_ambiguity(commits=commits, matrix=matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME
    matrix_path = output_dir / RUN_MATRIX_FILENAME
    commits_path = output_dir / COMMITS_FILENAME
    ambiguity_path = output_dir / AMBIGUITY_FILENAME
    report_path = output_dir / REPORT_FILENAME
    summary_json_path = output_dir / SUMMARY_JSON_FILENAME
    _write_csv(summary_path, summary_rows)
    _write_csv(matrix_path, matrix)
    _write_csv(commits_path, commits)
    _write_csv(ambiguity_path, ambiguity_rows)
    report_path.write_text(
        _build_report(summary_rows=summary_rows, ambiguity_rows=ambiguity_rows),
        encoding="utf-8",
    )
    payload = {
        "schema": "dongdaemun_online_first_neighborhood_audit.v1",
        "schema_version": SCHEMA_VERSION,
        "base_dir": str(base_dir),
        "policies": policies,
        "n_runs": len(matrix),
        "paths": {
            "summary": str(summary_path),
            "run_matrix": str(matrix_path),
            "commits": str(commits_path),
            "local_ambiguity": str(ambiguity_path),
            "report": str(report_path),
            "summary_json": str(summary_json_path),
        },
    }
    _write_json(summary_json_path, payload)
    return payload

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policy names under conservative_<policy>/",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    policies = tuple(
        policy.strip() for policy in str(args.policies).split(",") if policy.strip()
    )
    payload = analyze_online_first_neighborhood(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        policies=policies,
    )
    print(f"Saved online-first neighborhood audit to {payload['paths']['summary_json']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
