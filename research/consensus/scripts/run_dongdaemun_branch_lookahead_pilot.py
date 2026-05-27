"""Run staged Dongdaemun branch-lookahead Leiden pilots.

This runner executes the policy that the offline branch-lookahead analyzer
simulates: screen all branch candidates at iter5, promote a small beam to
iter10, then convergence-polish the selected top candidate.  It writes the same
row schema as ``run_leiden_random_refinement_profile.py`` for each actually
executed stage so the offline analyzer can consume pilot outputs directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import analyze_leiden_branch_lookahead as branch_analysis  # noqa: E402
import run_leiden_random_refinement_profile as profiler  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path("research/consensus/results/adaptive_refinement")
    / "dongdaemun_branch_lookahead_pilot_20260511"
)

POLICY_ITER5_TOP3 = "iter5_screen_all_top3"
POLICY_ITER5_TOP5 = "iter5_screen_all_top5"
POLICY_MARGIN_POLISH_TOP2 = "margin_polish_top2"
SUPPORTED_POLICIES = (
    POLICY_ITER5_TOP3,
    POLICY_ITER5_TOP5,
    POLICY_MARGIN_POLISH_TOP2,
)

STAGE_SUMMARY_FILENAME = "dongdaemun_branch_lookahead_stage_summary.csv"
SUMMARY_FILENAME = "dongdaemun_branch_lookahead_pilot_summary.json"
REPORT_FILENAME = "dongdaemun_branch_lookahead_pilot_report.md"

STAGE_SUMMARY_FIELDS = [
    "sample",
    "source_sample",
    "edge_layer",
    "summary_path",
    "policy_name",
    "n_stage1_candidates",
    "n_promoted_iter10",
    "n_polished_convergence",
    "iter5_candidate_ids",
    "iter10_promoted_candidate_ids",
    "convergence_polished_candidate_ids",
    "selected_before_polish_candidate_id",
    "selected_candidate_id",
    "selected_seed",
    "selected_randomness",
    "selected_budget",
    "selected_quality",
    "selected_max_doc_weight_ratio",
    "selected_n_above_max_doc_weight",
    "stage1_elapsed_sec",
    "iter10_elapsed_sec",
    "convergence_elapsed_sec",
    "total_elapsed_sec",
]


def _candidate_key(row: dict[str, Any]) -> tuple[int, float]:
    return (int(row["seed"]), float(row["randomness"]))


def _candidate_id_from_key(key: tuple[int, float]) -> str:
    seed, randomness = key
    return f"seed={seed}|randomness={randomness:g}"


def _candidate_ids(keys: Iterable[tuple[int, float]]) -> list[str]:
    return [_candidate_id_from_key(key) for key in keys]


def _rows_for_keys(
    rows_by_candidate: dict[tuple[int, float], dict[str, Any]],
    keys: Iterable[tuple[int, float]],
) -> list[dict[str, Any]]:
    return [rows_by_candidate[key] for key in keys if key in rows_by_candidate]


def _rows_by_candidate(rows: Iterable[dict[str, Any]]) -> dict[tuple[int, float], dict[str, Any]]:
    return {_candidate_key(row): row for row in rows}


def _best_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("supported") and row.get("quality") is not None]
    if not eligible:
        return None
    return min(eligible, key=branch_analysis._selection_sort_key)


def _sum_elapsed(rows: Iterable[dict[str, Any]]) -> float:
    return float(
        sum(
            elapsed
            for elapsed in (
                branch_analysis._finite_float(row.get("elapsed_sec")) for row in rows
            )
            if elapsed is not None
        )
    )


def promoted_keys_for_policy(
    policy_name: str,
    iter5_rows: list[dict[str, Any]],
) -> list[tuple[int, float]]:
    """Return iter10 promotion keys from iter5 screening rows."""

    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported branch-lookahead policy: {policy_name}")
    promote_k = 5 if policy_name == POLICY_ITER5_TOP5 else 3
    ranked = sorted(iter5_rows, key=branch_analysis._selection_sort_key)
    return [_candidate_key(row) for row in ranked[: min(promote_k, len(ranked))]]


def convergence_polish_keys_for_policy(
    policy_name: str,
    iter10_rows: list[dict[str, Any]],
) -> list[tuple[int, float]]:
    """Return convergence polish keys from promoted iter10 rows."""

    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported branch-lookahead policy: {policy_name}")
    ranked = sorted(iter10_rows, key=branch_analysis._selection_sort_key)
    if not ranked:
        return []
    polish_k = (
        branch_analysis._margin_convergence_polish_k(ranked)
        if policy_name == POLICY_MARGIN_POLISH_TOP2
        else 1
    )
    return [_candidate_key(row) for row in ranked[: min(polish_k, len(ranked))]]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: profiler._csv_value(row.get(field)) for field in fieldnames}
            )


def _run_budget_row(
    *,
    profile: profiler.ProfileInput,
    graph: Any,
    node_weights: Any,
    seed: int,
    randomness: float,
    budget: profiler.IterationBudget,
    trace_path: Path,
    trace_runs_path: Path,
    rows_jsonl_path: Path,
) -> dict[str, Any]:
    row = profiler._run_one(
        profile=profile,
        graph=graph,
        node_weights=node_weights,
        seed=int(seed),
        randomness=float(randomness),
        budget=budget,
        trace_path=trace_path,
        trace_runs_path=trace_runs_path,
    )
    profiler._append_jsonl(rows_jsonl_path, row)
    return row


def _stage_summary_row(
    *,
    profile: profiler.ProfileInput,
    policy_name: str,
    iter5_rows: list[dict[str, Any]],
    iter10_rows: list[dict[str, Any]],
    convergence_rows: list[dict[str, Any]],
    promoted_keys: list[tuple[int, float]],
    polish_keys: list[tuple[int, float]],
    selected_before_polish: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    selected_key = _candidate_key(selected)
    selected_before_key = _candidate_key(selected_before_polish)
    selected_budget = (
        "convergence" if selected in convergence_rows else "10"
    )
    return {
        "sample": profile.sample,
        "source_sample": profile.source_sample,
        "edge_layer": profile.edge_layer,
        "summary_path": profiler.pilot._rel(profile.summary_path),
        "policy_name": policy_name,
        "n_stage1_candidates": len(iter5_rows),
        "n_promoted_iter10": len(iter10_rows),
        "n_polished_convergence": len(convergence_rows),
        "iter5_candidate_ids": _candidate_ids(_candidate_key(row) for row in iter5_rows),
        "iter10_promoted_candidate_ids": _candidate_ids(promoted_keys),
        "convergence_polished_candidate_ids": _candidate_ids(polish_keys),
        "selected_before_polish_candidate_id": _candidate_id_from_key(selected_before_key),
        "selected_candidate_id": _candidate_id_from_key(selected_key),
        "selected_seed": selected_key[0],
        "selected_randomness": selected_key[1],
        "selected_budget": selected_budget,
        "selected_quality": selected.get("quality"),
        "selected_max_doc_weight_ratio": selected.get("max_doc_weight_ratio"),
        "selected_n_above_max_doc_weight": selected.get("n_above_max_doc_weight"),
        "stage1_elapsed_sec": _sum_elapsed(iter5_rows),
        "iter10_elapsed_sec": _sum_elapsed(iter10_rows),
        "convergence_elapsed_sec": _sum_elapsed(convergence_rows),
        "total_elapsed_sec": _sum_elapsed(iter5_rows)
        + _sum_elapsed(iter10_rows)
        + _sum_elapsed(convergence_rows),
    }


def _write_report(path: Path, payload: dict[str, Any], stage_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Dongdaemun Branch Lookahead Pilot",
        "",
        f"- Policy: `{payload['policy_name']}`",
        f"- Executed rows: {payload['n_rows']}",
        f"- Stage summaries: {len(stage_rows)}",
        "",
        "## Selected Branches",
        "",
        "| sample | layer | selected | budget | quality | max_doc_weight_ratio | n_above_max_doc_weight | total_elapsed_sec |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        lines.append(
            "| {sample} | {layer} | {selected} | {budget} | {quality:.6f} | {pressure:.6f} | {above} | {elapsed:.3f} |".format(
                sample=row.get("sample", ""),
                layer=row.get("edge_layer", ""),
                selected=row.get("selected_candidate_id", ""),
                budget=row.get("selected_budget", ""),
                quality=float(row.get("selected_quality") or 0.0),
                pressure=float(row.get("selected_max_doc_weight_ratio") or 0.0),
                above=int(row.get("selected_n_above_max_doc_weight") or 0),
                elapsed=float(row.get("total_elapsed_sec") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Stage Counts",
            "",
            "| sample | iter5_candidates | iter10_promoted | convergence_polished |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in stage_rows:
        lines.append(
            "| {sample} | {n5} | {n10} | {nc} |".format(
                sample=row.get("sample", ""),
                n5=row.get("n_stage1_candidates", ""),
                n10=row.get("n_promoted_iter10", ""),
                nc=row.get("n_polished_convergence", ""),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_branch_lookahead_pilot(
    *,
    summaries: tuple[Path, ...],
    output_dir: Path,
    seeds: tuple[int, ...],
    randomness_values: tuple[float, ...],
    policy_name: str = POLICY_ITER5_TOP3,
    resume: bool = False,
) -> dict[str, Any]:
    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported branch-lookahead policy: {policy_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_jsonl_path = output_dir / profiler.ROWS_JSONL_FILENAME
    rows_csv_path = output_dir / profiler.ROWS_CSV_FILENAME
    trace_path = output_dir / profiler.QUALITY_TRACE_FILENAME
    trace_runs_path = output_dir / profiler.QUALITY_TRACE_RUNS_FILENAME
    trace_summary_dir = output_dir / "quality_trace_summary"
    stage_summary_path = output_dir / STAGE_SUMMARY_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    report_path = output_dir / REPORT_FILENAME

    if not resume:
        for path in (rows_jsonl_path, rows_csv_path, trace_runs_path):
            if path.exists():
                path.unlink()

    iter5_budget = profiler._parse_n_iterations_value("5")
    iter10_budget = profiler._parse_n_iterations_value("10")
    convergence_budget = profiler._parse_n_iterations_value("convergence")
    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    with profiler._quality_trace_path_context(trace_path, resume=resume):
        for summary_path_input in summaries:
            profile = profiler._resolve_profile_input(summary_path_input)
            n_nodes = profiler.pilot._infer_n_nodes(profile.input_cfg)
            node_weights = profiler.pilot._load_node_weights(
                profile.input_cfg.node_weights_path,
                n_nodes,
            )
            graph = profiler.pilot._load_graph(profile.input_cfg, node_weights)

            iter5_rows: list[dict[str, Any]] = []
            for seed in seeds:
                for randomness in randomness_values:
                    row = _run_budget_row(
                        profile=profile,
                        graph=graph,
                        node_weights=node_weights,
                        seed=int(seed),
                        randomness=float(randomness),
                        budget=iter5_budget,
                        trace_path=trace_path,
                        trace_runs_path=trace_runs_path,
                        rows_jsonl_path=rows_jsonl_path,
                    )
                    iter5_rows.append(row)
                    rows.append(row)

            promoted_keys = promoted_keys_for_policy(policy_name, iter5_rows)
            iter10_rows: list[dict[str, Any]] = []
            for seed, randomness in promoted_keys:
                row = _run_budget_row(
                    profile=profile,
                    graph=graph,
                    node_weights=node_weights,
                    seed=seed,
                    randomness=randomness,
                    budget=iter10_budget,
                    trace_path=trace_path,
                    trace_runs_path=trace_runs_path,
                    rows_jsonl_path=rows_jsonl_path,
                )
                iter10_rows.append(row)
                rows.append(row)

            polish_keys = convergence_polish_keys_for_policy(policy_name, iter10_rows)
            convergence_rows: list[dict[str, Any]] = []
            for seed, randomness in polish_keys:
                row = _run_budget_row(
                    profile=profile,
                    graph=graph,
                    node_weights=node_weights,
                    seed=seed,
                    randomness=randomness,
                    budget=convergence_budget,
                    trace_path=trace_path,
                    trace_runs_path=trace_runs_path,
                    rows_jsonl_path=rows_jsonl_path,
                )
                convergence_rows.append(row)
                rows.append(row)

            selected_before_polish = _best_row(iter10_rows) or _best_row(iter5_rows)
            selected = _best_row(convergence_rows) or selected_before_polish
            if selected_before_polish is None or selected is None:
                raise RuntimeError(f"No successful branch candidates for {profile.sample}")
            stage_rows.append(
                _stage_summary_row(
                    profile=profile,
                    policy_name=policy_name,
                    iter5_rows=iter5_rows,
                    iter10_rows=iter10_rows,
                    convergence_rows=convergence_rows,
                    promoted_keys=promoted_keys,
                    polish_keys=polish_keys,
                    selected_before_polish=selected_before_polish,
                    selected=selected,
                )
            )

    trace_payload = profiler.quality_summary.summarize_quality_trace(
        trace_path=trace_path,
        runs_path=trace_runs_path,
        output_dir=trace_summary_dir,
        group_fields=("sample", "variant", "seed", "randomness", "requested_n_iterations"),
    )
    profiler._enrich_rows_with_trace_summary(rows, Path(trace_payload["paths"]["by_run"]))
    profiler._enrich_rows_with_iteration_budget_metrics(rows)
    profiler._write_jsonl(rows_jsonl_path, rows)
    profiler._write_csv(rows_csv_path, rows)
    iteration_budget_paths = profiler._write_iteration_budget_outputs(
        output_dir=output_dir,
        rows=rows,
    )
    _write_csv(stage_summary_path, stage_rows, STAGE_SUMMARY_FIELDS)

    payload = {
        "schema": "dongdaemun_branch_lookahead_pilot.v1",
        "output_dir": str(output_dir),
        "policy_name": policy_name,
        "grid": {
            "seeds": list(seeds),
            "randomness_values": list(randomness_values),
            "stage_budgets": ["5", "10", "convergence"],
        },
        "n_rows": len(rows),
        "n_stage_summaries": len(stage_rows),
        "selected_by_sample": {
            str(row["sample"]): {
                "edge_layer": row.get("edge_layer"),
                "selected_candidate_id": row.get("selected_candidate_id"),
                "selected_seed": row.get("selected_seed"),
                "selected_randomness": row.get("selected_randomness"),
                "selected_budget": row.get("selected_budget"),
                "selected_quality": row.get("selected_quality"),
                "selected_max_doc_weight_ratio": row.get("selected_max_doc_weight_ratio"),
                "selected_n_above_max_doc_weight": row.get("selected_n_above_max_doc_weight"),
                "total_elapsed_sec": row.get("total_elapsed_sec"),
            }
            for row in stage_rows
        },
        "paths": {
            "rows_jsonl": str(rows_jsonl_path),
            "rows_csv": str(rows_csv_path),
            "quality_trace": str(trace_path),
            "quality_trace_runs": str(trace_runs_path),
            "quality_trace_summary": trace_payload["paths"]["summary"],
            "iteration_budget_by_run": iteration_budget_paths["iteration_budget_by_run"],
            "iteration_budget_by_group": iteration_budget_paths["iteration_budget_by_group"],
            "stage_summary": str(stage_summary_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
    }
    profiler._write_json(summary_path, payload)
    _write_report(report_path, payload, stage_rows)
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
        "--policy",
        choices=SUPPORTED_POLICIES,
        default=POLICY_ITER5_TOP3,
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in profiler.DEFAULT_SEEDS),
        help="Comma-separated Leiden seeds.",
    )
    parser.add_argument(
        "--randomness-values",
        default=",".join(str(value) for value in profiler.DEFAULT_RANDOMNESS),
        help="Comma-separated refinement randomness values.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summaries = tuple(args.summaries) if args.summaries else profiler.DEFAULT_SUMMARIES
    payload = run_branch_lookahead_pilot(
        summaries=summaries,
        output_dir=args.output_dir,
        seeds=tuple(int(value) for value in profiler._parse_csv_tuple(args.seeds, cast=int)),
        randomness_values=tuple(
            float(value)
            for value in profiler._parse_csv_tuple(args.randomness_values, cast=float)
        ),
        policy_name=str(args.policy),
        resume=bool(args.resume),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
