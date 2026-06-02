#!/usr/bin/env python3
"""Replay frozen tiny CPM endpoints as Leiden initial memberships.

This is a diagnostic for the Track C tiny-demo method surface. It asks whether
endpoint signatures missed by the handle probe are stable under direct replay,
or whether they collapse during ordinary Leiden + CPM polish. It is not a fair
method comparison, a wall/pathway trace, a quality/cost claim, or a
NanoClustering generality claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.clustering.runner import LeidenRunner

from run_leiden_cpm_tiny_demo_seed_sweep import (
    _canonical_groups,
    _classify_mechanism,
    _graph_cases,
    _json_safe,
    _signature_id,
    _write_csv,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_BASELINE_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_demo_seed_sweep_20260531"
DEFAULT_METHOD_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_handle_method_v1_20260531"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_tiny_cpm_endpoint_replay_v1_20260531"

FROZEN_ENDPOINT_MANIFEST_CSV = "leiden_cpm_tiny_demo_frozen_endpoint_manifest.csv"
METHOD_ENDPOINT_HITS_CSV = "tiny_cpm_method_endpoint_hits.csv"
REPLAY_RUNS_CSV = "tiny_cpm_endpoint_replay_runs.csv"
REPLAY_SUMMARY_CSV = "tiny_cpm_endpoint_replay_summary.csv"
MISSING_DIAGNOSIS_CSV = "tiny_cpm_missing_endpoint_replay_diagnosis.csv"
GATE_MATRIX_CSV = "tiny_cpm_endpoint_replay_gate_matrix.csv"
SUMMARY_JSON = "tiny_cpm_endpoint_replay_summary.json"
CONFIG_JSON = "tiny_cpm_endpoint_replay_config.json"
REPORT_MD = "tiny_cpm_endpoint_replay_report.md"

CLAIM_BOUNDARY = (
    "Tiny CPM endpoint-replay diagnostic only; frozen endpoint signatures are "
    "used as initial memberships for ordinary Leiden + CPM polish, no custom "
    "method comparison, no route execution, no wall/pathway promotion, no "
    "basin-quality claim, no cost claim, and no NanoClustering generality claim."
)
ROUTE_EXECUTION_STATUS = "not_route_trace_endpoint_replay_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_trace"
METHOD_STATUS = "diagnostic_replay_not_method_claim"


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _with_claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _initial_membership_from_groups(
    *,
    node_names: list[str],
    groups: list[list[str]],
) -> list[int]:
    assigned: dict[str, int] = {}
    for label, group in enumerate(groups):
        for node in group:
            assigned[str(node)] = int(label)

    missing = sorted(set(node_names) - set(assigned))
    unknown = sorted(set(assigned) - set(node_names))
    if unknown:
        raise ValueError(f"endpoint signature contains unknown nodes: {unknown}")

    next_label = len(groups)
    for node in missing:
        assigned[node] = next_label
        next_label += 1
    return [assigned[node] for node in node_names]


def _manifest_by_family(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(family): group.set_index("endpoint_signature_id", drop=False)
        for family, group in manifest.groupby("family", sort=True)
    }


def _matched_endpoint(
    *,
    family: str,
    signature_id: str,
    manifest_by_family: dict[str, pd.DataFrame],
) -> tuple[str | None, str | None, str | None]:
    family_manifest = manifest_by_family[family]
    if signature_id not in family_manifest.index:
        return None, None, None
    matched = family_manifest.loc[signature_id]
    return (
        str(matched["frozen_endpoint_id"]),
        str(matched["baseline_role"]),
        str(matched["mechanism_read"]),
    )


def _replay_outcome(
    *,
    target_signature_id: str,
    result_signature_id: str,
    result_frozen_endpoint_id: str | None,
) -> str:
    if result_signature_id == target_signature_id:
        return "replayed_same_endpoint"
    if result_frozen_endpoint_id is not None:
        return "collapsed_to_other_frozen_endpoint"
    return "new_endpoint_after_replay"


def _run_replay(
    *,
    manifest: pd.DataFrame,
    replay_seeds: int,
    n_iterations: int,
) -> pd.DataFrame:
    cases = {case.family: case for case in _graph_cases()}
    frozen_by_family = _manifest_by_family(manifest)
    rows: list[dict[str, Any]] = []

    manifest_rows = manifest.sort_values(["family", "endpoint_rank_in_family"])
    for frozen in manifest_rows.itertuples(index=False):
        family = str(frozen.family)
        case = cases[family]
        graph = case.builder()
        node_names = list(map(str, graph.vs["name"]))
        runner = LeidenRunner(graph, objective="cpm", default_iterations=n_iterations)
        endpoint_groups = json.loads(str(frozen.endpoint_signature))
        initial = _initial_membership_from_groups(
            node_names=node_names,
            groups=endpoint_groups,
        )

        for replay_seed in range(replay_seeds):
            result = runner.run(
                float(frozen.gamma),
                seed=replay_seed,
                initial_membership=initial,
            )
            membership = list(map(int, result.membership))
            result_groups = _canonical_groups(graph, membership)
            result_signature_id = _signature_id(result_groups)
            (
                result_frozen_endpoint_id,
                result_baseline_role,
                result_manifest_mechanism_read,
            ) = _matched_endpoint(
                family=family,
                signature_id=result_signature_id,
                manifest_by_family=frozen_by_family,
            )
            result_mechanism_read = _classify_mechanism(family, graph, membership)
            rows.append(
                {
                    "family": family,
                    "target_frozen_endpoint_id": str(frozen.frozen_endpoint_id),
                    "target_endpoint_rank_in_family": int(frozen.endpoint_rank_in_family),
                    "target_baseline_role": str(frozen.baseline_role),
                    "target_is_recurrent_endpoint": bool(frozen.is_recurrent_endpoint),
                    "target_seed_count": int(frozen.seed_count),
                    "target_seed_share": float(frozen.seed_share),
                    "target_mechanism_read": str(frozen.mechanism_read),
                    "target_quality_median": float(frozen.quality_median),
                    "target_cluster_count_min": int(frozen.cluster_count_min),
                    "target_cluster_count_max": int(frozen.cluster_count_max),
                    "target_endpoint_signature_id": str(frozen.endpoint_signature_id),
                    "replay_seed": int(replay_seed),
                    "gamma": float(frozen.gamma),
                    "result_endpoint_signature_id": result_signature_id,
                    "result_frozen_endpoint_id": result_frozen_endpoint_id,
                    "result_baseline_role": result_baseline_role,
                    "result_manifest_mechanism_read": result_manifest_mechanism_read,
                    "result_mechanism_read": result_mechanism_read,
                    "replay_outcome": _replay_outcome(
                        target_signature_id=str(frozen.endpoint_signature_id),
                        result_signature_id=result_signature_id,
                        result_frozen_endpoint_id=result_frozen_endpoint_id,
                    ),
                    "result_cluster_count": int(result.cluster_count),
                    "result_quality": float(result.quality),
                    "quality_delta_vs_target_median": float(
                        result.quality - float(frozen.quality_median)
                    ),
                    "result_endpoint_signature": json.dumps(
                        result_groups,
                        sort_keys=True,
                    ),
                }
            )
    return _with_claim_columns(pd.DataFrame(rows).sort_values(["family", "target_endpoint_rank_in_family", "replay_seed"]))


def _summarize_replay(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "family",
        "target_frozen_endpoint_id",
        "target_endpoint_rank_in_family",
        "target_baseline_role",
        "target_is_recurrent_endpoint",
        "target_seed_count",
        "target_seed_share",
        "target_mechanism_read",
        "target_quality_median",
        "target_cluster_count_min",
        "target_cluster_count_max",
        "target_endpoint_signature_id",
    ]
    for keys, group in runs.groupby(group_cols, sort=True, dropna=False):
        key = dict(zip(group_cols, keys))
        same = group["replay_outcome"].eq("replayed_same_endpoint")
        collapsed = group["replay_outcome"].eq("collapsed_to_other_frozen_endpoint")
        new = group["replay_outcome"].eq("new_endpoint_after_replay")
        replay_count = int(len(group))
        same_rate = float(same.mean()) if replay_count else 0.0
        if same_rate == 1.0:
            stability_class = "stable_all_replays"
        elif same_rate >= 0.8:
            stability_class = "mostly_stable"
        elif bool(new.any()):
            stability_class = "unstable_creates_new_endpoint"
        else:
            stability_class = "unstable_collapses_to_other_frozen_endpoint"

        matched_ids = sorted(
            {
                str(value)
                for value in group["result_frozen_endpoint_id"].dropna().tolist()
            }
        )
        rows.append(
            {
                **key,
                "replay_count": replay_count,
                "same_endpoint_count": int(same.sum()),
                "same_endpoint_rate": same_rate,
                "collapsed_to_other_frozen_count": int(collapsed.sum()),
                "new_endpoint_after_replay_count": int(new.sum()),
                "unique_result_signature_count": int(
                    group["result_endpoint_signature_id"].nunique()
                ),
                "matched_result_frozen_endpoint_ids": ";".join(matched_ids),
                "result_mechanism_reads": ";".join(
                    sorted(set(group["result_mechanism_read"].astype(str)))
                ),
                "quality_delta_min": float(group["quality_delta_vs_target_median"].min()),
                "quality_delta_median": float(
                    group["quality_delta_vs_target_median"].median()
                ),
                "quality_delta_max": float(group["quality_delta_vs_target_median"].max()),
                "stability_class": stability_class,
            }
        )
    return _with_claim_columns(
        pd.DataFrame(rows).sort_values(["family", "target_endpoint_rank_in_family"])
    )


def _method_hit_ids(method_dir: Path) -> set[str]:
    path = method_dir / METHOD_ENDPOINT_HITS_CSV
    if not path.exists():
        return set()
    hits = pd.read_csv(path)
    if "frozen_endpoint_id" not in hits.columns:
        return set()
    return {str(value) for value in hits["frozen_endpoint_id"].dropna().tolist()}


def _missing_endpoint_diagnosis(
    *,
    manifest: pd.DataFrame,
    replay_summary: pd.DataFrame,
    method_dir: Path,
) -> pd.DataFrame:
    hit_ids = _method_hit_ids(method_dir)
    recurrent = manifest[manifest["is_recurrent_endpoint"].astype(bool)].copy()
    recurrent["method_v1_hit_status"] = recurrent["frozen_endpoint_id"].map(
        lambda endpoint: "hit_by_method_v1" if str(endpoint) in hit_ids else "missed_by_method_v1"
    )
    missed = recurrent[recurrent["method_v1_hit_status"].eq("missed_by_method_v1")].copy()
    if missed.empty:
        return _with_claim_columns(pd.DataFrame())

    merged = missed.merge(
        replay_summary,
        left_on="frozen_endpoint_id",
        right_on="target_frozen_endpoint_id",
        how="left",
        suffixes=("", "_replay"),
    )

    def diagnose(row: pd.Series) -> str:
        same_rate = float(row.get("same_endpoint_rate", 0.0))
        new_count = int(row.get("new_endpoint_after_replay_count", 0))
        collapsed_count = int(row.get("collapsed_to_other_frozen_count", 0))
        if same_rate == 1.0:
            return "method_handle_gap_stable_endpoint"
        if same_rate >= 0.8:
            return "probable_handle_gap_mostly_stable_endpoint"
        if new_count > 0:
            return "endpoint_replay_expands_universe_check_freeze"
        if collapsed_count > 0:
            return "not_stable_endpoint_collapses_under_polish"
        return "undiagnosed_missing_endpoint"

    merged["diagnosis"] = merged.apply(diagnose, axis=1)
    preferred = [
        "family",
        "frozen_endpoint_id",
        "endpoint_rank_in_family",
        "baseline_role",
        "mechanism_read",
        "seed_count",
        "seed_share",
        "quality_median",
        "method_v1_hit_status",
        "same_endpoint_rate",
        "stability_class",
        "matched_result_frozen_endpoint_ids",
        "result_mechanism_reads",
        "diagnosis",
    ]
    return _with_claim_columns(merged[preferred].sort_values(["family", "endpoint_rank_in_family"]))


def _gate_matrix(
    *,
    manifest: pd.DataFrame,
    runs: pd.DataFrame,
    replay_summary: pd.DataFrame,
    missing_diagnosis: pd.DataFrame,
    replay_seeds: int,
) -> pd.DataFrame:
    endpoint_count = int(manifest["frozen_endpoint_id"].nunique())
    expected_runs = endpoint_count * replay_seeds
    recurrent_summary = replay_summary[
        replay_summary["target_is_recurrent_endpoint"].astype(bool)
    ]
    min_recurrent_same_rate = (
        float(recurrent_summary["same_endpoint_rate"].min())
        if not recurrent_summary.empty
        else 0.0
    )
    unstable_recurrent = recurrent_summary[
        recurrent_summary["same_endpoint_rate"].lt(1.0)
    ]
    diffuse_missing = missing_diagnosis[
        missing_diagnosis["family"].eq("diffuse_fragment_star")
    ]
    stable_missing = missing_diagnosis[
        missing_diagnosis["diagnosis"].isin(
            [
                "method_handle_gap_stable_endpoint",
                "probable_handle_gap_mostly_stable_endpoint",
            ]
        )
    ]
    rows = [
        {
            "gate_id": "R1_endpoint_replay_executed",
            "gate_question": "Were all frozen tiny CPM endpoints replayed?",
            "evidence": f"endpoints={endpoint_count}, runs={len(runs)}, expected_runs={expected_runs}",
            "status": "pass" if len(runs) == expected_runs else "blocked_incomplete_replay_grid",
            "decision": "use_as_endpoint_stability_diagnostic",
            "next_action": "inspect recurrent endpoint stability before adding handles",
        },
        {
            "gate_id": "R2_recurrent_endpoint_stability",
            "gate_question": "Do recurrent frozen endpoints remain stable under direct replay?",
            "evidence": (
                f"min_recurrent_same_endpoint_rate={min_recurrent_same_rate:.6f}, "
                f"unstable_recurrent_count={len(unstable_recurrent)}"
            ),
            "status": "pass" if min_recurrent_same_rate == 1.0 else "caveat_required",
            "decision": "stable_recurrent_endpoints_can_be_treated_as_handle_targets_if_pass",
            "next_action": "separate handle gaps from endpoint instability",
        },
        {
            "gate_id": "R3_diffuse_missing_endpoint_diagnosis",
            "gate_question": "Are method-v1 diffuse misses diagnosed by replay?",
            "evidence": (
                f"diffuse_missing_recurrent_endpoints={len(diffuse_missing)}, "
                f"stable_or_mostly_stable_missing={len(stable_missing[stable_missing['family'].eq('diffuse_fragment_star')])}"
            ),
            "status": "pass" if not diffuse_missing.empty else "blocked_no_diffuse_miss_surface",
            "decision": "if_stable_then_current_hard_case_is_handle_design_gap",
            "next_action": "design coverage handles only for stable missed endpoints",
        },
        {
            "gate_id": "R4_method_gap_separated_from_endpoint_instability",
            "gate_question": "Can missed endpoints be classified before adding new candidates?",
            "evidence": (
                f"missing_recurrent_endpoints={len(missing_diagnosis)}, "
                f"stable_or_mostly_stable_missing={len(stable_missing)}"
            ),
            "status": "pass" if not missing_diagnosis.empty else "caveat_required",
            "decision": "use_replay_diagnosis_as_guardrail_for_next_handle_probe",
            "next_action": "avoid expanding handles toward unstable endpoint signatures",
        },
        {
            "gate_id": "R5_method_claim_gate",
            "gate_question": "Can endpoint replay claim a method improvement?",
            "evidence": "endpoint-signature replay diagnostic only",
            "status": "closed_excluded_by_design",
            "decision": "keep_algorithm_quality_cost_and_pathway_claims_closed",
            "next_action": "use only as diagnostic input to the next controlled method probe",
        },
    ]
    matrix = pd.DataFrame(rows)
    matrix["claim_boundary"] = CLAIM_BOUNDARY
    return matrix


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for _, row in rows.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", r"\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix = [f"\n_Showing {max_rows} of {len(frame)} rows._"] if len(frame) > max_rows else []
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    gate_matrix: pd.DataFrame,
    replay_summary: pd.DataFrame,
    missing_diagnosis: pd.DataFrame,
) -> None:
    recurrent = replay_summary[replay_summary["target_is_recurrent_endpoint"].astype(bool)]
    text = [
        "# Tiny CPM Endpoint Replay Diagnostic v1",
        "",
        f"- frozen_endpoint_count: `{summary['frozen_endpoint_count']}`",
        f"- replay_run_count: `{summary['replay_run_count']}`",
        f"- replay_seeds: `{summary['replay_seeds']}`",
        f"- min_recurrent_same_endpoint_rate: `{summary['min_recurrent_same_endpoint_rate']}`",
        f"- method_v1_missing_recurrent_endpoint_count: `{summary['method_v1_missing_recurrent_endpoint_count']}`",
        f"- stable_or_mostly_stable_missing_count: `{summary['stable_or_mostly_stable_missing_count']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "evidence", "status", "decision", "next_action"],
            max_rows=10,
        ),
        "",
        "## Recurrent Endpoint Replay Stability",
        "",
        _markdown_table(
            recurrent,
            [
                "family",
                "target_frozen_endpoint_id",
                "target_seed_count",
                "target_mechanism_read",
                "same_endpoint_rate",
                "stability_class",
                "matched_result_frozen_endpoint_ids",
            ],
            max_rows=30,
        ),
        "",
        "## Method-v1 Missed Recurrent Endpoint Diagnosis",
        "",
        _markdown_table(
            missing_diagnosis,
            [
                "family",
                "frozen_endpoint_id",
                "seed_count",
                "mechanism_read",
                "same_endpoint_rate",
                "stability_class",
                "diagnosis",
            ],
            max_rows=30,
        ),
        "",
        "## Read",
        "",
        "- Endpoint replay is a diagnostic, not a fair baseline-vs-method comparison.",
        "- Stable missed endpoints indicate a handle-design gap: the endpoint can survive direct Leiden polish, but v1 handles did not discover it.",
        "- Unstable missed endpoints should not become method targets until the endpoint universe or replay protocol is revised.",
        "- The next controlled method probe should add only mechanism-readable handles for stable missed endpoints, then compare again to the frozen random-restart curve.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def run_diagnostic(
    *,
    baseline_dir: Path,
    method_dir: Path,
    output_dir: Path,
    replay_seeds: int,
    n_iterations: int,
) -> dict[str, Any]:
    manifest = pd.read_csv(baseline_dir / FROZEN_ENDPOINT_MANIFEST_CSV)
    runs = _run_replay(
        manifest=manifest,
        replay_seeds=replay_seeds,
        n_iterations=n_iterations,
    )
    replay_summary = _summarize_replay(runs)
    missing_diagnosis = _missing_endpoint_diagnosis(
        manifest=manifest,
        replay_summary=replay_summary,
        method_dir=method_dir,
    )
    gate_matrix = _gate_matrix(
        manifest=manifest,
        runs=runs,
        replay_summary=replay_summary,
        missing_diagnosis=missing_diagnosis,
        replay_seeds=replay_seeds,
    )

    recurrent_summary = replay_summary[
        replay_summary["target_is_recurrent_endpoint"].astype(bool)
    ]
    stable_missing = missing_diagnosis[
        missing_diagnosis["diagnosis"].isin(
            [
                "method_handle_gap_stable_endpoint",
                "probable_handle_gap_mostly_stable_endpoint",
            ]
        )
    ]
    summary = {
        "frozen_endpoint_count": int(manifest["frozen_endpoint_id"].nunique()),
        "recurrent_endpoint_count": int(
            manifest[manifest["is_recurrent_endpoint"].astype(bool)][
                "frozen_endpoint_id"
            ].nunique()
        ),
        "replay_seeds": int(replay_seeds),
        "replay_run_count": int(len(runs)),
        "min_recurrent_same_endpoint_rate": float(
            recurrent_summary["same_endpoint_rate"].min()
        )
        if not recurrent_summary.empty
        else None,
        "method_v1_missing_recurrent_endpoint_count": int(len(missing_diagnosis)),
        "stable_or_mostly_stable_missing_count": int(len(stable_missing)),
        "gate_status_counts": {
            str(key): int(value)
            for key, value in gate_matrix["status"].value_counts().sort_index().to_dict().items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "inputs": {
            "baseline_dir": _rel(baseline_dir),
            "method_dir": _rel(method_dir),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(runs, output_dir / REPLAY_RUNS_CSV)
    _write_csv(replay_summary, output_dir / REPLAY_SUMMARY_CSV)
    _write_csv(missing_diagnosis, output_dir / MISSING_DIAGNOSIS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "baseline_dir": _rel(baseline_dir),
        "method_dir": _rel(method_dir),
        "output_dir": _rel(output_dir),
        "replay_seeds": int(replay_seeds),
        "n_iterations": int(n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        gate_matrix=gate_matrix,
        replay_summary=replay_summary,
        missing_diagnosis=missing_diagnosis,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--method-dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-seeds", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=-1)
    args = parser.parse_args()
    summary = run_diagnostic(
        baseline_dir=args.baseline_dir,
        method_dir=args.method_dir,
        output_dir=args.output_dir,
        replay_seeds=args.replay_seeds,
        n_iterations=args.n_iterations,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
