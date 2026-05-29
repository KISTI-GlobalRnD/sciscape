#!/usr/bin/env python3
"""Compare attachment-margin tunneling rows against vanilla Leiden controls."""

from __future__ import annotations

import argparse
import json
import math
import time
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
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _parse_n_iterations_values,
)
from evaluate_leiden_basin_branch_candidate_controls import (  # noqa: E402
    _reference_or_control_row,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    _rank_and_filter_prefix_rows,
)
from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    _load_case_context,
    _markdown_table,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_ATTACHMENT_DIR,
    PROBE_ROWS_FILENAME as ATTACHMENT_PROBE_ROWS_FILENAME,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
    _load_json,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    endpoint_distance,
    support_distance,
    v_only_support_nodes,
)
from sciscape.clustering.leiden_basin_search import state_distance  # noqa: E402

DEFAULT_OUTPUT_DIR = DEFAULT_ATTACHMENT_DIR.parent / (
    "basin_transition_attachment_margin_controls_field34_cc_c0_p6_p8_p10_v0"
)

CONTROL_ROWS_FILENAME = "attachment_margin_control_rows.csv"
SUMMARY_ROWS_FILENAME = "attachment_margin_control_summary_rows.csv"
SUMMARY_FILENAME = "attachment_margin_control_summary.json"
CONFIG_FILENAME = "attachment_margin_control_config.json"
REPORT_FILENAME = "attachment_margin_control_report.md"

def _parse_csv_tuple(value: str, *, cast: type = str) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())

def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

def _source_label_from_dir(source_move_dir: Path) -> str:
    config = _load_json(source_move_dir / "post_gate_recovery_move_config.json")
    if not config:
        return source_move_dir.name
    return f"p{int(config.get('prefix_rank', 0))}_{source_move_dir.name.split('_')[-2]}"

def _load_context_from_source_dir(source_move_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_json(source_move_dir / "post_gate_recovery_move_config.json")
    if not config:
        raise ValueError(f"Missing source config in {source_move_dir}")

    pair_id = str(config.get("pair_id", "c0-s11-r0.001"))
    prefix_rank = int(config["prefix_rank"])
    prefix_dir = Path(config["prefix_dir"])
    profile_batch_dir = Path(config["profile_batch_dir"])
    vanilla_dir = Path(config["vanilla_dir"])
    candidate_dirs = tuple(Path(path) for path in config["candidate_dirs"])

    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=(pair_id,),
        top_prefixes_per_case=max(prefix_rank, 10),
    )
    prefixes = _rank_and_filter_prefix_rows(
        prefixes,
        selected_prefix_ranks=(prefix_rank,),
    )
    if prefixes.empty:
        raise ValueError(f"No prefix row selected for {pair_id} rank {prefix_rank}")

    context = _load_case_context(
        prefix_row=prefixes.iloc[0],
        profile_batch_dir=profile_batch_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=vanilla_dir,
        baseline_iterations=int(config.get("baseline_iterations", 10)),
        candidate_polish_iterations=int(config.get("candidate_polish_iterations", 5)),
        resolution=float(config.get("resolution", 0.01)),
        randomness=float(config.get("randomness", 0.01)),
        perturb_seed_offset=1000,
    )
    meta = {
        "pair_id": pair_id,
        "prefix_rank": prefix_rank,
        "source_move_dir": str(source_move_dir),
        "resolution": float(config.get("resolution", 0.01)),
        "randomness": float(config.get("randomness", 0.01)),
        "baseline_iterations": int(config.get("baseline_iterations", 10)),
        "candidate_polish_iterations": int(
            config.get("candidate_polish_iterations", 5)
        ),
    }
    return context, meta

def _vanilla_distance_context(case_ctx: dict[str, Any]) -> dict[str, Any]:
    baseline = case_ctx["baseline"].membership
    candidate = case_ctx["candidate"].recreated.membership
    vanilla = case_ctx["vanilla"].membership
    candidate_support, vanilla_support, _target_nodes = v_only_support_nodes(
        baseline,
        candidate,
        vanilla,
    )
    vanilla_support_distance_to_candidate = support_distance(
        vanilla_support,
        candidate_support,
    )[0]
    vanilla_target_distance = state_distance(
        support_distance_value=float(vanilla_support_distance_to_candidate),
        endpoint_distance_value=endpoint_distance(
            vanilla,
            candidate,
            case_ctx["sketch_nodes"],
        ),
    )
    return {
        "vanilla_support_distance_to_candidate": float(
            vanilla_support_distance_to_candidate
        ),
        "vanilla_target_distance": float(vanilla_target_distance),
    }

def _operator_row(row: pd.Series) -> dict[str, Any]:
    return {
        "row_type": "operator",
        "variant": "attachment_margin_target_only",
        "source_case": str(row["source_case"]),
        "run_id": str(row["state_id"]),
        "seed": int(row["recovery_seed"]),
        "randomness": _safe_float(row.get("vanilla_randomness")),
        "requested_n_iterations": "",
        "iteration_mode": "bounded_local_polish",
        "elapsed_sec": _safe_float(row.get("path_elapsed_sec"), 0.0),
        "quality": _safe_float(row["state_quality"]),
        "delta_q_vs_vanilla": _safe_float(row["state_delta_q_vs_vanilla"]),
        "delta_q_vs_candidate": _safe_float(row["state_delta_q_vs_candidate"]),
        "support_size": int(_safe_float(row["state_support_size"], 0.0)),
        "support_distance_to_vanilla": _safe_float(
            row["state_support_distance_to_vanilla"]
        ),
        "support_distance_to_candidate": _safe_float(
            row["state_support_distance_to_candidate"]
        ),
        "endpoint_distance_to_vanilla": _safe_float(
            row["state_endpoint_distance_to_vanilla"]
        ),
        "endpoint_distance_to_candidate": _safe_float(
            row["state_endpoint_distance_to_candidate"]
        ),
        "candidate_progress_from_vanilla": _safe_float(
            row["state_candidate_progress_from_vanilla"]
        ),
        "target_distance": _safe_float(row["state_target_distance"]),
        "source_distance": _safe_float(row["state_source_distance"]),
        "target_progress_from_vanilla": _safe_float(
            row["state_target_progress_from_vanilla"]
        ),
        "q_wall": 0.0,
        "mutable_node_count": int(_safe_float(row["mutable_node_count"], 0.0)),
        "context_node_count": int(_safe_float(row["context_node_count"], 0.0)),
        "selected_k": int(_safe_float(row["selected_k"], 0.0)),
        "selected_node_ids": "" if pd.isna(row.get("selected_node_ids")) else str(row.get("selected_node_ids")),
        "gate_release_delta_q_gain": _safe_float(row["gate_release_delta_q_gain"]),
        "source_delta_q_vs_start": _safe_float(row["source_delta_q_vs_start"]),
    }

def _best_operator_rows(probe_rows: pd.DataFrame) -> pd.DataFrame:
    target_rows = probe_rows[
        probe_rows["action_mode"].astype(str).eq("target_only_margin")
    ].copy()
    if target_rows.empty:
        return target_rows
    ordered = target_rows.sort_values(
        [
            "source_case",
            "gate_release_delta_q_gain",
            "state_delta_q_vs_start",
            "state_support_distance_to_vanilla",
            "mutable_node_count",
        ],
        ascending=[True, False, False, False, True],
    )
    return ordered.groupby("source_case", sort=False).head(1).reset_index(drop=True)

def _control_rows_for_source(
    *,
    source_case: str,
    source_move_dir: Path,
    seeds: tuple[int, ...],
    randomness_values: tuple[float, ...],
    n_iterations_values: tuple[Any, ...],
) -> pd.DataFrame:
    case_ctx, meta = _load_context_from_source_dir(source_move_dir)
    dist_ctx = _vanilla_distance_context(case_ctx)
    baseline_membership = case_ctx["baseline"].membership
    candidate_membership = case_ctx["candidate"].recreated.membership
    vanilla_membership = case_ctx["vanilla"].membership
    rows: list[dict[str, Any]] = []

    common = {
        "baseline_membership": baseline_membership,
        "candidate_membership": candidate_membership,
        "vanilla_membership": vanilla_membership,
        "sketch_nodes": case_ctx["sketch_nodes"],
        "vanilla_support_distance_to_candidate": dist_ctx[
            "vanilla_support_distance_to_candidate"
        ],
        "vanilla_target_distance": dist_ctx["vanilla_target_distance"],
        "candidate_quality": float(case_ctx["candidate"].recreated.quality),
        "vanilla_quality": float(case_ctx["vanilla"].quality),
    }
    references = [
        (
            "reference_candidate",
            case_ctx["candidate"].recreated.membership,
            float(case_ctx["candidate"].recreated.quality),
        ),
        (
            "reference_vanilla",
            case_ctx["vanilla"].membership,
            float(case_ctx["vanilla"].quality),
        ),
    ]
    for variant, membership, quality in references:
        row = _reference_or_control_row(
            row_type="reference",
            variant=variant,
            run_id=f"{case_ctx['public_context']['case']}|{variant}",
            membership=membership,
            quality=quality,
            **common,
        )
        row.update(
            {
                "source_case": source_case,
                "source_move_dir": str(source_move_dir),
                "context_node_count": int(np.asarray(membership).size),
                "selected_k": "",
                "selected_node_ids": "",
                "gate_release_delta_q_gain": math.nan,
                "source_delta_q_vs_start": math.nan,
            }
        )
        rows.append(row)

    for seed in seeds:
        for control_randomness in randomness_values:
            for budget in n_iterations_values:
                start = time.perf_counter()
                result = case_ctx["graph"].run_leiden(
                    resolution=float(meta["resolution"]),
                    seed=int(seed),
                    n_iterations=int(budget.n_iterations),
                    randomness=float(control_randomness),
                )
                elapsed = time.perf_counter() - start
                row = _reference_or_control_row(
                    row_type="control",
                    variant="standard_leiden",
                    run_id=(
                        f"{case_ctx['public_context']['case']}|"
                        f"{source_case}|control|seed={int(seed)}|"
                        f"randomness={float(control_randomness):g}|n={budget.requested}"
                    ),
                    membership=np.asarray(result.membership, dtype=np.uint64),
                    quality=float(result.quality),
                    elapsed_sec=float(elapsed),
                    seed=int(seed),
                    randomness=float(control_randomness),
                    requested_n_iterations=str(budget.requested),
                    iteration_mode=str(budget.mode),
                    **common,
                )
                row.update(
                    {
                        "source_case": source_case,
                        "source_move_dir": str(source_move_dir),
                        "context_node_count": int(np.asarray(result.membership).size),
                        "selected_k": "",
                        "selected_node_ids": "",
                        "gate_release_delta_q_gain": math.nan,
                        "source_delta_q_vs_start": math.nan,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)

def _summarize_source(
    rows: pd.DataFrame,
    *,
    material_delta_q: float,
    support_margin: float,
    progress_margin: float,
) -> dict[str, Any]:
    source_case = str(rows["source_case"].iloc[0])
    operator = rows[rows["row_type"].eq("operator")].iloc[0]
    controls = rows[rows["row_type"].eq("control")].copy()
    best_quality = controls.sort_values(
        ["quality", "support_distance_to_vanilla", "target_progress_from_vanilla"],
        ascending=[False, False, False],
    ).iloc[0]
    best_support = controls.sort_values(
        ["support_distance_to_vanilla", "target_progress_from_vanilla", "quality"],
        ascending=[False, False, False],
    ).iloc[0]
    best_progress = controls.sort_values(
        ["target_progress_from_vanilla", "support_distance_to_vanilla", "quality"],
        ascending=[False, False, False],
    ).iloc[0]
    operator_randomness = float(operator["randomness"])
    same_randomness_controls = controls[
        np.isclose(controls["randomness"].astype(float), operator_randomness)
    ].copy()
    best_same_randomness = (
        same_randomness_controls.sort_values(
            ["quality", "support_distance_to_vanilla", "target_progress_from_vanilla"],
            ascending=[False, False, False],
        ).iloc[0]
        if not same_randomness_controls.empty
        else best_quality
    )
    directed_controls = controls[
        controls["target_progress_from_vanilla"].astype(float) > 0.0
    ].copy()
    directed_same_randomness_controls = same_randomness_controls[
        same_randomness_controls["target_progress_from_vanilla"].astype(float) > 0.0
    ].copy()
    dominance_mask = (
        (controls["quality"].astype(float) >= float(operator["quality"]))
        & (
            controls["support_distance_to_vanilla"].astype(float)
            >= float(operator["support_distance_to_vanilla"])
        )
        & (
            controls["target_progress_from_vanilla"].astype(float)
            >= float(operator["target_progress_from_vanilla"])
        )
    )
    quality_margin = float(operator["quality"]) - float(best_quality["quality"])
    support_margin_value = float(operator["support_distance_to_vanilla"]) - float(
        best_support["support_distance_to_vanilla"]
    )
    progress_margin_value = float(operator["target_progress_from_vanilla"]) - float(
        best_progress["target_progress_from_vanilla"]
    )
    if dominance_mask.any():
        verdict = "seed_control_dominates_operator"
    elif quality_margin >= float(material_delta_q) and progress_margin_value >= 0.0:
        verdict = "operator_material_quality_win"
    elif (
        float(operator["target_progress_from_vanilla"]) > 0.0
        and directed_controls.empty
        and quality_margin >= -float(material_delta_q)
    ):
        verdict = "operator_unique_directed_near_quality"
    elif (
        float(operator["target_progress_from_vanilla"]) > 0.0
        and directed_controls.empty
    ):
        verdict = "operator_unique_directed_quality_lag"
    elif (
        progress_margin_value >= float(progress_margin)
        and quality_margin >= -float(material_delta_q)
    ):
        verdict = "operator_progress_tradeoff"
    else:
        verdict = "operator_not_control_clear"

    return {
        "source_case": source_case,
        "operator_run_id": operator["run_id"],
        "operator_selected_k": int(operator["selected_k"]),
        "operator_selected_node_ids": operator["selected_node_ids"],
        "operator_quality": float(operator["quality"]),
        "operator_delta_q_vs_vanilla": float(operator["delta_q_vs_vanilla"]),
        "operator_support": float(operator["support_distance_to_vanilla"]),
        "operator_target_progress": float(operator["target_progress_from_vanilla"]),
        "operator_mutable_node_count": int(operator["mutable_node_count"]),
        "operator_elapsed_sec": float(operator["elapsed_sec"]),
        "control_rows": int(len(controls)),
        "best_quality_control_run_id": best_quality["run_id"],
        "best_quality_control_quality": float(best_quality["quality"]),
        "best_quality_control_support": float(
            best_quality["support_distance_to_vanilla"]
        ),
        "best_quality_control_target_progress": float(
            best_quality["target_progress_from_vanilla"]
        ),
        "best_quality_control_mutable_node_count": int(
            best_quality["mutable_node_count"]
        ),
        "same_randomness_control_rows": int(len(same_randomness_controls)),
        "best_same_randomness_control_run_id": best_same_randomness["run_id"],
        "best_same_randomness_control_quality": float(best_same_randomness["quality"]),
        "best_same_randomness_control_support": float(
            best_same_randomness["support_distance_to_vanilla"]
        ),
        "best_same_randomness_control_target_progress": float(
            best_same_randomness["target_progress_from_vanilla"]
        ),
        "operator_quality_minus_best_same_randomness_control": float(
            float(operator["quality"]) - float(best_same_randomness["quality"])
        ),
        "same_randomness_candidate_directed_control_rows": int(
            len(directed_same_randomness_controls)
        ),
        "best_support_control_run_id": best_support["run_id"],
        "best_support_control_quality": float(best_support["quality"]),
        "best_support_control_support": float(best_support["support_distance_to_vanilla"]),
        "best_support_control_target_progress": float(
            best_support["target_progress_from_vanilla"]
        ),
        "best_progress_control_run_id": best_progress["run_id"],
        "best_progress_control_quality": float(best_progress["quality"]),
        "best_progress_control_support": float(best_progress["support_distance_to_vanilla"]),
        "best_progress_control_target_progress": float(
            best_progress["target_progress_from_vanilla"]
        ),
        "operator_quality_minus_best_control": float(quality_margin),
        "operator_support_minus_best_control": float(support_margin_value),
        "operator_target_progress_minus_best_control": float(progress_margin_value),
        "candidate_directed_control_rows": int(len(directed_controls)),
        "control_dominates_operator_rows": int(dominance_mask.sum()),
        "material_delta_q": float(material_delta_q),
        "support_margin": float(support_margin),
        "progress_margin": float(progress_margin),
        "verdict": verdict,
    }

def _write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    lines = [
        "# Attachment-Margin Seed-Control Comparison",
        "",
        "This artifact compares compact attachment-margin target-only tunneling rows against same-case standard Leiden seed/iteration controls.",
        "",
        "## Config",
        "",
        "| key | value |",
        "| --- | --- |",
    ]
    for key in [
        "attachment_dir",
        "seeds",
        "randomness_values",
        "n_iterations_values",
        "material_delta_q",
        "support_margin",
        "progress_margin",
    ]:
        lines.append(f"| {key} | {config.get(key, '')} |")
    lines.extend(["", "## Summary", ""])
    summary_cols = [
        "source_case",
        "operator_selected_node_ids",
        "operator_delta_q_vs_vanilla",
        "operator_support",
        "operator_target_progress",
        "operator_mutable_node_count",
        "best_same_randomness_control_quality",
        "best_same_randomness_control_target_progress",
        "operator_quality_minus_best_same_randomness_control",
        "best_quality_control_quality",
        "best_quality_control_target_progress",
        "operator_quality_minus_best_control",
        "candidate_directed_control_rows",
        "verdict",
    ]
    lines.extend(_markdown_table(summary_rows[[c for c in summary_cols if c in summary_rows]], max_rows=50))
    lines.extend(["", "## Top Rows", ""])
    row_cols = [
        "source_case",
        "row_type",
        "variant",
        "seed",
        "randomness",
        "requested_n_iterations",
        "quality",
        "delta_q_vs_vanilla",
        "support_distance_to_vanilla",
        "target_progress_from_vanilla",
        "support_distance_to_candidate",
        "mutable_node_count",
        "elapsed_sec",
        "selected_node_ids",
    ]
    display = rows.sort_values(
        ["source_case", "row_type", "quality", "target_progress_from_vanilla"],
        ascending=[True, True, False, False],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=120))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- A compact tunneling row is not a Dongdaemun-refinement claim unless it beats seed/iteration controls on material and cost-adjusted value.",
            "- Positive target progress with lower QF remains a mechanism signal, not a quality win.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_evaluation(
    *,
    attachment_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    randomness_values: tuple[float, ...],
    n_iterations_values: tuple[Any, ...],
    material_delta_q: float,
    support_margin: float,
    progress_margin: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_rows = pd.read_csv(attachment_dir / ATTACHMENT_PROBE_ROWS_FILENAME)
    summary_rows = pd.read_csv(attachment_dir / ATTACHMENT_SUMMARY_ROWS_FILENAME)
    best_ops = _best_operator_rows(probe_rows)
    all_rows: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    for _, op_row in best_ops.iterrows():
        source_case = str(op_row["source_case"])
        source_match = summary_rows[summary_rows["source_case"].astype(str).eq(source_case)]
        if source_match.empty:
            raise ValueError(f"Missing source summary row for {source_case}")
        source_move_dir = Path(str(source_match.iloc[0]["source_move_dir"]))
        control_rows = _control_rows_for_source(
            source_case=source_case,
            source_move_dir=source_move_dir,
            seeds=seeds,
            randomness_values=randomness_values,
            n_iterations_values=n_iterations_values,
        )
        operator = pd.DataFrame([_operator_row(op_row)])
        combined = pd.concat([operator, control_rows], ignore_index=True, sort=False)
        all_rows.append(combined)
        summary.append(
            _summarize_source(
                combined,
                material_delta_q=material_delta_q,
                support_margin=support_margin,
                progress_margin=progress_margin,
            )
        )

    rows = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    control_summary = pd.DataFrame(summary)
    rows.to_csv(output_dir / CONTROL_ROWS_FILENAME, index=False)
    control_summary.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    config = {
        "attachment_dir": str(attachment_dir),
        "output_dir": str(output_dir),
        "seeds": [int(seed) for seed in seeds],
        "randomness_values": [float(value) for value in randomness_values],
        "n_iterations_values": [str(value.requested) for value in n_iterations_values],
        "material_delta_q": float(material_delta_q),
        "support_margin": float(support_margin),
        "progress_margin": float(progress_margin),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "leiden_basin_attachment_margin_controls.v0",
        "output_dir": str(output_dir),
        "source_count": int(len(control_summary)),
        "row_count": int(len(rows)),
        "verdict_counts": control_summary["verdict"].value_counts().to_dict()
        if not control_summary.empty
        else {},
        "paths": {
            "rows": str(output_dir / CONTROL_ROWS_FILENAME),
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        summary_rows=control_summary,
        config=config,
    )
    return result

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment-dir", type=Path, default=DEFAULT_ATTACHMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="11,42,73,101,137")
    parser.add_argument("--randomness-values", default="0.001,0.01")
    parser.add_argument("--n-iterations-values", default="1,10,convergence")
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--support-margin", type=float, default=0.01)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    summary = run_evaluation(
        attachment_dir=args.attachment_dir,
        output_dir=args.output_dir,
        seeds=_parse_csv_tuple(args.seeds, cast=int),
        randomness_values=_parse_csv_tuple(args.randomness_values, cast=float),
        n_iterations_values=_parse_n_iterations_values(args.n_iterations_values),
        material_delta_q=args.material_delta_q,
        support_margin=args.support_margin,
        progress_margin=args.progress_margin,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
