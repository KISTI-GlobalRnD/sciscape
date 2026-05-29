#!/usr/bin/env python3
"""Fill vanilla graph context for clean non-field34 distinct basin pairs.

This is a narrow Track C preparation step. It creates the minimal manifest for
two missing vanilla contexts and optionally runs the standard Leiden vanilla
sweep needed by the uniform wall-probe runner. It does not run pair routes,
promote wall claims, or evaluate basin quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _parse_n_iterations_value,
    collect_sweep,
)


BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_clean_distinct_vanilla_context_gap_fill_20260528"
)
CROSSFIELD_MANIFEST = (
    BASE_RESULT_DIR
    / "leiden_multibasin_crossfield_budget12_support_20260519/portfolio_batch_cases.csv"
)
FIELD30_MANIFEST = (
    BASE_RESULT_DIR
    / "leiden_multibasin_signature_field30_budget12_support_20260519/portfolio_batch_cases.csv"
)

GAP_MANIFEST_CSV = "clean_distinct_vanilla_context_gap_manifest.csv"
GAP_CONFIG_JSON = "clean_distinct_vanilla_context_gap_config.json"
GAP_SUMMARY_JSON = "clean_distinct_vanilla_context_gap_summary.json"
GAP_REPORT_MD = "clean_distinct_vanilla_context_gap_report.md"

TARGET_SPECS = (
    {
        "case_slug": (
            "field26_gcc_emb_full_knn30_bc_cosine_seed11_budget12_"
            "multifidelity_label_probe_only"
        ),
        "source_manifest": CROSSFIELD_MANIFEST,
        "panel_pair_ids": (
            "field26_gcc_emb_full_knn30_bc_cosine_budget12:c1-c2",
            "field26_gcc_emb_full_knn30_bc_cosine_budget12:c5-c10",
        ),
    },
    {
        "case_slug": (
            "field30_gcc_emb_full_knn30_emb_knn_seed11_budget12_"
            "multifidelity_label_probe_only"
        ),
        "source_manifest": FIELD30_MANIFEST,
        "panel_pair_ids": (
            "field30_gcc_emb_full_knn30_emb_knn_budget12:c6-c10",
            "field30_gcc_emb_full_knn30_emb_knn_budget12:c6-c11",
        ),
    },
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _target_manifest() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.Series] = []
    specs: list[dict[str, Any]] = []
    for spec in TARGET_SPECS:
        source_manifest = Path(spec["source_manifest"])
        manifest = _read_csv(source_manifest)
        matches = manifest[manifest["case_slug"].astype(str).eq(str(spec["case_slug"]))].copy()
        if matches.empty:
            raise ValueError(f"case_slug not found in {source_manifest}: {spec['case_slug']}")
        if len(matches) > 1:
            raise ValueError(f"case_slug is not unique in {source_manifest}: {spec['case_slug']}")
        row = matches.iloc[0].copy()
        row["gap_fill_source_manifest"] = _rel(source_manifest)
        row["gap_fill_panel_pair_ids"] = ";".join(spec["panel_pair_ids"])
        rows.append(row)
        specs.append(
            {
                "case_slug": str(spec["case_slug"]),
                "source_manifest": _rel(source_manifest),
                "panel_pair_ids": list(spec["panel_pair_ids"]),
                "graph_dir": str(row.get("graph_dir", "")),
                "field": int(row.get("field")),
                "method": str(row.get("method", "")),
            }
        )
    return pd.DataFrame([row.to_dict() for row in rows]), specs


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    sweep_summary: dict[str, Any] | None,
    manifest: pd.DataFrame,
) -> None:
    lines = [
        "# Leiden Basin Clean Distinct Vanilla Context Gap Fill",
        "",
        "Status: missing vanilla context gap fill prepared",
        "Date: 2026-05-28",
        "",
        "This artifact only fills standard Leiden graph context for clean non-field34 distinct basin pairs. It does not run routes, rank basins, or promote wall claims.",
        "",
        "## Target Cases",
        "",
        "| case_slug | field | method | panel_pair_ids |",
        "| --- | ---: | --- | --- |",
    ]
    for _, row in manifest.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_slug"]),
                    str(row["field"]),
                    str(row["method"]),
                    str(row["gap_fill_panel_pair_ids"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Vanilla Sweep",
            "",
            f"- run vanilla sweep: {summary['run_vanilla']}",
            f"- manifest rows: {summary['manifest_row_count']}",
            f"- expected unlocked panel pairs: {summary['expected_unlocked_panel_pair_count']}",
        ]
    )
    if sweep_summary:
        lines.extend(
            [
                f"- candidate cases: {sweep_summary['candidate_case_count']}",
                f"- rows: {sweep_summary['row_count']}",
                f"- new runs: {sweep_summary['run_count']}",
                f"- skipped existing runs: {sweep_summary['skipped_existing_count']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- These rows are runner context only.",
            "- Basin relation, wall evidence, and route-order gates must be checked in later artifacts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path, *, run_vanilla: bool, resume: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, specs = _target_manifest()
    manifest_path = output_dir / GAP_MANIFEST_CSV
    _write_csv(manifest, manifest_path)

    sweep_summary: dict[str, Any] | None = None
    if run_vanilla:
        sweep_summary = collect_sweep(
            case_manifest=manifest_path,
            target_rows_path=None,
            output_dir=output_dir,
            seeds=(11,),
            randomness_values=(0.0,),
            n_iterations_values=(_parse_n_iterations_value("10"),),
            target_classes=set(),
            fields=set(),
            methods=set(),
            max_cases=None,
            run_limit=None,
            resolution=0.01,
            compatible_sketches=False,
            resume=resume,
        )

    summary = {
        "status": "clean_distinct_vanilla_context_gap_fill_prepared",
        "date": "2026-05-28",
        "script": _rel(Path(__file__)),
        "output_dir": _rel(output_dir),
        "manifest": _rel(manifest_path),
        "manifest_row_count": int(len(manifest)),
        "target_cases": specs,
        "expected_unlocked_panel_pair_count": int(
            sum(len(spec["panel_pair_ids"]) for spec in specs)
        ),
        "run_vanilla": bool(run_vanilla),
        "resume": bool(resume),
        "sweep_summary": sweep_summary or {},
        "claim_boundary": (
            "Runner context only; no route execution, wall promotion, or basin-quality "
            "claim is made."
        ),
    }
    (output_dir / GAP_CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / GAP_SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(
        output_dir / GAP_REPORT_MD,
        summary=summary,
        sweep_summary=sweep_summary,
        manifest=manifest,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_dir,
                run_vanilla=not bool(args.skip_vanilla),
                resume=not bool(args.no_resume),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
