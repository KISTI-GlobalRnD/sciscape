#!/usr/bin/env python3
"""Compare NanoClustering anchor-release pilots across selection policies.

This analysis reads already executed anchor-release pilot outputs and compares
whether source-anchor and target-anchor local terminals remain distinct after
both are released into the same fixed-outside pair mask. It does not run
clustering, execute routes, promote wall/pathway claims, inspect basin quality,
or claim method success.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_RUN_DIRS = [
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_anchor_release_pilot_anchor_expand_seed0_20260601",
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_anchor_release_pilot_low_overlap_top12_seed0_20260601",
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_anchor_release_pilot_largest_pair_free_top12_seed0_20260601",
]
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_anchor_release_policy_comparison_20260601"
)

PAIR_ROWS_CSV = "nanoclustering_anchor_release_pilot_pair_rows.csv"
CONFIG_JSON = "nanoclustering_anchor_release_pilot_config.json"
SUMMARY_JSON = "nanoclustering_anchor_release_pilot_summary.json"

COMPARISON_PAIR_ROWS_CSV = "nanoclustering_anchor_release_policy_pair_rows.csv"
COMPARISON_RUN_ROWS_CSV = "nanoclustering_anchor_release_policy_run_rows.csv"
COMPARISON_SUMMARY_JSON = "nanoclustering_anchor_release_policy_summary.json"
COMPARISON_CONFIG_JSON = "nanoclustering_anchor_release_policy_config.json"
COMPARISON_REPORT_MD = "nanoclustering_anchor_release_policy_report.md"

PAIR_KEY = ["panel_case_id", "role_id", "target_handle_id", "method_seed"]
CLAIM_BOUNDARY = (
    "Anchor-release policy comparison only; reads completed pilot outputs and "
    "checks common-release collapse/non-collapse. It does not run clustering, "
    "execute new routes, promote wall/pathway claims, inspect basin quality, "
    "or claim method success."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _run_label(run_dir: Path, config: dict[str, Any]) -> str:
    policy = str(config.get("target_selection_policy") or "")
    if policy:
        return policy
    name = run_dir.name
    if "anchor_expand" in name:
        return "default_anchor_expand"
    return name.replace("leiden_basin_nanoclustering_anchor_release_pilot_", "")


def _domain_from_panel(panel_case_id: str) -> str:
    parts = str(panel_case_id).split("_")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    config = _read_json(run_dir / CONFIG_JSON)
    summary = _read_json(run_dir / SUMMARY_JSON)
    rows = _read_csv(run_dir / PAIR_ROWS_CSV)
    label = _run_label(run_dir, config)
    rows = rows.copy()
    rows["run_label"] = label
    rows["run_dir"] = _rel(run_dir)
    rows["selection_policy"] = str(config.get("target_selection_policy") or label)
    rows["domain"] = rows["panel_case_id"].astype(str).map(_domain_from_panel)
    return rows, config, summary


def _run_rows(
    *,
    loaded: list[tuple[Path, pd.DataFrame, dict[str, Any], dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir, pairs, config, summary in loaded:
        label = _run_label(run_dir, config)
        rows.append(
            {
                "run_label": label,
                "selection_policy": str(config.get("target_selection_policy") or label),
                "run_dir": _rel(run_dir),
                "release_pair_count": int(summary["release_pair_count"]),
                "unique_pair_count": int(pairs.drop_duplicates(PAIR_KEY).shape[0]),
                "anchor_pair_distinct_count": int(summary["anchor_pair_distinct_count"]),
                "release_pair_distinct_count": int(summary["release_pair_distinct_count"]),
                "release_pair_collapsed_count": int(summary["release_pair_collapsed_count"]),
                "release_pair_distinct_share": float(summary["release_pair_distinct_share"]),
                "unique_local_source_target_pair_mask_object_count": int(
                    summary["unique_local_source_target_pair_mask_object_count"]
                ),
                "unique_panel_case_count": int(summary["unique_panel_case_count"]),
                "unique_target_handle_count": int(summary["unique_target_handle_count"]),
                "branch_count": int(summary["branch_count"]),
                "role_count": int(summary["role_count"]),
                "common_release_free_node_count_median": float(
                    pairs["common_release_free_node_count"].median()
                ),
                "common_release_free_node_count_max": int(
                    pairs["common_release_free_node_count"].max()
                ),
                "release_target_doc_share_median": float(
                    pairs["target_release_target_best_cluster_doc_share"].median()
                ),
                "release_target_doc_share_max": float(
                    pairs["target_release_target_best_cluster_doc_share"].max()
                ),
                "total_route_seconds": float(summary["total_route_seconds"]),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _comparison_pair_rows(all_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = all_pairs.drop_duplicates(PAIR_KEY).copy()
    run_names = (
        all_pairs.groupby(PAIR_KEY, sort=False)["run_label"]
        .agg(lambda values: ";".join(sorted(set(values.astype(str)))))
        .reset_index(name="observed_in_run_labels")
    )
    rows = rows.merge(run_names, on=PAIR_KEY, how="left", validate="one_to_one")
    rows["target_release_target_doc_share_drop_vs_target_anchor"] = (
        rows["target_anchor_target_best_cluster_doc_share"]
        - rows["target_release_target_best_cluster_doc_share"]
    )
    rows["source_release_target_doc_share_drop_vs_source_anchor"] = (
        rows["source_anchor_target_best_cluster_doc_share"]
        - rows["source_release_target_best_cluster_doc_share"]
    )
    rows["release_equals_anchor_status"] = "not_compared_to_full_anchor_hash"
    rows["claim_boundary"] = CLAIM_BOUNDARY
    keep = [
        "panel_case_id",
        "panel_case_rank",
        "domain",
        "role_id",
        "target_handle_id",
        "method_seed",
        "observed_in_run_labels",
        "common_release_free_node_count",
        "source_node_count",
        "target_node_count",
        "pair_node_count",
        "source_anchor_target_best_cluster_doc_share",
        "target_anchor_target_best_cluster_doc_share",
        "source_release_target_best_cluster_doc_share",
        "target_release_target_best_cluster_doc_share",
        "target_release_target_doc_share_drop_vs_target_anchor",
        "source_release_target_doc_share_drop_vs_source_anchor",
        "source_release_source_best_cluster_doc_share",
        "target_release_source_best_cluster_doc_share",
        "anchor_pair_distinct",
        "release_pair_distinct",
        "release_collapse_status",
        "source_anchor_pair_changed_vs_initial",
        "target_anchor_pair_changed_vs_initial",
        "source_release_pair_changed_vs_initial",
        "target_release_pair_changed_vs_initial",
        "source_release_quality",
        "target_release_quality",
        "release_equals_anchor_status",
        "claim_boundary",
    ]
    return rows[keep].sort_values(
        ["domain", "panel_case_rank", "role_id", "common_release_free_node_count"],
        ascending=[True, True, True, False],
    )


def _summary(run_rows: pd.DataFrame, pair_rows: pd.DataFrame) -> dict[str, Any]:
    release_distinct = int(pair_rows["release_pair_distinct"].sum())
    collapsed = int(pair_rows["release_collapse_status"].eq("released_terminals_collapsed").sum())
    domain_rows = (
        pair_rows.groupby("domain", dropna=False)
        .agg(
            unique_pair_count=("target_handle_id", "size"),
            panel_case_count=("panel_case_rank", "nunique"),
            role_count=("role_id", "nunique"),
            target_handle_count=("target_handle_id", "nunique"),
            release_pair_distinct_count=("release_pair_distinct", "sum"),
            common_release_free_node_count_max=("common_release_free_node_count", "max"),
            release_target_doc_share_median=(
                "target_release_target_best_cluster_doc_share",
                "median",
            ),
        )
        .reset_index()
    )
    top_free = pair_rows.sort_values(
        "common_release_free_node_count", ascending=False
    ).head(10)
    return {
        "schema": "nanoclustering_anchor_release_policy_summary.v1",
        "status": "executed_anchor_release_policy_comparison",
        "run_count": int(len(run_rows)),
        "total_run_pair_rows": int(run_rows["release_pair_count"].sum()),
        "unique_pair_count": int(len(pair_rows)),
        "unique_panel_case_count": int(pair_rows["panel_case_id"].nunique()),
        "unique_role_count": int(pair_rows["role_id"].nunique()),
        "unique_target_handle_count": int(pair_rows["target_handle_id"].nunique()),
        "unique_domain_count": int(pair_rows["domain"].nunique()),
        "anchor_pair_distinct_count": int(pair_rows["anchor_pair_distinct"].sum()),
        "release_pair_distinct_count": release_distinct,
        "release_pair_collapsed_count": collapsed,
        "release_pair_distinct_share": (
            release_distinct / len(pair_rows) if len(pair_rows) else None
        ),
        "common_release_free_node_count_min": int(
            pair_rows["common_release_free_node_count"].min()
        ),
        "common_release_free_node_count_median": float(
            pair_rows["common_release_free_node_count"].median()
        ),
        "common_release_free_node_count_max": int(
            pair_rows["common_release_free_node_count"].max()
        ),
        "target_release_target_doc_share_median": float(
            pair_rows["target_release_target_best_cluster_doc_share"].median()
        ),
        "target_release_target_doc_share_max": float(
            pair_rows["target_release_target_best_cluster_doc_share"].max()
        ),
        "target_release_target_doc_share_drop_vs_target_anchor_median": float(
            pair_rows["target_release_target_doc_share_drop_vs_target_anchor"].median()
        ),
        "domain_rows": _json_safe(domain_rows.to_dict(orient="records")),
        "top_free_pair_rows": _json_safe(
            top_free[
                [
                    "panel_case_rank",
                    "domain",
                    "role_id",
                    "target_handle_id",
                    "common_release_free_node_count",
                    "source_node_count",
                    "target_node_count",
                    "target_release_target_best_cluster_doc_share",
                    "release_pair_distinct",
                ]
            ].to_dict(orient="records")
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _report(
    *,
    output_dir: Path,
    run_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# NanoClustering Anchor-Release Policy Comparison",
        "",
        f"- status: `{summary['status']}`",
        f"- run_count: {summary['run_count']}",
        f"- total_run_pair_rows: {summary['total_run_pair_rows']}",
        f"- unique_pair_count: {summary['unique_pair_count']}",
        f"- unique_panel_case_count: {summary['unique_panel_case_count']}",
        f"- anchor_pair_distinct_count: {summary['anchor_pair_distinct_count']}",
        f"- release_pair_distinct_count: {summary['release_pair_distinct_count']}",
        f"- release_pair_collapsed_count: {summary['release_pair_collapsed_count']}",
        f"- release_pair_distinct_share: {summary['release_pair_distinct_share']}",
        (
            "- common_release_free_node_count: "
            f"min={summary['common_release_free_node_count_min']}, "
            f"median={summary['common_release_free_node_count_median']}, "
            f"max={summary['common_release_free_node_count_max']}"
        ),
        (
            "- target_release_target_doc_share: "
            f"median={summary['target_release_target_doc_share_median']}, "
            f"max={summary['target_release_target_doc_share_max']}"
        ),
        (
            "- target_release_target_doc_share_drop_vs_target_anchor_median: "
            f"{summary['target_release_target_doc_share_drop_vs_target_anchor_median']}"
        ),
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Runs",
    ]
    for row in run_rows.sort_values("run_label").itertuples(index=False):
        lines.append(
            "- "
            f"{row.run_label}: pairs={row.release_pair_count}, "
            f"release_distinct={row.release_pair_distinct_count}, "
            f"collapsed={row.release_pair_collapsed_count}, "
            f"median_free={row.common_release_free_node_count_median}, "
            f"max_free={row.common_release_free_node_count_max}, "
            f"route_seconds={row.total_route_seconds}"
        )

    lines.extend(["", "## Domain Rows"])
    for row in summary["domain_rows"]:
        lines.append(
            "- "
            f"{row['domain']}: unique_pairs={row['unique_pair_count']}, "
            f"panels={row['panel_case_count']}, roles={row['role_count']}, "
            f"release_distinct={row['release_pair_distinct_count']}, "
            f"max_free={row['common_release_free_node_count_max']}, "
            f"median_release_target_share={row['release_target_doc_share_median']}"
        )

    lines.extend(["", "## Largest Free-Set Pairs"])
    for row in summary["top_free_pair_rows"]:
        lines.append(
            "- "
            f"panel={row['panel_case_rank']} {row['domain']} "
            f"{row['role_id']} -> {row['target_handle_id']}: "
            f"free={row['common_release_free_node_count']}, "
            f"source_nodes={row['source_node_count']}, "
            f"target_nodes={row['target_node_count']}, "
            f"release_target_share={row['target_release_target_best_cluster_doc_share']}, "
            f"release_distinct={row['release_pair_distinct']}"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "The repeated result is negative evidence for treating the current "
            "source-anchor/target-anchor local terminals as separate basin-wall "
            "objects. Anchor distinctness is easy to force, but it has not "
            "survived the common feasible-set release gate in these policies.",
            "",
            "This still does not disprove basin multiplicity. It narrows the "
            "failure to the present anchor-arm construction and says the next "
            "gate should search for release-stable terminal multiplicity before "
            "using wall/pathway language.",
            "",
            "## Artifacts",
            "",
            f"- `{_rel(output_dir / COMPARISON_PAIR_ROWS_CSV)}`",
            f"- `{_rel(output_dir / COMPARISON_RUN_ROWS_CSV)}`",
            f"- `{_rel(output_dir / COMPARISON_SUMMARY_JSON)}`",
            f"- `{_rel(output_dir / COMPARISON_CONFIG_JSON)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        dest="run_dirs",
        help=(
            "Anchor-release pilot result directory. May be provided multiple "
            "times. Defaults to the three 20260601 policy runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {_rel(DEFAULT_OUTPUT_DIR)}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [path.resolve() for path in (args.run_dirs or DEFAULT_RUN_DIRS)]
    output_dir = args.output_dir.resolve()
    loaded = []
    pair_frames = []
    for run_dir in run_dirs:
        pairs, config, summary = _load_run(run_dir)
        loaded.append((run_dir, pairs, config, summary))
        pair_frames.append(pairs)

    all_pairs = pd.concat(pair_frames, ignore_index=True)
    run_rows = _run_rows(loaded=loaded)
    pair_rows = _comparison_pair_rows(all_pairs)
    summary = _summary(run_rows, pair_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, output_dir / COMPARISON_PAIR_ROWS_CSV)
    _write_csv(run_rows, output_dir / COMPARISON_RUN_ROWS_CSV)
    (output_dir / COMPARISON_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_anchor_release_policy_comparison.v1",
        "run_dirs": [_rel(path) for path in run_dirs],
        "output_dir": _rel(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / COMPARISON_CONFIG_JSON).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / COMPARISON_REPORT_MD).write_text(
        _report(
            output_dir=output_dir,
            run_rows=run_rows,
            pair_rows=pair_rows,
            summary=summary,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
