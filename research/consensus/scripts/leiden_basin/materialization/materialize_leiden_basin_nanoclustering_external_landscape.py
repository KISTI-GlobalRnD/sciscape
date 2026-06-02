#!/usr/bin/env python3
"""Materialize NanoClustering external endpoint landscape diagnostics.

This is an external-data preparation step for Track C. It reads existing
NanoClustering hierarchy membership artifacts and separates comparable seed
ensembles from reference-only contrasts. It does not run clustering, execute
routes, promote wall/pathway claims, inspect basin quality/cost, or change
NanoClustering artifacts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
DEFAULT_NANO_ROOT = REPO_ROOT.parent / "1.1.4.KISTI_NanoClustering"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"

ENDPOINT_REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
PAIRWISE_ROWS_CSV = "nanoclustering_external_pairwise_landscape_rows.csv"
REFERENCE_CLUSTER_BY_SEED_CSV = "nanoclustering_external_reference_cluster_persistence_by_seed.csv"
REFERENCE_CLUSTER_SUMMARY_CSV = "nanoclustering_external_reference_cluster_persistence_summary.csv"
PRIOR_SUMMARY_CSV = "nanoclustering_external_prior_summary_rows.csv"
SUMMARY_JSON = "nanoclustering_external_landscape_summary.json"
REPORT_MD = "nanoclustering_external_landscape_report.md"
CONFIG_JSON = "nanoclustering_external_landscape_config.json"

CLAIM_BOUNDARY = (
    "External endpoint-landscape diagnostics only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)
QUALITY_COST_STATUS = "excluded_external_landscape_preparation"
ROUTE_EXECUTION_STATUS = "not_executed_external_membership_read_only"
WALL_PROMOTION_STATUS = "not_promoted_no_wall_pathway_trace"


@dataclass(frozen=True)
class EndpointSpec:
    run_id: str
    relative_path: str
    comparability_group: str
    comparison_family: str
    endpoint_role: str
    unit_col: str
    weight_col: str
    label_cols: tuple[str, ...]
    branch: str = ""
    seed: int | None = None
    pure_seed_ensemble: bool = False
    current_reference: bool = False
    notes: str = ""


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _endpoint_specs() -> list[EndpointSpec]:
    current_root = "outputs/specter2/_current"
    specs = [
        EndpointSpec(
            run_id="paris_active_micro3773",
            relative_path=(
                f"{current_root}/paris_preview_hierarchy_domain8_macro35_meso255_"
                "micro3773_20260527/nano_hierarchy_membership.parquet"
            ),
            comparability_group="paris_current_78049_reference_contrast",
            comparison_family="current_reference_contrast",
            endpoint_role="active_paris_reference_endpoint",
            unit_col="nano_id",
            weight_col="docs",
            label_cols=("micro_id", "meso_id", "macro_id", "domain_id"),
            current_reference=True,
            notes="Current Paris Domain8 micro3773 endpoint; reference only, not a seed ensemble.",
        ),
        EndpointSpec(
            run_id="paris_active_micro3906",
            relative_path=(
                f"{current_root}/paris_preview_hierarchy_domain8_macro35_meso255_"
                "micro3906_20260527/nano_hierarchy_membership.parquet"
            ),
            comparability_group="paris_current_78049_reference_contrast",
            comparison_family="current_reference_contrast",
            endpoint_role="same_upper_micro_refinement_contrast",
            unit_col="nano_id",
            weight_col="docs",
            label_cols=("micro_id", "meso_id", "macro_id", "domain_id"),
            notes="Same nano/docs and same upper partition as active micro3773, different micro cut.",
        ),
        EndpointSpec(
            run_id="sidecar_final_rust_seed001_macro24",
            relative_path=(
                f"{current_root}/sidecar_g0005_min250_gamma000650_core250_micro_g17_"
                "seed005_max68000_hardcap_final_macro24_20260513/rust_seed001/"
                "nano_hierarchy_membership.parquet"
            ),
            comparability_group="paris_current_78049_reference_contrast",
            comparison_family="branch_endpoint_contrast",
            endpoint_role="same_nano_docs_branch_contrast_endpoint",
            unit_col="nano_id",
            weight_col="docs",
            label_cols=("micro_id", "meso_id", "macro_id"),
            branch="rust",
            seed=1,
            notes="Same nano/docs as active Paris, but not a pure seed-only contrast.",
        ),
        EndpointSpec(
            run_id="sidecar_final_java_seed005_macro24",
            relative_path=(
                f"{current_root}/sidecar_g0005_min250_gamma000650_core250_micro_g17_"
                "seed005_max68000_hardcap_final_macro24_20260513/java_seed005/"
                "nano_hierarchy_membership.parquet"
            ),
            comparability_group="sidecar_final_macro24_unaligned_reference_only",
            comparison_family="reference_only_unaligned",
            endpoint_role="misaligned_nano_id_docs_reference_only",
            unit_col="nano_id",
            weight_col="docs",
            label_cols=("micro_id", "meso_id", "macro_id"),
            branch="java",
            seed=5,
            notes="Docs sum matches, but nano_id/docs alignment with active Paris is not exact.",
        ),
    ]
    for branch, unit_count in (("java", 78154), ("rust", 78119)):
        for seed in range(10):
            specs.append(
                EndpointSpec(
                    run_id=f"sidecar_{branch}_g0005_min250_gamma0p7_seed{seed:03d}",
                    relative_path=(
                        f"{current_root}/sidecar_g0005_min250_micro_seed_sweep_"
                        f"gamma0p7_20260512/{branch}/runs/seed{seed:03d}/"
                        "candidate_nano_to_micro_membership.parquet"
                    ),
                    comparability_group=(
                        f"sidecar_{branch}_g0005_min250_gamma0p7_candidate{unit_count}_"
                        "seed_ensemble"
                    ),
                    comparison_family="pure_seed_ensemble",
                    endpoint_role="seed_endpoint",
                    unit_col="original_cluster_id",
                    weight_col="doc_count",
                    label_cols=("candidate_micro_id",),
                    branch=branch,
                    seed=seed,
                    pure_seed_ensemble=True,
                    notes="Same branch candidate-node seed ensemble; pure seed contrast within branch.",
                )
            )
    return specs


def _load_endpoint(nano_root: Path, spec: EndpointSpec) -> pd.DataFrame:
    path = nano_root / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    columns = list(dict.fromkeys([spec.unit_col, spec.weight_col, *spec.label_cols]))
    frame = pq.read_table(path, columns=columns).to_pandas()
    frame = frame.rename(columns={spec.unit_col: "unit_id", spec.weight_col: "unit_weight"})
    frame["unit_id"] = frame["unit_id"].astype("int64")
    frame["unit_weight"] = frame["unit_weight"].astype("int64")
    for col in spec.label_cols:
        frame[col] = frame[col].astype("int64")
    return frame.sort_values(["unit_id", "unit_weight"]).reset_index(drop=True)


def _registry_rows(
    *,
    nano_root: Path,
    specs: list[EndpointSpec],
    memberships: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        frame = memberships[spec.run_id]
        row: dict[str, Any] = {
            "run_id": spec.run_id,
            "comparability_group": spec.comparability_group,
            "comparison_family": spec.comparison_family,
            "endpoint_role": spec.endpoint_role,
            "branch": spec.branch,
            "seed": spec.seed if spec.seed is not None else "",
            "pure_seed_ensemble": spec.pure_seed_ensemble,
            "current_reference": spec.current_reference,
            "relative_path": spec.relative_path,
            "absolute_path": str(nano_root / spec.relative_path),
            "unit_col": spec.unit_col,
            "weight_col": spec.weight_col,
            "label_cols": ";".join(spec.label_cols),
            "row_count": int(len(frame)),
            "unit_count": int(frame["unit_id"].nunique()),
            "unit_weight_sum": int(frame["unit_weight"].sum()),
            "unit_id_min": int(frame["unit_id"].min()),
            "unit_id_max": int(frame["unit_id"].max()),
            "unit_weight_min": int(frame["unit_weight"].min()),
            "unit_weight_max": int(frame["unit_weight"].max()),
            "route_execution_status": ROUTE_EXECUTION_STATUS,
            "wall_promotion_status": WALL_PROMOTION_STATUS,
            "quality_cost_status": QUALITY_COST_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
            "notes": spec.notes,
        }
        for col in spec.label_cols:
            row[f"{col}_count"] = int(frame[col].nunique())
            row[f"{col}_min"] = int(frame[col].min())
            row[f"{col}_max"] = int(frame[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["comparison_family", "comparability_group", "branch", "seed", "run_id"]
    )


def _metric_rows_for_group(
    specs: list[EndpointSpec],
    memberships: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_spec, right_spec in itertools.combinations(specs, 2):
        shared_labels = [col for col in left_spec.label_cols if col in set(right_spec.label_cols)]
        for label_col in shared_labels:
            left = memberships[left_spec.run_id][["unit_id", "unit_weight", label_col]].rename(
                columns={label_col: "left_label"}
            )
            right = memberships[right_spec.run_id][["unit_id", "unit_weight", label_col]].rename(
                columns={label_col: "right_label"}
            )
            aligned = left.merge(right, on=["unit_id", "unit_weight"], how="inner")
            min_rows = min(len(left), len(right))
            max_rows = max(len(left), len(right))
            alignment_share_min = len(aligned) / min_rows if min_rows else 0.0
            alignment_share_max = len(aligned) / max_rows if max_rows else 0.0
            status = (
                "metric_ready_exact_unit_weight_alignment"
                if min_rows and alignment_share_min >= 0.99 and alignment_share_max >= 0.99
                else "skipped_insufficient_unit_weight_alignment"
            )
            row: dict[str, Any] = {
                "comparability_group": left_spec.comparability_group,
                "comparison_family": left_spec.comparison_family,
                "left_run_id": left_spec.run_id,
                "right_run_id": right_spec.run_id,
                "left_branch": left_spec.branch,
                "right_branch": right_spec.branch,
                "left_seed": left_spec.seed if left_spec.seed is not None else "",
                "right_seed": right_spec.seed if right_spec.seed is not None else "",
                "level_label_col": label_col,
                "left_rows": int(len(left)),
                "right_rows": int(len(right)),
                "aligned_unit_weight_rows": int(len(aligned)),
                "alignment_share_of_smaller": alignment_share_min,
                "alignment_share_of_larger": alignment_share_max,
                "metric_status": status,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
            if status == "metric_ready_exact_unit_weight_alignment":
                left_labels = aligned["left_label"]
                right_labels = aligned["right_label"]
                row.update(
                    {
                        "left_cluster_count": int(left_labels.nunique()),
                        "right_cluster_count": int(right_labels.nunique()),
                        "ari": adjusted_rand_score(left_labels, right_labels),
                        "nmi_arithmetic": normalized_mutual_info_score(
                            left_labels,
                            right_labels,
                            average_method="arithmetic",
                        ),
                        "ami_arithmetic": adjusted_mutual_info_score(
                            left_labels,
                            right_labels,
                            average_method="arithmetic",
                        ),
                        "exact_label_equal_share": float((left_labels == right_labels).mean()),
                    }
                )
            rows.append(row)
    return rows


def _pairwise_rows(
    *,
    specs: list[EndpointSpec],
    memberships: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group_specs_iter in itertools.groupby(
        sorted(specs, key=lambda spec: spec.comparability_group),
        key=lambda spec: spec.comparability_group,
    ):
        group_specs = list(group_specs_iter)
        if len(group_specs) < 2:
            continue
        if group_specs[0].comparison_family == "pure_seed_ensemble":
            continue
        rows.extend(_metric_rows_for_group(group_specs, memberships))
    return pd.DataFrame(rows).sort_values(
        ["comparison_family", "comparability_group", "level_label_col", "left_run_id", "right_run_id"]
    )


def _existing_seed_pairwise_rows(nano_root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    current_root = nano_root / "outputs/specter2/_current"
    sources = [
        (
            "java",
            "sidecar_java_g0005_min250_gamma0p7_candidate78154_seed_ensemble",
            current_root
            / "sidecar_g0005_min250_micro_seed_sweep_gamma0p7_20260512/java/"
            "pairwise_seed_stability.csv",
        ),
        (
            "rust",
            "sidecar_rust_g0005_min250_gamma0p7_candidate78119_seed_ensemble",
            current_root
            / "sidecar_g0005_min250_micro_seed_sweep_gamma0p7_20260512/rust/"
            "pairwise_seed_stability.csv",
        ),
    ]
    registry_by_run = registry.set_index("run_id")
    rows: list[dict[str, Any]] = []
    for branch, group, path in sources:
        if not path.exists():
            continue
        frame = _read_csv(path)
        for record in frame.to_dict(orient="records"):
            left_seed = int(record["left_seed"])
            right_seed = int(record["right_seed"])
            left_run_id = f"sidecar_{branch}_g0005_min250_gamma0p7_seed{left_seed:03d}"
            right_run_id = f"sidecar_{branch}_g0005_min250_gamma0p7_seed{right_seed:03d}"
            left_rows = int(registry_by_run.loc[left_run_id, "row_count"])
            right_rows = int(registry_by_run.loc[right_run_id, "row_count"])
            rows.append(
                {
                    "comparability_group": group,
                    "comparison_family": "pure_seed_ensemble",
                    "left_run_id": left_run_id,
                    "right_run_id": right_run_id,
                    "left_branch": branch,
                    "right_branch": branch,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "level_label_col": "candidate_micro_id",
                    "left_rows": left_rows,
                    "right_rows": right_rows,
                    "aligned_unit_weight_rows": min(left_rows, right_rows),
                    "alignment_share_of_smaller": 1.0,
                    "alignment_share_of_larger": 1.0,
                    "metric_status": "metric_ready_existing_pairwise_seed_stability_csv",
                    "left_cluster_count": "",
                    "right_cluster_count": "",
                    "ari": _safe_float(record.get("ari")),
                    "nmi_arithmetic": _safe_float(record.get("nmi_arithmetic")),
                    "ami_arithmetic": None,
                    "exact_label_equal_share": None,
                    "source_pairwise_csv": str(path),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "quality_cost_status": QUALITY_COST_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def _reference_persistence_by_seed(
    *,
    specs: list[EndpointSpec],
    memberships: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_specs = [spec for spec in specs if spec.pure_seed_ensemble]
    for group, group_specs_iter in itertools.groupby(
        sorted(seed_specs, key=lambda spec: spec.comparability_group),
        key=lambda spec: spec.comparability_group,
    ):
        group_specs = list(group_specs_iter)
        reference = next(spec for spec in group_specs if spec.seed == 0)
        ref_frame = memberships[reference.run_id][
            ["unit_id", "unit_weight", "candidate_micro_id"]
        ].rename(columns={"candidate_micro_id": "ref_cluster_id"})
        ref_totals = ref_frame.groupby("ref_cluster_id", as_index=False).agg(
            ref_unit_count=("unit_id", "size"),
            ref_weight_sum=("unit_weight", "sum"),
        )
        for spec in group_specs:
            if spec.seed == 0:
                continue
            other = memberships[spec.run_id][
                ["unit_id", "unit_weight", "candidate_micro_id"]
            ].rename(columns={"candidate_micro_id": "run_cluster_id"})
            aligned = ref_frame.merge(other, on=["unit_id", "unit_weight"], how="inner")
            overlap = aligned.groupby(["ref_cluster_id", "run_cluster_id"], as_index=False).agg(
                overlap_unit_count=("unit_id", "size"),
                overlap_weight_sum=("unit_weight", "sum"),
            )
            overlap = overlap.merge(ref_totals, on="ref_cluster_id", how="left")
            overlap["share_ref_units"] = overlap["overlap_unit_count"] / overlap["ref_unit_count"]
            overlap["share_ref_weight"] = overlap["overlap_weight_sum"] / overlap["ref_weight_sum"]
            overlap = overlap.sort_values(
                ["ref_cluster_id", "share_ref_weight", "share_ref_units", "run_cluster_id"],
                ascending=[True, False, False, True],
            )
            best = overlap.drop_duplicates("ref_cluster_id", keep="first")
            for row in best.itertuples(index=False):
                rows.append(
                    {
                        "comparability_group": group,
                        "branch": spec.branch,
                        "reference_run_id": reference.run_id,
                        "comparison_run_id": spec.run_id,
                        "reference_seed": reference.seed,
                        "comparison_seed": spec.seed,
                        "ref_cluster_id": int(row.ref_cluster_id),
                        "best_run_cluster_id": int(row.run_cluster_id),
                        "ref_unit_count": int(row.ref_unit_count),
                        "ref_weight_sum": int(row.ref_weight_sum),
                        "overlap_unit_count": int(row.overlap_unit_count),
                        "overlap_weight_sum": int(row.overlap_weight_sum),
                        "best_share_ref_units": float(row.share_ref_units),
                        "best_share_ref_weight": float(row.share_ref_weight),
                        "route_execution_status": ROUTE_EXECUTION_STATUS,
                        "wall_promotion_status": WALL_PROMOTION_STATUS,
                        "quality_cost_status": QUALITY_COST_STATUS,
                        "claim_boundary": CLAIM_BOUNDARY,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["comparability_group", "branch", "ref_cluster_id", "comparison_seed"]
    )


def _reference_persistence_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    if by_seed.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["comparability_group", "branch", "ref_cluster_id"]
    for keys, group in by_seed.groupby(group_cols):
        comparability_group, branch, ref_cluster_id = keys
        rows.append(
            {
                "comparability_group": comparability_group,
                "branch": branch,
                "ref_cluster_id": int(ref_cluster_id),
                "comparison_seed_count": int(group["comparison_seed"].nunique()),
                "ref_unit_count": int(group["ref_unit_count"].iloc[0]),
                "ref_weight_sum": int(group["ref_weight_sum"].iloc[0]),
                "best_share_ref_weight_min": float(group["best_share_ref_weight"].min()),
                "best_share_ref_weight_q10": float(group["best_share_ref_weight"].quantile(0.1)),
                "best_share_ref_weight_median": float(group["best_share_ref_weight"].median()),
                "best_share_ref_weight_mean": float(group["best_share_ref_weight"].mean()),
                "best_share_ref_weight_max": float(group["best_share_ref_weight"].max()),
                "best_share_ref_units_min": float(group["best_share_ref_units"].min()),
                "best_share_ref_units_median": float(group["best_share_ref_units"].median()),
                "runs_ge80_weight": int((group["best_share_ref_weight"] >= 0.8).sum()),
                "runs_ge90_weight": int((group["best_share_ref_weight"] >= 0.9).sum()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "quality_cost_status": QUALITY_COST_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["comparability_group", "best_share_ref_weight_min", "ref_weight_sum"],
        ascending=[True, True, False],
    )


def _prior_summary_rows(nano_root: Path) -> pd.DataFrame:
    review_dir = nano_root / "outputs/specter2/_current/past_seed_nmi_ami_review_20260512"
    rows = []
    for source_name, filename in (
        ("computed_unweighted_review", "computed_summary.csv"),
        ("existing_weighted_context", "existing_weighted_summary.csv"),
    ):
        path = review_dir / filename
        if not path.exists():
            continue
        frame = _read_csv(path)
        frame.insert(0, "summary_source", source_name)
        frame["source_path"] = str(path)
        frame["claim_boundary"] = (
            "Prior NanoClustering review context only; keep separate from active "
            "Paris 78049 endpoint claims unless population identity is verified."
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame[columns].head(max_rows).copy()
    rows = rows.fillna("")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for record in rows.to_dict(orient="records"):
        body.append("| " + " | ".join(str(record[col]) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _pairwise_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    ready = pairwise[pairwise["metric_status"].astype(str).str.startswith("metric_ready")].copy()
    if ready.empty:
        return pd.DataFrame()
    return (
        ready.groupby(["comparison_family", "comparability_group", "level_label_col"], as_index=False)
        .agg(
            pair_count=("ari", "size"),
            ari_min=("ari", "min"),
            ari_mean=("ari", "mean"),
            ari_median=("ari", "median"),
            ari_max=("ari", "max"),
            nmi_min=("nmi_arithmetic", "min"),
            nmi_mean=("nmi_arithmetic", "mean"),
            nmi_median=("nmi_arithmetic", "median"),
            nmi_max=("nmi_arithmetic", "max"),
            ami_min=("ami_arithmetic", "min"),
            ami_mean=("ami_arithmetic", "mean"),
            ami_median=("ami_arithmetic", "median"),
            ami_max=("ami_arithmetic", "max"),
        )
        .sort_values(["comparison_family", "comparability_group", "level_label_col"])
    )


def _persistence_branch_summary(persistence_summary: pd.DataFrame) -> pd.DataFrame:
    if persistence_summary.empty:
        return pd.DataFrame()
    return (
        persistence_summary.groupby(["comparability_group", "branch"], as_index=False)
        .agg(
            ref_cluster_count=("ref_cluster_id", "size"),
            ref_weight_sum=("ref_weight_sum", "sum"),
            min_best_share_weight=("best_share_ref_weight_min", "min"),
            q10_best_share_weight=("best_share_ref_weight_min", lambda s: s.quantile(0.1)),
            median_best_share_weight=("best_share_ref_weight_median", "median"),
            clusters_never_ge80=(
                "runs_ge80_weight",
                lambda s: int((s == 0).sum()),
            ),
            clusters_all_ge80=(
                "runs_ge80_weight",
                lambda s: int((s == 9).sum()),
            ),
        )
        .sort_values(["comparability_group", "branch"])
    )


def _write_report(
    *,
    output_dir: Path,
    nano_root: Path,
    registry: pd.DataFrame,
    pairwise: pd.DataFrame,
    pair_summary: pd.DataFrame,
    persistence_summary: pd.DataFrame,
    prior_summary: pd.DataFrame,
) -> None:
    branch_summary = _persistence_branch_summary(persistence_summary)
    skipped = pairwise[~pairwise["metric_status"].astype(str).str.startswith("metric_ready")]
    worst_persistence = persistence_summary.sort_values(
        ["best_share_ref_weight_min", "ref_weight_sum"],
        ascending=[True, False],
    ).head(12)
    text = [
        "# NanoClustering External Endpoint Landscape",
        "",
        f"- nano_root: `{nano_root}`",
        f"- endpoint_rows: `{len(registry)}`",
        f"- pairwise_rows: `{len(pairwise)}`",
        f"- persistence_reference_clusters: `{len(persistence_summary)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Endpoint Families",
        "",
        _markdown_table(
            registry.groupby(["comparison_family", "comparability_group"], as_index=False).agg(
                endpoint_count=("run_id", "size"),
                unit_weight_sum_min=("unit_weight_sum", "min"),
                unit_weight_sum_max=("unit_weight_sum", "max"),
            ),
            ["comparison_family", "comparability_group", "endpoint_count", "unit_weight_sum_min", "unit_weight_sum_max"],
        ),
        "",
        "## Pairwise Metric Summary",
        "",
        _markdown_table(
            pair_summary,
            [
                "comparison_family",
                "comparability_group",
                "level_label_col",
                "pair_count",
                "ari_min",
                "ari_mean",
                "nmi_mean",
                "ami_mean",
            ],
        ),
        "",
        "## Skipped Pairwise Rows",
        "",
        _markdown_table(
            skipped,
            [
                "comparability_group",
                "left_run_id",
                "right_run_id",
                "level_label_col",
                "aligned_unit_weight_rows",
                "alignment_share_of_smaller",
                "metric_status",
            ],
            max_rows=12,
        ),
        "",
        "## Seed-Ensemble Reference Cluster Persistence",
        "",
        _markdown_table(
            branch_summary,
            [
                "comparability_group",
                "branch",
                "ref_cluster_count",
                "min_best_share_weight",
                "q10_best_share_weight",
                "median_best_share_weight",
                "clusters_never_ge80",
                "clusters_all_ge80",
            ],
        ),
        "",
        "## Most Volatile Reference Clusters",
        "",
        _markdown_table(
            worst_persistence,
            [
                "comparability_group",
                "branch",
                "ref_cluster_id",
                "ref_weight_sum",
                "best_share_ref_weight_min",
                "best_share_ref_weight_median",
                "runs_ge80_weight",
                "runs_ge90_weight",
            ],
            max_rows=12,
        ),
        "",
        "## Prior Summary Context",
        "",
        _markdown_table(
            prior_summary,
            [
                "summary_source",
                "case",
                "view",
                "run_count",
                "pair_count",
                "unit_count",
                "nmi_mean",
                "ami_mean",
            ],
            max_rows=12,
        ),
        "",
        "## Read",
        "",
        "- The active Paris 78049 endpoint is a current reference, not a multi-seed sample by itself.",
        "- The Java and Rust sidecar seed sweeps are the clean seed ensembles for first basin-existence checks.",
        "- Historical upper-level seed instability is retained as prior context, but it is not collapsed into the active Paris claim surface.",
        "- Wall/pathway claims remain unsupported here because these artifacts provide endpoint memberships, not optimizer-native route traces.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(nano_root: Path, output_dir: Path) -> dict[str, Any]:
    specs = _endpoint_specs()
    memberships = {spec.run_id: _load_endpoint(nano_root, spec) for spec in specs}
    registry = _registry_rows(nano_root=nano_root, specs=specs, memberships=memberships)
    pairwise = _pairwise_rows(specs=specs, memberships=memberships)
    existing_seed_pairwise = _existing_seed_pairwise_rows(nano_root, registry)
    if not existing_seed_pairwise.empty:
        pairwise = pd.concat([pairwise, existing_seed_pairwise], ignore_index=True, sort=False)
        pairwise = pairwise.sort_values(
            [
                "comparison_family",
                "comparability_group",
                "level_label_col",
                "left_run_id",
                "right_run_id",
            ]
        )
    pair_summary = _pairwise_summary(pairwise)
    persistence_by_seed = _reference_persistence_by_seed(specs=specs, memberships=memberships)
    persistence_summary = _reference_persistence_summary(persistence_by_seed)
    prior_summary = _prior_summary_rows(nano_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(registry, output_dir / ENDPOINT_REGISTRY_CSV)
    _write_csv(pairwise, output_dir / PAIRWISE_ROWS_CSV)
    _write_csv(persistence_by_seed, output_dir / REFERENCE_CLUSTER_BY_SEED_CSV)
    _write_csv(persistence_summary, output_dir / REFERENCE_CLUSTER_SUMMARY_CSV)
    _write_csv(prior_summary, output_dir / PRIOR_SUMMARY_CSV)

    branch_summary = _persistence_branch_summary(persistence_summary)
    summary = {
        "ok": True,
        "nano_root": str(nano_root),
        "output_dir": _rel(output_dir),
        "endpoint_count": int(len(registry)),
        "pairwise_row_count": int(len(pairwise)),
        "pairwise_ready_count": int(
            pairwise["metric_status"].astype(str).str.startswith("metric_ready").sum()
        ),
        "pairwise_status_counts": _count(pairwise, "metric_status"),
        "reference_cluster_persistence_by_seed_rows": int(len(persistence_by_seed)),
        "reference_cluster_persistence_summary_rows": int(len(persistence_summary)),
        "prior_summary_rows": int(len(prior_summary)),
        "endpoint_family_counts": _count(registry, "comparison_family"),
        "comparability_group_counts": _count(registry, "comparability_group"),
        "pairwise_summary": pair_summary.to_dict(orient="records"),
        "persistence_branch_summary": branch_summary.to_dict(orient="records"),
        "claim_boundary": CLAIM_BOUNDARY,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "quality_cost_status": QUALITY_COST_STATUS,
        "outputs": {
            "endpoint_registry_csv": _rel(output_dir / ENDPOINT_REGISTRY_CSV),
            "pairwise_rows_csv": _rel(output_dir / PAIRWISE_ROWS_CSV),
            "reference_cluster_by_seed_csv": _rel(output_dir / REFERENCE_CLUSTER_BY_SEED_CSV),
            "reference_cluster_summary_csv": _rel(output_dir / REFERENCE_CLUSTER_SUMMARY_CSV),
            "prior_summary_csv": _rel(output_dir / PRIOR_SUMMARY_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config = {
        "nano_root": str(nano_root),
        "output_dir": str(output_dir),
        "script": _rel(Path(__file__)),
        "endpoint_specs": [
            {
                "run_id": spec.run_id,
                "relative_path": spec.relative_path,
                "comparability_group": spec.comparability_group,
                "comparison_family": spec.comparison_family,
                "endpoint_role": spec.endpoint_role,
                "unit_col": spec.unit_col,
                "weight_col": spec.weight_col,
                "label_cols": list(spec.label_cols),
                "branch": spec.branch,
                "seed": spec.seed,
                "pure_seed_ensemble": spec.pure_seed_ensemble,
                "current_reference": spec.current_reference,
                "notes": spec.notes,
            }
            for spec in specs
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        nano_root=nano_root,
        registry=registry,
        pairwise=pairwise,
        pair_summary=pair_summary,
        persistence_summary=persistence_summary,
        prior_summary=prior_summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nano-root",
        type=Path,
        default=DEFAULT_NANO_ROOT,
        help="Path to the KISTI_NanoClustering checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Sciscape Track C external landscape outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(nano_root=args.nano_root.resolve(), output_dir=args.output_dir.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
