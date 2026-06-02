#!/usr/bin/env python3
"""Compare prior Track C data with the NanoClustering endpoint data.

This is a diagnostic contrast for the density hypothesis. It reads existing
summaries and membership-derived endpoint artifacts only. It does not run
clustering, execute routes, promote wall/pathway claims, or inspect basin
quality/cost.
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
DEFAULT_NANO_ROOT = REPO_ROOT.parent / "1.1.4.KISTI_NanoClustering"
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_dataset_contrast_20260530"
)

METHODOLOGY_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_20260529"
EXISTENCE_DIR = BASE_RESULT_DIR / "leiden_basin_existence_assumption_audit_20260529"
NANO_LANDSCAPE_DIR = BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
NANO_INVENTORY_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_fragmentation_boundary_inventory_20260530"
)
NANO_PAIR_CASE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_definition_core_pair_cases_20260530"
)

DATASET_CONTRAST_CSV = "dataset_contrast_rows.csv"
NETWORK_DENSITY_CSV = "network_density_rows.csv"
ENDPOINT_INSTABILITY_CSV = "endpoint_instability_rows.csv"
DENSITY_HYPOTHESIS_CSV = "density_hypothesis_assessment_rows.csv"
SUMMARY_JSON = "dataset_contrast_summary.json"
REPORT_MD = "dataset_contrast_report.md"
CONFIG_JSON = "dataset_contrast_config.json"

CLAIM_BOUNDARY = (
    "Dataset contrast and density-hypothesis diagnostic only; no route execution, "
    "wall/pathway promotion, basin-quality claim, cost claim, or directed-search claim."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _density(edge_count: int | float | None, node_count: int | float | None) -> float | None:
    if not edge_count or not node_count or node_count <= 1:
        return None
    possible = float(node_count) * float(node_count - 1) / 2.0
    return float(edge_count) / possible


def _avg_degree(edge_count: int | float | None, node_count: int | float | None) -> float | None:
    if not edge_count or not node_count:
        return None
    return 2.0 * float(edge_count) / float(node_count)


def _safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _network_density_rows(nano_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    prior_graph_summaries = [
        (
            "prior_sciscape_g016_gamma0p0085_cluster_graph",
            BASE_RESULT_DIR / "g016_gamma0p0085_cluster_graph_summary.json",
        ),
        (
            "prior_sciscape_bcrefresh_g0005_recguard_cluster_graph",
            BASE_RESULT_DIR / "bcrefresh_g0005_recguard_cluster_graph_summary.json",
        ),
    ]
    for dataset_id, path in prior_graph_summaries:
        data = _read_json(path)
        node_count = data.get("n_active_clusters") or data.get("n_clusters")
        graph_path = data.get("graph_path")
        graph_available = bool(graph_path and Path(str(graph_path)).exists())
        rows.append(
            {
                "dataset_id": dataset_id,
                "data_family": "prior_sciscape_cluster_graph_summary",
                "node_count": node_count,
                "edge_count": None,
                "avg_degree_if_undirected": None,
                "undirected_density": None,
                "edge_source_status": "edge_rows_unavailable_locally",
                "graph_path": graph_path or "",
                "graph_path_exists": graph_available,
                "total_doc_weight": data.get("total_doc_weight"),
                "total_internal_weight": data.get("total_internal_weight"),
                "total_external_weight": data.get("total_external_weight"),
                "external_internal_weight_ratio": (
                    float(data["total_external_weight"]) / float(data["total_internal_weight"])
                    if data.get("total_internal_weight")
                    else None
                ),
                "conductance_p50": _safe_get(data, "conductance", "p50"),
                "conductance_p90": _safe_get(data, "conductance", "p90"),
                "doc_weight_p50": _safe_get(data, "doc_weight", "p50"),
                "doc_weight_p90": _safe_get(data, "doc_weight", "p90"),
                "source_path": _rel(path),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    nano_reports = [
        (
            "current_nanoclustering_java_gamma0p7_candidate_graph",
            nano_root
            / "outputs/specter2/_current/sidecar_g0005_min250_micro_seed_sweep_gamma0p7_20260512/java/report.json",
        ),
        (
            "current_nanoclustering_rust_gamma0p7_candidate_graph",
            nano_root
            / "outputs/specter2/_current/sidecar_g0005_min250_micro_seed_sweep_gamma0p7_20260512/rust/report.json",
        ),
    ]
    for dataset_id, path in nano_reports:
        data = _read_json(path)
        node_count = int(_safe_get(data, "graph", "n_nodes"))
        edge_count = int(_safe_get(data, "graph", "n_edges"))
        rows.append(
            {
                "dataset_id": dataset_id,
                "data_family": "current_nanoclustering_candidate_graph",
                "node_count": node_count,
                "edge_count": edge_count,
                "avg_degree_if_undirected": _avg_degree(edge_count, node_count),
                "undirected_density": _density(edge_count, node_count),
                "edge_source_status": "edge_count_from_report",
                "graph_path": _safe_get(data, "inputs", "int_edges", default=""),
                "graph_path_exists": Path(str(_safe_get(data, "inputs", "int_edges", default=""))).exists(),
                "total_doc_weight": _safe_get(data, "graph", "total_docs"),
                "total_internal_weight": None,
                "total_external_weight": None,
                "external_internal_weight_ratio": None,
                "conductance_p50": None,
                "conductance_p90": None,
                "doc_weight_p50": _safe_get(data, "seed_rows", default=[{}])[0]
                .get("final_summary", {})
                .get("docs_median"),
                "doc_weight_p90": _safe_get(data, "seed_rows", default=[{}])[0]
                .get("final_summary", {})
                .get("docs_q90"),
                "source_path": str(path),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    paris_report = _read_json(
        nano_root
        / "outputs/specter2/_current/paris_nano_dendrogram_probe_top20_union_log1p_uniform_20260526/report.json"
    )
    node_count = int(_safe_get(paris_report, "graph", "node_count"))
    source_edges = int(_safe_get(paris_report, "graph", "source", "edge_rows")) - int(
        _safe_get(paris_report, "graph", "source", "self_rows", default=0)
    )
    sparse_edges = int(_safe_get(paris_report, "graph", "sparse_edges", "edge_pairs"))
    for dataset_id, edge_count, status in [
        ("current_paris_active_source_graph_nonself", source_edges, "source_edge_rows_minus_self_rows"),
        ("current_paris_active_top20_union_sparse_graph", sparse_edges, "top20_union_sparse_edge_pairs"),
    ]:
        rows.append(
            {
                "dataset_id": dataset_id,
                "data_family": "current_nanoclustering_paris_graph_probe",
                "node_count": node_count,
                "edge_count": edge_count,
                "avg_degree_if_undirected": _avg_degree(edge_count, node_count),
                "undirected_density": _density(edge_count, node_count),
                "edge_source_status": status,
                "graph_path": _safe_get(paris_report, "graph", "source", "int_edges", default="")
                if "source" in dataset_id
                else _safe_get(paris_report, "outputs", "sparse_edges_parquet", default=""),
                "graph_path_exists": None,
                "total_doc_weight": None,
                "total_internal_weight": None,
                "total_external_weight": None,
                "external_internal_weight_ratio": None,
                "conductance_p50": None,
                "conductance_p90": None,
                "doc_weight_p50": None,
                "doc_weight_p90": None,
                "source_path": str(
                    nano_root
                    / "outputs/specter2/_current/paris_nano_dendrogram_probe_top20_union_log1p_uniform_20260526/report.json"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    structural = _read_json(
        nano_root / "outputs/specter2/_current/final_rrf_nano_graph_20260429/structural_graph_summary.json"
    )
    node_count = int(structural["structural_nodes"])
    edge_count = int(structural["structural_undirected_edges"])
    rows.append(
        {
            "dataset_id": "current_final_rrf_structural_nano_graph",
            "data_family": "current_nanoclustering_full_structural_graph",
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_degree_if_undirected": _avg_degree(edge_count, node_count),
            "undirected_density": _density(edge_count, node_count),
            "edge_source_status": "structural_undirected_edges_from_summary",
            "graph_path": _safe_get(structural, "outputs", "nano_undirected_edges_structural", default=""),
            "graph_path_exists": None,
            "total_doc_weight": structural.get("structural_docs"),
            "total_internal_weight": None,
            "total_external_weight": None,
            "external_internal_weight_ratio": None,
            "conductance_p50": None,
            "conductance_p90": None,
            "doc_weight_p50": None,
            "doc_weight_p90": None,
            "source_path": str(
                nano_root / "outputs/specter2/_current/final_rrf_nano_graph_20260429/structural_graph_summary.json"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return pd.DataFrame(rows)


def _dataset_contrast_rows() -> pd.DataFrame:
    panel_summary = _read_json(METHODOLOGY_DIR / "precommitted_nonfield34_panel_v0_summary.json")
    enrich_summary = _read_json(METHODOLOGY_DIR / "methodology_v0_evidence_enrichment_summary.json")
    existence_summary = _read_json(EXISTENCE_DIR / "basin_existence_assumption_summary.json")
    nano_landscape = _read_json(NANO_LANDSCAPE_DIR / "nanoclustering_external_landscape_summary.json")
    nano_inventory = _read_json(NANO_INVENTORY_DIR / "nanoclustering_fragmentation_boundary_inventory_summary.json")
    nano_pair_cases = _read_json(NANO_PAIR_CASE_DIR / "nanoclustering_definition_core_pair_case_summary.json")

    rows = [
        {
            "contrast_axis": "case_universe",
            "prior_track_c": (
                f"{panel_summary['panel_case_count']} precommitted non-field34 cases, "
                f"{panel_summary['panel_pair_candidate_count']} accepted distinct pair candidates"
            ),
            "current_nanoclustering": (
                f"{nano_landscape['endpoint_count']} endpoint rows, "
                f"{nano_landscape['reference_cluster_persistence_summary_rows']} seed0 reference clusters"
            ),
            "read": "prior data is case/pair-panel evidence; current data is seed-ensemble endpoint cartography",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "contrast_axis": "full_membership_coverage",
            "prior_track_c": (
                f"{enrich_summary['endpoint_evidence_row_count']} endpoint evidence rows; "
                f"{enrich_summary['both_full_membership_cache_pair_count']} pairs with both full-membership caches"
            ),
            "current_nanoclustering": (
                f"{nano_landscape['reference_cluster_persistence_by_seed_rows']} reference-by-seed rows "
                "from aligned memberships"
            ),
            "read": "current data has much broader membership-level observability",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "contrast_axis": "candidate_basin_signal",
            "prior_track_c": (
                f"{existence_summary['strong_meaningful_distinct_pair_count']} strong and "
                f"{existence_summary['moderate_meaningful_distinct_pair_count']} moderate meaningful distinct pairs"
            ),
            "current_nanoclustering": (
                f"{nano_inventory['global_counts']['recurrent_strong_fragmentation_candidate']} recurrent strong "
                f"fragmentation candidates; {nano_pair_cases['endpoint_pair_event_count']} definition-core endpoint-pair events"
            ),
            "read": "current signal is repeated fragmentation families, not just support-local distinct pair rows",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "contrast_axis": "wall_pathway_readiness",
            "prior_track_c": "0 executable route candidates and 0 wall-promotion candidates under current gates",
            "current_nanoclustering": "membership-only endpoint-pair substrate; no route or wall/pathway trace",
            "read": "neither side currently supports a wall/pathway claim",
            "claim_boundary": CLAIM_BOUNDARY,
        },
    ]
    return pd.DataFrame(rows)


def _endpoint_instability_rows() -> pd.DataFrame:
    pairwise = _read_csv(NANO_LANDSCAPE_DIR / "nanoclustering_external_pairwise_landscape_rows.csv")
    persistence = _read_csv(
        NANO_LANDSCAPE_DIR / "nanoclustering_external_reference_cluster_persistence_summary.csv"
    )
    inventory = _read_csv(
        NANO_INVENTORY_DIR / "nanoclustering_fragmentation_boundary_cluster_inventory.csv"
    )
    pair_cases = _read_csv(NANO_PAIR_CASE_DIR / "nanoclustering_definition_core_pair_event_rows.csv")
    rows: list[dict[str, Any]] = []

    pure = pairwise[pairwise["comparison_family"].eq("pure_seed_ensemble")].copy()
    for branch, group in pure.groupby("left_branch", dropna=False):
        rows.append(
            {
                "metric_scope": "seed_pairwise_partition_similarity",
                "branch": branch,
                "row_count": int(len(group)),
                "ari_min": float(group["ari"].min()),
                "ari_median": float(group["ari"].median()),
                "ari_mean": float(group["ari"].mean()),
                "ari_max": float(group["ari"].max()),
                "nmi_median": float(group["nmi_arithmetic"].median()),
                "reference_cluster_count": None,
                "candidate_count": None,
                "event_count": None,
                "read": "seed-to-seed ARI is moderate while NMI stays high, suggesting broad hierarchy consistency with local boundary movement",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    for branch, group in persistence.groupby("branch"):
        rows.append(
            {
                "metric_scope": "reference_cluster_persistence",
                "branch": branch,
                "row_count": int(len(group)),
                "ari_min": None,
                "ari_median": None,
                "ari_mean": None,
                "ari_max": None,
                "nmi_median": None,
                "reference_cluster_count": int(len(group)),
                "candidate_count": int(group["best_share_ref_weight_min"].lt(0.5).sum()),
                "event_count": int(group["runs_ge80_weight"].lt(group["comparison_seed_count"]).sum()),
                "read": (
                    f"{int(group['best_share_ref_weight_min'].lt(0.5).sum())} references have at least one strong "
                    "fragmentation seed"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    for branch, group in inventory.groupby("branch"):
        rows.append(
            {
                "metric_scope": "fragmentation_rule_family",
                "branch": branch,
                "row_count": int(len(group)),
                "ari_min": None,
                "ari_median": None,
                "ari_mean": None,
                "ari_max": None,
                "nmi_median": None,
                "reference_cluster_count": int(len(group)),
                "candidate_count": int(group["is_recurrent_strong_fragmentation_candidate"].sum()),
                "event_count": int(group["strong_fragmentation_event_count"].sum()),
                "read": "recurrent strong fragmentation is a minority but nontrivial subset of seed0 references",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    for (tier, branch), group in pair_cases.groupby(["boundary_family_tier", "branch"]):
        rows.append(
            {
                "metric_scope": f"definition_core_pair_cases:{tier}",
                "branch": branch,
                "row_count": int(len(group)),
                "ari_min": None,
                "ari_median": None,
                "ari_mean": None,
                "ari_max": None,
                "nmi_median": None,
                "reference_cluster_count": int(group["family_id"].nunique()),
                "candidate_count": int(group["endpoint_pair_role"].eq("severe_definition_core_endpoint_pair").sum()),
                "event_count": int(len(group)),
                "read": ";".join(
                    f"{key}:{value}"
                    for key, value in sorted(group["boundary_pattern"].value_counts().to_dict().items())
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    return pd.DataFrame(rows)


def _density_hypothesis_rows(network_density: pd.DataFrame) -> pd.DataFrame:
    current_dense = network_density[
        network_density["data_family"].isin(
            {
                "current_nanoclustering_candidate_graph",
                "current_nanoclustering_paris_graph_probe",
                "current_nanoclustering_full_structural_graph",
            }
        )
        & network_density["edge_count"].notna()
        & network_density["dataset_id"].ne("current_paris_active_top20_union_sparse_graph")
    ]
    top_avg_degree = float(current_dense["avg_degree_if_undirected"].max())
    top_density = float(current_dense["undirected_density"].max())
    rows = [
        {
            "question": "is_current_network_dense",
            "assessment": "yes_for_full_candidate_graphs",
            "evidence": (
                f"current full/candidate graphs have avg_degree up to {top_avg_degree:.1f} "
                f"and undirected density up to {top_density:.4f}; top20-union sparse projection is far lower"
            ),
            "limitation": "prior graph edge rows are not locally available for direct density recomputation",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "question": "does_density_alone_explain_basin_fragmentation",
            "assessment": "not_proven",
            "evidence": (
                "fragmentation survives matched stable controls and separates into repeat_severe versus "
                "persistent_mixed endpoint-pair patterns"
            ),
            "limitation": (
                "we do not yet have a controlled same-node graph-density ablation; current comparison changes "
                "data universe, endpoint protocol, and membership coverage at the same time"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "question": "most_plausible_current_read",
            "assessment": "density_is_a_contributor_not_a_sufficient_explanation",
            "evidence": (
                "high degree and high weak-edge mass can create many near-boundary alternatives, but the observed "
                "signal is family-specific fragmentation/absorption rather than uniform seed noise"
            ),
            "limitation": "needs within-graph local cut/degree and density-thinning checks before promotion",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        {
            "question": "next_test",
            "assessment": "within_current_graph_control",
            "evidence": (
                "compare volatile/recurrent families against stable-like references on local weighted degree, "
                "cut ratio, top-neighbor concentration, and top20 sparse versus full graph membership behavior"
            ),
            "limitation": "this is a diagnostic plan, not a completed causal test",
            "claim_boundary": CLAIM_BOUNDARY,
        },
    ]
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    suffix = []
    if len(frame) > max_rows:
        suffix.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join([header, separator, *body, *suffix])


def _write_report(
    *,
    output_dir: Path,
    dataset_contrast: pd.DataFrame,
    network_density: pd.DataFrame,
    endpoint_instability: pd.DataFrame,
    density_hypothesis: pd.DataFrame,
) -> None:
    text = [
        "# NanoClustering Dataset Contrast And Density Diagnostic",
        "",
        f"- network_density_rows: `{len(network_density)}`",
        f"- dataset_contrast_rows: `{len(dataset_contrast)}`",
        f"- endpoint_instability_rows: `{len(endpoint_instability)}`",
        f"- density_hypothesis_rows: `{len(density_hypothesis)}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Dataset Contrast",
        "",
        _markdown_table(
            dataset_contrast,
            ["contrast_axis", "prior_track_c", "current_nanoclustering", "read"],
            max_rows=10,
        ),
        "",
        "## Network Density Rows",
        "",
        _markdown_table(
            network_density,
            [
                "dataset_id",
                "data_family",
                "node_count",
                "edge_count",
                "avg_degree_if_undirected",
                "undirected_density",
                "edge_source_status",
            ],
            max_rows=20,
        ),
        "",
        "## Endpoint Instability Rows",
        "",
        _markdown_table(
            endpoint_instability,
            [
                "metric_scope",
                "branch",
                "row_count",
                "ari_median",
                "nmi_median",
                "reference_cluster_count",
                "candidate_count",
                "event_count",
                "read",
            ],
            max_rows=30,
        ),
        "",
        "## Density Hypothesis Assessment",
        "",
        _markdown_table(
            density_hypothesis,
            ["question", "assessment", "evidence", "limitation"],
            max_rows=10,
        ),
        "",
        "## Read",
        "",
        "- The current NanoClustering full/candidate graph is dense at the nano level: about 78k nodes and about 105-107M full/candidate edges, implying average degree around 2.7k under an undirected interpretation.",
        "- The top20-union projection is not dense in that sense: about 1.14M edge pairs and average degree around 29.3. This gives us a natural density contrast inside the current data family.",
        "- Density is a plausible contributor because high weak-edge mass can make many local alternatives near-tied, but it is not yet a sufficient explanation for the basin signal.",
        "- The current signal is not uniform seed noise: repeat-severe core and persistent-mixed core produce different endpoint-pair archetypes, and matched stable controls already argued against pure size selection.",
        "- The next causal check should be within the current graph: compare volatile/recurrent families to stable-like references by local degree, cut ratio, top-neighbor concentration, and full versus top20-sparse behavior.",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(text) + "\n", encoding="utf-8")


def materialize(*, nano_root: Path, output_dir: Path) -> dict[str, Any]:
    dataset_contrast = _dataset_contrast_rows()
    network_density = _network_density_rows(nano_root)
    endpoint_instability = _endpoint_instability_rows()
    density_hypothesis = _density_hypothesis_rows(network_density)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(dataset_contrast, output_dir / DATASET_CONTRAST_CSV)
    _write_csv(network_density, output_dir / NETWORK_DENSITY_CSV)
    _write_csv(endpoint_instability, output_dir / ENDPOINT_INSTABILITY_CSV)
    _write_csv(density_hypothesis, output_dir / DENSITY_HYPOTHESIS_CSV)
    _write_report(
        output_dir=output_dir,
        dataset_contrast=dataset_contrast,
        network_density=network_density,
        endpoint_instability=endpoint_instability,
        density_hypothesis=density_hypothesis,
    )

    current_dense = network_density[
        network_density["data_family"].str.startswith("current_nanoclustering")
        & network_density["edge_count"].notna()
    ].copy()
    summary = {
        "ok": True,
        "nano_root": str(nano_root),
        "output_dir": _rel(output_dir),
        "network_density_row_count": int(len(network_density)),
        "dataset_contrast_row_count": int(len(dataset_contrast)),
        "endpoint_instability_row_count": int(len(endpoint_instability)),
        "density_hypothesis_row_count": int(len(density_hypothesis)),
        "current_dense_graph_max_avg_degree": float(
            current_dense["avg_degree_if_undirected"].max()
        ),
        "current_dense_graph_max_undirected_density": float(
            current_dense["undirected_density"].max()
        ),
        "prior_edge_density_directly_available": bool(
            network_density[
                network_density["data_family"].eq("prior_sciscape_cluster_graph_summary")
            ]["edge_count"].notna().any()
        ),
        "density_hypothesis_status": "plausible_contributor_not_proven_cause",
        "claim_boundary": CLAIM_BOUNDARY,
        "outputs": {
            "dataset_contrast_csv": _rel(output_dir / DATASET_CONTRAST_CSV),
            "network_density_csv": _rel(output_dir / NETWORK_DENSITY_CSV),
            "endpoint_instability_csv": _rel(output_dir / ENDPOINT_INSTABILITY_CSV),
            "density_hypothesis_csv": _rel(output_dir / DENSITY_HYPOTHESIS_CSV),
            "summary_json": _rel(output_dir / SUMMARY_JSON),
            "report_md": _rel(output_dir / REPORT_MD),
            "config_json": _rel(output_dir / CONFIG_JSON),
        },
    }
    config = {
        "script": _rel(Path(__file__)),
        "nano_root": str(nano_root),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-root", type=Path, default=DEFAULT_NANO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        nano_root=args.nano_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
