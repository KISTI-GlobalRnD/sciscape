"""Edge landscape: 2D year×year histograms per layer with weight analysis.

Shows how connections distribute across publication years for each layer,
revealing temporal patterns unique to each edge type:
  DC: triangular (recent → past citations)
  BC: diagonal band (same-era shared references)
  CC: broad spread (classic papers co-cited across eras)
  Emb: tight diagonal (textually similar = same era)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def compute_edge_year_matrix(
    edges: pl.DataFrame,
    year_map: Dict[str, int],
    *,
    year_range: Tuple[int, int] | None = None,
    weight_col: str = "rel_sum2",
) -> Dict[str, Any]:
    """Compute 2D year×year histogram of edges.

    Returns dict with:
    - count_matrix: (n_years × n_years) edge count
    - weight_matrix: (n_years × n_years) average weight
    - weight_sum_matrix: (n_years × n_years) total weight
    - years: list of year labels
    - marginal_x: count per year (uid1)
    - marginal_y: count per year (uid2)
    """
    years_present = sorted(set(y for y in year_map.values() if y and y > 0))
    if not years_present:
        return {"error": "no year data"}

    if year_range:
        yr_min, yr_max = year_range
    else:
        yr_min = min(years_present)
        yr_max = max(years_present)

    years = list(range(yr_min, yr_max + 1))
    n = len(years)
    yr_to_idx = {y: i for i, y in enumerate(years)}

    count_mat = np.zeros((n, n), dtype=np.int64)
    weight_sum = np.zeros((n, n), dtype=np.float64)

    for row in edges.iter_rows(named=True):
        u1, u2 = row["uid1"], row["uid2"]
        y1, y2 = year_map.get(u1), year_map.get(u2)
        if y1 is None or y2 is None or y1 < yr_min or y2 < yr_min:
            continue
        if y1 > yr_max or y2 > yr_max:
            continue
        i, j = yr_to_idx.get(y1), yr_to_idx.get(y2)
        if i is None or j is None:
            continue
        w = float(row.get(weight_col, 1.0))
        # Symmetric: count both directions
        count_mat[i, j] += 1
        count_mat[j, i] += 1
        weight_sum[i, j] += w
        weight_sum[j, i] += w

    # Average weight (avoid div by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_avg = np.where(count_mat > 0, weight_sum / count_mat, 0.0)

    return {
        "count_matrix": count_mat.tolist(),
        "weight_sum_matrix": weight_sum.tolist(),
        "weight_avg_matrix": weight_avg.tolist(),
        "years": years,
        "marginal_x": count_mat.sum(axis=1).tolist(),
        "marginal_y": count_mat.sum(axis=0).tolist(),
        "total_edges": int(count_mat.sum() // 2),
    }


def compute_multilayer_year_matrices(
    layers: Dict[str, pl.DataFrame],
    year_map: Dict[str, int],
    *,
    year_range: Tuple[int, int] | None = None,
    top_k: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Compute year×year matrices for multiple layers.

    Returns {layer_name: matrix_data}.
    """
    from ..linkage.filters import filter_top_k

    results = {}
    for name, df in layers.items():
        if df.height == 0:
            continue
        if top_k > 0:
            df = filter_top_k(df, top_k)
        results[name] = compute_edge_year_matrix(df, year_map, year_range=year_range)
        log.info("edge_year_matrix %s: %d edges", name, results[name]["total_edges"])

    return results


def compute_weight_quantile_maps(
    edges: pl.DataFrame,
    year_map: Dict[str, int],
    *,
    quantiles: Sequence[float] = (0.5, 0.9, 0.99),
    year_range: Tuple[int, int] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """2D year maps for different weight quantiles.

    For each quantile threshold:
      "Where do the top X% strongest edges connect?"

    Returns {quantile_label: matrix_data}.
    """
    w = edges["rel_sum2"].to_numpy()
    results = {}

    for q in quantiles:
        threshold = np.quantile(w, q)
        above = edges.filter(pl.col("rel_sum2") >= threshold)
        label = f"top_{int((1-q)*100)}pct"
        results[label] = compute_edge_year_matrix(above, year_map, year_range=year_range)
        results[label]["threshold"] = float(threshold)
        results[label]["n_above"] = above.height

    return results


def edge_landscape_to_plotly(
    layer_matrices: Dict[str, Dict[str, Any]],
    *,
    mode: str = "count",
) -> Dict[str, Any]:
    """Generate Plotly figures for edge landscape visualization.

    Parameters
    ----------
    layer_matrices : dict
        {layer_name: matrix_data} from compute_multilayer_year_matrices.
    mode : str
        "count" (edge count), "weight_avg" (average weight), "weight_sum" (total weight).

    Returns dict of Plotly figure JSONs.
    """
    matrix_key = {
        "count": "count_matrix",
        "weight_avg": "weight_avg_matrix",
        "weight_sum": "weight_sum_matrix",
    }[mode]

    figures = {}

    for name, data in layer_matrices.items():
        if "error" in data:
            continue

        mat = np.array(data[matrix_key])
        years = data["years"]
        marginal = data["marginal_x"]

        # Main heatmap
        figures[f"{name}_heatmap"] = {
            "data": [{
                "z": mat.tolist(),
                "x": years,
                "y": years,
                "type": "heatmap",
                "colorscale": "YlOrRd",
                "hoverongaps": False,
            }],
            "layout": {
                "title": f"{name.upper()} — Year × Year ({mode})",
                "xaxis": {"title": "Paper A year"},
                "yaxis": {"title": "Paper B year"},
                "height": 450,
                "width": 500,
            },
        }

        # Marginal histogram
        figures[f"{name}_marginal"] = {
            "data": [{
                "x": years,
                "y": marginal,
                "type": "bar",
                "marker": {"color": "#60a5fa"},
            }],
            "layout": {
                "title": f"{name.upper()} — Edges per year",
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "Edge count"},
                "height": 250,
                "width": 500,
            },
        }

    # Comparison: overlay marginals
    if len(layer_matrices) > 1:
        traces = []
        colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
        for i, (name, data) in enumerate(layer_matrices.items()):
            if "error" in data:
                continue
            traces.append({
                "x": data["years"],
                "y": data["marginal_x"],
                "name": name.upper(),
                "type": "scatter",
                "mode": "lines",
                "line": {"color": colors[i % len(colors)], "width": 2},
            })
        figures["comparison_marginal"] = {
            "data": traces,
            "layout": {
                "title": "Edges per year — All layers",
                "xaxis": {"title": "Year"},
                "yaxis": {"title": "Edge count"},
                "height": 300,
            },
        }

    return figures


def format_edge_landscape_report(
    layer_matrices: Dict[str, Dict[str, Any]],
) -> str:
    """Text summary of edge landscape analysis."""
    lines = ["=" * 60, "Edge Landscape Report (Year × Year)", "=" * 60]

    for name, data in layer_matrices.items():
        if "error" in data:
            continue
        mat = np.array(data["count_matrix"])
        years = data["years"]
        total = data["total_edges"]

        # Temporal spread: how concentrated around diagonal?
        n = len(years)
        diag_band = 0  # edges within ±3 years
        off_diag = 0
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= 3:
                    diag_band += mat[i, j]
                else:
                    off_diag += mat[i, j]
        total_cells = diag_band + off_diag
        diag_pct = 100 * diag_band / total_cells if total_cells else 0

        # Asymmetry: lower triangle (citing past) vs upper (citing future)
        lower = np.tril(mat, -1).sum()
        upper = np.triu(mat, 1).sum()
        asym = lower / (lower + upper) if (lower + upper) else 0.5

        lines.append(f"\n--- {name.upper()} ---")
        lines.append(f"  Total edges: {total:,}")
        lines.append(f"  Year range: {years[0]}–{years[-1]}")
        lines.append(f"  Diagonal band (±3yr): {diag_pct:.1f}%")
        lines.append(f"  Temporal asymmetry: {asym:.2f} (1.0=all past→present, 0.5=symmetric)")

    return "\n".join(lines)


__all__ = [
    "compute_edge_year_matrix",
    "compute_multilayer_year_matrices",
    "compute_weight_quantile_maps",
    "edge_landscape_to_plotly",
    "format_edge_landscape_report",
]
