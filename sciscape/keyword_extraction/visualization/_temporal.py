"""Temporal comparison visualization.

Provides interactive time-series charts for comparing keyword trends
across clusters — enabling users to identify emerging, declining, and
stable research topics over time.

Usage::

    from sciscape.keyword_extraction.visualization import (
        plot_temporal_heatmap,
        plot_cluster_trend_comparison,
    )
    fig = plot_temporal_heatmap(keywords_df)
    fig.show()
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _check_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "Visualization requires plotly. Install with: pip install plotly"
        )


def _parse_year_series(series: pd.Series) -> pd.Series:
    """Parse year series column (JSON strings or dicts)."""
    def _parse_one(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
    return series.apply(_parse_one)


def _extract_year_matrix(
    df: pd.DataFrame,
    metric: str = "pub_year_series",
    top_n: int = 20,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Build a terms × years matrix from the pipeline output.

    Returns (matrix_df, term_list, year_list).
    """
    if metric not in df.columns:
        return pd.DataFrame(), [], []

    # Select top terms by score
    top_df = df.nlargest(top_n, "score").copy()
    parsed = _parse_year_series(top_df[metric])

    # Collect all years
    all_years: set[int] = set()
    for d in parsed:
        all_years.update(int(k) for k in d.keys() if str(k).isdigit())

    if not all_years:
        return pd.DataFrame(), [], []

    years = sorted(all_years)
    terms = top_df["term"].tolist()

    mat = np.zeros((len(terms), len(years)))
    for i, d in enumerate(parsed):
        for j, y in enumerate(years):
            mat[i, j] = d.get(str(y), d.get(y, 0))

    matrix_df = pd.DataFrame(mat, index=terms, columns=years)
    return matrix_df, terms, years


def plot_temporal_heatmap(
    df: pd.DataFrame,
    metric: str = "pub_year_series",
    top_n: int = 30,
    title: str = "Keyword Temporal Heatmap",
    cluster_id: Optional[int] = None,
    normalize_rows: bool = False,
) -> "go.Figure":
    """Heatmap showing keyword frequency over years.

    Parameters
    ----------
    df : DataFrame
        Pipeline output with term, score, and temporal series columns.
    metric : str
        Temporal metric column name.
    top_n : int
        Number of top keywords to display.
    title : str
        Chart title.
    cluster_id : int, optional
        Filter to a specific cluster. If None, uses all clusters.
    normalize_rows : bool
        If True, normalize each row to [0, 1] range.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    import plotly.graph_objects as go

    subset = df[df["cluster_id"] == cluster_id] if cluster_id is not None else df
    matrix_df, terms, years = _extract_year_matrix(subset, metric=metric, top_n=top_n)

    if matrix_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    mat = matrix_df.values.copy()
    if normalize_rows:
        row_max = mat.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        mat = mat / row_max

    fig = go.Figure(go.Heatmap(
        z=mat,
        x=[str(y) for y in years],
        y=terms,
        colorscale="YlOrRd",
        hovertemplate="<b>%{y}</b><br>Year: %{x}<br>Value: %{z:.1f}<extra></extra>",
    ))

    cluster_label = f" (C{cluster_id})" if cluster_id is not None else ""
    fig.update_layout(
        title=f"{title}{cluster_label}",
        xaxis_title="Year",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(terms) * 25 + 150),
        template="plotly_white",
        margin=dict(l=150, r=40, t=60, b=60),
    )

    return fig


def plot_cluster_trend_comparison(
    df: pd.DataFrame,
    metric: str = "pub_year_series",
    top_n_per_cluster: int = 5,
    title: str = "Cluster Trend Comparison",
    aggregate: str = "sum",
) -> "go.Figure":
    """Compare temporal trends across clusters.

    Aggregates keyword time series per cluster and plots them as lines,
    making it easy to see which research areas are growing or declining.

    Parameters
    ----------
    df : DataFrame
        Pipeline output with cluster_id, term, score, and temporal columns.
    metric : str
        Temporal metric column.
    top_n_per_cluster : int
        Use top N keywords per cluster for aggregation.
    title : str
        Chart title.
    aggregate : str
        Aggregation method: ``"sum"`` or ``"mean"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    import plotly.graph_objects as go

    if metric not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    fig = go.Figure()
    clusters = sorted(df["cluster_id"].unique())

    for idx, cid in enumerate(clusters):
        cgrp = df[df["cluster_id"] == cid].nlargest(top_n_per_cluster, "score")
        parsed = _parse_year_series(cgrp[metric])

        # Aggregate per year
        year_totals: dict[int, list[float]] = {}
        for d in parsed:
            for k, v in d.items():
                y = int(k) if str(k).isdigit() else None
                if y is not None:
                    year_totals.setdefault(y, []).append(float(v))

        if not year_totals:
            continue

        years = sorted(year_totals.keys())
        if aggregate == "mean":
            values = [np.mean(year_totals[y]) for y in years]
        else:
            values = [sum(year_totals[y]) for y in years]

        # Cluster label from top keywords
        top3 = cgrp.nlargest(3, "score")["term"].tolist()
        label = f"C{cid}: {', '.join(top3[:3])}"

        fig.add_trace(go.Scatter(
            x=years, y=values,
            mode="lines+markers",
            name=label[:50],
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>C{cid}</b><br>Year: %{{x}}<br>Value: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Document Count" if aggregate == "sum" else "Average Count",
        height=550,
        template="plotly_white",
        legend=dict(font=dict(size=9)),
        hovermode="x unified",
    )

    return fig


__all__ = [
    "plot_temporal_heatmap",
    "plot_cluster_trend_comparison",
]
