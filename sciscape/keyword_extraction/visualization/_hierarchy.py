"""Hierarchical cluster visualization.

Generates interactive treemap and sunburst views of cluster → depth-level →
keyword relationships.  These drill-down charts expose the broad / mid /
specific keyword stratification that SciScape computes (not available in
VOSViewer).

Usage::

    from sciscape.keyword_extraction.visualization import (
        plot_cluster_treemap,
        plot_cluster_sunburst,
    )
    fig = plot_cluster_sunburst(keywords_df)
    fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ._data_prep import _keyword_label_col, _keyword_score_col

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _check_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "Visualization requires plotly. Install with: pip install plotly"
        )


_DEPTH_NAMES = {0: "Broad", 1: "Mid", 2: "Specific"}
_DEPTH_COLORS = {0: "#636EFA", 1: "#EF553B", 2: "#00CC96"}

_CLUSTER_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _build_hierarchy_df(
    df: pd.DataFrame,
    top_n_per_depth: int = 10,
) -> pd.DataFrame:
    """Build a flat hierarchy table: cluster → depth_level → term."""
    if "depth_level" not in df.columns:
        # Assign all to depth 0 if depth stage was not run
        df = df.copy()
        df["depth_level"] = 0

    rows: list[dict] = []
    label_col = _keyword_label_col(df)
    score_col = _keyword_score_col(df)
    for cid in sorted(df["cluster_id"].unique()):
        cgrp = df[df["cluster_id"] == cid]
        # Top terms per cluster (cluster label)
        top3 = cgrp.nlargest(3, score_col)[label_col].astype(str).tolist()
        cluster_label = f"C{cid}: {', '.join(top3)}"

        for dlvl in sorted(cgrp["depth_level"].dropna().unique()):
            dlvl = int(dlvl)
            dgrp = cgrp[cgrp["depth_level"] == dlvl]
            depth_label = _DEPTH_NAMES.get(dlvl, f"L{dlvl}")

            for _, row in dgrp.nlargest(top_n_per_depth, score_col).iterrows():
                rows.append({
                    "cluster": cluster_label,
                    "cluster_id": int(cid),
                    "depth": depth_label,
                    "depth_level": dlvl,
                    "term": row[label_col],
                    "score": row[score_col],
                    "frequency": row.get("frequency", 1),
                })

    return pd.DataFrame(rows)


def plot_cluster_treemap(
    df: pd.DataFrame,
    top_n_per_depth: int = 10,
    title: str = "Cluster Keyword Treemap",
    color_by: str = "depth",
) -> "go.Figure":
    """Interactive treemap: cluster → depth level → keyword.

    Parameters
    ----------
    df : DataFrame
        Pipeline output with cluster_id, term, score, depth_level columns.
    top_n_per_depth : int
        Max keywords per depth level per cluster.
    title : str
        Chart title.
    color_by : str
        ``"depth"`` colors by depth level; ``"cluster"`` colors by cluster.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    import plotly.express as px

    hdf = _build_hierarchy_df(df, top_n_per_depth=top_n_per_depth)

    if hdf.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    color_col = "depth" if color_by == "depth" else "cluster"
    color_map = None
    if color_by == "depth":
        color_map = {v: _DEPTH_COLORS.get(k, "#AB63FA") for k, v in _DEPTH_NAMES.items()}

    fig = px.treemap(
        hdf,
        path=["cluster", "depth", "term"],
        values="score",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={"score": ":.4f", "frequency": True},
        title=title,
    )

    fig.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_traces(
        textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>Score: %{customdata[0]:.4f}<br>Freq: %{customdata[1]}<extra></extra>",
    )

    return fig


def plot_cluster_sunburst(
    df: pd.DataFrame,
    top_n_per_depth: int = 10,
    title: str = "Cluster Keyword Sunburst",
    color_by: str = "depth",
) -> "go.Figure":
    """Interactive sunburst: cluster → depth level → keyword.

    Parameters
    ----------
    df : DataFrame
        Pipeline output with cluster_id, term, score, depth_level columns.
    top_n_per_depth : int
        Max keywords per depth level per cluster.
    title : str
        Chart title.
    color_by : str
        ``"depth"`` or ``"cluster"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    import plotly.express as px

    hdf = _build_hierarchy_df(df, top_n_per_depth=top_n_per_depth)

    if hdf.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text="No data to display",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    color_col = "depth" if color_by == "depth" else "cluster"
    color_map = None
    if color_by == "depth":
        color_map = {v: _DEPTH_COLORS.get(k, "#AB63FA") for k, v in _DEPTH_NAMES.items()}

    fig = px.sunburst(
        hdf,
        path=["cluster", "depth", "term"],
        values="score",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={"score": ":.4f", "frequency": True},
        title=title,
    )

    fig.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


__all__ = [
    "plot_cluster_treemap",
    "plot_cluster_sunburst",
]
