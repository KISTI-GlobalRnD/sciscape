"""Individual Plotly chart functions for programmatic use."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._data_prep import _build_cluster_labels, _parse_json_col

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _check_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "Visualization requires plotly. Install with: pip install plotly"
        )


def plot_cluster_keywords(df: pd.DataFrame, top_n: int = 15) -> "go.Figure":
    """Horizontal bar chart of top-N keywords per cluster."""
    _check_plotly()
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    clusters = sorted(df["cluster_id"].unique())
    labels = _build_cluster_labels(df)
    fig = make_subplots(rows=len(clusters), cols=1,
                        subplot_titles=[f"C{c}: {labels[c]}" for c in clusters],
                        vertical_spacing=0.08 / max(1, len(clusters)))
    colors = px.colors.qualitative.Plotly
    for idx, cid in enumerate(clusters):
        grp = df[df["cluster_id"] == cid].nlargest(top_n, "score").sort_values("score")
        fig.add_trace(go.Bar(y=grp["term"], x=grp["score"], orientation="h",
                             marker_color=colors[idx % len(colors)], showlegend=False,
                             hovertext=[f"<b>{t}</b><br>Score: {s:.4f}<br>Freq: {f:,}"
                                        for t, s, f in zip(grp["term"], grp["score"], grp["frequency"])],
                             hoverinfo="text"), row=idx + 1, col=1)
    fig.update_layout(title="Top Keywords per Cluster", height=350 * len(clusters), template="plotly_white")
    return fig


def plot_score_distribution(df: pd.DataFrame) -> "go.Figure":
    """Box plot of score distributions across clusters."""
    _check_plotly()
    import plotly.graph_objects as go

    labels = _build_cluster_labels(df)
    fig = go.Figure()
    for cid in sorted(df["cluster_id"].unique()):
        subset = df[df["cluster_id"] == cid]
        fig.add_trace(go.Box(y=subset["score"], name=f"C{cid}: {labels[cid]}",
                             boxpoints="all", jitter=0.4, hovertext=subset["term"], hoverinfo="y+text"))
    fig.update_layout(title="Score Distribution", yaxis_title="Score", height=500, template="plotly_white")
    return fig


def plot_temporal_trends(df: pd.DataFrame, metric: str = "pub_year_series", top_n: int = 10) -> "go.Figure":
    """Temporal trend lines per cluster."""
    _check_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    clusters = sorted(df["cluster_id"].unique())
    labels = _build_cluster_labels(df)
    fig = make_subplots(rows=len(clusters), cols=1,
                        subplot_titles=[f"C{c}: {labels[c]}" for c in clusters],
                        vertical_spacing=0.06 / max(1, len(clusters)))
    for idx, cid in enumerate(clusters):
        grp = df[df["cluster_id"] == cid].nlargest(top_n, "score").copy()
        series = _parse_json_col(grp[metric])
        for ti, (_, row) in enumerate(grp.iterrows()):
            d = series.loc[row.name]
            if not d:
                continue
            years = sorted(int(k) for k in d.keys())
            vals = [d.get(str(y), d.get(y, 0)) for y in years]
            fig.add_trace(go.Scatter(x=years, y=vals, mode="lines+markers", name=row["term"],
                                     visible=True if ti < 5 else "legendonly",
                                     legendgroup=f"c{cid}"), row=idx + 1, col=1)
    fig.update_layout(title="Temporal Trends", height=450 * len(clusters), template="plotly_white")
    return fig


def plot_depth_distribution(df: pd.DataFrame) -> "go.Figure":
    """Stacked bar of depth levels per cluster."""
    _check_plotly()
    import plotly.graph_objects as go

    if "depth_level" not in df.columns:
        raise ValueError("depth_level column not found")
    labels = _build_cluster_labels(df)
    clusters = sorted(df["cluster_id"].unique())
    depth_names = {0: "Broad", 1: "Mid", 2: "Specific"}
    depth_colors = {0: "#636EFA", 1: "#EF553B", 2: "#00CC96"}
    fig = go.Figure()
    for lvl in sorted(df["depth_level"].unique()):
        counts = [len(df[(df["cluster_id"] == c) & (df["depth_level"] == lvl)]) for c in clusters]
        fig.add_trace(go.Bar(x=[f"C{c}: {labels[c]}" for c in clusters], y=counts,
                             name=depth_names.get(lvl, f"L{lvl}"),
                             marker_color=depth_colors.get(lvl, "#AB63FA"), text=counts, textposition="inside"))
    fig.update_layout(title="Depth Distribution", barmode="stack", height=450, template="plotly_white")
    return fig


def plot_cross_cluster_terms(df: pd.DataFrame, min_clusters: int = 2) -> "go.Figure":
    """Heatmap of terms shared across clusters."""
    _check_plotly()
    import plotly.graph_objects as go

    labels = _build_cluster_labels(df)
    clusters = sorted(df["cluster_id"].unique())
    tc = df.groupby("term")["cluster_id"].nunique()
    shared = tc[tc >= min_clusters].index.tolist()
    if not shared:
        fig = go.Figure()
        fig.add_annotation(text="No shared terms", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    order = (df[df["term"].isin(shared)].groupby("term").agg(n=("cluster_id", "nunique"), ms=("score", "max"))
             .sort_values(["n", "ms"], ascending=[False, False]).index.tolist())
    mat = np.zeros((len(order), len(clusters)))
    for i, t in enumerate(order):
        for j, c in enumerate(clusters):
            r = df[(df["term"] == t) & (df["cluster_id"] == c)]
            if not r.empty:
                mat[i, j] = r["score"].iloc[0]
    fig = go.Figure(go.Heatmap(z=mat, x=[f"C{c}" for c in clusters], y=order, colorscale="YlOrRd"))
    fig.update_layout(title="Cross-Cluster Terms", height=max(400, len(order) * 25 + 150),
                      yaxis=dict(autorange="reversed"), template="plotly_white")
    return fig
