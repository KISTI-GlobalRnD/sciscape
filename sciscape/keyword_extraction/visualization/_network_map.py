"""Cluster network map visualization.

Generates an interactive 2D map of clusters based on keyword similarity,
with nodes sized by document count and colored by cluster.  This is the
primary "overview" visualization comparable to VOSViewer's density map.

Layout algorithm:
  1. Compute pairwise Jaccard similarity between clusters (shared keywords)
  2. Convert to distance matrix (1 - similarity)
  3. Apply MDS (metric multi-dimensional scaling) for 2D positions
  4. Render with Plotly as a scatter + edge plot

Usage::

    from sciscape.keyword_extraction.visualization import plot_cluster_map
    fig = plot_cluster_map(keywords_df, viz_data=pipeline.get_visualization_data())
    fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
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


def _cluster_keyword_sets(
    df: pd.DataFrame,
) -> Dict[int, set]:
    """Extract keyword sets per cluster."""
    result = {}
    label_col = _keyword_label_col(df)
    for cid, grp in df.groupby("cluster_id"):
        result[int(cid)] = set(grp[label_col].astype(str).tolist())
    return result


def _pairwise_jaccard(
    keyword_sets: Dict[int, set],
) -> Tuple[List[int], np.ndarray]:
    """Compute pairwise Jaccard similarity matrix between clusters."""
    cids = sorted(keyword_sets.keys())
    n = len(cids)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = keyword_sets[cids[i]]
            b = keyword_sets[cids[j]]
            union = a | b
            if union:
                jac = len(a & b) / len(union)
            else:
                jac = 0.0
            sim[i, j] = jac
            sim[j, i] = jac
        sim[i, i] = 1.0
    return cids, sim


def _mds_layout(
    sim: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D positions from similarity matrix using MDS.

    Falls back to random layout if sklearn is not available or MDS fails.
    """
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    try:
        from sklearn.manifold import MDS
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            max_iter=300,
            normalized_stress="auto",
        )
        pos = mds.fit_transform(dist)
    except Exception:
        # Fallback: random layout
        rng = np.random.RandomState(random_state)
        pos = rng.randn(sim.shape[0], 2)
    return pos


def _spring_layout(
    sim: np.ndarray,
    edge_threshold: float = 0.05,
    iterations: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Simple force-directed (Fruchterman-Reingold) layout.

    Used as fallback or alternative to MDS. Nodes repel each other;
    edges (similarity > threshold) attract.
    """
    n = sim.shape[0]
    rng = np.random.RandomState(random_state)
    pos = rng.randn(n, 2)

    k = 1.0 / np.sqrt(max(n, 1))  # optimal distance
    temp = 1.0

    for _ in range(iterations):
        # Repulsive forces (all pairs)
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = pos[i] - pos  # (n, 2)
            dist = np.sqrt((delta ** 2).sum(axis=1))
            dist = np.maximum(dist, 1e-4)
            force = (k ** 2) / dist
            force[i] = 0.0
            disp[i] = (delta * force[:, None]).sum(axis=0)

        # Attractive forces (edges above threshold)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] < edge_threshold:
                    continue
                delta = pos[j] - pos[i]
                dist = max(np.sqrt((delta ** 2).sum()), 1e-4)
                force = (dist ** 2) / k * sim[i, j]
                attract = delta / dist * force
                disp[i] += attract
                disp[j] -= attract

        # Apply displacement with temperature
        disp_mag = np.sqrt((disp ** 2).sum(axis=1))
        disp_mag = np.maximum(disp_mag, 1e-4)
        pos += disp / disp_mag[:, None] * np.minimum(disp_mag, temp)[:, None]
        temp *= 0.95

    # Normalize to [-1, 1]
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    if scale > 0:
        pos /= scale

    return pos


def plot_cluster_map(
    df: pd.DataFrame,
    viz_data: Optional[Dict] = None,
    layout: str = "mds",
    edge_threshold: float = 0.02,
    top_n_labels: int = 3,
    node_scale: float = 30.0,
    title: str = "Cluster Network Map",
) -> "go.Figure":
    """Generate interactive 2D cluster map.

    Parameters
    ----------
    df : DataFrame
        Pipeline output with cluster_id, term, score, frequency columns.
    viz_data : dict, optional
        From ``pipeline.get_visualization_data()``. Used for doc counts
        and cross-cluster metadata.
    layout : str
        Layout algorithm: ``"mds"`` (default) or ``"spring"``.
    edge_threshold : float
        Minimum Jaccard similarity to draw an edge between clusters.
    top_n_labels : int
        Number of top keywords to show in cluster labels.
    node_scale : float
        Base size multiplier for cluster nodes.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    import plotly.graph_objects as go

    # Compute cluster similarity
    keyword_sets = _cluster_keyword_sets(df)
    cids, sim = _pairwise_jaccard(keyword_sets)
    n = len(cids)

    if n == 0:
        fig = go.Figure()
        fig.add_annotation(text="No clusters to display",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Compute layout
    if layout == "spring":
        pos = _spring_layout(sim, edge_threshold=edge_threshold)
    else:
        pos = _mds_layout(sim)

    # Cluster metadata
    cluster_sizes = {}
    cluster_labels = {}
    label_col = _keyword_label_col(df)
    score_col = _keyword_score_col(df)
    for cid in cids:
        grp = df[df["cluster_id"] == cid]
        cluster_sizes[cid] = len(grp)
        top_terms = grp.nlargest(top_n_labels, score_col)[label_col].astype(str).tolist()
        cluster_labels[cid] = ", ".join(top_terms)

    # Use doc count from viz_data if available
    if viz_data and "_pipeline_config" in viz_data:
        pass  # could use n_documents per cluster

    # Node sizes proportional to keyword count (or doc count)
    sizes = np.array([cluster_sizes.get(c, 1) for c in cids], dtype=float)
    sizes = node_scale * (sizes / max(sizes.max(), 1)) ** 0.5
    sizes = np.maximum(sizes, 8)

    # Color palette
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    # Draw edges
    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= edge_threshold:
                edge_x.extend([pos[i, 0], pos[j, 0], None])
                edge_y.extend([pos[i, 1], pos[j, 1], None])

    fig = go.Figure()

    # Edge trace
    if edge_x:
        # Compute edge widths based on similarity
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= edge_threshold:
                    width = max(0.5, sim[i, j] * 5)
                    opacity = min(0.8, sim[i, j] * 2 + 0.1)
                    fig.add_trace(go.Scatter(
                        x=[pos[i, 0], pos[j, 0]],
                        y=[pos[i, 1], pos[j, 1]],
                        mode="lines",
                        line=dict(width=width, color=f"rgba(150,150,150,{opacity})"),
                        hoverinfo="text",
                        hovertext=f"C{cids[i]} ↔ C{cids[j]}: Jaccard={sim[i,j]:.3f}",
                        showlegend=False,
                    ))

    # Node trace
    for idx, cid in enumerate(cids):
        color = colors[idx % len(colors)]
        label = cluster_labels[cid]
        n_kw = cluster_sizes[cid]

        # Hover text with detailed info
        hover = (
            f"<b>Cluster {cid}</b><br>"
            f"Keywords: {n_kw}<br>"
            f"Top: {label}<br>"
        )
        # Add similarity info
        neighbors = []
        for j, other_cid in enumerate(cids):
            if other_cid != cid and sim[idx, j] >= edge_threshold:
                neighbors.append(f"C{other_cid} ({sim[idx, j]:.2f})")
        if neighbors:
            hover += f"Similar: {', '.join(neighbors[:5])}"

        fig.add_trace(go.Scatter(
            x=[pos[idx, 0]],
            y=[pos[idx, 1]],
            mode="markers+text",
            marker=dict(
                size=sizes[idx],
                color=color,
                line=dict(width=1.5, color="white"),
                opacity=0.85,
            ),
            text=f"C{cid}",
            textposition="top center",
            textfont=dict(size=10, color="#333"),
            hovertext=hover,
            hoverinfo="text",
            name=f"C{cid}: {label[:40]}",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        legend=dict(
            font=dict(size=9),
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="",
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        hovermode="closest",
        height=700,
        width=900,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_cluster_map_with_keywords(
    df: pd.DataFrame,
    viz_data: Optional[Dict] = None,
    layout: str = "mds",
    edge_threshold: float = 0.02,
    top_n_keywords: int = 5,
    title: str = "Cluster Keyword Map",
) -> "go.Figure":
    """Extended cluster map with keyword annotations.

    Same as ``plot_cluster_map`` but adds keyword labels around each
    cluster node for a denser, VOSViewer-like information display.
    """
    _check_plotly()
    import plotly.graph_objects as go

    keyword_sets = _cluster_keyword_sets(df)
    cids, sim = _pairwise_jaccard(keyword_sets)
    n = len(cids)

    if n == 0:
        fig = go.Figure()
        fig.add_annotation(text="No clusters", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    if layout == "spring":
        pos = _spring_layout(sim, edge_threshold=edge_threshold)
    else:
        pos = _mds_layout(sim)

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    fig = go.Figure()

    # Edges (lighter)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= edge_threshold:
                width = max(0.3, sim[i, j] * 4)
                fig.add_trace(go.Scatter(
                    x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]],
                    mode="lines",
                    line=dict(width=width, color="rgba(200,200,200,0.5)"),
                    hoverinfo="skip", showlegend=False,
                ))

    # Cluster nodes + keyword annotations
    for idx, cid in enumerate(cids):
        color = colors[idx % len(colors)]
        grp = df[df["cluster_id"] == cid]
        label_col = _keyword_label_col(df)
        score_col = _keyword_score_col(df)
        top_kw = grp.nlargest(top_n_keywords, score_col)
        n_kw = len(grp)

        # Main node
        fig.add_trace(go.Scatter(
            x=[pos[idx, 0]], y=[pos[idx, 1]],
            mode="markers",
            marker=dict(size=max(20, 10 + n_kw * 0.3), color=color,
                        line=dict(width=2, color="white"), opacity=0.7),
            hovertext=f"<b>C{cid}</b> ({n_kw} keywords)",
            hoverinfo="text",
            name=f"C{cid}",
        ))

        # Keyword labels in a small arc around the node
        for ki, (_, row) in enumerate(top_kw.iterrows()):
            angle = (2 * np.pi * ki / max(top_n_keywords, 1)) - np.pi / 2
            r = 0.08 + 0.02 * ki
            kx = pos[idx, 0] + r * np.cos(angle)
            ky = pos[idx, 1] + r * np.sin(angle)

            # Font size proportional to score
            fsize = max(7, min(11, 7 + row[score_col] * 20))
            fig.add_annotation(
                x=kx, y=ky, text=row[label_col],
                showarrow=False,
                font=dict(size=fsize, color=color),
                opacity=0.9,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        legend=dict(font=dict(size=9)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        hovermode="closest",
        height=800,
        width=1000,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


__all__ = [
    "plot_cluster_map",
    "plot_cluster_map_with_keywords",
]
