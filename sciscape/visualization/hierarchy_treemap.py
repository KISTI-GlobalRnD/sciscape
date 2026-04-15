"""Hierarchical treemap and sunburst visualization.

Renders nano → micro → meso → macro hierarchy as interactive
Plotly treemap or sunburst, with cluster sizes and labels.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def build_treemap_data(
    hierarchy_df: pl.DataFrame,
    level_labels: Dict[str, pl.DataFrame] | None = None,
    *,
    levels: Sequence[str] = ("macro", "meso", "micro", "nano"),
    value_col: str = "count",
) -> Dict[str, Any]:
    """Build treemap data from hierarchy membership + labels.

    Parameters
    ----------
    hierarchy_df : pl.DataFrame
        uid + cluster_{level} columns.
    level_labels : dict, optional
        {level: DataFrame(cluster, label)} from label_pipeline.
    levels : sequence
        Level order from coarsest to finest (treemap nesting order).

    Returns
    -------
    dict with ids, labels, parents, values for Plotly treemap.
    """
    present = [l for l in levels if f"cluster_{l}" in hierarchy_df.columns]
    if not present:
        return {"error": "no hierarchy levels found"}

    # Build label lookup
    label_map: Dict[str, Dict[int, str]] = {}
    for level in present:
        if level_labels and level in level_labels:
            ldf = level_labels[level]
            label_map[level] = dict(zip(
                ldf["cluster"].to_list(),
                ldf["label"].to_list(),
            ))
        else:
            label_map[level] = {}

    # Count papers per node at each level
    # Node ID format: "level:cluster_id"
    ids = [""]  # root
    labels = ["All Papers"]
    parents = [""]
    values = [0]  # will be sum

    # Build parent-child relationships
    # For each adjacent pair of levels: child_cluster → parent_cluster (majority)
    parent_mapping: Dict[str, Dict[int, int]] = {}  # level → {child_cid: parent_cid}
    for i in range(len(present) - 1):
        child_level = present[i + 1]  # finer
        parent_level = present[i]     # coarser
        child_col = f"cluster_{child_level}"
        parent_col = f"cluster_{parent_level}"

        mapping = {}
        groups = hierarchy_df.select(child_col, parent_col).group_by(child_col).agg(
            pl.col(parent_col).mode().first().alias("parent")
        )
        for row in groups.iter_rows(named=True):
            mapping[int(row[child_col])] = int(row["parent"])
        parent_mapping[child_level] = mapping

    # Add nodes level by level (coarsest first)
    total_papers = hierarchy_df.height

    for level_idx, level in enumerate(present):
        col = f"cluster_{level}"
        cluster_sizes = Counter(hierarchy_df[col].to_list())

        for cid in sorted(cluster_sizes.keys()):
            size = cluster_sizes[cid]
            node_id = f"{level}:{cid}"
            label = label_map.get(level, {}).get(int(cid), f"{level} C{cid}")
            # Truncate long labels
            if len(label) > 50:
                label = label[:47] + "..."

            # Parent
            if level_idx == 0:
                parent_id = ""  # root
            else:
                parent_level = present[level_idx - 1]
                parent_cid = parent_mapping.get(level, {}).get(int(cid), 0)
                parent_id = f"{parent_level}:{parent_cid}"

            ids.append(node_id)
            labels.append(f"{label}<br><sub>{size:,} papers</sub>")
            parents.append(parent_id)
            values.append(size)

    # Root value
    values[0] = total_papers

    return {
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": values,
        "levels": present,
        "total": total_papers,
    }


def treemap_to_plotly(data: Dict[str, Any], *, mode: str = "treemap") -> Dict[str, Any]:
    """Convert treemap data to Plotly figure JSON.

    Parameters
    ----------
    mode : str
        "treemap" or "sunburst".
    """
    if "error" in data:
        return data

    trace = {
        "type": mode,
        "ids": data["ids"],
        "labels": data["labels"],
        "parents": data["parents"],
        "values": data["values"],
        "branchvalues": "total",
        "textinfo": "label",
        "hovertemplate": "%{label}<extra></extra>",
    }

    if mode == "treemap":
        trace["tiling"] = {"packing": "squarify"}
        trace["marker"] = {"line": {"width": 1, "color": "white"}}

    return {
        "data": [trace],
        "layout": {
            "title": f"Cluster Hierarchy ({' → '.join(data['levels'])})",
            "height": 700,
            "margin": {"t": 40, "l": 0, "r": 0, "b": 0},
        },
    }


def save_treemap_html(
    data: Dict[str, Any],
    path: str,
    *,
    mode: str = "treemap",
    title: str = "SciScape Hierarchy",
) -> None:
    """Save interactive treemap as standalone HTML."""
    import json

    fig = treemap_to_plotly(data, mode=mode)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head><body>
<div id="chart" style="width:100%;height:700px;"></div>
<script>
var data = {json.dumps(fig["data"])};
var layout = {json.dumps(fig["layout"])};
Plotly.newPlot('chart', data, layout);
</script>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Saved %s → %s", mode, path)


def hierarchy_summary_table(
    data: Dict[str, Any],
    hierarchy_df: pl.DataFrame,
    level_labels: Dict[str, pl.DataFrame] | None = None,
) -> str:
    """Text summary of hierarchy structure."""
    present = data.get("levels", [])
    lines = ["=" * 60, "Hierarchy Summary", "=" * 60]

    for level in present:
        col = f"cluster_{level}"
        sizes = Counter(hierarchy_df[col].to_list())
        n_cl = len(sizes)
        avg = data["total"] // n_cl if n_cl else 0
        mx = max(sizes.values()) if sizes else 0

        lines.append(f"\n--- {level} ({n_cl} clusters, avg {avg:,}) ---")

        # Top clusters with labels
        top = sorted(sizes.items(), key=lambda x: -x[1])[:10]
        for cid, size in top:
            label = ""
            if level_labels and level in level_labels:
                ldf = level_labels[level]
                match = ldf.filter(pl.col("cluster") == cid)
                if match.height:
                    label = match["label"][0]
            if len(label) > 60:
                label = label[:57] + "..."
            lines.append(f"  C{cid}: {size:>6,} papers  {label}")

    return "\n".join(lines)


__all__ = [
    "build_treemap_data",
    "treemap_to_plotly",
    "save_treemap_html",
    "hierarchy_summary_table",
]
