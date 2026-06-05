"""Dashboard export — assembles data + template into HTML."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ._data_prep import prepare_cluster_data
from ._dashboard_template import _DASHBOARD_HTML_TEMPLATE


def _file_export_row(path: str | Path, *, root: Path, file_id: str, role: str, fmt: str = "html") -> dict:
    candidate = Path(path).resolve()
    try:
        rel_path = candidate.relative_to(root).as_posix()
    except ValueError:
        rel_path = candidate.as_posix()
    return {
        "file_id": file_id,
        "path": rel_path,
        "role": role,
        "format": fmt,
        "public_share_state": "local",
    }


def export_dashboard(
    df: pd.DataFrame,
    output_path: str = "keyword_dashboard.html",
    title: str = "SciScape Keyword Explorer",
    open_browser: bool = False,
    viz_data: Optional[Dict] = None,
    write_manifest: bool = True,
) -> str:
    """Generate a self-contained interactive HTML dashboard.

    Parameters
    ----------
    df : DataFrame
        Pipeline output from ``run_keyword_pipeline``.
    output_path : str
        Path for the HTML file.
    title : str
        Dashboard title.
    open_browser : bool
        Whether to open the file in a browser after generation.
    viz_data : dict, optional
        Supplementary visualization data from
        ``KeywordExtractionPipeline.get_visualization_data()``.
    write_manifest : bool
        Whether to write a manifest-backed export artifact next to the dashboard
        result root.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    from sciscape.artifacts import build_report_data_contract

    cluster_data = prepare_cluster_data(df, viz_data=viz_data)
    cluster_data["_sciscape"] = build_report_data_contract(cluster_data)
    data_json = json.dumps(cluster_data, ensure_ascii=False)

    html = _DASHBOARD_HTML_TEMPLATE
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{N_KEYWORDS}}", str(len(df)))
    html = html.replace("{{N_CLUSTERS}}", str(df["cluster_id"].nunique()))
    html = html.replace("{{DATA_JSON}}", data_json)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    if write_manifest:
        from sciscape.artifacts import write_export_manifest

        root = Path(abs_path).parent
        write_export_manifest(
            root,
            export_id="keyword_dashboard",
            export_family="viewer",
            export_kind="keyword_dashboard_html",
            primary_file=abs_path,
            source_artifacts=[
                {
                    "role": "embedded_report_data",
                    "artifact_ref": "embedded_report_data",
                    "path": "",
                    "feature_ref": "keyword",
                    "required": False,
                }
            ],
            feature_refs=["overview", "keyword", "cluster_map", "export"],
            compatibility={"target_tools": ["Browser", "SciScape"], "limitations": ["self-contained HTML"]},
            title=title,
        )

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")

    return abs_path


def export_report(
    df: pd.DataFrame,
    output_dir: str = "workspace/reports/sciscape_report",
    title: str = "SciScape Keyword Report",
    viz_data: Optional[Dict] = None,
    open_browser: bool = False,
    write_manifest: bool = True,
) -> List[str]:
    """Generate a full report with dashboard + Plotly visualization pages.

    Creates a directory with:
    - ``index.html`` — main keyword dashboard
    - ``network_map.html`` — cluster network map (MDS layout)
    - ``hierarchy.html`` — sunburst/treemap drill-down
    - ``temporal.html`` — temporal comparison charts

    Parameters
    ----------
    df : DataFrame
        Pipeline output from ``run_keyword_pipeline``.
    output_dir : str
        Directory for the report files.
    title : str
        Report title.
    viz_data : dict, optional
        Supplementary visualization data.
    open_browser : bool
        Whether to open the index page in a browser.
    write_manifest : bool
        Whether to write a manifest-backed report export artifact.

    Returns
    -------
    list[str]
        Absolute paths to all generated files.
    """
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "export_report requires plotly. Install with: pip install plotly"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    # 0. Save data.json (for external viewer / Vercel deploy)
    from sciscape.artifacts import build_report_data_contract

    cluster_data = prepare_cluster_data(df, viz_data=viz_data)
    cluster_data["_sciscape"] = build_report_data_contract(cluster_data)
    data_json_path = str(out / "data.json")
    with open(data_json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False)
    generated.append(os.path.abspath(data_json_path))

    # 1. Main dashboard
    dash_path = export_dashboard(
        df, output_path=str(out / "index.html"),
        title=title, viz_data=viz_data, write_manifest=False,
    )
    generated.append(dash_path)

    # 2. Network map
    from ._network_map import plot_cluster_map
    fig = plot_cluster_map(df, viz_data=viz_data, title=f"{title} — Cluster Map")
    net_path = str(out / "network_map.html")
    fig.write_html(net_path, include_plotlyjs="cdn")
    generated.append(os.path.abspath(net_path))

    # 3. Hierarchy (sunburst)
    from ._hierarchy import plot_cluster_sunburst, plot_cluster_treemap
    fig_sun = plot_cluster_sunburst(df, title=f"{title} — Sunburst")
    sun_path = str(out / "hierarchy_sunburst.html")
    fig_sun.write_html(sun_path, include_plotlyjs="cdn")
    generated.append(os.path.abspath(sun_path))

    fig_tree = plot_cluster_treemap(df, title=f"{title} — Treemap")
    tree_path = str(out / "hierarchy_treemap.html")
    fig_tree.write_html(tree_path, include_plotlyjs="cdn")
    generated.append(os.path.abspath(tree_path))

    # 4. Temporal
    from ._temporal import plot_temporal_heatmap, plot_cluster_trend_comparison
    if "pub_year_series" in df.columns:
        fig_heat = plot_temporal_heatmap(df, title=f"{title} — Temporal Heatmap")
        heat_path = str(out / "temporal_heatmap.html")
        fig_heat.write_html(heat_path, include_plotlyjs="cdn")
        generated.append(os.path.abspath(heat_path))

        fig_trend = plot_cluster_trend_comparison(df, title=f"{title} — Trends")
        trend_path = str(out / "temporal_trends.html")
        fig_trend.write_html(trend_path, include_plotlyjs="cdn")
        generated.append(os.path.abspath(trend_path))

    # 5. Index page linking all
    nav_links = "\n".join(
        f'    <li><a href="{Path(p).name}">{Path(p).stem.replace("_", " ").title()}</a></li>'
        for p in generated[1:]  # skip index itself
    )
    nav_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title} — Report Index</title>
<style>body{{font-family:sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;}}
a{{color:#0d6efd;text-decoration:none;}} a:hover{{text-decoration:underline;}}
li{{margin:.5rem 0;font-size:1.1rem;}}</style></head>
<body><h1>{title}</h1>
<p>{len(df)} keywords across {df["cluster_id"].nunique()} clusters</p>
<h2>Report Pages</h2>
<ul>
    <li><a href="index.html"><b>Interactive Dashboard</b></a></li>
{nav_links}
</ul>
<p style="color:#6c757d;margin-top:2rem;font-size:.85rem;">Generated by SciScape</p>
</body></html>"""
    nav_path = str(out / "report.html")
    with open(nav_path, "w", encoding="utf-8") as f:
        f.write(nav_html)
    generated.append(os.path.abspath(nav_path))

    if write_manifest:
        from sciscape.artifacts import write_export_manifest

        root = out.parent.resolve()
        files = [
            _file_export_row(path, root=root, file_id=Path(path).stem or f"file_{idx}", role="primary" if Path(path).name == "report.html" else "support", fmt=Path(path).suffix.lstrip(".") or "file")
            for idx, path in enumerate(generated, start=1)
        ]
        report_data_rel = _file_export_row(data_json_path, root=root, file_id="report_data", role="report_data", fmt="json")["path"]
        write_export_manifest(
            root,
            export_id="html_report",
            export_family="report",
            export_kind="html_report",
            primary_file=nav_path,
            source_artifacts=[
                {
                    "role": "report_data",
                    "artifact_ref": "report_data",
                    "path": report_data_rel,
                    "feature_ref": "overview",
                    "required": True,
                }
            ],
            feature_refs=["overview", "keyword", "cluster_map", "term_network", "export"],
            files=files,
            transforms=[
                {"transform_type": "prepare_cluster_data", "description": "Prepare report data.json from keyword table."},
                {"transform_type": "render_html_report", "description": "Render dashboard, charts, and report index."},
            ],
            compatibility={"target_tools": ["Browser", "SciScape"], "limitations": ["Plotly pages may require CDN access"]},
            title=title,
        )

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(nav_path)}")

    return generated


def export_viewer(
    output_path: str = "viewer.html",
    title: str = "SciScape Viewer",
    write_manifest: bool = True,
) -> str:
    """Generate a standalone viewer HTML that loads hosted or uploaded data.

    Deploy this single file to Vercel / GitHub Pages / any static host.
    The viewer auto-loads ``data.json`` from the same directory when hosted
    over HTTP(S), accepts an explicit ``?data=...`` URL, and also supports
    drag-and-drop upload of ``data.json`` or keyword CSV/TSV files.

    Parameters
    ----------
    output_path : str
        Path for the HTML file.
    title : str
        Viewer title.
    write_manifest : bool
        Whether to write a manifest-backed static viewer export artifact.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    from ._dashboard_template import _DASHBOARD_HTML_TEMPLATE

    html = _DASHBOARD_HTML_TEMPLATE
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{N_KEYWORDS}}", "0")
    html = html.replace("{{N_CLUSTERS}}", "0")
    # Replace the DATA assignment with null — triggers upload UI
    html = html.replace("{{DATA_JSON}}", "null")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    if write_manifest:
        from sciscape.artifacts import write_export_manifest

        root = Path(abs_path).parent
        write_export_manifest(
            root,
            export_id="static_viewer",
            export_family="viewer",
            export_kind="static_viewer_html",
            primary_file=abs_path,
            source_artifacts=[
                {
                    "role": "viewer_template",
                    "artifact_ref": "viewer_template",
                    "path": "",
                    "feature_ref": "export",
                    "required": False,
                }
            ],
            feature_refs=["overview", "keyword", "cluster_map", "export"],
            compatibility={"target_tools": ["Browser", "SciScape"], "limitations": ["loads data.json or uploaded data"]},
            title=title,
        )

    return abs_path
