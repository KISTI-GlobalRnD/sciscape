"""Dashboard export — assembles data + template into HTML."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import pandas as pd

from ._data_prep import prepare_cluster_data
from ._dashboard_template import _DASHBOARD_HTML_TEMPLATE


def export_dashboard(
    df: pd.DataFrame,
    output_path: str = "keyword_dashboard.html",
    title: str = "SciScape Keyword Explorer",
    open_browser: bool = False,
    viz_data: Optional[Dict] = None,
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

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    cluster_data = prepare_cluster_data(df, viz_data=viz_data)
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

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")

    return abs_path
