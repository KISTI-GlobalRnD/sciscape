"""Export cluster networks to standard graph formats (GEXF, GraphML).

Enables interoperability with Gephi, Cytoscape, and other tools.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _source_artifact(
    role: str,
    path: str | Path | None,
    *,
    result_root: Path,
    artifact_ref: str | None = None,
    feature_ref: str = "cluster_map",
    required: bool = True,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "role": role,
        "artifact_ref": artifact_ref or role,
        "feature_ref": feature_ref,
        "required": required,
    }
    if path is None:
        row["path"] = ""
        row["required"] = False
        return row
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    try:
        row["path"] = candidate.relative_to(result_root).as_posix()
    except ValueError:
        # Do not leak private absolute paths into public export manifests.
        row["path"] = ""
        row["required"] = False
    return row


def _write_graph_export_manifest(
    output_path: Path,
    *,
    fmt: str,
    result_root: str | Path | None,
    source_paths: Mapping[str, str | Path | None] | None,
) -> None:
    from sciscape.artifacts import write_export_manifest

    root = Path(result_root).expanduser().resolve() if result_root is not None else output_path.parent.resolve()
    source_paths = source_paths or {}
    source_artifacts = [
        _source_artifact("edges", source_paths.get("edges"), result_root=root, artifact_ref="edges"),
        _source_artifact("membership", source_paths.get("membership"), result_root=root, artifact_ref="membership"),
    ]
    if source_paths.get("abstracts") is not None:
        source_artifacts.append(
            _source_artifact(
                "records",
                source_paths.get("abstracts"),
                result_root=root,
                artifact_ref="records",
                feature_ref="overview",
                required=False,
            )
        )
    write_export_manifest(
        root,
        export_id=f"network_{fmt}",
        export_family="graph",
        export_kind=f"{fmt}_graph",
        primary_file=output_path,
        source_artifacts=source_artifacts,
        feature_refs=["cluster_map", "evidence", "export"],
        compatibility={
            "target_tools": ["Gephi"] if fmt == "gexf" else ["Cytoscape", "NetworkX"],
            "format_version": "GEXF 1.3" if fmt == "gexf" else "GraphML",
            "limitations": [],
        },
        transforms=[
            {"transform_type": "load_edge_table", "description": "Load SciScape edge table."},
            {"transform_type": f"write_{fmt}", "description": f"Write network as {fmt.upper()}."},
        ],
        title=f"SciScape {fmt.upper()} network export",
    )


def export_gexf(
    edges: pl.DataFrame,
    membership: pl.DataFrame | Dict[str, int],
    output_path: Path | str,
    *,
    keywords: pl.DataFrame | None = None,
    abstracts: pl.DataFrame | None = None,
    keyword_col: str = "keyword",
    cluster_col: str | None = None,
    write_manifest: bool = False,
    result_root: str | Path | None = None,
    source_paths: Mapping[str, str | Path | None] | None = None,
) -> Path:
    """Export cluster network as GEXF (Gephi format).

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table (uid1, uid2, rel_sum2).
    membership : pl.DataFrame or dict
        Node → cluster mapping. If DataFrame, must have 'uid' and a cluster_ column.
    output_path : Path
        Output GEXF file path.
    keywords : pl.DataFrame, optional
        Keywords per cluster for node labels.
    abstracts : pl.DataFrame, optional
        Paper metadata (uid, title, pubyear) for node attributes.
    """
    output_path = Path(output_path)

    # Parse membership
    if isinstance(membership, dict):
        uid_to_cluster = membership
    else:
        if cluster_col is None:
            cluster_col = next((c for c in membership.columns if c.startswith("cluster_")), "cluster")
        uid_to_cluster = dict(zip(
            membership["uid"].to_list(),
            membership[cluster_col].to_list(),
        ))

    # Parse metadata
    uid_to_title = {}
    uid_to_year = {}
    if abstracts is not None:
        uid_to_title = dict(zip(abstracts["uid"].to_list(), abstracts["title"].to_list()))
        if "pubyear" in abstracts.columns:
            uid_to_year = dict(zip(abstracts["uid"].to_list(), abstracts["pubyear"].to_list()))

    # Cluster labels from keywords
    cluster_labels = {}
    if keywords is not None:
        kw_cluster_col = next((c for c in keywords.columns if "cluster" in c.lower()), None)
        kw_keyword_col = next((c for c in keywords.columns if c.lower() in (keyword_col, "term", "keyword")), keyword_col)
        if kw_cluster_col and kw_keyword_col in keywords.columns:
            first_kw = keywords.group_by(kw_cluster_col).agg(pl.col(kw_keyword_col).first())
            cluster_labels = dict(zip(
                first_kw[kw_cluster_col].to_list(),
                first_kw[kw_keyword_col].to_list(),
            ))

    # Build GEXF XML
    all_uids = set(edges["uid1"].to_list()) | set(edges["uid2"].to_list())

    gexf = ET.Element("gexf", xmlns="http://gexf.net/1.3", version="1.3")
    graph = ET.SubElement(gexf, "graph", defaultedgetype="undirected", mode="static")

    # Node attributes declaration
    attrs = ET.SubElement(graph, "attributes", {"class": "node", "mode": "static"})
    ET.SubElement(attrs, "attribute", id="0", title="cluster", type="integer")
    ET.SubElement(attrs, "attribute", id="1", title="cluster_label", type="string")
    ET.SubElement(attrs, "attribute", id="2", title="title", type="string")
    ET.SubElement(attrs, "attribute", id="3", title="year", type="integer")

    # Nodes
    nodes_el = ET.SubElement(graph, "nodes")
    for uid in sorted(all_uids):
        cid = uid_to_cluster.get(uid, -1)
        label = cluster_labels.get(cid, f"cluster_{cid}")
        node = ET.SubElement(nodes_el, "node", id=str(uid), label=str(uid))
        attvals = ET.SubElement(node, "attvalues")
        ET.SubElement(attvals, "attvalue", {"for": "0", "value": str(cid)})
        ET.SubElement(attvals, "attvalue", {"for": "1", "value": str(label)})
        if uid in uid_to_title:
            ET.SubElement(attvals, "attvalue", {"for": "2", "value": str(uid_to_title[uid])[:200]})
        if uid in uid_to_year:
            yr = uid_to_year[uid]
            if yr:
                ET.SubElement(attvals, "attvalue", {"for": "3", "value": str(int(yr))})

    # Edges
    edges_el = ET.SubElement(graph, "edges")
    for i, (u1, u2, w) in enumerate(zip(
        edges["uid1"].to_list(), edges["uid2"].to_list(), edges["rel_sum2"].to_list()
    )):
        ET.SubElement(edges_el, "edge", id=str(i), source=str(u1), target=str(u2),
                      weight=f"{w:.6f}")

    # Write
    tree = ET.ElementTree(gexf)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    if write_manifest:
        _write_graph_export_manifest(
            output_path,
            fmt="gexf",
            result_root=result_root,
            source_paths=source_paths,
        )
    log.info("Exported GEXF: %d nodes, %d edges → %s", len(all_uids), edges.height, output_path)
    return output_path


def export_graphml(
    edges: pl.DataFrame,
    membership: pl.DataFrame | Dict[str, int],
    output_path: Path | str,
    *,
    abstracts: pl.DataFrame | None = None,
    cluster_col: str | None = None,
    write_manifest: bool = False,
    result_root: str | Path | None = None,
    source_paths: Mapping[str, str | Path | None] | None = None,
) -> Path:
    """Export cluster network as GraphML (Cytoscape format).

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table (uid1, uid2, rel_sum2).
    membership : pl.DataFrame or dict
        Node → cluster mapping.
    output_path : Path
        Output GraphML file path.
    """
    output_path = Path(output_path)

    if isinstance(membership, dict):
        uid_to_cluster = membership
    else:
        if cluster_col is None:
            cluster_col = next((c for c in membership.columns if c.startswith("cluster_")), "cluster")
        uid_to_cluster = dict(zip(
            membership["uid"].to_list(), membership[cluster_col].to_list(),
        ))

    uid_to_title = {}
    uid_to_year = {}
    if abstracts is not None:
        uid_to_title = dict(zip(abstracts["uid"].to_list(), abstracts["title"].to_list()))
        if "pubyear" in abstracts.columns:
            uid_to_year = dict(zip(abstracts["uid"].to_list(), abstracts["pubyear"].to_list()))

    all_uids = set(edges["uid1"].to_list()) | set(edges["uid2"].to_list())

    ns = "http://graphml.graphstruct.org/xmlns"
    root = ET.Element("graphml", xmlns=ns)

    # Attribute keys
    ET.SubElement(root, "key", {"id": "d0", "for": "node", "attr.name": "cluster", "attr.type": "int"})
    ET.SubElement(root, "key", {"id": "d1", "for": "node", "attr.name": "title", "attr.type": "string"})
    ET.SubElement(root, "key", {"id": "d2", "for": "node", "attr.name": "year", "attr.type": "int"})
    ET.SubElement(root, "key", {"id": "d3", "for": "edge", "attr.name": "weight", "attr.type": "double"})

    graph = ET.SubElement(root, "graph", id="G", edgedefault="undirected")

    for uid in sorted(all_uids):
        node = ET.SubElement(graph, "node", id=str(uid))
        cid = uid_to_cluster.get(uid, -1)
        d = ET.SubElement(node, "data", key="d0")
        d.text = str(cid)
        if uid in uid_to_title:
            d1 = ET.SubElement(node, "data", key="d1")
            d1.text = str(uid_to_title[uid])[:200]
        if uid in uid_to_year and uid_to_year[uid]:
            d2 = ET.SubElement(node, "data", key="d2")
            d2.text = str(int(uid_to_year[uid]))

    for i, (u1, u2, w) in enumerate(zip(
        edges["uid1"].to_list(), edges["uid2"].to_list(), edges["rel_sum2"].to_list()
    )):
        edge = ET.SubElement(graph, "edge", id=f"e{i}", source=str(u1), target=str(u2))
        d = ET.SubElement(edge, "data", key="d3")
        d.text = f"{w:.6f}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    if write_manifest:
        _write_graph_export_manifest(
            output_path,
            fmt="graphml",
            result_root=result_root,
            source_paths=source_paths,
        )
    log.info("Exported GraphML: %d nodes, %d edges → %s", len(all_uids), edges.height, output_path)
    return output_path


__all__ = ["export_gexf", "export_graphml"]
