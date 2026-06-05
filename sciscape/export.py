"""Export cluster networks to standard graph formats (GEXF, GraphML).

Enables interoperability with Gephi, Cytoscape, and other tools.
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping

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


def _membership_mapping(
    membership: pl.DataFrame | Dict[str, int],
    *,
    cluster_col: str | None = None,
) -> dict[Any, Any]:
    if isinstance(membership, dict):
        return membership
    if cluster_col is None:
        cluster_col = next((c for c in membership.columns if c.startswith("cluster_")), "cluster")
    return dict(zip(membership["uid"].to_list(), membership[cluster_col].to_list()))


def _edge_weight_column(edges: pl.DataFrame) -> str | None:
    for candidate in ("rel_sum2", "weight", "score", "similarity", "cosine", "edge_weight"):
        if candidate in edges.columns:
            return candidate
    for column in edges.columns:
        if column.endswith("_weight"):
            return column
    return None


def _write_tab_rows(path: Path, rows: list[list[Any]], *, header: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if header:
            writer.writerow(header)
        writer.writerows(rows)


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
    uid_to_cluster = _membership_mapping(membership, cluster_col=cluster_col)

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

    uid_to_cluster = _membership_mapping(membership, cluster_col=cluster_col)

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


def export_vosviewer_network(
    edges: pl.DataFrame,
    membership: pl.DataFrame | Dict[str, int],
    output_dir: Path | str,
    *,
    abstracts: pl.DataFrame | None = None,
    cluster_col: str | None = None,
    map_filename: str = "vosviewer_map.txt",
    network_filename: str = "vosviewer_network.txt",
    write_manifest: bool = True,
    result_root: str | Path | None = None,
    source_paths: Mapping[str, str | Path | None] | None = None,
) -> dict[str, Path]:
    """Export a VOSviewer-style map/network text-file pair.

    The map file uses VOSviewer item attributes. The network file stores
    source-id, target-id, and link strength rows without a header.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / map_filename
    network_path = output_dir / network_filename

    uid_to_cluster = _membership_mapping(membership, cluster_col=cluster_col)
    weight_col = _edge_weight_column(edges)
    edge_records = edges.to_dicts()
    all_uids = set(uid_to_cluster.keys())
    for row in edge_records:
        all_uids.add(row.get("uid1"))
        all_uids.add(row.get("uid2"))
    all_uids = {uid for uid in all_uids if uid is not None}
    uid_order = sorted(all_uids, key=lambda value: str(value))
    uid_to_item_id = {uid: index + 1 for index, uid in enumerate(uid_order)}

    raw_clusters = sorted({str(uid_to_cluster.get(uid, "missing")) for uid in uid_order})
    if len(raw_clusters) > 1000:
        raise ValueError("VOSviewer cluster IDs support at most 1000 clusters")
    cluster_to_vos = {cluster: index + 1 for index, cluster in enumerate(raw_clusters)}

    uid_to_title: dict[Any, str] = {}
    uid_to_year: dict[Any, Any] = {}
    if abstracts is not None and "uid" in abstracts.columns:
        if "title" in abstracts.columns:
            uid_to_title = dict(zip(abstracts["uid"].to_list(), abstracts["title"].to_list()))
        if "pubyear" in abstracts.columns:
            uid_to_year = dict(zip(abstracts["uid"].to_list(), abstracts["pubyear"].to_list()))

    pair_strength: dict[tuple[int, int], float] = {}
    for row in edge_records:
        uid1 = row.get("uid1")
        uid2 = row.get("uid2")
        if uid1 not in uid_to_item_id or uid2 not in uid_to_item_id or uid1 == uid2:
            continue
        raw_weight = row.get(weight_col) if weight_col else 1.0
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        item1 = uid_to_item_id[uid1]
        item2 = uid_to_item_id[uid2]
        key = tuple(sorted((item1, item2)))
        pair_strength[key] = pair_strength.get(key, 0.0) + weight

    item_id_to_uid = {item_id: uid for uid, item_id in uid_to_item_id.items()}
    link_count: dict[Any, int] = {uid: 0 for uid in uid_order}
    total_strength: dict[Any, float] = {uid: 0.0 for uid in uid_order}
    for (item1, item2), strength in pair_strength.items():
        uid1 = item_id_to_uid[item1]
        uid2 = item_id_to_uid[item2]
        link_count[uid1] = link_count.get(uid1, 0) + 1
        link_count[uid2] = link_count.get(uid2, 0) + 1
        total_strength[uid1] = total_strength.get(uid1, 0.0) + strength
        total_strength[uid2] = total_strength.get(uid2, 0.0) + strength

    map_header = [
        "id",
        "label",
        "description",
        "cluster",
        "weight<Links>",
        "weight<Total link strength>",
        "score<Avg. pub. year>",
    ]
    map_rows: list[list[Any]] = []
    for uid in uid_order:
        raw_cluster = str(uid_to_cluster.get(uid, "missing"))
        year = uid_to_year.get(uid)
        map_rows.append(
            [
                uid_to_item_id[uid],
                str(uid),
                str(uid_to_title.get(uid, "")),
                cluster_to_vos[raw_cluster],
                int(link_count.get(uid, 0)),
                f"{float(total_strength.get(uid, 0.0)):.6f}",
                "" if year in (None, "") else str(year),
            ]
        )

    network_rows = [
        [source_id, target_id, f"{strength:.6f}"]
        for (source_id, target_id), strength in sorted(pair_strength.items())
    ]
    _write_tab_rows(map_path, map_rows, header=map_header)
    _write_tab_rows(network_path, network_rows)

    manifest_path = None
    if write_manifest:
        from sciscape.artifacts import write_export_manifest

        root = Path(result_root).expanduser().resolve() if result_root is not None else output_dir.resolve()
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
        manifest = write_export_manifest(
            root,
            export_id="vosviewer_map_network",
            export_family="vosviewer",
            export_kind="vosviewer_map_network",
            primary_file=map_path,
            source_artifacts=source_artifacts,
            feature_refs=["cluster_map", "evidence", "export"],
            files=[
                {
                    "file_id": "map",
                    "path": map_path,
                    "role": "map",
                    "format": "txt",
                    "public_share_state": "local",
                },
                {
                    "file_id": "network",
                    "path": network_path,
                    "role": "network",
                    "format": "txt",
                    "public_share_state": "local",
                },
            ],
            transforms=[
                {"transform_type": "load_edge_table", "description": "Load SciScape edge table."},
                {"transform_type": "map_source_ids_to_vosviewer_ids", "description": "Assign 1-based VOSviewer item IDs."},
                {"transform_type": "write_vosviewer_map_network", "description": "Write VOSviewer-style map and network files."},
            ],
            compatibility={
                "target_tools": ["VOSviewer", "VOSviewer Online"],
                "format_version": "map/network text files",
                "limitations": ["layout coordinates are not exported", "paper labels use source uids"],
            },
            title="VOSviewer map/network export",
        )
        manifest_path = Path(manifest["manifest_path"])

    log.info("Exported VOSviewer map/network: %d nodes, %d links → %s", len(uid_order), len(network_rows), output_dir)
    result = {"map_path": map_path, "network_path": network_path}
    if manifest_path is not None:
        result["manifest_path"] = manifest_path
    return result


__all__ = ["export_gexf", "export_graphml", "export_vosviewer_network"]
