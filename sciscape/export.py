"""Export cluster networks to standard graph formats (GEXF, GraphML).

Enables interoperability with Gephi, Cytoscape, and other tools.
"""

from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
import zipfile
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
    selection: Mapping[str, Any] | None = None,
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
        selection=selection
        or {
            "scope": "full_result",
            "view": {"mode": "cluster_network_export", "surface": "graph_export"},
            "layer_state": {"network_format": fmt, "edge_table": "edges"},
            "thresholds": {},
            "filters": [],
        },
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


def _relative_existing_file(root: Path, path: Any) -> tuple[str, Path] | None:
    rel_path = "" if path is None else str(path).strip()
    if not rel_path or Path(rel_path).is_absolute():
        return None
    resolved = (root / rel_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return rel_path, resolved


def _add_bundle_file(
    files: list[dict[str, Any]],
    seen: set[str],
    *,
    root: Path,
    path: Any,
    role: str,
    source_export_id: str,
) -> None:
    resolved = _relative_existing_file(root, path)
    if resolved is None:
        return
    rel_path, file_path = resolved
    if rel_path in seen:
        return
    seen.add(rel_path)
    files.append(
        {
            "path": rel_path,
            "role": role,
            "source_export_id": source_export_id,
            "bytes": int(file_path.stat().st_size),
        }
    )


def _vosviewer_bundle_file_rows(root: Path, result_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    export_rows = result_manifest.get("exports") if isinstance(result_manifest.get("exports"), list) else []
    for export_row in export_rows:
        if not isinstance(export_row, Mapping):
            continue
        if export_row.get("export_family") != "vosviewer":
            continue
        if export_row.get("status") in {"missing", "blocked"}:
            continue
        export_id = str(export_row.get("export_id") or "vosviewer_export")
        for file_row in export_row.get("files") or []:
            if not isinstance(file_row, Mapping) or file_row.get("exists") is False:
                continue
            _add_bundle_file(
                files,
                seen,
                root=root,
                path=file_row.get("path"),
                role=str(file_row.get("role") or "file"),
                source_export_id=export_id,
            )
        _add_bundle_file(
            files,
            seen,
            root=root,
            path=export_row.get("export_manifest_ref"),
            role="export_manifest",
            source_export_id=export_id,
        )
        export_manifest_ref = str(export_row.get("export_manifest_ref") or "")
        if export_manifest_ref:
            qa_ref = Path(export_manifest_ref).parent / "export_qa.json"
            _add_bundle_file(
                files,
                seen,
                root=root,
                path=qa_ref.as_posix(),
                role="export_qa",
                source_export_id=export_id,
            )
    return files


def export_vosviewer_bundle(
    result_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    bundle_filename: str = "vosviewer_bundle.zip",
    inventory_filename: str = "vosviewer_bundle_inventory.json",
    write_manifest: bool = True,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Package manifest-backed VOSviewer exports into one zip bundle."""

    from sciscape.artifacts import load_result_manifest, write_export_manifest

    root = Path(result_root).expanduser().resolve()
    export_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else root / "exports" / "vosviewer_bundle"
    export_dir.mkdir(parents=True, exist_ok=True)

    result_manifest = load_result_manifest(root)
    bundle_files = _vosviewer_bundle_file_rows(root, result_manifest)
    if not bundle_files:
        raise ValueError("No manifest-backed VOSviewer export files are available for bundling.")

    inventory_path = export_dir / inventory_filename
    inventory_payload = {
        "schema_version": "sciscape_vosviewer_bundle_inventory_v1",
        "result_id": result_manifest.get("result_id"),
        "source": "result_manifest.exports",
        "file_count": len(bundle_files),
        "files": bundle_files,
    }
    inventory_path.write_text(json.dumps(inventory_payload, indent=2, sort_keys=True), encoding="utf-8")

    bundle_path = export_dir / bundle_filename
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in bundle_files:
            archive.write(root / row["path"], arcname=row["path"])
        archive.write(inventory_path, arcname=inventory_filename)

    manifest_path = None
    if write_manifest:
        manifest = write_export_manifest(
            root,
            export_id="vosviewer_bundle",
            export_family="bundle",
            export_kind="vosviewer_bundle",
            primary_file=bundle_path,
            source_artifacts=[
                {
                    "role": row["role"],
                    "artifact_ref": f"{row['source_export_id']}:{row['role']}",
                    "feature_ref": "export",
                    "path": row["path"],
                    "required": True,
                }
                for row in bundle_files
            ],
            feature_refs=["export"],
            selection=selection
            or {
                "scope": "manifest_backed_exports",
                "view": {"mode": "download_bundle", "surface": "web_download"},
                "filters": [{"field": "export_family", "op": "eq", "value": "vosviewer"}],
                "thresholds": {},
                "layer_state": {
                    "source_inventory": "result_manifest.exports",
                    "bundle_file_count": len(bundle_files),
                },
            },
            files=[
                {
                    "file_id": "bundle",
                    "path": bundle_path,
                    "role": "bundle",
                    "format": "zip",
                    "public_share_state": "local",
                },
                {
                    "file_id": "inventory",
                    "path": inventory_path,
                    "role": "inventory",
                    "format": "json",
                    "public_share_state": "local",
                },
            ],
            transforms=[
                {
                    "transform_type": "collect_result_manifest_vosviewer_exports",
                    "description": "Collect VOSviewer export files from result_manifest.exports.",
                },
                {
                    "transform_type": "write_vosviewer_bundle_zip",
                    "description": "Write one downloadable zip bundle for VOSviewer workflows.",
                    "parameters": {"file_count": len(bundle_files)},
                },
            ],
            compatibility={
                "target_tools": ["VOSviewer", "VOSviewer Online"],
                "format_version": "zip bundle of VOSviewer-compatible files",
                "limitations": ["bundle membership is derived only from manifest-backed VOSviewer exports"],
            },
            title="VOSviewer export bundle",
            format="zip",
            output_dir=export_dir,
        )
        manifest_path = Path(manifest["manifest_path"])

    result = {"bundle_path": bundle_path, "inventory_path": inventory_path}
    if manifest_path is not None:
        result["manifest_path"] = manifest_path
    return result


def _keyword_rule_manifest_path(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        return candidate / "rule_set_manifest.json", candidate
    return candidate, candidate.parent


def _keyword_rule_result_root(rule_dir: Path, result_root: str | Path | None) -> Path:
    if result_root is not None:
        return Path(result_root).expanduser().resolve()
    if rule_dir.parent.name == "rules":
        return rule_dir.parent.parent.resolve()
    return rule_dir.resolve()


def _read_keyword_rule_table(rule_dir: Path, manifest: Mapping[str, Any], output_key: str) -> Any:
    import pandas as pd

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    rel_path = str(outputs.get(output_key) or "")
    if not rel_path:
        return pd.DataFrame()
    path = rule_dir / rel_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _add_thesaurus_mapping(rows: dict[str, str], label: Any, replacement: Any) -> None:
    label_text = "" if label is None else str(label).strip()
    if not label_text:
        return
    replacement_text = "" if replacement is None else str(replacement).strip()
    existing = rows.get(label_text)
    if existing is None:
        rows[label_text] = replacement_text
        return
    if existing == replacement_text:
        return
    if existing and replacement_text:
        rows[label_text] = min(existing, replacement_text)
        return
    rows[label_text] = existing or replacement_text


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _split_rule_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(part).strip() for part in parsed if str(part).strip()]
    return [part.strip() for part in text.split("|") if part.strip()]


def _keyword_rule_thesaurus_rows(rules: Any, applications: Any, before_after: Any) -> list[list[str]]:
    rows: dict[str, str] = {}
    rules_by_id = {
        str(row.get("rule_id")): row
        for row in rules.to_dict("records")
        if row.get("rule_id") not in (None, "")
    }

    if before_after is not None and not before_after.empty:
        for row in before_after.to_dict("records"):
            label = row.get("term_before") or row.get("raw_term")
            replacement = row.get("term_after") or ""
            if _truthy(row.get("blocked")):
                _add_thesaurus_mapping(rows, label, "")
            elif row.get("parent_term"):
                _add_thesaurus_mapping(rows, label, row.get("parent_term"))
            elif label and replacement and str(label).strip() != str(replacement).strip():
                _add_thesaurus_mapping(rows, label, replacement)

    if applications is not None and not applications.empty:
        for row in applications.to_dict("records"):
            action = str(row.get("action") or "")
            if action == "keep_with_flag":
                continue
            rule = rules_by_id.get(str(row.get("rule_id")), {})
            label = row.get("display_label_before") or row.get("normalized_term_before") or row.get("raw_term")
            replacement = (
                rule.get("replacement")
                or row.get("display_label_after")
                or row.get("normalized_term_after")
                or ""
            )
            if action == "block":
                _add_thesaurus_mapping(rows, label, "")
            elif action == "group_under":
                target = replacement or row.get("display_label_after") or row.get("normalized_term_after")
                for variant in _split_rule_evidence(row.get("evidence_value")):
                    _add_thesaurus_mapping(rows, variant, target)
            elif action in {"normalize", "alias_to", "expand_to"}:
                _add_thesaurus_mapping(rows, label, replacement)

    return [[label, rows[label]] for label in sorted(rows)]


def export_vosviewer_thesaurus(
    rule_manifest: str | Path,
    output_dir: Path | str,
    *,
    thesaurus_filename: str = "vosviewer_thesaurus.txt",
    rule_set_filename: str = "sciscape_keyword_rules.tsv",
    write_manifest: bool = True,
    result_root: str | Path | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Export keyword cleaning rules as a VOSviewer thesaurus file.

    The VOSviewer thesaurus file has two tab-delimited columns: ``label`` and
    ``replace by``. Empty replacements represent stop/ignore rows.
    """

    from sciscape.artifacts import validate_keyword_rule_artifact

    manifest_path, rule_dir = _keyword_rule_manifest_path(rule_manifest)
    validation = validate_keyword_rule_artifact(manifest_path)
    if not validation.ok:
        raise ValueError(f"keyword rule artifact is blocked: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rules = _read_keyword_rule_table(rule_dir, manifest, "rules")
    applications = _read_keyword_rule_table(rule_dir, manifest, "applications")
    before_after = _read_keyword_rule_table(rule_dir, manifest, "before_after")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thesaurus_path = output_dir / thesaurus_filename
    rule_set_path = output_dir / rule_set_filename

    thesaurus_rows = _keyword_rule_thesaurus_rows(rules, applications, before_after)
    _write_tab_rows(thesaurus_path, thesaurus_rows, header=["label", "replace by"])

    rule_columns = [
        "rule_id",
        "rule_family",
        "match_type",
        "pattern",
        "replacement",
        "action",
        "destructive",
        "confidence_policy",
        "reason",
    ]
    rule_records = rules.to_dict("records") if rules is not None and not rules.empty else []
    rule_rows = [[row.get(column, "") for column in rule_columns] for row in rule_records]
    _write_tab_rows(rule_set_path, rule_rows, header=rule_columns)

    manifest_export_path = None
    if write_manifest:
        from sciscape.artifacts import write_export_manifest

        root = _keyword_rule_result_root(rule_dir, result_root)
        manifest_export = write_export_manifest(
            root,
            export_id="vosviewer_thesaurus",
            export_family="vosviewer",
            export_kind="vosviewer_thesaurus",
            primary_file=thesaurus_path,
            source_artifacts=[
                {
                    "role": "keyword_rules",
                    "artifact_ref": "keyword_rules",
                    "feature_ref": "keyword",
                    "path": manifest_path,
                    "required": True,
                }
            ],
            feature_refs=["keyword", "export"],
            selection=selection
            or {
                "scope": "keyword_rule_artifact",
                "view": {"mode": "cleaning_rules", "surface": "rule_export"},
                "filters": [{"field": "action", "op": "exclude", "value": "keep_with_flag"}],
                "thresholds": {},
                "layer_state": {
                    "rule_set_id": manifest.get("rule_set_id"),
                    "thesaurus_columns": ["label", "replace by"],
                },
            },
            files=[
                {
                    "file_id": "thesaurus",
                    "path": thesaurus_path,
                    "role": "thesaurus",
                    "format": "txt",
                    "public_share_state": "local",
                },
                {
                    "file_id": "rule_set",
                    "path": rule_set_path,
                    "role": "rule_set",
                    "format": "tsv",
                    "public_share_state": "local",
                },
            ],
            transforms=[
                {"transform_type": "validate_keyword_rule_artifact", "description": "Validate replayable keyword rule artifact."},
                {"transform_type": "write_vosviewer_thesaurus", "description": "Convert block, alias, acronym, and grouping rules to VOSviewer thesaurus rows."},
            ],
            compatibility={
                "target_tools": ["VOSviewer", "VOSviewer Online"],
                "format_version": "thesaurus text file",
                "limitations": ["review-only flags are not exported as replacement rules"],
            },
            title="VOSviewer keyword thesaurus export",
        )
        manifest_export_path = Path(manifest_export["manifest_path"])

    result = {"thesaurus_path": thesaurus_path, "rule_set_path": rule_set_path}
    if manifest_export_path is not None:
        result["manifest_path"] = manifest_export_path
    return result


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
    selection: Mapping[str, Any] | None = None,
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
    resolved_cluster_col = cluster_col
    if resolved_cluster_col is None and isinstance(membership, pl.DataFrame):
        resolved_cluster_col = next((c for c in membership.columns if c.startswith("cluster_")), "cluster")
    uid_to_cluster = _membership_mapping(membership, cluster_col=resolved_cluster_col)

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
            selection=selection,
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
    selection: Mapping[str, Any] | None = None,
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
            selection=selection,
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
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Export a VOSviewer-style map/network text-file pair.

    The map file uses VOSviewer item attributes. The network file stores
    source-id, target-id, and link strength rows without a header.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / map_filename
    network_path = output_dir / network_filename

    resolved_cluster_col = cluster_col
    if resolved_cluster_col is None and isinstance(membership, pl.DataFrame):
        resolved_cluster_col = next((c for c in membership.columns if c.startswith("cluster_")), "cluster")
    uid_to_cluster = _membership_mapping(membership, cluster_col=resolved_cluster_col)
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
            selection=selection
            or {
                "scope": "full_result",
                "view": {"mode": "vosviewer_map_network", "surface": "export"},
                "cluster_level": resolved_cluster_col or "mapping",
                "filters": [{"field": "link_strength", "op": "gt", "value": 0}],
                "thresholds": {"min_link_strength": 0},
                "layer_state": {
                    "map_file": map_filename,
                    "network_file": network_filename,
                    "edge_weight_column": weight_col or "unit_weight",
                },
            },
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


__all__ = [
    "export_gexf",
    "export_graphml",
    "export_vosviewer_bundle",
    "export_vosviewer_network",
    "export_vosviewer_thesaurus",
]
