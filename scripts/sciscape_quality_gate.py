#!/usr/bin/env python3
"""Lightweight release gates for SciScape demo and visualization surfaces."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.artifacts import (
    build_atlas_render_payload,
    default_artifact_contract_path,
    validate_result_root,
    write_artifact_contract,
    write_edge_evidence_samples,
    write_result_manifest,
)
from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
from sciscape.keyword_extraction.visualization import export_dashboard
from sciscape.landscape import LandscapeConfig, run_landscape
from sciscape.web.network_data import build_term_network_json


ARTIFACT_TERMS = {
    "class htmlview paragraph",
    "div class htmlview",
    "lt div gt",
    "get access",
    "journal article",
    "articles author",
    "works author",
    "author gsw google",
    "urology vol",
    "usepackage",
    "usepackage amsmath",
    "documentclass article",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_smoke_inputs(root: Path) -> tuple[Path, Path]:
    abstracts = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(8)],
            "title": [
                "Nanoparticle synthesis for stable catalysts",
                "Catalytic nanoparticle synthesis routes",
                "Nanocrystal growth and nanoparticle stability",
                "Encoded publisher page artifact",
                "Graph neural network traffic forecasting",
                "Spatio temporal graph neural network",
                "Graph neural network anomaly detection",
                "HTML and LaTeX page residue",
            ],
            "abstract": [
                "Nanoparticle synthesis improves catalytic stability and reaction selectivity.",
                "Catalyst nanoparticles show stable synthesis routes and high activity.",
                "Nanocrystal growth controls nanoparticle size and catalytic performance.",
                (
                    "&lt;div class=&quot;htmlview paragraph&quot;&gt; Get access Journal Article "
                    "Articles Author Works Author GSW Google Urology vol \\usepackage{amsmath}"
                ),
                "Graph neural networks improve traffic forecasting on road sensor networks.",
                "Spatio temporal graph neural network models capture dynamic traffic flows.",
                "Graph neural networks support anomaly detection and representation learning.",
                "documentclass article lt div gt class htmlview paragraph",
            ],
            "pubyear": [2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023],
        }
    )
    membership = pd.DataFrame(
        {
            "uid": [f"D{i}" for i in range(8)],
            "cluster": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    abstract_path = root / "abstracts.parquet"
    membership_path = root / "membership.parquet"
    abstracts.to_parquet(abstract_path, index=False)
    membership.to_parquet(membership_path, index=False)
    return abstract_path, membership_path


def run_smoke_gate() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sciscape_quality_gate_") as tmp:
        root = Path(tmp)
        abstract_path, membership_path = _write_smoke_inputs(root)
        cfg = KeywordExtractionConfig(
            abstract_path=abstract_path,
            membership_path=membership_path,
            cluster_level="cluster",
            include_title=True,
            title_weight=1.0,
            min_df_unigram=1,
            min_df_phrase=1,
            phrase_min_count_per_cluster=1,
            top_n_keywords=30,
            scoring_pool_factor=2.0,
            ngram_min=1,
            ngram_max=3,
            use_phrase_vectorizer=True,
            normalization_enabled=True,
            cooccurrence_enabled=True,
            cooccurrence_min_count=1,
            quality_rerank_enabled=True,
            n_jobs=1,
            verbose=False,
        )
        keywords = run_keyword_pipeline(cfg)
        _assert(not keywords.empty, "keyword smoke gate produced no keywords")

        terms = set(keywords["term"].str.lower())
        leaked = sorted(ARTIFACT_TERMS & terms)
        _assert(not leaked, f"artifact terms leaked into keywords: {leaked}")

        keyword_path = root / "keywords.parquet"
        keyword_cols = [
            col
            for col in ("cluster_id", "term", "display_label", "score", "frequency")
            if col in keywords.columns
        ]
        keywords[keyword_cols].to_parquet(keyword_path, index=False)
        term_network = build_term_network_json(
            keyword_path,
            top_k_per_cluster=10,
            min_cooc=1,
            max_terms=80,
        )
        _assert(term_network["nodes"], "term co-occurrence gate has no nodes")
        _assert(term_network["edges"], "term co-occurrence gate has no edges")

        dashboard_path = root / "dashboard.html"
        export_dashboard(
            keywords,
            output_path=str(dashboard_path),
            title="SciScape Quality Gate",
            selection={
                "scope": "quality_gate_smoke",
                "view": {"mode": "keyword_dashboard", "surface": "quality_gate_smoke"},
                "filters": [],
                "thresholds": {},
                "layer_state": {
                    "script": "sciscape_quality_gate.py",
                    "smoke": "keyword_quality",
                },
                "focus": {},
            },
        )
        html = dashboard_path.read_text(encoding="utf-8")
        _assert("SciScape Quality Gate" in html, "dashboard title missing")
        _assert("keywords-tier-summary" in html, "dashboard tier summary missing")
        _assert("cooccurrence-evidence" in html, "dashboard cooccurrence evidence missing")

        return {
            "status": "passed",
            "keywords": int(len(keywords)),
            "clusters": int(keywords["cluster_id"].nunique()),
            "term_network_nodes": int(len(term_network["nodes"])),
            "term_network_edges": int(len(term_network["edges"])),
        }


def validate_demo_outputs(
    *,
    root: Path,
    manifest_path: Path,
    allow_missing: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for name, preset in dict(manifest["presets"]).items():
        slug = str(preset["slug"])
        out_dir = root / slug
        expected = [Path(p) for p in preset.get("expected_artifacts", [])]
        missing = [p.as_posix() for p in expected if not (out_dir / p).exists()]
        if missing and not allow_missing:
            raise AssertionError(f"demo preset {name!r} is missing artifacts: {missing}")

        keyword_path = out_dir / "landscape" / "keywords.parquet"
        term_nodes = 0
        term_edges = 0
        if keyword_path.exists():
            kw = pd.read_parquet(keyword_path)
            _assert(not kw.empty, f"demo preset {name!r} has empty keywords")
            term_network = build_term_network_json(
                keyword_path,
                top_k_per_cluster=10,
                min_cooc=1,
                max_terms=150,
            )
            term_nodes = len(term_network["nodes"])
            term_edges = len(term_network["edges"])
            _assert(term_nodes > 0, f"demo preset {name!r} has no term nodes")
            _assert(term_edges > 0, f"demo preset {name!r} has no term co-occurrence edges")

        rows.append(
            {
                "preset": name,
                "slug": slug,
                "output_dir": str(out_dir),
                "status": "skipped_missing" if missing else "passed",
                "missing": missing,
                "term_network_nodes": term_nodes,
                "term_network_edges": term_edges,
            }
        )
    return {"status": "passed", "demos": rows}


def _write_web_demo_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Create a tiny local demo output that exercises the web demo-open path."""
    local_root = root / "workspace" / "examples_output"
    output_dir = local_root / "openalex_live_20260530_010203" / "quality_gate_demo"
    landscape_dir = output_dir / "landscape"
    report_dir = landscape_dir / "report"
    report_dir.mkdir(parents=True)

    abstract_path = output_dir / "abstracts.parquet"
    edge_path = output_dir / "edges.parquet"
    membership_path = landscape_dir / "membership.parquet"
    keyword_path = landscape_dir / "keywords.parquet"
    edge_evidence_path = landscape_dir / "edge_evidence_samples.json"

    pd.DataFrame(
        {
            "uid": ["W0", "W1", "W2", "W3"],
            "title": [
                "Perovskite interface passivation",
                "Perovskite solar cell stability",
                "Graph neural traffic forecasting",
                "Graph neural anomaly detection",
            ],
            "abstract": [
                "Interface passivation improves perovskite solar cell stability.",
                "Perovskite devices use passivation layers for stable performance.",
                "Graph neural networks forecast traffic over road sensor graphs.",
                "Graph neural networks detect anomalies in dynamic graphs.",
            ],
            "pubyear": [2021, 2022, 2021, 2022],
        }
    ).to_parquet(abstract_path, index=False)
    pd.DataFrame(
        {
            "uid1": ["W0", "W0", "W1", "W2"],
            "uid2": ["W1", "W2", "W3", "W3"],
            "rel_sum2": [2.0, 1.0, 1.0, 2.0],
        }
    ).to_parquet(edge_path, index=False)
    pd.DataFrame(
        {
            "uid": ["W0", "W1", "W2", "W3"],
            "cluster_nano": [0, 0, 1, 1],
        }
    ).to_parquet(membership_path, index=False)
    pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, 1, 1],
            "term": [
                "perovskite solar cells",
                "interface passivation",
                "device stability",
                "graph neural networks",
                "traffic forecasting",
                "anomaly detection",
            ],
            "display_label": [
                "perovskite solar cells",
                "interface passivation",
                "device stability",
                "graph neural networks",
                "traffic forecasting",
                "anomaly detection",
            ],
            "score": [0.95, 0.9, 0.8, 0.96, 0.88, 0.78],
            "frequency": [2, 2, 1, 2, 1, 1],
        }
    ).to_parquet(keyword_path, index=False)
    written_edge_evidence = write_edge_evidence_samples(
        edges_path=edge_path,
        membership_path=membership_path,
        abstracts_path=abstract_path,
        output_path=edge_evidence_path,
        max_relations=10,
        max_samples_per_relation=2,
    )
    _assert(written_edge_evidence == edge_evidence_path, "edge evidence fixture was not written")
    (report_dir / "data.json").write_text(
        json.dumps(
            {
                "0": {
                    "label": "perovskite solar cells, interface passivation",
                    "keywords": [
                        {"term": "perovskite solar cells"},
                        {"term": "interface passivation"},
                    ],
                },
                "1": {
                    "label": "graph neural networks, traffic forecasting",
                    "keywords": [
                        {"term": "graph neural networks"},
                        {"term": "traffic forecasting"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "index.html").write_text("<html><title>dashboard</title></html>", encoding="utf-8")
    (report_dir / "report.html").write_text("<html><title>report</title></html>", encoding="utf-8")
    write_result_manifest(output_dir, mode="demo")

    manifest_path = root / "demo_presets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_output_root": str(local_root / "openalex_live"),
                "presets": {
                    "quality_gate": {
                        "slug": "quality_gate_demo",
                        "title": "Quality Gate Demo",
                        "query": "quality gate demo",
                        "max_works": 4,
                        "expected_artifacts": [
                            "result_manifest.json",
                            "abstracts.parquet",
                            "edges.parquet",
                            "landscape/membership.parquet",
                            "landscape/keywords.parquet",
                            "landscape/edge_evidence_samples.json",
                            "landscape/report/data.json",
                            "landscape/report/index.html",
                            "landscape/report/report.html",
                        ],
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path, local_root, report_dir / "data.json"


def run_web_demo_smoke_gate() -> dict[str, Any]:
    """Exercise the web demo launcher and key visualization data endpoints."""
    with tempfile.TemporaryDirectory(prefix="sciscape_web_demo_gate_") as tmp:
        root = Path(tmp)
        manifest_path, local_root, primary_path = _write_web_demo_fixture(root)

        from fastapi.testclient import TestClient
        import sciscape.web.app as web_app
        from sciscape.web.jobstore import JobStore

        old_manifest = web_app._DEMO_MANIFEST_PATH
        old_roots = web_app._LOCAL_DATA_ROOTS
        old_jobs = web_app._jobs
        try:
            web_app._DEMO_MANIFEST_PATH = manifest_path
            web_app._LOCAL_DATA_ROOTS = [local_root]
            web_app._jobs = JobStore(root / "jobs.db")

            client = TestClient(web_app.app)
            home = client.get("/")
            _assert(home.status_code == 200, "web homepage did not load")
            _assert("Recommended Demos" in home.text, "web homepage lacks Recommended Demos")
            _assert("loadDemoPresets()" in home.text, "web homepage lacks demo refresh action")

            demo_response = client.get("/api/demo-presets")
            _assert(demo_response.status_code == 200, "demo preset endpoint failed")
            demos = demo_response.json().get("demos", [])
            _assert(len(demos) == 1, "demo preset endpoint returned unexpected demo count")
            demo = demos[0]
            _assert(demo["status"] == "available", "synthetic demo was not available")
            _assert(demo["primary_path"] == str(primary_path), "synthetic demo primary path mismatch")

            open_response = client.post("/api/local-data/open", json={"path": demo["primary_path"]})
            _assert(open_response.status_code == 200, "open local demo endpoint failed")
            job_id = open_response.json()["job_id"]

            job_response = client.get(f"/api/jobs/{job_id}")
            _assert(job_response.status_code == 200, "opened demo job status endpoint failed")
            job = job_response.json()
            _assert(job["status"] == "done", "opened demo job was not marked done")
            result = job["result"]
            _assert(result["landscape_rel_path"] == "landscape", "landscape rel path mismatch")
            contract = result["artifact_contract"]
            _assert(
                contract["counts"]["edge_evidence_artifacts"] == 1,
                "artifact contract did not detect edge evidence sidecar",
            )
            atlas = result.get("atlas", {})
            sample_count = sum(
                int(neighbor.get("sample_count") or 0)
                for node in atlas.get("nodes", [])
                for neighbor in node.get("neighbors", [])
            )
            _assert(sample_count > 0, "atlas neighbor evidence samples were not attached")

            network_response = client.get(f"/api/jobs/{job_id}/network")
            _assert(network_response.status_code == 200, "cluster network endpoint failed")
            network = network_response.json()
            _assert("error" not in network, f"cluster network error: {network.get('error')}")
            _assert(network["nodes"], "cluster network has no nodes")

            term_response = client.get(f"/api/jobs/{job_id}/term-network?top_k=3&min_cooc=1")
            _assert(term_response.status_code == 200, "term network endpoint failed")
            term_network = term_response.json()
            _assert("error" not in term_network, f"term network error: {term_network.get('error')}")
            _assert(term_network["nodes"], "term network has no nodes")
            _assert(term_network["edges"], "term network has no edges")

            report_response = client.get(f"/api/jobs/{job_id}/view/landscape/report/report.html")
            _assert(report_response.status_code == 200, "report view endpoint failed")
            data_response = client.get(f"/api/jobs/{job_id}/download/landscape/report/data.json")
            _assert(data_response.status_code == 200, "data download endpoint failed")

            return {
                "status": "passed",
                "demo_key": demo["key"],
                "job_id": job_id,
                "network_levels": int(len(network["nodes"])),
                "term_network_nodes": int(len(term_network["nodes"])),
                "term_network_edges": int(len(term_network["edges"])),
                "edge_evidence_samples": sample_count,
            }
        finally:
            web_app._DEMO_MANIFEST_PATH = old_manifest
            web_app._LOCAL_DATA_ROOTS = old_roots
            web_app._jobs = old_jobs


def _write_query_to_atlas_smoke_inputs(result_root: Path) -> tuple[Path, Path]:
    """Create a tiny query-shaped corpus with two interpretable communities."""
    result_root.mkdir(parents=True, exist_ok=True)
    abstract_path = result_root / "abstracts.parquet"
    edge_path = result_root / "edges.parquet"

    pd.DataFrame(
        {
            "uid": [f"Q{i}" for i in range(8)],
            "title": [
                "Perovskite interface passivation",
                "Perovskite solar cell stability",
                "Perovskite defect transport",
                "Perovskite tandem photovoltaic devices",
                "Graph neural traffic forecasting",
                "Graph neural anomaly detection",
                "Temporal graph road networks",
                "Graph embedding node classification",
            ],
            "abstract": [
                "Interface passivation improves perovskite solar cell stability and defect tolerance.",
                "Perovskite solar cells use passivation layers for stable photovoltaic performance.",
                "Defect transport and ion migration affect perovskite device reliability.",
                "Tandem perovskite devices improve photovoltaic conversion efficiency.",
                "Graph neural networks forecast traffic over road sensor graphs.",
                "Graph neural networks detect anomalies in dynamic traffic systems.",
                "Temporal graph forecasting models capture road network flows.",
                "Graph embedding networks learn representations for node classification.",
            ],
            "pubyear": [2021, 2022, 2023, 2024, 2021, 2022, 2023, 2024],
        }
    ).to_parquet(abstract_path, index=False)

    edges: list[dict[str, Any]] = []
    for group in ([0, 1, 2, 3], [4, 5, 6, 7]):
        for left_idx, left in enumerate(group):
            for right in group[left_idx + 1 :]:
                edges.append({"uid1": f"Q{left}", "uid2": f"Q{right}", "rel_sum2": 2.0})
    edges.append({"uid1": "Q3", "uid2": "Q4", "rel_sum2": 0.05})
    pd.DataFrame(edges).to_parquet(edge_path, index=False)
    return edge_path, abstract_path


def _atlas_visual_smoke_render_payload() -> dict[str, Any]:
    return build_atlas_render_payload(
        {
            "schema_version": "sciscape_atlas_payload_v1",
            "levels": ["macro", "micro"],
            "nodes": [
                {
                    "cluster_uid": "macro:0",
                    "level": "macro",
                    "cluster_id": 0,
                    "label": "Energy materials",
                    "short_label": "Energy materials",
                    "doc_count": 24,
                    "child_count": 2,
                    "x": -80,
                    "y": -20,
                },
                {
                    "cluster_uid": "micro:0",
                    "level": "micro",
                    "cluster_id": 0,
                    "parent_uid": "macro:0",
                    "label": "Perovskite passivation",
                    "short_label": "Perovskite passivation",
                    "doc_count": 14,
                    "keyword_count": 6,
                    "x": -30,
                    "y": 35,
                },
                {
                    "cluster_uid": "micro:1",
                    "level": "micro",
                    "cluster_id": 1,
                    "parent_uid": "macro:0",
                    "label": "Graph neural forecasting",
                    "short_label": "Graph neural forecasting",
                    "doc_count": 12,
                    "keyword_count": 5,
                    "x": 75,
                    "y": -10,
                },
            ],
            "edges": [
                {
                    "source_uid": "micro:0",
                    "target_uid": "micro:1",
                    "level": "micro",
                    "weight": 2.5,
                    "edge_count": 3,
                    "relation_label": "cross-topic bridge",
                }
            ],
        }
    )


def _synthetic_atlas_semantic_payload(*, node_count: int = 100, edge_count: int = 500) -> dict[str, Any]:
    """Build a deterministic Atlas payload for renderer contract/perf gates."""
    _assert(node_count >= 2, "synthetic Atlas payload needs at least two nodes")
    macro_count = max(1, min(10, node_count // 20 or 1))
    micro_count = node_count - macro_count
    _assert(micro_count > 0, "synthetic Atlas payload needs at least one micro node")

    nodes: list[dict[str, Any]] = []
    macro_positions: list[tuple[float, float]] = []
    for index in range(macro_count):
        angle = (2.0 * math.pi * index) / max(1, macro_count)
        x = round(70.0 * math.cos(angle), 6)
        y = round(70.0 * math.sin(angle), 6)
        macro_positions.append((x, y))
        nodes.append(
            {
                "cluster_uid": f"macro:{index}",
                "level": "macro",
                "cluster_id": index,
                "label": f"Macro topic {index}",
                "short_label": f"Macro {index}",
                "doc_count": 120 + index * 7,
                "child_count": max(1, micro_count // macro_count),
                "x": x,
                "y": y,
            }
        )

    for index in range(micro_count):
        parent_index = index % macro_count
        parent_x, parent_y = macro_positions[parent_index]
        spoke = index // macro_count
        angle = (2.0 * math.pi * (spoke % 19)) / 19.0
        ring = 18.0 + (spoke % 5) * 9.0
        x = round(parent_x + ring * math.cos(angle), 6)
        y = round(parent_y + ring * math.sin(angle), 6)
        nodes.append(
            {
                "cluster_uid": f"micro:{index}",
                "level": "micro",
                "cluster_id": index,
                "parent_uid": f"macro:{parent_index}",
                "label": f"Synthetic research topic {index}",
                "short_label": f"Topic {index}",
                "doc_count": 12 + (index % 23),
                "keyword_count": 3 + (index % 9),
                "neighbor_count": 4 + (index % 13),
                "representative_work_count": 2 + (index % 5),
                "x": x,
                "y": y,
            }
        )

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for span in range(1, micro_count):
        for source_index in range(micro_count):
            target_index = (source_index + span) % micro_count
            left, right = sorted((source_index, target_index))
            pair = (left, right)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append(
                {
                    "source_uid": f"micro:{left}",
                    "target_uid": f"micro:{right}",
                    "level": "micro",
                    "weight": 1.0 + (len(edges) % 17) / 3.0,
                    "edge_count": 1 + (len(edges) % 9),
                    "relation_label": "synthetic relation",
                    "same_parent": left % macro_count == right % macro_count,
                    "shared_terms": [f"term-{len(edges) % 7}", f"term-{(len(edges) + 3) % 11}"],
                    "sample_count": len(edges) % 4,
                }
            )
            if len(edges) >= edge_count:
                break
        if len(edges) >= edge_count:
            break

    _assert(len(edges) == edge_count, f"synthetic Atlas payload could only create {len(edges)} unique edges")
    return {
        "schema_version": "sciscape_atlas_payload_v1",
        "levels": ["macro", "micro"],
        "nodes": nodes,
        "edges": edges,
        "warnings": [],
    }


def _run_atlas_render_contract_gate(
    *,
    node_count: int,
    edge_count: int,
    max_build_ms: float,
    max_payload_json_bytes: int,
) -> dict[str, Any]:
    semantic_payload = _synthetic_atlas_semantic_payload(node_count=node_count, edge_count=edge_count)
    started = time.perf_counter()
    render_payload = build_atlas_render_payload(semantic_payload)
    build_ms = (time.perf_counter() - started) * 1000.0
    payload_json = json.dumps(render_payload, ensure_ascii=True, separators=(",", ":"))
    payload_json_bytes = len(payload_json.encode("utf-8"))
    macro_count = max(1, min(10, node_count // 20 or 1))

    _assert(
        render_payload["schema_version"] == "sciscape_atlas_render_payload_v1",
        "Atlas render contract gate schema mismatch",
    )
    _assert(render_payload["node_count"] == node_count, "Atlas render contract gate node count mismatch")
    _assert(render_payload["edge_count"] == edge_count, "Atlas render contract gate edge count mismatch")
    _assert(render_payload["label_count"] == node_count, "Atlas render contract gate label count mismatch")
    _assert(
        render_payload["hierarchy_edge_count"] >= node_count - macro_count,
        "Atlas render contract gate hierarchy rows were not generated",
    )
    _assert(
        render_payload.get("view", {}).get("coordinate_source") == "node_coordinates",
        "Atlas render contract gate should preserve supplied node coordinates",
    )
    render_layers = render_payload.get("layers", {})
    _assert(
        render_layers.get("nodes", {}).get("recommended_deck_layer") == "ScatterplotLayer",
        "Atlas render contract gate node layer mismatch",
    )
    _assert(
        render_layers.get("edges", {}).get("recommended_deck_layer") == "LineLayer",
        "Atlas render contract gate edge layer mismatch",
    )
    _assert(
        render_layers.get("labels", {}).get("recommended_deck_layer") == "TextLayer",
        "Atlas render contract gate label layer mismatch",
    )
    _assert(build_ms < max_build_ms, f"Atlas render payload build too slow: {build_ms:.3f}ms")
    _assert(
        payload_json_bytes < max_payload_json_bytes,
        f"Atlas render payload too large: {payload_json_bytes} bytes",
    )

    return {
        "status": "passed",
        "nodes": int(render_payload["node_count"]),
        "edges": int(render_payload["edge_count"]),
        "labels": int(render_payload["label_count"]),
        "hierarchy_edges": int(render_payload["hierarchy_edge_count"]),
        "coordinate_source": render_payload["view"]["coordinate_source"],
        "build_ms": round(build_ms, 3),
        "payload_json_bytes": payload_json_bytes,
        "recommended_layers": {
            "nodes": render_layers["nodes"]["recommended_deck_layer"],
            "edges": render_layers["edges"]["recommended_deck_layer"],
            "labels": render_layers["labels"]["recommended_deck_layer"],
        },
    }


def run_atlas_render_perf_smoke_gate() -> dict[str, Any]:
    """Validate renderer payload construction at a CI-scale Atlas map size."""
    return _run_atlas_render_contract_gate(
        node_count=100,
        edge_count=500,
        max_build_ms=1000.0,
        max_payload_json_bytes=2_000_000,
    )


def run_atlas_render_scale_smoke_gate() -> dict[str, Any]:
    """Validate renderer payload construction at a small-demo Atlas map size."""
    return _run_atlas_render_contract_gate(
        node_count=5000,
        edge_count=25000,
        max_build_ms=5000.0,
        max_payload_json_bytes=50_000_000,
    )


def _write_atlas_visual_smoke_html(path: Path, payload: dict[str, Any]) -> None:
    payload_json = json.dumps(payload, ensure_ascii=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SciScape Atlas visual smoke</title>
<script src="https://unpkg.com/deck.gl@^9.0.0/dist.min.js"></script>
<style>
html, body, #stage {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #f9fafc; }}
</style>
</head>
<body>
<div id="stage"></div>
<script>
const payload = {payload_json};
const nodes = payload.layers.nodes.rows;
const edges = payload.layers.edges.rows;
const labels = payload.layers.labels.rows;
const deckgl = new deck.Deck({{
  parent: document.getElementById('stage'),
  views: [new deck.OrthographicView()],
  initialViewState: {{target: [0, 0, 0], zoom: 1.35, minZoom: -4, maxZoom: 8}},
  controller: false,
  parameters: {{clearColor: [0.976, 0.98, 0.988, 1]}},
  useDevicePixels: 1,
  glOptions: {{preserveDrawingBuffer: true}},
  layers: [
    new deck.LineLayer({{
      id: 'smoke-edges',
      data: edges,
      getSourcePosition: d => d.source_position,
      getTargetPosition: d => d.target_position,
      getColor: [232, 168, 56, 210],
      getWidth: 4,
      widthUnits: 'pixels'
    }}),
    new deck.ScatterplotLayer({{
      id: 'smoke-nodes',
      data: nodes,
      getPosition: d => d.position,
      getRadius: d => Math.max(10, d.render_radius * 1.8),
      radiusUnits: 'pixels',
      stroked: true,
      filled: true,
      getFillColor: d => d.level === 'macro' ? [43, 168, 156, 230] : [74, 89, 118, 230],
      getLineColor: [255, 255, 255, 255],
      getLineWidth: 2,
      lineWidthUnits: 'pixels'
    }}),
    new deck.TextLayer({{
      id: 'smoke-labels',
      data: labels,
      getPosition: d => d.position,
      getText: d => d.text,
      getSize: 13,
      getColor: [10, 15, 26, 230],
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'top',
      getPixelOffset: [0, 12],
      sizeUnits: 'pixels'
    }})
  ],
  onAfterRender: () => {{ document.body.dataset.rendered = 'true'; }}
}});
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_atlas_interaction_smoke_html(path: Path, payload: dict[str, Any]) -> None:
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SciScape Atlas interaction smoke</title>
<script src="https://unpkg.com/deck.gl@^9.0.0/dist.min.js"></script>
<style>
html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #f9fafc; }}
#stage {{ width: 100vw; height: 100vh; background:
  linear-gradient(90deg, rgba(197, 207, 224, 0.18) 1px, transparent 1px),
  linear-gradient(180deg, rgba(197, 207, 224, 0.18) 1px, transparent 1px),
  #f9fafc; background-size: 28px 28px; }}
#result {{ display: none; }}
</style>
</head>
<body data-interaction-status="booting">
<div id="stage"></div>
<pre id="result"></pre>
<script type="application/json" id="payload">{payload_json}</script>
<script>
const startedAt = performance.now();
function report(payload) {{
  const encoded = btoa(JSON.stringify(payload));
  document.body.dataset.result = encoded;
  document.body.dataset.interactionStatus = payload.status || 'reported';
  document.getElementById('result').textContent = encoded;
}}
try {{
  const payloadEl = document.getElementById('payload');
  const payload = JSON.parse(payloadEl.textContent);
  payloadEl.textContent = '';
  const stage = document.getElementById('stage');
  const nodes = payload.layers.nodes.rows;
  const edges = payload.layers.edges.rows;
  const labels = payload.layers.labels.rows
    .slice()
    .sort((a, b) => Number(b.priority || 0) - Number(a.priority || 0))
    .slice(0, 160);
  const selected = nodes[nodes.length - 1] || nodes[0];
  let renderCount = 0;
  let completed = false;
  const selectedPosition = selected && Array.isArray(selected.position) ? selected.position : [0, 0];
  const deckgl = new deck.Deck({{
    parent: stage,
    views: [new deck.OrthographicView()],
    initialViewState: {{target: [0, 0, 0], zoom: -1.25, minZoom: -6, maxZoom: 8}},
    controller: true,
    parameters: {{clearColor: [0.976, 0.98, 0.988, 1]}},
    useDevicePixels: 1,
    glOptions: {{preserveDrawingBuffer: true}},
    layers: [
      new deck.LineLayer({{
        id: 'interaction-edges',
        data: edges,
        getSourcePosition: d => d.source_position,
        getTargetPosition: d => d.target_position,
        getColor: d => d.same_parent ? [43, 168, 156, 90] : [232, 168, 56, 82],
        getWidth: d => Math.max(1, Math.min(6, Number(d.render_width || 1))),
        widthUnits: 'pixels',
        pickable: true
      }}),
      new deck.ScatterplotLayer({{
        id: 'interaction-nodes',
        data: nodes,
        getPosition: d => d.position,
        getRadius: d => d.cluster_uid === selected.cluster_uid ? 18 : Math.max(5, Number(d.render_radius || 6)),
        radiusUnits: 'pixels',
        radiusMinPixels: 3,
        radiusMaxPixels: 36,
        stroked: true,
        filled: true,
        getFillColor: d => d.cluster_uid === selected.cluster_uid ? [10, 15, 26, 245] :
          (d.level === 'macro' ? [43, 168, 156, 215] : [74, 89, 118, 180]),
        getLineColor: [255, 255, 255, 235],
        getLineWidth: d => d.cluster_uid === selected.cluster_uid ? 2.5 : 1,
        lineWidthUnits: 'pixels',
        pickable: true
      }}),
      new deck.TextLayer({{
        id: 'interaction-labels',
        data: labels,
        getPosition: d => d.position,
        getText: d => d.text,
        getSize: d => d.cluster_uid === selected.cluster_uid ? 13 : 10,
        getColor: [10, 15, 26, 220],
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'bottom',
        getPixelOffset: [0, -8],
        sizeUnits: 'pixels',
        pickable: false
      }})
    ],
    onAfterRender: () => {{
      renderCount += 1;
      if (renderCount === 1) {{
        deckgl.setProps({{viewState: {{target: [selectedPosition[0], selectedPosition[1], 0], zoom: 0.45, minZoom: -6, maxZoom: 8}}}});
        return;
      }}
      if (renderCount < 2 || completed) return;
      completed = true;
      try {{
        const pick = deckgl.pickObject({{x: Math.round(stage.clientWidth / 2), y: Math.round(stage.clientHeight / 2), radius: 36}});
        const pickedUid = pick && pick.object && (pick.object.cluster_uid || pick.object.id || '');
        const status = pickedUid === selected.cluster_uid ? 'passed' : 'failed';
        report({{
          status,
          renderCount,
          selectedUid: selected.cluster_uid,
          pickedUid: pickedUid || '',
          nodes: nodes.length,
          edges: edges.length,
          labels: labels.length,
          cameraTarget: selectedPosition,
          elapsedMs: Math.round(performance.now() - startedAt)
        }});
      }} catch (error) {{
        report({{
          status: 'failed',
          reason: 'pickObject failed: ' + String(error && error.message || error),
          renderCount,
          selectedUid: selected.cluster_uid,
          pickedUid: '',
          nodes: nodes.length,
          edges: edges.length,
          labels: labels.length,
          cameraTarget: selectedPosition,
          elapsedMs: Math.round(performance.now() - startedAt)
        }});
      }}
    }}
  }});
  window.setTimeout(() => {{
    if (!document.body.dataset.result) {{
      report({{
        status: 'failed',
        reason: 'timeout before selected-node hit-test completed',
        renderCount,
        selectedUid: selected && selected.cluster_uid || '',
        pickedUid: '',
        nodes: nodes.length,
        edges: edges.length,
        labels: labels.length,
        elapsedMs: Math.round(performance.now() - startedAt)
      }});
    }}
  }}, 14000);
}} catch (error) {{
  report({{status: 'failed', reason: String(error && error.message || error), elapsedMs: Math.round(performance.now() - startedAt)}});
}}
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _atlas_inspector_smoke_result() -> dict[str, Any]:
    atlas = {
        "schema_version": "sciscape_atlas_payload_v1",
        "levels": ["macro", "micro"],
        "node_count": 4,
        "edge_count": 2,
        "warnings": [],
        "nodes": [
            {
                "cluster_uid": "macro:0",
                "level": "macro",
                "cluster_id": 0,
                "label": "Energy and network science",
                "short_label": "Energy networks",
                "doc_count": 52,
                "child_count": 3,
                "keyword_count": 4,
            },
            {
                "cluster_uid": "micro:0",
                "level": "micro",
                "cluster_id": 0,
                "parent_uid": "macro:0",
                "label": "Perovskite passivation",
                "short_label": "Perovskite passivation",
                "doc_count": 18,
                "doc_count_source": "membership:cluster+abstracts",
                "keyword_count": 4,
                "neighbor_count": 2,
                "representative_work_count": 2,
                "keywords": [
                    {"term": "perovskite solar cell", "score": 0.94, "rank": 1},
                    {"term": "surface passivation", "score": 0.83, "rank": 2},
                    {"term": "stability", "score": 0.72, "rank": 3},
                ],
                "representative_works": [
                    {
                        "uid": "W0",
                        "title": "Stable perovskite solar cells through surface passivation",
                        "year": 2023,
                        "cited_by_count": 42,
                    },
                    {
                        "uid": "W1",
                        "title": "Interface engineering for high efficiency perovskite devices",
                        "year": 2022,
                        "cited_by_count": 37,
                    },
                ],
                "representative_works_source": "abstracts",
                "lineage": [
                    {"cluster_uid": "macro:0", "level": "macro", "short_label": "Energy networks"},
                    {"cluster_uid": "micro:0", "level": "micro", "short_label": "Perovskite passivation"},
                ],
                "neighbors": [
                    {
                        "cluster_uid": "micro:1",
                        "label": "Graph neural forecasting",
                        "short_label": "Graph forecasting",
                        "level": "micro",
                        "weight": 3.25,
                        "edge_count": 4,
                        "shared_terms": ["graph neural network", "forecasting"],
                        "same_parent": False,
                        "relation_label": "cross-cluster",
                        "sample_count": 1,
                        "samples": [
                            {
                                "source_title": "Stable perovskite solar cells through surface passivation",
                                "target_title": "Graph neural networks for scientific trend forecasting",
                                "edge_type": "citation",
                                "weight": 1.5,
                            }
                        ],
                    },
                    {
                        "cluster_uid": "micro:2",
                        "label": "Catalyst synthesis",
                        "short_label": "Catalyst synthesis",
                        "level": "micro",
                        "weight": 1.5,
                        "edge_count": 2,
                        "shared_terms": [],
                        "same_parent": True,
                        "relation_label": "same-parent",
                        "sample_count": 0,
                        "samples": [],
                    },
                ],
            },
            {
                "cluster_uid": "micro:1",
                "level": "micro",
                "cluster_id": 1,
                "parent_uid": "macro:0",
                "label": "Graph neural forecasting",
                "short_label": "Graph forecasting",
                "doc_count": 14,
                "keyword_count": 3,
                "neighbor_count": 1,
                "keywords": [{"term": "graph neural network", "score": 0.91, "rank": 1}],
                "lineage": [
                    {"cluster_uid": "macro:0", "level": "macro", "short_label": "Energy networks"},
                    {"cluster_uid": "micro:1", "level": "micro", "short_label": "Graph forecasting"},
                ],
                "neighbors": [
                    {
                        "cluster_uid": "micro:0",
                        "label": "Perovskite passivation",
                        "short_label": "Perovskite passivation",
                        "level": "micro",
                        "weight": 3.25,
                        "edge_count": 4,
                        "shared_terms": ["graph neural network", "forecasting"],
                        "same_parent": False,
                        "relation_label": "cross-cluster",
                        "sample_count": 1,
                    }
                ],
            },
            {
                "cluster_uid": "micro:2",
                "level": "micro",
                "cluster_id": 2,
                "parent_uid": "macro:0",
                "label": "Catalyst synthesis",
                "short_label": "Catalyst synthesis",
                "doc_count": 10,
                "keyword_count": 3,
                "neighbor_count": 1,
                "keywords": [{"term": "catalyst synthesis", "score": 0.88, "rank": 1}],
                "lineage": [
                    {"cluster_uid": "macro:0", "level": "macro", "short_label": "Energy networks"},
                    {"cluster_uid": "micro:2", "level": "micro", "short_label": "Catalyst synthesis"},
                ],
                "neighbors": [
                    {
                        "cluster_uid": "micro:0",
                        "label": "Perovskite passivation",
                        "short_label": "Perovskite passivation",
                        "level": "micro",
                        "weight": 1.5,
                        "edge_count": 2,
                        "shared_terms": [],
                        "same_parent": True,
                        "relation_label": "same-parent",
                        "sample_count": 0,
                    }
                ],
            },
        ],
    }
    return {
        "result_state": "loaded",
        "features": {
            "cluster_map": True,
            "keyword": True,
            "term_network": True,
            "cooccurrence": True,
            "evidence": True,
            "quality": True,
            "export": True,
        },
        "feature_states": {
            "overview": "stable",
            "cluster_map": "stable",
            "keyword": "stable",
            "term_network": "stable",
            "cooccurrence": "stable",
            "evidence": "stable",
            "quality": "stable",
            "export": "stable",
            "temporal": "hidden",
            "evolution": "hidden",
            "narrative": "hidden",
        },
        "artifact_contract": {
            "ok": True,
            "counts": {
                "abstract_rows": 42,
                "membership_rows": 42,
                "keyword_rows": 12,
                "cooccurrence_rows": 9,
                "edge_evidence_artifacts": 1,
            },
            "features": {
                "cluster_map": True,
                "keyword": True,
                "term_network": True,
                "cooccurrence": True,
                "evidence": True,
                "quality": True,
            },
            "warnings": [],
        },
        "atlas": atlas,
    }


def _write_atlas_inspector_smoke_html(path: Path, result: dict[str, Any]) -> None:
    source = Path("sciscape/web/static/index.html").read_text(encoding="utf-8")
    source = source.replace(
        '<script src="https://d3js.org/d3.v7.min.js"></script>',
        "<script>window.d3 = window.d3 || {};</script>",
    )
    source = source.replace(
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
        "<script>window.Plotly = window.Plotly || {};</script>",
    )
    source = source.replace(
        '<script src="https://unpkg.com/deck.gl@^9.0.0/dist.min.js"></script>',
        "<script>window.deck = window.deck || {};</script>",
    )
    source = source.replace(
        "@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');\n\n",
        "",
    )
    result_json = json.dumps(result, ensure_ascii=True, separators=(",", ":"))
    harness = f"""
<script>
(function() {{
  const startedAt = performance.now();
  function report(payload) {{
    const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
    document.body.dataset.result = encoded;
    document.body.dataset.inspectorStatus = payload.status || 'reported';
  }}
  function fail(reason) {{
    report({{status: 'failed', reason: String(reason), elapsedMs: Math.round(performance.now() - startedAt)}});
  }}
  function assertSmoke(condition, reason) {{
    if (!condition) throw new Error(reason);
  }}
  function text(selector) {{
    const el = document.querySelector(selector);
    return el ? el.textContent : '';
  }}
  window.addEventListener('load', function() {{
    try {{
      const result = {result_json};
      const node = result.atlas.nodes.find(row => row.cluster_uid === 'micro:0');
      assertSmoke(node, 'fixture node micro:0 missing');
      currentJobId = 'atlas-inspector-smoke';
      currentAtlasLevel = 'micro';
      currentAtlasNodeUid = 'micro:0';
      currentAtlasNeighborUid = 'micro:1';
      currentAtlasViewMode = 'evidence';
      currentAtlasFocusMode = 'global';
      atlasUrlStateApplied = true;
      renderAtlasShell(result);
      const shell = document.getElementById('atlas-shell');
      assertSmoke(shell && shell.classList.contains('active'), 'Atlas shell did not become active');
      const model = buildAtlasInspectorEvidenceModel(node);
      assertSmoke(model.schema_version === 'sciscape_inspector_evidence_view_v1', 'inspector schema mismatch');
      assertSmoke(model.sections.relations.state === 'stable', 'relations section is not stable');
      assertSmoke(model.sections.works.state === 'stable', 'works section is not stable');
      assertSmoke(model.sections.qa.state === 'stable', 'qa section is not stable');
      assertSmoke(text('.atlas-inspector-title').includes('Perovskite'), 'selected inspector title missing');
      assertSmoke(text('.atlas-review-queue-title').includes('Review queue'), 'review queue title missing');
      assertSmoke(text('.atlas-review-queue-counts').includes('review 2'), 'review queue counts missing review targets');
      assertSmoke(document.querySelectorAll('.atlas-review-queue-row.review').length >= 1, 'review queue rows missing');
      const reviewFilterButton = document.querySelector('.atlas-review-queue-count.review');
      assertSmoke(reviewFilterButton, 'review queue filter button missing');
      reviewFilterButton.click();
      assertSmoke(currentAtlasReviewFilter === 'review', 'review filter did not update state');
      assertSmoke(new URL(window.location.href).searchParams.get('atlas_review') === 'review', 'review filter did not sync URL');
      assertSmoke(document.querySelectorAll('.atlas-evidence-node-row').length === 2, 'review filter did not narrow evidence rows');
      assertSmoke(document.querySelector('.atlas-review-queue-count.review.active'), 'review filter active marker missing');
      const nextTargetButton = document.querySelector('.atlas-review-queue-next');
      assertSmoke(nextTargetButton && !nextTargetButton.disabled, 'review queue next target button missing');
      assertSmoke(text('.atlas-review-title').includes('Cluster reading ready'), 'review checklist ready state missing');
      assertSmoke(text('.atlas-review-flags').includes('sample-backed'), 'review checklist sample-backed flag missing');
      assertSmoke(text('.atlas-review-packet-title').includes('Review packet'), 'review packet title missing');
      assertSmoke(text('.atlas-review-packet').includes('perovskite solar cell'), 'review packet meaning missing');
      assertSmoke(text('.atlas-review-packet').includes('Stable perovskite solar cells'), 'review packet works missing');
      assertSmoke(text('.atlas-review-packet').includes('1 samples'), 'review packet sampled relation missing');
      const reviewQueueRowsSeen = document.querySelectorAll('.atlas-review-queue-row').length;
      assertSmoke(text('.atlas-neighbor-evidence-status').includes('1 samples'), 'sample-backed neighbor status missing');
      assertSmoke(document.querySelectorAll('.atlas-neighbor-sample-row').length >= 1, 'neighbor sample row missing');
      const sampleRowsSeen = document.querySelectorAll('.atlas-neighbor-sample-row').length;
      const initialReviewState = text('.atlas-review-panel .atlas-review-state').trim();
      assertSmoke(text('#atlas-shell').includes('Stable perovskite solar cells'), 'representative work title missing');
      selectAtlasNeighborEvidence('micro:2');
      assertSmoke(currentAtlasNeighborUid === 'micro:2', 'neighbor selection did not update state');
      assertSmoke(text('.atlas-neighbor-evidence-status').includes('aggregate only'), 'aggregate-only status missing');
      assertSmoke(text('#atlas-shell').includes('Raw pair samples unavailable'), 'aggregate-only explanation missing');
      assertSmoke(text('.atlas-review-flags').includes('aggregate-only'), 'review checklist aggregate-only flag missing');
      assertSmoke(text('.atlas-review-packet').includes('aggregate only'), 'review packet aggregate fallback missing');
      assertSmoke(text('#atlas-shell').includes('same-parent'), 'aggregate relation label missing');
      const activeNextTargetButton = document.querySelector('.atlas-review-queue-next');
      assertSmoke(activeNextTargetButton && !activeNextTargetButton.disabled, 'review queue next target button missing after rerender');
      activeNextTargetButton.click();
      assertSmoke(currentAtlasNodeUid === 'micro:1' || currentAtlasNodeUid === 'micro:2', 'next target did not select a review node');
      assertSmoke(currentAtlasReviewFilter === 'review', 'next target changed review filter');
      const nextTargetUid = currentAtlasNodeUid;
      assertSmoke(!currentAtlasNeighborUid, 'neighbor state was not reset after node selection');
      assertSmoke(text('.atlas-inspector-title').includes('Graph'), 'selected node title did not update');
      assertSmoke(text('.atlas-review-title').includes('Cluster reading review'), 'review checklist partial state missing');
      report({{
        status: 'passed',
        schema: model.schema_version,
        selectedUid: currentAtlasNodeUid,
        relationState: model.sections.relations.state,
        worksState: model.sections.works.state,
        qaState: model.sections.qa.state,
        initialReviewState: initialReviewState,
        selectedReviewState: text('.atlas-review-panel .atlas-review-state').trim(),
        reviewQueueRows: reviewQueueRowsSeen,
        reviewFilter: currentAtlasReviewFilter,
        nextTargetUid: nextTargetUid,
        reviewPacketSeen: !!document.querySelector('.atlas-review-packet.review'),
        filteredEvidenceRows: document.querySelectorAll('.atlas-evidence-node-row').length,
        sampleRows: sampleRowsSeen,
        aggregateFallbackSeen: true,
        elapsedMs: Math.round(performance.now() - startedAt)
      }});
    }} catch (error) {{
      fail(error && error.message || error);
    }}
  }});
  window.setTimeout(function() {{
    if (!document.body.dataset.result) fail('timeout before inspector smoke completed');
  }}, 6000);
}})();
</script>
"""
    path.write_text(source.replace("</body>", harness + "\n</body>"), encoding="utf-8")


def _find_headless_browser() -> str | None:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _png_rgba_rows(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    _assert(data.startswith(b"\x89PNG\r\n\x1a\n"), "screenshot is not a PNG file")
    offset = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    _assert(width and height and bit_depth == 8 and color_type in {2, 6}, "unsupported screenshot PNG format")
    channels = 4 if color_type == 6 else 3
    row_len = int(width) * channels
    raw = zlib.decompress(bytes(idat))
    rows = bytearray()
    prev = bytearray(row_len)
    idx = 0
    for _ in range(int(height)):
        filter_type = raw[idx]
        idx += 1
        row = bytearray(raw[idx : idx + row_len])
        idx += row_len
        for i, value in enumerate(row):
            left = row[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 1:
                row[i] = (value + left) & 0xFF
            elif filter_type == 2:
                row[i] = (value + up) & 0xFF
            elif filter_type == 3:
                row[i] = (value + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                predictor = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                row[i] = (value + predictor) & 0xFF
            elif filter_type != 0:
                raise AssertionError(f"unsupported PNG filter type: {filter_type}")
        if channels == 3:
            expanded = bytearray()
            for i in range(0, len(row), 3):
                expanded.extend((row[i], row[i + 1], row[i + 2], 255))
            rows.extend(expanded)
        else:
            rows.extend(row)
        prev = row
    return int(width), int(height), bytes(rows)


def _non_background_pixel_ratio(path: Path, background: tuple[int, int, int] = (249, 250, 252)) -> float:
    width, height, rgba = _png_rgba_rows(path)
    colored = 0
    total = width * height
    for i in range(0, len(rgba), 4):
        r, g, b, a = rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]
        if a and max(abs(r - background[0]), abs(g - background[1]), abs(b - background[2])) > 8:
            colored += 1
    return colored / max(1, total)


def _extract_browser_data_result(dom: str) -> dict[str, Any]:
    match = re.search(r'data-result="([^"]+)"', dom)
    _assert(match is not None, "headless browser did not expose data-result")
    encoded = html.unescape(match.group(1))
    decoded = base64.b64decode(encoded).decode("utf-8")
    return json.loads(decoded)


def run_atlas_visual_smoke_gate() -> dict[str, Any]:
    browser = _find_headless_browser()
    if browser is None:
        return {"status": "skipped", "reason": "no google-chrome/chromium executable found"}
    with tempfile.TemporaryDirectory(prefix="sciscape_atlas_visual_gate_") as tmp:
        root = Path(tmp)
        html_path = root / "atlas_visual_smoke.html"
        screenshot_path = root / "atlas_visual_smoke.png"
        payload = _atlas_visual_smoke_render_payload()
        _write_atlas_visual_smoke_html(html_path, payload)
        cmd = [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=swiftshader",
            "--enable-unsafe-swiftshader",
            "--window-size=900,600",
            "--virtual-time-budget=7000",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            html_path.as_uri(),
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        _assert(completed.returncode == 0, f"headless browser visual smoke failed: {completed.stderr[-500:]}")
        _assert(screenshot_path.exists(), "headless browser did not write screenshot")
        ratio = _non_background_pixel_ratio(screenshot_path)
        _assert(ratio > 0.001, f"deck.gl visual smoke appears blank: non-background pixel ratio={ratio:.6f}")
        return {
            "status": "passed",
            "browser": browser,
            "nodes": int(payload["node_count"]),
            "edges": int(payload["edge_count"]),
            "labels": int(payload["label_count"]),
            "non_background_pixel_ratio": round(ratio, 6),
        }


def run_atlas_interaction_smoke_gate() -> dict[str, Any]:
    browser = _find_headless_browser()
    if browser is None:
        return {"status": "skipped", "reason": "no google-chrome/chromium executable found"}
    with tempfile.TemporaryDirectory(prefix="sciscape_atlas_interaction_gate_") as tmp:
        root = Path(tmp)
        html_path = root / "atlas_interaction_smoke.html"
        screenshot_path = root / "atlas_interaction_smoke.png"
        semantic_payload = _synthetic_atlas_semantic_payload(node_count=5000, edge_count=25000)
        payload = build_atlas_render_payload(semantic_payload)
        _write_atlas_interaction_smoke_html(html_path, payload)
        cmd = [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=swiftshader",
            "--enable-unsafe-swiftshader",
            "--window-size=1100,760",
            "--virtual-time-budget=22000",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            "--dump-dom",
            html_path.as_uri(),
        ]
        started = time.perf_counter()
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=70)
        browser_ms = (time.perf_counter() - started) * 1000.0
        _assert(completed.returncode == 0, f"headless browser interaction smoke failed: {completed.stderr[-500:]}")
        browser_result = _extract_browser_data_result(completed.stdout)
        _assert(browser_result.get("status") == "passed", f"deck.gl interaction smoke failed: {browser_result}")
        _assert(screenshot_path.exists(), "headless browser did not write interaction screenshot")
        ratio = _non_background_pixel_ratio(screenshot_path)
        _assert(ratio > 0.001, f"deck.gl interaction smoke appears blank: non-background pixel ratio={ratio:.6f}")
        return {
            "status": "passed",
            "browser": browser,
            "nodes": int(browser_result.get("nodes") or 0),
            "edges": int(browser_result.get("edges") or 0),
            "labels": int(browser_result.get("labels") or 0),
            "render_count": int(browser_result.get("renderCount") or 0),
            "selected_uid": str(browser_result.get("selectedUid") or ""),
            "picked_uid": str(browser_result.get("pickedUid") or ""),
            "browser_elapsed_ms": round(browser_ms, 3),
            "deck_elapsed_ms": int(browser_result.get("elapsedMs") or 0),
            "non_background_pixel_ratio": round(ratio, 6),
        }


def run_atlas_inspector_smoke_gate() -> dict[str, Any]:
    """Open the static web app in a browser and exercise the Atlas inspector."""
    browser = _find_headless_browser()
    if browser is None:
        return {"status": "skipped", "reason": "no google-chrome/chromium executable found"}
    with tempfile.TemporaryDirectory(prefix="sciscape_atlas_inspector_gate_") as tmp:
        root = Path(tmp)
        html_path = root / "atlas_inspector_smoke.html"
        screenshot_path = root / "atlas_inspector_smoke.png"
        result = _atlas_inspector_smoke_result()
        _write_atlas_inspector_smoke_html(html_path, result)
        cmd = [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--window-size=1200,900",
            "--virtual-time-budget=9000",
            "--run-all-compositor-stages-before-draw",
            f"--screenshot={screenshot_path}",
            "--dump-dom",
            html_path.as_uri(),
        ]
        started = time.perf_counter()
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=45)
        browser_ms = (time.perf_counter() - started) * 1000.0
        _assert(completed.returncode == 0, f"headless browser inspector smoke failed: {completed.stderr[-500:]}")
        browser_result = _extract_browser_data_result(completed.stdout)
        _assert(browser_result.get("status") == "passed", f"Atlas inspector browser smoke failed: {browser_result}")
        _assert(screenshot_path.exists(), "headless browser did not write inspector screenshot")
        ratio = _non_background_pixel_ratio(screenshot_path)
        _assert(ratio > 0.001, f"Atlas inspector smoke appears blank: non-background pixel ratio={ratio:.6f}")
        return {
            "status": "passed",
            "browser": browser,
            "schema": str(browser_result.get("schema") or ""),
            "selected_uid": str(browser_result.get("selectedUid") or ""),
            "relation_state": str(browser_result.get("relationState") or ""),
            "works_state": str(browser_result.get("worksState") or ""),
            "qa_state": str(browser_result.get("qaState") or ""),
            "initial_review_state": str(browser_result.get("initialReviewState") or ""),
            "selected_review_state": str(browser_result.get("selectedReviewState") or ""),
            "review_queue_rows": int(browser_result.get("reviewQueueRows") or 0),
            "review_filter": str(browser_result.get("reviewFilter") or ""),
            "next_target_uid": str(browser_result.get("nextTargetUid") or ""),
            "review_packet_seen": bool(browser_result.get("reviewPacketSeen")),
            "filtered_evidence_rows": int(browser_result.get("filteredEvidenceRows") or 0),
            "sample_rows": int(browser_result.get("sampleRows") or 0),
            "aggregate_fallback_seen": bool(browser_result.get("aggregateFallbackSeen")),
            "browser_elapsed_ms": round(browser_ms, 3),
            "dom_elapsed_ms": int(browser_result.get("elapsedMs") or 0),
            "non_background_pixel_ratio": round(ratio, 6),
        }


def run_p1_atlas_smoke_gate() -> dict[str, Any]:
    """Run a tiny full pipeline result and reopen it through the web Atlas API."""
    with tempfile.TemporaryDirectory(prefix="sciscape_p1_atlas_gate_") as tmp:
        root = Path(tmp)
        local_root = root / "workspace" / "examples_output"
        result_root = local_root / "query_to_atlas_smoke"
        landscape_dir = result_root / "landscape"
        edge_path, abstract_path = _write_query_to_atlas_smoke_inputs(result_root)

        pipeline_result = run_landscape(
            edge_path,
            abstract_path,
            landscape_dir,
            config=LandscapeConfig(
                force=True,
                gamma_pre=None,
                gamma_range=(1e-4, 1e-1),
                min_docs_per_cluster=2,
                n_hierarchy_levels=1,
                leiden_iterations=5,
                min_df_unigram=1,
                min_df_phrase=1,
                top_n_unigrams=30,
                top_n_keywords=20,
                ngram_range=(1, 3),
                n_jobs=1,
                edge_evidence_max_relations=20,
                edge_evidence_samples_per_relation=2,
                report_title="SciScape P1 Query-to-Atlas Smoke",
            ),
        )

        artifact_result = write_artifact_contract(result_root)
        manifest = write_result_manifest(
            result_root,
            mode="demo",
            source_overrides={
                "query": "synthetic query-to-atlas smoke",
                "max_works": 8,
                "input_kind": "synthetic_fixture",
            },
        )
        contract_path = default_artifact_contract_path(artifact_result)
        report_data_path = landscape_dir / "report" / "data.json"
        _assert(report_data_path.exists(), "pipeline did not write report data")
        _assert(artifact_result.ok, "pipeline result artifact contract is blocked")
        _assert(
            artifact_result.counts.get("cooccurrence_rows", 0) > 0,
            "pipeline result did not write co-occurrence artifact rows",
        )

        from fastapi.testclient import TestClient
        import sciscape.web.app as web_app
        from sciscape.web.jobstore import JobStore

        old_roots = web_app._LOCAL_DATA_ROOTS
        old_jobs = web_app._jobs
        try:
            web_app._LOCAL_DATA_ROOTS = [local_root]
            web_app._jobs = JobStore(root / "jobs.db")

            client = TestClient(web_app.app)
            open_response = client.post("/api/local-data/open", json={"path": str(report_data_path)})
            _assert(open_response.status_code == 200, "open full pipeline result endpoint failed")
            job_id = open_response.json()["job_id"]

            job_response = client.get(f"/api/jobs/{job_id}")
            _assert(job_response.status_code == 200, "pipeline job status endpoint failed")
            job = job_response.json()
            _assert(job["status"] == "done", "pipeline result was not marked done")
            result = job["result"]
            _assert(result["landscape_rel_path"] == "landscape", "pipeline landscape rel path mismatch")
            _assert(result.get("artifact_contract", {}).get("ok") is True, "web artifact contract is blocked")
            loaded_manifest = result.get("result_manifest", {})
            _assert(
                loaded_manifest.get("manifest_state") == "present",
                "web result manifest was not loaded from the pipeline fixture",
            )
            _assert(
                loaded_manifest.get("manifest_path") == "result_manifest.json",
                "web result manifest path was not canonical",
            )

            feature_states = result.get("feature_states", {})
            for feature in ("cluster_map", "keyword", "term_network", "cooccurrence", "evidence", "export"):
                _assert(
                    feature_states.get(feature) in {"stable", "beta"},
                    f"{feature} was not exposed by the pipeline manifest",
                )
            _assert(
                feature_states.get("cooccurrence") == "stable",
                "co-occurrence was not backed by a stable artifact",
            )
            _assert(
                feature_states.get("evidence") == "stable",
                "evidence was not backed by stable representative-work or edge-evidence artifacts",
            )

            atlas = result.get("atlas", {})
            atlas_nodes = [node for node in atlas.get("nodes", []) if isinstance(node, dict)]
            _assert(atlas_nodes, "pipeline Atlas payload has no nodes")
            neighbor_rows = [
                neighbor
                for node in atlas_nodes
                for neighbor in node.get("neighbors", [])
                if isinstance(neighbor, dict)
            ]
            _assert(neighbor_rows, "pipeline Atlas payload has no neighbor rows")
            aggregate_contract_rows = [
                neighbor
                for neighbor in neighbor_rows
                if str(neighbor.get("cluster_uid") or "").strip()
                and str(neighbor.get("relation_label") or "").strip()
                and neighbor.get("weight") is not None
                and neighbor.get("edge_count") is not None
            ]
            _assert(
                len(aggregate_contract_rows) == len(neighbor_rows),
                "pipeline Atlas neighbor rows are missing aggregate relation fields",
            )
            sampled_neighbor_rows = [
                neighbor
                for neighbor in neighbor_rows
                if int(neighbor.get("sample_count") or 0) > 0 or neighbor.get("samples")
            ]
            aggregate_only_neighbor_rows = [
                neighbor
                for neighbor in neighbor_rows
                if not neighbor.get("samples")
            ]
            shared_term_rows = [
                neighbor
                for neighbor in neighbor_rows
                if neighbor.get("shared_terms")
            ]
            shared_term_contract_rows = [
                neighbor
                for neighbor in neighbor_rows
                if isinstance(neighbor.get("shared_terms"), list)
            ]
            edge_evidence_samples = sum(
                int(neighbor.get("sample_count") or 0)
                for neighbor in neighbor_rows
            )
            _assert(sampled_neighbor_rows, "pipeline Atlas payload has no sampled neighbor rows")
            _assert(edge_evidence_samples > 0, "pipeline Atlas payload has no neighbor evidence")
            _assert(
                len(shared_term_contract_rows) == len(neighbor_rows),
                "pipeline Atlas neighbor rows are missing shared-term fields",
            )
            atlas_render_summary = result.get("atlas_render_summary", {})
            _assert(
                atlas_render_summary.get("schema_version") == "sciscape_atlas_render_payload_v1",
                "pipeline Atlas render summary was not attached",
            )
            _assert(
                atlas_render_summary.get("engine_family") == "deck.gl",
                "pipeline Atlas render summary did not target deck.gl",
            )

            atlas_render_response = client.get(f"/api/jobs/{job_id}/atlas-render")
            _assert(atlas_render_response.status_code == 200, "pipeline Atlas render endpoint failed")
            atlas_render = atlas_render_response.json()
            _assert(
                atlas_render.get("schema_version") == "sciscape_atlas_render_payload_v1",
                "pipeline Atlas render schema mismatch",
            )
            _assert(atlas_render.get("node_count", 0) >= 2, "pipeline Atlas render has too few nodes")
            _assert(atlas_render.get("label_count", 0) >= 2, "pipeline Atlas render has too few labels")
            _assert(
                atlas_render.get("view", {}).get("type") == "OrthographicView",
                "pipeline Atlas render does not expose an orthographic view",
            )
            render_layers = atlas_render.get("layers", {})
            _assert(
                render_layers.get("nodes", {}).get("recommended_deck_layer") == "ScatterplotLayer",
                "pipeline Atlas render node layer is not deck.gl scatterplot-ready",
            )
            _assert(
                render_layers.get("labels", {}).get("recommended_deck_layer") == "TextLayer",
                "pipeline Atlas render label layer is not deck.gl text-ready",
            )

            network_response = client.get(f"/api/jobs/{job_id}/network")
            _assert(network_response.status_code == 200, "pipeline cluster network endpoint failed")
            network = network_response.json()
            _assert("error" not in network, f"pipeline cluster network error: {network.get('error')}")
            _assert(network["nodes"], "pipeline cluster network has no nodes")

            term_response = client.get(f"/api/jobs/{job_id}/term-network?top_k=5&min_cooc=1")
            _assert(term_response.status_code == 200, "pipeline term network endpoint failed")
            term_network = term_response.json()
            _assert("error" not in term_network, f"pipeline term network error: {term_network.get('error')}")
            _assert(term_network["nodes"], "pipeline term network has no nodes")
            _assert(term_network["edges"], "pipeline term network has no edges")

            report_response = client.get(f"/api/jobs/{job_id}/view/landscape/report/report.html")
            _assert(report_response.status_code == 200, "pipeline report view endpoint failed")
            data_response = client.get(f"/api/jobs/{job_id}/download/landscape/report/data.json")
            _assert(data_response.status_code == 200, "pipeline data download endpoint failed")

            keywords = pipeline_result["keywords_df"]
            return {
                "status": "passed",
                "job_id": job_id,
                "output_dir": str(result_root),
                "keywords": int(len(keywords)),
                "clusters": int(keywords["cluster_id"].nunique()),
                "network_nodes": int(len(network["nodes"])),
                "term_network_nodes": int(len(term_network["nodes"])),
                "term_network_edges": int(len(term_network["edges"])),
                "cooccurrence_rows": int(artifact_result.counts.get("cooccurrence_rows", 0)),
                "edge_evidence_samples": int(edge_evidence_samples),
                "atlas_neighbor_rows": int(len(neighbor_rows)),
                "atlas_neighbor_aggregate_contract_rows": int(len(aggregate_contract_rows)),
                "atlas_neighbor_sampled_rows": int(len(sampled_neighbor_rows)),
                "atlas_neighbor_aggregate_only_rows": int(len(aggregate_only_neighbor_rows)),
                "atlas_neighbor_shared_term_rows": int(len(shared_term_rows)),
                "atlas_neighbor_shared_term_contract_rows": int(len(shared_term_contract_rows)),
                "atlas_render_nodes": int(atlas_render.get("node_count", 0)),
                "atlas_render_edges": int(atlas_render.get("edge_count", 0)),
                "atlas_render_labels": int(atlas_render.get("label_count", 0)),
                "atlas_render_coordinate_source": atlas_render.get("view", {}).get("coordinate_source"),
                "result_state": result["result_state"],
                "feature_states": feature_states,
                "artifact_contract_path": str(contract_path),
                "result_kind": manifest.result_kind,
            }
        finally:
            web_app._LOCAL_DATA_ROOTS = old_roots
            web_app._jobs = old_jobs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SciScape release quality gates.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a synthetic keyword/network/dashboard smoke gate.",
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=None,
        help="Validate generated outputs for the curated demo presets.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("examples/demo_presets.json"),
        help="Curated demo manifest path.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing demo outputs as skipped instead of failing.",
    )
    parser.add_argument(
        "--web-demo-smoke",
        action="store_true",
        help="Run a synthetic web demo launcher smoke gate without external data.",
    )
    parser.add_argument(
        "--p1-atlas-smoke",
        action="store_true",
        help="Run a tiny full pipeline and reopen it through the web Atlas API.",
    )
    parser.add_argument(
        "--atlas-visual-smoke",
        action="store_true",
        help="Render a tiny deck.gl Atlas map in headless Chrome and verify nonblank pixels.",
    )
    parser.add_argument(
        "--atlas-interaction-smoke",
        action="store_true",
        help="Render a 5k-node deck.gl Atlas map in headless Chrome and verify camera update plus hit-test.",
    )
    parser.add_argument(
        "--atlas-inspector-smoke",
        action="store_true",
        help="Open the static web app in headless Chrome and exercise Atlas inspector node/neighbor evidence.",
    )
    parser.add_argument(
        "--atlas-render-perf-smoke",
        action="store_true",
        help="Build a 100-node/500-edge Atlas render payload and verify CI-scale renderer contract metrics.",
    )
    parser.add_argument(
        "--atlas-render-scale-smoke",
        action="store_true",
        help="Build a 5k-node/25k-edge Atlas render payload and verify small-demo scale metrics.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Validate a SciScape result root, report directory, or report/data.json file.",
    )
    parser.add_argument(
        "--write-artifact-contract",
        action="store_true",
        help="Write qa/artifact_contract.json when used with --artifact-root.",
    )
    parser.add_argument(
        "--write-result-manifest",
        action="store_true",
        help="Write result_manifest.json when used with --artifact-root.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_smoke = args.smoke or (
        args.demo_root is None
        and args.artifact_root is None
        and not args.web_demo_smoke
        and not args.p1_atlas_smoke
        and not args.atlas_visual_smoke
        and not args.atlas_interaction_smoke
        and not args.atlas_inspector_smoke
        and not args.atlas_render_perf_smoke
        and not args.atlas_render_scale_smoke
    )
    results: dict[str, Any] = {"status": "passed", "gates": {}}
    if run_smoke:
        results["gates"]["smoke"] = run_smoke_gate()
    if args.demo_root is not None:
        results["gates"]["demo_outputs"] = validate_demo_outputs(
            root=args.demo_root,
            manifest_path=args.manifest,
            allow_missing=args.allow_missing,
        )
    if args.web_demo_smoke:
        results["gates"]["web_demo_smoke"] = run_web_demo_smoke_gate()
    if args.p1_atlas_smoke:
        results["gates"]["p1_atlas_smoke"] = run_p1_atlas_smoke_gate()
    if args.atlas_visual_smoke:
        results["gates"]["atlas_visual_smoke"] = run_atlas_visual_smoke_gate()
    if args.atlas_interaction_smoke:
        results["gates"]["atlas_interaction_smoke"] = run_atlas_interaction_smoke_gate()
    if args.atlas_inspector_smoke:
        results["gates"]["atlas_inspector_smoke"] = run_atlas_inspector_smoke_gate()
    if args.atlas_render_perf_smoke:
        results["gates"]["atlas_render_perf_smoke"] = run_atlas_render_perf_smoke_gate()
    if args.atlas_render_scale_smoke:
        results["gates"]["atlas_render_scale_smoke"] = run_atlas_render_scale_smoke_gate()
    if args.artifact_root is not None:
        if args.write_artifact_contract:
            artifact_result = write_artifact_contract(args.artifact_root)
            artifact_payload = artifact_result.to_dict()
            artifact_payload["artifact_contract_path"] = str(default_artifact_contract_path(artifact_result))
        else:
            artifact_payload = validate_result_root(args.artifact_root).to_dict()
        results["gates"]["artifact_contract"] = artifact_payload
        if args.write_result_manifest:
            manifest_result = write_result_manifest(args.artifact_root)
            manifest_payload = manifest_result.to_dict()
            manifest_payload["result_manifest_path"] = str(Path(artifact_payload["result_root"]) / "result_manifest.json")
            results["gates"]["result_manifest"] = manifest_payload
        if not artifact_payload["ok"]:
            results["status"] = "failed"
            if args.json:
                print(json.dumps(results, indent=2, sort_keys=True))
            raise SystemExit(1)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print("SciScape quality gates passed:")
        for name, payload in results["gates"].items():
            print(f"  - {name}: {json.dumps(payload, sort_keys=True)}")


if __name__ == "__main__":
    main()
