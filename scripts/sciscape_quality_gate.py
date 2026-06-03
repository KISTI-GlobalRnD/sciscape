#!/usr/bin/env python3
"""Lightweight release gates for SciScape demo and visualization surfaces."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.artifacts import (
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
        export_dashboard(keywords, output_path=str(dashboard_path), title="SciScape Quality Gate")
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

            atlas = result.get("atlas", {})
            _assert(atlas.get("nodes"), "pipeline Atlas payload has no nodes")
            edge_evidence_samples = sum(
                int(neighbor.get("sample_count") or 0)
                for node in atlas.get("nodes", [])
                for neighbor in node.get("neighbors", [])
            )
            _assert(edge_evidence_samples > 0, "pipeline Atlas payload has no neighbor evidence")

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
                "edge_evidence_samples": int(edge_evidence_samples),
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
