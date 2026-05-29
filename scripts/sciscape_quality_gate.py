#!/usr/bin/env python3
"""Lightweight release gates for SciScape demo and visualization surfaces."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
from sciscape.keyword_extraction.visualization import export_dashboard
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_smoke = args.smoke or args.demo_root is None
    results: dict[str, Any] = {"status": "passed", "gates": {}}
    if run_smoke:
        results["gates"]["smoke"] = run_smoke_gate()
    if args.demo_root is not None:
        results["gates"]["demo_outputs"] = validate_demo_outputs(
            root=args.demo_root,
            manifest_path=args.manifest,
            allow_missing=args.allow_missing,
        )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print("SciScape quality gates passed:")
        for name, payload in results["gates"].items():
            print(f"  - {name}: {json.dumps(payload, sort_keys=True)}")


if __name__ == "__main__":
    main()
