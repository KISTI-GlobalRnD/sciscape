#!/usr/bin/env python3
"""Live OpenAlex demo presets for SciScape full-cycle workflows.

The presets are intentionally small enough for a laptop run, but large enough
to produce useful clusters and keyword summaries. They call the public
OpenAlex pipeline first, then run the landscape step with demo-friendly
cluster and keyword settings.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sciscape.landscape import LandscapeConfig, run_landscape
from sciscape.openalex import OpenAlexPipelineConfig, run_openalex_pipeline


DEMO_MANIFEST_PATH = Path(__file__).with_name("demo_presets.json")


@dataclass(frozen=True)
class DemoPreset:
    slug: str
    title: str
    query: str
    filters: dict[str, str]
    max_works: int = 1000
    edge_types: tuple[str, ...] = ("dc", "bc")
    bc_topk: int = 50
    min_shared_refs: int = 1
    min_docs: int = 30
    top_n_keywords: int = 30
    combine_strategy: str = "consensus"
    combine_top_k: int | str = "auto"
    auto_gamma: bool = True
    auto_gamma_target: float = 3.0


def load_demo_manifest(path: Path = DEMO_MANIFEST_PATH) -> dict[str, Any]:
    """Load the curated demo manifest used by docs, examples, and gates."""
    return json.loads(path.read_text(encoding="utf-8"))


def _preset_from_manifest(record: dict[str, Any]) -> DemoPreset:
    return DemoPreset(
        slug=str(record["slug"]),
        title=str(record["title"]),
        query=str(record.get("query", "")),
        filters={str(k): str(v) for k, v in dict(record.get("filters", {})).items()},
        max_works=int(record.get("max_works", 1000)),
        edge_types=tuple(str(v) for v in record.get("edge_types", ("dc", "bc"))),
        bc_topk=int(record.get("bc_topk", 50)),
        min_shared_refs=int(record.get("min_shared_refs", 1)),
        min_docs=int(record.get("min_docs", 30)),
        top_n_keywords=int(record.get("top_n_keywords", 30)),
        combine_strategy=str(record.get("combine_strategy", "consensus")),
        combine_top_k=record.get("combine_top_k", "auto"),
        auto_gamma=bool(record.get("auto_gamma", True)),
        auto_gamma_target=float(record.get("auto_gamma_target", 3.0)),
    )


def load_demo_presets(path: Path = DEMO_MANIFEST_PATH) -> dict[str, DemoPreset]:
    manifest = load_demo_manifest(path)
    return {
        str(name): _preset_from_manifest(record)
        for name, record in dict(manifest["presets"]).items()
    }


PRESETS: dict[str, DemoPreset] = load_demo_presets()


def _build_parser() -> argparse.ArgumentParser:
    manifest = load_demo_manifest()
    parser = argparse.ArgumentParser(
        description=(
            "Run live OpenAlex SciScape demos for perovskite solar cells and "
            "graph neural networks."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=[*PRESETS.keys(), "both"],
        default="perovskite",
        help="Demo dataset preset to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(str(manifest.get("default_output_root", "workspace/examples_output/openalex_live"))),
        help="Root directory for demo outputs.",
    )
    parser.add_argument(
        "--max-works",
        type=int,
        default=None,
        help="Override preset max works.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="OpenAlex polite-pool email. Recommended for repeated runs.",
    )
    parser.add_argument(
        "--skip-landscape",
        action="store_true",
        help="Fetch works and build edges only.",
    )
    parser.add_argument(
        "--force-landscape",
        action="store_true",
        help="Ignore cached landscape intermediates.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print preset metadata and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without contacting OpenAlex.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable info logging.",
    )
    return parser


def _selected_presets(name: str) -> list[DemoPreset]:
    if name == "both":
        return [PRESETS["perovskite"], PRESETS["gnn"]]
    return [PRESETS[name]]


def _preset_summary(preset: DemoPreset, max_works: int | None = None) -> dict[str, Any]:
    data = asdict(preset)
    if max_works is not None:
        data["max_works"] = max_works
    return data


def _run_one(
    preset: DemoPreset,
    *,
    output_root: Path,
    email: str | None,
    max_works: int | None,
    skip_landscape: bool,
    force_landscape: bool,
) -> dict[str, Any]:
    out_dir = output_root / preset.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_max_works = max_works or preset.max_works

    def progress(msg: str) -> None:
        print(f"[{preset.slug}] {msg}")

    print(f"\n==> {preset.title}")
    print(f"Output: {out_dir}")

    fetch_cfg = OpenAlexPipelineConfig(
        query=preset.query,
        filters=dict(preset.filters),
        max_works=effective_max_works,
        email=email,
        edge_types=preset.edge_types,
        bc_topk=preset.bc_topk,
        min_shared_refs=preset.min_shared_refs,
        output_dir=out_dir,
        run_landscape=False,
        progress=progress,
    )
    fetch_result = run_openalex_pipeline(fetch_cfg)

    landscape_dir: Path | None = None
    if not skip_landscape:
        if fetch_result.edges_path is None:
            raise RuntimeError(f"No edge table was built for preset {preset.slug}")

        layer_paths = {
            edge_type: out_dir / f"edges_{edge_type}.parquet"
            for edge_type in preset.edge_types
            if (out_dir / f"edges_{edge_type}.parquet").exists()
        }
        landscape_dir = out_dir / "landscape"
        landscape_cfg = LandscapeConfig(
            min_docs_per_cluster=preset.min_docs,
            top_n_keywords=preset.top_n_keywords,
            layer_paths=layer_paths or None,
            combine_strategy=preset.combine_strategy,
            combine_top_k=preset.combine_top_k,
            auto_gamma=preset.auto_gamma,
            auto_gamma_target=preset.auto_gamma_target,
            force=force_landscape,
            report_title=f"SciScape Demo: {preset.title}",
            progress=progress,
        )
        run_landscape(
            fetch_result.edges_path,
            fetch_result.abstracts_path,
            landscape_dir,
            config=landscape_cfg,
        )

    summary = {
        "preset": preset.slug,
        "title": preset.title,
        "query": preset.query,
        "filters": preset.filters,
        "max_works": effective_max_works,
        "n_works": fetch_result.n_works,
        "n_edges": fetch_result.n_edges,
        "abstracts_path": str(fetch_result.abstracts_path) if fetch_result.abstracts_path else None,
        "edges_path": str(fetch_result.edges_path) if fetch_result.edges_path else None,
        "landscape_dir": str(landscape_dir) if landscape_dir else None,
    }
    summary_path = out_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary: {summary_path}")
    if landscape_dir:
        print(f"Report: {landscape_dir / 'report' / 'report.html'}")
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    presets = _selected_presets(args.preset)

    if args.list_presets:
        print(json.dumps({p.slug: _preset_summary(p) for p in PRESETS.values()}, indent=2))
        return

    if args.dry_run:
        print(json.dumps([_preset_summary(p, args.max_works) for p in presets], indent=2))
        return

    summaries = [
        _run_one(
            preset,
            output_root=args.output_root,
            email=args.email,
            max_works=args.max_works,
            skip_landscape=args.skip_landscape,
            force_landscape=args.force_landscape,
        )
        for preset in presets
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_root / "openalex_live_demo_summary.json"
    combined_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
