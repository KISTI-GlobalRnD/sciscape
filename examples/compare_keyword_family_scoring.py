#!/usr/bin/env python3
"""Compare keyword family representative scoring on existing demo outputs.

This script re-runs keyword extraction from cached ``abstracts_subset.parquet``
and ``membership.parquet`` files, once with family-aware representative scoring
disabled and once with it enabled. It is intended for local quality review, not
for regenerating OpenAlex data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sciscape.keyword_extraction import (
    KeywordExtractionConfig,
    KeywordExtractionPipeline,
    keyword_diagnostics,
    score_before_after,
)
from sciscape.keyword_extraction.depth import DepthConfig
from sciscape.keyword_extraction.term_network import TermNetworkConfig


PRESET_SLUGS = (
    "perovskite_solar_cells_2020_2024",
    "graph_neural_networks_2020_2024",
)

LANDSCAPE_CANDIDATES = (
    "landscape_representative_latest",
    "landscape_refined",
    "landscape",
)


@dataclass(frozen=True)
class RunInput:
    slug: str
    landscape_dir: Path
    abstracts_path: Path
    membership_path: Path
    top_n_keywords: int


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if not isinstance(value, (list, dict, tuple, set)) and pd.isna(value):
        return None
    return str(value)


def _jsonable_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable_nested(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable_nested(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == "object":
            out[column] = out[column].map(_jsonable_nested)
    return out


def _read_top_n_keywords(landscape_dir: Path, fallback: int) -> int:
    keywords_path = landscape_dir / "keywords.parquet"
    if not keywords_path.exists():
        return int(fallback)
    df = pd.read_parquet(keywords_path, columns=["cluster_id"])
    if df.empty:
        return int(fallback)
    return int(df.groupby("cluster_id").size().max())


def _find_inputs(input_root: Path, slugs: list[str], fallback_top_n: int) -> list[RunInput]:
    inputs: list[RunInput] = []
    for slug in slugs:
        preset_dir = input_root / slug
        for candidate in LANDSCAPE_CANDIDATES:
            landscape_dir = preset_dir / candidate
            abstracts_path = landscape_dir / "abstracts_subset.parquet"
            membership_path = landscape_dir / "membership.parquet"
            if abstracts_path.exists() and membership_path.exists():
                inputs.append(
                    RunInput(
                        slug=slug,
                        landscape_dir=landscape_dir,
                        abstracts_path=abstracts_path,
                        membership_path=membership_path,
                        top_n_keywords=_read_top_n_keywords(landscape_dir, fallback_top_n),
                    )
                )
                break
        else:
            raise FileNotFoundError(
                f"No usable landscape output found for {slug} under {input_root}. "
                f"Checked: {', '.join(LANDSCAPE_CANDIDATES)}"
            )
    return inputs


def _build_keyword_config(
    run_input: RunInput,
    *,
    family_enabled: bool,
    top_n_keywords: int,
    n_jobs: int,
    family_weight: float,
    family_max_bonus: float,
    verbose: bool,
) -> KeywordExtractionConfig:
    return KeywordExtractionConfig(
        abstract_path=run_input.abstracts_path,
        membership_path=run_input.membership_path,
        include_title=True,
        title_weight=2.0,
        min_df_unigram=5,
        min_df_phrase=3,
        use_phrase_vectorizer=True,
        ngram_min=2,
        ngram_max=3,
        phrase_min_count_per_cluster=5,
        top_n_unigrams=200,
        top_n_keywords=top_n_keywords,
        scoring_pool_factor=1.5,
        normalization_enabled=True,
        norm_plural_merge_enabled=True,
        academic_stopwords_enabled=True,
        artifact_filter_enabled=True,
        cross_cluster_penalty_enabled=True,
        cross_cluster_penalty_min_count=2,
        quality_diagnostics_enabled=True,
        quality_rerank_enabled=True,
        fragment_suppression_enabled=True,
        cooccurrence_enabled=True,
        cooccurrence_min_count=3,
        term_network=TermNetworkConfig(
            enabled=True,
            layers=["string", "token", "cooccurrence"],
            merge_threshold=0.5,
        ),
        auto_merge_enabled=True,
        short_term_expansion_enabled=True,
        depth=DepthConfig(enabled=True, n_levels=3),
        quality_family_representative_enabled=family_enabled,
        quality_family_representative_weight=family_weight,
        quality_family_representative_max_bonus=family_max_bonus,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def _run_keywords(config: KeywordExtractionConfig) -> pd.DataFrame:
    pipeline = KeywordExtractionPipeline(config)
    return pipeline.run()


def _score_column(df: pd.DataFrame) -> str:
    for column in ("representative_score", "quality_score", "score"):
        if column in df.columns:
            return column
    raise ValueError("Keyword dataframe has no usable score column")


def _label(row: pd.Series) -> str:
    value = row.get("display_label")
    if isinstance(value, str) and value.strip():
        return value
    return str(row.get("term", ""))


def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_col = _score_column(out)
    sort_cols = ["cluster_id", score_col]
    ascending = [True, False]
    if "score" in out.columns and score_col != "score":
        sort_cols.append("score")
        ascending.append(False)
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    out["rank"] = out.groupby("cluster_id").cumcount() + 1
    out["report_label"] = out.apply(_label, axis=1)
    return out


def _top_labels(df: pd.DataFrame, cluster_id: int, n: int) -> list[str]:
    group = df[df["cluster_id"] == cluster_id].head(n)
    return [str(value) for value in group["report_label"].tolist()]


def _family_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "representative_family_multiplier" not in df.columns:
        return {
            "boosted_terms": 0,
            "boosted_ratio": 0.0,
            "multiplier_mean": None,
            "multiplier_p90": None,
            "multiplier_max": None,
        }
    multiplier = pd.to_numeric(df["representative_family_multiplier"], errors="coerce").fillna(1.0)
    boosted = multiplier > 1.0 + 1e-12
    boosted_values = multiplier[boosted]
    return {
        "boosted_terms": int(boosted.sum()),
        "boosted_ratio": float(boosted.mean()) if len(boosted) else 0.0,
        "multiplier_mean": float(boosted_values.mean()) if not boosted_values.empty else None,
        "multiplier_p90": float(boosted_values.quantile(0.9)) if not boosted_values.empty else None,
        "multiplier_max": float(boosted_values.max()) if not boosted_values.empty else None,
    }


def _top_family_terms(df: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    if "representative_family_multiplier" not in df.columns:
        return []
    score_col = _score_column(df)
    columns = [
        "cluster_id",
        "rank",
        "term",
        "report_label",
        score_col,
        "representative_family_multiplier",
    ]
    for optional in (
        "representative_family_child_count",
        "representative_family_member_count",
        "representative_family_avg_child_coverage",
    ):
        if optional in df.columns:
            columns.append(optional)
    out = df[pd.to_numeric(df["representative_family_multiplier"], errors="coerce").fillna(1.0) > 1.0]
    out = out.sort_values(
        ["representative_family_multiplier", score_col],
        ascending=[False, False],
        kind="mergesort",
    )
    return out[columns].head(limit).to_dict(orient="records")


def _rank_delta_summary(off_ranked: pd.DataFrame, on_ranked: pd.DataFrame) -> dict[str, Any]:
    if "representative_family_multiplier" not in on_ranked.columns:
        return {"shared_terms": 0, "boosted_shared_terms": 0, "boosted_terms_ranked_up": 0}
    merged = off_ranked[["cluster_id", "term", "rank"]].merge(
        on_ranked[["cluster_id", "term", "rank", "representative_family_multiplier"]],
        on=["cluster_id", "term"],
        suffixes=("_off", "_on"),
        how="inner",
    )
    if merged.empty:
        return {"shared_terms": 0, "boosted_shared_terms": 0, "boosted_terms_ranked_up": 0}
    boosted = pd.to_numeric(
        merged["representative_family_multiplier"],
        errors="coerce",
    ).fillna(1.0) > 1.0
    ranked_up = boosted & (merged["rank_on"] < merged["rank_off"])
    return {
        "shared_terms": int(len(merged)),
        "boosted_shared_terms": int(boosted.sum()),
        "boosted_terms_ranked_up": int(ranked_up.sum()),
        "boosted_terms_ranked_up_ratio": float(ranked_up.sum() / max(1, boosted.sum())),
        "boosted_rank_delta_mean": (
            float((merged.loc[boosted, "rank_off"] - merged.loc[boosted, "rank_on"]).mean())
            if boosted.any()
            else None
        ),
    }


def _compare_keywords(off: pd.DataFrame, on: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    off_ranked = _ranked(off)
    on_ranked = _ranked(on)
    cluster_ids = sorted(set(off_ranked["cluster_id"].astype(int)) | set(on_ranked["cluster_id"].astype(int)))
    rows: list[dict[str, Any]] = []
    top3_jaccards: list[float] = []

    for cluster_id in cluster_ids:
        off_top1 = _top_labels(off_ranked, cluster_id, 1)
        on_top1 = _top_labels(on_ranked, cluster_id, 1)
        off_top3 = _top_labels(off_ranked, cluster_id, 3)
        on_top3 = _top_labels(on_ranked, cluster_id, 3)
        off_set = set(off_top3)
        on_set = set(on_top3)
        denom = len(off_set | on_set)
        top3_jaccard = len(off_set & on_set) / denom if denom else 1.0
        top3_jaccards.append(top3_jaccard)
        rows.append(
            {
                "cluster_id": cluster_id,
                "off_top1": off_top1[0] if off_top1 else None,
                "on_top1": on_top1[0] if on_top1 else None,
                "top1_changed": bool(off_top1 != on_top1),
                "off_top3": " | ".join(off_top3),
                "on_top3": " | ".join(on_top3),
                "top3_jaccard": top3_jaccard,
            }
        )

    changes = pd.DataFrame(rows)
    summary = {
        "n_clusters": int(len(cluster_ids)),
        "n_keywords_off": int(len(off)),
        "n_keywords_on": int(len(on)),
        "score_column": _score_column(on_ranked),
        "top1_changed_clusters": int(changes["top1_changed"].sum()) if not changes.empty else 0,
        "top1_changed_ratio": float(changes["top1_changed"].mean()) if not changes.empty else 0.0,
        "top3_changed_clusters": int((changes["top3_jaccard"] < 1.0).sum()) if not changes.empty else 0,
        "avg_top3_jaccard": float(np.mean(top3_jaccards)) if top3_jaccards else 1.0,
        "family_stats": _family_stats(on_ranked),
        "rank_delta": _rank_delta_summary(off_ranked, on_ranked),
        "top_family_boosted_terms": _top_family_terms(on_ranked),
    }
    return summary, changes


def _metric_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> float | None:
    before_value = before.get(key)
    after_value = after.get(key)
    if before_value is None or after_value is None:
        return None
    return float(after_value) - float(before_value)


def _run_one(
    run_input: RunInput,
    *,
    output_root: Path,
    top_n_override: int | None,
    n_jobs: int,
    family_weight: float,
    family_max_bonus: float,
    sample_clusters: int | None,
    verbose: bool,
) -> dict[str, Any]:
    top_n_keywords = int(top_n_override or run_input.top_n_keywords)
    out_dir = output_root / run_input.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==> {run_input.slug}")
    print(f"Input: {run_input.landscape_dir}")
    print(f"top_n_keywords={top_n_keywords}")

    off_cfg = _build_keyword_config(
        run_input,
        family_enabled=False,
        top_n_keywords=top_n_keywords,
        n_jobs=n_jobs,
        family_weight=family_weight,
        family_max_bonus=family_max_bonus,
        verbose=verbose,
    )
    on_cfg = _build_keyword_config(
        run_input,
        family_enabled=True,
        top_n_keywords=top_n_keywords,
        n_jobs=n_jobs,
        family_weight=family_weight,
        family_max_bonus=family_max_bonus,
        verbose=verbose,
    )

    print("Running keyword extraction: family scoring OFF")
    off_df = _run_keywords(off_cfg)
    off_path = out_dir / "keywords_family_off.parquet"
    _parquet_safe(off_df).to_parquet(off_path, index=False)

    print("Running keyword extraction: family scoring ON")
    on_df = _run_keywords(on_cfg)
    on_path = out_dir / "keywords_family_on.parquet"
    _parquet_safe(on_df).to_parquet(on_path, index=False)

    comparison, cluster_changes = _compare_keywords(off_df, on_df)
    cluster_changes_path = out_dir / "top_label_changes.csv"
    cluster_changes.to_csv(cluster_changes_path, index=False)

    before_after = score_before_after(
        off_df,
        on_df,
        sample_clusters=sample_clusters,
        seed=0,
    )
    off_diag = keyword_diagnostics(off_df, sample_clusters=sample_clusters, seed=0).to_dict()
    on_diag = keyword_diagnostics(on_df, sample_clusters=sample_clusters, seed=0).to_dict()
    comparison["diagnostics"] = {
        "quality_score": before_after,
        "deltas": {
            "review_flag_ratio": _metric_delta(off_diag, on_diag, "review_flag_ratio"),
            "representative_diversity_ratio": _metric_delta(
                off_diag,
                on_diag,
                "representative_diversity_ratio",
            ),
            "family_compression_ratio": _metric_delta(off_diag, on_diag, "family_compression_ratio"),
            "unresolved_short_form_ratio": _metric_delta(
                off_diag,
                on_diag,
                "unresolved_short_form_ratio",
            ),
        },
    }
    comparison["paths"] = {
        "keywords_family_off": off_path,
        "keywords_family_on": on_path,
        "top_label_changes": cluster_changes_path,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(
        "Summary: "
        f"top1_changed={comparison['top1_changed_clusters']}/{comparison['n_clusters']}, "
        f"avg_top3_jaccard={comparison['avg_top3_jaccard']:.3f}, "
        f"boosted_terms={comparison['family_stats']['boosted_terms']}"
    )
    return {
        "slug": run_input.slug,
        "input_dir": run_input.landscape_dir,
        "output_dir": out_dir,
        "top_n_keywords": top_n_keywords,
        **comparison,
    }


def _format_ratio(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _write_report(output_root: Path, results: list[dict[str, Any]]) -> Path:
    lines = [
        "# Keyword Family Scoring Comparison",
        "",
        "Family-aware representative scoring was toggled on/off while reusing the",
        "same cached abstracts and memberships from the OpenAlex demo landscapes.",
        "",
        "| Dataset | Clusters | Top-1 changed | Avg top-3 Jaccard | Boosted terms | Boosted ranked up | Diagnostic score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        rank_delta = result["rank_delta"]
        lines.append(
            "| "
            f"{result['slug']} | "
            f"{result['n_clusters']} | "
            f"{result['top1_changed_clusters']}/{result['n_clusters']} | "
            f"{result['avg_top3_jaccard']:.3f} | "
            f"{result['family_stats']['boosted_terms']} | "
            f"{rank_delta.get('boosted_terms_ranked_up', 0)}/"
            f"{rank_delta.get('boosted_shared_terms', 0)} | "
            f"{result['diagnostics']['quality_score']['total_score']:.2f} |"
        )

    lines.extend(["", "## Dataset Notes", ""])
    for result in results:
        lines.extend(
            [
                f"### {result['slug']}",
                "",
                f"- Input: `{result['input_dir']}`",
                f"- Output: `{result['output_dir']}`",
                f"- Top-1 changed clusters: {result['top1_changed_clusters']} / {result['n_clusters']}",
                f"- Avg top-3 Jaccard: {result['avg_top3_jaccard']:.3f}",
                f"- Family boosted terms: {result['family_stats']['boosted_terms']} "
                f"({_format_ratio(result['family_stats']['boosted_ratio'])})",
                f"- Max family multiplier: {_format_ratio(result['family_stats']['multiplier_max'])}",
                f"- Boosted shared terms ranked up: "
                f"{result['rank_delta'].get('boosted_terms_ranked_up', 0)} / "
                f"{result['rank_delta'].get('boosted_shared_terms', 0)}",
                f"- Review flag ratio delta: "
                f"{_format_ratio(result['diagnostics']['deltas']['review_flag_ratio'])}",
                f"- Representative diversity ratio delta: "
                f"{_format_ratio(result['diagnostics']['deltas']['representative_diversity_ratio'])}",
                "",
            ]
        )
        boosted_terms = result.get("top_family_boosted_terms", [])[:5]
        if boosted_terms:
            lines.extend(["Top boosted terms:", ""])
            for row in boosted_terms:
                lines.append(
                    "- "
                    f"cluster {row['cluster_id']} rank {row['rank']}: "
                    f"{row['report_label']} "
                    f"(x{float(row['representative_family_multiplier']):.3f})"
                )
            lines.append("")

    path = output_root / "family_scoring_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare keyword family representative scoring on cached SciScape demo outputs.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("examples_output/keyword_representative_mmr_20260528_102627"),
        help="Root containing demo preset directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for comparison artifacts. Defaults to examples_output/family_scoring_eval_<timestamp>.",
    )
    parser.add_argument(
        "--preset",
        choices=[*PRESET_SLUGS, "both"],
        default="both",
        help="Dataset preset to compare.",
    )
    parser.add_argument(
        "--top-n-keywords",
        type=int,
        default=None,
        help="Override per-cluster keyword count inferred from existing outputs.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs passed to the keyword pipeline.",
    )
    parser.add_argument(
        "--family-weight",
        type=float,
        default=0.08,
        help="Family representative scoring weight.",
    )
    parser.add_argument(
        "--family-max-bonus",
        type=float,
        default=0.15,
        help="Maximum family representative multiplier bonus.",
    )
    parser.add_argument(
        "--sample-clusters",
        type=int,
        default=None,
        help="Cluster sample count for diagnostics. Defaults to all clusters.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose keyword pipeline logging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    slugs = list(PRESET_SLUGS) if args.preset == "both" else [args.preset]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or Path("examples_output") / f"family_scoring_eval_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = _find_inputs(args.input_root, slugs, fallback_top_n=args.top_n_keywords or 30)
    results = [
        _run_one(
            run_input,
            output_root=output_root,
            top_n_override=args.top_n_keywords,
            n_jobs=args.n_jobs,
            family_weight=args.family_weight,
            family_max_bonus=args.family_max_bonus,
            sample_clusters=args.sample_clusters,
            verbose=args.verbose,
        )
        for run_input in inputs
    ]

    combined_path = output_root / "family_scoring_comparison.json"
    combined_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    report_path = _write_report(output_root, results)
    print(f"\nCombined JSON: {combined_path}")
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
