"""Validate semantic coherence for hierarchy postprocess field expansion.

This script compares ``small_only`` and ``two_stage_quality_first`` effective
source memberships from the six-field expansion using local title/abstract
text.  It joins membership node indices back to OpenAlex work ids, builds a
field-local TF-IDF matrix, and reports cluster centroid coherence metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hierarchy_postprocess import DEFAULT_OUTPUT_DIR, _markdown_table, _write_table  # noqa: E402

EXPECTED_FIELDS = (12, 15, 18, 26, 30, 34)
DEFAULT_SEEDS = (11, 42, 73)
DEFAULT_POLICIES = ("small_only", "two_stage_quality_first")
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "field_expansion_source_seed_effects.csv"
DEFAULT_RUN_ROOT = DEFAULT_OUTPUT_DIR / "semantic_coherence_validation"
DEFAULT_MIN_CLUSTER_TEXT_DOCS = 10
DEFAULT_MAX_FEATURES = 50_000


@dataclass(frozen=True)


class SampleConfig:
    sample: str
    field_id: int
    text_path: Path
    edge_path: Path | None
    node_mapping_path: Path | None
    existing_manifest_path: Path | None
    src_col: str = "uid1"
    dst_col: str = "uid2"


def _repo_path(path: str | Path | None) -> Path | None:
    if path is None or (isinstance(path, float) and pd.isna(path)):
        return None
    resolved = Path(str(path))
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _field_id_from_sample(sample: str) -> int:
    match = re.search(r"field_?(\d+)", str(sample))
    if not match:
        raise ValueError(f"Could not parse field id from sample: {sample}")
    return int(match.group(1))


def _sample_config(sample: str) -> SampleConfig:
    field_id = _field_id_from_sample(sample)
    text_path = REPO_ROOT / "data" / "openalex_metadata" / f"field_{field_id}" / "works_text.parquet"
    if sample == "field34_combo_dc_bc_cc_sum":
        return SampleConfig(
            sample=sample,
            field_id=field_id,
            text_path=text_path,
            edge_path=REPO_ROOT
            / "data"
            / "linktype_edges"
            / f"field_{field_id}"
            / "combo_dc+bc+cc_sum.parquet",
            node_mapping_path=REPO_ROOT / "data" / "linktype_edges" / f"field_{field_id}" / "node_mapping.parquet",
            existing_manifest_path=None,
            src_col="src",
            dst_col="dst",
        )
    return SampleConfig(
        sample=sample,
        field_id=field_id,
        text_path=text_path,
        edge_path=REPO_ROOT
        / "data"
        / "linktype_edges_gcc"
        / f"field_{field_id}"
        / "emb_full_knn30.parquet",
        node_mapping_path=REPO_ROOT / "data" / "linktype_edges_gcc" / f"field_{field_id}" / "node_mapping.parquet",
        existing_manifest_path=DEFAULT_OUTPUT_DIR
        / "field_expansion_runs"
        / sample
        / "graph"
        / "node_manifest.parquet",
        src_col="uid1",
        dst_col="uid2",
    )


def _read_membership(path: Path) -> pd.DataFrame:
    membership = pd.read_parquet(path, columns=["node_idx", "cluster"])
    if not membership["node_idx"].is_monotonic_increasing:
        membership = membership.sort_values("node_idx", kind="mergesort").reset_index(drop=True)
    return membership


def _membership_node_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _normalize_manifest_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = set(manifest.columns)
    if {"node_idx", "uid"}.issubset(cols):
        out = manifest[["node_idx", "uid"]].rename(columns={"uid": "work_id"})
    elif {"idx", "work_id"}.issubset(cols):
        out = manifest[["idx", "work_id"]].rename(columns={"idx": "node_idx"})
    elif {"node_idx", "work_id"}.issubset(cols):
        out = manifest[["node_idx", "work_id"]]
    else:
        raise ValueError(f"Unsupported node manifest columns: {manifest.columns.tolist()}")
    out = out.copy()
    out["node_idx"] = out["node_idx"].astype(np.int64)
    out["work_id"] = out["work_id"].astype("string")
    return out


def _manifest_from_mapping_if_exact(config: SampleConfig, n_nodes: int) -> pd.DataFrame | None:
    if config.node_mapping_path is None or not config.node_mapping_path.exists():
        return None
    if int(pq.ParquetFile(config.node_mapping_path).metadata.num_rows) != int(n_nodes):
        return None
    return _normalize_manifest_columns(pd.read_parquet(config.node_mapping_path))


def _manifest_from_existing_if_exact(config: SampleConfig, n_nodes: int) -> pd.DataFrame | None:
    if config.existing_manifest_path is None or not config.existing_manifest_path.exists():
        return None
    if int(pq.ParquetFile(config.existing_manifest_path).metadata.num_rows) != int(n_nodes):
        return None
    return _normalize_manifest_columns(pd.read_parquet(config.existing_manifest_path))


def _derive_manifest_from_edges(config: SampleConfig, cache_path: Path) -> pd.DataFrame:
    if config.edge_path is None or not config.edge_path.exists():
        raise FileNotFoundError(f"Missing edge parquet for {config.sample}: {config.edge_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    edges_lf = pl.scan_parquet(config.edge_path).select([config.src_col, config.dst_col])
    uid_lf = pl.concat(
        [
            edges_lf.select(pl.col(config.src_col).alias("work_id")),
            edges_lf.select(pl.col(config.dst_col).alias("work_id")),
        ],
        how="vertical",
    )
    (
        uid_lf.unique()
        .sort("work_id")
        .with_row_index("node_idx")
        .with_columns(pl.col("node_idx").cast(pl.UInt32))
        .select(["node_idx", "work_id"])
        .sink_parquet(cache_path, compression="zstd")
    )
    return _normalize_manifest_columns(pd.read_parquet(cache_path))


def _load_manifest(config: SampleConfig, n_nodes: int, output_dir: Path, force_manifest: bool) -> tuple[pd.DataFrame, str]:
    cache_path = output_dir / "manifests" / config.sample / "node_manifest.parquet"
    if cache_path.exists() and not force_manifest:
        manifest = _normalize_manifest_columns(pd.read_parquet(cache_path))
        if len(manifest) == n_nodes:
            return manifest, "cached"

    existing = _manifest_from_existing_if_exact(config, n_nodes)
    if existing is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(cache_path, index=False)
        return existing, "existing_manifest"

    mapped = _manifest_from_mapping_if_exact(config, n_nodes)
    if mapped is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mapped.to_parquet(cache_path, index=False)
        return mapped, "node_mapping"

    derived = _derive_manifest_from_edges(config, cache_path)
    if len(derived) != n_nodes:
        raise ValueError(
            f"Derived manifest for {config.sample} has {len(derived)} nodes, "
            f"but membership has {n_nodes}"
        )
    return derived, "edge_derived"


def _load_text_docs(config: SampleConfig, manifest: pd.DataFrame) -> pd.DataFrame:
    if not config.text_path.exists():
        raise FileNotFoundError(f"Missing works_text parquet for {config.sample}: {config.text_path}")
    text = pd.read_parquet(config.text_path, columns=["work_id", "title", "abstract"])
    text["work_id"] = text["work_id"].astype("string")
    joined = manifest.merge(text, on="work_id", how="inner", validate="one_to_one")
    title = joined["title"].fillna("").astype(str)
    abstract = joined["abstract"].fillna("").astype(str)
    joined["text"] = (title + "\n" + abstract).str.replace(r"\s+", " ", regex=True).str.strip()
    joined = joined[joined["text"].str.len() > 0].copy()
    joined = joined.sort_values("node_idx", kind="mergesort").reset_index(drop=True)
    return joined[["node_idx", "work_id", "text"]]


def _build_tfidf(texts: pd.Series, *, max_features: int, min_df: int, ngram_max: int) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    params: dict[str, Any] = {
        "lowercase": True,
        "strip_accents": "unicode",
        "stop_words": "english",
        "max_features": int(max_features),
        "min_df": int(min_df),
        "max_df": 0.85,
        "ngram_range": (1, int(ngram_max)),
        "sublinear_tf": True,
        "norm": "l2",
        "dtype": np.float32,
        "token_pattern": r"(?u)\b[^\W\d_][^\W_]{2,}\b",
    }
    for attempted_min_df in (int(min_df), 2, 1):
        params["min_df"] = max(1, attempted_min_df)
        try:
            vectorizer = TfidfVectorizer(**params)
            matrix = vectorizer.fit_transform(texts.tolist()).tocsr()
            return matrix, vectorizer
        except ValueError:
            if attempted_min_df == 1:
                raise
    raise RuntimeError("unreachable")


def _labels_for_docs(membership: pd.DataFrame, node_idx: np.ndarray) -> np.ndarray:
    member_node_idx = membership["node_idx"].to_numpy(dtype=np.int64, copy=False)
    clusters = membership["cluster"].to_numpy(dtype=np.int64, copy=False)
    if len(member_node_idx) and np.array_equal(member_node_idx, np.arange(len(member_node_idx), dtype=np.int64)):
        return clusters[node_idx]
    mapping = pd.Series(clusters, index=member_node_idx)
    labels = mapping.reindex(node_idx)
    if labels.isna().any():
        missing = int(labels.isna().sum())
        raise ValueError(f"Membership is missing {missing} text node indices")
    return labels.to_numpy(dtype=np.int64)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _cluster_coherence(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    min_cluster_docs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if matrix.shape[0] != labels.shape[0]:
        raise ValueError("TF-IDF row count and label count differ")

    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    unique_labels, starts, counts = np.unique(
        sorted_labels,
        return_index=True,
        return_counts=True,
    )
    inverse = np.empty_like(order)
    repeated_positions = np.repeat(np.arange(len(unique_labels), dtype=np.int64), counts)
    inverse[order] = repeated_positions
    indicator = sparse.csr_matrix(
        (
            np.ones(labels.shape[0], dtype=np.float32),
            (inverse, np.arange(labels.shape[0], dtype=np.int64)),
        ),
        shape=(len(unique_labels), labels.shape[0]),
    )
    cluster_sums = (indicator @ matrix).tocsr()
    cluster_norms = np.sqrt(cluster_sums.multiply(cluster_sums).sum(axis=1)).A1
    inv_norms = np.zeros_like(cluster_norms, dtype=np.float64)
    nonzero = cluster_norms > 0.0
    inv_norms[nonzero] = 1.0 / cluster_norms[nonzero]
    centroids = cluster_sums.multiply(inv_norms[:, None]).tocsr()
    cluster_rows: list[dict[str, Any]] = []

    for pos, (cluster, start, count) in enumerate(zip(unique_labels, starts, counts, strict=True)):
        doc_count = int(count)
        if doc_count < int(min_cluster_docs) or cluster_norms[pos] <= 0.0:
            continue
        row_idx = order[int(start) : int(start) + doc_count]
        sub = matrix[row_idx]
        centroid = centroids.getrow(pos)
        sims = np.asarray(sub.multiply(centroid).sum(axis=1)).ravel()
        sum_data = cluster_sums.getrow(pos).data.astype(np.float64, copy=False)
        total_mass = float(sum_data.sum())
        if total_mass > 0.0:
            if sum_data.size <= 10:
                top10_mass = float(sum_data.sum())
            else:
                top10_mass = float(np.partition(sum_data, -10)[-10:].sum())
            top10_share = top10_mass / total_mass
        else:
            top10_share = float("nan")
        cluster_rows.append(
            {
                "cluster": int(cluster),
                "text_doc_count": doc_count,
                "mean_doc_centroid_cosine": float(np.mean(sims)),
                "median_doc_centroid_cosine": float(np.median(sims)),
                "p10_doc_centroid_cosine": float(np.quantile(sims, 0.10)),
                "p90_doc_centroid_cosine": float(np.quantile(sims, 0.90)),
                "top10_term_share": float(top10_share),
            }
        )

    cluster_df = pd.DataFrame(cluster_rows)
    if cluster_df.empty:
        metrics = {
            "n_clusters_eval": 0,
            "eval_doc_count": 0,
            "weighted_mean_doc_centroid_cosine": float("nan"),
            "unweighted_mean_cluster_cosine": float("nan"),
            "median_cluster_cosine": float("nan"),
            "p10_cluster_cosine": float("nan"),
            "p90_cluster_cosine": float("nan"),
            "weighted_mean_top10_term_share": float("nan"),
            "unweighted_mean_top10_term_share": float("nan"),
            "mean_cluster_text_docs": float("nan"),
            "max_cluster_text_docs": 0,
        }
        return metrics, cluster_df

    weights = cluster_df["text_doc_count"].to_numpy(dtype=np.float64)
    values = cluster_df["mean_doc_centroid_cosine"].to_numpy(dtype=np.float64)
    top10_values = cluster_df["top10_term_share"].to_numpy(dtype=np.float64)
    metrics = {
        "n_clusters_eval": int(len(cluster_df)),
        "eval_doc_count": int(cluster_df["text_doc_count"].sum()),
        "weighted_mean_doc_centroid_cosine": _weighted_mean(values, weights),
        "unweighted_mean_cluster_cosine": float(np.mean(values)),
        "median_cluster_cosine": float(np.median(values)),
        "p10_cluster_cosine": float(np.quantile(values, 0.10)),
        "p90_cluster_cosine": float(np.quantile(values, 0.90)),
        "weighted_mean_top10_term_share": _weighted_mean(top10_values, weights),
        "unweighted_mean_top10_term_share": float(np.mean(top10_values)),
        "mean_cluster_text_docs": float(np.mean(weights)),
        "max_cluster_text_docs": int(np.max(weights)),
    }
    return metrics, cluster_df


def _field_metrics(
    *,
    config: SampleConfig,
    rows: pd.DataFrame,
    output_dir: Path,
    min_cluster_docs: int,
    max_features: int,
    min_df: int,
    ngram_max: int,
    force_manifest: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    first_membership = _repo_path(rows.iloc[0]["membership_path"])
    if first_membership is None:
        raise ValueError(f"Missing membership path for {config.sample}")
    n_nodes = _membership_node_count(first_membership)
    manifest, manifest_source = _load_manifest(config, n_nodes, output_dir, force_manifest)
    docs = _load_text_docs(config, manifest)
    matrix, vectorizer = _build_tfidf(
        docs["text"],
        max_features=max_features,
        min_df=min_df,
        ngram_max=ngram_max,
    )
    node_idx = docs["node_idx"].to_numpy(dtype=np.int64, copy=False)

    effect_rows: list[dict[str, Any]] = []
    cluster_tables: list[pd.DataFrame] = []
    for _, row in rows.iterrows():
        membership_path = _repo_path(row["membership_path"])
        if membership_path is None or not membership_path.exists():
            raise FileNotFoundError(f"Missing membership path: {row['membership_path']}")
        membership = _read_membership(membership_path)
        if len(membership) != n_nodes:
            raise ValueError(
                f"{config.sample} membership {membership_path} has {len(membership)} rows, "
                f"expected {n_nodes}"
            )
        labels = _labels_for_docs(membership, node_idx)
        total_clusters = int(membership["cluster"].nunique())
        text_clusters = int(pd.Series(labels).nunique())
        label_counts = pd.Series(labels).value_counts()
        small_text_clusters = int((label_counts < int(min_cluster_docs)).sum())
        metrics, cluster_df = _cluster_coherence(
            matrix,
            labels,
            min_cluster_docs=min_cluster_docs,
        )
        effect_row = {
            "sample": config.sample,
            "field": int(config.field_id),
            "seed": int(row["seed"]),
            "policy": str(row["policy"]),
            "membership_path": str(membership_path),
            "manifest_source": manifest_source,
            "n_graph_nodes": int(n_nodes),
            "n_text_docs_joined": int(len(docs)),
            "text_coverage_vs_graph": float(len(docs) / n_nodes) if n_nodes else float("nan"),
            "n_tfidf_features": int(len(vectorizer.vocabulary_)),
            "n_clusters_total": total_clusters,
            "n_clusters_with_text": text_clusters,
            "small_text_clusters_below_min": small_text_clusters,
            "eval_doc_share_of_text": float(metrics["eval_doc_count"] / len(docs)) if len(docs) else float("nan"),
            **metrics,
        }
        effect_rows.append(effect_row)
        if not cluster_df.empty:
            cluster_df = cluster_df.copy()
            cluster_df.insert(0, "policy", str(row["policy"]))
            cluster_df.insert(0, "seed", int(row["seed"]))
            cluster_df.insert(0, "field", int(config.field_id))
            cluster_df.insert(0, "sample", config.sample)
            cluster_tables.append(cluster_df)

    metadata = {
        "sample": config.sample,
        "field": int(config.field_id),
        "manifest_source": manifest_source,
        "n_graph_nodes": int(n_nodes),
        "n_text_docs_joined": int(len(docs)),
        "text_coverage_vs_graph": float(len(docs) / n_nodes) if n_nodes else float("nan"),
        "n_tfidf_features": int(len(vectorizer.vocabulary_)),
    }
    effects = pd.DataFrame(effect_rows)
    clusters = pd.concat(cluster_tables, ignore_index=True) if cluster_tables else pd.DataFrame()
    return effects, clusters, metadata


def _policy_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, group in effects.groupby("policy", sort=True):
        rows.append(
            {
                "policy": policy,
                "n_runs": int(len(group)),
                "mean_weighted_doc_centroid_cosine": float(
                    group["weighted_mean_doc_centroid_cosine"].mean()
                ),
                "median_weighted_doc_centroid_cosine": float(
                    group["weighted_mean_doc_centroid_cosine"].median()
                ),
                "mean_unweighted_cluster_cosine": float(group["unweighted_mean_cluster_cosine"].mean()),
                "mean_eval_doc_share_of_text": float(group["eval_doc_share_of_text"].mean()),
                "mean_text_coverage_vs_graph": float(group["text_coverage_vs_graph"].mean()),
                "mean_weighted_top10_term_share": float(group["weighted_mean_top10_term_share"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _quality_first_pairwise(effects: pd.DataFrame) -> pd.DataFrame:
    key = ["sample", "field", "seed"]
    small = effects[effects["policy"] == "small_only"].copy()
    quality = effects[effects["policy"] == "two_stage_quality_first"].copy()
    merged = quality.merge(
        small,
        on=key,
        suffixes=("_quality_first", "_small_only"),
        validate="one_to_one",
    )
    delta_cols = [
        "weighted_mean_doc_centroid_cosine",
        "unweighted_mean_cluster_cosine",
        "median_cluster_cosine",
        "p10_cluster_cosine",
        "p90_cluster_cosine",
        "weighted_mean_top10_term_share",
        "unweighted_mean_top10_term_share",
        "eval_doc_share_of_text",
        "n_clusters_eval",
        "eval_doc_count",
    ]
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        out = {column: row[column] for column in key}
        for col in delta_cols:
            out[f"{col}_small_only"] = row[f"{col}_small_only"]
            out[f"{col}_quality_first"] = row[f"{col}_quality_first"]
            out[f"delta_{col}"] = row[f"{col}_quality_first"] - row[f"{col}_small_only"]
        out["coherence_non_decreasing"] = bool(
            out["delta_weighted_mean_doc_centroid_cosine"] >= -1e-12
        )
        out["top10_term_share_non_increasing"] = bool(
            out["delta_weighted_mean_top10_term_share"] <= 1e-12
        )
        rows.append(out)
    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def _field_breakdown(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (field, sample), group in pairwise.groupby(["field", "sample"], sort=True):
        rows.append(
            {
                "field": int(field),
                "sample": sample,
                "n_pairs": int(len(group)),
                "mean_delta_weighted_doc_centroid_cosine": float(
                    group["delta_weighted_mean_doc_centroid_cosine"].mean()
                ),
                "median_delta_weighted_doc_centroid_cosine": float(
                    group["delta_weighted_mean_doc_centroid_cosine"].median()
                ),
                "coherence_non_decreasing_pairs": int(group["coherence_non_decreasing"].sum()),
                "mean_delta_unweighted_cluster_cosine": float(
                    group["delta_unweighted_mean_cluster_cosine"].mean()
                ),
                "mean_delta_weighted_top10_term_share": float(
                    group["delta_weighted_mean_top10_term_share"].mean()
                ),
                "top10_term_share_non_increasing_pairs": int(
                    group["top10_term_share_non_increasing"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_outputs(effects: pd.DataFrame, pairwise: pd.DataFrame) -> dict[str, Any]:
    fields = tuple(sorted(int(v) for v in effects["field"].unique()))
    expected_pairs = len(EXPECTED_FIELDS) * len(DEFAULT_SEEDS)
    checks = {
        "expected_fields_present": fields == EXPECTED_FIELDS,
        "pairwise_rows": int(len(pairwise)),
        "expected_pairwise_rows": int(expected_pairs),
        "pairwise_row_count_ok": int(len(pairwise)) == int(expected_pairs),
        "main_metric_non_null": bool(
            effects["weighted_mean_doc_centroid_cosine"].notna().all()
            and pairwise["delta_weighted_mean_doc_centroid_cosine"].notna().all()
        ),
        "all_text_coverage_positive": bool((effects["text_coverage_vs_graph"] > 0).all()),
    }
    checks["passed"] = bool(
        checks["expected_fields_present"]
        and checks["pairwise_row_count_ok"]
        and checks["main_metric_non_null"]
        and checks["all_text_coverage_positive"]
    )
    return checks


def _plot_pairwise(pairwise: pd.DataFrame, output_path: Path) -> None:
    field_summary = _field_breakdown(pairwise)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].bar(
        field_summary["field"].astype(str),
        field_summary["mean_delta_weighted_doc_centroid_cosine"],
        color="#31688e",
    )
    axes[0].set_title("Mean coherence delta by field")
    axes[0].set_xlabel("Field")
    axes[0].set_ylabel("quality_first - small_only")

    jitter = np.linspace(-0.08, 0.08, len(DEFAULT_SEEDS))
    seed_to_jitter = {seed: jitter[i] for i, seed in enumerate(DEFAULT_SEEDS)}
    x_positions = {field: i for i, field in enumerate(sorted(pairwise["field"].unique()))}
    for seed, group in pairwise.groupby("seed", sort=True):
        xs = [x_positions[int(field)] + seed_to_jitter.get(int(seed), 0.0) for field in group["field"]]
        axes[1].scatter(
            xs,
            group["delta_weighted_mean_doc_centroid_cosine"],
            label=f"seed {int(seed)}",
            s=35,
            alpha=0.85,
        )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(list(x_positions.values()), [str(field) for field in x_positions])
    axes[1].set_title("Seed-level coherence deltas")
    axes[1].set_xlabel("Field")
    axes[1].set_ylabel("quality_first - small_only")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Semantic coherence validation from title/abstract TF-IDF", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(
    *,
    output_dir: Path,
    effects: pd.DataFrame,
    pairwise: pd.DataFrame,
    policy_summary: pd.DataFrame,
    field_breakdown: pd.DataFrame,
    validation: dict[str, Any],
    metadata: list[dict[str, Any]],
) -> None:
    mean_delta = float(pairwise["delta_weighted_mean_doc_centroid_cosine"].mean())
    non_decreasing = int(pairwise["coherence_non_decreasing"].sum())
    n_pairs = int(len(pairwise))
    top10_mean_delta = float(pairwise["delta_weighted_mean_top10_term_share"].mean())
    coverage_min = float(effects["text_coverage_vs_graph"].min())
    coverage_max = float(effects["text_coverage_vs_graph"].max())
    report = [
        "# Semantic Coherence Validation",
        "",
        "This validation compares `small_only` and `two_stage_quality_first` effective source memberships across the six-field expansion.",
        "It uses field-local title/abstract TF-IDF and cluster centroid cosine coherence, conditional on documents with available text.",
        "",
        "## Readout",
        "",
        f"- Validation passed: `{validation['passed']}`.",
        f"- Pairwise quality_first vs small_only rows: `{n_pairs}/{validation['expected_pairwise_rows']}`.",
        f"- Mean weighted doc-centroid coherence delta: `{mean_delta:.6f}`.",
        f"- Non-decreasing coherence pairs: `{non_decreasing}/{n_pairs}`.",
        f"- Mean weighted top-10 term-share delta: `{top10_mean_delta:.6f}`.",
        f"- Text coverage vs graph ranges from `{coverage_min:.4f}` to `{coverage_max:.4f}`; interpret as a text-available subset check, not full-graph semantic coverage.",
        "",
        "## Policy Summary",
        "",
        _markdown_table(policy_summary.round(6)),
        "",
        "## Field Breakdown",
        "",
        _markdown_table(field_breakdown.round(6)),
        "",
        "## Field Text Metadata",
        "",
        _markdown_table(pd.DataFrame(metadata).round(6)),
        "",
        "## Caveat",
        "",
        "OpenAlex text availability is much smaller than the graph node set for several fields, so this is a local semantic sanity check over available title/abstract documents.",
        "The quantitative hierarchy evidence remains the primary support for balance and propagation claims.",
        "",
    ]
    (output_dir / "semantic_coherence_report.md").write_text("\n".join(report), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--fields", default=",".join(str(v) for v in EXPECTED_FIELDS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--min-cluster-text-docs", type=int, default=DEFAULT_MIN_CLUSTER_TEXT_DOCS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--skip-cluster-table", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = _repo_path(args.output_dir)
    run_root = _repo_path(args.run_root)
    input_path = _repo_path(args.input)
    assert output_dir is not None and run_root is not None and input_path is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    fields = {int(v) for v in str(args.fields).split(",") if str(v).strip()}
    seeds = {int(v) for v in str(args.seeds).split(",") if str(v).strip()}
    effects_input = pd.read_csv(input_path)
    effects_input = effects_input[
        effects_input["membership_role"].eq("effective")
        & effects_input["policy"].isin(DEFAULT_POLICIES)
        & effects_input["seed"].astype(int).isin(seeds)
    ].copy()
    effects_input["field"] = effects_input["sample"].map(_field_id_from_sample)
    effects_input = effects_input[effects_input["field"].isin(fields)].copy()
    if effects_input.empty:
        raise ValueError(f"No matching effective membership rows in {input_path}")

    all_effects: list[pd.DataFrame] = []
    all_clusters: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for sample, sample_rows in effects_input.groupby("sample", sort=True):
        config = _sample_config(sample)
        print(f"[semantic-coherence] evaluating {sample}", flush=True)
        sample_effects, sample_clusters, sample_metadata = _field_metrics(
            config=config,
            rows=sample_rows,
            output_dir=run_root,
            min_cluster_docs=int(args.min_cluster_text_docs),
            max_features=int(args.max_features),
            min_df=int(args.min_df),
            ngram_max=int(args.ngram_max),
            force_manifest=bool(args.force_manifest),
        )
        all_effects.append(sample_effects)
        if not args.skip_cluster_table and not sample_clusters.empty:
            all_clusters.append(sample_clusters)
        metadata.append(sample_metadata)
        print(
            "[semantic-coherence] completed "
            f"{sample}: text_docs={sample_metadata['n_text_docs_joined']} "
            f"features={sample_metadata['n_tfidf_features']}",
            flush=True,
        )

    effects = pd.concat(all_effects, ignore_index=True)
    effects = effects.sort_values(["field", "seed", "policy"]).reset_index(drop=True)
    pairwise = _quality_first_pairwise(effects)
    summary = _policy_summary(effects)
    field_breakdown = _field_breakdown(pairwise)
    validation = _validate_outputs(effects, pairwise)

    _write_table(effects, output_dir / "semantic_coherence_effects.csv")
    _write_table(summary, output_dir / "semantic_coherence_policy_summary.csv")
    _write_table(pairwise, output_dir / "semantic_coherence_quality_first_vs_small_only.csv")
    _write_table(field_breakdown, output_dir / "semantic_coherence_field_breakdown.csv")
    if all_clusters:
        cluster_metrics = pd.concat(all_clusters, ignore_index=True)
        _write_table(cluster_metrics, output_dir / "semantic_coherence_cluster_metrics.csv")

    summary_payload = {
        "validation": validation,
        "metadata": metadata,
        "mean_delta_weighted_doc_centroid_cosine": float(
            pairwise["delta_weighted_mean_doc_centroid_cosine"].mean()
        ),
        "coherence_non_decreasing_pairs": int(pairwise["coherence_non_decreasing"].sum()),
        "n_pairwise": int(len(pairwise)),
        "mean_delta_weighted_top10_term_share": float(
            pairwise["delta_weighted_mean_top10_term_share"].mean()
        ),
    }
    (output_dir / "semantic_coherence_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        effects=effects,
        pairwise=pairwise,
        policy_summary=summary,
        field_breakdown=field_breakdown,
        validation=validation,
        metadata=metadata,
    )
    if not args.skip_figure:
        _plot_pairwise(pairwise, output_dir / "figure11_semantic_coherence.png")

    if not validation["passed"]:
        raise SystemExit(f"Semantic coherence validation failed: {validation}")


if __name__ == "__main__":
    main()
