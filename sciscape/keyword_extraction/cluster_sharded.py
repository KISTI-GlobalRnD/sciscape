"""Cluster-sharded keyword extraction engine.

This opt-in engine keeps the legacy CountVectorizer pipeline untouched.  It is
designed for large Nano-level runs where the safe execution unit is a cluster
shard rather than a document row batch or a dense cluster x term matrix.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .abbreviations import build_abbreviation_lookup, extract_parenthetical_abbreviations
from .config import KeywordExtractionConfig
from .extraction import _DataSource, _effective_n_jobs
from .quality import annotate_keyword_quality, write_keyword_quality_residual_report
from .utils import _looks_like_metadata_artifact_term, _normalize_text_basic

try:  # optional but already required by parquet-heavy workflows
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_ARROW = True
except Exception:  # pragma: no cover
    pa = None
    pq = None
    _HAS_ARROW = False


_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_ACRONYM_LIKE_RE = re.compile(r"^(?:[a-z]{2,6}|\d+[a-z]{1,4}|[a-z]{1,4}\d+)$")
_TERM_SEP = "|"


@dataclass(frozen=True)
class ClusterShard:
    """One cluster-shard execution unit."""

    shard_id: int
    cluster_ids: tuple[int, ...]
    doc_count: int
    cap_sum: int
    max_cluster_doc_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cluster_ids"] = list(self.cluster_ids)
        return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def current_rss_mb() -> Optional[float]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        rss_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except Exception:
        return None
    return float(rss_pages * page_size) / (1024.0 * 1024.0)


def stable_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def done_matches(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "complete" and payload.get("fingerprint") == fingerprint


def resolve_cluster_sharded_output_dir(config: KeywordExtractionConfig) -> Path:
    if config.cluster_sharded_output_dir is not None:
        return Path(config.cluster_sharded_output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("workspace/artifacts/keyword_cluster_sharded") / stamp


def adaptive_candidate_cap(doc_count: int, config: KeywordExtractionConfig) -> int:
    """Return a bounded candidate-pool cap for one cluster."""

    floor = int(config.candidate_pool_floor)
    hard_max = int(config.candidate_pool_hard_max)
    if doc_count <= 250:
        raw = floor
    else:
        raw = floor + 128.0 * math.log2(max(1.0, float(doc_count) / 250.0))
    if doc_count >= 5_000:
        raw = max(raw, float(config.candidate_pool_large))
    if doc_count >= 10_000:
        raw = max(raw, float(config.candidate_pool_large) + 256.0)
    round_unit = 64.0 if hard_max >= 64 else 1.0
    rounded = int(math.ceil(raw / round_unit) * round_unit)
    return max(floor, min(hard_max, rounded))


def build_cluster_shard_manifest(
    config: KeywordExtractionConfig,
    data_source: Optional[_DataSource] = None,
) -> dict[str, Any]:
    """Build a document-balanced cluster shard manifest."""

    source = data_source or _DataSource(config)
    membership = source.membership_map()
    counts = membership.value_counts().sort_index()
    cluster_caps = {
        int(cluster_id): adaptive_candidate_cap(int(doc_count), config)
        for cluster_id, doc_count in counts.items()
    }
    records = [
        {
            "cluster_id": int(cluster_id),
            "doc_count": int(doc_count),
            "candidate_cap": int(cluster_caps[int(cluster_id)]),
        }
        for cluster_id, doc_count in counts.items()
    ]
    target_docs = max(1, int(config.target_docs_per_shard))
    max_clusters = max(1, int(config.max_clusters_per_shard))
    large_single = bool(config.large_cluster_single_shard)

    shards: list[ClusterShard] = []
    current_clusters: list[int] = []
    current_docs = 0
    current_cap = 0
    current_max_docs = 0

    def flush() -> None:
        nonlocal current_clusters, current_docs, current_cap, current_max_docs
        if not current_clusters:
            return
        shards.append(
            ClusterShard(
                shard_id=len(shards),
                cluster_ids=tuple(current_clusters),
                doc_count=int(current_docs),
                cap_sum=int(current_cap),
                max_cluster_doc_count=int(current_max_docs),
            )
        )
        current_clusters = []
        current_docs = 0
        current_cap = 0
        current_max_docs = 0

    for record in sorted(records, key=lambda row: (-int(row["doc_count"]), int(row["cluster_id"]))):
        cluster_id = int(record["cluster_id"])
        doc_count = int(record["doc_count"])
        cap = int(record["candidate_cap"])
        if large_single and doc_count >= target_docs:
            flush()
            shards.append(
                ClusterShard(
                    shard_id=len(shards),
                    cluster_ids=(cluster_id,),
                    doc_count=doc_count,
                    cap_sum=cap,
                    max_cluster_doc_count=doc_count,
                )
            )
            continue
        would_overflow_docs = current_clusters and current_docs + doc_count > target_docs
        would_overflow_clusters = len(current_clusters) >= max_clusters
        if would_overflow_docs or would_overflow_clusters:
            flush()
        current_clusters.append(cluster_id)
        current_docs += doc_count
        current_cap += cap
        current_max_docs = max(current_max_docs, doc_count)
    flush()

    return {
        "schema_version": "sciscape_keyword_cluster_shards_v1",
        "created_at_utc": utc_now_iso(),
        "cluster_level": config.cluster_level,
        "total_clusters": int(len(counts)),
        "total_docs": int(counts.sum()),
        "target_docs_per_shard": int(config.target_docs_per_shard),
        "max_clusters_per_shard": int(config.max_clusters_per_shard),
        "candidate_pool_floor": int(config.candidate_pool_floor),
        "candidate_pool_hard_max": int(config.candidate_pool_hard_max),
        "candidate_row_budget": {
            "target": int(config.global_candidate_row_target),
            "warning": int(config.global_candidate_row_warning),
            "hard_stop": int(config.global_candidate_row_hard_stop),
        },
        "unique_term_budget": {
            "target": int(config.global_unique_term_target),
            "warning": int(config.global_unique_term_warning),
            "hard_stop": int(config.global_unique_term_hard_stop),
        },
        "clusters": records,
        "shards": [shard.to_dict() for shard in shards],
    }


def _numeric_summary(values: Iterable[int]) -> dict[str, float | int | None]:
    data = [int(value) for value in values]
    if not data:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    arr = np.asarray(data, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def run_cluster_sharded_preflight(config: KeywordExtractionConfig) -> dict[str, Any]:
    """Write a cheap membership-only budget report for the V2 keyword engine."""

    output_dir = resolve_cluster_sharded_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_cluster_shard_manifest(config)
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    clusters = list(manifest.get("clusters", []))
    shards = list(manifest.get("shards", []))
    expected_candidate_rows = int(sum(int(row.get("candidate_cap", 0)) for row in clusters))
    warning = int(config.global_candidate_row_warning)
    hard_stop = int(config.global_candidate_row_hard_stop)
    target = int(config.global_candidate_row_target)
    if expected_candidate_rows > hard_stop:
        status = "hard_stop"
    elif expected_candidate_rows > warning:
        status = "warning"
    elif expected_candidate_rows > target:
        status = "above_target"
    else:
        status = "ok"

    summary: dict[str, Any] = {
        "schema_version": "sciscape_keyword_cluster_sharded_preflight_v1",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "should_run_full_extraction": expected_candidate_rows <= hard_stop,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "abstract_path": str(config.abstract_path),
        "membership_path": str(config.membership_path),
        "cluster_level": config.cluster_level,
        "total_clusters": int(manifest.get("total_clusters", 0)),
        "total_docs": int(manifest.get("total_docs", 0)),
        "shard_count": int(len(shards)),
        "expected_candidate_rows_upper_bound": expected_candidate_rows,
        "candidate_row_budget": manifest.get("candidate_row_budget", {}),
        "unique_term_budget": manifest.get("unique_term_budget", {}),
        "candidate_row_target_exceeded": expected_candidate_rows > target,
        "candidate_row_warning": expected_candidate_rows > warning,
        "candidate_row_hard_stop": expected_candidate_rows > hard_stop,
        "cluster_doc_count_stats": _numeric_summary(int(row.get("doc_count", 0)) for row in clusters),
        "candidate_cap_stats": _numeric_summary(int(row.get("candidate_cap", 0)) for row in clusters),
        "shard_doc_count_stats": _numeric_summary(int(row.get("doc_count", 0)) for row in shards),
        "shard_cap_sum_stats": _numeric_summary(int(row.get("cap_sum", 0)) for row in shards),
        "shard_cluster_count_stats": _numeric_summary(len(row.get("cluster_ids", [])) for row in shards),
        "largest_clusters": sorted(
            (
                {
                    "cluster_id": int(row.get("cluster_id", -1)),
                    "doc_count": int(row.get("doc_count", 0)),
                    "candidate_cap": int(row.get("candidate_cap", 0)),
                }
                for row in clusters
            ),
            key=lambda row: (-row["doc_count"], row["cluster_id"]),
        )[:10],
    }
    summary_path = output_dir / "preflight_summary.json"
    summary["preflight_summary_path"] = str(summary_path)
    write_json_atomic(summary_path, summary)
    return summary


def _manifest_shards(manifest: Mapping[str, Any]) -> list[ClusterShard]:
    return [
        ClusterShard(
            shard_id=int(row["shard_id"]),
            cluster_ids=tuple(int(cid) for cid in row.get("cluster_ids", [])),
            doc_count=int(row.get("doc_count", 0)),
            cap_sum=int(row.get("cap_sum", 0)),
            max_cluster_doc_count=int(row.get("max_cluster_doc_count", 0)),
        )
        for row in manifest.get("shards", [])
    ]


def _requested_shard_ids(config: KeywordExtractionConfig, manifest: Mapping[str, Any]) -> set[int] | None:
    raw_ids = config.cluster_sharded_shard_ids
    if raw_ids is None:
        return None
    requested = {int(value) for value in raw_ids}
    available = {
        int(row["shard_id"])
        for row in manifest.get("shards", [])
        if isinstance(row, Mapping) and row.get("shard_id") is not None
    }
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"unknown cluster-sharded shard IDs: {unknown}")
    return requested


def _candidate_shard_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "candidates" / f"candidate_shard_{int(shard_id):04d}.parquet"


def _final_shard_path(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "final" / f"keyword_shard_{int(shard_id):04d}.parquet"


def _candidate_paths_for_manifest(
    manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    require_existing: bool = False,
) -> list[Path]:
    paths = [
        _candidate_shard_path(output_dir, int(row["shard_id"]))
        for row in manifest.get("shards", [])
        if isinstance(row, Mapping) and row.get("shard_id") is not None
    ]
    if require_existing:
        missing = [path for path in paths if not path.exists() or not path.with_suffix(".done.json").exists()]
        if missing:
            sample = ", ".join(str(path) for path in missing[:5])
            suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise RuntimeError(
                "selected cluster-sharded rerun requires complete candidate files "
                f"for every shard; missing {sample}{suffix}"
            )
    return paths


def _cluster_cap_map(manifest: Mapping[str, Any]) -> dict[int, int]:
    return {
        int(row["cluster_id"]): int(row["candidate_cap"])
        for row in manifest.get("clusters", [])
    }


def _raw_document_batches(config: KeywordExtractionConfig) -> Iterator[pd.DataFrame]:
    """Yield normalized raw title/abstract/year rows joined to cluster ids."""

    source = _DataSource(config)
    membership = source.membership_map()
    uid_col = config.uid_col
    title_col = config.title_col
    abstract_col = config.abstract_col
    year_col = config.year_col

    if _HAS_ARROW and config.use_pyarrow_streaming:
        pf = pq.ParquetFile(str(config.abstract_path))
        names = set(pf.schema_arrow.names)
        columns = [uid_col, abstract_col]
        if title_col in names:
            columns.append(title_col)
        if year_col in names:
            columns.append(year_col)
        for row_group in range(pf.num_row_groups):
            docs = pf.read_row_group(row_group, columns=columns).to_pandas()
            docs["cluster_id"] = docs[uid_col].map(membership)
            docs = docs.dropna(subset=["cluster_id"])
            if docs.empty:
                continue
            if title_col not in docs.columns:
                docs[title_col] = ""
            if year_col not in docs.columns:
                docs[year_col] = pd.NA
            yield pd.DataFrame(
                {
                    uid_col: docs[uid_col].to_numpy(),
                    "cluster_id": docs["cluster_id"].astype(int).to_numpy(),
                    title_col: docs[title_col].fillna("").map(_normalize_text_basic).to_numpy(),
                    abstract_col: docs[abstract_col].fillna("").map(_normalize_text_basic).to_numpy(),
                    year_col: pd.to_numeric(docs[year_col], errors="coerce").astype("Int64").to_numpy(),
                }
            )
        return

    columns = [uid_col, abstract_col]
    try:
        docs = pd.read_parquet(config.abstract_path)
    except Exception:
        docs = pd.read_parquet(config.abstract_path, columns=columns)
    docs["cluster_id"] = docs[uid_col].map(membership)
    docs = docs.dropna(subset=["cluster_id"])
    if docs.empty:
        return
    if title_col not in docs.columns:
        docs[title_col] = ""
    if year_col not in docs.columns:
        docs[year_col] = pd.NA
    yield pd.DataFrame(
        {
            uid_col: docs[uid_col].to_numpy(),
            "cluster_id": docs["cluster_id"].astype(int).to_numpy(),
            title_col: docs[title_col].fillna("").map(_normalize_text_basic).to_numpy(),
            abstract_col: docs[abstract_col].fillna("").map(_normalize_text_basic).to_numpy(),
            year_col: pd.to_numeric(docs[year_col], errors="coerce").astype("Int64").to_numpy(),
        }
    )


def materialize_document_shards(
    config: KeywordExtractionConfig,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Scan source documents once and write cluster-shard parquet files."""

    if not _HAS_ARROW:
        raise RuntimeError("cluster_sharded keyword engine requires pyarrow for document shard writing")

    doc_dir = output_dir / "doc_shards"
    doc_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = stable_digest(
        {
            "stage": "document_sharding",
            "manifest": manifest,
            "abstract_path": str(config.abstract_path),
            "membership_path": str(config.membership_path),
            "title_col": config.title_col,
            "abstract_col": config.abstract_col,
            "year_col": config.year_col,
        }
    )
    shards = _manifest_shards(manifest)
    shard_paths = [doc_dir / f"doc_shard_{shard.shard_id:04d}.parquet" for shard in shards]
    done_paths = [path.with_suffix(".done.json") for path in shard_paths]
    if config.scoring_shard_resume and all(
        path.exists() and done_matches(done, fingerprint)
        for path, done in zip(shard_paths, done_paths)
    ):
        return shard_paths
    for path, done in zip(shard_paths, done_paths):
        path.unlink(missing_ok=True)
        done.unlink(missing_ok=True)

    cluster_to_shard = {
        cluster_id: shard.shard_id
        for shard in shards
        for cluster_id in shard.cluster_ids
    }
    writers: dict[int, Any] = {}
    row_counts: Counter[int] = Counter()
    uid_col = config.uid_col
    columns = [uid_col, "cluster_id", config.title_col, config.abstract_col, config.year_col]

    try:
        for batch in _raw_document_batches(config):
            if batch.empty:
                continue
            batch = batch.copy()
            batch["_shard_id"] = batch["cluster_id"].map(cluster_to_shard)
            batch = batch.dropna(subset=["_shard_id"])
            if batch.empty:
                continue
            batch["_shard_id"] = batch["_shard_id"].astype(int)
            for shard_id, group in batch.groupby("_shard_id", sort=False):
                shard_id_int = int(shard_id)
                out = group[columns].reset_index(drop=True)
                out[config.year_col] = pd.to_numeric(
                    out[config.year_col],
                    errors="coerce",
                ).astype("Int64")
                table = pa.Table.from_pandas(out, preserve_index=False)
                path = shard_paths[shard_id_int]
                writer = writers.get(shard_id_int)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                    writers[shard_id_int] = writer
                writer.write_table(table)
                row_counts[shard_id_int] += len(out)
    finally:
        for writer in writers.values():
            writer.close()

    empty_columns = {
        uid_col: pd.Series(dtype=object),
        "cluster_id": pd.Series(dtype="int64"),
        config.title_col: pd.Series(dtype=object),
        config.abstract_col: pd.Series(dtype=object),
        config.year_col: pd.Series(dtype="Int64"),
    }
    for shard in shards:
        path = shard_paths[shard.shard_id]
        if not path.exists():
            pd.DataFrame(empty_columns).to_parquet(path, index=False)
        write_json_atomic(
            path.with_suffix(".done.json"),
            {
                "schema_version": "sciscape_keyword_doc_shard_done_v1",
                "created_at_utc": utc_now_iso(),
                "status": "complete",
                "fingerprint": fingerprint,
                "shard_id": int(shard.shard_id),
                "rows": int(row_counts[shard.shard_id]),
                "path": str(path),
            },
        )
    return shard_paths


def _tokens(text: object, config: KeywordExtractionConfig, stopwords: set[str]) -> list[str]:
    raw = _normalize_text_basic(text)
    if config.lowercase:
        raw = raw.lower()
    tokens = _TOKEN_RE.findall(raw)
    return [tok for tok in tokens if tok and tok not in stopwords]


def _ngrams(tokens: list[str], config: KeywordExtractionConfig) -> list[str]:
    terms: list[str] = list(tokens)
    if config.use_phrase_vectorizer and config.ngram_max >= 2:
        lo = max(2, int(config.ngram_min))
        hi = max(lo, int(config.ngram_max))
        for n in range(lo, hi + 1):
            if len(tokens) < n:
                continue
            terms.extend(" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))
    return terms


def _artifact_risk(term: str, artifact_res: Iterable[re.Pattern[str]]) -> float:
    if not term:
        return 1.0
    if _looks_like_metadata_artifact_term(term):
        return 1.0
    if any(pattern.search(term) for pattern in artifact_res):
        return 1.0
    tokens = term.split()
    if not tokens:
        return 1.0
    if len(term) <= 2 and term not in {"ai", "ml", "dl", "2d", "3d", "uv", "ir", "ph"}:
        return 1.0
    if len(tokens) > 8:
        return 0.75
    if any(tok.isdigit() for tok in tokens) and sum(tok.isalpha() for tok in tokens) < len(tokens) * 0.5:
        return 0.8
    return 0.0


def _is_acronym_like(term: str) -> bool:
    return " " not in term and bool(_ACRONYM_LIKE_RE.match(term)) and term not in ENGLISH_STOP_WORDS


def _candidate_hint(row: Mapping[str, Any]) -> float:
    phrase_bonus = 0.25 if " " in str(row["term"]) else 0.0
    recent_bonus = 0.18 if "recent" in str(row.get("channel_flags", "")) else 0.0
    return (
        math.log1p(float(row.get("local_tf", 0)))
        + 0.35 * math.log1p(float(row.get("local_doc_df", 0)))
        + 0.65 * math.log1p(float(row.get("title_tf", 0)))
        + phrase_bonus
        + recent_bonus
        - 2.0 * float(row.get("artifact_risk", 0.0))
    )


def _stats_candidate_hint(term: str, row: Mapping[str, Any]) -> float:
    flags = set(row.get("channel_flags", set()))
    if " " in term:
        flags.add("phrase_ngram")
    if int(row.get("title_tf", 0)) > 0:
        flags.add("title_weighted")
    return _candidate_hint(
        {
            "term": term,
            "local_tf": int(row.get("local_tf", 0)),
            "local_doc_df": int(row.get("local_doc_df", 0)),
            "title_tf": int(row.get("title_tf", 0)),
            "artifact_risk": float(row.get("artifact_risk", 0.0)),
            "channel_flags": _TERM_SEP.join(sorted(str(flag) for flag in flags)),
        }
    )


def _tracked_term_count(cluster_stats: Mapping[int, Mapping[str, Any]]) -> int:
    return int(sum(len(stats) for stats in cluster_stats.values()))


def _prune_cluster_stats(
    cluster_stats: dict[int, dict[str, dict[str, Any]]],
    cap_map: Mapping[int, int],
    config: KeywordExtractionConfig,
) -> int:
    multiplier = max(1, int(config.candidate_mining_prune_multiplier))
    removed_total = 0
    for cluster_id, stats in list(cluster_stats.items()):
        cap = int(cap_map.get(cluster_id, adaptive_candidate_cap(0, config)))
        limit = max(cap, cap * multiplier)
        if len(stats) <= limit:
            continue
        keep_terms = {
            term
            for term, _score in sorted(
                ((term, _stats_candidate_hint(term, row)) for term, row in stats.items()),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        }
        before = len(stats)
        for term in list(stats.keys()):
            if term not in keep_terms:
                del stats[term]
        removed_total += before - len(stats)
    return int(removed_total)


def _select_cluster_candidates(
    cluster_id: int,
    stats: dict[str, dict[str, Any]],
    cap: int,
) -> list[dict[str, Any]]:
    if not stats:
        return []
    max_year = max((int(row["last_year"]) for row in stats.values() if row.get("last_year") is not None), default=None)
    rows: list[dict[str, Any]] = []
    for term, row in stats.items():
        flags = set(row["channel_flags"])
        if " " in term:
            flags.add("phrase_ngram")
        if int(row["title_tf"]) > 0:
            flags.add("title_weighted")
        if _is_acronym_like(term):
            flags.add("acronym_like")
        if max_year is not None and row.get("last_year") is not None and int(row["last_year"]) >= max_year - 2:
            flags.add("recent")
        out = {
            "cluster_id": int(cluster_id),
            "term": term,
            "local_tf": int(row["local_tf"]),
            "local_doc_df": int(row["local_doc_df"]),
            "title_tf": int(row["title_tf"]),
            "abstract_tf": int(row["abstract_tf"]),
            "first_year": row.get("first_year"),
            "last_year": row.get("last_year"),
            "channel_flags": _TERM_SEP.join(sorted(flags)),
            "artifact_risk": float(row["artifact_risk"]),
        }
        out["candidate_score_hint"] = _candidate_hint(out)
        rows.append(out)

    def top(channel: str, quota: int, key) -> list[dict[str, Any]]:
        if quota <= 0:
            return []
        selected = [row for row in rows if channel in str(row["channel_flags"]).split(_TERM_SEP)]
        selected.sort(key=key, reverse=True)
        return selected[:quota]

    quotas = {
        "frequency": max(1, int(cap * 0.34)),
        "title_weighted": max(1, int(cap * 0.18)),
        "phrase_ngram": max(1, int(cap * 0.28)),
        "acronym_like": max(1, int(cap * 0.08)),
        "recent": max(1, int(cap * 0.12)),
    }
    selected: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["local_tf"], item["local_doc_df"]), reverse=True)[: quotas["frequency"]]:
        row["channel_flags"] = _merge_flags(row["channel_flags"], "frequency")
        selected[row["term"]] = row
    for row in top("title_weighted", quotas["title_weighted"], lambda item: (item["title_tf"], item["candidate_score_hint"])):
        selected[row["term"]] = row
    for row in top("phrase_ngram", quotas["phrase_ngram"], lambda item: (item["local_doc_df"], item["local_tf"])):
        selected[row["term"]] = row
    for row in top("acronym_like", quotas["acronym_like"], lambda item: (item["local_doc_df"], item["local_tf"])):
        selected[row["term"]] = row
    for row in top("recent", quotas["recent"], lambda item: (item["last_year"] or -1, item["local_doc_df"])):
        selected[row["term"]] = row

    if len(selected) < min(cap, len(rows)):
        for row in sorted(rows, key=lambda item: item["candidate_score_hint"], reverse=True):
            selected.setdefault(row["term"], row)
            if len(selected) >= min(cap, len(rows)):
                break
    chosen = list(selected.values())
    chosen.sort(key=lambda item: item["candidate_score_hint"], reverse=True)
    return chosen[:cap]


def _merge_flags(existing: str, flag: str) -> str:
    flags = {part for part in str(existing).split(_TERM_SEP) if part}
    flags.add(flag)
    return _TERM_SEP.join(sorted(flags))


def mine_candidate_shard(
    config: KeywordExtractionConfig,
    shard: Mapping[str, Any],
    doc_shard_path: Path,
    output_dir: Path,
) -> Path:
    """Mine one cluster-shard candidate parquet file."""

    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shard_id = int(shard["shard_id"])
    out_path = candidate_dir / f"candidate_shard_{shard_id:04d}.parquet"
    done_path = out_path.with_suffix(".done.json")
    progress_path = out_path.with_suffix(".progress.json")
    abbreviation_path = candidate_dir / f"abbreviation_shard_{shard_id:04d}.json"
    fingerprint = stable_digest(
        {
            "stage": "candidate_mining_abbrev_v1",
            "shard": shard,
            "doc_shard_path": str(doc_shard_path),
            "ngram_min": config.ngram_min,
            "ngram_max": config.ngram_max,
            "token_pattern": config.token_pattern,
            "lowercase": config.lowercase,
            "use_phrase_vectorizer": config.use_phrase_vectorizer,
            "artifact_filter_patterns": list(config.artifact_filter_patterns),
            "candidate_pool_hard_max": config.candidate_pool_hard_max,
            "candidate_mining_prune_interval_docs": config.candidate_mining_prune_interval_docs,
            "candidate_mining_prune_multiplier": config.candidate_mining_prune_multiplier,
            "abbreviation_dictionary_enabled": config.abbreviation_dictionary_enabled,
            "abbreviation_max_long_form_words": config.abbreviation_max_long_form_words,
        }
    )
    if (
        config.scoring_shard_resume
        and out_path.exists()
        and done_matches(done_path, fingerprint)
        and (not config.abbreviation_dictionary_enabled or abbreviation_path.exists())
    ):
        return out_path
    out_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    abbreviation_path.unlink(missing_ok=True)

    stopwords = config.build_stopword_set()
    artifact_res = [re.compile(pattern, re.IGNORECASE) for pattern in config.artifact_filter_patterns]
    cap_map = {
        int(row["cluster_id"]): int(row["candidate_cap"])
        for row in shard.get("clusters", [])
    }
    if not cap_map:
        cap_map = {int(cid): adaptive_candidate_cap(0, config) for cid in shard.get("cluster_ids", [])}

    start = time.monotonic()
    peak_rss_mb = current_rss_mb()

    def emit_progress(
        status: str,
        rows_processed: int,
        rows_total: int,
        cluster_stats: Mapping[int, Mapping[str, Any]],
        pruned_terms: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        nonlocal peak_rss_mb
        rss_mb = current_rss_mb()
        if rss_mb is not None:
            peak_rss_mb = max(float(peak_rss_mb or 0.0), float(rss_mb))
        elapsed = max(0.0, time.monotonic() - start)
        payload: dict[str, Any] = {
            "schema_version": "sciscape_keyword_candidate_shard_progress_v1",
            "updated_at_utc": utc_now_iso(),
            "status": status,
            "fingerprint": fingerprint,
            "shard_id": shard_id,
            "doc_shard_path": str(doc_shard_path),
            "output_path": str(out_path),
            "rows_total": int(rows_total),
            "rows_processed": int(rows_processed),
            "percent": float(rows_processed / rows_total * 100.0) if rows_total else 100.0,
            "clusters_seen": int(len(cluster_stats)),
            "terms_tracked": _tracked_term_count(cluster_stats),
            "pruned_terms": int(pruned_terms),
            "elapsed_sec": elapsed,
            "rows_per_sec": float(rows_processed / elapsed) if elapsed > 0 else None,
            "rss_mb": rss_mb,
            "peak_rss_mb": peak_rss_mb,
        }
        if extra:
            payload.update(dict(extra))
        write_json_atomic(progress_path, payload)

    docs = pd.read_parquet(doc_shard_path)
    rows_total = int(len(docs))
    abbreviation_rows = 0
    if config.abbreviation_dictionary_enabled:
        evidence = extract_parenthetical_abbreviations(
            docs,
            uid_col=config.uid_col,
            cluster_col="cluster_id",
            title_col=config.title_col,
            abstract_col=config.abstract_col,
            max_long_form_words=config.abbreviation_max_long_form_words,
        )
        abbreviation_rows = int(len(evidence))
        _write_abbreviation_evidence_json(abbreviation_path, evidence, shard_id=shard_id, fingerprint=fingerprint)
    else:
        _write_abbreviation_evidence_json(
            abbreviation_path,
            pd.DataFrame(),
            shard_id=shard_id,
            fingerprint=fingerprint,
        )
    cluster_stats: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    progress_interval = max(1, int(config.candidate_mining_progress_interval_docs))
    prune_interval = max(1, int(config.candidate_mining_prune_interval_docs))
    pruned_terms = 0
    processed = 0
    emit_progress("running", 0, rows_total, cluster_stats, pruned_terms)
    try:
        for processed, row in enumerate(docs.itertuples(index=False), start=1):
            cluster_id = int(getattr(row, "cluster_id"))
            title = getattr(row, config.title_col)
            abstract = getattr(row, config.abstract_col)
            year_value = getattr(row, config.year_col)
            year = None
            if pd.notna(year_value):
                try:
                    year = int(year_value)
                except (TypeError, ValueError):
                    year = None
            title_terms = _ngrams(_tokens(title, config, stopwords), config) if config.include_title else []
            abstract_terms = _ngrams(_tokens(abstract, config, stopwords), config)
            doc_terms = set(title_terms) | set(abstract_terms)
            stats = cluster_stats[cluster_id]
            for term, count in Counter(title_terms).items():
                term_stats = stats.setdefault(term, _empty_term_stats(term, artifact_res))
                term_stats["local_tf"] += int(count)
                term_stats["title_tf"] += int(count)
            for term, count in Counter(abstract_terms).items():
                term_stats = stats.setdefault(term, _empty_term_stats(term, artifact_res))
                term_stats["local_tf"] += int(count)
                term_stats["abstract_tf"] += int(count)
            for term in doc_terms:
                term_stats = stats.setdefault(term, _empty_term_stats(term, artifact_res))
                term_stats["local_doc_df"] += 1
                if year is not None:
                    if term_stats["first_year"] is None or year < int(term_stats["first_year"]):
                        term_stats["first_year"] = year
                    if term_stats["last_year"] is None or year > int(term_stats["last_year"]):
                        term_stats["last_year"] = year
            if processed % prune_interval == 0:
                pruned_terms += _prune_cluster_stats(cluster_stats, cap_map, config)
            if processed % progress_interval == 0:
                emit_progress("running", processed, rows_total, cluster_stats, pruned_terms)
    except Exception as exc:
        emit_progress(
            "failed",
            processed,
            rows_total,
            cluster_stats,
            pruned_terms,
            extra={"error": repr(exc)},
        )
        raise
    pruned_terms += _prune_cluster_stats(cluster_stats, cap_map, config)
    emit_progress("selecting", rows_total, rows_total, cluster_stats, pruned_terms)

    candidate_rows: list[dict[str, Any]] = []
    for cluster_id, stats in cluster_stats.items():
        cap = int(cap_map.get(cluster_id, adaptive_candidate_cap(int(len(stats)), config)))
        candidate_rows.extend(_select_cluster_candidates(cluster_id, stats, cap))

    columns = [
        "cluster_id",
        "term",
        "local_tf",
        "local_doc_df",
        "title_tf",
        "abstract_tf",
        "first_year",
        "last_year",
        "channel_flags",
        "artifact_risk",
        "candidate_score_hint",
    ]
    candidate_df = pd.DataFrame(candidate_rows, columns=columns)
    emit_progress(
        "writing",
        rows_total,
        rows_total,
        cluster_stats,
        pruned_terms,
        extra={"candidate_rows": int(len(candidate_df))},
    )
    candidate_df.to_parquet(out_path, index=False)
    elapsed = max(0.0, time.monotonic() - start)
    write_json_atomic(
        done_path,
        {
            "schema_version": "sciscape_keyword_candidate_shard_done_v1",
            "created_at_utc": utc_now_iso(),
            "status": "complete",
            "fingerprint": fingerprint,
            "shard_id": shard_id,
            "clusters": int(candidate_df["cluster_id"].nunique()) if not candidate_df.empty else 0,
            "rows": int(len(candidate_df)),
            "source_rows": int(rows_total),
            "abbreviation_evidence_rows": int(abbreviation_rows),
            "abbreviation_evidence_path": str(abbreviation_path),
            "elapsed_sec": elapsed,
            "rows_per_sec": float(rows_total / elapsed) if elapsed > 0 else None,
            "rss_mb": current_rss_mb(),
            "peak_rss_mb": peak_rss_mb,
            "terms_tracked": _tracked_term_count(cluster_stats),
            "pruned_terms": int(pruned_terms),
            "path": str(out_path),
        },
    )
    emit_progress(
        "complete",
        rows_total,
        rows_total,
        cluster_stats,
        pruned_terms,
        extra={"candidate_rows": int(len(candidate_df))},
    )
    return out_path


def _write_abbreviation_evidence_json(
    path: Path,
    evidence: pd.DataFrame,
    *,
    shard_id: int,
    fingerprint: str,
) -> None:
    records = [] if evidence is None or evidence.empty else json.loads(evidence.to_json(orient="records"))
    write_json_atomic(
        path,
        {
            "schema_version": "sciscape_keyword_abbreviation_evidence_shard_v1",
            "created_at_utc": utc_now_iso(),
            "status": "complete",
            "fingerprint": fingerprint,
            "shard_id": int(shard_id),
            "rows": int(len(records)),
            "evidence": records,
        },
    )


def _read_abbreviation_evidence_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("evidence", [])
    return [dict(record) for record in records if isinstance(record, Mapping)]


def _empty_term_stats(term: str, artifact_res: Iterable[re.Pattern[str]]) -> dict[str, Any]:
    return {
        "local_tf": 0,
        "local_doc_df": 0,
        "title_tf": 0,
        "abstract_tf": 0,
        "first_year": None,
        "last_year": None,
        "channel_flags": set(),
        "artifact_risk": _artifact_risk(term, artifact_res),
    }


def run_candidate_mining(
    config: KeywordExtractionConfig,
    manifest: Mapping[str, Any],
    doc_shard_paths: list[Path],
    output_dir: Path,
    active_shard_ids: set[int] | None = None,
) -> list[Path]:
    shards = manifest.get("shards", [])
    cluster_rows = {
        int(row["cluster_id"]): row
        for row in manifest.get("clusters", [])
    }
    enriched_shards = []
    for shard in shards:
        shard_id = int(shard["shard_id"])
        if active_shard_ids is not None and shard_id not in active_shard_ids:
            continue
        shard_copy = dict(shard)
        shard_copy["clusters"] = [cluster_rows[int(cid)] for cid in shard_copy.get("cluster_ids", [])]
        enriched_shards.append(shard_copy)

    n_jobs = _effective_n_jobs(config.n_jobs)
    backend = config.parallel_backend
    if backend == "sequential" or n_jobs == 1:
        return [
            mine_candidate_shard(config, shard, doc_shard_paths[int(shard["shard_id"])], output_dir)
            for shard in enriched_shards
        ]
    prefer = "threads" if backend == "threading" else "processes"
    return Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(mine_candidate_shard)(config, shard, doc_shard_paths[int(shard["shard_id"])], output_dir)
        for shard in enriched_shards
    )


def build_global_term_stats(
    config: KeywordExtractionConfig,
    candidate_paths: list[Path],
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Aggregate candidate rows into a term-level global stats table."""

    global_dir = output_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    out_path = global_dir / "global_term_stats.parquet"
    done_path = out_path.with_suffix(".done.json")
    candidate_done = []
    for path in candidate_paths:
        done_path_candidate = path.with_suffix(".done.json")
        if done_path_candidate.exists():
            try:
                candidate_done.append(json.loads(done_path_candidate.read_text(encoding="utf-8")))
            except Exception:
                candidate_done.append({"path": str(path)})
        else:
            candidate_done.append({"path": str(path)})
    fingerprint = stable_digest(
        {
            "stage": "global_term_stats",
            "manifest_total_clusters": manifest.get("total_clusters"),
            "candidate_done": candidate_done,
            "global_unique_term_target": config.global_unique_term_target,
            "global_unique_term_hard_stop": config.global_unique_term_hard_stop,
        }
    )
    if config.scoring_shard_resume and out_path.exists() and done_matches(done_path, fingerprint):
        return pd.read_parquet(out_path)

    stats: dict[str, dict[str, Any]] = {}
    total_candidate_rows = 0
    for path in candidate_paths:
        df = pd.read_parquet(path)
        total_candidate_rows += len(df)
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            term = str(row.term)
            term_stats = stats.setdefault(
                term,
                {
                    "term": term,
                    "total_tf": 0,
                    "total_doc_df": 0,
                    "cluster_df": 0,
                    "max_cluster_tf": 0,
                    "first_year": None,
                    "last_year": None,
                    "artifact_risk": 0.0,
                },
            )
            local_tf = int(row.local_tf)
            term_stats["total_tf"] += local_tf
            term_stats["total_doc_df"] += int(row.local_doc_df)
            term_stats["cluster_df"] += 1
            term_stats["max_cluster_tf"] = max(int(term_stats["max_cluster_tf"]), local_tf)
            term_stats["artifact_risk"] = max(float(term_stats["artifact_risk"]), float(row.artifact_risk))
            if pd.notna(row.first_year):
                year = int(row.first_year)
                if term_stats["first_year"] is None or year < int(term_stats["first_year"]):
                    term_stats["first_year"] = year
            if pd.notna(row.last_year):
                year = int(row.last_year)
                if term_stats["last_year"] is None or year > int(term_stats["last_year"]):
                    term_stats["last_year"] = year

    if total_candidate_rows > int(config.global_candidate_row_hard_stop):
        raise RuntimeError(
            "cluster-sharded candidate rows exceeded hard stop: "
            f"{total_candidate_rows} > {config.global_candidate_row_hard_stop}"
        )
    if len(stats) > int(config.global_unique_term_hard_stop):
        raise RuntimeError(
            "cluster-sharded unique terms exceeded hard stop: "
            f"{len(stats)} > {config.global_unique_term_hard_stop}"
        )

    entropy_acc: Counter[str] = Counter()
    for path in candidate_paths:
        df = pd.read_parquet(path, columns=["term", "local_tf"])
        if df.empty:
            continue
        for row in df.itertuples(index=False):
            total_tf = max(1, int(stats[str(row.term)]["total_tf"]))
            p = float(row.local_tf) / float(total_tf)
            if p > 0:
                entropy_acc[str(row.term)] += p * math.log(p)

    rows = []
    for term, row in stats.items():
        out = dict(row)
        out["cluster_entropy"] = float(-entropy_acc.get(term, 0.0))
        out["term_keep_score"] = (
            math.log1p(float(out["total_doc_df"]))
            + 0.25 * math.log1p(float(out["cluster_df"]))
            + 0.15 * math.log1p(float(out["max_cluster_tf"]))
            - 3.0 * float(out["artifact_risk"])
        )
        rows.append(out)
    global_df = pd.DataFrame(rows)
    if global_df.empty:
        global_df = pd.DataFrame(
            columns=[
                "term_id",
                "term",
                "total_tf",
                "total_doc_df",
                "cluster_df",
                "max_cluster_tf",
                "first_year",
                "last_year",
                "artifact_risk",
                "cluster_entropy",
                "term_keep_score",
            ]
        )
    else:
        global_df = global_df[global_df["artifact_risk"] < 1.0].copy()
        if len(global_df) > int(config.global_unique_term_target):
            global_df = global_df.sort_values("term_keep_score", ascending=False).head(
                int(config.global_unique_term_target)
            )
        global_df = global_df.sort_values(["term_keep_score", "term"], ascending=[False, True]).reset_index(drop=True)
        global_df.insert(0, "term_id", np.arange(len(global_df), dtype=np.int64))

    global_df.to_parquet(out_path, index=False)
    allowed_path = global_dir / "allowed_terms.parquet"
    global_df[["term_id", "term"]].to_parquet(allowed_path, index=False)
    write_json_atomic(
        done_path,
        {
            "schema_version": "sciscape_keyword_global_term_stats_done_v1",
            "created_at_utc": utc_now_iso(),
            "status": "complete",
            "fingerprint": fingerprint,
            "candidate_rows": int(total_candidate_rows),
            "unique_terms": int(len(stats)),
            "allowed_terms": int(len(global_df)),
            "candidate_row_warning": bool(total_candidate_rows > int(config.global_candidate_row_warning)),
            "unique_term_warning": bool(len(stats) > int(config.global_unique_term_warning)),
            "path": str(out_path),
        },
    )
    return global_df


def build_cluster_sharded_abbreviation_lookup(
    config: KeywordExtractionConfig,
    candidate_paths: list[Path],
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    """Build a quality-stage abbreviation lookup from candidate-shard evidence."""

    if not config.abbreviation_dictionary_enabled:
        return {"global": {}, "cluster": {}}, "disabled"

    records: list[dict[str, Any]] = []
    for candidate_path in candidate_paths:
        shard_id = _path_shard_id(candidate_path)
        evidence_path = candidate_path.parent / f"abbreviation_shard_{shard_id:04d}.json"
        records.extend(_read_abbreviation_evidence_json(evidence_path))

    global_dir = output_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = global_dir / "abbreviation_evidence.json"
    write_json_atomic(
        evidence_path,
        {
            "schema_version": "sciscape_keyword_abbreviation_evidence_global_v1",
            "created_at_utc": utc_now_iso(),
            "status": "complete",
            "rows": int(len(records)),
            "evidence": records,
        },
    )
    if not records:
        lookup = {"global": {}, "cluster": {}}
        return lookup, _abbreviation_lookup_digest(lookup)

    evidence = pd.DataFrame(records)
    lookup = build_abbreviation_lookup(
        evidence,
        min_support_docs=config.abbreviation_min_support_docs,
        min_cluster_support_docs=config.abbreviation_min_cluster_support_docs,
        min_top_support_ratio=config.abbreviation_min_top_support_ratio,
    )
    digest = _abbreviation_lookup_digest(lookup)
    write_json_atomic(
        global_dir / "abbreviation_lookup.done.json",
        {
            "schema_version": "sciscape_keyword_abbreviation_lookup_done_v1",
            "created_at_utc": utc_now_iso(),
            "status": "complete",
            "fingerprint": digest,
            "evidence_rows": int(len(records)),
            "global_entries": int(len(lookup.get("global", {}))),
            "cluster_entries": int(len(lookup.get("cluster", {}))),
            "evidence_path": str(evidence_path),
        },
    )
    return lookup, digest


def _path_shard_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _abbreviation_lookup_digest(lookup: Mapping[str, Any]) -> str:
    global_items = sorted(
        (str(short), dict(value))
        for short, value in lookup.get("global", {}).items()
        if isinstance(value, Mapping)
    )
    cluster_items = sorted(
        (int(key[0]), str(key[1]), dict(value))
        for key, value in lookup.get("cluster", {}).items()
        if isinstance(key, tuple) and len(key) == 2 and isinstance(value, Mapping)
    )
    return stable_digest({"global": global_items, "cluster": cluster_items})


def score_candidate_shard(
    config: KeywordExtractionConfig,
    candidate_path: Path,
    global_stats: pd.DataFrame,
    n_clusters: int,
    output_dir: Path,
    abbreviation_lookup: Optional[Mapping[str, Any]] = None,
    abbreviation_lookup_digest: str = "",
) -> Path:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shard_id = _path_shard_id(candidate_path)
    out_path = final_dir / f"keyword_shard_{shard_id:04d}.parquet"
    flagged_path = _flagged_final_path(out_path)
    done_path = out_path.with_suffix(".done.json")
    candidate_done_path = candidate_path.with_suffix(".done.json")
    candidate_done = {}
    if candidate_done_path.exists():
        try:
            candidate_done = json.loads(candidate_done_path.read_text(encoding="utf-8"))
        except Exception:
            candidate_done = {"path": str(candidate_path)}
    global_digest = hashlib.sha256()
    if not global_stats.empty:
        digest_cols = ["term", "term_id", "cluster_df", "artifact_risk"]
        if "cluster_entropy" in global_stats.columns:
            digest_cols.append("cluster_entropy")
        for row in global_stats[digest_cols].itertuples(index=False):
            global_digest.update(str(row).encode("utf-8"))
            global_digest.update(b"\0")
    fingerprint = stable_digest(
        {
            "stage": "final_scoring_quality_v10",
            "candidate_done": candidate_done,
            "global_stats_digest": global_digest.hexdigest(),
            "n_clusters": n_clusters,
            "top_n_keywords": config.top_n_keywords,
            "quality_diagnostics_enabled": config.quality_diagnostics_enabled,
            "quality_rerank_enabled": config.quality_rerank_enabled,
            "quality_global_term_threshold": config.quality_global_term_threshold,
            "quality_phrase_preference_weight": config.quality_phrase_preference_weight,
            "quality_artifact_demotion_weight": config.quality_artifact_demotion_weight,
            "quality_single_token_shadow_penalty": config.quality_single_token_shadow_penalty,
            "abbreviation_lookup_digest": abbreviation_lookup_digest,
        }
    )
    if config.scoring_shard_resume and out_path.exists() and flagged_path.exists() and done_matches(done_path, fingerprint):
        return out_path

    candidates = pd.read_parquet(candidate_path)
    if candidates.empty or global_stats.empty:
        empty = pd.DataFrame(columns=_final_columns(include_quality=config.quality_diagnostics_enabled))
        empty.to_parquet(out_path, index=False)
        empty.to_parquet(flagged_path, index=False)
        write_json_atomic(done_path, _final_done_payload(shard_id, out_path, 0, fingerprint, flagged_path=flagged_path))
        return out_path

    stats = global_stats.set_index("term")
    candidates = candidates[candidates["term"].isin(stats.index)].copy()
    if candidates.empty:
        empty = pd.DataFrame(columns=_final_columns(include_quality=config.quality_diagnostics_enabled))
        empty.to_parquet(out_path, index=False)
        empty.to_parquet(flagged_path, index=False)
        write_json_atomic(done_path, _final_done_payload(shard_id, out_path, 0, fingerprint, flagged_path=flagged_path))
        return out_path

    candidates["cluster_total_tf"] = candidates.groupby("cluster_id")["local_tf"].transform("sum").clip(lower=1)
    candidates["term_id"] = candidates["term"].map(stats["term_id"]).astype(int)
    candidates["cluster_df"] = candidates["term"].map(stats["cluster_df"]).astype(float)
    candidates["global_artifact_risk"] = candidates["term"].map(stats["artifact_risk"]).astype(float)
    if "cluster_entropy" in stats.columns:
        raw_entropy = candidates["term"].map(stats["cluster_entropy"]).astype(float)
        entropy_denom = np.log(candidates["cluster_df"].clip(lower=2.0).astype(float))
        candidates["global_cluster_entropy"] = (raw_entropy / entropy_denom).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 1.0)
    else:
        candidates["global_cluster_entropy"] = 0.0
    tf = candidates["local_tf"].astype(float) / candidates["cluster_total_tf"].astype(float)
    idf = np.log((1.0 + float(n_clusters)) / (1.0 + candidates["cluster_df"].astype(float))) + 1.0
    specificity = 1.0 - (candidates["cluster_df"].astype(float) / max(1.0, float(n_clusters)))
    phrase_bonus = candidates["term"].str.contains(" ", regex=False).astype(float) * 0.08
    title_bonus = np.log1p(candidates["title_tf"].astype(float)) * 0.08
    recent_bonus = candidates["channel_flags"].astype(str).str.contains("recent", regex=False).astype(float) * 0.06
    candidates["score"] = (
        (tf * idf)
        + 0.15 * specificity
        + phrase_bonus
        + title_bonus
        + recent_bonus
        - candidates["global_artifact_risk"].astype(float)
    )
    candidates["frequency"] = candidates["local_tf"].astype(int)
    candidates["doc_coverage"] = candidates["local_doc_df"].astype(int)
    candidates["artifact_risk"] = candidates[["artifact_risk", "global_artifact_risk"]].max(axis=1)
    candidates = _apply_quality_refinement(config, candidates, n_clusters, abbreviation_lookup=abbreviation_lookup)
    flagged_top = _select_top_candidates(candidates, config, clean_view=False)
    flagged_final = _finalize_top_candidates(
        flagged_top,
        config=config,
        n_clusters=n_clusters,
        include_quality=config.quality_diagnostics_enabled,
    )
    flagged_final.to_parquet(flagged_path, index=False)

    top = _select_top_candidates(candidates, config, clean_view=True)
    top["rank"] = top.groupby("cluster_id").cumcount() + 1
    top["tier"] = top.apply(lambda row: _keyword_tier(row, config, n_clusters), axis=1)
    top["pub_year_series"] = top.apply(_compact_year_series_placeholder, axis=1)
    top["keyword_engine"] = "cluster_sharded"
    top = top.rename(columns={"channel_flags": "candidate_channel_flags"})
    final = top[_final_columns(include_quality=config.quality_diagnostics_enabled)].reset_index(drop=True)
    final.to_parquet(out_path, index=False)
    write_json_atomic(done_path, _final_done_payload(shard_id, out_path, len(final), fingerprint, flagged_path=flagged_path))
    return out_path


def _apply_quality_refinement(
    config: KeywordExtractionConfig,
    candidates: pd.DataFrame,
    n_clusters: int,
    abbreviation_lookup: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    if candidates.empty or not config.quality_diagnostics_enabled:
        return candidates
    return annotate_keyword_quality(
        candidates,
        rerank=bool(config.quality_rerank_enabled),
        global_n_clusters=n_clusters,
        term_cluster_count_col="cluster_df",
        term_entropy_col="global_cluster_entropy",
        global_term_threshold=config.quality_global_term_threshold,
        global_term_penalty=config.quality_global_term_penalty,
        entropy_penalty=config.quality_cross_cluster_entropy_penalty,
        phrase_preference_weight=config.quality_phrase_preference_weight,
        artifact_demotion_weight=config.quality_artifact_demotion_weight,
        acronym_demotion_weight=config.quality_acronym_demotion_weight,
        formula_demotion_weight=config.quality_formula_demotion_weight,
        single_token_shadow_penalty=config.quality_single_token_shadow_penalty,
        cluster_specific_bonus=config.quality_cluster_specific_bonus,
        min_multiplier=config.quality_min_multiplier,
        acronym_max_length=config.quality_acronym_max_length,
        network_roles_enabled=config.quality_network_roles_enabled,
        abbreviation_lookup=abbreviation_lookup,
        family_representative_enabled=config.quality_family_representative_enabled,
        family_representative_weight=config.quality_family_representative_weight,
        family_representative_max_bonus=config.quality_family_representative_max_bonus,
    )


def _select_top_candidates(
    candidates: pd.DataFrame,
    config: KeywordExtractionConfig,
    *,
    clean_view: bool = True,
) -> pd.DataFrame:
    top_n = int(config.top_n_keywords)
    if candidates.empty:
        return candidates.copy()
    if not config.quality_rerank_enabled or "quality_score" not in candidates.columns:
        ordered = candidates.sort_values(
            ["cluster_id", "score", "candidate_score_hint"],
            ascending=[True, False, False],
        )
        return ordered.groupby("cluster_id", sort=False).head(top_n).copy()

    if "representative_score" not in candidates.columns or "keyword_label_tier" not in candidates.columns:
        ordered = candidates.sort_values(
            ["cluster_id", "quality_score", "score", "candidate_score_hint"],
            ascending=[True, False, False, False],
        )
        return ordered.groupby("cluster_id", sort=False).head(top_n).copy()

    if clean_view:
        candidates = _drop_review_rows_when_possible(candidates)
    representative_budget = min(top_n, max(1, int(math.ceil(top_n * 0.35))))
    label_tier = candidates["keyword_label_tier"].fillna("").astype(str)
    primary_mask = label_tier.str.startswith("primary")
    primary = (
        candidates[primary_mask]
        .sort_values(
            ["cluster_id", "representative_score", "quality_score", "score", "candidate_score_hint"],
            ascending=[True, False, False, False, False],
        )
        .groupby("cluster_id", sort=False)
        .head(representative_budget)
        .copy()
    )
    remaining = candidates.drop(index=primary.index)
    fill = (
        remaining.sort_values(
            ["cluster_id", "quality_score", "score", "candidate_score_hint"],
            ascending=[True, False, False, False],
        )
        .groupby("cluster_id", sort=False)
        .head(top_n)
        .copy()
    )
    primary["_selection_bucket"] = 0
    primary["_selection_score"] = primary["representative_score"].astype(float)
    fill["_selection_bucket"] = 1
    fill["_selection_score"] = fill["quality_score"].astype(float)
    selected = pd.concat([primary, fill], ignore_index=False)
    selected = selected.sort_values(
        ["cluster_id", "_selection_bucket", "_selection_score", "score", "candidate_score_hint"],
        ascending=[True, True, False, False, False],
    )
    selected = selected.groupby("cluster_id", sort=False).head(top_n).copy()
    return selected.drop(columns=["_selection_bucket", "_selection_score"], errors="ignore")


def _flagged_final_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.flagged{out_path.suffix}")


def _finalize_top_candidates(
    top: pd.DataFrame,
    *,
    config: KeywordExtractionConfig,
    n_clusters: int,
    include_quality: bool,
) -> pd.DataFrame:
    if top.empty:
        return pd.DataFrame(columns=_final_columns(include_quality=include_quality))
    top = top.copy()
    top["rank"] = top.groupby("cluster_id").cumcount() + 1
    top["tier"] = top.apply(lambda row: _keyword_tier(row, config, n_clusters), axis=1)
    top["pub_year_series"] = top.apply(_compact_year_series_placeholder, axis=1)
    top["keyword_engine"] = "cluster_sharded"
    top = top.rename(columns={"channel_flags": "candidate_channel_flags"})
    return top[_final_columns(include_quality=include_quality)].reset_index(drop=True)


def _drop_review_rows_when_possible(candidates: pd.DataFrame) -> pd.DataFrame:
    if "clean_view_action" in candidates.columns:
        actions = candidates["clean_view_action"].fillna("").astype(str)
        clean = candidates[~actions.isin({"drop_from_candidates", "hide_from_clean"})].copy()
    else:
        label_tier = candidates["keyword_label_tier"].fillna("").astype(str)
        clean = candidates[~label_tier.str.startswith("review_")].copy()
    if clean.empty:
        return candidates
    missing_clusters = set(candidates["cluster_id"].unique()) - set(clean["cluster_id"].unique())
    if not missing_clusters:
        return clean
    fallback = candidates[candidates["cluster_id"].isin(missing_clusters)]
    return pd.concat([clean, fallback], ignore_index=False)


_BASE_FINAL_COLUMNS = [
    "cluster_id",
    "term",
    "score",
    "frequency",
    "doc_coverage",
    "rank",
    "tier",
    "term_id",
    "cluster_df",
    "candidate_channel_flags",
    "artifact_risk",
    "candidate_score_hint",
    "pub_year_series",
    "keyword_engine",
]


_QUALITY_FINAL_COLUMNS = [
    "raw_term",
    "normalized_term",
    "display_label",
    "quality_score",
    "quality_multiplier",
    "quality_flags",
    "quality_risk_family",
    "quality_flag_basis",
    "quality_flag_confidence",
    "clean_view_action",
    "quality_decision_trace",
    "representative_score",
    "representative_multiplier",
    "representative_rank",
    "representative_role",
    "representative_flags",
    "keyword_label_tier",
    "representative_family_child_count",
    "representative_family_member_count",
    "representative_family_avg_child_coverage",
    "representative_family_multiplier",
    "keyword_scope",
    "keyword_cluster_count",
    "keyword_cluster_ratio",
    "abbreviation_status",
    "abbreviation_target",
    "abbreviation_confidence",
    "abbreviation_source",
    "abbreviation_support_docs",
    "abbreviation_cluster_support_docs",
    "abbreviation_top_support_ratio",
    "abbreviation_ambiguity_type",
    "network_role",
    "network_score",
    "network_flags",
]


def _final_columns(include_quality: bool = True) -> list[str]:
    if not include_quality:
        return list(_BASE_FINAL_COLUMNS)
    return [*_BASE_FINAL_COLUMNS, *_QUALITY_FINAL_COLUMNS]


def _compact_year_series_placeholder(row: pd.Series) -> dict[str, int]:
    first = row.get("first_year")
    last = row.get("last_year")
    freq = int(row.get("frequency", row.get("local_tf", 0)))
    if pd.isna(first) and pd.isna(last):
        return {}
    if pd.notna(last):
        return {str(int(last)): freq}
    return {str(int(first)): freq}


def _keyword_tier(row: pd.Series, config: KeywordExtractionConfig, n_clusters: int) -> str:
    if float(row.get("global_artifact_risk", row.get("artifact_risk", 0.0))) >= 1.0:
        return "drop"
    cluster_ratio = float(row.get("cluster_df", 0.0)) / max(1.0, float(n_clusters))
    if cluster_ratio >= float(config.quality_global_term_threshold):
        return "review_broad"
    rank = int(row.get("rank", 999_999))
    if rank <= max(3, int(math.ceil(config.top_n_keywords * 0.2))):
        return "primary"
    flags = str(row.get("channel_flags", row.get("candidate_channel_flags", "")))
    if "recent" in flags:
        return "emerging"
    if int(row.get("cluster_df", 0)) <= 2:
        return "niche"
    return "supporting"


def _final_done_payload(
    shard_id: int,
    out_path: Path,
    rows: int,
    fingerprint: str = "",
    flagged_path: Optional[Path] = None,
) -> dict[str, Any]:
    return {
        "schema_version": "sciscape_keyword_final_shard_done_v1",
        "created_at_utc": utc_now_iso(),
        "status": "complete",
        "fingerprint": fingerprint,
        "shard_id": int(shard_id),
        "rows": int(rows),
        "path": str(out_path),
        "flagged_path": str(flagged_path) if flagged_path is not None else "",
    }


def run_final_scoring(
    config: KeywordExtractionConfig,
    candidate_paths: list[Path],
    global_stats: pd.DataFrame,
    manifest: Mapping[str, Any],
    output_dir: Path,
    aggregate_candidate_paths: Optional[list[Path]] = None,
) -> pd.DataFrame:
    n_clusters = int(manifest.get("total_clusters", 0))
    abbreviation_lookup, abbreviation_digest = build_cluster_sharded_abbreviation_lookup(
        config,
        candidate_paths,
        output_dir,
    )
    n_jobs = _effective_n_jobs(config.n_jobs)
    backend = config.parallel_backend
    if backend == "sequential" or n_jobs == 1:
        final_paths = [
            score_candidate_shard(
                config,
                path,
                global_stats,
                n_clusters,
                output_dir,
                abbreviation_lookup=abbreviation_lookup,
                abbreviation_lookup_digest=abbreviation_digest,
            )
            for path in candidate_paths
        ]
    else:
        prefer = "threads" if backend == "threading" else "processes"
        final_paths = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(score_candidate_shard)(
                config,
                path,
                global_stats,
                n_clusters,
                output_dir,
                abbreviation_lookup=abbreviation_lookup,
                abbreviation_lookup_digest=abbreviation_digest,
            )
            for path in candidate_paths
        )
    if aggregate_candidate_paths is None:
        aggregate_final_paths = final_paths
    else:
        aggregate_final_paths = [
            _final_shard_path(output_dir, _path_shard_id(path))
            for path in aggregate_candidate_paths
        ]
        missing = [path for path in aggregate_final_paths if not path.exists()]
        if missing:
            sample = ", ".join(str(path) for path in missing[:5])
            suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise RuntimeError(
                "selected cluster-sharded rerun cannot rebuild final keywords "
                f"because final shard output is missing: {sample}{suffix}"
            )

    frames = [pd.read_parquet(path) for path in aggregate_final_paths]
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=_final_columns(include_quality=config.quality_diagnostics_enabled))
    )
    if not result.empty:
        result = result.sort_values(["cluster_id", "rank"]).reset_index(drop=True)
    result_path = output_dir / "keywords.parquet"
    result.to_parquet(result_path, index=False)
    flagged_frames = [
        pd.read_parquet(_flagged_final_path(path))
        for path in aggregate_final_paths
        if _flagged_final_path(path).exists()
    ]
    flagged_result = (
        pd.concat(flagged_frames, ignore_index=True)
        if flagged_frames
        else pd.DataFrame(columns=_final_columns(include_quality=config.quality_diagnostics_enabled))
    )
    if not flagged_result.empty:
        flagged_result = flagged_result.sort_values(["cluster_id", "rank"]).reset_index(drop=True)
    flagged_result_path = output_dir / "keywords_flagged.parquet"
    flagged_result.to_parquet(flagged_result_path, index=False)
    if config.quality_diagnostics_enabled:
        write_keyword_quality_residual_report(
            flagged_result if not flagged_result.empty else result,
            output_dir / "qa",
            top_rank=max(10, int(config.top_n_keywords)),
            max_rows=500,
        )
    return result


def _keyword_rule_source_artifact(role: str, path: Path, root: Path) -> dict[str, str]:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        artifact_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        artifact_path = str(resolved_path)
    return {"role": role, "path": artifact_path}


def run_cluster_sharded_keyword_pipeline(
    config: KeywordExtractionConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> pd.DataFrame:
    """Run the opt-in cluster-sharded keyword engine."""

    output_dir = resolve_cluster_sharded_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback("cluster_shard_manifest", 0, 1)
    manifest = build_cluster_shard_manifest(config)
    active_shard_ids = _requested_shard_ids(config, manifest)
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    if progress_callback:
        progress_callback("cluster_shard_manifest", 1, 1)
        progress_callback("document_sharding", 0, 1)
    doc_shards = materialize_document_shards(config, manifest, output_dir)
    active_shard_count = len(active_shard_ids) if active_shard_ids is not None else len(doc_shards)
    if progress_callback:
        progress_callback("document_sharding", 1, 1)
        progress_callback("candidate_mining", 0, active_shard_count)
    candidate_paths = run_candidate_mining(config, manifest, doc_shards, output_dir, active_shard_ids=active_shard_ids)
    aggregate_candidate_paths = _candidate_paths_for_manifest(
        manifest,
        output_dir,
        require_existing=active_shard_ids is not None,
    )
    if progress_callback:
        progress_callback("candidate_mining", len(candidate_paths), active_shard_count)
        progress_callback("global_term_stats", 0, 1)
    global_stats = build_global_term_stats(config, aggregate_candidate_paths, manifest, output_dir)
    if progress_callback:
        progress_callback("global_term_stats", 1, 1)
        progress_callback("final_scoring", 0, len(candidate_paths))
    result = run_final_scoring(
        config,
        candidate_paths,
        global_stats,
        manifest,
        output_dir,
        aggregate_candidate_paths=aggregate_candidate_paths if active_shard_ids is not None else None,
    )
    if progress_callback:
        progress_callback("final_scoring", len(candidate_paths), len(candidate_paths))
    keyword_rule_artifact: dict[str, Any] | None = None
    if config.keyword_rule_artifact_enabled:
        from .rule_artifact import write_keyword_cleaning_rule_artifacts

        if progress_callback:
            progress_callback("keyword_rule_artifact", 0, 1)
        rule_root = Path(config.keyword_rule_result_root) if config.keyword_rule_result_root is not None else output_dir
        keyword_rule_artifact = write_keyword_cleaning_rule_artifacts(
            rule_root,
            keywords=result,
            rule_set_id=config.keyword_rule_set_id,
            output_dir=rule_root / "rules" / config.keyword_rule_set_id,
            source_artifacts=[
                _keyword_rule_source_artifact("keywords", output_dir / "keywords.parquet", rule_root),
                _keyword_rule_source_artifact("keywords_flagged", output_dir / "keywords_flagged.parquet", rule_root),
                _keyword_rule_source_artifact("cluster_sharded_manifest", manifest_path, rule_root),
            ],
        )
        if progress_callback:
            progress_callback("keyword_rule_artifact", 1, 1)
    write_json_atomic(
        output_dir / "run_summary.json",
        {
            "schema_version": "sciscape_keyword_cluster_sharded_run_summary_v1",
            "created_at_utc": utc_now_iso(),
            "status": "complete",
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "candidate_shards": len(candidate_paths),
            "active_shard_ids": sorted(active_shard_ids) if active_shard_ids is not None else None,
            "aggregate_candidate_shards": len(aggregate_candidate_paths),
            "global_terms": int(len(global_stats)),
            "final_rows": int(len(result)),
            "keyword_rule_manifest_path": (
                str(keyword_rule_artifact["manifest_path"]) if keyword_rule_artifact is not None else None
            ),
        },
    )
    return result
