#!/usr/bin/env python3
"""Roll up display-clean Nano keywords to all Science Atlas hierarchy levels.

The script assumes the Nano IDs in ``dashboard/tables/nano_terms_topk.parquet``
match ``dashboard/tables/cluster_lineage.parquet``.  It rebuilds
``core/atlas_cluster_terms.parquet`` from the clean Nano table and the current
lineage, replacing stale row sources such as old dashboard terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEVEL_ORDER = {"domain": 0, "macro": 1, "meso": 2, "micro": 3, "nano": 4}
UPPER_LEVELS = (
    ("micro", "micro_id", "sciscape_clean_v10_rollup_micro"),
    ("meso", "meso_id", "sciscape_clean_v10_rollup_meso"),
    ("macro", "macro_id", "sciscape_clean_v10_rollup_macro"),
    ("domain", "domain_id", "sciscape_clean_v10_rollup_domain"),
)
NANO_COLUMNS = [
    "cluster_uid",
    "level",
    "cluster_id",
    "term",
    "term_count",
    "term_doc_count",
    "representative_doc_count",
    "score",
    "rank",
]
LINEAGE_COLUMNS = [
    "hierarchy_version",
    "nano_id",
    "nano_docs",
    "micro_id",
    "meso_id",
    "macro_id",
    "domain_id",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_nano_terms(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=NANO_COLUMNS)
    frame = frame.copy()
    frame["cluster_uid"] = frame["cluster_uid"].astype(str)
    frame["level"] = frame["level"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype("int64")
    frame["term"] = frame["term"].fillna("").astype(str).str.strip()
    frame["term_count"] = pd.to_numeric(frame["term_count"], errors="coerce").fillna(0).astype("int64")
    frame["term_doc_count"] = pd.to_numeric(frame["term_doc_count"], errors="coerce").fillna(0).astype("int64")
    frame["representative_doc_count"] = (
        pd.to_numeric(frame["representative_doc_count"], errors="coerce").fillna(0).astype("int64")
    )
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).astype("float64")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce").fillna(9999).astype("int64")
    frame = frame[frame["term"].ne("")].copy()
    return frame


def read_lineage(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=LINEAGE_COLUMNS)
    frame = frame.copy()
    for column in ("nano_id", "nano_docs", "micro_id", "meso_id", "macro_id", "domain_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["hierarchy_version"] = frame["hierarchy_version"].astype(str)
    return frame


def validate_nano_lineage(nano: pd.DataFrame, lineage: pd.DataFrame) -> dict[str, Any]:
    if not nano["level"].eq("nano").all():
        levels = sorted(nano["level"].dropna().unique().tolist())
        raise ValueError(f"nano term table contains non-nano levels: {levels}")

    nano_ids = set(int(value) for value in nano["cluster_id"].unique())
    lineage_ids = set(int(value) for value in lineage["nano_id"].unique())
    missing_lineage = nano_ids - lineage_ids
    missing_terms = lineage_ids - nano_ids
    if missing_lineage or missing_terms:
        raise ValueError(
            "Nano ID mismatch: "
            f"terms_without_lineage={len(missing_lineage)}, "
            f"lineage_without_terms={len(missing_terms)}"
        )
    versions = lineage["hierarchy_version"].value_counts().to_dict()
    if len(versions) != 1:
        raise ValueError(f"lineage contains multiple hierarchy versions: {versions}")
    return {
        "nano_cluster_count": len(nano_ids),
        "lineage_cluster_count": len(lineage_ids),
        "hierarchy_version": next(iter(versions)),
        "hierarchy_version_rows": versions,
    }


def parent_stats(lineage: pd.DataFrame, parent_col: str) -> pd.DataFrame:
    stats = (
        lineage.groupby(parent_col, as_index=False)
        .agg(parent_doc_count=("nano_docs", "sum"), parent_nano_count=("nano_id", "nunique"))
        .rename(columns={parent_col: "parent_id"})
    )
    stats["parent_doc_count"] = stats["parent_doc_count"].astype("int64")
    stats["parent_nano_count"] = stats["parent_nano_count"].astype("int64")
    return stats


def rollup_level(source: pd.DataFrame, lineage: pd.DataFrame, *, level: str, parent_col: str, top_n: int) -> pd.DataFrame:
    parent_count = int(lineage[parent_col].nunique())
    stats = parent_stats(lineage, parent_col)

    group = source[[parent_col, "cluster_id", "term", "term_count", "term_doc_count", "score", "rank"]].copy()
    doc_weight = group["term_doc_count"].clip(lower=1).astype("float64")
    rank_cap = max(int(top_n), 1)
    rank_signal = ((rank_cap + 1 - group["rank"].clip(lower=1, upper=rank_cap)) / rank_cap).astype("float64")
    group["_score_doc_weight"] = group["score"].astype("float64") * doc_weight
    group["_rank_doc_weight"] = rank_signal * doc_weight

    rolled = (
        group.groupby([parent_col, "term"], as_index=False)
        .agg(
            term_count=("term_count", "sum"),
            term_doc_count=("term_doc_count", "sum"),
            contributing_nano_count=("cluster_id", "nunique"),
            score_doc_weight=("_score_doc_weight", "sum"),
            rank_doc_weight=("_rank_doc_weight", "sum"),
        )
        .rename(columns={parent_col: "parent_id"})
    )
    if rolled.empty:
        return pd.DataFrame(columns=["cluster_uid", "term", "rank", "score", "evidence_channel"])

    term_parent_df = rolled.groupby("term")["parent_id"].nunique().rename("term_parent_df")
    rolled = rolled.merge(term_parent_df, on="term", how="left").merge(stats, on="parent_id", how="left")

    denominator = rolled["term_doc_count"].clip(lower=1).astype("float64")
    mean_score = rolled["score_doc_weight"] / denominator
    mean_rank_signal = rolled["rank_doc_weight"] / denominator
    coverage = (rolled["term_doc_count"] / rolled["parent_doc_count"].clip(lower=1)).clip(lower=0.0, upper=1.0)
    nano_support = (
        rolled["contributing_nano_count"] / rolled["parent_nano_count"].clip(lower=1)
    ).clip(lower=0.0, upper=1.0)
    specificity = np.log1p((parent_count + 1.0) / (rolled["term_parent_df"].astype("float64") + 1.0))

    rolled["_raw_score"] = (
        mean_score.clip(lower=0.0)
        * (0.40 + 0.60 * mean_rank_signal.clip(lower=0.0, upper=1.0))
        * (0.60 + np.sqrt(coverage))
        * (0.70 + np.sqrt(nano_support))
        * specificity
        * np.log1p(rolled["term_doc_count"].astype("float64").clip(lower=0.0))
    )
    rolled = rolled.sort_values(
        ["parent_id", "_raw_score", "term_doc_count", "contributing_nano_count", "term"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    top = rolled.groupby("parent_id", sort=False).head(top_n).copy()
    top["rank"] = top.groupby("parent_id", sort=False).cumcount().add(1).astype("int64")
    max_score = top.groupby("parent_id")["_raw_score"].transform("max").replace(0.0, 1.0)
    top["score"] = (top["_raw_score"] / max_score).astype("float64")
    top["cluster_uid"] = level + ":" + top["parent_id"].astype("int64").astype(str)
    top["evidence_channel"] = f"sciscape_clean_v10_rollup_{level}"
    return top[["cluster_uid", "term", "rank", "score", "evidence_channel"]]


def build_core_terms(nano: pd.DataFrame, lineage: pd.DataFrame, top_n_upper: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    joined = nano.merge(
        lineage[["nano_id", "micro_id", "meso_id", "macro_id", "domain_id"]],
        left_on="cluster_id",
        right_on="nano_id",
        how="inner",
        validate="many_to_one",
    )

    nano_core = nano[["cluster_uid", "term", "rank", "score"]].copy()
    nano_core["evidence_channel"] = "sciscape_clean_v10"

    parts = []
    level_summaries: dict[str, Any] = {}
    for level, parent_col, _channel in UPPER_LEVELS:
        rolled = rollup_level(joined, lineage, level=level, parent_col=parent_col, top_n=top_n_upper)
        parts.append(rolled)
        level_summaries[level] = {
            "rows": int(len(rolled)),
            "clusters": int(rolled["cluster_uid"].nunique()) if not rolled.empty else 0,
            "min_terms": int(rolled.groupby("cluster_uid").size().min()) if not rolled.empty else 0,
            "max_terms": int(rolled.groupby("cluster_uid").size().max()) if not rolled.empty else 0,
        }

    out = pd.concat([*parts, nano_core], ignore_index=True)
    prefix = out["cluster_uid"].astype(str).str.split(":", n=1).str[0]
    ident = out["cluster_uid"].astype(str).str.split(":", n=1).str[1].astype("int64")
    out["_level_order"] = prefix.map(LEVEL_ORDER).fillna(99).astype("int64")
    out["_id"] = ident
    out = out.sort_values(["_level_order", "_id", "rank", "term"], kind="mergesort").drop(columns=["_level_order", "_id"])
    out["rank"] = out["rank"].astype("int64")
    out["score"] = out["score"].astype("float64")

    level_summaries["nano"] = {
        "rows": int(len(nano_core)),
        "clusters": int(nano_core["cluster_uid"].nunique()),
        "min_terms": int(nano_core.groupby("cluster_uid").size().min()),
        "max_terms": int(nano_core.groupby("cluster_uid").size().max()),
    }
    return out, level_summaries


def prefix_counts(frame: pd.DataFrame) -> dict[str, int]:
    prefix = frame["cluster_uid"].astype(str).str.split(":", n=1).str[0]
    return {str(k): int(v) for k, v in prefix.value_counts().sort_index().items()}


def channel_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["evidence_channel"].value_counts().sort_index().items()}


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False, row_group_size=100_000)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_checksum_file(path: Path, relpath: str, digest: str) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    replaced = False
    out_lines = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1] == relpath:
            out_lines.append(f"{digest}  {relpath}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append(f"{digest}  {relpath}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n")
    os.replace(tmp, path)
    return True


def sample_terms(frame: pd.DataFrame, cluster_uid: str, limit: int = 10) -> list[str]:
    rows = frame.loc[frame["cluster_uid"].astype(str).eq(cluster_uid)].sort_values("rank").head(limit)
    return [str(value) for value in rows["term"].tolist()]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Sciscape Clean Hierarchy Rollup",
        "",
        f"- Created UTC: `{summary['created_at_utc']}`",
        f"- Datapack: `{summary['datapack_dir']}`",
        f"- Hierarchy version: `{summary['lineage_validation']['hierarchy_version']}`",
        f"- Apply: `{summary['applied']}`",
        f"- Top-N upper: `{summary['top_n_upper']}`",
        "",
        "## Row Counts",
        "",
    ]
    for key, value in summary["new_prefix_rows"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Evidence Channels", ""])
    for key, value in summary["new_evidence_channels"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Samples", ""])
    for key, values in summary["sample_terms"].items():
        lines.append(f"- `{key}`: {', '.join(values)}")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    datapack_dir = args.datapack_dir.resolve()
    nano_path = (args.nano_terms or datapack_dir / "dashboard" / "tables" / "nano_terms_topk.parquet").resolve()
    lineage_path = datapack_dir / "dashboard" / "tables" / "cluster_lineage.parquet"
    core_terms_path = datapack_dir / "core" / "atlas_cluster_terms.parquet"
    qa_dir = datapack_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    stamp = utc_stamp()
    nano = read_nano_terms(nano_path)
    lineage = read_lineage(lineage_path)
    validation = validate_nano_lineage(nano, lineage)

    old_terms = pd.read_parquet(core_terms_path)
    new_terms, level_summaries = build_core_terms(nano, lineage, int(args.top_n_upper))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datapack_dir": str(datapack_dir),
        "nano_terms_path": str(nano_path),
        "lineage_path": str(lineage_path),
        "core_terms_path": str(core_terms_path),
        "top_n_upper": int(args.top_n_upper),
        "applied": bool(args.apply),
        "lineage_validation": validation,
        "old_rows": int(len(old_terms)),
        "old_prefix_rows": prefix_counts(old_terms),
        "old_evidence_channels": channel_counts(old_terms),
        "new_rows": int(len(new_terms)),
        "new_prefix_rows": prefix_counts(new_terms),
        "new_evidence_channels": channel_counts(new_terms),
        "level_summaries": level_summaries,
        "sample_terms": {
            "domain:0": sample_terms(new_terms, "domain:0"),
            "macro:0": sample_terms(new_terms, "macro:0"),
            "meso:0": sample_terms(new_terms, "meso:0"),
            "micro:0": sample_terms(new_terms, "micro:0"),
            "nano:0": sample_terms(new_terms, "nano:0"),
            "nano:79650": sample_terms(new_terms, "nano:79650"),
        },
    }

    preview_path = qa_dir / f"sciscape_clean_rollup_core_terms_preview_{stamp}.parquet"
    if args.write_preview:
        write_parquet_atomic(preview_path, new_terms)
        summary["preview_path"] = str(preview_path)

    if args.apply:
        backup_dir = qa_dir / f"sciscape_clean_rollup_backup_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(core_terms_path, backup_dir / "atlas_cluster_terms.parquet")
        for rel in ("core/CHECKSUMS.sha256", "CHECKSUMS.sha256"):
            source = datapack_dir / rel
            if source.exists():
                backup_target = backup_dir / rel.replace("/", "__")
                shutil.copy2(source, backup_target)
        write_parquet_atomic(core_terms_path, new_terms)
        digest = sha256_file(core_terms_path)
        update_checksum_file(datapack_dir / "core" / "CHECKSUMS.sha256", "atlas_cluster_terms.parquet", digest)
        update_checksum_file(datapack_dir / "CHECKSUMS.sha256", "core/atlas_cluster_terms.parquet", digest)
        summary["backup_dir"] = str(backup_dir)
        summary["new_core_terms_sha256"] = digest

    summary_json = qa_dir / f"sciscape_clean_rollup_summary_{stamp}.json"
    summary_md = qa_dir / f"sciscape_clean_rollup_summary_{stamp}.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_summary(summary_md, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapack-dir", type=Path, required=True)
    parser.add_argument("--nano-terms", type=Path, default=None)
    parser.add_argument("--top-n-upper", type=int, default=20)
    parser.add_argument("--write-preview", action="store_true", help="Write candidate core terms into qa before apply")
    parser.add_argument("--apply", action="store_true", help="Replace core/atlas_cluster_terms.parquet after backup")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
