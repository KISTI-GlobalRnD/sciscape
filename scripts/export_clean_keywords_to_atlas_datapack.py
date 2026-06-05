#!/usr/bin/env python3
"""Export display-clean SciScape keywords into a Science Atlas datapack.

This script is intentionally conservative: it drops only high-confidence
rendering/metadata artifacts, and otherwise demotes broad or genre-like terms
so the Atlas display can prefer more specific cluster terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_COLUMNS = [
    "cluster_id",
    "term",
    "normalized_term",
    "rank",
    "score",
    "frequency",
    "doc_coverage",
    "cluster_df",
    "quality_score",
    "quality_flags",
    "quality_risk_family",
    "clean_view_action",
    "keyword_label_tier",
    "artifact_risk",
    "representative_score",
    "representative_role",
    "keyword_scope",
    "keyword_cluster_count",
    "keyword_cluster_ratio",
    "abbreviation_status",
    "abbreviation_target",
    "network_role",
    "network_score",
]

MATH_RENDER_RE = re.compile(
    r"\b(?:mathrm|mathbf|mathit|mathcal|overline|underline|textit|textrm)\b"
)
HTML_OR_METADATA_RE = re.compile(
    r"\b(?:htmlview|lt div|gt lt|class htmlview|get access|articles author|works author|author gsw|urology vol|journal article|downloaded from)\b"
)

DOCUMENT_GENRE_TERMS = {
    "book review",
    "book reviews",
    "case report",
    "case reports",
    "case study",
    "case studies",
    "literature review",
    "meta analysis",
    "narrative review",
    "review essay",
    "scoping review",
    "systematic review",
    "systematic reviews",
}

BROAD_DISPLAY_TERMS = {
    "high performance",
    "large scale",
    "long term",
    "low cost",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def term_tokens(value: object) -> list[str]:
    return [token for token in str(value or "").lower().split() if token]


def has_adjacent_repeat(tokens: list[str]) -> bool:
    return any(tokens[index] == tokens[index + 1] for index in range(len(tokens) - 1))


def is_contiguous_subsequence(shorter: list[str], longer: list[str]) -> bool:
    if not shorter or len(shorter) >= len(longer):
        return False
    width = len(shorter)
    return any(longer[index : index + width] == shorter for index in range(len(longer) - width + 1))


def scope_multiplier(cluster_ratio: float) -> float:
    if cluster_ratio >= 0.20:
        return 0.18
    if cluster_ratio >= 0.15:
        return 0.25
    if cluster_ratio >= 0.10:
        return 0.35
    if cluster_ratio >= 0.05:
        return 0.55
    if cluster_ratio >= 0.02:
        return 0.75
    if cluster_ratio >= 0.01:
        return 0.90
    return 1.0


def cleaning_decision(row: pd.Series) -> tuple[bool, str, float]:
    term = str(row.term or "").strip()
    lowered = term.lower()
    tokens = term_tokens(term)
    reasons: list[str] = []

    if not term:
        return True, "blank", 0.0
    if float(row.artifact_risk or 0.0) > 0:
        return True, "artifact_risk", 0.0
    if MATH_RENDER_RE.search(lowered):
        return True, "math_render_fragment", 0.0
    if HTML_OR_METADATA_RE.search(lowered):
        return True, "html_or_metadata_fragment", 0.0

    multiplier = scope_multiplier(float(row.keyword_cluster_ratio or 0.0))
    if multiplier < 1.0:
        reasons.append("shared_or_broad_scope")

    if lowered in DOCUMENT_GENRE_TERMS:
        multiplier *= 0.25
        reasons.append("document_genre")
    elif lowered in BROAD_DISPLAY_TERMS:
        multiplier *= 0.40
        reasons.append("broad_display_term")

    if has_adjacent_repeat(tokens):
        multiplier *= 0.35 if len(tokens) <= 3 else 0.60
        reasons.append("adjacent_repeated_token")

    return False, "|".join(reasons) if reasons else "clean", multiplier


def select_display_terms(group: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidates = group.loc[~group["clean_drop"]].copy()
    if candidates.empty:
        candidates = group.copy()

    candidates["token_count"] = candidates["term"].map(lambda value: len(term_tokens(value)))
    candidates["sort_score"] = candidates["display_score"] * (
        1.0 + candidates["token_count"].clip(lower=1, upper=4).sub(1) * 0.04
    )
    candidates = candidates.sort_values(
        ["sort_score", "display_score", "doc_coverage", "rank", "term"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )

    selected_indices: list[int] = []
    selected_tokens: list[list[str]] = []
    selected_norms: set[str] = set()

    for row in candidates.itertuples():
        tokens = term_tokens(row.term)
        norm = " ".join(tokens)
        if not norm or norm in selected_norms:
            continue
        if any(is_contiguous_subsequence(tokens, kept) for kept in selected_tokens):
            continue
        selected_indices.append(row.Index)
        selected_tokens.append(tokens)
        selected_norms.add(norm)
        if len(selected_indices) >= top_n:
            break

    if len(selected_indices) < top_n:
        for row in candidates.itertuples():
            if row.Index in selected_indices:
                continue
            tokens = term_tokens(row.term)
            norm = " ".join(tokens)
            if not norm or norm in selected_norms:
                continue
            selected_indices.append(row.Index)
            selected_norms.add(norm)
            if len(selected_indices) >= top_n:
                break

    selected = candidates.loc[selected_indices].copy()
    selected["clean_rank"] = range(1, len(selected) + 1)
    return selected


def select_display_indices(sorted_group: pd.DataFrame, top_n: int) -> list[int]:
    selected_indices: list[int] = []
    selected_tokens: list[list[str]] = []
    selected_norms: set[str] = set()

    for row in sorted_group.itertuples():
        tokens = term_tokens(row.term)
        norm = " ".join(tokens)
        if not norm or norm in selected_norms:
            continue
        if any(is_contiguous_subsequence(tokens, kept) for kept in selected_tokens):
            continue
        selected_indices.append(row.Index)
        selected_tokens.append(tokens)
        selected_norms.add(norm)
        if len(selected_indices) >= top_n:
            break

    if len(selected_indices) < top_n:
        for row in sorted_group.itertuples():
            if row.Index in selected_indices:
                continue
            tokens = term_tokens(row.term)
            norm = " ".join(tokens)
            if not norm or norm in selected_norms:
                continue
            selected_indices.append(row.Index)
            selected_norms.add(norm)
            if len(selected_indices) >= top_n:
                break

    return selected_indices


def pattern_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    terms = frame["term"].fillna("").astype(str).str.lower()
    patterns = {
        "math_render_fragment": r"\b(?:mathrm|mathbf|mathit|mathcal|overline|underline|textit|textrm)\b",
        "html_or_metadata_fragment": r"\b(?:htmlview|lt div|gt lt|class htmlview|get access|articles author|works author|author gsw|urology vol|journal article|downloaded from)\b",
        "document_genre": r"^(?:book reviews?|case reports?|case stud(?:y|ies)|literature review|meta analysis|narrative review|review essay|scoping review|systematic reviews?)$",
        "adjacent_repeated_token": r"\b(\w+)\s+\1\b",
    }
    out: dict[str, dict[str, int]] = {}
    for name, pattern in patterns.items():
        mask = terms.str.contains(pattern, regex=True, na=False)
        out[name] = {
            "rows": int(mask.sum()),
            "top10": int((mask & (frame["rank"] <= 10)).sum()),
            "top1": int((mask & (frame["rank"] == 1)).sum()),
            "clusters": int(frame.loc[mask, "cluster_id"].nunique()),
        }
    return out


def top10_subphrase_summary(frame: pd.DataFrame) -> dict[str, int]:
    total_pairs = 0
    clusters = 0
    for _, group in frame.loc[frame["rank"] <= 10, ["cluster_id", "rank", "term"]].groupby("cluster_id"):
        tokens = [(int(row.rank), term_tokens(row.term)) for row in group.itertuples(index=False)]
        cluster_pairs = 0
        for left_rank, left_tokens in tokens:
            for right_rank, right_tokens in tokens:
                if left_rank == right_rank:
                    continue
                if is_contiguous_subsequence(left_tokens, right_tokens):
                    cluster_pairs += 1
        if cluster_pairs:
            clusters += 1
            total_pairs += cluster_pairs
    return {"clusters": clusters, "pairs": total_pairs}


def read_keywords(path: Path) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [column for column in DEFAULT_COLUMNS if column in available]
    frame = pq.read_table(path, columns=columns).to_pandas()
    for column in DEFAULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def build_clean_export(
    keyword_path: Path,
    datapack_dir: Path,
    output_dir: Path,
    top_n: int,
    family_suppression: bool,
    subphrase_summary: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = read_keywords(keyword_path)
    frame["term"] = frame["term"].fillna("").astype(str).str.strip()
    frame["cluster_id"] = frame["cluster_id"].astype("int64")
    frame["rank"] = frame["rank"].astype("int64")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    frame["representative_score"] = pd.to_numeric(frame["representative_score"], errors="coerce").fillna(frame["score"])
    frame["doc_coverage"] = pd.to_numeric(frame["doc_coverage"], errors="coerce").fillna(0).astype("int64")
    frame["frequency"] = pd.to_numeric(frame["frequency"], errors="coerce").fillna(frame["doc_coverage"]).astype("int64")
    frame["cluster_df"] = pd.to_numeric(frame["cluster_df"], errors="coerce").fillna(0)
    frame["keyword_cluster_ratio"] = pd.to_numeric(frame["keyword_cluster_ratio"], errors="coerce").fillna(0.0)
    frame["artifact_risk"] = pd.to_numeric(frame["artifact_risk"], errors="coerce").fillna(0.0)

    term_lower = frame["term"].str.lower()
    blank_mask = term_lower.eq("")
    artifact_mask = frame["artifact_risk"].gt(0)
    math_mask = term_lower.str.contains(MATH_RENDER_RE, regex=True, na=False)
    metadata_mask = term_lower.str.contains(HTML_OR_METADATA_RE, regex=True, na=False)
    drop_mask = blank_mask | artifact_mask | math_mask | metadata_mask

    drop_reason = pd.Series("clean", index=frame.index, dtype="object")
    drop_reason.loc[blank_mask] = "blank"
    drop_reason.loc[artifact_mask] = "artifact_risk"
    drop_reason.loc[math_mask] = "math_render_fragment"
    drop_reason.loc[metadata_mask] = "html_or_metadata_fragment"

    multiplier = pd.Series(1.0, index=frame.index, dtype="float64")
    ratio = frame["keyword_cluster_ratio"]
    multiplier.loc[ratio.ge(0.20)] = 0.18
    multiplier.loc[ratio.ge(0.15) & ratio.lt(0.20)] = 0.25
    multiplier.loc[ratio.ge(0.10) & ratio.lt(0.15)] = 0.35
    multiplier.loc[ratio.ge(0.05) & ratio.lt(0.10)] = 0.55
    multiplier.loc[ratio.ge(0.02) & ratio.lt(0.05)] = 0.75
    multiplier.loc[ratio.ge(0.01) & ratio.lt(0.02)] = 0.90

    demotion_reason = pd.Series("clean", index=frame.index, dtype="object")

    def add_reason(mask: pd.Series, reason: str) -> None:
        current = demotion_reason.loc[mask]
        prefix = current.where(current.ne("clean"), "")
        demotion_reason.loc[mask] = prefix.where(prefix.eq(""), prefix + "|") + reason

    broad_scope_mask = multiplier.lt(1.0)
    add_reason(broad_scope_mask, "shared_or_broad_scope")

    genre_mask = term_lower.isin(DOCUMENT_GENRE_TERMS)
    multiplier.loc[genre_mask] *= 0.25
    add_reason(genre_mask, "document_genre")

    broad_term_mask = term_lower.isin(BROAD_DISPLAY_TERMS)
    multiplier.loc[broad_term_mask] *= 0.40
    add_reason(broad_term_mask, "broad_display_term")

    repeat_mask = term_lower.str.contains(r"\b(\w+)\s+\1\b", regex=True, na=False)
    short_repeat_mask = repeat_mask & term_lower.map(lambda value: len(term_tokens(value)) <= 3)
    multiplier.loc[short_repeat_mask] *= 0.35
    multiplier.loc[repeat_mask & ~short_repeat_mask] *= 0.60
    add_reason(repeat_mask, "adjacent_repeated_token")

    frame["clean_drop"] = drop_mask
    frame["clean_reason"] = demotion_reason.where(~drop_mask, drop_reason)
    frame["clean_multiplier"] = multiplier.where(~drop_mask, 0.0)
    frame["display_score"] = frame["representative_score"].fillna(frame["score"]) * frame["clean_multiplier"]
    frame["token_count"] = term_lower.str.count(r"\s+").add(1).where(term_lower.ne(""), 0)
    frame["sort_score"] = frame["display_score"] * (
        1.0 + frame["token_count"].clip(lower=1, upper=4).sub(1) * 0.04
    )

    candidates = frame.loc[~frame["clean_drop"]].sort_values(
        ["cluster_id", "sort_score", "display_score", "doc_coverage", "rank", "term"],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    )
    if family_suppression:
        selected_indices: list[int] = []
        for _, group in candidates.groupby("cluster_id", sort=False):
            selected_indices.extend(select_display_indices(group, top_n=top_n))
        clean = frame.loc[selected_indices].copy()
    else:
        clean = candidates.groupby("cluster_id", sort=False).head(top_n).copy()
    clean["rank"] = clean.groupby("cluster_id", sort=False).cumcount().add(1).astype("int64")
    clean["cluster_uid"] = "nano:" + clean["cluster_id"].astype(str)
    clean["evidence_channel"] = "sciscape_clean_v10"

    core_nano = clean[["cluster_uid", "term", "rank", "display_score", "evidence_channel"]].rename(
        columns={"display_score": "score"}
    )
    core_nano["score"] = core_nano["score"].astype("float64")

    node_path = datapack_dir / "core" / "atlas_cluster_nodes.parquet"
    node_columns = ["cluster_uid", "level", "cluster_id", "doc_count"]
    nodes = pq.read_table(node_path, columns=node_columns).to_pandas()
    nano_docs = (
        nodes.loc[nodes["level"].eq("nano"), ["cluster_id", "doc_count"]]
        .assign(cluster_id=lambda x: x["cluster_id"].astype("int64"))
        .set_index("cluster_id")["doc_count"]
        .to_dict()
    )
    dashboard_nano = pd.DataFrame(
        {
            "cluster_uid": clean["cluster_uid"],
            "level": "nano",
            "cluster_id": clean["cluster_id"].astype("int64"),
            "ngram_n": clean["term"].map(lambda value: min(max(len(term_tokens(value)), 1), 3)).astype("int32"),
            "term": clean["term"],
            "term_count": clean["frequency"].astype("int64"),
            "term_doc_count": clean["doc_coverage"].astype("int64"),
            "representative_doc_count": clean["cluster_id"].map(nano_docs).fillna(clean["doc_coverage"]).astype("int64"),
            "score": clean["display_score"].astype("float64"),
            "rank": clean["rank"].astype("int64"),
        }
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "keyword_path": str(keyword_path),
        "datapack_dir": str(datapack_dir),
        "top_n": top_n,
        "input_rows": int(len(frame)),
        "input_clusters": int(frame["cluster_id"].nunique()),
        "clean_rows": int(len(clean)),
        "clean_clusters": int(clean["cluster_id"].nunique()),
        "clusters_with_non_top_n": int((clean.groupby("cluster_id").size() != top_n).sum()),
        "dropped_rows": int(frame["clean_drop"].sum()),
        "drop_reasons": frame.loc[frame["clean_drop"], "clean_reason"].value_counts().to_dict(),
        "demotion_reasons": frame.loc[~frame["clean_drop"], "clean_reason"].value_counts().head(30).to_dict(),
        "before_pattern_counts": pattern_counts(frame.rename(columns={"clean_rank": "rank"})),
        "after_pattern_counts": pattern_counts(clean),
        "family_suppression": family_suppression,
        "before_top10_subphrase": top10_subphrase_summary(frame) if subphrase_summary else None,
        "after_top10_subphrase": top10_subphrase_summary(clean) if subphrase_summary else None,
        "top1_terms_after": clean.loc[clean["rank"].eq(1), "term"].value_counts().head(30).to_dict(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    core_nano.to_parquet(output_dir / "atlas_cluster_terms_nano_sciscape_clean.parquet", index=False)
    dashboard_nano.to_parquet(output_dir / "nano_terms_topk_sciscape_clean.parquet", index=False)
    clean.to_parquet(output_dir / "keyword_clean_display_terms.parquet", index=False)
    (output_dir / "keyword_clean_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    write_summary_markdown(summary, output_dir / "keyword_clean_export_summary.md")
    return core_nano, dashboard_nano, summary


def write_summary_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Keyword Clean Export Summary",
        "",
        f"- Created UTC: `{summary['created_at_utc']}`",
        f"- Input rows: `{summary['input_rows']}`",
        f"- Input clusters: `{summary['input_clusters']}`",
        f"- Clean rows: `{summary['clean_rows']}`",
        f"- Clean clusters: `{summary['clean_clusters']}`",
        f"- Clusters with non-top-n output: `{summary['clusters_with_non_top_n']}`",
        f"- Dropped rows: `{summary['dropped_rows']}`",
        "",
        "## Drop Reasons",
        "",
    ]
    for key, value in dict(summary.get("drop_reasons", {})).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Before Pattern Counts", ""])
    append_pattern_table(lines, dict(summary["before_pattern_counts"]))
    lines.extend(["", "## After Pattern Counts", ""])
    append_pattern_table(lines, dict(summary["after_pattern_counts"]))
    lines.extend(
        [
            "",
            "## Subphrase Summary",
            "",
            f"- Family suppression: `{summary['family_suppression']}`",
            f"- Before top10 clusters with subphrase pairs: `{(summary['before_top10_subphrase'] or {}).get('clusters', 'not_computed')}`",
            f"- Before top10 subphrase pairs: `{(summary['before_top10_subphrase'] or {}).get('pairs', 'not_computed')}`",
            f"- After top10 clusters with subphrase pairs: `{(summary['after_top10_subphrase'] or {}).get('clusters', 'not_computed')}`",
            f"- After top10 subphrase pairs: `{(summary['after_top10_subphrase'] or {}).get('pairs', 'not_computed')}`",
            "",
            "## Top1 Terms After",
            "",
        ]
    )
    for key, value in dict(summary.get("top1_terms_after", {})).items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n")


def append_pattern_table(lines: list[str], counts: dict[str, dict[str, int]]) -> None:
    lines.append("| pattern | rows | top10 | top1 | clusters |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, values in counts.items():
        lines.append(
            f"| `{name}` | {values['rows']} | {values['top10']} | {values['top1']} | {values['clusters']} |"
        )


def merge_core_terms(datapack_dir: Path, core_nano: pd.DataFrame) -> pd.DataFrame:
    original = pq.read_table(datapack_dir / "core" / "atlas_cluster_terms.parquet").to_pandas()
    non_nano = original.loc[~original["cluster_uid"].astype(str).str.startswith("nano:")].copy()
    return pd.concat([non_nano, core_nano], ignore_index=True)


def backup_files(paths: Iterable[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)


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
    out_lines: list[str] = []
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
    tmp.replace(path)
    return True


def backup_checksum_files(datapack_dir: Path, backup_dir: Path) -> None:
    for relpath in (
        "CHECKSUMS.sha256",
        "core/CHECKSUMS.sha256",
        "dashboard/CHECKSUMS.sha256",
        "dashboard/tables/CHECKSUMS.sha256",
    ):
        source = datapack_dir / relpath
        if source.exists():
            shutil.copy2(source, backup_dir / relpath.replace("/", "__"))


def update_datapack_checksums(datapack_dir: Path, core_path: Path, dashboard_path: Path) -> dict[str, object]:
    core_digest = sha256_file(core_path)
    dashboard_digest = sha256_file(dashboard_path)
    updates = [
        (
            datapack_dir / "CHECKSUMS.sha256",
            "core/atlas_cluster_terms.parquet",
            core_digest,
        ),
        (
            datapack_dir / "core" / "CHECKSUMS.sha256",
            "atlas_cluster_terms.parquet",
            core_digest,
        ),
        (
            datapack_dir / "CHECKSUMS.sha256",
            "dashboard/tables/nano_terms_topk.parquet",
            dashboard_digest,
        ),
        (
            datapack_dir / "dashboard" / "CHECKSUMS.sha256",
            "tables/nano_terms_topk.parquet",
            dashboard_digest,
        ),
        (
            datapack_dir / "dashboard" / "tables" / "CHECKSUMS.sha256",
            "nano_terms_topk.parquet",
            dashboard_digest,
        ),
    ]
    checksum_updates = []
    for checksum_path, relpath, digest in updates:
        checksum_updates.append(
            {
                "checksum_path": str(checksum_path),
                "relpath": relpath,
                "updated": update_checksum_file(checksum_path, relpath, digest),
            }
        )
    return {
        "core_sha256": core_digest,
        "dashboard_sha256": dashboard_digest,
        "checksum_updates": checksum_updates,
    }


def apply_to_datapack(
    datapack_dir: Path,
    core_nano: pd.DataFrame,
    dashboard_nano: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    stamp = utc_stamp()
    backup_dir = datapack_dir / "qa" / f"keyword_clean_export_backup_{stamp}"
    core_path = datapack_dir / "core" / "atlas_cluster_terms.parquet"
    dashboard_path = datapack_dir / "dashboard" / "tables" / "nano_terms_topk.parquet"
    backup_files([core_path, dashboard_path], backup_dir)
    backup_checksum_files(datapack_dir, backup_dir)

    merged_core = merge_core_terms(datapack_dir, core_nano)
    merged_core.to_parquet(output_dir / "atlas_cluster_terms_sciscape_clean_merged.parquet", index=False)
    merged_core.to_parquet(core_path, index=False)
    dashboard_nano.to_parquet(dashboard_path, index=False)
    checksum_result = update_datapack_checksums(datapack_dir, core_path, dashboard_path)

    marker = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backup_dir": str(backup_dir),
        "core_path": str(core_path),
        "dashboard_path": str(dashboard_path),
        "merged_core_rows": int(len(merged_core)),
        "dashboard_nano_rows": int(len(dashboard_nano)),
        "export_dir": str(output_dir),
        **checksum_result,
    }
    marker_path = datapack_dir / "qa" / f"keyword_clean_export_applied_{stamp}.json"
    marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False))
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", type=Path, required=True, help="SciScape full keyword parquet")
    parser.add_argument("--datapack-dir", type=Path, required=True, help="Science Atlas datapack root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for clean export artifacts")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--family-suppression",
        action="store_true",
        help="Use slower cluster-local subphrase suppression before top-N selection",
    )
    parser.add_argument(
        "--subphrase-summary",
        action="store_true",
        help="Compute slower before/after top10 subphrase-pair summary",
    )
    parser.add_argument("--apply", action="store_true", help="Backup and replace datapack term files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    core_nano, dashboard_nano, summary = build_clean_export(
        keyword_path=args.keywords,
        datapack_dir=args.datapack_dir,
        output_dir=args.output_dir,
        top_n=int(args.top_n),
        family_suppression=bool(args.family_suppression),
        subphrase_summary=bool(args.subphrase_summary),
    )
    result: dict[str, object] = {"summary": summary}
    if args.apply:
        result["applied"] = apply_to_datapack(args.datapack_dir, core_nano, dashboard_nano, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
