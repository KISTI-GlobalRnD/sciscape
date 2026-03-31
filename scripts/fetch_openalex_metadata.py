#!/usr/bin/env python3
"""Fetch title, abstract, authors, and concepts from OpenAlex API.

OpenAlex API supports filter-based batch queries (up to 200 works per page).
We use the pipe-separated work ID filter to batch requests efficiently.

Usage:
    python scripts/fetch_openalex_metadata.py --field 34
    python scripts/fetch_openalex_metadata.py --field 34 --batch-size 50 --polite-email you@example.com
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch")

DATA_ROOT = Path.home() / "Desktop/HDD/local_map_analysis_data/processed/outputs/data"
NODE_DIR = DATA_ROOT / "oa26_gcc_only_k30"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "openalex_metadata"

API_BASE = "https://api.openalex.org/works"
# OpenAlex polite pool: add email for higher rate limit
POLITE_EMAIL = None

# Fields to request from API (minimize payload)
API_SELECT = ",".join([
    "id", "title", "publication_year", "type", "language",
    "abstract_inverted_index",
    "authorships",
    "concepts",
    "topics",
    "cited_by_count", "referenced_works_count",
])


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    # inverted_index: {"word": [pos1, pos2, ...], ...}
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def extract_authors(authorships: list[dict]) -> list[dict]:
    """Extract author info from authorships field."""
    authors = []
    for a in authorships:
        author = a.get("author", {})
        institutions = a.get("institutions", [])
        authors.append({
            "author_id": author.get("id", ""),
            "author_name": author.get("display_name", ""),
            "institution_ids": [inst.get("id", "") for inst in institutions],
            "institution_names": [inst.get("display_name", "") for inst in institutions],
        })
    return authors


def extract_concepts(concepts: list[dict]) -> list[dict]:
    """Extract concept info (id, name, level, score)."""
    return [
        {
            "concept_id": c.get("id", ""),
            "concept_name": c.get("display_name", ""),
            "level": c.get("level", -1),
            "score": c.get("score", 0.0),
        }
        for c in concepts
    ]


def extract_topics(topics: list[dict]) -> list[dict]:
    """Extract topic info."""
    return [
        {
            "topic_id": t.get("id", ""),
            "topic_name": t.get("display_name", ""),
            "subfield_name": t.get("subfield", {}).get("display_name", ""),
            "field_name": t.get("field", {}).get("display_name", ""),
            "domain_name": t.get("domain", {}).get("display_name", ""),
        }
        for t in topics
    ]


def fetch_batch(work_ids: list[str], email: str | None = None) -> list[dict]:
    """Fetch a batch of works from OpenAlex API.

    Uses the pipe-separated filter: filter=openalex:W123|W456|...
    Max ~50 IDs per request to stay under URL length limits.
    """
    # Build filter with full OpenAlex URLs
    oa_ids = "|".join(f"https://openalex.org/{wid}" for wid in work_ids)
    params = {
        "filter": f"openalex:{oa_ids}",
        "select": API_SELECT,
        "per_page": 200,
    }
    if email:
        params["mailto"] = email

    results = []
    page = 1
    while True:
        params["page"] = page
        resp = requests.get(API_BASE, params=params, timeout=30)

        if resp.status_code == 429:
            log.warning("Rate limited, waiting 5s...")
            time.sleep(5)
            continue

        resp.raise_for_status()
        data = resp.json()
        batch_results = data.get("results", [])
        results.extend(batch_results)

        if len(results) >= data.get("meta", {}).get("count", 0):
            break
        page += 1

    return results


def process_works(raw_works: list[dict]) -> list[dict]:
    """Process raw API results into flat records."""
    records = []
    for w in raw_works:
        oa_id = w.get("id", "")
        # Extract W-number from full URL
        work_id = oa_id.split("/")[-1] if oa_id else ""

        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        authors = extract_authors(w.get("authorships", []))
        concepts = extract_concepts(w.get("concepts", []))
        topics = extract_topics(w.get("topics", []))

        records.append({
            "work_id": work_id,
            "title": w.get("title", ""),
            "abstract": abstract,
            "publication_year": w.get("publication_year"),
            "type": w.get("type", ""),
            "language": w.get("language", ""),
            "cited_by_count": w.get("cited_by_count", 0),
            "referenced_works_count": w.get("referenced_works_count", 0),
            "n_authors": len(authors),
            "authors_json": json.dumps(authors, ensure_ascii=False),
            "concepts_json": json.dumps(concepts, ensure_ascii=False),
            "topics_json": json.dumps(topics, ensure_ascii=False),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch OpenAlex metadata")
    parser.add_argument("--field", type=int, default=34)
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of work IDs per API request (max ~50 for URL limits)")
    parser.add_argument("--polite-email", type=str, default=None,
                        help="Email for OpenAlex polite pool (higher rate limit)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial output")
    args = parser.parse_args()

    # Load work IDs from node metadata
    node_path = NODE_DIR / f"field_{args.field}_nodes_oa_meta_gcc_k30.parquet"
    nodes = pl.read_parquet(node_path, columns=["work_id"])
    all_work_ids = nodes["work_id"].unique().sort().to_list()
    log.info("Field %d: %d unique work IDs to fetch", args.field, len(all_work_ids))

    # Check for existing partial results
    out_dir = OUT_DIR / f"field_{args.field}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "works_metadata.parquet"

    fetched_ids = set()
    existing_records = []
    if args.resume and out_path.exists():
        existing = pl.read_parquet(out_path)
        fetched_ids = set(existing["work_id"].to_list())
        existing_records = existing.to_dicts()
        log.info("Resuming: %d already fetched", len(fetched_ids))

    remaining = [wid for wid in all_work_ids if wid not in fetched_ids]
    log.info("Remaining to fetch: %d", len(remaining))

    if not remaining:
        log.info("All work IDs already fetched!")
        return

    # Fetch in batches
    all_records = list(existing_records)
    n_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(remaining))
        batch_ids = remaining[start:end]

        try:
            raw = fetch_batch(batch_ids, args.polite_email)
            processed = process_works(raw)
            all_records.extend(processed)

            fetched_count = len(all_records)
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                log.info("  Batch %d/%d done (%d/%d total)",
                         batch_idx + 1, n_batches, fetched_count, len(all_work_ids))
                # Save checkpoint
                df = pl.DataFrame(all_records)
                df.write_parquet(out_path)

        except Exception as e:
            log.error("Batch %d failed: %s. Saving checkpoint.", batch_idx, e)
            if all_records:
                df = pl.DataFrame(all_records)
                df.write_parquet(out_path)
            raise

        # Rate limiting: ~10 req/sec for polite pool, ~1 req/sec otherwise
        if args.polite_email:
            time.sleep(0.1)
        else:
            time.sleep(1.0)

    # Final save
    df = pl.DataFrame(all_records)
    df.write_parquet(out_path)

    # Summary
    n_with_title = df.filter(pl.col("title").is_not_null() & (pl.col("title") != "")).height
    n_with_abstract = df.filter(pl.col("abstract").is_not_null() & (pl.col("abstract") != "")).height
    n_with_authors = df.filter(pl.col("n_authors") > 0).height

    log.info("\n=== SUMMARY ===")
    log.info("Field %d: %d works fetched", args.field, len(df))
    log.info("  With title:    %d (%.1f%%)", n_with_title, 100 * n_with_title / len(df))
    log.info("  With abstract: %d (%.1f%%)", n_with_abstract, 100 * n_with_abstract / len(df))
    log.info("  With authors:  %d (%.1f%%)", n_with_authors, 100 * n_with_authors / len(df))
    log.info("Output: %s", out_path)


if __name__ == "__main__":
    main()
