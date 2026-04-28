"""Extract abbreviation dictionary from paper abstracts.

Scans "full name (ABBR)" patterns in abstracts to build an
abbreviation → full name mapping. No external API needed.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import polars as pl

log = logging.getLogger(__name__)

# Patterns: "Full Name (ABBR)" — captures multi-word name + uppercase abbreviation
_PATTERN_PAREN = re.compile(
    r'([A-Z][a-z]+(?:[\s-]+[a-z]+){1,6})\s*\(([A-Z][A-Z0-9]{1,9})\)'
)
# Also: "(ABBR) full name" at sentence start
_PATTERN_PAREN_REV = re.compile(
    r'\(([A-Z][A-Z0-9]{1,9})\)\s+([a-z]+(?:\s+[a-z]+){1,6})'
)


def extract_abbreviations(
    abstracts: pl.DataFrame | Sequence[str],
    *,
    min_count: int = 2,
    title_col: str = "title",
    abstract_col: str = "abstract",
) -> Dict[str, str]:
    """Extract abbreviation → full name mapping from abstracts.

    Parameters
    ----------
    abstracts : pl.DataFrame or list of str
        If DataFrame, uses title + abstract columns.
    min_count : int
        Minimum occurrences to keep (filters noise).

    Returns
    -------
    dict
        {abbreviation: canonical_full_name}
    """
    if isinstance(abstracts, pl.DataFrame):
        texts = []
        for col in [abstract_col, title_col]:
            if col in abstracts.columns:
                texts.extend(abstracts[col].to_list())
    else:
        texts = list(abstracts)

    # Count all (abbr, full) pairs
    pair_counts: Counter = Counter()
    for text in texts:
        if not text:
            continue
        for pattern in [_PATTERN_PAREN, _PATTERN_PAREN_REV]:
            for m in pattern.finditer(text):
                if pattern == _PATTERN_PAREN:
                    full, abbr = m.group(1).strip(), m.group(2).strip()
                else:
                    abbr, full = m.group(1).strip(), m.group(2).strip()

                if len(abbr) < 2 or len(full.split()) < 2:
                    continue
                pair_counts[(abbr, full.lower())] += 1

    # For each abbreviation, pick the most common full name
    abbr_candidates: Dict[str, Counter] = defaultdict(Counter)
    for (abbr, full), count in pair_counts.items():
        if count >= min_count:
            # Clean: remove leading articles
            full_clean = re.sub(r'^(the|a|an)\s+', '', full).strip()
            abbr_candidates[abbr][full_clean] += count

    result = {}
    for abbr, candidates in abbr_candidates.items():
        best_full = candidates.most_common(1)[0][0]
        result[abbr] = best_full

    log.info("Extracted %d abbreviations from %d texts (min_count=%d)",
             len(result), len(texts), min_count)
    return result


def abbreviation_table(abbr_dict: Dict[str, str]) -> pl.DataFrame:
    """Convert abbreviation dict to DataFrame."""
    return pl.DataFrame({
        "abbreviation": list(abbr_dict.keys()),
        "full_name": list(abbr_dict.values()),
    }).sort("abbreviation")


def expand_labels_with_abbreviations(
    labels: List[str],
    abbr_dict: Dict[str, str],
    *,
    mode: str = "append",
) -> List[str]:
    """Expand abbreviations in cluster labels.

    Parameters
    ----------
    mode : str
        "append": "DFT analysis" → "DFT (density functional theory) analysis"
        "replace": "DFT analysis" → "density functional theory analysis"
        "keep": no change, just for lookup
    """
    result = []
    for label in labels:
        new_label = label
        for abbr, full in abbr_dict.items():
            # Only expand if abbreviation appears as whole word
            pattern = r'\b' + re.escape(abbr) + r'\b'
            if re.search(pattern, new_label):
                if mode == "append":
                    new_label = re.sub(pattern, f"{abbr} ({full})", new_label, count=1)
                elif mode == "replace":
                    new_label = re.sub(pattern, full, new_label, count=1)
        result.append(new_label)
    return result


__all__ = [
    "extract_abbreviations",
    "abbreviation_table",
    "expand_labels_with_abbreviations",
]
