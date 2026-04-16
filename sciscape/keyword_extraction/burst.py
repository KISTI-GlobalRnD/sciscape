"""Keyword burst detection.

Identifies keywords with sudden increases in frequency,
signaling emerging research topics. Uses a simplified
Kleinberg-inspired model + relative growth rate.

Two methods:
  1. growth_rate: year-over-year relative frequency change
  2. kleinberg: two-state automaton burst detection
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


@dataclass
class BurstPeriod:
    """One burst period for a keyword."""
    keyword: str
    start_year: int
    end_year: int
    intensity: float  # burst strength (higher = stronger burst)


@dataclass
class BurstResult:
    """Result of burst detection."""
    bursts: List[BurstPeriod]
    keyword_timeseries: Dict[str, Dict[int, float]]  # keyword → {year: freq}
    years: List[int]

    def top_bursts(self, n: int = 20) -> List[BurstPeriod]:
        return sorted(self.bursts, key=lambda b: -b.intensity)[:n]

    def active_bursts(self, year: int) -> List[BurstPeriod]:
        return [b for b in self.bursts if b.start_year <= year <= b.end_year]

    def summary(self, n: int = 10) -> str:
        lines = [f"Burst Detection: {len(self.bursts)} bursts found"]
        for b in self.top_bursts(n):
            lines.append(f"  [{b.start_year}-{b.end_year}] {b.keyword} (intensity={b.intensity:.2f})")
        return "\n".join(lines)


def detect_bursts_growth(
    keywords_df: pl.DataFrame,
    abstracts_df: pl.DataFrame,
    *,
    keyword_col: str = "keyword",
    cluster_col: str | None = None,
    min_freq: int = 5,
    growth_threshold: float = 2.0,
    min_years: int = 2,
) -> BurstResult:
    """Detect keyword bursts using relative growth rate.

    A keyword is "bursting" if its frequency in year t is >= growth_threshold
    times its average frequency in previous years.

    Parameters
    ----------
    keywords_df : pl.DataFrame
        Keywords with cluster assignments.
    abstracts_df : pl.DataFrame
        Abstracts with uid, pubyear, and text.
    keyword_col : str
        Column name for keywords.
    min_freq : int
        Minimum total frequency to consider.
    growth_threshold : float
        Minimum ratio current/average_past to trigger burst.
    min_years : int
        Minimum burst duration (years) to report.

    Returns
    -------
    BurstResult
    """
    # Build keyword × year frequency matrix
    if "pubyear" not in abstracts_df.columns:
        log.warning("No pubyear column in abstracts, cannot detect bursts")
        return BurstResult(bursts=[], keyword_timeseries={}, years=[])

    # Get keyword-document mapping
    kw_list = keywords_df[keyword_col].to_list()
    all_keywords = set(kw_list)

    # Count keyword occurrences per year from abstracts
    uid_to_year = dict(zip(
        abstracts_df["uid"].to_list(),
        [int(y) if y else 0 for y in abstracts_df["pubyear"].to_list()],
    ))

    # Simple approach: check if keyword appears in title/abstract per year
    kw_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: Dict[int, int] = defaultdict(int)

    for row in abstracts_df.select("uid", "title", "abstract", "pubyear").iter_rows():
        uid, title, abstract, year = row
        if not year or year <= 0:
            continue
        year = int(year)
        text = f"{title or ''} {abstract or ''}".lower()
        year_totals[year] += 1
        for kw in all_keywords:
            if kw.lower() in text:
                kw_year_counts[kw][year] += 1

    years = sorted(year_totals.keys())
    if len(years) < 3:
        return BurstResult(bursts=[], keyword_timeseries={}, years=years)

    # Compute relative frequency (per-year normalized)
    kw_timeseries: Dict[str, Dict[int, float]] = {}
    for kw, yc in kw_year_counts.items():
        total = sum(yc.values())
        if total < min_freq:
            continue
        ts = {}
        for y in years:
            count = yc.get(y, 0)
            denom = year_totals.get(y, 1)
            ts[y] = count / denom  # relative frequency
        kw_timeseries[kw] = ts

    # Detect bursts: growth_threshold × running average
    bursts: List[BurstPeriod] = []
    for kw, ts in kw_timeseries.items():
        in_burst = False
        burst_start = 0
        burst_intensity = 0.0

        for i, y in enumerate(years):
            freq = ts.get(y, 0)
            # Running average of previous years
            past = [ts.get(py, 0) for py in years[:i]]
            avg_past = np.mean(past) if past else 0

            if avg_past > 0 and freq / avg_past >= growth_threshold:
                if not in_burst:
                    burst_start = y
                    in_burst = True
                burst_intensity = max(burst_intensity, freq / avg_past)
            else:
                if in_burst and (y - 1 - burst_start + 1) >= min_years:
                    bursts.append(BurstPeriod(
                        keyword=kw, start_year=burst_start,
                        end_year=y - 1, intensity=round(burst_intensity, 2),
                    ))
                in_burst = False
                burst_intensity = 0.0

        # Close open burst
        if in_burst and (years[-1] - burst_start + 1) >= min_years:
            bursts.append(BurstPeriod(
                keyword=kw, start_year=burst_start,
                end_year=years[-1], intensity=round(burst_intensity, 2),
            ))

    log.info("Burst detection: %d keywords → %d bursts", len(kw_timeseries), len(bursts))
    return BurstResult(bursts=bursts, keyword_timeseries=kw_timeseries, years=years)


def detect_bursts_kleinberg(
    keyword_counts: Dict[str, Dict[int, int]],
    year_totals: Dict[int, int],
    *,
    s: float = 2.0,
    gamma: float = 1.0,
) -> List[BurstPeriod]:
    """Simplified Kleinberg burst detection (two-state automaton).

    Parameters
    ----------
    keyword_counts : dict
        {keyword: {year: count}}
    year_totals : dict
        {year: total_docs}
    s : float
        State transition cost scaling (higher = fewer bursts).
    gamma : float
        Penalty for state transitions.

    Returns
    -------
    list of BurstPeriod
    """
    years = sorted(year_totals.keys())
    n_years = len(years)
    if n_years < 3:
        return []

    bursts = []
    for kw, yc in keyword_counts.items():
        total_count = sum(yc.get(y, 0) for y in years)
        total_docs = sum(year_totals.get(y, 0) for y in years)
        if total_count < 3 or total_docs == 0:
            continue

        p_base = total_count / total_docs  # base rate
        p_burst = min(p_base * s, 0.99)  # burst rate (capped)

        # Viterbi-like: find periods where burst state is optimal
        # State 0 = normal, State 1 = burst
        state = 0
        burst_start = 0
        max_ratio = 0.0

        for y in years:
            count = yc.get(y, 0)
            n = year_totals.get(y, 1)
            p_obs = count / n if n > 0 else 0

            if p_obs > p_burst * 0.8:  # likely burst
                if state == 0:
                    burst_start = y
                    state = 1
                max_ratio = max(max_ratio, p_obs / max(p_base, 1e-10))
            else:
                if state == 1:
                    duration = y - burst_start
                    if duration >= 2:
                        bursts.append(BurstPeriod(
                            keyword=kw, start_year=burst_start,
                            end_year=y - 1, intensity=round(max_ratio, 2),
                        ))
                    state = 0
                    max_ratio = 0.0

        if state == 1:
            duration = years[-1] - burst_start + 1
            if duration >= 2:
                bursts.append(BurstPeriod(
                    keyword=kw, start_year=burst_start,
                    end_year=years[-1], intensity=round(max_ratio, 2),
                ))

    return bursts


__all__ = ["detect_bursts_growth", "detect_bursts_kleinberg", "BurstResult", "BurstPeriod"]
