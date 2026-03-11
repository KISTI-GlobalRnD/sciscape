"""Temporal metrics computation (Stage 10: temporal) for keyword extraction pipeline."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, MutableMapping, Tuple

import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

TokenCounter = Counter[str]
YearCounter = Counter[int]
TermYearCounter = MutableMapping[str, YearCounter]
ClusterTermYearCounter = MutableMapping[int, TermYearCounter]


class TemporalMixin:
    """Mixin providing temporal series computation for KeywordExtractionPipeline."""

    def _compute_year_series(self, top_df: pd.DataFrame) -> ClusterTermYearCounter:
        if top_df.empty:
            return defaultdict(lambda: defaultdict(Counter))

        cfg = self.config
        self._log("Stage 3: computing year series for %d rows", len(top_df))
        token_to_targets_uni: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        token_to_targets_phrase: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        for row in top_df.itertuples(index=False):
            cluster_id = int(row.cluster_id)
            canonical = str(row.term)
            sources = getattr(row, "source_terms", None)
            if sources is None or (isinstance(sources, float) and math.isnan(sources)):  # type: ignore[arg-type]
                sources_list: List[str] = [canonical]
            elif isinstance(sources, str):
                sources_list = [sources]
            else:
                sources_list = [str(token) for token in sources if str(token).strip()]
            if not sources_list:
                sources_list = [canonical]
            unique_sources: List[str] = []
            for token in sources_list:
                token_str = token.strip()
                if not token_str or token_str in unique_sources:
                    continue
                unique_sources.append(token_str)
                if " " in token_str:
                    token_to_targets_phrase[token_str].append((cluster_id, canonical))
                else:
                    token_to_targets_uni[token_str].append((cluster_id, canonical))

        uni_vocab = sorted(token_to_targets_uni.keys())
        phrase_vocab = sorted(token_to_targets_phrase.keys())

        vec_uni = CountVectorizer(
            lowercase=cfg.lowercase,
            stop_words=self.stopwords_list,
            token_pattern=cfg.token_pattern,
            strip_accents=cfg.strip_accents,
            vocabulary={t: i for i, t in enumerate(uni_vocab)} if uni_vocab else None,
            ngram_range=(1, 1),
            dtype=np.int32,
        ) if uni_vocab else None

        vec_phrase = CountVectorizer(
            lowercase=cfg.lowercase,
            stop_words=self.stopwords_list,
            token_pattern=cfg.token_pattern,
            strip_accents=cfg.strip_accents,
            vocabulary={t: i for i, t in enumerate(phrase_vocab)} if phrase_vocab else None,
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            dtype=np.int32,
        ) if phrase_vocab else None

        idx2term_uni = np.array(uni_vocab, dtype=str) if uni_vocab else None
        idx2term_phrase = np.array(phrase_vocab, dtype=str) if phrase_vocab else None

        term_year: ClusterTermYearCounter = defaultdict(lambda: defaultdict(Counter))
        denom_tokens: Dict[int, Counter[int]] = defaultdict(Counter)
        full_uni_vec = self.vec_uni

        for batch in self._data.batch_iter():
            clusters = batch["cluster_id"].astype(int).to_numpy()
            years = batch["pubyear"].to_numpy()
            texts = batch["text"].tolist()

            if full_uni_vec is not None:
                doc_totals = np.asarray(full_uni_vec.transform(texts).sum(axis=1)).ravel()
                for idx, total in enumerate(doc_totals):
                    if total == 0:
                        continue
                    year = years[idx]
                    if pd.isna(year):
                        continue
                    cid = int(clusters[idx])
                    denom_tokens[cid][int(year)] += int(total)

            if vec_uni is not None:
                X = vec_uni.transform(texts).tocoo()
                for r, c, v in zip(X.row, X.col, X.data):
                    year = years[r]
                    if pd.isna(year):
                        continue
                    cid = int(clusters[r])
                    token = idx2term_uni[c]  # type: ignore[index]
                    for target_cid, canonical in token_to_targets_uni.get(token, []):
                        if target_cid != cid:
                            continue
                        term_year[cid][canonical][int(year)] += int(v)

            if vec_phrase is not None:
                X = vec_phrase.transform(texts).tocoo()
                for r, c, v in zip(X.row, X.col, X.data):
                    year = years[r]
                    if pd.isna(year):
                        continue
                    cid = int(clusters[r])
                    token = idx2term_phrase[c]  # type: ignore[index]
                    for target_cid, canonical in token_to_targets_phrase.get(token, []):
                        if target_cid != cid:
                            continue
                        term_year[cid][canonical][int(year)] += int(v)

        self.cluster_year_token_denoms = denom_tokens
        self._log("Stage 3: year series computed for %d clusters", len(term_year))
        return term_year

    def _build_time_series_metrics(self, top_df: pd.DataFrame, term_year: ClusterTermYearCounter) -> pd.DataFrame:
        if top_df.empty:
            return top_df.assign(
                year_denominators=[],
                ppm_series=[],
                loglift_series=[],
                bayesian_log_odds_series=[],
            )

        global_term_counts: Dict[str, Counter[int]] = defaultdict(Counter)
        for term_year_map in term_year.values():
            for term, year_counter in term_year_map.items():
                for year, count in year_counter.items():
                    global_term_counts[str(term)][int(year)] += int(count)

        global_year_denoms: Dict[int, int] = defaultdict(int)
        for year_map in self.cluster_year_token_denoms.values():
            for year, denom in year_map.items():
                global_year_denoms[int(year)] += int(denom)

        alpha = 0.5
        prior = 0.5

        pub_year_list: List[Dict[int, int]] = []
        year_denom_list: List[Dict[int, int]] = []
        ppm_list: List[Dict[int, float]] = []
        loglift_list: List[Dict[int, float]] = []
        bayes_list: List[Dict[int, float]] = []

        for row_idx, row in enumerate(top_df.itertuples(index=False)):
            cid = int(row.cluster_id)
            term = str(row.term)
            year_counts_raw = getattr(row, "pub_year_series", None)
            if not year_counts_raw:
                year_counts_raw = term_year.get(cid, {}).get(term, {})
            year_items = sorted(
                ((int(k), int(v)) for k, v in dict(year_counts_raw).items() if v),
                key=lambda kv: kv[0],
            )
            year_counts_sorted: Dict[int, int] = OrderedDict(year_items)
            denom_map = self.cluster_year_token_denoms.get(cid, {})

            year_denoms = OrderedDict((year, int(denom_map.get(year, 0))) for year in year_counts_sorted)
            ppm_series: Dict[int, float] = {}
            loglift_series: Dict[int, float] = {}
            bayes_series: Dict[int, float] = {}

            global_counts_map = global_term_counts.get(term, {})

            for year, count in year_counts_sorted.items():
                year_int = int(year)
                denom_c = denom_map.get(year_int, 0)
                global_count = global_counts_map.get(year_int, 0)
                global_denom = global_year_denoms.get(year_int, 0)

                ppm_series[year_int] = 1e6 * count / denom_c if denom_c else float("nan")

                if denom_c > 0 and global_denom > 0:
                    p_cluster = (count + alpha) / (denom_c + alpha)
                    p_global = (global_count + alpha) / (global_denom + alpha)
                    loglift_series[year_int] = float(np.log(p_cluster) - np.log(p_global))

                    p_year = (global_count + alpha) / (global_denom + alpha)
                    theta_cluster = (count + prior * p_year) / (denom_c + prior)
                    theta_global = (global_count + prior * p_year) / (global_denom + prior)
                    theta_cluster = float(np.clip(theta_cluster, 1e-9, 1 - 1e-9))
                    theta_global = float(np.clip(theta_global, 1e-9, 1 - 1e-9))
                    bayes_series[year_int] = float(
                        (np.log(theta_cluster) - np.log(1 - theta_cluster))
                        - (np.log(theta_global) - np.log(1 - theta_global))
                    )
                else:
                    loglift_series[year_int] = float("nan")
                    bayes_series[year_int] = float("nan")

            pub_year_list.append(year_counts_sorted)
            year_denom_list.append(year_denoms)
            ppm_list.append(ppm_series)
            loglift_list.append(loglift_series)
            bayes_list.append(bayes_series)

        return top_df.assign(
            pub_year_series=pub_year_list,
            year_denominators=year_denom_list,
            ppm_series=ppm_list,
            loglift_series=loglift_list,
            bayesian_log_odds_series=bayes_list,
        )
