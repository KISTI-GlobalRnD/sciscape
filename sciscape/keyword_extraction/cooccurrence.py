"""Term co-occurrence collection (Stage 6).

Scans documents once to build a term x term co-occurrence matrix for
selected (top-K) terms only. This 2-pass approach avoids the O(V^2) cost
of building co-occurrence for the full vocabulary.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def collect_cooccurrence(
    texts_iter: Iterator[List[str]],
    selected_terms: Sequence[str],
    lowercase: bool = True,
    token_pattern: str = r"(?u)\b\w\w+\b",
    strip_accents: Optional[str] = None,
    stopwords: Optional[List[str]] = None,
    min_cooc_count: int = 1,
) -> sp.csr_matrix:
    """Build a term x term co-occurrence matrix from document batches.

    Parameters
    ----------
    texts_iter : iterator of list[str]
        Yields batches of document texts.
    selected_terms : list of str
        Terms to track co-occurrence for (typically top-K).
    min_cooc_count : int
        Minimum co-occurrence count to keep (sparsity filter).

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric matrix of shape (n_terms, n_terms) with co-occurrence counts.
    """
    if not selected_terms:
        return sp.csr_matrix((0, 0), dtype=np.int64)

    # Separate unigrams and phrases for matching
    uni_terms = [t for t in selected_terms if " " not in t]
    phrase_terms = [t for t in selected_terms if " " in t]

    term_to_idx: Dict[str, int] = {t: i for i, t in enumerate(selected_terms)}
    n = len(selected_terms)

    # Build vectorizers for matching
    vec_uni = None
    if uni_terms:
        vec_uni = CountVectorizer(
            lowercase=lowercase,
            token_pattern=token_pattern,
            strip_accents=strip_accents,
            stop_words=stopwords,
            vocabulary={t: i for i, t in enumerate(uni_terms)},
            ngram_range=(1, 1),
            binary=True,
            dtype=np.int32,
        )

    vec_phrase = None
    if phrase_terms:
        # Determine ngram range from phrase lengths
        min_n = min(len(t.split()) for t in phrase_terms)
        max_n = max(len(t.split()) for t in phrase_terms)
        vec_phrase = CountVectorizer(
            lowercase=lowercase,
            token_pattern=token_pattern,
            strip_accents=strip_accents,
            stop_words=stopwords,
            vocabulary={t: i for i, t in enumerate(phrase_terms)},
            ngram_range=(min_n, max_n),
            binary=True,
            dtype=np.int32,
        )

    # Accumulate co-occurrence counts
    cooc = sp.lil_matrix((n, n), dtype=np.int64)

    for batch_texts in texts_iter:
        if not batch_texts:
            continue

        n_docs = len(batch_texts)
        present_indices_per_doc: List[List[int]] = [[] for _ in range(n_docs)]

        if vec_uni is not None:
            X_uni = vec_uni.transform(batch_texts)
            for doc_idx in range(X_uni.shape[0]):
                row = X_uni.getrow(doc_idx)
                present_indices_per_doc[doc_idx] = [
                    term_to_idx[uni_terms[j]] for j in row.indices
                ]

        if vec_phrase is not None:
            X_phrase = vec_phrase.transform(batch_texts)
            for doc_idx in range(X_phrase.shape[0]):
                row = X_phrase.getrow(doc_idx)
                present_indices_per_doc[doc_idx].extend(
                    term_to_idx[phrase_terms[j]] for j in row.indices
                )

        # For each document, count co-occurrences between present terms
        for indices in present_indices_per_doc:
            if len(indices) < 2:
                continue
            for i_pos in range(len(indices)):
                for j_pos in range(i_pos + 1, len(indices)):
                    ti, tj = indices[i_pos], indices[j_pos]
                    cooc[ti, tj] += 1
                    cooc[tj, ti] += 1

    result = cooc.tocsr()

    # Apply minimum count filter
    if min_cooc_count > 1:
        result.data[result.data < min_cooc_count] = 0
        result.eliminate_zeros()

    return result


__all__ = ["collect_cooccurrence"]
