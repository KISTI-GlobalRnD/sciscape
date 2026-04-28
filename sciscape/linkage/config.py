"""Configuration for link-type edge construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence


class Normalization(str, Enum):
    """Normalization methods for BC/CC coupling weights."""

    RAW = "raw"
    COSINE = "cosine"
    ASSOC_STRENGTH = "assoc_strength"


class DCNormalization(str, Enum):
    """Normalization methods for direct citation weights."""

    BINARY = "binary"
    FRACTIONAL = "fractional"


class CombineMethod(str, Enum):
    """Methods for combining multiple link-type edge sets.

    Layer-agnostic (treat all layers equally):
        SUM, MAX, NOISY_OR, MIN

    Consensus (reward multi-layer agreement):
        CONSENSUS — ``(Σ wᵢ / n) × consensus_count``

    Weighted:
        WEIGHTED_SUM — requires ``weights`` dict in ``combine_edges()``

    Multiplicative (both signals must agree):
        PRODUCT, GEOMETRIC_MEAN, HARMONIC_MEAN
    """

    # ── Layer-agnostic ────────────────────────────────────────────
    SUM = "sum"             # Σ w_i  (after [0,1] normalization)
    MAX = "max"             # max(w_i)
    NOISY_OR = "noisy_or"   # 1 - Π(1 - w_i)
    MIN = "min"             # min(w_i)  — intersection-like

    # ── Consensus ────────────────────────────────────────────────
    CONSENSUS = "consensus"  # (Σ w_i / n) × consensus_count

    # ── Weighted ──────────────────────────────────────────────────
    WEIGHTED_SUM = "weighted_sum"  # Σ α_i · w_i

    # ── Multiplicative ────────────────────────────────────────────
    PRODUCT = "product"             # Π w_i
    GEOMETRIC_MEAN = "geometric_mean"  # (Π w_i)^(1/n)
    HARMONIC_MEAN = "harmonic_mean"    # n / Σ(1/w_i)


@dataclass
class LinkageConfig:
    """Configuration for link-type edge construction from citation data.

    Parameters
    ----------
    citing_col : str
        Column name for the citing paper ID.
    cited_col : str
        Column name for the cited paper ID.
    cited_in_set_col : str or None
        Column indicating whether the cited paper is within the field.
        If None, all citations are treated as in-set.
    min_shared : int
        Minimum shared references (BC) or citers (CC) to keep an edge.
    dc_norms : sequence of DCNormalization
        Which DC normalizations to compute.
    bc_norms : sequence of Normalization
        Which BC normalizations to compute.
    cc_norms : sequence of Normalization
        Which CC normalizations to compute.
    """

    citing_col: str = "citing_work_id"
    cited_col: str = "cited_work_id"
    cited_in_set_col: Optional[str] = "cited_in_set"

    # BC/CC minimum shared count — papers must share at least this many
    # references (BC) or citers (CC) to form an edge.
    # BC default 3: sharing only 2 refs in a large field is likely noise.
    # CC default 2: co-citation is sparser, keep standard bibliometric threshold.
    bc_min_shared: int = 3
    cc_min_shared: int = 2

    # Per-row top-k for BC/CC matmul (requires sparse_dot_topn).
    # When set, M @ M.T keeps only top-k entries per row during multiplication,
    # avoiding full result materialisation at large scale.
    # Default 500: keeps top-500 strongest couplings per paper — sufficient
    # for downstream top-k filtering (typically k=30) while bounding memory.
    bc_topk: Optional[int] = 500
    cc_topk: Optional[int] = 500

    dc_norms: Sequence[DCNormalization] = (DCNormalization.BINARY, DCNormalization.FRACTIONAL)
    bc_norms: Sequence[Normalization] = (Normalization.RAW, Normalization.COSINE, Normalization.ASSOC_STRENGTH)
    cc_norms: Sequence[Normalization] = (Normalization.RAW, Normalization.COSINE, Normalization.ASSOC_STRENGTH)


__all__ = [
    "CombineMethod",
    "DCNormalization",
    "LinkageConfig",
    "Normalization",
]
