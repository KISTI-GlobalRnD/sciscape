"""Pre-partition and cascade search for bottom-up hierarchical Leiden.

Pre-partition assembles small "Lego blocks" at high γ — tightly connected
sub-groups that are unlikely to be split at any lower resolution.  The
graph is then contracted (each block → one super-node), and a γ-cascade
with hot-start finds target clusters on the much smaller contracted graph.

This is analogous to pre-assembling Lego pieces before arranging them
into larger structures: individual papers form blocks, blocks form
nano-clusters, nano-clusters form micro-clusters, and so on.

Functions are pure — caching is the caller's responsibility.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, TYPE_CHECKING

import igraph as ig
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .runner import LeidenRunner, RustLeidenRunner

log = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────


@dataclass(frozen=True)
class PrepartitionResult:
    """Result of high-γ block initialization."""

    pre_membership: List[int]
    gamma_pre: float
    seed: int | None
    n_nodes: int
    n_parts: int
    n_singletons: int
    pre_sizes: Dict[int, int]

    @property
    def node_sizes_list(self) -> List[int]:
        """Per-supernode sizes for the contracted graph (0-indexed order)."""
        return [self.pre_sizes[i] for i in range(self.n_parts)]


@dataclass(frozen=True)
class CascadeResult:
    """Result of a γ-cascade search on the contracted graph."""

    membership: List[int]
    gamma: float
    quality: float
    n_clusters: int
    cascade_path: List[float]
    hot_started: bool


# ── Block initialization ─────────────────────────────────────


def prepartition(
    runner: "LeidenRunner | RustLeidenRunner",
    gamma_pre: float,
    *,
    seed: int | None = None,
) -> PrepartitionResult:
    """Form Lego blocks via high-γ Leiden on the full graph.

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the original graph.
    gamma_pre : float
        High resolution parameter for block formation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PrepartitionResult
        Block membership and metadata.
    """
    result = runner.run(gamma_pre, seed=seed, n_iterations=-1)
    mem = result.membership

    counts = Counter(mem)
    n_parts = len(counts)
    n_singletons = sum(1 for s in counts.values() if s == 1)

    # Renumber to 0-based contiguous IDs (vectorized)
    unique_ids = sorted(counts.keys())
    max_id = max(unique_ids) if unique_ids else 0
    remap_arr = np.empty(max_id + 1, dtype=np.intp)
    for new, old in enumerate(unique_ids):
        remap_arr[old] = new
    mem_remapped = remap_arr[np.array(mem)].tolist()
    sizes_remapped = {remap_arr[k]: v for k, v in counts.items()}

    log.info(
        "Pre-partition: γ=%.1e → %d parts (%d singletons)",
        gamma_pre, n_parts, n_singletons,
    )

    return PrepartitionResult(
        pre_membership=mem_remapped,
        gamma_pre=gamma_pre,
        seed=seed,
        n_nodes=len(mem),
        n_parts=n_parts,
        n_singletons=n_singletons,
        pre_sizes=sizes_remapped,
    )


def contract_graph(
    runner: "LeidenRunner | RustLeidenRunner",
    parts: PrepartitionResult,
) -> tuple["ig.Graph | RustLeidenRunner", "LeidenRunner | RustLeidenRunner"]:
    """Contract the original graph using block membership.

    Returns the contracted graph/runner pair.
    For RustLeidenRunner, contract() returns a new runner directly.
    """
    from .runner import RustLeidenRunner

    if isinstance(runner, RustLeidenRunner):
        contracted_runner = runner.contract(parts.pre_membership)
        return contracted_runner, contracted_runner
    else:
        contracted = runner.contract(
            parts.pre_membership,
            combine_weights="sum",
            keep_loops=True,
        )
        contracted_runner = runner.clone_for_graph(contracted)
        return contracted, contracted_runner


# ── Cascade search ────────────────────────────────────────────


def cascade_search(
    runner: "LeidenRunner | RustLeidenRunner",
    parts: PrepartitionResult,
    gamma_targets: Sequence[float],
    *,
    seed: int | None = None,
    hot_start: bool = True,
) -> CascadeResult:
    """Run γ-cascade with hot-start on the contracted graph.

    Starts from the highest γ in *gamma_targets* and descends,
    using each result as ``initial_membership`` for the next level.

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the **original** graph.
    blocks : PrepartitionResult
        Pre-computed block initialization.
    gamma_targets : sequence of float
        Target γ values, will be sorted descending internally.
    seed : int, optional
        Random seed.
    hot_start : bool
        If True (default), expand the best contracted result and
        refine on the original graph.

    Returns
    -------
    CascadeResult
        Final membership mapped to original node indices.
    """
    contracted, contracted_runner = contract_graph(runner, parts)
    node_sizes = parts.node_sizes_list
    gammas = sorted(gamma_targets, reverse=True)

    prev_mem = None
    best_result = None

    for gamma in gammas:
        if gamma >= parts.gamma_pre:
            log.warning(
                "Skipping γ=%.1e ≥ γ_block=%.1e (would violate monotonicity)",
                gamma,
                parts.gamma_pre,
            )
            continue

        kwargs = dict(
            resolution=gamma,
            seed=seed,
            n_iterations=-1,
            node_sizes=node_sizes,
        )
        if prev_mem is not None:
            kwargs["initial_membership"] = prev_mem

        result = contracted_runner.run(**kwargs)
        prev_mem = result.membership

        n_cl = len(set(result.membership))
        log.info(
            "  Cascade γ=%.1e: %d clusters, Q=%.2f",
            gamma,
            n_cl,
            result.quality,
        )
        best_result = result

    if best_result is None:
        raise ValueError("No valid γ targets (all ≥ γ_block)")

    # Expand to original node indices
    mem_expanded = _expand_membership(
        best_result.membership, parts.pre_membership
    )

    # Hot start: refine on original graph
    final_gamma = gammas[-1] if gammas[-1] < parts.gamma_pre else gammas[0]
    did_hot_start = False

    if hot_start:
        hot_result = runner.run(
            final_gamma,
            seed=seed,
            n_iterations=-1,
            initial_membership=mem_expanded,
        )
        mem_expanded = hot_result.membership
        quality = hot_result.quality
        did_hot_start = True
    else:
        quality = best_result.quality

    return CascadeResult(
        membership=mem_expanded,
        gamma=final_gamma,
        quality=quality,
        n_clusters=len(set(mem_expanded)),
        cascade_path=[g for g in gammas if g < parts.gamma_pre],
        hot_started=did_hot_start,
    )


# ── Persistence ───────────────────────────────────────────────

_META_PREFIX = "prepartition."


def save_prepartition(
    parts: PrepartitionResult,
    path: Path,
    uids: Sequence[str],
    *,
    source: str | None = None,
) -> None:
    """Save pre-partition membership as a Parquet file with metadata.

    Parameters
    ----------
    parts : PrepartitionResult
        Pre-partition result.
    path : Path
        Output parquet file path.
    uids : sequence of str
        Node UIDs matching the order of ``parts.pre_membership``.
    source : str, optional
        Path to the edge file used (for provenance tracking).
    """
    df = pl.DataFrame(
        {
            "uid": list(uids),
            "part": parts.pre_membership,
        }
    )

    metadata = {
        f"{_META_PREFIX}gamma_pre": str(parts.gamma_pre),
        f"{_META_PREFIX}seed": str(parts.seed) if parts.seed is not None else "",
        f"{_META_PREFIX}n_nodes": str(parts.n_nodes),
        f"{_META_PREFIX}n_edges": "",  # filled by caller if desired
        f"{_META_PREFIX}n_parts": str(parts.n_parts),
        f"{_META_PREFIX}n_singletons": str(parts.n_singletons),
        f"{_META_PREFIX}source": source or "",
        f"{_META_PREFIX}created": datetime.now(timezone.utc).isoformat(),
    }

    path = Path(path)
    # polars sink_parquet doesn't support custom metadata directly;
    # write via pyarrow for metadata embedding.
    table = df.to_arrow()
    existing_meta = table.schema.metadata or {}
    existing_meta.update({k.encode(): v.encode() for k, v in metadata.items()})
    table = table.replace_schema_metadata(existing_meta)

    import pyarrow.parquet as pq

    pq.write_table(table, str(path))
    log.info("Pre-partition saved: %s (%d parts)", path, parts.n_parts)


def load_prepartition(path: Path) -> PrepartitionResult | None:
    """Load cached blocks from a Parquet file.

    Returns None if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None

    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    raw_meta = table.schema.metadata or {}
    meta = {
        k.decode(): v.decode()
        for k, v in raw_meta.items()
        if k.decode().startswith(_META_PREFIX)
    }

    if f"{_META_PREFIX}gamma_pre" not in meta:
        log.warning("No block_init metadata in %s", path)
        return None

    df = pl.from_arrow(table)
    pre_mem = df["part"].to_list()
    gamma_pre = float(meta[f"{_META_PREFIX}gamma_pre"])
    seed_str = meta.get(f"{_META_PREFIX}seed", "")
    seed = int(seed_str) if seed_str else None

    counts = Counter(pre_mem)

    return PrepartitionResult(
        pre_membership=pre_mem,
        gamma_pre=gamma_pre,
        seed=seed,
        n_nodes=len(pre_mem),
        n_parts=len(counts),
        n_singletons=sum(1 for s in counts.values() if s == 1),
        pre_sizes=dict(counts),
    )


def load_prepartition_metadata(path: Path) -> dict | None:
    """Load only the metadata from a blocks Parquet file (without reading data).

    Returns a dict of metadata fields, or None if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return None

    import pyarrow.parquet as pq

    schema = pq.read_schema(str(path))
    raw_meta = schema.metadata or {}
    prefix = _META_PREFIX
    return {
        k.decode().removeprefix(prefix): v.decode()
        for k, v in raw_meta.items()
        if k.decode().startswith(prefix)
    }


def is_cache_valid(
    path: Path,
    gamma_pre: float,
    n_nodes: int,
    source: str | None = None,
) -> bool:
    """Check if cached blocks are valid for the given parameters."""
    meta = load_prepartition_metadata(path)
    if meta is None:
        return False

    if float(meta.get("gamma_pre", -1)) != gamma_pre:
        return False
    if int(meta.get("n_nodes", -1)) != n_nodes:
        return False
    if source and meta.get("source", "") != source:
        return False

    return True


# ── Helpers ───────────────────────────────────────────────────


def _expand_membership(
    contracted_mem: Sequence[int],
    pre_mem: Sequence[int],
) -> List[int]:
    """Map contracted-graph membership back to original node indices."""
    return np.array(contracted_mem)[np.array(pre_mem)].tolist()


__all__ = [
    "PrepartitionResult",
    "CascadeResult",
    "prepartition",
    "contract_graph",
    "cascade_search",
    "save_prepartition",
    "load_prepartition",
    "load_prepartition_metadata",
    "is_cache_valid",
]
