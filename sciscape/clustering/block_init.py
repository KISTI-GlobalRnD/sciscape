"""Block initialization and cascade search for bottom-up hierarchical Leiden.

High-γ Leiden produces dense "Lego blocks" (including size-1 singletons).
The graph is then contracted to super-nodes, and a γ-cascade with hot-start
finds target clusters efficiently on the much smaller contracted graph.

Functions are pure — caching is the caller's responsibility.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, TYPE_CHECKING

import igraph as ig
import polars as pl

if TYPE_CHECKING:
    from .runner import LeidenRunner, LeidenRunResult

log = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────


@dataclass(frozen=True)
class BlockInitResult:
    """Result of high-γ block initialization."""

    block_membership: List[int]
    gamma_block: float
    seed: int | None
    n_nodes: int
    n_blocks: int
    n_singletons: int
    block_sizes: Dict[int, int]

    @property
    def node_sizes_list(self) -> List[int]:
        """Per-supernode sizes for the contracted graph (0-indexed order)."""
        return [self.block_sizes[i] for i in range(self.n_blocks)]


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


def block_init(
    runner: "LeidenRunner",
    gamma_block: float,
    *,
    seed: int | None = None,
) -> BlockInitResult:
    """Form Lego blocks via high-γ Leiden on the full graph.

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the original graph.
    gamma_block : float
        High resolution parameter for block formation.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BlockInitResult
        Block membership and metadata.
    """
    result = runner.run(gamma_block, seed=seed, n_iterations=-1)
    mem = result.membership

    counts = Counter(mem)
    n_blocks = len(counts)
    n_singletons = sum(1 for s in counts.values() if s == 1)

    # Renumber to 0-based contiguous IDs
    unique_ids = sorted(counts.keys())
    remap = {old: new for new, old in enumerate(unique_ids)}
    mem_remapped = [remap[b] for b in mem]
    sizes_remapped = {remap[k]: v for k, v in counts.items()}

    log.info(
        "Block init: γ=%.1e → %d blocks (%d singletons), %.1fs",
        gamma_block,
        n_blocks,
        n_singletons,
        0.0,  # caller can time externally
    )

    return BlockInitResult(
        block_membership=mem_remapped,
        gamma_block=gamma_block,
        seed=seed,
        n_nodes=len(mem),
        n_blocks=n_blocks,
        n_singletons=n_singletons,
        block_sizes=sizes_remapped,
    )


def contract_graph(
    runner: "LeidenRunner",
    blocks: BlockInitResult,
) -> tuple[ig.Graph, "LeidenRunner"]:
    """Contract the original graph using block membership.

    Returns the contracted graph (no self-loops) and a runner for it.
    """
    contracted = runner.contract(
        blocks.block_membership,
        combine_weights="sum",
        keep_loops=True,  # simplify(loops=True) removes self-loops
    )
    contracted_runner = runner.clone_for_graph(contracted)
    return contracted, contracted_runner


# ── Cascade search ────────────────────────────────────────────


def cascade_search(
    runner: "LeidenRunner",
    blocks: BlockInitResult,
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
    blocks : BlockInitResult
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
    contracted, contracted_runner = contract_graph(runner, blocks)
    node_sizes = blocks.node_sizes_list
    gammas = sorted(gamma_targets, reverse=True)

    prev_mem = None
    best_result = None

    for gamma in gammas:
        if gamma >= blocks.gamma_block:
            log.warning(
                "Skipping γ=%.1e ≥ γ_block=%.1e (would violate monotonicity)",
                gamma,
                blocks.gamma_block,
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
        best_result.membership, blocks.block_membership
    )

    # Hot start: refine on original graph
    final_gamma = gammas[-1] if gammas[-1] < blocks.gamma_block else gammas[0]
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
        cascade_path=[g for g in gammas if g < blocks.gamma_block],
        hot_started=did_hot_start,
    )


# ── Persistence ───────────────────────────────────────────────

_META_PREFIX = "block_init."


def save_blocks(
    blocks: BlockInitResult,
    path: Path,
    uids: Sequence[str],
    *,
    source: str | None = None,
) -> None:
    """Save block membership as a Parquet file with metadata.

    Parameters
    ----------
    blocks : BlockInitResult
        Block initialization result.
    path : Path
        Output parquet file path.
    uids : sequence of str
        Node UIDs matching the order of ``blocks.block_membership``.
    source : str, optional
        Path to the edge file used (for provenance tracking).
    """
    df = pl.DataFrame(
        {
            "uid": list(uids),
            "block": blocks.block_membership,
        }
    )

    metadata = {
        f"{_META_PREFIX}gamma_block": str(blocks.gamma_block),
        f"{_META_PREFIX}seed": str(blocks.seed) if blocks.seed is not None else "",
        f"{_META_PREFIX}n_nodes": str(blocks.n_nodes),
        f"{_META_PREFIX}n_edges": "",  # filled by caller if desired
        f"{_META_PREFIX}n_blocks": str(blocks.n_blocks),
        f"{_META_PREFIX}n_singletons": str(blocks.n_singletons),
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
    log.info("Blocks saved: %s (%d blocks)", path, blocks.n_blocks)


def load_blocks(path: Path) -> BlockInitResult | None:
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

    if f"{_META_PREFIX}gamma_block" not in meta:
        log.warning("No block_init metadata in %s", path)
        return None

    df = pl.from_arrow(table)
    block_mem = df["block"].to_list()
    gamma_block = float(meta[f"{_META_PREFIX}gamma_block"])
    seed_str = meta.get(f"{_META_PREFIX}seed", "")
    seed = int(seed_str) if seed_str else None

    counts = Counter(block_mem)

    return BlockInitResult(
        block_membership=block_mem,
        gamma_block=gamma_block,
        seed=seed,
        n_nodes=len(block_mem),
        n_blocks=len(counts),
        n_singletons=sum(1 for s in counts.values() if s == 1),
        block_sizes=dict(counts),
    )


def load_blocks_metadata(path: Path) -> dict | None:
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
    gamma_block: float,
    n_nodes: int,
    source: str | None = None,
) -> bool:
    """Check if cached blocks are valid for the given parameters."""
    meta = load_blocks_metadata(path)
    if meta is None:
        return False

    if float(meta.get("gamma_block", -1)) != gamma_block:
        return False
    if int(meta.get("n_nodes", -1)) != n_nodes:
        return False
    if source and meta.get("source", "") != source:
        return False

    return True


# ── Helpers ───────────────────────────────────────────────────


def _expand_membership(
    contracted_mem: Sequence[int],
    block_mem: Sequence[int],
) -> List[int]:
    """Map contracted-graph membership back to original node indices."""
    return [contracted_mem[block_mem[i]] for i in range(len(block_mem))]


__all__ = [
    "BlockInitResult",
    "CascadeResult",
    "block_init",
    "contract_graph",
    "cascade_search",
    "save_blocks",
    "load_blocks",
    "load_blocks_metadata",
    "is_cache_valid",
]
