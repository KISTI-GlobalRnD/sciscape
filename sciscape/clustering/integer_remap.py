"""String-UID to integer remapping for large-scale graph pipelines.

Converts parquet edge tables with string UIDs into integer-indexed
edges plus a node manifest. The output is cached on disk to avoid
repeated remapping.

This is the prerequisite for the Java Leiden backend which requires
0-indexed integer edge lists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemapResult:
    """Result of string-to-integer UID remapping."""

    n_nodes: int
    n_edges: int
    node_manifest_path: Path
    int_edges_path: Path


def integer_remap(
    edges: pl.DataFrame | Path,
    output_dir: Path,
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
    overwrite: bool = False,
) -> RemapResult:
    """Remap string UIDs to 0-indexed integers and save to parquet.

    Parameters
    ----------
    edges : pl.DataFrame or Path
        Edge table or path to parquet file.
    output_dir : Path
        Directory to write ``node_manifest.parquet`` and ``int_edges.parquet``.
    uid1_col, uid2_col, weight_col : str
        Column names in the edge table.
    overwrite : bool
        If False and output files exist, skip remapping.

    Returns
    -------
    RemapResult
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "node_manifest.parquet"
    int_edges_path = output_dir / "int_edges.parquet"

    # Check cache
    if not overwrite and manifest_path.exists() and int_edges_path.exists():
        manifest = pl.read_parquet(manifest_path)
        n_edges = pl.scan_parquet(int_edges_path).select(pl.len()).collect().item()
        log.info("integer_remap: using cached files in %s", output_dir)
        return RemapResult(
            n_nodes=manifest.height,
            n_edges=n_edges,
            node_manifest_path=manifest_path,
            int_edges_path=int_edges_path,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load edges
    if isinstance(edges, (str, Path)):
        edges = pl.read_parquet(edges)

    log.info("integer_remap: %d edges, building node manifest...", edges.height)

    # Use Polars Categorical encoding for fast string → int mapping.
    # StringCache ensures uid1 and uid2 share the same integer codes,
    # eliminating the need for separate unique+sort+join steps (~1.5x faster).
    with pl.StringCache():
        edges_cat = edges.with_columns(
            pl.col(uid1_col).cast(pl.Categorical).alias("_uid1_cat"),
            pl.col(uid2_col).cast(pl.Categorical).alias("_uid2_cat"),
        )
        src = edges_cat["_uid1_cat"].to_physical().cast(pl.Int32)
        dst = edges_cat["_uid2_cat"].to_physical().cast(pl.Int32)
        categories = edges_cat["_uid1_cat"].cat.get_categories()

    n_nodes = categories.len()
    manifest = pl.DataFrame({
        "node_idx": np.arange(n_nodes, dtype=np.int32),
        "uid": categories,
    })
    manifest.write_parquet(manifest_path, compression="zstd")
    log.info("integer_remap: %d unique nodes → %s", n_nodes, manifest_path)

    int_edges = pl.DataFrame({
        "src": src,
        "dst": dst,
        "weight": edges[weight_col].cast(pl.Float64),
    })

    int_edges.write_parquet(int_edges_path, compression="zstd")
    n_edges = int_edges.height
    log.info("integer_remap: %d int edges → %s", n_edges, int_edges_path)

    return RemapResult(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_manifest_path=manifest_path,
        int_edges_path=int_edges_path,
    )


def load_manifest(path: Path) -> pl.DataFrame:
    """Load a node manifest parquet (node_idx, uid)."""
    return pl.read_parquet(path)


def join_back_uids(
    membership: np.ndarray | list[int],
    manifest: pl.DataFrame | Path,
) -> pl.DataFrame:
    """Join integer membership back to string UIDs.

    Parameters
    ----------
    membership : array-like
        Cluster assignment per node_idx (length = n_nodes).
    manifest : pl.DataFrame or Path
        Node manifest with ``node_idx`` and ``uid`` columns.

    Returns
    -------
    pl.DataFrame
        Columns: ``uid``, ``cluster``.
    """
    if isinstance(manifest, (str, Path)):
        manifest = pl.read_parquet(manifest)

    mem = np.asarray(membership, dtype=np.int32)
    return manifest.with_columns(
        pl.Series("cluster", mem),
    ).select("uid", "cluster")


def integer_remap_memory(
    edges: pl.DataFrame,
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
) -> tuple:
    """In-memory integer remap (no disk I/O).

    Returns (src, dst, weight, n_nodes, uids) where:
    - src, dst: numpy uint32 arrays
    - weight: numpy float64 array
    - n_nodes: int
    - uids: list of str (index → uid mapping)
    """
    with pl.StringCache():
        cats = edges.with_columns(
            pl.col(uid1_col).cast(pl.Categorical).alias("_c1"),
            pl.col(uid2_col).cast(pl.Categorical).alias("_c2"),
        )
        src = cats["_c1"].to_physical().to_numpy().astype(np.uint32)
        dst = cats["_c2"].to_physical().to_numpy().astype(np.uint32)
        categories = cats["_c1"].cat.get_categories()

    n_nodes = categories.len()
    uids = categories.to_list()
    w = edges[weight_col].to_numpy().astype(np.float64)
    log.info("integer_remap_memory: %d edges, %d nodes (no disk I/O)", len(w), n_nodes)
    return src, dst, w, n_nodes, uids


__all__ = ["RemapResult", "integer_remap", "integer_remap_memory", "join_back_uids", "load_manifest"]
