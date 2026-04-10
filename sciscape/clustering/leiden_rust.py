"""Rust backend wrapper for Leiden clustering via sciscape-leiden.

Drop-in replacement for leiden_java.py functions using the Rust
native module. Much faster than Java (no JVM startup, no file I/O)
and no JDK dependency.

Requires::

    pip install sciscape-leiden

Or build from source::

    cd sciscape-leiden && maturin develop --release
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

try:
    import sciscape_leiden as _rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _check_available():
    if not RUST_AVAILABLE:
        raise ImportError(
            "sciscape-leiden Rust module not installed. "
            "Install with: pip install sciscape-leiden "
            "Or build: cd sciscape-leiden && maturin develop --release"
        )


@dataclass(frozen=True)
class RustLeidenResult:
    """Result of a Rust Leiden clustering run."""
    membership: np.ndarray
    quality: float
    n_clusters: int


def run_leiden_rust(
    edge_path: Path | None = None,
    *,
    resolution: float,
    n_nodes: int | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    seed: int = 0,
    n_iterations: int = 10,
    n_starts: int = 1,
    randomness: float = 0.01,
    initial_membership: np.ndarray | None = None,
    fixed_nodes: np.ndarray | None = None,
) -> RustLeidenResult:
    """Run Leiden clustering via the Rust backend.

    Accepts either file path (parquet with src/dst/weight columns)
    or pre-loaded numpy arrays.

    Parameters
    ----------
    edge_path : Path, optional
        Path to int_edges.parquet (columns: src, dst, weight).
    resolution : float
        CPM resolution parameter.
    n_nodes : int, optional
        Total number of nodes. Required if using edge_path.
    edges_src, edges_dst, edges_weight : numpy arrays, optional
        Pre-loaded edge arrays. Alternative to edge_path.
    seed, n_iterations, n_starts, randomness
        Leiden parameters.
    initial_membership : numpy array, optional
        Initial cluster assignment (uint64).
    fixed_nodes : numpy array, optional
        Boolean mask of nodes that cannot change cluster.

    Returns
    -------
    RustLeidenResult
    """
    _check_available()

    # Load from parquet if needed
    if edges_src is None:
        if edge_path is None:
            raise ValueError("Provide either edge_path or edges_src/dst/weight")
        df = pl.read_parquet(edge_path)
        edges_src = df["src"].to_numpy().astype(np.uint32)
        edges_dst = df["dst"].to_numpy().astype(np.uint32)
        edges_weight = df["weight"].to_numpy().astype(np.float64)
        if n_nodes is None:
            n_nodes = int(max(edges_src.max(), edges_dst.max())) + 1

    edges_src = np.ascontiguousarray(edges_src, dtype=np.uint32)
    edges_dst = np.ascontiguousarray(edges_dst, dtype=np.uint32)
    edges_weight = np.ascontiguousarray(edges_weight, dtype=np.float64)

    if n_nodes is None:
        n_nodes = int(max(edges_src.max(), edges_dst.max())) + 1

    membership, quality, n_clusters = _rust.run_leiden(
        n_nodes=n_nodes,
        src=edges_src,
        dst=edges_dst,
        weights=edges_weight,
        resolution=resolution,
        n_iterations=n_iterations,
        n_starts=n_starts,
        randomness=randomness,
        seed=seed,
        initial_membership=initial_membership,
        fixed_nodes=fixed_nodes,
    )

    log.info(
        "leiden_rust: %d nodes → %d clusters (γ=%.6g, Q=%.4f)",
        n_nodes, n_clusters, resolution, quality,
    )

    return RustLeidenResult(
        membership=membership,
        quality=quality,
        n_clusters=n_clusters,
    )


def postprocess_small_clusters_rust(
    *,
    resolution: float,
    min_size: int,
    membership: np.ndarray,
    n_nodes: int | None = None,
    edge_path: Path | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    seed: int = 0,
    n_iterations: int = 10,
    randomness: float = 0.01,
) -> RustLeidenResult:
    """Reassign small clusters using constrained Leiden (Rust backend)."""
    _check_available()

    if edges_src is None:
        if edge_path is None:
            raise ValueError("Provide either edge_path or edges_src/dst/weight")
        df = pl.read_parquet(edge_path)
        edges_src = df["src"].to_numpy().astype(np.uint32)
        edges_dst = df["dst"].to_numpy().astype(np.uint32)
        edges_weight = df["weight"].to_numpy().astype(np.float64)
        if n_nodes is None:
            n_nodes = int(max(edges_src.max(), edges_dst.max())) + 1

    edges_src = np.ascontiguousarray(edges_src, dtype=np.uint32)
    edges_dst = np.ascontiguousarray(edges_dst, dtype=np.uint32)
    edges_weight = np.ascontiguousarray(edges_weight, dtype=np.float64)
    membership = np.ascontiguousarray(membership, dtype=np.uint64)

    if n_nodes is None:
        n_nodes = len(membership)

    result_mem, n_clusters = _rust.run_postprocess(
        n_nodes=n_nodes,
        src=edges_src,
        dst=edges_dst,
        weights=edges_weight,
        membership=membership,
        resolution=resolution,
        min_size=min_size,
        n_iterations=n_iterations,
        randomness=randomness,
        seed=seed,
    )

    changed = int(np.sum(result_mem != membership))
    log.info(
        "postprocess_rust: %d nodes changed, %d clusters (min_size=%d)",
        changed, n_clusters, min_size,
    )

    return RustLeidenResult(
        membership=result_mem,
        quality=0.0,  # not computed in postprocess
        n_clusters=n_clusters,
    )


__all__ = [
    "RUST_AVAILABLE",
    "RustLeidenResult",
    "run_leiden_rust",
    "postprocess_small_clusters_rust",
]
