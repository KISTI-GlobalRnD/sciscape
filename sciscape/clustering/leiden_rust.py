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
from typing import Any, Sequence

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


@dataclass(frozen=True)
class RustPostprocessResult:
    """Result of Rust postprocessing with round-by-round monitoring."""
    membership: np.ndarray
    n_clusters: int
    changed_at_round: np.ndarray  # empty when change-trace tracking is disabled
    rounds: list  # list of dicts with per-round info


@dataclass(frozen=True)
class RustGraphHandle:
    """Opaque handle for reusing a Rust CSR graph across repeated runs."""

    native: Any
    n_nodes: int
    n_edges: int
    edge_path: Path | None = None
    loaded_from: str = "arrays"


def _ensure_edge_arrays(
    edge_path: Path | None,
    n_nodes: int | None,
    edges_src: np.ndarray | None,
    edges_dst: np.ndarray | None,
    edges_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if edges_src is None:
        if edge_path is None:
            raise ValueError("Provide either edge_path or edges_src/dst/weight")
        df = pl.read_parquet(edge_path)
        edges_src = df["src"].to_numpy().astype(np.uint32)
        edges_dst = df["dst"].to_numpy().astype(np.uint32)
        edges_weight = df["weight"].to_numpy().astype(np.float64)
        if n_nodes is None:
            n_nodes = int(max(edges_src.max(), edges_dst.max())) + 1

    src = np.ascontiguousarray(edges_src, dtype=np.uint32)
    dst = np.ascontiguousarray(edges_dst, dtype=np.uint32)
    weight = np.ascontiguousarray(edges_weight, dtype=np.float64)

    if n_nodes is None:
        n_nodes = int(max(src.max(), dst.max())) + 1
    return src, dst, weight, n_nodes


def _binary_sidecar_paths(edge_path: Path) -> tuple[Path, Path, Path]:
    edge_path = Path(edge_path)
    return (
        edge_path.parent / "src.u32.bin",
        edge_path.parent / "dst.u32.bin",
        edge_path.parent / "weight.f64.bin",
    )


def _infer_n_nodes_for_edge_path(edge_path: Path) -> int | None:
    manifest_path = Path(edge_path).parent / "node_manifest.parquet"
    if not manifest_path.exists():
        return None
    return int(pl.scan_parquet(manifest_path).select(pl.len()).collect().item())


def _ensure_node_weights(node_weights: np.ndarray | None) -> np.ndarray | None:
    if node_weights is None:
        return None
    return np.ascontiguousarray(node_weights, dtype=np.float64)


def _membership_raw_sidecar_path(membership_path: Path) -> Path:
    membership_path = Path(membership_path)
    return membership_path.with_name(f"{membership_path.name}.u64.bin")


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for parquet membership sidecars. "
            "Install pyarrow or use .npy/.u64.bin membership files."
        ) from exc
    return pq


def _split_membership_path_spec(membership_path: Path | str) -> tuple[Path, str | None]:
    raw = str(membership_path)
    if "#" in raw:
        path_str, column = raw.rsplit("#", 1)
        if not column:
            raise ValueError(f"empty membership column specifier: {membership_path}")
        return Path(path_str), column
    return Path(raw), None


def _membership_column_sidecar_path(membership_path: Path, column: str | None) -> Path:
    if column is None or column in {"cluster", "membership"}:
        return _membership_raw_sidecar_path(membership_path)
    return membership_path.with_name(f"{membership_path.name}.{column}.u64.bin")


def _ensure_membership_array(membership: np.ndarray) -> np.ndarray:
    arr = membership if isinstance(membership, np.ndarray) else np.asarray(membership)
    if arr.ndim != 1:
        raise ValueError(f"membership must be 1-dimensional, got shape {arr.shape}")
    if arr.dtype != np.uint64:
        arr = arr.astype(np.uint64, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.uint64)
    return arr


def write_membership_raw_sidecar(
    membership_path: Path,
    membership: np.ndarray,
    *,
    column: str | None = None,
) -> Path:
    membership_path = Path(membership_path)
    sidecar_path = _membership_column_sidecar_path(membership_path, column)
    arr = _ensure_membership_array(membership)
    arr.tofile(sidecar_path)
    return sidecar_path


def _stream_parquet_membership_to_sidecar(
    membership_path: Path,
    sidecar_path: Path,
    *,
    column: str | None = None,
) -> Path:
    pq = _import_pyarrow_parquet()
    parquet = pq.ParquetFile(membership_path)
    names = parquet.schema.names
    if column is not None:
        if column not in names:
            raise ValueError(f"membership parquet missing requested column '{column}': {membership_path}")
        col = column
    else:
        col = next((name for name in ("cluster", "membership") if name in names), None)
        if col is None:
            cluster_cols = [name for name in names if name.startswith("cluster_")]
            if len(cluster_cols) == 1:
                col = cluster_cols[0]
    if col is None:
        raise ValueError(
            f"membership parquet must contain 'cluster'/'membership' or specify a single cluster_* column: {membership_path}"
        )
    with sidecar_path.open("wb") as fh:
        for batch in parquet.iter_batches(columns=[col], batch_size=1_000_000):
            chunk = _ensure_membership_array(batch.column(0).to_numpy())
            chunk.tofile(fh)
    return sidecar_path


def write_membership_sidecars_for_dataframe(
    membership_path: Path,
    frame: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> list[Path]:
    membership_path = Path(membership_path)
    if columns is None:
        columns = [
            col
            for col in frame.columns
            if col in {"cluster", "membership"} or col.startswith("cluster_")
        ]
    written: list[Path] = []
    for col in columns:
        if col not in frame.columns:
            raise ValueError(f"membership column '{col}' not found in frame for {membership_path}")
        written.append(write_membership_raw_sidecar(membership_path, frame[col].to_numpy(), column=col))
    return written


def _load_membership_path(membership_path: Path | str) -> np.ndarray:
    membership_path, column = _split_membership_path_spec(membership_path)
    if membership_path.name.endswith(".u64.bin"):
        size = membership_path.stat().st_size
        if size % 8 != 0:
            raise ValueError(f"u64 binary membership file length not divisible by 8: {membership_path}")
        return np.memmap(membership_path, dtype=np.uint64, mode="r", shape=(size // 8,))
    if membership_path.suffix == ".npy":
        if column is not None:
            raise ValueError(f"column specifier not supported for .npy membership path: {membership_path}")
        return _ensure_membership_array(np.load(membership_path, mmap_mode="r"))
    if membership_path.suffix == ".parquet":
        sidecar_path = _membership_column_sidecar_path(membership_path, column)
        if sidecar_path.exists() and sidecar_path.stat().st_mtime >= membership_path.stat().st_mtime:
            return _load_membership_path(sidecar_path)
        _stream_parquet_membership_to_sidecar(membership_path, sidecar_path, column=column)
        return _load_membership_path(sidecar_path)
    raise ValueError(f"unsupported membership_path suffix: {membership_path.suffix}")


def load_graph_rust(
    edge_path: Path | None = None,
    *,
    n_nodes: int | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    node_weights: np.ndarray | None = None,
) -> RustGraphHandle:
    """Build and keep a Rust CSR graph handle for repeated Leiden runs."""
    _check_available()

    if edge_path is not None and edges_src is None:
        src_path, dst_path, weight_path = _binary_sidecar_paths(Path(edge_path))
        if src_path.exists() and dst_path.exists() and weight_path.exists():
            resolved_n_nodes = n_nodes
            if resolved_n_nodes is None:
                resolved_n_nodes = _infer_n_nodes_for_edge_path(Path(edge_path))
            if resolved_n_nodes is not None:
                native = _rust.load_graph_raw_files(
                    n_nodes=resolved_n_nodes,
                    src_path=str(src_path),
                    dst_path=str(dst_path),
                    weights_path=str(weight_path),
                )
                return RustGraphHandle(
                    native=native,
                    n_nodes=resolved_n_nodes,
                    n_edges=int(native.n_edges),
                    edge_path=Path(edge_path),
                    loaded_from="raw_files",
                )

    src, dst, weight, resolved_n_nodes = _ensure_edge_arrays(
        edge_path, n_nodes, edges_src, edges_dst, edges_weight
    )
    nw = _ensure_node_weights(node_weights)
    native = _rust.load_graph(
        n_nodes=resolved_n_nodes,
        src=src,
        dst=dst,
        weights=weight,
        node_weights=nw,
    )
    return RustGraphHandle(
        native=native,
        n_nodes=resolved_n_nodes,
        n_edges=int(native.n_edges),
        edge_path=Path(edge_path) if edge_path is not None else None,
        loaded_from="edge_path" if edge_path is not None and edges_src is None else "arrays",
    )


def run_leiden_rust_handle(
    handle: RustGraphHandle,
    *,
    resolution: float,
    seed: int = 0,
    n_iterations: int = 10,
    n_starts: int = 1,
    randomness: float = 0.01,
    initial_membership: np.ndarray | None = None,
    initial_membership_path: Path | None = None,
    fixed_nodes: np.ndarray | None = None,
) -> RustLeidenResult:
    """Run Leiden clustering against a preloaded Rust graph handle."""
    _check_available()
    if initial_membership is None and initial_membership_path is not None:
        initial_membership = _load_membership_path(initial_membership_path)
    if initial_membership is not None:
        initial_membership = _ensure_membership_array(initial_membership)
    if fixed_nodes is not None:
        fixed_nodes = np.ascontiguousarray(fixed_nodes, dtype=bool)

    membership, quality, n_clusters = _rust.run_leiden_handle(
        graph=handle.native,
        resolution=resolution,
        n_iterations=n_iterations,
        n_starts=n_starts,
        randomness=randomness,
        seed=seed,
        initial_membership=initial_membership,
        fixed_nodes=fixed_nodes,
    )
    log.info(
        "leiden_rust_handle: %d nodes → %d clusters (γ=%.6g, Q=%.4f)",
        handle.n_nodes, n_clusters, resolution, quality,
    )
    return RustLeidenResult(
        membership=membership,
        quality=quality,
        n_clusters=n_clusters,
    )


def contract_graph_rust_handle(
    handle: RustGraphHandle,
    membership: np.ndarray,
) -> tuple[RustGraphHandle, np.ndarray | None]:
    """Contract a Rust graph handle by membership and return the reduced handle."""
    _check_available()
    mem = np.ascontiguousarray(membership, dtype=np.uint64)
    native, node_weights = _rust.contract_graph_handle(
        graph=handle.native,
        membership=mem,
        materialize_node_weights=False,
    )
    reduced = RustGraphHandle(
        native=native,
        n_nodes=int(native.n_nodes),
        n_edges=int(native.n_edges),
        edge_path=None,
        loaded_from="contracted_handle",
    )
    return reduced, None if node_weights is None else np.asarray(node_weights, dtype=np.float64)


def summarize_membership_rust_handle(
    handle: RustGraphHandle,
    membership: np.ndarray | None = None,
    *,
    membership_path: Path | None = None,
) -> tuple[int, int, float, float]:
    """Return (n_clusters, max_size, max_weight, total_weight) for a membership."""
    _check_available()
    if membership is None:
        if membership_path is None:
            raise ValueError("Provide either membership or membership_path")
        membership = _load_membership_path(membership_path)
    mem = _ensure_membership_array(membership)
    return _rust.summarize_membership_handle(graph=handle.native, membership=mem)


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
    node_weights: np.ndarray | None = None,
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

    handle = load_graph_rust(
        edge_path=edge_path,
        n_nodes=n_nodes,
        edges_src=edges_src,
        edges_dst=edges_dst,
        edges_weight=edges_weight,
        node_weights=node_weights,
    )
    return run_leiden_rust_handle(
        handle,
        resolution=resolution,
        seed=seed,
        n_iterations=n_iterations,
        n_starts=n_starts,
        randomness=randomness,
        initial_membership=initial_membership,
        fixed_nodes=fixed_nodes,
    )


def postprocess_small_clusters_rust_handle(
    *,
    handle: RustGraphHandle,
    resolution: float,
    min_size: int = 0,
    min_weight: float = 0.0,
    membership: np.ndarray | None = None,
    membership_path: Path | None = None,
    seed: int = 0,
    n_iterations: int = 10,
    randomness: float = 0.01,
    max_rounds: int = 5,
    gamma_decay: float = 0.5,
    use_greedy: bool = True,
    greedy_anchor_only: bool = False,
    greedy_fallback_to_small: bool = False,
    greedy_max_weight: float = 0.0,
    use_component_merge: bool = True,
    component_max_weight: float = 0.0,
    track_changed_rounds: bool = True,
) -> RustPostprocessResult:
    """Postprocess against a preloaded Rust graph handle."""
    _check_available()
    if membership is None:
        if membership_path is None:
            raise ValueError("Provide either membership or membership_path")
        membership = _load_membership_path(membership_path)
    membership = _ensure_membership_array(membership)

    result_mem, n_clusters, changed_at, rounds = _rust.run_postprocess_handle(
        graph=handle.native,
        membership=membership,
        resolution=resolution,
        min_size=min_size,
        n_iterations=n_iterations,
        randomness=randomness,
        seed=seed,
        min_weight=min_weight,
        max_rounds=int(max_rounds),
        gamma_decay=float(gamma_decay),
        use_greedy=bool(use_greedy),
        greedy_anchor_only=bool(greedy_anchor_only),
        greedy_fallback_to_small=bool(greedy_fallback_to_small),
        greedy_max_weight=float(greedy_max_weight),
        use_component_merge=bool(use_component_merge),
        component_max_weight=float(component_max_weight),
        track_changed_rounds=bool(track_changed_rounds),
    )

    changed = int(np.sum(result_mem != membership))
    threshold_str = (
        f"min_weight={min_weight}" if min_weight > 0
        else f"min_size={min_size}"
    )
    for r in rounds:
        log.info(
            "postprocess_handle round %d: γ=%.4e, method=%s, small: %d→%d, "
            "merged: %d, total: %d, max_size: %d, max_weight: %.1f",
            r["round"], r["gamma"], r["method"],
            r["n_small_before"], r["n_small_after"],
            r["n_merged"], r["n_total_clusters"], r["max_cluster_size"],
            r["max_cluster_weight"],
        )
    log.info(
        "postprocess_rust_handle: %d nodes changed, %d clusters (%s, %d rounds)",
        changed, n_clusters, threshold_str, len(rounds),
    )

    return RustPostprocessResult(
        membership=result_mem,
        n_clusters=n_clusters,
        changed_at_round=changed_at,
        rounds=rounds,
    )


def postprocess_small_clusters_rust(
    *,
    resolution: float,
    min_size: int = 0,
    min_weight: float = 0.0,
    membership: np.ndarray,
    n_nodes: int | None = None,
    edge_path: Path | None = None,
    edges_src: np.ndarray | None = None,
    edges_dst: np.ndarray | None = None,
    edges_weight: np.ndarray | None = None,
    node_weights: np.ndarray | None = None,
    seed: int = 0,
    n_iterations: int = 10,
    randomness: float = 0.01,
    max_rounds: int = 5,
    gamma_decay: float = 0.5,
    use_greedy: bool = True,
    greedy_anchor_only: bool = False,
    greedy_fallback_to_small: bool = False,
    greedy_max_weight: float = 0.0,
    use_component_merge: bool = True,
    component_max_weight: float = 0.0,
    track_changed_rounds: bool = True,
) -> RustPostprocessResult:
    """Reassign small clusters using constrained Leiden (Rust backend).

    Threshold semantics:
    - If ``node_weights`` is provided and ``min_weight > 0``, clusters are
      considered "small" when their total node_weight < min_weight (doc_count).
    - Otherwise, raw node count < min_size is used.
    """
    _check_available()

    handle = load_graph_rust(
        edge_path=edge_path,
        n_nodes=n_nodes,
        edges_src=edges_src,
        edges_dst=edges_dst,
        edges_weight=edges_weight,
        node_weights=node_weights,
    )
    return postprocess_small_clusters_rust_handle(
        handle=handle,
        resolution=resolution,
        min_size=min_size,
        min_weight=min_weight,
        membership=membership,
        seed=seed,
        n_iterations=n_iterations,
        randomness=randomness,
        max_rounds=max_rounds,
        gamma_decay=gamma_decay,
        use_greedy=use_greedy,
        greedy_anchor_only=greedy_anchor_only,
        greedy_fallback_to_small=greedy_fallback_to_small,
        greedy_max_weight=greedy_max_weight,
        use_component_merge=use_component_merge,
        component_max_weight=component_max_weight,
        track_changed_rounds=track_changed_rounds,
    )


__all__ = [
    "RUST_AVAILABLE",
    "RustGraphHandle",
    "RustLeidenResult",
    "RustPostprocessResult",
    "contract_graph_rust_handle",
    "load_graph_rust",
    "summarize_membership_rust_handle",
    "run_leiden_rust_handle",
    "run_leiden_rust",
    "postprocess_small_clusters_rust_handle",
    "postprocess_small_clusters_rust",
]
