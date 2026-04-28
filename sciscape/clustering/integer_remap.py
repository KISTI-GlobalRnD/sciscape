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
from uuid import uuid4

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

_PARQUET_ROW_GROUP_SIZE = 1_000_000
_PARQUET_SIDECAR_BATCH_SIZE = 1_000_000


@dataclass(frozen=True)
class RemapResult:
    """Result of string-to-integer UID remapping."""

    n_nodes: int
    n_edges: int
    node_manifest_path: Path
    int_edges_path: Path
    src_path: Path | None = None
    dst_path: Path | None = None
    weight_path: Path | None = None

    @property
    def sidecar_paths(self) -> tuple[Path, Path, Path]:
        """Raw u32/u32/f64 sidecar paths for this remap result."""
        if self.src_path is not None and self.dst_path is not None and self.weight_path is not None:
            return self.src_path, self.dst_path, self.weight_path
        return int_edge_sidecar_paths(self.int_edges_path)


def int_edge_sidecar_paths(int_edges_path: Path) -> tuple[Path, Path, Path]:
    """Return raw sidecar paths for an ``int_edges.parquet`` file."""
    int_edges_path = Path(int_edges_path)
    return (
        int_edges_path.parent / "src.u32.bin",
        int_edges_path.parent / "dst.u32.bin",
        int_edges_path.parent / "weight.f64.bin",
    )


def _valid_int_edge_sidecars(src_path: Path, dst_path: Path, weight_path: Path) -> bool:
    if not (src_path.exists() and dst_path.exists() and weight_path.exists()):
        return False
    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    weight_size = weight_path.stat().st_size
    if src_size % np.dtype(np.uint32).itemsize != 0:
        return False
    if dst_size % np.dtype(np.uint32).itemsize != 0:
        return False
    if weight_size % np.dtype(np.float64).itemsize != 0:
        return False
    return (
        src_size // np.dtype(np.uint32).itemsize
        == dst_size // np.dtype(np.uint32).itemsize
        == weight_size // np.dtype(np.float64).itemsize
    )


def _sidecar_edge_count(weight_path: Path) -> int:
    return weight_path.stat().st_size // np.dtype(np.float64).itemsize


def write_int_edge_sidecars(
    int_edges: pl.DataFrame,
    int_edges_path: Path,
) -> tuple[Path, Path, Path]:
    """Write raw u32/u32/f64 edge sidecars next to ``int_edges_path``."""
    return write_int_edge_sidecars_from_series(
        int_edges["src"],
        int_edges["dst"],
        int_edges["weight"],
        int_edges_path,
    )


def _series_to_numpy_for_sidecar(series: pl.Series, dtype: np.dtype) -> np.ndarray:
    """Return a contiguous numpy view/copy suitable for raw sidecar writing."""
    rechunked = series.rechunk()
    try:
        arr = rechunked.to_numpy(allow_copy=False)
    except RuntimeError:
        arr = rechunked.to_numpy(allow_copy=True)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr, dtype=dtype)


def write_int_edge_sidecars_from_series(
    src: pl.Series,
    dst: pl.Series,
    weight: pl.Series,
    int_edges_path: Path,
) -> tuple[Path, Path, Path]:
    """Write raw u32/u32/f64 sidecars from numeric Polars series."""
    src_path, dst_path, weight_path = int_edge_sidecar_paths(int_edges_path)
    _series_to_numpy_for_sidecar(src, np.dtype(np.uint32)).tofile(src_path)
    _series_to_numpy_for_sidecar(dst, np.dtype(np.uint32)).tofile(dst_path)
    _series_to_numpy_for_sidecar(weight, np.dtype(np.float64)).tofile(weight_path)
    return src_path, dst_path, weight_path


def write_int_edge_sidecars_from_parquet(
    int_edges_path: Path,
    *,
    batch_size: int = _PARQUET_SIDECAR_BATCH_SIZE,
) -> tuple[Path, Path, Path]:
    """Write raw u32/u32/f64 sidecars from int edge parquet in batches."""
    import pyarrow.parquet as pq

    src_path, dst_path, weight_path = int_edge_sidecar_paths(int_edges_path)
    parquet = pq.ParquetFile(int_edges_path)
    with (
        src_path.open("wb") as src_fh,
        dst_path.open("wb") as dst_fh,
        weight_path.open("wb") as weight_fh,
    ):
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["src", "dst", "weight"],
        ):
            src = np.ascontiguousarray(
                batch.column("src").to_numpy(zero_copy_only=False),
                dtype=np.uint32,
            )
            dst = np.ascontiguousarray(
                batch.column("dst").to_numpy(zero_copy_only=False),
                dtype=np.uint32,
            )
            weight = np.ascontiguousarray(
                batch.column("weight").to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            src.tofile(src_fh)
            dst.tofile(dst_fh)
            weight.tofile(weight_fh)
    return src_path, dst_path, weight_path


def ensure_int_edge_sidecars(int_edges_path: Path) -> tuple[Path, Path, Path]:
    """Materialize raw edge sidecars used by the Rust backend.

    The sidecars avoid a second full parquet -> Polars -> numpy conversion when
    building the Rust CSR graph for large clustering runs.
    """
    int_edges_path = Path(int_edges_path)
    src_path, dst_path, weight_path = int_edge_sidecar_paths(int_edges_path)
    if not int_edges_path.exists():
        if _valid_int_edge_sidecars(src_path, dst_path, weight_path):
            return src_path, dst_path, weight_path
        raise FileNotFoundError(
            f"{int_edges_path} does not exist and valid raw sidecars were not found"
        )
    if (
        _valid_int_edge_sidecars(src_path, dst_path, weight_path)
        and src_path.stat().st_mtime >= int_edges_path.stat().st_mtime
        and dst_path.stat().st_mtime >= int_edges_path.stat().st_mtime
        and weight_path.stat().st_mtime >= int_edges_path.stat().st_mtime
    ):
        return src_path, dst_path, weight_path

    return write_int_edge_sidecars_from_parquet(int_edges_path)


def _integer_remap_parquet_lazy(
    edge_path: Path,
    output_dir: Path,
    *,
    uid1_col: str,
    uid2_col: str,
    weight_col: str,
) -> RemapResult:
    """Remap a parquet edge table using Polars lazy execution."""
    manifest_path = output_dir / "node_manifest.parquet"
    int_edges_path = output_dir / "int_edges.parquet"

    edges_lf = pl.scan_parquet(edge_path).select([uid1_col, uid2_col, weight_col])
    uid_lf = pl.concat(
        [
            edges_lf.select(pl.col(uid1_col).alias("uid")),
            edges_lf.select(pl.col(uid2_col).alias("uid")),
        ],
        how="vertical",
    )
    (
        uid_lf
        .unique()
        .sort("uid")
        .with_row_index("node_idx")
        .with_columns(pl.col("node_idx").cast(pl.UInt32))
        .select(["node_idx", "uid"])
        .sink_parquet(
            manifest_path,
            compression="zstd",
            row_group_size=_PARQUET_ROW_GROUP_SIZE,
        )
    )

    n_nodes = int(pl.scan_parquet(manifest_path).select(pl.len()).collect().item())
    if n_nodes > np.iinfo(np.uint32).max + 1:
        raise ValueError(
            f"integer_remap supports at most {np.iinfo(np.uint32).max + 1} nodes, "
            f"got {n_nodes}"
        )

    manifest_lf = pl.scan_parquet(manifest_path)
    src_map = manifest_lf.select(
        pl.col("uid").alias(uid1_col),
        pl.col("node_idx").alias("src"),
    )
    dst_map = manifest_lf.select(
        pl.col("uid").alias(uid2_col),
        pl.col("node_idx").alias("dst"),
    )
    (
        edges_lf
        .join(src_map, on=uid1_col, how="left")
        .join(dst_map, on=uid2_col, how="left")
        .select(
            pl.col("src").cast(pl.UInt32),
            pl.col("dst").cast(pl.UInt32),
            pl.col(weight_col).cast(pl.Float64).alias("weight"),
        )
        .sink_parquet(
            int_edges_path,
            compression="zstd",
            row_group_size=_PARQUET_ROW_GROUP_SIZE,
        )
    )

    n_edges = int(pl.scan_parquet(int_edges_path).select(pl.len()).collect().item())
    src_path, dst_path, weight_path = write_int_edge_sidecars_from_parquet(int_edges_path)
    log.info("integer_remap: %d unique nodes → %s", n_nodes, manifest_path)
    log.info("integer_remap: %d int edges → %s", n_edges, int_edges_path)
    return RemapResult(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_manifest_path=manifest_path,
        int_edges_path=int_edges_path,
        src_path=src_path,
        dst_path=dst_path,
        weight_path=weight_path,
    )


def _integer_remap_parquet_rust(
    edge_path: Path,
    output_dir: Path,
    *,
    uid1_col: str,
    uid2_col: str,
    weight_col: str,
    write_int_edges: bool = True,
) -> RemapResult | None:
    """Use the Rust parquet remapper when the installed extension exposes it."""
    try:
        import sciscape_leiden as _rust
    except ImportError:
        return None

    remap_fn_name = (
        "rust_integer_remap_parquet"
        if write_int_edges
        else "rust_integer_remap_parquet_sidecars"
    )
    remap_fn = getattr(_rust, remap_fn_name, None)
    if remap_fn is None:
        return None

    result = remap_fn(
        str(edge_path),
        str(output_dir),
        uid1_col,
        uid2_col,
        weight_col,
    )
    if len(result) == 4:
        n_nodes, n_edges, manifest_path, int_edges_path = result
        src_path, dst_path, weight_path = int_edge_sidecar_paths(Path(int_edges_path))
    else:
        n_nodes, n_edges, manifest_path, int_edges_path, src_path, dst_path, weight_path = result
    return RemapResult(
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
        node_manifest_path=Path(manifest_path),
        int_edges_path=Path(int_edges_path),
        src_path=Path(src_path),
        dst_path=Path(dst_path),
        weight_path=Path(weight_path),
    )


def _integer_remap_dataframe_rust(
    edges: pl.DataFrame,
    output_dir: Path,
    *,
    uid1_col: str,
    uid2_col: str,
    weight_col: str,
    write_int_edges: bool = True,
) -> RemapResult | None:
    """Use the Rust parquet remapper for an in-memory edge DataFrame."""
    try:
        import sciscape_leiden as _rust
    except ImportError:
        return None
    remap_fn_name = (
        "rust_integer_remap_parquet"
        if write_int_edges
        else "rust_integer_remap_parquet_sidecars"
    )
    if getattr(_rust, remap_fn_name, None) is None:
        return None

    tmp_path = output_dir / f"_rust_remap_input_{uuid4().hex}.parquet"
    try:
        edges.select([uid1_col, uid2_col, weight_col]).write_parquet(
            tmp_path,
            compression="zstd",
            row_group_size=_PARQUET_ROW_GROUP_SIZE,
        )
        return _integer_remap_parquet_rust(
            tmp_path,
            output_dir,
            uid1_col=uid1_col,
            uid2_col=uid2_col,
            weight_col=weight_col,
            write_int_edges=write_int_edges,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def integer_remap(
    edges: pl.DataFrame | Path,
    output_dir: Path,
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
    overwrite: bool = False,
    write_int_edges: bool = True,
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
    src_path, dst_path, weight_path = int_edge_sidecar_paths(int_edges_path)

    # Check cache
    cache_has_edges = int_edges_path.exists() if write_int_edges else _valid_int_edge_sidecars(
        src_path,
        dst_path,
        weight_path,
    )
    if not overwrite and manifest_path.exists() and cache_has_edges:
        n_nodes = pl.scan_parquet(manifest_path).select(pl.len()).collect().item()
        if int_edges_path.exists():
            n_edges = pl.scan_parquet(int_edges_path).select(pl.len()).collect().item()
        else:
            n_edges = _sidecar_edge_count(weight_path)
        log.info("integer_remap: using cached files in %s", output_dir)
        return RemapResult(
            n_nodes=n_nodes,
            n_edges=n_edges,
            node_manifest_path=manifest_path,
            int_edges_path=int_edges_path,
            src_path=src_path,
            dst_path=dst_path,
            weight_path=weight_path,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(edges, (str, Path)) and Path(edges).suffix.lower() == ".parquet":
        log.info(
            "integer_remap: using rust parquet %sremap for %s",
            "sidecar-only " if not write_int_edges else "",
            edges,
        )
        try:
            rust_result = _integer_remap_parquet_rust(
                Path(edges),
                output_dir,
                uid1_col=uid1_col,
                uid2_col=uid2_col,
                weight_col=weight_col,
                write_int_edges=write_int_edges,
            )
        except Exception as exc:
            log.warning(
                "integer_remap: rust parquet remap failed; falling back to "
                "Polars lazy remap: %s",
                exc,
            )
        else:
            if rust_result is not None:
                log.info(
                    "integer_remap: %d unique nodes → %s",
                    rust_result.n_nodes,
                    rust_result.node_manifest_path,
                )
                log.info(
                    "integer_remap: %d int edges → %s",
                    rust_result.n_edges,
                    rust_result.int_edges_path,
                )
                return rust_result

        log.info("integer_remap: using Polars lazy parquet remap for %s", edges)
        return _integer_remap_parquet_lazy(
            Path(edges),
            output_dir,
            uid1_col=uid1_col,
            uid2_col=uid2_col,
            weight_col=weight_col,
        )

    if isinstance(edges, pl.DataFrame):
        log.info(
            "integer_remap: using rust dataframe %sremap via temporary parquet",
            "sidecar-only " if not write_int_edges else "",
        )
        try:
            rust_result = _integer_remap_dataframe_rust(
                edges,
                output_dir,
                uid1_col=uid1_col,
                uid2_col=uid2_col,
                weight_col=weight_col,
                write_int_edges=write_int_edges,
            )
        except Exception as exc:
            log.warning(
                "integer_remap: rust dataframe remap failed; falling back to "
                "Polars in-memory remap: %s",
                exc,
            )
        else:
            if rust_result is not None:
                log.info(
                    "integer_remap: %d unique nodes → %s",
                    rust_result.n_nodes,
                    rust_result.node_manifest_path,
                )
                log.info(
                    "integer_remap: %d int edges → %s",
                    rust_result.n_edges,
                    rust_result.int_edges_path,
                )
                return rust_result

    # Load only the columns needed for remapping. In the normal Rust/Java
    # large-graph path this can be a parquet path, avoiding an eager full-edge
    # load in run_pipeline.
    if isinstance(edges, (str, Path)):
        edges = pl.read_parquet(
            edges,
            columns=[uid1_col, uid2_col, weight_col],
        )
    else:
        edges = edges.select([uid1_col, uid2_col, weight_col])

    log.info("integer_remap: %d edges, building node manifest...", edges.height)

    # Use Polars Categorical encoding for fast string → int mapping.
    # StringCache ensures uid1 and uid2 share the same integer codes,
    # eliminating the need for separate unique+sort+join steps (~1.5x faster).
    with pl.StringCache():
        uid1_cat = edges[uid1_col].cast(pl.Categorical)
        uid2_cat = edges[uid2_col].cast(pl.Categorical)
        src = uid1_cat.to_physical().cast(pl.UInt32)
        dst = uid2_cat.to_physical().cast(pl.UInt32)
        categories = uid1_cat.cat.get_categories()

    n_nodes = categories.len()
    if n_nodes > np.iinfo(np.uint32).max + 1:
        raise ValueError(
            f"integer_remap supports at most {np.iinfo(np.uint32).max + 1} nodes, "
            f"got {n_nodes}"
        )
    manifest = pl.DataFrame({
        "node_idx": np.arange(n_nodes, dtype=np.uint32),
        "uid": categories,
    })
    manifest.write_parquet(manifest_path, compression="zstd")
    log.info("integer_remap: %d unique nodes → %s", n_nodes, manifest_path)

    weight = edges[weight_col].cast(pl.Float64)
    int_edges = pl.DataFrame({
        "src": src,
        "dst": dst,
        "weight": weight,
    })

    int_edges.write_parquet(int_edges_path, compression="zstd")
    src_path, dst_path, weight_path = write_int_edge_sidecars_from_series(src, dst, weight, int_edges_path)
    n_edges = int_edges.height
    log.info("integer_remap: %d int edges → %s", n_edges, int_edges_path)

    return RemapResult(
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_manifest_path=manifest_path,
        int_edges_path=int_edges_path,
        src_path=src_path,
        dst_path=dst_path,
        weight_path=weight_path,
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


__all__ = [
    "RemapResult",
    "ensure_int_edge_sidecars",
    "int_edge_sidecar_paths",
    "integer_remap",
    "integer_remap_memory",
    "join_back_uids",
    "load_manifest",
    "write_int_edge_sidecars",
    "write_int_edge_sidecars_from_parquet",
    "write_int_edge_sidecars_from_series",
]
