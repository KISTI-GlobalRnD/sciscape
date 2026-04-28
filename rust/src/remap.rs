//! Rust parquet remapping for string UID edge lists.

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ahash::AHashMap;
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow_array::{LargeStringArray, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use crate::Graph;

const BATCH_SIZE: usize = 1_000_000;

type UidMap = AHashMap<Arc<str>, u32>;

#[derive(Debug, Clone)]
pub struct RemapOutput {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub node_manifest_path: PathBuf,
    pub int_edges_path: PathBuf,
    pub src_path: PathBuf,
    pub dst_path: PathBuf,
    pub weight_path: PathBuf,
}

#[derive(Debug)]
pub struct RemapGraphOutput {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub node_manifest_path: PathBuf,
    pub int_edges_path: PathBuf,
    pub graph: Graph,
}

fn parquet_err<E: std::fmt::Display>(context: &str, err: E) -> String {
    format!("{}: {}", context, err)
}

fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build()
}

fn record_reader(path: &Path, columns: &[&str]) -> Result<ParquetRecordBatchReader, String> {
    let file = File::open(path).map_err(|e| parquet_err("open parquet", e))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| parquet_err("read parquet metadata", e))?;
    let arrow_schema = builder.schema().clone();
    let parquet_schema = builder.metadata().file_metadata().schema_descr();
    let mut indices = Vec::with_capacity(columns.len());
    for column in columns {
        indices.push(
            arrow_schema
                .index_of(column)
                .map_err(|_| format!("parquet column not found: {}", column))?,
        );
    }
    let projection = ProjectionMask::roots(parquet_schema, indices);
    builder
        .with_batch_size(BATCH_SIZE)
        .with_projection(projection)
        .build()
        .map_err(|e| parquet_err("build parquet batch reader", e))
}

fn column(batch: &RecordBatch, name: &str) -> Result<ArrayRef, String> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("record batch column not found: {}", name))?;
    Ok(batch.column(idx).clone())
}

enum StringColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl<'a> StringColumn<'a> {
    fn value(&self, row: usize, column: &str) -> Result<&'a str, String> {
        match self {
            Self::Utf8(values) => {
                if values.is_null(row) {
                    return Err(format!("null UID in column {} at row {}", column, row));
                }
                Ok(values.value(row))
            }
            Self::LargeUtf8(values) => {
                if values.is_null(row) {
                    return Err(format!("null UID in column {} at row {}", column, row));
                }
                Ok(values.value(row))
            }
        }
    }
}

fn string_column<'a>(array: &'a ArrayRef, column: &str) -> Result<StringColumn<'a>, String> {
    match array.data_type() {
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| format!("failed to read {} as Utf8", column))?;
            Ok(StringColumn::Utf8(values))
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| format!("failed to read {} as LargeUtf8", column))?;
            Ok(StringColumn::LargeUtf8(values))
        }
        other => Err(format!(
            "unsupported UID column type for {}: {:?}",
            column, other
        )),
    }
}

enum WeightColumn<'a> {
    Float64(&'a Float64Array),
    Float32(&'a Float32Array),
    Int64(&'a Int64Array),
    Int32(&'a Int32Array),
}

impl WeightColumn<'_> {
    fn value(&self, row: usize, column: &str) -> Result<f64, String> {
        match self {
            Self::Float64(values) => {
                if values.is_null(row) {
                    return Err(format!("null weight in column {} at row {}", column, row));
                }
                Ok(values.value(row))
            }
            Self::Float32(values) => {
                if values.is_null(row) {
                    return Err(format!("null weight in column {} at row {}", column, row));
                }
                Ok(values.value(row) as f64)
            }
            Self::Int64(values) => {
                if values.is_null(row) {
                    return Err(format!("null weight in column {} at row {}", column, row));
                }
                Ok(values.value(row) as f64)
            }
            Self::Int32(values) => {
                if values.is_null(row) {
                    return Err(format!("null weight in column {} at row {}", column, row));
                }
                Ok(values.value(row) as f64)
            }
        }
    }
}

fn weight_column<'a>(array: &'a ArrayRef, column: &str) -> Result<WeightColumn<'a>, String> {
    match array.data_type() {
        DataType::Float64 => {
            let values = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| format!("failed to read {} as Float64", column))?;
            Ok(WeightColumn::Float64(values))
        }
        DataType::Float32 => {
            let values = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| format!("failed to read {} as Float32", column))?;
            Ok(WeightColumn::Float32(values))
        }
        DataType::Int64 => {
            let values = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| format!("failed to read {} as Int64", column))?;
            Ok(WeightColumn::Int64(values))
        }
        DataType::Int32 => {
            let values = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| format!("failed to read {} as Int32", column))?;
            Ok(WeightColumn::Int32(values))
        }
        other => Err(format!(
            "unsupported weight column type for {}: {:?}",
            column, other
        )),
    }
}

fn intern_uid(uid: &str, uid_to_idx: &mut UidMap, uids: &mut Vec<Arc<str>>) -> Result<u32, String> {
    if let Some(&idx) = uid_to_idx.get(uid) {
        return Ok(idx);
    }
    if uids.len() > u32::MAX as usize {
        return Err(format!(
            "integer remap supports at most {} nodes",
            u32::MAX as u64 + 1
        ));
    }
    let idx = uids.len() as u32;
    let owned: Arc<str> = Arc::from(uid);
    uids.push(owned.clone());
    uid_to_idx.insert(owned, idx);
    Ok(idx)
}

fn intern_uid_with_degree(
    uid: &str,
    uid_to_idx: &mut UidMap,
    uids: &mut Vec<Arc<str>>,
    degree: &mut Vec<u64>,
) -> Result<u32, String> {
    if let Some(&idx) = uid_to_idx.get(uid) {
        return Ok(idx);
    }
    let idx = intern_uid(uid, uid_to_idx, uids)?;
    degree.push(0);
    Ok(idx)
}

fn write_pod_slice<T: Copy, W: Write>(writer: &mut W, values: &[T]) -> Result<(), String> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    writer
        .write_all(bytes)
        .map_err(|e| parquet_err("write raw sidecar", e))
}

fn write_manifest(path: &Path, uids: &[Arc<str>]) -> Result<(), String> {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("node_idx", DataType::UInt32, false),
        Field::new("uid", DataType::Utf8, false),
    ]));
    let file = File::create(path).map_err(|e| parquet_err("create node manifest", e))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(writer_props()))
        .map_err(|e| parquet_err("create node manifest writer", e))?;

    for (chunk_idx, chunk) in uids.chunks(BATCH_SIZE).enumerate() {
        let start = chunk_idx * BATCH_SIZE;
        let node_idx: Vec<u32> = (start..start + chunk.len()).map(|i| i as u32).collect();
        let uid_refs: Vec<&str> = chunk.iter().map(AsRef::as_ref).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(node_idx)) as ArrayRef,
                Arc::new(StringArray::from(uid_refs)) as ArrayRef,
            ],
        )
        .map_err(|e| parquet_err("build node manifest batch", e))?;
        writer
            .write(&batch)
            .map_err(|e| parquet_err("write node manifest batch", e))?;
    }
    writer
        .close()
        .map_err(|e| parquet_err("close node manifest writer", e))?;
    Ok(())
}

fn sidecar_paths(int_edges_path: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let parent = int_edges_path.parent().unwrap_or_else(|| Path::new("."));
    (
        parent.join("src.u32.bin"),
        parent.join("dst.u32.bin"),
        parent.join("weight.f64.bin"),
    )
}

fn remove_edge_artifacts(int_edges_path: &Path) -> Result<(), String> {
    if int_edges_path.exists() {
        std::fs::remove_file(int_edges_path)
            .map_err(|e| parquet_err("remove stale int edges parquet", e))?;
    }
    let (src_path, dst_path, weight_path) = sidecar_paths(int_edges_path);
    for path in [src_path, dst_path, weight_path] {
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| parquet_err("remove stale raw sidecar", e))?;
        }
    }
    Ok(())
}

fn write_int_edge_outputs(
    edge_path: &Path,
    int_edges_path: &Path,
    uid_to_idx: &UidMap,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
    write_parquet: bool,
) -> Result<(usize, PathBuf, PathBuf, PathBuf), String> {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("src", DataType::UInt32, false),
        Field::new("dst", DataType::UInt32, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let mut parquet_writer = if write_parquet {
        let file = File::create(int_edges_path).map_err(|e| parquet_err("create int edges", e))?;
        Some(
            ArrowWriter::try_new(file, schema.clone(), Some(writer_props()))
                .map_err(|e| parquet_err("create int edge writer", e))?,
        )
    } else {
        if int_edges_path.exists() {
            std::fs::remove_file(int_edges_path)
                .map_err(|e| parquet_err("remove stale int edges parquet", e))?;
        }
        None
    };
    let (src_path, dst_path, weight_path) = sidecar_paths(int_edges_path);
    let mut src_file =
        BufWriter::new(File::create(&src_path).map_err(|e| parquet_err("create src sidecar", e))?);
    let mut dst_file =
        BufWriter::new(File::create(&dst_path).map_err(|e| parquet_err("create dst sidecar", e))?);
    let mut weight_file = BufWriter::new(
        File::create(&weight_path).map_err(|e| parquet_err("create weight sidecar", e))?,
    );

    let mut reader = record_reader(edge_path, &[uid1_col, uid2_col, weight_col])?;
    let mut n_edges = 0usize;
    while let Some(batch) = reader.next() {
        let batch = batch.map_err(|e| parquet_err("read edge batch", e))?;
        let uid1 = column(&batch, uid1_col)?;
        let uid2 = column(&batch, uid2_col)?;
        let weight_col_array = column(&batch, weight_col)?;
        let uid1_values = string_column(&uid1, uid1_col)?;
        let uid2_values = string_column(&uid2, uid2_col)?;
        let weight_values = weight_column(&weight_col_array, weight_col)?;

        let triples: Vec<(u32, u32, f64)> = (0..batch.num_rows())
            .into_par_iter()
            .map(|row| {
                let u1 = uid1_values.value(row, uid1_col)?;
                let u2 = uid2_values.value(row, uid2_col)?;
                let s = *uid_to_idx
                    .get(u1)
                    .ok_or_else(|| format!("uid not found in manifest: {}", u1))?;
                let d = *uid_to_idx
                    .get(u2)
                    .ok_or_else(|| format!("uid not found in manifest: {}", u2))?;
                Ok((s, d, weight_values.value(row, weight_col)?))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut src = Vec::with_capacity(triples.len());
        let mut dst = Vec::with_capacity(triples.len());
        let mut weight = Vec::with_capacity(triples.len());
        for (s, d, w) in triples {
            src.push(s);
            dst.push(d);
            weight.push(w);
        }

        write_pod_slice(&mut src_file, &src)?;
        write_pod_slice(&mut dst_file, &dst)?;
        write_pod_slice(&mut weight_file, &weight)?;

        let n_rows = src.len();
        if let Some(writer) = parquet_writer.as_mut() {
            let out_batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(UInt32Array::from(src)) as ArrayRef,
                    Arc::new(UInt32Array::from(dst)) as ArrayRef,
                    Arc::new(Float64Array::from(weight)) as ArrayRef,
                ],
            )
            .map_err(|e| parquet_err("build int edge batch", e))?;
            writer
                .write(&out_batch)
                .map_err(|e| parquet_err("write int edge batch", e))?;
            n_edges += out_batch.num_rows();
        } else {
            n_edges += n_rows;
        }
    }

    src_file
        .flush()
        .map_err(|e| parquet_err("flush src sidecar", e))?;
    dst_file
        .flush()
        .map_err(|e| parquet_err("flush dst sidecar", e))?;
    weight_file
        .flush()
        .map_err(|e| parquet_err("flush weight sidecar", e))?;
    if let Some(writer) = parquet_writer {
        writer
            .close()
            .map_err(|e| parquet_err("close int edge writer", e))?;
    }
    Ok((n_edges, src_path, dst_path, weight_path))
}

fn filled_vec<T: Clone>(len: usize, value: T, context: &str) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| parquet_err(context, e))?;
    values.resize(len, value);
    Ok(values)
}

fn read_int_edges_to_graph(
    edge_path: &Path,
    uid_to_idx: &UidMap,
    n_nodes: usize,
    n_edges: usize,
    mut degree: Vec<u64>,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
) -> Result<Graph, String> {
    if degree.len() != n_nodes {
        return Err(format!(
            "degree length {} does not match node count {}",
            degree.len(),
            n_nodes
        ));
    }

    let mut first_neighbor_index = filled_vec(
        n_nodes + 1,
        0u64,
        "reserve graph first-neighbor-index vector",
    )?;

    let mut running = 0u64;
    for node in 0..n_nodes {
        first_neighbor_index[node] = running;
        let d = degree[node];
        degree[node] = running;
        running += d;
    }
    first_neighbor_index[n_nodes] = running;
    let expected_directed = n_edges
        .checked_mul(2)
        .ok_or_else(|| "directed edge count overflow during graph build".to_string())?;
    if running as usize != expected_directed {
        return Err(format!(
            "remap degree mismatch: directed edges={}, expected={}",
            running, expected_directed
        ));
    }

    let n_directed_edges = running as usize;
    let mut neighbors = filled_vec(n_directed_edges, 0u32, "reserve graph neighbor vector")?;
    let mut edge_weights =
        filled_vec(n_directed_edges, 0.0f64, "reserve graph edge-weight vector")?;
    let mut offset = degree;

    let mut reader = record_reader(edge_path, &[uid1_col, uid2_col, weight_col])?;
    while let Some(batch) = reader.next() {
        let batch = batch.map_err(|e| parquet_err("read edge batch", e))?;
        let uid1 = column(&batch, uid1_col)?;
        let uid2 = column(&batch, uid2_col)?;
        let weight_col_array = column(&batch, weight_col)?;
        let uid1_values = string_column(&uid1, uid1_col)?;
        let uid2_values = string_column(&uid2, uid2_col)?;
        let weight_values = weight_column(&weight_col_array, weight_col)?;

        let triples: Vec<(u32, u32, f64)> = (0..batch.num_rows())
            .into_par_iter()
            .map(|row| {
                let u1 = uid1_values.value(row, uid1_col)?;
                let u2 = uid2_values.value(row, uid2_col)?;
                let s = *uid_to_idx
                    .get(u1)
                    .ok_or_else(|| format!("uid not found in manifest: {}", u1))?;
                let d = *uid_to_idx
                    .get(u2)
                    .ok_or_else(|| format!("uid not found in manifest: {}", u2))?;
                Ok((s, d, weight_values.value(row, weight_col)?))
            })
            .collect::<Result<Vec<_>, String>>()?;

        for (s, d, w) in triples {
            let s_idx = s as usize;
            let d_idx = d as usize;

            let pos_s = offset[s_idx] as usize;
            neighbors[pos_s] = d;
            edge_weights[pos_s] = w;
            offset[s_idx] += 1;

            let pos_d = offset[d_idx] as usize;
            neighbors[pos_d] = s;
            edge_weights[pos_d] = w;
            offset[d_idx] += 1;
        }
    }

    Ok(Graph {
        n_nodes,
        n_edges: n_directed_edges,
        first_neighbor_index,
        neighbors,
        edge_weights,
        node_weights: filled_vec(n_nodes, 1.0f64, "reserve graph node-weight vector")?,
        self_loop_weights: filled_vec(n_nodes, 0.0f64, "reserve graph self-loop vector")?,
    })
}

pub fn integer_remap_parquet(
    edge_path: &Path,
    output_dir: &Path,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
) -> Result<RemapOutput, String> {
    integer_remap_parquet_with_options(edge_path, output_dir, uid1_col, uid2_col, weight_col, true)
}

pub fn integer_remap_parquet_with_options(
    edge_path: &Path,
    output_dir: &Path,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
    write_int_edges_parquet: bool,
) -> Result<RemapOutput, String> {
    create_dir_all(output_dir).map_err(|e| parquet_err("create remap output dir", e))?;
    let node_manifest_path = output_dir.join("node_manifest.parquet");
    let int_edges_path = output_dir.join("int_edges.parquet");

    let mut uid_to_idx: UidMap = UidMap::new();
    let mut uids: Vec<Arc<str>> = Vec::new();
    let mut reader = record_reader(edge_path, &[uid1_col, uid2_col])?;
    while let Some(batch) = reader.next() {
        let batch = batch.map_err(|e| parquet_err("read UID batch", e))?;
        let uid1 = column(&batch, uid1_col)?;
        let uid2 = column(&batch, uid2_col)?;
        let uid1_values = string_column(&uid1, uid1_col)?;
        let uid2_values = string_column(&uid2, uid2_col)?;
        for row in 0..batch.num_rows() {
            let u1 = uid1_values.value(row, uid1_col)?;
            let u2 = uid2_values.value(row, uid2_col)?;
            intern_uid(u1, &mut uid_to_idx, &mut uids)?;
            intern_uid(u2, &mut uid_to_idx, &mut uids)?;
        }
    }

    write_manifest(&node_manifest_path, &uids)?;
    let (n_edges, src_path, dst_path, weight_path) = write_int_edge_outputs(
        edge_path,
        &int_edges_path,
        &uid_to_idx,
        uid1_col,
        uid2_col,
        weight_col,
        write_int_edges_parquet,
    )?;

    Ok(RemapOutput {
        n_nodes: uids.len(),
        n_edges,
        node_manifest_path,
        int_edges_path,
        src_path,
        dst_path,
        weight_path,
    })
}

pub fn integer_remap_parquet_to_graph(
    edge_path: &Path,
    output_dir: &Path,
    uid1_col: &str,
    uid2_col: &str,
    weight_col: &str,
) -> Result<RemapGraphOutput, String> {
    create_dir_all(output_dir).map_err(|e| parquet_err("create remap output dir", e))?;
    let node_manifest_path = output_dir.join("node_manifest.parquet");
    let int_edges_path = output_dir.join("int_edges.parquet");
    remove_edge_artifacts(&int_edges_path)?;

    let mut uid_to_idx: UidMap = UidMap::new();
    let mut uids: Vec<Arc<str>> = Vec::new();
    let mut degree: Vec<u64> = Vec::new();
    let mut n_edges = 0usize;
    let mut reader = record_reader(edge_path, &[uid1_col, uid2_col])?;
    while let Some(batch) = reader.next() {
        let batch = batch.map_err(|e| parquet_err("read UID batch", e))?;
        let uid1 = column(&batch, uid1_col)?;
        let uid2 = column(&batch, uid2_col)?;
        let uid1_values = string_column(&uid1, uid1_col)?;
        let uid2_values = string_column(&uid2, uid2_col)?;
        for row in 0..batch.num_rows() {
            let u1 = uid1_values.value(row, uid1_col)?;
            let u2 = uid2_values.value(row, uid2_col)?;
            let s = intern_uid_with_degree(u1, &mut uid_to_idx, &mut uids, &mut degree)?;
            let d = intern_uid_with_degree(u2, &mut uid_to_idx, &mut uids, &mut degree)?;
            degree[s as usize] += 1;
            degree[d as usize] += 1;
            n_edges = n_edges
                .checked_add(1)
                .ok_or_else(|| "edge count overflow during integer remap".to_string())?;
        }
    }

    write_manifest(&node_manifest_path, &uids)?;
    let graph = read_int_edges_to_graph(
        edge_path,
        &uid_to_idx,
        uids.len(),
        n_edges,
        degree,
        uid1_col,
        uid2_col,
        weight_col,
    )?;

    Ok(RemapGraphOutput {
        n_nodes: uids.len(),
        n_edges,
        node_manifest_path,
        int_edges_path,
        graph,
    })
}
