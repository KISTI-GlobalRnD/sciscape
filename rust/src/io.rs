//! File I/O for edge lists and clustering results.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Read a TSV edge list (src, dst, weight).
///
/// Returns (src_vec, dst_vec, weight_vec, n_nodes).
pub fn read_edge_list<P: AsRef<Path>>(
    path: P,
    weighted: bool,
) -> std::io::Result<(Vec<u32>, Vec<u32>, Vec<f64>, usize)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut src = Vec::new();
    let mut dst = Vec::new();
    let mut weights = Vec::new();
    let mut max_node: u32 = 0;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let s: u32 = parts[0].parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let d: u32 = parts[1].parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let w: f64 = if weighted && parts.len() > 2 {
            parts[2].parse().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?
        } else {
            1.0
        };

        src.push(s);
        dst.push(d);
        weights.push(w);
        max_node = max_node.max(s).max(d);
    }

    let n_nodes = (max_node + 1) as usize;
    Ok((src, dst, weights, n_nodes))
}

/// Write clustering as TSV (node_id, cluster_id).
pub fn write_clustering<P: AsRef<Path>>(
    path: P,
    clusters: &[usize],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for (node, &cid) in clusters.iter().enumerate() {
        writeln!(writer, "{}\t{}", node, cid)?;
    }
    Ok(())
}

/// Read clustering from TSV (node_id, cluster_id).
pub fn read_clustering<P: AsRef<Path>>(
    path: P,
    n_nodes: usize,
) -> std::io::Result<Vec<usize>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut clusters = vec![0usize; n_nodes];

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let node: usize = parts[0].parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        let cid: usize = parts[1].parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        if node < n_nodes {
            clusters[node] = cid;
        }
    }

    Ok(clusters)
}

/// Read fixed nodes file (one node index per line).
pub fn read_fixed_nodes<P: AsRef<Path>>(
    path: P,
    n_nodes: usize,
) -> std::io::Result<Vec<bool>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut fixed = vec![false; n_nodes];

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let node: usize = line.parse().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        if node < n_nodes {
            fixed[node] = true;
        }
    }

    Ok(fixed)
}
