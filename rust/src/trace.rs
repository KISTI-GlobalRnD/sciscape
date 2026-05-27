//! Lightweight tracing helpers for long-running Leiden jobs.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::OnceLock;

const DEFAULT_TRACE_MIN_DIRECTED_EDGES: usize = 1_000_000;

fn env_setting(name: &str) -> &'static str {
    static TRACE: OnceLock<String> = OnceLock::new();
    static TRACE_RSS: OnceLock<String> = OnceLock::new();
    static TRACE_MIN_EDGES: OnceLock<String> = OnceLock::new();

    match name {
        "SCISCAPE_LEIDEN_TRACE" => TRACE
            .get_or_init(|| std::env::var(name).unwrap_or_default())
            .as_str(),
        "SCISCAPE_LEIDEN_TRACE_RSS" => TRACE_RSS
            .get_or_init(|| std::env::var(name).unwrap_or_default())
            .as_str(),
        "SCISCAPE_LEIDEN_TRACE_MIN_DIRECTED_EDGES" => TRACE_MIN_EDGES
            .get_or_init(|| std::env::var(name).unwrap_or_default())
            .as_str(),
        _ => "",
    }
}

fn truthy(value: &str) -> bool {
    !matches!(value, "" | "0" | "false" | "False" | "FALSE")
}

pub(crate) fn enabled() -> bool {
    truthy(env_setting("SCISCAPE_LEIDEN_TRACE"))
}

pub(crate) fn verbose() -> bool {
    matches!(env_setting("SCISCAPE_LEIDEN_TRACE"), "verbose")
}

pub(crate) fn min_directed_edges() -> usize {
    static MIN_EDGES: OnceLock<usize> = OnceLock::new();
    *MIN_EDGES.get_or_init(|| {
        env_setting("SCISCAPE_LEIDEN_TRACE_MIN_DIRECTED_EDGES")
            .parse::<usize>()
            .ok()
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_TRACE_MIN_DIRECTED_EDGES)
    })
}

pub(crate) fn should_trace_edges(directed_edges: usize) -> bool {
    enabled() && (verbose() || directed_edges >= min_directed_edges())
}

pub(crate) fn emit(args: fmt::Arguments<'_>) {
    if enabled() {
        eprintln!("[sciscape_leiden] {}", args);
    }
}

fn open_ddm_trace_file(path: &str) -> Option<File> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return None;
        }
    }
    OpenOptions::new().create(true).append(true).open(path).ok()
}

pub(crate) fn ddm_candidate_trace_enabled() -> bool {
    !std::env::var("SCISCAPE_DDM_CANDIDATE_TRACE_PATH")
        .unwrap_or_default()
        .is_empty()
}

pub(crate) fn ddm_candidate_trace_run_id() -> Option<String> {
    std::env::var("SCISCAPE_DDM_CANDIDATE_TRACE_RUN_ID")
        .ok()
        .filter(|value| !value.is_empty())
}

pub(crate) fn emit_ddm_candidate_trace(args: fmt::Arguments<'_>) {
    let path = std::env::var("SCISCAPE_DDM_CANDIDATE_TRACE_PATH").unwrap_or_default();
    if path.is_empty() {
        return;
    };
    let epoch = std::env::var("SCISCAPE_DDM_CANDIDATE_TRACE_EPOCH").unwrap_or_default();
    static FILE: OnceLock<Mutex<Option<(String, String, File)>>> = OnceLock::new();
    let file = FILE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = file.lock() {
        let needs_open = match guard.as_ref() {
            Some((current_path, current_epoch, _)) => {
                current_path != &path || current_epoch != &epoch
            }
            None => true,
        };
        if needs_open {
            *guard = open_ddm_trace_file(&path).map(|fh| (path.clone(), epoch.clone(), fh));
        }
        let Some((_, _, fh)) = guard.as_mut() else {
            return;
        };
        let _ = writeln!(fh, "{}", args);
    }
}

pub(crate) fn ddm_quality_trace_enabled() -> bool {
    !std::env::var("SCISCAPE_DDM_QUALITY_TRACE_PATH")
        .unwrap_or_default()
        .is_empty()
}

pub(crate) fn ddm_quality_trace_run_id() -> Option<String> {
    std::env::var("SCISCAPE_DDM_QUALITY_TRACE_RUN_ID")
        .ok()
        .filter(|value| !value.is_empty())
}

pub(crate) fn emit_ddm_quality_trace(args: fmt::Arguments<'_>) {
    let path = std::env::var("SCISCAPE_DDM_QUALITY_TRACE_PATH").unwrap_or_default();
    if path.is_empty() {
        return;
    };
    let epoch = std::env::var("SCISCAPE_DDM_QUALITY_TRACE_EPOCH").unwrap_or_default();
    static FILE: OnceLock<Mutex<Option<(String, String, File)>>> = OnceLock::new();
    let file = FILE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = file.lock() {
        let needs_open = match guard.as_ref() {
            Some((current_path, current_epoch, _)) => {
                current_path != &path || current_epoch != &epoch
            }
            None => true,
        };
        if needs_open {
            *guard = open_ddm_trace_file(&path).map(|fh| (path.clone(), epoch.clone(), fh));
        }
        let Some((_, _, fh)) = guard.as_mut() else {
            return;
        };
        let _ = writeln!(fh, "{}", args);
    }
}

pub(crate) fn ddm_trajectory_trace_enabled() -> bool {
    !std::env::var("SCISCAPE_DDM_TRAJECTORY_TRACE_PATH")
        .unwrap_or_default()
        .is_empty()
}

pub(crate) fn ddm_trajectory_trace_run_id() -> Option<String> {
    std::env::var("SCISCAPE_DDM_TRAJECTORY_TRACE_RUN_ID")
        .ok()
        .filter(|value| !value.is_empty())
}

pub(crate) fn emit_ddm_trajectory_trace(args: fmt::Arguments<'_>) {
    let path = std::env::var("SCISCAPE_DDM_TRAJECTORY_TRACE_PATH").unwrap_or_default();
    if path.is_empty() {
        return;
    };
    let epoch = std::env::var("SCISCAPE_DDM_TRAJECTORY_TRACE_EPOCH").unwrap_or_default();
    static FILE: OnceLock<Mutex<Option<(String, String, File)>>> = OnceLock::new();
    let file = FILE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = file.lock() {
        let needs_open = match guard.as_ref() {
            Some((current_path, current_epoch, _)) => {
                current_path != &path || current_epoch != &epoch
            }
            None => true,
        };
        if needs_open {
            *guard = open_ddm_trace_file(&path).map(|fh| (path.clone(), epoch.clone(), fh));
        }
        let Some((_, _, fh)) = guard.as_mut() else {
            return;
        };
        let _ = writeln!(fh, "{}", args);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DdmLocalMoveFocusNodes {
    target_nodes: Vec<usize>,
    neighbor_nodes: Vec<usize>,
}

impl DdmLocalMoveFocusNodes {
    pub(crate) fn role_for(&self, node: usize) -> Option<&'static str> {
        if self.target_nodes.binary_search(&node).is_ok() {
            return Some("target");
        }
        if self.neighbor_nodes.binary_search(&node).is_ok() {
            return Some("neighbor");
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.target_nodes.is_empty() && self.neighbor_nodes.is_empty()
    }
}

fn parse_node_list(value: &str) -> Vec<usize> {
    let mut nodes: Vec<usize> = value
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

pub(crate) fn ddm_local_move_focus_nodes() -> Option<DdmLocalMoveFocusNodes> {
    let focus = DdmLocalMoveFocusNodes {
        target_nodes: parse_node_list(
            &std::env::var("SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES").unwrap_or_default(),
        ),
        neighbor_nodes: parse_node_list(
            &std::env::var("SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES").unwrap_or_default(),
        ),
    };
    if focus.is_empty() {
        None
    } else {
        Some(focus)
    }
}

pub(crate) fn json_f64(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

pub(crate) fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

pub(crate) fn json_string_option(value: Option<String>) -> String {
    value
        .as_deref()
        .map(json_string)
        .unwrap_or_else(|| "null".to_string())
}

pub(crate) fn json_usize_option(value: Option<usize>) -> String {
    value
        .map(|item| item.to_string())
        .unwrap_or_else(|| "null".to_string())
}

pub(crate) fn leiden_quality_trace_enabled() -> bool {
    !std::env::var("SCISCAPE_LEIDEN_QUALITY_TRACE_PATH")
        .unwrap_or_default()
        .is_empty()
}

pub(crate) fn leiden_quality_trace_run_id() -> Option<String> {
    std::env::var("SCISCAPE_LEIDEN_QUALITY_TRACE_RUN_ID")
        .ok()
        .filter(|value| !value.is_empty())
}

pub(crate) fn leiden_quality_trace_target_max_weight() -> Option<f64> {
    std::env::var("SCISCAPE_LEIDEN_QUALITY_TRACE_TARGET_MAX_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

pub(crate) fn emit_leiden_quality_trace(args: fmt::Arguments<'_>) {
    let path = std::env::var("SCISCAPE_LEIDEN_QUALITY_TRACE_PATH").unwrap_or_default();
    if path.is_empty() {
        return;
    };
    let epoch = std::env::var("SCISCAPE_LEIDEN_QUALITY_TRACE_EPOCH").unwrap_or_default();
    static FILE: OnceLock<Mutex<Option<(String, String, File)>>> = OnceLock::new();
    let file = FILE.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = file.lock() {
        let needs_open = match guard.as_ref() {
            Some((current_path, current_epoch, _)) => {
                current_path != &path || current_epoch != &epoch
            }
            None => true,
        };
        if needs_open {
            *guard = open_ddm_trace_file(&path).map(|fh| (path.clone(), epoch.clone(), fh));
        }
        let Some((_, _, fh)) = guard.as_mut() else {
            return;
        };
        let _ = writeln!(fh, "{}", args);
    }
}

fn rss_enabled() -> bool {
    truthy(env_setting("SCISCAPE_LEIDEN_TRACE_RSS"))
}

fn parse_status_kb(line: &str, field: &str) -> Option<u64> {
    line.strip_prefix(field)?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
}

pub(crate) fn memory_fields() -> String {
    if !rss_enabled() {
        return String::new();
    }

    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return String::new();
    };

    let mut rss_kb = None;
    let mut hwm_kb = None;
    for line in status.lines() {
        if rss_kb.is_none() {
            rss_kb = parse_status_kb(line, "VmRSS:");
        }
        if hwm_kb.is_none() {
            hwm_kb = parse_status_kb(line, "VmHWM:");
        }
        if rss_kb.is_some() && hwm_kb.is_some() {
            break;
        }
    }

    match (rss_kb, hwm_kb) {
        (Some(rss), Some(hwm)) => format!(
            " rss_mb={:.1} hwm_mb={:.1}",
            rss as f64 / 1024.0,
            hwm as f64 / 1024.0
        ),
        (Some(rss), None) => format!(" rss_mb={:.1}", rss as f64 / 1024.0),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_parse_node_list_ignores_empty_and_invalid_values() {
        assert_eq!(parse_node_list(""), Vec::<usize>::new());
        assert_eq!(parse_node_list("bad, , -1"), Vec::<usize>::new());
        assert_eq!(parse_node_list("3, 1, 3,2,bad"), vec![1, 2, 3]);
    }

    #[test]
    fn test_local_move_focus_nodes_parse_env_and_target_precedence() {
        let _guard = ENV_LOCK.lock().unwrap();
        let previous_target = std::env::var("SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES").ok();
        let previous_neighbor = std::env::var("SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES").ok();

        std::env::set_var("SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES", "2,4,bad");
        std::env::set_var("SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES", "4,5");
        let focus = ddm_local_move_focus_nodes().unwrap();
        assert_eq!(focus.role_for(2), Some("target"));
        assert_eq!(focus.role_for(4), Some("target"));
        assert_eq!(focus.role_for(5), Some("neighbor"));
        assert_eq!(focus.role_for(6), None);

        match previous_target {
            Some(value) => std::env::set_var("SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES", value),
            None => std::env::remove_var("SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES"),
        }
        match previous_neighbor {
            Some(value) => std::env::set_var("SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES", value),
            None => std::env::remove_var("SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES"),
        }
    }
}
