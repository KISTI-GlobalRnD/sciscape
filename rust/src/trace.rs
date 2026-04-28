//! Lightweight tracing helpers for long-running Leiden jobs.

use std::fmt;
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
