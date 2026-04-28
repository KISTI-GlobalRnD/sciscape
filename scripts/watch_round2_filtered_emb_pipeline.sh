#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-/tmp/aligned_openalex_text_round2_chunked}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/research/consensus/results/logs_gpu_emb}"
if [[ "$#" -gt 0 ]]; then
  FIELDS=("$@")
else
  FIELDS=(18 26)
fi

MIN_TEXT_LEN="${MIN_TEXT_LEN:-20}"
MIN_ABSTRACT_LEN="${MIN_ABSTRACT_LEN:-20}"
MIN_TITLE_LEN="${MIN_TITLE_LEN:-0}"
K="${K:-30}"

mkdir -p "${LOG_DIR}"

extract_pattern() {
  local field="$1"
  printf 'extract_emb_aligned_text.py --fields %s --out-root %s' "${field}" "${OUT_ROOT}"
}

wait_for_extract_or_restart() {
  local field="$1"
  local works_text="${OUT_ROOT}/field_${field}/works_text.parquet"

  if [[ -f "${works_text}" ]]; then
    echo "[field_${field}] aligned text already exists: ${works_text}"
    return 0
  fi

  local pattern
  pattern="$(extract_pattern "${field}")"
  local pid
  pid="$(pgrep -f "${pattern}" || true)"

  if [[ -n "${pid}" ]]; then
    echo "[field_${field}] waiting for running extraction pid=${pid}"
    while kill -0 "${pid}" 2>/dev/null; do
      if [[ -f "${works_text}" ]]; then
        echo "[field_${field}] aligned text appeared during wait"
        return 0
      fi
      sleep 30
    done
    if [[ -f "${works_text}" ]]; then
      echo "[field_${field}] aligned text completed"
      return 0
    fi
    echo "[field_${field}] extraction pid=${pid} exited without output; restarting"
  fi

  echo "[field_${field}] starting extraction"
  "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/extract_emb_aligned_text.py" \
    --fields "${field}" \
    --out-root "${OUT_ROOT}"
}

run_filtered_knn() {
  local field="$1"
  local works_text="${OUT_ROOT}/field_${field}/works_text.parquet"
  local out_parquet="${REPO_ROOT}/data/linktype_edges_gcc/field_${field}/emb_full_knn${K}_textfilt_txt${MIN_TEXT_LEN}_abs${MIN_ABSTRACT_LEN}_reqabs.parquet"

  if [[ -f "${out_parquet}" ]]; then
    echo "[field_${field}] filtered knn already exists: ${out_parquet}"
    return 0
  fi

  echo "[field_${field}] building filtered kNN"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_emb_knn_edges.py" \
    --field "${field}" \
    --k "${K}" \
    --device cuda \
    --filter-text \
    --works-text-path "${works_text}" \
    --min-text-len "${MIN_TEXT_LEN}" \
    --min-title-len "${MIN_TITLE_LEN}" \
    --min-abstract-len "${MIN_ABSTRACT_LEN}" \
    --require-abstract
}

for field in "${FIELDS[@]}"; do
  wait_for_extract_or_restart "${field}"
  run_filtered_knn "${field}"
done

echo "round2 filtered embedding pipeline complete"
