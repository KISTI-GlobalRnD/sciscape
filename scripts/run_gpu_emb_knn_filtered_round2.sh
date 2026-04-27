#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ALIGNED_TEXT_ROOT="${ALIGNED_TEXT_ROOT:-}"
MIN_TEXT_LEN="${MIN_TEXT_LEN:-20}"
MIN_ABSTRACT_LEN="${MIN_ABSTRACT_LEN:-20}"
MIN_TITLE_LEN="${MIN_TITLE_LEN:-0}"
MIN_METADATA_MATCH="${MIN_METADATA_MATCH:-0.95}"
REQUIRE_ABSTRACT="${REQUIRE_ABSTRACT:-1}"
K="${K:-30}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/linktype_edges_gcc}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/research/consensus/results/logs_gpu_emb}"

if [[ -z "${ALIGNED_TEXT_ROOT}" ]]; then
  echo "ALIGNED_TEXT_ROOT must point to the embedding-aligned works_text root" >&2
  echo "Example: export ALIGNED_TEXT_ROOT=/srv/openalex_aligned_text" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

FIELDS_AND_GPUS=(
  "18:0"
  "30:1"
  "26:2"
)

declare -a pids=()

for pair in "${FIELDS_AND_GPUS[@]}"; do
  field="${pair%%:*}"
  gpu="${pair##*:}"
  gcc_mapping="${OUT_ROOT}/field_${field}/node_mapping.parquet"
  works_text="${ALIGNED_TEXT_ROOT}/field_${field}/works_text.parquet"
  log_path="${LOG_DIR}/field_${field}_k${K}_gpu${gpu}.log"

  if [[ ! -f "${gcc_mapping}" ]]; then
    echo "Missing GCC mapping: ${gcc_mapping}" >&2
    exit 1
  fi
  if [[ ! -f "${works_text}" ]]; then
    echo "Missing aligned works_text: ${works_text}" >&2
    exit 1
  fi

  cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/gpu_build_emb_knn.py"
    --field "${field}"
    --k "${K}"
    --gpu "${gpu}"
    --gcc-mapping "${gcc_mapping}"
    --out-dir "${OUT_ROOT}/field_${field}"
    --filter-text
    --works-text-path "${works_text}"
    --min-text-len "${MIN_TEXT_LEN}"
    --min-title-len "${MIN_TITLE_LEN}"
    --min-abstract-len "${MIN_ABSTRACT_LEN}"
    --min-metadata-match "${MIN_METADATA_MATCH}"
  )

  if [[ "${REQUIRE_ABSTRACT}" == "1" ]]; then
    cmd+=(--require-abstract)
  fi

  echo "[launch] field_${field} gpu=${gpu} log=${log_path}"
  printf '  %q ' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more GPU runs failed. Check ${LOG_DIR}/*.log" >&2
  exit "${status}"
fi

echo "All filtered GPU k-NN runs completed successfully."
echo "Expected outputs per field:"
echo "  emb_full_knn${K}_textfilt_txt${MIN_TEXT_LEN}_abs${MIN_ABSTRACT_LEN}_reqabs.parquet"
echo "  emb_full_knn${K}_textfilt_txt${MIN_TEXT_LEN}_abs${MIN_ABSTRACT_LEN}_reqabs.metadata.json"
