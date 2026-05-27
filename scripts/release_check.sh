#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if command -v uv >/dev/null 2>&1; then
  PY=(uv run --extra dev python)
  MATURIN=(uv run --extra dev maturin)
else
  PY=(python)
  MATURIN=(maturin)
fi

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run git diff --check
run git diff --cached --check
run cargo test --manifest-path rust/Cargo.toml --release
run cargo test --manifest-path rust-text/Cargo.toml --release
run "${MATURIN[@]}" develop --manifest-path rust/Cargo.toml
run "${MATURIN[@]}" develop --manifest-path rust-text/Cargo.toml
run "${PY[@]}" -m pytest -q

printf '\n==> sciscape CLI help\n'
"${PY[@]}" -m sciscape.cli --help >/dev/null

printf '\nRelease check passed.\n'
