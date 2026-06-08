#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${SCRIPT_DIR}/lib/data.sh" ]]; then
  # shellcheck source=lib/data.sh
  source "${SCRIPT_DIR}/lib/data.sh"
  data_source_workspace_env "${REPO_ROOT}"
else
  if [[ -f "${REPO_ROOT}/.workspace-env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.workspace-env.sh"
  fi
  if [[ -n "${CARGO_HOME:-}" && -f "${CARGO_HOME}/env" ]]; then
    # shellcheck disable=SC1090
    source "${CARGO_HOME}/env"
  elif [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
  fi

  data_need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
      echo "corpus-prep: required command not found: $1" >&2
      return 1
    }
  }
fi

bin="${REPO_ROOT}/tools/corpus-prep/target/release/corpus-prep"
if [[ ! -x "${bin}" || "${REPO_ROOT}/tools/corpus-prep/src/main.rs" -nt "${bin}" || "${REPO_ROOT}/tools/corpus-prep/Cargo.toml" -nt "${bin}" ]]; then
  data_need_cmd cargo
  (cd "${REPO_ROOT}/tools/corpus-prep" && cargo build --release)
fi

extra="${CORPUS_PREP_EXTRA:-${PREP_CORPUS_RUST_EXTRA:-${PREP_FINEWEB_RUST_EXTRA:-}}}"
if [[ -n "${extra}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${extra})
  exec "${bin}" "${extra_args[@]}" "$@"
else
  exec "${bin}" "$@"
fi
