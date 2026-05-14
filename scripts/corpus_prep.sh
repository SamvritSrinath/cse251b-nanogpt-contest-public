#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/data.sh
source "${SCRIPT_DIR}/lib/data.sh"
data_source_workspace_env "${REPO_ROOT}"

bin="${REPO_ROOT}/tools/corpus-prep/target/release/corpus-prep"
if [[ ! -x "${bin}" ]]; then
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
