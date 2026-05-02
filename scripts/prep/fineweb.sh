# shellcheck shell=bash
# FineWeb-Edu → REPO_ROOT/data/fineweb-edu (Rust corpus-prep preferred, else Python).

# shellcheck source=prep/corpus.sh
source "${SCRIPT_DIR}/prep/corpus.sh"

prep_build_fineweb_edu() {
  prep_require_repo_root || return 1
  local out_dir="${REPO_ROOT}/data/fineweb-edu"
  local py="${PYTHON_BIN}"

  mkdir -p "${REPO_ROOT}/data"
  cd "${REPO_ROOT}" || return 1

  local use_rust="${USE_RUST_FINEWEB_PREP:-1}"
  if [[ "${use_rust}" != "0" ]] && command -v cargo >/dev/null 2>&1; then
    prep_corpus_hf_build \
      "${out_dir}" \
      "HuggingFaceFW/fineweb-edu" \
      "sample-10BT" \
      "edufineweb" \
      "text"
  else
    if [[ "${use_rust}" != "0" ]]; then
      echo "prep: cargo not found; using Python prep_fineweb.py" >&2
    fi
    # shellcheck disable=SC2086
    "${py}" scripts/prep_fineweb.py --out "${out_dir}" ${PREP_FINEWEB_EXTRA:-}
  fi
}
