# shellcheck shell=bash
# Shared Rust corpus-prep (tools/corpus-prep). Requires common.sh and REPO_ROOT (from prep_data.sh).

prep_corpus_rust_bin() {
  prep_require_repo_root || return 1
  local rust_crate="${REPO_ROOT}/tools/corpus-prep"
  export CARGO_TARGET_DIR="${rust_crate}/target"
  echo "${CARGO_TARGET_DIR}/release/corpus-prep"
}

prep_corpus_rust_ensure() {
  local bin
  bin="$(prep_corpus_rust_bin)"
  if [[ ! -x "${bin}" ]]; then
    (cd "${REPO_ROOT}/tools/corpus-prep" && cargo build --release)
  fi
}

# Hugging Face Hub parquet corpus → uint16 .bin (GPT-2). Extra CLI: PREP_CORPUS_RUST_EXTRA or PREP_FINEWEB_RUST_EXTRA.
prep_corpus_hf_build() {
  local out_dir="$1"
  local hf_dataset="$2"
  local hf_subset="$3"
  local shard_prefix="$4"
  local text_column="${5:-text}"
  prep_corpus_rust_ensure || return 1
  local bin
  bin="$(prep_corpus_rust_bin)"
  mkdir -p "${REPO_ROOT}/data"
  cd "${REPO_ROOT}" || return 1
  local xtra="${PREP_CORPUS_RUST_EXTRA:-${PREP_FINEWEB_RUST_EXTRA:-}}"
  # shellcheck disable=SC2086
  "${bin}" \
    --out "${out_dir}" \
    --hf-dataset "${hf_dataset}" \
    --hf-subset "${hf_subset}" \
    --shard-prefix "${shard_prefix}" \
    --text-column "${text_column}" \
    ${xtra}
}

# Local recursive *.parquet → uint16 .bin (same tokenizer / shard layout).
prep_corpus_local_build() {
  local out_dir="$1"
  local parquet_root="$2"
  local shard_prefix="$3"
  local text_column="${4:-text}"
  prep_corpus_rust_ensure || return 1
  local bin
  bin="$(prep_corpus_rust_bin)"
  mkdir -p "${REPO_ROOT}/data"
  cd "${REPO_ROOT}" || return 1
  local xtra="${PREP_CORPUS_RUST_EXTRA:-${PREP_FINEWEB_RUST_EXTRA:-}}"
  # shellcheck disable=SC2086
  "${bin}" \
    --out "${out_dir}" \
    --local-parquet-dir "${parquet_root}" \
    --shard-prefix "${shard_prefix}" \
    --text-column "${text_column}" \
    ${xtra}
}
