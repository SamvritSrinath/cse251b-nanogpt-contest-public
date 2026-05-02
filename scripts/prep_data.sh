#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# What is the difference?
# -----------------------------------------------------------------------------
# Rust (tools/corpus-prep): Hub or local parquet → GPT-2 uint16 .bin; FineWeb uses defaults in prep/fineweb.sh.
# Python (scripts/prep_fineweb.py): HuggingFace datasets + tiktoken + mp.Pool.
# Both emit the same *layout*: raw uint16 .bin shards for numpy memmap in src/data.py.
# Karpathy's upstream fineweb.py is closest to the Python path here (often .npy there);
# this repo standardizes on raw .bin.
#
# GCS workflow (optional)
# -----------------------------------------------------------------------------
# Export GCS_DATA_ROOT=gs://YOUR_BUCKET/SOME_PREFIX with the same layout as the repo,
# e.g. gs://my-bucket/cse251b-run1/data/fineweb-edu/*.bin
#
# For each path in data.sources from PREP_CONFIG (default configs/full.yaml):
#   - If GCS already has .bin shards under ${GCS_DATA_ROOT}/<that path>/ → pull only.
#   - Else → run a local builder when we ship one; then upload (see GCS_PUSH_AFTER_PREP).
#
# Env (common)
#   PREP_CONFIG            YAML to read data.sources from (default: configs/full.yaml)
#   PREP_DATA_SOURCES      Optional comma-separated paths (e.g. data/fineweb-edu,data/foo)
#                          If set, skips YAML parsing (no PyYAML needed on that run).
#   PREP_FORCE_LOCAL=1     Ignore GCS presence; always rebuild what has a local builder
#   GCS_DATA_ROOT          gs://bucket/prefix (optional; enables pull/push)
#   GCS_PUSH_AFTER_PREP=0  After a local build, do not upload to GCS (default: upload when GCS_DATA_ROOT set)
#   PYTHON_BIN, FALLBACK_PYTHON_BIN
#
# Env (FineWeb / data/fineweb-edu only)
#   USE_RUST_FINEWEB_PREP=0  Force Python prep_fineweb.py
#   PREP_FINEWEB_EXTRA       Extra args for Python backend
#   PREP_CORPUS_RUST_EXTRA   Extra args for tools/corpus-prep (any HF/local run)
#   PREP_FINEWEB_RUST_EXTRA  Legacy alias for PREP_CORPUS_RUST_EXTRA when unset
#
# Other mixture paths: pre-upload .bin shards to GCS, or from local/HF parquet run
#   tools/corpus-prep/target/release/corpus-prep --help
# and/or add a small scripts/prep/<name>.sh plus a branch in prep_build_local_for_relpath below.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
export SCRIPT_DIR

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${FALLBACK_PYTHON_BIN:-python3}"
fi
export PYTHON_BIN

# shellcheck source=prep/common.sh
source "${SCRIPT_DIR}/prep/common.sh"
# shellcheck source=prep/fineweb.sh
source "${SCRIPT_DIR}/prep/fineweb.sh"

PREP_CONFIG="${PREP_CONFIG:-configs/full.yaml}"
if [[ "${PREP_CONFIG}" = /* ]]; then
  CFG_PATH="${PREP_CONFIG}"
else
  CFG_PATH="${REPO_ROOT}/${PREP_CONFIG}"
fi

prep_build_local_for_relpath() {
  local rel="${1#/}"
  local base
  base="$(basename "${rel}")"
  case "${base}" in
    fineweb-edu)
      prep_build_fineweb_edu
      ;;
    *)
      echo "prep: no bundled local builder for '${rel}'." >&2
      echo "prep: Place uint16 .bin shards under ${REPO_ROOT}/${rel} or upload to \${GCS_DATA_ROOT}/${rel}/" >&2
      echo "prep: To add a builder, create scripts/prep/<name>.sh and extend prep_build_local_for_relpath in prep_data.sh." >&2
      return 1
      ;;
  esac
}

mkdir -p "${REPO_ROOT}/data"

have_gcs=0
if [[ -n "${GCS_DATA_ROOT:-}" ]]; then
  if prep_have_gcloud; then
    have_gcs=1
    export GCS_DATA_ROOT="${GCS_DATA_ROOT%/}"
  else
    echo "prep: GCS_DATA_ROOT is set but gcloud is not on PATH; skipping GCS pull/push" >&2
  fi
fi

prep_resolve_source_list() {
  if [[ -n "${PREP_DATA_SOURCES:-}" ]]; then
    echo "prep: using PREP_DATA_SOURCES (comma-separated); ignoring PREP_CONFIG for the list" >&2
    tr ',' '\n' <<< "${PREP_DATA_SOURCES}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d'
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prep/list_data_sources.py" "${CFG_PATH}"
  fi
}

while IFS= read -r rel; do
  [[ -n "${rel}" ]] || continue
  echo "prep: === ${rel} ===" >&2
  pulled=0
  if [[ "${have_gcs}" -eq 1 ]] && [[ "${PREP_FORCE_LOCAL:-0}" == "0" ]]; then
    gs_uri="$(prep_gcs_uri_for_relpath "${rel}")"
    if prep_gcs_dir_has_bins "${gs_uri}"; then
      prep_gcs_pull_source "${rel}"
      pulled=1
    else
      echo "prep: GCS has no .bin shards at ${gs_uri} (will build if a local recipe exists)" >&2
    fi
  fi
  if [[ "${pulled}" -eq 0 ]]; then
    prep_build_local_for_relpath "${rel}"
    if [[ "${have_gcs}" -eq 1 ]] && [[ "${GCS_PUSH_AFTER_PREP:-1}" != "0" ]]; then
      prep_gcs_push_source "${rel}"
    fi
  fi
done < <(prep_resolve_source_list)

if [[ -n "${PREP_DATA_SOURCES:-}" ]]; then
  echo "prep: finished (PREP_DATA_SOURCES)"
else
  echo "prep: finished for config ${PREP_CONFIG}"
fi
