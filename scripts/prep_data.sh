#!/usr/bin/env bash
set -euo pipefail

# Canonical config-driven data prep entrypoint.
#
# It resolves data.sources from PREP_CONFIG (or PREP_DATA_SOURCES), optionally pulls an
# already-tokenized source tree from GCS, and otherwise delegates local preparation to the
# canonical ingestion script:
#   ./scripts/ingest_data.sh
#
# Common env:
#   PREP_CONFIG             YAML to read data.sources from (default: configs/full.yaml)
#   PREP_DATA_SOURCES       Optional comma-separated repo-relative paths, e.g.
#                           data/fineweb-edu,data/openwebtext
#   PREP_FORCE_LOCAL=1      Ignore GCS presence and rebuild locally
#   GCS_DATA_ROOT           Optional gs://bucket/prefix laid out like the repo data tree
#   GCS_PUSH_AFTER_PREP=0   Do not upload locally built sources back to GCS
#   PYTHON_BIN, FALLBACK_PYTHON_BIN
#
# FineWeb defaults (for relpath data/fineweb-edu):
#   USE_RUST_FINEWEB_PREP=0 Force the legacy Python scripts/prep_fineweb.py fallback
#   PREP_FINEWEB_HF_INCLUDE Optional include glob; default sample/10BT/*.parquet
#   PREP_FINEWEB_HF_REVISION
#   PREP_FINEWEB_TEXT_COLUMN
#   PREP_FINEWEB_LOCAL_PARQUET_DIR
#   PREP_FINEWEB_OUT_DIR
#   PREP_FINEWEB_KEEP_RAW=1 Keep downloaded parquet cache after sharding
#   PREP_FINEWEB_NO_CLEAN_OUTPUT=1
#
# Generic HF parquet builder (opt-in for exactly one relpath):
#   PREP_GENERIC_HF_RELPATH    Repo-relative source path to satisfy, e.g. data/openwebtext
#   PREP_GENERIC_HF_DATASET    HF dataset repo id
#   PREP_GENERIC_HF_INCLUDE    Optional HF include glob; empty means full snapshot
#   PREP_GENERIC_HF_EXCLUDE    Optional HF exclude glob
#   PREP_GENERIC_HF_REVISION   HF revision (default: main)
#   PREP_GENERIC_HF_TEXT_COLUMN
#   PREP_GENERIC_HF_SHARD_PREFIX
#   PREP_GENERIC_HF_PARQUET_DIR
#   PREP_GENERIC_HF_OUT_DIR
#   PREP_GENERIC_HF_KEEP_RAW=1
#   PREP_GENERIC_HF_NO_CLEAN_OUTPUT=1

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

PREP_CONFIG="${PREP_CONFIG:-configs/full.yaml}"
if [[ "${PREP_CONFIG}" = /* ]]; then
  CFG_PATH="${PREP_CONFIG}"
else
  CFG_PATH="${REPO_ROOT}/${PREP_CONFIG}"
fi

prep_resolve_source_list() {
  if [[ -n "${PREP_DATA_SOURCES:-}" ]]; then
    echo "prep: using PREP_DATA_SOURCES; ignoring PREP_CONFIG for source discovery" >&2
    tr ',' '\n' <<< "${PREP_DATA_SOURCES}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d'
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prep/list_data_sources.py" "${CFG_PATH}"
  fi
}

prep_run_cmd() {
  echo "prep: $*" >&2
  "$@"
}

prep_build_fineweb_for_relpath() {
  local rel="${1#/}"
  local out_dir="${PREP_FINEWEB_OUT_DIR:-${REPO_ROOT}/${rel}}"

  if [[ "${USE_RUST_FINEWEB_PREP:-1}" == "0" ]]; then
    mkdir -p "$(dirname "${out_dir}")"
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prep_fineweb.py" --out "${out_dir}" ${PREP_FINEWEB_EXTRA:-}
    return 0
  fi

  local cmd=(
    "${SCRIPT_DIR}/ingest_data.sh"
    --hf-dataset "HuggingFaceFW/fineweb-edu"
    --include "${PREP_FINEWEB_HF_INCLUDE:-sample/10BT/*.parquet}"
    --hf-revision "${PREP_FINEWEB_HF_REVISION:-main}"
    --source-name "fineweb-edu"
    --shard-prefix "edufineweb"
    --text-column "${PREP_FINEWEB_TEXT_COLUMN:-text}"
    --out-dir "${out_dir}"
  )
  if [[ -n "${PREP_FINEWEB_LOCAL_PARQUET_DIR:-}" ]]; then
    cmd+=(--raw-cache-dir "${PREP_FINEWEB_LOCAL_PARQUET_DIR}")
  fi
  if [[ "${PREP_FINEWEB_KEEP_RAW:-0}" == "1" ]]; then
    cmd+=(--keep-raw-parquet)
  fi
  if [[ "${PREP_FINEWEB_NO_CLEAN_OUTPUT:-0}" == "1" ]]; then
    cmd+=(--no-clean-output)
  fi
  prep_run_cmd "${cmd[@]}"
}

prep_build_generic_hf_for_relpath() {
  local rel="${1#/}"
  local generic_rel="${PREP_GENERIC_HF_RELPATH:-}"
  [[ -n "${generic_rel}" ]] || return 1
  generic_rel="${generic_rel#/}"
  [[ "${rel}" == "${generic_rel}" ]] || return 1

  local dataset="${PREP_GENERIC_HF_DATASET:-}"
  [[ -n "${dataset}" ]] || {
    echo "prep: PREP_GENERIC_HF_RELPATH is set but PREP_GENERIC_HF_DATASET is empty" >&2
    return 1
  }

  local base
  base="$(basename "${rel}")"
  local cmd=(
    "${SCRIPT_DIR}/ingest_data.sh"
    --hf-dataset "${dataset}"
    --hf-revision "${PREP_GENERIC_HF_REVISION:-main}"
    --source-name "${base}"
    --text-column "${PREP_GENERIC_HF_TEXT_COLUMN:-text}"
    --shard-prefix "${PREP_GENERIC_HF_SHARD_PREFIX:-$(tr '-' '_' <<< "${base}")}"
    --out-dir "${PREP_GENERIC_HF_OUT_DIR:-${REPO_ROOT}/${rel}}"
  )
  if [[ -n "${PREP_GENERIC_HF_INCLUDE:-}" ]]; then
    cmd+=(--include "${PREP_GENERIC_HF_INCLUDE}")
  fi
  if [[ -n "${PREP_GENERIC_HF_EXCLUDE:-}" ]]; then
    cmd+=(--exclude "${PREP_GENERIC_HF_EXCLUDE}")
  fi
  if [[ -n "${PREP_GENERIC_HF_PARQUET_DIR:-}" ]]; then
    cmd+=(--raw-cache-dir "${PREP_GENERIC_HF_PARQUET_DIR}")
  fi
  if [[ "${PREP_GENERIC_HF_KEEP_RAW:-0}" == "1" ]]; then
    cmd+=(--keep-raw-parquet)
  fi
  if [[ "${PREP_GENERIC_HF_NO_CLEAN_OUTPUT:-0}" == "1" ]]; then
    cmd+=(--no-clean-output)
  fi
  prep_run_cmd "${cmd[@]}"
}

prep_build_local_for_relpath() {
  local rel="${1#/}"
  case "$(basename "${rel}")" in
    fineweb-edu)
      prep_build_fineweb_for_relpath "${rel}"
      ;;
    *)
      if prep_build_generic_hf_for_relpath "${rel}"; then
        :
      else
        echo "prep: no bundled local builder for '${rel}'." >&2
        echo "prep: Place uint16 .bin shards under ${REPO_ROOT}/${rel} or upload to \${GCS_DATA_ROOT}/${rel}/" >&2
        echo "prep: Or set PREP_GENERIC_HF_RELPATH=${rel} plus PREP_GENERIC_HF_DATASET=... to drive the generic HF path." >&2
        return 1
      fi
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
    echo "prep: GCS_DATA_ROOT is set but gcloud is not on PATH; skipping GCS sync" >&2
  fi
fi

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
      echo "prep: no token shards found at ${gs_uri}; building locally" >&2
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
