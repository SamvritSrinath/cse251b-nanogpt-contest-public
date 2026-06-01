#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/ingest_data.sh --hf-dataset REPO_ID [options]
  ./scripts/ingest_data.sh --local-parquet-dir DIR [options]

Canonical data-ingestion entrypoint for this repository:
1. optionally list/download a Hugging Face dataset snapshot,
2. recurse for parquet files,
3. tokenize into raw GPT-2 uint16 .bin shards,
4. clean temporary parquet cache after success by default for HF downloads.

Source options:
  --hf-dataset REPO_ID      Hugging Face dataset repo id.
  --local-parquet-dir DIR   Existing local parquet tree to tokenize.
  --include GLOB            Optional HF include glob.
  --exclude GLOB            Optional HF exclude glob.
  --hf-revision REV         HF revision, branch, tag, or commit. Default: main.

Output options:
  --source-name NAME        Logical source name. Default: derived from dataset or local dir.
  --data-root PATH          Root for managed raw cache and output dirs. Default: <repo>/data.
  --raw-cache-dir PATH      Explicit managed parquet cache dir for HF downloads.
  --out-dir PATH            Explicit output dir for token shards.
  --text-column NAME        Parquet text column. Default: text.
  --shard-prefix NAME       Output shard prefix. Default: source-name with '-' mapped to '_'.
  --max-parquet-files N     Forwarded to corpus-prep for smoke tests.
  --split-mode MODE         corpus-prep split mode. Default: train-only.
  --emit-doc-index          Emit <shard>.bin.docs.json sidecars.
  --doc-id-column NAME      Optional metadata column for doc ids.
  --title-column NAME       Optional metadata column for titles.
  --section-column NAME     Optional metadata column for sections.

Behavior:
  --dry-run                 List what would be downloaded/read, then exit.
  --keep-raw-parquet        Keep the managed HF parquet cache after successful sharding.
  --clean-hf-cache-after-source
  --clean-cache-after-success
                            Cleanup aliases; both keep the default raw-cache cleanup enabled.
  --delete-local-parquet    Remove --local-parquet-dir after successful sharding.
  --no-clean-raw-cache      Do not clear the managed HF parquet cache before download.
  --no-clean-output         Do not clear the output shard dir before tokenization.
  --min-free-gb N           Require at least N GB free on the repo root filesystem before ingest.
  --gcs-uri URI             Optional gs:// destination to upload the final shard dir.
  -h, --help                Show this help text.

Examples:
  ./scripts/ingest_data.sh \
    --hf-dataset HuggingFaceFW/fineweb-edu \
    --include "sample/10BT/*.parquet" \
    --source-name fineweb-edu \
    --shard-prefix edufineweb

  ./scripts/ingest_data.sh \
    --hf-dataset some-org/openwebtext \
    --source-name openwebtext \
    --dry-run

  ./scripts/ingest_data.sh \
    --local-parquet-dir /mnt/disks/data/raw/openwebtext \
    --source-name openwebtext
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/data.sh
source "${SCRIPT_DIR}/lib/data.sh"
data_source_workspace_env "${REPO_ROOT}"

HF_DATASET=""
LOCAL_PARQUET_DIR=""
INCLUDE_GLOB=""
EXCLUDE_GLOB=""
HF_REVISION="main"
SOURCE_NAME=""
DATA_ROOT="${REPO_ROOT}/data"
RAW_CACHE_DIR=""
OUT_DIR=""
TEXT_COLUMN="text"
SHARD_PREFIX=""
MAX_PARQUET_FILES=""
SPLIT_MODE="train-only"
EMIT_DOC_INDEX=0
DOC_ID_COLUMN=""
TITLE_COLUMN=""
SECTION_COLUMN=""
DRY_RUN=0
KEEP_RAW_PARQUET=0
DELETE_LOCAL_PARQUET=0
CLEAN_RAW_CACHE=1
CLEAN_OUTPUT=1
GCS_URI=""
MIN_FREE_GB=""

print_disk_usage() {
  local label="$1"
  echo "ingest-data: disk usage (${label})"
  df -h /
  du -sh "${DATA_ROOT}" 2>/dev/null || true
  if [[ -n "${RAW_CACHE_DIR}" ]]; then du -sh "${RAW_CACHE_DIR}" 2>/dev/null || true; fi
  if [[ -n "${OUT_DIR}" ]]; then du -sh "${OUT_DIR}" 2>/dev/null || true; fi
}

check_min_free_gb() {
  local min_gb="$1"
  local available
  available="$(df -BG "${REPO_ROOT}" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
  if [[ -z "${available}" ]]; then
    echo "ingest-data: could not determine free disk for ${REPO_ROOT}" >&2
    exit 1
  fi
  if (( available < min_gb )); then
    echo "ingest-data: only ${available} GB free on repo filesystem; need at least ${min_gb} GB" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-dataset)
      HF_DATASET="${2:-}"
      shift 2
      ;;
    --local-parquet-dir)
      LOCAL_PARQUET_DIR="${2:-}"
      shift 2
      ;;
    --include)
      INCLUDE_GLOB="${2:-}"
      shift 2
      ;;
    --exclude)
      EXCLUDE_GLOB="${2:-}"
      shift 2
      ;;
    --hf-revision)
      HF_REVISION="${2:-}"
      shift 2
      ;;
    --source-name)
      SOURCE_NAME="${2:-}"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="${2:-}"
      shift 2
      ;;
    --raw-cache-dir)
      RAW_CACHE_DIR="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --text-column)
      TEXT_COLUMN="${2:-}"
      shift 2
      ;;
    --shard-prefix)
      SHARD_PREFIX="${2:-}"
      shift 2
      ;;
    --max-parquet-files)
      MAX_PARQUET_FILES="${2:-}"
      shift 2
      ;;
    --split-mode)
      SPLIT_MODE="${2:-}"
      shift 2
      ;;
    --emit-doc-index)
      EMIT_DOC_INDEX=1
      shift
      ;;
    --doc-id-column)
      DOC_ID_COLUMN="${2:-}"
      shift 2
      ;;
    --title-column)
      TITLE_COLUMN="${2:-}"
      shift 2
      ;;
    --section-column)
      SECTION_COLUMN="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --keep-raw-parquet)
      KEEP_RAW_PARQUET=1
      shift
      ;;
    --clean-hf-cache-after-source|--clean-cache-after-success)
      KEEP_RAW_PARQUET=0
      shift
      ;;
    --delete-local-parquet)
      DELETE_LOCAL_PARQUET=1
      shift
      ;;
    --no-clean-raw-cache)
      CLEAN_RAW_CACHE=0
      shift
      ;;
    --no-clean-output)
      CLEAN_OUTPUT=0
      shift
      ;;
    --gcs-uri)
      GCS_URI="${2:-}"
      shift 2
      ;;
    --min-free-gb)
      MIN_FREE_GB="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ingest-data: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${HF_DATASET}" && -n "${LOCAL_PARQUET_DIR}" ]]; then
  echo "ingest-data: choose either --hf-dataset or --local-parquet-dir, not both" >&2
  exit 2
fi
if [[ -z "${HF_DATASET}" && -z "${LOCAL_PARQUET_DIR}" ]]; then
  echo "ingest-data: one of --hf-dataset or --local-parquet-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "${SOURCE_NAME}" ]]; then
  if [[ -n "${HF_DATASET}" ]]; then
    SOURCE_NAME="$(basename "${HF_DATASET}")"
  else
    SOURCE_NAME="$(basename "${LOCAL_PARQUET_DIR}")"
  fi
fi
SOURCE_NAME="$(data_sanitize_name "${SOURCE_NAME}")"

if [[ -z "${RAW_CACHE_DIR}" && -n "${HF_DATASET}" ]]; then
  RAW_CACHE_DIR="${DATA_ROOT}/raw-parquets/${SOURCE_NAME}"
fi
if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${DATA_ROOT}/${SOURCE_NAME}"
fi
if [[ -z "${SHARD_PREFIX}" ]]; then
  SHARD_PREFIX="$(printf '%s' "${SOURCE_NAME}" | tr '-' '_')"
fi

if [[ -n "${MIN_FREE_GB}" ]]; then
  check_min_free_gb "${MIN_FREE_GB}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -n "${HF_DATASET}" ]]; then
    data_hf_snapshot_download \
      "${HF_DATASET}" \
      "${RAW_CACHE_DIR}" \
      "${INCLUDE_GLOB}" \
      "${HF_REVISION}" \
      "${EXCLUDE_GLOB}" \
      "1"
  else
    [[ -d "${LOCAL_PARQUET_DIR}" ]] || {
      echo "ingest-data: local parquet dir not found: ${LOCAL_PARQUET_DIR}" >&2
      exit 1
    }
    count="$(data_count_parquet_files "${LOCAL_PARQUET_DIR}")"
    echo "ingest-data: local parquet dry-run from ${LOCAL_PARQUET_DIR} (${count} file(s))"
    data_list_parquet_files "${LOCAL_PARQUET_DIR}"
  fi
  exit 0
fi

print_disk_usage "before ${SOURCE_NAME}"

if [[ -n "${HF_DATASET}" ]]; then
  data_need_cmd cargo
  data_hf_cli_bin >/dev/null
  if [[ "${CLEAN_RAW_CACHE}" == "1" ]]; then
    data_reset_dir "${RAW_CACHE_DIR}" "raw parquet cache" "${DATA_ROOT}"
  else
    mkdir -p "${RAW_CACHE_DIR}"
  fi

  data_hf_snapshot_download \
    "${HF_DATASET}" \
    "${RAW_CACHE_DIR}" \
    "${INCLUDE_GLOB}" \
    "${HF_REVISION}" \
    "${EXCLUDE_GLOB}" \
    "0"

  PARQUET_ROOT="${RAW_CACHE_DIR}"
else
  data_need_cmd cargo
  [[ -d "${LOCAL_PARQUET_DIR}" ]] || {
    echo "ingest-data: local parquet dir not found: ${LOCAL_PARQUET_DIR}" >&2
    exit 1
  }
  PARQUET_ROOT="${LOCAL_PARQUET_DIR}"
fi

parquet_count="$(data_count_parquet_files "${PARQUET_ROOT}")"
if [[ "${parquet_count}" == "0" ]]; then
  echo "ingest-data: no *.parquet files found under ${PARQUET_ROOT}" >&2
  exit 1
fi

if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
  data_reset_dir "${OUT_DIR}" "token shard output" "${DATA_ROOT}"
else
  mkdir -p "${OUT_DIR}"
fi

echo "ingest-data: tokenizing ${parquet_count} parquet file(s) from ${PARQUET_ROOT}"
cmd=(
  "${SCRIPT_DIR}/corpus_prep.sh"
  --local-parquet-dir "${PARQUET_ROOT}"
  --out "${OUT_DIR}"
  --shard-prefix "${SHARD_PREFIX}"
  --text-column "${TEXT_COLUMN}"
  --split-mode "${SPLIT_MODE}"
  --source-name "${SOURCE_NAME}"
)
if [[ -n "${MAX_PARQUET_FILES}" ]]; then
  cmd+=(--max-parquet-files "${MAX_PARQUET_FILES}")
fi
if [[ "${EMIT_DOC_INDEX}" == "1" ]]; then
  cmd+=(--emit-doc-index)
fi
if [[ -n "${DOC_ID_COLUMN}" ]]; then
  cmd+=(--doc-id-column "${DOC_ID_COLUMN}")
fi
if [[ -n "${TITLE_COLUMN}" ]]; then
  cmd+=(--title-column "${TITLE_COLUMN}")
fi
if [[ -n "${SECTION_COLUMN}" ]]; then
  cmd+=(--section-column "${SECTION_COLUMN}")
fi
"${cmd[@]}"

if [[ -n "${HF_DATASET}" ]] && [[ "${KEEP_RAW_PARQUET}" != "1" ]]; then
  data_remove_tree "${RAW_CACHE_DIR}" "raw parquet cache" "${DATA_ROOT}"
fi

if [[ -z "${HF_DATASET}" ]] && [[ "${DELETE_LOCAL_PARQUET}" == "1" ]]; then
  data_remove_tree "${LOCAL_PARQUET_DIR}" "local parquet source" "${DATA_ROOT}"
fi

if [[ -n "${GCS_URI}" ]]; then
  data_need_cmd gcloud
  echo "ingest-data: uploading ${OUT_DIR} -> ${GCS_URI}"
  gcloud storage cp --recursive "${OUT_DIR}" "${GCS_URI}"
fi

print_disk_usage "after ${SOURCE_NAME}"
echo "ingest-data: done"
echo "ingest-data: source-name=${SOURCE_NAME}"
echo "ingest-data: out-dir=${OUT_DIR}"
if [[ -n "${HF_DATASET}" ]]; then
  echo "ingest-data: raw parquet cache cleaned"
fi
