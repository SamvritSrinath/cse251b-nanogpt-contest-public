#!/usr/bin/env bash
set -euo pipefail

# Download and tokenize FineWeb-Edu shards using Karpathy's build-nanogpt flow.
# The default source is HuggingFaceFW/fineweb-edu, which build-nanogpt fetches
# and tokenizes into GPT-2-compatible .bin shards for this repository.
# Usage: ./scripts/prep_data.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/build-nanogpt"
OUT_DIR="${REPO_ROOT}/data/fineweb-edu"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" && ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is set but is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "/home/zeus/miniconda3/envs/cloudspace/bin/python" ]]; then
    PYTHON_BIN="/home/zeus/miniconda3/envs/cloudspace/bin/python"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python || true)"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find Python. Set PYTHON_BIN=/path/to/python and retry." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/data"

if [[ ! -d "${UPSTREAM_DIR}" ]]; then
  git clone https://github.com/karpathy/build-nanogpt.git "${UPSTREAM_DIR}"
fi

cd "${UPSTREAM_DIR}"
"${PYTHON_BIN}" fineweb.py

mkdir -p "${OUT_DIR}"
find "${UPSTREAM_DIR}" -type f -name "*.bin" -print0 | while IFS= read -r -d '' shard; do
  cp "${shard}" "${OUT_DIR}/"
done

echo "Copied FineWeb-Edu token shards to ${OUT_DIR}"
echo "Optional secondary corpora can live in sibling directories such as data/openwebtext/."
