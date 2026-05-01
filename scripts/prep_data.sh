#!/usr/bin/env bash
set -euo pipefail

# Download and tokenize FineWeb-Edu shards using Karpathy's build-nanogpt flow.
# The default source is HuggingFaceFW/fineweb-edu, which build-nanogpt fetches
# and tokenizes into GPT-2-compatible .bin shards for this repository.
# Usage: ./scripts/prep_data.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/build-nanogpt"
OUT_DIR="${REPO_ROOT}/data/fineweb-edu"
# PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
PYTHON_BIN="/home/zeus/miniconda3/envs/cloudspace/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${FALLBACK_PYTHON_BIN:-python3.11}"
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
