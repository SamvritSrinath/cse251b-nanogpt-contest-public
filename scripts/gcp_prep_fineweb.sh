#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/ingest_data.sh" \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --include "sample/10BT/*.parquet" \
  --source-name fineweb-edu \
  --shard-prefix edufineweb \
  "$@"
