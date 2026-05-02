#!/usr/bin/env bash
set -euo pipefail

# Run a configured experiment from the repo-local virtual environment.
# Usage: ./scripts/run_experiment.sh configs/baseline.yaml --notes "smoke test"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${REPO_ROOT}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]] && [[ ! -x "/home/zeus/miniconda3/envs/cloudspace/bin/python" ]]; then
  echo "Expected ${REPO_ROOT}/.venv/bin/python. Create the virtualenv first." >&2
  exit 1
fi

CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: ./scripts/run_experiment.sh <config.yaml> [extra train.py args...]" >&2
  exit 1
fi
shift
if [[ -x "/home/zeus/miniconda3/envs/cloudspace/bin/python" ]]; then
  PYTHON_BIN="/home/zeus/miniconda3/envs/cloudspace/bin/python"
else
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi
"${PYTHON_BIN}" src/train.py --config "${CONFIG_PATH}" "$@"
