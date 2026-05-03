#!/usr/bin/env bash
set -euo pipefail

# Run a configured experiment from the Studio conda env, repo-local venv, or PYTHON_BIN.
# Usage: ./scripts/run_experiment.sh configs/baseline.yaml --notes "smoke test"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: ./scripts/run_experiment.sh <config.yaml> [extra train.py args...]" >&2
  exit 1
fi
shift

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

"${PYTHON_BIN}" src/train.py --config "${CONFIG_PATH}" "$@"
