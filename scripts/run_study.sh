#!/usr/bin/env bash
set -euo pipefail

# Run a grid or Optuna study from the repo-local virtual environment.
# Usage: ./scripts/run_study.sh configs/studies/grid_smoke.yaml

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${REPO_ROOT}"

# if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
#   echo "Expected ${REPO_ROOT}/.venv/bin/python. Create the virtualenv first." >&2
#   exit 1
# fi

STUDY_PATH="${1:-}"
if [[ -z "${STUDY_PATH}" ]]; then
  echo "Usage: ./scripts/run_study.sh <study.yaml>" >&2
  exit 1
fi

# "${REPO_ROOT}/.venv/bin/python" -m src.search --study "${STUDY_PATH}"
python -m src.search --study "${STUDY_PATH}"
