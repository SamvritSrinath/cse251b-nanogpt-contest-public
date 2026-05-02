#!/usr/bin/env bash
set -euo pipefail

# Upload an exported submission bundle to a HuggingFace model repo.
# Usage: ./scripts/submit_hf.sh username/cse251b-group-XX submission/<run-id>/best

if [[ $# -ne 2 ]]; then
  echo "Usage: ./scripts/submit_hf.sh <hf_repo_id> <submission_dir>" >&2
  exit 1
fi

HF_REPO_ID="$1"
SUBMISSION_DIR="$2"

if [[ ! -d "${SUBMISSION_DIR}" ]]; then
  echo "Submission directory not found: ${SUBMISSION_DIR}" >&2
  exit 1
fi

huggingface-cli upload "${HF_REPO_ID}" "${SUBMISSION_DIR}" . --repo-type model
