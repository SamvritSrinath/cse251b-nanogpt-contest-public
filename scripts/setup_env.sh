#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_env.sh data [options]
  ./scripts/setup_env.sh train [options]

Canonical environment bootstrap for this repository.

Roles:
  data    Install Rust + Python tooling for dataset ingestion and tokenization.
  train   Install the Python training environment.

Options:
  --workspace-root PATH     Root for caches/tool homes. Default: <repo>.
  --env-file PATH           Shell env file to write and source. Default: <repo>/.workspace-env.sh
  --venv PATH               Virtualenv path. Default: <repo>/.venv
  --skip-apt                Skip apt-get update/install.
  --skip-corpus-build       Do not prebuild tools/corpus-prep on the data role.
  --torch-index-url URL     Install torch/torchvision/torchaudio from this index if torch
                            is not already visible inside the venv.
  -h, --help                Show this help text.

Notes:
  - The written env file pins mutable state onto the chosen workspace root:
    HF cache, pip cache, cargo/rustup homes, and conda package/env directories.
  - apt packages remain system-level; they cannot be relocated onto the mounted disk.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROLE="${1:-}"
if [[ "${ROLE}" == "-h" || "${ROLE}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "${ROLE}" ]]; then
  usage >&2
  exit 2
fi
shift

WORKSPACE_ROOT="${REPO_ROOT}"
ENV_FILE="${REPO_ROOT}/.workspace-env.sh"
VENV_DIR="${REPO_ROOT}/.venv"
SKIP_APT=0
SKIP_CORPUS_BUILD=0
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace-root)
      WORKSPACE_ROOT="${2:-}"
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-corpus-build)
      SKIP_CORPUS_BUILD=1
      shift
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "setup-env: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${ROLE}" in
  data|train)
    ;;
  *)
    echo "setup-env: role must be 'data' or 'train'" >&2
    exit 2
    ;;
esac

write_workspace_env() {
  local workspace_root="$1"
  local env_file="$2"
  mkdir -p "${workspace_root}" "$(dirname "${env_file}")"
  cat > "${env_file}" <<EOF
export WORKSPACE_ROOT="${workspace_root}"
export XDG_CACHE_HOME="${workspace_root}/.cache"
export PIP_CACHE_DIR="${workspace_root}/.cache/pip"
export HF_HOME="${workspace_root}/.hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export CARGO_HOME="${workspace_root}/.cargo"
export RUSTUP_HOME="${workspace_root}/.rustup"
export CONDA_PKGS_DIRS="${workspace_root}/.conda/pkgs"
export CONDA_ENVS_PATH="${workspace_root}/.conda/envs"
export PATH="${workspace_root}/.cargo/bin:\$PATH"
EOF
}

write_workspace_env "${WORKSPACE_ROOT}" "${ENV_FILE}"
# shellcheck disable=SC1090
source "${ENV_FILE}"

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
fi

if [[ "${SKIP_APT}" != "1" ]]; then
  ${SUDO} apt-get update
  if [[ "${ROLE}" == "data" ]]; then
    ${SUDO} apt-get install -y build-essential curl git libssl-dev pkg-config python3-pip python3-venv tmux
  else
    ${SUDO} apt-get install -y git python3-pip python3-venv tmux
  fi
fi

command -v python3 >/dev/null 2>&1 || {
  echo "setup-env: python3 is required" >&2
  exit 1
}
command -v git >/dev/null 2>&1 || {
  echo "setup-env: git is required" >&2
  exit 1
}

# Pin Rust to the workspace CARGO_HOME/RUSTUP_HOME from the env file. If we only checked
# `command -v cargo`, a user-level cargo would skip rustup-init while RUSTUP_HOME still
# pointed at an empty workspace .rustup, and builds would fail with "no default toolchain".
if [[ "${ROLE}" == "data" ]] && [[ ! -x "${CARGO_HOME}/bin/cargo" ]]; then
  command -v curl >/dev/null 2>&1 || {
    echo "setup-env: curl is required to install rustup" >&2
    exit 1
  }
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

if [[ -f "${CARGO_HOME}/env" ]]; then
  # shellcheck disable=SC1090
  source "${CARGO_HOME}/env"
fi

# rustup can be on PATH with no default toolchain (fresh multi-user installs, or interrupted
# setup). cargo then fails with: "rustup could not choose a version of cargo to run".
if [[ "${ROLE}" == "data" ]]; then
  _rustup=""
  if [[ -x "${CARGO_HOME}/bin/rustup" ]]; then
    _rustup="${CARGO_HOME}/bin/rustup"
  elif command -v rustup >/dev/null 2>&1; then
    _rustup="$(command -v rustup)"
  fi
  if [[ -n "${_rustup}" ]]; then
    "${_rustup}" toolchain install stable --profile minimal
    "${_rustup}" default stable
  fi
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

ACTIVATE_PATH="${VENV_DIR}/bin/activate"
if [[ ! -f "${ACTIVATE_PATH}" ]]; then
  echo "setup-env: missing activate script: ${ACTIVATE_PATH}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ACTIVATE_PATH}"
python -m pip install --upgrade pip wheel

if [[ "${ROLE}" == "data" ]]; then
  command -v cargo >/dev/null 2>&1 || {
    echo "setup-env: cargo is required for data role" >&2
    exit 1
  }
  python -m pip install PyYAML hf_transfer huggingface_hub
  if [[ "${SKIP_CORPUS_BUILD}" != "1" ]]; then
    (cd "${REPO_ROOT}/tools/corpus-prep" && cargo build --release)
  fi
  echo "setup-env: data role ready"
  echo "setup-env: source ${ENV_FILE}"
  echo "setup-env: source ${ACTIVATE_PATH}"
  echo "setup-env: next step: ./scripts/ingest_data.sh --hf-dataset HuggingFaceFW/fineweb-edu --include \"sample/10BT/*.parquet\" --source-name fineweb-edu --shard-prefix edufineweb"
else
  python -m pip install PyYAML huggingface_hub numpy optuna tqdm
  if ! python -c "import torch" >/dev/null 2>&1; then
    if [[ -n "${TORCH_INDEX_URL}" ]]; then
      python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
    else
      echo "setup-env: torch is not visible inside ${VENV_DIR}" >&2
      echo "setup-env: rerun with --torch-index-url https://download.pytorch.org/whl/cu121" >&2
      exit 1
    fi
  fi
  echo "setup-env: train role ready"
  echo "setup-env: source ${ENV_FILE}"
  echo "setup-env: source ${ACTIVATE_PATH}"
  echo "setup-env: next step: ./scripts/run_experiment.sh configs/small.yaml --notes \"smoke test\""
fi

