#!/usr/bin/env bash
# shellcheck shell=bash

data_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
  echo "${script_dir}"
}

data_source_workspace_env() {
  local repo_root="${1:-}"
  if [[ -n "${repo_root}" ]] && [[ -f "${repo_root}/.workspace-env.sh" ]]; then
    local old_path="${PATH:-}"
    local old_cargo_home="${CARGO_HOME-}"
    local old_rustup_home="${RUSTUP_HOME-}"
    local old_xdg_cache_home="${XDG_CACHE_HOME-}"
    local old_pip_cache_dir="${PIP_CACHE_DIR-}"
    local old_hf_home="${HF_HOME-}"
    local old_conda_pkgs_dirs="${CONDA_PKGS_DIRS-}"
    local old_conda_envs_path="${CONDA_ENVS_PATH-}"
    local old_workspace_root="${WORKSPACE_ROOT-}"
    # shellcheck disable=SC1090
    source "${repo_root}/.workspace-env.sh"
    if [[ -n "${WORKSPACE_ROOT:-}" && ! -d "${WORKSPACE_ROOT}" ]]; then
      echo "data: ignoring stale .workspace-env.sh WORKSPACE_ROOT=${WORKSPACE_ROOT}" >&2
      export PATH="${old_path}"
      if [[ -n "${old_cargo_home}" ]]; then export CARGO_HOME="${old_cargo_home}"; else unset CARGO_HOME; fi
      if [[ -n "${old_rustup_home}" ]]; then export RUSTUP_HOME="${old_rustup_home}"; else unset RUSTUP_HOME; fi
      if [[ -n "${old_xdg_cache_home}" ]]; then export XDG_CACHE_HOME="${old_xdg_cache_home}"; else unset XDG_CACHE_HOME; fi
      if [[ -n "${old_pip_cache_dir}" ]]; then export PIP_CACHE_DIR="${old_pip_cache_dir}"; else unset PIP_CACHE_DIR; fi
      if [[ -n "${old_hf_home}" ]]; then export HF_HOME="${old_hf_home}"; else unset HF_HOME; fi
      if [[ -n "${old_conda_pkgs_dirs}" ]]; then export CONDA_PKGS_DIRS="${old_conda_pkgs_dirs}"; else unset CONDA_PKGS_DIRS; fi
      if [[ -n "${old_conda_envs_path}" ]]; then export CONDA_ENVS_PATH="${old_conda_envs_path}"; else unset CONDA_ENVS_PATH; fi
      if [[ -n "${old_workspace_root}" ]]; then export WORKSPACE_ROOT="${old_workspace_root}"; else unset WORKSPACE_ROOT; fi
    fi
  fi

  local cargo_env=""
  if [[ -n "${CARGO_HOME:-}" ]] && [[ -f "${CARGO_HOME}/env" ]]; then
    cargo_env="${CARGO_HOME}/env"
  elif [[ -f "${HOME}/.cargo/env" ]]; then
    cargo_env="${HOME}/.cargo/env"
  fi
  if [[ -n "${cargo_env}" ]]; then
    # shellcheck disable=SC1090
    source "${cargo_env}"
  fi
}

data_need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "data: required command not found: $1" >&2
    return 1
  }
}

data_have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

data_sanitize_name() {
  printf '%s' "$1" | tr '/:' '-' | tr -c 'A-Za-z0-9._-' '-'
}

data_hf_cli_bin() {
  if data_have_cmd hf; then
    echo "hf"
  elif data_have_cmd huggingface-cli; then
    echo "huggingface-cli"
  else
    echo "data: neither 'hf' nor 'huggingface-cli' is available on PATH" >&2
    return 1
  fi
}

data_hf_snapshot_download() {
  local repo_id="$1"
  local local_dir="$2"
  local include_glob="${3:-}"
  local revision="${4:-main}"
  local exclude_glob="${5:-}"
  local dry_run="${6:-0}"
  local cli=""

  cli="$(data_hf_cli_bin)" || return 1
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

  local cmd=("${cli}" download "${repo_id}" --repo-type dataset --local-dir "${local_dir}" --revision "${revision}")
  if [[ -n "${include_glob}" ]]; then
    cmd+=(--include "${include_glob}")
  fi
  if [[ -n "${exclude_glob}" ]]; then
    cmd+=(--exclude "${exclude_glob}")
  fi
  if [[ "${dry_run}" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  if [[ "${dry_run}" != "1" ]]; then
    mkdir -p "${local_dir}"
  fi
  "${cmd[@]}"
}

data_list_parquet_files() {
  local root="$1"
  [[ -d "${root}" ]] || return 0
  find "${root}" -type f -name '*.parquet' | LC_ALL=C sort
}

data_count_parquet_files() {
  local root="$1"
  data_list_parquet_files "${root}" | wc -l | tr -d '[:space:]'
}

data_assert_safe_rm_target() {
  local path="$1"
  local repo_root="${2:-}"
  [[ -n "${path}" ]] || {
    echo "data: refusing to remove an empty path" >&2
    return 1
  }
  case "${path}" in
    /|.|..)
      echo "data: refusing to remove unsafe path '${path}'" >&2
      return 1
      ;;
  esac
  if [[ -n "${repo_root}" ]]; then
    case "${path}" in
      "${repo_root}"|"${repo_root}/data")
        echo "data: refusing to remove repo root level path '${path}'" >&2
        return 1
        ;;
    esac
  fi
}

data_reset_dir() {
  local path="$1"
  local label="${2:-directory}"
  local repo_root="${3:-}"
  data_assert_safe_rm_target "${path}" "${repo_root}" || return 1
  if [[ -e "${path}" ]]; then
    echo "data: clearing ${label}: ${path}"
    rm -rf -- "${path}"
  fi
  mkdir -p "${path}"
}

data_remove_tree() {
  local path="$1"
  local label="${2:-directory}"
  local repo_root="${3:-}"
  data_assert_safe_rm_target "${path}" "${repo_root}" || return 1
  if [[ -e "${path}" ]]; then
    echo "data: removing ${label}: ${path}"
    rm -rf -- "${path}"
  fi
}
