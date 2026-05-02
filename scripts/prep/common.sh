# shellcheck shell=bash
# Shared helpers for prep_data.sh (GCS + paths). Expects REPO_ROOT to be set by the caller.

prep_require_repo_root() {
  if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "prep: internal error: REPO_ROOT is unset" >&2
    return 1
  fi
}

# Remote URI for a repo-relative directory (e.g. data/fineweb-edu).
# GCS_DATA_ROOT must look like gs://bucket or gs://bucket/some/prefix (no trailing slash enforced).
prep_gcs_uri_for_relpath() {
  local root="${GCS_DATA_ROOT%/}"
  local rel="${1#/}"
  echo "${root}/${rel}"
}

prep_have_gcloud() {
  command -v gcloud >/dev/null 2>&1
}

# Return 0 if the gs://... tree appears to contain at least one .bin shard.
prep_gcs_dir_has_bins() {
  local gs_dir="${1%/}"
  prep_have_gcloud || return 1
  if gcloud storage ls "${gs_dir}/**" 2>/dev/null | grep -m1 -qE '\.bin$'; then
    return 0
  fi
  if gcloud storage ls "${gs_dir}/" 2>/dev/null | grep -m1 -qE '\.bin$'; then
    return 0
  fi
  return 1
}

prep_gcs_pull_source() {
  local rel="${1#/}"
  local gs_uri
  gs_uri="$(prep_gcs_uri_for_relpath "${rel}")"
  local parent="${REPO_ROOT}/$(dirname "${rel}")"
  mkdir -p "${parent}"
  echo "prep: pulling ${gs_uri} → ${REPO_ROOT}/${rel}" >&2
  gcloud storage cp --recursive "${gs_uri}" "${parent}/"
}

prep_gcs_push_source() {
  local rel="${1#/}"
  local local_dir="${REPO_ROOT}/${rel}"
  local gs_parent
  gs_parent="$(prep_gcs_uri_for_relpath "$(dirname "${rel}")")"
  [[ -d "${local_dir}" ]] || {
    echo "prep: cannot push missing directory ${local_dir}" >&2
    return 1
  }
  echo "prep: uploading ${local_dir} → ${gs_parent}/" >&2
  gcloud storage cp --recursive "${local_dir}" "${gs_parent}/"
}
