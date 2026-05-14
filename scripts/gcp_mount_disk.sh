#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/gcp_mount_disk.sh --device /dev/disk/by-id/google-nanodata --mount-point /mnt/disks/nano-data-parse [--persist]
  ./scripts/gcp_mount_disk.sh --device /dev/sdb --mount-point /mnt/disks/nano-data-parse --format-if-needed --persist

Mount a GCP persistent disk, optionally formatting it first if it is blank.

Options:
  --device PATH             Block device or /dev/disk/by-id path to mount.
  --mount-point PATH        Target directory, e.g. /mnt/disks/nano-data-parse.
  --owner USER              Chown the mounted directory to this user (default: current user).
  --format-if-needed        Run mkfs.ext4 if the device has no filesystem yet.
  --persist                 Append an /etc/fstab entry using the disk UUID.
  -h, --help                Show this help text.

Notes:
  - Do not pass --format-if-needed on a disk that already contains data.
  - GCP does not auto-mount attached persistent disks for you.
EOF
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "mount-disk: required command not found: $1" >&2
    exit 1
  }
}

DEVICE=""
MOUNT_POINT=""
OWNER_USER="${USER}"
FORMAT_IF_NEEDED=0
PERSIST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --mount-point)
      MOUNT_POINT="${2:-}"
      shift 2
      ;;
    --owner)
      OWNER_USER="${2:-}"
      shift 2
      ;;
    --format-if-needed)
      FORMAT_IF_NEEDED=1
      shift
      ;;
    --persist)
      PERSIST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "mount-disk: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${DEVICE}" ]] || {
  echo "mount-disk: --device is required" >&2
  usage >&2
  exit 2
}
[[ -n "${MOUNT_POINT}" ]] || {
  echo "mount-disk: --mount-point is required" >&2
  usage >&2
  exit 2
}

need_cmd findmnt
need_cmd lsblk
need_cmd blkid

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
fi

if [[ ! -e "${DEVICE}" ]]; then
  echo "mount-disk: device not found: ${DEVICE}" >&2
  exit 1
fi

DEVICE_REAL="$(readlink -f "${DEVICE}" 2>/dev/null || realpath "${DEVICE}" 2>/dev/null || echo "${DEVICE}")"

existing_target="$(findmnt -rn -S "${DEVICE_REAL}" -o TARGET || true)"
if [[ -n "${existing_target}" ]]; then
  if [[ "${existing_target}" != "${MOUNT_POINT}" ]]; then
    echo "mount-disk: ${DEVICE_REAL} is already mounted at ${existing_target}" >&2
    exit 1
  fi
  echo "mount-disk: ${DEVICE_REAL} is already mounted at ${MOUNT_POINT}"
else
  fs_type="$(lsblk -no FSTYPE "${DEVICE_REAL}" | head -n 1 | tr -d '[:space:]')"
  if [[ -z "${fs_type}" ]]; then
    if [[ "${FORMAT_IF_NEEDED}" != "1" ]]; then
      echo "mount-disk: ${DEVICE_REAL} has no filesystem." >&2
      echo "mount-disk: re-run with --format-if-needed if this is a brand-new empty disk." >&2
      exit 1
    fi
    echo "mount-disk: formatting ${DEVICE_REAL} as ext4"
    ${SUDO} mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "${DEVICE_REAL}"
    fs_type="ext4"
  fi

  current_source="$(findmnt -rn -T "${MOUNT_POINT}" -o SOURCE || true)"
  if [[ -n "${current_source}" ]]; then
    echo "mount-disk: mount point already in use: ${MOUNT_POINT} (${current_source})" >&2
    exit 1
  fi

  ${SUDO} mkdir -p "${MOUNT_POINT}"
  ${SUDO} mount -o discard,defaults "${DEVICE_REAL}" "${MOUNT_POINT}"
fi

if id "${OWNER_USER}" >/dev/null 2>&1; then
  ${SUDO} chown "${OWNER_USER}:${OWNER_USER}" "${MOUNT_POINT}"
fi

if [[ "${PERSIST}" == "1" ]]; then
  uuid="$(${SUDO} blkid -s UUID -o value "${DEVICE_REAL}")"
  fs_type="$(lsblk -no FSTYPE "${DEVICE_REAL}" | head -n 1 | tr -d '[:space:]')"
  if [[ -z "${uuid}" || -z "${fs_type}" ]]; then
    echo "mount-disk: could not resolve UUID/filesystem for ${DEVICE_REAL}" >&2
    exit 1
  fi
  if ! ${SUDO} grep -Fq "UUID=${uuid} " /etc/fstab; then
    printf 'UUID=%s %s %s discard,defaults,nofail 0 2\n' "${uuid}" "${MOUNT_POINT}" "${fs_type}" \
      | ${SUDO} tee -a /etc/fstab >/dev/null
    echo "mount-disk: added /etc/fstab entry for UUID=${uuid}"
  else
    echo "mount-disk: /etc/fstab already contains UUID=${uuid}"
  fi
fi

echo "mount-disk: mounted ${DEVICE_REAL} at ${MOUNT_POINT}"
findmnt -rn -T "${MOUNT_POINT}" -o SOURCE,TARGET,FSTYPE,OPTIONS
