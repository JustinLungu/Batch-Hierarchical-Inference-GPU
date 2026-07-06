#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config/experiment.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  while IFS='=' read -r name value; do
    [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!name+x}" ]]; then
      export "${name}=${value}"
    fi
  done < "$CONFIG_FILE"
fi

EXPECA_IMAGE_NAMESPACE="${EXPECA_IMAGE_NAMESPACE:-}"
EXPECA_IMAGE_TAG="${EXPECA_IMAGE_TAG:-cpu-amd64-001}"
EXPECA_IMAGE_PLATFORM="${EXPECA_IMAGE_PLATFORM:-linux/amd64}"

failures=0

check_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    echo "OK   command: ${command_name}"
  else
    echo "MISS command: ${command_name}"
    failures=$((failures + 1))
  fi
}

check_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "OK   path: ${path}"
  else
    echo "MISS path: ${path}"
    failures=$((failures + 1))
  fi
}

echo "Checking ExPECA public-IP CPU baseline prerequisites"
echo

check_command docker
check_command curl

if docker buildx version >/dev/null 2>&1; then
  echo "OK   docker buildx"
else
  echo "WARN docker buildx is unavailable"
  echo "     build_expeca_cpu_images.sh will fall back to plain docker build."
  echo "     That is fine on native amd64 Linux, but buildx is safer for cross-platform builds."
fi

echo
check_path app/edge_server/Dockerfile.edge_server
check_path app/edge_device/Dockerfile.edge_device
check_path data/models/sml
check_path data/models/lml
check_path data/datasets/imagenette/val_renamed
check_path data/datasets/imagenetV2

echo
if [[ -n "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "OK   EXPECA_IMAGE_NAMESPACE=${EXPECA_IMAGE_NAMESPACE}"
else
  echo "MISS EXPECA_IMAGE_NAMESPACE is empty"
  echo "     Set it to your Docker Hub or registry namespace before build/push."
  failures=$((failures + 1))
fi

echo "INFO EXPECA_IMAGE_TAG=${EXPECA_IMAGE_TAG}"
echo "INFO EXPECA_IMAGE_PLATFORM=${EXPECA_IMAGE_PLATFORM}"

echo
if [[ "$failures" -eq 0 ]]; then
  echo "All ExPECA CPU prerequisites look ready."
else
  echo "${failures} prerequisite check(s) failed."
  echo
  echo "Typical preparation order:"
  echo "  scripts/download_dataset.sh"
  echo "  scripts/download_models.sh --all"
  echo "  scripts/prepare_expeca_author_layout.sh"
  echo "  EXPECA_IMAGE_NAMESPACE=your_namespace scripts/check_expeca_cpu_prereqs.sh"
  exit 1
fi
