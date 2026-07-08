#!/usr/bin/env bash
set -euo pipefail

source scripts/lib/load_env.sh

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

echo "Checking ExPECA thesis reproduction CPU image prerequisites"
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
  echo "     Set it in config/experiment.env before build/push."
  failures=$((failures + 1))
fi

echo "INFO EXPECA_IMAGE_TAG=${EXPECA_IMAGE_TAG}"
echo "INFO EXPECA_IMAGE_PLATFORM=${EXPECA_IMAGE_PLATFORM}"

echo
if [[ "$failures" -eq 0 ]]; then
  echo "All ExPECA CPU thesis reproduction prerequisites look ready."
else
  echo "${failures} prerequisite check(s) failed."
  echo
  echo "Typical preparation order:"
  echo "  scripts/download_dataset.sh"
  echo "  scripts/download_models.sh --all"
  echo "  scripts/prepare_expeca_author_layout.sh"
  echo "  edit config/experiment.env"
  echo "  scripts/check_expeca_cpu_prereqs.sh"
  exit 1
fi
