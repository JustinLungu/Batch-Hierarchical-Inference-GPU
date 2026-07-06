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

if [[ -z "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "EXPECA_IMAGE_NAMESPACE is required."
  echo "Set it in config/experiment.env before running this script."
  exit 1
fi

EDGE_SERVER_IMAGE="${EDGE_SERVER_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-server:${EXPECA_IMAGE_TAG}}"
EDGE_DEVICE_IMAGE="${EDGE_DEVICE_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-device:${EXPECA_IMAGE_TAG}}"

echo "Building ExPECA CPU images from the original Dockerfiles"
echo "  platform:    ${EXPECA_IMAGE_PLATFORM}"
echo "  edge server: ${EDGE_SERVER_IMAGE}"
echo "  edge device: ${EDGE_DEVICE_IMAGE}"
echo

scripts/check_expeca_cpu_prereqs.sh

if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --platform "$EXPECA_IMAGE_PLATFORM" \
    -f app/edge_server/Dockerfile.edge_server \
    -t "$EDGE_SERVER_IMAGE" \
    .

  docker buildx build \
    --platform "$EXPECA_IMAGE_PLATFORM" \
    -f app/edge_device/Dockerfile.edge_device \
    -t "$EDGE_DEVICE_IMAGE" \
    .
else
  echo
  echo "docker buildx is unavailable; using plain docker build."
  echo "This builds for the host architecture."
  echo

  docker build \
    -f app/edge_server/Dockerfile.edge_server \
    -t "$EDGE_SERVER_IMAGE" \
    .

  docker build \
    -f app/edge_device/Dockerfile.edge_device \
    -t "$EDGE_DEVICE_IMAGE" \
    .
fi

echo
echo "Built images:"
echo "  ${EDGE_SERVER_IMAGE}"
echo "  ${EDGE_DEVICE_IMAGE}"
echo
echo "Push them with:"
echo "  scripts/push_expeca_cpu_images.sh"
