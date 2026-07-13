#!/usr/bin/env bash
set -euo pipefail

source scripts/lib/load_env.sh

EXPECA_IMAGE_NAMESPACE="${EXPECA_IMAGE_NAMESPACE:-}"
EXPECA_GPU_EDGE_SERVER_IMAGE_TAG="${EXPECA_GPU_EDGE_SERVER_IMAGE_TAG:-gpu-amd64-001}"
EXPECA_IMAGE_PLATFORM="${EXPECA_IMAGE_PLATFORM:-linux/amd64}"

if [[ -z "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "EXPECA_IMAGE_NAMESPACE is required."
  echo "Set it in config/experiment.env before running this script."
  exit 1
fi

EDGE_SERVER_IMAGE="${EDGE_SERVER_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-server:${EXPECA_GPU_EDGE_SERVER_IMAGE_TAG}}"

echo "Building ExPECA GPU edge-server image"
echo "  platform:    ${EXPECA_IMAGE_PLATFORM}"
echo "  edge server: ${EDGE_SERVER_IMAGE}"
echo

if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --platform "$EXPECA_IMAGE_PLATFORM" \
    -f app/edge_server/Dockerfile.edge_server.gpu \
    -t "$EDGE_SERVER_IMAGE" \
    .
else
  echo
  echo "docker buildx is unavailable; using plain docker build."
  echo "This builds for the host architecture."
  echo

  docker build \
    -f app/edge_server/Dockerfile.edge_server.gpu \
    -t "$EDGE_SERVER_IMAGE" \
    .
fi

echo
echo "Built GPU edge-server image:"
echo "  ${EDGE_SERVER_IMAGE}"
echo
echo "Push it with:"
echo "  scripts/push_expeca_gpu_server_image.sh"
