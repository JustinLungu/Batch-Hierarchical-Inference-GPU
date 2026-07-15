#!/usr/bin/env bash
set -euo pipefail

source scripts/lib/load_env.sh

EXPECA_IMAGE_NAMESPACE="${EXPECA_IMAGE_NAMESPACE:-}"
EXPECA_EDGE_DEVICE_ARM64_IMAGE_TAG="${EXPECA_EDGE_DEVICE_ARM64_IMAGE_TAG:-cpu-arm64-001}"
EXPECA_ARM64_IMAGE_PLATFORM="${EXPECA_ARM64_IMAGE_PLATFORM:-linux/arm64}"

if [[ -z "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "EXPECA_IMAGE_NAMESPACE is required."
  echo "Set it in config/experiment.env before running this script."
  exit 1
fi

EDGE_DEVICE_ARM64_IMAGE="${EXPECA_EDGE_DEVICE_ARM64_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-device:${EXPECA_EDGE_DEVICE_ARM64_IMAGE_TAG}}"

echo "Building ExPECA Raspberry Pi edge-device image"
echo "  platform:    ${EXPECA_ARM64_IMAGE_PLATFORM}"
echo "  edge device: ${EDGE_DEVICE_ARM64_IMAGE}"
echo

if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --platform "$EXPECA_ARM64_IMAGE_PLATFORM" \
    --load \
    -f app/edge_device/Dockerfile.edge_device \
    -t "$EDGE_DEVICE_ARM64_IMAGE" \
    .
else
  echo
  echo "docker buildx is unavailable; using plain docker build."
  echo "This requires Docker/binfmt support for ${EXPECA_ARM64_IMAGE_PLATFORM} on this host."
  echo

  docker build \
    --platform "$EXPECA_ARM64_IMAGE_PLATFORM" \
    -f app/edge_device/Dockerfile.edge_device \
    -t "$EDGE_DEVICE_ARM64_IMAGE" \
    .
fi

echo
echo "Built Raspberry Pi edge-device image:"
echo "  ${EDGE_DEVICE_ARM64_IMAGE}"
echo
echo "Push it with:"
echo "  scripts/push_expeca_raspberry_pi_image.sh"
