#!/usr/bin/env bash
set -euo pipefail

source scripts/lib/load_env.sh

EXPECA_IMAGE_NAMESPACE="${EXPECA_IMAGE_NAMESPACE:-}"
EXPECA_EDGE_DEVICE_ARM64_IMAGE_TAG="${EXPECA_EDGE_DEVICE_ARM64_IMAGE_TAG:-cpu-arm64-001}"

if [[ -z "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "EXPECA_IMAGE_NAMESPACE is required."
  echo "Set it in config/experiment.env before running this script."
  exit 1
fi

EDGE_DEVICE_ARM64_IMAGE="${EXPECA_EDGE_DEVICE_ARM64_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-device:${EXPECA_EDGE_DEVICE_ARM64_IMAGE_TAG}}"

echo "Pushing ExPECA Raspberry Pi edge-device image"
echo "  ${EDGE_DEVICE_ARM64_IMAGE}"
echo

docker push "$EDGE_DEVICE_ARM64_IMAGE"

echo
echo "Raspberry Pi edge-device image pushed. Use this image for the edge-device container:"
echo "  ${EDGE_DEVICE_ARM64_IMAGE}"
