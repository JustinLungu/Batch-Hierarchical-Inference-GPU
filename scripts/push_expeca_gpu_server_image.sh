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
EXPECA_GPU_EDGE_SERVER_IMAGE_TAG="${EXPECA_GPU_EDGE_SERVER_IMAGE_TAG:-gpu-amd64-001}"

if [[ -z "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "EXPECA_IMAGE_NAMESPACE is required."
  echo "Set it in config/experiment.env before running this script."
  exit 1
fi

EDGE_SERVER_IMAGE="${EDGE_SERVER_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-server:${EXPECA_GPU_EDGE_SERVER_IMAGE_TAG}}"

echo "Pushing ExPECA GPU edge-server image"
echo "  ${EDGE_SERVER_IMAGE}"
echo

docker push "$EDGE_SERVER_IMAGE"

echo
echo "GPU edge-server image pushed. Use this image for the edge-server container:"
echo "  ${EDGE_SERVER_IMAGE}"
