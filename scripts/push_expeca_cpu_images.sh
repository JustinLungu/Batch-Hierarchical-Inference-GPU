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

if [[ -z "$EXPECA_IMAGE_NAMESPACE" ]]; then
  echo "EXPECA_IMAGE_NAMESPACE is required."
  echo "Example:"
  echo "  EXPECA_IMAGE_NAMESPACE=your_dockerhub_user scripts/push_expeca_cpu_images.sh"
  exit 1
fi

EDGE_SERVER_IMAGE="${EDGE_SERVER_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-server:${EXPECA_IMAGE_TAG}}"
EDGE_DEVICE_IMAGE="${EDGE_DEVICE_IMAGE:-${EXPECA_IMAGE_NAMESPACE}/hi-framework-edge-device:${EXPECA_IMAGE_TAG}}"

echo "Pushing ExPECA CPU images"
echo "  ${EDGE_SERVER_IMAGE}"
echo "  ${EDGE_DEVICE_IMAGE}"
echo

docker push "$EDGE_SERVER_IMAGE"
docker push "$EDGE_DEVICE_IMAGE"

echo
echo "Images pushed. Use these image names in notebooks/ExPECA_HI_setup_Public_IP.ipynb:"
echo "  ${EDGE_SERVER_IMAGE}"
echo "  ${EDGE_DEVICE_IMAGE}"
