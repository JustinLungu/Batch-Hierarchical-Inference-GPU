#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config/experiment.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

DATASET_DIR="${DATASET_DIR:-data/datasets}"
DATASET_NAME="${DATASET_NAME:-imagenette2-160}"
DATASET_URL="${DATASET_URL:-https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz}"
ARCHIVE_PATH="${DATASET_DIR}/${DATASET_NAME}.tgz"
TARGET_PATH="${DATASET_DIR}/${DATASET_NAME}"

mkdir -p "$DATASET_DIR"

if [[ -d "$TARGET_PATH" ]]; then
  echo "Dataset already exists at ${TARGET_PATH}"
  exit 0
fi

echo "Downloading ${DATASET_NAME}"
echo "  from: ${DATASET_URL}"
echo "  to:   ${ARCHIVE_PATH}"

if command -v curl >/dev/null 2>&1; then
  curl -L "$DATASET_URL" -o "$ARCHIVE_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget "$DATASET_URL" -O "$ARCHIVE_PATH"
else
  echo "Neither curl nor wget is installed."
  exit 1
fi

echo "Extracting dataset into ${DATASET_DIR}"
tar -xzf "$ARCHIVE_PATH" -C "$DATASET_DIR"

echo
echo "Dataset ready at ${TARGET_PATH}"
echo "Validation split for controllers:"
echo "  ${TARGET_PATH}/val"
