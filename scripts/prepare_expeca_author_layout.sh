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

SOURCE_IMAGENETTE_VAL="${SOURCE_IMAGENETTE_VAL:-data/datasets/imagenette2-160/val}"
TARGET_IMAGENETTE_VAL="${TARGET_IMAGENETTE_VAL:-data/datasets/imagenette/val_renamed}"
IMAGENETV2_DIR="${IMAGENETV2_DIR:-data/datasets/imagenetV2}"
MODE="${MODE:-copy}"

usage() {
  cat <<'EOF'
Usage: scripts/prepare_expeca_author_layout.sh

Creates the dataset paths expected by the original ExPECA edge-device
Dockerfile:

  data/datasets/imagenette/val_renamed/
  data/datasets/imagenetV2/

Defaults copy Imagenette from data/datasets/imagenette2-160/val.

Environment overrides:
  SOURCE_IMAGENETTE_VAL   Source validation split.
  TARGET_IMAGENETTE_VAL   Target path copied into the Docker image.
  IMAGENETV2_DIR          Directory created so Docker COPY succeeds.
  MODE                    copy or symlink. Default: copy.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$SOURCE_IMAGENETTE_VAL" ]]; then
  echo "Missing source dataset: ${SOURCE_IMAGENETTE_VAL}"
  echo "Run scripts/download_dataset.sh first, or set SOURCE_IMAGENETTE_VAL."
  exit 1
fi

mkdir -p "$(dirname "$TARGET_IMAGENETTE_VAL")"
mkdir -p "$IMAGENETV2_DIR"
touch "${IMAGENETV2_DIR}/.dockerkeep"

if [[ -e "$TARGET_IMAGENETTE_VAL" ]]; then
  echo "Dataset target already exists: ${TARGET_IMAGENETTE_VAL}"
else
  case "$MODE" in
    copy)
      echo "Copying ${SOURCE_IMAGENETTE_VAL} to ${TARGET_IMAGENETTE_VAL}"
      cp -a "$SOURCE_IMAGENETTE_VAL" "$TARGET_IMAGENETTE_VAL"
      ;;
    symlink)
      echo "Symlinking ${TARGET_IMAGENETTE_VAL} -> ${SOURCE_IMAGENETTE_VAL}"
      ln -s "$(realpath "$SOURCE_IMAGENETTE_VAL")" "$TARGET_IMAGENETTE_VAL"
      ;;
    *)
      echo "Unsupported MODE='${MODE}'. Use copy or symlink."
      exit 1
      ;;
  esac
fi

echo
echo "Author Dockerfile dataset paths are ready:"
echo "  ${TARGET_IMAGENETTE_VAL}"
echo "  ${IMAGENETV2_DIR}"
echo
echo "For the first ExPECA CPU baseline, select this dataset inside /app/start.sh:"
echo "  imagenette/val_renamed/"
