#!/usr/bin/env bash
set -euo pipefail

source scripts/lib/load_env.sh

DATASET_DIR="${DATASET_DIR:-data/datasets}"
DOWNLOAD_TARGET="configured"

usage() {
  cat <<'EOF'
Usage: scripts/download_dataset.sh [--all|--imagenette|--imagenetv2]

Downloads datasets used by this repository.

Default:
  Download the dataset implied by SAMPLE_PATH in config/experiment.env.

Options:
  --all          Download all supported datasets.
  --imagenette   Download Imagenette 160px for smoke tests.
  --imagenetv2   Download ImageNetV2 Matched Frequency for thesis reproduction.
  --help, -h     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      DOWNLOAD_TARGET="all"
      shift
      ;;
    --imagenette)
      DOWNLOAD_TARGET="imagenette"
      shift
      ;;
    --imagenetv2)
      DOWNLOAD_TARGET="imagenetv2"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

count_images() {
  find "$1" -type f \( \
    -iname '*.jpg' -o \
    -iname '*.jpeg' -o \
    -iname '*.png' -o \
    -iname '*.webp' \
  \) | wc -l
}

count_class_dirs() {
  find "$1" -mindepth 1 -maxdepth 1 -type d | wc -l
}

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$output"
  else
    echo "Neither curl nor wget is installed."
    exit 1
  fi
}

extract_tar_auto() {
  local archive="$1"
  local destination="$2"
  if gzip -t "$archive" >/dev/null 2>&1; then
    tar -xzf "$archive" -C "$destination"
  else
    tar -xf "$archive" -C "$destination"
  fi
}

download_imagenette() {
  local name="${IMAGENETTE_NAME:-imagenette2-160}"
  local url="${IMAGENETTE_URL:-https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz}"
  local archive_path="${DATASET_DIR}/${name}.tgz"
  local target_path="${DATASET_DIR}/${name}"

  mkdir -p "$DATASET_DIR"

  if [[ -d "$target_path" ]]; then
    echo "Imagenette already exists at ${target_path}"
    return
  fi

  echo "Downloading Imagenette"
  echo "  from: ${url}"
  echo "  to:   ${archive_path}"
  download_file "$url" "$archive_path"

  echo "Extracting Imagenette into ${DATASET_DIR}"
  tar -xzf "$archive_path" -C "$DATASET_DIR"

  echo "Imagenette ready:"
  echo "  ${target_path}/val"
}

validate_imagenetv2() {
  local path="$1"
  local image_count
  local class_count
  image_count="$(count_images "$path" | tr -d ' ')"
  class_count="$(count_class_dirs "$path" | tr -d ' ')"

  if [[ "$image_count" != "10000" || "$class_count" != "1000" ]]; then
    echo "Invalid ImageNetV2 layout at ${path}"
    echo "  class directories: ${class_count} (expected 1000)"
    echo "  images:            ${image_count} (expected 10000)"
    return 1
  fi

  echo "Validated ImageNetV2 dataset:"
  echo "  ${path}"
  echo "  class directories: ${class_count}"
  echo "  images:            ${image_count}"
}

download_imagenetv2() {
  local root="${IMAGENETV2_DIR:-${DATASET_DIR}/imagenetV2}"
  local split="${IMAGENETV2_SPLIT:-matched-frequency-format-val}"
  local url="${IMAGENETV2_URL:-https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz}"
  local target_path="${root}/${split}"
  local archive_path="${root}/imagenetv2-${split}.tar.gz"
  local tmp_extract_dir="${root}/.extracting-${split}"

  mkdir -p "$root"

  if [[ -d "$target_path" ]]; then
    validate_imagenetv2 "$target_path"
    return
  fi

  if [[ -f "$archive_path" ]] \
    && ! gzip -t "$archive_path" >/dev/null 2>&1 \
    && ! tar -tf "$archive_path" >/dev/null 2>&1; then
    echo "Removing invalid partial archive: ${archive_path}"
    rm -f "$archive_path"
  fi

  if [[ ! -f "$archive_path" ]]; then
    echo "Downloading ImageNetV2 ${split}"
    echo "  from: ${url}"
    echo "  to:   ${archive_path}"
    download_file "$url" "$archive_path"
  else
    echo "ImageNetV2 archive already exists at ${archive_path}"
  fi

  rm -rf "$tmp_extract_dir"
  mkdir -p "$tmp_extract_dir"

  echo "Extracting ImageNetV2 archive..."
  extract_tar_auto "$archive_path" "$tmp_extract_dir"

  local extracted_root
  extracted_root="$(find "$tmp_extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -z "$extracted_root" ]]; then
    echo "Could not find extracted dataset root in ${tmp_extract_dir}"
    exit 1
  fi

  rm -rf "$target_path"
  mv "$extracted_root" "$target_path"
  rm -rf "$tmp_extract_dir"

  validate_imagenetv2 "$target_path"
}

case "$DOWNLOAD_TARGET" in
  all)
    download_imagenette
    download_imagenetv2
    ;;
  imagenette)
    download_imagenette
    ;;
  imagenetv2)
    download_imagenetv2
    ;;
  configured)
    if [[ "${SAMPLE_PATH:-}" == *"imagenetV2"* ]]; then
      download_imagenetv2
    else
      download_imagenette
    fi
    ;;
esac

echo
echo "Dataset preparation complete."
