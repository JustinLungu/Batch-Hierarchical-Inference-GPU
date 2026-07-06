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

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
DOWNLOAD_ALL=false

usage() {
  cat <<'EOF'
Usage: scripts/download_models.sh [--all]

Downloads torchvision model state_dict files into data/models/.

Default:
  Download the configured SML/LML pair from config/experiment.env.

Options:
  --all, -a   Download every architecture currently supported by the repo.
  --help, -h  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all|-a)
      DOWNLOAD_ALL=true
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

SML_ARCH="${SML_ARCH:-mobilenet_v3_large}"
SML_MODEL="${SML_MODEL:-data/models/sml/mobilenet_v3_large_imagenet1k_v2.pth}"
LML_ARCH="${LML_ARCH:-wide_resnet50_2}"
LML_MODEL="${LML_MODEL:-data/models/lml/Wide_ResNet50_2_Weights_IMAGENET1K_V2.pth}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}"
  echo "Run scripts/setup_env.sh first, or set PYTHON_BIN."
  exit 1
fi

"$PYTHON_BIN" - "$DOWNLOAD_ALL" "$SML_ARCH" "$SML_MODEL" "$LML_ARCH" "$LML_MODEL" <<'PY'
import os
import sys
import torch
from torchvision import models

download_all = sys.argv[1].lower() == "true"
sml_arch, sml_path, lml_arch, lml_path = sys.argv[2:6]

MODEL_SPECS = {
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "data/models/sml/mobilenet_v3_large_imagenet1k_v2.pth",
    ),
    "efficientnet_b3": (
        models.efficientnet_b3,
        models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "data/models/sml/EfficientNet_B3_Weights_IMAGENET1K_V1.pth",
    ),
    "resnet34": (
        models.resnet34,
        models.ResNet34_Weights.IMAGENET1K_V1,
        "data/models/sml/ResNet34_Weights_IMAGENET1K_V1.pth",
    ),
    "wide_resnet50_2": (
        models.wide_resnet50_2,
        models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,
        "data/models/lml/Wide_ResNet50_2_Weights_IMAGENET1K_V2.pth",
    ),
    "efficientnet_v2_l": (
        models.efficientnet_v2_l,
        models.EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        "data/models/lml/EfficientNet_V2_L_Weights_IMAGENET1K_V1.pth",
    ),
    "vit_h_14": (
        models.vit_h_14,
        models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
        "data/models/lml/ViT_H_14_Weights_IMAGENET1K_SWAG_E2E_V1.pth",
    ),
}

def save_weights(arch: str, path: str) -> None:
    if arch not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise SystemExit(f"Unsupported architecture '{arch}'. Supported: {supported}")

    if os.path.exists(path):
        print(f"Model already exists at {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    constructor, weights, _ = MODEL_SPECS[arch]
    print(f"Downloading torchvision weights for {arch}")
    model = constructor(weights=weights)
    torch.save(model.state_dict(), path)
    print(f"Saved {arch} state_dict to {path}")

if download_all:
    for arch, (_, _, path) in MODEL_SPECS.items():
        save_weights(arch, path)
else:
    save_weights(sml_arch, sml_path)
    save_weights(lml_arch, lml_path)
PY

echo
if [[ "$DOWNLOAD_ALL" == true ]]; then
  echo "All supported models are ready under data/models/"
else
  echo "Models ready:"
  echo "  SML: ${SML_MODEL}"
  echo "  LML: ${LML_MODEL}"
fi
