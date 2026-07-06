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

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install it first, then rerun this script: https://docs.astral.sh/uv/"
  exit 1
fi

echo "Syncing project environment with Python ${PYTHON_VERSION}"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --python "$PYTHON_VERSION"

echo
echo "Environment ready."
echo "Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
