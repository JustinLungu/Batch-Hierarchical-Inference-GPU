#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config/experiment.env}"
if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
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
