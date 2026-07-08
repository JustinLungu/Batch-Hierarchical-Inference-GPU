#!/usr/bin/env bash
set -euo pipefail

source scripts/lib/load_env.sh

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install uv first, then rerun this script."
  exit 1
fi

echo "Syncing ExPECA notebook dependencies into ${VENV_DIR}"
uv sync

echo
echo "ExPECA notebook environment ready."
echo "Restart the notebook kernel before rerunning the import cell."
