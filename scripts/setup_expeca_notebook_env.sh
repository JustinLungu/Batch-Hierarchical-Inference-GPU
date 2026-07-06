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

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install uv first, then rerun this script."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}"
  echo "Run scripts/setup_env.sh first, or set PYTHON_BIN."
  exit 1
fi

echo "Installing ExPECA notebook dependencies into ${PYTHON_BIN}"
uv pip install \
  --python "$PYTHON_BIN" \
  jedi \
  loguru \
  git+https://github.com/KTH-EXPECA/python-chi

echo
echo "ExPECA notebook environment ready."
echo "Restart the notebook kernel before rerunning the import cell."
