#!/usr/bin/env bash

load_env_stack() {
  local defaults_file="${DEFAULT_CONFIG_FILE:-config/defaults.env}"
  local experiment_file="${CONFIG_FILE:-config/experiment.env}"
  local protected_names=" ${BHI_ENV_OVERRIDES:-} "

  remember_existing_env() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    while IFS='=' read -r name _value; do
      name="${name#"${name%%[![:space:]]*}"}"
      name="${name%"${name##*[![:space:]]}"}"
      [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
      [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
      if [[ -n "${!name+x}" ]]; then
        protected_names+="${name} "
      fi
    done < "$file"
  }

  load_env_file() {
    local file="$1"
    local override_file_values="$2"
    [[ -f "$file" ]] || return 0
    while IFS='=' read -r name value; do
      name="${name#"${name%%[![:space:]]*}"}"
      name="${name%"${name##*[![:space:]]}"}"
      [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
      [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
      if [[ "$protected_names" == *" ${name} "* ]]; then
        continue
      fi
      if [[ "$override_file_values" == "true" || -z "${!name+x}" ]]; then
        export "${name}=${value}"
      fi
    done < "$file"
  }

  remember_existing_env "$defaults_file"
  remember_existing_env "$experiment_file"
  load_env_file "$defaults_file" false
  load_env_file "$experiment_file" true
}

load_env_stack
