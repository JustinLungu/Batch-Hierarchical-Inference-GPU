import subprocess
import time
from pathlib import Path

import pandas as pd
import requests


def load_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def require_config(config: dict[str, str], key: str) -> str:
    import os

    value = os.environ.get(key, config.get(key))
    if value is None or value == "":
        raise RuntimeError(f"Missing required config value: {key}")
    return value


def require_config_bool(config: dict[str, str], key: str) -> bool:
    value = require_config(config, key).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def wait_for_server(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/docs", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"Server did not become ready: {url}")


def start_process(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)


def seconds_between(df: pd.DataFrame, end: str, start: str) -> pd.Series:
    if end not in df.columns or start not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    return df[end] - df[start]


def format_mean_seconds(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "n/a"
    return f"~{clean.mean():.4f}s"


def format_median_seconds(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "n/a"
    return f"~{clean.median():.4f}s"
