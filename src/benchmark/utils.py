from pathlib import Path

import pandas as pd


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
