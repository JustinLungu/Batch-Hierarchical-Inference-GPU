import mimetypes
import os
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


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


def is_valid_image(image_path: str) -> bool:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        return False

    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def true_class_label(image_path: str, class_index: int) -> int:
    parent_name = Path(image_path).parent.name
    if parent_name.isdigit():
        return int(parent_name)
    if parent_name.startswith("class_"):
        suffix = parent_name.removeprefix("class_")
        if suffix.isdigit():
            return int(suffix)
    return class_index


def count_true(data: pd.DataFrame, column: str) -> int | None:
    if column not in data.columns:
        return None
    return int(data[column].astype(str).str.lower().eq("true").sum())


def numeric_mean(data: pd.DataFrame, column: str) -> float | None:
    if column not in data.columns:
        return None
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def numeric_median(data: pd.DataFrame, column: str) -> float | None:
    if column not in data.columns:
        return None
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def offloaded_mask(timing: pd.DataFrame) -> pd.Series:
    if "Offloaded" in timing.columns:
        return timing["Offloaded"].astype(str).str.lower().eq("true")
    if "LML Prediction" in timing.columns:
        return pd.to_numeric(timing["LML Prediction"], errors="coerce").notna()
    if "lml_inference_s" in timing.columns:
        return pd.to_numeric(timing["lml_inference_s"], errors="coerce").notna()
    return pd.Series([False] * len(timing), index=timing.index)


def group_accuracy(
    true_class: pd.Series,
    prediction: pd.Series,
    mask: pd.Series,
) -> float | None:
    valid = true_class.notna() & prediction.notna() & mask
    if not valid.any():
        return None
    return float((prediction[valid] == true_class[valid]).mean())


def duration_sum_mean(
    timing: pd.DataFrame,
    columns: list[str],
    mask: pd.Series | None = None,
) -> float:
    if timing.empty:
        return 0.0
    selected = timing if mask is None else timing[mask]
    if selected.empty:
        return 0.0
    total = pd.Series([0.0] * len(selected), index=selected.index)
    for column in columns:
        if column in selected.columns:
            total += pd.to_numeric(selected[column], errors="coerce").fillna(0.0)
    return float(total.mean())


def series_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def optional_float(value) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def format_seconds(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}s"


def format_float(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def format_percent(value) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def approximate_throughput(timing: pd.DataFrame) -> float | None:
    start = pd.to_numeric(
        timing["ts_sml_inference_start"], errors="coerce"
    ).min()
    end_candidates = timing["ts_results_received_from_offloading_module"].fillna(
        timing.get("ts_results_saved_not_offloaded")
    )
    end = pd.to_numeric(end_candidates, errors="coerce").max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        return None
    return len(timing) / (end - start)


def apply_thesis_axes_style(axis) -> None:
    axis.grid(axis="y", linestyle="--", alpha=0.6)
    axis.set_axisbelow(True)


def add_figure_caption(figure, figure_id: str, title: str) -> None:
    figure.subplots_adjust(bottom=0.18)
    figure.text(0.42, 0.035, figure_id, ha="right", fontsize=14, fontweight="bold")
    figure.text(0.50, 0.035, title, ha="left", fontsize=14, fontweight="bold")


def annotate_bars(axis, bars, fmt="{:.1f}", rotation=25, color="black") -> None:
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height) or height == 0:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=8,
            color=color,
        )
