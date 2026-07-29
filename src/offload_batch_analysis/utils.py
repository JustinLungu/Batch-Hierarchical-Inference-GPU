import numpy as np
import pandas as pd


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Batch measurements contain invalid values for {column}.")
    return values.astype(float)


def single_numeric_value(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"Expected one value for {column} within a configuration, "
            f"found {len(values)}."
        )
    return float(values[0])


def correlation(
    x_values: pd.Series,
    y_values: pd.Series,
    method: str,
) -> float:
    if x_values.nunique() < 2 or y_values.nunique() < 2:
        return float("nan")
    if method == "spearman":
        x_values = x_values.rank(method="average")
        y_values = y_values.rank(method="average")
    return float(x_values.corr(y_values, method="pearson"))


def linear_trend(
    x_values: pd.Series,
    y_values: pd.Series,
) -> tuple[float, float]:
    if x_values.nunique() < 2 or y_values.nunique() < 2:
        return float("nan"), float("nan")

    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = slope * x_values + intercept
    residual_sum = float(((y_values - predicted) ** 2).sum())
    total_sum = float(((y_values - y_values.mean()) ** 2).sum())
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return float(slope), r_squared


def single_batch_value(
    batch: pd.DataFrame,
    column: str,
    batch_id: object,
) -> float:
    values = pd.to_numeric(batch[column], errors="coerce").dropna()
    unique_values = values.unique()
    if len(unique_values) != 1:
        raise ValueError(
            f"Server batch {batch_id!r} has {len(unique_values)} distinct "
            f"values for {column}; expected one shared request-level timestamp."
        )
    return float(unique_values[0])


def minimum_batch_value(
    batch: pd.DataFrame,
    column: str,
    batch_id: object,
) -> float:
    values = pd.to_numeric(batch[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Server batch {batch_id!r} has no value for {column}.")
    return float(values.min())


def maximum_batch_value(
    batch: pd.DataFrame,
    column: str,
    batch_id: object,
) -> float:
    values = pd.to_numeric(batch[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Server batch {batch_id!r} has no value for {column}.")
    return float(values.max())
