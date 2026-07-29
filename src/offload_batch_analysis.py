from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OffloadBatchContext:
    config_id: str
    controller_batch_size: int


class OffloadBatchAnalyzer:
    """Collapse per-sample raw rows into one measurement per server request."""

    REQUIRED_COLUMNS = {
        "ts_sample_sent_to_edge_server",
        "ts_sample_received_at_edge_server",
        "ts_lml_inference_start",
        "ts_lml_inference_end",
        "ts_results_sent_to_edge_device",
    }

    OUTPUT_COLUMNS = [
        "config",
        "controller_batch_size",
        "edge_server_batch_id",
        "actual_server_batch_size",
        "server_received_at",
        "server_results_sent_at",
        "lml_micro_batches_observed",
        "server_queue_or_preprocess_s",
        "lml_wall_time_s",
        "server_postprocess_s",
        "server_response_time_s",
        "per_image_server_time_s",
        "effective_server_throughput_samples_s",
    ]

    GROUPED_SUMMARY_COLUMNS = [
        "config",
        "controller_batch_size",
        "actual_server_batch_size",
        "batch_count",
        "batch_share_percent",
        "response_time_mean_s",
        "response_time_median_s",
        "response_time_std_s",
        "response_time_p25_s",
        "response_time_p75_s",
        "response_time_p95_s",
        "per_image_time_mean_s",
        "per_image_time_median_s",
        "per_image_time_p25_s",
        "per_image_time_p75_s",
        "per_image_time_p95_s",
        "throughput_mean_samples_s",
        "throughput_median_samples_s",
        "throughput_p25_samples_s",
        "throughput_p75_samples_s",
        "throughput_p95_samples_s",
        "lml_wall_time_mean_s",
        "micro_batches_mean",
    ]

    TREND_COLUMNS = [
        "config",
        "controller_batch_size",
        "request_count",
        "actual_batch_size_min",
        "actual_batch_size_max",
        "actual_batch_size_mean",
        "actual_batch_size_median",
        "response_time_pearson_r",
        "response_time_spearman_r",
        "response_time_slope_s_per_sample",
        "response_time_linear_r_squared",
        "per_image_time_spearman_r",
        "throughput_spearman_r",
    ]

    def extract_batch_measurements(
        self,
        context: OffloadBatchContext,
        raw_results: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_raw_columns(raw_results)
        offloaded = raw_results.dropna(subset=["ts_sample_sent_to_edge_server"]).copy()
        if offloaded.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        for column in self.REQUIRED_COLUMNS:
            offloaded[column] = pd.to_numeric(offloaded[column], errors="coerce")

        rows = []
        grouped = offloaded.groupby("ts_sample_sent_to_edge_server", sort=True)
        for batch_id, (_, batch) in enumerate(grouped):
            received_at = self.single_batch_value(
                batch, "ts_sample_received_at_edge_server", batch_id
            )
            results_sent_at = self.single_batch_value(
                batch, "ts_results_sent_to_edge_device", batch_id
            )
            lml_start = self.minimum_batch_value(
                batch, "ts_lml_inference_start", batch_id
            )
            lml_end = self.maximum_batch_value(batch, "ts_lml_inference_end", batch_id)
            response_time = results_sent_at - received_at
            batch_size = len(batch)
            micro_batches = batch[
                ["ts_lml_inference_start", "ts_lml_inference_end"]
            ].drop_duplicates()
            rows.append(
                {
                    "config": context.config_id,
                    "controller_batch_size": context.controller_batch_size,
                    "edge_server_batch_id": batch_id,
                    "actual_server_batch_size": batch_size,
                    "server_received_at": received_at,
                    "server_results_sent_at": results_sent_at,
                    "lml_micro_batches_observed": len(micro_batches),
                    "server_queue_or_preprocess_s": lml_start - received_at,
                    "lml_wall_time_s": lml_end - lml_start,
                    "server_postprocess_s": results_sent_at - lml_end,
                    "server_response_time_s": response_time,
                    "per_image_server_time_s": response_time / batch_size,
                    "effective_server_throughput_samples_s": (
                        batch_size / response_time if response_time > 0 else pd.NA
                    ),
                }
            )

        return pd.DataFrame(rows, columns=self.OUTPUT_COLUMNS)

    def summarize_by_batch_size(
        self,
        measurements: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_measurement_columns(measurements)
        if measurements.empty:
            return pd.DataFrame(columns=self.GROUPED_SUMMARY_COLUMNS)

        rows = []
        for (config_id, controller_size, actual_size), group in measurements.groupby(
            ["config", "controller_batch_size", "actual_server_batch_size"],
            sort=True,
        ):
            config_request_count = int(
                (measurements["config"] == config_id).sum()
            )
            response = self.numeric_series(group, "server_response_time_s")
            per_image = self.numeric_series(group, "per_image_server_time_s")
            throughput = self.numeric_series(
                group, "effective_server_throughput_samples_s"
            )
            lml_wall = self.numeric_series(group, "lml_wall_time_s")
            micro_batches = self.numeric_series(group, "lml_micro_batches_observed")
            rows.append(
                {
                    "config": config_id,
                    "controller_batch_size": int(controller_size),
                    "actual_server_batch_size": int(actual_size),
                    "batch_count": len(group),
                    "batch_share_percent": 100.0 * len(group) / config_request_count,
                    "response_time_mean_s": response.mean(),
                    "response_time_median_s": response.median(),
                    "response_time_std_s": response.std(),
                    "response_time_p25_s": response.quantile(0.25),
                    "response_time_p75_s": response.quantile(0.75),
                    "response_time_p95_s": response.quantile(0.95),
                    "per_image_time_mean_s": per_image.mean(),
                    "per_image_time_median_s": per_image.median(),
                    "per_image_time_p25_s": per_image.quantile(0.25),
                    "per_image_time_p75_s": per_image.quantile(0.75),
                    "per_image_time_p95_s": per_image.quantile(0.95),
                    "throughput_mean_samples_s": throughput.mean(),
                    "throughput_median_samples_s": throughput.median(),
                    "throughput_p25_samples_s": throughput.quantile(0.25),
                    "throughput_p75_samples_s": throughput.quantile(0.75),
                    "throughput_p95_samples_s": throughput.quantile(0.95),
                    "lml_wall_time_mean_s": lml_wall.mean(),
                    "micro_batches_mean": micro_batches.mean(),
                }
            )

        return pd.DataFrame(rows, columns=self.GROUPED_SUMMARY_COLUMNS)

    def calculate_config_trends(
        self,
        measurements: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validate_measurement_columns(measurements)
        if measurements.empty:
            return pd.DataFrame(columns=self.TREND_COLUMNS)

        rows = []
        for config_id, group in measurements.groupby("config", sort=True):
            batch_size = self.numeric_series(group, "actual_server_batch_size")
            response = self.numeric_series(group, "server_response_time_s")
            per_image = self.numeric_series(group, "per_image_server_time_s")
            throughput = self.numeric_series(
                group, "effective_server_throughput_samples_s"
            )
            slope, r_squared = self.linear_trend(batch_size, response)
            rows.append(
                {
                    "config": config_id,
                    "controller_batch_size": int(
                        self.single_numeric_value(group, "controller_batch_size")
                    ),
                    "request_count": len(group),
                    "actual_batch_size_min": int(batch_size.min()),
                    "actual_batch_size_max": int(batch_size.max()),
                    "actual_batch_size_mean": batch_size.mean(),
                    "actual_batch_size_median": batch_size.median(),
                    "response_time_pearson_r": self.correlation(
                        batch_size, response, "pearson"
                    ),
                    "response_time_spearman_r": self.correlation(
                        batch_size, response, "spearman"
                    ),
                    "response_time_slope_s_per_sample": slope,
                    "response_time_linear_r_squared": r_squared,
                    "per_image_time_spearman_r": self.correlation(
                        batch_size, per_image, "spearman"
                    ),
                    "throughput_spearman_r": self.correlation(
                        batch_size, throughput, "spearman"
                    ),
                }
            )

        return pd.DataFrame(rows, columns=self.TREND_COLUMNS)

    @classmethod
    def validate_raw_columns(cls, raw_results: pd.DataFrame) -> None:
        missing = sorted(cls.REQUIRED_COLUMNS.difference(raw_results.columns))
        if missing:
            raise ValueError(
                "Raw edge-device results are missing required batch-analysis columns: "
                + ", ".join(missing)
            )

    @classmethod
    def validate_measurement_columns(cls, measurements: pd.DataFrame) -> None:
        missing = sorted(set(cls.OUTPUT_COLUMNS).difference(measurements.columns))
        if missing:
            raise ValueError(
                "Batch measurements are missing required columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Batch measurements contain invalid values for {column}.")
        return values.astype(float)

    @staticmethod
    def single_numeric_value(frame: pd.DataFrame, column: str) -> float:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise ValueError(
                f"Expected one value for {column} within a configuration, "
                f"found {len(values)}."
            )
        return float(values[0])

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def minimum_batch_value(
        batch: pd.DataFrame,
        column: str,
        batch_id: object,
    ) -> float:
        values = pd.to_numeric(batch[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Server batch {batch_id!r} has no value for {column}.")
        return float(values.min())

    @staticmethod
    def maximum_batch_value(
        batch: pd.DataFrame,
        column: str,
        batch_id: object,
    ) -> float:
        values = pd.to_numeric(batch[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Server batch {batch_id!r} has no value for {column}.")
        return float(values.max())
