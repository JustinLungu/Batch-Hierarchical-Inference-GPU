from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def validate_raw_columns(cls, raw_results: pd.DataFrame) -> None:
        missing = sorted(cls.REQUIRED_COLUMNS.difference(raw_results.columns))
        if missing:
            raise ValueError(
                "Raw edge-device results are missing required batch-analysis columns: "
                + ", ".join(missing)
            )

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
