from pathlib import Path

import pandas as pd

from .constants import TIMING_COLUMNS, TIMING_DURATIONS, TIMING_OUTPUT_COLUMNS
from .utils import (
    approximate_throughput,
    count_true,
    format_mean_seconds,
    format_median_seconds,
    numeric_mean,
    numeric_median,
    seconds_between,
)


class BenchmarkResultProcessor:
    """Convert raw service output into normalized timing data and run metrics."""

    def __init__(
        self,
        *,
        raw_results_csv: Path,
        timing_results_csv: Path,
        run_metadata: dict,
    ):
        self.raw_results_csv = raw_results_csv
        self.timing_results_csv = timing_results_csv
        self.run_metadata = run_metadata

    def load_raw_results(self) -> pd.DataFrame:
        results = pd.read_csv(self.raw_results_csv)
        if "UUID" in results.columns:
            results = results.drop_duplicates(subset=["UUID"], keep="last")
        for column in TIMING_COLUMNS:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        return results

    @staticmethod
    def add_timing_durations(results: pd.DataFrame) -> pd.DataFrame:
        timing = results.copy()
        for output_column, (end_column, start_column) in TIMING_DURATIONS.items():
            timing[output_column] = seconds_between(timing, end_column, start_column)

        offloaded_total = seconds_between(
            timing,
            "ts_results_received_from_offloading_module",
            "ts_sml_inference_start",
        )
        local_total = seconds_between(
            timing,
            "ts_results_saved_not_offloaded",
            "ts_sml_inference_start",
        )
        timing["total_tracked_latency_s"] = offloaded_total.fillna(local_total)

        if "ts_sample_sent_to_edge_server" in timing.columns:
            batch_keys = timing["ts_sample_sent_to_edge_server"].fillna(-1)
            timing["edge_server_batch_id"] = pd.factorize(batch_keys)[0]
            timing.loc[batch_keys == -1, "edge_server_batch_id"] = pd.NA

        return timing

    def write_timing_csv(self, timing: pd.DataFrame) -> None:
        available_columns = [
            column for column in TIMING_OUTPUT_COLUMNS if column in timing.columns
        ]
        output = timing[available_columns].copy()

        for column in output.columns:
            if column.endswith("_s"):
                values = pd.to_numeric(output[column], errors="coerce")
                output[column] = values.map(
                    lambda value: "" if pd.isna(value) else f"{value:.6f}"
                )

        if "edge_server_batch_id" in output.columns:
            batch_ids = pd.to_numeric(output["edge_server_batch_id"], errors="coerce")
            output["edge_server_batch_id"] = batch_ids.map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )

        output.to_csv(self.timing_results_csv, index=False)

    def build_summary(self, timing: pd.DataFrame) -> str:
        lines = [
            f"Run: {self.run_metadata['run_name']}",
            f"Rows: {len(timing)}",
        ]

        if "Offloaded" in timing.columns:
            offloaded = timing["Offloaded"].astype(str).str.lower().eq("true")
            lines.append(f"Offloaded: {offloaded.sum()} / {len(timing)}")

        if "Buffered" in timing.columns:
            buffered = timing["Buffered"].astype(str).str.lower().eq("true")
            lines.append(f"Still buffered: {buffered.sum()} / {len(timing)}")

        batch_sizes = self.edge_server_batch_sizes(timing)
        if "edge_server_batch_id" in timing.columns:
            lines.append(f"Edge-server batches observed: {len(batch_sizes)}")
            lines.append(f"Edge-server batch sizes: {batch_sizes}")

        lines.append(
            "Total tracked latency median: "
            f"{format_median_seconds(timing['total_tracked_latency_s'])}"
        )
        lines.append(
            f"SML inference mean: {format_mean_seconds(timing['sml_inference_s'])}"
        )
        lines.append(
            f"LML inference mean: {format_mean_seconds(timing['lml_inference_s'])}"
        )
        lines.append(
            f"Offload roundtrip: {format_mean_seconds(timing['offload_roundtrip_s'])}"
        )

        throughput = approximate_throughput(timing)
        if throughput is not None:
            lines.append(f"Approx throughput: ~{throughput:.2f} samples/s")

        return "\n".join(lines) + "\n"

    def aggregate_metrics(self, timing: pd.DataFrame) -> dict:
        batch_sizes = self.edge_server_batch_sizes(timing)
        metadata = self.run_metadata
        return {
            **metadata,
            "rows": int(len(timing)),
            "offloaded": count_true(timing, "Offloaded"),
            "still_buffered": count_true(timing, "Buffered"),
            "edge_server_batches_observed": len(batch_sizes),
            "edge_server_batch_size_min": min(batch_sizes) if batch_sizes else None,
            "edge_server_batch_size_median": (
                float(pd.Series(batch_sizes).median()) if batch_sizes else None
            ),
            "edge_server_batch_size_max": max(batch_sizes) if batch_sizes else None,
            "total_latency_median_s": numeric_median(
                timing, "total_tracked_latency_s"
            ),
            "total_latency_mean_s": numeric_mean(timing, "total_tracked_latency_s"),
            "sml_inference_mean_s": numeric_mean(timing, "sml_inference_s"),
            "lml_inference_mean_s": numeric_mean(timing, "lml_inference_s"),
            "offload_roundtrip_mean_s": numeric_mean(timing, "offload_roundtrip_s"),
            "throughput_samples_s": approximate_throughput(timing),
        }

    @staticmethod
    def edge_server_batch_sizes(timing: pd.DataFrame) -> list[int]:
        if "edge_server_batch_id" not in timing.columns:
            return []
        return (
            timing.dropna(subset=["edge_server_batch_id"])
            .groupby("edge_server_batch_id")
            .size()
            .tolist()
        )
