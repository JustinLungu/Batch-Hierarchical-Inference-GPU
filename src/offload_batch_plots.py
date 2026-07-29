from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class OffloadBatchPlotter:
    """Plot request latency scaling against the actual offloaded batch size."""

    CONFIG_COLORS = {
        "005": "#1f77b4",
        "006": "#ff7f0e",
        "007": "#2ca02c",
    }

    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir

    def write_plots(
        self,
        measurements: pd.DataFrame,
        grouped_summary: pd.DataFrame,
    ) -> list[Path]:
        matplotlib_cache = Path("/tmp") / "matplotlib-offload-batch-analysis"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        import matplotlib.pyplot as plt

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plots = [
            self.write_metric_plot(
                plt,
                measurements,
                grouped_summary,
                raw_column="server_response_time_s",
                median_column="response_time_median_s",
                p25_column="response_time_p25_s",
                p75_column="response_time_p75_s",
                title="Server Response Time by Actual Offload Batch Size",
                ylabel="Server response time (s)",
                filename="server_response_time_by_batch_size.png",
            ),
            self.write_metric_plot(
                plt,
                measurements,
                grouped_summary,
                raw_column="per_image_server_time_s",
                median_column="per_image_time_median_s",
                p25_column="per_image_time_p25_s",
                p75_column="per_image_time_p75_s",
                title="Per-Image Server Time by Actual Offload Batch Size",
                ylabel="Server time per image (s/image)",
                filename="per_image_server_time_by_batch_size.png",
            ),
            self.write_metric_plot(
                plt,
                measurements,
                grouped_summary,
                raw_column="effective_server_throughput_samples_s",
                median_column="throughput_median_samples_s",
                p25_column="throughput_p25_samples_s",
                p75_column="throughput_p75_samples_s",
                title="Effective Server Throughput by Actual Offload Batch Size",
                ylabel="Effective throughput (samples/s)",
                filename="server_throughput_by_batch_size.png",
            ),
        ]
        return plots

    def write_metric_plot(
        self,
        plt,
        measurements: pd.DataFrame,
        grouped_summary: pd.DataFrame,
        *,
        raw_column: str,
        median_column: str,
        p25_column: str,
        p75_column: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        figure, axis = plt.subplots(figsize=(10, 6))
        for config_id in sorted(measurements["config"].unique()):
            color = self.CONFIG_COLORS.get(config_id)
            raw = measurements[measurements["config"] == config_id].sort_values(
                "actual_server_batch_size"
            )
            summary = grouped_summary[
                grouped_summary["config"] == config_id
            ].sort_values("actual_server_batch_size")
            axis.scatter(
                raw["actual_server_batch_size"],
                raw[raw_column],
                color=color,
                alpha=0.12,
                s=15,
                edgecolors="none",
            )
            axis.plot(
                summary["actual_server_batch_size"],
                summary[median_column],
                color=color,
                marker="o",
                linewidth=2,
                markersize=4,
                label=f"Config {config_id} median",
            )
            axis.fill_between(
                summary["actual_server_batch_size"],
                summary[p25_column],
                summary[p75_column],
                color=color,
                alpha=0.15,
            )

        axis.set_title(title)
        axis.set_xlabel("Actual server batch size (offloaded samples/request)")
        axis.set_ylabel(ylabel)
        axis.grid(linestyle="--", alpha=0.5)
        axis.set_axisbelow(True)
        axis.legend()
        figure.tight_layout()

        path = self.plots_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
