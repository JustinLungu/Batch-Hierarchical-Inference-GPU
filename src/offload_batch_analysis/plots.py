from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class OffloadBatchPlotter:
    """Create focused plots for actual offloaded batch-size behavior."""

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
        trends: pd.DataFrame,
    ) -> list[Path]:
        matplotlib_cache = Path("/tmp") / "matplotlib-offload-batch-analysis"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        import matplotlib.pyplot as plt

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.write_batch_distribution(plt, measurements, grouped_summary),
            self.write_response_trend(plt, measurements, grouped_summary, trends),
            self.write_per_image_trend(
                plt, measurements, grouped_summary, trends
            ),
        ]

    def write_batch_distribution(
        self,
        plt,
        measurements: pd.DataFrame,
        grouped_summary: pd.DataFrame,
    ) -> Path:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for axis, config_id in zip(axes, self.config_ids(measurements)):
            summary = self.config_summary(grouped_summary, config_id)
            config_measurements = measurements[
                measurements["config"].astype(str).str.zfill(3) == config_id
            ]
            controller_size = int(summary["controller_batch_size"].iloc[0])
            mean_batch_size = config_measurements["actual_server_batch_size"].mean()
            axis.bar(
                summary["actual_server_batch_size"],
                summary["batch_share_percent"],
                color=self.CONFIG_COLORS[config_id],
                width=0.85,
                alpha=0.85,
            )
            axis.axvline(
                mean_batch_size,
                color="#222222",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean = {mean_batch_size:.1f}",
            )
            axis.set_title(
                f"Config {config_id}\nController batch size = {controller_size}"
            )
            axis.set_xlabel("Actual offloaded batch size")
            axis.set_ylabel("Share of server requests (%)")
            axis.tick_params(axis="y", labelleft=True)
            axis.grid(axis="y", linestyle="--", alpha=0.45)
            axis.set_axisbelow(True)
            axis.legend(frameon=False, loc="upper right")

        figure.suptitle("Observed Server Batch-Size Distribution", fontsize=16)
        figure.tight_layout()
        return self.save(plt, figure, "actual_batch_size_distribution.png")

    def write_response_trend(
        self,
        plt,
        measurements: pd.DataFrame,
        grouped_summary: pd.DataFrame,
        trends: pd.DataFrame,
    ) -> Path:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for axis, config_id in zip(axes, self.config_ids(measurements)):
            summary = self.config_summary(grouped_summary, config_id)
            config_measurements = measurements[
                measurements["config"].astype(str).str.zfill(3) == config_id
            ]
            trend = self.config_trend(trends, config_id)
            x = summary["actual_server_batch_size"]
            mean = summary["response_time_mean_s"]
            confidence = self.confidence_interval(
                summary["response_time_std_s"], summary["batch_count"]
            )
            color = self.CONFIG_COLORS[config_id]

            axis.errorbar(
                x,
                mean,
                yerr=confidence,
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                capsize=3,
                label="Observed mean (95% CI)",
            )
            slope = trend["response_time_slope_s_per_sample"]
            intercept = (
                config_measurements["server_response_time_s"].mean()
                - slope * config_measurements["actual_server_batch_size"].mean()
            )
            axis.plot(
                x,
                intercept + slope * x,
                color="#222222",
                linestyle="--",
                linewidth=1.4,
                label="Linear fit",
            )
            axis.text(
                0.04,
                0.95,
                (
                    f"Requests = {int(trend['request_count'])}\n"
                    f"Slope = {slope:.3f} s/sample\n"
                    f"$R^2$ = {trend['response_time_linear_r_squared']:.3f}"
                ),
                transform=axis.transAxes,
                va="top",
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
            )
            axis.set_title(f"Config {config_id}")
            axis.set_xlabel("Actual offloaded batch size")
            axis.set_ylabel("Mean response time (s)")
            axis.grid(linestyle="--", alpha=0.45)
            axis.set_axisbelow(True)
            axis.legend(frameon=False, loc="lower right")

        figure.suptitle(
            "Mean Response Time from Server Arrival to Result Send", fontsize=16
        )
        figure.tight_layout()
        return self.save(plt, figure, "mean_server_response_trend.png")

    def write_per_image_trend(
        self,
        plt,
        measurements: pd.DataFrame,
        grouped_summary: pd.DataFrame,
        trends: pd.DataFrame,
    ) -> Path:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for axis, config_id in zip(axes, self.config_ids(grouped_summary)):
            summary = self.config_summary(grouped_summary, config_id)
            config_measurements = measurements[
                measurements["config"].astype(str).str.zfill(3) == config_id
            ]
            trend = self.config_trend(trends, config_id)
            x = summary["actual_server_batch_size"]
            mean = summary["per_image_time_mean_s"]
            confidence = self.confidence_interval(
                summary["per_image_time_std_s"], summary["batch_count"]
            )
            color = self.CONFIG_COLORS[config_id]
            axis.errorbar(
                x,
                mean,
                yerr=confidence,
                color=color,
                fmt="none",
                capsize=3,
                alpha=0.7,
            )
            axis.scatter(
                x,
                mean,
                color=color,
                s=18 + 3 * summary["batch_count"].pow(0.5),
                label="Observed mean (95% CI)",
                zorder=3,
            )
            slope = config_measurements[
                ["actual_server_batch_size", "per_image_server_time_s"]
            ].cov().iloc[0, 1] / config_measurements[
                "actual_server_batch_size"
            ].var()
            intercept = (
                config_measurements["per_image_server_time_s"].mean()
                - slope * config_measurements["actual_server_batch_size"].mean()
            )
            axis.plot(
                x,
                intercept + slope * x,
                color="#222222",
                linestyle="--",
                linewidth=1.4,
                label="Linear fit",
            )
            axis.text(
                0.04,
                0.95,
                f"Spearman $r_s$ = {trend['per_image_time_spearman_r']:.3f}",
                transform=axis.transAxes,
                va="top",
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
            )
            axis.set_title(f"Config {config_id}")
            axis.set_xlabel("Actual offloaded batch size")
            axis.set_ylabel("Mean server time per image (s/image)")
            axis.tick_params(axis="y", labelleft=True)
            axis.grid(linestyle="--", alpha=0.45)
            axis.set_axisbelow(True)
            axis.legend(frameon=False, loc="lower right")

        figure.suptitle(
            "Mean Per-Image Server Time (marker size reflects request count)",
            fontsize=16,
        )
        figure.tight_layout()
        return self.save(plt, figure, "mean_per_image_server_time_trend.png")

    @staticmethod
    def confidence_interval(std: pd.Series, count: pd.Series) -> pd.Series:
        return (1.96 * std / count.pow(0.5)).fillna(0.0)

    @staticmethod
    def config_ids(data: pd.DataFrame) -> list[str]:
        return sorted(data["config"].astype(str).str.zfill(3).unique())

    @staticmethod
    def config_summary(summary: pd.DataFrame, config_id: str) -> pd.DataFrame:
        config = summary[summary["config"].astype(str).str.zfill(3) == config_id]
        return config.sort_values("actual_server_batch_size")

    @staticmethod
    def config_trend(trends: pd.DataFrame, config_id: str) -> pd.Series:
        config = trends[trends["config"].astype(str).str.zfill(3) == config_id]
        return config.iloc[0]

    def save(self, plt, figure, filename: str) -> Path:
        path = self.plots_dir / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return path
