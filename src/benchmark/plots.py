import os
from pathlib import Path

import pandas as pd

from .classification_plots import ClassificationPlotter
from .performance_plots import PerformancePlotter


class BenchmarkPlotter:
    """Create all benchmark figures from already aggregated data."""

    def __init__(self, plots_dir: Path, benchmark_defaults: dict[str, str]):
        self.plots_dir = plots_dir
        self.classification = ClassificationPlotter(
            plots_dir,
            benchmark_defaults,
        )
        self.performance = PerformancePlotter(plots_dir)

    def write_plots(
        self,
        summary: pd.DataFrame,
        latency_breakdown: pd.DataFrame,
        threshold_trajectory: pd.DataFrame,
        offloading_distribution: pd.DataFrame,
        per_sample_latency: pd.DataFrame,
    ) -> list[Path]:
        matplotlib_cache = Path("/tmp") / "matplotlib-benchmark"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return []

        for stale_plot in self.plots_dir.glob("*.png"):
            stale_plot.unlink()

        return [
            self.classification.write_accuracy_comparison_plot(plt, summary),
            self.classification.write_offloading_distribution_plot(
                plt,
                offloading_distribution,
            ),
            self.classification.write_threshold_value_updates_plot(
                plt,
                threshold_trajectory,
            ),
            self.performance.write_per_sample_latency_plot(
                plt,
                per_sample_latency,
            ),
            self.performance.write_latency_breakdown_plot(
                plt,
                latency_breakdown,
            ),
            self.performance.write_throughput_processing_time_plot(
                plt,
                summary,
                per_sample_latency,
            ),
        ]
