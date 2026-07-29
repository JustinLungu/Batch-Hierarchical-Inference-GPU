from pathlib import Path

import pandas as pd

from .utils import add_figure_caption, annotate_bars, apply_thesis_axes_style


class PerformancePlotter:
    def __init__(self, plots_dir: Path):
        self.plots_dir = plots_dir

    def write_per_sample_latency_plot(
        self, plt, per_sample_latency: pd.DataFrame
    ) -> Path:
        import numpy as np

        configs = per_sample_latency["config"].tolist()
        x_values = np.arange(len(configs))
        width = 0.22
        series = [
            ("System Combined", "system_combined_s", "#1f77b4", -width),
            ("Offloaded Samples", "offloaded_samples_s", "#ff7f0e", 0),
            ("Not Offloaded Samples", "not_offloaded_samples_s", "#2ca02c", width),
        ]
        figure, axis = plt.subplots(figsize=(11, 7))
        for label, column, color, offset in series:
            values = pd.to_numeric(per_sample_latency[column], errors="coerce")
            bars = axis.bar(x_values + offset, values, width, label=label, color=color)
            annotate_bars(axis, bars, fmt="{:.2f}", rotation=0)

        axis.set_title("Per-Sample Latency Comparison")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Latency (s)")
        axis.set_xticks(x_values)
        axis.set_xticklabels(configs)
        axis.legend(loc="upper left")
        apply_thesis_axes_style(axis)
        add_figure_caption(figure, "Figure 5-4", "Per-Sample Latency Comparison")

        path = self.plots_dir / "figure_5_4_per_sample_latency_comparison.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_latency_breakdown_plot(self, plt, latency: pd.DataFrame) -> Path:
        step_columns = [
            "step_1_ed_processing_s",
            "step_2_ed_offload_buffer_s",
            "step_3_ed_to_es_communication_s",
            "step_4_es_processing_s",
            "step_5_es_to_ed_communication_s",
            "step_6_ed_result_saving_s",
        ]
        labels = [
            "Step 1: ED Processing",
            "Step 2: ED Offload Buffer",
            "Step 3: ED to ES Communication",
            "Step 4: ES Processing",
            "Step 5: ES to ED Communication",
            "Step 6: ED Result Saving",
        ]
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ]

        figure, axis = plt.subplots(figsize=(11, 7))
        bottoms = pd.Series([0.0] * len(latency))
        x_values = latency["config"].tolist()
        for column, label, color in zip(step_columns, labels, colors):
            values = pd.to_numeric(latency[column], errors="coerce").fillna(0.0)
            bars = axis.bar(x_values, values, bottom=bottoms, label=label, color=color)
            for bar_index, bar in enumerate(bars):
                height = bar.get_height()
                if height < 0.05:
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms.iloc[bar_index] + height / 2,
                    f"{height:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            bottoms += values

        for index, total in enumerate(bottoms):
            axis.text(
                index,
                total,
                f"{total:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        axis.set_title("Latency Breakdown (Absolute)")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Time (s)")
        axis.legend(loc="upper left")
        apply_thesis_axes_style(axis)
        add_figure_caption(figure, "Figure 5-5", "Latency Breakdown")

        path = self.plots_dir / "figure_5_5_latency_breakdown.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_throughput_processing_time_plot(
        self, plt, summary: pd.DataFrame, per_sample_latency: pd.DataFrame
    ) -> Path:
        import numpy as np

        merged = summary.merge(
            per_sample_latency,
            left_on="thesis_config",
            right_on="config",
            how="left",
        )
        configs = merged["thesis_config"].tolist()
        x_values = np.arange(len(configs))
        throughput = pd.to_numeric(
            merged["throughput_samples_s"], errors="coerce"
        ).fillna(0.0)
        seconds_per_sample = throughput.map(lambda value: 1.0 / value if value else 0.0)
        per_sample = pd.to_numeric(
            merged["system_combined_s"], errors="coerce"
        ).fillna(0.0)

        figure, axis_left = plt.subplots(figsize=(11, 7))
        bars = axis_left.bar(
            x_values,
            throughput,
            width=0.6,
            color="#1f77b4",
            alpha=0.7,
            label="Samples per Second",
        )
        axis_left.set_xlabel("Configuration")
        axis_left.set_ylabel("Throughput (samples/s)", color="#1f77b4")
        axis_left.tick_params(axis="y", labelcolor="#1f77b4")
        axis_left.set_xticks(x_values)
        axis_left.set_xticklabels(configs)
        axis_left.grid(axis="y", linestyle="--", alpha=0.35)
        axis_left.set_axisbelow(True)

        axis_right = axis_left.twinx()
        line_seconds, = axis_right.plot(
            x_values,
            seconds_per_sample,
            color="#006b4f",
            marker="o",
            label="Seconds per Sample",
        )
        line_latency, = axis_right.plot(
            x_values,
            per_sample,
            color="#d95f02",
            marker="o",
            label="Per-Sample Latency",
        )
        axis_right.set_ylabel("Time (s)", color="#d95f02")
        axis_right.tick_params(axis="y", labelcolor="#d95f02")

        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            axis_left.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1f77b4",
            )
        for x_value, value in zip(x_values, seconds_per_sample):
            if value > 0:
                axis_right.text(
                    x_value,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#006b4f",
                )
        for x_value, value in zip(x_values, per_sample):
            if value > 0:
                axis_right.text(
                    x_value,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#d95f02",
                )

        axis_left.set_title("System Throughput and Processing Times")
        handles = [bars, line_seconds, line_latency]
        labels = [handle.get_label() for handle in handles]
        axis_left.legend(handles, labels, loc="upper center")
        add_figure_caption(figure, "Figure 5-6", "Throughput and Processing Time")

        path = self.plots_dir / "figure_5_6_throughput_processing_time.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
