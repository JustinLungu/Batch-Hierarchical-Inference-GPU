import os
from pathlib import Path

import pandas as pd


class ThesisPlotter:
    def __init__(self, plots_dir: Path, thesis_base: dict[str, str]):
        self.plots_dir = plots_dir
        self.thesis_base = thesis_base

    def write_plots(
        self,
        summary: pd.DataFrame,
        latency_breakdown: pd.DataFrame,
        threshold_trajectory: pd.DataFrame,
        offloading_distribution: pd.DataFrame,
        per_sample_latency: pd.DataFrame,
    ) -> list[Path]:
        matplotlib_cache = Path("/tmp") / "matplotlib-thesis-reproduction"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return []

        for stale_plot in self.plots_dir.glob("*.png"):
            stale_plot.unlink()

        return [
            self.write_accuracy_comparison_plot(plt, summary),
            self.write_offloading_distribution_plot(plt, offloading_distribution),
            self.write_threshold_value_updates_plot(plt, threshold_trajectory),
            self.write_per_sample_latency_plot(plt, per_sample_latency),
            self.write_latency_breakdown_plot(plt, latency_breakdown),
            self.write_throughput_processing_time_plot(plt, summary, per_sample_latency),
        ]

    @staticmethod
    def apply_thesis_axes_style(axis) -> None:
        axis.grid(axis="y", linestyle="--", alpha=0.6)
        axis.set_axisbelow(True)

    @staticmethod
    def add_figure_caption(figure, figure_id: str, title: str) -> None:
        figure.subplots_adjust(bottom=0.18)
        figure.text(0.42, 0.035, figure_id, ha="right", fontsize=14, fontweight="bold")
        figure.text(0.50, 0.035, title, ha="left", fontsize=14, fontweight="bold")

    @staticmethod
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

    def write_accuracy_comparison_plot(self, plt, summary: pd.DataFrame) -> Path:
        import numpy as np

        configs = summary["thesis_config"].tolist()
        x_values = np.arange(len(configs))
        width = 0.18
        series = [
            ("System Overall", "accuracy", "#1f77b4", -1.5 * width),
            ("S-M-L - All Samples", "sml_accuracy", "#ff7f0e", -0.5 * width),
            ("S-M-L - Not Offloaded Samples", "sml_accuracy_not_offloaded", "#2ca02c", 0.5 * width),
            ("L-M-L - Offloaded Samples", "lml_accuracy_offloaded", "#d62728", 1.5 * width),
        ]

        figure, axis = plt.subplots(figsize=(11, 7))
        max_accuracy = 0.0
        for label, column, color, offset in series:
            if column in summary:
                values = pd.to_numeric(summary[column], errors="coerce") * 100.0
            else:
                values = pd.Series([float("nan")] * len(summary), index=summary.index)
            if not values.dropna().empty:
                max_accuracy = max(max_accuracy, float(values.max()))
            bars = axis.bar(x_values + offset, values, width, label=label, color=color)
            self.annotate_bars(axis, bars)

        axis.set_title("Accuracy Comparison")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Accuracy (%)")
        axis.set_xticks(x_values)
        axis.set_xticklabels(configs)
        axis.set_ylim(0, max(95, max_accuracy + 8))
        axis.legend(loc="lower left")
        self.apply_thesis_axes_style(axis)
        self.add_figure_caption(figure, "Figure 5-1", "Accuracy Comparison")

        path = self.plots_dir / "figure_5_1_accuracy_comparison.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_offloading_distribution_plot(
        self, plt, distribution: pd.DataFrame
    ) -> Path:
        thesis_configs = distribution[distribution["config"].isin(["003", "004", "005", "006", "007"])]
        configs = thesis_configs["config"].tolist()
        stack = [
            ("True Positive (SML wrong + Offloaded)", "true_positive_percent", "#006400"),
            ("True Negative (SML correct + Not offloaded)", "true_negative_percent", "#2ca02c"),
            ("False Positive (SML correct + Offloaded)", "false_positive_percent", "#e18124"),
            ("False Negative (SML wrong + Not offloaded)", "false_negative_percent", "#d62728"),
        ]

        figure, axis = plt.subplots(figsize=(11, 7))
        bottoms = pd.Series([0.0] * len(thesis_configs), index=thesis_configs.index)
        for label, column, color in stack:
            values = pd.to_numeric(thesis_configs[column], errors="coerce").fillna(0.0)
            bars = axis.bar(configs, values, bottom=bottoms, label=label, color=color)
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                if height <= 0:
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms.iloc[idx] + height / 2,
                    f"{height:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            bottoms += values

        axis.set_title("Offloading Classification Distribution")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Samples (%)")
        axis.set_ylim(0, 100)
        axis.legend(loc="upper left", bbox_to_anchor=(0.02, -0.08))
        self.apply_thesis_axes_style(axis)
        self.add_figure_caption(figure, "Figure 5-2", "Offloading Decision Distributions")

        path = self.plots_dir / "figure_5_2_offloading_decision_distributions.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_threshold_value_updates_plot(
        self, plt, threshold_trajectory: pd.DataFrame
    ) -> Path:
        figure, axis = plt.subplots(figsize=(12, 5))
        colors = {
            "004": "#1f77b4",
            "005": "#ff7f0e",
            "006": "#2ca02c",
            "007": "#d62728",
        }
        if not threshold_trajectory.empty:
            for config, group in threshold_trajectory.groupby("config"):
                values = pd.to_numeric(group["decision_threshold"], errors="coerce")
                values = values.fillna(
                    pd.to_numeric(group["adaptive_threshold_after_update"], errors="coerce")
                ).dropna()
                if values.empty:
                    continue
                x_values = pd.Series(range(len(values)), index=values.index)
                if len(values) > 1:
                    x_values = x_values / (len(values) - 1)
                smooth = values.rolling(window=max(1, min(25, len(values) // 5)), min_periods=1).mean()
                std = values.rolling(window=max(2, min(25, len(values) // 5)), min_periods=1).std().fillna(0.0)
                color = colors.get(str(config), None)
                axis.plot(x_values, smooth, label=f"Config {config}", color=color, linewidth=1.2)
                axis.fill_between(
                    x_values,
                    (smooth - std).clip(lower=0),
                    (smooth + std).clip(upper=1),
                    color=color,
                    alpha=0.12,
                )

        fixed_threshold = float(self.thesis_base.get("FIXED_THRESHOLD_VALUE", 0.3888))
        axis.axhline(fixed_threshold, color="gray", linewidth=0.8, alpha=0.6, label="Fixed Threshold")
        axis.set_title("Threshold Over Update")
        axis.set_xlabel("Normalized Update Sequence")
        axis.set_ylabel("Threshold Value")
        axis.set_ylim(0.34, 0.84)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["First", "Last"])
        axis.legend(loc="upper right")
        self.apply_thesis_axes_style(axis)
        self.add_figure_caption(figure, "Figure 5-3", "Threshold Value Updates")

        path = self.plots_dir / "figure_5_3_threshold_value_updates.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

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
            self.annotate_bars(axis, bars, fmt="{:.2f}", rotation=0)

        axis.set_title("Per-Sample Latency Comparison")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Latency (s)")
        axis.set_xticks(x_values)
        axis.set_xticklabels(configs)
        axis.legend(loc="upper left")
        self.apply_thesis_axes_style(axis)
        self.add_figure_caption(figure, "Figure 5-4", "Per-Sample Latency Comparison")

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
        self.apply_thesis_axes_style(axis)
        self.add_figure_caption(figure, "Figure 5-5", "Latency Breakdown")

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
        self.add_figure_caption(figure, "Figure 5-6", "Throughput and Processing Time")

        path = self.plots_dir / "figure_5_6_throughput_processing_time.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path


    @staticmethod
    def format_float(value) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"

    @staticmethod
    def format_percent(value) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.1f}%"
