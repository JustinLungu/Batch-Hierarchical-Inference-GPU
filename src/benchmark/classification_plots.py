from pathlib import Path

import pandas as pd

from .utils import add_figure_caption, annotate_bars, apply_thesis_axes_style


class ClassificationPlotter:
    def __init__(self, plots_dir: Path, benchmark_defaults: dict[str, str]):
        self.plots_dir = plots_dir
        self.benchmark_defaults = benchmark_defaults

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
            annotate_bars(axis, bars)

        axis.set_title("Accuracy Comparison")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Accuracy (%)")
        axis.set_xticks(x_values)
        axis.set_xticklabels(configs)
        axis.set_ylim(0, max(95, max_accuracy + 8))
        axis.legend(loc="lower left")
        apply_thesis_axes_style(axis)
        add_figure_caption(figure, "Figure 5-1", "Accuracy Comparison")

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
        apply_thesis_axes_style(axis)
        add_figure_caption(figure, "Figure 5-2", "Offloading Decision Distributions")

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

        fixed_threshold = float(self.benchmark_defaults.get("FIXED_THRESHOLD_VALUE", 0.3888))
        axis.axhline(fixed_threshold, color="gray", linewidth=0.8, alpha=0.6, label="Fixed Threshold")
        axis.set_title("Threshold Over Update")
        axis.set_xlabel("Normalized Update Sequence")
        axis.set_ylabel("Threshold Value")
        axis.set_ylim(0.34, 0.84)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["First", "Last"])
        axis.legend(loc="upper right")
        apply_thesis_axes_style(axis)
        add_figure_caption(figure, "Figure 5-3", "Threshold Value Updates")

        path = self.plots_dir / "figure_5_3_threshold_value_updates.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

