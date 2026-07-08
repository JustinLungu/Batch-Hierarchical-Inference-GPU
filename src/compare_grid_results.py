import os
from pathlib import Path

import pandas as pd

from constants import REPO_ROOT
from utils import load_env_file, require_config


CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", "config/experiment.env"))
SUMMARY_FILENAME = "summary.csv"
COMPARISON_DIRNAME = "comparison_expeca_public_ip"
METRICS = {
    "throughput_samples_s": "Throughput (samples/s)",
    "total_latency_median_s": "Median tracked latency (s)",
    "total_latency_mean_s": "Mean tracked latency (s)",
    "sml_inference_mean_s": "SML inference mean (s)",
    "lml_inference_mean_s": "LML inference mean (s)",
    "offload_roundtrip_mean_s": "Offload roundtrip mean (s)",
}

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def main() -> int:
    comparison = GridComparison()
    comparison.run()
    return 0


class GridComparison:
    def __init__(self):
        os.chdir(REPO_ROOT)
        self.config = load_env_file(CONFIG_FILE)
        self.results_dir = Path(require_config(self.config, "RESULTS_DIR"))
        self.output_dir = self.results_dir / COMPARISON_DIRNAME
        self.combined_csv = self.output_dir / "combined_grid_summary.csv"
        self.best_csv = self.output_dir / "best_by_metric.csv"
        self.summary_md = self.output_dir / "summary.md"
        self.plots_dir = self.output_dir / "plots"

    def run(self) -> None:
        summaries = self.load_grid_summaries()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        combined = pd.concat(summaries, ignore_index=True)
        combined = self.normalize_columns(combined)
        combined.to_csv(self.combined_csv, index=False)

        best = self.best_rows(combined)
        best.to_csv(self.best_csv, index=False)

        pivot_paths = self.write_pivot_csvs(combined)
        plot_paths = self.write_plots(combined)
        self.write_summary(combined, best, pivot_paths, plot_paths)

        print(f"Wrote comparison folder: {self.output_dir}")
        print(f"Wrote combined grid CSV: {self.combined_csv}")
        print(f"Wrote best-by-metric CSV: {self.best_csv}")
        print(f"Wrote summary: {self.summary_md}")
        print(f"Wrote {len(pivot_paths)} pivot CSV(s).")
        if plot_paths:
            print(f"Wrote {len(plot_paths)} plot(s) under: {self.plots_dir}")
        else:
            print("No plots written. Install matplotlib with `uv sync` to enable PNG plots.")

    def load_grid_summaries(self) -> list[pd.DataFrame]:
        paths = sorted(self.results_dir.glob("analysis_expeca_public_ip_*_grid/summary.csv"))
        summaries = []
        for path in paths:
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame["grid_summary_csv"] = str(path)
            frame["grid_analysis_folder"] = str(path.parent)
            summaries.append(frame)

        if not summaries:
            raise RuntimeError(
                "No grid summary CSVs found. Expected files like "
                "results/analysis_expeca_public_ip_cpu_grid/summary.csv."
            )
        return summaries

    @staticmethod
    def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy()
        numeric_columns = [
            "batch_size",
            "controller_batch_size",
            "rows",
            "offloaded",
            "still_buffered",
            "edge_server_batches_observed",
            *METRICS.keys(),
        ]
        for column in numeric_columns:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized.sort_values(["device", "controller_batch_size", "batch_size"])

    @staticmethod
    def best_rows(data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for device, device_data in data.groupby("device"):
            for metric in METRICS:
                if metric not in device_data.columns or device_data[metric].dropna().empty:
                    continue
                if metric == "throughput_samples_s":
                    index = device_data[metric].idxmax()
                    direction = "max"
                else:
                    index = device_data[metric].idxmin()
                    direction = "min"
                row = device_data.loc[index]
                rows.append(
                    {
                        "device": device,
                        "metric": metric,
                        "criterion": direction,
                        "value": row[metric],
                        "batch_size": row["batch_size"],
                        "controller_batch_size": row["controller_batch_size"],
                        "rows": row["rows"],
                        "analysis_folder": row.get("analysis_folder"),
                    }
                )
        return pd.DataFrame(rows)

    def write_plots(self, data: pd.DataFrame) -> list[Path]:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return []

        plot_paths = []
        for metric, label in METRICS.items():
            if metric not in data.columns or data[metric].dropna().empty:
                continue
            plot_paths.append(self.write_line_plot(plt, data, metric, label))
            plot_paths.extend(self.write_heatmaps(plt, data, metric, label))
        return plot_paths

    def write_pivot_csvs(self, data: pd.DataFrame) -> list[Path]:
        paths = []
        for metric in METRICS:
            if metric not in data.columns or data[metric].dropna().empty:
                continue
            for device, group in data.groupby("device"):
                pivot = self.metric_pivot(group, metric)
                if pivot.empty:
                    continue
                path = self.output_dir / f"{device}_{metric}_pivot.csv"
                pivot.to_csv(path)
                paths.append(path)
        return paths

    def write_line_plot(self, plt, data: pd.DataFrame, metric: str, label: str) -> Path:
        figure, axis = plt.subplots(figsize=(8, 5))
        for (device, controller_batch), group in data.groupby(
            ["device", "controller_batch_size"]
        ):
            group = group.sort_values("batch_size")
            axis.plot(
                group["batch_size"],
                group[metric],
                marker="o",
                label=f"{device}, controller={int(controller_batch)}",
            )

        axis.set_title(label)
        axis.set_xlabel("Server batch size")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()

        path = self.plots_dir / f"{metric}_line.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_heatmaps(self, plt, data: pd.DataFrame, metric: str, label: str) -> list[Path]:
        paths = []
        for device, group in data.groupby("device"):
            pivot = self.metric_pivot(group, metric)
            if pivot.empty:
                continue

            figure, axis = plt.subplots(figsize=(8, 4.5))
            image = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
            axis.set_title(f"{device}: {label}")
            axis.set_xlabel("Server batch size")
            axis.set_ylabel("Controller batch size")
            axis.set_xticks(range(len(pivot.columns)))
            axis.set_xticklabels([str(int(value)) for value in pivot.columns])
            axis.set_yticks(range(len(pivot.index)))
            axis.set_yticklabels([str(int(value)) for value in pivot.index])

            for row_index, controller_batch in enumerate(pivot.index):
                for column_index, batch_size in enumerate(pivot.columns):
                    value = pivot.loc[controller_batch, batch_size]
                    if pd.notna(value):
                        axis.text(
                            column_index,
                            row_index,
                            f"{value:.3g}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=8,
                        )

            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            path = self.plots_dir / f"{device}_{metric}_heatmap.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
        return paths

    @staticmethod
    def metric_pivot(data: pd.DataFrame, metric: str) -> pd.DataFrame:
        return data.pivot_table(
            index="controller_batch_size",
            columns="batch_size",
            values=metric,
            aggfunc="mean",
        ).sort_index()

    def write_summary(
        self,
        data: pd.DataFrame,
        best: pd.DataFrame,
        pivot_paths: list[Path],
        plot_paths: list[Path],
    ) -> None:
        lines = [
            "# ExPECA Grid Comparison",
            "",
            f"Grid rows: {len(data)}",
            f"Devices: {', '.join(sorted(data['device'].astype(str).unique()))}",
            f"Combined CSV: `{self.combined_csv}`",
            f"Best-by-metric CSV: `{self.best_csv}`",
            "",
            "## Best Settings",
            "",
        ]

        if best.empty:
            lines.append("No best settings could be computed.")
        else:
            lines.extend(
                [
                    "| Device | Metric | Criterion | Value | Batch | Controller Batch |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for _, row in best.iterrows():
                lines.append(
                    "| "
                    f"{row['device']} | "
                    f"{row['metric']} | "
                    f"{row['criterion']} | "
                    f"{row['value']:.6f} | "
                    f"{int(row['batch_size'])} | "
                    f"{int(row['controller_batch_size'])} |"
                )

        lines.extend(["", "## Pivot CSVs", ""])
        if pivot_paths:
            for path in pivot_paths:
                lines.append(f"- `{path}`")
        else:
            lines.append("No pivot CSVs written.")

        lines.extend(["", "## Plots", ""])
        if plot_paths:
            for path in plot_paths:
                lines.append(f"- `{path}`")
        else:
            lines.append("No plots written because `matplotlib` is not installed.")

        self.summary_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
