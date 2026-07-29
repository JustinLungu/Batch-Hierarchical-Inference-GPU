import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .constants import CONFIG_FILE, DEFAULT_CONFIG_FILE, REPO_ROOT, RUN_METADATA_FILENAME, TIMING_RESULTS_FILENAME
from .config import (
    BENCHMARK_DEFAULTS_FILE,
    config_value,
    configuration_label,
    load_configurations,
    output_dir_name,
    print_dry_run,
    validate_assets,
)
from .metrics import BenchmarkMetrics
from .plots import BenchmarkPlotter
from .public_ip import PublicIpExperiment
from .report import BenchmarkReportWriter
from .utils import load_env_file, require_config


class BenchmarkRunner:
    def __init__(
        self,
        dry_run: bool = False,
        plot_only: bool = False,
    ):
        os.chdir(REPO_ROOT)
        self.dry_run = dry_run
        self.plot_only = plot_only
        self.config = load_env_file(DEFAULT_CONFIG_FILE)
        self.config.update(load_env_file(CONFIG_FILE))
        self.device = require_config(self.config, "DEVICE")
        self.benchmark_defaults = load_env_file(BENCHMARK_DEFAULTS_FILE)
        self.configurations = load_configurations(
            selection=config_value(self.config, "THESIS_CONFIGS_TO_RUN", "all")
        )
        self.results_dir = Path(require_config(self.config, "RESULTS_DIR"))
        self.sample_limit = config_value(self.config, "CONTROLLER_MAX_SAMPLES", "all")
        self.started_at = datetime.now(timezone.utc)
        self.output_dir = self.results_dir / output_dir_name(self.config, self.device)
        self.summary_csv = self.output_dir / "summary.csv"
        self.latency_breakdown_csv = self.output_dir / "latency_breakdown.csv"
        self.threshold_trajectory_csv = self.output_dir / "threshold_trajectory.csv"
        self.offloading_distribution_csv = (
            self.output_dir / "offloading_distribution.csv"
        )
        self.per_sample_latency_csv = self.output_dir / "per_sample_latency.csv"
        self.summary_md = self.output_dir / "summary.md"
        self.metadata_json = self.output_dir / RUN_METADATA_FILENAME
        self.plots_dir = self.output_dir / "plots"
        self.metrics = BenchmarkMetrics()

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        validate_assets(self.benchmark_defaults)

        if self.plot_only:
            return self.regenerate_plots_from_csv()

        if self.dry_run:
            print_dry_run(
                device=self.device,
                sample_limit=self.sample_limit,
                benchmark_defaults=self.benchmark_defaults,
                configurations=self.configurations,
            )
            return 0

        print(
            "Running thesis reproduction configurations "
            f"{configuration_label(self.configurations)} on DEVICE={self.device}."
        )
        print(f"Sample limit: {self.sample_limit}")
        print()

        rows = []
        latency_rows = []
        threshold_rows = []
        offloading_distribution_rows = []
        per_sample_latency_rows = []
        for index, benchmark_config in enumerate(self.configurations, start=1):
            print(
                f"[{index}/{len(self.configurations)}] "
                f"Config {benchmark_config.config_id}: {benchmark_config.description}"
            )
            config_output_dir = self.output_dir / f"config_{benchmark_config.config_id}"
            if config_output_dir.exists():
                shutil.rmtree(config_output_dir)
            experiment = PublicIpExperiment(
                config_overrides=benchmark_config.overrides(
                    self.benchmark_defaults, self.sample_limit
                ),
                config_output_dir=config_output_dir,
            )
            try:
                experiment.start_services()
                experiment.send_config()
                experiment.send_samples()
                experiment.download_remote_results()
                timing = experiment.post_process_results()
            finally:
                experiment.stop_services()

            row = experiment.results.aggregate_metrics(timing)
            row.update(
                {
                    "thesis_config": benchmark_config.config_id,
                    "description": benchmark_config.description,
                    "decision_method": benchmark_config.decision_method,
                    "offloading_strategy": benchmark_config.offloading_strategy,
                    "fixed_threshold_value": float(
                        benchmark_config.fixed_threshold_value
                    ),
                    **self.metrics.accuracy_metrics(timing),
                    **self.metrics.summary_communication_metrics(timing, row),
                }
            )
            rows.append(row)
            latency_rows.append(self.metrics.latency_breakdown_row(benchmark_config, timing))
            threshold_rows.extend(self.metrics.threshold_trajectory_rows(benchmark_config, timing))
            offloading_distribution_rows.append(
                self.metrics.offloading_distribution_row(benchmark_config, timing)
            )
            per_sample_latency_rows.append(
                self.metrics.per_sample_latency_row(benchmark_config, timing)
            )
            print()

        summary = pd.DataFrame(rows).sort_values("thesis_config")
        latency_breakdown = pd.DataFrame(latency_rows).sort_values("config")
        threshold_trajectory = pd.DataFrame(threshold_rows)
        offloading_distribution = pd.DataFrame(offloading_distribution_rows).sort_values(
            "config"
        )
        per_sample_latency = pd.DataFrame(per_sample_latency_rows).sort_values("config")

        summary.to_csv(self.summary_csv, index=False)
        latency_breakdown.to_csv(self.latency_breakdown_csv, index=False)
        threshold_trajectory.to_csv(self.threshold_trajectory_csv, index=False)
        offloading_distribution.to_csv(self.offloading_distribution_csv, index=False)
        per_sample_latency.to_csv(self.per_sample_latency_csv, index=False)
        plot_paths = BenchmarkPlotter(self.plots_dir, self.benchmark_defaults).write_plots(
            summary,
            latency_breakdown,
            threshold_trajectory,
            offloading_distribution,
            per_sample_latency,
        )
        report = BenchmarkReportWriter(self)
        report.write_summary_md(summary)
        report.write_metadata(summary)

        print(f"Wrote thesis reproduction folder: {self.output_dir}")
        print(f"Wrote aggregate CSV: {self.summary_csv}")
        print(f"Wrote latency breakdown CSV: {self.latency_breakdown_csv}")
        print(f"Wrote threshold trajectory CSV: {self.threshold_trajectory_csv}")
        print(f"Wrote offloading distribution CSV: {self.offloading_distribution_csv}")
        print(f"Wrote per-sample latency CSV: {self.per_sample_latency_csv}")
        print(f"Wrote aggregate summary: {self.summary_md}")
        print(f"Wrote aggregate metadata: {self.metadata_json}")
        if plot_paths:
            print(f"Wrote {len(plot_paths)} plot(s): {self.plots_dir}")
        return 0

    def regenerate_plots_from_csv(self) -> int:
        required = [
            self.summary_csv,
            self.threshold_trajectory_csv,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise RuntimeError(
                "Cannot regenerate plots because existing result CSV(s) are missing:\n"
                f"{formatted}\n"
                "Run `.venv/bin/python src/run_benchmark.py` first."
            )

        latency_rows = []
        for config in self.configurations:
            timing = self.read_config_timing(config.config_id)
            latency_rows.append(self.metrics.latency_breakdown_row(config, timing))
        pd.DataFrame(latency_rows).sort_values("config").to_csv(
            self.latency_breakdown_csv, index=False
        )
        print(f"Rebuilt thesis-style latency breakdown: {self.latency_breakdown_csv}")

        if not self.offloading_distribution_csv.exists():
            rows = []
            for config in self.configurations:
                timing = self.read_config_timing(config.config_id)
                rows.append(self.metrics.offloading_distribution_row(config, timing))
            pd.DataFrame(rows).sort_values("config").to_csv(
                self.offloading_distribution_csv, index=False
            )
            print(f"Backfilled missing CSV: {self.offloading_distribution_csv}")

        if not self.per_sample_latency_csv.exists():
            rows = []
            for config in self.configurations:
                timing = self.read_config_timing(config.config_id)
                rows.append(self.metrics.per_sample_latency_row(config, timing))
            pd.DataFrame(rows).sort_values("config").to_csv(
                self.per_sample_latency_csv, index=False
            )
            print(f"Backfilled missing CSV: {self.per_sample_latency_csv}")

        threshold_trajectory = pd.read_csv(
            self.threshold_trajectory_csv, dtype={"config": str}
        )

        summary = pd.read_csv(self.summary_csv, dtype={"thesis_config": str})
        summary = self.backfill_summary_accuracy_columns(summary)
        summary.to_csv(self.summary_csv, index=False)

        plot_paths = BenchmarkPlotter(self.plots_dir, self.benchmark_defaults).write_plots(
            summary,
            pd.read_csv(self.latency_breakdown_csv, dtype={"config": str}),
            threshold_trajectory,
            pd.read_csv(self.offloading_distribution_csv, dtype={"config": str}),
            pd.read_csv(self.per_sample_latency_csv, dtype={"config": str}),
        )
        report = BenchmarkReportWriter(self)
        report.write_summary_md(summary)
        report.write_metadata(summary)
        print(f"Regenerated summary and metadata: {self.output_dir}")
        print(f"Regenerated {len(plot_paths)} thesis-style plot(s): {self.plots_dir}")
        for path in plot_paths:
            print(f"  {path}")
        return 0

    def backfill_summary_accuracy_columns(self, summary: pd.DataFrame) -> pd.DataFrame:
        accuracy_columns = [
            "accuracy",
            "sml_accuracy",
            "lml_accuracy_offloaded",
            "sml_accuracy_not_offloaded",
            "correct",
        ]
        needs_backfill = any(column not in summary for column in accuracy_columns)
        if not needs_backfill:
            needs_backfill = summary[accuracy_columns].isna().any().any()
        if not needs_backfill:
            return summary

        output = summary.copy()
        for column in accuracy_columns:
            if column not in output:
                output[column] = pd.NA

        for config in self.configurations:
            mask = output["thesis_config"].astype(str).str.zfill(3) == config.config_id
            if not mask.any():
                continue
            metrics = self.metrics.accuracy_metrics(self.read_config_timing(config.config_id))
            for column, value in metrics.items():
                if column in output:
                    output.loc[mask, column] = value

        print(f"Backfilled missing accuracy column(s) in: {self.summary_csv}")
        return output

    def read_config_timing(self, config_id: str) -> pd.DataFrame:
        timing_csv = self.output_dir / f"config_{config_id}" / TIMING_RESULTS_FILENAME
        if not timing_csv.exists():
            raise RuntimeError(f"Missing per-config timing CSV: {timing_csv}")
        return pd.read_csv(timing_csv)
