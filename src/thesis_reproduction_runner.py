import csv
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from constants import CONFIG_FILE, DEFAULT_CONFIG_FILE, REPO_ROOT, RUN_METADATA_FILENAME, TIMING_RESULTS_FILENAME
from thesis_models import THESIS_CONFIG_FILE, THESIS_REPRODUCTION_FILE, ThesisConfiguration
from thesis_metrics import ThesisMetrics
from thesis_plots import ThesisPlotter
from thesis_public_ip_run import ThesisPublicIpRun
from thesis_report import ThesisReportWriter
from utils import load_env_file, require_config


class ThesisReproductionRunner:
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
        self.thesis_base = load_env_file(THESIS_REPRODUCTION_FILE)
        self.configurations = self.load_thesis_configurations()
        self.results_dir = Path(require_config(self.config, "RESULTS_DIR"))
        self.sample_limit = self.config_value("CONTROLLER_MAX_SAMPLES", "all")
        self.started_at = datetime.now(timezone.utc)
        self.output_dir = self.results_dir / self.output_dir_name()
        self.summary_csv = self.output_dir / "summary.csv"
        self.latency_breakdown_csv = self.output_dir / "latency_breakdown.csv"
        self.communication_efficiency_csv = (
            self.output_dir / "communication_efficiency.csv"
        )
        self.threshold_trajectory_csv = self.output_dir / "threshold_trajectory.csv"
        self.offloading_distribution_csv = (
            self.output_dir / "offloading_distribution.csv"
        )
        self.per_sample_latency_csv = self.output_dir / "per_sample_latency.csv"
        self.summary_md = self.output_dir / "summary.md"
        self.metadata_json = self.output_dir / RUN_METADATA_FILENAME
        self.plots_dir = self.output_dir / "plots"
        self.metrics = ThesisMetrics()

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.validate_assets()

        if self.plot_only:
            return self.regenerate_plots_from_csv()

        if self.dry_run:
            self.print_dry_run()
            return 0

        print(
            "Running thesis reproduction configurations "
            f"{self.config_id_label()} on DEVICE={self.device}."
        )
        print(f"Sample limit: {self.sample_limit}")
        print()

        rows = []
        latency_rows = []
        communication_rows = []
        threshold_rows = []
        offloading_distribution_rows = []
        per_sample_latency_rows = []
        for index, thesis_config in enumerate(self.configurations, start=1):
            print(
                f"[{index}/{len(self.configurations)}] "
                f"Config {thesis_config.config_id}: {thesis_config.description}"
            )
            config_output_dir = self.output_dir / f"config_{thesis_config.config_id}"
            if config_output_dir.exists():
                shutil.rmtree(config_output_dir)
            run = ThesisPublicIpRun(
                config_overrides=thesis_config.overrides(
                    self.thesis_base, self.sample_limit
                ),
                config_output_dir=config_output_dir,
            )
            try:
                run.start_services()
                run.send_config()
                run.send_samples()
                run.download_remote_results()
                timing = run.post_process_results()
            finally:
                run.stop_services()

            row = run.aggregate_metrics(timing)
            row.update(
                {
                    "thesis_config": thesis_config.config_id,
                    "description": thesis_config.description,
                    "decision_method": thesis_config.decision_method,
                    "offloading_strategy": thesis_config.offloading_strategy,
                    "fixed_threshold_value": float(
                        thesis_config.fixed_threshold_value
                    ),
                    **self.metrics.accuracy_metrics(timing),
                    **self.metrics.summary_communication_metrics(timing, row),
                }
            )
            rows.append(row)
            latency_rows.append(self.metrics.latency_breakdown_row(thesis_config, timing))
            communication_rows.append(
                self.metrics.communication_efficiency_row(thesis_config, timing, row)
            )
            threshold_rows.extend(self.metrics.threshold_trajectory_rows(thesis_config, timing))
            offloading_distribution_rows.append(
                self.metrics.offloading_distribution_row(thesis_config, timing)
            )
            per_sample_latency_rows.append(
                self.metrics.per_sample_latency_row(thesis_config, timing)
            )
            print()

        summary = pd.DataFrame(rows).sort_values("thesis_config")
        latency_breakdown = pd.DataFrame(latency_rows).sort_values("config")
        communication_efficiency = pd.DataFrame(communication_rows).sort_values("config")
        communication_efficiency = self.metrics.add_communication_baselines(
            communication_efficiency
        )
        threshold_trajectory = pd.DataFrame(threshold_rows)
        offloading_distribution = pd.DataFrame(offloading_distribution_rows).sort_values(
            "config"
        )
        per_sample_latency = pd.DataFrame(per_sample_latency_rows).sort_values("config")

        summary.to_csv(self.summary_csv, index=False)
        latency_breakdown.to_csv(self.latency_breakdown_csv, index=False)
        communication_efficiency.to_csv(self.communication_efficiency_csv, index=False)
        threshold_trajectory.to_csv(self.threshold_trajectory_csv, index=False)
        offloading_distribution.to_csv(self.offloading_distribution_csv, index=False)
        per_sample_latency.to_csv(self.per_sample_latency_csv, index=False)
        plot_paths = ThesisPlotter(self.plots_dir, self.thesis_base).write_plots(
            summary,
            latency_breakdown,
            communication_efficiency,
            threshold_trajectory,
            offloading_distribution,
            per_sample_latency,
        )
        report = ThesisReportWriter(self)
        report.write_summary_md(summary)
        report.write_metadata(summary)

        print(f"Wrote thesis reproduction folder: {self.output_dir}")
        print(f"Wrote aggregate CSV: {self.summary_csv}")
        print(f"Wrote latency breakdown CSV: {self.latency_breakdown_csv}")
        print(f"Wrote communication efficiency CSV: {self.communication_efficiency_csv}")
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
            self.communication_efficiency_csv,
            self.threshold_trajectory_csv,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise RuntimeError(
                "Cannot regenerate plots because existing result CSV(s) are missing:\n"
                f"{formatted}\n"
                "Run `.venv/bin/python src/run_thesis_reproduction.py` first."
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

        plot_paths = ThesisPlotter(self.plots_dir, self.thesis_base).write_plots(
            summary,
            pd.read_csv(self.latency_breakdown_csv, dtype={"config": str}),
            pd.read_csv(self.communication_efficiency_csv, dtype={"config": str}),
            threshold_trajectory,
            pd.read_csv(self.offloading_distribution_csv, dtype={"config": str}),
            pd.read_csv(self.per_sample_latency_csv, dtype={"config": str}),
        )
        report = ThesisReportWriter(self)
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

    def load_thesis_configurations(self) -> list[ThesisConfiguration]:
        if not THESIS_CONFIG_FILE.exists():
            raise RuntimeError(f"Missing thesis config table: {THESIS_CONFIG_FILE}")

        with THESIS_CONFIG_FILE.open(newline="") as config_file:
            reader = csv.DictReader(config_file)
            required_columns = {
                "config_id",
                "decision_method",
                "offloading_strategy",
                "controller_batch_size",
                "batch_size",
                "fixed_threshold_value",
                "description",
            }
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise RuntimeError(
                    f"{THESIS_CONFIG_FILE} is missing column(s): {missing}"
                )
            configs = [ThesisConfiguration.from_csv_row(row) for row in reader]

        if [config.config_id for config in configs] != [
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
        ]:
            raise RuntimeError(
                f"{THESIS_CONFIG_FILE} must define configs 001 through 007 in order."
            )
        selected_config_ids = self.selected_config_ids()
        if selected_config_ids is None:
            return configs

        selected_configs = [
            config for config in configs if config.config_id in selected_config_ids
        ]
        missing = selected_config_ids - {config.config_id for config in selected_configs}
        if missing:
            raise ValueError(
                "THESIS_CONFIGS_TO_RUN contains unknown config id(s): "
                f"{', '.join(sorted(missing))}"
            )
        return selected_configs

    def selected_config_ids(self) -> set[str] | None:
        raw_value = self.config_value("THESIS_CONFIGS_TO_RUN", "all").strip()
        if raw_value.lower() in {"all", "001-007", "1-7"}:
            return None
        return {
            item.strip().zfill(3)
            for item in raw_value.replace(";", ",").split(",")
            if item.strip()
        }

    def config_id_label(self) -> str:
        return ",".join(config.config_id for config in self.configurations)

    def output_dir_name(self) -> str:
        configured = self.config_value("THESIS_OUTPUT_DIR", "")
        if configured:
            return configured
        if self.device == "cuda":
            return "thesis_reproduction_gpu"
        return "thesis_reproduction"

    def validate_assets(self) -> None:
        required_keys = ["SAMPLE_PATH", "SML_MODEL", "LML_MODEL"]
        missing = [
            self.thesis_base[key]
            for key in required_keys
            if key not in self.thesis_base or not Path(self.thesis_base[key]).exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise RuntimeError(
                "Missing thesis reproduction asset(s):\n"
                f"{formatted}\n"
                "Run `scripts/download_dataset.sh --imagenetv2` and "
                "`scripts/download_models.sh --all` first."
            )

    def print_dry_run(self) -> None:
        print("Thesis reproduction configuration:")
        print(f"  DEVICE={self.device}")
        print(f"  CONTROLLER_MAX_SAMPLES={self.sample_limit}")
        print(f"  SAMPLE_PATH={self.thesis_base['SAMPLE_PATH']}")
        print(f"  SML_ARCH={self.thesis_base['SML_ARCH']}")
        print(f"  LML_ARCH={self.thesis_base['LML_ARCH']}")
        print()
        for config in self.configurations:
            print(
                f"{config.config_id}: "
                f"DECISION_METHOD={config.decision_method}, "
                f"OFFLOADING_STRATEGY={config.offloading_strategy}, "
                f"CONTROLLER_BATCH_SIZE={config.controller_batch_size}, "
                f"BATCH_SIZE={config.batch_size}, "
                f"FIXED_THRESHOLD_VALUE={config.fixed_threshold_value}"
            )

    def config_value(self, key: str, default: str) -> str:
        return os.environ.get(key, self.config.get(key, default)).strip()
