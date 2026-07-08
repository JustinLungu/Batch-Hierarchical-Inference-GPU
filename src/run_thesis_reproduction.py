import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from constants import (
    CONFIG_FILE,
    DEFAULT_CONFIG_FILE,
    RAW_RESULTS_COPY_FILENAME,
    REPO_ROOT,
    RUN_METADATA_FILENAME,
    SUMMARY_FILENAME,
    TIMING_RESULTS_FILENAME,
)
from experiment_runner import ExperimentRunner
from utils import load_env_file, require_config


THESIS_CONFIG_FILE = Path("config/thesis_configs.csv")
THESIS_REPRODUCTION_FILE = Path("config/thesis_reproduction.env")


@dataclass(frozen=True)
class ThesisConfiguration:
    config_id: str
    decision_method: str
    offloading_strategy: str
    controller_batch_size: int
    batch_size: int
    fixed_threshold_value: str
    description: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "ThesisConfiguration":
        return cls(
            config_id=row["config_id"].strip(),
            decision_method=row["decision_method"].strip(),
            offloading_strategy=row["offloading_strategy"].strip(),
            controller_batch_size=int(row["controller_batch_size"]),
            batch_size=int(row["batch_size"]),
            fixed_threshold_value=row["fixed_threshold_value"].strip(),
            description=row["description"].strip(),
        )

    def overrides(self, thesis_base: dict[str, str], sample_limit: str) -> dict[str, str]:
        return {
            **thesis_base,
            "DECISION_METHOD": self.decision_method,
            "OFFLOADING_STRATEGY": self.offloading_strategy,
            "FIXED_THRESHOLD_VALUE": self.fixed_threshold_value,
            "CONTROLLER_BATCH_SIZE": str(self.controller_batch_size),
            "BATCH_SIZE": str(self.batch_size),
            "CONTROLLER_MAX_SAMPLES": sample_limit,
        }


class ThesisPublicIpRun(ExperimentRunner):
    MODE = "expeca_public_ip_thesis"
    RUN_LABEL = "thesis"
    ANALYSIS_LABEL = "thesis"

    def __init__(
        self,
        config_overrides: dict[str, str],
        config_output_dir: Path,
    ):
        super().__init__(config_overrides=config_overrides)
        self.analysis_dir = config_output_dir
        self.timing_results_csv = self.analysis_dir / TIMING_RESULTS_FILENAME
        self.summary_md = self.analysis_dir / SUMMARY_FILENAME
        self.metadata_json = self.analysis_dir / RUN_METADATA_FILENAME
        self.raw_results_copy = self.analysis_dir / RAW_RESULTS_COPY_FILENAME

    def start_services(self) -> None:
        print("Using already-running ExPECA public-IP containers.")
        self.check_remote_service(self.edge_server_url, "edge server")
        self.check_remote_service(self.edge_device_url, "edge device")

    def stop_services(self) -> None:
        print("Leaving ExPECA containers running.")

    def download_remote_results(self) -> None:
        print("Downloading remote edge-device results...")
        response = requests.get(f"{self.edge_device_url}/results", timeout=60)
        response.raise_for_status()
        self.raw_results_csv.parent.mkdir(parents=True, exist_ok=True)
        temp_results_csv = self.raw_results_csv.with_suffix(".tmp")
        temp_results_csv.write_bytes(response.content)
        temp_results_csv.replace(self.raw_results_csv)
        print(f"Downloaded remote results to: {self.raw_results_csv}")

    @staticmethod
    def check_remote_service(url: str, label: str) -> None:
        response = requests.get(f"{url}/logs", timeout=30)
        response.raise_for_status()
        print(f"{label} reachable: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the thesis reproduction configurations on ExPECA public-IP containers."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate assets and print the seven thesis configurations without sending requests.",
    )
    args = parser.parse_args()
    return ThesisReproductionRunner(dry_run=args.dry_run).run()


class ThesisReproductionRunner:
    def __init__(self, dry_run: bool = False):
        os.chdir(REPO_ROOT)
        self.dry_run = dry_run
        self.config = load_env_file(DEFAULT_CONFIG_FILE)
        self.config.update(load_env_file(CONFIG_FILE))
        self.thesis_base = load_env_file(THESIS_REPRODUCTION_FILE)
        self.configurations = self.load_thesis_configurations()
        self.device = require_config(self.config, "DEVICE")
        self.results_dir = Path(require_config(self.config, "RESULTS_DIR"))
        self.sample_limit = self.config_value("CONTROLLER_MAX_SAMPLES", "all")
        self.started_at = datetime.now(timezone.utc)
        self.output_dir = self.results_dir / "thesis_reproduction"
        self.summary_csv = self.output_dir / "summary.csv"
        self.latency_breakdown_csv = self.output_dir / "latency_breakdown.csv"
        self.communication_efficiency_csv = (
            self.output_dir / "communication_efficiency.csv"
        )
        self.threshold_trajectory_csv = self.output_dir / "threshold_trajectory.csv"
        self.summary_md = self.output_dir / "summary.md"
        self.metadata_json = self.output_dir / RUN_METADATA_FILENAME
        self.plots_dir = self.output_dir / "plots"

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.validate_assets()

        if self.dry_run:
            self.print_dry_run()
            return 0

        print(
            "Running thesis reproduction configurations "
            f"001-007 on DEVICE={self.device}."
        )
        print(f"Sample limit: {self.sample_limit}")
        print()

        rows = []
        latency_rows = []
        communication_rows = []
        threshold_rows = []
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
                    **self.accuracy_metrics(timing),
                    **self.summary_communication_metrics(timing, row),
                }
            )
            rows.append(row)
            latency_rows.append(self.latency_breakdown_row(thesis_config, timing))
            communication_rows.append(
                self.communication_efficiency_row(thesis_config, timing, row)
            )
            threshold_rows.extend(self.threshold_trajectory_rows(thesis_config, timing))
            print()

        summary = pd.DataFrame(rows).sort_values("thesis_config")
        latency_breakdown = pd.DataFrame(latency_rows).sort_values("config")
        communication_efficiency = pd.DataFrame(communication_rows).sort_values("config")
        communication_efficiency = self.add_communication_baselines(
            communication_efficiency
        )
        threshold_trajectory = pd.DataFrame(threshold_rows)

        summary.to_csv(self.summary_csv, index=False)
        latency_breakdown.to_csv(self.latency_breakdown_csv, index=False)
        communication_efficiency.to_csv(self.communication_efficiency_csv, index=False)
        threshold_trajectory.to_csv(self.threshold_trajectory_csv, index=False)
        plot_paths = self.write_plots(
            latency_breakdown, communication_efficiency, threshold_trajectory
        )
        self.write_summary_md(summary, latency_breakdown, communication_efficiency)
        self.write_metadata(summary, plot_paths)

        print(f"Wrote thesis reproduction folder: {self.output_dir}")
        print(f"Wrote aggregate CSV: {self.summary_csv}")
        print(f"Wrote latency breakdown CSV: {self.latency_breakdown_csv}")
        print(f"Wrote communication efficiency CSV: {self.communication_efficiency_csv}")
        print(f"Wrote threshold trajectory CSV: {self.threshold_trajectory_csv}")
        print(f"Wrote aggregate summary: {self.summary_md}")
        print(f"Wrote aggregate metadata: {self.metadata_json}")
        if plot_paths:
            print(f"Wrote {len(plot_paths)} plot(s): {self.plots_dir}")
        return 0

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
        return configs

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
                f"FIXED_THRESHOLD_VALUE={config.fixed_threshold_value}"
            )

    def latency_breakdown_row(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> dict:
        step_1 = self.duration_sum_mean(
            timing, ["sml_inference_s", "offload_decision_s"]
        )
        step_2 = self.duration_sum_mean(timing, ["edge_buffer_wait_s"])
        step_3 = self.duration_sum_mean(timing, ["edge_to_server_network_s"])
        step_4 = self.duration_sum_mean(
            timing,
            ["server_queue_or_preprocess_s", "lml_inference_s", "server_postprocess_s"],
        )
        step_5 = self.duration_sum_mean(timing, ["server_to_edge_network_s"])
        step_6 = self.duration_sum_mean(timing, ["edge_receive_to_saved_s"])
        total = step_1 + step_2 + step_3 + step_4 + step_5 + step_6
        tracked = self.numeric_mean(timing, "total_tracked_latency_s")

        return {
            "config": config.config_id,
            "decision_method": config.decision_method,
            "offloading_strategy": config.offloading_strategy,
            "controller_batch_size": config.controller_batch_size,
            "step_1_ed_processing_s": step_1,
            "step_2_ed_offload_buffer_s": step_2,
            "step_3_ed_to_es_communication_s": step_3,
            "step_4_es_processing_s": step_4,
            "step_5_es_to_ed_communication_s": step_5,
            "step_6_ed_result_saving_s": step_6,
            "latency_breakdown_total_s": total,
            "tracked_latency_mean_s": tracked,
            "tracked_latency_median_s": self.numeric_median(
                timing, "total_tracked_latency_s"
            ),
        }

    def accuracy_metrics(self, timing: pd.DataFrame) -> dict:
        if timing.empty or "True Class" not in timing.columns:
            return {
                "accuracy": None,
                "sml_accuracy": None,
                "lml_accuracy_offloaded": None,
                "correct": None,
            }

        true_class = pd.to_numeric(timing["True Class"], errors="coerce")
        sml_prediction = pd.to_numeric(timing.get("SML Prediction"), errors="coerce")
        lml_prediction = pd.to_numeric(timing.get("LML Prediction"), errors="coerce")
        offloaded = timing.get("Offloaded", pd.Series([False] * len(timing)))
        offloaded = offloaded.astype(str).str.lower().eq("true")
        final_prediction = sml_prediction.copy()
        final_prediction.loc[offloaded] = lml_prediction.loc[offloaded]

        valid_final = true_class.notna() & final_prediction.notna()
        valid_sml = true_class.notna() & sml_prediction.notna()
        valid_lml = true_class.notna() & lml_prediction.notna() & offloaded

        correct = (final_prediction[valid_final] == true_class[valid_final]).sum()
        sml_correct = (sml_prediction[valid_sml] == true_class[valid_sml]).sum()
        lml_correct = (lml_prediction[valid_lml] == true_class[valid_lml]).sum()

        return {
            "accuracy": float(correct / valid_final.sum()) if valid_final.any() else None,
            "sml_accuracy": float(sml_correct / valid_sml.sum()) if valid_sml.any() else None,
            "lml_accuracy_offloaded": (
                float(lml_correct / valid_lml.sum()) if valid_lml.any() else None
            ),
            "correct": int(correct),
        }

    def communication_efficiency_row(
        self,
        config: ThesisConfiguration,
        timing: pd.DataFrame,
        summary_row: dict,
    ) -> dict:
        rows = len(timing)
        offloaded = self.count_true(timing, "Offloaded") or 0
        transmissions = int(summary_row.get("edge_server_batches_observed") or 0)
        average_offload_batch = offloaded / transmissions if transmissions else 0.0
        offload_ratio = offloaded / rows if rows else 0.0

        return {
            "config": config.config_id,
            "decision_method": config.decision_method,
            "offloading_strategy": config.offloading_strategy,
            "controller_batch_size": config.controller_batch_size,
            "rows": rows,
            "offloaded_samples": offloaded,
            "offload_ratio": offload_ratio,
            "offload_transmissions": transmissions,
            "average_offload_batch_size": average_offload_batch,
            "transmission_reduction_vs_individual_percent": (
                100.0 * (1.0 - transmissions / offloaded) if offloaded else 0.0
            ),
        }

    def summary_communication_metrics(
        self, timing: pd.DataFrame, summary_row: dict
    ) -> dict:
        rows = len(timing)
        offloaded = self.count_true(timing, "Offloaded") or 0
        transmissions = int(summary_row.get("edge_server_batches_observed") or 0)
        return {
            "offload_ratio": offloaded / rows if rows else 0.0,
            "offload_transmissions": transmissions,
            "average_offload_batch_size": (
                offloaded / transmissions if transmissions else 0.0
            ),
        }

    def threshold_trajectory_rows(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> list[dict]:
        if config.decision_method != "adaptive_threshold":
            return []

        rows = []
        for sample_index, (_, row) in enumerate(timing.iterrows(), start=1):
            rows.append(
                {
                    "config": config.config_id,
                    "sample_index": sample_index,
                    "filename": row.get("Filename"),
                    "sml_confidence": self.optional_float(row.get("SML Confidence")),
                    "offloaded": str(row.get("Offloaded")).lower() == "true",
                    "decision_threshold": self.optional_float(
                        row.get("Decision Threshold")
                    ),
                    "adaptive_threshold_after_update": self.optional_float(
                        row.get("Adaptive Threshold After Update")
                    ),
                    "threshold_update_duration_s": self.optional_float(
                        row.get("ts_threshold_updated")
                    ),
                }
            )
        return rows

    def add_communication_baselines(self, communication: pd.DataFrame) -> pd.DataFrame:
        output = communication.copy()
        config_004 = output.loc[output["config"] == "004", "offload_transmissions"]
        baseline = int(config_004.iloc[0]) if not config_004.empty else None
        if baseline and baseline > 0:
            output["transmission_reduction_vs_config_004_percent"] = output[
                "offload_transmissions"
            ].map(lambda value: 100.0 * (1.0 - float(value) / baseline))
        else:
            output["transmission_reduction_vs_config_004_percent"] = pd.NA
        return output

    def write_plots(
        self,
        latency_breakdown: pd.DataFrame,
        communication_efficiency: pd.DataFrame,
        threshold_trajectory: pd.DataFrame,
    ) -> list[Path]:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return []

        return [
            self.write_latency_breakdown_plot(plt, latency_breakdown),
            self.write_latency_breakdown_relative_plot(plt, latency_breakdown),
            self.write_communication_efficiency_plot(plt, communication_efficiency),
            self.write_throughput_plot(plt),
            self.write_threshold_trajectory_plot(plt, threshold_trajectory),
        ]

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

        figure, axis = plt.subplots(figsize=(11, 6))
        bottoms = pd.Series([0.0] * len(latency))
        x_values = latency["config"].tolist()
        for column, label in zip(step_columns, labels):
            values = pd.to_numeric(latency[column], errors="coerce").fillna(0.0)
            axis.bar(x_values, values, bottom=bottoms, label=label)
            bottoms += values

        for index, total in enumerate(bottoms):
            axis.text(index, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

        axis.set_title("Latency Breakdown (Absolute)")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Time (s)")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()

        path = self.plots_dir / "latency_breakdown_absolute.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_latency_breakdown_relative_plot(self, plt, latency: pd.DataFrame) -> Path:
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

        values = latency[step_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        totals = values.sum(axis=1).replace(0, pd.NA)
        percentages = values.div(totals, axis=0).fillna(0.0) * 100.0

        figure, axis = plt.subplots(figsize=(11, 6))
        bottoms = pd.Series([0.0] * len(latency))
        x_values = latency["config"].tolist()
        for column, label in zip(step_columns, labels):
            column_values = percentages[column]
            axis.bar(x_values, column_values, bottom=bottoms, label=label)
            bottoms += column_values

        axis.set_title("Latency Breakdown (Relative)")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Share of latency (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()

        path = self.plots_dir / "latency_breakdown_relative.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_communication_efficiency_plot(
        self, plt, communication: pd.DataFrame
    ) -> Path:
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(
            communication["config"],
            communication["offload_transmissions"],
            color="#4C78A8",
        )
        axis.set_title("Offload Transmissions by Configuration")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Offload transmissions")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()

        path = self.plots_dir / "communication_efficiency.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_throughput_plot(self, plt) -> Path:
        summary = pd.read_csv(self.summary_csv)
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(summary["thesis_config"], summary["throughput_samples_s"], color="#59A14F")
        axis.set_title("Throughput by Configuration")
        axis.set_xlabel("Configuration")
        axis.set_ylabel("Samples/s")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()

        path = self.plots_dir / "throughput.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_threshold_trajectory_plot(
        self, plt, threshold_trajectory: pd.DataFrame
    ) -> Path:
        figure, axis = plt.subplots(figsize=(10, 5))
        if not threshold_trajectory.empty:
            for config, group in threshold_trajectory.groupby("config"):
                values = pd.to_numeric(
                    group["decision_threshold"], errors="coerce"
                ).dropna()
                if values.empty:
                    continue
                axis.plot(
                    group.loc[values.index, "sample_index"],
                    values,
                    label=f"Config {config}",
                )

        axis.set_title("Adaptive Threshold Trajectory")
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Threshold")
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
        figure.tight_layout()

        path = self.plots_dir / "threshold_trajectory.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def write_summary_md(
        self,
        summary: pd.DataFrame,
        latency_breakdown: pd.DataFrame,
        communication_efficiency: pd.DataFrame,
    ) -> None:
        lines = [
            f"Run: thesis_reproduction_{self.device}",
            f"Configurations: {len(summary)}",
            f"Sample limit: {self.sample_limit}",
            f"Dataset: {self.thesis_base['SAMPLE_PATH']}",
            f"SML: {self.thesis_base['SML_ARCH']}",
            f"LML: {self.thesis_base['LML_ARCH']}",
            "",
            "| Config | Decision | Strategy | Controller Batch | Rows | Offloaded | Accuracy | Throughput | Latency Median | LML Mean |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for _, row in summary.iterrows():
            offloaded = "n/a"
            if pd.notna(row["offloaded"]):
                offloaded = str(int(row["offloaded"]))
            lines.append(
                "| "
                f"{row['thesis_config']} | "
                f"{row['decision_method']} | "
                f"{row['offloading_strategy']} | "
                f"{int(row['controller_batch_size'])} | "
                f"{int(row['rows'])} | "
                f"{offloaded} | "
                f"{self.format_percent(row['accuracy'])} | "
                f"{self.format_float(row['throughput_samples_s'])} | "
                f"{self.format_seconds(row['total_latency_median_s'])} | "
                f"{self.format_seconds(row['lml_inference_mean_s'])} |"
            )

        lines.extend(
            [
                "",
                "Latency breakdown CSV:",
                f"`{self.latency_breakdown_csv}`",
                "",
                "Communication efficiency CSV:",
                f"`{self.communication_efficiency_csv}`",
                "",
                "Threshold trajectory CSV:",
                f"`{self.threshold_trajectory_csv}`",
                "",
                "| Config | Offload Transmissions | Avg Offload Batch | Reduction vs Config 004 |",
                "|---|---:|---:|---:|",
            ]
        )

        for _, row in communication_efficiency.iterrows():
            reduction = row.get("transmission_reduction_vs_config_004_percent")
            reduction_label = "n/a" if pd.isna(reduction) else f"{float(reduction):.2f}%"
            lines.append(
                "| "
                f"{row['config']} | "
                f"{int(row['offload_transmissions'])} | "
                f"{float(row['average_offload_batch_size']):.2f} | "
                f"{reduction_label} |"
            )

        self.summary_md.write_text("\n".join(lines) + "\n")

    def write_metadata(self, summary: pd.DataFrame, plot_paths: list[Path]) -> None:
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "run_name": f"thesis_reproduction_{self.device}",
            "mode": "thesis_reproduction_public_ip",
            "device": self.device,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_s": (finished_at - self.started_at).total_seconds(),
            "default_config_file": str(DEFAULT_CONFIG_FILE),
            "experiment_config_file": str(CONFIG_FILE),
            "thesis_reproduction_file": str(THESIS_REPRODUCTION_FILE),
            "thesis_config_file": str(THESIS_CONFIG_FILE),
            "thesis_base": self.thesis_base,
            "sample_limit": self.sample_limit,
            "configs": [
                {
                    "config_id": config.config_id,
                    "decision_method": config.decision_method,
                    "offloading_strategy": config.offloading_strategy,
                    "controller_batch_size": config.controller_batch_size,
                    "batch_size": config.batch_size,
                    "fixed_threshold_value": float(config.fixed_threshold_value),
                    "description": config.description,
                }
                for config in self.configurations
            ],
            "outputs": {
                "analysis_folder": str(self.output_dir),
                "summary_csv": str(self.summary_csv),
                "latency_breakdown_csv": str(self.latency_breakdown_csv),
                "communication_efficiency_csv": str(
                    self.communication_efficiency_csv
                ),
                "threshold_trajectory_csv": str(self.threshold_trajectory_csv),
                "summary_md": str(self.summary_md),
                "metadata_json": str(self.metadata_json),
                "plots": [str(path) for path in plot_paths],
            },
            "result_count": int(len(summary)),
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

    def config_value(self, key: str, default: str) -> str:
        return os.environ.get(key, self.config.get(key, default)).strip()

    @staticmethod
    def duration_sum_mean(timing: pd.DataFrame, columns: list[str]) -> float:
        if timing.empty:
            return 0.0
        total = pd.Series([0.0] * len(timing), index=timing.index)
        for column in columns:
            if column in timing.columns:
                total += pd.to_numeric(timing[column], errors="coerce").fillna(0.0)
        return float(total.mean())

    @staticmethod
    def numeric_mean(data: pd.DataFrame, column: str) -> float | None:
        if column not in data.columns:
            return None
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.mean())

    @staticmethod
    def numeric_median(data: pd.DataFrame, column: str) -> float | None:
        if column not in data.columns:
            return None
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.median())

    @staticmethod
    def count_true(data: pd.DataFrame, column: str) -> int | None:
        if column not in data.columns:
            return None
        return int(data[column].astype(str).str.lower().eq("true").sum())

    @staticmethod
    def optional_float(value) -> float | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return None
        return float(numeric)

    @staticmethod
    def format_seconds(value) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}s"

    @staticmethod
    def format_float(value) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.2f}"

    @staticmethod
    def format_percent(value) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100.0:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
