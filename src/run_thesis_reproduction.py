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


class AdaptiveThresholdReplay:
    def __init__(self, beta: float = 0.5, eta: float = 0.06, quantize_step: float = 0.01):
        self.beta = beta
        self.eta = eta
        self.quantize_step = quantize_step
        self.p_intervals = [0.0, 0.75, 1.0]
        self.weights = [1.0, 1.0]

    def update_thresholds(self, confidence_score: float, correct_classification: int) -> None:
        import bisect
        import math

        confidence_score = round(confidence_score / self.quantize_step) * self.quantize_step
        confidence_score = min(max(confidence_score, 0.0), 1.0)

        idx = bisect.bisect_right(self.p_intervals, confidence_score) - 1
        if confidence_score not in self.p_intervals:
            self.p_intervals.insert(idx + 1, confidence_score)
            self.weights.insert(idx + 1, self.weights[idx])

        for index in range(len(self.weights)):
            if self.p_intervals[index + 1] <= confidence_score:
                cost = self.beta if correct_classification == 0 else 0.0
            else:
                cost = self.beta if correct_classification == 1 else 0.0
            self.weights[index] *= math.exp(-self.eta * cost)

        total_weight = sum(
            (self.p_intervals[index + 1] - self.p_intervals[index]) * self.weights[index]
            for index in range(len(self.weights))
        )
        self.weights = [weight / total_weight for weight in self.weights]

    def get_threshold(self) -> float:
        cumulative_weight = 0.0
        total_weight = sum(
            (self.p_intervals[index + 1] - self.p_intervals[index]) * self.weights[index]
            for index in range(len(self.weights))
        )

        for index in range(len(self.weights)):
            cumulative_weight += (
                self.p_intervals[index + 1] - self.p_intervals[index]
            ) * self.weights[index]
            if cumulative_weight >= 0.5 * total_weight:
                return self.p_intervals[index + 1]
        return 1.0


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
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate thesis-style plots from existing CSV outputs without rerunning ExPECA.",
    )
    parser.add_argument(
        "--reconstruct-thresholds",
        action="store_true",
        help=(
            "With --plot-only, reconstruct missing threshold values from timing CSVs. "
            "Default behavior preserves only values logged by the edge-device."
        ),
    )
    args = parser.parse_args()
    return ThesisReproductionRunner(
        dry_run=args.dry_run,
        plot_only=args.plot_only,
        reconstruct_thresholds=args.reconstruct_thresholds,
    ).run()


class ThesisReproductionRunner:
    def __init__(
        self,
        dry_run: bool = False,
        plot_only: bool = False,
        reconstruct_thresholds: bool = False,
    ):
        os.chdir(REPO_ROOT)
        self.dry_run = dry_run
        self.plot_only = plot_only
        self.reconstruct_thresholds = reconstruct_thresholds
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
        self.offloading_distribution_csv = (
            self.output_dir / "offloading_distribution.csv"
        )
        self.per_sample_latency_csv = self.output_dir / "per_sample_latency.csv"
        self.summary_md = self.output_dir / "summary.md"
        self.metadata_json = self.output_dir / RUN_METADATA_FILENAME
        self.plots_dir = self.output_dir / "plots"

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
            f"001-007 on DEVICE={self.device}."
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
            offloading_distribution_rows.append(
                self.offloading_distribution_row(thesis_config, timing)
            )
            per_sample_latency_rows.append(
                self.per_sample_latency_row(thesis_config, timing)
            )
            print()

        summary = pd.DataFrame(rows).sort_values("thesis_config")
        latency_breakdown = pd.DataFrame(latency_rows).sort_values("config")
        communication_efficiency = pd.DataFrame(communication_rows).sort_values("config")
        communication_efficiency = self.add_communication_baselines(
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
        plot_paths = self.write_plots(
            summary,
            latency_breakdown,
            communication_efficiency,
            threshold_trajectory,
            offloading_distribution,
            per_sample_latency,
        )
        self.write_summary_md(summary, latency_breakdown, communication_efficiency)
        self.write_metadata(summary, plot_paths)

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
            self.latency_breakdown_csv,
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

        if not self.offloading_distribution_csv.exists():
            rows = []
            for config in self.configurations:
                timing = self.read_config_timing(config.config_id)
                rows.append(self.offloading_distribution_row(config, timing))
            pd.DataFrame(rows).sort_values("config").to_csv(
                self.offloading_distribution_csv, index=False
            )
            print(f"Backfilled missing CSV: {self.offloading_distribution_csv}")

        if not self.per_sample_latency_csv.exists():
            rows = []
            for config in self.configurations:
                timing = self.read_config_timing(config.config_id)
                rows.append(self.per_sample_latency_row(config, timing))
            pd.DataFrame(rows).sort_values("config").to_csv(
                self.per_sample_latency_csv, index=False
            )
            print(f"Backfilled missing CSV: {self.per_sample_latency_csv}")

        threshold_trajectory = pd.read_csv(
            self.threshold_trajectory_csv, dtype={"config": str}
        )
        if self.threshold_trajectory_needs_rebuild(threshold_trajectory):
            if not self.reconstruct_thresholds:
                print(
                    "Threshold trajectory has no logged threshold values. "
                    "Preserving original CSV; rerun with a rebuilt edge-device image "
                    "or pass --reconstruct-thresholds for a best-effort offline replay."
                )
            else:
                threshold_trajectory = self.reconstruct_threshold_trajectory_from_timing()
                threshold_trajectory.to_csv(self.threshold_trajectory_csv, index=False)
                print(f"Reconstructed threshold trajectory: {self.threshold_trajectory_csv}")

        summary = pd.read_csv(self.summary_csv, dtype={"thesis_config": str})
        summary = self.backfill_summary_accuracy_columns(summary)
        summary.to_csv(self.summary_csv, index=False)

        plot_paths = self.write_plots(
            summary,
            pd.read_csv(self.latency_breakdown_csv, dtype={"config": str}),
            pd.read_csv(self.communication_efficiency_csv, dtype={"config": str}),
            threshold_trajectory,
            pd.read_csv(self.offloading_distribution_csv, dtype={"config": str}),
            pd.read_csv(self.per_sample_latency_csv, dtype={"config": str}),
        )
        print(f"Regenerated {len(plot_paths)} thesis-style plot(s): {self.plots_dir}")
        for path in plot_paths:
            print(f"  {path}")
        return 0

    def reconstruct_threshold_trajectory_from_timing(self) -> pd.DataFrame:
        rows = []
        for config in self.configurations:
            rows.extend(
                self.threshold_trajectory_rows(
                    config, self.read_config_timing(config.config_id)
                )
            )
        return pd.DataFrame(rows)

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
            metrics = self.accuracy_metrics(self.read_config_timing(config.config_id))
            for column, value in metrics.items():
                if column in output:
                    output.loc[mask, column] = value

        print(f"Backfilled missing accuracy column(s) in: {self.summary_csv}")
        return output

    @staticmethod
    def threshold_trajectory_needs_rebuild(threshold_trajectory: pd.DataFrame) -> bool:
        if threshold_trajectory.empty:
            return True
        threshold_columns = [
            column
            for column in [
                "decision_threshold",
                "adaptive_threshold_after_update",
            ]
            if column in threshold_trajectory
        ]
        if not threshold_columns:
            return True
        values = threshold_trajectory[threshold_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        return not values.notna().any().any()

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
        offloaded = self.offloaded_mask(timing)
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
            "sml_accuracy_not_offloaded": self.group_accuracy(
                true_class, sml_prediction, ~offloaded
            ),
            "correct": int(correct),
        }

    @staticmethod
    def group_accuracy(
        true_class: pd.Series, prediction: pd.Series, mask: pd.Series
    ) -> float | None:
        valid = true_class.notna() & prediction.notna() & mask
        if not valid.any():
            return None
        return float((prediction[valid] == true_class[valid]).mean())

    @staticmethod
    def offloaded_mask(timing: pd.DataFrame) -> pd.Series:
        if "Offloaded" in timing.columns:
            return timing["Offloaded"].astype(str).str.lower().eq("true")
        if "LML Prediction" in timing.columns:
            return pd.to_numeric(timing["LML Prediction"], errors="coerce").notna()
        if "lml_inference_s" in timing.columns:
            return pd.to_numeric(timing["lml_inference_s"], errors="coerce").notna()
        return pd.Series([False] * len(timing), index=timing.index)

    def communication_efficiency_row(
        self,
        config: ThesisConfiguration,
        timing: pd.DataFrame,
        summary_row: dict,
    ) -> dict:
        rows = len(timing)
        offloaded = int(self.offloaded_mask(timing).sum())
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
        offloaded = int(self.offloaded_mask(timing).sum())
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
        adaptive_replay = AdaptiveThresholdReplay()
        offloaded = self.offloaded_mask(timing)
        for sample_index, (_, row) in enumerate(timing.iterrows(), start=1):
            confidence = self.optional_float(row.get("SML Confidence"))
            decision_threshold = self.optional_float(row.get("Decision Threshold"))
            if decision_threshold is None:
                decision_threshold = adaptive_replay.get_threshold()

            adaptive_threshold_after_update = self.optional_float(
                row.get("Adaptive Threshold After Update")
            )
            sample_offloaded = bool(offloaded.loc[row.name])
            if sample_offloaded and confidence is not None:
                true_class = self.optional_float(row.get("True Class"))
                sml_prediction = self.optional_float(row.get("SML Prediction"))
                lml_prediction = self.optional_float(row.get("LML Prediction"))
                if true_class is not None and sml_prediction is not None:
                    correct_classification = int(sml_prediction == true_class)
                    adaptive_replay.update_thresholds(confidence, correct_classification)
                    if adaptive_threshold_after_update is None:
                        adaptive_threshold_after_update = adaptive_replay.get_threshold()
                elif sml_prediction is not None and lml_prediction is not None:
                    correct_classification = int(lml_prediction == sml_prediction)
                    adaptive_replay.update_thresholds(confidence, correct_classification)
                    if adaptive_threshold_after_update is None:
                        adaptive_threshold_after_update = adaptive_replay.get_threshold()

            rows.append(
                {
                    "config": config.config_id,
                    "sample_index": sample_index,
                    "filename": row.get("Filename"),
                    "sml_confidence": confidence,
                    "offloaded": sample_offloaded,
                    "decision_threshold": decision_threshold,
                    "adaptive_threshold_after_update": adaptive_threshold_after_update,
                    "threshold_update_duration_s": self.optional_float(
                        row.get("ts_threshold_updated")
                    ),
                }
            )
        return rows

    def offloading_distribution_row(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> dict:
        true_class = pd.to_numeric(timing.get("True Class"), errors="coerce")
        sml_prediction = pd.to_numeric(timing.get("SML Prediction"), errors="coerce")
        offloaded = self.offloaded_mask(timing)
        sml_correct = true_class.notna() & sml_prediction.notna() & (
            sml_prediction == true_class
        )
        sml_wrong = true_class.notna() & sml_prediction.notna() & (
            sml_prediction != true_class
        )
        total = max(len(timing), 1)

        true_positive = int((sml_wrong & offloaded).sum())
        true_negative = int((sml_correct & ~offloaded).sum())
        false_positive = int((sml_correct & offloaded).sum())
        false_negative = int((sml_wrong & ~offloaded).sum())

        return {
            "config": config.config_id,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive_percent": 100.0 * true_positive / total,
            "true_negative_percent": 100.0 * true_negative / total,
            "false_positive_percent": 100.0 * false_positive / total,
            "false_negative_percent": 100.0 * false_negative / total,
        }

    def per_sample_latency_row(
        self, config: ThesisConfiguration, timing: pd.DataFrame
    ) -> dict:
        offloaded = self.offloaded_mask(timing)
        latency = pd.to_numeric(timing["total_tracked_latency_s"], errors="coerce")
        return {
            "config": config.config_id,
            "system_combined_s": self.series_mean(latency),
            "offloaded_samples_s": self.series_mean(latency[offloaded]),
            "not_offloaded_samples_s": self.series_mean(latency[~offloaded]),
        }

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
        summary: pd.DataFrame,
        latency_breakdown: pd.DataFrame,
        communication_efficiency: pd.DataFrame,
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
    def series_mean(values: pd.Series) -> float | None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return None
        return float(numeric.mean())

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
