import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from constants import CONFIG_FILE, DEFAULT_CONFIG_FILE, REPO_ROOT, RUN_METADATA_FILENAME
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
        self.analysis_dir = (
            self.results_dir / f"analysis_thesis_reproduction_{self.device}"
        )
        self.summary_csv = self.analysis_dir / "summary.csv"
        self.summary_md = self.analysis_dir / "summary.md"
        self.metadata_json = self.analysis_dir / RUN_METADATA_FILENAME

    def run(self) -> int:
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
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
        for index, thesis_config in enumerate(self.configurations, start=1):
            print(
                f"[{index}/{len(self.configurations)}] "
                f"Config {thesis_config.config_id}: {thesis_config.description}"
            )
            run = ThesisPublicIpRun(
                config_overrides=thesis_config.overrides(
                    self.thesis_base, self.sample_limit
                ),
                analysis_suffix=(
                    f"{thesis_config.config_id}_"
                    f"{thesis_config.decision_method}_"
                    f"{thesis_config.offloading_strategy}_"
                    f"controllerbatch{thesis_config.controller_batch_size}"
                ),
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
                }
            )
            rows.append(row)
            print()

        summary = pd.DataFrame(rows).sort_values("thesis_config")
        summary.to_csv(self.summary_csv, index=False)
        self.write_summary_md(summary)
        self.write_metadata(summary)

        print(f"Wrote thesis reproduction folder: {self.analysis_dir}")
        print(f"Wrote aggregate CSV: {self.summary_csv}")
        print(f"Wrote aggregate summary: {self.summary_md}")
        print(f"Wrote aggregate metadata: {self.metadata_json}")
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
                f"BATCH_SIZE={config.batch_size}, "
                f"FIXED_THRESHOLD_VALUE={config.fixed_threshold_value}"
            )

    def write_summary_md(self, summary: pd.DataFrame) -> None:
        lines = [
            f"Run: thesis_reproduction_{self.device}",
            f"Configurations: {len(summary)}",
            f"Sample limit: {self.sample_limit}",
            f"Dataset: {self.thesis_base['SAMPLE_PATH']}",
            f"SML: {self.thesis_base['SML_ARCH']}",
            f"LML: {self.thesis_base['LML_ARCH']}",
            "",
            "| Config | Decision | Strategy | Controller Batch | Rows | Offloaded | Throughput | Latency Median | LML Mean |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
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
                f"{self.format_float(row['throughput_samples_s'])} | "
                f"{self.format_seconds(row['total_latency_median_s'])} | "
                f"{self.format_seconds(row['lml_inference_mean_s'])} |"
            )

        self.summary_md.write_text("\n".join(lines) + "\n")

    def write_metadata(self, summary: pd.DataFrame) -> None:
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
                "analysis_folder": str(self.analysis_dir),
                "summary_csv": str(self.summary_csv),
                "summary_md": str(self.summary_md),
                "metadata_json": str(self.metadata_json),
            },
            "result_count": int(len(summary)),
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

    def config_value(self, key: str, default: str) -> str:
        return os.environ.get(key, self.config.get(key, default)).strip()

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


if __name__ == "__main__":
    raise SystemExit(main())
