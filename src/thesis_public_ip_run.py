from pathlib import Path

import requests

from constants import (
    RAW_RESULTS_COPY_FILENAME,
    RUN_METADATA_FILENAME,
    SUMMARY_FILENAME,
    TIMING_RESULTS_FILENAME,
)
from experiment_runner import ExperimentRunner


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


