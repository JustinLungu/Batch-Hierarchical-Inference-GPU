import shutil
from pathlib import Path

import pandas as pd
import requests

from constants import RAW_RESULTS_COPY_FILENAME, TIMING_RESULTS_FILENAME
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
        self.raw_results_copy = self.analysis_dir / RAW_RESULTS_COPY_FILENAME

    def post_process_results(self) -> pd.DataFrame:
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.raw_results_csv, self.raw_results_copy)
        raw_results = self.load_raw_results()
        timing_results = self.add_timing_durations(raw_results)
        self.write_timing_csv(timing_results)

        print(self.build_summary(timing_results))
        print(f"Wrote analysis folder: {self.analysis_dir}")
        print(f"Copied raw CSV: {self.raw_results_copy}")
        print(f"Wrote timing CSV: {self.timing_results_csv}")
        return timing_results

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


