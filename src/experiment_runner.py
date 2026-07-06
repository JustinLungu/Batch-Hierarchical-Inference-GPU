import json
import mimetypes
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from torchvision import datasets

from constants import (
    ANALYSIS_DIRNAME,
    CONFIG_FILE,
    EDGE_DEVICE_RESULTS_FILENAME,
    EDGE_DEVICE_SCRIPT,
    EDGE_SERVER_SCRIPT,
    RAW_RESULTS_COPY_FILENAME,
    REPO_ROOT,
    RUN_METADATA_FILENAME,
    SUMMARY_FILENAME,
    TIMING_COLUMNS,
    TIMING_DURATIONS,
    TIMING_OUTPUT_COLUMNS,
    TIMING_RESULTS_FILENAME,
)
from utils import (
    format_mean_seconds,
    format_median_seconds,
    load_env_file,
    require_config,
    require_config_bool,
    seconds_between,
)


class ExperimentRunner:
    MODE = "experiment"
    RUN_LABEL = "experiment"
    ANALYSIS_LABEL = "experiment"

    def __init__(self):
        os.chdir(REPO_ROOT)
        self.config = self.load_config()
        self.started_at = datetime.now(timezone.utc)
        self.run_id = self.started_at.strftime("%Y%m%dT%H%M%SZ")

        self.edge_device_host = require_config(self.config, "EDGE_DEVICE_IP")
        self.edge_server_host = require_config(self.config, "EDGE_SERVER_IP")
        self.edge_device_port = require_config(self.config, "EDGE_DEVICE_PORT")
        self.edge_server_port = require_config(self.config, "EDGE_SERVER_PORT")
        self.edge_device_url = f"http://{self.edge_device_host}:{self.edge_device_port}"
        self.edge_server_url = f"http://{self.edge_server_host}:{self.edge_server_port}"

        results_dir = Path(require_config(self.config, "RESULTS_DIR"))
        self.raw_results_csv = results_dir / EDGE_DEVICE_RESULTS_FILENAME
        self.batch_size = int(require_config(self.config, "BATCH_SIZE"))
        self.controller_batch_size = int(require_config(self.config, "CONTROLLER_BATCH_SIZE"))
        self.controller_max_samples = self.parse_controller_max_samples()
        self.device = require_config(self.config, "DEVICE")
        self.flush_final_batch = require_config_bool(self.config, "FLUSH_FINAL_BATCH")

        self.run_name = self._build_run_name()
        self.analysis_dir = results_dir / self._analysis_dirname()
        self.timing_results_csv = self.analysis_dir / TIMING_RESULTS_FILENAME
        self.summary_md = self.analysis_dir / SUMMARY_FILENAME
        self.metadata_json = self.analysis_dir / RUN_METADATA_FILENAME
        self.raw_results_copy = self.analysis_dir / RAW_RESULTS_COPY_FILENAME

    def config_files(self) -> list[Path]:
        return [CONFIG_FILE]

    def load_config(self) -> dict[str, str]:
        config: dict[str, str] = {}
        for config_file in self.config_files():
            config.update(load_env_file(config_file))
        return config

    def parse_controller_max_samples(self) -> int | None:
        raw_value = os.environ.get(
            "CONTROLLER_MAX_SAMPLES",
            self.config.get("CONTROLLER_MAX_SAMPLES", str(self.controller_batch_size)),
        )
        value = raw_value.strip().lower()
        if value in {"all", "full"}:
            return None
        try:
            max_samples = int(value)
        except ValueError as exc:
            raise ValueError(
                "CONTROLLER_MAX_SAMPLES must be a positive integer, 'all', or 'full'."
            ) from exc
        if max_samples <= 0:
            raise ValueError("CONTROLLER_MAX_SAMPLES must be positive when it is numeric.")
        return max_samples

    def run(self) -> int:
        completed = False
        try:
            self.start_services()
            self.send_config()
            self.send_samples()
            print(f"Raw results saved by edge device: {self.raw_results_csv}")
            completed = True
        finally:
            self.stop_services()

        if completed:
            self.post_process_results()
        return 0

    def start_services(self) -> None:
        raise NotImplementedError

    def stop_services(self) -> None:
        raise NotImplementedError

    def send_config(self) -> None:
        print("Sending configuration...")
        experiment_config = self.experiment_config()
        server_response = requests.post(
            f"{self.edge_server_url}/config", json=experiment_config, timeout=120
        )
        device_response = requests.post(
            f"{self.edge_device_url}/config", json=experiment_config, timeout=120
        )
        server_response.raise_for_status()
        device_response.raise_for_status()

    def experiment_config(self) -> dict:
        return {
            "sample_path": require_config(self.config, "SAMPLE_PATH"),
            "sml_model": require_config(self.config, "SML_MODEL"),
            "sml_architecture": require_config(self.config, "SML_ARCH"),
            "lml_model": require_config(self.config, "LML_MODEL"),
            "lml_architecture": require_config(self.config, "LML_ARCH"),
            "decision_method": require_config(self.config, "DECISION_METHOD"),
            "fixed_threshold_value": float(require_config(self.config, "FIXED_THRESHOLD_VALUE")),
            "offloading_strategy": require_config(self.config, "OFFLOADING_STRATEGY"),
            "batch_size": self.batch_size,
            "batch_wait_time": float(require_config(self.config, "BATCH_WAIT_TIME")),
            "controller_batch_size": self.controller_batch_size,
        }

    def send_samples(self) -> None:
        experiment_config = self.experiment_config()
        image_records = self.collect_image_records(
            experiment_config["sample_path"], self.controller_max_samples
        )

        total_samples = len(image_records)
        print(
            f"Sending {total_samples} samples to edge device "
            f"in controller batches of {self.controller_batch_size}..."
        )

        total_results = 0
        for start in range(0, total_samples, self.controller_batch_size):
            batch = image_records[start : start + self.controller_batch_size]
            is_final_batch = start + self.controller_batch_size >= total_samples
            files, metadata = self.build_request_payload(batch)
            response = requests.post(
                f"{self.edge_device_url}/predict",
                files=files,
                data={
                    "metadata": json.dumps(metadata),
                    "flush_final_batch": str(
                        self.flush_final_batch and is_final_batch
                    ).lower(),
                },
                timeout=max(120, len(batch) * 60),
            )
            response.raise_for_status()
            response_data = response.json()
            if isinstance(response_data, dict):
                response_data = [response_data]
            total_results += len(response_data)
            print(
                f"Controller batch {start // self.controller_batch_size + 1}: "
                f"sent {len(batch)} sample(s), received {len(response_data)} result row(s)."
            )

        print(f"Edge device returned {total_results} result rows in total.")

    def collect_image_records(
        self, sample_path: str, max_samples: int | None
    ) -> list[tuple[str, int]]:
        dataset = datasets.ImageFolder(sample_path)
        image_records = []

        for image_path, class_index in dataset.imgs:
            if max_samples is not None and len(image_records) >= max_samples:
                break
            if self.is_valid_image(image_path):
                image_records.append((image_path, class_index))

        if not image_records:
            raise RuntimeError(f"No valid images found in {sample_path}.")
        if max_samples is not None and len(image_records) < max_samples:
            raise RuntimeError(
                f"Only found {len(image_records)} valid images in {sample_path}; "
                f"need {max_samples}."
            )
        return image_records

    def build_request_payload(
        self, image_records: list[tuple[str, int]]
    ) -> tuple[list, list[dict]]:
        files = []
        metadata = []

        for image_path, class_index in image_records:
            image_name = os.path.basename(image_path)
            mime_type, _ = mimetypes.guess_type(image_path)
            with open(image_path, "rb") as image_file:
                files.append(("files", (image_name, image_file.read(), mime_type)))
            metadata.append(
                {
                    "UUID": str(uuid.uuid4()),
                    "Filename": image_name,
                    "True Class": class_index,
                }
            )
        return files, metadata

    def post_process_results(self) -> None:
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.raw_results_csv, self.raw_results_copy)
        raw_results = self.load_raw_results()
        timing_results = self.add_timing_durations(raw_results)

        self.write_timing_csv(timing_results)
        summary = self.build_summary(timing_results)
        self.summary_md.write_text(summary)
        self.write_metadata(timing_results)

        print(summary)
        print(f"Wrote analysis folder: {self.analysis_dir}")
        print(f"Copied raw CSV: {self.raw_results_copy}")
        print(f"Wrote timing CSV: {self.timing_results_csv}")
        print(f"Wrote summary: {self.summary_md}")
        print(f"Wrote metadata: {self.metadata_json}")

    def load_raw_results(self) -> pd.DataFrame:
        results = pd.read_csv(self.raw_results_csv)
        for column in TIMING_COLUMNS:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        return results

    def add_timing_durations(self, results: pd.DataFrame) -> pd.DataFrame:
        timing = results.copy()
        for output_column, (end_column, start_column) in TIMING_DURATIONS.items():
            timing[output_column] = seconds_between(timing, end_column, start_column)

        offloaded_total = seconds_between(
            timing, "ts_results_received_from_offloading_module", "ts_sml_inference_start"
        )
        local_total = seconds_between(
            timing, "ts_results_saved_not_offloaded", "ts_sml_inference_start"
        )
        timing["total_tracked_latency_s"] = offloaded_total.fillna(local_total)

        if "ts_sample_sent_to_edge_server" in timing.columns:
            batch_keys = timing["ts_sample_sent_to_edge_server"].fillna(-1)
            timing["edge_server_batch_id"] = pd.factorize(batch_keys)[0]
            timing.loc[batch_keys == -1, "edge_server_batch_id"] = pd.NA

        return timing

    def write_timing_csv(self, timing: pd.DataFrame) -> None:
        available_columns = [column for column in TIMING_OUTPUT_COLUMNS if column in timing.columns]
        output = timing[available_columns].copy()

        for column in output.columns:
            if column.endswith("_s"):
                values = pd.to_numeric(output[column], errors="coerce")
                output[column] = values.map(lambda value: "" if pd.isna(value) else f"{value:.6f}")

        if "edge_server_batch_id" in output.columns:
            batch_ids = pd.to_numeric(output["edge_server_batch_id"], errors="coerce")
            output["edge_server_batch_id"] = batch_ids.map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )

        output.to_csv(self.timing_results_csv, index=False)

    def build_summary(self, timing: pd.DataFrame) -> str:
        lines = [
            f"Run: {self.run_name}",
            f"Rows: {len(timing)}",
        ]

        if "Offloaded" in timing.columns:
            offloaded = timing["Offloaded"].astype(str).str.lower().eq("true")
            lines.append(f"Offloaded: {offloaded.sum()} / {len(timing)}")

        if "Buffered" in timing.columns:
            buffered = timing["Buffered"].astype(str).str.lower().eq("true")
            lines.append(f"Still buffered: {buffered.sum()} / {len(timing)}")

        if "edge_server_batch_id" in timing.columns:
            batch_sizes = (
                timing.dropna(subset=["edge_server_batch_id"])
                .groupby("edge_server_batch_id")
                .size()
                .tolist()
            )
            lines.append(f"Edge-server batches observed: {len(batch_sizes)}")
            lines.append(f"Edge-server batch sizes: {batch_sizes}")

        lines.append(
            "Total tracked latency median: "
            f"{format_median_seconds(timing['total_tracked_latency_s'])}"
        )
        lines.append(f"SML inference mean: {format_mean_seconds(timing['sml_inference_s'])}")
        lines.append(f"LML inference mean: {format_mean_seconds(timing['lml_inference_s'])}")
        lines.append(f"Offload roundtrip: {format_mean_seconds(timing['offload_roundtrip_s'])}")

        throughput = self.approx_throughput(timing)
        if throughput is not None:
            lines.append(f"Approx throughput: ~{throughput:.2f} samples/s")

        return "\n".join(lines) + "\n"

    def write_metadata(self, timing: pd.DataFrame) -> None:
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "mode": self.MODE,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_s": (finished_at - self.started_at).total_seconds(),
            "python_executable": sys.executable,
            "command": " ".join(sys.argv),
            "git_commit": self.git_commit(),
            "services": {
                "edge_device_url": self.edge_device_url,
                "edge_server_url": self.edge_server_url,
                "edge_device_script": str(EDGE_DEVICE_SCRIPT),
                "edge_server_script": str(EDGE_SERVER_SCRIPT),
            },
            "experiment": {
                "device": self.device,
                "sample_path": require_config(self.config, "SAMPLE_PATH"),
                "sml_architecture": require_config(self.config, "SML_ARCH"),
                "sml_model": require_config(self.config, "SML_MODEL"),
                "lml_architecture": require_config(self.config, "LML_ARCH"),
                "lml_model": require_config(self.config, "LML_MODEL"),
                "decision_method": require_config(self.config, "DECISION_METHOD"),
                "offloading_strategy": require_config(self.config, "OFFLOADING_STRATEGY"),
                "fixed_threshold_value": float(
                    require_config(self.config, "FIXED_THRESHOLD_VALUE")
                ),
                "batch_size": self.batch_size,
                "controller_batch_size": self.controller_batch_size,
                "controller_max_samples": self.controller_max_samples_label,
                "flush_final_batch": self.flush_final_batch,
                "batch_wait_time": float(require_config(self.config, "BATCH_WAIT_TIME")),
            },
            "outputs": {
                "source_raw_results_csv": str(self.raw_results_csv),
                "analysis_folder": str(self.analysis_dir),
                "raw_results_copy": str(self.raw_results_copy),
                "timing_results_csv": str(self.timing_results_csv),
                "summary_md": str(self.summary_md),
                "metadata_json": str(self.metadata_json),
            },
            "result_counts": {
                "rows": int(len(timing)),
                "offloaded": self.count_true(timing, "Offloaded"),
                "still_buffered": self.count_true(timing, "Buffered"),
            },
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

    def _build_run_name(self) -> str:
        flush_label = "flush" if self.flush_final_batch else "no_flush"
        return (
            f"{self.RUN_LABEL}_{self.device}"
            f"_serverbatch{self.batch_size}"
            f"_controllerbatch{self.controller_batch_size}"
            f"_samples{self.controller_max_samples_label}"
            f"_{flush_label}"
        )

    @property
    def controller_max_samples_label(self) -> str:
        if self.controller_max_samples is None:
            return "all"
        return str(self.controller_max_samples)

    def _analysis_dirname(self) -> str:
        return f"{ANALYSIS_DIRNAME}_{self.ANALYSIS_LABEL}_{self.device}"

    @staticmethod
    def is_valid_image(image_path: str) -> bool:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            return False

        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    @staticmethod
    def git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    @staticmethod
    def count_true(data: pd.DataFrame, column: str) -> int | None:
        if column not in data.columns:
            return None
        return int(data[column].astype(str).str.lower().eq("true").sum())

    @staticmethod
    def approx_throughput(timing: pd.DataFrame) -> float | None:
        start = pd.to_numeric(timing["ts_sml_inference_start"], errors="coerce").min()
        end_candidates = timing["ts_results_received_from_offloading_module"].fillna(
            timing.get("ts_results_saved_not_offloaded")
        )
        end = pd.to_numeric(end_candidates, errors="coerce").max()
        if pd.isna(start) or pd.isna(end) or end <= start:
            return None
        return len(timing) / (end - start)
