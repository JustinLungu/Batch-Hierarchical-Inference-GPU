import json
import mimetypes
import os
import uuid
from pathlib import Path

import requests
from torchvision import datasets

from .constants import (
    CONFIG_FILE,
    DEFAULT_CONFIG_FILE,
    EDGE_DEVICE_RESULTS_FILENAME,
    RAW_RESULTS_COPY_FILENAME,
    REPO_ROOT,
    TIMING_RESULTS_FILENAME,
)
from .results import BenchmarkResultProcessor
from .utils import (
    is_valid_image,
    load_env_file,
    require_config,
    require_config_bool,
    true_class_label,
)


class BenchmarkExperiment:
    MODE = "experiment"
    RUN_LABEL = "experiment"

    def __init__(
        self,
        config_overrides: dict[str, str],
        output_dir: Path,
    ):
        os.chdir(REPO_ROOT)
        self.config: dict[str, str] = {}
        for config_file in (DEFAULT_CONFIG_FILE, CONFIG_FILE):
            self.config.update(load_env_file(config_file))
        if config_overrides:
            self.config.update(config_overrides)
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
        self.analysis_dir = output_dir
        self.timing_results_csv = self.analysis_dir / TIMING_RESULTS_FILENAME
        self.raw_results_copy = self.analysis_dir / RAW_RESULTS_COPY_FILENAME
        self.results = BenchmarkResultProcessor(
            raw_results_csv=self.raw_results_csv,
            timing_results_csv=self.timing_results_csv,
            run_metadata=self.result_metadata(),
        )

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
        experiment_config = {
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
        for key in (
            "LML_BATCHING_MODE",
            "LML_INITIAL_BATCH_SIZE",
            "LML_MIN_BATCH_SIZE",
            "LML_MAX_BATCH_SIZE",
            "LML_GPU_MEMORY_FRACTION",
            "LML_OOM_RETRY",
        ):
            if key in self.config:
                experiment_config[key.lower()] = self.config[key]
        return experiment_config

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
            if is_valid_image(image_path):
                image_records.append((image_path, true_class_label(image_path, class_index)))

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

    def result_metadata(self) -> dict:
        return {
            "run_name": self.run_name,
            "mode": self.MODE,
            "device": self.device,
            "batch_size": self.batch_size,
            "controller_batch_size": self.controller_batch_size,
            "controller_max_samples": self.controller_max_samples_label,
            "flush_final_batch": self.flush_final_batch,
            "analysis_folder": str(self.analysis_dir),
            "timing_results_csv": str(self.timing_results_csv),
        }

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

