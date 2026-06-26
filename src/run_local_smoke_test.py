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
    SMOKE_LOG_DIR,
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
    start_process,
    wait_for_server,
)


class LocalSmokeTest:
    def __init__(self):
        os.chdir(REPO_ROOT)
        self.config = load_env_file(CONFIG_FILE)
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
        self.device = require_config(self.config, "DEVICE")
        self.flush_final_batch = require_config_bool(self.config, "FLUSH_FINAL_BATCH")

        self.run_name = self._build_run_name()
        self.analysis_dir = results_dir / f"{ANALYSIS_DIRNAME}_local_{self.device}"
        self.timing_results_csv = self.analysis_dir / TIMING_RESULTS_FILENAME
        self.summary_md = self.analysis_dir / SUMMARY_FILENAME
        self.metadata_json = self.analysis_dir / RUN_METADATA_FILENAME
        self.raw_results_copy = self.analysis_dir / RAW_RESULTS_COPY_FILENAME

        self.processes: list[subprocess.Popen] = []

    def run(self) -> int:
        try:
            self._start_services()
            self._send_config()
            self._send_samples()
            print(f"Raw results saved by edge device: {self.raw_results_csv}")
        finally:
            self._stop_services()
        return 0

    def _start_services(self) -> None:
        env = self._service_env()
        python = sys.executable

        print("Starting edge server...")
        self.processes.append(
            start_process(
                [python, str(EDGE_SERVER_SCRIPT)],
                SMOKE_LOG_DIR / "edge_server_stdout.log",
                env,
            )
        )
        wait_for_server(self.edge_server_url)

        print("Starting edge device...")
        self.processes.append(
            start_process(
                [python, str(EDGE_DEVICE_SCRIPT)],
                SMOKE_LOG_DIR / "edge_device_stdout.log",
                env,
            )
        )
        wait_for_server(self.edge_device_url)

    def _service_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["DEVICE"] = self.device
        env["EDGE_SERVER_IP"] = self.edge_server_host
        env["EDGE_DEVICE_PORT"] = self.edge_device_port
        env["EDGE_SERVER_PORT"] = self.edge_server_port
        return env

    def _build_run_name(self) -> str:
        flush_label = "flush" if self.flush_final_batch else "no_flush"
        return (
            f"local_python_{self.device}"
            f"_serverbatch{self.batch_size}"
            f"_controllerbatch{self.controller_batch_size}"
            f"_{flush_label}"
        )

    def _send_config(self) -> None:
        print("Sending configuration...")
        experiment_config = self._experiment_config()
        server_response = requests.post(
            f"{self.edge_server_url}/config", json=experiment_config, timeout=120
        )
        device_response = requests.post(
            f"{self.edge_device_url}/config", json=experiment_config, timeout=120
        )
        server_response.raise_for_status()
        device_response.raise_for_status()

    def _experiment_config(self) -> dict:
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

    def _send_samples(self) -> None:
        experiment_config = self._experiment_config()
        print(f"Sending {self.controller_batch_size} samples to edge device...")
        files, metadata = self._collect_image_batch(
            experiment_config["sample_path"], self.controller_batch_size
        )
        response = requests.post(
            f"{self.edge_device_url}/predict",
            files=files,
            data={
                "metadata": json.dumps(metadata),
                "flush_final_batch": str(self.flush_final_batch).lower(),
            },
            timeout=max(120, self.controller_batch_size * 60),
        )
        response.raise_for_status()
        print(f"Edge device returned {len(response.json())} result rows.")

    def _collect_image_batch(self, sample_path: str, batch_size: int) -> tuple[list, list[dict]]:
        dataset = datasets.ImageFolder(sample_path)
        files = []
        metadata = []

        for image_path, class_index in dataset.imgs:
            if len(files) >= batch_size:
                break
            if not self._is_valid_image(image_path):
                continue

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

        if len(files) < batch_size:
            raise RuntimeError(
                f"Only found {len(files)} valid images in {sample_path}; need {batch_size}."
            )
        return files, metadata

    def _is_valid_image(self, image_path: str) -> bool:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            return False

        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def post_process_results(self) -> None:
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.raw_results_csv, self.raw_results_copy)
        raw_results = self._load_raw_results()
        timing_results = self._add_timing_durations(raw_results)

        self._write_timing_csv(timing_results)
        summary = self._build_summary(timing_results)
        self.summary_md.write_text(summary)
        self._write_metadata(timing_results)

        print(summary)
        print(f"Wrote analysis folder: {self.analysis_dir}")
        print(f"Copied raw CSV: {self.raw_results_copy}")
        print(f"Wrote timing CSV: {self.timing_results_csv}")
        print(f"Wrote summary: {self.summary_md}")
        print(f"Wrote metadata: {self.metadata_json}")

    def _load_raw_results(self) -> pd.DataFrame:
        results = pd.read_csv(self.raw_results_csv)
        for column in TIMING_COLUMNS:
            if column in results.columns:
                results[column] = pd.to_numeric(results[column], errors="coerce")
        return results

    def _add_timing_durations(self, results: pd.DataFrame) -> pd.DataFrame:
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

    def _write_timing_csv(self, timing: pd.DataFrame) -> None:
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

    def _build_summary(self, timing: pd.DataFrame) -> str:
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

        throughput = self._approx_throughput(timing)
        if throughput is not None:
            lines.append(f"Approx throughput: ~{throughput:.2f} samples/s")

        return "\n".join(lines) + "\n"

    def _write_metadata(self, timing: pd.DataFrame) -> None:
        finished_at = datetime.now(timezone.utc)
        metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "mode": "local_python_scripts",
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_s": (finished_at - self.started_at).total_seconds(),
            "python_executable": sys.executable,
            "command": " ".join(sys.argv),
            "git_commit": self._git_commit(),
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
                "offloaded": self._count_true(timing, "Offloaded"),
                "still_buffered": self._count_true(timing, "Buffered"),
            },
        }
        self.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

    def _git_commit(self) -> str | None:
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

    def _count_true(self, data: pd.DataFrame, column: str) -> int | None:
        if column not in data.columns:
            return None
        return int(data[column].astype(str).str.lower().eq("true").sum())

    def _approx_throughput(self, timing: pd.DataFrame) -> float | None:
        start = pd.to_numeric(timing["ts_sml_inference_start"], errors="coerce").min()
        end_candidates = timing["ts_results_received_from_offloading_module"].fillna(
            timing.get("ts_results_saved_not_offloaded")
        )
        end = pd.to_numeric(end_candidates, errors="coerce").max()
        if pd.isna(start) or pd.isna(end) or end <= start:
            return None
        return len(timing) / (end - start)

    def _stop_services(self) -> None:
        for process in reversed(self.processes):
            process.terminate()
        for process in reversed(self.processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    smoke_test = LocalSmokeTest()
    smoke_test.run()
    smoke_test.post_process_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
