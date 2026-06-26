import json
import mimetypes
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from torchvision import datasets


def load_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def config_value(config: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key, config.get(key, default))


def config_bool(config: dict[str, str], key: str, default: bool) -> bool:
    value = config_value(config, key, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def wait_for_server(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/docs", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"Server did not become ready: {url}")


def start_process(command: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)


def collect_image_batch(sample_path: str, batch_size: int) -> tuple[list, list[dict]]:
    dataset = datasets.ImageFolder(sample_path)
    files = []
    metadata = []

    for image_path, class_index in dataset.imgs:
        if len(files) >= batch_size:
            break

        image_name = os.path.basename(image_path)
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            continue

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            continue

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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    config = load_env_file(Path(os.environ.get("CONFIG_FILE", "config/experiment.env")))

    edge_device_port = config_value(config, "EDGE_DEVICE_PORT", "8000")
    edge_server_port = config_value(config, "EDGE_SERVER_PORT", "8001")
    edge_device_url = f"http://127.0.0.1:{edge_device_port}"
    edge_server_url = f"http://127.0.0.1:{edge_server_port}"

    batch_size = int(config_value(config, "BATCH_SIZE", "2"))
    controller_batch_size = int(config_value(config, "CONTROLLER_BATCH_SIZE", str(batch_size)))
    flush_final_batch = config_bool(config, "FLUSH_FINAL_BATCH", True)

    experiment_config = {
        "sample_path": config_value(config, "SAMPLE_PATH", "data/datasets/imagenette2-160/val"),
        "sml_model": config_value(config, "SML_MODEL", "data/models/sml/mobilenet_v3_large_imagenet1k_v2.pth"),
        "sml_architecture": config_value(config, "SML_ARCH", "mobilenet_v3_large"),
        "lml_model": config_value(config, "LML_MODEL", "data/models/lml/Wide_ResNet50_2_Weights_IMAGENET1K_V2.pth"),
        "lml_architecture": config_value(config, "LML_ARCH", "wide_resnet50_2"),
        "decision_method": config_value(config, "DECISION_METHOD", "always_offload"),
        "fixed_threshold_value": float(config_value(config, "FIXED_THRESHOLD_VALUE", "0.7")),
        "offloading_strategy": config_value(config, "OFFLOADING_STRATEGY", "size_based_batching"),
        "batch_size": batch_size,
        "batch_wait_time": float(config_value(config, "BATCH_WAIT_TIME", "3.0")),
        "controller_batch_size": controller_batch_size,
    }

    env = os.environ.copy()
    env["DEVICE"] = config_value(config, "DEVICE", "auto")
    env["EDGE_SERVER_IP"] = "127.0.0.1"
    env["EDGE_DEVICE_PORT"] = edge_device_port
    env["EDGE_SERVER_PORT"] = edge_server_port

    python = sys.executable
    processes: list[subprocess.Popen] = []

    try:
        print("Starting edge server...")
        processes.append(
            start_process(
                [python, "app/edge_server/edge_server.py"],
                Path("/tmp/bhi-local-smoke/edge_server_stdout.log"),
                env,
            )
        )
        wait_for_server(edge_server_url)

        print("Starting edge device...")
        processes.append(
            start_process(
                [python, "app/edge_device/edge_device.py"],
                Path("/tmp/bhi-local-smoke/edge_device_stdout.log"),
                env,
            )
        )
        wait_for_server(edge_device_url)

        print("Sending configuration...")
        server_response = requests.post(
            f"{edge_server_url}/config", json=experiment_config, timeout=120
        )
        device_response = requests.post(
            f"{edge_device_url}/config", json=experiment_config, timeout=120
        )
        server_response.raise_for_status()
        device_response.raise_for_status()

        print(f"Sending {controller_batch_size} samples to edge device...")
        files, metadata = collect_image_batch(
            experiment_config["sample_path"], controller_batch_size
        )
        response = requests.post(
            f"{edge_device_url}/predict",
            files=files,
            data={
                "metadata": json.dumps(metadata),
                "flush_final_batch": str(flush_final_batch).lower(),
            },
            timeout=max(120, controller_batch_size * 60),
        )
        response.raise_for_status()
        payload = response.json()

        print(json.dumps(payload, indent=2))

        results_response = requests.get(f"{edge_device_url}/results", timeout=30)
        if results_response.status_code == 200:
            Path("results/Full_results.csv").write_bytes(results_response.content)
            print("Saved results/Full_results.csv")
        else:
            print(f"No results CSV available yet: HTTP {results_response.status_code}")

        return 0
    finally:
        for process in reversed(processes):
            process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
