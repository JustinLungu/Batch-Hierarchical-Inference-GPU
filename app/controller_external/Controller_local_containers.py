import os
import re
import uuid
import time
import json
import requests
import mimetypes
from datetime import datetime
from dotenv import load_dotenv
from torchvision import datasets
from PIL import Image, UnidentifiedImageError

# Load environment variables from .env file
load_dotenv()

# Configuration: Endpoints and file paths for the Edge Device and Edge Server
EDGE_DEVICE_IP = "127.0.0.1"
EDGE_SERVER_IP = "127.0.0.1"
EDGE_DEVICE_PORT = os.getenv("EDGE_DEVICE_PORT", "8000")
EDGE_SERVER_PORT = os.getenv("EDGE_SERVER_PORT", "8001")

EDGE_DEVICE = f"http://{EDGE_DEVICE_IP}:{EDGE_DEVICE_PORT}"
EDGE_SERVER = f"http://{EDGE_SERVER_IP}:{EDGE_SERVER_PORT}"

EDGE_DEVICE_PREDICT = f"{EDGE_DEVICE}/predict"
EDGE_DEVICE_CONFIG = f"{EDGE_DEVICE}/config"
EDGE_DEVICE_LOG = f"{EDGE_DEVICE}/logs"
EDGE_DEVICE_RESULTS = f"{EDGE_DEVICE}/results"

EDGE_SERVER_CONFIG = f"{EDGE_SERVER}/config"
EDGE_SERVER_LOG = f"{EDGE_SERVER}/logs"

# Configuration: File paths for results and logs
SYSTEM_LOG_FILE = "results/Full_log.txt"
SYSTEM_RESULTS_FILE = "results/Full_results.csv"

# Configuration: Experiment parameters
experiment_config = {

    # Configuration: Experiment parameters
    "sample_path": "data/datasets/imagenette/val_subset_custom", # Loaded dataset
    "sml_model": "data/models/sml/small_model_state_dict.pth", # SML for Edge Device
    "sml_architecture": "mobilenet_v2_custom", # Options: "mobilenet_v2", "resnet50" ...
    "lml_model": "data/models/lml/large_model_state_dict.pth", # LML for Edge Server
    "lml_architecture": "mobilenet_v2_custom", # Options: "mobilenet_v2", "resnet50" ...

    # Configuration: Offloading parameters
    "decision_method": "adaptive_threshold", # Options: "never_offload", "always_offload", "fixed_threshold", "adaptive_threshold"
    "fixed_threshold_value": 0.7, # Options: 0.0 to 1.0
    "offloading_strategy": "send_individually", # Options: "send_individually", "size_based_batching", "buffer_based_batching", "combined_batching"
    "wait_time_between_samples": 0.1, # (Seconds) Wait time for sending samples to Edge Device

    # Configuration: Batch processing parameters
    "batch_size": 5, # Number of samples in a batch
    "batch_wait_time": 3.0 # Batch wait time (Seconds)
}

# Function: Send experiment parameters configuration to Edge Device and Edge Server
def send_experiment_config():
    print("Sending configuration to Edge Device and Edge Server...")
    try:
        r1 = requests.post(EDGE_DEVICE_CONFIG, json=experiment_config, timeout=30)
        r2 = requests.post(EDGE_SERVER_CONFIG, json=experiment_config, timeout=30)
        if r1.status_code == 200 and r2.status_code == 200:
            print("Configuration sent successfully.")
        else:
            if r1.status_code != 200:
                print(f"ERROR: Edge Device config failed: {r1.text}")
            if r2.status_code != 200:
                print(f"ERROR: Edge Server config failed: {r2.text}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to send config: {e}")

# Function: Send request with retry logic
def send_request_with_retry(url, files, data, max_retries, retry_delay, timeout):
    retries = 0
    while retries <= max_retries:
        try:
            if retries > 0:
                print(f"Retry attempt {retries}/{max_retries}...")
            response = requests.post(url, files=files, data=data, timeout=timeout)
            return response
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries > max_retries:
                print(f"ERROR: All retry attempts failed. Last error: {e}")
                return None
            print(f"Request failed: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

# Function: Collect logs from Edge Device and Edge Server
def collect_and_save_logs():
    def fetch_logs(url, label):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return [(datetime.min, f"[{label}] Failed to fetch logs: {resp.text}")]
            lines = resp.text.splitlines()
        except requests.exceptions.RequestException as e:
            return [(datetime.min, f"[{label}] Error fetching logs: {e}")]
        entries = []
        for line in lines:
            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d{3})", line.strip())
            ts = datetime.min
            if match:
                for fmt in ["%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        ts = datetime.strptime(match.group(1), fmt)
                        break
                    except ValueError:
                        continue
            entries.append((ts, f"[{label}] {line.strip()}"))
        return entries

    entries = fetch_logs(EDGE_DEVICE_LOG, "EdgeDevice") + fetch_logs(EDGE_SERVER_LOG, "EdgeServer")
    entries.sort(key=lambda x: x[0])
    logs = "\n".join(e for _, e in entries)
    os.makedirs(os.path.dirname(SYSTEM_LOG_FILE), exist_ok=True)
    with open(SYSTEM_LOG_FILE, "w") as f:
        f.write(logs)
    print(f"System logs saved to {SYSTEM_LOG_FILE}")

# Function: Download and save results
def download_and_save_results():
    try:
        response = requests.get(EDGE_DEVICE_RESULTS, timeout=10, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(SYSTEM_RESULTS_FILE), exist_ok=True)
            with open(SYSTEM_RESULTS_FILE, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Results CSV downloaded to {SYSTEM_RESULTS_FILE}")
        else:
            print(f"Failed to download results CSV. Status: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Error downloading results CSV: {e}")

# Main function to run the experiment
if __name__ == "__main__":
    send_experiment_config(experiment_config)

    dataset = datasets.ImageFolder(experiment_config["sample_path"])
    image_files = []
    for img_path, class_index in dataset.imgs:
        folder = os.path.basename(os.path.dirname(img_path))
        try:
            if folder.startswith("class_"):
                true_label = int(folder.split("_")[1])
            else:
                true_label = class_index
        except (IndexError, ValueError):
            print(f"WARNING: Cannot parse label from folder '{folder}' for file {img_path}")
            true_label = None
        image_files.append((img_path, true_label))

    wait_time = experiment_config.get("wait_time_between_samples", 0.5)
    results = []

    if input("Start generating samples to Edge Device? (y/n): ").strip().lower() != "y":
        print("Experiment cancelled.")
        exit(0)

    for image_path, true_label in image_files:
        image_name = os.path.basename(image_path)
        print(f"Sending: {image_name} True Class: {true_label}")

        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, IOError) as e:
            print(f"ERROR: Invalid image file {image_path}, skipped.")
            continue

        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                print(f"ERROR: Unsupported MIME type for {image_path}, skipped.")
                continue

            sample_uuid = str(uuid.uuid4())
            files = [("files", (image_name, image_bytes, mime_type))]
            metadata = [{"UUID": sample_uuid, "Filename": image_name, "True Class": true_label}]
            data = {"metadata": json.dumps(metadata)}

            response = send_request_with_retry(
                EDGE_DEVICE_PREDICT,
                files=files,
                data=data,
                max_retries=5,
                retry_delay=5,
                timeout=10
            )

            if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
                res_data = response.json()
                if isinstance(res_data, dict):
                    res_data = [res_data]
                for item in res_data:
                    if item.get("UUID") == sample_uuid:
                        item["True Class"] = true_label
                        results.append(item)
                        print(f"Received: {item}")
                        break
            else:
                print(f"ERROR: Bad response for {image_name}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network issue for {image_name}: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected error for {image_name}: {e}")

        time.sleep(wait_time)

    collect_and_save_logs()
    download_and_save_results()
    print("Experiment completed.")