import os
import re
import uuid
import time
import json
import random
import requests
import mimetypes
from datetime import datetime
from dotenv import load_dotenv
from torchvision import datasets
from PIL import Image, UnidentifiedImageError

# Import architecture registries
from model_registry import sml_model_registry as sml_registry
from model_registry import lml_model_registry as lml_registry

load_dotenv()

# Validate model registries
if not isinstance(sml_registry, dict) or not sml_registry:
    print("ERROR: SML model registry is missing or invalid.")
    raise RuntimeError("SML model registry is missing or invalid.")

if not isinstance(lml_registry, dict) or not lml_registry:
    print("ERROR: LML model registry is missing or invalid.")
    raise RuntimeError("LML model registry is missing or invalid.")

# Configuration: Endpoints and file paths for the Edge Device and Edge Server
EDGE_DEVICE_IP = os.getenv("EDGE_DEVICE_IP", "127.0.0.1")
EDGE_SERVER_IP = os.getenv("EDGE_SERVER_IP", "127.0.0.1")
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

# Function: Choose from a list of options
def choose_from_list(prompt, options):
    for idx, opt in enumerate(options, start=1):
        print(f"{idx}. {opt}")
    while True:
        choice = input(f"{prompt} (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please try again.")

# Function: Choose a file
def choose_file(prompt, directory):
    if os.path.isdir(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        files = []
    if not files:
        return input(f"{prompt} (enter path): ").strip()
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")
    while True:
        choice = input(f"{prompt} (1-{len(files)} or path): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(directory, files[int(choice) - 1])
        if choice:
            return choice
        print("Invalid choice. Please try again.")

# Function: Get a float input with validation
def get_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Invalid number. Please try again.")

# Function: Get an integer input with validation
def get_int(prompt):
    while True:
        try:
            return int(input(prompt).strip())
        except ValueError:
            print("Invalid number. Please try again.")

# Function: Configure experiment parameters
def interactive_config():
    config = {}
    datasets_path = input("--- Dataset path (example: imagenette/val_renamed/): ").strip()
    config["sample_path"] = os.path.join("data/datasets", datasets_path)
    shuffle_choice = input("--- Shuffle dataset input order (y/n)? ").strip().lower()
    config["shuffle_input"] = shuffle_choice == "y"
    config["wait_time_between_samples"] = get_float("--- Sample generation wait time (seconds): ")
    config["controller_batch_size"] = get_int("--- Controller batch size: ")
    config["sml_architecture"] = choose_from_list("--- SML model: ", list(sml_registry.keys()))
    config["sml_model"] = sml_registry[config["sml_architecture"]]["path"]
    config["lml_architecture"] = choose_from_list("--- LML model: ", list(lml_registry.keys()))
    config["lml_model"] = lml_registry[config["lml_architecture"]]["path"]
    decision_options = ["never_offload", "always_offload", "fixed_threshold", "adaptive_threshold"]
    config["decision_method"] = choose_from_list("--- Offloading decision method ", decision_options)
    config["fixed_threshold_value"] = get_float("--- Enter fixed threshold value (0.0 to 1.0): ")
    strategy_options = ["send_individually", "dynamic_batching", "size_based_batching"]
    config["offloading_strategy"] = choose_from_list("--- Offloading sending method ", strategy_options)
    config["batch_size"] = get_int("--- Offload batch size: ")
    config["batch_wait_time"] = get_float("--- Offload batch wait time (s): ")
    return config

# Function: Send experiment parameters configuration to Edge Device and Edge Server
def send_experiment_config(cfg):
    print("Sending configuration to Edge Device and Edge Server...")
    try:
        r1 = requests.post(EDGE_DEVICE_CONFIG, json=cfg, timeout=30)
        r2 = requests.post(EDGE_SERVER_CONFIG, json=cfg, timeout=30)
        if r1.status_code == 200 and r2.status_code == 200:
            print("Configuration sent successfully.")
        else:
            if r1.status_code != 200:
                print(f"ERROR: Edge Device config failed: {r1.text}")
            if r2.status_code != 200:
                print(f"ERROR: Edge Server config failed: {r2.text}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to send config: {e}")

# Function: Send request
def send_request(url, files, data):
    try:
        response = requests.post(url, files=files, data=data, timeout=600)
        return response
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        return None

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
        resp = requests.get(EDGE_DEVICE_RESULTS, timeout=10, stream=True)
        if resp.status_code == 200:
            os.makedirs(os.path.dirname(SYSTEM_RESULTS_FILE), exist_ok=True)
            with open(SYSTEM_RESULTS_FILE, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Results saved to {SYSTEM_RESULTS_FILE}")
        else:
            print(f"ERROR: Results download failed: HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"ERROR: Results download error: {e}")

# Main function to run the experiment
if __name__ == "__main__":
    experiment_config = interactive_config()
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

    if experiment_config.get("shuffle_input"):
        random.shuffle(image_files)

    wait_time = experiment_config.get("wait_time_between_samples", 0.5)
    controller_batch_size = experiment_config.get("controller_batch_size", 1)
    results = []

    if input("Start generating samples to Edge Device? (y/n): ").strip().lower() != "y":
        print("Experiment cancelled.")
        exit(0)

    batch_files = []
    batch_metadata = []

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
            batch_files.append(("files", (image_name, image_bytes, mime_type)))
            batch_metadata.append({"UUID": sample_uuid, "Filename": image_name, "True Class": true_label})

            if len(batch_files) >= controller_batch_size:
                data = {"metadata": json.dumps(batch_metadata)}
                response = send_request(EDGE_DEVICE_PREDICT, files=batch_files, data=data)
                if response and response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
                    res_data = response.json()
                    if isinstance(res_data, dict):
                        res_data = [res_data]
                    for item in res_data:
                        results.append(item)
                        print(f"Received: {item}")
                else:
                    print(f"ERROR: Bad batch response: {response.text if response else 'No response'}")
                batch_files = []
                batch_metadata = []
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network issue for {image_name}: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected error for {image_name}: {e}")

    if batch_files:
        data = {"metadata": json.dumps(batch_metadata)}
        response = send_request(EDGE_DEVICE_PREDICT, files=batch_files, data=data)
        if response and response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
            res_data = response.json()
            if isinstance(res_data, dict):
                res_data = [res_data]
            for item in res_data:
                results.append(item)
                print(f"Received: {item}")
        else:
            print(f"ERROR: Bad batch response: {response.text if response else 'No response'}")

    collect_and_save_logs()
    download_and_save_results()
    print("Experiment completed.")