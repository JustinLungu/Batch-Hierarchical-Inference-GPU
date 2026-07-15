import os
import time
import json
import torch
import uvicorn
import logging
import asyncio
import torch.nn as nn
from PIL import Image
from typing import List
from dotenv import load_dotenv
from torchvision import transforms, models
from fastapi.responses import PlainTextResponse
from fastapi import FastAPI, UploadFile, File, Form
from torchvision.transforms import InterpolationMode

# Load environment variables
load_dotenv()

# Configuration:
EDGE_SERVER_PORT = os.getenv("EDGE_SERVER_PORT", "8001")
LOGS_TXT = "results/EdgeServer.log"

# Supported LML model and architecture registry
model_registry = {
    "wide_resnet50_2": {
        "constructor": lambda: models.wide_resnet50_2(weights=None),
        "replace_classifier": lambda model: setattr(model, "fc", nn.Linear(model.fc.in_features, 1000)),
        "transform": transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    },
    "efficientnet_v2_l": {
        "constructor": lambda: models.efficientnet_v2_l(weights=None),
        "replace_classifier": lambda model: setattr(model.classifier, "1", nn.Linear(model.classifier[1].in_features, 1000)),
        "transform": transforms.Compose([
            transforms.Resize(480, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    },
    "vit_h_14": {
        "constructor": lambda: models.vit_h_14(weights=None, image_size=518),
        "replace_classifier": lambda model: setattr(model.heads, "head", nn.Linear(model.heads.head.in_features, 1000)),
        "transform": transforms.Compose([
            transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    # Add more architectures as needed here...
}

# Function: Clear results folder and reinitialize logger
def clear_results_and_logs():
    os.makedirs("results", exist_ok=True)
    for filename in os.listdir("results"):
        file_path = os.path.join("results", filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")

    # Reinitialize logger
    global logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = configure_logging()

# Function: Logging configuration
def configure_logging():
    os.makedirs("results", exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(LOGS_TXT)
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# Function: Log information asynchronously
async def async_log_info(message: str):
    await asyncio.to_thread(logger.info, message)

# Function: Load LML model and architecture using registry
def load_model(checkpoint_path):
    arch = config.get("lml_architecture")
    if arch not in model_registry:
        logger.error(f"Unsupported architecture: {arch}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model file not found: {checkpoint_path}")

    entry = model_registry[arch]
    model = entry["constructor"]()
    entry["replace_classifier"](model)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    global predefined_transform
    predefined_transform = entry["transform"]

    return model

# Function: Resolve compute device from environment
def resolve_device():
    requested_device = os.getenv("DEVICE", "auto").strip().lower()
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError("DEVICE must be one of: auto, cpu, cuda")

# Function: Describe runtime compute backend for logs
def device_diagnostics():
    diagnostics = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "selected_device": str(device),
    }
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        diagnostics["cuda_device_index"] = current_device
        diagnostics["cuda_device_name"] = torch.cuda.get_device_name(current_device)
        diagnostics["cuda_device_capability"] = ".".join(
            str(value) for value in torch.cuda.get_device_capability(current_device)
        )
    return diagnostics


def config_value(key):
    env_key = key.upper()
    if key in config:
        return config[key]
    if env_key in os.environ:
        return os.environ[env_key]
    raise ValueError(
        f"Missing required LML batching setting '{env_key}'. "
        "Set it in config/experiment.env before sending configuration."
    )


def config_bool(key):
    value = str(config_value(key)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def config_int(key, minimum=None):
    value = int(config_value(key))
    if minimum is not None:
        value = max(minimum, value)
    return value


def config_float(key, minimum=None, maximum=None):
    value = float(config_value(key))
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def lml_batching_settings():
    mode = str(config_value("lml_batching_mode")).strip().lower()
    if mode == "auto":
        mode = "adaptive" if device.type == "cuda" else "sequential"
    if mode in {"off", "false", "disabled", "none"}:
        mode = "sequential"
    if mode not in {"sequential", "fixed", "adaptive"}:
        raise ValueError("lml_batching_mode must be one of: auto, sequential, fixed, adaptive")

    min_batch_size = config_int("lml_min_batch_size", minimum=1)
    initial_batch_size = config_int("lml_initial_batch_size", minimum=min_batch_size)
    max_batch_size = config_int("lml_max_batch_size", minimum=min_batch_size)
    initial_batch_size = min(initial_batch_size, max_batch_size)

    return {
        "mode": mode,
        "initial_batch_size": initial_batch_size,
        "min_batch_size": min_batch_size,
        "max_batch_size": max_batch_size,
        "gpu_memory_fraction": config_float(
            "lml_gpu_memory_fraction", minimum=0.1, maximum=1.0
        ),
        "oom_retry": config_bool("lml_oom_retry"),
    }


def cuda_memory_snapshot():
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "cuda_memory_free_bytes": free_bytes,
        "cuda_memory_total_bytes": total_bytes,
        "cuda_memory_used_bytes": used_bytes,
        "cuda_memory_used_fraction": used_bytes / total_bytes if total_bytes else None,
    }


def is_cuda_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and device.type == "cuda"
    )


def next_successful_batch_size(current_batch_size, settings, processed_batch_size):
    if settings["mode"] != "adaptive":
        return current_batch_size
    if device.type != "cuda":
        return current_batch_size
    if processed_batch_size < current_batch_size:
        return current_batch_size
    return min(settings["max_batch_size"], max(current_batch_size + 1, current_batch_size * 2))


def smaller_batch_size(current_batch_size, settings):
    return max(settings["min_batch_size"], current_batch_size // 2)


def fit_batch_size_to_memory(current_batch_size, settings):
    if settings["mode"] != "adaptive" or device.type != "cuda":
        return current_batch_size

    memory = cuda_memory_snapshot()
    used_fraction = memory.get("cuda_memory_used_fraction")
    if used_fraction is None:
        return current_batch_size

    while (
        current_batch_size > settings["min_batch_size"]
        and used_fraction > settings["gpu_memory_fraction"]
    ):
        current_batch_size = smaller_batch_size(current_batch_size, settings)

    return current_batch_size


async def preprocess_upload_file(file):
    def load_and_transform():
        file.file.seek(0)
        image = Image.open(file.file).convert("RGB")
        return predefined_transform(image)

    return await asyncio.to_thread(load_and_transform)


async def run_lml_micro_batch(batch_files, uuid_map, settings, requested_batch_size):
    ts_lml_inference_start = time.time()
    tensors = [await preprocess_upload_file(file) for file in batch_files]
    image_tensor = torch.stack(tensors, dim=0).to(device)

    memory_before = cuda_memory_snapshot()
    with torch.no_grad():
        output = lml_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    memory_after = cuda_memory_snapshot()
    ts_lml_inference_end = time.time()

    batch_results = []
    for index, file in enumerate(batch_files):
        confidence_score = confidence[index].item()
        predicted_label = prediction[index].item()
        sample_uuid = uuid_map.get(file.filename)

        await async_log_info(
            "Sample: "
            f"{file.filename}, Prediction: {predicted_label}, "
            f"Confidence: {confidence_score:.2f}, "
            f"Batch: {len(batch_files)}, "
            f"Time: {ts_lml_inference_end - ts_lml_inference_start:.3f}s"
        )

        result = {
            "UUID": sample_uuid,
            "Filename": file.filename,
            "LML Prediction": predicted_label,
            "LML Confidence": confidence_score,
            "ts_sample_received_at_edge_server": None,
            "ts_lml_inference_start": ts_lml_inference_start,
            "ts_lml_inference_end": ts_lml_inference_end,
            "LML Batching Mode": settings["mode"],
            "LML Requested Batch Size": requested_batch_size,
            "LML Actual Batch Size": len(batch_files),
            "LML Batch Device": str(device),
            "LML CUDA Memory Used Fraction Before": memory_before.get(
                "cuda_memory_used_fraction"
            ),
            "LML CUDA Memory Used Fraction After": memory_after.get(
                "cuda_memory_used_fraction"
            ),
        }
        batch_results.append(result)

    return batch_results


async def run_lml_batches(files, uuid_map):
    settings = lml_batching_settings()
    if settings["mode"] == "sequential":
        current_batch_size = 1
    else:
        current_batch_size = settings["initial_batch_size"]

    results = []
    index = 0
    while index < len(files):
        current_batch_size = fit_batch_size_to_memory(current_batch_size, settings)
        requested_batch_size = current_batch_size
        batch_size = min(current_batch_size, len(files) - index)
        batch_files = files[index : index + batch_size]

        try:
            batch_results = await run_lml_micro_batch(
                batch_files, uuid_map, settings, requested_batch_size
            )
        except Exception as exc:
            if (
                settings["oom_retry"]
                and is_cuda_oom(exc)
                and current_batch_size > settings["min_batch_size"]
            ):
                logger.warning(
                    "CUDA OOM with LML batch size %s. Retrying with smaller batch.",
                    current_batch_size,
                )
                torch.cuda.empty_cache()
                current_batch_size = smaller_batch_size(current_batch_size, settings)
                continue
            raise

        results.extend(batch_results)
        index += batch_size
        current_batch_size = next_successful_batch_size(
            current_batch_size, settings, batch_size
        )

    return results

# Initialize: Global variables
config = {}
lml_model = None
cached_config = {}

# Initialize: Device configuration
device = resolve_device()

# Initialize: Logging
logger = configure_logging()

# Initialize: FastAPI
app = FastAPI()

# Endpoint: Collect logs
@app.get("/logs", response_class=PlainTextResponse)
async def get_logs():
    log_path = LOGS_TXT
    if not os.path.exists(log_path):
        return "Log file not found."
    with open(log_path, "r") as f:
        return f.read()

# Endpoint: Receive and process configuration
@app.post("/config")
async def update_config(new_config: dict):
    global config, lml_model, cached_config

    # Clear previous results and logs for new configuration
    clear_results_and_logs()

    config = new_config
    lml_model = config.get("lml_model", None)
    lml_architecture = config.get("lml_architecture", None)

    if not lml_model or not os.path.exists(lml_model):
        raise FileNotFoundError(f"Model file not found: {lml_model}")
    if not lml_architecture:
        raise ValueError("LML model architecture not found")

    lml_model = load_model(lml_model)

    cached_config["lml_model"] = lml_model
    cached_config["lml_architecture"] = lml_architecture
    cached_config["lml_batching"] = lml_batching_settings()

    await async_log_info(
        "Edge Server: Configuration received and updated successfully. "
        f"LML batching: {json.dumps(cached_config['lml_batching'])}"
    )
    return {"message": "Edge Server: Configuration updated successfully"}

# Endpoint: Receive and process samples
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...), metadata: str = Form(None)):
    global lml_model
    if lml_model is None:
        logger.error("Model not loaded. Configuration missing...")
        return {"error": "Model not loaded. Configuration missing..."}
    
    # ts_sample_received_at_edge_server: Request received at Edge Server time
    ts_sample_received_at_edge_server = time.time()

    uuid_map = {}
    try:
        meta_list = json.loads(metadata)
        uuid_map = {entry["Filename"]: entry["UUID"] for entry in meta_list}
    except Exception as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        return {"error": "Invalid metadata format"}

    results = await run_lml_batches(files, uuid_map)
    for result in results:
        result["ts_sample_received_at_edge_server"] = ts_sample_received_at_edge_server

    # ts_results_sent_to_edge_device: Results sent to Edge Device time
    ts_results_sent_to_edge_device = time.time()
    for result in results:
        result["ts_results_sent_to_edge_device"] = ts_results_sent_to_edge_device

    return results

# Main function: Start Edge Server
if __name__ == "__main__":
    clear_results_and_logs()
    logger.info(f"Edge Server: Starting... (Device: {device}, Port: {EDGE_SERVER_PORT})")
    logger.info(f"Edge Server: Runtime diagnostics: {json.dumps(device_diagnostics())}")
    uvicorn.run(app, host="0.0.0.0", port=int(EDGE_SERVER_PORT))
