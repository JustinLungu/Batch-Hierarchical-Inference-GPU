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

    await async_log_info(f"Edge Server: Configuration received and updated successfully.")
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

    results = []

    for file in files:

        # ts_lml_inference_start: LML inference start time
        ts_lml_inference_start = time.time()

        # Preprocess image
        image = await asyncio.to_thread(lambda: Image.open(file.file).convert("RGB"))
        image_tensor = predefined_transform(image).unsqueeze(0).to(device)

        # LML inference
        with torch.no_grad():
            output = lml_model(image_tensor)
            confidence, prediction = torch.max(torch.softmax(output, dim=1), 1)

        # ts_lml_inference_end: LML inference end time
        ts_lml_inference_end = time.time()

        confidence_score = confidence.item()
        predicted_label = prediction.item()
        sample_uuid = uuid_map.get(file.filename)

        # Log results for LML inference
        await async_log_info(f"Sample: {file.filename}, Prediction: {predicted_label}, Confidence: {confidence_score:.2f}, Time: {ts_lml_inference_end - ts_lml_inference_start:.3f}s")

        result = {
            "UUID": sample_uuid,
            "Filename": file.filename,
            "LML Prediction": predicted_label,
            "LML Confidence": confidence_score,
            "ts_sample_received_at_edge_server": ts_sample_received_at_edge_server,
            "ts_lml_inference_start": ts_lml_inference_start,
            "ts_lml_inference_end": ts_lml_inference_end
        }

        results.append(result)

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
