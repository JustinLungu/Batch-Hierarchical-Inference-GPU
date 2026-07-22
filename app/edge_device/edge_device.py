import os
import time
import json
import torch
import uvicorn
import logging
import asyncio
import pandas as pd
import torch.nn as nn
from PIL import Image
from io import BytesIO
from typing import List
from dotenv import load_dotenv
from torchvision import transforms, models
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, Form
from torchvision.transforms import InterpolationMode
import offloading_decision_maker
from sample_offloading_method import offload_sample_method, send_batch_to_server, flush_size_buffer

# Load environment variables
load_dotenv()

# Configuration:
EDGE_SERVER_IP = os.getenv("EDGE_SERVER_IP", "127.0.0.1")
EDGE_DEVICE_PORT = os.getenv("EDGE_DEVICE_PORT", "8000")
EDGE_SERVER_PORT = os.getenv("EDGE_SERVER_PORT", "8001")
EDGE_SERVER = f"http://{EDGE_SERVER_IP}:{EDGE_SERVER_PORT}"
RESULTS_CSV = "results/EdgeDevice_results.csv"
LOGS_TXT = "results/EdgeDevice.log"

# Supported SML model and architecture registry
model_registry = {
    "mobilenet_v3_large": {
        "constructor": lambda: models.mobilenet_v3_large(weights=None),
        "replace_classifier": lambda model: setattr(model.classifier, "3", nn.Linear(model.classifier[3].in_features, 1000)),
        "transform": transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    },
    "efficientnet_b3": {
        "constructor": lambda: models.efficientnet_b3(weights=None),
        "replace_classifier": lambda model: setattr(model.classifier, "1", nn.Linear(model.classifier[1].in_features, 1000)),
        "transform": transforms.Compose([
            transforms.Resize(320, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    },
    "resnet34": {
        "constructor": lambda: models.resnet34(weights=None),
        "replace_classifier": lambda model: setattr(model, "fc", nn.Linear(model.fc.in_features, 1000)),
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    },
    # Add more architectures as needed here...
}

# Class: BufferedUploadFile
class BufferedUploadFile:
    def __init__(self, file, filename, content_type, uuid):
        self.file = file
        self.filename = filename
        self.content_type = content_type
        self.uuid = uuid

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


def clear_offloading_buffers():
    for attribute in ("batch_buffer_size", "dynamic_buffer"):
        if hasattr(offload_sample_method, attribute):
            setattr(offload_sample_method, attribute, [])

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

# Function: Write CSV asynchronously
async def write_csv_async(df_new_row: pd.DataFrame, results_csv: str, update=False):
    if update and os.path.exists(results_csv):
        df_existing = await asyncio.to_thread(pd.read_csv, results_csv)
        for _, row in df_new_row.iterrows():
            uuid = row["UUID"]
            if uuid in df_existing["UUID"].values:
                df_existing.loc[df_existing["UUID"] == uuid] = row
            else:
                row_df = pd.DataFrame([row]).dropna(axis=1, how='all')
                if not row_df.empty:
                    df_existing = pd.concat([df_existing, row_df], ignore_index=True)
        await asyncio.to_thread(df_existing.to_csv, results_csv, index=False)
    else:
        await asyncio.to_thread(df_new_row.to_csv, results_csv, mode="a",
                              header=not os.path.exists(results_csv), index=False)

# Function: Log information asynchronously
async def async_log_info(message: str):
    await asyncio.to_thread(logger.info, message)

# Function: Handle offloaded results
async def update_batch_results(batch_results, ts_sample_sent_to_edge_server, ts_results_received_from_edge_server):
    rows_to_update = []
    for res in batch_results:
        uuid_match = res.get("UUID")
        if uuid_match and uuid_match in results:
            entry = results[uuid_match]
            entry["LML Prediction"] = res.get("LML Prediction")
            entry["LML Confidence"] = res.get("LML Confidence")
            entry["ts_sample_sent_to_edge_server"] = ts_sample_sent_to_edge_server
            entry["ts_sample_received_at_edge_server"] = res.get("ts_sample_received_at_edge_server")
            entry["ts_lml_inference_start"] = res.get("ts_lml_inference_start")
            entry["ts_lml_inference_end"] = res.get("ts_lml_inference_end")
            entry["ts_results_sent_to_edge_device"] = res.get("ts_results_sent_to_edge_device")
            entry["ts_results_received_from_edge_server"] = ts_results_received_from_edge_server
            entry["ts_results_received_from_offloading_module"] = time.time()
            entry["Offloaded"] = True
            entry["Buffered"] = False
            
            # Adaptive threshold update feedback loop
            if cached_config.get("decision_method") == "adaptive_threshold":
                correct_classification = 1 if res.get("LML Prediction") == entry["SML Prediction"] else 0
                t8_start = time.time()
                offloading_decision_maker.adaptive_threshold_model.update_thresholds(
                    entry["SML Confidence"], correct_classification
                )
                t8_end = time.time()

                # ts_threshold_updated: Adaptive threshold update time
                entry["ts_threshold_updated"] = t8_end - t8_start
                entry["Adaptive Threshold After Update"] = (
                    offloading_decision_maker.get_adaptive_threshold()
                )
                await async_log_info(f"Sample: {entry['Filename']}, Threshold updated in: {entry['ts_threshold_updated']:.3f}s")
            
            await async_log_info(f"Sample: {entry['Filename']}, Processing completed")
            
            rows_to_update.append(entry)
    
    # Update the CSV file with the new results
    if rows_to_update:
        df_update = pd.DataFrame(rows_to_update)
        await write_csv_async(df_update, RESULTS_CSV, update=True)

# Function: Load SML model and architecture using registry
def load_model(checkpoint_path):
    arch = config.get("sml_architecture")
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

# Initialize: Global variables
config = {}
sml_model = None
cached_config = {}
results = {}

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
    
# Endpoint: Collect results CSV
@app.get("/results")
async def download_results():
    if not os.path.exists(RESULTS_CSV):
        return PlainTextResponse("CSV file not found.", status_code=404)
    return FileResponse(RESULTS_CSV, media_type="text/csv", filename="EdgeDevice_results.csv")

# Endpoint: Configure Edge Device
@app.post("/config")
async def update_config(new_config: dict):
    global config, sml_model, cached_config

    # Clear previous results and logs for new configuration
    clear_results_and_logs()
    clear_offloading_buffers()
    offloading_decision_maker.reset_adaptive_threshold_model()

    config = new_config

    sml_model = config.get("sml_model", None)
    sml_architecture = config.get("sml_architecture", None)

    if not sml_model or not os.path.exists(sml_model):
        raise FileNotFoundError(f"Model file not found: {sml_model}")
    if not sml_architecture:
        raise ValueError("SML model architecture not found")

    sml_model = load_model(sml_model)

    cached_config["controller_batch_size"] = config.get("controller_batch_size")
    cached_config["sml_model"] = sml_model
    cached_config["sml_architecture"] = sml_architecture
    cached_config["decision_method"] = config.get("decision_method")
    cached_config["fixed_threshold_value"] = config.get("fixed_threshold_value")
    cached_config["offloading_strategy"] = config.get("offloading_strategy")
    cached_config["batch_size"] = config.get("batch_size")
    cached_config["batch_wait_time"] = config.get("batch_wait_time")

    await async_log_info(f"Edge Device: Configuration received and updated successfully.")
    return {"message": "Edge Device: Configuration updated successfully"}

# Endpoint: Inference samples
@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...),
    flush_final_batch: bool = Form(False),
):
    global sml_model
    if sml_model is None:
        logger.error("Model not loaded. Configuration missing...")
        return {"error": "Model not loaded. Configuration missing..."}
    
    try:
        meta_list = json.loads(metadata)
        metadata_map = {entry["Filename"]: entry for entry in meta_list}
    except Exception as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        return {"error": "Invalid metadata format"}
    
    # Cached configuration parameters
    decision_method = cached_config.get("decision_method")
    fixed_threshold = cached_config.get("fixed_threshold_value")
    offloading_strategy = cached_config.get("offloading_strategy")
    batch_size = cached_config.get("batch_size")
    batch_wait_time = cached_config.get("batch_wait_time")
    edge_server_predict_url = (EDGE_SERVER + "/predict")

    pending_dynamic = []
    processed_uuids = []

    for file in files:

        # ts_sml_inference_start: SML inference start time
        ts_sml_inference_start = time.time()

        file_content = await file.read()
        buffered_file = BytesIO(file_content)
        buffered_file.name = file.filename
        content_type = file.content_type or "application/octet-stream"

        # Preprocess image
        image = await asyncio.to_thread(lambda: Image.open(BytesIO(file_content)).convert("RGB"))
        image_tensor = predefined_transform(image).unsqueeze(0).to(device)

        # SML inference
        with torch.no_grad():
            output = sml_model(image_tensor)
            confidence, prediction = torch.max(torch.softmax(output, dim=1), 1)

        # ts_sml_inference_end: SML inference end time
        ts_sml_inference_end = time.time()

        # Extract SML confidence score and predicted label
        confidence_score = confidence.item()
        predicted_label = prediction.item()

        # Get metadata for the sample
        entry = metadata_map.get(file.filename, {})
        sample_uuid = entry.get("UUID")
        true_label = entry.get("True Class")

        # Log results for SML inference
        await async_log_info(
            f"Sample: {file.filename}, Prediction: {predicted_label}, Confidence: {confidence_score:.2f}, Time: {ts_sml_inference_end - ts_sml_inference_start:.3f}s"
        )

        # Make offloading decision
        decision_threshold = None
        if decision_method == "fixed_threshold":
            decision_threshold = fixed_threshold
        elif decision_method == "adaptive_threshold":
            decision_threshold = offloading_decision_maker.get_adaptive_threshold()

        offload_decision = offloading_decision_maker.make_offloading_decision(
            file.filename, confidence_score, decision_method, fixed_threshold
        )

        # ts_offload_decision_made: Offloading decision completed time
        ts_offload_decision_made = time.time()

        # initialize results for sample based on UUID
        results[sample_uuid] = {
            "UUID": sample_uuid,
            "Filename": file.filename,
            "True Class": true_label,
            "SML Prediction": predicted_label,
            "SML Confidence": confidence_score,
            "Offloaded": offload_decision,
            "Buffered": False,
            "LML Prediction": None,
            "LML Confidence": None,
            "Decision Threshold": decision_threshold,
            "Adaptive Threshold After Update": None,
            "ts_sml_inference_start": ts_sml_inference_start,
            "ts_sml_inference_end": ts_sml_inference_end,
            "ts_offload_decision_made": ts_offload_decision_made,
            "ts_results_saved_not_offloaded": None,
            "ts_sample_sent_to_offloading": None,
            "ts_sample_sent_to_edge_server": None,
            "ts_sample_received_at_edge_server": None,
            "ts_lml_inference_start": None,
            "ts_lml_inference_end": None,
            "ts_results_sent_to_edge_device": None,
            "ts_results_received_from_edge_server": None,
            "ts_results_received_from_offloading_module": None,
            "ts_threshold_updated": None,
        }

        processed_uuids.append(sample_uuid)

        # If offloaded decision True:
        if offload_decision:
            buffered_upload_file = BufferedUploadFile(buffered_file, file.filename, content_type, sample_uuid)
            
            # ts_sample_sent_to_offloading: Sample sent for offloading
            results[sample_uuid]["ts_sample_sent_to_offloading"] = time.time()

            # If using dynamic batching, add to pending dynamic list
            if offloading_strategy == "dynamic_batching":
                pending_dynamic.append((buffered_upload_file, sample_uuid))
                results[sample_uuid]["Buffered"] = True
                logging.info(f"Sample: {file.filename}, Added to dynamic batching buffer, Current size: {len(pending_dynamic)}")
            # Send sample to offloading module
            else:
                offload_result = offload_sample_method(
                    buffered_upload_file,
                    offloading_strategy,
                    edge_server_predict_url,
                    batch_size,
                    batch_wait_time,
                )
                if offload_result is not None:
                    result, ts_sent, ts_recv = offload_result
                    batch_results = result if isinstance(result, list) else [result]
                    await update_batch_results(batch_results, ts_sent, ts_recv)
                else:
                    results[sample_uuid]["Buffered"] = True
        # If offloaded decision False:
        else:
            # ts_results_saved_not_offloaded: Timestamp for Final result completed for non-offloaded sample
            results[sample_uuid]["ts_results_saved_not_offloaded"] = time.time()
            await async_log_info(f"Sample: {file.filename}, Processing completed")

    # If offloading logic is dynamic batching and there are samples waiting in the buffer:
    if offloading_strategy == "dynamic_batching" and pending_dynamic:
        offload_result = send_batch_to_server(pending_dynamic, edge_server_predict_url)
        if offload_result is not None:
            result, ts_sent, ts_recv = offload_result
            batch_results = result if isinstance(result, list) else [result]
            await update_batch_results(batch_results, ts_sent, ts_recv)
            for _, uid in pending_dynamic:
                results[uid]["Buffered"] = False
        else:
            results[sample_uuid]["Buffered"] = True

    # If fixed size-based batching leaves a final partial batch, send it only
    # when the caller explicitly says this request is the end of the input.
    if offloading_strategy == "size_based_batching" and flush_final_batch:
        offload_result = flush_size_buffer(edge_server_predict_url)
        if offload_result is not None:
            result, ts_sent, ts_recv = offload_result
            batch_results = result if isinstance(result, list) else [result]
            await update_batch_results(batch_results, ts_sent, ts_recv)

    response_payload = []
    for uid in processed_uuids:
        if not results[uid].get("Buffered", False):
            response_payload.append(results[uid])
        else:
            response_payload.append({"message": f"Sample '{uid}' added to buffer"})

        if not results[uid].get("Buffered", False) and not results[uid].get("Offloaded", False):
            df_new_row = pd.DataFrame([results[uid]])
            await write_csv_async(df_new_row, RESULTS_CSV)

    return response_payload

# Main function: Start Edge Device
if __name__ == "__main__":
    clear_results_and_logs()
    logger.info(f"Edge Device: Starting... (Device: {device}, Port: {EDGE_DEVICE_PORT})")
    uvicorn.run(app, host="0.0.0.0", port=int(EDGE_DEVICE_PORT))
