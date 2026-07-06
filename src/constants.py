import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", "config/experiment.env"))
EXPECA_PUBLIC_IP_CONFIG_FILE = Path(
    os.environ.get("EXPECA_PUBLIC_IP_CONFIG_FILE", "config/expeca_public_ip.env")
)
SMOKE_LOG_DIR = Path("/tmp/bhi-local-smoke")

EDGE_SERVER_SCRIPT = Path("app/edge_server/edge_server.py")
EDGE_DEVICE_SCRIPT = Path("app/edge_device/edge_device.py")
EDGE_DEVICE_RESULTS_FILENAME = "EdgeDevice_results.csv"

DOCKER_NETWORK = "edge_net"
DOCKER_EDGE_SERVER_CONTAINER = "edge_server"
DOCKER_EDGE_DEVICE_CONTAINER = "edge_device"
DOCKER_EDGE_SERVER_IMAGE = "batch-hi-edge-server:local"
DOCKER_EDGE_DEVICE_IMAGE = "batch-hi-edge-device:local"
DOCKER_EDGE_SERVER_DOCKERFILE = Path("docker/local/Dockerfile.edge_server.cpu")
DOCKER_EDGE_DEVICE_DOCKERFILE = Path("docker/local/Dockerfile.edge_device.cpu")

ANALYSIS_DIRNAME = "analysis"
TIMING_RESULTS_FILENAME = "timing_results.csv"
SUMMARY_FILENAME = "summary.md"
RUN_METADATA_FILENAME = "run_metadata.json"
RAW_RESULTS_COPY_FILENAME = "raw_edge_device_results.csv"

TIMING_COLUMNS = [
    "ts_sml_inference_start",
    "ts_sml_inference_end",
    "ts_offload_decision_made",
    "ts_results_saved_not_offloaded",
    "ts_sample_sent_to_offloading",
    "ts_sample_sent_to_edge_server",
    "ts_sample_received_at_edge_server",
    "ts_lml_inference_start",
    "ts_lml_inference_end",
    "ts_results_sent_to_edge_device",
    "ts_results_received_from_edge_server",
    "ts_results_received_from_offloading_module",
    "ts_threshold_updated",
]

TIMING_OUTPUT_COLUMNS = [
    "UUID",
    "Filename",
    "sml_inference_s",
    "offload_decision_s",
    "edge_buffer_wait_s",
    "edge_to_server_network_s",
    "server_queue_or_preprocess_s",
    "lml_inference_s",
    "server_postprocess_s",
    "server_to_edge_network_s",
    "offload_roundtrip_s",
    "edge_receive_to_saved_s",
    "total_tracked_latency_s",
    "edge_server_batch_id",
]

TIMING_DURATIONS = {
    "sml_inference_s": ("ts_sml_inference_end", "ts_sml_inference_start"),
    "offload_decision_s": ("ts_offload_decision_made", "ts_sml_inference_end"),
    "edge_buffer_wait_s": ("ts_sample_sent_to_edge_server", "ts_sample_sent_to_offloading"),
    "edge_to_server_network_s": (
        "ts_sample_received_at_edge_server",
        "ts_sample_sent_to_edge_server",
    ),
    "server_queue_or_preprocess_s": (
        "ts_lml_inference_start",
        "ts_sample_received_at_edge_server",
    ),
    "lml_inference_s": ("ts_lml_inference_end", "ts_lml_inference_start"),
    "server_postprocess_s": ("ts_results_sent_to_edge_device", "ts_lml_inference_end"),
    "server_to_edge_network_s": (
        "ts_results_received_from_edge_server",
        "ts_results_sent_to_edge_device",
    ),
    "offload_roundtrip_s": (
        "ts_results_received_from_edge_server",
        "ts_sample_sent_to_edge_server",
    ),
    "edge_receive_to_saved_s": (
        "ts_results_received_from_offloading_module",
        "ts_results_received_from_edge_server",
    ),
}
