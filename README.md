# Batch Hierarchical Inference Framework (Batch HI-Framework)

This repository contains the code and documentation for a two-layer Hierarchical Inference (HI) framework. The framework facilitates the development, testing, and deployment of inference tasks split between a resource-constrained edge device and a more powerful edge server.

The framework is designed around the ExPECA testbed environment at KTH Royal Institute of Technology, but it can be adapted for other environments and hardware setups.
It is built as part of an MSc thesis project at KTH.

## Overview

The core idea is to run a lightweight Small Model (SML) on an **Edge Device** for initial, fast inference. Based on configurable criteria (e.g., confidence thresholds), the task can be offloaded to an **Edge Server** running a more complex, Large Model (LML) for higher accuracy. This balances latency, resource usage, and accuracy. An **External Controller** script manages the overall experiment flow, including configuration, data feeding, and results collection.

## System Architecture

The framework consists of the following main components:

**Edge Device (`app/edge_device/`)**:

- Runs a lightweight model (SML).
- Receives input data (e.g., images) via a FastAPI endpoint (`/predict`).
- Performs initial inference with the SML.
- Decides whether to offload based on the chosen `decision_method` (`offloading_decision_maker.py`).
- If offloading, sends the data to the Edge Server using the selected `offloading_strategy` (`sample_offloading_method.py`).
- Logs detailed performance metrics (timings, predictions, confidence).
- Configurable via a FastAPI endpoint (`/config`) and environment variables.

**Edge Server (`app/edge_server/`)**:

- Runs a computationally intensive model (LML).
- Receives potentially batched data from the Edge Device via a FastAPI endpoint (`/predict`).
- Performs inference using the LML.
- Returns the results (prediction, confidence) to the Edge Device.
- Logs detailed performance metrics (timings, predictions, confidence).
- Configurable via a FastAPI endpoint (`/config`) and environment variables.

**Offloading Logic (`app/edge_device/`)**:

- `offloading_decision_maker.py` - Implements strategies to decide *if* a sample should be offloaded (e.g., based on SML confidence).
- `sample_offloading_method.py` - Implements strategies for *how* samples are sent to the server (e.g., individually or batched).

**External Controller (`app/controller_external/`)**:

- Orchestrates the experiments.
- Sends configuration details (model paths, strategies, etc.) to the Edge Device and Edge Server.
- Feeds input samples to the Edge Device.
- Collects and aggregates results from the logs.

**Internal Controller (`app/controller_internal/`)**:

- Similar to the external controller, but is built into the Edge Device container.
- Useful for setups where there is no direct access to the Edge Device.

## Features

**Offloading Decision Logics:**

- `never_offload` - Only local inference on the Edge Device (S-ML).
- `always_offload` - Both inference on the Edge Device (S-ML) and the Edge Server (L-ML).
- `fixed_threshold` - Selective offloading based on fixed confidence threshold.
- `adaptive_threshold` - Selective offloading based on adaptive confidence threshold.
  - NOTE: `offloading_decision_maker.py` for details on the adaptive threshold.

**Sample Batching Logics:**

- `send_individually` - Samples are sent immediately and individually.
- `size_based_batching` - Batch is sent when a set number of samples is collected.
- `buffer_based_batching` - Batch is sent when a set time has passed since the first sample was collected.
- `combined_batching` - Batch is sent when either a set number of samples or a set time has passed.

**Logging:**

| Field | Description |
|-------|-------------|
| `UUID` | Unique identifier for each inference task |
| `Filename` | Input file that was processed |
| `True Class` | Actual classification label |
| `SML Prediction` | Small Model prediction result |
| `SML Confidence` | Small Model confidence score |
| `Offloaded` | Whether the task was offloaded to the server |
| `Buffered` | Whether the task was held in a batch buffer |
| `LML Prediction` | Large Model prediction result |
| `LML Confidence` | Large Model confidence score |
| `ts_sml_inference_start` | Timestamp when SML inference started |
| `ts_sml_inference_end` | Timestamp when SML inference completed |
| `ts_offload_decision_made` | Timestamp when offloading decision was made |
| `ts_results_saved_not_offloaded` | Timestamp when results were saved (non-offloaded cases) |
| `ts_sample_sent_to_offloading` | Timestamp when sample was sent to offloading module |
| `ts_sample_sent_to_edge_server` | Timestamp when sample was sent to edge server |
| `ts_sample_received_at_edge_server` | Timestamp when sample arrived at edge server |
| `ts_lml_inference_start` | Timestamp when LML inference started |
| `ts_lml_inference_end` | Timestamp when LML inference completed |
| `ts_results_sent_to_edge_device` | Timestamp when results were sent back to edge device |
| `ts_results_received_from_edge_server` | Timestamp when results arrived from edge server |
| `ts_results_received_from_offloading_module` | Timestamp when results were received from offloading module |
| `ts_threshold_updated` | Processing time for updating the adaptive threshold |
| `ts_total_time` | Total processing time for the inference task |

**Model Architecture Support:**

- Support for extending available sml and lml models in `edge_device.py` and `edge_server.py`.
  - Models are located in `data/models/sml/` and `data/models/lml/`.
- Support for extending available architectures in `edge_device.py` and `edge_server.py`.

## Project Structure

```plaintext
Hierarchical-Inference-Framework/
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   ├── edge_device/
│   │   ├── edge_device.py                   # Main Edge Device logic (SML inference, API)
│   │   ├── offloading_decision_maker.py     # Logic for making offloading decisions
│   │   ├── sample_offloading_method.py      # Logic for batching and sending samples
│   │   ├── entrypoint_edge_device.sh        # Docker entrypoint
│   │   └── Dockerfile.edge_device           # Dockerfile for Edge Device
│   ├── edge_server/
│   │   ├── edge_server.py                   # Main Edge Server logic (LML inference, API)
│   │   ├── entrypoint_edge_server.sh        # Docker entrypoint
│   │   └── Dockerfile.edge_server           # Dockerfile for Edge Server
│   ├── controller_internal/
│   │   ├── controller_inside_container.py   # Controller for container execution inside Edge Device
│   │   ├── model_registry.py                # SML and LML model registry for internal controller
│   │   └── start.sh.               # Script to start the internal controller
│   └── controller_external/
│       ├── controller_local_scripts.py      # Controller for local script execution
│       ├── controller_local_containers.py   # Controller for local container execution
│       └── controller_testbed_containers.py # Controller for testbed container execution
│
├── config/
│   └── .env                                 # Environment variables (ports, IPs)
│
├── data/
│   ├── datasets/
│   │   ├── imagenette/                      # Imagenette dataset
│   │   └── imagenetV2/                      # ImageNetV2 dataset
│   └── models/
│       ├── lml/                             # LML models
│       └── sml/                             # SML models
│
├── documentation/
│   ├── ExPECA_HI_setup.ipynb                # Notebooks for ExPECA setup
│   └── ExPECA_raspberry_setup.md            # ExPECA Raspberry Pi setup
│
├── results/                                 # Output directory for logs and results
│   ├── Full system results and logs
│   ├── Edge Device results and logs
│   └── Edge Server results and logs
│
└── scripts_containers/                      # Helper scripts for Docker building and management
    ├── build_edge_device_amd64.ps1
    ├── build_edge_device_arm64.ps1
    ├── build_edge_server_amd64.ps1
    ├── start_local_containers_amd64.ps1
    ├── start_local_containers_arm64.ps1
    └── tag_and_push.ps1
```

## Setup and Installation

**1. Clone the Repository:**

```bash
git clone <your-repository-url>
cd Hierarchical-Inference-Framework
```

**2. Install Dependencies:**
Install the required packages:

```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables:**
Create the `config/.env` file and modify it:

```dotenv
# config/.env
EDGE_DEVICE_IP=edge_device
EDGE_SERVER_IP=edge_server
EDGE_DEVICE_PORT=8000
EDGE_SERVER_PORT=8001
DNS_IP=
GATEWAY_IP=
PASS=
```

**4. Prepare Data and Models:**

- Place your datasets in the `data/datasets/` directory.
- Place your pre-trained SML models in `data/models/sml/`.
- Place your pre-trained LML models in `data/models/lml/`.
- Ensure the model architectures you use are supported in the `load_model` functions within `edge_device.py` and `edge_server.py`, or add support for new architectures.

## Configuration

Configuration is managed through environment variables and parameters passed via the external controller scripts.

- **Environment Variables**: Define network settings (IPs, ports) as described in the Setup section.
- **Experiment Parameters**: The controller scripts (`app/controller_external/`) define the specific parameters for each experiment run:

| Parameter                   | Description                                                                      | Example Value                           |
| :-------------------------- | :------------------------------------------------------------------------------- | :-------------------------------------- |
| `sample_path`               | Path to the input samples (e.g., images).                                        | `"data/datasets/imagenette/"`           |
| `sml_model`                 | Path to the SML file for the Edge Device.                                        | `"data/models/sml/mobilenetv2_sml.pth"` |
| `sml_architecture`          | Architecture name of the SML model (must match keys in `edge_device.py`).        | `"mobilenet_v2"`                        |
| `lml_model`                 | Path to the LML file for the Edge Server.                                        | `"data/models/lml/resnet50_lml.pth"`    |
| `lml_architecture`          | Architecture name of the LML model (must match keys in `edge_server.py`).        | `"resnet50"`                            |
| `decision_method`           | Selected offloading decision logic.                                              | `"adaptive_threshold"`                  |
| `fixed_threshold_value`     | Static confidence threshold (used if `decision_method` is `fixed_threshold`).    | `0.85`                                  |
| `offloading_strategy`       | Selected sample batching logic.                                                  | `"size_based_batching"`                 |
| `batch_size`                | Number of samples per batch (for size-based batching).                           | `10`                                    |
| `batch_wait_time`           | Max time to wait before sending a batch (for time-based batching).               | `0.5` (seconds)                         |
| `wait_time_between_samples` | Delay between sending samples from the controller to the Edge Device.            | `0.1` (seconds)                         |

## Usage: Running Experiments

Choose one of the following methods to run your experiments:

1. Locally with Python scripts
2. Locally with Docker containers
3. Remotely with Docker containers (e.g., ExPECA Testbed)
4. Remotely with Internal Controller (e.g., ExPECA Testbed)

### 1. Locally with Python Scripts

Simple setup, no Docker required. Good for development, initial testing and debugging.

**Steps:**

1. Open three separate terminals.
2. In terminal 1, start the Edge Server:

    ```bash
    python app/edge_server/edge_server.py
    ```

3. In terminal 2, start the Edge Device:

    ```bash
    python app/edge_device/edge_device.py
    ```

4. Configure your experiment parameters within `app/controller_external/controller_local_scripts.py`.
5. Run the controller script in terminal 3:

    ```bash
    python app/controller_external/controller_local_scripts.py
    ```

6. Results and logs will be saved in the `results/` directory.

### 2. Locally with Docker Containers

For testing the framework in a containerized environment before deploying to remote hardware.

**Steps:**

1. Ensure Docker Desktop (or Docker Engine on Linux) is installed and running.
2. Build containers using the provided PowerShell script:
    - There are amd64 and arm64 versions of the containers available.

    ```powershell
    # On Windows with PowerShell
    .\scripts\build_edge_device_amd64.ps1
    .\scripts\build_edge_device_arm64.ps1
    .\scripts\build_edge_server_amd64.ps1
    ```

3. Start containers using the provided PowerShell script:

    ```powershell
    # On Windows with PowerShell
    .\scripts\start_local_containers_amd64.ps1
    .\scripts\start_local_containers_arm64.ps1
    ```

4. Configure your experiment parameters within `app/controller_external/controller_local_containers.py`.
5. Run the controller script:

    ```bash
    python app/controller_external/controller_local_containers.py
    ```

6. Results and logs will be saved in the `results/` directory.
7. Stop the containers when done.

### 3. Remotely with Docker Containers (e.g., ExPECA Testbed)

**Steps:**

1. Build the Docker images locally (as in step 2.2 above).
2. Tag and push the images to a container registry (e.g., Docker Hub).

    ```powershell
    # On Windows with PowerShell (after updating script with your Docker Hub username)
    .\scripts\tag_push.ps1
    ```

3. Set up the remote environment. For ExPECA use `notebooks/ExPECA_HI_setup_Public_IP.ipynb` and follow the instructions in the notebook.
4. Configure your experiment parameters within `app/controller_external/controller_local_containers.py`.
5. Run the controller script:

    ```bash
    python app/controller_external/controller_local_containers.py
    ```

6. Results and logs will be saved in the `results/` directory.
7. Stop the containers when done.
8. Remove the leases on the ExPECA testbed.

### 4. Remotely with Internal Controller (e.g., ExPECA Testbed)

1. Build the Docker images locally (as in step 2.2 above).
2. Tag and push the images to a container registry (e.g., Docker Hub).
3. Set up the remote environment. For ExPECA use `notebooks/ExPECA_HI_setup_EP5G.ipynb` and follow the instructions in the notebook.

## Resources

List of resources used in the project:

## Resources

List of main resources used in the project:

- [PyTorch](https://pytorch.org/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [Imagenette](https://github.com/fastai/imagenette)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [ExPECA Testbed](https://expeca.proj.kth.se/)

## Future Work / Improvements

- Implement more advanced dynamic batching strategies.
- Implement more sophisticated adaptive offloading decision algorithms.
- Expand support for more model architectures and ML frameworks.
- Support different data types beyond images.
- Enhance monitoring and visualization.
