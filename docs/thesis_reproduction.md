# Thesis Reproduction Workflow

This document is the main runbook for reproducing the seven configurations from
the thesis and then repeating the same reproduction with a GPU edge server.

The workflow is:

```text
laptop runner
  -> ExPECA edge-device container
      -> ExPECA edge-server container
```

The edge-device always runs the small model. The edge-server runs the large
model. Switching from CPU to GPU should only change the edge-server compute
device and image, not the experiment definitions.

## 1. Prepare Local Assets

Set up the Python environment and download the thesis dataset/models:

```bash
scripts/setup_env.sh
source .venv/bin/activate
scripts/download_dataset.sh --imagenetv2
scripts/download_models.sh --all
scripts/prepare_expeca_author_layout.sh
scripts/setup_expeca_notebook_env.sh
```

The thesis reproduction requires:

```text
data/datasets/imagenetV2/matched-frequency-format-val
data/models/sml/mobilenet_v3_large_imagenet1k_v2.pth
data/models/lml/ViT_H_14_Weights_IMAGENET1K_SWAG_E2E_V1.pth
```

`scripts/download_dataset.sh --imagenetv2` validates that ImageNetV2 contains
1000 class folders and 10000 images.

## 2. Understand the Config Files

There are four config files:

```text
config/defaults.env
config/experiment.env
config/thesis_reproduction.env
config/thesis_configs.csv
```

`config/defaults.env` contains stable paths, download URLs, ports, and default
Docker image tags.

`config/experiment.env` is the runtime file you edit before a run:

```env
DEVICE=cpu
EDGE_DEVICE_IP=...
EDGE_SERVER_IP=...
CONTROLLER_MAX_SAMPLES=all
EXPECA_IMAGE_NAMESPACE=...
EXPECA_EDGE_SERVER_DEVICE=cpu
```

Use `CONTROLLER_MAX_SAMPLES=4` or another small number for a quick validation
run. Use `CONTROLLER_MAX_SAMPLES=all` for full thesis reproduction.

`config/thesis_reproduction.env` fixes the thesis dataset and model pair:

```text
ImageNetV2 Matched Frequency
MobileNetV3-Large
ViT-H/14
```

`config/thesis_configs.csv` defines the seven thesis configurations. These are
data, not Python code:

```text
001 never_offload
002 always_offload + send_individually
003 fixed_threshold + send_individually, threshold 0.3888
004 adaptive_threshold + send_individually
005 adaptive_threshold + dynamic_batching, controller batch 5
006 adaptive_threshold + dynamic_batching, controller batch 15
007 adaptive_threshold + dynamic_batching, controller batch 45
```

Config `002` is intentionally kept. In the thesis it is the accuracy upper bound
and high-latency baseline. For our GPU work it is also the cleanest way to
isolate large-model/server behavior because every sample is sent to the LML.

## 3. Build and Push CPU Images

Log in to Docker Hub or your container registry:

```bash
docker login
```

Set your namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE
```

Build and push the CPU images:

```bash
scripts/check_expeca_cpu_prereqs.sh
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

## 4. Start CPU Containers on ExPECA

Open:

```text
notebooks/ExPECA_HI_setup_Public_IP.ipynb
```

Run the authentication/setup cells first. Use the same image namespace/tag you
pushed. For the CPU reproduction:

```python
EDGE_SERVER_DEVICE = "cpu"
EDGE_DEVICE_DEVICE = "cpu"
```

Create the edge-server container first and record its public IP. Then create the
edge-device container and record its public IP. The edge-device environment must
receive the edge-server public IP.

Confirm both services are reachable:

```bash
curl http://EDGE_SERVER_PUBLIC_IP:8001/logs
curl http://EDGE_DEVICE_PUBLIC_IP:8000/logs
```

Then put those public IPs in `config/experiment.env`:

```env
DEVICE=cpu
EDGE_SERVER_IP=EDGE_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=EDGE_DEVICE_PUBLIC_IP
EXPECA_EDGE_SERVER_DEVICE=cpu
```

## 5. Preview the Thesis Run

Before launching a long run, print the resolved configuration table:

```bash
.venv/bin/python src/run_thesis_reproduction.py --dry-run
```

This validates local thesis assets and prints configs `001` through `007`.

## 6. Run the CPU Thesis Reproduction

For a short validation:

```env
CONTROLLER_MAX_SAMPLES=4
```

For the full reproduction:

```env
CONTROLLER_MAX_SAMPLES=all
```

Run:

```bash
.venv/bin/python src/run_thesis_reproduction.py
```

The runner executes all seven configurations in order. Each configuration sends
a fresh `/config` request to the edge-device and edge-server. The edge-device
clears previous results, old offload buffers, and adaptive-threshold state for
each configuration.

Aggregate outputs are written to:

```text
results/analysis_thesis_reproduction_cpu/summary.csv
results/analysis_thesis_reproduction_cpu/summary.md
results/analysis_thesis_reproduction_cpu/run_metadata.json
```

Each individual configuration also writes a detailed analysis folder named with
its config number and strategy.

## 7. Repeat With GPU

After the CPU thesis reproduction works, build and push the GPU edge-server image:

```bash
scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

In the ExPECA notebook, keep the edge-device on CPU and switch only the
edge-server to the GPU image/device:

```python
EDGE_SERVER_DEVICE = "cuda"
EDGE_DEVICE_DEVICE = "cpu"
```

Update `config/experiment.env`:

```env
DEVICE=cuda
EDGE_SERVER_IP=GPU_EDGE_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=EDGE_DEVICE_PUBLIC_IP
EXPECA_EDGE_SERVER_DEVICE=cuda
CONTROLLER_MAX_SAMPLES=all
```

Run the same command:

```bash
.venv/bin/python src/run_thesis_reproduction.py
```

GPU outputs are written to:

```text
results/analysis_thesis_reproduction_cuda/
```

The CPU and GPU runs are directly comparable because they use the same thesis
dataset, model pair, and seven configuration definitions.
