# Batch Hierarchical Inference GPU Experiments

This repository contains the Batch Hierarchical Inference framework adapted for thesis reproduction on ExPECA and GPU offloading benchmarks.

The main pipeline is:

```text
local controller -> ExPECA edge-device container -> ExPECA CPU/GPU edge-server container
```

The edge-device runs the SML. The edge-server runs the LML. Results are saved as thesis-style CSV summaries and plots.

## Quickstart

For normal use, edit only `config/experiment.env`.

### 1. Set Up Python

```bash
scripts/setup_env.sh
source .venv/bin/activate
scripts/setup_expeca_notebook_env.sh
```

If VS Code complains about the notebook kernel, restart the kernel after running the setup script.

### 2. Download Data And Models

```bash
scripts/download_dataset.sh --imagenetv2
scripts/download_models.sh --all
scripts/prepare_expeca_author_layout.sh
```

The thesis reproduction uses:

```text
Dataset: data/datasets/imagenetV2/matched-frequency-format-val
SML:     mobilenet_v3_large
LML:     vit_h_14
```

### 3. Configure The Run

Open `config/experiment.env` and set your Docker namespace, public IPs, device, sample count, and config list.

CPU example:

```env
EXPECA_IMAGE_NAMESPACE=<your-dockerhub-or-registry-namespace>
DEVICE=cpu
EXPECA_EDGE_SERVER_DEVICE=cpu
CONTROLLER_MAX_SAMPLES=100
THESIS_CONFIGS_TO_RUN=all
```

GPU example:

```env
EXPECA_IMAGE_NAMESPACE=<your-dockerhub-or-registry-namespace>
DEVICE=cuda
EXPECA_EDGE_SERVER_DEVICE=cuda
CONTROLLER_MAX_SAMPLES=all
THESIS_CONFIGS_TO_RUN=002,003,004,005,006,007
LML_BATCHING_MODE=auto
LML_INITIAL_BATCH_SIZE=16
LML_MIN_BATCH_SIZE=1
LML_MAX_BATCH_SIZE=256
LML_GPU_MEMORY_FRACTION=0.9
LML_OOM_RETRY=true
```

Config `001` is skipped for GPU because it never offloads to the server.

### 4. Build And Push Images

CPU baseline images:

```bash
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

GPU edge-server image:

```bash
scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

Raspberry Pi / ARM64 edge-device image:

```bash
scripts/build_expeca_raspberry_pi_image.sh
scripts/push_expeca_raspberry_pi_image.sh
```

The ARM64 build requires Docker Buildx and ARM64 emulation on an amd64 laptop.

### 5. Start ExPECA Containers

CPU reproduction notebook:

```text
notebooks/ExPECA_HI_setup_Public_IP.ipynb
```

GPU server plus Raspberry Pi edge-device notebook:

```text
notebooks/ExPECA_HI_setup_GPU_RaspberryPi_Public_IP.ipynb
```

Run the notebook cells to reserve workers and create containers. Then copy the printed public IPs into `config/experiment.env`:

```env
EDGE_SERVER_IP=<edge-server-public-ip>
EDGE_DEVICE_IP=<edge-device-public-ip>
```

Check that both services are reachable:

```bash
curl http://$EDGE_SERVER_IP:8001/logs
curl http://$EDGE_DEVICE_IP:8000/logs
```

### 6. Run The Benchmark

Dry-run first:

```bash
.venv/bin/python src/run_thesis_reproduction.py --dry-run
```

Run the experiment:

```bash
.venv/bin/python src/run_thesis_reproduction.py
```

Regenerate plots from existing CSV outputs:

```bash
.venv/bin/python src/run_thesis_reproduction.py --plot-only
```

## Results

CPU results are written to:

```text
results/thesis_reproduction/
```

GPU results are written to:

```text
results/thesis_reproduction_gpu/
```

Each run contains an aggregate summary, metadata, thesis-style plot CSVs/images, and per-config raw edge-device plus derived timing CSVs.

## Thesis Configurations

| Config | Decision Method | Offloading Strategy | Controller Batch |
|---|---|---|---:|
| `001` | `never_offload` | `send_individually` | 1 |
| `002` | `always_offload` | `send_individually` | 1 |
| `003` | `fixed_threshold` | `send_individually` | 1 |
| `004` | `adaptive_threshold` | `send_individually` | 1 |
| `005` | `adaptive_threshold` | `dynamic_batching` | 5 |
| `006` | `adaptive_threshold` | `dynamic_batching` | 15 |
| `007` | `adaptive_threshold` | `dynamic_batching` | 45 |

These are defined in `config/thesis_configs.csv`. Fixed thesis dataset/model choices live in `config/thesis_reproduction.env`. Runtime values live in `config/experiment.env`.

## Repository Layout

```text
app/
  edge_device/              FastAPI edge-device service, SML, offloading logic
  edge_server/              FastAPI edge-server service, LML, CPU/GPU inference

config/
  defaults.env              Stable paths, ports, image tags, dataset/model paths
  experiment.env            Active runtime settings, IPs, CPU/GPU selection
  thesis_configs.csv        Thesis config matrix 001-007
  thesis_reproduction.env   Fixed thesis dataset/model choices

docs/                       Detailed thesis and ExPECA notes
notebooks/                  ExPECA setup notebooks
scripts/                    Setup, download, build, and push helpers
src/                        Controller, analysis, and plotting code
```

## Extra Documentation

- `docs/thesis_batch_hi_summary.md`: detailed thesis summary.
- `docs/thesis_reproduction.md`: detailed CPU/GPU reproduction notes.
- `scripts/README.md`: helper script overview.
- `src/README.md`: runner and analysis internals.
