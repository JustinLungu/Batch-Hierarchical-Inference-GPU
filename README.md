# Batch Hierarchical Inference GPU Experiments

This repository contains the Batch Hierarchical Inference framework adapted for
reproducing the thesis experiments on ExPECA and benchmarking GPU offloading.

The current workflow is centered on:

- a Raspberry Pi / ARM64 edge-device container running the SML,
- an ExPECA CPU or GPU edge-server container running the LML,
- a local Python controller in `src/run_thesis_reproduction.py`,
- reproducible configuration in `config/`,
- Docker image build/push helpers in `scripts/`,
- thesis-style CSV summaries and plots in `results/`.

## Main Goal

The experiment reproduces the seven thesis configurations and extends them to
GPU LML inference. For GPU benchmarking, config `001` is skipped because it
never offloads to the server.

The important configuration IDs are:

| Config | Decision Method | Offloading Strategy | Controller Batch |
|---|---|---|---:|
| `001` | `never_offload` | `send_individually` | 1 |
| `002` | `always_offload` | `send_individually` | 1 |
| `003` | `fixed_threshold` | `send_individually` | 1 |
| `004` | `adaptive_threshold` | `send_individually` | 1 |
| `005` | `adaptive_threshold` | `dynamic_batching` | 5 |
| `006` | `adaptive_threshold` | `dynamic_batching` | 15 |
| `007` | `adaptive_threshold` | `dynamic_batching` | 45 |

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

docs/                       Thesis and ExPECA run documentation
notebooks/                  ExPECA setup notebooks
scripts/                    Setup, download, build, and push helpers
src/                        Controller, analysis, and plotting code
```

## Environment

This project uses `uv` and `pyproject.toml` for the local Python environment.

```bash
scripts/setup_env.sh
source .venv/bin/activate
```

Notebook dependencies are part of the same environment. If VS Code asks to
install `ipykernel`, restart the kernel first; `ipykernel` is already managed by
`uv`.

## Data And Models

Download the thesis dataset and model checkpoints:

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

## Docker Images

Set your Docker Hub namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE
```

Build and push CPU baseline images:

```bash
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

Build and push the GPU edge-server image:

```bash
scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

Build and push the Raspberry Pi / ARM64 edge-device image:

```bash
scripts/build_expeca_raspberry_pi_image.sh
scripts/push_expeca_raspberry_pi_image.sh
```

The ARM64 build requires Docker Buildx and ARM64 emulation on an amd64 laptop.

## ExPECA Notebooks

Use:

```text
notebooks/ExPECA_HI_setup_GPU_RaspberryPi_Public_IP.ipynb
```

for the GPU/Raspberry Pi benchmark.

The GPU notebook creates separate containers:

```text
hi-gpu-edge-server
hi-gpu-edge-device
```

so existing CPU containers named `hi-edge-server` and `hi-edge-device` can stay
alive.

After the notebook creates both containers, copy the printed public IPs into
`config/experiment.env`:

```env
DEVICE=cuda
EXPECA_EDGE_SERVER_DEVICE=cuda
EDGE_SERVER_IP=GPU_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=RASPBERRY_PI_PUBLIC_IP
THESIS_CONFIGS_TO_RUN=002,003,004,005,006,007
```

## Run Thesis Reproduction

Dry-run first:

```bash
.venv/bin/python src/run_thesis_reproduction.py --dry-run
```

Run the experiment:

```bash
.venv/bin/python src/run_thesis_reproduction.py
```

Outputs are written to:

```text
results/thesis_reproduction/       CPU run
results/thesis_reproduction_gpu/   GPU run
```

Each run contains:

```text
summary.csv
summary.md
run_metadata.json
latency_breakdown.csv
offloading_distribution.csv
per_sample_latency.csv
threshold_trajectory.csv
plots/
config_00*/raw_edge_device_results.csv
config_00*/timing_results.csv
```

Regenerate plots from existing CSV outputs:

```bash
.venv/bin/python src/run_thesis_reproduction.py --plot-only
```

## GPU LML Batching

The GPU edge-server supports memory-aware LML batching through
`config/experiment.env`:

```env
LML_BATCHING_MODE=auto
LML_INITIAL_BATCH_SIZE=16
LML_MIN_BATCH_SIZE=1
LML_MAX_BATCH_SIZE=256
LML_GPU_MEMORY_FRACTION=0.9
LML_OOM_RETRY=true
```

`auto` runs sequentially on CPU and adaptively on CUDA.

## Useful Documentation

- `docs/thesis_batch_hi_summary.md`: detailed thesis summary.
- `docs/thesis_reproduction.md`: CPU/GPU reproduction runbook.
- `scripts/README.md`: setup and Docker helper script overview.
- `src/README.md`: runner and analysis output details.
