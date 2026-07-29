# Batch Hierarchical Inference GPU Experiments

This repository extends the Batch Hierarchical Inference framework for
reproducible CPU and GPU experiments on ExPECA.

The runtime pipeline is:

```text
local benchmark controller
  -> ExPECA edge-device container: MobileNetV3-Large SML
      -> ExPECA CPU/GPU edge-server container: ViT-H/14 LML
```

The benchmark reproduces the seven thesis configurations, records the full
per-sample timeline, generates thesis-style figures, and supports additional
analysis of the actual dynamic offload batches.

## What This Repository Adds

- Automated ImageNetV2 and model downloads with validation.
- Centralized runtime, model, dataset, and configuration-matrix files.
- Reproducible CPU, CUDA, and ARM64 Raspberry Pi container builds.
- ExPECA deployment notebooks for public-IP CPU, GPU/Raspberry Pi, and EP5G
  setups.
- A non-interactive runner for configurations `001`-`007`.
- Per-sample timing, accuracy, offloading, batching, and throughput analysis.
- Thesis-style plots reconstructed from the timestamps saved by the services.
- Memory-aware CUDA micro-batching with configurable limits and OOM recovery.
- Post-processing for actual offload batch sizes in configurations `005`-`007`.

## Local Setup

### 1. Create The Environment

```bash
scripts/setup_env.sh
source .venv/bin/activate
scripts/setup_expeca_notebook_env.sh
```

Restart the VS Code/Jupyter kernel after updating the notebook environment.

### 2. Download The Dataset And Models

```bash
scripts/download_dataset.sh --imagenetv2
scripts/download_models.sh --all
scripts/prepare_expeca_author_layout.sh
```

The benchmark uses:

```text
Dataset: ImageNetV2 Matched Frequency, 10,000 images
SML:     MobileNetV3-Large
LML:     ViT-H/14
```

The download script validates that ImageNetV2 contains 1,000 class directories
and 10,000 images.

### 3. Configure The Runtime

Edit `config/experiment.env`. This is the normal user-editable configuration
file. Set:

- `DEVICE`: `cpu` or `cuda`.
- `EDGE_SERVER_IP` and `EDGE_DEVICE_IP`: values printed by the deployment
  notebook.
- `CONTROLLER_MAX_SAMPLES`: a small integer for validation or `all`.
- `THESIS_CONFIGS_TO_RUN`: `all` for CPU or usually
  `002,003,004,005,006,007` for GPU.
- `EXPECA_IMAGE_NAMESPACE`: your registry namespace.
- `EXPECA_EDGE_SERVER_DEVICE`: `cpu` or `cuda`.
- `LML_*`: server-side inference batching limits.

Stable paths and image tags live in `config/defaults.env`. The fixed thesis
dataset/model pair lives in `config/thesis_reproduction.env`, and the seven
experiment definitions live in `config/thesis_configs.csv`.

## Run The Benchmark

Building images, reserving ExPECA workers, creating containers, running CPU/GPU
experiments, and releasing resources are documented in:

[CPU/GPU benchmark runbook](docs/thesis_reproduction.md)

After the remote services are running and their IPs are configured:

```bash
.venv/bin/python src/run_benchmark.py --dry-run
.venv/bin/python src/run_benchmark.py
```

Regenerate plots without rerunning ExPECA:

```bash
.venv/bin/python src/run_benchmark.py --plot-only
```

## Configuration Matrix

| Config | Decision Method | Offloading Strategy | Controller Batch |
|---|---|---|---:|
| `001` | `never_offload` | `send_individually` | 1 |
| `002` | `always_offload` | `send_individually` | 1 |
| `003` | `fixed_threshold` | `send_individually` | 1 |
| `004` | `adaptive_threshold` | `send_individually` | 1 |
| `005` | `adaptive_threshold` | `dynamic_batching` | 5 |
| `006` | `adaptive_threshold` | `dynamic_batching` | 15 |
| `007` | `adaptive_threshold` | `dynamic_batching` | 45 |

For `dynamic_batching`, the controller batch is processed by the edge device
and only the selected subset is sent to the server. It does not wait until 5,
15, or 45 offloaded samples have accumulated. This matches the original
application logic and the average offload batches reported in the thesis.

## Main Findings

The completed full-dataset runs support the central result: GPU inference
substantially reduces the large-model server bottleneck.

- In always-offload config `002`, recorded mean LML inference decreased from
  approximately `2.27 s` on CPU to `0.27 s` on GPU, an approximately `8.5x`
  reduction.
- Config `002` throughput increased from approximately `0.40` to `0.93`
  samples/s.
- GPU throughput increased with dynamic controller batches, reaching
  approximately `2.47` samples/s in config `007`.
- Larger dynamic batches take longer per server request, but process more
  samples together and improve effective throughput.
- Public-IP communication produced much smaller communication components than
  the thesis EP5G path.

These are not perfectly controlled end-to-end hardware comparisons. The CPU
and GPU runs used different edge-device placements, and public-IP networking
does not reproduce the thesis radio path. Config `002` is the clearest
server-side CPU/GPU comparison because every image follows the LML path.
Adaptive configurations can also diverge when small prediction differences
alter the threshold trajectory.

## Actual Offload Batch Analysis

Analyze how edge-server response time changes with the actual selected batch
sizes in configs `005`-`007` without rerunning ExPECA:

```bash
# GPU results, using the default directory
.venv/bin/python src/analyze_offload_batches.py

# A specific CPU result directory
.venv/bin/python src/analyze_offload_batches.py results/CPU_thesis_reproduction
```

The analysis reports request time, per-image time, effective throughput,
grouped statistics, and trends. See
[`src/offload_batch_analysis/README.md`](src/offload_batch_analysis/README.md)
for output definitions.

## Repository Layout

```text
app/         Edge-device and edge-server FastAPI services
config/      Runtime settings and experiment definitions
data/        Downloaded datasets and model checkpoints
docs/        Detailed runbook and thesis notes
notebooks/   ExPECA resource deployment notebooks
results/     Raw, derived, aggregate, and plotted experiment outputs
scripts/     Environment, download, build, and push helpers
src/         Benchmark runner and post-processing packages
```

Further documentation:

- [`docs/thesis_reproduction.md`](docs/thesis_reproduction.md): complete CPU/GPU
  execution runbook.
- [`docs/thesis_batch_hi_summary.md`](docs/thesis_batch_hi_summary.md): detailed
  account of the original thesis.
- [`notebooks/README.md`](notebooks/README.md): notebook selection and resource
  lifecycle.
- [`app/README.md`](app/README.md): runtime request flow and service behavior.
- [`scripts/README.md`](scripts/README.md): helper script reference.
- [`src/README.md`](src/README.md): runner and analysis package overview.
