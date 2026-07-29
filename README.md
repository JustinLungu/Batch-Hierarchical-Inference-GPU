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

## First-Time Local Setup

### 1. Install `uv`

This project uses `uv` for Python installation, dependency resolution, the
locked environment, and `.venv` creation. Do not install the project
dependencies manually with `pip`.

Install `uv` on Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell if necessary, then verify:

```bash
uv --version
```

Other installation methods are documented in the
[`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Create And Activate `.venv`

From the repository root:

```bash
scripts/setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` runs `uv sync` with the configured Python version. `uv` creates
`.venv`, installs the locked dependencies from `uv.lock`, and includes the
Jupyter/ExPECA packages declared in `pyproject.toml`.

In VS Code, select `.venv/bin/python` as the notebook kernel and restart the
kernel after resyncing the environment.

### 3. Download The Dataset And Models

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

## Continue With The Full Runbook

Local setup alone is not enough to launch the experiment. A first-time user
must also create a Docker registry account, build and push the correct
CPU/GPU/ARM64 images, obtain ExPECA credentials, reserve workers, create the
containers, and copy their public IPs into the runtime configuration.

Follow the complete guide before running the benchmark:

**[CPU/GPU benchmark runbook](docs/thesis_reproduction.md)**

After completing the runbook and confirming that both remote services are
reachable:

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

For the original thesis summary and complete CPU/GPU execution instructions, see
[`docs/`](docs/).
