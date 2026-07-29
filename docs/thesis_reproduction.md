# CPU/GPU Benchmark Runbook

This is the operational guide for running the benchmark on ExPECA. Complete
the environment, dataset, and model setup in the
[root README](../README.md#first-time-local-setup) before starting here.

The deployment is:

```text
local benchmark controller
  -> ExPECA edge-device container
      -> ExPECA edge-server container
```

The edge device always runs the SML on CPU. CPU and GPU experiments change the
edge-server image and compute device while retaining the same dataset, model
pair, and experiment definitions.

## Source Material And Credit

This repository reproduces and extends the work provided in:

- Original implementation: [h3nkk44/Batch-Hierarchical-Inference](https://github.com/h3nkk44/Batch-Hierarchical-Inference)
- Original thesis: [DiVA full-text PDF](https://www.diva-portal.org/smash/get/diva2:2035067/FULLTEXT01.pdf)

The thesis configurations, terminology, and comparison figures are based on
these sources.

## First-Time Prerequisites

Before building or deploying anything, prepare the following accounts and
tools.

### Local Tools

- Git.
- Docker Engine with a running Docker daemon.
- Docker Buildx when building the ARM64 Raspberry Pi image from an amd64
  machine.
- `uv` and the project `.venv`, prepared through the
  [root setup](../README.md#first-time-local-setup).
- VS Code/Jupyter with `.venv/bin/python` selected when running notebooks.

Verify the important commands:

```bash
docker --version
docker info
docker buildx version
uv --version
.venv/bin/python --version
```

The normal CPU build does not require cross-platform emulation. The Raspberry
Pi image does: `scripts/build_expeca_raspberry_pi_image.sh` stops with a clear
error when Buildx is unavailable.

### Container Registry Account

Create a [Docker Hub](https://hub.docker.com/) account or use another registry that ExPECA can access.
The build scripts publish:

```text
<namespace>/hi-framework-edge-server:<cpu-or-gpu-tag>
<namespace>/hi-framework-edge-device:<amd64-or-arm64-tag>
```

Set the namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=<registry-namespace>
```

The ExPECA workers must be able to pull these images. Public Docker Hub
repositories are the simplest option. If private images are required, configure
registry authentication according to the ExPECA deployment policy.

Authenticate locally before pushing:

```bash
docker login
```

Create the repositories in the registry first if your registry does not create
them automatically on the initial push.

### ExPECA Access

You need:

1. An ExPECA/Chameleon account.
2. Access to an ExPECA project.
3. Permission to reserve the required CPU, Raspberry Pi, and GPU workers.
4. An OpenRC credential file downloaded from the [ExPECA/Chameleon API Access page](https://testbed.expeca.proj.kth.se/project/api_access/openrc/).

Keep the OpenRC file in the repository root or beside the notebooks so the
authentication cells can find it. OpenRC files contain credentials and must
never be committed; the repository ignores `*-openrc.sh`.

The notebooks reserve shared infrastructure. Confirm worker availability and
release every container and lease when the run is complete.

## 1. Understand The Configuration Files

The benchmark reads four files:

| File | Responsibility |
|---|---|
| `config/defaults.env` | Stable paths, ports, download URLs, and image tags. |
| `config/experiment.env` | Active device, public IPs, sample limit, selected configs, registry namespace, and GPU batching settings. |
| `config/thesis_reproduction.env` | Fixed ImageNetV2, MobileNetV3-Large, and ViT-H/14 choices. |
| `config/thesis_configs.csv` | Configuration matrix `001`-`007`. |

Normally, edit only `config/experiment.env`.

Use a small `CONTROLLER_MAX_SAMPLES` for validation and `all` for a full run.
Config `001` is omitted from GPU runs because it never calls the edge server.

### CPU Settings

```env
DEVICE=cpu
EXPECA_EDGE_SERVER_DEVICE=cpu
CONTROLLER_MAX_SAMPLES=all
THESIS_CONFIGS_TO_RUN=all
LML_BATCHING_MODE=sequential
```

Without `THESIS_OUTPUT_DIR`, CPU results default to:

```text
results/thesis_reproduction/
```

### GPU Settings

```env
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

Without `THESIS_OUTPUT_DIR`, GPU results default to:

```text
results/thesis_reproduction_gpu/
```

`auto` resolves to sequential inference on CPU and adaptive micro-batching on
CUDA. GPU micro-batching divides an already-received offload request into
inference chunks; it does not change the edge device's offloading decisions or
network request size.

## 2. Build And Push Images

Log in to Docker Hub or the configured registry:

```bash
docker login
```

Set your namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=<registry-namespace>
```

### CPU Edge Server And Device

```bash
scripts/check_expeca_cpu_prereqs.sh
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

### GPU Edge Server

```bash
scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

### Raspberry Pi / ARM64 Edge Device

```bash
scripts/build_expeca_raspberry_pi_image.sh
scripts/push_expeca_raspberry_pi_image.sh
```

Building ARM64 on an amd64 laptop requires Docker Buildx and ARM64 emulation.
Image names and tags are printed after successful build and push operations.

Rebuild and recreate containers whenever runtime code under `app/`, dependency
files, model files, or Dockerfiles change. Analysis-only changes under `src/`
do not require new container images.

## 3. Deploy A CPU Experiment

Open:

```text
notebooks/ExPECA_HI_setup_Public_IP.ipynb
```

Follow the notebook in order:

1. Authenticate with your local OpenRC file.
2. Inspect stale containers and leases.
3. Reserve and create the CPU edge server.
4. Record its public IP.
5. Reserve and create the edge device.
6. Pass the edge-server IP to the edge-device container.
7. Wait for both services to report `Running`.

For the closest thesis hardware layout, use a Raspberry Pi/ARM worker for the
edge device and a separate CPU worker for the edge server. Placing both
containers on one server is suitable for smoke testing but changes edge
processing, contention, and network timing.

## 4. Deploy A GPU Experiment

Open:

```text
notebooks/ExPECA_HI_setup_GPU_RaspberryPi_Public_IP.ipynb
```

The notebook is prepared for:

- GPU edge server on `worker-05`, using the CUDA image and NVIDIA runtime.
- Raspberry Pi/ARM64 edge device on `worker-21`, using the ARM64 image.
- The controller on the local laptop.

Run the cells in order, confirm that both containers reach `Running`, and check
the edge-server logs for:

```text
Device: cuda
```

Worker assignments are shared infrastructure and may change. Confirm current
availability before reserving them.

For an EP5G rather than public-IP experiment, use
`notebooks/ExPECA_HI_setup_EP5G.ipynb`. Its network path is materially
different, so do not compare its communication timings as though they came
from the public-IP setup.

## 5. Configure And Verify The Endpoints

Copy the public IPs printed by the notebook into `config/experiment.env`:

```env
EDGE_SERVER_IP=<edge-server-public-ip>
EDGE_DEVICE_IP=<edge-device-public-ip>
```

Verify both services:

```bash
curl http://$EDGE_SERVER_IP:8001/logs
curl http://$EDGE_DEVICE_IP:8000/logs
```

The services should return startup logs. Before a full GPU run, verify that the
edge-server diagnostics report CUDA rather than CPU.

The notebooks create infrastructure only. Do not start `/app/start.sh`; the
supported experiment controller is the local benchmark runner.

## 6. Validate And Run

Validate local assets and print the resolved configuration matrix:

```bash
.venv/bin/python src/run_benchmark.py --dry-run
```

For an initial end-to-end check, temporarily set:

```env
CONTROLLER_MAX_SAMPLES=4
```

After the small run succeeds, restore:

```env
CONTROLLER_MAX_SAMPLES=all
```

Run:

```bash
.venv/bin/python src/run_benchmark.py
```

The controller runs locally and sends data to the remote ExPECA containers.
Keep the laptop powered, awake, online, and the process running until the
experiment finishes.

Each configuration sends a fresh `/config` request. The services reset their
results, offload buffers, and adaptive-threshold state before processing the
next configuration.

## 7. Inspect The Results

The result directory contains aggregate reports and plot data, thesis-style
figures, and one folder per completed configuration. Each config folder keeps
the raw edge-device result CSV and the derived timing CSV needed for further
analysis.

The generated figures are:

1. Accuracy comparison.
2. Offloading decision distributions.
3. Adaptive-threshold trajectories.
4. Per-sample latency comparison.
5. Six-step latency breakdown.
6. Throughput and processing time.

Figure 5-5 uses the offloaded execution path for offloading-dependent
components. In particular, configs `002`-`004` must not average missing
server-side timings from non-offloaded samples into those components.

Regenerate figures from existing aggregate CSV data without rerunning ExPECA:

```bash
.venv/bin/python src/run_benchmark.py --plot-only
```

If the threshold figure is empty, inspect the per-config raw CSV for `Decision
Threshold` and `Adaptive Threshold After Update`. Images built from older
edge-device code may not record those columns.

## 8. Analyze Actual Dynamic Batches

Configs `005`-`007` use controller batches of 5, 15, and 45. The actual server
batch contains only the samples selected for offloading from that request.

Analyze GPU results:

```bash
.venv/bin/python src/analyze_offload_batches.py
```

Analyze another result directory:

```bash
.venv/bin/python src/analyze_offload_batches.py results/thesis_reproduction
```

The analysis writes request-level measurements, grouped batch-size statistics,
trend calculations, and plots under the selected result directory's
`offload_batch_analysis/` folder.

## 9. Interpret Differences From The Thesis

Exact thesis values are not expected unless the complete hardware, network,
software, and measurement environment is reproduced.

- A regular ExPECA worker runs the edge SML much faster than the thesis
  Raspberry Pi.
- The Raspberry Pi/ARM worker produces edge-processing values closer to the
  thesis hardware.
- Public-IP communication is much faster than the thesis EP5G/radio path, so
  ED-to-ES and ES-to-ED components can be nearly invisible.
- CUDA changes the dominant ES-processing component and allows several images
  to share one model forward pass.
- CPU model timing can vary with CPU allocation, thread behavior, dependency
  versions, container runtime, and load.
- Adaptive runs can follow different trajectories when prediction differences
  change subsequent thresholds and offloading decisions.

The most defensible CPU/GPU server comparison is config `002`, because every
sample is offloaded and threshold behavior cannot change the selected workload.

## 10. Release ExPECA Resources

After collecting results:

1. Destroy the edge-device and edge-server containers created for the run.
2. Release their worker leases.
3. Release the GPU worker immediately when it is no longer needed.
4. Verify in the notebook or ExPECA dashboard that no active resource remains.

Use the GPU notebook's dedicated release section when releasing only the GPU
edge server. Do not destroy containers or reservations owned by other users.
