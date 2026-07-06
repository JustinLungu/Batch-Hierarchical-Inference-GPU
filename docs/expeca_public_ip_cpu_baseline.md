# ExPECA Public-IP CPU Baseline

This guide prepares the first remote ExPECA run using public IPs and CPU-only
containers. The goal is to prove that the remote containers boot, accept
configuration, communicate with each other, and return timing/results data
before moving to GPU.

## What This Runs

```text
laptop/controller
  -> ExPECA edge-device container on public IP
      -> ExPECA edge-server container on public IP
```

Both containers run with `DEVICE=cpu`.

## 1. Create Registry Access

ExPECA workers pull images from a container registry. Docker Hub is the simplest
default path.

1. Create or log in to a Docker Hub account.
2. Log in locally from the project terminal:

   ```bash
   docker login
   ```

3. Open `config/experiment.env` and set the image constants:

   ```env
   EXPECA_IMAGE_NAMESPACE=YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE
   EXPECA_IMAGE_TAG=cpu-amd64-001
   EXPECA_IMAGE_PLATFORM=linux/amd64
   ```

The scripts will create these image names:

```text
YOUR_NAMESPACE/hi-framework-edge-server:YOUR_TAG
YOUR_NAMESPACE/hi-framework-edge-device:YOUR_TAG
```

For Docker Hub, `<namespace>` is normally your Docker Hub username. For another
registry, use the full namespace expected by that registry.

## 2. Prepare Local Assets

Set up the Python environment, download a small dataset, and download all model
checkpoints supported by the current helper scripts:

```bash
scripts/setup_env.sh
source .venv/bin/activate
scripts/download_dataset.sh
scripts/download_models.sh --all
```

The ExPECA Docker build also needs the dataset layout expected by the original
edge-device Dockerfile:

```bash
scripts/prepare_expeca_author_layout.sh
```

That script prepares:

```text
data/datasets/imagenette/val_renamed/
data/datasets/imagenetV2/
```

`imagenetV2` may only contain a placeholder file for the CPU baseline. That is
fine; it exists so Docker can copy the expected directory.

## 3. Prepare the Notebook Environment

The ExPECA notebook uses the `python-chi` bindings and `loguru`. Install those
into the selected project environment:

```bash
scripts/setup_expeca_notebook_env.sh
```

Then restart the notebook kernel in VS Code/Jupyter so it sees the installed
packages.

## 4. Check Build Prerequisites

Run the preflight checker:

```bash
scripts/check_expeca_cpu_prereqs.sh
```

Fix any missing dataset/model paths before building.

## 5. Build and Push CPU Images

Build the CPU images from the repository Dockerfiles:

```bash
scripts/build_expeca_cpu_images.sh
```

Push them to the registry:

```bash
scripts/push_expeca_cpu_images.sh
```

If the edge-device image build fails while copying a dataset directory, rerun:

```bash
scripts/prepare_expeca_author_layout.sh
```

Then rebuild.

## 6. Configure the ExPECA Notebook

Open:

```text
notebooks/ExPECA_HI_setup_Public_IP.ipynb
```

Run the authentication cell first. It expects an OpenRC file downloaded from the
ExPECA dashboard. Keep that file private; files matching `*-openrc.sh` are
ignored by git.

Run the import/setup cell next. The notebook now includes helper functions for:

- listing containers by name;
- destroying stale containers by UUID;
- polling container creation until `Running` or `Error`;
- inspecting logs without relying on ambiguous name lookup.

Before creating containers, run the `Cleanup stale containers` cell. If a
previous run was interrupted, ExPECA may still have `hi-edge-server` or
`hi-edge-device` containers.

Common stale-container cleanup:

```python
destroy_named_containers("hi-edge-server", statuses={"Error", "Creating"})
destroy_named_containers("hi-edge-device", statuses={"Error", "Creating"})
```

Full reset when you intentionally want to recreate everything:

```python
destroy_named_containers("hi-edge-device")
destroy_named_containers("hi-edge-server")
```

Destroy the edge-device first, then the edge-server, so the server IP does not
change underneath a still-running device.

## 7. Set Notebook Image Names

In the notebook import/setup cell, set:

```python
EXPECA_IMAGE_NAMESPACE = "YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE"
EXPECA_IMAGE_TAG = "cpu-amd64-001"
```

Use the same namespace/tag you pushed in step 5. The notebook derives
`EDGE_SERVER_IMAGE` and `EDGE_DEVICE_AMD64_IMAGE` from those values and passes
them into the ExPECA container creation cells.

For the first CPU baseline, use the worker-node/amd64 edge-device option rather
than the Raspberry Pi/arm64 option.

Both container environments should include:

```python
"DEVICE": "cpu"
```

The edge-device environment must also receive the edge-server public IP through
the notebook variable:

```python
"EDGE_SERVER_IP": edge_server_pub_ip
```

Run the edge-server creation cell first. When it prints `status: Running`,
record the server public IP. Then run the worker-node/amd64 edge-device
creation cell and record the device public IP.

Do not rerun a creation cell if the matching container is already `Running`.
If the notebook says a container name already exists, inspect it first. Reuse it
if it is healthy, or destroy it if you want a clean rerun.

## 8. Confirm Containers Are Reachable

From the project terminal:

```bash
curl http://EDGE_SERVER_PUBLIC_IP:8001/logs
curl http://EDGE_DEVICE_PUBLIC_IP:8000/logs
```

The logs should show:

```text
Edge Server: Starting... (Device: cpu, Port: 8001)
Edge Device: Starting... (Device: cpu, Port: 8000)
```

## 9. Configure the CPU Experiment

Open `config/experiment.env` and enter the public IPs printed by the notebook.
The same file also controls whether this is a tiny check or the full selected
dataset:

```env
DEVICE=cpu
EDGE_SERVER_IP=EDGE_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=EDGE_DEVICE_PUBLIC_IP
EDGE_SERVER_PORT=8001
EDGE_DEVICE_PORT=8000
BATCH_SIZE=4
CONTROLLER_BATCH_SIZE=4
CONTROLLER_MAX_SAMPLES=all
FLUSH_FINAL_BATCH=true
```

`BATCH_SIZE` is the edge-server offload batch size. `CONTROLLER_BATCH_SIZE` is
how many images the laptop sends to the edge-device in each request.
`CONTROLLER_MAX_SAMPLES` is the total number of images sent by the controller.
Use a small integer such as `4` for a quick check, or `all` to run the whole
dataset selected by `SAMPLE_PATH`.

Then run:

```bash
.venv/bin/python src/run_expeca_public_ip_test.py
```

The runner:

- checks both remote `/logs` endpoints;
- sends experiment config to `/config` on both containers;
- sends images to the edge-device `/predict` endpoint in repeated controller
  batches until `CONTROLLER_MAX_SAMPLES` is reached;
- lets the edge-device offload to the edge-server;
- downloads `/results` from the edge-device;
- writes local timing analysis.

The config request clears previous container logs/results. That is expected and
helps keep each baseline run clean.

This is the recommended non-interactive replacement for the notebook's final
manual `/app/start.sh` instructions. It still uses the same edge-device and
edge-server runtime APIs, but makes full runs and batch-size sweeps repeatable
from one terminal command.

## 10. Run a Batch-Size Grid

Once the single CPU run works, sweep server-side batch sizes with the same
already-running ExPECA containers.

Configure the grid in `config/experiment.env`:

```env
BATCH_SIZE_GRID=1,2,4,8,16,32,64
CONTROLLER_BATCH_SIZE=64
CONTROLLER_BATCH_SIZE_GRID=
BATCH_GRID_PAIR_MODE=product
CONTROLLER_MAX_SAMPLES=all
```

With `CONTROLLER_BATCH_SIZE_GRID` empty, the grid runner tests each
`BATCH_SIZE_GRID` value while keeping `CONTROLLER_BATCH_SIZE` fixed. This is
usually the clearest first experiment because only the server offload batch
size changes.

For a quick check before launching the full dataset, set:

```env
CONTROLLER_MAX_SAMPLES=64
BATCH_SIZE_GRID=1,2,4
```

Then run:

```bash
.venv/bin/python src/run_expeca_batch_grid.py
```

Each grid item writes its own detailed analysis folder:

```text
results/analysis_expeca_public_ip_cpu_serverbatch1_controllerbatch64/
results/analysis_expeca_public_ip_cpu_serverbatch2_controllerbatch64/
...
```

The grid runner also writes aggregate comparison files:

```text
results/analysis_expeca_public_ip_cpu_grid/summary.csv
results/analysis_expeca_public_ip_cpu_grid/summary.md
results/analysis_expeca_public_ip_cpu_grid/run_metadata.json
```

The aggregate CSV is the file to use for plotting CPU vs GPU later. It includes
batch size, controller batch size, rows, throughput, latency, inference time,
offload roundtrip, and links to each detailed analysis folder.

If you want to vary both server batch size and controller request size, set
`CONTROLLER_BATCH_SIZE_GRID` too:

```env
BATCH_SIZE_GRID=4,8,16
CONTROLLER_BATCH_SIZE_GRID=32,64
BATCH_GRID_PAIR_MODE=product
```

`product` tests all combinations. Use `zip` only when you want positional
pairs, for example `4 with 32`, `8 with 64`, and `16 with 128`.

## 11. Read the Results

After a successful run, inspect:

```text
results/analysis_expeca_public_ip_cpu/summary.md
results/analysis_expeca_public_ip_cpu/timing_results.csv
results/analysis_expeca_public_ip_cpu/run_metadata.json
results/analysis_expeca_public_ip_cpu/raw_edge_device_results.csv
```

The summary should include rows like:

```text
Rows: 4
Offloaded: 4 / 4
Edge-server batches observed: 1
Edge-server batch sizes: [4]
Approx throughput: ~... samples/s
```

## Success Criteria

The CPU baseline is complete when:

- edge-server `/logs` is reachable;
- edge-device `/logs` is reachable;
- logs show both services started with `Device: cpu`;
- both containers accept `/config`;
- edge-device `/predict` returns at least one result;
- edge-device `/results` downloads a CSV;
- local analysis is written under `results/analysis_expeca_public_ip_cpu/`.

After this, the next step is GPU: build a CUDA-capable edge-server image,
deploy it on a GPU-capable ExPECA worker, set `DEVICE=cuda`, verify CUDA inside
the server logs/container, and rerun the same fixed-batch experiments.

## GPU Server Preparation

The public ExPECA inventory page does not currently list GPU/accelerator fields
for the worker nodes, so the GPU worker name must be confirmed separately with
the ExPECA contact/supervisor. The rest of the GPU path can be prepared before
that worker is known.

Build and push the CUDA-enabled edge-server image:

```bash
# In config/experiment.env:
# EXPECA_GPU_EDGE_SERVER_IMAGE_TAG=gpu-amd64-001

scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

In `notebooks/ExPECA_HI_setup_Public_IP.ipynb`, switch only the edge-server:

```python
EDGE_SERVER_IMAGE_TAG = "gpu-amd64-001"
EDGE_SERVER_DEVICE = "cuda"
EDGE_DEVICE_AMD64_IMAGE_TAG = "cpu-amd64-001"
EDGE_DEVICE_DEVICE = "cpu"
```

Then reserve/create the edge-server on the confirmed GPU-capable worker. The
edge-server logs should include CUDA diagnostics like:

```text
Edge Server: Runtime diagnostics: {"cuda_available": true, ...}
```

Do not treat a GPU run as valid unless the logs confirm `cuda_available: true`
and show a CUDA device name.
