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

## 9. Configure the Tiny CPU Baseline

Open `config/expeca_public_ip.env` and enter the public IPs printed by the
notebook:

```env
DEVICE=cpu
EDGE_SERVER_IP=EDGE_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=EDGE_DEVICE_PUBLIC_IP
EDGE_SERVER_PORT=8001
EDGE_DEVICE_PORT=8000
BATCH_SIZE=4
CONTROLLER_BATCH_SIZE=4
FLUSH_FINAL_BATCH=true
```

Then run:

```bash
.venv/bin/python src/run_expeca_public_ip_test.py
```

The runner:

- checks both remote `/logs` endpoints;
- sends experiment config to `/config` on both containers;
- sends one controller batch of images to the edge-device `/predict` endpoint;
- lets the edge-device offload to the edge-server;
- downloads `/results` from the edge-device;
- writes local timing analysis.

The config request clears previous container logs/results. That is expected and
helps keep each baseline run clean.

## 10. Read the Results

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
