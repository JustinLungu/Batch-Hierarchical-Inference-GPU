# ExPECA Public-IP CPU Baseline

This guide prepares the first remote ExPECA run while staying close to the
author's original setup. The goal is not GPU yet. The goal is to prove that the
remote containers boot, accept configuration, communicate, and return results.

## What This Runs

```text
laptop/controller
  -> ExPECA edge-device container on public IP
      -> ExPECA edge-server container on public IP
```

Both containers run with `DEVICE=cpu` for the baseline.

## 1. Prepare Local Assets

Set up Python, download the small dataset, and download model checkpoints:

```bash
scripts/setup_env.sh
scripts/download_dataset.sh
scripts/download_models.sh --all
```

The original edge-device Dockerfile expects an older dataset layout:

```text
data/datasets/imagenette/val_renamed/
data/datasets/imagenetV2/
```

Create that layout from the downloaded Imagenette data:

```bash
scripts/prepare_expeca_author_layout.sh
```

## 2. Configure Image Names

Choose a Docker Hub or container-registry namespace. You can edit
`config/experiment.env` or override it per command:

```bash
export EXPECA_IMAGE_NAMESPACE=justin157
export EXPECA_IMAGE_TAG=cpu-amd64-001
```

The resulting image names are:

```text
justin157/hi-framework-edge-server:cpu-amd64-001
justin157/hi-framework-edge-device:cpu-amd64-001
```

The author's old Docker Hub tags were checked and were not publicly pullable:

```bash
docker pull h3nkk44/hi-framework-edge-server:latest_amd64
docker pull h3nkk44/hi-framework-edge-device:latest_amd64
```

Both returned `not found`, so this workflow builds and pushes our own images
instead of relying on the previous private/unavailable images.

## 3. Check Prerequisites

```bash
scripts/check_expeca_cpu_prereqs.sh
```

Fix any missing paths before building.

## 4. Build and Push CPU Images

```bash
EXPECA_IMAGE_NAMESPACE=justin157 scripts/build_expeca_cpu_images.sh
EXPECA_IMAGE_NAMESPACE=justin157 scripts/push_expeca_cpu_images.sh
```

These scripts build from the author's original Dockerfiles:

```text
app/edge_server/Dockerfile.edge_server
app/edge_device/Dockerfile.edge_device
```

If the edge-device build fails at:

```text
COPY data/datasets/imagenetV2/ ./data/datasets/imagenetV2/
```

rerun:

```bash
scripts/prepare_expeca_author_layout.sh
```

That script creates the expected dataset layout and a placeholder file so the
empty `imagenetV2` directory is included in Docker's build context.

## 5. Start Containers on ExPECA

Open:

```text
notebooks/ExPECA_HI_setup_Public_IP.ipynb
```

Run the authentication/setup cells, then use the public-IP container sections.

Replace the author's image names:

```python
image = "h3nkk44/hi-framework-edge-server:latest_amd64"
image = "h3nkk44/hi-framework-edge-device:latest_amd64"
```

with your pushed images:

```python
image = "justin157/hi-framework-edge-server:cpu-amd64-001"
image = "justin157/hi-framework-edge-device:cpu-amd64-001"
```

For the first baseline, use the worker-node/amd64 edge-device option rather
than the Raspberry Pi/arm64 option.

Add `DEVICE=cpu` to both container environments:

```python
environment = {
    "DNS_IP": "8.8.8.8",
    "GATEWAY_IP": "130.237.11.97",
    "PASS": "expeca",
    "DEVICE": "cpu",
}
```

For the edge-device container, keep the author's server IP handoff:

```python
"EDGE_SERVER_IP": edge_server_pub_ip
```

Record both public IPs printed by the notebook.

## 6. Confirm Containers Are Reachable

From your laptop:

```bash
curl http://EDGE_SERVER_PUBLIC_IP:8001/logs
curl http://EDGE_DEVICE_PUBLIC_IP:8000/logs
```

Successful responses mean the containers are running and reachable.

## 7. Run a Tiny Baseline Experiment

Use a very small test first:

```text
DEVICE=cpu
BATCH_SIZE=4
CONTROLLER_BATCH_SIZE=4
FLUSH_FINAL_BATCH=true
```

The author-provided external controller for testbed runs is:

```text
app/controller_external/Controller_testbed_containers.py
```

It has old hard-coded IPs. Before using it, replace those IPs with the public
IPs printed by the notebook.

If using the internal controller from the edge-device console, run:

```bash
/app/start.sh
```

When prompted for the dataset path, use:

```text
imagenette/val_renamed/
```

## Success Criteria

The CPU baseline is ready when:

- edge-device `/logs` is reachable;
- edge-server `/logs` is reachable;
- both containers accept `/config`;
- edge-device `/predict` returns at least one result;
- `/results` downloads a CSV;
- logs show the services started with `Device: cpu`.

After this, the next step is GPU: a CUDA-capable server image, a GPU-capable
ExPECA worker, `DEVICE=cuda`, and explicit verification that CUDA is available
inside the edge-server container.
