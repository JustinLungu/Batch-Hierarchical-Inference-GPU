# ExPECA Deployment Notebooks

This directory contains the infrastructure notebooks used to reserve ExPECA
resources and start the remote edge-device and edge-server containers. The
notebooks prepare the deployment; the benchmark itself is started separately
from the repository root with `src/run_benchmark.py`.

## Choose A Notebook

| File | Use it for |
|---|---|
| `ExPECA_HI_setup_Public_IP.ipynb` | CPU experiments over public IPs. It can deploy the edge device to an ARM64 Raspberry Pi worker or a regular amd64 worker. |
| `ExPECA_HI_setup_GPU_RaspberryPi_Public_IP.ipynb` | The main GPU benchmark: GPU edge server on `worker-05` and ARM64 Raspberry Pi edge device on `worker-21`, both reached through public IPs. |
| `ExPECA_HI_setup_EP5G.ipynb` | The original private-5G/EP5G deployment path. Keep this notebook for experiments that need the radio network rather than public-IP networking. |
| `ExPECA_raspberry_setup.md` | Additional Raspberry Pi preparation and troubleshooting notes. |

Worker assignments can change with ExPECA availability. Treat the worker names
in a notebook as experiment configuration, not permanent infrastructure.

## Before Opening A Notebook

From the repository root:

```bash
scripts/setup_env.sh
source .venv/bin/activate
```

Restart the VS Code/Jupyter kernel after updating the environment.

You also need:

1. An ExPECA account and project access.
2. An OpenRC file downloaded from the ExPECA/Chameleon API Access page.
3. The required CPU, GPU, or ARM64 images built and pushed to your container
   registry.
4. The image namespace and tags configured in `config/experiment.env` and
   `config/defaults.env`.

OpenRC files contain credentials. Files matching `notebooks/*-openrc.sh` are
ignored by Git and must remain local.

## Normal Workflow

Run notebook cells deliberately and in order:

1. Load the OpenRC credentials and authenticate.
2. import the ExPECA client and notebook helpers.
3. Inspect existing containers and remove only stale resources you own.
4. Reserve the required edge-server worker.
5. Create the edge-server container and wait for `Running`.
6. Reserve the edge-device worker.
7. Create the edge-device container and wait for `Running`.
8. Check both `/logs` endpoints.
9. Copy the printed public IPs to `config/experiment.env`.
10. Run a dry-run and then the benchmark from a terminal.

```env
EDGE_SERVER_IP=<edge-server-public-ip>
EDGE_DEVICE_IP=<edge-device-public-ip>
```

```bash
.venv/bin/python src/run_benchmark.py --dry-run
.venv/bin/python src/run_benchmark.py
```

The controller runs on the laptop. Keep the laptop awake and the process
running until the benchmark finishes.

## Verifying The Deployment

For a public-IP deployment:

```bash
curl http://$EDGE_SERVER_IP:8001/logs
curl http://$EDGE_DEVICE_IP:8000/logs
```

A `200` response and service startup logs mean the containers are ready. A
container remaining in `Creating` for several minutes, or changing to `Error`
or `Deleted`, is a deployment issue rather than a benchmark result.

The GPU notebook also prints PyTorch/CUDA diagnostics from the edge-server
logs. Confirm that the selected device is `cuda` before starting a full GPU
run.

## Cleanup

The notebooks reserve shared infrastructure. When finished:

1. Stop or destroy the containers created by the notebook.
2. Release the corresponding worker leases.
3. Release the GPU worker promptly so other users can reserve it.

Use the GPU notebook's dedicated release section when only the GPU edge server
should be released. Do not delete unrelated containers or leases.

## Operational Notes

- Avoid `Run All` when a notebook already has active reservations or
  containers. Re-running creation cells can create duplicate or stale records.
- Inspect resource status before cleanup. Old `Deleted` records may remain
  visible even though they are no longer running.
- Public-IP and EP5G runs use different network paths, so their communication
  latency results are not directly interchangeable.
- Notebook outputs can contain project names, public IPs, resource IDs, and
  logs. Clear outputs before committing notebook changes.

