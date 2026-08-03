# ExPECA GPU YOLO Training Handover

## Purpose

This document is a self-contained implementation guide for adapting a project
to train YOLO models on an ExPECA GPU worker. It describes the required files,
their responsibilities, the ExPECA resource lifecycle, and the commands needed
to build, deploy, train, recover, export results, and release the worker.

The intended flow is:

```text
Developer laptop
  -> build and push a GPU training image
  -> reserve an ExPECA GPU worker
  -> create an NVIDIA-enabled container
  -> start a persistent remote training session

ExPECA GPU container
  -> verify CUDA
  -> access or download the dataset
  -> train YOLO
  -> write logs and checkpoints
  -> export artifacts before cleanup
```

The notebook provisions and inspects infrastructure. Training itself runs
inside the ExPECA container, not on the laptop.

## ExPECA Mental Model

ExPECA exposes physical testbed workers through a Chameleon/OpenStack control
plane. A training deployment has several separate resources:

- **Project credentials:** the OpenRC values and password authenticate API
  calls for one ExPECA project.
- **Worker lease:** reserves a physical worker for a fixed duration. The lease
  must be active before a container can use that worker.
- **Registry image:** ExPECA pulls the Docker image by name and tag. The image
  must be accessible to the platform and built for the worker architecture.
- **Container:** runs the training image on the reserved worker. Its lifecycle
  is separate from the worker lease.
- **NVIDIA allocation:** the container must use the NVIDIA runtime and request
  a GPU through its resource label.
- **Network attachment:** a public IP, free worker interface, network, CIDR,
  and gateway make SSH access possible.
- **Persistent artifacts:** the container filesystem is temporary. Important
  results must be stored on persistent storage or copied out before cleanup.

Creating, destroying, and reserving resources are asynchronous operations.
The API call returning or timing out does not necessarily mean the remote
operation has finished. Always inspect the actual resource status before
retrying.

Destroying a container does not automatically release its worker lease.
Releasing a lease does not safely export container data. Treat those as
separate explicit steps.

## Prerequisites

The operator needs:

1. An ExPECA account and project with permission to reserve the GPU worker.
2. An OpenRC credential file for that project.
3. Confirmation of the currently available GPU worker and lease window.
4. Docker with the `buildx` component.
5. A container registry account whose image ExPECA can pull.
6. `uv`, Jupyter, SSH, and `rsync` on the laptop.
7. An SSH key pair, preferably Ed25519.
8. A YOLO-compatible dataset and dataset YAML.
9. Enough persistent storage for the dataset, logs, and checkpoints.

The laptop does not need an NVIDIA GPU. It builds and pushes the image,
controls ExPECA, transfers data, and connects over SSH. CUDA execution happens
on the remote GPU worker.

## End-to-End Workflow

Follow this order:

1. Add and lock the training dependencies.
2. Create the Dockerfile, entrypoint, runtime verification, and training code.
3. Build the image for `linux/amd64`.
4. Push an immutable image tag and record its digest.
5. Load OpenRC credentials in the notebook.
6. Inspect existing leases and containers.
7. Reserve the GPU worker.
8. Allocate a public IP and free worker interface.
9. Create the container with NVIDIA runtime and GPU resource label.
10. Verify `nvidia-smi` and `torch.cuda.is_available()`.
11. Transfer or mount the dataset and training configuration.
12. Run a one-batch or one-epoch smoke test.
13. Verify checkpoint export and resume.
14. Start the full run inside `tmux`.
15. Monitor the remote process, storage, and reservation expiry.
16. Export and verify all artifacts.
17. Destroy the container.
18. Release the worker lease.
19. Confirm that no active resource remains.

## Recommended Project Structure

Add only the files needed by the target project:

```text
config/
  expeca.env.example
  training.env.example
docker/
  Dockerfile.expeca-gpu
  entrypoint.sh
docs/
  expeca_gpu_training.md
notebooks/
  ExPECA_GPU_training.ipynb
scripts/
  build_expeca_gpu_image.sh
  push_expeca_gpu_image.sh
  upload_training_inputs.sh
  train_yolo.sh
  export_training_artifacts.sh
src/
  train.py
  verify_runtime.py
tests/
  test_training_config.py
.dockerignore
.gitignore
pyproject.toml
```

Do not add separate wrappers that only call one other function. Keep the
command entry points short and put reusable logic in normal Python modules.

## Configuration

Separate infrastructure settings from experiment settings.

### `config/expeca.env.example`

```bash
# Docker registry. The real namespace belongs in an ignored local env file.
EXPECA_IMAGE_NAMESPACE=your-dockerhub-account
EXPECA_GPU_IMAGE_TAG=yolo-gpu-amd64-001
EXPECA_IMAGE_PLATFORM=linux/amd64

# Confirm current availability with the ExPECA administrators or inventory.
EXPECA_GPU_WORKER=worker-05
EXPECA_GPU_COUNT=1

EXPECA_CONTAINER_NAME=yolo-training-gpu
EXPECA_PUBLIC_NETWORK=serverpublic
EXPECA_PUBLIC_GATEWAY=130.237.11.97
EXPECA_DNS_SERVER=8.8.8.8
```

Worker names, available network interfaces, gateways, and GPU availability
are deployment facts, not permanent constants. Confirm them before each
reservation.

### `config/training.env.example`

```bash
YOLO_MODEL=yolo11n.pt
YOLO_DATA=/workspace/data/dataset.yaml
YOLO_EPOCHS=100
YOLO_IMAGE_SIZE=640
YOLO_BATCH_SIZE=16
YOLO_DEVICE=0
YOLO_WORKERS=8
YOLO_SEED=42
YOLO_RUN_NAME=baseline
YOLO_OUTPUT_DIR=/workspace/outputs
YOLO_RESUME=false
YOLO_RESUME_CHECKPOINT=/workspace/outputs/baseline/weights/last.pt
YOLO_SAVE_PERIOD=10
```

The actual configuration should be copied to ignored local files. Never
commit OpenRC credentials, registry tokens, passwords, private keys, dataset
secrets, or signed download URLs.

## GPU Training Image

Use a pinned GPU-capable base image compatible with the NVIDIA driver exposed
by ExPECA. Record the exact image digest, Python version, CUDA version,
PyTorch version, and YOLO package version.

A reasonable Dockerfile shape is:

```dockerfile
FROM <pinned-pytorch-cuda-runtime-image>

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-server tmux git curl ca-certificates iproute2 && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd /workspace/outputs /workspace/data /workspace/config

COPY pyproject.toml uv.lock ./
RUN pip install uv && \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh /entrypoint.sh
RUN uv sync --frozen --no-dev && \
    chmod +x /entrypoint.sh scripts/*.sh

ENV PATH="/workspace/.venv/bin:${PATH}"

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
```

Adapt dependency installation to the target repository. If `uv` cannot use
the CUDA PyTorch index correctly, install the pinned CUDA PyTorch wheels in a
dedicated Docker step and install the remaining locked dependencies
afterward.

Do not bake a large dataset or produced checkpoints into the image. That
creates very large registry layers, slow pulls, and expensive rebuilds. Put
only code and dependencies in the image.

Keep `.dockerignore` strict:

```text
.git
.venv
__pycache__
.pytest_cache
data
outputs
runs
*.pt
*.onnx
```

Model weights should only be excluded if the container downloads them or they
are supplied through persistent storage.

## Container Entrypoint

The entrypoint should make the container available for an interactive or
detached training session. It should not automatically begin an expensive
training run.

Example responsibilities:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /var/run/sshd /workspace/data /workspace/outputs /workspace/config

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
    install -d -m 700 /root/.ssh
    printf '%s\n' "${SSH_PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
elif [[ -n "${ROOT_PASSWORD:-}" ]]; then
    echo "root:${ROOT_PASSWORD}" | chpasswd
    sed -i 's/^#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
fi

nvidia-smi || true
python -m src.verify_runtime

/usr/sbin/sshd
exec sleep infinity
```

Prefer an injected SSH public key or another ExPECA-supported access method.
Do not commit a static root password. If temporary password access is the only
available mechanism, inject the secret at deployment time and rotate it.

## CUDA Verification

Create `src/verify_runtime.py` so deployment fails visibly when the container
does not have usable CUDA:

```python
import json
import sys

import torch


def main() -> int:
    available = torch.cuda.is_available()
    report = {
        "torch_version": torch.__version__,
        "cuda_available": available,
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if available else 0,
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if available
        else [],
    }
    print(json.dumps(report, indent=2))
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Also inspect `nvidia-smi` inside the deployed container. A successful image
build does not prove that ExPECA exposed a GPU to the container.

## Build and Push Scripts

The image must match the GPU worker architecture, normally `linux/amd64`.
The build script should be complete and load its configuration directly:

```bash
#!/usr/bin/env bash
set -euo pipefail

set -a
source config/expeca.env
set +a

image="${EXPECA_IMAGE_NAMESPACE}/yolo-training:${EXPECA_GPU_IMAGE_TAG}"

docker buildx build \
  --platform "${EXPECA_IMAGE_PLATFORM:-linux/amd64}" \
  --load \
  -f docker/Dockerfile.expeca-gpu \
  -t "${image}" \
  .

printf 'Built %s\n' "${image}"
```

Create a separate push script:

```bash
#!/usr/bin/env bash
set -euo pipefail

set -a
source config/expeca.env
set +a

image="${EXPECA_IMAGE_NAMESPACE}/yolo-training:${EXPECA_GPU_IMAGE_TAG}"
docker image inspect "${image}" >/dev/null
docker push "${image}"
```

Use immutable versioned tags for real experiments. Reusing one tag can cause
confusion about which image ExPECA pulled.

Operational details:

- Cross-platform builds require `docker buildx` and host emulation where
  applicable.
- A transient Docker Hub `502 Bad Gateway` can usually be handled by retrying
  the push; already-uploaded layers are reused.
- `Layer already exists` is normal and does not indicate a corrupt image.
- Large model and dataset layers make every build and push much slower.
- Confirm the final pushed digest and record it with the training run.

## ExPECA Deployment Notebook

Create `notebooks/ExPECA_GPU_training.ipynb`. Install its local dependencies
before opening it:

```bash
uv add --dev ipykernel jupyter loguru requests
uv pip install "git+https://github.com/KTH-EXPECA/python-chi"
uv run python -m ipykernel install --user \
  --name expeca-yolo \
  --display-name "ExPECA YOLO"
```

Select the `ExPECA YOLO` kernel in the notebook.

### Cell 1: load OpenRC credentials

Download the project OpenRC shell file from the ExPECA API Access page, keep
it outside version control, and set `OPENRC_PATH`:

```python
import os
import re
from getpass import getpass
from pathlib import Path

OPENRC_PATH = Path("../your-project-openrc.sh")

if not OPENRC_PATH.exists():
    raise FileNotFoundError(OPENRC_PATH)

pattern = r'export\s+(\w+)\s*=\s*("[^"]+"|[^"\n]+)'
for name, value in re.findall(pattern, OPENRC_PATH.read_text()):
    os.environ[name] = value.strip().strip('"')

os.environ["OS_PASSWORD"] = getpass("ExPECA password: ")
print("OpenRC environment loaded.")
```

Add `*-openrc.sh` to `.gitignore`.

### Cell 2: deployment configuration

Keep all values that a user may change in this one cell:

```python
from pathlib import Path

DOCKER_NAMESPACE = "your-dockerhub-account"
GPU_IMAGE_TAG = "yolo-gpu-amd64-001"
GPU_IMAGE = f"{DOCKER_NAMESPACE}/yolo-training:{GPU_IMAGE_TAG}"

GPU_WORKER_NAME = "worker-05"  # Confirm current GPU worker availability.
CONTAINER_NAME = "yolo-training-gpu"
GPU_COUNT = 1
LEASE_DAYS = 1
LEASE_HOURS = 0

PUBLIC_NETWORK = "serverpublic"
PUBLIC_GATEWAY = "130.237.11.97"
DNS_SERVER = "8.8.8.8"
PUBLIC_IP_INDEX = 0

SSH_PUBLIC_KEY_PATH = Path.home() / ".ssh" / "id_ed25519.pub"
if not SSH_PUBLIC_KEY_PATH.exists():
    raise FileNotFoundError(
        f"Create an SSH key first: ssh-keygen -t ed25519; missing {SSH_PUBLIC_KEY_PATH}"
    )
SSH_PUBLIC_KEY = SSH_PUBLIC_KEY_PATH.read_text().strip()

print("Image:", GPU_IMAGE)
print("Worker:", GPU_WORKER_NAME)
print("Container:", CONTAINER_NAME)
```

The worker name and network values are deployment-specific. Confirm the
available GPU worker, public network, gateway, and interface allocation before
creating the container.

### Cell 3: imports and lifecycle helpers

```python
import json
import time

from loguru import logger
import chi.container
import chi.network
from chi.expeca import (
    get_available_publicips,
    get_worker_interfaces,
    list_reservations,
    reserve,
    unreserve_byid,
)

INACTIVE_STATUSES = {"Deleted", "Error"}


def containers_named(name, include_inactive=True):
    matches = [
        container
        for container in chi.container.list_containers()
        if container.name == name
    ]
    if include_inactive:
        return matches
    return [
        container
        for container in matches
        if container.status not in INACTIVE_STATUSES
    ]


def print_container(container):
    print(
        "name=", container.name,
        "uuid=", container.uuid,
        "status=", container.status,
        "reason=", getattr(container, "status_reason", None),
        "image=", getattr(container, "image", None),
        "addresses=", getattr(container, "addresses", None),
    )


def inspect_named_containers(name, include_logs=False):
    matches = containers_named(name)
    if not matches:
        print(f"No containers named {name}.")
        return []
    for container in matches:
        print_container(container)
        if include_logs:
            try:
                print(chi.container.get_logs(container.uuid))
            except Exception as exc:
                print("Logs unavailable:", repr(exc))
    return matches


def reserve_worker(worker_name, days=1, hours=0):
    lease_name = worker_name + "-lease"
    for lease in list_reservations(brief=True):
        if lease["name"] == lease_name and lease.get("status") == "ACTIVE":
            print("Reusing active lease:", lease["reservation_id"])
            return lease["reservation_id"]

    created = reserve(
        {
            "type": "device",
            "name": worker_name,
            "duration": {"days": days, "hours": hours},
        }
    )
    reservation_id = created["reservations"][0]["id"]
    print("Created reservation:", reservation_id)
    return reservation_id


def choose_public_ip(index=0):
    available = get_available_publicips()
    if index >= len(available):
        raise IndexError(f"IP index {index} unavailable; candidates: {available}")
    print("Available public IPs:", available)
    return available[index]


def choose_worker_interface(worker_name):
    interface_map = list(get_worker_interfaces(worker_name).values())[0]
    available = [
        name
        for name, data in interface_map.items()
        if not data.get("connections", [])
    ]
    if not available:
        print(json.dumps(interface_map, indent=2))
        raise RuntimeError(f"No free interface on {worker_name}")
    print("Available worker interfaces:", available)
    return available[0]


def create_with_progress(name, kwargs, poll_seconds=15, timeout_seconds=1200):
    active = containers_named(name, include_inactive=False)
    if active:
        for container in active:
            print_container(container)
        raise RuntimeError(f"An active container named {name} already exists.")

    try:
        created = chi.container.create_container(
            name=name,
            start_timeout=60,
            **kwargs,
        )
        container_uuid = created.uuid
    except Exception:
        active = containers_named(name, include_inactive=False)
        if len(active) != 1:
            raise
        container_uuid = active[0].uuid
        logger.warning("Creation call did not finish, but one active container exists.")

    deadline = time.time() + timeout_seconds
    while True:
        container = chi.container.get_container(container_uuid)
        print(
            time.strftime("%H:%M:%S"),
            "status=", container.status,
            "reason=", getattr(container, "status_reason", None),
            "addresses=", getattr(container, "addresses", None),
        )
        if container.status == "Running":
            return container
        if container.status in INACTIVE_STATUSES:
            raise RuntimeError(
                f"Container failed: {container.status}; "
                f"{getattr(container, 'status_reason', None)}"
            )
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {name}; last status={container.status}"
            )
        time.sleep(poll_seconds)


def wait_until_inactive(uuids, poll_seconds=5, timeout_seconds=180):
    remaining = set(uuids)
    deadline = time.time() + timeout_seconds
    while remaining:
        active_uuids = {
            container.uuid
            for container in chi.container.list_containers()
            if container.status not in INACTIVE_STATUSES
        }
        remaining &= active_uuids
        if not remaining:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Removal timed out: {sorted(remaining)}")
        print("Waiting for removal:", sorted(remaining))
        time.sleep(poll_seconds)


def destroy_named_containers(name):
    matches = containers_named(name)
    if not matches:
        print(f"No containers named {name}.")
        return
    uuids = []
    for container in matches:
        if container.status in INACTIVE_STATUSES:
            continue
        print("Destroying:", container.uuid, container.status)
        chi.container.destroy_container(container.uuid)
        uuids.append(container.uuid)
    wait_until_inactive(uuids)
```

### Cell 4: inspect before creating

```python
inspect_named_containers(CONTAINER_NAME, include_logs=True)
print(json.dumps(list_reservations(brief=True), indent=2))
```

Do not continue if another active container has the same name unless it is the
container intentionally being reused.

### Cell 5: reserve the GPU worker

```python
gpu_reservation_id = reserve_worker(
    GPU_WORKER_NAME,
    days=LEASE_DAYS,
    hours=LEASE_HOURS,
)
```

### Cell 6: create the NVIDIA-enabled container

```python
public_ip = choose_public_ip(PUBLIC_IP_INDEX)
worker_interface = choose_worker_interface(GPU_WORKER_NAME)
public_network = chi.network.get_network(PUBLIC_NETWORK)

gpu_container = create_with_progress(
    CONTAINER_NAME,
    {
        "image": GPU_IMAGE,
        "reservation_id": gpu_reservation_id,
        "runtime": "nvidia",
        "environment": {
            "DNS_IP": DNS_SERVER,
            "GATEWAY_IP": PUBLIC_GATEWAY,
            "SSH_PUBLIC_KEY": SSH_PUBLIC_KEY,
        },
        "mounts": [],
        "nets": [{"network": public_network["id"]}],
        "labels": {
            "networks.1.interface": worker_interface,
            "networks.1.ip": public_ip + "/27",
            "networks.1.gateway": PUBLIC_GATEWAY,
            "capabilities.privileged": "true",
            "resources.limits.nvidia_com_gpu": str(GPU_COUNT),
        },
    },
)

print("GPU container public IP:", public_ip)
print("SSH command:", f"ssh root@{public_ip}")
```

The essential GPU settings are `runtime="nvidia"` and the
`resources.limits.nvidia_com_gpu` label. Without both, the container may start
without access to CUDA.

### Cell 7: inspect startup

```python
inspect_named_containers(CONTAINER_NAME, include_logs=True)
```

Then verify from the laptop:

```bash
ssh root@GPU_CONTAINER_PUBLIC_IP \
  'nvidia-smi && python -m src.verify_runtime'
```

Do not start training unless both diagnostics succeed.

### Cell 8: cleanup and release

Run this only after exporting every required checkpoint and log:

```python
destroy_named_containers(CONTAINER_NAME)

released = False
for lease in list_reservations(brief=True):
    if (
        lease["name"] == GPU_WORKER_NAME + "-lease"
        and lease.get("status") == "ACTIVE"
    ):
        lease_id = lease.get("id")
        reservation_id = lease.get("reservation_id")
        try:
            unreserve_byid(lease_id)
        except Exception:
            unreserve_byid(reservation_id)
        released = True

if not released:
    print("No active GPU lease found.")

inspect_named_containers(CONTAINER_NAME)
print(json.dumps(list_reservations(brief=True), indent=2))
```

### Asynchronous operation rules

Container operations are asynchronous:

- A creation cell can time out while ExPECA continues processing the request.
- Interrupting a notebook does not necessarily cancel the remote operation.
- Never immediately create a duplicate container after a timeout.
- First list containers by name and inspect `Creating`, `Running`, `Error`,
  `Deleted`, and the failure reason.
- A container shown as `Deleted` may remain visible as a historical record.
- If a resource is stuck in `Creating` or cannot be removed, the issue may be
  on the ExPECA control plane rather than in the Docker image.

Never create a duplicate immediately after a timeout. First list containers by
name and inspect `Creating`, `Running`, `Error`, `Deleted`, addresses, and the
failure reason. Use unique names for unrelated experiments.

## Data and Artifact Persistence

An ExPECA container is temporary. Assume its local filesystem disappears when
the container is destroyed.

Before running a full training job, choose and test one persistence method:

- an ExPECA volume mounted into the container;
- ExPECA object storage;
- a remote dataset store accessed by the container; or
- explicit `rsync`/`scp` transfer to and from the laptop.

For a first implementation, `rsync` is the simplest complete path. Add the
container IP to the ignored `config/expeca.env` after deployment:

```bash
EXPECA_GPU_IP=130.237.11.xxx
```

Create `scripts/upload_training_inputs.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

set -a
source config/expeca.env
set +a

: "${EXPECA_GPU_IP:?Set EXPECA_GPU_IP in config/expeca.env}"

ssh "root@${EXPECA_GPU_IP}" \
  'mkdir -p /workspace/data /workspace/config /workspace/outputs'

rsync -avP --delete data/ "root@${EXPECA_GPU_IP}:/workspace/data/"
rsync -avP config/training.env \
  "root@${EXPECA_GPU_IP}:/workspace/config/training.env"
```

For very large datasets, replace repeated laptop uploads with an ExPECA volume
or remote dataset store. Verify the dataset YAML after transfer because its
train/validation paths must be valid inside the container.

Ultralytics normally writes this output layout:

```text
outputs/<run-name>/
  weights/
    best.pt
    last.pt
  args.yaml
  results.csv
  results.png
  confusion_matrix.png
  environment.json
outputs/training.log
```

At minimum, `environment.json` should contain:

- Git commit;
- Docker image name and digest;
- ExPECA worker and container name;
- GPU model;
- Python, CUDA, PyTorch, and YOLO versions;
- dataset identity/version;
- random seed;
- effective training arguments;
- start/end timestamps.

Create `scripts/export_training_artifacts.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

set -a
source config/expeca.env
source config/training.env
set +a

: "${EXPECA_GPU_IP:?Set EXPECA_GPU_IP in config/expeca.env}"
: "${YOLO_RUN_NAME:?Set YOLO_RUN_NAME in config/training.env}"

destination="artifacts/${YOLO_RUN_NAME}"
mkdir -p "${destination}"

rsync -avP \
  "root@${EXPECA_GPU_IP}:/workspace/outputs/${YOLO_RUN_NAME}/" \
  "${destination}/"

rsync -avP \
  "root@${EXPECA_GPU_IP}:/workspace/outputs/training.log" \
  "${destination}/training.log"

test -s "${destination}/weights/best.pt"
test -s "${destination}/weights/last.pt"
printf 'Artifacts exported to %s\n' "${destination}"
```

Add `artifacts/`, `outputs/`, and the real training environment file to
`.gitignore`. Test artifact export and resume before committing to a long run.

## Training Entry Point

`scripts/train_yolo.sh` should be the one operator-facing command inside the
container:

```bash
#!/usr/bin/env bash
set -euo pipefail

training_config="${TRAINING_CONFIG:-/workspace/config/training.env}"
test -f "${training_config}"

set -a
source "${training_config}"
set +a

python -m src.verify_runtime
python -m src.train
```

Create `src/train.py`:

```python
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def as_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return normalized == "true"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    data = Path(required("YOLO_DATA"))
    if not data.exists():
        raise FileNotFoundError(data)

    output_dir = Path(os.getenv("YOLO_OUTPUT_DIR", "/workspace/outputs"))
    run_name = os.getenv("YOLO_RUN_NAME", "baseline")
    run_dir = output_dir / run_name
    resume = as_bool("YOLO_RESUME")
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"{run_dir} already contains results; choose a new run name or resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    model_source = os.getenv("YOLO_MODEL", "yolo11n.pt")
    if resume:
        model_source = required("YOLO_RESUME_CHECKPOINT")
        if not Path(model_source).exists():
            raise FileNotFoundError(model_source)

    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "model_source": model_source,
        "data": str(data),
        "run_name": run_name,
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("YOLO_")
        },
    }
    (run_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    model = YOLO(model_source)
    model.train(
        data=str(data),
        epochs=int(os.getenv("YOLO_EPOCHS", "100")),
        imgsz=int(os.getenv("YOLO_IMAGE_SIZE", "640")),
        batch=int(os.getenv("YOLO_BATCH_SIZE", "16")),
        device=os.getenv("YOLO_DEVICE", "0"),
        workers=int(os.getenv("YOLO_WORKERS", "8")),
        seed=int(os.getenv("YOLO_SEED", "42")),
        project=str(output_dir),
        name=run_name,
        exist_ok=True,
        resume=resume,
        save=True,
        save_period=int(os.getenv("YOLO_SAVE_PERIOD", "10")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Adapt model-specific arguments if the project uses a different YOLO API, but
preserve validation, metadata, deterministic settings, checkpointing, and
resume behavior.

## Running Without Keeping the Laptop Open

Starting training directly from a laptop terminal ties the client connection
to the process. Instead, SSH into the ExPECA container and use `tmux`:

```bash
ssh root@GPU_CONTAINER_PUBLIC_IP
tmux new -s yolo-training
cd /workspace
./scripts/train_yolo.sh 2>&1 | tee outputs/training.log
```

Detach with `Ctrl-b`, then `d`. Reconnect later with:

```bash
ssh root@GPU_CONTAINER_PUBLIC_IP
tmux attach -t yolo-training
```

Once the process is running inside `tmux`, the laptop may disconnect or close.
Training continues while all of the following remain true:

- the ExPECA container is running;
- the worker reservation has not expired;
- the training process has not failed;
- storage has not filled;
- the platform has not reclaimed or restarted the resource.

For stronger unattended operation, use a supervised process or job mechanism
supported by the platform.

## Monitoring

Useful commands inside the remote container:

```bash
nvidia-smi
watch -n 2 nvidia-smi
ps aux
tail -f /workspace/outputs/training.log
df -h
du -sh /workspace/outputs
```

Monitor GPU memory, utilization, temperature, storage, process health, epoch
progress, validation metrics, and checkpoint creation.

## Cleanup Procedure

Never release the worker before preserving the outputs.

1. Confirm training finished or intentionally stop it.
2. Export checkpoints, logs, metrics, and metadata.
3. Verify the exported files can be opened and that checkpoint sizes are
   plausible.
4. Record the final image digest and run configuration.
5. Destroy the exact named container.
6. Poll until it is no longer active.
7. Release the exact worker lease.
8. List reservations and containers once more to confirm cleanup.

Container cleanup and lease cleanup are separate operations.

## Common Failure Modes

### Container remains in `Creating`

Inspect existing containers and wait. Do not submit duplicate creation
requests. If the status remains unchanged for a long time, capture the UUID,
worker, timestamps, status, and reason for the ExPECA administrators.

### Container becomes `Error` or `Deleted`

Inspect the reason and logs. Distinguish image pull failures, platform
scheduling failures, invalid network labels, and entrypoint failures.

### `torch.cuda.is_available()` is false

Check all of:

- the container used `runtime="nvidia"`;
- the GPU resource label was present;
- the worker actually has an available GPU;
- `nvidia-smi` works in the container;
- CUDA PyTorch was installed rather than a CPU-only wheel;
- the image CUDA runtime is compatible with the host driver.

### Registry pull fails

Confirm the repository is public or ExPECA has credentials, the tag exists,
the architecture is correct, and the pushed digest completed successfully.

### Training stops when the laptop disconnects

The process was still attached to the SSH session. Run it under `tmux` or a
supervisor.

### Checkpoints disappear

They were stored only in the temporary container. Use persistent storage or
export them before cleanup.

### CUDA out of memory

Reduce batch size, image size, or model size. Consider gradient accumulation
if the training framework supports it. Record any adjustment because it
changes the experiment configuration.

## Minimum Validation Sequence

Before the full run:

1. Unit-test configuration parsing and invalid values.
2. Test missing dataset and output-path errors.
3. Build the image.
4. Run it on any local NVIDIA host if available.
5. Deploy to ExPECA and verify CUDA.
6. Run one training batch.
7. Run a one-epoch smoke test on a small dataset subset.
8. Verify logs, metrics, and all checkpoint files.
9. Export the smoke-test artifacts.
10. Resume from `last.pt` and confirm training continues.
11. Test container and lease cleanup.

Only then start the long training run.

## Definition of Done

The adaptation is complete when:

- a new user can build and push the pinned image from documentation;
- the notebook can reserve the configured GPU worker and create an
  NVIDIA-enabled container;
- CUDA diagnostics pass inside that container;
- training can start from one documented command;
- training survives a laptop disconnection;
- metrics and checkpoints are written throughout the run;
- a stopped run can resume from an exported checkpoint;
- all artifacts are copied to persistent storage;
- container and lease cleanup are documented and verified;
- no secrets or large generated artifacts are committed.
