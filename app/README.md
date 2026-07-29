# Runtime Applications

This directory contains the two FastAPI services that run inside the ExPECA
containers:

```text
local benchmark controller
        |
        v
edge-device service (SML and offloading decision)
        |
        v
edge-server service (LML on CPU or GPU)
```

Experiment orchestration, aggregate analysis, and plotting live under `src/`.
Changes under `app/` affect the recorded experiment behavior and should
therefore be treated more carefully than post-processing changes.

## Directory Layout

```text
app/
  edge_device/
    edge_device.py
    offloading_decision_maker.py
    sample_offloading_method.py
    Dockerfile.edge_device
    entrypoint_edge_device.sh

  edge_server/
    edge_server.py
    Dockerfile.edge_server
    Dockerfile.edge_server.gpu
    entrypoint_edge_server.sh
```

The edge device listens on port `8000`. The edge server listens on port `8001`.

## Request Flow

For each benchmark configuration:

1. The local controller sends configuration payloads to both `/config`
   endpoints.
2. The edge server loads the configured large model (LML).
3. The edge device loads the configured small model (SML), resets its results,
   batching state, and adaptive-threshold state.
4. The controller sends one controller batch to the edge device's `/predict`
   endpoint.
5. The edge device runs SML inference and makes an offloading decision for
   every image.
6. Selected images are sent to the edge server according to the configured
   offloading strategy.
7. The edge server runs LML inference and returns predictions plus server-side
   timestamps.
8. The edge device merges the LML results with its SML results and writes the
   raw result CSV.
9. The controller downloads that CSV from the edge device's `/results`
   endpoint for post-processing.

## Edge Device

`edge_device/edge_device.py` owns:

- SML model loading and inference.
- Per-image confidence and prediction recording.
- Offloading decisions.
- Request-level batching and calls to the edge server.
- Adaptive-threshold feedback.
- Raw timestamps and `EdgeDevice_results.csv`.

Its HTTP endpoints are:

| Endpoint | Purpose |
|---|---|
| `POST /config` | Load the SML and reset state for a new configuration. |
| `POST /predict` | Process a controller batch and optionally offload selected samples. |
| `GET /results` | Download the current raw edge-device result CSV. |
| `GET /logs` | Read the edge-device service log. |

### Decision Methods

`edge_device/offloading_decision_maker.py` implements:

- `never_offload`
- `always_offload`
- `fixed_threshold`
- `adaptive_threshold`

For thesis-compatible adaptive runs, the feedback signal follows the original
implementation: agreement between the SML and LML predictions.

### Offloading Strategies

`edge_device/sample_offloading_method.py` implements:

- `send_individually`: send each selected image in its own server request.
- `dynamic_batching`: collect the selected subset from the current controller
  request and send that subset together at the end of the request.
- `size_based_batching`: keep selected samples in a persistent buffer and send
  them when `batch_size` is reached.

`controller_batch_size` and `batch_size` are different controls.
`controller_batch_size` determines how many images arrive at the edge device in
one request. `batch_size` is used by `size_based_batching` as its offload-buffer
target. The thesis configurations `005`-`007` use `dynamic_batching`, so their
actual server batch is the number selected for offloading from each controller
batch, not necessarily 5, 15, or 45.

## Edge Server

`edge_server/edge_server.py` owns:

- LML model loading.
- CPU or CUDA device selection.
- Image preprocessing and LML inference.
- Memory-aware GPU micro-batching.
- Server-side inference and request timestamps.
- Runtime diagnostics written to the service log.

Its HTTP endpoints are:

| Endpoint | Purpose |
|---|---|
| `POST /config` | Load the LML and resolve LML batching settings. |
| `POST /predict` | Run LML inference for an offloaded request. |
| `GET /logs` | Read the edge-server service log and device diagnostics. |

### GPU Micro-Batching

GPU micro-batching divides one received offload request into model-inference
chunks. It does not change which samples the edge device selected or the actual
network request batch size.

The relevant settings are centralized in `config/experiment.env`:

- `LML_BATCHING_MODE`: `auto`, `sequential`, `fixed`, or `adaptive`.
- `LML_INITIAL_BATCH_SIZE`: first attempted inference chunk size.
- `LML_MIN_BATCH_SIZE`: lower retry bound.
- `LML_MAX_BATCH_SIZE`: upper growth bound.
- `LML_GPU_MEMORY_FRACTION`: memory-pressure limit used by adaptive mode.
- `LML_OOM_RETRY`: retry a CUDA out-of-memory failure with a smaller chunk.

With `auto`, CUDA uses adaptive batching while CPU uses sequential inference.
After a successful CUDA batch, adaptive mode may grow the next chunk up to the
maximum. Under memory pressure or an OOM, it reduces the chunk size down to the
minimum and retries when enabled.

## Timing Data

The services record the raw events needed by the benchmark analysis, including:

- SML inference start and end.
- Offloading decision time.
- Edge-device send and response times.
- Edge-server request receipt and response send times.
- LML inference start and end.
- Result completion times.

The app stores timestamps rather than thesis figures. The benchmark package
under `src/benchmark/` derives latency components and plots from these raw
events.

## Containers

- `edge_device/Dockerfile.edge_device` builds the edge-device service. The
  build scripts target amd64 for regular workers or ARM64 for the Raspberry Pi.
- `edge_server/Dockerfile.edge_server` builds the CPU edge server.
- `edge_server/Dockerfile.edge_server.gpu` builds the CUDA-enabled GPU edge
  server.
- The entrypoint scripts configure container networking/SSH and start the
  corresponding FastAPI service.

Use the scripts under `scripts/` to build and push images rather than invoking
the Dockerfiles manually. See `scripts/README.md` for the supported commands.

