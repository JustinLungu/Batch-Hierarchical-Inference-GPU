# Thesis Reproduction Workflow

This document is the main runbook for reproducing the seven configurations from
the thesis and then repeating the same reproduction with a GPU edge server.

The workflow is:

```text
laptop runner
  -> ExPECA edge-device container
      -> ExPECA edge-server container
```

The edge-device always runs the small model. The edge-server runs the large
model. Switching from CPU to GPU should only change the edge-server compute
device and image, not the experiment definitions.

## 1. Prepare Local Assets

Set up the Python environment and download the thesis dataset/models:

```bash
scripts/setup_env.sh
source .venv/bin/activate
scripts/download_dataset.sh --imagenetv2
scripts/download_models.sh --all
scripts/prepare_expeca_author_layout.sh
scripts/setup_expeca_notebook_env.sh
```

The thesis reproduction requires:

```text
data/datasets/imagenetV2/matched-frequency-format-val
data/models/sml/mobilenet_v3_large_imagenet1k_v2.pth
data/models/lml/ViT_H_14_Weights_IMAGENET1K_SWAG_E2E_V1.pth
```

`scripts/download_dataset.sh --imagenetv2` validates that ImageNetV2 contains
1000 class folders and 10000 images.

## 2. Understand the Config Files

There are four config files:

```text
config/defaults.env
config/experiment.env
config/thesis_reproduction.env
config/thesis_configs.csv
```

`config/defaults.env` contains stable paths, download URLs, ports, and default
Docker image tags.

`config/experiment.env` is the runtime file you edit before a run:

```env
DEVICE=cpu
EDGE_DEVICE_IP=...
EDGE_SERVER_IP=...
CONTROLLER_MAX_SAMPLES=all
EXPECA_IMAGE_NAMESPACE=...
EXPECA_EDGE_SERVER_DEVICE=cpu
```

Use `CONTROLLER_MAX_SAMPLES=4` or another small number for a quick validation
run. Use `CONTROLLER_MAX_SAMPLES=all` for full thesis reproduction.

`config/thesis_reproduction.env` fixes the thesis dataset and model pair:

```text
ImageNetV2 Matched Frequency
MobileNetV3-Large
ViT-H/14
```

`config/thesis_configs.csv` defines the seven thesis configurations. These are
data, not Python code:

```text
001 never_offload
002 always_offload + send_individually
003 fixed_threshold + send_individually, threshold 0.3888
004 adaptive_threshold + send_individually
005 adaptive_threshold + dynamic_batching, controller batch 5
006 adaptive_threshold + dynamic_batching, controller batch 15
007 adaptive_threshold + dynamic_batching, controller batch 45
```

For Configs `005` through `007`, the thesis batch size is the input/controller
batch size. That means the important field is:

```text
CONTROLLER_BATCH_SIZE=5
CONTROLLER_BATCH_SIZE=15
CONTROLLER_BATCH_SIZE=45
```

These configs use `OFFLOADING_STRATEGY=dynamic_batching`. They are not the
size-based batching grid we used earlier.

Config `002` is intentionally kept. In the thesis it is the accuracy upper bound
and high-latency baseline. For our GPU work it is also the cleanest way to
isolate large-model/server behavior because every sample is sent to the LML.

## 3. Build and Push CPU Images

Log in to Docker Hub or your container registry:

```bash
docker login
```

Set your namespace in `config/experiment.env`:

```env
EXPECA_IMAGE_NAMESPACE=YOUR_DOCKERHUB_USERNAME_OR_REGISTRY_NAMESPACE
```

Build and push the CPU images:

```bash
scripts/check_expeca_cpu_prereqs.sh
scripts/build_expeca_cpu_images.sh
scripts/push_expeca_cpu_images.sh
```

## 4. Start CPU Containers on ExPECA

Open:

```text
notebooks/ExPECA_HI_setup_Public_IP.ipynb
```

Run the authentication/setup cells first. Use the same image namespace/tag you
pushed. For the CPU reproduction:

```python
EDGE_SERVER_DEVICE = "cpu"
EDGE_DEVICE_DEVICE = "cpu"
```

Create the edge-server container first and record its public IP. Then create the
edge-device container and record its public IP. The edge-device environment must
receive the edge-server public IP.

Confirm both services are reachable:

```bash
curl http://EDGE_SERVER_PUBLIC_IP:8001/logs
curl http://EDGE_DEVICE_PUBLIC_IP:8000/logs
```

Then put those public IPs in `config/experiment.env`:

```env
DEVICE=cpu
EDGE_SERVER_IP=EDGE_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=EDGE_DEVICE_PUBLIC_IP
EXPECA_EDGE_SERVER_DEVICE=cpu
```

## 5. Preview the Thesis Run

Before launching a long run, print the resolved configuration table:

```bash
.venv/bin/python src/run_thesis_reproduction.py --dry-run
```

This validates local thesis assets and prints configs `001` through `007`.

## 6. Run the CPU Thesis Reproduction

For a short validation:

```env
CONTROLLER_MAX_SAMPLES=4
```

For the full reproduction:

```env
CONTROLLER_MAX_SAMPLES=all
```

Run the closest CPU reproduction with explicit CPU overrides:

```bash
DEVICE=cpu \
EXPECA_EDGE_SERVER_DEVICE=cpu \
THESIS_CONFIGS_TO_RUN=all \
THESIS_OUTPUT_DIR=CPU_thesis_reproduction \
LML_BATCHING_MODE=sequential \
.venv/bin/python src/run_thesis_reproduction.py
```

The runner executes all seven configurations in order. Each configuration sends
a fresh `/config` request to the edge-device and edge-server. The edge-device
clears previous results, old offload buffers, and adaptive-threshold state for
each configuration.

For the closest thesis match, run the edge-device on the Raspberry Pi/ARM worker
and the CPU edge-server on a separate CPU worker. Running both containers on the
same worker is useful for smoke tests, but it changes queueing, network timing,
and resource contention.

Aggregate outputs are written to the configured output directory. With the
command above, that is:

```text
results/CPU_thesis_reproduction/summary.csv
results/CPU_thesis_reproduction/summary.md
results/CPU_thesis_reproduction/run_metadata.json
results/CPU_thesis_reproduction/latency_breakdown.csv
results/CPU_thesis_reproduction/offloading_distribution.csv
results/CPU_thesis_reproduction/per_sample_latency.csv
results/CPU_thesis_reproduction/threshold_trajectory.csv
results/CPU_thesis_reproduction/plots/
```

Each individual configuration writes a detailed folder:

```text
results/CPU_thesis_reproduction/config_001/
results/CPU_thesis_reproduction/config_002/
results/CPU_thesis_reproduction/config_003/
results/CPU_thesis_reproduction/config_004/
results/CPU_thesis_reproduction/config_005/
results/CPU_thesis_reproduction/config_006/
results/CPU_thesis_reproduction/config_007/
```

Each config folder contains the raw edge-device CSV and the derived timing CSV
for that one configuration. The aggregate `summary.md` and `run_metadata.json`
store the human-readable report and run provenance once at the top level.

`latency_breakdown.csv` is the table behind the thesis-style stacked latency
plot. It contains:

```text
Step 1: ED Processing
Step 2: ED Offload Buffer
Step 3: ED to ES Communication
Step 4: ES Processing
Step 5: ES to ED Communication
Step 6: ED Result Saving
```

`summary.csv` includes accuracy, SML accuracy, LML accuracy for offloaded
samples, offloading ratio, throughput, and latency aggregates for each config.

`threshold_trajectory.csv` records the adaptive threshold seen by each sample in
Configs `004` through `007`, plus the post-update threshold when an offloaded
sample updates the adaptive model. For thesis reproduction, the adaptive update follows the original implementation:
it treats `LML Prediction == SML Prediction` as the correctness feedback.

The plot folder contains thesis-style figures:

```text
figure_5_1_accuracy_comparison.png
figure_5_2_offloading_decision_distributions.png
figure_5_3_threshold_value_updates.png
figure_5_4_per_sample_latency_comparison.png
figure_5_5_latency_breakdown.png
figure_5_6_throughput_processing_time.png
```

Regenerate only the figures from existing CSV outputs with:

```bash
.venv/bin/python src/run_thesis_reproduction.py --plot-only
```

If `figure_5_3_threshold_value_updates.png` is empty, check whether the
per-config raw/timing CSVs contain `Decision Threshold` or
`Adaptive Threshold After Update`. Older edge-device images did not save these
columns. The plot-only command intentionally uses only values logged by the
edge-device, so rebuild and recreate the edge-device container if those columns
are absent.

## 7. Repeat With GPU

After the CPU thesis reproduction works, build and push the GPU edge-server image:

```bash
scripts/build_expeca_gpu_server_image.sh
scripts/push_expeca_gpu_server_image.sh
```

In the ExPECA notebook, keep the edge-device on CPU and switch only the
edge-server to the GPU image/device:

```python
EDGE_SERVER_DEVICE = "cuda"
EDGE_DEVICE_DEVICE = "cpu"
```

Update `config/experiment.env`:

```env
DEVICE=cuda
EDGE_SERVER_IP=GPU_EDGE_SERVER_PUBLIC_IP
EDGE_DEVICE_IP=EDGE_DEVICE_PUBLIC_IP
EXPECA_EDGE_SERVER_DEVICE=cuda
CONTROLLER_MAX_SAMPLES=all
```

Run the GPU benchmark configs:

```bash
DEVICE=cuda \
EXPECA_EDGE_SERVER_DEVICE=cuda \
THESIS_CONFIGS_TO_RUN=002,003,004,005,006,007 \
THESIS_OUTPUT_DIR=thesis_reproduction_gpu \
.venv/bin/python src/run_thesis_reproduction.py
```

GPU outputs are written to:

```text
results/thesis_reproduction_gpu/
```

The CPU and GPU runs are directly comparable when they use the same thesis
dataset, model pair, configuration definitions, and edge-device worker. Config
`001` is skipped for GPU because it never sends work to the edge-server.
