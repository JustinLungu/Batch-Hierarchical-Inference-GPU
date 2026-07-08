# Python Utilities

This folder contains Python utilities for local development and experiment
orchestration. Shell setup/download helpers live in `scripts/`.

The local smoke test code is split by responsibility:

- `experiment_runner.py` owns the shared controller, timing, summary, and metadata flow.
- `run_local_smoke_test.py` starts/stops local Python service processes.
- `run_local_docker_test.py` starts/stops local Docker service containers.
- `run_expeca_public_ip_test.py` runs one remote ExPECA public-IP experiment
  against already-running containers.
- `run_expeca_batch_grid.py` runs repeated remote ExPECA experiments across
  configured batch sizes and writes an aggregate comparison table.
- `compare_grid_results.py` combines CPU/GPU grid summaries and writes
  comparison CSVs, pivot tables, and optional plots.
- `constants.py` keeps local paths, filenames, and timing column definitions.
- `utils.py` keeps small reusable helpers for config, processes, and timing math.

## Local Smoke Test

Run the local Python-script path with one small fixed batch:

```bash
source .venv/bin/activate
python src/run_local_smoke_test.py
```

Set `DEVICE` in `config/experiment.env` to choose compute mode:

```text
DEVICE=auto  # use CUDA if available, otherwise CPU
DEVICE=cpu   # force CPU
DEVICE=cuda  # require CUDA, fail if unavailable
```

The smoke test starts:

```text
edge_server.py
edge_device.py
```

Then it sends one configured sample batch through:

```text
controller utility -> edge_device.py -> edge_server.py
```

It reads stable defaults from `config/defaults.env` and active run choices from
`config/experiment.env`. The edge-device service writes
raw results to `results/EdgeDevice_results.csv`.
After inference finishes, the same file automatically analyzes the raw results
and writes a device-specific analysis folder such as `results/analysis_local_cpu/`.

For this local utility:

- `CONTROLLER_BATCH_SIZE` is how many images the utility sends to the edge device.
- `BATCH_SIZE` is how many offloaded images the edge device sends to the edge server.
- `FLUSH_FINAL_BATCH=true` sends any final partial edge-server batch at the end
  of the request.

Outputs from the analysis step:

```text
results/analysis_local_cpu/summary.md
results/analysis_local_cpu/timing_results.csv
results/analysis_local_cpu/run_metadata.json
results/analysis_local_cpu/raw_edge_device_results.csv
```

## Local Docker Test

Run the same controller and analysis flow, but with `edge_server.py` and
`edge_device.py` inside Docker containers:

```bash
source .venv/bin/activate
python src/run_local_docker_test.py
```

The Docker runner builds the local images if they do not exist, starts the
`edge_server` and `edge_device` containers on the `edge_net` Docker network,
sends the configured sample batch from the host Python process, and removes the
containers when the run finishes.

Outputs from the Docker analysis step:

```text
results/analysis_local_docker_cpu/summary.md
results/analysis_local_docker_cpu/timing_results.csv
results/analysis_local_docker_cpu/run_metadata.json
results/analysis_local_docker_cpu/raw_edge_device_results.csv
```

## ExPECA Batch Grid

After the ExPECA public-IP containers are running, configure the sweep in
`config/experiment.env`:

```env
BATCH_SIZE_GRID=1,2,4,8,16,32,64
CONTROLLER_BATCH_SIZE=64
CONTROLLER_BATCH_SIZE_GRID=
CONTROLLER_MAX_SAMPLES=all
```

Then run:

```bash
source .venv/bin/activate
python src/run_expeca_batch_grid.py
```

With `CONTROLLER_BATCH_SIZE_GRID` empty, the grid changes only `BATCH_SIZE`.
That is the recommended first comparison for CPU vs GPU because the controller
request size stays fixed.

Aggregate outputs:

```text
results/analysis_expeca_public_ip_cpu_grid/summary.csv
results/analysis_expeca_public_ip_cpu_grid/summary.md
results/analysis_expeca_public_ip_cpu_grid/run_metadata.json
```

Each grid item also writes its own detailed folder, named by server and
controller batch size.

## Grid Comparison

After one or more grid runs exist, combine them with:

```bash
source .venv/bin/activate
python src/compare_grid_results.py
```

It discovers folders named like:

```text
results/analysis_expeca_public_ip_cpu_grid/
results/analysis_expeca_public_ip_gpu_grid/
```

and writes:

```text
results/comparison_expeca_public_ip/combined_grid_summary.csv
results/comparison_expeca_public_ip/best_by_metric.csv
results/comparison_expeca_public_ip/*_pivot.csv
results/comparison_expeca_public_ip/summary.md
```

If `matplotlib` is installed, it also writes line plots and heatmaps under:

```text
results/comparison_expeca_public_ip/plots/
```
