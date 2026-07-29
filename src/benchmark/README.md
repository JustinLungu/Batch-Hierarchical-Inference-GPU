# Benchmark Package

This package implements the main CPU/GPU benchmark pipeline. It runs the
configured experiment matrix against edge-device and edge-server containers
that are already running on ExPECA, derives comparable timing metrics, and
produces the aggregate reports and thesis-style figures.

Use the public entrypoint from the repository root:

```bash
.venv/bin/python src/run_benchmark.py
```

Modules in this package are implementation details and are not intended to be
run directly.

## Pipeline

For every selected configuration, the package:

1. Loads stable defaults, active runtime settings, model/dataset settings, and
   the configuration matrix.
2. Verifies that the edge-device and edge-server HTTP services are reachable.
3. Sends the resolved experiment configuration to both services.
4. Reads images from the configured dataset and sends them to the edge device
   in controller batches.
5. Downloads the raw edge-device result CSV after the configuration finishes.
6. Derives named durations from the recorded timestamps.
7. Calculates per-configuration accuracy, offloading, latency, throughput, and
   threshold metrics.
8. Writes aggregate CSV files, a Markdown summary, run metadata, and Figures
   5-1 through 5-6.

The runner leaves the remote containers running. Container reservation,
creation, and cleanup belong to the ExPECA notebooks.

## Module Map

### `../run_benchmark.py`

Defines the public command-line interface and creates `BenchmarkRunner`.

- `--dry-run`: validate assets and print the resolved configurations.
- `--plot-only`: rebuild reports and plots from existing result CSV files.

### `runner.py`

Orchestrates the complete configuration matrix. `BenchmarkRunner` owns the
outer loop, per-configuration result directories, aggregate data frames,
report generation, and plot generation.

This is the high-level workflow layer. HTTP request details and metric formulas
belong in the modules below.

### `config.py`

Represents one row from `config/thesis_configs.csv` as
`BenchmarkConfiguration`. It combines that row with the fixed dataset/model
settings and the requested sample limit.

### `public_ip.py`

Specializes one experiment for already-running ExPECA public-IP services. It
checks service availability, downloads the remote result CSV, and delegates
post-processing to `results.py`.

### `experiment.py`

Contains the network and request mechanics for one benchmark configuration:

- loading runtime configuration;
- preparing multipart image requests;
- sending controller batches;
- collecting images and labels;
- sending service configuration.

`controller_batch_size` controls how many images one controller request sends
to the edge device. The actual edge-server batch can be smaller because
`dynamic_batching` forwards only the samples selected for offloading.

### `results.py`

Converts one configuration's raw timestamps into named durations, writes its
normalized timing CSV, prints the run summary, and provides the run-level
metrics consumed by the aggregate analysis.

### `metrics.py`

Calculates aggregate values used by the reports and plots:

- accuracy for the complete system, SML, and offloaded LML samples;
- offloading classification distribution;
- offload ratio and observed transmission count;
- threshold trajectories;
- per-sample latency;
- the six-step latency breakdown.

The Figure 5-5 server path is averaged over offloaded samples. Local samples do
not contribute zero-valued server durations to that average.

### `plots.py`, `classification_plots.py`, and `performance_plots.py`

`plots.py` is the small figure orchestrator. Classification, offloading, and
threshold figures live in `classification_plots.py`; latency and throughput
figures live in `performance_plots.py`. Plotting remains separate from metric
calculation so existing results can be replotted without contacting ExPECA.

### `report.py`

Writes the human-readable aggregate summary and run metadata.

### `constants.py`

Defines package-wide paths, raw timestamp columns, derived timing columns, and
the mapping from timestamp pairs to duration names.

### `utils.py`

Contains shared helpers for environment parsing, image discovery, numeric
aggregation, timestamp subtraction, duration formatting, and consistent plot
styling.

## Configuration Inputs

The package reads:

```text
config/defaults.env
config/experiment.env
config/thesis_reproduction.env
config/thesis_configs.csv
```

Environment variables override values read from the `.env` files. The most
important active settings are the device, public IPs, selected configuration
IDs, sample limit, and output directory.

## Result Structure

Each configuration directory contains the untouched downloaded edge-device
CSV and a normalized timing CSV. The parent result directory contains the
aggregate tables, summary, metadata, and plot directory.

Existing aggregate outputs can be regenerated with:

```bash
.venv/bin/python src/run_benchmark.py --plot-only
```

## Maintenance Rules

- Preserve raw timestamp columns; downstream analysis depends on them.
- Put new experiment orchestration in `runner.py`, request mechanics in
  `experiment.py` or `public_ip.py`, per-run result conversion in `results.py`,
  metric formulas in `metrics.py`, and rendering in the plot modules.
- Do not add interactive prompts to this package. The benchmark is designed to
  run unattended.
- Keep result schema changes explicit because existing CPU/GPU result folders
  are valid inputs to `--plot-only` and the offload-batch analysis package.
