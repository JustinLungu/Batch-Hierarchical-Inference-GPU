# Offload Batch Analysis Package

This package is a post-processing tool for benchmark configurations `005`,
`006`, and `007`. It studies how edge-server performance changes with the
**actual number of samples offloaded in one request**.

It does not contact ExPECA and does not rerun an experiment. It reads existing
`raw_edge_device_results.csv` files produced by the benchmark package.

Use the public entrypoint from the repository root:

```bash
# GPU result directory is the default
.venv/bin/python src/analyze_offload_batches.py

# Analyze a specific CPU or GPU result directory
.venv/bin/python src/analyze_offload_batches.py results/CPU_thesis_reproduction
```

## Why This Analysis Exists

The controller sends fixed batches of 5, 15, or 45 images to the edge device
for configurations `005`, `006`, and `007`. With `dynamic_batching`, only the
subset selected by the offloading decision is sent to the edge server.

Consequently, a controller batch of 45 can produce server requests containing
different actual batch sizes, such as 17, 20, or 22 samples. This package
compares those naturally occurring server batch sizes without changing the
experiment logic.

## Data Flow

1. `src/analyze_offload_batches.py` loads the configured controller batch size
   for configs `005`-`007`.
2. It reads each configuration's raw edge-device CSV.
3. `OffloadBatchAnalyzer` groups sample rows that share
   `ts_sample_sent_to_edge_server`; those rows belong to one server request.
4. Each request becomes one measurement with its actual batch size and server
   timing values.
5. Measurements are grouped by configuration and actual batch size.
6. Correlations, linear trends, grouped statistics, and plots are written
   beside the source results.

## Timing Boundary

The primary response-time measurement is:

```text
ts_results_sent_to_edge_device - ts_sample_received_at_edge_server
```

It measures time inside the server request boundary, from arrival at the edge
server until the response leaves the edge server. It excludes edge-to-server
and server-to-edge network transit.

The package also separates:

- server queue or preprocessing time;
- LML wall-clock inference time;
- server postprocessing time;
- response time per image;
- effective server throughput.

## Module Map

### `../analyze_offload_batches.py`

Loads result paths and configuration context, runs the analysis, writes CSV
files and plots, and prints a short trend summary.

### `analyzer.py`

Contains the analysis logic:

- `OffloadBatchContext` identifies the configuration and controller batch size.
- `extract_batch_measurements()` collapses per-sample rows into one row per
  edge-server request.
- `summarize_by_batch_size()` calculates counts, means, medians, standard
  deviations, percentiles, per-image time, throughput, and micro-batch values.
- `calculate_config_trends()` calculates Pearson/Spearman correlations and a
  linear response-time slope for each configuration.

The analyzer validates required columns and rejects inconsistent timestamps
within a single inferred server request.

### `utils.py`

Contains the numeric extraction, correlation, trend, and batch-consistency
helpers shared by the analyzer.

### `plots.py`

Creates three comparisons against actual server batch size:

- total server response time;
- server time per image;
- effective server throughput.

Scatter points represent individual server requests. Lines represent grouped
medians, and shaded regions represent the 25th-75th percentile range.

## Outputs

The package writes:

```text
<result-directory>/offload_batch_analysis/
  batch_measurements.csv
  batch_size_summary.csv
  config_trends.csv
  plots/
```

- `batch_measurements.csv`: one row per observed edge-server request.
- `batch_size_summary.csv`: grouped statistics for each config and actual
  server batch size.
- `config_trends.csv`: correlation and linear-trend values per config.

## Interpreting Trends

- A positive response-time correlation means larger batches take longer in
  total.
- A negative per-image-time correlation means batching reduces cost per image.
- A positive throughput correlation means the server processes more images per
  second as the batch grows.

Correlation alone does not prove causation. Sparse batch sizes, warm-up,
micro-batch splitting, GPU contention, and outliers can all affect the
observed trend.

## Assumptions

- Rows sharing `ts_sample_sent_to_edge_server` represent one server request.
- Required server timestamps are present and numeric.
- The source files come from the same application instrumentation used by the
  benchmark package.
- Configurations `005`-`007` remain defined in `config/thesis_configs.csv`.
