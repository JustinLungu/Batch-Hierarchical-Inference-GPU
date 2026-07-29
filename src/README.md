# Python Tools

Run the CPU/GPU benchmark against already-running ExPECA public-IP containers:

```bash
.venv/bin/python src/run_benchmark.py
```

Preview the resolved configurations or regenerate plots without rerunning ExPECA:

```bash
.venv/bin/python src/run_benchmark.py --dry-run
.venv/bin/python src/run_benchmark.py --plot-only
```

The benchmark reads `config/defaults.env`, `config/experiment.env`,
`config/thesis_reproduction.env`, and `config/thesis_configs.csv`. Results are
written to the directory selected by `THESIS_OUTPUT_DIR`.

Analyze actual server batch sizes in an existing CPU or GPU result directory:

```bash
.venv/bin/python src/analyze_offload_batches.py [results-directory]
```

## Packages

- `benchmark/`: configuration loading, ExPECA requests, timing derivation,
  aggregate metrics, reports, and plots for the main CPU/GPU experiment.
- `offload_batch_analysis/`: grouped trends and plots for actual dynamic
  offload batch sizes.

The two root Python files own their command-line arguments and call into the
corresponding packages. Package modules are implementation details and are not
run directly.
