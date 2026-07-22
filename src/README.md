# Python Runner

This folder now has one experiment entrypoint:

```bash
python src/run_thesis_reproduction.py
```

That command reproduces the seven thesis configurations on already-running
ExPECA public-IP containers. It uses:

```text
config/defaults.env
config/experiment.env
config/thesis_reproduction.env
config/thesis_configs.csv
```

Preview the resolved setup without contacting ExPECA:

```bash
python src/run_thesis_reproduction.py --dry-run
```

Outputs are written under:

```text
results/thesis_reproduction/config_001/raw_edge_device_results.csv
results/thesis_reproduction/config_001/timing_results.csv
...
results/thesis_reproduction/config_007/raw_edge_device_results.csv
results/thesis_reproduction/config_007/timing_results.csv
results/thesis_reproduction/summary.csv
results/thesis_reproduction/summary.md
results/thesis_reproduction/run_metadata.json
results/thesis_reproduction/latency_breakdown.csv
results/thesis_reproduction/offloading_distribution.csv
results/thesis_reproduction/per_sample_latency.csv
results/thesis_reproduction/threshold_trajectory.csv
results/thesis_reproduction/plots/
```

Regenerate only the thesis-style plots from existing CSV outputs:

```bash
python src/run_thesis_reproduction.py --plot-only
```

The plot folder contains Figure 5-1 through Figure 5-6 equivalents using the
same visual structure as the thesis.

The runnable entrypoint stays intentionally small. The reusable implementation lives in:

- `run_thesis_reproduction.py`: CLI only; parses flags and calls the runner.
- `thesis_reproduction_runner.py`: orchestrates configs, per-config runs, and aggregate CSV files.
- `thesis_public_ip_run.py`: one ExPECA public-IP run against already-running containers.
- `thesis_models.py`: thesis config rows.
- `thesis_metrics.py`: thesis accuracy, latency, communication, threshold, and batching metrics.
- `thesis_plots.py`: Figure 5-1 through Figure 5-6 plotting code.
- `thesis_report.py`: aggregate Markdown summary and metadata JSON writer.
- `experiment_runner.py`: shared sample sending, timing analysis, and per-run outputs.
- `constants.py`: local paths, output filenames, and timing column definitions.
- `utils.py`: config, process, and timing helpers.

The goal of the current repo is thesis reproduction first, then the same
reproduction with a GPU edge server. Older exploratory runner scripts were
removed to keep that path obvious.
