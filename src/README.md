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
results/thesis_reproduction/config_001/
...
results/thesis_reproduction/config_007/
results/thesis_reproduction/summary.csv
results/thesis_reproduction/latency_breakdown.csv
results/thesis_reproduction/communication_efficiency.csv
results/thesis_reproduction/threshold_trajectory.csv
results/thesis_reproduction/plots/
```

Regenerate only the thesis-style plots from existing CSV outputs:

```bash
python src/run_thesis_reproduction.py --plot-only
```

The plot folder contains Figure 5-1 through Figure 5-6 equivalents using the
same visual structure as the thesis.

The reusable implementation lives in:

- `experiment_runner.py`: shared sample sending, timing analysis, summaries, and
  metadata.
- `constants.py`: local paths, output filenames, and timing column definitions.
- `utils.py`: config, process, and timing helpers.

The goal of the current repo is thesis reproduction first, then the same
reproduction with a GPU edge server. Older exploratory runner scripts were
removed to keep that path obvious.
