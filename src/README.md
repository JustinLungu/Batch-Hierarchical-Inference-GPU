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

The reusable implementation lives in:

- `experiment_runner.py`: shared sample sending, timing analysis, summaries, and
  metadata.
- `constants.py`: local paths, output filenames, and timing column definitions.
- `utils.py`: config, process, and timing helpers.

The goal of the current repo is thesis reproduction first, then the same
reproduction with a GPU edge server. Older exploratory runner scripts were
removed to keep that path obvious.
