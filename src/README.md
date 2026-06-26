# Python Utilities

This folder contains Python utilities for local development and experiment
orchestration. Shell setup/download helpers live in `scripts/`.

The local smoke test code is split by responsibility:

- `run_local_smoke_test.py` runs the workflow and post-processes the results.
- `constants.py` keeps paths, defaults, and timing column definitions.
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

It reads defaults from `config/experiment.env`. The edge-device service writes
raw results to `results/EdgeDevice_results.csv`.
After inference finishes, the same file automatically analyzes the raw results
and writes `results/analysis/summary.md` and
`results/analysis/timing_results.csv`.

For this local utility:

- `CONTROLLER_BATCH_SIZE` is how many images the utility sends to the edge device.
- `BATCH_SIZE` is how many offloaded images the edge device sends to the edge server.
- `FLUSH_FINAL_BATCH=true` sends any final partial edge-server batch at the end
  of the request.

Outputs from the analysis step:

```text
results/analysis/timing_results.csv
results/analysis/summary.md
```
